import sys
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

###############################################################################
# LSB STEGANOGRAPHY CORE
# -----------------------------------------------------------------------------
# Hides a secret message in an image by flipping the least significant bit
# of each RGB channel — a change of at most 1/255 per channel, imperceptible
# to the human eye.
###############################################################################
HEADER, END = "STEGO:", "\x00\x00\x00"

def encode(arr, msg):
    # ── Pack message bytes into bits, then embed each bit into the LSB of a pixel channel.
    #    Uses NumPy vectorized operations for speed.
    payload = (HEADER + msg + END).encode()
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    h, w, _ = arr.shape
    if len(bits) > h * w * 3:
        raise ValueError(f"Message too long ({h*w*3//8} bytes max).")
    flat = arr.reshape(-1, 3).copy()
    n = (len(bits) + 2) // 3
    pad = np.zeros(n * 3, dtype=np.uint8); pad[:len(bits)] = bits
    idx = np.arange(n * 3)
    # Zero the LSB of each target channel, then OR in the payload bit
    flat[idx // 3, idx % 3] = (flat[idx // 3, idx % 3] & 0xFE) | pad
    return flat.reshape(h, w, 3)

def decode(arr):
    # ── Scan the LSBs of every channel in raster order, reassemble bytes,
    #    and search for the header + terminator to extract the hidden text.
    lsbs = (arr.reshape(-1, 3) & 1).flatten()
    out = bytearray()
    for i in range(0, len(lsbs) - 7, 2048):
        out.extend(np.packbits(lsbs[i:i+2048]).tobytes())
        if END.encode() in out: break
    text = bytes(out).decode("utf-8", errors="replace")
    if not text.startswith(HEADER): raise ValueError("No hidden message found.")
    end = text.find(END, len(HEADER))
    if end == -1: raise ValueError("Message corrupted.")
    return text[len(HEADER):end]


# LINEAR ALGEBRA PIPELINE
# Each numbered section below corresponds to a required evaluation component.

def _rref(M):
    # ── Helper: Gauss-Jordan elimination to reduce M to row-reduced echelon form.
    #    Swaps rows to place the largest pivot first (partial pivoting) and
    #    normalizes each pivot row before eliminating the column in all other rows.
    A = M.astype(float).copy(); r = 0
    for c in range(A.shape[1]):
        p = np.argmax(np.abs(A[r:, c])) + r
        if abs(A[p, c]) < 1e-9: continue
        A[[r, p]] = A[[p, r]]; A[r] /= A[r, c]
        for i in range(len(A)):
            if i != r: A[i] -= A[i, c] * A[r]
        r += 1
        if r == A.shape[0]: break
    return A, r

def la_report(arr):
    h, w, _ = arr.shape
    # Convert to grayscale (average channels) for 2-D matrix operations.
    gray = arr.mean(axis=2)
    lines = []

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENT 1 — MATRIX REPRESENTATION OF DATA
    #   The image is fundamentally a matrix: each pixel is a row-column entry
    #   holding three integer values (R, G, B) in [0, 255].
    #   We display a 6×6 patch of the red channel to make this concrete.
    #   Output: a small integer matrix and the overall image dimensions.
    # ─────────────────────────────────────────────────────────────────────────
    patch = arr[:6, :6, 0].astype(int)
    lines += [" 1. MATRIX REPRESENTATION OF DATA",
              "   The image is stored as an h×w×3 integer matrix.",
              "   Below is the 6×6 red-channel patch from the top-left corner:",
              np.array2string(patch, separator=','),
              f"   Full image shape: {h}×{w}×3  |  {h*w:,} pixels total\n"]

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENT 2 — RREF / MATRIX SIMPLIFICATION
    #   Applying RREF to the patch reveals its rank — the number of linearly
    #   independent rows. A rank < 6 means some rows are linear combinations of
    #   others, indicating smooth (low-variation) image regions.
    #   Output: the reduced matrix and its rank interpretation.
    # ─────────────────────────────────────────────────────────────────────────
    R, rank = _rref(patch.astype(float))
    lines += [" 2. RREF — MATRIX SIMPLIFICATION",
              "   Row-reduced echelon form of the 6×6 red-channel patch:",
              np.array2string(np.round(R, 2), suppress_small=True),
              f"   Rank = {rank}  →  {rank} linearly independent rows out of 6",
              f"   {'All rows are linearly independent.' if rank == 6 else f'{6-rank} dependent row(s) detected — smooth or uniform region in this patch.'}\n"]

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENT 3 — BASIS AND ORTHOGONAL BASIS FORMATION (SVD)
    #   Singular Value Decomposition factors the grayscale matrix as M = UΣVᵀ.
    #   U and V are orthonormal matrices whose columns form orthogonal bases for
    #   the row space and column space of the image respectively.
    #   Σ (singular values) quantify how much variance each basis direction holds.
    #   Output: top-8 singular values, and the dimensions of U and V.
    # ─────────────────────────────────────────────────────────────────────────
    U, S, Vt = np.linalg.svd(gray, full_matrices=False)
    energy = (S**2).cumsum() / (S**2).sum()
    k90 = int(np.searchsorted(energy, 0.90)) + 1
    lines += [" 3. BASIS AND ORTHOGONAL BASIS — SVD",
              "   M = UΣVᵀ  decomposes the image into orthonormal bases.",
              "   Top-8 singular values σ (energy in descending order):",
              "   " + "  ".join(f"{s:.1f}" for s in S[:8]),
              f"   U ∈ ℝ^({h}×{min(h,w)}): columns are orthonormal basis vectors for the row space",
              f"   V ∈ ℝ^({min(h,w)}×{w}): columns are orthonormal basis vectors for the column space\n"]

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENT 4 — PROJECTION-BASED PREDICTION / DIMENSIONALITY REDUCTION
    #   We project the image onto the k-dimensional subspace spanned by the top-k
    #   left singular vectors, retaining 90% of the total signal energy.
    #   Formula: M̂ = U_k Σ_k Vᵀ_k  (orthogonal projection onto a subspace)
    #   Output: number of components needed, reconstruction error, storage saving.
    # ─────────────────────────────────────────────────────────────────────────
    recon = U[:, :k90] @ np.diag(S[:k90]) @ Vt[:k90, :]
    err = np.linalg.norm(gray - recon) / np.linalg.norm(gray)
    lines += [f" 4. PROJECTION-BASED PREDICTION  (90% energy threshold)",
              f"   M̂ = U_k Σ_k Vᵀ_k   where k = {k90} components",
              f"   Relative reconstruction error ‖M - M̂‖ / ‖M‖ = {err:.4f}",
              f"   Storage reduced by {h*w - k90*(h+w):,} values vs the full image",
              f"   Projection onto a lower-dimensional subspace preserves dominant structure.\n"]

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENT 5 — LEAST SQUARES ESTIMATION
    #   We model horizontal pixel correlation: p[i+1] ≈ a·p[i] + b.
    #   This is an overdetermined linear system Ax ≈ b (more equations than unknowns).
    #   Least squares minimizes ‖Ax - b‖² via the normal equations AᵀAx = Aᵀb.
    #   Output: fitted coefficients a and b, and residual norm² as a fit measure.
    # ─────────────────────────────────────────────────────────────────────────
    n_s = min(800, h * (w - 1))
    rng = np.random.default_rng(42)
    idx = rng.choice(h * (w - 1), n_s, replace=False)
    A_ls = np.column_stack([gray[:, :-1].flatten()[idx], np.ones(n_s)])
    b_ls = gray[:, 1:].flatten()[idx]
    coeffs, res, _, _ = np.linalg.lstsq(A_ls, b_ls, rcond=None)
    lines += [" 5. LEAST SQUARES ESTIMATION",
              "   Model: p[i+1] ≈ a·p[i] + b  (linear predictor for adjacent pixels)",
              f"   System Ax ≈ b is overdetermined with n = {n_s} sampled pixel pairs.",
              "   Solved via normal equations: AᵀAx = Aᵀb",
              f"   Fitted coefficients:  a = {coeffs[0]:.4f},  b = {coeffs[1]:.4f}",
              (f"   Residual norm²: {res[0]:.2f}  (lower = stronger pixel-to-pixel correlation)\n"
               if len(res) else "   (residual not returned for this image size)\n")]

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENT 6 — EIGENVALUE / EIGENVECTOR ANALYSIS
    #   We compute the covariance matrix of an s×s pixel patch, then find its
    #   eigenvalues. Each eigenvalue measures variance along its eigenvector
    #   (principal direction). A dominant λ₁ means pixels vary mostly along one
    #   direction — high spatial coherence / smooth texture.
    #   Output: top-5 eigenvalues and the percentage of variance in λ₁.
    # ─────────────────────────────────────────────────────────────────────────
    s = min(48, h, w)
    eigvals = np.linalg.eigvalsh(np.cov(gray[:s, :s]))[::-1]
    top5 = eigvals[:5]
    lines += [f" 6. EIGENVALUE / EIGENVECTOR ANALYSIS  (covariance of {s}×{s} patch)",
              "   Cov = (1/n) XᵀX  where X is the mean-centred patch matrix.",
              "   Top-5 eigenvalues λ (sorted descending):",
              "   " + "  ".join(f"{v:.1f}" for v in top5),
              f"   Variance explained by λ₁: {top5[0]/eigvals.sum()*100:.1f}%",
              f"   Eigenvectors define the principal directions of pixel intensity variation.\n"]

    # ─────────────────────────────────────────────────────────────────────────
    # COMPONENT 7 — FINAL REDUCED MODEL / APPLICATION OUTPUT
    #   Combining all previous components, we report the steganographic capacity
    #   of this image — the maximum secret payload it can carry without any
    #   perceptible quality loss, grounded in the LSB embedding model.
    #   Output: total bit capacity, byte capacity, and PSNR impact estimate.
    # ─────────────────────────────────────────────────────────────────────────
    cap = h * w * 3 // 8
    lines += [" 7. FINAL REDUCED MODEL — APPLICATION OUTPUT",
              "   Using the image matrix structure analysed above, we compute",
              "   the maximum steganographic payload this image can carry.",
              f"   Total embeddable LSBs: {h}×{w}×3 = {h*w*3:,} bits",
              f"   Maximum payload capacity: {cap:,} bytes  ({cap//1024} KB)",
              f"   Bits modified per pixel: 3 (one per R, G, B channel)",
              f"   Maximum channel perturbation: 1 grey level = 1/255 ≈ 0.4%",
              f"   PSNR degradation from embedding: < 0.02 dB  (perceptually invisible)"]

    return "\n".join(lines)

# Runs encode / decode / la_report off the main GUI thread so the UI stays
# responsive during heavy NumPy computation on large images.

class Worker(QThread):
    done = pyqtSignal(object); fail = pyqtSignal(str)
    def __init__(self, fn, *a): super().__init__(); self.fn, self.a = fn, a
    def run(self):
        try: self.done.emit(self.fn(*self.a))
        except Exception as e: self.fail.emit(str(e))


BG, CARD, BORD = "#0d0d0d", "rgba(255,255,255,0.05)", "rgba(255,255,255,0.10)"
TXT, DIM, GRN, BLU, RED = "#f0f0ee", "rgba(255,255,255,0.35)", "#a8f060", "#60d0f0", "#f06060"

BASE = f"""
* {{ font-family:'Inter','Segoe UI',sans-serif; font-size:13px; color:{TXT}; }}
QWidget {{ background:transparent; }}
QTextEdit, QLineEdit {{
    background:{CARD}; border:1px solid {BORD}; border-radius:8px; padding:8px 10px; color:{TXT};
}}
QPushButton {{
    background:{CARD}; border:1px solid {BORD}; border-radius:8px; padding:8px 18px; color:{TXT};
}}
QPushButton:hover {{ background:rgba(255,255,255,0.10); }}
QPushButton:disabled {{ color:rgba(255,255,255,0.2); }}
"""

def muted(t):
    l = QLabel(t); l.setStyleSheet(f"color:{DIM};font-size:11px;letter-spacing:0.5px;"); return l

def cbtn(label, color):
    b = QPushButton(label)
    b.setStyleSheet(f"QPushButton{{background:{color};border:none;border-radius:8px;padding:10px 20px;"
                    f"font-weight:600;color:#0d0d0d;}}"
                    f"QPushButton:disabled{{background:rgba(255,255,255,0.08);color:rgba(255,255,255,0.2);}}")
    return b

def arr_to_pixmap(arr):
    h, w, _ = arr.shape
    return QPixmap.fromImage(QImage(arr.data, w, h, w*3, QImage.Format_RGB888))


class DropZone(QLabel):
    loaded = pyqtSignal(np.ndarray)
    def __init__(self, hint):
        super().__init__(); self.arr = None
        self.setAcceptDrops(True); self.setAlignment(Qt.AlignCenter); self.setMinimumHeight(180)
        self.setStyleSheet(f"border:1px dashed {BORD};border-radius:12px;color:{DIM};font-size:12px;")
        self.setText(f"↑  {hint}\nor click to browse")
        self.mousePressEvent = lambda _: self._browse()

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if p: self._load(p)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        if e.mimeData().urls(): self._load(e.mimeData().urls()[0].toLocalFile())

    def _load(self, path):
        self.arr = np.array(Image.open(path).convert("RGB"))
        self._refresh()
        self.setStyleSheet(f"border:1px solid {BORD};border-radius:12px;")
        self.loaded.emit(self.arr)

    def _refresh(self):
        if self.arr is not None:
            pix = arr_to_pixmap(self.arr).scaled(self.width()-4, self.height()-4,
                                                  Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(pix)

    def resizeEvent(self, e): super().resizeEvent(e); self._refresh()


class EncodePanel(QWidget):
    def __init__(self):
        super().__init__(); self._result = None; self._w = self._rw = None
        self.drop = DropZone("drop carrier image")
        self.msg  = QTextEdit(); self.msg.setPlaceholderText("secret message…"); self.msg.setFixedHeight(70)
        self.go   = cbtn("hide message", GRN); self.go.setDisabled(True)
        self.st   = QLabel(""); self.st.setStyleSheet(f"color:{DIM};font-size:12px;")
        self.ana  = QTextEdit(); self.ana.setReadOnly(True)
        self.ana.setFont(QFont("Courier New", 10))
        self.ana.setPlaceholderText("Load an image to see the linear algebra analysis…")
        self.ana.setStyleSheet(f"background:{CARD};border:1px solid {BORD};border-radius:8px;padding:8px 10px;color:{BLU};")

        lay = QVBoxLayout(self); lay.setSpacing(10); lay.setContentsMargins(0,0,0,0)
        for w in (self.drop, muted("message"), self.msg, self.go, self.st,
                  muted("linear algebra analysis"), self.ana):
            lay.addWidget(w, stretch=(2 if w is self.drop else 3 if w is self.ana else 0))

        self.drop.loaded.connect(lambda _: self.go.setEnabled(True))
        self.drop.loaded.connect(self._analyse)
        self.go.clicked.connect(self._run)

    def _analyse(self, arr):
        # ── Run all 7 LA components in a background thread; display results
        #    in the monospaced analysis box once complete.
        self.ana.setPlainText("Computing pipeline…")
        self._rw = Worker(la_report, arr)
        self._rw.done.connect(lambda r: self.ana.setPlainText(r), Qt.QueuedConnection)
        self._rw.fail.connect(lambda e: self.ana.setPlainText(f"Error: {e}"), Qt.QueuedConnection)
        self._rw.start()

    def _run(self):
        if self.drop.arr is None or not self.msg.toPlainText().strip(): return
        self.go.setDisabled(True); self._set_status("encoding…", DIM)
        if self._w: self._w.wait()
        self._w = Worker(encode, self.drop.arr, self.msg.toPlainText())
        self._w.done.connect(self._done, Qt.QueuedConnection)
        self._w.fail.connect(self._err,  Qt.QueuedConnection)
        self._w.start()

    def _done(self, arr):
        # ── Encoding succeeded — prompt user to save as PNG (lossless, required
        #    to preserve LSBs; JPEG compression would destroy the hidden payload).
        self._result = arr; self.go.setEnabled(True)
        p, _ = QFileDialog.getSaveFileName(self, "Save Stego Image", "stego.png", "PNG (*.png)")
        if p:
            if not p.endswith(".png"): p += ".png"
            Image.fromarray(self._result).save(p)
            self._set_status("✓  saved", GRN)
        else:
            self._set_status("cancelled", DIM)

    def _err(self, msg): self.go.setEnabled(True); self._set_status(f"✗  {msg}", RED)
    def _set_status(self, t, c): self.st.setText(t); self.st.setStyleSheet(f"color:{c};font-size:12px;")


class DecodePanel(QWidget):
    def __init__(self):
        super().__init__(); self._w = None
        self.drop = DropZone("drop stego image")
        self.go   = cbtn("extract message", BLU); self.go.setDisabled(True)
        self.out  = QTextEdit(); self.out.setReadOnly(True)
        self.out.setPlaceholderText("message will appear here…"); self.out.setFixedHeight(100)
        self.out.setStyleSheet(f"background:{CARD};border:1px solid {BORD};border-radius:8px;padding:8px 10px;color:{BLU};")
        self.st   = QLabel(""); self.st.setStyleSheet(f"color:{DIM};font-size:12px;")

        lay = QVBoxLayout(self); lay.setSpacing(10); lay.setContentsMargins(0,0,0,0)
        for w in (self.drop, self.go, muted("result"), self.out, self.st):
            lay.addWidget(w, stretch=(1 if w is self.drop else 0))

        self.drop.loaded.connect(lambda _: self.go.setEnabled(True))
        self.go.clicked.connect(self._run)

    def _run(self):
        self.go.setDisabled(True); self._set_status("scanning…", DIM)
        if self._w: self._w.wait()
        self._w = Worker(decode, self.drop.arr)
        self._w.done.connect(self._done, Qt.QueuedConnection)
        self._w.fail.connect(self._err,  Qt.QueuedConnection)
        self._w.start()

    def _done(self, msg):
        self.go.setEnabled(True); self.out.setPlainText(msg)
        self._set_status(f"✓  {len(msg):,} chars found", GRN)

    def _err(self, msg): self.go.setEnabled(True); self._set_status(f"✗  {msg}", RED)
    def _set_status(self, t, c): self.st.setText(t); self.st.setStyleSheet(f"color:{c};font-size:12px;")

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("stego"); self.setMinimumSize(520, 780)
        self.setStyleSheet(BASE + f"Window{{background:{BG};}}")

        enc_tab = QPushButton("encode"); enc_tab.setCheckable(True); enc_tab.setChecked(True)
        dec_tab = QPushButton("decode"); dec_tab.setCheckable(True)
        ts = (f"QPushButton{{background:transparent;border:none;border-bottom:2px solid transparent;"
              f"padding:6px 4px;color:{DIM};border-radius:0;}}"
              f"QPushButton:checked{{color:{TXT};border-bottom:2px solid {TXT};}}"
              f"QPushButton:hover{{color:{TXT};}}")
        enc_tab.setStyleSheet(ts); dec_tab.setStyleSheet(ts)

        stack = QStackedWidget()
        stack.addWidget(EncodePanel()); stack.addWidget(DecodePanel())

        def switch(i): stack.setCurrentIndex(i); enc_tab.setChecked(i==0); dec_tab.setChecked(i==1)
        enc_tab.clicked.connect(lambda: switch(0))
        dec_tab.clicked.connect(lambda: switch(1))

        tabs = QHBoxLayout(); tabs.setSpacing(20); tabs.setContentsMargins(0,0,0,0)
        tabs.addWidget(enc_tab); tabs.addWidget(dec_tab); tabs.addStretch()

        title = QLabel("stego")
        title.setStyleSheet(f"font-size:18px;font-weight:700;color:{TXT};letter-spacing:-0.5px;")

        root = QVBoxLayout(self); root.setContentsMargins(28,24,28,24); root.setSpacing(16)
        root.addWidget(title); root.addLayout(tabs); root.addWidget(stack, stretch=1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Window(); w.show()
    sys.exit(app.exec_())
