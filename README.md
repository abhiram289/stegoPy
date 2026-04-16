# StegoPy — LSB Steganography with Linear Algebra Analysis

A Python-based application that hides secret messages inside images using Least Significant Bit (LSB) steganography, combined with a Linear Algebra analysis pipeline to study image structure and embedding capacity.


## Features

- Hide secret messages inside images (PNG format)
- Extract hidden messages from stego images
- Linear Algebra analysis pipeline:
  - Matrix representation of image data
  - RREF (Row Reduction)
  - Singular Value Decomposition (SVD)
  - Projection and dimensionality reduction
  - Least Squares estimation
- GUI built using PyQt5
- Multithreaded execution for responsive UI


## Concept Overview

### LSB Steganography

Each pixel contains three channels (Red, Green, Blue). The least significant bit of each channel is modified to store message bits. This results in a change of at most 1 in pixel value, which is not perceptible to the human eye.


The pipeline analyzes the image as a matrix to:

- Determine embedding capacity
- Understand structure and redundancy
- Analyze variance and correlations


### Encoding Process

1. Load an image
2. Enter the secret message
3. Click "Hide Message"
4. Save the output as a PNG file


### Decoding Process

1. Load the stego image
2. Click "Extract Message"
3. The hidden message will be displayed


### Working Summary

Encoding:
Message → Bits → Embedded into LSB of pixels → Stego Image

Decoding:
Extract LSB → Reconstruct Bits → Convert to Text


## Team

[Abhiram - PES1UG24AM151](https://github.com/abhiram289),
[Nishkal - PES1UG24AM181](https://github.com/Nishkal00),
[Sinan - PES1UG24AM165](https://github.com/Mohd-Sinan7)



