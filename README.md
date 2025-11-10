# Snapshot Hyperspectral Imaging via Compressive Sensing and Implicit Neural Representation

This repository contains the implementation of our method for **snapshot hyperspectral image reconstruction** using **implicit neural representations (INRs)**, presented in:

> **J. Boondicharern, A. Vazifeh, and J. W. Fleischer**,  
> *Snapshot Hyperspectral Imaging via Compressive Sensing and Implicit Neural Representation,*  
> *Optica Imaging Congress 2025 (3D, DH, COSI, IS, pcAOP, RadIT), Technical Digest Series (Optica Publishing Group, 2025), paper CTu1B.5.*

---

## 🧭 Overview

Traditional hyperspectral imaging (HSI) techniques—such as spatial or spectral scanning—trade off between spatial resolution, spectral fidelity, and acquisition speed.  
Snapshot hyperspectral systems aim to capture all spectral bands **in a single exposure**, but this results in sparse and underdetermined data.

In our approach, we address this challenge by combining **compressive sensing** with **implicit neural representations**, using a **sinusoidal activation network (SIREN)** to model the hyperspectral cube as a continuous function of spatial–spectral coordinates.

This allows us to:
- Reconstruct full hyperspectral cubes from a **single 2D coded measurement**.  
- Perform both **interpolation and extrapolation** across spectral dimensions.  
- Achieve high-quality reconstructions **without external supervision** or prior spectral correlations.

---

## ⚙️ Method

1. **Coded Aperture Encoding**  
   - The sensor captures one random spectral band per pixel using binary aperture masks.  
   - Each mask corresponds to a subset of the total pixels, effectively compressing the spectral data.

2. **Implicit Neural Representation (INR)**  
   - The hyperspectral scene is modeled as a continuous function  
     \[
     f_\theta(x, y, \lambda) \rightarrow I(x, y, \lambda)
     \]
     parameterized by a **SIREN network** (5 hidden layers, 512 neurons each).  
   - Sinusoidal activations allow the model to capture high-frequency spatial and spectral details.

3. **Optimization**  
   - The network learns directly from sparse pixel–wavelength measurements.  
   - Reconstruction loss is measured via **Mean Squared Error (MSE)** between predicted and observed pixels.

4. **Output**  
   - The trained INR implicitly represents the full hyperspectral cube.  
   - Any wavelength can be queried continuously, enabling dense reconstruction or visualization.

---

## 📊 Results

We evaluated the method across four benchmark hyperspectral datasets:

| Dataset | Dimensions | Spectral Range (nm) | Validation Loss | SSIM | PSNR (dB) |
|----------|-------------|--------------------|----------------|-------|-----------|
| PaviaU | 610×340×103 | 430–860 | 0.0031 | 0.78 | 25.1 |
| HS-SOD | 1024×768×81 | 380–720 | 0.0004 | 0.94 | 33.3 |
| CZ-hsdb | 1390×1040×10 | 420–720 | 0.0006 | 0.93 | 32.1 |
| Dermatology | 1000×1000×100 | 450–900 | 0.0001 | 0.93 | 41.9 |

*(Results section to be expanded — visual examples and code outputs will be added later.)*

---

## 🧩 Key Features

- ✅ Snapshot reconstruction from **single exposure**
- ✅ Fully **unsupervised optimization**
- ✅ Continuous **spatial–spectral modeling**
- ✅ Built on **SIREN activations** for fine detail recovery
- ✅ Robust to different spectral sampling ratios

---

## 📁 Repository Structure (suggested)

