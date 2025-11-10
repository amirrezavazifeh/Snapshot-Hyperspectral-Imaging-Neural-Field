# Snapshot Hyperspectral Imaging via Compressive Sensing and Implicit Neural Representation

This repository contains the implementation of our method for **snapshot hyperspectral image reconstruction** using **implicit neural representations (INRs)**, presented in:

> **J. Boondicharern, A. Vazifeh, and J. W. Fleischer**,  
> *Snapshot Hyperspectral Imaging via Compressive Sensing and Implicit Neural Representation,*  
> *Computational Optical Sensing and Imaging (COSI), Optica Imaging Congress 2025*  

---

## 🧭 Overview

**Hyperspectral Imaging (HSI)** captures images across many narrow and contiguous spectral bands.  
Unlike RGB imaging, which captures only three color channels, HSI provides a rich spectral signature at every pixel, enabling powerful applications in **remote sensing**, **environmental monitoring**, **precision agriculture**, and **medical diagnostics**.

However, conventional HSI systems face a fundamental **trade-off between spatial, spectral, and temporal resolution**:

- **Spatial Scanning (push-broom systems):** captures one line of the scene at a time, requiring motion of the object or sensor.  
- **Spectral Scanning (filter-based systems):** sequentially captures one wavelength at a time, leading to long acquisition times and temporal misalignment.  
- **Snapshot Imaging:** captures all wavelengths in a single exposure using multiplexed optics or coded apertures — but produces **sparse, spectrally mixed measurements**.

The last category—**snapshot hyperspectral imaging**—offers high-speed acquisition but poses a major reconstruction challenge.  
Each pixel typically measures only one wavelength, making the recovery of the full 3D hyperspectral cube from a 2D snapshot an **ill-posed inverse problem**.  

---

## ⚙️ Method

In our approach, we address this challenge by combining **compressive sensing** with **implicit neural representations**, using a **sinusoidal activation network (SIREN)** to model the hyperspectral cube as a continuous function of spatial–spectral coordinates.

This allows us to:

- **Reconstruct** full hyperspectral cubes from a single 2D coded measurement.  
- **Perform both interpolation and extrapolation** across spectral dimensions.  
- **Achieve high-quality reconstructions** without external supervision or prior spectral correlations.

### ✳️ Core Idea

Each snapshot image is treated as a set of **sparse spectral observations**, where only a fraction of pixels correspond to each wavelength due to coded apertures or filter mosaics.  
We model the true hyperspectral signal as a **continuous function**:

\[
f_\theta(x, y, \lambda) \rightarrow I(x, y, \lambda)
\]

where \((x, y, \lambda)\) are spatial–spectral coordinates, and \(f_\theta\) is a **SIREN network** (sinusoidal MLP).  
The sinusoidal activation captures high-frequency spatial and spectral structures that ReLU-based networks typically miss.

The model is trained to minimize a **mean squared reconstruction loss** between predicted intensities and the sparse coded measurements.  
Once optimized, \(f_\theta\) becomes a **fully continuous hyperspectral field**, from which any wavelength or combination can be queried and visualized at arbitrary resolution.

<p align="center">
  <img src="Code/Method.png" alt="Method Overview" width="700">
  <br>
  <em>Figure: Overview of the proposed snapshot hyperspectral reconstruction framework using implicit neural representation.</em>
</p>

---

## 🎥 Results

We demonstrate robust hyperspectral reconstruction across several benchmark and real-world datasets.  
The method produces smooth, high-fidelity reconstructions from heavily undersampled single-shot measurements.

*(Visualization or GIF of reconstruction results will be added here.)*

---

## 🧩 Key Features

- ✅ **Single-shot reconstruction** from sparse coded measurements  
- ✅ **Fully unsupervised** optimization (no pretraining or priors)  
- ✅ **Continuous spatial–spectral representation** via INRs  
- ✅ **SIREN activations** for fine detail and spectral smoothness  
- ✅ **Sensor-agnostic design**, compatible with various aperture codes

---

## 📚 Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{boondicharern2025snapshot,
  title     = {Snapshot Hyperspectral Imaging via Compressive Sensing and Implicit Neural Representation},
  author    = {Jatearoon Boondicharern and Amir Reza Vazifeh and Jason W. Fleischer},
  booktitle = {Computational Optical Sensing and Imaging (COSI), Optica Imaging Congress 2025},
  series    = {Technical Digest Series},
  publisher = {Optica Publishing Group},
  year      = {2025},
  paper     = {CTu1B.5}
}
