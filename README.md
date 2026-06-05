# Deep Image Prior for X-Ray Image Denoising

**Author:** Waris Ali — M.Phil Physics (Radiation Protection & Health Physics),
**Context:** Deep Learning for Medical Imaging

---

## Overview

This project implements the **Deep Image Prior (DIP)** technique for X-ray image denoising — a zero-shot, unsupervised method that requires **no training dataset**.

The key insight (Ulyanov et al., 2018): a randomly-initialised convolutional network, when optimised to fit a single noisy image, recovers the clean signal *before* fitting the noise — because convolutional architectures encode a strong prior toward natural image structure.

### Why X-Ray Imaging?

investigated quality control of diagnostic X-ray systems and patient radiation dose optimisation. A central challenge in X-ray imaging is the **dose–quality trade-off**: reducing radiation dose increases noise. DIP addresses this directly — it can restore image quality from low-dose (high-noise) acquisitions without requiring paired clean/noisy training data. This is precisely the clinical problem in radiation protection.

---

## Method

```
Random noise z  →  CNN (Hourglass/U-Net)  →  Reconstructed image x̂
                        ↑ optimised to fit noisy observation y
```

- **Input:** Fixed random tensor z (never updated)
- **Network:** Lightweight hourglass encoder–decoder (U-Net style)
- **Loss:** MSE between network output and noisy image
- **Early stopping:** PSNR monitored — training stopped at peak (before overfitting to noise)

This formulation is an **inverse problem**: recover x from y = x + n, where n is noise. The network architecture itself acts as the regulariser — no explicit prior or training data needed.

---

## Results

| Metric | Noisy Input | DIP Reconstruction |
|--------|------------|-------------------|
| PSNR   | ~20.5 dB   | ~29–31 dB          |
| SSIM   | ~0.40      | ~0.82–0.87         |

*(Shepp-Logan phantom, σ=25 Gaussian noise, 800 iterations)*

![Result](results/dip_xray_result.png)

# Default (800 iterations, noise σ=25)

# Custom settings
python deep_image_prior.py --iters 1000 --noise_std 30 --save_steps 100

# Results saved to results/dip_xray_result.png
## Project Structure

```
deep_image_prior_xray
├── requirements.txt       # Dependencies
├── results/
│   └── dip_xray_result.png
└── README.md



## References
1. Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2018). Deep Image Prior. *CVPR*.
2. Heckel, R., & Hand, P. (2020). Deep Decoder: Concise Image Representations from Untrained Non-convolutional Networks. *ICLR*.
3. Heckel, R., Reinhard, M. (2019). Regularising Linear Inverse Problems with Convolutional Neural Networks. *arXiv*.
