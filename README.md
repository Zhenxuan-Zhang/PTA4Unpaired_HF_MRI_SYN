# Pretext Task Adversarial Learning for Unpaired Low-field to Ultra High-field MRI Synthesis

> 🚀 A deep learning framework for unpaired high-field MRI synthesis.

![PTA-Task](fig1_miccai.png)

---

## 📖 Table of Contents
- [Abstract](#abstract)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)

---

## 🧠 Abstract
High-field MRI synthesis holds significant potential in overcoming data scarcity for downstream tasks (e.g., segmentation). Low-field MRI images often suffer from reduced signal-to-noise ratio (SNR) and spatial resolution, making high-field synthesis crucial for better clinical insights. However, ensuring anatomical accuracy and preserving fine details in synthetic high-field MRI remains challenging.

We introduce the **Pretext Task Adversarial (PTA) Learning Framework**, which includes:

- **Slice-wise Gap Perception (SGP) Network**: Aligns slice inconsistencies between low-field and high-field datasets using contrastive learning.
- **Local Structure Correction (LSC) Network**: Enhances anatomical structures by restoring locally rotated and masked images.
- **Pretext Task-Guided Adversarial Training**: Incorporates a discriminator and additional supervision to improve realism.

Extensive experiments on low-field to ultra high-field MRI synthesis demonstrate **state-of-the-art performance**, achieving **16.892 FID, 1.933 IS, and 0.324 MS-SSIM**, enabling the generation of high-quality high-field-like MRI data to enhance training datasets for downstream applications.

🔗 **GitHub Repository:** [PTA4Unpaired_HF_MRI_SYN](https://github.com/Zhenxuan-Zhang/PTA4Unpaired_HF_MRI_SYN)

---

## ⚙️ Installation
```bash
git clone https://github.com/Zhenxuan-Zhang/PTA4Unpaired_HF_MRI_SYN.git
cd PTA4Unpaired_HF_MRI_SYN
pip install -r requirements.txt
```

---

## 🚀 Usage
```python
import torch
from model import PTA_Network

# Load pre-trained model
model = PTA_Network()
model.load_state_dict(torch.load('pta_checkpoint.pth'))
model.eval()

# Perform inference
input_image = torch.randn(1, 1, 256, 256)  # Example input
output_image = model(input_image)
```

---

## 📂 Dataset
Our framework is evaluated on diverse datasets:
- **Low-field MRI datasets:** M4RAW, LISA
- **High-field MRI datasets:** fastMRI, HCP1200

More details on data preparation can be found in the [dataset documentation](dataset/README.md).

---

## 🏗 Model Architecture
The PTA learning framework consists of:
1. **SGP Network:** Mitigates inter-slice misalignment via contrastive learning.
2. **LSC Network:** Enhances fine-grained anatomical details through local corrections.
3. **CycleGAN-based Adversarial Training:** Ensures realism and structure preservation.

---

## 📊 Results
| Method  | FID ↓ | IS ↑ | MS-SSIM ↓ |
|---------|------|------|----------|
| Syn-GAN  | 171.009 | 1.131 | 0.989 |
| ESR-GAN  | 184.045 | 1.627 | 0.406 |
| CycleGAN | 61.470 | 2.068 | 0.201 |
| **PTA (Ours)** | **16.892** | **1.933** | **0.324** |

PTA achieves superior synthesis quality, maintaining both **fidelity and diversity**, outperforming existing methods in high-field MRI generation.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 📞 Contact
For inquiries or collaborations, open an issue on [GitHub](https://github.com/Zhenxuan-Zhang/PTA4Unpaired_HF_MRI_SYN/issues).
