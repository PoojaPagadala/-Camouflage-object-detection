
# Camouflaged Object Detection using SegFormer-B2

This project implements a deep learning-based solution for detecting and segmenting **camouflaged objects** in complex scenes. The model uses **SegFormer-B2**, a transformer-based semantic segmentation architecture, trained on high-quality datasets like **COD10K**. It targets real-world applications in surveillance, defense, wildlife monitoring, and medical imaging.

---

## 📘 Introduction

Camouflaged Object Detection (COD) aims to identify objects that blend into their surroundings due to similar textures, patterns, or colors. This makes traditional detection methods ineffective. COD is critical in fields like defense, search and rescue, autonomous navigation, and ecology.

This project leverages **SegFormer-B2**, a powerful vision transformer model, to enhance accuracy, generalization, and robustness in detecting camouflaged objects across diverse backgrounds.

---

## 🧠 Model Architecture

We use **SegFormer-B2**, which combines:
- **MiT-B2 (Mix Vision Transformer)** as encoder for multiscale global context
- A **lightweight MLP decoder** for efficient high-resolution output

Key features:
- Overlapping patch embedding
- Efficient self-attention
- Multiscale hierarchical representation
- Fast and memory-efficient inference

---

## 🧰 Dataset

### COD10K
- **Size**: 10,000 images (6000 train / 4000 test)
- **Categories**: 78 categories (animals, insects, nature, etc.)
- **Annotations**: Binary ground-truth masks
- **Purpose**: Model training and validation

### NC4K
- **Size**: 4,121 real-world camouflaged images
- **Purpose**: Cross-dataset evaluation for generalization testing

---

## 🧼 Preprocessing & Augmentations

- **Resize** to 512x512
- **Flip** (horizontal, p=0.5)
- **ColorJitter** (brightness, contrast, saturation)
- **Brightness & Contrast** (p=0.2)
- **Shift, Scale, Rotate** (limits: ±5%, ±10%, ±15°)
- **Normalization**: mean=0.5, std=0.5
- **ToTensorV2** for PyTorch compatibility

---

## 🏋️ Training Configuration

- **Model**: SegFormer-B2 (from NVIDIA ADE checkpoint)
- **Epochs**: 20
- **Batch Size**: 8
- **Optimizer**: AdamW
- **LR Scheduler**: CosineAnnealingLR
- **Loss**: `BCE + (1 - Tversky) + (1 - Dice)`
- **Mixed Precision**: Enabled (via `torch.cuda.amp`)
- **Hardware**: Trained on GPU (Kaggle/Colab)

---

## 📈 Evaluation Metrics

| Metric             | Train   | Val     | Test    |
|--------------------|---------|---------|---------|
| IoU                | 0.8499  | 0.7725  | 0.7747  |
| Dice Score         | 0.8893  | 0.8172  | 0.8177  |
| Precision          | 0.9536  | 0.9190  | 0.9107  |
| Recall             | 0.8813  | 0.8266  | 0.8323  |
| Fβ Score           | 0.9017  | 0.8331  | 0.8382  |
| MAE                | 0.0092  | 0.0193  | 0.0184  |
| S-measure          | 0.8696  | 0.7949  | 0.7962  |
| E-measure          | 0.9908  | 0.9807  | 0.9816  |

---

## 📊 Comparative Analysis (on COD10K)

| Method              | S     | F     | E     | M (MAE) |
|---------------------|-------|-------|-------|---------|
| SINet               | 0.769 | 0.636 | 0.806 | 0.049   |
| PF-Net              | 0.800 | 0.660 | 0.877 | 0.040   |
| ZoomNet             | 0.819 | 0.742 | 0.864 | 0.032   |
| TCU-Net             | 0.880 | 0.804 | 0.941 | 0.020   |
| **Proposed (Ours)** | **0.896** | **0.838** | **0.981** | **0.018** |

---

## 🖼 Visual Results

- Ground truth masks vs. predicted segmentation
- Successful detection of flatfish, sea creatures, camouflaged soldiers, etc.
- Visual overlay comparisons on both COD10K and NC4K samples

---

## 🧪 Key Features & Contributions

- Developed a **SegFormer-B2 pipeline** for COD
- Trained and tested on **realistic datasets**
- Achieved **state-of-the-art accuracy** in multiple metrics
- Applied **custom loss**, **attention-based fusion**, and **augmentation**
- Supports **binary segmentation** for complex camouflage patterns

---

## 🛠 Tech Stack

- **Language**: Python 3.7+
- **Framework**: PyTorch
- **Libraries**: NumPy, Pandas, OpenCV, Albumentations
- **Dev Platforms**: Kaggle, Colab, Jupyter Notebook

---

## 💻 System Requirements

- **GPU**: Recommended (NVIDIA with CUDA support)
- **RAM**: Minimum 16GB
- **Storage**: SSD with at least 256 GB
- **Software**: Python ≥ 3.7, PyTorch ≥ 1.10, torchvision, albumentations

---

## 📂 Project Structure
PoojaPagadala/
├── model/ # Trained models or model-related files
├── templates/ # HTML templates for the frontend
├── LICENSE # License file
├── README.md # Project overview and instructions
├── app.py # Flask application entry point
├── requirements.txt # Python dependencies
└── segb2-COD_cleaned.ipynb # Jupyter Notebook for data preprocessing/model training
---

## License

This project is licensed under the [MIT License](./LICENSE).  
See the `LICENSE` file for details.

---

## 👥 Authors

- Pagadala Pooja 
- Kavya Sudha Gorrepati  
- Akuthota Meghana  
- Vuyyala Likhitha   
- Vemula Teena Mounika  

**Supervisor**: Dr. Habila Basumatary  
*Indian Institute of Information Technology, Pune*

---

## 🔮 Future Scope

- Extend COD models to **medical imaging** (e.g., tumor detection)
- Integrate **thermal, IR, LiDAR** data for complex environments
- Optimize for **real-time inference on edge devices**
- Develop **interactive web apps** or **mobile deployments**

---

## 🔑 Keywords

Camouflaged Object Detection, SegFormer, MiT-B2, Semantic Segmentation, Deep Learning, Transformer, Vision Transformer, PyTorch, COD10K, NC4K, Attention Mechanism, Binary Mask

