# 🖼️ Image Classifier

This project is a simple **image classification tool** using **PyTorch** and **Torchvision**.  
It predicts the class of a given image using a pre-trained model.

---

## 🚀 Features

- Classifies images using a pre-trained model (e.g., ResNet18 or MobileNet)
- Accepts **local image files** or **image URLs** for classification
- Uses **CPU only** (no GPU required)

---

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/ankita-sen-cyber/your-repo-name.git
cd your-repo-name
```


2. **Create a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 📦 Requirements

- Python 3.8+

- torch

- torchvision

- torchaudio

- requests (if using image URLs)

## 📷 Usage

### Classify a local image: 

```bash 
python image_classifier.py --image path_to_your_image.jpg
```
### Classify an image from a URL:

```bash
python image_classifier.py --url https://example.com/dog.jpg
```
## 🧠 Model Details

- Uses pre-trained models from torchvision.models

- Models are trained on ImageNet

- No additional training required; direct inference

## 📂 Project Structure

```bash
Project/
│
├── image_classifier.py   # Main script
├── requirements.txt      # Dependencies
├── README.md              # Project documentation
└── .venv/                 # Virtual environment (excluded from Git)
```
## ✨ Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)