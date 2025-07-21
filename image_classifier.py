import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# Load Pretrained ResNet50
model = models.resnet50(pretrained=True)
model.eval()

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Load image from URL or local path
def load_image(img_path_or_url):
    if img_path_or_url.startswith('http'):
        response = requests.get(img_path_or_url)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(img_path_or_url)
    return img.convert('RGB')

# Predict
def classify(img_path_or_url):
    img = load_image(img_path_or_url)
    input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)

    # Get probabilities
    probs = torch.nn.functional.softmax(output[0], dim=0)

    # Load labels
    labels = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt').text.splitlines()

    # Print top 5
    top5 = torch.topk(probs, 5)
    for idx, val in zip(top5.indices, top5.values):
        print(f"{labels[idx]}: {val.item()*100:.2f}%")

# Example usage
if __name__ == "__main__":
    img_url = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg"  # Replace with your image path or URL
    classify(img_url)
