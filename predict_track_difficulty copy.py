import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Note: Ensure the model is retrained using the binary classification dataset before using this script for prediction.
class_names = ['easy', 'hard']

class TurnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

def predict_image(image_path, model_path="models/turnnet.pt"):
    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        return

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    print("Tensor after normalization:", img_tensor)

    model = TurnNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        outputs = model(img_tensor)
        print("Model raw outputs:", outputs)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]
        print(f"Predicted Difficulty: {label.upper()}")

if __name__ == "__main__":
    image_path = input("Enter image path: ").strip()
    predict_image(image_path)
    print("Prediction complete.")
    print("Class labels:", class_names)