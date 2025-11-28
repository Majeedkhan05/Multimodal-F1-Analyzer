import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
    data_dir = "turn_dataset_augmented"
    batch_size = 8
    epochs = 75
    learning_rate = 0.001
    model_path = "models/turnnet.pt"
    os.makedirs("models", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        # Filter out the 'medium' class
        dataset.classes = [cls for cls in dataset.classes if cls != 'medium']
        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}
        dataset.samples = [s for s in dataset.samples if os.path.basename(os.path.dirname(s[0])) in dataset.class_to_idx]
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    class_names = dataset.classes
    print("Classes:", class_names)
    print("Number of classes:", len(class_names))
    try:
        images, _ = next(iter(dataloader))
        print("Sample image batch shape:", images.shape)
    except Exception as e:
        print(f"Error fetching a batch: {e}")
        return

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

    model = TurnNet()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    print("Model architecture:")
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {correct/total:.2f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()