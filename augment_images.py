import os
import uuid
from PIL import Image
from torchvision import transforms

# Categories to augment
categories = ['easy', 'hard']

# Define augmentation pipeline
augment = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomPerspective(distortion_scale=0.3, p=1.0),
    transforms.ToTensor()
])

# Save multiple versions
def save_augmented_images(image_path, output_dir, num_augments=4):
    img = Image.open(image_path).convert("RGB")
    for _ in range(num_augments):
        aug_tensor = augment(img)
        aug_img = transforms.ToPILImage()(aug_tensor)
        unique_name = f"{uuid.uuid4().hex[:6]}.png"
        aug_img.save(os.path.join(output_dir, unique_name))

# Run on all categories
for category in categories:
    source_folder = f'turn_dataset/{category}'
    target_folder = f'turn_dataset_augmented/{category}'
    os.makedirs(target_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(source_folder, filename)
            save_augmented_images(image_path, target_folder)