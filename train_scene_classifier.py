import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

def train_binary_cnn(dataset_dir, output_model_path, num_epochs=10, batch_size=32):
    print("--- 🧠 TRAINING SCENE CLASSIFIER (ResNet18) ---")
    
    # 1. Setup Device (Uses GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Data Augmentation and Normalization
    # ResNet strictly requires 224x224 images and specific color normalization
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Flips images to make the AI smarter
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load Dataset
    if not os.path.exists(dataset_dir):
        print(f"❌ Error: Dataset folder '{dataset_dir}' not found!")
        return

    full_dataset = datasets.ImageFolder(dataset_dir, transform=data_transforms)
    class_names = full_dataset.classes
    print(f"Found classes: {class_names}")

    # 4. Split into 80% Training / 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Training on {train_size} images, Validating on {val_size} images.\n")

    # 5. Initialize Pre-trained ResNet18
    model = models.resnet18(pretrained=True)
    
    # Modify the final layer to output exactly 2 classes (Binary)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 6. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7. The Training Loop
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
                dataset_size = train_size
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
                dataset_size = val_size

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Save the best model weights
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), output_model_path)

        print()

    print(f"✅ Training complete! Best Validation Accuracy: {best_acc:.4f}")
    print(f"💾 Model saved to: {output_model_path}")

if __name__ == "__main__":
    # Point this to the folder you created in Task 1.2
    DATASET_DIR = "data/cnn_dataset" 
    
    # This is the final weight file vision_agent.py will use!
    OUTPUT_MODEL = "scene_classifier.pth" 
    
    train_binary_cnn(DATASET_DIR, OUTPUT_MODEL, num_epochs=4)