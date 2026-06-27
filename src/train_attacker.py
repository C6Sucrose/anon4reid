import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


class Market1501TrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.pid_to_label = {}

        filenames = sorted([
            f for f in os.listdir(root)
            if f.endswith('.jpg') and not f.startswith('-1') and not f.startswith('0000')
        ])

        for fname in filenames:
            pid = int(fname.split('_')[0])
            if pid not in self.pid_to_label:
                self.pid_to_label[pid] = len(self.pid_to_label)
            self.samples.append((os.path.join(root, fname), self.pid_to_label[pid]))

        self.num_classes = len(self.pid_to_label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, "data", "raw", "Market-1501", "bounding_box_test")
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "id_attacker.pth")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = Market1501TrainDataset(train_dir, transform=transform)
    num_classes = dataset.num_classes
    print(f"Training set: {len(dataset)} images, {num_classes} identities")

    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Early stopping on training loss plateau
    max_epochs = 30
    patience = 4
    best_loss = float('inf')
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch:02d}/{max_epochs}  loss={epoch_loss:.4f}  acc={epoch_acc:.2f}%")

        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            stale = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved checkpoint (best loss so far)")
        else:
            stale += 1
            if stale >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nTraining complete. Best model saved to {save_path}")

    # Save training metadata
    meta = {
        "num_classes": num_classes,
        "pid_to_label": {str(k): v for k, v in dataset.pid_to_label.items()},
        "best_train_loss": best_loss,
    }
    meta_path = os.path.join(model_dir, "id_attacker_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == '__main__':
    main()
