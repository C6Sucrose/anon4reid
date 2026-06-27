import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


ANON_DATASETS = {
    "Blur": "Market-1501-Blur",
    "Edge": "Market-1501-Edge",
    "RAD": "Market-1501-RAD",
    "SDInpaint": "Market-1501-SDInpaint",
}


class Market1501QueryDataset(Dataset):
    def __init__(self, query_dir, pid_to_label, transform=None):
        self.transform = transform
        self.samples = []

        for fname in sorted(os.listdir(query_dir)):
            if not fname.endswith('.jpg'):
                continue
            pid = int(fname.split('_')[0])
            if pid == -1 or pid == 0:
                continue
            if pid not in pid_to_label:
                continue
            self.samples.append((os.path.join(query_dir, fname), pid_to_label[pid]))

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
    processed_dir = os.path.join(base_dir, "data", "processed")
    model_dir = os.path.join(base_dir, "models")
    output_dir = os.path.join(base_dir, "outputs", "results")
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "results_privacy.json")

    weights_path = os.path.join(model_dir, "id_attacker.pth")
    meta_path = os.path.join(model_dir, "id_attacker_meta.json")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Attacker weights not found at {weights_path}. Run train_attacker.py first.")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Attacker metadata not found at {meta_path}. Run train_attacker.py first.")

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    num_classes = meta["num_classes"]
    pid_to_label = {int(k): v for k, v in meta["pid_to_label"].items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load existing results to merge (Rule 3)
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    with torch.no_grad():
        for label, dataset_dirname in ANON_DATASETS.items():
            query_dir = os.path.join(processed_dir, dataset_dirname, "query")
            if not os.path.isdir(query_dir):
                print(f"SKIP: {query_dir} not found")
                continue

            print(f"\n{'='*60}")
            print(f"Privacy Evaluation: {label}")
            print(f"{'='*60}")

            dataset = Market1501QueryDataset(query_dir, pid_to_label, transform=transform)
            loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)

            correct = 0
            total = 0

            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            accuracy = (correct / total) * 100 if total > 0 else 0.0
            all_results[label] = {
                "Top-1_Accuracy": round(accuracy, 4),
                "Correct": correct,
                "Total": total,
            }
            print(f"[{label}] Top-1 Accuracy: {accuracy:.2f}% ({correct}/{total})")

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
