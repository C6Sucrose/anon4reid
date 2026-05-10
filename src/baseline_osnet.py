import os
import json
import torch
import torchreid

from torchreid.reid.data.datasets.image.market1501 import Market1501

class CustomMarket1501(Market1501):
    dataset_dir = 'Market-1501'

torchreid.reid.data.register_image_dataset('custom_market1501', CustomMarket1501)

def main():
    # ─── 1. Paths & Configuration ──────────────────────────────────────────
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "data", "raw")
    output_dir = os.path.join(base_dir, "outputs", "results")
    
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "results_baseline.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ─── 2. Data Loading ───────────────────────────────────────────────────
    # Torchreid expects root to be the directory containing the dataset folder.
    print("Loading Market-1501 dataset...")
    
    datamanager = torchreid.reid.data.ImageDataManager(
        root=dataset_dir,
        sources='custom_market1501',
        targets='custom_market1501',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop'],
        workers=0
    )

    # ─── 3. Model Instantiation ────────────────────────────────────────────
    print("Building OSNet model...")
    model = torchreid.reid.models.build_model(
        name='osnet_x1_0',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=False # Set to false because we want market1501 weights, not ImageNet
    )
    
    # OSNet specific Market-1501 weights link. 
    # Download manually via browser if gdown fails and place in outputs/
    weights_path = os.path.join(output_dir, "osnet_x1_0_market1501.pth")
    if not os.path.exists(weights_path):
        import gdown
        print(f"Downloading pre-trained Market-1501 OSNet weights...")
        # Deep-person-reid official Google Drive ID for osnet_x1_0_market1501
        file_id = "1L0OqPzF8fkk9sXj0R_D3jYJtB-6Z4PVr" 
        try:
            gdown.download(id=file_id, output=weights_path, quiet=False)
        except Exception as e:
            print("Failed to auto-download weights. Please manually download OSNet Market1501 weights")
            print("and place the .pth file at:", weights_path)
            # Fallback for the execution so it doesn't crash entirely if it fails to download
            pass
            
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        torchreid.reid.utils.torchtools.load_pretrained_weights(model, weights_path)
    
    model = model.to(device)

    # ─── 4. Evaluation Engine (with TQDM Progress Bar) ─────────────────────
    optimizer = torchreid.reid.optim.build_optimizer(model, optim='adam', lr=0.0003)
    
    engine = torchreid.reid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        label_smooth=True
    )

    # Run testing
    from tqdm import tqdm
    
    print("\nRunning baseline evaluation on original Market-1501...")
    
    # We call engine._evaluate directly to retrieve BOTH rank1 and mAP
    # We wrap the data loaders in tqdm to provide an estimated completion time bar
    query_loader = datamanager.test_loader['custom_market1501']['query']
    gallery_loader = datamanager.test_loader['custom_market1501']['gallery']
    
    rank1, mAP = engine._evaluate(
        dataset_name='custom_market1501',
        query_loader=tqdm(query_loader, desc="Extracting Query", leave=True),
        gallery_loader=tqdm(gallery_loader, desc="Extracting Gallery", leave=True),
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False
    )
    
    # ─── 5. Results Serialization ──────────────────────────────────────────
    # Save the output metrics specifically
    results = {
        "Baseline": {
            "mAP": float(mAP),
            "Rank-1": float(rank1)
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nBaseline evaluation complete.")
    print(f"Results saved to: {results_file}")
    print(f"mAP: {mAP:.4f} | Rank-1: {rank1:.4f}")


if __name__ == '__main__':
    main()
