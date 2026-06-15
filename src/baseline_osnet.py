import os
import json
import torch
import numpy as np
import torch.nn.functional as F
import torchreid

from tqdm import tqdm
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

    # ─── 4. Evaluation with Full CMC Extraction ────────────────────────────
    optimizer = torchreid.reid.optim.build_optimizer(model, optim='adam', lr=0.0003)

    engine = torchreid.reid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        label_smooth=True
    )

    print("\nRunning baseline evaluation on original Market-1501...")

    query_loader = datamanager.test_loader['custom_market1501']['query']
    gallery_loader = datamanager.test_loader['custom_market1501']['gallery']

    cmc, mAP = _evaluate_full_cmc(
        engine,
        tqdm(query_loader, desc="Extracting Query", leave=True),
        tqdm(gallery_loader, desc="Extracting Gallery", leave=True),
        dist_metric='euclidean',
        normalize_feature=False,
    )

    # ─── 5. Results Serialization ──────────────────────────────────────────
    rank1  = float(cmc[0])   # cmc[0] = Rank-1
    rank5  = float(cmc[4])   # cmc[4] = Rank-5
    rank10 = float(cmc[9])   # cmc[9] = Rank-10
    rank20 = float(cmc[19])  # cmc[19] = Rank-20

    results = {
        "Baseline": {
            "mAP": float(mAP),
            "Rank-1": rank1,
            "Rank-5": rank5,
            "Rank-10": rank10,
            "Rank-20": rank20,
            "CMC_curve": [float(cmc[i]) for i in range(min(20, len(cmc)))]
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nBaseline evaluation complete.")
    print(f"Results saved to: {results_file}")
    print(f"mAP: {mAP:.4f} | Rank-1: {rank1:.4f} | Rank-5: {rank5:.4f} | Rank-10: {rank10:.4f} | Rank-20: {rank20:.4f}")


@torch.no_grad()
def _evaluate_full_cmc(
    engine,
    query_loader,
    gallery_loader,
    dist_metric='euclidean',
    normalize_feature=False,
):
    """
    Evaluates the model and returns the full CMC array + mAP,
    instead of just Rank-1 + mAP as engine._evaluate() does.
    Replicates the engine's internal pipeline but captures cmc[k] for all k.
    """
    def _extract(loader):
        f_, pids_, camids_ = [], [], []
        for data in loader:
            imgs, pids, camids = engine.parse_data_for_eval(data)
            if engine.use_gpu:
                imgs = imgs.cuda()
            features = engine.extract_features(imgs)
            features = features.cpu()
            f_.append(features)
            pids_.extend(pids.tolist())
            camids_.extend(camids.tolist())
        f_ = torch.cat(f_, 0)
        pids_ = np.asarray(pids_)
        camids_ = np.asarray(camids_)
        return f_, pids_, camids_

    print('Extracting features from query set ...')
    qf, q_pids, q_camids = _extract(query_loader)
    print(f'Done, obtained {qf.size(0)}-by-{qf.size(1)} matrix')

    print('Extracting features from gallery set ...')
    gf, g_pids, g_camids = _extract(gallery_loader)
    print(f'Done, obtained {gf.size(0)}-by-{gf.size(1)} matrix')

    if normalize_feature:
        print('Normalizing features with L2 norm ...')
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)

    print(f'Computing distance matrix with metric={dist_metric} ...')
    distmat = torchreid.reid.metrics.compute_distance_matrix(
        qf, gf, dist_metric
    )
    distmat = distmat.numpy()

    print('Computing CMC and mAP ...')
    cmc, mAP = torchreid.reid.metrics.evaluate_rank(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        max_rank=50,
        use_metric_cuhk03=False
    )

    print('** Results **')
    print('mAP: {:.1%}'.format(mAP))
    for r in [1, 5, 10, 20]:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

    return cmc, mAP


if __name__ == '__main__':
    main()
