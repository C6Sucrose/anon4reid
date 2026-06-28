import os
import json
import torch
import numpy as np
import torch.nn.functional as F
import torchreid

from tqdm import tqdm
from torchreid.reid.data.datasets.image.market1501 import Market1501
from torchreid.reid.utils.rerank import re_ranking

ANON_DATASETS = {
    "Blur": "Market-1501-Blur",
    "Edge": "Market-1501-Edge",
    "RAD": "Market-1501-RAD",
    "SDInpaint": "Market-1501-SDInpaint",
}


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    output_dir = os.path.join(base_dir, "outputs", "results")
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "results_utility.json")

    weights_path = os.path.join(output_dir, "osnet_x1_0_market1501.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"OSNet weights not found at {weights_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load existing results to merge (append, not overwrite per Rule 3)
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for label, dataset_dirname in ANON_DATASETS.items():
        dataset_path = os.path.join(processed_dir, dataset_dirname)
        if not os.path.isdir(dataset_path):
            print(f"SKIP: {dataset_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating Utility: {label} ({dataset_dirname})")
        print(f"{'='*60}")

        reg_name = f"utility_{label.lower()}"

        # Dynamically create and register a dataset class for this variant
        cls = type(
            f"Market1501_{label}",
            (Market1501,),
            {"dataset_dir": dataset_dirname},
        )
        torchreid.reid.data.register_image_dataset(reg_name, cls)

        datamanager = torchreid.reid.data.ImageDataManager(
            root=processed_dir,
            sources=reg_name,
            targets=reg_name,
            height=256,
            width=128,
            batch_size_train=32,
            batch_size_test=100,
            transforms=['random_flip', 'random_crop'],
            workers=0,
        )

        model = torchreid.reid.models.build_model(
            name='osnet_x1_0',
            num_classes=datamanager.num_train_pids,
            loss='softmax',
            pretrained=False,
        )
        torchreid.reid.utils.torchtools.load_pretrained_weights(model, weights_path)
        model = model.to(device)

        optimizer = torchreid.reid.optim.build_optimizer(model, optim='adam', lr=0.0003)
        engine = torchreid.reid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer=optimizer, label_smooth=True
        )

        query_loader = datamanager.test_loader[reg_name]['query']
        gallery_loader = datamanager.test_loader[reg_name]['gallery']

        cmc, mAP = _evaluate_full_cmc(
            engine,
            tqdm(query_loader, desc=f"[{label}] Query"),
            tqdm(gallery_loader, desc=f"[{label}] Gallery"),
        )

        all_results[label] = {
            "mAP": float(mAP),
            "Rank-1": float(cmc[0]),
            "Rank-5": float(cmc[4]),
            "Rank-10": float(cmc[9]),
            "Rank-20": float(cmc[19]),
        }

        print(f"[{label}] mAP={mAP:.4f}  R1={cmc[0]:.4f}  R5={cmc[4]:.4f}  R10={cmc[9]:.4f}  R20={cmc[19]:.4f}")

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to {results_file}")


@torch.no_grad()
def _evaluate_full_cmc(engine, query_loader, gallery_loader, dist_metric='euclidean', normalize_feature=False):
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

    qf, q_pids, q_camids = _extract(query_loader)
    gf, g_pids, g_camids = _extract(gallery_loader)

    if normalize_feature:
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)

    q_g_dist = torchreid.reid.metrics.compute_distance_matrix(qf, gf, dist_metric).numpy()
    q_q_dist = torchreid.reid.metrics.compute_distance_matrix(qf, qf, dist_metric).numpy()
    g_g_dist = torchreid.reid.metrics.compute_distance_matrix(gf, gf, dist_metric).numpy()

    print('Applying k-reciprocal re-ranking ...')
    distmat = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    cmc, mAP = torchreid.reid.metrics.evaluate_rank(
        distmat, q_pids, g_pids, q_camids, g_camids,
        max_rank=50, use_metric_cuhk03=False,
    )

    return cmc, mAP


if __name__ == '__main__':
    main()
