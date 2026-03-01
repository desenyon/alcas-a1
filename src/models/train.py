"""
ALCAS - Ensemble Training v4
Changes from v3:
  - Train on random split (80/10/10), report on temporal test
  - Target normalization to N(0,1)
  - OneCycleLR scheduler
  - hidden_dim=384, dropout=0.15
  - epochs=150, patience=25
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from models.affinity_model import AffinityModel, count_parameters
from data.dataloader import get_dataloader

CONFIG = {
    'seeds':         [42, 7, 13, 99, 2024, 314],
    'batch_size':    32,
    'grad_accum':    4,
    'num_workers':   4,
    'lr':            3e-4,
    'weight_decay':  1e-4,
    'epochs':        150,
    'patience':      25,
    'hidden_dim':    384,
    'dropout':       0.15,
    'grad_clip':     1.0,
    'huber_delta':   1.0,
}

GRAPHS_DIR = Path('data/processed/graphs')
MODELS_DIR = Path('results/models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def compute_metrics(preds, targets):
    preds   = np.array(preds,   dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    return {
        'rmse':    float(np.sqrt(np.mean((preds - targets) ** 2))),
        'mae':     float(np.mean(np.abs(preds - targets))),
        'r2':      float(r2_score(targets, preds)),
        'pearson': float(pearsonr(preds, targets)[0]),
    }


def get_target_stats(loader):
    """Compute mean and std of training targets for normalization."""
    ys = []
    for batch in loader:
        ys.extend(batch['y'].numpy())
    ys = np.array(ys)
    return float(ys.mean()), float(ys.std())


def train_one_seed(seed, train_loader, val_loader, y_mean, y_std):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    out_dir = MODELS_DIR / f'seed_{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AffinityModel(
        hidden_dim = CONFIG['hidden_dim'],
        dropout    = CONFIG['dropout'],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'],
    )

    steps_per_epoch = max(1, len(train_loader) // CONFIG['grad_accum'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = CONFIG['lr'],
        epochs          = CONFIG['epochs'],
        steps_per_epoch = steps_per_epoch,
        pct_start       = 0.1,
        anneal_strategy = 'cos',
    )

    criterion = nn.HuberLoss(delta=CONFIG['huber_delta'])
    scaler    = torch.amp.GradScaler('cuda')

    y_mean_t = torch.tensor(y_mean, dtype=torch.float, device=DEVICE)
    y_std_t  = torch.tensor(y_std,  dtype=torch.float, device=DEVICE)

    best_val_r2      = -float('inf')
    patience_counter = 0
    history          = []

    print(f"\n{'='*60}")
    print(f"Seed {seed} | Params: {count_parameters(model):,}")
    print(f"y_mean={y_mean:.3f} y_std={y_std:.3f}")
    print(f"{'='*60}")

    for epoch in range(1, CONFIG['epochs'] + 1):
        t0 = time.time()

        # ---- TRAIN ----
        model.train()
        train_losses = []
        step = 0
        optimizer.zero_grad()

        for batch in train_loader:
            batch  = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
            y_norm = (batch['y'] - y_mean_t) / y_std_t

            with torch.amp.autocast('cuda'):
                preds = model(batch)
                loss  = criterion(preds, y_norm)

            scaler.scale(loss / CONFIG['grad_accum']).backward()
            train_losses.append(loss.item())
            step += 1

            if step % CONFIG['grad_accum'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        if step % CONFIG['grad_accum'] != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # ---- VALIDATE ----
        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch  = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}
                with torch.amp.autocast('cuda'):
                    preds = model(batch)
                # Denormalize
                preds_denorm = preds.float() * y_std_t + y_mean_t
                val_preds.extend(preds_denorm.cpu().numpy())
                val_targets.extend(batch['y'].cpu().numpy())

        train_loss  = float(np.mean(train_losses))
        val_metrics = compute_metrics(val_preds, val_targets)
        elapsed     = time.time() - t0
        lr_now      = optimizer.param_groups[0]['lr']

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'lr': lr_now,
            **{f'val_{k}': v for k, v in val_metrics.items()},
        })

        print(
            f"Ep {epoch:3d} | loss={train_loss:.4f} | "
            f"R²={val_metrics['r2']:.4f} | r={val_metrics['pearson']:.4f} | "
            f"RMSE={val_metrics['rmse']:.4f} | lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        if val_metrics['r2'] > best_val_r2:
            best_val_r2      = val_metrics['r2']
            patience_counter = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_metrics': val_metrics,
                'y_mean':      y_mean,
                'y_std':       y_std,
                'config':      CONFIG,
                'seed':        seed,
            }, out_dir / 'best_model.pt')
            print(f"  -> Saved (R²={val_metrics['r2']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"Early stopping at epoch {epoch}")
                break

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(out_dir / 'best_model.pt', weights_only=False)
    m    = ckpt['val_metrics']
    print(f"\nSeed {seed} best: R²={m['r2']:.4f} | r={m['pearson']:.4f} | RMSE={m['rmse']:.4f}")
    return m


def evaluate_ensemble(loader, label='test'):
    models = []
    for seed in CONFIG['seeds']:
        ckpt_path = MODELS_DIR / f'seed_{seed}' / 'best_model.pt'
        if not ckpt_path.exists():
            continue
        ckpt  = torch.load(ckpt_path, weights_only=False)
        model = AffinityModel(
            hidden_dim=CONFIG['hidden_dim'], dropout=CONFIG['dropout'],
        ).to(DEVICE)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        y_mean = ckpt['y_mean']
        y_std  = ckpt['y_std']
        models.append((model, y_mean, y_std))

    print(f"\nEnsemble of {len(models)} models on {label} set...")

    all_preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            seed_preds = []
            for model, ym, ys in models:
                with torch.amp.autocast('cuda'):
                    p = model(batch).float()
                seed_preds.append((p * ys + ym).cpu())
            all_preds.append(torch.stack(seed_preds, dim=0))
            targets.extend(batch['y'].cpu().numpy())

    all_preds = torch.cat(all_preds, dim=1).numpy()
    targets   = np.array(targets)
    mean_pred = all_preds.mean(axis=0)
    std_pred  = all_preds.std(axis=0)
    metrics   = compute_metrics(mean_pred, targets)

    print(f"\nEnsemble {label} metrics:")
    print(f"  R²:           {metrics['r2']:.4f}")
    print(f"  Pearson r:    {metrics['pearson']:.4f}")
    print(f"  RMSE:         {metrics['rmse']:.4f} pKd")
    print(f"  MAE:          {metrics['mae']:.4f} pKd")
    print(f"  Uncertainty:  {std_pred.mean():.4f} pKd")

    np.save(MODELS_DIR / f'ensemble_{label}_preds.npy',   mean_pred)
    np.save(MODELS_DIR / f'ensemble_{label}_std.npy',     std_pred)
    np.save(MODELS_DIR / f'ensemble_{label}_targets.npy', targets)
    with open(MODELS_DIR / f'ensemble_{label}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == '__main__':
    print("=" * 60)
    print("ALCAS Ensemble Training v4")
    print(f"Seeds: {CONFIG['seeds']}")
    print(f"Batch: {CONFIG['batch_size']} x {CONFIG['grad_accum']} = {CONFIG['batch_size']*CONFIG['grad_accum']} effective")
    print(f"hidden_dim={CONFIG['hidden_dim']} dropout={CONFIG['dropout']}")
    print("=" * 60)

    # Train on RANDOM split
    train_loader = get_dataloader(
        str(GRAPHS_DIR / 'train.pkl'),
        batch_size=CONFIG['batch_size'], shuffle=True,
        num_workers=CONFIG['num_workers'],
    )
    val_loader = get_dataloader(
        str(GRAPHS_DIR / 'val.pkl'),
        batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'],
    )

    # Temporal test for final scientific reporting
    temporal_test_loader = get_dataloader(
        str(GRAPHS_DIR / 'temporal/test.pkl'),
        batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'],
    )
    random_test_loader = get_dataloader(
        str(GRAPHS_DIR / 'test.pkl'),
        batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'],
    )

    # Compute normalization stats from training set only
    print("\nComputing target normalization stats...")
    y_mean, y_std = get_target_stats(train_loader)
    print(f"y_mean={y_mean:.4f} y_std={y_std:.4f}")
    with open(MODELS_DIR / 'target_stats.json', 'w') as f:
        json.dump({'y_mean': y_mean, 'y_std': y_std}, f)

    # Train all seeds
    all_metrics = {}
    for seed in CONFIG['seeds']:
        all_metrics[seed] = train_one_seed(seed, train_loader, val_loader, y_mean, y_std)

    print(f"\n{'='*60}")
    print("PER-SEED SUMMARY")
    print(f"{'='*60}")
    r2s = [m['r2'] for m in all_metrics.values()]
    rs  = [m['pearson'] for m in all_metrics.values()]
    for seed, m in all_metrics.items():
        print(f"  Seed {seed:4d}: R²={m['r2']:.4f} | r={m['pearson']:.4f} | RMSE={m['rmse']:.4f}")
    print(f"\n  Mean R²:      {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")
    print(f"  Mean Pearson: {np.mean(rs):.4f} +/- {np.std(rs):.4f}")

    # Evaluate ensemble on both test sets
    evaluate_ensemble(random_test_loader,   label='random_test')
    evaluate_ensemble(temporal_test_loader, label='temporal_test')

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")