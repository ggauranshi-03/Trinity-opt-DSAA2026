"""Optuna hyperparameter search for Trinity on the small transformer.

Why this exists: manual one-variable-at-a-time tuning (GC off, sensor speed-up,
then damping) fixed the "K-FAC never turns on" and "K-FAC stalls" failure modes,
but the resulting config still trains slower AND generalizes worse than plain
Adam/SGD on train_transformer_small_dsaa.py (see trinity_transformer_new.log vs
adam_transformer_new.log / sgd_transformer_new.log). Trinity has ~10 real degrees
of freedom and only one (damping) has had anything like a controlled search, while
lr/weight_decay/beta_max were just inherited from Adam's config. This runs a joint
search over the knobs most likely to matter, using validation accuracy (not train
loss -- train loss was actively misleading last time: lower train loss correlated
with WORSE val acc) as the objective.

Usage (one worker per GPU, sharing one Optuna study via the sqlite file):
    CUDA_VISIBLE_DEVICES=0 python3 tune_trinity_transformer_small.py --n_trials 5 --gpu 0
    CUDA_VISIBLE_DEVICES=1 python3 tune_trinity_transformer_small.py --n_trials 5 --gpu 1
    ...
"""
import argparse
import time

import optuna
import torch
import torch.nn as nn

from datasets_lt import get_dataloaders, PerClassEvaluator
from train_transformer_small_dsaa import CIFAR100MTransformerSmall
from trinity_optimizer import Trinity

STUDY_NAME = "trinity_transformer_small"
TRIAL_EPOCHS = 25   # enough for SO to engage (~epoch 6-10) and the val-acc trend to separate


def objective(trial, device):
    torch.manual_seed(42)
    trainloader, testloader, img_num_list, num_classes = get_dataloaders(
        'cifar10', 0.01, batch_size=128, num_workers=2)
    evaluator = PerClassEvaluator(num_classes, img_num_list)
    torch.manual_seed(42)
    model = CIFAR100MTransformerSmall(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    lr = trial.suggest_float('lr', 3e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    damping = trial.suggest_float('damping', 0.05, 1.0, log=True)
    beta_max = trial.suggest_float('beta_max', 0.05, 0.5)
    rho_lo = trial.suggest_float('rho_lo', 0.05, 0.25)
    rho_hi = rho_lo + trial.suggest_float('rho_gap', 0.05, 0.3)

    optimizer = Trinity(
        model, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay,
        damping=damping, kfac_stat_interval=10, kfac_inv_interval=100, kfac_decay=0.95,
        max_kfac_dim=4800, sensor_interval=75, sensor_probes=4, sensor_nblocks=4,
        sensor_eps=0.01, ema_alpha=0.4, rho_lo=rho_lo, rho_hi=rho_hi, dwell_time=500,
        beta_max=beta_max, grad_ema_alpha=0.99, escape_nu_thr=0.75, escape_gamma_g=0.3,
        escape_r=1e-3, escape_lock=200, use_gc=False, use_so=True, use_zo=True,
        use_escape=True, use_grafting=True, force_so_always=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRIAL_EPOCHS)

    best_val_acc = 0.0
    for epoch in range(TRIAL_EPOCHS):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            def closure(_i=inputs, _t=targets):
                with torch.no_grad():
                    return criterion(model(_i), _t)
            optimizer.step(closure)
        scheduler.step()

        model.eval()
        evaluator.reset()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                out = model(inputs)
                _, pred = out.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
                evaluator.update(pred, targets)
        val_acc = 100. * correct / total
        best_val_acc = max(best_val_acc, val_acc)

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    diag = optimizer.get_diagnostics()
    trial.set_user_attr('n_so', diag['n_so'])
    trial.set_user_attr('n_blocks', diag['n_blocks'])
    trial.set_user_attr('final_val_acc', val_acc)
    return best_val_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_trials', type=int, default=5)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--worker_id', type=int, default=0,
                     help="Each worker gets its own sqlite file to avoid the "
                          "concurrent-write crash multiple processes hitting one "
                          ".db file causes; merge_studies.py combines them after.")
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    storage = f"sqlite:///tune_worker_{args.worker_id}.db"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=8)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=storage, direction='maximize',
        pruner=pruner, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=None))

    t0 = time.time()
    study.optimize(lambda t: objective(t, device), n_trials=args.n_trials)
    print(f"worker on gpu {args.gpu} finished {args.n_trials} trials in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
