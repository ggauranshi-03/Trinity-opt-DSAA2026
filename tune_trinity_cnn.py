"""Optuna hyperparameter search for Trinity on ResNet-18 / WideResNet-101-2.

Both currently use Trinity's shared config block unchanged (lr=0.001 inherited from
Adam, sensor_nblocks=2/sensor_interval=100/ema_alpha=0.7 tuned by hand for ResNet-18's
block count). Baselines (logs/trinity.log, logs/trinity_wresnet.log): Trinity edges out
Adam/RMSprop on both but loses to SGD (ResNet-18: 59.97% vs SGD's 60.86%; WideResNet:
60.95% vs SGD's 65.02%, a much bigger gap).

Block-count check (measured): ResNet-18 has 21 Trinity blocks (close to what the shared
sensor defaults were tuned for) but WideResNet-101-2 has 105 -- a 5x mismatch, the same
shape of problem that made K-FAC never engage on the 74-block transformer. So unlike
ResNet-18 (6-knob search: lr/weight_decay/damping/beta_max/rho_lo/rho_hi), WideResNet's
search also includes sensor_nblocks/sensor_interval/ema_alpha.

Usage (one worker per GPU, each writes its own sqlite file -- see
tune_trinity_transformer_small.py's comment for why: concurrent writers to one sqlite
file crash Optuna):
    CUDA_VISIBLE_DEVICES=0 python3 tune_trinity_cnn.py --model resnet18 --n_trials 5 --worker_id 0
    CUDA_VISIBLE_DEVICES=1 python3 tune_trinity_cnn.py --model wide_resnet --n_trials 5 --worker_id 1
Then: python3 merge_tune_results.py --study <name>
"""
import argparse
import time

import optuna
import torch
import torch.nn as nn
import torchvision.models as models

from datasets_lt import get_dataloaders, PerClassEvaluator
from trinity_optimizer import Trinity

MODEL_CTORS = {
    'resnet18': lambda nc: models.resnet18(weights=None, num_classes=nc),
    'wide_resnet': lambda nc: models.wide_resnet101_2(weights=None, num_classes=nc),
}
TRIAL_EPOCHS = {'resnet18': 30, 'wide_resnet': 30}


def objective(trial, model_name, device):
    torch.manual_seed(42)
    trainloader, testloader, img_num_list, num_classes = get_dataloaders(
        'cifar10', 0.01, batch_size=128, num_workers=2)
    evaluator = PerClassEvaluator(num_classes, img_num_list)
    torch.manual_seed(42)
    model = MODEL_CTORS[model_name](num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    lr = trial.suggest_float('lr', 3e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    damping = trial.suggest_float('damping', 0.005, 0.5, log=True)
    beta_max = trial.suggest_float('beta_max', 0.05, 0.5)
    rho_lo = trial.suggest_float('rho_lo', 0.05, 0.25)
    rho_hi = rho_lo + trial.suggest_float('rho_gap', 0.05, 0.3)

    kwargs = dict(
        lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay,
        damping=damping, kfac_stat_interval=10, kfac_inv_interval=100, kfac_decay=0.95,
        max_kfac_dim=4800, sensor_probes=4, rho_lo=rho_lo, rho_hi=rho_hi, dwell_time=500,
        beta_max=beta_max, grad_ema_alpha=0.99, escape_nu_thr=0.75, escape_gamma_g=0.3,
        escape_r=1e-3, escape_lock=200, use_gc=True, use_so=True, use_zo=True,
        use_escape=True, use_grafting=True, force_so_always=False)

    if model_name == 'wide_resnet':
        # 105 blocks vs ResNet-18's 21 -- same sensor-coverage problem the transformer
        # had, so these three are searched too instead of left at the ResNet-tuned
        # shared defaults (sensor_nblocks=2/sensor_interval=100/ema_alpha=0.7).
        kwargs['sensor_nblocks'] = trial.suggest_int('sensor_nblocks', 2, 12)
        kwargs['sensor_interval'] = trial.suggest_int('sensor_interval', 30, 100)
        kwargs['ema_alpha'] = trial.suggest_float('ema_alpha', 0.2, 0.7)
    else:
        kwargs['sensor_nblocks'] = 2
        kwargs['sensor_interval'] = 100
        kwargs['ema_alpha'] = 0.7

    optimizer = Trinity(model, **kwargs)
    epochs = TRIAL_EPOCHS[model_name]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    for epoch in range(epochs):
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
    ap.add_argument('--model', required=True, choices=['resnet18', 'wide_resnet'])
    ap.add_argument('--n_trials', type=int, default=5)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--worker_id', type=int, required=True)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    storage = f"sqlite:///tune_{args.model}_worker_{args.worker_id}.db"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=10)
    study = optuna.create_study(
        study_name=f'trinity_{args.model}', storage=storage, direction='maximize',
        pruner=pruner, load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=None))

    t0 = time.time()
    study.optimize(lambda t: objective(t, args.model, device), n_trials=args.n_trials)
    print(f"[{args.model}] worker {args.worker_id} finished {args.n_trials} trials in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
