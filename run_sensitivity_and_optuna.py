import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time
import argparse
import wandb
from datasets_lt import get_dataloaders, PerClassEvaluator
from run_full_ablation import Trinity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial, args, trainloader, testloader, num_classes):
    # Hyperparameter sweep space
    if args.sweep_param == 'zo_scale':
        zo_scale = trial.suggest_categorical('zo_scale', [0.01, 0.05, 0.1, 0.2, 0.5])
        kfac_interval = 10
        damping = 1e-2
    elif args.sweep_param == 'kfac_interval':
        zo_scale = 0.1
        kfac_interval = trial.suggest_categorical('kfac_interval', [1, 5, 10, 20, 50])
        damping = 1e-2
    elif args.sweep_param == 'damping':
        zo_scale = 0.1
        kfac_interval = 10
        damping = trial.suggest_categorical('damping', [1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    else: # all
        zo_scale = trial.suggest_float('zo_scale', 0.01, 0.5)
        kfac_interval = trial.suggest_categorical('kfac_interval', [5, 10, 20])
        damping = trial.suggest_float('damping', 1e-4, 1.0, log=True)

    model = models.resnet18(weights=None, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Trinity(model, lr=1e-3, zo_scale=zo_scale, kfac_interval=kfac_interval, damping=damping)

    val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            def closure(_inp=inputs, _tgt=targets):
                with torch.no_grad():
                    return criterion(model(_inp), _tgt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(closure)
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct / total
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_param', type=str, default='all', choices=['zo_scale', 'kfac_interval', 'damping', 'all'])
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--imb_factor', type=float, default=0.01)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10) # short epochs for tuning
    args = parser.parse_args()

    trainloader, testloader, _, num_classes = get_dataloaders(args.dataset, args.imb_factor)

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
    )
    
    study.optimize(lambda t: objective(t, args, trainloader, testloader, num_classes), n_trials=args.trials)

    print(f"\n--- Best Trial for {args.sweep_param} sweep ---")
    print(f"  Value: {study.best_trial.value:.2f}%")
    print(f"  Params: {study.best_trial.params}")

    # Plotting code could go here, but optuna-dashboard is typically used to visualize

if __name__ == '__main__':
    main()
