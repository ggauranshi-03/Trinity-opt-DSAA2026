import os
import time
import argparse

import torch
import torch.nn as nn
import torchvision.models as models
import wandb

from datasets_lt import get_dataloaders, PerClassEvaluator
from optimizer_factory import load_config, build_optimizer, OPTIMIZER_CHOICES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='trinity', choices=OPTIMIZER_CHOICES)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output_dir', type=str, default='results_dsaa2026')
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg, tr_cfg = cfg['dataset'], cfg['training']
    epochs = tr_cfg['epochs']

    torch.manual_seed(tr_cfg.get('seed', 42))

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(
        args.output_dir,
        f"wideresnet_{args.optimizer}_{ds_cfg['name']}_imb{ds_cfg['imb_factor']}.csv")
    with open(csv_path, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,head_acc,med_acc,tail_acc,time_s\n")

    trainloader, testloader, img_num_list, num_classes = get_dataloaders(
        ds_cfg['name'], ds_cfg['imb_factor'], batch_size=ds_cfg['batch_size'],
        num_workers=ds_cfg['num_workers'])
    evaluator = PerClassEvaluator(num_classes, img_num_list)

    model = models.wide_resnet101_2(weights=None, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer, scheduler, needs_closure = build_optimizer(args.optimizer, model, cfg, epochs)

    run = wandb.init(project=f"{cfg['wandb']['project']}-wideresnet",
                      group=f"{ds_cfg['name']}_imb{ds_cfg['imb_factor']}",
                      name=args.optimizer,
                      config={**vars(args), **ds_cfg, **tr_cfg, 'optimizer': args.optimizer})

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        epoch_start = time.time()

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=tr_cfg['grad_clip_norm'])

            if needs_closure:
                def closure(_inp=inputs, _tgt=targets):
                    with torch.no_grad():
                        return criterion(model(_inp), _tgt)
                optimizer.step(closure)
            else:
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        train_loss = total_loss / total

        # Eval
        model.eval()
        evaluator.reset()
        val_loss, v_total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                v_total += targets.size(0)
                evaluator.update(predicted, targets)

        val_metrics = evaluator.compute()
        val_acc = val_metrics['overall_acc']
        val_loss = val_loss / v_total

        scheduler.step()
        epoch_time = time.time() - epoch_start

        diag = optimizer.get_diagnostics() if hasattr(optimizer, 'get_diagnostics') else None
        diag_str = ""
        if diag is not None:
            diag_str = (f" | SO: {diag['n_so']}/{diag['n_blocks']}, "
                        f"rho: {diag['rho_mean']:.3f}, nu: {diag['nu_mean']:.3f}, "
                        f"escape: {diag['escape_fires']}, "
                        f"kfac_inv: {diag['kfac_inv_success']}/{diag['kfac_inv_success']+diag['kfac_inv_fail']}, "
                        f"graft_scale: {diag['grafting_scale_mean']:.3f}")

        print(f"[{args.optimizer}] Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%, Head: {val_metrics.get('many_shot_acc', 0):.2f}%, "
              f"Med: {val_metrics.get('medium_shot_acc', 0):.2f}%, "
              f"Tail: {val_metrics.get('few_shot_acc', 0):.2f}%{diag_str}")

        with open(csv_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f},"
                    f"{val_metrics.get('many_shot_acc', 0):.2f},"
                    f"{val_metrics.get('medium_shot_acc', 0):.2f},"
                    f"{val_metrics.get('few_shot_acc', 0):.2f},{epoch_time:.1f}\n")

        log_dict = {
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'val/head_acc': val_metrics.get('many_shot_acc', 0),
            'val/med_acc': val_metrics.get('medium_shot_acc', 0),
            'val/tail_acc': val_metrics.get('few_shot_acc', 0),
        }
        if diag is not None:
            log_dict.update({f'trinity/{k}': v for k, v in diag.items()})
        wandb.log(log_dict)

    wandb.finish()


if __name__ == '__main__':
    main()
