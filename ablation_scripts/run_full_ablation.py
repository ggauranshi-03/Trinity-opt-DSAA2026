import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time
import wandb
import psutil
import os
import argparse
from datasets_lt import get_dataloaders, PerClassEvaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-export Trinity from dedicated optimizer module
from trinity_optimizer import Trinity


# --- Main Script ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='Full', choices=['FO', 'ZO', 'SO', 'FO_ZO', 'FO_SO', 'Full'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--imb_factor', type=float, default=0.01)
    parser.add_argument('--output_dir', type=str, default='results_dsaa2026')
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"ablation_{args.variant}_{args.dataset}_imb{args.imb_factor}.csv")
    with open(csv_path, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,head_acc,med_acc,tail_acc,time_s\n")

    trainloader, testloader, img_num_list, num_classes = get_dataloaders(args.dataset, args.imb_factor)
    evaluator = PerClassEvaluator(num_classes, img_num_list)

    model = models.resnet18(weights=None, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    use_fo = args.variant in ['FO', 'FO_ZO', 'FO_SO', 'Full']
    use_so = args.variant in ['SO', 'FO_SO', 'Full']
    use_zo = args.variant in ['ZO', 'FO_ZO', 'Full']

    # For SO only without Adam, we still use Adam as the base update here because KFAC in Trinity preconditions Adam. 
    # ZO only uses Adam with zeroed FO gradients.
    
    optimizer = Trinity(model, lr=1e-3, use_fo=use_fo, use_so=use_so, use_zo=use_zo)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run = wandb.init(project="Trinity-Ablation", name=f"Ablation_{args.variant}_{args.epochs}ep", config=args)

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        epoch_start = time.time()
        
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
            optimizer.step(closure if use_zo else None)

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

        print(f"[{args.variant}] Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}% (Time: {epoch_time:.1f}s){diag_str}")

        with open(csv_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f},{val_metrics.get('many_shot_acc', 0):.2f},{val_metrics.get('medium_shot_acc', 0):.2f},{val_metrics.get('few_shot_acc', 0):.2f},{epoch_time:.1f}\n")

        log_dict = {
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'val/head_acc': val_metrics.get('many_shot_acc', 0),
            'val/med_acc': val_metrics.get('medium_shot_acc', 0),
            'val/tail_acc': val_metrics.get('few_shot_acc', 0),
            'time/epoch_s': epoch_time
        }
        if diag is not None:
            log_dict.update({f'trinity/{k}': v for k, v in diag.items()})
        wandb.log(log_dict)

    wandb.finish()

if __name__ == '__main__':
    main()
