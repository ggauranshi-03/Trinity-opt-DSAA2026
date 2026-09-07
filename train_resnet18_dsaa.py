import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time
import wandb
import argparse
from datasets_lt import get_dataloaders, PerClassEvaluator
from trinity_optimizer import Trinity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--imb_factor', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--output_dir', type=str, default='results_dsaa2026')
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"resnet18_{args.dataset}_imb{args.imb_factor}.csv")
    with open(csv_path, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,head_acc,med_acc,tail_acc,time_s\n")

    trainloader, testloader, img_num_list, num_classes = get_dataloaders(args.dataset, args.imb_factor)
    evaluator = PerClassEvaluator(num_classes, img_num_list)

    model = models.resnet18(weights=None, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = Trinity(model, lr=1e-3, zo_scale=0.1, kfac_interval=10, damping=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run = wandb.init(project="Trinity-Benchmark", name=f"ResNet18_{args.dataset}_imb{args.imb_factor}", config=args)

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
            optimizer.step(closure)

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
        
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%, Head: {val_metrics.get('many_shot_acc', 0):.2f}%, Med: {val_metrics.get('medium_shot_acc', 0):.2f}%, Tail: {val_metrics.get('few_shot_acc', 0):.2f}%")
        
        with open(csv_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f},{val_metrics.get('many_shot_acc', 0):.2f},{val_metrics.get('medium_shot_acc', 0):.2f},{val_metrics.get('few_shot_acc', 0):.2f},{epoch_time:.1f}\n")
            
        wandb.log({
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'val/head_acc': val_metrics.get('many_shot_acc', 0),
            'val/med_acc': val_metrics.get('medium_shot_acc', 0),
            'val/tail_acc': val_metrics.get('few_shot_acc', 0),
        })

    wandb.finish()

if __name__ == '__main__':
    main()
