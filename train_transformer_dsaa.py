import os
import time
import argparse

import torch
import torch.nn as nn
import wandb

from datasets_lt import get_dataloaders, PerClassEvaluator
from optimizer_factory import load_config, build_optimizer, OPTIMIZER_CHOICES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CIFAR100MTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.patch_size = 4
        self.embed_dim = 768

        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 64 + 1, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=12,
            dim_feedforward=3072,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)

        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.head(x)


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
        f"transformer_{args.optimizer}_{ds_cfg['name']}_imb{ds_cfg['imb_factor']}.csv")
    with open(csv_path, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,head_acc,med_acc,tail_acc,time_s\n")

    trainloader, testloader, img_num_list, num_classes = get_dataloaders(
        ds_cfg['name'], ds_cfg['imb_factor'], batch_size=ds_cfg['batch_size'],
        num_workers=ds_cfg['num_workers'])
    evaluator = PerClassEvaluator(num_classes, img_num_list)

    model = CIFAR100MTransformer(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer, scheduler, needs_closure = build_optimizer(args.optimizer, model, cfg, epochs)

    run = wandb.init(project="Trinity-Benchmark",
                      name=f"ViT_{args.optimizer}_{ds_cfg['name']}_imb{ds_cfg['imb_factor']}",
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

        print(f"[{args.optimizer}] Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%, Head: {val_metrics.get('many_shot_acc', 0):.2f}%, "
              f"Med: {val_metrics.get('medium_shot_acc', 0):.2f}%, "
              f"Tail: {val_metrics.get('few_shot_acc', 0):.2f}%")

        with open(csv_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f},"
                    f"{val_metrics.get('many_shot_acc', 0):.2f},"
                    f"{val_metrics.get('medium_shot_acc', 0):.2f},"
                    f"{val_metrics.get('few_shot_acc', 0):.2f},{epoch_time:.1f}\n")

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
