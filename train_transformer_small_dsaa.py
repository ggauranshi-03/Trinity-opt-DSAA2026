import os
import time
import argparse

import torch
import torch.nn as nn
import wandb

from datasets_lt import get_dataloaders, PerClassEvaluator
from optimizer_factory import load_config, build_optimizer, OPTIMIZER_CHOICES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Explicit linear modules so Trinity discovers and preconditions Q, K, V, and Out
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, dim_feedforward=1536):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiheadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

    def forward(self, x):
        # norm_first (Pre-LN)
        x = x + self.self_attn(self.norm1(x))
        x = x + self.linear2(self.activation(self.linear1(self.norm2(x))))
        return x


class CIFAR100MTransformerSmall(nn.Module):
    """Same architecture family as CIFAR100MTransformer (train_transformer_dsaa.py),
    shrunk from depth=12/embed_dim=768 (74 Trinity blocks) to depth=6/embed_dim=384
    (38 blocks: patch_embed + 6*(q,k,v,out,linear1,linear2) + head).

    Why: the 74-block version made Trinity's ZO sensor too slow to cover every block
    (sensor_nblocks/sensor_interval were tuned for ResNet-18's ~18-20 blocks), so
    K-FAC never turned on within a normal training budget. Halving the depth and
    embed_dim roughly halves the block count, closing most of that gap while
    keeping this a genuine transformer (attention + MLP blocks, LayerNorm, no
    convolutional inductive bias) for an architectural-generality comparison.
    """
    def __init__(self, num_classes=10, depth=6, embed_dim=384, num_heads=6, dim_feedforward=1536):
        super().__init__()
        self.patch_size = 4
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 64 + 1, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=self.embed_dim, num_heads=num_heads, dim_feedforward=dim_feedforward)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='trinity', choices=OPTIMIZER_CHOICES)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output_dir', type=str, default='results_dsaa2026')
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=1536)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg, tr_cfg = cfg['dataset'], cfg['training']
    epochs = tr_cfg['epochs']

    torch.manual_seed(tr_cfg.get('seed', 42))

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(
        args.output_dir,
        f"transformer_small_{args.optimizer}_{ds_cfg['name']}_imb{ds_cfg['imb_factor']}.csv")
    with open(csv_path, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,head_acc,med_acc,tail_acc,time_s\n")

    trainloader, testloader, img_num_list, num_classes = get_dataloaders(
        ds_cfg['name'], ds_cfg['imb_factor'], batch_size=ds_cfg['batch_size'],
        num_workers=ds_cfg['num_workers'])
    evaluator = PerClassEvaluator(num_classes, img_num_list)

    model = CIFAR100MTransformerSmall(
        num_classes=num_classes, depth=args.depth, embed_dim=args.embed_dim,
        num_heads=args.num_heads, dim_feedforward=args.dim_feedforward).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer, scheduler, needs_closure = build_optimizer(args.optimizer, model, cfg, epochs,
                                                           model_name='transformer_small')

    run = wandb.init(project=f"{cfg['wandb']['project']}-transformer-small",
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
