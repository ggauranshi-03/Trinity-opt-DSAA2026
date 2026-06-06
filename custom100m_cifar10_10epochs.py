import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import optuna
import time
import matplotlib
import wandb
import psutil, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────── Data ──────────────────────────────────────────
class LongTailedCIFAR10(datasets.CIFAR10):
    def __init__(self, imb_factor=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.train:
            img_num_list = self._get_img_num_per_cls(self.targets, imb_factor)
            self._gen_imbalanced_data(img_num_list)

    def _get_img_num_per_cls(self, targets, imb_factor):
        num_cls = len(set(targets))
        img_max = len(targets) / num_cls
        img_num_per_cls = []
        for cls_idx in range(num_cls):
            num = img_max * (imb_factor**(cls_idx / (num_cls - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def _gen_imbalanced_data(self, img_num_per_cls):
        new_data, new_targets = [], []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
            
        self.data = np.vstack(new_data)
        self.targets = new_targets
        print(f"[*] Generated Long-Tailed CIFAR-10. Total train samples: {len(self.targets)}")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Imbalance factor: 0.01 = 100:1 ratio between majority and minority classes
trainset    = LongTailedCIFAR10(imb_factor=0.01, root='./data', train=True,  download=True, transform=transform_train)
testset     = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True,  num_workers=2, pin_memory=True)
testloader  = DataLoader(testset,  batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
# ─────────────────────────────── Model ─────────────────────────────────────────
class CIFAR100MTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # CIFAR is 32x32. Using 4x4 patches gives us 64 total sequence tokens.
        self.patch_size = 4
        self.embed_dim = 768  # Standard dim for ~100M parameter transformers

        # 1. Linear Patch Projection
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        # 2. Learnable Class Token & Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 64 + 1, self.embed_dim))

        # 3. Transformer Backbone (~85M params: 12 layers, 12 heads, 3072 FFN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=12,
            dim_feedforward=3072,
            activation="gelu",
            batch_first=True,
            norm_first=True # Pre-LN is critical for stabilizing deep transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        
        # 4. Classification Head
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        
        # Extract patches: (B, 3, 32, 32) -> (B, 768, 8, 8) -> (B, 768, 64) -> (B, 64, 768)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Prepend the CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed

        # Pass through Transformer layers
        x = self.transformer(x)
        
        # Extract the CLS token for the final classification layer
        x = self.norm(x[:, 0])
        return self.head(x)

def get_model(num_classes=10):
    # Initializes the ~86M parameter Transformer
    model = CIFAR100MTransformer(num_classes=num_classes)
    return model.to(device)

# ─────────────────────────────── Hybrid ZFO Optimizer ──────────────────────────
class Trinity(torch.optim.Optimizer):
    """
    Hybrid ZFO: Adam (FO) + K-FAC preconditioning (SO) + Two-point SPSA (ZO)

    ── First-order (FO) ───────────────────────────────────────────────────────
    Adam with bias-corrected first and second moment tracking (betas, eps).
    Serves as the base update rule for ALL parameters.  Reads lr from
    param_groups so it is compatible with PyTorch LR schedulers.

    ── Second-order (SO): K-FAC ───────────────────────────────────────────────
    For every Linear and Conv2d layer the optimizer maintains two Kronecker
    factors via an EMA:
        A  ← EMA of (input_activations.T @ input_activations / N)
        G  ← EMA of (grad_outputs.T    @ grad_outputs    / N)
    Every kfac_interval steps these factors are refreshed and their damped
    inverses recomputed (pi-scaled Tikhonov damping).  The preconditioned
    gradient  G_inv @ grad_mat @ A_inv  is then fed into Adam *instead of*
    the raw gradient — no double-counting, no instability.
    Conv2d layers use im2col (nn.Unfold) to obtain the correct A matrix.
    Layers whose Kronecker-factor dimension exceeds max_kfac_dim are skipped
    and fall back to plain Adam (avoids OOM for very wide layers).

    ── Zeroth-order (ZO): two-point SPSA ─────────────────────────────────────
    Every zo_interval steps a two-point finite-difference gradient estimate
        ĝ_ZO ≈ [f(θ+εz) − f(θ-εz)] / (2ε) · z,   z ~ N(0, I)
    is added (scaled by zo_scale) to the backprop gradient *before* the Adam
    update.  This augments gradient signal in flat / noisy regions and
    provides a small exploration bonus.  A guard flag (_collecting=False)
    prevents the ZO forward passes from overwriting the KFAC activation
    buffers needed for the SO component.
    """

    def __init__(self, model,
                 lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4,
                 # K-FAC
                 damping=1e-2, kfac_interval=10, kfac_decay=0.95,
                 max_kfac_dim=2400,
                 # ZO-SPSA
                 zo_interval=20, zo_epsilon=1e-3, zo_scale=0.1):

        params   = [p for p in model.parameters() if p.requires_grad]
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        self.model        = model
        self.betas        = betas
        self.eps          = eps
        self.weight_decay = weight_decay
        self.damping      = damping
        self.kfac_interval = kfac_interval
        self.kfac_decay    = kfac_decay
        self.max_kfac_dim  = max_kfac_dim
        self.zo_interval   = zo_interval
        self.zo_epsilon    = zo_epsilon
        self.zo_scale      = zo_scale

        self._step       = 0
        self._collecting = True   # set False during ZO eval to protect KFAC buffers

        # K-FAC per-module storage
        self._m_A  = {}   # last forward activation (from fwd hook)
        self._m_G  = {}   # last grad-output        (from bwd hook)
        self._kA   = {}   # EMA A-factor
        self._kG   = {}   # EMA G-factor
        self._Ai   = {}   # damped inverse of A-factor
        self._Gi   = {}   # damped inverse of G-factor
        self._mods = {}   # id(module) → module

        self._register_hooks()

        # ids of params owned by KFAC-eligible modules (to avoid duplicate Adam updates)
        self._kfac_ids = set()
        for m in self._mods.values():
            self._kfac_ids.add(id(m.weight))
            if m.bias is not None:
                self._kfac_ids.add(id(m.bias))

    # ── Hooks ────────────────────────────────────────────────────────────────
    def _register_hooks(self):
        for m in self.model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                mid = id(m)
                self._mods[mid] = m
                m.register_forward_hook(self._fwd(mid))
                m.register_full_backward_hook(self._bwd(mid))

    def _fwd(self, mid):
        def h(mod, inp, out):
            if self._collecting:
                self._m_A[mid] = inp[0].detach()
        return h

    def _bwd(self, mid):
        def h(mod, gin, gout):
            if self._collecting:
                self._m_G[mid] = gout[0].detach()
        return h

    # ── K-FAC: factor update ─────────────────────────────────────────────────
    def _update_kfac(self):
        # --- Refresh Kronecker factors via EMA ---
        for mid, m in self._mods.items():
            A_raw = self._m_A.get(mid)
            G_raw = self._m_G.get(mid)
            if A_raw is None or G_raw is None:
                continue

            if isinstance(m, nn.Linear):
                A = A_raw.view(-1, A_raw.size(-1))           # (N, in_features)
                G = G_raw.view(-1, G_raw.size(-1))           # (N, out_features)
                if m.bias is not None:
                    A = torch.cat([A, A.new_ones(A.size(0), 1)], dim=1)

            else:  # Conv2d — im2col via nn.Unfold
                B   = A_raw.size(0)
                unf = nn.Unfold(kernel_size=m.kernel_size, padding=m.padding,
                                stride=m.stride, dilation=m.dilation).to(A_raw.device)
                Auf = unf(A_raw)                             # (B, C_in·kH·kW, L)
                A   = Auf.permute(0, 2, 1).reshape(-1, Auf.size(1))   # (B·L, C_in·kH·kW)
                if m.bias is not None:
                    A = torch.cat([A, A.new_ones(A.size(0), 1)], dim=1)
                G = (G_raw.view(B, G_raw.size(1), -1)
                          .permute(0, 2, 1)
                          .reshape(-1, G_raw.size(1)))        # (B·L, C_out)

            n  = A.size(0)
            Af = (A.t() @ A) / n
            Gf = (G.t() @ G) / n

            # Skip layers whose factor dimension would exceed the memory budget
            if Af.size(0) > self.max_kfac_dim or Gf.size(0) > self.max_kfac_dim:
                continue

            d = self.kfac_decay
            if mid not in self._kA:
                self._kA[mid] = Af;  self._kG[mid] = Gf
            else:
                self._kA[mid] = d * self._kA[mid] + (1 - d) * Af
                self._kG[mid] = d * self._kG[mid] + (1 - d) * Gf

        # --- Recompute inverses with pi-scaled Tikhonov damping ---
        for mid in list(self._kA.keys()):
            A, G = self._kA[mid], self._kG[mid]
            # pi balances the damping between the two factors
            pi = ((torch.trace(A) * G.size(0)) /
                  (torch.trace(G) * A.size(0) + 1e-8)).sqrt().clamp(0.01, 100.)
            dA = (self.damping ** 0.5) * pi
            dG = (self.damping ** 0.5) / pi
            try:
                self._Ai[mid] = torch.linalg.inv(
                    A + dA * torch.eye(A.size(0), device=A.device))
                self._Gi[mid] = torch.linalg.inv(
                    G + dG * torch.eye(G.size(0), device=G.device))
            except RuntimeError:
                pass    # keep stale inverse on numerical failure

    # ── K-FAC: precondition gradient ─────────────────────────────────────────
    def _precond(self, mid, m, gw, gb):
        """Return K-FAC-preconditioned (grad_w, grad_b); identity if no inverse yet."""
        if mid not in self._Ai:
            return gw, gb
        Ai, Gi = self._Ai[mid], self._Gi[mid]

        if isinstance(m, nn.Linear):
            if gb is not None:
                C = torch.cat([gw, gb.unsqueeze(1)], dim=1)   # (out, in+1)
                P = Gi @ C @ Ai
                return P[:, :-1], P[:, -1]
            return Gi @ gw @ Ai, None

        else:  # Conv2d
            Gw = gw.view(gw.size(0), -1)                      # (C_out, C_in·kH·kW)
            if gb is not None:
                C = torch.cat([Gw, gb.unsqueeze(1)], dim=1)
                P = Gi @ C @ Ai
                return P[:, :-1].view_as(gw), P[:, -1]
            return (Gi @ Gw @ Ai).view_as(gw), None

    # ── Adam helper ───────────────────────────────────────────────────────────
    def _adam(self, p, grad):
        s = self.state[p]
        if not s:
            s['step'] = 0
            s['m'] = torch.zeros_like(p)
            s['v'] = torch.zeros_like(p)
        s['step'] += 1
        t  = s['step']
        b1, b2 = self.betas
        lr = self.param_groups[0]['lr']    # read dynamically → LR schedulers work

        if self.weight_decay:
            grad = grad + self.weight_decay * p.data

        s['m'].mul_(b1).add_(grad, alpha=1 - b1)
        s['v'].mul_(b2).addcmul_(grad, grad, value=1 - b2)
        alpha = lr * (1 - b2 ** t) ** 0.5 / (1 - b1 ** t)
        p.data.addcdiv_(s['m'], s['v'].sqrt().add_(self.eps), value=-alpha)

    # ── Main step ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def step(self, closure=None):
        self._step += 1
        do_kfac = (self._step % self.kfac_interval == 0)
        do_zo   = (self._step % self.zo_interval   == 0)

        # ── ZO: two-point SPSA gradient augmentation ──────────────────────────
        if do_zo and closure is not None:
            params = self.param_groups[0]['params']
            saved  = [p.data.clone() for p in params]
            zs     = [torch.randn_like(p) for p in params]

            self._collecting = False       # freeze KFAC buffers during ZO eval

            for p, z in zip(params, zs):   # θ + ε·z
                p.data.add_(self.zo_epsilon * z)
            with torch.enable_grad():
                lp = float(closure())

            for p, z, s in zip(params, zs, saved):   # θ − ε·z
                p.data.copy_(s - self.zo_epsilon * z)
            with torch.enable_grad():
                lm = float(closure())

            for p, s in zip(params, saved):   # restore θ
                p.data.copy_(s)

            self._collecting = True

            # ĝ_ZO = [(f+ − f−) / 2ε] · z  (two-point: ~2× lower variance than one-point)
            scale = (lp - lm) / (2.0 * self.zo_epsilon) * self.zo_scale
            for p, z in zip(params, zs):
                if p.grad is not None:
                    # p.grad.data.add_(scale * z)
                    p.grad.data.add_(scale / (z.numel() ** 0.5) * z)

        # ── SO: refresh K-FAC factors and inverses ────────────────────────────
        if do_kfac:
            self._update_kfac()

        # ── FO + SO: preconditioned Adam for KFAC-eligible layers ─────────────
        # K-FAC preconditioning REPLACES the raw gradient (not added on top),
        # so there is no double-counting and no instability.
        for mid, m in self._mods.items():
            w, b = m.weight, m.bias
            if w.grad is None:
                continue
            gw = w.grad.data.clone()
            gb = b.grad.data.clone() if (b is not None and b.grad is not None) else None

            # Apply K-FAC preconditioning (returns identity if inverses not ready yet)
            gw, gb = self._precond(mid, m, gw, gb)

            self._adam(w, gw)
            if b is not None and gb is not None:
                self._adam(b, gb)

        # ── FO: plain Adam for remaining params (BatchNorm γ/β, etc.) ─────────
        for p in self.param_groups[0]['params']:
            if id(p) in self._kfac_ids or p.grad is None:
                continue
            self._adam(p, p.grad.data.clone())

        return None


def train_one_epoch(model, optimizer, criterion, dataloader, device,
                    epoch=0, run=None, opt_name=""):
    model.train()
    total_loss, correct, total = 0, 0, 0
    batch_times = []

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        t0 = time.time()
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()

        def closure(_inp=inputs, _tgt=targets):
            with torch.no_grad():
                return criterion(model(_inp), _tgt)

        # gradient norm BEFORE clipping (raw signal health)
        raw_grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(closure)

        batch_time = time.time() - t0
        batch_times.append(batch_time)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # ── W&B batch-level log ────────────────────────────────────────────
        if run is not None and batch_idx % 50 == 0:
            step = epoch * len(dataloader) + batch_idx
            log_dict = {
                f"{opt_name}/batch/loss":          loss.item(),
                f"{opt_name}/batch/grad_norm_raw": raw_grad_norm,
                f"{opt_name}/batch/time_s":        batch_time,
                f"{opt_name}/batch/lr":            optimizer.param_groups[0]['lr'],
                "global_step":                     step,
            }
            # ZO-specific stats
            if isinstance(optimizer, Trinity):
                log_dict[f"{opt_name}/batch/zo_step"]   = int(optimizer._step % optimizer.zo_interval == 0)
                log_dict[f"{opt_name}/batch/kfac_step"]  = int(optimizer._step % optimizer.kfac_interval == 0)
                log_dict[f"{opt_name}/batch/opt_step"]   = optimizer._step
            run.log(log_dict)

    avg_batch_time = sum(batch_times) / len(batch_times)
    return total_loss / total, 100. * correct / total, avg_batch_time

def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / total, 100. * correct / total

# ─────────────────────────────── Optimizer factory ─────────────────────────────
def make_optimizer(name, model, params):
    lr = params['lr']
    wd = params.get('weight_decay', 1e-4)
    if name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    elif name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    elif name == 'hybrid':
        return Trinity(
            model, lr=lr, weight_decay=wd,
            damping        = params.get('damping',       1e-2),
            kfac_interval  = params.get('kfac_interval', 10),
            zo_scale       = params.get('zo_scale',      0.1),
        )
    raise ValueError(f"Unknown optimizer: {name}")


# ─────────────────────────────── Hyperparameter tuning ─────────────────────────
def objective(trial, opt_name):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    params = dict(lr=lr, weight_decay=wd)

    if opt_name == 'hybrid':
        # Expose the hybrid-specific hyperparameters to Optuna as well
        params['damping']       = trial.suggest_float('damping', 1e-3, 0.1, log=True)
        params['kfac_interval'] = trial.suggest_categorical('kfac_interval', [5, 10, 20])
        params['zo_scale']      = trial.suggest_float('zo_scale', 0.05, 0.5)

    model     = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(opt_name, model, params)

    val_acc = 0.0
    for epoch in range(3):   # short search: 3 epochs per trial
        train_one_epoch(model, optimizer, criterion, trainloader, device)
        _, val_acc = evaluate(model, criterion, testloader, device)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return val_acc


def run_tuning(opt_name, n_trials=10):
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
    study.optimize(lambda t: objective(t, opt_name), n_trials=n_trials)
    return study.best_params


def run_experiment(optimizer_name, params, epochs=30, wandb_project="hybrid-zfo"):
    # ── Init W&B run ─────────────────────────────────────────────────────────
    run = wandb.init(
        project=wandb_project,
        name=f"{optimizer_name}",
        group="final_training",
        config={
            "optimizer":   optimizer_name,
            "epochs":      epochs,
            "batch_size":  128,
            "dataset":     "CIFAR-10",
            "model":       "ResNet-18",
            **params,
        },
        reinit="finish_previous",
    )

    model     = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(optimizer_name, model, params)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Log model graph + watch gradients/parameters
    wandb.watch(model, criterion, log="all", log_freq=100)

    # Count parameters
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.summary["total_params"]     = total_params
    run.summary["trainable_params"] = trainable_params

    train_losses, val_accs = [], []
    cumulative_time = 0.0
    start_wall = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        tl, train_acc, avg_batch_time = train_one_epoch(
            model, optimizer, criterion, trainloader, device,
            epoch=epoch, run=run, opt_name=optimizer_name
        )
        vl, va = evaluate(model, criterion, testloader, device)
        scheduler.step()

        epoch_time     = time.time() - epoch_start
        cumulative_time += epoch_time

        # ── GPU/CPU memory ──────────────────────────────────────────────────
        mem_log = {}
        if torch.cuda.is_available():
            mem_log["system/gpu_mem_allocated_MB"] = torch.cuda.memory_allocated() / 1e6
            mem_log["system/gpu_mem_reserved_MB"]  = torch.cuda.memory_reserved()  / 1e6
            mem_log["system/gpu_mem_peak_MB"]      = torch.cuda.max_memory_allocated() / 1e6
        mem_log["system/cpu_ram_used_GB"] = psutil.Process(os.getpid()).memory_info().rss / 1e9
        mem_log["system/cpu_percent"]     = psutil.cpu_percent()

        # ── Per-epoch W&B log ───────────────────────────────────────────────
        run.log({
            # Performance
            f"{optimizer_name}/train/loss":         tl,
            f"{optimizer_name}/train/accuracy":     train_acc,
            f"{optimizer_name}/val/loss":           vl,
            f"{optimizer_name}/val/accuracy":       va,
            # Learning rate
            f"{optimizer_name}/lr":                 optimizer.param_groups[0]['lr'],
            # Computational cost
            f"{optimizer_name}/time/epoch_s":       epoch_time,
            f"{optimizer_name}/time/cumulative_s":  cumulative_time,
            f"{optimizer_name}/time/avg_batch_s":   avg_batch_time,
            # Throughput
            f"{optimizer_name}/throughput/samples_per_s": len(trainset) / epoch_time,
            # Epoch
            "epoch": epoch + 1,
            **mem_log,
        })

        train_losses.append(tl)
        val_accs.append(va)
        print(f"  [{optimizer_name:8s}] epoch {epoch+1:3d}  "
              f"loss={tl:.4f}  val_acc={va:.2f}%  time={epoch_time:.1f}s")

    total_time = time.time() - start_wall

    # ── Run summary ──────────────────────────────────────────────────────────
    run.summary[f"best_val_acc"]   = max(val_accs)
    run.summary[f"final_val_acc"]  = val_accs[-1]
    run.summary[f"final_train_loss"] = train_losses[-1]
    run.summary[f"total_time_s"]   = total_time

    # ── Final comparison table (as W&B Table artifact) ───────────────────────
    table = wandb.Table(columns=["epoch", "train_loss", "val_accuracy"])
    for i, (l, a) in enumerate(zip(train_losses, val_accs)):
        table.add_data(i + 1, l, a)
    run.log({f"{optimizer_name}/training_curve_table": table})

    wandb.finish()
    return train_losses, val_accs, total_time

# ─────────────────────────────── Ablation classes ───────────────────────────────
class HybridNoKFAC(Trinity):
    """Ablation: ZO + FO only — K-FAC preconditioning disabled (identity preconditioner)."""
    def _precond(self, mid, m, gw, gb):
        return gw, gb   # no-op: returns raw gradient unchanged


class HybridNoZO(Trinity):
    """Ablation: FO + K-FAC only — ZO augmentation step skipped entirely."""
    @torch.no_grad()
    def step(self, closure=None):
        self._step += 1
        if self._step % self.kfac_interval == 0:
            self._update_kfac()

        for mid, m in self._mods.items():
            w, b = m.weight, m.bias
            if w.grad is None:
                continue
            gw = w.grad.data.clone()
            gb = b.grad.data.clone() if (b is not None and b.grad is not None) else None
            gw, gb = self._precond(mid, m, gw, gb)
            self._adam(w, gw)
            if b is not None and gb is not None:
                self._adam(b, gb)

        for p in self.param_groups[0]['params']:
            if id(p) in self._kfac_ids or p.grad is None:
                continue
            self._adam(p, p.grad.data.clone())
        return None


def run_ablation(epochs=20, wandb_project="hybrid-zfo"):
    base_params = dict(lr=0.001, weight_decay=1e-4,
                       damping=0.01, kfac_interval=10, zo_scale=0.1)
    configs = {
        'Hybrid (full)':     Trinity,
        'Hybrid – no K-FAC': HybridNoKFAC,
        'Hybrid – no ZO':    HybridNoZO,
    }
    results = {}

    for name, cls in configs.items():
        print(f"\n── Ablation: {name} ──")

        run = wandb.init(
            project=wandb_project,
            name=f"ablation_{name.replace(' ', '_').replace('–','no')}",
            group="ablation",
            config={"ablation_variant": name, "epochs": epochs, **base_params},
            reinit=True,
        )

        model     = get_model()
        criterion = nn.CrossEntropyLoss()
        opt       = cls(model, **base_params)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        losses, accs = [], []

        for epoch in range(epochs):
            epoch_start = time.time()
            tl, train_acc, avg_bt = train_one_epoch(
                model, opt, criterion, trainloader, device,
                epoch=epoch, run=run, opt_name=name
            )
            _, va = evaluate(model, criterion, testloader, device)
            scheduler.step()
            epoch_time = time.time() - epoch_start

            losses.append(tl)
            accs.append(va)
            print(f"  Epoch {epoch+1:3d}: val_acc={va:.2f}%")

            mem_log = {}
            if torch.cuda.is_available():
                mem_log["system/gpu_mem_allocated_MB"] = torch.cuda.memory_allocated() / 1e6

            run.log({
                f"ablation/{name}/train_loss":   tl,
                f"ablation/{name}/val_accuracy": va,
                f"ablation/{name}/epoch_time_s": epoch_time,
                "epoch": epoch + 1,
                **mem_log,
            })

        run.summary["best_val_acc"]  = max(accs)
        run.summary["final_val_acc"] = accs[-1]
        wandb.finish()
        results[name] = (losses, accs)

    return results
# ─────────────────────────────── Main ──────────────────────────────────────────
def main():
    TUNING_TRIALS   = 10
    TRAIN_EPOCHS    = 10
    WANDB_PROJECT   = "Hybrid-ZFS"   # ← set your project name

    # ── Hyperparameter tuning (unchanged, no W&B needed here) ────────────────
    print("=" * 60 + "\n  Hyperparameter Tuning\n" + "=" * 60)
    best = {}
    for name in ['hybrid', 'sgd', 'adam', 'rmsprop']:
        print(f"\n  Tuning {name} …")
        best[name] = run_tuning(name, n_trials=TUNING_TRIALS)
        print(f"  Best {name}: {best[name]}")

    # ── Full training ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\n  Final Training\n" + "=" * 60)
    results = {}
    for name in ['hybrid', 'sgd', 'adam', 'rmsprop']:
        print(f"\n── {name.upper()} ──")
        l, a, t = run_experiment(name, best[name],
                                 epochs=TRAIN_EPOCHS,
                                 wandb_project=WANDB_PROJECT)
        results[name] = dict(losses=l, accs=a, time=t)

    # ── Cross-optimizer summary run ───────────────────────────────────────────
    summary_run = wandb.init(
        project=WANDB_PROJECT,
        name="optimizer_comparison_summary",
        group="summary",
        reinit=True,
    )

    # Comparison table
    cmp_table = wandb.Table(columns=["optimizer", "best_val_acc", "final_val_acc",
                                      "final_train_loss", "total_time_s"])
    for name, d in results.items():
        cmp_table.add_data(name, max(d['accs']), d['accs'][-1],
                           d['losses'][-1], d['time'])

    # Loss & accuracy line plots as W&B custom charts
    epochs_axis = list(range(1, TRAIN_EPOCHS + 1))
    loss_table  = wandb.Table(columns=["epoch"] + list(results.keys()))
    acc_table   = wandb.Table(columns=["epoch"] + list(results.keys()))
    for i in range(TRAIN_EPOCHS):
        loss_table.add_data(i + 1, *[results[n]['losses'][i] for n in results])
        acc_table.add_data( i + 1, *[results[n]['accs'][i]   for n in results])

    summary_run.log({
        "comparison/optimizer_table":    cmp_table,
        "comparison/train_loss_curves":  loss_table,
        "comparison/val_acc_curves":     acc_table,
    })

    # Bar charts for final metrics
    summary_run.log({
        "comparison/final_val_acc_bar": wandb.plot.bar(
            wandb.Table(
                data=[[n, d['accs'][-1]]  for n, d in results.items()],
                columns=["optimizer", "val_acc"]
            ),
            "optimizer", "val_acc", title="Final Val Accuracy by Optimizer"
        ),
        "comparison/total_time_bar": wandb.plot.bar(
            wandb.Table(
                data=[[n, d['time']] for n, d in results.items()],
                columns=["optimizer", "time_s"]
            ),
            "optimizer", "time_s", title="Total Training Time (s)"
        ),
    })
    wandb.finish()

    # ── Local matplotlib plots (unchanged) ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, d in results.items():
        axes[0].plot(d['losses'], label=name)
        axes[1].plot(d['accs'],   label=name)
    for ax, title, yl in zip(axes,
            ['Training Loss', 'Validation Accuracy (%)'],
            ['Loss', 'Accuracy (%)']):
        ax.set_xlabel('Epoch');  ax.set_ylabel(yl)
        ax.set_title(title);     ax.legend()
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150)
    print("Saved: comparison.png")

    print("\n" + "=" * 60)
    for name, d in results.items():
        print(f"  {name:10s}: val_acc={d['accs'][-1]:.2f}%  time={d['time']:.0f}s")

    # ── Ablation study ────────────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\n  Ablation Studies\n" + "=" * 60)
    abl = run_ablation(epochs=20, wandb_project=WANDB_PROJECT)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, (losses, accs) in abl.items():
        axes[0].plot(losses, label=name)
        axes[1].plot(accs,   label=name)
    for ax, title, yl in zip(axes,
            ['Training Loss (Ablation)', 'Validation Accuracy (Ablation)'],
            ['Loss', 'Accuracy (%)']):
        ax.set_xlabel('Epoch');  ax.set_ylabel(yl)
        ax.set_title(title);     ax.legend()
    plt.tight_layout()
    plt.savefig('ablation.png', dpi=150)
    print("Saved: ablation.png")

if __name__ == '__main__':
    main()