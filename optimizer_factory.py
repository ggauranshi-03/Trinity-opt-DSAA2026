import yaml
import torch
from trinity_optimizer import Trinity

OPTIMIZER_CHOICES = ['adam', 'rmsprop', 'sgd', 'trinity']


def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_optimizer(name, model, cfg, epochs, model_name=None):
    """Build an optimizer + CosineAnnealingLR scheduler from config.yaml settings.

    Returns (optimizer, scheduler, needs_closure) where `needs_closure` tells the
    training loop whether optimizer.step() must be called with a closure (Trinity's
    ZO sensor needs to recompute the loss; the standard baselines do not).

    `model_name` (e.g. 'transformer') selects a per-architecture override block
    under `optimizers.<name>.overrides.<model_name>` in config.yaml, applied on top
    of that optimizer's shared hyperparameters. See config.yaml's trinity.overrides
    for why the transformer needs different sensor/GC settings than ResNet-18.
    """
    name = name.lower()
    if name not in cfg['optimizers']:
        raise ValueError(f"No hyperparameters for optimizer '{name}' in config.yaml "
                          f"(available: {list(cfg['optimizers'].keys())})")
    p = dict(cfg['optimizers'][name])
    overrides = p.pop('overrides', {}) or {}
    if model_name and model_name in overrides:
        p.update(overrides[model_name])

    if name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=p['lr'], betas=tuple(p['betas']),
            eps=p['eps'], weight_decay=p['weight_decay'])
        needs_closure = False

    elif name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=p['lr'], alpha=p['alpha'], eps=p['eps'],
            momentum=p.get('momentum', 0.0), weight_decay=p['weight_decay'])
        needs_closure = False

    elif name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=p['lr'], momentum=p.get('momentum', 0.9),
            nesterov=p.get('nesterov', True), weight_decay=p['weight_decay'])
        needs_closure = False

    elif name == 'trinity':
        optimizer = Trinity(
            model, lr=p['lr'], betas=tuple(p.get('betas', (0.9, 0.999))),
            eps=p.get('eps', 1e-8), weight_decay=p.get('weight_decay', 1e-4),
            damping=p.get('damping', 1e-2),
            kfac_stat_interval=p.get('kfac_stat_interval', 10),
            kfac_inv_interval=p.get('kfac_inv_interval', 100),
            kfac_decay=p.get('kfac_decay', 0.95),
            max_kfac_dim=p.get('max_kfac_dim', 4800),
            sensor_interval=p.get('sensor_interval', 100),
            sensor_probes=p.get('sensor_probes', 4),
            sensor_nblocks=p.get('sensor_nblocks', 2),
            sensor_eps=p.get('sensor_eps', 1e-2),
            ema_alpha=p.get('ema_alpha', 0.7),
            rho_lo=p.get('rho_lo', 0.15), rho_hi=p.get('rho_hi', 0.35),
            dwell_time=p.get('dwell_time', 500), beta_max=p.get('beta_max', 0.3),
            grad_ema_alpha=p.get('grad_ema_alpha', 0.99),
            escape_nu_thr=p.get('escape_nu_thr', 0.75),
            escape_gamma_g=p.get('escape_gamma_g', 0.3),
            escape_r=p.get('escape_r', 1e-3), escape_lock=p.get('escape_lock', 200),
            use_gc=p.get('use_gc', True), use_so=p.get('use_so', True),
            use_zo=p.get('use_zo', True), use_escape=p.get('use_escape', True),
            use_grafting=p.get('use_grafting', True),
            force_so_always=p.get('force_so_always', False))
        needs_closure = True

    else:
        raise ValueError(f"Unknown optimizer '{name}'")

    warmup_epochs = cfg.get('training', {}).get('warmup_epochs', 0)
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler, needs_closure
