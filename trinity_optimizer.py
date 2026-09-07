import torch
import torch.nn as nn

class Trinity(torch.optim.Optimizer):
    """
    Trinity Optimizer: A Tri-Level Hybrid Optimizer for Deep Learning.
    
    Combines:
    1. First-Order (FO): Adaptive momentum (AdamW-style parameter update).
    2. Second-Order (SO): Kronecker-factored Approximate Curvature (K-FAC) preconditioning.
    3. Zeroth-Order (ZO): Periodic gradient-free directional perturbations to escape saddle points.
    
    Args:
        model (nn.Module): The model containing parameters to optimize.
        lr (float): Learning rate (default: 1e-3).
        betas (tuple): Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float): Term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float): Weight decay coefficient (default: 1e-4).
        damping (float): Tikhonov damping factor for K-FAC matrix inversion (default: 1e-2).
        kfac_interval (int): Frequency (in steps) of updating K-FAC curvature matrices (default: 10).
        kfac_decay (float): Running average decay factor for K-FAC covariance statistics (default: 0.95).
        max_kfac_dim (int): Maximum dimension for K-FAC factorization to limit memory overhead (default: 2400).
        zo_interval (int): Frequency (in steps) of Zeroth-Order directional exploration (default: 20).
        zo_epsilon (float): Base perturbation scale for Zeroth-Order finite differences (default: 1e-3).
        zo_scale (float): Scaling factor for Zeroth-Order gradient augmentation (default: 0.1).
        use_fo (bool): Enable first-order gradient updates (default: True).
        use_so (bool): Enable second-order K-FAC preconditioning (default: True).
        use_zo (bool): Enable zeroth-order exploration (default: True).
    """
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4,
                 damping=1e-2, kfac_interval=10, kfac_decay=0.95, max_kfac_dim=2400,
                 zo_interval=20, zo_epsilon=1e-3, zo_scale=0.1, 
                 use_fo=True, use_so=True, use_zo=True):
        params = [p for p in model.parameters() if p.requires_grad]
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        self.model = model
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.damping = damping
        self.kfac_interval = kfac_interval
        self.kfac_decay = kfac_decay
        self.max_kfac_dim = max_kfac_dim
        self.zo_interval = zo_interval
        self.zo_epsilon = zo_epsilon
        self.zo_scale = zo_scale

        self.use_fo = use_fo
        self.use_so = use_so
        self.use_zo = use_zo

        self._step = 0
        self._collecting = True

        self._m_A = {}
        self._m_G = {}
        self._kA = {}
        self._kG = {}
        self._Ai = {}
        self._Gi = {}
        self._mods = {}

        if self.use_so:
            self._register_hooks()

        self._kfac_ids = set()
        for m in self._mods.values():
            self._kfac_ids.add(id(m.weight))
            if m.bias is not None:
                self._kfac_ids.add(id(m.bias))

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

    def _update_kfac(self):
        for mid, m in self._mods.items():
            A_raw = self._m_A.get(mid)
            G_raw = self._m_G.get(mid)
            if A_raw is None or G_raw is None:
                continue

            if isinstance(m, nn.Linear):
                A = A_raw.view(-1, A_raw.size(-1))
                G = G_raw.view(-1, G_raw.size(-1))
                if m.bias is not None:
                    A = torch.cat([A, A.new_ones(A.size(0), 1)], dim=1)
            else:
                B = A_raw.size(0)
                unf = nn.Unfold(kernel_size=m.kernel_size, padding=m.padding,
                                stride=m.stride, dilation=m.dilation).to(A_raw.device)
                Auf = unf(A_raw)
                A = Auf.permute(0, 2, 1).reshape(-1, Auf.size(1))
                if m.bias is not None:
                    A = torch.cat([A, A.new_ones(A.size(0), 1)], dim=1)
                G = (G_raw.view(B, G_raw.size(1), -1).permute(0, 2, 1).reshape(-1, G_raw.size(1)))

            n = A.size(0)
            Af = (A.t() @ A) / n
            Gf = (G.t() @ G) / n

            if Af.size(0) > self.max_kfac_dim or Gf.size(0) > self.max_kfac_dim:
                continue

            d = self.kfac_decay
            if mid not in self._kA:
                self._kA[mid] = Af; self._kG[mid] = Gf
            else:
                self._kA[mid] = d * self._kA[mid] + (1 - d) * Af
                self._kG[mid] = d * self._kG[mid] + (1 - d) * Gf

        for mid in list(self._kA.keys()):
            A, G = self._kA[mid], self._kG[mid]
            trace_A = torch.trace(A).clamp(min=0)
            trace_G = torch.trace(G).clamp(min=0)
            pi = ((trace_A * G.size(0)) / (trace_G * A.size(0) + 1e-8)).sqrt().clamp(0.01, 100.)
            pi = torch.nan_to_num(pi, nan=1.0)
            dA = (self.damping ** 0.5) * pi
            dG = (self.damping ** 0.5) / pi
            try:
                A_inv = torch.linalg.inv((A + dA * torch.eye(A.size(0), device=A.device)).cpu()).to(A.device)
                G_inv = torch.linalg.inv((G + dG * torch.eye(G.size(0), device=G.device)).cpu()).to(G.device)
                self._Ai[mid] = A_inv
                self._Gi[mid] = G_inv
            except RuntimeError:
                pass

    def _precond(self, mid, m, gw, gb):
        if not self.use_so or mid not in self._Ai:
            return gw, gb
        Ai, Gi = self._Ai[mid], self._Gi[mid]

        if isinstance(m, nn.Linear):
            if gb is not None:
                C = torch.cat([gw, gb.unsqueeze(1)], dim=1)
                P = Gi @ C @ Ai
                return P[:, :-1], P[:, -1]
            return Gi @ gw @ Ai, None
        else:
            Gw = gw.view(gw.size(0), -1)
            if gb is not None:
                C = torch.cat([Gw, gb.unsqueeze(1)], dim=1)
                P = Gi @ C @ Ai
                return P[:, :-1].view_as(gw), P[:, -1]
            return (Gi @ Gw @ Ai).view_as(gw), None

    def _adam(self, p, grad):
        s = self.state[p]
        if not s:
            s['step'] = 0
            s['m'] = torch.zeros_like(p)
            s['v'] = torch.zeros_like(p)
        s['step'] += 1
        t = s['step']
        b1, b2 = self.betas
        lr = self.param_groups[0]['lr']

        if self.weight_decay:
            grad = grad + self.weight_decay * p.data

        s['m'].mul_(b1).add_(grad, alpha=1 - b1)
        s['v'].mul_(b2).addcmul_(grad, grad, value=1 - b2)
        alpha = lr * (1 - b2 ** t) ** 0.5 / (1 - b1 ** t)
        p.data.addcdiv_(s['m'], s['v'].sqrt().add_(self.eps), value=-alpha)

    @torch.no_grad()
    def step(self, closure=None):
        self._step += 1
        do_kfac = self.use_so and (self._step % self.kfac_interval == 0)
        do_zo = self.use_zo and (self._step % self.zo_interval == 0)

        # Optional: ZO gradient augmentation
        if do_zo and closure is not None:
            params = self.param_groups[0]['params']
            saved = [p.data.clone() for p in params]
            zs = [torch.randn_like(p) for p in params]

            # Use a robust epsilon that scales with the total number of parameters
            total_numel = sum(p.numel() for p in params)
            robust_eps = self.zo_epsilon / (total_numel ** 0.5)

            self._collecting = False
            for p, z in zip(params, zs):
                p.data.add_(robust_eps * z)
            with torch.enable_grad():
                try:
                    lp = float(closure())
                except:
                    lp = 100.0

            for p, z, s in zip(params, zs, saved):
                p.data.copy_(s - robust_eps * z)
            with torch.enable_grad():
                try:
                    lm = float(closure())
                except:
                    lm = 100.0

            for p, s in zip(params, saved):
                p.data.copy_(s)
            self._collecting = True

            # Clamp lp and lm to prevent massive inf spikes
            lp = min(max(lp, 0.0), 100.0)
            lm = min(max(lm, 0.0), 100.0)
            diff = lp - lm

            if not torch.isfinite(torch.tensor(diff)):
                diff = 0.0

            scale = diff / (2.0 * robust_eps) * self.zo_scale
            
            # Dynamically normalize ZO gradient norm to cap at 1.0 to prevent overriding FO
            zo_grad_norm = abs(scale) * (total_numel ** 0.5)
            if zo_grad_norm > 1.0 and zo_grad_norm > 0:
                scale = scale / zo_grad_norm

            for p, z in zip(params, zs):
                if p.grad is not None:
                    # If not using FO, we might want to ONLY use ZO gradient, so we zero out p.grad first
                    if not self.use_fo:
                        p.grad.data.zero_()
                    p.grad.data.add_(scale * z)

        if not self.use_fo and not self.use_zo:
            pass

        if do_kfac:
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
