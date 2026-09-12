import torch
import torch.nn as nn


class Trinity(torch.optim.Optimizer):
    """
    Trinity Optimizer: forward-only ZO curvature sensing for selective natural-gradient
    (K-FAC) preconditioning, with an AdamW actuator and grafted magnitude.

    Implements Algorithms 1-3 of the Trinity paper plan:
      - ZO-PROBE (Alg. 2): per-block forward-only second-difference curvature sensor,
        producing a debiased effective-rank ratio rho_hat in (0,1] and a negative-
        curvature fraction nu_hat.
      - CONTROLLER (Alg. 3): hysteresis + minimum dwell time + global inversion budget
        (beta_max) turns per-block K-FAC preconditioning on/off based on rho_hat.
      - Update engine: a shadow AdamW state is always maintained per block; when a
        block is in SO mode, K-FAC-preconditioned gradients feed a second Adam state;
        the direction comes from the active mode, the magnitude is grafted from the
        shadow AdamW step (Frobenius-norm matched) so switching modes never jumps the
        effective step size.
      - ZO escape actuator: when a block's sensed negative-curvature fraction is high
        and its gradient norm is small, a single perturbed-GD kick is applied and a
        lockout counter prevents immediate re-triggering.

    `step()` requires a `closure()` that recomputes the loss under `torch.no_grad()`
    for the SAME minibatch used to produce `.grad` this step (see training scripts).
    """

    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4,
                 damping=1e-2, kfac_stat_interval=10, kfac_inv_interval=100, kfac_decay=0.95,
                 max_kfac_dim=4800,
                 sensor_interval=100, sensor_probes=4, sensor_nblocks=2, sensor_eps=1e-2,
                 ema_alpha=0.7, rho_lo=0.15, rho_hi=0.35, dwell_time=500, beta_max=0.3,
                 escape_nu_thr=0.75, escape_gamma_g=0.3, escape_r=1e-3, escape_lock=200,
                 grad_ema_alpha=0.99,
                 use_gc=True, use_so=True, use_zo=True, use_escape=True, use_grafting=True,
                 force_so_always=False, delta=1e-12):
        params = [p for p in model.parameters() if p.requires_grad]
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        self.model = model
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.damping = damping
        self.kfac_stat_interval = kfac_stat_interval
        self.kfac_inv_interval = kfac_inv_interval
        self.kfac_decay = kfac_decay
        self.max_kfac_dim = max_kfac_dim

        self.sensor_interval = sensor_interval
        self.sensor_probes = sensor_probes
        self.sensor_nblocks = sensor_nblocks
        self.sensor_eps = sensor_eps

        self.ema_alpha = ema_alpha
        self.rho_lo = rho_lo
        self.rho_hi = rho_hi
        self.dwell_time = dwell_time
        self.beta_max = beta_max

        # Separate, slower coefficient for the gradient-norm reference gbar. The
        # pseudocode writes only "gbar <- EMA(||g||)" with no coefficient, and reusing
        # alpha_ema=0.7 gives gbar an effective window of ~3 steps -- so the escape
        # condition ||g|| <= gamma_g*gbar (gamma_g=0.3) would require a >3x gradient
        # collapse within ~3 steps and would essentially never fire. gbar is meant to
        # be a stable "typical recent gradient scale" reference, hence a long horizon.
        self.grad_ema_alpha = grad_ema_alpha

        self.escape_nu_thr = escape_nu_thr
        self.escape_gamma_g = escape_gamma_g
        self.escape_r = escape_r
        self.escape_lock = escape_lock

        self.use_gc = use_gc
        self.use_so = use_so
        self.use_zo = use_zo
        self.use_escape = use_escape
        self.use_grafting = use_grafting
        self.force_so_always = force_so_always
        self.delta = delta

        self._step = 0
        self._collecting = True  # F_collect: gates KFAC stat hooks off during ZO probing
        self._rr_ptr = 0

        # Diagnostics: cumulative counters since the last get_diagnostics() call, so the
        # training loop can log, per epoch, which of the three paths (plain AdamW /
        # K-FAC-preconditioned / ZO escape-perturbation) blocks are actually taking,
        # and how the grafting scale (direction-from-active, magnitude-from-shadow
        # ratio) is behaving. Without this the only visibility from outside is the
        # loss curve, which conflates all of these together.
        self._diag_blocks_probed = 0
        self._diag_escape_fires = 0
        self._diag_kfac_inv_success = 0
        self._diag_kfac_inv_fail = 0
        self._diag_scale_sum = 0.0
        self._diag_scale_count = 0
        self._diag_scale_min = None
        self._diag_scale_max = None

        self._m_A = {}
        self._m_G = {}
        self._kA = {}
        self._kG = {}
        self._Ai = {}
        self._Gi = {}
        self._mods = {}

        # Block discovery (which Linear/Conv2d weight+bias pairs form a "block") is
        # independent of use_so: ZO sensing/escape and gradient centralization are all
        # block-oriented and must work even when K-FAC itself is disabled (e.g. the
        # "escape only" ablation variant). Only the A/G-capturing hooks (needed
        # exclusively by K-FAC) are gated by use_so.
        self._discover_blocks()
        if self.use_so:
            self._register_hooks()

        self._kfac_ids = set()
        for m in self._mods.values():
            self._kfac_ids.add(id(m.weight))
            if self._block_bias(m) is not None:
                self._kfac_ids.add(id(m.bias))

        # SO-eligible = within the K-FAC factor-dimension budget (d_max). Blocks that
        # exceed it are permanently FO (N10): excluded from the controller's candidate
        # pool/budget AND from the ZO sensing round-robin, not just from inversion.
        self._so_eligible_ids = (
            [mid for mid, m in self._mods.items() if self._is_so_eligible(m)]
            if self.use_so else []
        )
        so_eligible_set = set(self._so_eligible_ids)

        self._block_state = {}
        for mid, m in self._mods.items():
            bb = self._block_bias(m)
            d_l = m.weight.numel() + (bb.numel() if bb is not None else 0)
            eligible = mid in so_eligible_set
            self._block_state[mid] = dict(
                mode='SO' if (self.use_so and self.force_so_always and eligible) else 'FO',
                so_eligible=eligible,
                rho_ema=1.0, nu=0.0, last_switch=0, lock=0, grad_norm_ema=0.0,
                d_l=d_l,
                shadow_w={}, shadow_b={} if bb is not None else None,
                active_w={}, active_b={} if bb is not None else None,
            )

        self._other_state = {}

        if self.use_so and len(self._mods) > 0:
            total = sum(m.weight.numel() + self._bias_numel(m)
                        for m in self._mods.values())
            excluded = sum(m.weight.numel() + self._bias_numel(m)
                           for mid, m in self._mods.items() if mid not in so_eligible_set)
            frac = excluded / max(total, 1)
            if frac > 0:
                print(f"[Trinity] d_max={self.max_kfac_dim}: {frac:.3%} of trackable "
                      f"parameters permanently excluded from K-FAC (kept on AdamW).")

    # ------------------------------------------------------------------ hooks

    def _discover_blocks(self):
        for m in self.model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                self._mods[id(m)] = m

    def _register_hooks(self):
        for mid, m in self._mods.items():
            m.register_forward_hook(self._fwd(mid))
            m.register_full_backward_hook(self._bwd(mid))

    def _block_bias(self, m):
        """The block's bias, or None if absent or frozen.

        A frozen bias is not part of the optimizable block (it is not even in the
        optimizer's param list): it must not be perturbed by the sensor or the escape
        actuator, must not contribute a ones-column to the A factor, and must not
        receive an update. Every site that asks "does this block have a bias?" must
        agree, or the A factor's dimension and the preconditioner's input disagree.
        """
        b = m.bias
        return b if (b is not None and b.requires_grad) else None

    def _bias_numel(self, m):
        b = self._block_bias(m)
        return b.numel() if b is not None else 0

    def _kfac_factor_dims(self, m):
        # A-side (input) and G-side (output) Kronecker factor dims, as formed in
        # _update_kfac_stats: for Linear these are in/out features (+1 for bias on
        # the A side); for Conv2d, in_channels*kH*kW and out_channels.
        if isinstance(m, nn.Linear):
            a_dim = m.weight.shape[1] + (1 if self._block_bias(m) is not None else 0)
            g_dim = m.weight.shape[0]
        else:
            a_dim = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3] + \
                (1 if self._block_bias(m) is not None else 0)
            g_dim = m.weight.shape[0]
        return a_dim, g_dim

    def _is_so_eligible(self, m):
        a_dim, g_dim = self._kfac_factor_dims(m)
        return a_dim <= self.max_kfac_dim and g_dim <= self.max_kfac_dim

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

    # -------------------------------------------------------- BN/dropout probe mode

    def _set_probe_mode(self, enable):
        if enable:
            self._bn_saved_momentum = {}
            self._dropout_saved_training = {}
            for mod in self.model.modules():
                if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    self._bn_saved_momentum[id(mod)] = mod.momentum
                    mod.momentum = 0.0
                elif isinstance(mod, nn.modules.dropout._DropoutNd):
                    self._dropout_saved_training[id(mod)] = mod.training
                    mod.eval()
        else:
            for mod in self.model.modules():
                if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if id(mod) in self._bn_saved_momentum:
                        mod.momentum = self._bn_saved_momentum[id(mod)]
                elif isinstance(mod, nn.modules.dropout._DropoutNd):
                    if self._dropout_saved_training.get(id(mod), True):
                        mod.train()

    # -------------------------------------------------------------- ZO-PROBE (Alg. 2)

    def _zo_probe(self, mid, closure):
        m = self._mods[mid]
        w, b = m.weight, self._block_bias(m)

        self._collecting = False
        self._set_probe_mode(True)

        L0 = float(closure())
        cs = []
        for _ in range(self.sensor_probes):
            zw = torch.randn_like(w)
            zb = torch.randn_like(b) if b is not None else None

            w.data.add_(self.sensor_eps * zw)
            if b is not None:
                b.data.add_(self.sensor_eps * zb)
            Lp = float(closure())

            w.data.add_(-2.0 * self.sensor_eps * zw)
            if b is not None:
                b.data.add_(-2.0 * self.sensor_eps * zb)
            Lm = float(closure())

            w.data.add_(self.sensor_eps * zw)
            if b is not None:
                b.data.add_(self.sensor_eps * zb)

            c = (Lp - 2.0 * L0 + Lm) / (self.sensor_eps ** 2)
            cs.append(c)

        self._set_probe_mode(False)
        self._collecting = True

        cs_t = torch.tensor(cs, dtype=torch.float64)
        c_mean = cs_t.mean().item()
        m_probes = len(cs)
        c_var = cs_t.var(unbiased=True).item() if m_probes > 1 else 0.0

        r_eff = 2.0 * max(c_mean ** 2 - c_var / max(m_probes, 1), 0.0) / (c_var + self.delta)
        d_l = self._block_state[mid]['d_l']
        rho_hat = min(max(r_eff / d_l, 1.0 / d_l), 1.0)
        nu_hat = sum(1 for c in cs if c < 0) / m_probes
        return rho_hat, nu_hat

    # -------------------------------------------------------------- CONTROLLER (Alg. 3)

    def _controller(self, t):
        if self.force_so_always:
            return
        eligible_ids = self._so_eligible_ids
        B = len(eligible_ids)
        if B == 0:
            return
        cand = set()
        for mid in eligible_ids:
            info = self._block_state[mid]
            # tau_dwell is measured in TRAINING STEPS, not controller invocations.
            # The controller only runs every sensor_interval steps, so incrementing a
            # counter per invocation would make tau_dwell=500 mean 500*100 = 50k steps
            # -- longer than an entire 200-300 epoch run on LT-CIFAR-10, and no block
            # would ever switch to SO. The C3 sweep {0,100,500,2000} is only coherent
            # in steps.
            if t - info['last_switch'] < self.dwell_time:
                if info['mode'] == 'SO':
                    cand.add(mid)
                continue
            if info['mode'] == 'FO' and info['rho_ema'] < self.rho_lo:
                cand.add(mid)
            if info['mode'] == 'SO' and info['rho_ema'] <= self.rho_hi:
                cand.add(mid)

        budget = int(self.beta_max * B)
        if len(cand) > budget:
            cand = set(sorted(cand, key=lambda m: self._block_state[m]['rho_ema'])[:budget])

        for mid in eligible_ids:
            info = self._block_state[mid]
            new_mode = 'SO' if mid in cand else 'FO'
            if new_mode != info['mode']:
                if new_mode == 'SO':
                    self._warm_start_active(info)
                info['last_switch'] = t
                info['mode'] = new_mode

    def _clone_state(self, src):
        if not src:
            return {}
        return {'step': src['step'], 'm': src['m'].clone(), 'v': src['v'].clone()}

    def _warm_start_active(self, info):
        # FO->SO transition: snapshot the shadow AdamW moments into a FRESH,
        # independent dict for the active (K-FAC) Adam state (N8: "starts warm
        # rather than from zero"). Must be an independent copy, not an alias —
        # aliasing the two dicts would let the active branch's later updates
        # (on preconditioned gradients) silently corrupt the shadow AdamW state,
        # which is required to always track plain-AdamW-only statistics for
        # grafting.
        info['active_w'] = self._clone_state(info['shadow_w'])
        if info['shadow_b'] is not None:
            info['active_b'] = self._clone_state(info['shadow_b'])

    # ---------------------------------------------------------- ZO escape actuator

    def _escape_check(self, mid):
        info = self._block_state[mid]
        if info['lock'] > 0:
            return
        if info['nu'] < self.escape_nu_thr:
            return
        m = self._mods[mid]
        w, b = m.weight, self._block_bias(m)
        if w.grad is None:
            return
        gn2 = w.grad.data.norm().item() ** 2
        if b is not None and b.grad is not None:
            gn2 += b.grad.data.norm().item() ** 2
        grad_norm = gn2 ** 0.5
        if grad_norm > self.escape_gamma_g * info['grad_norm_ema'] + 1e-12:
            return

        # u ~ Uniform(S^{d_l-1}) over the JOINT [w; b] block, scaled by the block's
        # combined parameter norm (Alg. 1, lines 22-23) — one direction/norm across
        # the whole block, not two independent per-tensor perturbations.
        uw = torch.randn_like(w)
        ub = torch.randn_like(b) if b is not None else None
        u_norm = self._frob(uw, ub).clamp_min(1e-12)
        theta_norm = self._frob(w.data, b.data if b is not None else None)
        step_scale = (self.escape_r * theta_norm / u_norm).item()

        w.data.add_(step_scale * uw)
        if b is not None:
            b.data.add_(step_scale * ub)

        for st in (info['shadow_w'], info['shadow_b'], info['active_w'], info['active_b']):
            if st and 'm' in st:
                st['m'].mul_(0.5)

        info['lock'] = self.escape_lock
        self._diag_escape_fires += 1

    # ---------------------------------------------------------------- K-FAC (Alg. 1, Phase 3)

    def _update_kfac_stats(self):
        for mid, m in self._mods.items():
            A_raw = self._m_A.get(mid)
            G_raw = self._m_G.get(mid)
            if A_raw is None or G_raw is None:
                continue

            if isinstance(m, nn.Linear):
                A = A_raw.view(-1, A_raw.size(-1))
                G = G_raw.view(-1, G_raw.size(-1))
                if self._block_bias(m) is not None:
                    A = torch.cat([A, A.new_ones(A.size(0), 1)], dim=1)
            else:
                B = A_raw.size(0)
                unf = nn.Unfold(kernel_size=m.kernel_size, padding=m.padding,
                                stride=m.stride, dilation=m.dilation).to(A_raw.device)
                Auf = unf(A_raw)
                A = Auf.permute(0, 2, 1).reshape(-1, Auf.size(1))
                if self._block_bias(m) is not None:
                    A = torch.cat([A, A.new_ones(A.size(0), 1)], dim=1)
                G = (G_raw.view(B, G_raw.size(1), -1).permute(0, 2, 1).reshape(-1, G_raw.size(1)))

            n = A.size(0)
            Af = (A.t() @ A) / n
            Gf = (G.t() @ G) / n

            if Af.size(0) > self.max_kfac_dim or Gf.size(0) > self.max_kfac_dim:
                continue

            d = self.kfac_decay
            if mid not in self._kA:
                self._kA[mid] = Af
                self._kG[mid] = Gf
            else:
                self._kA[mid] = d * self._kA[mid] + (1 - d) * Af
                self._kG[mid] = d * self._kG[mid] + (1 - d) * Gf

    def _update_kfac_inverses(self):
        for mid, info in self._block_state.items():
            if info['mode'] != 'SO' or mid not in self._kA:
                continue
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
                self._diag_kfac_inv_success += 1
            except RuntimeError:
                self._diag_kfac_inv_fail += 1

    def _precond(self, mid, m, gw, gb):
        Ai, Gi = self._Ai[mid], self._Gi[mid]
        # The A factor carries an appended ones-column iff the MODULE has a bias (see
        # _update_kfac_stats), so this branch must key off m.bias, NOT off gb: if a
        # bias exists but has no grad (a frozen bias), keying off gb picks the
        # no-bias path and Ai's dimension no longer matches -> shape error.
        Gw = gw if isinstance(m, nn.Linear) else gw.view(gw.size(0), -1)
        if self._block_bias(m) is not None:
            gb_col = gb.unsqueeze(1) if gb is not None else Gw.new_zeros(Gw.size(0), 1)
            P = Gi @ torch.cat([Gw, gb_col], dim=1) @ Ai
            pw = P[:, :-1]
            pb = P[:, -1] if gb is not None else None
        else:
            pw, pb = Gi @ Gw @ Ai, None
        return (pw if isinstance(m, nn.Linear) else pw.reshape_as(gw)), pb

    # ----------------------------------------------------------------- AdamW core

    def _adam_delta(self, grad, state):
        if not state:
            state['step'] = 0
            state['m'] = torch.zeros_like(grad)
            state['v'] = torch.zeros_like(grad)
        state['step'] += 1
        t = state['step']
        b1, b2 = self.betas
        state['m'].mul_(b1).add_(grad, alpha=1 - b1)
        state['v'].mul_(b2).addcmul_(grad, grad, value=1 - b2)
        mhat = state['m'] / (1 - b1 ** t)
        vhat = state['v'] / (1 - b2 ** t)
        return mhat / (vhat.sqrt() + self.eps)

    def _frob(self, a, b):
        n = (a.double() ** 2).sum()
        if b is not None:
            n = n + (b.double() ** 2).sum()
        return n.sqrt()

    # ---------------------------------------------------------------------- step

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise ValueError("Trinity requires a closure() that recomputes the loss "
                              "under torch.no_grad() for the current minibatch (used by "
                              "the ZO curvature sensor).")
        self._step += 1
        t = self._step
        lr = self.param_groups[0]['lr']

        # Gradient centralization (Yong et al.) on the RAW gradient, restricted to the
        # tracked Linear/Conv2d weight blocks (ndim >= 2 block weights) — not every
        # ndim>=2 parameter in the model, which would also catch non-layer tensors
        # like a ViT's cls_token/pos_embed that GC was never intended for.
        if self.use_gc:
            for m in self._mods.values():
                w = m.weight
                if w.grad is not None and w.grad.dim() >= 2:
                    g = w.grad.data
                    g.sub_(g.mean(dim=tuple(range(1, g.dim())), keepdim=True))

        # ---- Phase 1: ZO sensing (round-robin, every sensor_interval steps) ----
        # Sensing itself only needs use_zo: nu_hat feeds the escape actuator even
        # when K-FAC (use_so) is disabled (e.g. the "escape only" ablation variant).
        # The CONTROLLER (which turns SO mode on/off) only makes sense when use_so.
        S_t = []
        if self.use_zo and t % self.sensor_interval == 0 and len(self._mods) > 0:
            # Which blocks to rotate through: nu_hat (escape trigger) is meaningful for
            # EVERY block, rho_hat (controller) only for SO-eligible ones. Sensor
            # overhead is fixed by sensor_nblocks regardless of which blocks are picked,
            # so when escape is on we rotate over all blocks -- restricting to
            # SO-eligible blocks would save nothing while leaving d_max-excluded blocks
            # (e.g. ResNet-18's layer4, ~75% of its parameters) unable to ever escape.
            if self.use_escape:
                block_ids = list(self._mods.keys())
            elif self.use_so:
                block_ids = self._so_eligible_ids
            else:
                block_ids = []
            n_blocks = len(block_ids)
            for _ in range(min(self.sensor_nblocks, n_blocks)):
                mid = block_ids[self._rr_ptr % n_blocks]
                self._rr_ptr += 1
                rho_hat, nu_hat = self._zo_probe(mid, closure)
                info = self._block_state[mid]
                info['rho_ema'] = self.ema_alpha * info['rho_ema'] + (1 - self.ema_alpha) * rho_hat
                info['nu'] = nu_hat
                S_t.append(mid)
                self._diag_blocks_probed += 1
            if self.use_so:
                self._controller(t)

        # ---- Phase 2: ZO escape actuator + lockout/grad-norm EMA bookkeeping ----
        if self.use_escape:
            for mid in S_t:
                self._escape_check(mid)
        for mid, m in self._mods.items():
            info = self._block_state[mid]
            info['lock'] = max(info['lock'] - 1, 0)
            gn2 = 0.0
            if m.weight.grad is not None:
                gn2 += m.weight.grad.data.norm().item() ** 2
            bb = self._block_bias(m)
            if bb is not None and bb.grad is not None:
                gn2 += bb.grad.data.norm().item() ** 2
            a = self.grad_ema_alpha
            info['grad_norm_ema'] = a * info['grad_norm_ema'] + (1 - a) * (gn2 ** 0.5)

        # ---- Phase 3: K-FAC factors + gated inverses ----
        if self.use_so and t % self.kfac_stat_interval == 0:
            self._update_kfac_stats()
        if self.use_so and t % self.kfac_inv_interval == 0:
            self._update_kfac_inverses()

        # ---- Phase 4: update engine with grafting ----
        for mid, m in self._mods.items():
            info = self._block_state[mid]
            w, b = m.weight, self._block_bias(m)
            if w.grad is None:
                continue
            gw = w.grad.data.clone()
            gb = b.grad.data.clone() if (b is not None and b.grad is not None) else None

            dw_ad = self._adam_delta(gw, info['shadow_w'])
            db_ad = self._adam_delta(gb, info['shadow_b']) if gb is not None else None

            if info['mode'] == 'SO' and mid in self._Ai:
                pw, pb = self._precond(mid, m, gw, gb)
                dw_act = self._adam_delta(pw, info['active_w'])
                db_act = self._adam_delta(pb, info['active_b']) if pb is not None else None
            else:
                # FO mode: just reuse the shadow AdamW delta this step. Do NOT alias
                # info['active_w'] to info['shadow_w'] here — that would make a later
                # SO-mode _adam_delta(pw, info['active_w']) call mutate the SAME dict
                # object as shadow_w, corrupting the shadow AdamW trajectory that
                # grafting depends on. Warm-starting active on FO->SO transition is
                # handled once, by _warm_start_active() in _controller().
                dw_act, db_act = dw_ad, db_ad

            if self.use_grafting:
                norm_ad = self._frob(dw_ad, db_ad)
                norm_act = self._frob(dw_act, db_act)
                scale = (norm_ad / (norm_act + self.delta)).to(dw_act.dtype)
            else:
                scale = 1.0

            scale_val = scale.item() if torch.is_tensor(scale) else scale
            self._diag_scale_sum += scale_val
            self._diag_scale_count += 1
            self._diag_scale_min = scale_val if self._diag_scale_min is None else min(self._diag_scale_min, scale_val)
            self._diag_scale_max = scale_val if self._diag_scale_max is None else max(self._diag_scale_max, scale_val)

            w.data.add_(-lr * (scale * dw_act + self.weight_decay * w.data))
            if b is not None and db_act is not None:
                b.data.add_(-lr * (scale * db_act.to(b.dtype) + self.weight_decay * b.data))

        # non-block (permanently FO) params: norms, standalone biases, 1-D tensors
        for p in self.param_groups[0]['params']:
            if id(p) in self._kfac_ids or p.grad is None:
                continue
            st = self._other_state.setdefault(id(p), {})
            d = self._adam_delta(p.grad.data.clone(), st)
            p.data.add_(-lr * (d + self.weight_decay * p.data))

        return None

    def get_diagnostics(self, reset=True):
        """Which of the three paths blocks are actually taking, right now.

        - mode counts: a LIVE snapshot of how many blocks are currently sitting in
          FO (plain AdamW) vs SO (K-FAC-preconditioned) mode.
        - escape_fires / blocks_probed / kfac_inv_*: CUMULATIVE counts of events
          since the last call (the "third path" — ZO perturbation kicks — plus how
          much sensing/inversion activity actually happened), so a training loop
          can call this once per epoch and log a per-epoch rate.
        - grafting_scale_*: the direction-from-active/magnitude-from-shadow ratio
          actually applied at the update step, aggregated over the same window.
          Near 1.0 means "active mode's own update already had shadow-AdamW-like
          magnitude"; far from 1.0 means grafting is doing real rescaling work
          (or, if consistently tiny, that SO-mode updates are being crushed).

        Call this once per epoch (reset=True, the default) so the numbers describe
        "this epoch," not a running average since training started.
        """
        n_blocks = len(self._block_state)
        n_so = sum(1 for i in self._block_state.values() if i['mode'] == 'SO')
        rho_vals = [i['rho_ema'] for i in self._block_state.values() if i['so_eligible']]
        nu_vals = [i['nu'] for i in self._block_state.values()]

        diag = {
            'n_blocks': n_blocks,
            'n_so': n_so,
            'n_fo': n_blocks - n_so,
            'so_frac': n_so / max(n_blocks, 1),
            'rho_mean': sum(rho_vals) / len(rho_vals) if rho_vals else float('nan'),
            'rho_min': min(rho_vals) if rho_vals else float('nan'),
            'rho_max': max(rho_vals) if rho_vals else float('nan'),
            'nu_mean': sum(nu_vals) / len(nu_vals) if nu_vals else float('nan'),
            'blocks_probed': self._diag_blocks_probed,
            'escape_fires': self._diag_escape_fires,
            'kfac_inv_success': self._diag_kfac_inv_success,
            'kfac_inv_fail': self._diag_kfac_inv_fail,
            'grafting_scale_mean': (self._diag_scale_sum / self._diag_scale_count
                                     if self._diag_scale_count else float('nan')),
            'grafting_scale_min': self._diag_scale_min if self._diag_scale_min is not None else float('nan'),
            'grafting_scale_max': self._diag_scale_max if self._diag_scale_max is not None else float('nan'),
        }

        if reset:
            self._diag_blocks_probed = 0
            self._diag_escape_fires = 0
            self._diag_kfac_inv_success = 0
            self._diag_kfac_inv_fail = 0
            self._diag_scale_sum = 0.0
            self._diag_scale_count = 0
            self._diag_scale_min = None
            self._diag_scale_max = None

        return diag
