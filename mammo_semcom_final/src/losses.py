import torch, torch.nn.functional as F
EPS = 1e-6
LAM = dict(attr=.5, pair=.1, orth=.1, F=.5, C=.5, B=.5, M=.5)    # lambda_1..3, lambda_F, lambda_C, lambda_B, lambda_M
# CLEAN_V2 localizer parameters (defaults = the specified formulas). peak = (start epoch, max weight, ramp epochs)
LOC = dict(pos_w=10., tv_a=.3, tv_b=.7, r=24, temp=1., peak=(5, .3, 3))


# ---------- 1. Localization: CLEAN_V2 ----------
def lam_peak(ep, start=5, wmax=.3, ramp=3):
    """default: 0 for epochs 0-4, 0.1 / 0.2 / 0.3 from epoch 5 on."""
    return 0. if ep < start else min(wmax, wmax * (ep - start + 1) / ramp)


def loc_loss(z, y, c, lam, P=LOC):
    """z: logits (B,1,H,W); y: binary mask; c: (B,2) lesion centre (row,col); lam: peak weight; P: parameters."""
    p = torch.sigmoid(z)
    t = F.avg_pool2d(y, 15, 1, 7).clamp(0, 1)                                    # soft target
    bce = -(P["pos_w"] * t * torch.log(p + EPS) + (1 - t) * torch.log(1 - p + EPS)).mean()
    d = (1, 2, 3)
    tp, fp, fn = (p * t).sum(d), (p * (1 - t)).sum(d), ((1 - p) * t).sum(d)
    tversky = (1 - (tp + 1) / (tp + P["tv_a"] * fp + P["tv_b"] * fn + 1)).mean()
    if lam == 0:
        return bce + tversky
    H, W = z.shape[-2:]
    yy = torch.arange(H, device=z.device)[None, :, None]; xx = torch.arange(W, device=z.device)[None, None, :]
    near = ((yy - c[:, 0, None, None]) ** 2 + (xx - c[:, 1, None, None]) ** 2 <= P["r"] ** 2).flatten(1)
    mass = torch.exp(F.log_softmax(z.flatten(1) / P["temp"], 1)).masked_fill(~near, 0).sum(1)
    return bce + tversky + lam * (-torch.log(mass + EPS)).mean()


# ---------- 2. Lesion-vs-background terms ----------
def wbce(logit, y, alpha):
    p = torch.sigmoid(logit)
    return -(alpha * y * torch.log(p + EPS) + (1 - y) * torch.log(1 - p + EPS)).mean()


def lb_extra(lL, lB, lF, y, alpha, lam=LAM, rho=.1, delta=.2):
    """L_LB minus WBCE(pL) (that term is already the base loss, so it is not counted twice)."""
    pL, pB, pF = map(torch.sigmoid, (lL, lB, lF))
    cons = ((pF - pL.detach()) ** 2).mean()
    bg = ((pB - rho * y) ** 2).mean()
    margin = (F.relu(pB - pL + delta) ** 2).mean()
    return lam["F"] * wbce(lF, y, alpha) + lam["C"] * cons + lam["B"] * bg + lam["M"] * margin


# ---------- 3. Phenotype-diagnostic two-branch ----------
def attr_loss(u, heads, A):
    """A: (B,R) descriptor codes, -1 = not observed."""
    tot, n = 0., 0
    for r, g in enumerate(heads):
        m = A[:, r] >= 0
        if m.any():
            tot = tot + F.cross_entropy(g(u[m]), A[m, r], reduction="sum"); n += int(m.sum())
    return tot / (n + EPS)


def pair_loss(u, v, y, t, pid, A, m=1., tau=.5):
    """Pairs inside the batch: same lesion type, opposite label, different patient."""
    both = (A >= 0)[:, None] & (A >= 0)[None]
    nO = both.sum(-1)
    D = ((A[:, None] != A[None]) & both).sum(-1) / nO.clamp(min=1)
    w = torch.exp(-D / tau)
    E = ((u[:, None] - u[None]) ** 2).sum(-1) + F.relu(m - (((v[:, None] - v[None]) ** 2).sum(-1) + 1e-8).sqrt()) ** 2
    valid = (y[:, None] != y[None]) & (pid[:, None] != pid[None]) & (nO > 0)
    per_type = []
    for k in (0, 1):                                                             # 0 = mass, 1 = calcification
        vm = valid & (t[:, None] == k) & (t[None] == k)
        if vm.any():
            per_type.append((w * E * vm).sum() / ((w * vm).sum() + EPS))
    return torch.stack(per_type).mean() if per_type else u.sum() * 0


def orth_loss(u, v):
    C = (u - u.mean(0)).T @ (v - v.mean(0)) / len(u)
    return (C ** 2).sum()
