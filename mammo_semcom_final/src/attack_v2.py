"""Adversarial robustness of the receiver.
Pixel attacks : white-box PGD on the classifier / localizer, transfer attack from a surrogate, 2-stage end-to-end PGD, adaptive attack through a blur defence.
Tag attacks   : the semantic priority tags are attacked (read and erase, invert, inject fake high priority) and defended (encrypt, decoy, randomise)."""
import numpy as np, torch, torch.nn.functional as F
from .train import DEVICE, MEAN3, STD3

_M, _S = torch.tensor(MEAN3).unsqueeze(0).to(DEVICE), torch.tensor(STD3).unsqueeze(0).to(DEVICE)


# ============================================================ pixel attacks
def blur_t(x, sigma=1.0):
    """Differentiable Gaussian blur (the receiver's denoising defence; lets the attacker adapt to it). x: (B,1,H,W)."""
    r = int(3 * sigma)
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype); k = torch.exp(-k ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    x = F.pad(x, (r, r, r, r), mode="reflect")
    return F.conv2d(F.conv2d(x, k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))


def crop_input(x, hm, c, cfg):
    """Differentiable version of extract_patch + to_input2. x, hm: (B,1,H,W); c: (B,2) crop centres -> (B,3,out,out)."""
    P, h = cfg.patch, cfg.patch // 2
    z = F.pad(torch.cat([x, hm], 1), (h, h, h, h))
    p = torch.stack([z[b, :, int(round(float(cy))):int(round(float(cy))) + P, int(round(float(cx))):int(round(float(cx))) + P]
                     for b, (cy, cx) in enumerate(c)])
    p = F.interpolate(p, size=(cfg.patch_out, cfg.patch_out), mode="bilinear", align_corners=False)
    return (torch.cat([p[:, :1], p[:, :1], p[:, 1:]], 1) - _M) / _S


def pgd(grad_fn, x, eps, steps=10, start=None):
    """L_inf PGD (steps=1 -> FGSM) inside the eps-ball around x, optionally starting from `start`. 8-bit rounded at the end."""
    a, xa = (eps / 4 if steps > 1 else eps), (x if start is None else start).clone()
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        g = grad_fn(xa)
        xa = torch.min(torch.max(xa.detach() + a * g.sign(), x - eps), x + eps).clamp(0, 1)
    return torch.round(xa * 255) / 255


def attack(kind, x, y, hm, c, masks, localizers, clf, cfg, eps, steps=10, pre=None, start=None):
    """kind 'clf': maximise the classifier BCE on the lesion crop at centres c (map channel hm).
    kind 'loc': maximise the localizer-ensemble BCE against the true mask (pushes the peak away from the lesion).
    pre: defence applied by the receiver before the models (e.g. blur_t) -> adaptive attack. Tensors (B,1,H,W) on DEVICE."""
    pre = pre or (lambda t: t)
    if kind == "clf":
        yy = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
        gf = lambda xa: torch.autograd.grad(F.binary_cross_entropy_with_logits(clf(crop_input(pre(xa), hm, c, cfg))[2], yy), xa)[0]
    else:
        gf = lambda xa: sum(torch.autograd.grad(F.binary_cross_entropy_with_logits(m(pre(xa)), masks), xa)[0] for m in localizers)
    return pgd(gf, x, eps, steps, start)


def attack_e2e(receiver, x, y, masks, cfg, eps, steps=10):
    """2-stage end-to-end PGD through the whole receiver: stage 1 attacks the localizer with eps/2; the receiver then re-locates the
    lesion on the perturbed image; stage 2 attacks the classifier at that new crop (and its new map), continuing inside the full eps-ball."""
    a1 = attack("loc", x, y, None, None, masks, receiver.loc, receiver.clf, cfg, eps / 2, steps)
    with torch.no_grad():
        h1 = torch.stack([torch.sigmoid(m(a1)) for m in receiver.loc]).mean(0)
    r = receiver.diagnose(a1[:, 0].cpu().numpy())
    return attack("clf", x, y, h1, np.stack([r["cy"], r["cx"]], 1), masks, receiver.loc, receiver.clf, cfg, eps, steps, start=a1)


def perturb_quality(clean, adv, n=20):
    """PSNR (all images) and SSIM (first n images) between clean and perturbed images: shows how visible the attack is."""
    from skimage.metrics import structural_similarity as ssim
    ps = float(np.mean([10 * np.log10(1 / (((c - a) ** 2).mean() + 1e-12)) for c, a in zip(clean, adv)]))
    ss = float(np.mean([ssim(c, a, data_range=1.0) for c, a in zip(clean[:n], adv[:n])]))
    return ps, ss


# ============================================================ tag attacks / defences (numpy only)
def public_tags(imp, defence, rng, gamma=.5):
    """The priority tags the network (and an eavesdropper) sees. imp: true block importance (>=0 sent, -1 not sent;
    lesion-level blocks have imp >= 2, background < 1).
    none: tags = importance | decoy: extra background blocks are tagged top priority (2x the lesion-level count, >=4)
    random: importance + noise (range gamma*3), so the ranking no longer exactly reveals the lesion."""
    out = imp.copy().astype(np.float64); f = out.ravel(); idx = np.flatnonzero(f >= 0)
    if defence == "decoy" and len(idx):
        n_top = int((f[idx] >= 2).sum()); low = idx[f[idx] < 1]
        d = min(len(low), max(2 * n_top, 4))
        if d: f[rng.choice(low, d, replace=False)] = f[idx].max()
    elif defence == "random" and len(idx):
        f[idx] += rng.random(len(idx)) * gamma * 3.0
    return out


def tag_attack(kind, imp_true, tags, k, rng, reads_tags=True):
    """Returns (tags used by the network, blocks erased in transit or None).
    erase : erase k blocks - the k highest-tagged ones if the attacker reads the tags (= the lesion), else k random sent blocks
    invert: rewrite the tags so the ranking is reversed (lesion becomes lowest priority)
    inject: tag k blocks as the highest priority (background blocks if the tags are readable) to crowd out the lesion"""
    if kind == "none":
        return tags, None
    flat = tags.ravel(); idx = np.flatnonzero(imp_true.ravel() >= 0); k = min(k, len(idx))
    if kind == "erase":
        sel = idx[np.lexsort((rng.random(len(idx)), -flat[idx]))][:k] if reads_tags else rng.choice(idx, k, replace=False)
        lost = np.zeros(flat.size, bool); lost[sel] = True
        return tags, lost.reshape(tags.shape)
    f = tags.copy(); ff = f.ravel()
    if kind == "invert":
        ff[idx] = ff[idx].max() - ff[idx]
    elif kind == "inject":
        pool = idx[flat[idx] < 1] if reads_tags and (flat[idx] < 1).sum() >= k else idx
        ff[rng.choice(pool, k, replace=False)] = ff[idx].max() + 1.0
    else:
        raise ValueError(kind)
    return f, None
