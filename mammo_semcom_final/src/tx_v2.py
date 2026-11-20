"""V2 transmission: clinically guided block priorities + a network model with loss, bandwidth, latency and jitter.
Built on channel.py (tx_block_mask, block_importance, choose_lost, apply_loss) so the old experiment logic is unchanged."""
import numpy as np, pandas as pd, cv2, torch
from .channel import tx_block_mask, uniform_importance, block_importance, choose_lost, apply_loss
from .metrics import auc_safe, roi_bbox, roi_psnr
from .pipeline import diagnose_all
from .train import DEVICE
from .train_v2 import to_input2

NET = dict(p=0., burst=False, bw=1000., lat=.05, jit=.01, dl=1.)   # loss rate, bursty?, blocks/s, latency s, jitter s, deadline s
SCEN = {"ideal": {},
        "random 10%": dict(p=.1), "random 30%": dict(p=.3), "random 50%": dict(p=.5),
        "burst 10%": dict(p=.1, burst=True), "burst 30%": dict(p=.3, burst=True), "burst 50%": dict(p=.5, burst=True),
        "bandwidth 100 blk/s": dict(bw=100.), "bandwidth 50 blk/s": dict(bw=50.),
        "latency 500ms + jitter 300ms": dict(lat=.5, jit=.3), "burst 30% + 100 blk/s": dict(p=.3, burst=True, bw=100.)}
STRATEGIES = [("uniform", "none"), ("conventional", "conventional"), ("proposed", "clinical")]   # (strategy, importance source)


# ---------------- sender: block priorities ----------------
def saliency(clf, img01, heat, block):
    """Classification relevance of each block: |d logit / d image| of the full view, averaged per block, scaled to [0,1]."""
    f = lambda a: cv2.resize(a.astype(np.float32), (img01.shape[1] // 2, img01.shape[0] // 2), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(to_input2(f(img01), f(heat))[None]).to(DEVICE).requires_grad_(True)
    clf.eval(); clf(x)[2].sum().backward()
    g = cv2.resize(x.grad[0, :2].abs().sum(0).cpu().numpy(), (img01.shape[1], img01.shape[0]))
    ny, nx = img01.shape[0] // block, img01.shape[1] // block
    s = g[:ny * block, :nx * block].reshape(ny, block, nx, block).mean((1, 3))
    return s / (s.max() + 1e-8)


def importance_sets(cfg, img01, heat, clf):
    """none = no priority | conventional = generic block contrast (not clinical) | clinical = lesion map levels (2/1/0) + classifier saliency."""
    out = {"none": [], "conventional": [], "clinical": []}
    b = cfg.block
    for im, h in zip(img01, heat):
        tx = tx_block_mask(im, b); ny, nx = tx.shape
        out["none"].append(uniform_importance(tx))
        c = im[:ny * b, :nx * b].reshape(ny, b, nx, b).std((1, 3))
        out["conventional"].append(np.where(tx, c / (c.max() + 1e-8), -1.))
        lvl = block_importance(h.astype(np.float32), tx, b, near=cfg.near_blocks).astype(np.float32)
        out["clinical"].append(np.where(tx, lvl + .9 * saliency(clf, im, h, b), -1.))
    return out


# ---------------- network ----------------
def deliver(imp, strategy, net, rng, burst_len=6.):
    """One image through the network. imp >= 0: block is sent, -1: not sent.
    uniform      : blocks sent in random order, random or bursty loss
    conventional / proposed : priority-aware network, blocks sent by priority, lowest priority dropped first
    Blocks that arrive after the deadline (bandwidth, latency, jitter) are lost as well."""
    n_ = {**NET, **net}
    flat = imp.ravel(); idx = np.flatnonzero(flat >= 0); n = len(idx)
    prio = strategy != "uniform"
    order = idx[np.argsort(-flat[idx], kind="stable")] if prio else rng.permutation(idx)
    base = n_["lat"] + (np.arange(n) + 1) / n_["bw"]
    t = base + np.abs(rng.normal(0, n_["jit"], n))                       # arrival time of every block
    late = np.zeros(flat.size, bool); late[order[t > n_["dl"]]] = True; late = late.reshape(imp.shape)
    scheme = "semantic" if prio else ("burst" if n_["burst"] else "uniform")
    lost = choose_lost(np.where(late, -1, imp), n_["p"], scheme, rng, burst_len) | late
    ok = ~lost.ravel()[order]
    return lost, dict(loss_rate=lost.sum() / max(n, 1), delay_ms=1e3 * t[ok].mean() if ok.any() else np.nan,
                      jitter_ms=1e3 * (t - base).std(), thr_KBps=ok.sum() * 32 * 32 / 1024 / n_["dl"])


def run_network(cfg, img01, masks_u8, y, receiver, imps, thr, scenarios=SCEN, strategies=STRATEGIES, reps=1):
    N = len(img01); gt = [m > 0 for m in masks_u8]; bbox = [roi_bbox(g) for g in gt]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49)); gt_dil = [cv2.dilate(g.astype(np.uint8), k) > 0 for g in gt]
    rows = []
    for sname, net in scenarios.items():
        for strat, src in strategies:
            for rep in range(reps):
                dmg, st, rp, ps = np.empty_like(img01), [], np.full(N, np.nan), np.zeros(N)
                for i in range(N):
                    lost, s = deliver(imps[src][i], strat, net, np.random.default_rng(cfg.seed + 10007 * rep + i), cfg.burst_len)
                    dmg[i], _ = apply_loss(img01[i], lost, cfg.block, cfg.recon); st.append(s)
                    ps[i] = 10 * np.log10(1 / (((img01[i] - dmg[i]) ** 2).mean() + 1e-10))
                    if bbox[i]: rp[i] = roi_psnr(img01[i], dmg[i], bbox[i])
                r = diagnose_all(receiver, dmg)
                hit = np.array([gt_dil[i][int(round(r["cy"][i])), int(round(r["cx"][i]))] for i in range(N)])
                rows.append(dict(scenario=sname, strategy=strat, rep=rep, AUC=auc_safe(y, r["p"]), ACC=float(((r["p"] >= thr) == y).mean()),
                                 loc_hit=hit.mean(), PSNR=ps.mean(), ROI_PSNR=np.nanmean(rp), **pd.DataFrame(st).mean().to_dict()))
        print("done:", sname)
    return pd.DataFrame(rows)
