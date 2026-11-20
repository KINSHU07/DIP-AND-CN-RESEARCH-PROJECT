"""V2 training on top of the existing project: CLEAN_V2 localizer (tunable), classifier with selectable losses, receiver."""
import os, re, json, zipfile
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import roc_curve
from .train import (DEVICE, _autocast, _scaler, seed_everything, extract_patch, to01, MEAN3, STD3, _worker_rng,
                    SegDS, predict_heatmaps, hit_rate, dice_np, localize)
from .models import UNetR18
from .metrics import auc_safe
from .losses import *


# ============================================================ localizer: CLEAN_V2 (tunable) + ensemble
ACT = {"relu": nn.ReLU, "leaky": lambda: nn.LeakyReLU(.1), "silu": nn.SiLU, "gelu": nn.GELU}


def set_activation(m, act):
    """Replace the ReLUs of the decoder (the pretrained encoder keeps its own)."""
    for d in (m.d4, m.d3, m.d2, m.d1):
        for i, l in enumerate(d.net):
            if isinstance(l, nn.ReLU):
                d.net[i] = ACT[act]()
    return m


def load_localizer_v2(cfg, name, act="relu"):
    m = set_activation(UNetR18(False), act).to(DEVICE)
    m.load_state_dict(torch.load(cfg.path("models", f"localizer_{name}.pt"), map_location=DEVICE))
    return m.eval()


def image_centres(abn_df, n):
    """(cy, cx) of the largest lesion of every image."""
    c = abn_df.sort_values("area").groupby("img_idx").tail(1).set_index("img_idx")[["cy", "cx"]]
    return c.reindex(range(n)).fillna(0).values.astype(np.float32)


class SegDS2(SegDS):
    def __init__(self, *a, centres, **k):
        super().__init__(*a, **k); self.c = centres

    def __getitem__(self, k):
        x, y = super().__getitem__(k)
        return x, y, torch.from_numpy(self.c[self.idxs[k]])


def train_localizer_v2(cfg, images, masks, centres, train_idx, val_idx, seed, name=None, P=None, act="relu", epochs=None, verbose=True):
    """Returns (model, val hit-rate, val Dice) of the best epoch. name=None -> not saved (used for the parameter sweep)."""
    P = {**LOC, **(P or {})}; E = epochs or cfg.loc_epochs
    seed_everything(seed)
    m = set_activation(UNetR18(cfg.pretrained), act).to(DEVICE)
    dl = DataLoader(SegDS2(images, masks, train_idx, cfg, True, False, centres=centres), batch_size=cfg.loc_bs,
                    shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.loc_lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E)
    scaler, best, state, bs = _scaler(), -1.0, None, (0, 0)
    vi, vm = [images[i] for i in val_idx], [masks[i] for i in val_idx]
    for ep in range(E):
        m.train(); lam = lam_peak(ep, *P["peak"])
        for x, y, c in dl:
            x, y, c = x.to(DEVICE), y.to(DEVICE), c.to(DEVICE)
            with _autocast():
                lg = m(x)
            loss = loc_loss(lg.float(), y, c, lam, P)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        heat = predict_heatmaps(m, vi); hr, dc = hit_rate(heat, vm), dice_np(heat, vm)
        if verbose:
            print(f"seed {seed} ep {ep + 1:2d}/{E} | loss {loss.item():.3f} | val hit {hr:.3f} | val Dice {dc:.3f}")
        if hr + dc >= best:
            best, bs, state = hr + dc, (hr, dc), {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    if name:
        torch.save(m.state_dict(), cfg.path("models", f"localizer_{name}.pt"))
    return m, bs[0], bs[1]


def ensemble_heat(localizers, imgs, chunk=64):
    """P(i) = mean over the M localizers of sigmoid(z_i). Returns float16 (N,H,W)."""
    out = np.zeros((len(imgs),) + imgs[0].shape, np.float16)
    for s in range(0, len(imgs), chunk):
        out[s:s + chunk] = np.mean([predict_heatmaps(l, imgs[s:s + chunk]).astype(np.float32) for l in localizers], 0)
    return out


# ============================================================ clinical descriptors (read from the CBIS case CSVs inside the ZIP)
DESC = ["mass_shape", "mass_margins", "calc_type", "calc_distribution"]


def descriptors(zip_path, abn_df, images_df):
    """Returns A (n_abn, 4) descriptor codes (-1 = not observed) and K (number of categories per descriptor)."""
    raws = []
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            mt = re.match(r"(mass|calc)_case_description_(train|test)_set\.csv", os.path.basename(n).lower())
            if mt:
                d = pd.read_csv(z.open(n)); d.columns = [c.strip().lower().replace(" ", "_") for c in d.columns]
                d["lesion_type"] = mt.group(1); raws.append(d)
    raw = pd.concat(raws, ignore_index=True).rename(columns={"left_or_right_breast": "side", "image_view": "view"})
    raw["pathology"], raw["subtlety"] = raw.pathology.astype(str).str.upper(), raw.subtlety.astype(float)
    for c in DESC:
        if c not in raw:
            raw[c] = np.nan
    key = ["patient_id", "side", "view", "lesion_type", "pathology", "subtlety"]
    raw = raw.drop_duplicates(key)[key + DESC]
    a = abn_df.merge(images_df[["img_idx", "side", "view"]], on="img_idx", how="left")
    a["pathology"], a["subtlety"] = a.pathology.astype(str).str.upper(), a.subtlety.astype(float)
    a = a.merge(raw, on=key, how="left"); assert len(a) == len(abn_df)
    A = [pd.factorize(a[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan}))[0] for c in DESC]
    print("descriptor observed per column:", {c: int((x >= 0).sum()) for c, x in zip(DESC, A)}, "of", len(a))
    return np.stack(A, 1).astype(np.int64), [max(int(x.max()) + 1, 1) for x in A]


# ============================================================ classifier
def to_input2(p, m):
    """Channels = (lesion image, lesion image, localization map): keeps the ImageNet stem and adds the localization guidance."""
    return ((np.stack([p, p, m]) - MEAN3) / STD3).astype(np.float32)


class ClfDS(Dataset):
    """One abnormality -> lesion view xL (patch), lesion-removed view xB, full view xF (+ label, type, patient, descriptors)."""
    def __init__(self, images, heat, abn, A, cfg, train):
        self.images, self.heat, self.abn, self.A, self.cfg, self.train = images, heat, abn.reset_index(drop=True), A, cfg, train
        self.rng = None
        self.y = self.abn.label.values
        self.t = self.abn.lesion_type.str.startswith("calc").astype(int).values
        self.pid = pd.factorize(self.abn.patient_id)[0]

    def __len__(self):
        return len(self.abn)

    def __getitem__(self, i):
        r, rng, c = self.abn.iloc[i], _worker_rng(self), self.cfg
        img, hm = to01(self.images[int(r.img_idx)]), self.heat[int(r.img_idx)].astype(np.float32)
        cy, cx = float(r.cy), float(r.cx)
        if self.train:
            j = .25 * c.patch; cy += rng.uniform(-j, j); cx += rng.uniform(-j, j)
        pl, ml = [extract_patch(a, cy, cx, c.patch, c.patch_out) for a in (img, hm)]
        if self.train:
            if rng.random() < .5: pl, ml = pl[:, ::-1], ml[:, ::-1]
            if rng.random() < .5: pl, ml = pl[::-1], ml[::-1]
            pl = np.clip(pl * rng.uniform(.9, 1.1) + rng.uniform(-.05, .05), 0, 1)
        xf, mf = [cv2.resize(a, (c.img_w // 2, c.img_h // 2), interpolation=cv2.INTER_AREA) for a in (img, hm)]
        xb, mb, h, y0, x0 = xf.copy(), mf.copy(), c.patch // 4, int(cy // 2), int(cx // 2)
        xb[max(y0 - h, 0):y0 + h, max(x0 - h, 0):x0 + h] = 0; mb[max(y0 - h, 0):y0 + h, max(x0 - h, 0):x0 + h] = 0   # lesion window removed
        T = lambda p, m: torch.from_numpy(to_input2(np.ascontiguousarray(p), np.ascontiguousarray(m)))
        return (T(pl, ml), T(xb, mb), T(xf, mf), torch.tensor(float(self.y[i])), torch.tensor(int(self.t[i])),
                torch.tensor(int(self.pid[i])), torch.from_numpy(self.A[i]))


class TwoBranch(nn.Module):
    """ResNet-18 backbone -> phenotype embedding u, diagnostic embedding v, p = sigmoid(w^T v + b), descriptor heads g_r(u)."""
    def __init__(self, K, pretrained=True, emb=64):
        super().__init__()
        r = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        for n, p in r.named_parameters():                       # same as the old classifier: only layer3/layer4 are fine-tuned
            p.requires_grad = n.startswith(("layer3", "layer4"))
        r.fc = nn.Identity(); self.bb = r
        self.P, self.D, self.w = nn.Linear(512, emb), nn.Linear(512, emb), nn.Linear(emb, 1)
        self.g = nn.ModuleList([nn.Linear(emb, k) for k in K])

    def forward(self, x):
        h = self.bb(x)
        v = F.normalize(self.D(h), dim=1)
        return F.normalize(self.P(h), dim=1), v, self.w(v).squeeze(1)


def _prob(m, x, tta=True):
    """p_L; test-time augmentation = mean over the 4 flips (the classifier was trained with these flips)."""
    fl = [[], [3], [2], [2, 3]] if tta else [[]]
    return torch.stack([torch.sigmoid(m(x.flip(f) if f else x)[2].float()) for f in fl]).mean(0)


@torch.no_grad()
def predict_lesion(m, ds, bs=64, tta=True):
    m.eval(); out = []
    for xl, *_ in DataLoader(ds, batch_size=bs, num_workers=2):
        with _autocast():
            out.append(_prob(m, xl.to(DEVICE), tta).cpu())
    return torch.cat(out).numpy()


MODES = {"wbce": "weighted BCE only (normal)", "lb": "WBCE + lesion-vs-background (consistency, background, margin)",
         "pheno": "two-branch phenotype (WBCE + attribute + pair + orthogonality)", "full": "all terms"}


def train_classifier_v2(cfg, images, heat, abn, A, K, mode="full", lam=None, epochs=None, name=None):
    """Returns (model, val-threshold, best val AUC). mode in MODES; lam overrides entries of LAM."""
    lam = {**LAM, **(lam or {})}; use_ph, use_lb = mode in ("pheno", "full"), mode in ("lb", "full"); E = epochs or cfg.clf_epochs
    seed_everything(cfg.seed)
    idx = {s: np.where(abn.split.values == s)[0] for s in ("train", "val")}
    mk = lambda s, train: ClfDS(images, heat, abn.iloc[idx[s]], A[idx[s]], cfg, train)
    dl = DataLoader(mk("train", True), batch_size=cfg.clf_bs, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    val_ds = mk("val", False); yv = val_ds.y
    ytr = abn.label.values[idx["train"]]; alpha = float((ytr == 0).sum() / (ytr == 1).sum())     # N_benign / N_malignant (train only)
    m = TwoBranch(K, cfg.pretrained).to(DEVICE)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=cfg.clf_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E)
    scaler, best, state = _scaler(), -1.0, None
    for ep in range(E):
        m.train()
        for xl, xb, xf, y, t, pid, Ab in dl:
            xl, xb, xf, y, t, pid, Ab = [a.to(DEVICE) for a in (xl, xb, xf, y, t, pid, Ab)]
            with _autocast():
                u, v, lL = m(xl)
                if use_lb:
                    lF, lB = m(xf)[2], m(xb)[2]
            u, v, lL = u.float(), v.float(), lL.float()
            loss = wbce(lL, y, alpha)
            if use_ph:
                loss = loss + lam["attr"] * attr_loss(u, m.g, Ab) + lam["pair"] * pair_loss(u, v, y, t, pid, Ab) + lam["orth"] * orth_loss(u, v)
            if use_lb:
                loss = loss + lb_extra(lL, lB.float(), lF.float(), y, alpha, lam)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        pv = predict_lesion(m, val_ds, tta=False); auc = auc_safe(yv, pv); score = 0. if np.isnan(auc) else auc
        print(f"[{name or mode}] epoch {ep + 1:2d}/{E} | loss {loss.item():.3f} | val AUC {auc:.4f}")
        if score >= best:
            best, state = score, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    bpv = predict_lesion(m, val_ds, tta=True)
    fpr, tpr, th = roc_curve(yv, bpv); thr = float(th[np.argmax(tpr - fpr)])                  # threshold chosen on validation only
    if name:
        torch.save(m.state_dict(), cfg.path("models", f"clf_{name}.pt"))
        json.dump(dict(K=K, thr=thr, mode=mode), open(cfg.path("models", f"clf_{name}.json"), "w"))
    return m, thr, best


def load_classifier_v2(cfg, name="v2"):
    meta = json.load(open(cfg.path("models", f"clf_{name}.json")))
    m = TwoBranch(meta["K"], False).to(DEVICE)
    m.load_state_dict(torch.load(cfg.path("models", f"clf_{name}.pt"), map_location=DEVICE))
    return m.eval(), meta


# ============================================================ receiver (same idea as pipeline.Receiver)
class ReceiverV2:
    """damaged image -> localizer ensemble -> peak -> lesion crop (+ localization map) -> classifier."""
    def __init__(self, cfg, localizers, clf, sigma=5.0, tta=True):
        self.cfg, self.loc, self.clf, self.sigma, self.tta = cfg, localizers, clf, sigma, tta

    @torch.no_grad()
    def diagnose(self, imgs01):
        c = self.cfg
        heat = np.mean([predict_heatmaps(l, imgs01).astype(np.float32) for l in self.loc], 0)
        locs = [localize(h, self.sigma) for h in heat]
        x = np.stack([to_input2(extract_patch(im, cy, cx, c.patch, c.patch_out), extract_patch(h, cy, cx, c.patch, c.patch_out))
                      for im, h, (cy, cx, _) in zip(imgs01, heat, locs)])
        with _autocast():
            p = _prob(self.clf, torch.from_numpy(x).to(DEVICE), self.tta).cpu().numpy()
        q = np.clip(p, 1e-6, 1 - 1e-6)
        return dict(p=p, ent=-(q * np.log(q) + (1 - q) * np.log(1 - q)), mi=np.zeros_like(p),
                    cy=np.array([l[0] for l in locs]), cx=np.array([l[1] for l in locs]), loc_conf=np.array([l[2] for l in locs]))
