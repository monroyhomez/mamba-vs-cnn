"""Bucle de entrenamiento/evaluación. Implementa los controles que corrigen los
bugs documentados del pipeline anterior:
  - métrica de checkpoint ÚNICA y consistente (cfg['monitor'])
  - todas las métricas persistidas a CSV por época (no solo stdout)
  - validación de que train_loss != 0
  - AMP bfloat16, grad clip, EMA, label smoothing, cosine+warmup
  - mixup/cutmix opcional (activado por dataset desde la config)
"""
import copy
import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import classification_metrics


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def _cosine_warmup(step, total_steps, warmup_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


def _mixup_cutmix(x, y, num_classes, mixup, cutmix):
    """Devuelve (x, target_soft). Aplica mixup O cutmix por batch si están activos."""
    y1 = F.one_hot(y, num_classes).float()
    if mixup <= 0 and cutmix <= 0:
        return x, y1
    perm = torch.randperm(x.size(0), device=x.device)
    use_cutmix = cutmix > 0 and (mixup <= 0 or torch.rand(1).item() < 0.5)
    if use_cutmix:
        lam = np.random.beta(cutmix, cutmix)
        h, w = x.shape[2:]
        rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = np.random.randint(h), np.random.randint(w)
        y0, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
        x0, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
        x[:, :, y0:y2, x0:x2] = x[perm, :, y0:y2, x0:x2]
        lam = 1 - ((y2 - y0) * (x2 - x0) / (h * w))
    else:
        lam = np.random.beta(mixup, mixup)
        x = lam * x + (1 - lam) * x[perm]
    target = lam * y1 + (1 - lam) * y1[perm]
    return x, target


class Trainer:
    def __init__(self, model, cfg, device, out_dir):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        t = cfg["train"]
        self.amp_dtype = torch.bfloat16 if t["amp_dtype"] == "bfloat16" else torch.float16
        self.opt = torch.optim.AdamW(
            model.parameters(), lr=cfg["optimizer"]["lr"],
            weight_decay=cfg["optimizer"]["weight_decay"], betas=tuple(cfg["optimizer"]["betas"]))
        self.ema = EMA(model, t["ema_decay"]) if t["ema"] else None
        self.num_classes = cfg["num_classes"]
        self.monitor = cfg["monitor"]
        self.csv_path = os.path.join(out_dir, "training_log.csv")
        self._csv_header_written = False

    def _log_csv(self, row):
        write_header = not self._csv_header_written
        with open(self.csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
                self._csv_header_written = True
            w.writerow(row)

    def train_one_epoch(self, loader, epoch, total_steps, warmup_steps, global_step):
        self.model.train()
        t = self.cfg["train"]
        aug = self.cfg["augment"]
        base_lr, min_lr = self.cfg["optimizer"]["lr"], self.cfg["scheduler"]["min_lr"]
        running = 0.0
        n = 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            lr = _cosine_warmup(global_step, total_steps, warmup_steps, base_lr, min_lr)
            for g in self.opt.param_groups:
                g["lr"] = lr
            xm, target = _mixup_cutmix(x, y, self.num_classes, aug["mixup"], aug["cutmix"])
            self.opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=self.amp_dtype):
                logits = self.model(xm)
                logp = F.log_softmax(logits, dim=1)
                # soft target + label smoothing
                ls = t["label_smoothing"]
                target = target * (1 - ls) + ls / self.num_classes
                loss = -(target * logp).sum(1).mean()
            loss.backward()
            if t["grad_clip"]:
                nn.utils.clip_grad_norm_(self.model.parameters(), t["grad_clip"])
            self.opt.step()
            if self.ema:
                self.ema.update(self.model)
            running += loss.item() * x.size(0)
            n += x.size(0)
            global_step += 1
        train_loss = running / max(1, n)
        assert train_loss > 0, "train_loss == 0: revisar el bucle (bug previo)."
        return train_loss, global_step, lr

    @torch.no_grad()
    def evaluate(self, loader, prefix="val", use_ema=True):
        model = self.ema.shadow if (self.ema and use_ema) else self.model
        model.eval()
        probs, gts, loss_sum, n = [], [], 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=self.amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
            gts.append(y.cpu().numpy())
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
        probs = np.concatenate(probs)
        gts = np.concatenate(gts)
        m = classification_metrics(gts, probs, self.num_classes, prefix=prefix)
        m[f"{prefix}_loss"] = loss_sum / max(1, n)
        return m

    def fit(self, train_loader, val_loader):
        t = self.cfg["train"]
        total_steps = t["epochs"] * len(train_loader)
        warmup_steps = self.cfg["scheduler"]["warmup_epochs"] * len(train_loader)
        best, best_epoch, no_improve, global_step = -1e9, -1, 0, 0
        for epoch in range(t["epochs"]):
            train_loss, global_step, lr = self.train_one_epoch(
                train_loader, epoch, total_steps, warmup_steps, global_step)
            val = self.evaluate(val_loader, "val")                 # métrica EMA (la monitorizada)
            if self.ema:                                            # además, métrica de pesos vivos (raw)
                val.update(self.evaluate(val_loader, "val_raw", use_ema=False))
            row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, **val}
            self._log_csv(row)
            # Monitor/checkpoint sobre pesos RAW (vivos), NO EMA: con fine-tuning corto el
            # EMA arranca frío y, si el modelo converge y hace early-stop pronto (típico en
            # CNN), el monitor EMA queda congelado en ~azar y corta el run prematuramente.
            mon = self.monitor.replace("val_", "val_raw_", 1) if self.ema else self.monitor
            score = val[mon]
            print(f"ep{epoch:03d} loss={train_loss:.4f} {self.monitor}={score:.4f}")
            if score > best:
                best, best_epoch, no_improve = score, epoch, 0
                ckpt = {"model": (self.ema.shadow if self.ema else self.model).state_dict(),
                        "epoch": epoch, "metric": score}
                if self.ema:                                        # guarda también los pesos vivos (no-EMA)
                    ckpt["model_raw"] = self.model.state_dict()
                torch.save(ckpt, os.path.join(self.out_dir, "best.pth"))
            else:
                no_improve += 1
                if no_improve >= t["early_stop_patience"]:
                    print(f"Early stop en epoch {epoch} (mejor={best_epoch}, {self.monitor}={best:.4f})")
                    break
        return {"best_epoch": best_epoch, f"best_{self.monitor}": best}
