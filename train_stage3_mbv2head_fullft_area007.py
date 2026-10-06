"""Stage3 MobileNetV2(0.35x) Head — Full Path Fine-Tuning, area007.

frozen best.pt (stage3_mbv2head_035x_frozen) 를 시작점으로
Stem~Stage3 backbone 전체를 fine-tuning한다.

frozen 스크립트와 완전히 분리된 fullft 전용 스크립트.
- backbone lr 그룹: early(Stem~S2) / stage3 / head 세 그룹
- BN 처리: prefix BN running stats 고정, head BN은 학습
- pos_weight: 기본 없음, --pos-weight 로 활성화

실행 예:
python train_stage3_mbv2head_fullft_area007.py \
  --fastvit-root ../../ \
  --frozen-checkpoint ../../checkpoints/split_gate_check/stage3_mbv2head_035x_frozen_area007_seed42_v1/best.pt \
  --train ../../split_gate/person_manifests_area007/train.json \
  --dev   ../../split_gate/person_manifests_area007/dev.json \
  --test  ../../split_gate/person_manifests_area007/test.json \
  --output ../../checkpoints/split_gate_check/stage3_mbv2head_035x_fullft_area007_seed42_v2 \
  --epochs 20 \
  --batch-size 32 \
  --img-size 256 \
  --head-lr 0.045 \
  --lr-decay 0.98 \
  --early-lr 0.000003 \
  --stage3-lr 0.00001 \
  --weight-decay 0.00004 \
  --dropout 0.2 \
  --width-mult 0.35 \
  --target-recall 0.99 \
  --seed 42 \
  --workers 4 \
  --device cuda
"""
import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


# ── 전처리 ────────────────────────────────────────────────────────────

class Letterbox:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        scale = self.size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        image = image.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (124, 116, 104))
        canvas.paste(image, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


class PersonDataset(Dataset):
    def __init__(self, manifest: str, size: int, training: bool = False):
        data = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.policy   = data["policy"]
        self.all_rows = data["samples"]
        self.rows = (
            [r for r in self.all_rows if r["label"] >= 0]
            if training else self.all_rows
        )
        ops = [Letterbox(size)]
        if training:
            ops.append(transforms.RandomHorizontalFlip())
        ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        with Image.open(r["path"]) as im:
            if im.size != (r["width"], r["height"]):
                raise ValueError(f"Image size mismatch: {r['path']}")
            x = self.transform(im.convert("RGB"))
        return x, torch.tensor(r["label"], dtype=torch.float32), i


# ── MobileNetV2(0.35x) Head ───────────────────────────────────────────

def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, expand_ratio: int = 6):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden_ch = _make_divisible(in_ch * expand_ratio)
        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden_ch, hidden_ch, 3, stride=stride,
                      padding=1, groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.use_residual else self.conv(x)


class MBV2Head(nn.Module):
    def __init__(self, in_channels: int = 256,
                 width_mult: float = 0.35, dropout: float = 0.2):
        super().__init__()
        c1      = _make_divisible(in_channels * width_mult)
        last_ch = _make_divisible(1280 * width_mult)
        self.features = nn.Sequential(
            InvertedResidual(in_channels, c1,  stride=1, expand_ratio=1),
            InvertedResidual(c1,          c1,  stride=1, expand_ratio=6),
            nn.Conv2d(c1, last_ch, 1, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU6(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_ch, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x).squeeze(1)


# ── FullFT 전용 모델 ──────────────────────────────────────────────────

class Stage3GateFullFT(nn.Module):
    """
    frozen best.pt 를 로드해서 backbone 전체를 fine-tuning하는 전용 모델.

    frozen 스크립트와 분리된 이유:
    - requires_grad 제어를 init 시점에 명시적으로 처리
    - train() 메서드에서 prefix BN만 eval() 고정, head BN은 학습
    - forward에서 gradient 차단 없이 prefix 전체 연산
    """
    def __init__(self, backbone, head: MBV2Head):
        super().__init__()
        # prefix: Stem~Stage3
        self.stem    = backbone.patch_embed
        self.stage1  = backbone.network[0]
        self.ds1     = backbone.network[1]   # Stage1→2 downsample
        self.stage2  = backbone.network[2]
        self.ds2     = backbone.network[3]   # Stage2→3 downsample
        self.stage3  = backbone.network[4]
        self.head    = head

        # 모든 파라미터 학습 가능으로 설정
        # (frozen best.pt 로드 후 덮어씌워도 requires_grad=True 유지)
        for p in self.parameters():
            p.requires_grad_(True)

    def _prefix_bn_eval(self):
        """prefix(backbone) BN의 running stats 고정. head BN은 학습 유지."""
        for m in [self.stem, self.stage1, self.ds1,
                  self.stage2, self.ds2, self.stage3]:
            for layer in m.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._prefix_bn_eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gradient 차단 없이 전체 경로 실행
        x = self.stem(x)
        x = self.stage1(x)
        x = self.ds1(x)
        x = self.stage2(x)
        x = self.ds2(x)
        x = self.stage3(x)
        if x.shape[1] != 256:
            raise RuntimeError(f"Expected 256ch from Stage3, got {x.shape[1]}")
        return self.head(x)

    def param_groups(self, early_lr: float, stage3_lr: float,
                     head_lr: float) -> list:
        """lr 그룹 반환: early(Stem~S2) / stage3 / head"""
        early_params = (
            list(self.stem.parameters()) +
            list(self.stage1.parameters()) +
            list(self.ds1.parameters()) +
            list(self.stage2.parameters()) +
            list(self.ds2.parameters())
        )
        return [
            {"params": early_params,                  "lr": early_lr},
            {"params": list(self.stage3.parameters()), "lr": stage3_lr},
            {"params": list(self.head.parameters()),   "lr": head_lr},
        ]


# ── Checkpoint 로드 ────────────────────────────────────────────────────

def load_frozen_checkpoint(model: Stage3GateFullFT,
                           frozen_ckpt_path: str,
                           img_size: int) -> None:
    """
    frozen best.pt 에서 가중치를 로드.
    prefix + head 전체를 복원한 뒤 requires_grad를 True로 강제 설정.
    """
    ckpt = torch.load(frozen_ckpt_path, map_location="cpu", weights_only=True)

    # 기본 검증
    if ckpt.get("stage") != 3:
        raise ValueError(f"stage=3 checkpoint 필요. got {ckpt.get('stage')}")
    if ckpt.get("img_size") != img_size:
        raise ValueError(
            f"img_size 불일치: ckpt={ckpt.get('img_size')}, arg={img_size}"
        )

    saved = ckpt["model"]   # prefix.* / head.* 키

    # Stage3GateFullFT는 prefix를 stem/stage1/ds1/stage2/ds2/stage3로 분리
    # frozen 스크립트는 prefix.0~5 키로 저장 → 매핑 필요
    prefix_map = {
        "prefix.0.": "stem.",
        "prefix.1.": "stage1.",
        "prefix.2.": "ds1.",
        "prefix.3.": "stage2.",
        "prefix.4.": "ds2.",
        "prefix.5.": "stage3.",
    }
    remapped = {}
    for k, v in saved.items():
        new_k = k
        for old_prefix, new_prefix in prefix_map.items():
            if k.startswith(old_prefix):
                new_k = new_prefix + k[len(old_prefix):]
                break
        remapped[new_k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        # head 구조가 같으면 missing 없어야 함
        print(f"⚠ missing keys: {missing[:5]}", flush=True)
    if unexpected:
        print(f"⚠ unexpected keys: {unexpected[:5]}", flush=True)

    # requires_grad를 명시적으로 True로 강제
    # (frozen 저장 당시 requires_grad 상태와 무관하게 전체 학습 가능)
    for p in model.parameters():
        p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Loaded from frozen ckpt. "
          f"Trainable: {trainable:,} / Total: {total:,} "
          f"(모두 학습 가능 — fullft)", flush=True)
    assert trainable == total, "일부 파라미터가 frozen 상태. 코드 확인 필요"


# ── Manifest 검증 ──────────────────────────────────────────────────────

def validate_manifests(paths: dict) -> dict:
    datasets, seen_paths, seen_ids, policy = {}, set(), set(), None
    for split, path in paths.items():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        pol  = data["policy"]
        if (pol.get("label_rule") != "person_size_trigger_v1"
                or pol.get("size_metric") != "area"
                or not math.isclose(float(pol.get("min_ratio", -1)), 0.07)):
            raise ValueError(f"{split}: area007 manifest 필요. got {pol}")
        if policy is not None and pol != policy:
            raise ValueError("Train/dev/test policies differ")
        policy = pol
        rows   = data["samples"]
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"{split}: label 0과 1 모두 필요")
        for r in rows:
            key = str(Path(r["path"]).resolve())
            iid = str(r["image_id"])
            if key in seen_paths or iid in seen_ids:
                raise ValueError(f"Duplicate: {key}")
            seen_paths.add(key)
            seen_ids.add(iid)
            if not Path(key).is_file():
                raise FileNotFoundError(key)
            if r["group"] not in {
                "target_person", "below_size_person_only", "no_person"}:
                raise ValueError(f"Unexpected group: {r['group']}")
            if int(r["label"]) != int(r["group"] == "target_person"):
                raise ValueError("Label/group mismatch")
        datasets[split] = {
            "count":  len(rows),
            "labels": dict(Counter(r["label"] for r in rows)),
            "groups": dict(Counter(r["group"] for r in rows)),
        }
    return datasets


# ── 지표 ──────────────────────────────────────────────────────────────

def select_threshold(y, scores, target_recall):
    positives = sorted(s for t, s in zip(y, scores) if t == 1)
    if not positives or not any(t == 0 for t in y):
        raise ValueError("Both labels required")
    needed = math.ceil(target_recall * len(positives))
    return positives[len(positives) - needed]


def compute_metrics(y, scores, threshold):
    pairs = [(int(t), s >= threshold) for t, s in zip(y, scores) if t >= 0]
    tp = sum(t == 1 and p     for t, p in pairs)
    fn = sum(t == 1 and not p for t, p in pairs)
    tn = sum(t == 0 and not p for t, p in pairs)
    fp = sum(t == 0 and p     for t, p in pairs)
    def sd(a, b): return a / b if b else None
    return dict(
        threshold=threshold, tp=tp, fn=fn, tn=tn, fp=fp,
        recall=sd(tp, tp+fn), negative_rejection=sd(tn, tn+fp),
        precision=sd(tp, tp+fp), accuracy=sd(tp+tn, len(pairs)),
        deep_pass_rate_labeled=sd(tp+fp, len(pairs)),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, scores = [], []
    for x, y, _ in loader:
        s = model(x.to(device)).sigmoid().cpu().tolist()
        ys.extend(y.tolist())
        scores.extend(s if isinstance(s, list) else [s])
    if not all(math.isfinite(s) for s in scores):
        raise RuntimeError("Non-finite score")
    return ys, scores


def save_report(dest, dataset, y, scores, threshold, split, epoch):
    dest.mkdir(parents=True, exist_ok=True)
    report = dict(
        split=split, stage=3, head_type="mbv2_035x_fullft",
        checkpoint_epoch=epoch,
        label_definition="at least one valid person bbox area >= 7%",
        policy=dataset.policy,
        labeled=compute_metrics(y, scores, threshold),
        groups={},
    )
    for group in sorted({r["group"] for r in dataset.rows}):
        idx    = [i for i, r in enumerate(dataset.rows) if r["group"] == group]
        gy, gs = [y[i] for i in idx], [scores[i] for i in idx]
        passed = sum(s >= threshold for s in gs)
        report["groups"][group] = dict(
            count=len(idx), passed=passed, exited=len(idx)-passed,
            pass_rate=passed/len(idx) if idx else 0,
            exit_rate=1-passed/len(idx) if idx else 0,
            metrics=compute_metrics(gy, gs, threshold),
        )
    report["all_images_pass_rate"] = (
        sum(s >= threshold for s in scores) / len(scores)
    )
    Path(dest / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (dest / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "image_id", "label", "group",
                         "max_person_area", "score", "threshold", "pass", "outcome"])
        for r, score in zip(dataset.rows, scores):
            passed  = score >= threshold
            outcome = ("TP" if passed else "FN") if r["label"] == 1 \
                      else ("FP" if passed else "TN")
            writer.writerow([r["path"], r["image_id"], r["label"], r["group"],
                             r.get("max_person_area"), score, threshold,
                             int(passed), outcome])
    print(split, json.dumps(report["labeled"]), flush=True)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Stage3 MBV2(0.35x) Head — Full Path Fine-Tuning, area007"
    )
    for name in ("fastvit-root", "frozen-checkpoint",
                 "train", "dev", "test", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--img-size",      type=int,   default=256)
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--width-mult",    type=float, default=0.35)
    p.add_argument("--dropout",       type=float, default=0.2)
    p.add_argument("--head-lr",       type=float, default=0.045)
    p.add_argument("--lr-decay",      type=float, default=0.98)
    p.add_argument("--early-lr",      type=float, default=3e-6)
    p.add_argument("--stage3-lr",     type=float, default=1e-5)
    p.add_argument("--weight-decay",  type=float, default=0.00004)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--pos-weight",    type=float, default=None,
                   help="BCE pos_weight. 기본 없음. 불균형 보정 시 1.738 권장.")
    p.add_argument("--target-recall", type=float, default=0.99)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--log-every",     type=int,   default=200)
    a = p.parse_args()

    if not 0 < a.target_recall <= 1:
        p.error("--target-recall must be in (0, 1]")
    if min(a.epochs, a.batch_size, a.img_size) < 1:
        p.error("epochs/batch-size/img-size must be >= 1")

    dest = Path(a.output)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    # ── Manifest 검증 ─────────────────────────────────────────────────
    manifest_info = validate_manifests(
        dict(train=a.train, dev=a.dev, test=a.test)
    )

    # ── 데이터 통계 ────────────────────────────────────────────────────
    train_data   = json.loads(Path(a.train).read_text(encoding="utf-8"))
    label_counts = Counter(r["label"] for r in train_data["samples"])
    n_pos = label_counts[1]
    n_neg = label_counts[0]
    print(f"Train labels — pos: {n_pos}, neg: {n_neg}, "
          f"neg/pos ratio: {n_neg/n_pos:.4f}", flush=True)

    # ── FastViT 로드 (구조만, 가중치는 frozen ckpt에서) ───────────────
    sys.path.insert(0, str(Path(a.fastvit_root).resolve()))
    from models.fastvit import fastvit_sa12
    backbone = fastvit_sa12(pretrained=False, inference_mode=False)

    # ── 모델 조립 ──────────────────────────────────────────────────────
    head  = MBV2Head(in_channels=256, width_mult=a.width_mult,
                     dropout=a.dropout)
    model = Stage3GateFullFT(backbone, head)
    del backbone

    load_frozen_checkpoint(model, a.frozen_checkpoint, a.img_size)
    model = model.to(device)

    # ── Dataloader ─────────────────────────────────────────────────────
    train_ds = PersonDataset(a.train, a.img_size, training=True)
    dev_ds   = PersonDataset(a.dev,   a.img_size)

    def make_loader(ds, shuffle=False):
        return DataLoader(
            ds, batch_size=a.batch_size, shuffle=shuffle,
            num_workers=a.workers, pin_memory=(device.type == "cuda"),
        )

    train_loader = make_loader(train_ds, shuffle=True)
    dev_loader   = make_loader(dev_ds)

    # ── Config 저장 ────────────────────────────────────────────────────
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    config = dict(
        vars(a),
        stage=3, head_type="mbv2_035x_fullft",
        frozen_checkpoint=str(Path(a.frozen_checkpoint).resolve()),
        trainable_parameters=trainable,
        total_parameters=total,
        manifests=manifest_info,
        policy=train_ds.policy,
    )
    Path(dest / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Trainable: {trainable:,} / Total: {total:,}", flush=True)

    # ── Optimizer ──────────────────────────────────────────────────────
    groups = model.param_groups(a.early_lr, a.stage3_lr, a.head_lr)
    optimizer = torch.optim.RMSprop(
        groups, alpha=0.9, momentum=0.9, weight_decay=a.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=a.lr_decay
    )

    # ── Loss ───────────────────────────────────────────────────────────
    if a.pos_weight is not None:
        pw = torch.tensor(float(a.pos_weight), device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        print(f"Loss: BCEWithLogitsLoss(pos_weight={a.pos_weight})", flush=True)
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Loss: BCEWithLogitsLoss (pos_weight 없음)", flush=True)

    # ── Best 저장 ──────────────────────────────────────────────────────
    def save_best(epoch, threshold, result):
        torch.save(dict(
            model={k: v.detach().cpu() for k, v in model.state_dict().items()},
            threshold=threshold, epoch=epoch,
            policy=train_ds.policy, img_size=a.img_size,
            stage=3, head_type="mbv2_035x_fullft",
            width_mult=a.width_mult, dropout=a.dropout,
            config=config, dev_metrics=result,
        ), dest / "best.pt")

    # ── Epoch 0 베이스라인 ─────────────────────────────────────────────
    y_dev, scores_dev = evaluate(model, dev_loader, device)
    threshold = select_threshold(y_dev, scores_dev, a.target_recall)
    baseline  = compute_metrics(y_dev, scores_dev, threshold)
    baseline.update(epoch=0, train_loss=None)
    history  = [baseline]
    best_tn  = baseline["tn"]
    save_best(0, threshold, baseline)
    print(f"Epoch 0 baseline: {json.dumps(baseline)}", flush=True)
    Path(dest / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    # ── 학습 루프 ──────────────────────────────────────────────────────
    for epoch in range(1, a.epochs + 1):
        model.train()   # prefix BN eval() 고정, head BN 학습
        total_loss = 0.0

        for step, (x, y, _) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss   = criterion(logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: epoch {epoch} step {step}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                a.grad_clip, error_if_nonfinite=True,
            )
            optimizer.step()
            total_loss += loss.item() * len(y)

            if step % a.log_every == 0:
                lr_now = optimizer.param_groups[2]["lr"]  # head lr
                print(f"epoch {epoch}/{a.epochs} "
                      f"step {step}/{len(train_loader)} "
                      f"loss={loss.item():.6f} head_lr={lr_now:.2e}",
                      flush=True)

        scheduler.step()

        y_dev, scores_dev = evaluate(model, dev_loader, device)
        threshold = select_threshold(y_dev, scores_dev, a.target_recall)
        result    = compute_metrics(y_dev, scores_dev, threshold)
        result.update(epoch=epoch, train_loss=total_loss / len(train_ds))
        history.append(result)
        print(json.dumps(result), flush=True)

        if result["tn"] > best_tn:
            best_tn = result["tn"]
            save_best(epoch, threshold, result)

        Path(dest / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    # ── 최종 평가 ──────────────────────────────────────────────────────
    saved = torch.load(dest / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    model = model.to(device)

    Path(dest / "selection.json").write_text(
        json.dumps(dict(
            best_epoch=saved["epoch"],
            threshold=saved["threshold"],
            target_recall=a.target_recall,
            dev_metrics=saved["dev_metrics"],
            initial_dev_metrics=baseline,
            improved_on_dev=saved["dev_metrics"]["tn"] > baseline["tn"],
            test_used_for_selection=False,
        ), indent=2),
        encoding="utf-8",
    )

    y_dev, scores_dev = evaluate(model, dev_loader, device)
    save_report(dest / "dev", dev_ds, y_dev, scores_dev,
                saved["threshold"], "dev", saved["epoch"])

    test_ds = PersonDataset(a.test, a.img_size)
    y_test, scores_test = evaluate(model, make_loader(test_ds), device)
    save_report(dest / "test", test_ds, y_test, scores_test,
                saved["threshold"], "test", saved["epoch"])

    torch.save({
        k: saved[k] for k in (
            "model", "threshold", "epoch", "policy",
            "img_size", "stage", "head_type", "width_mult", "dropout",
        )
    }, dest / "person_stage3_mbv2_fullft_unit.pt")

    Path(dest / "completed.json").write_text(
        json.dumps(dict(
            stage=3, head_type="mbv2_035x_fullft",
            width_mult=a.width_mult, min_ratio=0.07,
            seed=a.seed, epochs=a.epochs,
        ), indent=2),
        encoding="utf-8",
    )
    print(f"Done → {dest}", flush=True)


if __name__ == "__main__":
    main()
