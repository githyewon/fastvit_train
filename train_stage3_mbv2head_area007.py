"""Stage3 MobileNetV2(0.35x) Head — area007 manifest, frozen or fullft backbone.

MobileNetV2(0.35x) head를 FastViT Stage3 뒤에 배치.
VWW 논문 기준으로 설계하되 area007 데이터 불균형(1.74:1)을 pos_weight로 보정.

학습 모드:
  --mode frozen   : backbone 완전 frozen, head만 학습
  --mode fullft   : backbone(Stem~Stage3) 전체 fine-tuning

실행 예 (frozen):
python train_stage3_mbv2head_area007.py \
  --fastvit-root ../../ \
  --checkpoint ../../weights/fastvit_sa12.pth.tar \
  --train ../../split_gate/person_manifests_area007/train.json \
  --dev   ../../split_gate/person_manifests_area007/dev.json \
  --test  ../../split_gate/person_manifests_area007/test.json \
  --output ../../checkpoints/split_gate_check/stage3_mbv2head_035x_frozen_area007_seed42_v1 \
  --mode frozen \
  --epochs 30 \
  --batch-size 96 \
  --img-size 256 \
  --head-lr 0.045 \
  --lr-decay 0.98 \
  --weight-decay 0.00004 \
  --dropout 0.2 \
  --width-mult 0.35 \
  --target-recall 0.99 \
  --seed 42 \
  --workers 4 \
  --device cuda

실행 예 (pos weight):
python train_stage3_mbv2head_area007.py \
  --fastvit-root ../../ \
  --checkpoint ../../weights/fastvit_sa12.pth.tar \
  --train ../../split_gate/person_manifests_area007/train.json \
  --dev   ../../split_gate/person_manifests_area007/dev.json \
  --test  ../../split_gate/person_manifests_area007/test.json \
  --output ../../checkpoints/split_gate_check/stage3_mbv2head_035x_frozen_posweight_area007_seed42_v1 \
  --pos-weight 1.738 \
  --mode frozen \
  --epochs 30 \
  --batch-size 96 \
  --img-size 256 \
  --head-lr 0.045 \
  --lr-decay 0.98 \
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
import torch.nn.functional as F
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
    """
    MobileNetV2 Inverted Residual Block.

    1×1 Conv(expand) → 3×3 DWConv → 1×1 Conv(project, no activation)
    expand_ratio=1 이면 expansion 단계 생략 (첫 블록 관례).
    stride=1 + in_ch==out_ch 일 때만 residual 연결.
    """
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
    """
    FastViT Stage3 feature (B, 256, H, W) → 이진 person logit.

    MobileNetV2(0.35x) 방식:
      in_ch=256  →  c1=88(=256×0.35 divisor8)
      InvertedResidual(expand=1) → InvertedResidual(expand=6)
      → Conv1×1 → BN → ReLU6 → last_ch=448(=1280×0.35)
      → GAP → Dropout(0.2) → Linear(448, 1)

    기존 GAP+MLP(33K) 대비: Depthwise Conv가 공간 패턴을 처리한 뒤
    GAP으로 압축하므로 bbox 크기 정보가 더 잘 보존됨.
    파라미터: 약 120K
    """
    def __init__(
        self,
        in_channels: int = 256,
        width_mult:  float = 0.35,
        dropout:     float = 0.2,
    ):
        super().__init__()
        c1      = _make_divisible(in_channels * width_mult)   # 88
        last_ch = _make_divisible(1280 * width_mult)           # 448

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

        # 가중치 초기화 (MobileNetV2 관례)
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

        # bias: MobileNetV2 오리지널 기본값 0 유지
        # (log-odds prior 초기화는 MobileNetV2 설계에 없음)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (B, last_ch, H, W)
        x = self.pool(x).flatten(1)   # (B, last_ch)
        return self.classifier(x).squeeze(1)   # (B,)


# ── Stage3Gate (frozen / fullft 공용) ─────────────────────────────────

class Stage3Gate(nn.Module):
    """
    prefix (Stem~Stage3) + MBV2Head.

    mode="frozen": prefix requires_grad=False, head만 학습
    mode="fullft": prefix도 학습 (lr 그룹별 분리)
    """
    def __init__(self, backbone, head: MBV2Head, mode: str = "frozen"):
        super().__init__()
        self.prefix = nn.Sequential(
            backbone.patch_embed,    # Stem
            backbone.network[0],     # Stage1
            backbone.network[1],     # downsample
            backbone.network[2],     # Stage2
            backbone.network[3],     # downsample
            backbone.network[4],     # Stage3
        )
        self.head = head
        self.mode = mode
        if mode == "frozen":
            self.prefix.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # prefix BN running stats 항상 고정
        for m in self.prefix.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        # head BN은 학습 모드 유지 (head 안의 InvertedResidual BN)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "frozen":
            with torch.no_grad():
                feat = self.prefix(x)
        else:
            feat = self.prefix(x)
        if feat.shape[1] != 256:
            raise RuntimeError(f"Expected 256ch from Stage3, got {feat.shape[1]}")
        return self.head(feat)


# ── Checkpoint 로드 ────────────────────────────────────────────────────

def load_checkpoint(model: Stage3Gate, path: str, mode: str) -> None:
    """
    frozen: fastvit 원본 checkpoint → prefix 가중치만 로드
    fullft: frozen 단계의 best.pt → 전체(prefix+head) 로드
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    if mode == "frozen":
        # 원본 fastvit checkpoint 또는 이전 stage3 best.pt
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        # key prefix 정리 (module./model./backbone. 제거)
        clean = {}
        for k, v in state.items():
            while any(k.startswith(p) for p in ("module.", "model.", "backbone.")):
                k = k.split(".", 1)[1]
            clean[k] = v
        # prefix.* 키로 재매핑
        needed = {
            k: v for k, v in model.state_dict().items()
            if k.startswith("prefix.")
        }
        # fastvit 원본 키(patch_embed.*, network.*) → prefix.* 매핑
        remap = {}
        prefix_parts = [
            "patch_embed", "network.0", "network.1",
            "network.2",   "network.3", "network.4",
        ]
        for i, part in enumerate(prefix_parts):
            for k in list(clean.keys()):
                if k.startswith(part + ".") or k == part:
                    new_k = f"prefix.{i}." + k[len(part)+1:]
                    remap[new_k] = clean[k]

        # best.pt 형식(이미 prefix.* 키)인 경우 그대로 사용
        prefix_from_bestpt = {
            k: v for k, v in clean.items() if k.startswith("prefix.")
        }
        source = prefix_from_bestpt if prefix_from_bestpt else remap

        missing = [k for k in needed if k not in source]
        if missing:
            raise RuntimeError(f"Backbone keys missing: {missing[:5]}")
        model.load_state_dict(source, strict=False)
        print(f"Backbone (prefix) loaded: {len(source)} keys. "
              f"Head initialized fresh (MBV2 0.35x).", flush=True)

    else:  # fullft: frozen best.pt에서 전체 로드
        if "model" not in ckpt:
            raise ValueError("fullft mode requires frozen-mode best.pt "
                             "(keys: model, threshold, ...)")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        # head 구조가 바뀐 경우 head.* 키 불일치는 허용
        non_head_missing = [k for k in missing if not k.startswith("head.")]
        if non_head_missing:
            raise RuntimeError(f"Backbone keys missing: {non_head_missing[:5]}")
        print(f"Full model loaded from frozen best.pt. "
              f"head missing keys: {[k for k in missing if k.startswith('head.')][:3]}",
              flush=True)


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
            raise ValueError("Train/dev/test label policies differ")
        policy = pol
        rows   = data["samples"]
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"{split}: label 0과 1 모두 필요")
        for r in rows:
            key      = str(Path(r["path"]).resolve())
            image_id = str(r["image_id"])
            if key in seen_paths or image_id in seen_ids:
                raise ValueError(f"Duplicate: {key}")
            seen_paths.add(key)
            seen_ids.add(image_id)
            if not Path(key).is_file():
                raise FileNotFoundError(key)
            if r["group"] not in {
                "target_person", "below_size_person_only", "no_person"
            }:
                raise ValueError(f"Unexpected group: {r['group']}")
            if int(r["label"]) != int(r["group"] == "target_person"):
                raise ValueError("Label/group mismatch")
        datasets[split] = {
            "count":  len(rows),
            "labels": dict(Counter(r["label"] for r in rows)),
            "groups": dict(Counter(r["group"] for r in rows)),
        }
    return datasets


# ── 지표 계산 ──────────────────────────────────────────────────────────

def select_threshold(y: list, scores: list, target_recall: float) -> float:
    positives = sorted(s for t, s in zip(y, scores) if t == 1)
    if not positives or not any(t == 0 for t in y):
        raise ValueError("Both labels required")
    needed = math.ceil(target_recall * len(positives))
    return positives[len(positives) - needed]


def compute_metrics(y: list, scores: list, threshold: float) -> dict:
    pairs = [(int(t), s >= threshold) for t, s in zip(y, scores) if t >= 0]
    tp = sum(t == 1 and p  for t, p in pairs)
    fn = sum(t == 1 and not p for t, p in pairs)
    tn = sum(t == 0 and not p for t, p in pairs)
    fp = sum(t == 0 and p  for t, p in pairs)
    def sd(a, b): return a / b if b else None
    return dict(
        threshold=threshold, tp=tp, fn=fn, tn=tn, fp=fp,
        recall=sd(tp, tp+fn), negative_rejection=sd(tn, tn+fp),
        precision=sd(tp, tp+fp), accuracy=sd(tp+tn, len(pairs)),
        deep_pass_rate_labeled=sd(tp+fp, len(pairs)),
    )


# ── Evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys, scores = [], []
    for x, y, _ in loader:
        s = model(x.to(device)).sigmoid().cpu().tolist()
        ys.extend(y.tolist())
        scores.extend(s if isinstance(s, list) else [s])
    if not all(math.isfinite(s) for s in scores):
        raise RuntimeError("Non-finite score")
    return ys, scores


# ── Report 저장 ────────────────────────────────────────────────────────

def save_report(dest: Path, dataset: PersonDataset, y: list,
                scores: list, threshold: float, split: str, epoch: int):
    dest.mkdir(parents=True, exist_ok=True)
    report = dict(
        split=split, stage=3, head_type="mbv2_035x",
        checkpoint_epoch=epoch,
        label_definition="at least one valid person bbox area >= 7%",
        policy=dataset.policy,
        labeled=compute_metrics(y, scores, threshold),
        groups={},
    )
    for group in sorted({r["group"] for r in dataset.rows}):
        idx  = [i for i, r in enumerate(dataset.rows) if r["group"] == group]
        gy   = [y[i] for i in idx]
        gs   = [scores[i] for i in idx]
        passed = sum(s >= threshold for s in gs)
        report["groups"][group] = dict(
            count=len(idx), passed=passed, exited=len(idx)-passed,
            pass_rate=passed/len(idx) if idx else 0,
            exit_rate=1-passed/len(idx) if idx else 0,
            metrics=compute_metrics(gy, gs, threshold),
        )
    report["all_images_pass_rate"] = sum(s >= threshold for s in scores) / len(scores)
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
        description="Stage3 MobileNetV2(0.35x) head — area007, frozen or fullft"
    )
    for name in ("fastvit-root", "checkpoint", "train", "dev", "test", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--mode",         choices=["frozen", "fullft"], default="frozen")
    p.add_argument("--img-size",     type=int,   default=256)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=96)
    p.add_argument("--workers",      type=int,   default=4)
    p.add_argument("--width-mult",   type=float, default=0.35)
    p.add_argument("--dropout",      type=float, default=0.2)
    # MobileNetV2 기준 LR 설정
    p.add_argument("--head-lr",      type=float, default=0.045,
                   help="VWW 논문 기준 initial lr")
    p.add_argument("--lr-decay",     type=float, default=0.98,
                   help="VWW 논문 기준 per-epoch decay")
    p.add_argument("--weight-decay", type=float, default=0.00004,
                   help="VWW 논문 기준 weight decay")
    # fullft 전용 backbone lr
    p.add_argument("--early-lr",     type=float, default=3e-6)
    p.add_argument("--stage3-lr",    type=float, default=1e-5)
    p.add_argument("--pos-weight",   type=float, default=None,
                   help="BCE pos_weight. None=사용 안 함(VWW 오리지널). "
                        "area007 불균형 보정 시 1.738 권장.")
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--target-recall",type=float, default=0.99)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--log-every",    type=int,   default=200)
    a = p.parse_args()

    # ── 검증 ──────────────────────────────────────────────────────────
    if not 0 < a.target_recall <= 1:
        p.error("--target-recall must be in (0, 1]")
    if min(a.epochs, a.batch_size, a.img_size) < 1:
        p.error("epochs/batch-size/img-size must be >= 1")
    if not (0.0 < a.width_mult <= 1.0):
        p.error("--width-mult must be in (0, 1]")
    if not (0.0 <= a.dropout < 1.0):
        p.error("--dropout must be in [0, 1)")

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

    # ── 데이터 통계 (manifest 실측) ────────────────────────────────────
    train_data   = json.loads(Path(a.train).read_text(encoding="utf-8"))
    label_counts = Counter(r["label"] for r in train_data["samples"])
    n_pos        = label_counts[1]
    n_neg        = label_counts[0]
    n_total      = n_pos + n_neg
    pos_weight_val = n_neg / n_pos   # ≈ 1.738 (area007 기준)
    print(f"Train labels — pos: {n_pos}, neg: {n_neg}, "
          f"neg/pos ratio: {pos_weight_val:.4f} "
          f"(pos_weight={'사용 안 함' if a.pos_weight is None else a.pos_weight})",
          flush=True)

    # ── FastViT 로드 ───────────────────────────────────────────────────
    sys.path.insert(0, str(Path(a.fastvit_root).resolve()))
    from models.fastvit import fastvit_sa12
    backbone = fastvit_sa12(pretrained=False, inference_mode=False)

    # ── 모델 조립 ──────────────────────────────────────────────────────
    head = MBV2Head(
        in_channels=256,
        width_mult=a.width_mult,
        dropout=a.dropout,
    )
    model = Stage3Gate(backbone, head, mode=a.mode)
    load_checkpoint(model, a.checkpoint, mode=a.mode)
    model = model.to(device)
    del backbone

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
        stage=3, head_type="mbv2_035x",
        architecture=(
            f"Stem-Stage3({'frozen' if a.mode=='frozen' else 'fullft'}) → "
            f"InvRes(exp=1) → InvRes(exp=6) → Conv1×1 → GAP → "
            f"Dropout({a.dropout}) → Linear(448,1)"
        ),
        pos_weight=a.pos_weight,
        bias_init="default_zero (MobileNetV2 original)",
        trainable_parameters=trainable,
        total_parameters=total,
        manifests=manifest_info,
        policy=train_ds.policy,
        lr_schedule="MobileNetV2 exponential decay (per-epoch)",
    )
    Path(dest / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Trainable: {trainable:,} / Total: {total:,}", flush=True)
    print(f"Training labels: {Counter(r['label'] for r in train_ds.rows)}",
          flush=True)

    # ── Optimizer ──────────────────────────────────────────────────────
    # VWW 논문 기준: RMSProp(decay=0.9, momentum=0.9), lr=0.045, decay 0.98/epoch
    # PyTorch RMSprop에는 momentum이 없으므로 Adam으로 근사
    # (VWW TF 구현에서도 Adam 사용 가능)
    if a.mode == "frozen":
        optimizer = torch.optim.RMSprop(
            [p for p in model.parameters() if p.requires_grad],
            lr=a.head_lr,
            alpha=0.9,          # decay
            momentum=0.9,
            weight_decay=a.weight_decay,
        )
    else:  # fullft: 파라미터 그룹별 lr
        early  = list(model.prefix[:5].parameters())   # Stem~S2 downsample
        stage3 = list(model.prefix[5].parameters())    # Stage3
        head_p = list(model.head.parameters())
        optimizer = torch.optim.RMSprop([
            {"params": early,  "lr": a.early_lr},
            {"params": stage3, "lr": a.stage3_lr},
            {"params": head_p, "lr": a.head_lr},
        ], alpha=0.9, momentum=0.9, weight_decay=a.weight_decay)

    # 매 epoch마다 lr × decay (VWW 논문 기준)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=a.lr_decay
    )

    # ── Loss: BCE + pos_weight ─────────────────────────────────────────
    # pos_weight 설정
    # --pos-weight 없음(None): VWW 오리지널 그대로, 순수 MBV2 head 효과 확인
    # --pos-weight 1.738:    area007 불균형(36:64) 보정 버전
    if a.pos_weight is not None:
        pw = torch.tensor(float(a.pos_weight), device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        print(f"Loss: BCEWithLogitsLoss(pos_weight={a.pos_weight})", flush=True)
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Loss: BCEWithLogitsLoss (pos_weight 없음, VWW 오리지널)", flush=True)

    # ── Best 저장 함수 ─────────────────────────────────────────────────
    def save_best(epoch: int, threshold: float, result: dict):
        torch.save(
            dict(
                model={k: v.detach().cpu()
                       for k, v in model.state_dict().items()},
                threshold=threshold, epoch=epoch,
                policy=train_ds.policy, img_size=a.img_size,
                stage=3, head_type="mbv2_035x",
                width_mult=a.width_mult, dropout=a.dropout,
                config=config, dev_metrics=result,
            ),
            dest / "best.pt",
        )

    # ── Epoch 0 베이스라인 ─────────────────────────────────────────────
    y_dev, scores_dev = evaluate(model, dev_loader, device)
    threshold = select_threshold(y_dev, scores_dev, a.target_recall)
    baseline  = compute_metrics(y_dev, scores_dev, threshold)
    baseline.update(epoch=0, train_loss=None)
    history   = [baseline]
    best_tn   = baseline["tn"]
    save_best(0, threshold, baseline)
    print(f"Epoch 0 baseline: {json.dumps(baseline)}", flush=True)
    Path(dest / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    # ── 학습 루프 ──────────────────────────────────────────────────────
    for epoch in range(1, a.epochs + 1):
        model.train()
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
                lr_now = optimizer.param_groups[-1]["lr"]   # head lr
                print(f"epoch {epoch}/{a.epochs} step {step}/{len(train_loader)} "
                      f"loss={loss.item():.6f} lr={lr_now:.2e}", flush=True)

        scheduler.step()   # per-epoch lr decay

        # ── Epoch 평가 ────────────────────────────────────────────────
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

    # Dev 리포트
    y_dev, scores_dev = evaluate(model, dev_loader, device)
    save_report(dest / "dev", dev_ds, y_dev, scores_dev,
                saved["threshold"], "dev", saved["epoch"])

    # Test 리포트
    test_ds     = PersonDataset(a.test, a.img_size)
    test_loader = make_loader(test_ds)
    y_test, scores_test = evaluate(model, test_loader, device)
    save_report(dest / "test", test_ds, y_test, scores_test,
                saved["threshold"], "test", saved["epoch"])

    # Unit 저장
    unit = {k: saved[k] for k in (
        "model", "threshold", "epoch", "policy",
        "img_size", "stage", "head_type", "width_mult", "dropout",
    )}
    unit["format"] = "stage3_area007_mbv2head_unit_v1"
    torch.save(unit, dest / "person_stage3_mbv2_unit.pt")

    Path(dest / "completed.json").write_text(
        json.dumps(dict(
            stage=3, head_type="mbv2_035x", width_mult=a.width_mult,
            min_ratio=0.07, mode=a.mode, seed=a.seed, epochs=a.epochs,
        ), indent=2),
        encoding="utf-8",
    )
    print(f"Done → {dest}", flush=True)


if __name__ == "__main__":
    main()
