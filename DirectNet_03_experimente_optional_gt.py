# -*- coding: utf-8 -*-
"""
DirectNet — experimente complete, cu ground truth opțional.

Fișierul HSI .mat trebuie să fie în același folder cu scriptul.
Ground truth-ul NU este obligatoriu. Dacă există în fișierul .mat, codul îl
folosește pentru ROC-AUC, PR-AUC, Precision, Recall, F1, FP și FN. Dacă nu
există, codul continuă normal și produce hărți de scor, predicții pe percentile,
analiza pragurilor și exemple cu cele mai puternice anomalii candidate.
"""

import copy
import gc
import random
import shutil
import time
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset


# =============================================================================
# 1. CONFIGURARE
# =============================================================================

SEED = 42
QUICK_MODE = False

# Fișierul trebuie să fie în același folder cu scriptul.
DATA_FILENAME = "pavia.mat"

# None = alege automat cel mai mare array numeric 3D din fișier.
CUBE_KEY = None

# "auto" sau una dintre: "HWB", "BHW", "HBW", "WHB", "WBH", "BWH".
# Dacă orientarea automată greșește, setează manual forma din fișier.
CUBE_LAYOUT = "auto"

# Dacă există o hartă 2D cu puține valori și aceeași dimensiune spațială,
# poate fi folosită automat ca ground truth. Poți pune False pentru a o ignora.
USE_GROUND_TRUTH_IF_AVAILABLE = True
GROUND_TRUTH_KEY = None

# Fără ground truth, pragul binar este ales prin percentila scorurilor.
DEFAULT_THRESHOLD_PERCENTILE = 99.5

if QUICK_MODE:
    MAX_EPOCHS = 35
    TRAIN_SAMPLES_PER_EPOCH = 5000
    VALIDATION_SAMPLES = 1800
    PATIENCE = 7
else:
    MAX_EPOCHS = 80
    TRAIN_SAMPLES_PER_EPOCH = 10000
    VALIDATION_SAMPLES = 2500
    PATIENCE = 12

BATCH_SIZE = 100
INFERENCE_BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
HIDDEN_CHANNELS = 64

WIN_VALUES = [1, 3, 5, 7, 9]
WOUT_VALUES = [15, 19, 23]
REFERENCE_WOUT = 19

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / DATA_FILENAME
OUTPUT_DIR = SCRIPT_DIR / "directnet_experiments"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch:", torch.__version__)
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
print("Quick mode:", QUICK_MODE)


# =============================================================================
# 2. ÎNCĂRCAREA CUBULUI HSI; GROUND TRUTH-UL ESTE OPȚIONAL
# =============================================================================

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Nu am găsit fișierul local: {DATA_PATH}\n"
        f"Copiază {DATA_FILENAME} în același folder cu scriptul sau schimbă "
        "variabila DATA_FILENAME."
    )


def numeric_arrays_from_mat(path):
    mat_data = loadmat(path)
    return {
        key: np.asarray(value)
        for key, value in mat_data.items()
        if not key.startswith("__")
        and isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.number)
    }


def apply_layout(cube, layout):
    layout = layout.upper()
    if sorted(layout) != ["B", "H", "W"] or len(layout) != 3:
        raise ValueError("CUBE_LAYOUT trebuie să fie auto sau o permutare a literelor H, W, B.")
    permutation = [layout.index("H"), layout.index("W"), layout.index("B")]
    return np.transpose(cube, permutation)


def ground_truth_likelihood(array):
    """Scor mic = candidat mai plauzibil pentru o hartă de etichete."""
    squeezed = np.squeeze(array)
    if squeezed.ndim != 2:
        return None
    unique = np.unique(squeezed)
    if unique.size < 2 or unique.size > 32:
        return None
    integer_penalty = 0 if np.allclose(unique, np.round(unique)) else 100
    binary_penalty = 0 if set(unique.tolist()).issubset({0, 1}) else 10
    return integer_penalty + binary_penalty + unique.size


def find_gt_for_shape(arrays, spatial_shape, excluded_key=None, forced_key=None):
    if forced_key is not None:
        if forced_key not in arrays:
            raise KeyError(f"GROUND_TRUTH_KEY='{forced_key}' nu există în fișierul .mat.")
        raw = np.squeeze(arrays[forced_key])
        if raw.shape == spatial_shape:
            return forced_key, raw
        if raw.T.shape == spatial_shape:
            return forced_key, raw.T
        raise ValueError(
            f"Ground truth-ul {forced_key} are forma {raw.shape}, dar imaginea are {spatial_shape}."
        )

    candidates = []
    for key, value in arrays.items():
        if key == excluded_key:
            continue
        score = ground_truth_likelihood(value)
        if score is None:
            continue
        raw = np.squeeze(value)
        if raw.shape == spatial_shape:
            candidates.append((score, key, raw))
        elif raw.T.shape == spatial_shape:
            candidates.append((score + 1, key, raw.T))
    if not candidates:
        return None
    _, key, raw = min(candidates, key=lambda item: item[0])
    return key, raw


def orient_cube_and_find_gt(cube, arrays, cube_key):
    # Orientare manuală.
    if CUBE_LAYOUT.lower() != "auto":
        oriented = apply_layout(cube, CUBE_LAYOUT)
        gt_match = None
        if USE_GROUND_TRUTH_IF_AVAILABLE:
            gt_match = find_gt_for_shape(
                arrays,
                oriented.shape[:2],
                excluded_key=cube_key,
                forced_key=GROUND_TRUTH_KEY,
            )
        return oriented, gt_match, f"manual: {CUBE_LAYOUT}"

    # Dacă există ground truth, îl putem folosi pentru a identifica axele H și W.
    if USE_GROUND_TRUTH_IF_AVAILABLE:
        for perm in permutations(range(3)):
            candidate = np.transpose(cube, perm)
            gt_match = find_gt_for_shape(
                arrays,
                candidate.shape[:2],
                excluded_key=cube_key,
                forced_key=GROUND_TRUTH_KEY,
            )
            if gt_match is not None:
                return candidate, gt_match, f"automat prin potrivire cu {gt_match[0]}"

    # Fără ground truth: presupunem că axa spectrală este axa cea mai mică.
    # Pentru cuburi neobișnuite, setează CUBE_LAYOUT manual.
    spectral_axis = int(np.argmin(cube.shape))
    oriented = np.moveaxis(cube, spectral_axis, -1)
    return oriented, None, f"automat fără GT; axa spectrală presupusă={spectral_axis}"


arrays = numeric_arrays_from_mat(DATA_PATH)
cube_candidates = {
    key: value
    for key, value in arrays.items()
    if value.ndim == 3
}
if not cube_candidates:
    raise RuntimeError("Fișierul .mat nu conține niciun array numeric 3D.")

if CUBE_KEY is not None:
    if CUBE_KEY not in cube_candidates:
        raise KeyError(f"CUBE_KEY='{CUBE_KEY}' nu există sau nu este un array numeric 3D.")
    cube_key, cube_raw = CUBE_KEY, cube_candidates[CUBE_KEY]
else:
    cube_key, cube_raw = max(cube_candidates.items(), key=lambda item: item[1].size)

hsi, gt_match, orientation_info = orient_cube_and_find_gt(cube_raw, arrays, cube_key)
hsi = np.asarray(hsi, dtype=np.float32)
hsi = np.nan_to_num(hsi, nan=0.0, posinf=0.0, neginf=0.0)

H, W, B0 = hsi.shape
band_std_raw = hsi.reshape(-1, B0).std(axis=0)
valid_bands = band_std_raw > 1e-8
if not np.any(valid_bands):
    raise RuntimeError("Toate benzile au variație nulă.")
hsi = hsi[:, :, valid_bands]
H, W, B = hsi.shape

pixels = hsi.reshape(-1, B)
band_mean = pixels.mean(axis=0, keepdims=True)
band_std = np.maximum(pixels.std(axis=0, keepdims=True), 1e-8)
pixels_z = (pixels - band_mean) / band_std
hsi_z = pixels_z.reshape(H, W, B).astype(np.float32)

HAS_GT = gt_match is not None
if HAS_GT:
    gt_key, gt = gt_match
    gt = np.asarray(gt)
    unique_gt = set(np.unique(gt).tolist())
    gt_binary = gt.astype(np.uint8) if unique_gt.issubset({0, 1}) else (gt > 0).astype(np.uint8)
    y_true = gt_binary.ravel()
else:
    gt_key = None
    gt_binary = None
    y_true = None

print("Fișier:", DATA_PATH)
print("Cheie HSI:", cube_key)
print("Formă inițială:", cube_raw.shape)
print("Orientare:", orientation_info)
print("HSI final:", hsi_z.shape)
print("Ground truth disponibil:", HAS_GT)
if HAS_GT:
    print("Cheie GT:", gt_key)
    print("Anomalii GT:", int(gt_binary.sum()))
    print("Procent anomalie GT:", 100 * gt_binary.mean(), "%")
else:
    print(
        "Nu s-a găsit ground truth. Metricile supravegheate vor fi omise; "
        "anomaliile vor fi selectate prin percentile ale scorului."
    )


# =============================================================================
# 3. PSEUDOCOLOR PCA
# =============================================================================

def percentile_stretch(image, low=2, high=98):
    output = np.zeros_like(image, dtype=np.float32)
    for channel in range(image.shape[-1]):
        band = image[..., channel]
        lo, hi = np.percentile(band, [low, high])
        if hi > lo:
            output[..., channel] = np.clip((band - lo) / (hi - lo), 0, 1)
    return output


pca = PCA(n_components=3, random_state=SEED)
pca_rgb = pca.fit_transform(pixels_z).reshape(H, W, 3)
pca_rgb = percentile_stretch(pca_rgb)

plt.figure(figsize=(7, 7))
plt.imshow(pca_rgb)
plt.title("Pseudocolor PCA")
plt.axis("off")
plt.savefig(FIGURE_DIR / "pseudocolor_pca.png", dpi=200, bbox_inches="tight")
plt.close()

if HAS_GT:
    plt.figure(figsize=(7, 7))
    plt.imshow(gt_binary)
    plt.title("Ground truth")
    plt.axis("off")
    plt.savefig(FIGURE_DIR / "ground_truth.png", dpi=200, bbox_inches="tight")
    plt.close()


# =============================================================================
# 4. METRICI ȘI REZUMAT NESUPRAVEGHEAT
# =============================================================================

def supervised_metrics(y, scores):
    roc_auc = roc_auc_score(y, scores)
    average_precision = average_precision_score(y, scores)
    fpr, tpr, thresholds = roc_curve(y, scores)
    best_index = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[best_index])
    prediction = (scores >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "threshold": threshold,
        "threshold_method": "Youden",
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "prediction": prediction,
        "flagged_pixels": int(prediction.sum()),
        "flagged_percent": float(100 * prediction.mean()),
        "fpr": fpr,
        "tpr": tpr,
    }


def unsupervised_summary(scores, percentile=DEFAULT_THRESHOLD_PERCENTILE):
    scores = np.asarray(scores, dtype=np.float64)
    q25, median, q75, p95, p99, p995 = np.percentile(
        scores, [25, 50, 75, 95, 99, 99.5]
    )
    iqr = max(q75 - q25, 1e-12)
    threshold = float(np.percentile(scores, percentile))
    prediction = (scores >= threshold).astype(np.uint8)
    return {
        "median_score": float(median),
        "p95_score": float(p95),
        "p99_score": float(p99),
        "p99_5_score": float(p995),
        "max_score": float(scores.max()),
        "tail_contrast": float((p995 - median) / iqr),
        "threshold": threshold,
        "threshold_method": f"percentile_{percentile}",
        "prediction": prediction,
        "flagged_pixels": int(prediction.sum()),
        "flagged_percent": float(100 * prediction.mean()),
    }


def evaluate_scores(scores):
    unsup = unsupervised_summary(scores)
    if HAS_GT:
        sup = supervised_metrics(y_true, scores)
        return {**unsup, **sup}
    return {
        **unsup,
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "tp": np.nan,
        "tn": np.nan,
        "fp": np.nan,
        "fn": np.nan,
        "fpr": None,
        "tpr": None,
    }


# =============================================================================
# 5. BASELINE GRX
# =============================================================================

def global_rx_scores(pixels_standardized, regularization=1e-3):
    mean_vector = pixels_standardized.mean(axis=0, keepdims=True)
    centered = pixels_standardized - mean_vector
    covariance = np.atleast_2d(np.cov(centered, rowvar=False))
    scale = np.trace(covariance) / covariance.shape[0]
    covariance_regularized = covariance + regularization * scale * np.eye(covariance.shape[0])
    inverse_covariance = np.linalg.pinv(covariance_regularized)
    return np.einsum("ij,jk,ik->i", centered, inverse_covariance, centered, optimize=True)


grx_start = time.perf_counter()
grx_scores_flat = global_rx_scores(pixels_z)
grx_time = time.perf_counter() - grx_start
grx_eval = evaluate_scores(grx_scores_flat)
grx_score_map = grx_scores_flat.reshape(H, W)
grx_prediction_map = grx_eval["prediction"].reshape(H, W)

np.save(OUTPUT_DIR / "grx_scores.npy", grx_scores_flat)
np.save(OUTPUT_DIR / "grx_score_map.npy", grx_score_map)
np.save(OUTPUT_DIR / "grx_prediction_map.npy", grx_prediction_map)

plt.figure(figsize=(7, 7))
plt.imshow(grx_score_map)
plt.title("Harta scorurilor GRX")
plt.axis("off")
plt.savefig(FIGURE_DIR / "grx_score_map.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 7))
plt.imshow(grx_prediction_map)
plt.title(f"Predicție GRX — {grx_eval['threshold_method']}")
plt.axis("off")
plt.savefig(FIGURE_DIR / "grx_binary_prediction.png", dpi=200, bbox_inches="tight")
plt.close()

print("GRX timp:", grx_time, "s")
if HAS_GT:
    print("GRX ROC-AUC:", grx_eval["roc_auc"])
    print("GRX AP / PR-AUC:", grx_eval["average_precision"])
else:
    print("GRX pixeli marcați:", grx_eval["flagged_pixels"])


# =============================================================================
# 6. DATASET BLIND-BLOCK ȘI ARHITECTURA DIRECTNET
# =============================================================================

class BlindBlockPatchDataset(Dataset):
    def __init__(self, hsi_hwc, wout, win, deterministic=False, seed=42):
        if hsi_hwc.ndim != 3:
            raise ValueError("HSI trebuie să aibă forma H x W x B.")
        if wout % 2 != 1 or win % 2 != 1:
            raise ValueError("Wout și Win trebuie să fie impare.")
        if win > wout:
            raise ValueError("Win nu poate fi mai mare decât Wout.")

        self.hsi = torch.from_numpy(hsi_hwc).float()
        self.h, self.w, self.b = hsi_hwc.shape
        self.wout = int(wout)
        self.win = int(win)
        self.pad = self.wout // 2
        self.center = self.wout // 2
        self.deterministic = bool(deterministic)
        self.seed = int(seed)

        chw = self.hsi.permute(2, 0, 1).unsqueeze(0)
        self.padded = F.pad(
            chw, (self.pad, self.pad, self.pad, self.pad), mode="reflect"
        ).squeeze(0)

        inner_half = self.win // 2
        inner_coords, outer_coords = [], []
        for row in range(self.wout):
            for col in range(self.wout):
                in_inner = (
                    abs(row - self.center) <= inner_half
                    and abs(col - self.center) <= inner_half
                )
                (inner_coords if in_inner else outer_coords).append((row, col))

        self.inner_coords = torch.tensor(inner_coords, dtype=torch.long)
        self.outer_coords = torch.tensor(outer_coords, dtype=torch.long)

    def __len__(self):
        return self.h * self.w

    def __getitem__(self, index):
        row = index // self.w
        col = index % self.w
        patch = self.padded[
            :, row:row + self.wout, col:col + self.wout
        ].clone()
        target = self.hsi[row, col].clone()

        generator = None
        if self.deterministic:
            generator = torch.Generator()
            generator.manual_seed(self.seed + int(index))

        selected_ids = torch.randint(
            0,
            len(self.outer_coords),
            (len(self.inner_coords),),
            generator=generator,
        )
        source_coords = self.outer_coords[selected_ids]
        patch[
            :, self.inner_coords[:, 0], self.inner_coords[:, 1]
        ] = patch[:, source_coords[:, 0], source_coords[:, 1]]
        return patch, target, int(index)


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return x + residual


class DirectNet(nn.Module):
    def __init__(self, bands, wout, hidden_channels=64):
        super().__init__()
        if (wout - 7) % 4 != 0:
            raise ValueError("Wout trebuie să satisfacă Wout = 4*Nr + 7.")
        self.center = wout // 2
        self.n_res_blocks = (wout - 7) // 4
        self.conv_in = nn.Conv2d(bands, hidden_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(self.n_res_blocks)]
        )
        self.conv_penultimate = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.bn_penultimate = nn.BatchNorm2d(hidden_channels)
        self.conv_out = nn.Conv2d(hidden_channels, bands, 3, padding=1)

    def forward(self, x):
        low_level = F.relu(self.conv_in(x), inplace=True)
        high_level = self.blocks(low_level)
        high_level = self.bn_penultimate(self.conv_penultimate(high_level))
        reconstructed_patch = self.conv_out(low_level + high_level)
        return reconstructed_patch[:, :, self.center, self.center]


# =============================================================================
# 7. TRAIN / VALIDATION ȘI ANTRENARE
# =============================================================================

all_indices = np.arange(H * W)
split_rng = np.random.default_rng(SEED)
split_rng.shuffle(all_indices)
n_validation = min(VALIDATION_SAMPLES, max(1, int(0.1 * len(all_indices))))
validation_indices = all_indices[:n_validation]
training_indices = all_indices[n_validation:]

print("Train:", len(training_indices))
print("Validation:", len(validation_indices))


def train_and_evaluate_config(wout, win, seed=SEED, verbose=True):
    if win > wout:
        raise ValueError("Win trebuie să fie <= Wout.")
    if (wout - 7) % 4 != 0:
        raise ValueError("Wout trebuie să fie de forma 4*Nr+7.")

    set_seed(seed)
    config_name = f"wout{wout}_win{win}"
    checkpoint_path = CHECKPOINT_DIR / f"{config_name}.pt"

    train_base = BlindBlockPatchDataset(hsi_z, wout, win, deterministic=False, seed=seed)
    val_base = BlindBlockPatchDataset(hsi_z, wout, win, deterministic=True, seed=seed + 10000)
    train_subset = Subset(train_base, training_indices.tolist())
    val_subset = Subset(val_base, validation_indices.tolist())

    sampler_generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(
        train_subset,
        replacement=True,
        num_samples=TRAIN_SAMPLES_PER_EPOCH,
        generator=sampler_generator,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = DirectNet(B, wout, HIDDEN_CHANNELS).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    best_state = None
    best_val = float("inf")
    wait = 0
    train_history, val_history = [], []
    start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_sum, train_count = 0.0, 0
        for patches, targets, _ in train_loader:
            patches = patches.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(patches)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_sum += loss.item() * patches.shape[0]
            train_count += patches.shape[0]
        train_loss = train_sum / train_count

        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for patches, targets, _ in val_loader:
                patches = patches.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                loss = criterion(model(patches), targets)
                val_sum += loss.item() * patches.shape[0]
                val_count += patches.shape[0]
        val_loss = val_sum / val_count
        scheduler.step(val_loss)
        train_history.append(train_loss)
        val_history.append(val_loss)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if verbose:
            print(
                f"{config_name} | epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"wait={wait}/{PATIENCE}"
            )
        if wait >= PATIENCE:
            print("Early stopping.")
            break

    training_time = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError(f"Nu s-a salvat un model valid pentru {config_name}.")

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": best_state,
            "wout": wout,
            "win": win,
            "bands": B,
            "best_val_loss": best_val,
            "train_history": train_history,
            "val_history": val_history,
            "band_mean": band_mean,
            "band_std": band_std,
        },
        checkpoint_path,
    )

    eval_dataset = BlindBlockPatchDataset(
        hsi_z, wout, win, deterministic=True, seed=seed + 20000
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    reconstructed = np.zeros((H * W, B), dtype=np.float32)
    model.eval()
    inference_start = time.perf_counter()
    with torch.no_grad():
        for patches, _, indices in eval_loader:
            predictions = model(patches.to(DEVICE, non_blocking=True)).cpu().numpy()
            reconstructed[indices.numpy()] = predictions
    inference_time = time.perf_counter() - inference_start

    scores = np.linalg.norm(hsi_z.reshape(-1, B) - reconstructed, axis=1)
    evaluation = evaluate_scores(scores)

    row = {
        "config": config_name,
        "wout": int(wout),
        "win": int(win),
        "n_res_blocks": int((wout - 7) // 4),
        "epochs_ran": len(train_history),
        "best_val_loss": float(best_val),
        "roc_auc": evaluation["roc_auc"],
        "average_precision": evaluation["average_precision"],
        "precision": evaluation["precision"],
        "recall": evaluation["recall"],
        "f1": evaluation["f1"],
        "tp": evaluation["tp"],
        "tn": evaluation["tn"],
        "fp": evaluation["fp"],
        "fn": evaluation["fn"],
        "median_score": evaluation["median_score"],
        "p95_score": evaluation["p95_score"],
        "p99_score": evaluation["p99_score"],
        "p99_5_score": evaluation["p99_5_score"],
        "max_score": evaluation["max_score"],
        "tail_contrast": evaluation["tail_contrast"],
        "threshold": evaluation["threshold"],
        "threshold_method": evaluation["threshold_method"],
        "flagged_pixels": evaluation["flagged_pixels"],
        "flagged_percent": evaluation["flagged_percent"],
        "training_time_s": float(training_time),
        "inference_time_s": float(inference_time),
        "checkpoint": str(checkpoint_path),
    }

    artifacts = {
        "scores": scores,
        "score_map": scores.reshape(H, W),
        "prediction": evaluation["prediction"],
        "prediction_map": evaluation["prediction"].reshape(H, W),
        "fpr": evaluation["fpr"],
        "tpr": evaluation["tpr"],
        "train_history": np.asarray(train_history),
        "val_history": np.asarray(val_history),
        "reconstructed": reconstructed,
    }

    del model, train_base, val_base, eval_dataset
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return row, artifacts


# =============================================================================
# 8. SWEEP WIN
# =============================================================================

win_results = []
win_artifacts = {}

for win in WIN_VALUES:
    print("\n" + "=" * 80)
    row, artifacts = train_and_evaluate_config(REFERENCE_WOUT, win)
    win_results.append(row)
    win_artifacts[row["config"]] = artifacts

win_df = pd.DataFrame(win_results).sort_values("win").reset_index(drop=True)
print(win_df.to_string(index=False))
win_df.to_csv(OUTPUT_DIR / "win_sweep_results.csv", index=False)

if HAS_GT:
    plt.figure(figsize=(8, 5))
    plt.plot(win_df["win"], win_df["roc_auc"], marker="o")
    plt.xlabel("Win")
    plt.ylabel("ROC-AUC")
    plt.title("Impactul Win asupra ROC-AUC")
    plt.grid(alpha=0.3)
    plt.savefig(FIGURE_DIR / "win_vs_roc_auc.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(win_df["win"], win_df["average_precision"], marker="o")
    plt.xlabel("Win")
    plt.ylabel("Average Precision / PR-AUC")
    plt.title("Impactul Win asupra PR-AUC")
    plt.grid(alpha=0.3)
    plt.savefig(FIGURE_DIR / "win_vs_pr_auc.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(win_df["win"], win_df["fp"], marker="o", label="FP")
    plt.plot(win_df["win"], win_df["fn"], marker="o", label="FN")
    plt.xlabel("Win")
    plt.ylabel("Număr pixeli")
    plt.title("FP și FN în funcție de Win")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(FIGURE_DIR / "win_vs_fp_fn.png", dpi=200, bbox_inches="tight")
    plt.close()
else:
    plt.figure(figsize=(8, 5))
    plt.plot(win_df["win"], win_df["tail_contrast"], marker="o")
    plt.xlabel("Win")
    plt.ylabel("Contrast robust al cozii superioare")
    plt.title("Separarea nesupravegheată a scorurilor în funcție de Win")
    plt.grid(alpha=0.3)
    plt.savefig(FIGURE_DIR / "win_vs_tail_contrast.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(win_df["win"], win_df["best_val_loss"], marker="o")
    plt.xlabel("Win")
    plt.ylabel("Best validation L1 loss")
    plt.title("Eroarea de validare în funcție de Win")
    plt.grid(alpha=0.3)
    plt.savefig(FIGURE_DIR / "win_vs_validation_loss.png", dpi=200, bbox_inches="tight")
    plt.close()


# =============================================================================
# 9. SWEEP WOUT
# =============================================================================

if HAS_GT:
    best_win_row = win_df.loc[win_df["average_precision"].idxmax()]
    selection_reason = "Average Precision"
else:
    best_win_row = win_df.loc[win_df["tail_contrast"].idxmax()]
    selection_reason = "contrastul robust al cozii scorurilor"

BEST_WIN = int(best_win_row["win"])
print(f"Cel mai bun Win după {selection_reason}:", BEST_WIN)

wout_results = []
wout_artifacts = {}

for wout in WOUT_VALUES:
    config_name = f"wout{wout}_win{BEST_WIN}"
    if config_name in win_artifacts:
        existing = win_df[win_df["config"] == config_name].iloc[0].to_dict()
        wout_results.append(existing)
        wout_artifacts[config_name] = win_artifacts[config_name]
        print("Refolosesc:", config_name)
    else:
        print("\n" + "=" * 80)
        row, artifacts = train_and_evaluate_config(wout, BEST_WIN)
        wout_results.append(row)
        wout_artifacts[row["config"]] = artifacts

wout_df = pd.DataFrame(wout_results).sort_values("wout").reset_index(drop=True)
print(wout_df.to_string(index=False))
wout_df.to_csv(OUTPUT_DIR / "wout_sweep_results.csv", index=False)

if HAS_GT:
    plt.figure(figsize=(8, 5))
    plt.plot(wout_df["wout"], wout_df["roc_auc"], marker="o")
    plt.xlabel("Wout")
    plt.ylabel("ROC-AUC")
    plt.title("Impactul Wout asupra ROC-AUC")
    plt.grid(alpha=0.3)
    plt.savefig(FIGURE_DIR / "wout_vs_roc_auc.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(wout_df["wout"], wout_df["average_precision"], marker="o")
    plt.xlabel("Wout")
    plt.ylabel("Average Precision / PR-AUC")
    plt.title("Impactul Wout asupra PR-AUC")
    plt.grid(alpha=0.3)
    plt.savefig(FIGURE_DIR / "wout_vs_pr_auc.png", dpi=200, bbox_inches="tight")
    plt.close()
else:
    plt.figure(figsize=(8, 5))
    plt.plot(wout_df["wout"], wout_df["tail_contrast"], marker="o")
    plt.xlabel("Wout")
    plt.ylabel("Contrast robust al cozii superioare")
    plt.title("Separarea nesupravegheată a scorurilor în funcție de Wout")
    plt.grid(alpha=0.3)
    plt.savefig(FIGURE_DIR / "wout_vs_tail_contrast.png", dpi=200, bbox_inches="tight")
    plt.close()

plt.figure(figsize=(8, 5))
plt.plot(wout_df["wout"], wout_df["inference_time_s"], marker="o")
plt.xlabel("Wout")
plt.ylabel("Timp inferență (s)")
plt.title("Costul de inferență în funcție de Wout")
plt.grid(alpha=0.3)
plt.savefig(FIGURE_DIR / "wout_vs_inference_time.png", dpi=200, bbox_inches="tight")
plt.close()


# =============================================================================
# 10. CONFIGURAȚIA FINALĂ
# =============================================================================

all_results_df = (
    pd.concat([win_df, wout_df], ignore_index=True)
    .drop_duplicates("config")
    .reset_index(drop=True)
)

if HAS_GT:
    best_row = all_results_df.loc[all_results_df["average_precision"].idxmax()]
    final_selection_metric = "Average Precision"
else:
    best_row = all_results_df.loc[all_results_df["tail_contrast"].idxmax()]
    final_selection_metric = "contrastul robust al cozii scorurilor"

BEST_CONFIG = str(best_row["config"])
all_results_df.to_csv(OUTPUT_DIR / "all_experiment_results.csv", index=False)

all_artifacts = {**win_artifacts, **wout_artifacts}
best_artifacts = all_artifacts[BEST_CONFIG]
best_scores = best_artifacts["scores"]
best_score_map = best_artifacts["score_map"]
best_prediction = best_artifacts["prediction"]
best_prediction_map = best_artifacts["prediction_map"]

np.save(OUTPUT_DIR / "best_scores.npy", best_scores)
np.save(OUTPUT_DIR / "best_score_map.npy", best_score_map)
np.save(OUTPUT_DIR / "best_prediction_map.npy", best_prediction_map)
np.save(OUTPUT_DIR / "best_reconstructed_spectra.npy", best_artifacts["reconstructed"])

print(f"Configurația finală după {final_selection_metric}:", BEST_CONFIG)
print(best_row)

plt.figure(figsize=(8, 5))
plt.plot(best_artifacts["train_history"], label="Train")
plt.plot(best_artifacts["val_history"], label="Validation")
plt.xlabel("Epocă")
plt.ylabel("L1 loss")
plt.title(f"Curbele de antrenare — {BEST_CONFIG}")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig(FIGURE_DIR / "best_training_curves.png", dpi=200, bbox_inches="tight")
plt.close()


# =============================================================================
# 11. COMPARAȚIA GRX VS DIRECTNET
# =============================================================================

if HAS_GT:
    best_fpr, best_tpr, _ = roc_curve(y_true, best_scores)
    grx_fpr, grx_tpr, _ = roc_curve(y_true, grx_scores_flat)

    plt.figure(figsize=(8, 6))
    plt.plot(grx_fpr, grx_tpr, label=f"GRX — AUC={grx_eval['roc_auc']:.4f}")
    plt.plot(best_fpr, best_tpr, label=f"{BEST_CONFIG} — AUC={best_row['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], "--", label="Aleator")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Comparația ROC finală")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(FIGURE_DIR / "final_roc_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    grx_precision_curve, grx_recall_curve, _ = precision_recall_curve(
        y_true, grx_scores_flat
    )
    best_precision_curve, best_recall_curve, _ = precision_recall_curve(
        y_true, best_scores
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        grx_recall_curve,
        grx_precision_curve,
        label=f"GRX — AP={grx_eval['average_precision']:.4f}",
    )
    plt.plot(
        best_recall_curve,
        best_precision_curve,
        label=f"{BEST_CONFIG} — AP={best_row['average_precision']:.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Comparația Precision–Recall")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(FIGURE_DIR / "final_precision_recall_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
else:
    # Scările GRX și DirectNet sunt diferite, deci le normalizăm robust la mediană.
    grx_normalized = grx_scores_flat / (np.median(grx_scores_flat) + 1e-12)
    best_normalized = best_scores / (np.median(best_scores) + 1e-12)
    upper = max(
        np.percentile(grx_normalized, 99.5),
        np.percentile(best_normalized, 99.5),
    )

    plt.figure(figsize=(8, 6))
    plt.hist(
        grx_normalized,
        bins=100,
        range=(0, upper),
        density=True,
        alpha=0.5,
        label="GRX / mediană",
    )
    plt.hist(
        best_normalized,
        bins=100,
        range=(0, upper),
        density=True,
        alpha=0.5,
        label=f"{BEST_CONFIG} / mediană",
    )
    plt.xlabel("Scor normalizat")
    plt.ylabel("Densitate")
    plt.title("Distribuția scorurilor fără ground truth")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        FIGURE_DIR / "final_score_distribution_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# 12. HĂRȚILE FINALE ȘI DISTRIBUȚIA SCORURILOR
# =============================================================================

score_display = (
    best_score_map - best_score_map.min()
) / (best_score_map.max() - best_score_map.min() + 1e-12)

plt.figure(figsize=(7, 7))
plt.imshow(score_display)
plt.title(f"Harta de anomalii — {BEST_CONFIG}")
plt.axis("off")
plt.savefig(FIGURE_DIR / "best_anomaly_map.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 7))
plt.imshow(best_prediction_map)
plt.title(f"Predicție binară — {best_row['threshold_method']}")
plt.axis("off")
plt.savefig(FIGURE_DIR / "best_binary_prediction.png", dpi=200, bbox_inches="tight")
plt.close()

if HAS_GT:
    background_scores = best_scores[y_true == 0]
    anomaly_scores = best_scores[y_true == 1]

    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [background_scores, anomaly_scores],
        tick_labels=["Fundal", "Anomalie"],
        showfliers=False,
    )
    plt.ylabel("Scor de anomalie")
    plt.title(f"Separabilitatea — {BEST_CONFIG}")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(
        FIGURE_DIR / "best_separability_boxplot.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
else:
    threshold = float(best_row["threshold"])
    upper = np.percentile(best_scores, 99.9)
    plt.figure(figsize=(8, 5))
    plt.hist(best_scores, bins=120, range=(0, upper))
    plt.axvline(threshold, linestyle="--", label=f"Prag={threshold:.4f}")
    plt.xlabel("Scor de anomalie")
    plt.ylabel("Număr pixeli")
    plt.title(f"Distribuția scorurilor — {BEST_CONFIG}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        FIGURE_DIR / "best_score_histogram.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


# =============================================================================
# 13. ANALIZA PRAGURILOR
# =============================================================================

percentiles_to_test = [95, 97, 98, 99, 99.25, 99.5, 99.7, 99.8, 99.9]
threshold_rows = []

for percentile in percentiles_to_test:
    threshold = float(np.percentile(best_scores, percentile))
    prediction = (best_scores >= threshold).astype(np.uint8)
    row = {
        "percentile": percentile,
        "threshold": threshold,
        "flagged_pixels": int(prediction.sum()),
        "flagged_percent": float(100 * prediction.mean()),
    }

    if HAS_GT:
        tn, fp, fn, tp = confusion_matrix(
            y_true, prediction, labels=[0, 1]
        ).ravel()
        row.update({
            "precision": precision_score(y_true, prediction, zero_division=0),
            "recall": recall_score(y_true, prediction, zero_division=0),
            "f1": f1_score(y_true, prediction, zero_division=0),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        })
    else:
        selected_scores = best_scores[prediction == 1]
        row.update({
            "mean_selected_score": float(selected_scores.mean()),
            "median_selected_score": float(np.median(selected_scores)),
        })

    threshold_rows.append(row)

threshold_df = pd.DataFrame(threshold_rows)
print(threshold_df.to_string(index=False))
threshold_df.to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)

if HAS_GT:
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["percentile"], threshold_df["precision"], marker="o", label="Precision")
    plt.plot(threshold_df["percentile"], threshold_df["recall"], marker="o", label="Recall")
    plt.plot(threshold_df["percentile"], threshold_df["f1"], marker="o", label="F1")
    plt.xlabel("Percentila pragului")
    plt.ylabel("Scor")
    plt.title("Compromisul Precision–Recall")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(
        FIGURE_DIR / "threshold_precision_recall_f1.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["percentile"], threshold_df["fp"], marker="o", label="FP")
    plt.plot(threshold_df["percentile"], threshold_df["fn"], marker="o", label="FN")
    plt.xlabel("Percentila pragului")
    plt.ylabel("Număr pixeli")
    plt.title("Impactul pragului asupra FP și FN")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(FIGURE_DIR / "threshold_fp_fn.png", dpi=200, bbox_inches="tight")
    plt.close()
else:
    plt.figure(figsize=(8, 5))
    plt.plot(
        threshold_df["percentile"],
        threshold_df["threshold"],
        marker="o",
    )
    plt.xlabel("Percentila pragului")
    plt.ylabel("Valoarea threshold-ului")
    plt.title("Threshold-ul scorului în funcție de percentilă")
    plt.grid(alpha=0.3)
    plt.savefig(
        FIGURE_DIR / "threshold_value_by_percentile.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # Salvăm câteva hărți alternative pentru alegerea vizuală a pragului.
    for percentile in [99, 99.5, 99.9]:
        threshold = np.percentile(best_scores, percentile)
        prediction_map = (best_scores >= threshold).reshape(H, W)
        plt.figure(figsize=(7, 7))
        plt.imshow(prediction_map)
        plt.title(f"Predicție la percentila {percentile}")
        plt.axis("off")
        safe_name = str(percentile).replace(".", "_")
        plt.savefig(
            FIGURE_DIR / f"prediction_percentile_{safe_name}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()


# =============================================================================
# 14. CAZURI FP/FN SAU CELE MAI PUTERNICE ANOMALII CANDIDATE
# =============================================================================

def normalize_grayscale(array):
    array = np.asarray(array, dtype=np.float32)
    lo, hi = float(array.min()), float(array.max())
    return np.zeros_like(array) if hi <= lo else (array - lo) / (hi - lo)


def crop_bounds(row, col, radius):
    return (
        max(0, row - radius),
        min(H, row + radius + 1),
        max(0, col - radius),
        min(W, col + radius + 1),
    )


def mark_pixel_with_cross(image, row, col):
    output = image.copy()
    r0, r1 = max(0, row - 2), min(output.shape[0], row + 3)
    c0, c1 = max(0, col - 2), min(output.shape[1], col + 3)
    output[row, c0:c1] = 1.0
    output[r0:r1, col] = 1.0
    return output


def build_case_panel(flat_index, case_name, crop_radius=10):
    row, col = int(flat_index // W), int(flat_index % W)
    r0, r1, c0, c1 = crop_bounds(row, col, crop_radius)
    local_row, local_col = row - r0, col - c0

    pca_crop = mark_pixel_with_cross(
        pca_rgb[r0:r1, c0:c1], local_row, local_col
    )
    score_crop = normalize_grayscale(best_score_map[r0:r1, c0:c1])
    score_rgb = np.repeat(score_crop[..., None], 3, axis=2)
    score_rgb = mark_pixel_with_cross(score_rgb, local_row, local_col)
    pred_rgb = np.repeat(
        best_prediction_map[r0:r1, c0:c1, None].astype(np.float32),
        3,
        axis=2,
    )
    separator = np.ones((pca_crop.shape[0], 2, 3), dtype=np.float32)

    if HAS_GT:
        gt_rgb = np.repeat(
            gt_binary[r0:r1, c0:c1, None].astype(np.float32),
            3,
            axis=2,
        )
        panel = np.concatenate(
            [pca_crop, separator, score_rgb, separator, gt_rgb, separator, pred_rgb],
            axis=1,
        )
        title = (
            f"{case_name} | coord=({row}, {col}) | score={best_scores[flat_index]:.4f} | "
            f"GT={int(y_true[flat_index])} | pred={int(best_prediction[flat_index])}\n"
            "Pseudocolor | Scor | Ground truth | Predicție"
        )
    else:
        panel = np.concatenate(
            [pca_crop, separator, score_rgb, separator, pred_rgb],
            axis=1,
        )
        title = (
            f"{case_name} | coord=({row}, {col}) | score={best_scores[flat_index]:.4f} | "
            f"pred={int(best_prediction[flat_index])}\n"
            "Pseudocolor | Scor local | Predicție"
        )

    return panel, title


if HAS_GT:
    fp_indices = np.where((y_true == 0) & (best_prediction == 1))[0]
    fn_indices = np.where((y_true == 1) & (best_prediction == 0))[0]
    selected_fp = fp_indices[np.argsort(best_scores[fp_indices])[::-1]][:3]
    selected_fn = list(fn_indices[np.argsort(best_scores[fn_indices])[::-1]][:2])
    if len(fn_indices) >= 3:
        hardest_fn = int(fn_indices[np.argmin(best_scores[fn_indices])])
        if hardest_fn not in selected_fn:
            selected_fn.append(hardest_fn)

    for number, flat_index in enumerate(selected_fp, start=1):
        panel, title = build_case_panel(int(flat_index), f"False Positive {number}")
        plt.figure(figsize=(14, 4))
        plt.imshow(panel)
        plt.title(title)
        plt.axis("off")
        plt.savefig(
            FIGURE_DIR / f"false_positive_{number}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    for number, flat_index in enumerate(selected_fn, start=1):
        panel, title = build_case_panel(int(flat_index), f"False Negative {number}")
        plt.figure(figsize=(14, 4))
        plt.imshow(panel)
        plt.title(title)
        plt.axis("off")
        plt.savefig(
            FIGURE_DIR / f"false_negative_{number}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()
else:
    # Fără GT nu putem spune FP sau FN. Arătăm cele mai mari scoruri.
    top_indices = np.argsort(best_scores)[::-1][:6]
    candidate_rows = []
    for number, flat_index in enumerate(top_indices, start=1):
        row, col = divmod(int(flat_index), W)
        candidate_rows.append({
            "rank": number,
            "flat_index": int(flat_index),
            "row": row,
            "col": col,
            "score": float(best_scores[flat_index]),
            "predicted_anomaly": int(best_prediction[flat_index]),
        })
        panel, title = build_case_panel(
            int(flat_index), f"Anomalie candidată {number}"
        )
        plt.figure(figsize=(11, 4))
        plt.imshow(panel)
        plt.title(title)
        plt.axis("off")
        plt.savefig(
            FIGURE_DIR / f"top_anomaly_candidate_{number}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    pd.DataFrame(candidate_rows).to_csv(
        OUTPUT_DIR / "top_anomaly_candidates.csv", index=False
    )


# =============================================================================
# 15. TABEL FINAL ȘI ARHIVĂ ZIP
# =============================================================================

grx_row = {
    "config": "GRX",
    "wout": np.nan,
    "win": np.nan,
    "n_res_blocks": 0,
    "epochs_ran": 0,
    "best_val_loss": np.nan,
    "roc_auc": grx_eval["roc_auc"],
    "average_precision": grx_eval["average_precision"],
    "precision": grx_eval["precision"],
    "recall": grx_eval["recall"],
    "f1": grx_eval["f1"],
    "tp": grx_eval["tp"],
    "tn": grx_eval["tn"],
    "fp": grx_eval["fp"],
    "fn": grx_eval["fn"],
    "median_score": grx_eval["median_score"],
    "p95_score": grx_eval["p95_score"],
    "p99_score": grx_eval["p99_score"],
    "p99_5_score": grx_eval["p99_5_score"],
    "max_score": grx_eval["max_score"],
    "tail_contrast": grx_eval["tail_contrast"],
    "threshold": grx_eval["threshold"],
    "threshold_method": grx_eval["threshold_method"],
    "flagged_pixels": grx_eval["flagged_pixels"],
    "flagged_percent": grx_eval["flagged_percent"],
    "training_time_s": 0.0,
    "inference_time_s": grx_time,
    "checkpoint": "-",
}

presentation_table = pd.concat(
    [pd.DataFrame([grx_row]), all_results_df], ignore_index=True
)

if HAS_GT:
    presentation_columns = [
        "config",
        "wout",
        "win",
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "fp",
        "fn",
        "training_time_s",
        "inference_time_s",
    ]
    presentation_table = presentation_table[presentation_columns].sort_values(
        "average_precision", ascending=False
    )
else:
    presentation_columns = [
        "config",
        "wout",
        "win",
        "best_val_loss",
        "tail_contrast",
        "median_score",
        "p99_5_score",
        "threshold",
        "threshold_method",
        "flagged_pixels",
        "flagged_percent",
        "training_time_s",
        "inference_time_s",
    ]
    presentation_table = presentation_table[presentation_columns].sort_values(
        "tail_contrast", ascending=False
    )

presentation_table = presentation_table.reset_index(drop=True)
print(presentation_table.to_string(index=False))
presentation_table.to_csv(
    OUTPUT_DIR / "presentation_results_table.csv", index=False
)

run_info = {
    "data_file": str(DATA_PATH),
    "cube_key": cube_key,
    "original_cube_shape": str(tuple(cube_raw.shape)),
    "final_hsi_shape": str(tuple(hsi_z.shape)),
    "orientation": orientation_info,
    "has_ground_truth": HAS_GT,
    "ground_truth_key": gt_key,
    "best_config": BEST_CONFIG,
    "selection_metric": final_selection_metric,
    "default_threshold_percentile": DEFAULT_THRESHOLD_PERCENTILE,
}
pd.DataFrame([run_info]).to_csv(OUTPUT_DIR / "run_info.csv", index=False)

archive_base = SCRIPT_DIR / "directnet_experiments"
archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=OUTPUT_DIR)

print("Arhivă creată:", archive_path)
print("Rezultatele sunt salvate în:", OUTPUT_DIR)
if not HAS_GT:
    print(
        "IMPORTANT: fără ground truth, pixelii marcați sunt doar anomalii candidate. "
        "Nu pot fi calculate ROC-AUC, Precision, Recall, FP sau FN."
    )
