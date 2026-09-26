"""MPDD-2026 label maps, train/validation splits and per-participant feature banks."""
from __future__ import annotations

import csv
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from torch.utils.data import Dataset


# Relative data paths resolve against this directory.
PROJECT_ROOT = Path(__file__).resolve().parent


EPS = 1e-6


TARGET_T = 128


PAIR_COUNT = 4


GAIT_KEEP_DIM = 9


CLASSIFICATION_TASK_TO_COLUMN = {
    "binary": "label2",
    "ternary": "label3",
}


REGRESSION_TASK = "regression"


AUDIO_FEATURE_ALIASES = {
    "mfcc": ("mfcc", "mfcc64"),
    "opensmile": ("opensmile",),
    "wav2vec": ("wav2vec", "wav2vec2", "wav2vec2-FRA"),
}


VIDEO_FEATURE_ALIASES = {
    "densenet": ("densenet",),
    "resnet": ("resnet",),
    "openface": ("openface",),
}


YOUNG_AUDIO_EVENT_NAMES = {
    1: "E1.npy",
    2: "E2.npy",
    3: "E3.npy",
}


YOUNG_VIDEO_EVENT_NAMES = {
    1: ("event_1.npy", "event_1/event_1_all.npy"),
    2: ("event_2.npy", "event_2/event_2_all.npy"),
    3: ("event_3.npy", "event_3/event_3_all.npy"),
}


def normalize_phq_target(value: float | int) -> float:
    return float(np.log1p(max(float(value), 0.0)))


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _infer_split_counterpart(path: Path) -> Path | None:
    path_str = str(path)
    if path.name == "split_labels_train.csv" and "MPDD-AVG2026-trainval" in path_str:
        return Path(path_str.replace("MPDD-AVG2026-trainval", "MPDD-AVG2026-test")).with_name("split_labels_test.csv")
    if path.name == "split_labels_test.csv" and "MPDD-AVG2026-test" in path_str:
        return Path(path_str.replace("MPDD-AVG2026-test", "MPDD-AVG2026-trainval")).with_name("split_labels_train.csv")
    return None


def _development_id(value: Any) -> int:
    number = float(value)
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError('development patient IDs must be finite integers')
    return int(number)


def _development_path(path: Path, root: Path | None = None) -> Path:
    resolved = path.resolve()
    if 'MPDD-AVG2026-test' in resolved.parts or resolved.name == 'split_labels_test.csv':
        raise ValueError('development scope prohibits held-out paths')
    if root is not None and not resolved.is_relative_to(root.resolve()):
        raise ValueError('development path escapes declared training root')
    return resolved


def _development_map(mapping: dict) -> dict:
    result = {}
    for key, value in mapping.items():
        normalized = _development_id(key)
        if normalized in result:
            raise ValueError('duplicate normalized development patient ID')
        result[normalized] = value
    return result


def load_split_rows(split_csv: str | Path, data_scope: str = "all") -> list[dict[str, str]]:
    if data_scope not in ("all", "development"):
        raise ValueError('data_scope must be all or development')
    csv_path = resolve_project_path(split_csv)
    if data_scope == "development":
        csv_path = _development_path(csv_path)
    if data_scope == "development" and (csv_path.name == "split_labels_test.csv"
                                         or "MPDD-AVG2026-test" in csv_path.parts):
        raise ValueError('development scope requires a training split CSV')
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if data_scope == "development":
        seen = set()
        for row in rows:
            person_id = _development_id(row.get("ID"))
            if person_id in seen:
                raise ValueError('duplicate patient ID in development labels')
            seen.add(person_id)
            if row.get("split", "").strip().lower() != "train":
                raise ValueError('development scope accepts only train rows')
        return rows

    split_values = {row.get("split", "").strip().lower() for row in rows}
    if {"train", "test"}.issubset(split_values):
        return rows

    counterpart = _infer_split_counterpart(csv_path)
    if counterpart is None or not counterpart.exists():
        return rows

    with open(counterpart, "r", encoding="utf-8-sig", newline="") as handle:
        counterpart_rows = list(csv.DictReader(handle))
    return rows + counterpart_rows


def get_label_column(task: str, regression_label: str = "label2") -> str:
    if task == REGRESSION_TASK:
        if regression_label not in {"label2", "label3"}:
            raise ValueError(f"Unsupported regression_label: {regression_label}")
        return regression_label
    if task in CLASSIFICATION_TASK_TO_COLUMN:
        return CLASSIFICATION_TASK_TO_COLUMN[task]
    raise ValueError(f"Unsupported task: {task}")


def get_task_label(row: dict[str, str], task: str, regression_label: str = "label2") -> int:
    column = get_label_column(task, regression_label)
    if column not in row or str(row[column]).strip() == "":
        raise KeyError(f"Column {column} not found for task={task}")
    return int(float(row[column]))


def get_phq9_target(row: dict[str, str]) -> float:
    if "PHQ-9" not in row or str(row["PHQ-9"]).strip() == "":
        raise KeyError("Column PHQ-9 not found in split csv")
    return float(row["PHQ-9"])


def load_task_maps(
    split_csv: str | Path,
    task: str,
    regression_label: str = "label2",
    data_scope: str = "all",
) -> dict[str, Any]:
    rows = load_split_rows(split_csv, data_scope=data_scope)
    train_map: dict[int, int] = {}
    test_map: dict[int, int] = {}
    source_split_map: dict[int, str] = {}
    train_phq_map: dict[int, float] = {}
    test_phq_map: dict[int, float] = {}

    for row in rows:
        person_id = (_development_id(row["ID"]) if data_scope == "development"
                     else int(row["ID"]))
        split_name = row["split"].strip().lower()
        label = get_task_label(row, task, regression_label)
        phq9 = get_phq9_target(row)
        if data_scope == "development":
            raw_label = float(row[get_label_column(task, regression_label)])
            if not np.isfinite(raw_label) or not raw_label.is_integer() or not np.isfinite(phq9):
                raise ValueError('development labels must be finite integers and PHQ-9 finite')
        source_split_map[person_id] = split_name
        if split_name == "train":
            train_map[person_id] = label
            train_phq_map[person_id] = phq9
        else:
            test_map[person_id] = label
            test_phq_map[person_id] = phq9

    payload: dict[str, Any] = {
        "train_map": train_map,
        "test_map": test_map,
        "source_split_map": source_split_map,
        "rows": rows,
        "split_label": get_label_column(task, regression_label),
        "train_phq_map": train_phq_map,
        "test_phq_map": test_phq_map,
    }
    return payload


def _normalize(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return np.clip((array - mean) / std, -5.0, 5.0).astype(np.float32)


def _resize(array: np.ndarray, target_len: int) -> torch.Tensor:
    tensor = torch.from_numpy(array.astype(np.float32)).transpose(0, 1).unsqueeze(0)
    tensor = F.interpolate(tensor, size=target_len, mode="linear", align_corners=False)
    return tensor.squeeze(0).transpose(0, 1).contiguous()


def _resolve_split_data_roots(data_root: Path) -> dict[str, Path]:
    parent_name = data_root.parent.name
    if parent_name == "MPDD-AVG2026-trainval":
        dataset_root = data_root.parent.parent
        test_root = dataset_root / "MPDD-AVG2026-test" / data_root.name
        return {
            "train": data_root,
            "test": test_root if test_root.exists() else data_root,
        }
    if parent_name == "MPDD-AVG2026-test":
        dataset_root = data_root.parent.parent
        train_root = dataset_root / "MPDD-AVG2026-trainval" / data_root.name
        return {
            "train": train_root if train_root.exists() else data_root,
            "test": data_root,
        }
    return {"train": data_root, "test": data_root}


def _resolve_modality_base(split_root: Path, modality: str, split_name: str) -> Path:
    candidates = [split_root / modality, split_root / modality.capitalize()]
    for base in candidates:
        if not base.exists():
            continue
        split_candidate = base / split_name
        return split_candidate if split_candidate.exists() else base
    return candidates[0]


def _resolve_feature_dir(split_root: Path, modality: str, split_name: str, feature_name: str) -> tuple[Path, str]:
    if modality == "audio":
        aliases = AUDIO_FEATURE_ALIASES.get(feature_name, (feature_name,))
    elif modality == "video":
        aliases = VIDEO_FEATURE_ALIASES.get(feature_name, (feature_name,))
    else:
        raise ValueError(f"Unknown modality={modality}")

    modality_root = _resolve_modality_base(split_root, modality, split_name)
    for alias in aliases:
        candidate = modality_root / alias
        if candidate.exists():
            return candidate, alias
    return modality_root / aliases[0], aliases[0]


def _resolve_gait_root(split_root: Path, split_name: str) -> Path:
    for gait_dir in ("IMU-ELDER", "IMU-Elder", "IMU-Young", "IMU"):
        candidate = split_root / gait_dir
        if not candidate.exists():
            continue
        split_candidate = candidate / split_name
        return split_candidate if split_candidate.exists() else candidate
    return split_root / "IMU" / split_name


def _resolve_gait_file(gait_root: Path, person_id: int) -> Path:
    candidates = (
        gait_root / f"{person_id}.npy",
        gait_root / str(person_id) / f"{person_id}.npy",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _discover_pair_files(root: Path, modality: str) -> dict[int, Path]:
    index_map: dict[int, Path] = {}
    if not root.exists():
        return index_map

    pattern = re.compile(r"^A_(\d+)\.npy$") if modality == "audio" else re.compile(r"^V_(\d+)\.npy$")
    for file_path in sorted(root.glob("*.npy")):
        match = pattern.match(file_path.name)
        if not match:
            continue
        pair_idx = int(match.group(1))
        if 1 <= pair_idx <= PAIR_COUNT:
            index_map[pair_idx] = file_path
    return index_map


def _discover_young_pair_files(audio_dir: Path, video_dir: Path, video_feature: str) -> tuple[dict[int, Path], dict[int, Path]]:
    audio_map: dict[int, Path] = {}
    video_map: dict[int, Path] = {}
    for pair_idx, audio_name in YOUNG_AUDIO_EVENT_NAMES.items():
        audio_file = audio_dir / audio_name
        if not audio_file.is_file():
            continue
        for video_candidate_name in YOUNG_VIDEO_EVENT_NAMES[pair_idx]:
            video_file = video_dir / video_candidate_name
            if video_file.is_file():
                audio_map[pair_idx] = audio_file
                video_map[pair_idx] = video_file
                break
    return audio_map, video_map


def _load_personality_map(personality_npy: str | Path) -> dict[int, np.ndarray]:
    path = resolve_project_path(personality_npy)
    if path.is_dir():
        path = path / "descriptions_embeddings_with_ids.npy"
    if not path.exists():
        return {}
    data = np.load(str(path), allow_pickle=True)
    return {int(item["id"]): np.asarray(item["embedding"], dtype=np.float32) for item in data}


def _infer_feature_dim(roots: list[Path], max_dim: int | None = None,
                       *, files: list[Path] | None = None, training_root: Path | None = None) -> int:
    for root in (roots if files is None else [None]):
        if root is not None and not root.exists():
            continue
        for file_path in (root.rglob("*.npy") if files is None else files):
            if training_root is not None:
                file_path = _development_path(file_path, training_root)
            if not file_path.is_file() or file_path.stat().st_size == 0:
                continue
            try:
                array = np.asarray(np.load(str(file_path), allow_pickle=True), dtype=np.float32)
            except Exception:
                continue
            if array.ndim >= 2 and array.shape[-1] > 0:
                dim = int(array.shape[-1])
                return min(dim, max_dim) if max_dim is not None else dim
            if array.ndim == 1 and array.shape[0] > 0:
                dim = int(array.shape[0])
                return min(dim, max_dim) if max_dim is not None else dim
    return 1


def _load_feature_array(path: Path, fallback_dim: int, max_dim: int | None = None) -> np.ndarray:
    if path.is_file() and path.stat().st_size > 0:
        try:
            array = _normalize(np.load(str(path), allow_pickle=True))
            if max_dim is not None:
                if array.ndim >= 2 and array.shape[-1] > max_dim:
                    array = array[..., :max_dim]
                elif array.ndim == 1 and array.shape[0] > max_dim:
                    array = array[:max_dim]
            return array
        except Exception:
            pass
    effective_dim = min(fallback_dim, max_dim) if max_dim is not None else fallback_dim
    return np.zeros((1, effective_dim), dtype=np.float32)


class MPDDElderDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        label_map: dict[int, int],
        source_split_map: dict[int, str],
        subtrack: str,
        task: str,
        audio_feature: str,
        video_feature: str,
        personality_npy: str | Path,
        phq_map: dict[int, float] | None = None,
        target_t: int = TARGET_T,
        zero_personality: bool = False,
        data_scope: str = "all",
    ) -> None:
        if data_scope not in ("all", "development"):
            raise ValueError('data_scope must be all or development')
        self.data_root = resolve_project_path(data_root)
        if data_scope == "development":
            self.data_root = _development_path(self.data_root)
            get_label_column(task)
            for feature in (audio_feature, video_feature):
                if not isinstance(feature, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", feature):
                    raise ValueError("development feature names must be simple source aliases")
            if "MPDD-AVG2026-test" in self.data_root.parts:
                raise ValueError('development scope requires a trainval feature root')
            normalized_sources = {key: str(value).strip().lower()
                                  for key, value in _development_map(source_split_map).items()}
            normalized_labels = _development_map(label_map)
            if set(normalized_sources) != set(normalized_labels) or any(
                    value != "train" for value in normalized_sources.values()):
                raise ValueError('development labels and source splits must match and be train-only')
            if any(not np.isfinite(float(value)) or float(value) != int(float(value))
                   for value in normalized_labels.values()):
                raise ValueError('development labels must be finite integers')
            if phq_map is not None:
                phq_map = _development_map(phq_map)
                if set(phq_map) != set(normalized_labels) or any(
                        not np.isfinite(float(value)) for value in phq_map.values()):
                    raise ValueError('development PHQ targets must be finite and match patient IDs')
            label_map, source_split_map = normalized_labels, normalized_sources
            self.split_data_roots = {"train": self.data_root}
        else:
            self.split_data_roots = _resolve_split_data_roots(self.data_root)
        self.data_scope = data_scope
        self.label_map = label_map
        self.phq_map = phq_map or {}
        self.source_split_map = source_split_map
        self.subtrack = subtrack
        self.task = task
        self.audio_feature = audio_feature
        self.video_feature = video_feature
        self.target_t = target_t
        self.zero_personality = zero_personality
        self.has_phq_target = phq_map is not None
        self.is_young = any(root.name.lower() == "young" for root in self.split_data_roots.values())
        self.need_av = subtrack in ("A-V+P", "A-V-G+P")
        self.need_gait = subtrack in ("A-V-G+P", "G+P")
        self.audio_roots: dict[str, Path] = {}
        self.video_roots: dict[str, Path] = {}

        if self.need_av:
            resolved_audio = []
            resolved_video = []
            for split_name in self.split_data_roots:
                split_root = self.split_data_roots[split_name]
                audio_root, audio_alias = _resolve_feature_dir(split_root, "audio", split_name, audio_feature)
                video_root, video_alias = _resolve_feature_dir(split_root, "video", split_name, video_feature)
                self.audio_roots[split_name] = audio_root
                self.video_roots[split_name] = video_root
                if audio_root.exists():
                    resolved_audio.append(audio_alias)
                if video_root.exists():
                    resolved_video.append(video_alias)
            self.resolved_audio_feature = ",".join(sorted(set(resolved_audio))) if resolved_audio else audio_feature
            self.resolved_video_feature = ",".join(sorted(set(resolved_video))) if resolved_video else video_feature
        else:
            self.resolved_audio_feature = ""
            self.resolved_video_feature = ""

        self.gait_roots = {
            split_name: _resolve_gait_root(self.split_data_roots[split_name], split_name)
            for split_name in self.split_data_roots
        }
        if data_scope == "development":
            for roots in (self.audio_roots, self.video_roots, self.gait_roots):
                for path in roots.values():
                    _development_path(path, self.data_root)
            self.samples = self._collect_samples()
            def scoped_dim(key, max_dim=None):
                files = []
                for sample in self.samples:
                    value = sample.get(key, {})
                    files.extend(value.values() if isinstance(value, dict) else [value])
                return _infer_feature_dim([], max_dim, files=files, training_root=self.data_root)
            self.audio_dim_hint = scoped_dim('audio_map') if self.need_av else 0
            self.video_dim_hint = scoped_dim('video_map') if self.need_av else 0
            self.gait_dim_hint = scoped_dim('gait_file', GAIT_KEEP_DIM) if self.need_gait else 0
            if zero_personality:
                self.personality_map = {}
            else:
                personality_path = _development_path(resolve_project_path(personality_npy), self.data_root)
                if personality_path.is_dir():
                    personality_path = _development_path(
                        personality_path / 'descriptions_embeddings_with_ids.npy', self.data_root)
                self.personality_map = _load_personality_map(personality_path)
        else:
            self.audio_dim_hint = _infer_feature_dim(list(self.audio_roots.values())) if self.need_av else 0
            self.video_dim_hint = _infer_feature_dim(list(self.video_roots.values())) if self.need_av else 0
            self.gait_dim_hint = _infer_feature_dim(list(self.gait_roots.values()), max_dim=GAIT_KEEP_DIM) if self.need_gait else 0
            self.personality_map = _load_personality_map(personality_npy)
            self.samples = self._collect_samples()
        self._feature_tensor_cache: dict[tuple[str, int, int | None], torch.Tensor] = {}
        if not self.samples:
            raise RuntimeError(
                f"No valid samples found for task={task}, subtrack={subtrack}, "
                f"audio_feature={audio_feature}, video_feature={video_feature}"
            )

    def _load_resized_feature(self, path: Path, fallback_dim: int, max_dim: int | None = None) -> torch.Tensor:
        if self.data_scope == "development":
            path = _development_path(path, self.data_root)
        key = (path.as_posix(), int(fallback_dim), max_dim)
        cached = self._feature_tensor_cache.get(key)
        if cached is None:
            cached = _resize(_load_feature_array(path, fallback_dim, max_dim=max_dim), self.target_t)
            self._feature_tensor_cache[key] = cached
        return cached

    def _collect_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for person_id, label in sorted(self.label_map.items()):
            source_split = self.source_split_map.get(person_id, "train").lower()
            sample: dict[str, Any] = {
                "pid": person_id,
                "label": int(label),
                "source_split": source_split,
            }

            if self.has_phq_target:
                if person_id not in self.phq_map:
                    continue
                sample["phq9"] = normalize_phq_target(self.phq_map[person_id])

            if self.need_gait:
                gait_file = _resolve_gait_file(self.gait_roots[source_split], person_id)
                if self.data_scope == "development":
                    _development_path(gait_file, self.data_root)
                if not gait_file.exists():
                    continue
                sample["gait_file"] = gait_file

            if self.need_av:
                audio_dir = self.audio_roots[source_split] / str(person_id)
                video_dir = self.video_roots[source_split] / str(person_id)
                if self.data_scope == "development":
                    _development_path(audio_dir, self.data_root)
                    _development_path(video_dir, self.data_root)
                if self.is_young:
                    audio_map, video_map = _discover_young_pair_files(audio_dir, video_dir, self.video_feature)
                else:
                    audio_map = _discover_pair_files(audio_dir, "audio")
                    video_map = _discover_pair_files(video_dir, "video")
                if self.data_scope == "development":
                    for path in [*audio_map.values(), *video_map.values()]:
                        _development_path(path, self.data_root)
                shared_indices = sorted(set(audio_map) & set(video_map))
                if not shared_indices:
                    continue
                sample["audio_map"] = audio_map
                sample["video_map"] = video_map
                sample["pair_indices"] = shared_indices[:PAIR_COUNT]

            samples.append(sample)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        person_id = int(sample["pid"])
        label = torch.tensor(int(sample["label"]), dtype=torch.long)

        result: dict[str, torch.Tensor] = {
            "pid": torch.tensor(person_id, dtype=torch.long),
            "label": label,
        }

        if self.has_phq_target:
            result["phq9"] = torch.tensor(float(sample["phq9"]), dtype=torch.float32)

        if self.need_av:
            audio_pairs: list[torch.Tensor] = []
            video_pairs: list[torch.Tensor] = []
            pair_mask: list[float] = []
            audio_map: dict[int, Path] = sample["audio_map"]
            video_map: dict[int, Path] = sample["video_map"]
            pair_indices: list[int] = sample["pair_indices"]
            for pair_idx in pair_indices:
                audio_pairs.append(self._load_resized_feature(audio_map[pair_idx], self.audio_dim_hint))
                video_pairs.append(self._load_resized_feature(video_map[pair_idx], self.video_dim_hint))
                pair_mask.append(1.0)

            audio_dim = int(audio_pairs[0].shape[-1])
            video_dim = int(video_pairs[0].shape[-1])
            while len(audio_pairs) < PAIR_COUNT:
                audio_pairs.append(torch.zeros(self.target_t, audio_dim, dtype=torch.float32))
                video_pairs.append(torch.zeros(self.target_t, video_dim, dtype=torch.float32))
                pair_mask.append(0.0)

            result["audio"] = torch.stack(audio_pairs)
            result["video"] = torch.stack(video_pairs)
            result["pair_mask"] = torch.tensor(pair_mask, dtype=torch.float32)

        if self.need_gait:
            result["gait"] = self._load_resized_feature(sample["gait_file"], self.gait_dim_hint, max_dim=GAIT_KEEP_DIM)

        if self.zero_personality:
            personality = np.zeros(1024, dtype=np.float32)
        else:
            personality = self.personality_map.get(person_id, np.zeros(1024, dtype=np.float32))
        result["personality"] = torch.from_numpy(personality.astype(np.float32))
        return result


POOL_SPLITS = {"", "train", "val"}


def _load_train_rows(split_csv: str | Path) -> list[dict[str, str]]:
    csv_path = resolve_project_path(split_csv)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Split CSV is empty: {csv_path}")

    train_rows = [row for row in rows if row.get("split", "train").strip().lower() in POOL_SPLITS]
    if not train_rows:
        raise ValueError(f"No train rows found in split CSV: {csv_path}")
    return train_rows


def create_train_val_split(
    split_csv: str | Path,
    task: str,
    val_ratio: float = 0.1,
    regression_label: str = "label2",
    seed: int | None = None,
    respect_split_column: bool = False,
) -> dict[str, Any]:
    rows = _load_train_rows(split_csv)
    fixed_val_rows = [row for row in rows if row.get("split", "train").strip().lower() == "val"]
    if respect_split_column and fixed_val_rows:
        fixed_val_ids = {int(row["ID"]) for row in fixed_val_rows}
        train_id_split = [int(row["ID"]) for row in rows if int(row["ID"]) not in fixed_val_ids]
        val_id_split = [int(row["ID"]) for row in fixed_val_rows]
    else:
        sample_ids = [int(row["ID"]) for row in rows]
        sample_labels = [get_task_label(row, task, regression_label) for row in rows]

        if len(sample_ids) < 2:
            raise ValueError("At least two train samples are required to create a train/val split.")
        if not 0.0 < val_ratio < 1.0:
            raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

        label_counts = Counter(int(label) for label in sample_labels)
        splitter: StratifiedShuffleSplit | ShuffleSplit
        if label_counts and min(label_counts.values()) >= 2:
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=1.0 - val_ratio, random_state=seed)
            train_indices, val_indices = next(splitter.split(sample_ids, sample_labels))
        else:
            splitter = ShuffleSplit(n_splits=1, train_size=1.0 - val_ratio, random_state=seed)
            train_indices, val_indices = next(splitter.split(sample_ids))

        train_id_split = [sample_ids[index] for index in train_indices]
        val_id_split = [sample_ids[index] for index in val_indices]

    train_id_split = sorted(int(item) for item in train_id_split)
    val_id_split = sorted(int(item) for item in val_id_split)
    train_id_set = set(train_id_split)
    val_id_set = set(val_id_split)

    source_split_map = {int(row["ID"]): "train" for row in rows}
    train_map = {int(row["ID"]): get_task_label(row, task, regression_label) for row in rows if int(row["ID"]) in train_id_set}
    val_map = {int(row["ID"]): get_task_label(row, task, regression_label) for row in rows if int(row["ID"]) in val_id_set}

    payload = {
        "train_ids": train_id_split,
        "val_ids": val_id_split,
        "train_map": train_map,
        "val_map": val_map,
        "source_split_map": source_split_map,
        "rows": rows,
        "seed": seed,
        "respect_split_column": respect_split_column,
        "split_label": regression_label if task == REGRESSION_TASK else ("label2" if task == "binary" else "label3"),
        "train_phq_map": {int(row["ID"]): get_phq9_target(row) for row in rows if int(row["ID"]) in train_id_set},
        "val_phq_map": {int(row["ID"]): get_phq9_target(row) for row in rows if int(row["ID"]) in val_id_set},
    }
    return payload


def masked_pair_tensor(item: dict[str, Any], key: str) -> np.ndarray:
    values = item[key].detach().cpu().numpy().astype(np.float32)
    mask = item.get("pair_mask")
    if mask is None:
        return values.reshape(-1, values.shape[-1])
    keep = mask.detach().cpu().numpy().astype(np.float32) > 0.5
    if not keep.any():
        keep = np.ones(values.shape[0], dtype=bool)
    return values[keep].reshape(-1, values.shape[-1])


def pair_tensor_and_mask(item: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    values = item[key].detach().cpu().numpy().astype(np.float32)
    mask = item.get("pair_mask")
    if mask is None:
        keep = np.ones(values.shape[0], dtype=bool)
    else:
        keep = mask.detach().cpu().numpy().astype(np.float32) > 0.5
    if not keep.any():
        keep = np.ones(values.shape[0], dtype=bool)
    return values, keep


def stats_block(values: np.ndarray, include_extrema: bool = False) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    blocks = [values.mean(axis=0), values.std(axis=0)]
    if include_extrema:
        blocks.extend([np.percentile(values, 25, axis=0), np.percentile(values, 75, axis=0)])
    return np.concatenate(blocks).astype(np.float32)


def event_stats_block(values: np.ndarray, keep: np.ndarray, include_extrema: bool = False, include_delta: bool = False) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.asarray(keep, dtype=bool)
    if not keep.any():
        keep = np.ones(values.shape[0], dtype=bool)

    valid_values = values[keep].reshape(-1, values.shape[-1])
    event_template = stats_block(values[0], include_extrema=include_extrema)
    blocks = [stats_block(valid_values, include_extrema=include_extrema)]

    for pair_idx in range(values.shape[0]):
        if pair_idx < keep.shape[0] and keep[pair_idx]:
            blocks.append(stats_block(values[pair_idx], include_extrema=include_extrema))
        else:
            blocks.append(np.zeros_like(event_template))

    if include_delta:
        event_means = values.mean(axis=1)
        delta_template = np.zeros(values.shape[-1], dtype=np.float32)
        for pair_idx in range(max(values.shape[0] - 1, 0)):
            valid_pair = pair_idx < keep.shape[0] and pair_idx + 1 < keep.shape[0]
            if valid_pair and keep[pair_idx] and keep[pair_idx + 1]:
                blocks.append((event_means[pair_idx + 1] - event_means[pair_idx]).astype(np.float32))
            else:
                blocks.append(delta_template)

    return np.concatenate(blocks).astype(np.float32)


def personality_block(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "p_l2":
        denom = np.linalg.norm(values)
        return (values / denom).astype(np.float32) if denom > 1e-6 else values
    mean = float(values.mean())
    std = float(values.std())
    return ((values - mean) / max(std, 1e-6)).astype(np.float32)


def sample_vector(item: dict[str, Any], mode: str) -> np.ndarray:
    include_extrema = "_q" in mode
    blocks: list[np.ndarray] = []
    if not mode.startswith("gp"):
        if "event" in mode:
            audio_values, keep = pair_tensor_and_mask(item, "audio")
            video_values, _ = pair_tensor_and_mask(item, "video")
            include_delta = "delta" in mode
            blocks.append(event_stats_block(audio_values, keep, include_extrema=include_extrema, include_delta=include_delta))
            blocks.append(event_stats_block(video_values, keep, include_extrema=include_extrema, include_delta=include_delta))
        else:
            blocks.append(stats_block(masked_pair_tensor(item, "audio"), include_extrema=include_extrema))
            blocks.append(stats_block(masked_pair_tensor(item, "video"), include_extrema=include_extrema))
    blocks.append(stats_block(item["gait"].detach().cpu().numpy(), include_extrema=include_extrema))
    blocks.append(personality_block(item["personality"].detach().cpu().numpy(), "p_l2" if "pl2" in mode else "p_z"))
    pair_mask = item.get("pair_mask")
    if pair_mask is not None:
        blocks.append(np.asarray([float(pair_mask.sum().item())], dtype=np.float32))
    return np.concatenate(blocks).astype(np.float32)


def dataset_matrix(dataset: MPDDElderDataset, mode: str) -> tuple[list[int], np.ndarray]:
    ids: list[int] = []
    rows: list[np.ndarray] = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        ids.append(int(item["pid"].item()))
        rows.append(sample_vector(item, mode))
    return ids, np.vstack(rows).astype(np.float32)


def _development_bank_ids(values) -> list[int]:
    ids = []
    for value in values:
        try:
            number = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError('development bank IDs must be finite integers') from exc
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError('development bank IDs must be finite integers')
        ids.append(int(number))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('development bank IDs must be nonempty and unique')
    return ids


def validate_development_bank(ids, matrix, expected_ids):
    ids = _development_bank_ids(ids)
    expected = _development_bank_ids(expected_ids)
    if set(ids) != set(expected):
        raise ValueError('development bank must cover the exact expected training patient set')
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] == 0:
        raise ValueError('development bank requires one nonempty feature row per patient')
    if not np.isfinite(matrix).all():
        raise ValueError('development bank features must be finite')
    return ids, matrix


def build_feature_bank(
    track: str,
    audio_feature: str,
    video_feature: str,
    mode: str,
    task_maps: dict[str, Any],
    data_root: str,
    personality_npy: str,
    data_scope: str = "all",
) -> dict[str, Any]:
    if data_scope not in ("all", "development"):
        raise ValueError("data_scope must be all or development")
    scope_kwargs = {"data_scope": "development"} if data_scope == "development" else {}
    train_dataset = MPDDElderDataset(
        data_root=data_root,
        label_map=task_maps["train_map"],
        source_split_map=task_maps["source_split_map"],
        subtrack="A-V-G+P",
        task="binary",
        audio_feature=audio_feature,
        video_feature=video_feature,
        personality_npy=personality_npy,
        target_t=128,
        **scope_kwargs,
    )
    if data_scope == "development":
        train_ids, train_x = dataset_matrix(train_dataset, mode)
        train_ids, train_x = validate_development_bank(
            train_ids, train_x, task_maps['train_map'])
        return dict(track=track, audio_feature=audio_feature, video_feature=video_feature,
                    mode=mode, train_ids=train_ids, train_x=train_x, data_scope='development')
    test_dataset = MPDDElderDataset(
        data_root=data_root,
        label_map=task_maps["test_map"],
        source_split_map=task_maps["source_split_map"],
        subtrack="A-V-G+P",
        task="binary",
        audio_feature=audio_feature,
        video_feature=video_feature,
        personality_npy=personality_npy,
        target_t=128,
    )
    train_ids, train_x = dataset_matrix(train_dataset, mode)
    test_ids, test_x = dataset_matrix(test_dataset, mode)
    return {
        "track": track,
        "audio_feature": audio_feature,
        "video_feature": video_feature,
        "mode": mode,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "train_x": train_x,
        "test_x": test_x,
    }


def _safe_z(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    std = float(values.std())
    if std < 1e-6:
        return values * 0.0
    return (values - float(values.mean())) / std


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left_z = _safe_z(left)
    right_z = _safe_z(right)
    denom = float(np.linalg.norm(left_z) * np.linalg.norm(right_z))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(left_z, right_z) / denom)


def _lag_corr(left: np.ndarray, right: np.ndarray, lag: int) -> float:
    if lag < 0:
        return _safe_corr(left[:lag], right[-lag:])
    if lag > 0:
        return _safe_corr(left[lag:], right[:-lag])
    return _safe_corr(left, right)


def _series_stats(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return np.zeros(8, dtype=np.float32)
    slope = float(values[-1] - values[0]) if values.size > 1 else 0.0
    return np.asarray(
        [
            float(values.mean()),
            float(values.std()),
            float(np.percentile(values, 10)),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 50)),
            float(np.percentile(values, 75)),
            float(np.percentile(values, 90)),
            slope,
        ],
        dtype=np.float32,
    )


def _energy(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.sqrt(np.mean(values * values, axis=-1) + 1e-8).astype(np.float32)


def _personality_summary(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(
        [
            float(values.mean()),
            float(values.std()),
            float(np.percentile(values, 10)),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 50)),
            float(np.percentile(values, 75)),
            float(np.percentile(values, 90)),
            float(np.linalg.norm(values) / max(np.sqrt(values.size), 1.0)),
            float(np.max(np.abs(values))) if values.size else 0.0,
        ],
        dtype=np.float32,
    )


def alignment_vector(item: dict[str, Any], mode: str) -> np.ndarray:
    audio_values, keep = pair_tensor_and_mask(item, "audio")
    video_values, _ = pair_tensor_and_mask(item, "video")
    gait_values = np.nan_to_num(item["gait"].detach().cpu().numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    gait_energy = _energy(gait_values)

    blocks: list[np.ndarray] = []
    pair_audio_means: list[float] = []
    pair_video_means: list[float] = []
    pair_av_corrs: list[float] = []
    pair_ag_corrs: list[float] = []
    pair_vg_corrs: list[float] = []
    pair_count = 0
    max_pairs = audio_values.shape[0]
    for pair_idx in range(max_pairs):
        valid = pair_idx < keep.shape[0] and bool(keep[pair_idx])
        if valid:
            pair_count += 1
            audio_energy = _energy(audio_values[pair_idx])
            video_energy = _energy(video_values[pair_idx])
            av_corr = _safe_corr(audio_energy, video_energy)
            ag_corr = _safe_corr(audio_energy, gait_energy)
            vg_corr = _safe_corr(video_energy, gait_energy)
            pair_audio_means.append(float(audio_energy.mean()))
            pair_video_means.append(float(video_energy.mean()))
            pair_av_corrs.append(av_corr)
            pair_ag_corrs.append(ag_corr)
            pair_vg_corrs.append(vg_corr)
            pair_block = [
                _series_stats(audio_energy),
                _series_stats(video_energy),
                _series_stats(np.abs(_safe_z(audio_energy) - _safe_z(video_energy))),
                np.asarray(
                    [
                        av_corr,
                        _lag_corr(audio_energy, video_energy, -8),
                        _lag_corr(audio_energy, video_energy, -4),
                        _lag_corr(audio_energy, video_energy, 4),
                        _lag_corr(audio_energy, video_energy, 8),
                        ag_corr,
                        vg_corr,
                        float(audio_energy.mean() / max(float(video_energy.mean()), 1e-6)),
                    ],
                    dtype=np.float32,
                ),
            ]
            blocks.append(np.concatenate(pair_block).astype(np.float32))
        else:
            blocks.append(np.zeros(32, dtype=np.float32))

    if pair_audio_means:
        a_event = np.asarray(pair_audio_means, dtype=np.float32)
        v_event = np.asarray(pair_video_means, dtype=np.float32)
        av_event_corr = _safe_corr(a_event, v_event)
        event_delta = _safe_z(a_event) - _safe_z(v_event)
    else:
        a_event = np.zeros(1, dtype=np.float32)
        v_event = np.zeros(1, dtype=np.float32)
        av_event_corr = 0.0
        event_delta = np.zeros(1, dtype=np.float32)

    blocks.extend(
        [
            _series_stats(gait_energy),
            _series_stats(np.asarray(pair_av_corrs, dtype=np.float32)),
            _series_stats(np.asarray(pair_ag_corrs, dtype=np.float32)),
            _series_stats(np.asarray(pair_vg_corrs, dtype=np.float32)),
            _series_stats(event_delta),
            np.asarray([float(pair_count), av_event_corr, float(a_event.mean()), float(v_event.mean())], dtype=np.float32),
            _personality_summary(item["personality"].detach().cpu().numpy()),
        ]
    )
    if mode == "alignment_ext":
        audio_all = audio_values[keep].reshape(-1, audio_values.shape[-1]) if keep.any() else audio_values.reshape(-1, audio_values.shape[-1])
        video_all = video_values[keep].reshape(-1, video_values.shape[-1]) if keep.any() else video_values.reshape(-1, video_values.shape[-1])
        blocks.extend([_series_stats(_energy(audio_all)), _series_stats(_energy(video_all))])
    return np.concatenate(blocks).astype(np.float32)


def dataset_alignment_matrix(dataset: MPDDElderDataset, mode: str) -> tuple[list[int], np.ndarray]:
    ids: list[int] = []
    rows: list[np.ndarray] = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        ids.append(int(item["pid"].item()))
        rows.append(alignment_vector(item, mode))
    return ids, np.vstack(rows).astype(np.float32)


def build_alignment_bank(
    track: str,
    audio_feature: str,
    video_feature: str,
    mode: str,
    task_maps: dict[str, Any],
    data_root: str,
    personality_npy: str,
    data_scope: str = "all",
) -> dict[str, Any]:
    if data_scope not in ("all", "development"):
        raise ValueError("data_scope must be all or development")
    scope_kwargs = {"data_scope": "development"} if data_scope == "development" else {}
    train_dataset = MPDDElderDataset(
        data_root=data_root,
        label_map=task_maps["train_map"],
        source_split_map=task_maps["source_split_map"],
        subtrack="A-V-G+P",
        task="binary",
        audio_feature=audio_feature,
        video_feature=video_feature,
        personality_npy=personality_npy,
        target_t=128,
        **scope_kwargs,
    )
    if data_scope == "development":
        train_ids, train_x = dataset_alignment_matrix(train_dataset, mode)
        train_ids, train_x = validate_development_bank(
            train_ids, train_x, task_maps['train_map'])
        return dict(track=track, audio_feature=audio_feature, video_feature=video_feature,
                    mode=mode, train_ids=train_ids, train_x=train_x, data_scope='development')
    test_dataset = MPDDElderDataset(
        data_root=data_root,
        label_map=task_maps["test_map"],
        source_split_map=task_maps["source_split_map"],
        subtrack="A-V-G+P",
        task="binary",
        audio_feature=audio_feature,
        video_feature=video_feature,
        personality_npy=personality_npy,
        target_t=128,
    )
    train_ids, train_x = dataset_alignment_matrix(train_dataset, mode)
    test_ids, test_x = dataset_alignment_matrix(test_dataset, mode)
    return {
        "track": track,
        "audio_feature": audio_feature,
        "video_feature": video_feature,
        "mode": mode,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "train_x": train_x,
        "test_x": test_x,
    }


def bank_key(bank: dict[str, Any]) -> str:
    return f"{bank['audio_feature']}_{bank['video_feature']}_{bank['mode']}"


def row_map(bank: dict[str, Any], split: str) -> dict[int, np.ndarray]:
    id_key = f"{split}_ids"
    x_key = f"{split}_x"
    return {int(pid): bank[x_key][idx] for idx, pid in enumerate(bank[id_key])}


def concatenate_banks(name: str, banks: list[dict[str, Any]],
                      data_scope: str = "all") -> dict[str, Any]:
    if data_scope not in ("all", "development"):
        raise ValueError('data_scope must be all or development')
    if data_scope == "development":
        if not banks or any(bank.get('data_scope') != 'development' for bank in banks):
            raise ValueError('development concatenation requires declared development banks')
        expected = banks[0]['train_ids']
        train_maps = []
        for bank in banks:
            if bank['track'] != banks[0]['track']:
                raise ValueError('development banks must belong to the same track')
            ids, matrix = validate_development_bank(bank['train_ids'], bank['train_x'], expected)
            train_maps.append(dict(zip(ids, matrix)))
        train_ids = sorted(train_maps[0])
        train_x = np.vstack([np.concatenate([mapping[pid] for mapping in train_maps])
                             for pid in train_ids]).astype(np.float32)
        validate_development_bank(train_ids, train_x, expected)
        return dict(track=banks[0]['track'], audio_feature='fusion', video_feature=name,
                    mode='concat', parts=[bank_key(bank) for bank in banks],
                    train_ids=train_ids, train_x=train_x, data_scope='development')
    train_maps = [row_map(bank, "train") for bank in banks]
    test_maps = [row_map(bank, "test") for bank in banks]
    train_ids = sorted(set.intersection(*(set(mapping) for mapping in train_maps)))
    test_ids = sorted(set.intersection(*(set(mapping) for mapping in test_maps)))
    if not train_ids or not test_ids:
        raise ValueError(f"Empty fused bank: {name}")

    train_x = np.vstack([np.concatenate([mapping[pid] for mapping in train_maps]).astype(np.float32) for pid in train_ids])
    test_x = np.vstack([np.concatenate([mapping[pid] for mapping in test_maps]).astype(np.float32) for pid in test_ids])
    return {
        "track": banks[0]["track"],
        "audio_feature": "fusion",
        "video_feature": name,
        "mode": "concat",
        "parts": [bank_key(bank) for bank in banks],
        "train_ids": train_ids,
        "test_ids": test_ids,
        "train_x": train_x.astype(np.float32),
        "test_x": test_x.astype(np.float32),
    }
