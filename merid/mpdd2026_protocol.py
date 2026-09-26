"""Validates the fixed MPDD-2026 development folds shared by every candidate."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


def _patient_id(value):
    text = str(value).strip()
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError('patient ID must be finite and integral') from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError('patient ID must be finite and integral')
    return str(int(number))


def _sort_ids(values):
    return sorted(values, key=lambda value: int(value))


def _fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DevelopmentFoldMembership:
    fold: int
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    file_sha256: str


@dataclass(frozen=True)
class DevelopmentFoldProtocol:
    task: str
    n_folds: int
    patient_ids: tuple[str, ...]
    validation_fold: tuple[tuple[str, int], ...]
    folds: tuple[DevelopmentFoldMembership, ...]
    protocol_fingerprint: str
    evidence_fingerprint: str


@dataclass(frozen=True)
class DevelopmentFoldView:
    fold: int
    outer_train_ids: tuple[str, ...]
    outer_validation_ids: tuple[str, ...]
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    protocol_fingerprint: str
    source_ids_fingerprint: str


def load_fixed_fold_protocol(split_root, task, expected_patient_ids, n_folds):
    if not isinstance(task, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', task):
        raise ValueError('task must be a simple directory name')
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 1:
        raise ValueError('n_folds must be a positive integer')
    expected_list = [_patient_id(value) for value in expected_patient_ids]
    if not expected_list or len(expected_list) != len(set(expected_list)):
        raise ValueError('expected patient IDs must be nonempty and unique')
    expected = set(expected_list)
    root = Path(split_root)
    folds = []
    validation_counts = {patient: 0 for patient in expected}
    validation_fold = {}
    for number in range(n_folds):
        path = root / task / ('fold%d.csv' % number)
        if not path.is_file():
            raise FileNotFoundError('fixed fold file missing: ' + str(path))
        raw = path.read_bytes()
        with io.StringIO(raw.decode('utf-8-sig'), newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {'ID', 'split'} <= set(reader.fieldnames):
                raise ValueError('fold CSV must contain ID and split fields')
            membership = {}
            for row in reader:
                patient = _patient_id(row.get('ID'))
                if patient in membership:
                    raise ValueError('duplicate patient ID within fold')
                status = str(row.get('split', '')).strip().lower()
                if status not in ('train', 'val'):
                    raise ValueError('fold split status must be train or val')
                membership[patient] = status
        if set(membership) != expected:
            raise ValueError('each fold must contain the exact expected patient set')
        train = _sort_ids(patient for patient, status in membership.items() if status == 'train')
        validation = _sort_ids(patient for patient, status in membership.items() if status == 'val')
        if not train or not validation:
            raise ValueError('each fold must have nonempty train and validation partitions')
        if set(train) & set(validation):
            raise ValueError('fold train and validation patients overlap')
        for patient in validation:
            validation_counts[patient] += 1
            validation_fold[patient] = number
        folds.append(dict(fold=number, train_ids=train, validation_ids=validation,
                          file_sha256=hashlib.sha256(raw).hexdigest()))
    if any(count != 1 for count in validation_counts.values()):
        raise ValueError('validation membership must cover every patient exactly once across folds')
    patient_ids = _sort_ids(expected)
    canonical = dict(task=task, n_folds=n_folds, patient_ids=patient_ids,
                     validation_fold={patient: validation_fold[patient] for patient in patient_ids})
    return dict(**canonical, folds=folds, protocol_fingerprint=_fingerprint(canonical))


def _protocol_canonical(protocol):
    validation_fold = dict(protocol.validation_fold)
    return dict(
        task=protocol.task,
        n_folds=protocol.n_folds,
        patient_ids=list(protocol.patient_ids),
        validation_fold={patient: validation_fold[patient] for patient in protocol.patient_ids},
    )


def _evidence_canonical(protocol):
    return dict(
        **_protocol_canonical(protocol),
        folds=[dict(
            fold=fold.fold,
            train_ids=list(fold.train_ids),
            validation_ids=list(fold.validation_ids),
            file_sha256=fold.file_sha256,
        ) for fold in protocol.folds],
    )


def load_development_fold_protocol(split_root, task, expected_patient_ids, n_folds):
    raw = load_fixed_fold_protocol(split_root, task, expected_patient_ids, n_folds)
    folds = tuple(DevelopmentFoldMembership(
        fold=int(item['fold']),
        train_ids=tuple(item['train_ids']),
        validation_ids=tuple(item['validation_ids']),
        file_sha256=str(item['file_sha256']),
    ) for item in raw['folds'])
    protocol = DevelopmentFoldProtocol(
        task=raw['task'],
        n_folds=raw['n_folds'],
        patient_ids=tuple(raw['patient_ids']),
        validation_fold=tuple((patient, int(raw['validation_fold'][patient]))
                              for patient in raw['patient_ids']),
        folds=folds,
        protocol_fingerprint=raw['protocol_fingerprint'],
        evidence_fingerprint='',
    )
    return DevelopmentFoldProtocol(
        **{field: getattr(protocol, field) for field in (
            'task', 'n_folds', 'patient_ids', 'validation_fold', 'folds',
            'protocol_fingerprint')},
        evidence_fingerprint=_fingerprint(_evidence_canonical(protocol)),
    )


def _normalized_unique_ids(values, name):
    normalized = tuple(_patient_id(value) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(name + ' must contain unique patient IDs')
    return normalized


def _validate_development_protocol(protocol):
    if not isinstance(protocol, DevelopmentFoldProtocol):
        raise TypeError('protocol must be a DevelopmentFoldProtocol')
    if (not isinstance(protocol.task, str)
            or not re.fullmatch(r'[A-Za-z0-9_-]+', protocol.task)):
        raise ValueError('development protocol task is invalid')
    if (not isinstance(protocol.n_folds, int) or isinstance(protocol.n_folds, bool)
            or protocol.n_folds < 1):
        raise ValueError('development protocol fold count is invalid')
    if not isinstance(protocol.patient_ids, tuple) or not protocol.patient_ids:
        raise ValueError('development protocol patient IDs must be a nonempty tuple')
    if (any(not isinstance(patient, str) or _patient_id(patient) != patient
            for patient in protocol.patient_ids)
            or len(protocol.patient_ids) != len(set(protocol.patient_ids))
            or protocol.patient_ids != tuple(_sort_ids(protocol.patient_ids))):
        raise ValueError('development protocol patient IDs must be canonical, sorted, and unique')
    if not isinstance(protocol.validation_fold, tuple):
        raise ValueError('development protocol validation mapping must be a tuple')
    mapping_keys = [item[0] for item in protocol.validation_fold
                    if isinstance(item, tuple) and len(item) == 2]
    if (len(mapping_keys) != len(protocol.validation_fold)
            or len(mapping_keys) != len(set(mapping_keys))
            or set(mapping_keys) != set(protocol.patient_ids)
            or tuple(mapping_keys) != protocol.patient_ids):
        raise ValueError('development protocol validation mapping must cover unique patient IDs')
    validation_map = dict(protocol.validation_fold)
    if any(not isinstance(fold, int) or isinstance(fold, bool)
           or fold < 0 or fold >= protocol.n_folds for fold in validation_map.values()):
        raise ValueError('development protocol validation mapping has an invalid fold')
    canonical = _protocol_canonical(protocol)
    if _fingerprint(canonical) != protocol.protocol_fingerprint:
        raise ValueError('development protocol fingerprint mismatch')
    if _fingerprint(_evidence_canonical(protocol)) != protocol.evidence_fingerprint:
        raise ValueError('development protocol evidence fingerprint mismatch')
    expected = set(protocol.patient_ids)
    if len(protocol.folds) != protocol.n_folds:
        raise ValueError('development protocol fold count mismatch')
    validation_counts = {patient: 0 for patient in expected}
    for number, fold in enumerate(protocol.folds):
        if not isinstance(fold, DevelopmentFoldMembership):
            raise ValueError('development protocol contains an invalid fold')
        if fold.fold != number:
            raise ValueError('development protocol fold order mismatch')
        if (not isinstance(fold.train_ids, tuple)
                or not isinstance(fold.validation_ids, tuple)
                or len(fold.train_ids) != len(set(fold.train_ids))
                or len(fold.validation_ids) != len(set(fold.validation_ids))
                or any(_patient_id(patient) != patient
                       for patient in fold.train_ids + fold.validation_ids)):
            raise ValueError('development protocol fold IDs must be canonical and unique tuples')
        if (fold.train_ids != tuple(_sort_ids(fold.train_ids))
                or fold.validation_ids != tuple(_sort_ids(fold.validation_ids))):
            raise ValueError('development protocol fold IDs must be sorted')
        if not re.fullmatch(r'[0-9a-f]{64}', fold.file_sha256):
            raise ValueError('development protocol fold file hash is invalid')
        if not fold.train_ids or not fold.validation_ids:
            raise ValueError('development protocol folds require nonempty partitions')
        train, validation = set(fold.train_ids), set(fold.validation_ids)
        if train & validation or train | validation != expected:
            raise ValueError('development protocol fold membership mismatch')
        for patient in validation:
            validation_counts[patient] += 1
            if validation_map[patient] != number:
                raise ValueError('development protocol validation mapping disagrees with folds')
    if any(count != 1 for count in validation_counts.values()):
        raise ValueError('development protocol validation coverage mismatch')


def iter_development_folds(
    protocol,
    source_ids,
    *,
    train_subset_ids=None,
    validation_subset_ids=None,
):
    _validate_development_protocol(protocol)
    source = _normalized_unique_ids(source_ids, 'source IDs')
    if set(source) != set(protocol.patient_ids):
        raise ValueError('source IDs must exactly cover development protocol patients')
    source_index = {patient: index for index, patient in enumerate(source)}
    cohort = set(protocol.patient_ids)

    def subset(values, name):
        if values is None:
            return cohort
        result = set(_normalized_unique_ids(values, name))
        if not result <= cohort:
            raise ValueError(name + ' must be a subset of protocol patients')
        return result

    train_subset = subset(train_subset_ids, 'training subset')
    validation_subset = subset(validation_subset_ids, 'validation subset')
    source_fingerprint = _fingerprint(list(source))
    views = []
    for fold in protocol.folds:
        train_ids = tuple(patient for patient in fold.train_ids if patient in train_subset)
        validation_ids = tuple(patient for patient in fold.validation_ids
                               if patient in validation_subset)
        views.append(DevelopmentFoldView(
            fold=fold.fold,
            outer_train_ids=fold.train_ids,
            outer_validation_ids=fold.validation_ids,
            train_ids=train_ids,
            validation_ids=validation_ids,
            train_indices=tuple(source_index[patient] for patient in train_ids),
            validation_indices=tuple(source_index[patient] for patient in validation_ids),
            protocol_fingerprint=protocol.protocol_fingerprint,
            source_ids_fingerprint=source_fingerprint,
        ))
    return tuple(views)


def compare_protocol_partitions(left, right):
    def partition(value):
        if isinstance(value, DevelopmentFoldProtocol):
            _validate_development_protocol(value)
            return (value.n_folds, tuple(value.patient_ids), tuple(value.validation_fold))
        patient_ids = tuple(value.get('patient_ids', ()))
        validation = value.get('validation_fold', {})
        return (value.get('n_folds'), patient_ids,
                tuple((patient, validation.get(patient)) for patient in patient_ids))

    if partition(left) != partition(right):
        raise ValueError('fixed-fold task-family partitions differ')
    return True
