"""Frozen-centroid primitive baseline with training-only scaling and RAM caps."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from fpcode.behavior.observations.representations import native_feature_coordinates
from fpcode.behavior.unit_adapters import from_frame_labels
from fpcode.contracts.behavior_units import UnitStream, UnitType
from fpcode.contracts.sessions import SessionKey
from fpcode.contracts.signals import SignalSeries
from fpcode.contracts.timelines import frozen_array

IMPLEMENTATION_STATUS = "implemented"


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _matrix(features):
    if not isinstance(features, SignalSeries):
        raise ValueError("features must be a SignalSeries")
    values, valid = features.values, features.valid
    return (values[:, None], valid[:, None]) if values.ndim == 1 else (values, valid)


def _batch_size(requested, budget, fixed_bytes, k, f):
    _positive(requested, "batch_size")
    _positive(budget, "memory_budget_bytes")
    per_row = 16*k+64*f+32
    available = budget-fixed_bytes
    if available < per_row:
        raise ValueError("primitive operation memory budget cannot hold one distance batch")
    return min(int(requested), available//per_row)


def _assign(x, centers):
    """Squared distances through BLAS; no batch×K×features broadcast."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            d = x @ centers.T
            d *= -2.
            d += np.sum(x*x, axis=1)[:, None]
            d += np.sum(centers*centers, axis=1)[None, :]
            np.maximum(d, 0., out=d)
            return np.argmin(d, axis=1)
    except FloatingPointError as exc:
        raise ValueError("primitive distances exceed supported float64 range") from exc


@dataclass(frozen=True)
class CentroidPrimitiveSegmenter:
    model_id: str
    k: int
    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...]
    centers: np.ndarray
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    catalog: tuple[UnitType, ...]
    memory_budget_bytes: int = 256*1024*1024

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a nonempty string")
        k = _positive(self.k, "k")
        _positive(self.memory_budget_bytes, "memory_budget_bytes")
        names, units, catalog = tuple(self.feature_names), tuple(self.feature_units), tuple(self.catalog)
        f = len(names)
        if not f or len(set(names)) != f or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("feature names must be nonempty and unique")
        if len(units) != f or any(not isinstance(u, str) or not u for u in units):
            raise ValueError("feature units must match names")
        if len(catalog) != k or any(not isinstance(t, UnitType) for t in catalog) or len({t.type_id for t in catalog}) != k:
            raise ValueError("catalog must contain k unique UnitType records")
        if 32*(k*f+f) > self.memory_budget_bytes:
            raise ValueError("primitive parameter memory budget exceeded")
        centers, mean, std = (frozen_array(a, dtype=float) for a in (self.centers, self.scaler_mean, self.scaler_std))
        if centers.shape != (k, f) or mean.shape != (f,) or std.shape != (f,):
            raise ValueError("fitted array shapes disagree with feature schema/k")
        if any(not np.isfinite(a).all() for a in (centers, mean, std)) or np.any(std <= 0):
            raise ValueError("fitted arrays must be finite and scales positive")
        for name, value in (("k", k), ("feature_names", names), ("feature_units", units), ("catalog", catalog),
                            ("centers", centers), ("scaler_mean", mean), ("scaler_std", std)):
            object.__setattr__(self, name, value)

    @classmethod
    def fit(cls, features: SignalSeries, *, k: int, model_id="vq_baseline", selected_features=None,
            seed=42, max_iter=100, max_samples=50000, batch_size=10000,
            memory_budget_bytes=256*1024*1024):
        """Fit sampled training evidence; budgets cover new working arrays.

        Input SignalSeries storage already owned by the caller is excluded. Batch
        size is reduced to fit the declared cap, and impossible minimum allocations
        fail before construction. Empty/duplicate clusters retain a seeded finite
        prototype; they do not manufacture distinct behaviour classes.
        """
        for name, value in (("k", k), ("max_iter", max_iter), ("max_samples", max_samples),
                            ("batch_size", batch_size), ("memory_budget_bytes", memory_budget_bytes)):
            _positive(value, name)
        values, valid = _matrix(features)
        all_names = [s.name for s in features.signals]
        names = tuple(all_names) if selected_features is None else tuple(selected_features)
        if not names or len(set(names)) != len(names) or any(n not in all_names for n in names):
            raise ValueError("selected feature schema must be nonempty, unique and present")
        cols = [all_names.index(n) for n in names]
        units = tuple(features.signals[c].unit for c in cols)
        n, f = len(values), len(cols)
        if 12*n > memory_budget_bytes:
            raise ValueError("primitive training eligibility memory budget exceeded")
        mask = np.ones(n, bool)
        for c in cols:
            mask &= valid[:, c] & np.isfinite(values[:, c])
        indices = np.flatnonzero(mask)
        n_train = min(len(indices), int(max_samples))
        if n_train < k:
            raise ValueError(f"insufficient valid training samples ({n_train}) for k={k}")
        fixed = 12*n + 32*n_train*f + 32*k*f + 16*n_train + 64*f
        batch = _batch_size(batch_size, memory_budget_bytes, fixed, k, f)
        rng = np.random.default_rng(seed)
        if len(indices) > n_train:
            indices = rng.choice(indices, size=n_train, replace=False)
        x = np.column_stack([values[indices, c] for c in cols])
        mean, std = np.mean(x, axis=0), np.std(x, axis=0)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("training normalization exceeds supported float64 range")
        std = np.where(std > 1e-12, std, 1.)
        x = (x-mean)/std
        centers = _kmeans_plus_plus_init(x, k, rng)
        for _ in range(max_iter):
            labels = np.empty(n_train, dtype=np.int64)
            for lo in range(0, n_train, batch):
                labels[lo:lo+batch] = _assign(x[lo:lo+batch], centers)
            sums = np.zeros_like(centers)
            np.add.at(sums, labels, x)
            counts = np.bincount(labels, minlength=k)
            updated = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
            empty = np.flatnonzero(counts == 0)
            if len(empty):
                updated[empty] = x[rng.integers(0, n_train, size=len(empty))]
            shift = float(np.max(np.linalg.norm(updated-centers, axis=1)))
            centers = updated
            if shift < 1e-4:
                break
        return cls(model_id, k, names, units, centers, mean, std,
                   tuple(UnitType(str(i), f"primitive cluster {i}") for i in range(k)), memory_budget_bytes)

    def predict_labels(self, features: SignalSeries, *, batch_size=10000, memory_budget_bytes=None):
        """Named, unit-checked held-out assignment; unsupported rows remain -1."""
        values, valid = _matrix(features)
        names = [s.name for s in features.signals]
        cols = []
        for name, unit in zip(self.feature_names, self.feature_units):
            if name not in names:
                raise ValueError(f"required feature {name} missing from input signals")
            c = names.index(name)
            if features.signals[c].unit != unit:
                raise ValueError(f"feature {name} unit differs; explicit calibration is required")
            cols.append(c)
        budget = self.memory_budget_bytes if memory_budget_bytes is None else memory_budget_bytes
        batch = _batch_size(batch_size, budget, 8*len(values)+32*self.k*len(cols), self.k, len(cols))
        labels = np.full(len(values), -1, dtype=np.int64)
        for lo in range(0, len(values), batch):
            hi = min(lo+batch, len(values))
            x, v = values[lo:hi][:, cols], valid[lo:hi][:, cols]
            supported = np.all(v, axis=1) & np.isfinite(x).all(axis=1)
            if np.any(supported):
                normalized = (x[supported]-self.scaler_mean)/self.scaler_std
                labels[lo:hi][supported] = _assign(normalized, self.centers)
        return labels

    def segment_stream(self, features: SignalSeries, *, session: SessionKey, clock=None, fps=None,
                       native_frames=None, physical_time_origin_s=None, native_frame_origin=0,
                       batch_size=10000) -> UnitStream:
        """RLE on declared native frames or a caller-specified affine clock origin.

        FPS alone cannot establish a physical clock's frame-zero origin. Physical
        times must have an explicit sample mapping or physical_time_origin_s.
        Fractional frame positions are rejected, not rounded into observations.
        """
        if fps is None or isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("segmentation requires explicit finite positive native fps")
        clock = features.time.clock if clock is None else clock
        if clock != features.time.clock:
            raise ValueError("clock differs; an explicit time-axis association is required")
        if native_frames is None and features.time.unit != "frame":
            if features.time.unit not in ("s", "ms", "ns"):
                raise ValueError(f"unsupported physical time unit: {features.time.unit}")
            scale = {"s": 1.0, "ms": 1e-3, "ns": 1e-9}[features.time.unit]
            if (physical_time_origin_s is None or isinstance(physical_time_origin_s, bool)
                    or not isinstance(physical_time_origin_s, Real) or not math.isfinite(physical_time_origin_s)):
                raise ValueError("physical times require an explicit native mapping or physical clock origin")
            if isinstance(native_frame_origin, bool) or not isinstance(native_frame_origin, Integral) or native_frame_origin < 0:
                raise ValueError("native_frame_origin must be a nonnegative integer")
            coordinates = ((features.time.values.astype(np.longdouble)
                            - np.longdouble(physical_time_origin_s)/scale)*scale*fps + native_frame_origin)
            nearest = np.rint(coordinates)
            if not np.allclose(coordinates, nearest, rtol=0, atol=1e-8):
                raise ValueError("physical clock mapping produces fractional native frames")
            native_frames = nearest
        frames = native_feature_coordinates(features, native_frames)
        if features.time.unit != "frame" and len(frames) > 1:
            scale = {"s": 1., "ms": 1e-3, "ns": 1e-9}[features.time.unit]
            if not np.allclose(np.diff(features.time.values.astype(np.longdouble))*scale,
                               np.diff(frames)/fps, rtol=1e-6, atol=1e-9):
                raise ValueError("physical sample times disagree with native mapping/fps")
        labels = self.predict_labels(features, batch_size=batch_size)
        changes = np.count_nonzero((labels[1:] != labels[:-1]) | (np.diff(frames) != 1)) + bool(len(labels))
        if 256*changes + 16*len(labels) > self.memory_budget_bytes:
            raise ValueError("primitive RLE output exceeds the memory budget")
        return from_frame_labels(frames, labels, session=session, model_id=self.model_id,
                                 clock=clock, fps=fps, catalog=self.catalog)


def _kmeans_plus_plus_init(x, k, rng):
    centers = np.empty((k, x.shape[1]), dtype=x.dtype)
    centers[0] = x[rng.integers(len(x))]
    distances = np.sum((x-centers[0])**2, axis=1)
    for i in range(1, k):
        total = distances.sum()
        probabilities = distances/total if total > 1e-12 else np.full(len(x), 1./len(x))
        centers[i] = x[rng.choice(len(x), p=probabilities)]
        distances = np.minimum(distances, np.sum((x-centers[i])**2, axis=1))
    return centers


__all__ = ["IMPLEMENTATION_STATUS", "CentroidPrimitiveSegmenter"]
