"""Frozen-centroid primitive baseline with training-only scaling and RAM caps."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType

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


@dataclass(frozen=True)
class WholeBoutCodebook:
    """Training-only feature/joint/parent-subtype identities on unchanged intervals.

    Parent IDs are categories. Joint distances use a separately weighted one-hot
    block; no arithmetic on the numerical value of an ID is performed. Missing
    feature rows and unseen categorical parents remain unsupported (empty string).
    Resource caps cover owned arrays; caller-owned input storage is excluded.
    """

    route: str
    feature_names: tuple[str, ...]
    location: np.ndarray
    scale: np.ndarray
    parents: tuple[str, ...]
    centers: dict
    parent_weight: float = 1.
    memory_budget_bytes: int = 256*1024**2

    def __post_init__(self):
        names, parents = tuple(self.feature_names), tuple(self.parents)
        budget = _positive(self.memory_budget_bytes, "memory_budget_bytes")
        if (self.route not in {"feature", "joint", "within_parent"} or not names
                or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names)
                or not parents or len(set(parents)) != len(parents)
                or any(not isinstance(p, str) or not p for p in parents)):
            raise ValueError("codebook schema requires supported route and unique nonempty feature/parent names")
        if (isinstance(self.parent_weight, bool) or not isinstance(self.parent_weight, Real)
                or not np.isfinite(self.parent_weight) or self.parent_weight < 0):
            raise ValueError("parent_weight must be finite and nonnegative")
        expected_keys = set(parents) if self.route == "within_parent" else {"all"}
        if not isinstance(self.centers, (dict, MappingProxyType)) or set(self.centers) != expected_keys:
            raise ValueError("codebook center groups differ from declared route/parents")
        f = len(names) + (len(parents) if self.route == "joint" else 0)
        raw_arrays = [np.asarray(self.location), np.asarray(self.scale)] + [np.asarray(c) for c in self.centers.values()]
        if 32*sum(a.size for a in raw_arrays) > budget:
            raise MemoryError("whole-bout parameter array budget exceeded")
        location, scale = frozen_array(self.location, dtype=float), frozen_array(self.scale, dtype=float)
        centers = {p: frozen_array(c, dtype=float) for p, c in self.centers.items()}
        if (location.shape != (len(names),) or scale.shape != location.shape
                or not np.isfinite(location).all() or not np.isfinite(scale).all() or np.any(scale <= 0)
                or any(c.ndim != 2 or not len(c) or c.shape[1] != f or not np.isfinite(c).all()
                       for c in centers.values())):
            raise ValueError("codebook fitted arrays disagree with schema or finite positive scales")
        if 4*(location.nbytes+scale.nbytes+sum(c.nbytes for c in centers.values())) > budget:
            raise MemoryError("whole-bout parameter array budget exceeded")
        for name, value in (("feature_names", names), ("parents", parents), ("location", location),
                            ("scale", scale), ("centers", MappingProxyType(centers)),
                            ("parent_weight", float(self.parent_weight)), ("memory_budget_bytes", budget)):
            object.__setattr__(self, name, value)

    @classmethod
    def fit(cls, values, parent_ids, *, feature_names, route="feature", clusters=8,
            seed=101, iterations=40, parent_weight=1., memory_budget_bytes=256*1024**2):
        values, parent_ids = np.asarray(values, dtype=float), np.asarray(parent_ids, dtype=str)
        _positive(clusters, "clusters")
        _positive(iterations, "iterations")
        _positive(memory_budget_bytes, "memory_budget_bytes")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        feature_names = tuple(feature_names)
        if (route not in {"feature", "joint", "within_parent"} or values.ndim != 2
                or parent_ids.shape != (len(values),) or values.shape[1] != len(feature_names)
                or not len(feature_names) or len(set(feature_names)) != len(feature_names)
                or any(not isinstance(n, str) or not n for n in feature_names)
                or isinstance(parent_weight, bool) or not isinstance(parent_weight, Real)
                or not np.isfinite(parent_weight) or parent_weight < 0):
            raise ValueError("explicit whole-bout feature schema, categorical parents and supported creator required")
        valid = np.isfinite(values).all(axis=1) & (parent_ids != "")
        if valid.sum() < clusters:
            raise ValueError("insufficient finite training bouts for codebook")
        parents = tuple(sorted(set(parent_ids[valid])))
        f = len(feature_names) + (len(parents) if route == "joint" else 0)
        estimated = len(values)*(64*f+32) + 64*clusters*f
        if estimated > memory_budget_bytes:
            raise MemoryError("whole-bout training array budget exceeded")
        x = values[valid]
        location = np.median(x, axis=0)
        scale = np.quantile(x, .75, axis=0)-np.quantile(x, .25, axis=0)
        scale = np.where(scale > 1e-9, scale, 1.)
        x = np.clip((x-location)/scale, -8, 8)/np.sqrt(values.shape[1])
        ids = parent_ids[valid]
        if route == "joint":
            crosswalk = {p: i for i, p in enumerate(parents)}
            cat = np.zeros((len(ids), len(parents)))
            cat[np.arange(len(ids)), [crosswalk[p] for p in ids]] = np.sqrt(parent_weight/2)
            x = np.column_stack([x, cat])
        rng = np.random.default_rng(seed)
        centers = {}
        groups = parents if route == "within_parent" else ("all",)
        for parent in groups:
            part = x[ids == parent] if route == "within_parent" else x
            k = min(clusters, max(1, len(part)//5)) if route == "within_parent" else clusters
            fitted = _kmeans_plus_plus_init(part, k, rng)
            batch = _batch_size(10000, memory_budget_bytes, estimated, k, f)
            for _ in range(iterations):
                labels = np.empty(len(part), dtype=np.int64)
                for lo in range(0, len(part), batch):
                    labels[lo:lo+batch] = _assign(part[lo:lo+batch], fitted)
                sums = np.zeros_like(fitted)
                np.add.at(sums, labels, part)
                count = np.bincount(labels, minlength=k)
                updated = np.divide(sums, count[:, None], out=fitted.copy(), where=count[:, None] > 0)
                done = np.max(np.abs(updated-fitted)) < 1e-5
                fitted = updated
                if done:
                    break
            centers[parent] = frozen_array(fitted, dtype=float)
        return cls(route, tuple(feature_names), frozen_array(location, dtype=float),
                   frozen_array(scale, dtype=float), parents, centers, float(parent_weight), memory_budget_bytes)

    def assign(self, values, parent_ids, *, batch_size=10000, memory_budget_bytes=None):
        values, ids = np.asarray(values, dtype=float), np.asarray(parent_ids, dtype=str)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names) or ids.shape != (len(values),):
            raise ValueError("assignment schema differs from fitted whole-bout codebook")
        budget = self.memory_budget_bytes if memory_budget_bytes is None else memory_budget_bytes
        width = max(map(len, self.catalog_ids))
        k = max(len(c) for c in self.centers.values())
        f = len(self.feature_names)+(len(self.parents) if self.route == "joint" else 0)
        fixed = len(values)*width*4 + sum(c.nbytes for c in self.centers.values())
        batch = _batch_size(batch_size, budget, fixed, k, f)
        output = np.full(len(values), "", dtype=f"U{width}")
        crosswalk = {p: i for i, p in enumerate(self.parents)}
        for lo in range(0, len(values), batch):
            hi = min(lo+batch, len(values))
            valid = np.isfinite(values[lo:hi]).all(axis=1)
            positions = np.flatnonzero(valid)+lo
            x = np.clip((values[positions]-self.location)/self.scale, -8, 8)/np.sqrt(len(self.feature_names))
            if self.route == "joint":
                supported = np.array([p in crosswalk for p in ids[positions]], dtype=bool)
                positions, x = positions[supported], x[supported]
                cat = np.zeros((len(positions), len(self.parents)))
                cat[np.arange(len(positions)), [crosswalk[p] for p in ids[positions]]] = np.sqrt(self.parent_weight/2)
                x = np.column_stack([x, cat])
            if self.route == "within_parent":
                for parent, centers in self.centers.items():
                    use = ids[positions] == parent
                    if np.any(use):
                        output[positions[use]] = [f"{parent}:v{c}" for c in _assign(x[use], centers)]
            elif len(x):
                output[positions] = [f"type_{c}" for c in _assign(x, self.centers["all"])]
        return output

    @property
    def catalog_ids(self):
        if self.route == "within_parent":
            return tuple(f"{p}:v{i}" for p in self.parents for i in range(len(self.centers[p])))
        return tuple(f"type_{i}" for i in range(len(self.centers["all"])))


__all__ = ["IMPLEMENTATION_STATUS", "CentroidPrimitiveSegmenter", "WholeBoutCodebook"]
