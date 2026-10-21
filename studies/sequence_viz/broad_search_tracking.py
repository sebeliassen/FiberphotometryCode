"""Bounded live diagnostic leaders with complete parameter/result provenance.

Study policy, not a universal scientific score. All observations remain in SQLite
and original job artifacts; the shortlist contains provisional diagnostic leaders
and a Pareto subset of those leaders within each cohort/method.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path


def _mean(rows, field):
    values = [(r.get(field), r.get("occurrences", 0)) for r in rows]
    values = [(float(v), int(n)) for v, n in values if v is not None and n > 0 and math.isfinite(v)]
    total = sum(n for _, n in values)
    return sum(v*n for v, n in values)/total if total else None


def result_metrics(result):
    """Keep support, coherence, separation and fragmentation as separate metrics."""
    s = result.get("summary", {})
    types = result.get("quality", {}).get("types", [])
    catalogs = result.get("quality", {}).get("catalogues", [])
    within = _mean(types, "mean_within_sequence_agreement")
    between = _mean(types, "mean_other_type_sequence_agreement")
    support = _mean(types, "known_animals")
    observed = sum(r.get("observed_types", 0) for r in catalogs)
    singleton = sum(r.get("singleton_types", 0) for r in catalogs)
    targets = result.get("conditions", {}).get("targets", [])
    return dict(coverage=s.get("coverage_fraction"), agreement=within,
        separation=within-between if within is not None and between is not None else None,
        animal_support=support/max(1, s.get("held_animals", 0)) if support is not None else None,
        singleton_fraction=singleton/observed if observed else None,
        selected_calls=s.get("selected_calls", 0), observed_types=observed,
        feature_support=s.get("feature_support_fraction"),
        mean_distinct_constituents=_mean(types, "mean_distinct_lower_types"),
        hsmm_convergence_fraction=sum(bool(t["fit_diagnostics"].get("converged")) for t in targets)/len(targets) if targets else None,
        consumed_features=result.get("quality", {}).get("physical", {}).get("features_used_in_construction", []))


class CandidateTracker:
    """Use the existing single parent writer; do not retain all results in RAM."""

    columns = ("coverage", "agreement", "separation", "animal_support", "singleton_fraction")

    def __init__(self, connection, output, atomic_writer):
        self.connection, self.output, self.write = connection, Path(output), atomic_writer
        self.updated_at = -math.inf
        connection.execute("""CREATE TABLE IF NOT EXISTS candidate_metrics(
            candidate_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, cohort TEXT NOT NULL,
            method TEXT NOT NULL, route TEXT NOT NULL, status TEXT NOT NULL,
            artifact TEXT UNIQUE NOT NULL, config_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
            coverage REAL, agreement REAL, separation REAL, animal_support REAL,
            singleton_fraction REAL, selected_calls INTEGER, seed_stability REAL)""")
        connection.execute("CREATE INDEX IF NOT EXISTS candidates_group ON candidate_metrics(cohort,method,route)")
        connection.commit()

    def add(self, job, result, artifact):
        artifact = str(Path(artifact).resolve())
        key = hashlib.sha256(artifact.encode()).hexdigest()
        m = result_metrics(result)
        recipe = job["recipe"]
        self.connection.execute("""INSERT OR IGNORE INTO candidate_metrics
            (candidate_id,job_id,cohort,method,route,status,artifact,config_json,metrics_json,
             coverage,agreement,separation,animal_support,singleton_fraction,selected_calls)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (key, job["job_id"], job["cohort"], recipe["method"], recipe.get("route", "unspecified"), result["status"], artifact,
             json.dumps(job, sort_keys=True, allow_nan=False), json.dumps(m, sort_keys=True, allow_nan=False),
             *(m[c] for c in self.columns), m["selected_calls"]))
        self.connection.commit()

    def add_pair(self, pair):
        value = pair.get("comparison", {}).get("matched_type_temporal_f1")
        if value is None:
            return
        self.connection.executemany("UPDATE candidate_metrics SET seed_stability=? WHERE job_id=?",
            [(value, pair[k]["job_id"]) for k in ("reference", "alternative")])
        self.connection.commit()

    def rebuild(self, reference_outputs=()):
        """Recover missing compact rows after interruption, including baseline runs."""
        locations = [self.output, *(Path(p) for p in reference_outputs)]
        known = {r[0] for r in self.connection.execute("SELECT artifact FROM candidate_metrics")}
        for location in locations:
            for artifact in sorted((location/"jobs").glob("*.json")):
                if str(artifact.resolve()) not in known:
                    payload = json.loads(artifact.read_text())
                    self.add(payload["job"], payload["result"], artifact)
            for path in (location/"seed_stability").glob("*.json"):
                self.add_pair(json.loads(path.read_text()))

    def export(self, *, force=False, interval_s=60):
        if not force and time.monotonic()-self.updated_at < interval_s:
            return
        self.updated_at = time.monotonic()
        groups = self.connection.execute("SELECT DISTINCT cohort,method,route FROM candidate_metrics ORDER BY 1,2,3").fetchall()
        leaders, pool = [], {}
        for cohort, method, route in groups:
            for metric in (*self.columns, "seed_stability"):
                direction = "ASC" if metric == "singleton_fraction" else "DESC"
                rows = self.connection.execute(f"""SELECT candidate_id,config_json,metrics_json,artifact,seed_stability
                    FROM candidate_metrics WHERE cohort=? AND method=? AND route=? AND status='SUCCESS'
                    AND selected_calls>=3 AND {metric} IS NOT NULL
                    ORDER BY {metric} {direction},candidate_id LIMIT 3""", (cohort, method, route)).fetchall()
                ids = []
                for key, config, metrics, artifact, stability in rows:
                    pool[key] = dict(candidate_id=key, job=json.loads(config), metrics=json.loads(metrics),
                                     artifact=artifact, seed_stability=stability)
                    ids.append(key)
                leaders.append(dict(cohort=cohort, method=method, route=route, metric=metric, candidates=ids))
        pareto = defaultdict(list)
        for item in pool.values():
            m = item["metrics"]
            if any(m[c] is None for c in ("coverage", "agreement", "animal_support", "singleton_fraction")):
                continue
            vector = (m["coverage"], m["agreement"], m["animal_support"], 1-m["singleton_fraction"])
            key = (item["job"]["cohort"], item["job"]["recipe"]["method"])
            pareto[key].append((vector, item["candidate_id"]))
        panels = []
        for (cohort, method), candidates in sorted(pareto.items()):
            front = []
            for vector, candidate_id in candidates:
                if not any(all(x >= y for x, y in zip(other, vector)) and any(x > y for x, y in zip(other, vector))
                           for other, other_id in candidates if other_id != candidate_id):
                    front.append(candidate_id)
            panels.append(dict(cohort=cohort, method=method, candidates=front))
        count = self.connection.execute("SELECT COUNT(*) FROM candidate_metrics").fetchone()[0]
        self.write(self.output/"best_candidates.json", dict(tracked_results=count,
            policy="Provisional leaders per cohort/method/route; Pareto subset among leaders, not exhaustive global frontier; no phenotype/DA score or semantic winner",
            eligibility="SUCCESS and at least3selected calls; missing metrics remain missing",
            metrics=list(self.columns)+["seed_stability"], leaders=leaders, pareto_among_leaders=panels,
            candidates=list(pool.values()), updated_at_unix=time.time()))
