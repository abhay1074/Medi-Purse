"""Reference live retriever for the fusion v4 SHAP-8 system.

Loads the shipped weights and ranks the full 1,464-disease catalog for one raw
case text. This is the inference path the production backend should copy; it
does not use any precomputed per-case artifacts.

    from reference_retriever import Retriever
    r = Retriever("weights/retriever")
    out = r.rank(case_text, k=10)
    out["top"]        # [{"disease_id", "name", "score", "rank"}, ...]
    out["trace"]      # per-candidate component scores (for explanations)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np


FEATURES = [
    "disease_svd_0", "rank_min", "tfidf_z", "tfidf_minmax",
    "tfidf_rank", "svd_cosine", "tfidf_x_bow", "tfidf_recip_rank",
]


def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / max(hi - lo, 1e-9)


def _zscore(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / max(float(a.std()), 1e-9)


def _core(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


class Retriever:
    def __init__(self, asset_dir: str | Path):
        a = Path(asset_dir)
        self.diseases = json.load((a / "diseases.json").open(encoding="utf-8"))["diseases"]
        self.ids = [d["id"] for d in self.diseases]
        self.names = [d["name"] for d in self.diseases]
        self.index = {d: i for i, d in enumerate(self.ids)}

        self.tfidf = joblib.load(a / "tfidf_vectorizer.joblib")
        self.clf = joblib.load(a / "tfidf_classifier.joblib")
        self.bow = joblib.load(a / "bow_vectorizer_mass80.pkl")
        from scipy import sparse
        self.centroids = sparse.load_npz(a / "bow_centroids_mass80.npz").tocsr()
        self.vocab = self.bow.get_feature_names_out()
        self.removed = set(json.load((a / "prune_sets_mass80.json").open())["mass80"])
        self.svd = joblib.load(a / "svd32.joblib")
        self.disease_svd = np.load(a / "disease_svd32.npy").astype(np.float32)
        self.booster = lgb.Booster(model_file=str(a / "gbm_v4_shap8.txt"))
        cfg = json.load((a / "retriever_config.json").open(encoding="utf-8"))
        self.bow_weight = float(cfg.get("bow_weight", 0.6))
        self.gbm_weight = float(cfg.get("gbm_weight", 0.4))
        self.pad = float(cfg.get("pool_pad", -0.3))
        self.k_pool = int(cfg.get("k_pool", 100))
        assert len(self.ids) == self.centroids.shape[0] == len(self.disease_svd)

    def prune(self, text: str) -> str:
        return " ".join(t for t in str(text).split() if _core(t) not in self.removed)

    def rank(self, text: str, k: int = 10) -> dict:
        n_dis = len(self.ids)
        Xt = self.tfidf.transform([text])
        proba = np.asarray(self.clf.predict_proba(Xt)[0], dtype=np.float32)

        pool = np.argpartition(-proba, self.k_pool)[: self.k_pool]
        pool = pool[np.argsort(-proba[pool], kind="stable")]

        Xb = self.bow.transform([self.prune(text)])
        S = (Xb @ self.centroids.T).toarray().astype(np.float32)[0]
        bow_order = np.argsort(-S)
        bow_rank = np.empty(n_dis, dtype=np.int32)
        bow_rank[bow_order] = np.arange(1, n_dis + 1)

        tf = proba[pool]
        tf_rank = np.arange(1, self.k_pool + 1, dtype=np.float32)
        tf_mm = _minmax(tf)
        tf_z = _zscore(tf)
        bs = S[pool]
        bmm = _minmax(bs)
        br = bow_rank[pool].astype(np.float32)

        case_svd = self.svd.transform(Xb).astype(np.float32)[0]
        dv = self.disease_svd[pool]
        inter = case_svd[None, :] * dv
        denom = np.maximum(np.linalg.norm(case_svd) *
                           np.maximum(np.linalg.norm(dv, axis=1), 1e-9), 1e-9)
        svd_cos = inter.sum(axis=1) / denom

        cols = {
            "disease_svd_0": dv[:, 0],
            "rank_min": np.minimum(tf_rank, br),
            "tfidf_z": tf_z,
            "tfidf_minmax": tf_mm,
            "tfidf_rank": tf_rank,
            "svd_cosine": svd_cos,
            "tfidf_x_bow": tf_mm * bmm,
            "tfidf_recip_rank": 1.0 / np.maximum(tf_rank, 1.0),
        }
        X8 = np.stack([cols[f] for f in FEATURES], axis=1).astype(np.float32)
        pred = self.booster.predict(X8).astype(np.float32)
        pred_mm = _minmax(pred)

        mm = np.full(n_dis, np.nan, dtype=np.float32)
        mm[pool] = pred_mm
        mmb = _minmax(S)
        full = (self.bow_weight * mmb
                + self.gbm_weight * (np.where(np.isnan(mm), self.pad, mm) + 1e-4 * mmb))
        order = np.argsort(-full, kind="stable")[:k]

        case_norm = " " + re.sub(r"[^a-z0-9]+", " ", str(text).lower()) + " "
        trace = []
        for j in order:
            j = int(j)
            contrib = Xb.multiply(self.centroids[j]).tocsr()
            take = np.argsort(-contrib.data, kind="stable")[:8]
            matched = []
            for z in take:
                term = str(self.vocab[contrib.indices[z]])
                term_norm = re.sub(r"[^a-z0-9]+", " ", term.lower()).strip()
                if term_norm and f" {term_norm} " in case_norm:
                    matched.append({"term": term, "weight": float(contrib.data[z])})
            matched = matched[:5]
            pos = np.where(pool == j)[0]
            gbm = []
            if len(pos):
                c = self.booster.predict(X8[pos], pred_contrib=True)[0]
                pairs = sorted(zip(FEATURES, c[:-1]), key=lambda x: -abs(x[1]))
                gbm = [{"feature": f, "contribution": float(v)} for f, v in pairs[:3]]
            trace.append({
                "disease_id": self.ids[j],
                "tfidf_score": float(proba[j]),
                "tfidf_rank": int((proba > proba[j]).sum()) + 1,
                "bow_score": float(S[j]),
                "bow_rank": int(bow_rank[j]),
                "pooled": bool(len(pos)),
                "matched_terms": matched,
                "gbm_contributions": gbm,
            })

        return {
            "top": [
                {"rank": int(r + 1), "disease_id": self.ids[j],
                 "name": self.names[j], "score": float(full[j])}
                for r, j in enumerate(order)
            ],
            "trace": trace,
            "catalog_size": n_dis,
        }
