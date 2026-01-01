
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from math import radians, sin, cos, asin, sqrt


# ---------------------------------------------
# Utility: Haversine distance (km)
# ---------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c


# ---------------------------------------------
# 1) Prepare interactions with time decay & weak signals
# ---------------------------------------------
def prepare_interactions(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "gmap_id",
    rating_col: str = "rating",
    time_col: str = "review_time",
    text_col: Optional[str] = "text",
    pics_col: Optional[str] = "pics",
    resp_col: Optional[str] = "resp",
    half_life_days: float = 180.0,
    min_clip: float = 0.0,
    max_clip: float = 1.0
) -> pd.DataFrame:
    """
    Returns a clean interactions dataframe with columns:
      [user_col, item_col, 'x_ui', 'timestamp']
    where x_ui is the decayed, clipped interaction strength in [0,1].
    """
    df = df.copy()

    # Ensure time is datetime
    if time_col in df.columns:
        df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        # If no time available, use a constant timestamp to avoid NaT issues.
        df["timestamp"] = pd.Timestamp("2000-01-01")

    # Base r_ui from rating (scaled 1-5 -> 0-1) or implicit 0.6 if review exists
    r_ui = pd.Series(0.0, index=df.index, dtype=float)
    if rating_col in df.columns:
        # If rating is present, map 1..5 to 0..1 (robust to missing)
        r = pd.to_numeric(df[rating_col], errors="coerce")
        r_scaled = (r - 1.0) / 4.0  # 1->0, 5->1
        r_ui = r_scaled.fillna(0.0)
    else:
        r_ui = pd.Series(0.0, index=df.index, dtype=float)

    # If a review row exists without a rating, treat as implicit positive 0.6
    implicit_mask = r_ui.isna() | (r_ui == 0.0)
    if text_col and text_col in df.columns:
        implicit_mask = implicit_mask & df[text_col].notna()
    r_ui = r_ui.mask(implicit_mask, 0.6)

    # Weak bonuses
    bonus = pd.Series(0.0, index=df.index, dtype=float)
    if text_col and text_col in df.columns:
        bonus += (df[text_col].notna()).astype(float) * 0.05
    if pics_col and pics_col in df.columns:
        bonus += (df[pics_col].notna()).astype(float) * 0.05
    if resp_col and resp_col in df.columns:
        bonus += (df[resp_col].notna()).astype(float) * 0.02

    base = (r_ui.fillna(0.0) + bonus).clip(lower=min_clip, upper=max_clip)

    # Time decay with half-life
    latest_time = df["timestamp"].max()
    delta_days = (latest_time - df["timestamp"]).dt.days.clip(lower=0).fillna(0)
    wt = 0.5 ** (delta_days / max(half_life_days, 1e-6))

    x_ui = (base * wt).clip(lower=min_clip, upper=max_clip)

    interactions = df[[user_col, item_col]].copy()
    interactions["x_ui"] = x_ui
    interactions["timestamp"] = df["timestamp"]
    interactions.dropna(subset=[user_col, item_col], inplace=True)

    return interactions


# ---------------------------------------------
# 2) Build user–item matrix
# ---------------------------------------------
def build_user_item_matrix(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "gmap_id",
    value_col: str = "x_ui",
    min_user_interactions: int = 2,
    min_item_interactions: int = 2
) -> Tuple[csr_matrix, pd.Index, pd.Index]:
    df = interactions[[user_col, item_col, value_col]].copy()

    # Filter sparse users/items
    if min_user_interactions > 1:
        user_counts = df.groupby(user_col)[item_col].count()
        keep_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df[user_col].isin(keep_users)]
    if min_item_interactions > 1:
        item_counts = df.groupby(item_col)[user_col].count()
        keep_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df[item_col].isin(keep_items)]

    if df.empty:
        return csr_matrix((0,0)), pd.Index([]), pd.Index([])

    df[user_col] = df[user_col].astype("category")
    df[item_col] = df[item_col].astype("category")

    user_index = df[user_col].cat.categories
    item_index = df[item_col].cat.categories

    rows = df[user_col].cat.codes.values
    cols = df[item_col].cat.codes.values
    data = df[value_col].astype(float).values

    X = csr_matrix((data, (rows, cols)), shape=(len(user_index), len(item_index)))
    return X, user_index, item_index


# ---------------------------------------------
# 3) Fit item-based KNN and compute item–item similarities
# ---------------------------------------------
def fit_item_knn(
    X: csr_matrix,
    item_index: pd.Index,
    n_neighbors: int = 50,
    metric: str = "cosine"
) -> Tuple[NearestNeighbors, np.ndarray, np.ndarray]:
    """
    Returns:
      knn_model: fitted NearestNeighbors
      item_sim:  dense item–item similarity matrix for top-n neighbors per item (others 0)
      item_nbrs: neighbor indices for each item (including self at 0 distance for many metrics)
    """
    if X.shape[1] == 0:
        return None, np.zeros((0,0)), np.zeros((0,0))

    knn = NearestNeighbors(n_neighbors=min(n_neighbors, X.shape[1]), metric=metric, algorithm="auto")
    knn.fit(X.T)

    dists, indices = knn.kneighbors(X.T, return_distance=True)  # (n_items, n_neighbors)

    if metric == "cosine":
        sims = 1.0 - dists
    else:
        sims = 1.0 / (1.0 + dists)

    n_items = X.shape[1]
    item_sim = np.zeros((n_items, n_items), dtype=np.float32)

    for i in range(n_items):
        nbr_idx = indices[i]
        nbr_sim = sims[i]
        item_sim[i, nbr_idx] = np.maximum(nbr_sim, 0.0)

    return knn, item_sim, indices


# ---------------------------------------------
# Internal: score vector for a single user via neighbor aggregation
# ---------------------------------------------
def _score_items_for_user(
    user_row: np.ndarray,
    item_sim: np.ndarray
) -> np.ndarray:
    numer = item_sim @ user_row
    denom = (np.abs(item_sim).sum(axis=1) + 1e-12)
    scores = numer / denom
    return np.asarray(scores).ravel()


# ---------------------------------------------
# 4) Recommend for a single user
# ---------------------------------------------
def recommend_for_user(
    user_id,
    X: csr_matrix,
    user_index: pd.Index,
    item_index: pd.Index,
    item_sim: np.ndarray,
    topn: int = 10,
    already_consumed: bool = True,
    item_meta_df: Optional[pd.DataFrame] = None,
    geography_boost: Optional[Dict] = None,
    user_home_state: Optional[str] = None
) -> pd.DataFrame:
    if len(user_index) == 0 or len(item_index) == 0:
        return pd.DataFrame(columns=["gmap_id","score"])

    if user_id not in set(user_index.tolist()):
        return pd.DataFrame(columns=["gmap_id","score"])

    u = int(np.where(user_index == user_id)[0][0])
    user_row = X[u].toarray().ravel()
    scores = _score_items_for_user(user_row, item_sim)

    if already_consumed:
        seen_mask = user_row > 0
        scores[seen_mask] = -np.inf

    if item_meta_df is not None and geography_boost and geography_boost.get("enabled", False):
        meta = item_meta_df.drop_duplicates("gmap_id").copy()
        meta = meta.set_index("gmap_id").reindex(item_index, copy=False)

        state_weight = float(geography_boost.get("state_weight", 0.0))
        if state_weight and user_home_state and "state" in meta.columns:
            same_state = (meta["state"] == user_home_state).astype(float).fillna(0.0).values
            scores = scores + state_weight * same_state

        dist_weight = float(geography_boost.get("distance_km_weight", 0.0))
        max_km = float(geography_boost.get("max_km", 30.0))
        if dist_weight and {"latitude","longitude"}.issubset(meta.columns):
            seen_idx = np.where(user_row > 0)[0]
            if len(seen_idx) > 0:
                seen_meta = meta.iloc[seen_idx][["latitude","longitude"]].dropna()
                if len(seen_meta) > 0:
                    u_lat = seen_meta["latitude"].astype(float).mean()
                    u_lon = seen_meta["longitude"].astype(float).mean()
                    lat = meta["latitude"].astype(float)
                    lon = meta["longitude"].astype(float)
                    d = []
                    for la, lo in zip(lat, lon):
                        if pd.isna(la) or pd.isna(lo):
                            d.append(np.nan)
                        else:
                            d.append(haversine_km(u_lat, u_lon, la, lo))
                    d = pd.Series(d, index=meta.index).fillna(max_km * 2)
                    prox = (max_km - d.clip(upper=max_km)) / max_km
                    prox = prox.clip(lower=0.0, upper=1.0).values
                    scores = scores + dist_weight * prox

    if np.all(np.isneginf(scores)):
        return pd.DataFrame(columns=["gmap_id","score"])

    k = min(topn, np.isfinite(scores).sum())
    top_idx = np.argpartition(-scores, kth=max(k-1, 0))[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    top_items = item_index[top_idx]
    top_scores = scores[top_idx]

    out = pd.DataFrame({"gmap_id": top_items, "score": top_scores})
    if item_meta_df is not None:
        out = out.merge(
            item_meta_df.drop_duplicates("gmap_id"),
            on="gmap_id",
            how="left"
        )

    return out.reset_index(drop=True)


# ---------------------------------------------
# 5) Recommend for all users
# ---------------------------------------------
def recommend_for_all_users(
    X: csr_matrix,
    user_index: pd.Index,
    item_index: pd.Index,
    item_sim: np.ndarray,
    topn: int = 10
) -> Dict:
    all_recs = {}
    for u, uid in enumerate(user_index):
        user_row = X[u].toarray().ravel()
        scores = _score_items_for_user(user_row, item_sim)
        scores[user_row > 0] = -np.inf  # exclude seen
        k = min(topn, np.isfinite(scores).sum())
        if k == 0:
            all_recs[uid] = []
            continue
        top_idx = np.argpartition(-scores, kth=max(k-1, 0))[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        all_recs[uid] = list(zip(item_index[top_idx].tolist(), scores[top_idx].tolist()))
    return all_recs


# ---------------------------------------------
# Metrics helpers
# ---------------------------------------------
def _precision_at_k(pred, truth, K):
    if K == 0: return 0.0
    return len(pred[:K] & truth) / float(K)

def _recall_at_k(pred, truth, K):
    if len(truth) == 0: return np.nan
    return len(pred[:K] & truth) / float(len(truth))

def _ndcg_at_k(pred_list, truth_set, K):
    dcg = 0.0
    for i, item in enumerate(pred_list[:K]):
        if item in truth_set:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(truth_set), K)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return 0.0 if idcg == 0 else dcg / idcg


# ---------------------------------------------
# 6) Time-aware evaluation (train <= cutoff, test > cutoff)
# ---------------------------------------------
def evaluate_time_split(
    df_full: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    user_col: str = "user_id",
    item_col: str = "gmap_id",
    rating_col: str = "rating",
    time_col: str = "review_time",
    K_list: List[int] = [5,10]
) -> pd.DataFrame:
    df_full = df_full.copy()
    df_full["timestamp"] = pd.to_datetime(df_full[time_col], errors="coerce")
    train_df = df_full[df_full["timestamp"] <= cutoff_date]
    test_df  = df_full[df_full["timestamp"] > cutoff_date]

    interactions = prepare_interactions(
        train_df,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
        time_col=time_col
    )
    if interactions.empty:
        return pd.DataFrame({"metric": [], "K": [], "value": []})

    X, user_index, item_index = build_user_item_matrix(interactions)
    if X.shape[0] == 0 or X.shape[1] == 0:
        return pd.DataFrame({"metric": [], "K": [], "value": []})

    _, item_sim, _ = fit_item_knn(X, item_index, n_neighbors=min(50, X.shape[1]))

    test_df = test_df[[user_col, item_col]].dropna().drop_duplicates()
    test_users = set(test_df[user_col].unique()).intersection(set(user_index))

    results = []
    for K in K_list:
        precs, recs, ndcgs = [], [], []
        for uid in test_users:
            u_idx = int(np.where(user_index == uid)[0][0])
            user_row = X[u_idx].toarray().ravel()
            scores = _score_items_for_user(user_row, item_sim)
            scores[user_row > 0] = -np.inf

            k = min(K, np.isfinite(scores).sum())
            if k == 0:
                continue

            top_idx = np.argpartition(-scores, kth=max(k-1, 0))[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            pred_items = set(item_index[top_idx].tolist())
            pred_list  = item_index[top_idx].tolist()

            truth_items = set(test_df.loc[test_df[user_col] == uid, item_col].tolist())

            precs.append(_precision_at_k(pred_items, truth_items, K))
            r = _recall_at_k(pred_items, truth_items, K)
            if not np.isnan(r):
                recs.append(r)
            ndcgs.append(_ndcg_at_k(pred_list, truth_items, K))

        if len(precs) == 0:
            p_mean, r_mean, nd_mean = np.nan, np.nan, np.nan
        else:
            p_mean = float(np.nanmean(precs))
            r_mean = float(np.nanmean(recs)) if len(recs) else np.nan
            nd_mean = float(np.nanmean(ndcgs))

        results.extend([
            {"metric": "Precision", "K": K, "value": p_mean},
            {"metric": "Recall",    "K": K, "value": r_mean},
            {"metric": "NDCG",      "K": K, "value": nd_mean},
        ])

    return pd.DataFrame(results)
