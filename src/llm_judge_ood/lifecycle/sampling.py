from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans


def entropy(probabilities: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def active_label_sample(
    *,
    eligible_document_indices: np.ndarray,
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    budget: int,
    exclude_indices: set[int] | None = None,
    seed: int = 42,
    eligible_multiplier: int = 4,
) -> np.ndarray:
    eligible_document_indices = np.asarray(eligible_document_indices, dtype=int)
    if exclude_indices:
        eligible_document_indices = np.asarray(
            [idx for idx in eligible_document_indices if int(idx) not in exclude_indices], dtype=int
        )
    if budget <= 0 or eligible_document_indices.size == 0:
        return np.zeros(0, dtype=int)
    if eligible_document_indices.size <= budget:
        return eligible_document_indices
    uncertainty = entropy(probabilities[eligible_document_indices])
    keep_n = min(eligible_document_indices.size, max(int(budget), int(budget) * int(eligible_multiplier)))
    shortlist = eligible_document_indices[np.argsort(-uncertainty, kind="stable")[:keep_n]]
    if shortlist.size <= budget:
        return shortlist
    local_embeddings = np.asarray(embeddings, dtype=np.float32)[shortlist]
    distinct_embeddings = int(np.unique(local_embeddings, axis=0).shape[0])
    if distinct_embeddings <= 1:
        return shortlist[: int(budget)]
    n_clusters = min(int(budget), local_embeddings.shape[0], distinct_embeddings)
    labels = KMeans(n_clusters=n_clusters, n_init=5, random_state=seed).fit_predict(local_embeddings)
    selected: list[int] = []
    # KMeans may return fewer occupied labels than requested when eligible documents
    # have duplicate embeddings.  Iterate over actual labels, then use the
    # uncertainty-ranked shortlist to fill any unused budget.
    for cluster_id in sorted(np.unique(labels).tolist()):
        members = np.where(labels == cluster_id)[0]
        if members.size == 0:
            continue
        centroid = local_embeddings[members].mean(axis=0)
        member_distances = np.linalg.norm(local_embeddings[members] - centroid, axis=1)
        selected.append(int(shortlist[members[int(np.argmin(member_distances))]]))
    if len(selected) < int(budget):
        selected_set = set(selected)
        selected.extend(
            int(index)
            for index in shortlist.tolist()
            if int(index) not in selected_set
        )
    return np.asarray(selected[:budget], dtype=int)


def stratified_active_label_sample(
    *,
    eligible_document_indices: np.ndarray,
    strata: np.ndarray,
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    budget: int,
    seed: int = 42,
) -> np.ndarray:
    """Run uncertainty/diversity sampling while covering document clusters."""

    eligible_document_indices = np.asarray(eligible_document_indices, dtype=int)
    strata = np.asarray(strata).astype(str)
    if budget <= 0 or eligible_document_indices.size == 0:
        return np.zeros(0, dtype=int)
    if eligible_document_indices.size <= budget:
        return eligible_document_indices
    unique = sorted(set(strata[eligible_document_indices].tolist()))
    if len(unique) <= 1:
        return active_label_sample(
            eligible_document_indices=eligible_document_indices,
            embeddings=embeddings,
            probabilities=probabilities,
            budget=budget,
            seed=seed,
        )
    rng = np.random.default_rng(int(seed))
    order = [unique[index] for index in rng.permutation(len(unique)).tolist()]
    base, remainder = divmod(int(budget), len(order))
    selected: list[int] = []
    for position, value in enumerate(order):
        quota = base + int(position < remainder)
        if quota <= 0:
            continue
        local = eligible_document_indices[strata[eligible_document_indices] == value]
        picked = active_label_sample(
            eligible_document_indices=local,
            embeddings=embeddings,
            probabilities=probabilities,
            budget=quota,
            seed=int(seed) + position + 1,
        )
        selected.extend(picked.astype(int).tolist())
    if len(selected) < int(budget):
        selected_set = set(selected)
        remaining = np.asarray(
            [index for index in eligible_document_indices.tolist() if int(index) not in selected_set],
            dtype=int,
        )
        fill = active_label_sample(
            eligible_document_indices=remaining,
            embeddings=embeddings,
            probabilities=probabilities,
            budget=int(budget) - len(selected),
            seed=int(seed) + len(order) + 1,
        )
        selected.extend(fill.astype(int).tolist())
    return np.asarray(selected[: int(budget)], dtype=int)


def stratified_random_sample(indices: np.ndarray, strata: np.ndarray, *, budget: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices, dtype=int)
    strata = np.asarray(strata).astype(str)
    if indices.size <= budget:
        return indices
    selected: list[int] = []
    unique = sorted(set(strata[indices].tolist()))
    per = max(1, budget // max(len(unique), 1))
    for value in unique:
        local = indices[strata[indices] == value]
        take = min(per, len(local), budget - len(selected))
        if take > 0:
            selected.extend(rng.choice(local, size=take, replace=False).astype(int).tolist())
    remaining = [idx for idx in indices.tolist() if idx not in set(selected)]
    if len(selected) < budget and remaining:
        selected.extend(rng.choice(np.asarray(remaining), size=min(len(remaining), budget - len(selected)), replace=False).astype(int).tolist())
    return np.asarray(selected[:budget], dtype=int)
