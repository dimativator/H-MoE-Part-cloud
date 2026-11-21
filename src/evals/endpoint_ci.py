"""Percentile intervals from the Stage 3 reporting protocol."""
import os

import numpy as np
import torch.nn.functional as F


def state_dict(rows):
    result = {}
    for doc_id, cont_id, value in rows:
        if doc_id is None or cont_id is None or value is None:
            continue
        result.setdefault(int(doc_id), {}).setdefault(int(cont_id), float(value))
    return result


def per_document_values(metric) -> tuple[str, np.ndarray]:
    loglikelihoods = state_dict(metric.loglikelihoods)
    labels = {
        int(doc_id): int(value)
        for doc_id, _, value in metric.labels
        if doc_id is not None and value is not None
    }
    if metric.metric_type == "bpb":
        bpbs = state_dict(metric.bpbs)
        values = [bpbs[doc_id][0] for doc_id in sorted(loglikelihoods)]
        return "bpb_v2", np.asarray(values, dtype=np.float64)

    values = []
    for doc_id in sorted(loglikelihoods):
        choices = loglikelihoods[doc_id]
        ordered = [choices[index] for index in range(len(choices))]
        prediction = int(np.argmax(np.asarray(ordered)))
        values.append(float(prediction == labels[doc_id]))
    return f"{metric.metric_type}_v2", np.asarray(values, dtype=np.float64)


def bootstrap_means(values, replicates, seed, *, binary=False, chunk_size=128):
    rng = np.random.default_rng(seed)
    if binary:
        successes = int(values.sum())
        return rng.binomial(values.size, successes / values.size, size=replicates) / values.size
    output = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        output[start:stop] = values[indices].mean(axis=1)
    return output


def summarize(values, replicates, seed, *, binary=False):
    bootstrap = bootstrap_means(values, replicates, seed, binary=binary)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "n": int(values.size),
        "point_estimate": float(values.mean()),
        "bootstrap_mean": float(bootstrap.mean()),
        "bootstrap_std": float(bootstrap.std(ddof=1)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }, bootstrap


def enabled():
    return os.environ.get("STAGE3_ENDPOINT_CI") == "1"


def sequence_losses(logits, targets):
    losses = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none")
    return losses.view_as(targets).mean(dim=1).detach().cpu().tolist()


def interval(prefix, values, binary=False):
    values = np.asarray(values, dtype=np.float64)
    summary, _ = summarize(values, 100_000, 20260828, binary=binary)
    return {f"endpoint-ci/{prefix}/{key}": summary[key]
            for key in ("n", "point_estimate", "ci95_low", "ci95_high")}
