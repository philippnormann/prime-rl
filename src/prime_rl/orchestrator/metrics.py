import pandas as pd


def get_scoped_rollout_stats(
    by_example_mean: pd.DataFrame,
    scope: str,
    *,
    stats_with_min: tuple[str, ...],
    stats_without_min: tuple[str, ...],
) -> dict[str, float]:
    to_log = {}
    for col in stats_with_min:
        to_log[f"{col}/{scope}/mean"] = by_example_mean[col].mean()
        to_log[f"{col}/{scope}/max"] = by_example_mean[col].max()
        to_log[f"{col}/{scope}/min"] = by_example_mean[col].min()
    for col in stats_without_min:
        to_log[f"{col}/{scope}/mean"] = by_example_mean[col].mean()
        to_log[f"{col}/{scope}/max"] = by_example_mean[col].max()
    return to_log


def get_prefixed_reward_stats(reward_by_example_mean: pd.Series, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}/mean": reward_by_example_mean.mean(),
        f"{prefix}/max": reward_by_example_mean.max(),
        f"{prefix}/min": reward_by_example_mean.min(),
    }


def compute_global_rollout_metrics(metrics_by_example_mean: pd.DataFrame) -> dict[str, float]:
    """Return metrics/all/* only for metrics that are defined on every example in the batch."""
    to_log = {}
    for metric in metrics_by_example_mean.columns:
        metric_values = metrics_by_example_mean[metric]
        if not metric_values.notna().all():
            continue
        to_log[f"metrics/all/{metric}"] = metric_values.mean()
    return to_log
