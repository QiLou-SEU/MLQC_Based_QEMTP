import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils_quantum.PauliDecompsition import PauliDecompsiotion


DEFAULT_BASE_SEED = 20260606
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("random_symmetric_decomposition_benchmark2.csv")
EXPERIMENT_CONFIG = {
    "standard_dense": {"method": "standard", "backend": "dense"},
    "standard_sparse": {"method": "standard", "backend": "sparse"},
    "standard_diag_only": {"method": "standard", "backend": "diag_only"},
    "mlqc": {"method": "mlqc", "backend": "sparse"},
}
COMMON_METRIC_FIELDS = [
    "avg_compute_time_sec",
    "avg_term_count",
    "avg_rel_fro_error",
    "avg_sparse_conversion_time_sec",
    "avg_core_compute_time_sec",
]
MLQC_EXTRA_FIELDS = [
    "avg_perm_time_sec",
    "avg_svd_time_sec",
    "avg_subblock_expand_time_sec",
    "avg_combine_time_sec",
    "avg_effective_rank",
]


def build_random_symmetric_matrix(qubit_num, trial_index, base_seed=DEFAULT_BASE_SEED):
    dim = 1 << qubit_num
    seed = base_seed + qubit_num * 1000 + trial_index
    rng = np.random.default_rng(seed)
    random_matrix = rng.standard_normal((dim, dim))
    return (random_matrix + random_matrix.T) / 2.0


def relative_frobenius_error(reference_matrix, reconstructed_matrix):
    numerator = np.linalg.norm(reference_matrix - reconstructed_matrix, ord="fro")
    denominator = np.linalg.norm(reference_matrix, ord="fro")
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return float(numerator / denominator)


def measure_decomposition(matrix, method, backend, reference_matrix=None):
    decomposer = PauliDecompsiotion(matrix)
    terms = decomposer.pauli_decomposition(method=method, backend=backend)
    profile_stats = decomposer.get_profile_stats()

    reconstructed_matrix = decomposer.redecomposition(terms)
    reference = matrix if reference_matrix is None else reference_matrix
    rel_fro_error = relative_frobenius_error(reference, reconstructed_matrix)

    metrics = {
        "avg_compute_time_sec": float(profile_stats.get("compute_time_sec", 0.0)),
        "avg_term_count": float(len(terms)),
        "avg_rel_fro_error": rel_fro_error,
        "avg_sparse_conversion_time_sec": float(profile_stats.get("standard_sparse_conversion_time_sec", 0.0)),
        "avg_core_compute_time_sec": float(profile_stats.get("standard_sparse_core_compute_time_sec", 0.0)),
    }
    if method == "mlqc":
        metrics.update(
            {
                "avg_perm_time_sec": float(profile_stats.get("mlqc_perm_time_sec", 0.0)),
                "avg_svd_time_sec": float(profile_stats.get("mlqc_svd_time_sec", 0.0)),
                "avg_subblock_expand_time_sec": float(profile_stats.get("mlqc_subblock_expand_time_sec", 0.0)),
                "avg_combine_time_sec": float(profile_stats.get("mlqc_combine_time_sec", 0.0)),
                "avg_effective_rank": float(profile_stats.get("mlqc_effective_rank", 0.0)),
            }
        )
    return metrics


def benchmark(qubit_min, qubit_max, trials, base_seed, experiments):
    rows = []
    for qubit_num in range(qubit_min, qubit_max + 1):
        dim = 1 << qubit_num
        print(f"[random] qubit_num={qubit_num}, matrix_dim={dim}, trials={trials}")
        metrics_by_label = {label: [] for label in experiments}

        for trial in range(1, trials + 1):
            matrix = build_random_symmetric_matrix(qubit_num, trial, base_seed=base_seed)
            for label in experiments:
                config_item = EXPERIMENT_CONFIG[label]
                metrics_by_label[label].append(
                    measure_decomposition(
                        matrix,
                        method=config_item["method"],
                        backend=config_item["backend"],
                    )
                )

        row = {
            "benchmark_type": "random_symmetric",
            "matrix_dim": dim,
            "qubit_num": qubit_num,
            "trials": trials,
        }
        summary_parts = []
        for label in experiments:
            metric_list = metrics_by_label[label]
            row[f"{label}_avg_compute_time_sec"] = sum(item["avg_compute_time_sec"] for item in metric_list) / trials
            row[f"{label}_avg_term_count"] = sum(item["avg_term_count"] for item in metric_list) / trials
            row[f"{label}_avg_rel_fro_error"] = sum(item["avg_rel_fro_error"] for item in metric_list) / trials
            row[f"{label}_avg_sparse_conversion_time_sec"] = sum(item["avg_sparse_conversion_time_sec"] for item in metric_list) / trials
            row[f"{label}_avg_core_compute_time_sec"] = sum(item["avg_core_compute_time_sec"] for item in metric_list) / trials
            if label == "mlqc":
                for key in MLQC_EXTRA_FIELDS:
                    row[f"{label}_{key}"] = sum(item[key] for item in metric_list) / trials
                summary_parts.append(
                    f"{label}: compute={row[f'{label}_avg_compute_time_sec']:.6f}s, "
                    f"terms={row[f'{label}_avg_term_count']:.0f}, "
                    f"err={row[f'{label}_avg_rel_fro_error']:.6e}, "
                    f"expand={row[f'{label}_avg_subblock_expand_time_sec']:.6f}s, "
                    f"combine={row[f'{label}_avg_combine_time_sec']:.6f}s"
                )
            else:
                summary = (
                    f"{label}: compute={row[f'{label}_avg_compute_time_sec']:.6f}s, "
                    f"terms={row[f'{label}_avg_term_count']:.0f}, "
                    f"err={row[f'{label}_avg_rel_fro_error']:.6e}"
                )
                if label == "standard_sparse":
                    summary += (
                        f", sparse_conv={row[f'{label}_avg_sparse_conversion_time_sec']:.6f}s, "
                        f"core={row[f'{label}_avg_core_compute_time_sec']:.6f}s"
                    )
                summary_parts.append(summary)
        print("[random] " + " | ".join(summary_parts))
        rows.append(row)
    return rows


def build_fieldnames(experiments):
    fieldnames = ["benchmark_type", "matrix_dim", "qubit_num", "trials"]
    for label in experiments:
        fieldnames.extend([f"{label}_{name}" for name in COMMON_METRIC_FIELDS])
        if label == "mlqc":
            fieldnames.extend([f"{label}_{name}" for name in MLQC_EXTRA_FIELDS])
    return fieldnames


def write_rows(rows, output_path, experiments):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = build_fieldnames(experiments)
    try:
        import pandas as pd

        pd.DataFrame(rows, columns=fieldnames).to_csv(output_path, index=False)
    except Exception:
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def parse_experiments(experiments_text):
    labels = [item.strip() for item in experiments_text.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one experiment must be specified.")
    invalid = [label for label in labels if label not in EXPERIMENT_CONFIG]
    if invalid:
        raise ValueError(f"Unknown experiments: {invalid}")
    seen = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    return seen


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Pauli decomposition methods on random symmetric matrices.")
    parser.add_argument("--qubit-min", type=int, default=10)
    parser.add_argument("--qubit-max", type=int, default=10)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiments", type=str, default="standard_diag_only,mlqc")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.qubit_min < 1:
        raise ValueError("qubit_min must be >= 1.")
    if args.qubit_max < args.qubit_min:
        raise ValueError("qubit_max must be >= qubit_min.")
    if args.trials < 1:
        raise ValueError("trials must be >= 1.")

    experiments = parse_experiments(args.experiments)
    rows = benchmark(
        qubit_min=args.qubit_min,
        qubit_max=args.qubit_max,
        trials=args.trials,
        base_seed=args.base_seed,
        experiments=experiments,
    )
    write_rows(rows, args.output, experiments)
    print(f"[random] wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
