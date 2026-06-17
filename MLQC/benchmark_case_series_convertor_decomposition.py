import argparse
import csv
import importlib
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.config as config
import utils.dataloader as dl
import utils.emtcls as emtcls
from utils_quantum.PauliDecompsition import PauliDecompsiotion


DEFAULT_OUTPUT = Path(__file__).resolve().with_name("case_series_convertor_decomposition_benchmark2.csv")
EXPECTED_DIMENSION_FACTOR = 8
EXPECTED_DIMENSION_OFFSET = 3
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


def pad_matrix_to_power_of_two(matrix):
    original_dim = int(matrix.shape[0])
    padded_dim = 1 << int(math.ceil(math.log2(original_dim)))
    if padded_dim == original_dim:
        return np.asarray(matrix, dtype=np.complex128), original_dim, padded_dim
    padded_matrix = np.eye(padded_dim, dtype=np.complex128)
    padded_matrix[:original_dim, :original_dim] = np.asarray(matrix, dtype=np.complex128)
    return padded_matrix, original_dim, padded_dim


def expected_original_dim(n_value):
    return EXPECTED_DIMENSION_FACTOR * int(n_value) + EXPECTED_DIMENSION_OFFSET


def load_case_matrix(n_value):
    config.N = int(n_value)
    case_module = importlib.import_module("EMTcases.Case_Series_Convertor")
    case_module = importlib.reload(case_module)

    dl.elemet_list = case_module.elemet_list
    dl.vol_list = case_module.vol_list
    dl.vol_pu = case_module.vol_pu

    network = emtcls.Network()
    matrix = np.asarray(network.build_system_matrix(force=True), dtype=np.complex128)
    return {
        "requested_N": int(n_value),
        "loaded_config_N": int(case_module.N),
        "original_dim": int(matrix.shape[0]),
        "matrix": matrix,
    }


def verify_case_dimension(metadata):
    expected_dim = expected_original_dim(metadata["requested_N"])
    if metadata["loaded_config_N"] != metadata["requested_N"]:
        raise RuntimeError(
            f"N injection failed: requested {metadata['requested_N']}, "
            f"but Case_Series_Convertor loaded {metadata['loaded_config_N']}."
        )
    if metadata["original_dim"] != expected_dim:
        raise RuntimeError(
            f"Unexpected matrix dimension for N={metadata['requested_N']}: "
            f"got {metadata['original_dim']}, expected {expected_dim}."
        )


def precheck_n20():
    metadata = load_case_matrix(20)
    _, original_dim, padded_dim = pad_matrix_to_power_of_two(metadata["matrix"])
    verify_case_dimension(metadata)
    print(
        "[case] precheck passed:",
        {
            "requested_N": metadata["requested_N"],
            "loaded_config_N": metadata["loaded_config_N"],
            "original_dim": original_dim,
            "padded_dim": padded_dim,
        },
    )


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
    if reconstructed_matrix.shape != reference.shape:
        reconstructed_matrix = reconstructed_matrix[: reference.shape[0], : reference.shape[1]]
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


def benchmark_case_series(n_values, trials, experiments):
    rows = []
    observed_dims = {}
    for n_value in n_values:
        print(f"[case] N={n_value}, trials={trials}")
        metrics_by_label = {label: [] for label in experiments}
        metadata_first = None
        padded_dim_first = None
        qubit_num_first = None

        for _ in range(trials):
            metadata = load_case_matrix(n_value)
            verify_case_dimension(metadata)
            padded_matrix, original_dim, padded_dim = pad_matrix_to_power_of_two(metadata["matrix"])
            qubit_num = int(math.log2(padded_dim))

            previous_dim = observed_dims.get(n_value)
            if previous_dim is None:
                observed_dims[n_value] = original_dim
            elif previous_dim != original_dim:
                raise RuntimeError(
                    f"Inconsistent original_dim for N={n_value}: previous {previous_dim}, current {original_dim}."
                )

            if metadata_first is None:
                metadata_first = metadata
                padded_dim_first = padded_dim
                qubit_num_first = qubit_num

            for label in experiments:
                config_item = EXPERIMENT_CONFIG[label]
                metrics_by_label[label].append(
                    measure_decomposition(
                        padded_matrix,
                        method=config_item["method"],
                        backend=config_item["backend"],
                        reference_matrix=metadata["matrix"],
                    )
                )

        row = {
            "benchmark_type": "case_series_convertor",
            "case_name": "Case_Series_Convertor",
            "N": n_value,
            "original_dim": metadata_first["original_dim"],
            "padded_dim": padded_dim_first,
            "qubit_num": qubit_num_first,
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
        print("[case] " + " | ".join(summary_parts))
        rows.append(row)
    return rows


def build_fieldnames(experiments):
    fieldnames = ["benchmark_type", "case_name", "N", "original_dim", "padded_dim", "qubit_num", "trials"]
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
    parser = argparse.ArgumentParser(description="Benchmark Pauli decomposition methods on Case_Series_Convertor matrices.")
    parser.add_argument("--n-start", type=int, default=10)
    parser.add_argument("--n-stop", type=int, default=70)
    parser.add_argument("--n-step", type=int, default=10)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-precheck", action="store_true")
    parser.add_argument("--experiments", type=str, default="standard_sparse, mlqc")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_start <= 0:
        raise ValueError("n_start must be > 0.")
    if args.n_stop < args.n_start:
        raise ValueError("n_stop must be >= n_start.")
    if args.n_step <= 0:
        raise ValueError("n_step must be > 0.")
    if args.trials < 1:
        raise ValueError("trials must be >= 1.")

    experiments = parse_experiments(args.experiments)
    # if not args.skip_precheck:
    #     precheck_n20()

    n_values = list(range(args.n_start, args.n_stop + 1, args.n_step))
    rows = benchmark_case_series(n_values=n_values, trials=args.trials, experiments=experiments)
    write_rows(rows, args.output, experiments)
    print(f"[case] wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
