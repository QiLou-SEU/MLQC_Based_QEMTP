# Matrix Low-dimensional Qubit Casting Based Quantum Electromagnetic Transient Network Simulation Program

This repository accompanies the paper **"Matrix Low-dimensional Qubit Casting Based Quantum Electromagnetic Transient Network Simulation Program"**.

The work studies quantum electromagnetic transient simulation (QEMTP) for converter-dominated power systems and proposes a **matrix low-dimensional qubit casting (MLQC)** method to reduce the preprocessing burden of variational quantum linear solvers (VQLS). The public materials in this repository focus on the **core MLQC versus conventional Pauli decomposition comparison**, together with representative benchmark data used to illustrate the preprocessing improvement.

## Overview

In the paper, the main motivation is that classical electromagnetic transient programs become increasingly expensive as network size grows, while existing quantum approaches still suffer from heavy matrix preprocessing and circuit-construction overhead. The proposed MLQC method addresses this bottleneck by exploiting low-dimensional structure in the admittance matrix before Pauli expansion.

This public repository is intended to provide:

- the core Pauli/MLQC decomposition implementation used in the study;
- benchmark scripts for decomposition-time comparison;
- representative result data for random symmetric matrices and converter-related case matrices;
- sample simulation result files for figure reproduction and result inspection.

## Public Release Scope

At this stage, the repository primarily releases the **core decomposition comparison content** around MLQC and Pauli-based preprocessing.

The **full EMT simulation codebase**, including the complete end-to-end simulation workflow and some supporting modules used in the paper, is **currently under release coordination** and will be added in a future update.

**Coming soon.**

## Repository Structure

```text
.
|-- PauliDecompsition.py
|-- benchmarks/
|   |-- benchmark_random_symmetric_decomposition.py
|   `-- benchmark_case_series_convertor_decomposition.py
`-- data/
    |-- mlqc_decomposition/
    |   |-- random_symmetric_decomposition_benchmark.csv
    |   `-- case_series_convertor_decomposition_benchmark.csv
    `-- simulation/
        |-- Buck/
        `-- convertor/
```

## Included Materials

### 1. Core decomposition implementation

`PauliDecompsition.py` contains the main decomposition logic:

- standard Pauli decomposition backends;
- MLQC-based decomposition;
- reconstruction utilities for validation;
- profiling fields for timing breakdown analysis.

### 2. Benchmark scripts

`benchmarks/benchmark_random_symmetric_decomposition.py`

- benchmarks random symmetric matrices across different qubit numbers;
- compares compute time, term count, and reconstruction error;
- records MLQC timing breakdown such as permutation, SVD, subblock expansion, and combination time.

`benchmarks/benchmark_case_series_convertor_decomposition.py`

- benchmarks decomposition on converter-related case matrices;
- pads matrices to powers of two for quantum representation;
- compares standard sparse Pauli decomposition with MLQC.

### 3. Benchmark data

`data/mlqc_decomposition/random_symmetric_decomposition_benchmark.csv`

- benchmark results on random symmetric matrices;
- shows that MLQC becomes much faster as matrix dimension increases while preserving negligible reconstruction error.

`data/mlqc_decomposition/case_series_convertor_decomposition_benchmark.csv`

- benchmark results on the series-converter case matrices;
- demonstrates a substantial reduction in preprocessing time for practical EMT-related matrices.

### 4. Simulation result files

The `data/simulation/` directory contains representative voltage/current result files and plotting resources corresponding to the paper figures. These files are provided for result inspection and figure-level comparison.

## Main Takeaway from the Released Results

The released benchmark tables show the central message of the paper:

- MLQC preserves reconstruction accuracy at near machine precision;
- MLQC significantly reduces decomposition/preprocessing time compared with direct Pauli expansion;
- the advantage becomes more evident as the matrix dimension grows;
- the benefit also appears on converter-related EMT case matrices, not only on random test matrices.

For example, in the released CSV results:

- on random symmetric matrices with dimension `1024`, the recorded average decomposition time is about `15552.56 s` for the standard diagonal-only Pauli route and about `109.72 s` for MLQC;
- on the released converter case with `N = 70`, the recorded average decomposition time is about `1491.39 s` for standard sparse Pauli decomposition and about `0.75 s` for MLQC.

## Citation

If you use this repository, please cite the corresponding paper.

```bibtex
@article{lou_mlqc_qemtp,
  title={Matrix Low-dimensional Qubit Casting Based Quantum Electromagnetic Transient Network Simulation Program},
  author={Lou, Qi and Xu, Yijun and Gu, Wei},
  journal={Under review / to appear},
  year={2026}
}
```

Please replace the citation entry with the final published bibliographic information when available.

## Notes

- This repository currently emphasizes the **MLQC vs. Pauli preprocessing comparison**, which is the core publicly released component at this stage.
- A more complete release of the QEMTP simulation pipeline is being prepared and will be updated here when ready.

