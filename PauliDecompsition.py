import time

import numpy as np
from scipy import sparse

from utils_quantum.pyqpanda3_compat import QCircuit, Y, apply_circuit_control


class PauliDecompsiotion:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.complex128)
        self.size = int(self.matrix.shape[0])
        self.qubits_num = int(np.log2(self.size))
        self.pauli_dict = {
            0: np.array([[1, 0], [0, 1]], dtype=np.complex128),
            1: np.array([[0, 1], [1, 0]], dtype=np.complex128),
            2: np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            3: np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }
        self.pauli_sparse_dict = {
            key: sparse.csr_matrix(value)
            for key, value in self.pauli_dict.items()
        }
        self._mlqc_dense_pauli_cache_by_n = {}
        self._profile_stats = {}

    def pauli_basis(self, index):
        if len(index) == 0:
            return np.array([[1]], dtype=np.complex128)
        mat = self.pauli_dict[index[0]]
        for gate_index in index[1:]:
            mat = np.kron(mat, self.pauli_dict[gate_index])
        return mat

    def _pauli_basis_sparse(self, index):
        if len(index) == 0:
            return sparse.csr_matrix([[1]], dtype=np.complex128)
        mat = self.pauli_sparse_dict[index[0]]
        for gate_index in index[1:]:
            mat = sparse.kron(mat, self.pauli_sparse_dict[gate_index], format="csr")
        return mat

    def _pauli_basis_dense_cached(self, index, n=None):
        if n is None:
            n = len(index)
        n = int(n)
        idx_tuple = tuple(index)
        cache_n = self._mlqc_dense_pauli_cache_by_n.setdefault(n, {})
        cached = cache_n.get(idx_tuple)
        if cached is not None:
            return cached

        if n == 0:
            mat = np.array([[1]], dtype=np.complex128)
        elif n == 1:
            mat = self.pauli_dict[idx_tuple[0]]
        else:
            mat = np.kron(
                self.pauli_dict[idx_tuple[0]],
                self._pauli_basis_dense_cached(idx_tuple[1:], n=n - 1),
            )
        cache_n[idx_tuple] = mat
        return mat

    def _generate_indices(self, n):
        for value in range(4 ** n):
            idx = [0] * n
            tmp = value
            for pos in range(n - 1, -1, -1):
                idx[pos] = tmp % 4
                tmp //= 4
            yield idx

    def _reset_profile_stats(self):
        self._profile_stats = {
            "compute_time_sec": 0.0,
            "standard_sparse_conversion_time_sec": 0.0,
            "standard_sparse_core_compute_time_sec": 0.0,
            "mlqc_perm_time_sec": 0.0,
            "mlqc_svd_time_sec": 0.0,
            "mlqc_subblock_expand_time_sec": 0.0,
            "mlqc_combine_time_sec": 0.0,
            "mlqc_effective_rank": 0.0,
        }

    def get_profile_stats(self):
        return dict(self._profile_stats)

    @staticmethod
    def _trace_from_dense_rows_and_columns(matrix_dense, pauli_dense):
        trace_sum = 0.0 + 0.0j
        dim = matrix_dense.shape[0]
        for diag_index in range(dim):
            trace_sum += matrix_dense[diag_index, :] @ pauli_dense[:, diag_index]
        return trace_sum

    def _pauli_expand_dense(self, matrix, n, even_only=False):
        result = []
        matrix_dense = np.asarray(matrix, dtype=np.complex128)
        norm_factor = 2 ** n

        for idx in self._generate_indices(n):
            if even_only and np.count_nonzero(np.array(idx) == 2) % 2 != 0:
                continue
            pauli_dense = self.pauli_basis(idx)
            coeff = np.trace(matrix_dense @ pauli_dense) / norm_factor
            if abs(coeff) > 1e-12:
                result.append([idx, coeff])
        return result

    def _pauli_expand_sparse(self, matrix, n, even_only=False):
        result = []
        convert_start = time.perf_counter()
        matrix_sparse = sparse.csr_matrix(np.asarray(matrix, dtype=np.complex128))
        self._profile_stats["standard_sparse_conversion_time_sec"] += time.perf_counter() - convert_start
        norm_factor = 2 ** n

        for idx in self._generate_indices(n):
            if even_only and np.count_nonzero(np.array(idx) == 2) % 2 != 0:
                continue
            convert_start = time.perf_counter()
            pauli_sparse = self._pauli_basis_sparse(idx)
            self._profile_stats["standard_sparse_conversion_time_sec"] += time.perf_counter() - convert_start
            compute_start = time.perf_counter()
            coeff = (matrix_sparse @ pauli_sparse).diagonal().sum() / norm_factor
            self._profile_stats["standard_sparse_core_compute_time_sec"] += time.perf_counter() - compute_start
            if abs(coeff) > 1e-12:
                result.append([idx, coeff])
        return result

    def _pauli_expand_diag_only(self, matrix, n, even_only=False):
        result = []
        matrix_dense = np.asarray(matrix, dtype=np.complex128)
        norm_factor = 2 ** n

        for idx in self._generate_indices(n):
            if even_only and np.count_nonzero(np.array(idx) == 2) % 2 != 0:
                continue
            pauli_dense = self.pauli_basis(idx)
            coeff = self._trace_from_dense_rows_and_columns(matrix_dense, pauli_dense) / norm_factor
            if abs(coeff) > 1e-12:
                result.append([idx, coeff])
        return result

    def _pauli_expand_small_dense(self, matrix, n, even_only=False):
        result = []
        matrix_dense = np.asarray(matrix, dtype=np.complex128)
        norm_factor = 2 ** n

        for idx in self._generate_indices(n):
            if even_only and np.count_nonzero(np.array(idx) == 2) % 2 != 0:
                continue
            pauli_dense = self._pauli_basis_dense_cached(idx, n=n)
            compute_start = time.perf_counter()
            coeff = np.trace(matrix_dense @ pauli_dense) / norm_factor
            elapsed = time.perf_counter() - compute_start
            self._profile_stats["mlqc_subblock_expand_time_sec"] += elapsed
            if abs(coeff) > 1e-12:
                result.append((tuple(idx), coeff))
        return result

    def _standard_decomp(self, backend="sparse"):
        if backend == "dense":
            return self._pauli_expand_dense(self.matrix, self.qubits_num)
        if backend == "sparse":
            return self._pauli_expand_sparse(self.matrix, self.qubits_num)
        if backend == "diag_only":
            return self._pauli_expand_diag_only(self.matrix, self.qubits_num)
        raise ValueError(f"Unknown standard backend: {backend}")

    def _mlqc_decomp(self):
        def permute(mat, a_dim, b_dim):
            ans = np.zeros((a_dim * a_dim, b_dim * b_dim), dtype=np.complex128)
            for col in range(a_dim):
                for row in range(a_dim):
                    ans[a_dim * col + row, :] = mat[
                        row * b_dim:(row + 1) * b_dim,
                        col * b_dim:(col + 1) * b_dim,
                    ].reshape(b_dim * b_dim, order="F")
            return ans

        n = self.qubits_num
        dim = 2 ** n
        b_dim = int(2 ** (n // 2))
        a_dim = dim // b_dim

        perm_start = time.perf_counter()
        mat_tilde = permute(self.matrix, a_dim, b_dim)
        self._profile_stats["mlqc_perm_time_sec"] += time.perf_counter() - perm_start

        svd_start = time.perf_counter()
        u, s, v = np.linalg.svd(mat_tilde, full_matrices=False)
        self._profile_stats["mlqc_svd_time_sec"] += time.perf_counter() - svd_start

        coeff_dict = {}
        effective_rank = 0
        for rank_index, singular_value in enumerate(s):
            if singular_value < 1e-10:
                continue

            effective_rank += 1
            a_hat = np.sqrt(singular_value) * u[:, rank_index].reshape(a_dim, a_dim, order="F")
            b_hat = np.sqrt(singular_value) * v[rank_index, :].reshape(b_dim, b_dim, order="F")
            q_a = int(np.log2(a_dim))
            q_b = int(np.log2(b_dim))
            even_flag = np.allclose(a_hat, a_hat.T)

            set_a = self._pauli_expand_small_dense(a_hat, q_a, even_only=even_flag)
            set_b = self._pauli_expand_small_dense(b_hat, q_b, even_only=even_flag)

            combine_start = time.perf_counter()
            for left_idx, left_coeff in set_a:
                for right_idx, right_coeff in set_b:
                    idx = left_idx + right_idx
                    coeff_dict[idx] = coeff_dict.get(idx, 0.0 + 0.0j) + left_coeff * right_coeff
            self._profile_stats["mlqc_combine_time_sec"] += time.perf_counter() - combine_start

        self._profile_stats["mlqc_effective_rank"] = float(effective_rank)
        return [[list(index), coeff] for index, coeff in coeff_dict.items() if abs(coeff) > 1e-12]

    def pauli_decomposition(self, method="standard", backend="sparse"):
        self._reset_profile_stats()
        total_start = time.perf_counter()
        if method == "standard":
            result = self._standard_decomp(backend=backend)
        elif method == "mlqc":
            result = self._mlqc_decomp()
        else:
            raise ValueError(f"Unknown Pauli decomposition method: {method}")
        self._profile_stats["compute_time_sec"] = time.perf_counter() - total_start
        return result

    def redecomposition(self, decomposition_set):
        mat = np.zeros_like(self.matrix, dtype=np.complex128)
        for index, coeff in decomposition_set:
            mat += coeff * self.pauli_basis(index)
        if np.max(np.abs(np.imag(mat))) < 1e-10:
            mat = np.real(mat)
        return mat

    @staticmethod
    def CY(cqubit, qubit, dag=0):
        circ = QCircuit()
        circ << Y(qubit)
        controlled_circ = apply_circuit_control(circ, cqubit)
        return controlled_circ if dag == 0 else controlled_circ.dagger()
