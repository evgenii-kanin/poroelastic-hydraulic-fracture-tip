from functools import lru_cache
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=None)
def load_integrals():
    """
    Load precomputed integral matrices.

    The matrices correspond to the discrete collocation representation of the
    S_e, P_e, and P_s operators used in the paper. The precomputed matrix for
    the Cauchy operator is also included. The files are loaded only once per
    Python session because this function is cached.

    The S_s operator is not stored separately because, in this formulation,
    the same discrete coupling matrix is used for S_s and P_e.
    """

    folder = REPO_ROOT / "data" / "precomputed_integrals"

    return {
        "s_e_matrix_0": np.load(folder / "s_e_matrix_0_beta_0_25.npy"),
        "s_e_matrix_inf": np.load(folder / "s_e_matrix_inf_beta_0_25.npy"),
        "s_e_near_field_integral": np.load(folder / "s_e_near_field_integral_beta_0_25.npy"),
        "cauchy_matrix_0": np.load(folder / "cauchy_matrix_0.npy"),
        "cauchy_matrix_inf": np.load(folder / "cauchy_matrix_inf.npy"),
        "cauchy_near_field_integral": np.load(folder / "cauchy_near_field_integral.npy"),
        "p_e_matrix_0": np.load(folder / "p_e_matrix_0.npy"),
        "p_e_matrix_inf": np.load(folder / "p_e_matrix_inf.npy"),
        "p_e_near_field_integral": np.load(folder / "p_e_near_field_integral.npy"),
        "p_s_matrix_0": np.load(folder / "p_s_matrix_0.npy"),
        "p_s_matrix_inf": np.load(folder / "p_s_matrix_inf.npy"),
        "p_s_near_field_integral": np.load(folder / "p_s_near_field_integral.npy")
    }