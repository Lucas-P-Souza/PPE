"""
Aides de validation pour les matrices FEM (M, C, K).

Extrait de fem/solver.py pour isoler les responsabilités.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

try:
    from . import debug as dbg  # type: ignore
except Exception:
    try:
        from digital_twin.back_end.utils import debug as dbg  # type: ignore
    except Exception:
        class _DbgNoOp:
            @staticmethod
            def is_enabled() -> bool: return False
            @staticmethod
            def dprint(*args, **kwargs): pass
        dbg = _DbgNoOp()  # type: ignore


def _is_square_same_shape(M: np.ndarray, C: np.ndarray, K: np.ndarray) -> Tuple[bool, str, int]:
    for name, A in (("M", M), ("C", C), ("K", K)):
        if not isinstance(A, np.ndarray):
            return False, f"{name} n'est pas un numpy.ndarray", 0
        if A.ndim != 2:
            return False, f"{name} n'est pas 2D (ndim={A.ndim})", 0
        if A.shape[0] != A.shape[1]:
            return False, f"{name} n'est pas carrée (shape={A.shape})", 0
    if not (M.shape == C.shape == K.shape):
        return False, f"M, C, K ont des formats différents: M={M.shape}, C={C.shape}, K={K.shape}", 0
    return True, "OK", M.shape[0]


def _bc_applied_at_ends(A: np.ndarray, atol: float = 1e-12, *, allow_diag_zero: bool = False) -> Tuple[bool, str]:
    n = A.shape[0]
    for i in (0, n - 1):
        row = A[i, :].copy(); row[i] = 0.0
        col = A[:, i].copy(); col[i] = 0.0
        if not (np.all(np.abs(row) <= atol) and np.all(np.abs(col) <= atol)):
            max_off = max(np.max(np.abs(row)), np.max(np.abs(col)))
            return False, f"Extrémité {i}: lignes/colonnes non nulles hors diagonale (max hors diag={max_off:.2e})"
        if allow_diag_zero:
            if not (np.isclose(A[i, i], 0.0, atol=1e-9) or np.isclose(A[i, i], 1.0, atol=1e-9)):
                return False, f"Extrémité {i}: diagonale ni 0 ni 1 (A[{i},{i}]={A[i,i]:.3e})"
        else:
            if not np.isclose(A[i, i], 1.0, atol=1e-9):
                return False, f"Extrémité {i}: diagonale non ≈ 1.0 (A[{i},{i}]={A[i,i]:.3e})"
    return True, "OK"


def _is_symmetric(A: np.ndarray, rtol: float = 1e-8, atol: float = 1e-10) -> bool:
    return np.allclose(A, A.T, rtol=rtol, atol=atol)


def valider_mck(M: np.ndarray, C: np.ndarray, K: np.ndarray, *, verbose: bool = True) -> None:
    """Valide basiquement M, C, K: dimensions, symétrie et CL fixes aux extrémités.

    - Vérifie que M, C, K sont carrées et de même taille.
    - Informe (debug) la symétrie NumPy allclose.
    - Contrôle des conditions de bord: lignes/colonnes 0 et N-1 nulles (hors diag),
      diagonale ≈ 1.0 pour M/K; 0.0 ou 1.0 accepté pour C.
    """
    ok, msg, n = _is_square_same_shape(M, C, K)
    if not ok:
        raise ValueError(f"[ERREUR] Échec de la vérification des dimensions: {msg}")

    if verbose and dbg.is_enabled():
        dbg.dprint(f"Taille des matrices: {M.shape} (carrées)")
        dbg.dprint(f"Nombre de degrés de liberté (n): {n}")

    if verbose and dbg.is_enabled():
        dbg.dprint(f"Symétrie: M={_is_symmetric(M)}, C={_is_symmetric(C)}, K={_is_symmetric(K)}")

    for name, A in (("M", M), ("C", C), ("K", K)):
        allow_zero = True if name == "C" else False
        ok_bc, msg_bc = _bc_applied_at_ends(A, allow_diag_zero=allow_zero)
        if verbose and dbg.is_enabled():
            status = "OK" if ok_bc else "NOK"
            dbg.dprint(f"CL aux extrémités dans {name}: {status} — {msg_bc}")
        if not ok_bc:
            raise ValueError(f"[ERREUR] Les conditions aux limites ne semblent pas correctes dans {name}: {msg_bc}")


# Alias rétro-compatible (PT)
validar_mck = valider_mck
