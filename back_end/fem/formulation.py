"""Formulation FEM 1D (corde tendue) — prise en charge maillage uniforme ou non uniforme.

Objectifs
---------
Fournir des routines d'assemblage des matrices globales :
 - Masse   (M)
 - Raideur (K)
 - Amortissement de Rayleigh (C = α M + β K)

Points clés
-----------
1. Deux cas de discrétisation traités par une API unifiée :
    - Maillage uniforme : nombre de nœuds + longueur totale -> dx constant.
    - Maillage non uniforme : vecteur explicite des longueurs élémentaires dx_i.
2. Matrices locales (élément e de longueur dx_e) :
        M_e = (μ * dx_e / 6) [[2, 1], [1, 2]]
        K_e = (T / dx_e)     [[1, -1], [-1, 1]]
   (On NE construit plus C par somme des C_e — la matrice d'amortissement globale
    est formée APRÈS assemblage via C = α M + β K.)
3. Assemblage additif standard sur les sous-blocs (i,i), (i,i+1), (i+1,i), (i+1,i+1).
4. Conditions aux limites *fixes* optionnelles (nœuds 0 et N-1 bloqués) appliquées
   directement à M et K (puis C est calculé ensuite).
5. Calcul global de l'amortissement : C = α M + β K après (éventuelle) application des CL.
6. Retour possible d'un dictionnaire de métadonnées (dx_min, dx_max, etc.).

Conventions
-----------
 - n_nodes = nombre total de nœuds.
 - n_elems = n_nodes - 1.
 - Indices d'éléments: e relie les nœuds e et e+1.

Sécurité & Validation
---------------------
 - Tous les dx doivent être > 0.
 - n_nodes >= 2.

API publique
------------
 - assemble_system_matrices_nonuniform(dx_vector, ...)
 - assemble_system_matrices(n_nodes, length, ...)
 - assemble_mkc(uniform_length=None, n_nodes=None, dx_vector=None, ...)
      -> Fait la déduction du mode automatiquement.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# __all__ : définit l'API publique du module (symbols exportés par `from .. import *`).
# On le place ici (juste après les imports) pour qu'il soit visible immédiatement
# et éviter de le "perdre" en bas parmi les fonctions.
# ---------------------------------------------------------------------------
__all__ = [
    "assemble_system_matrices",
    "assemble_system_matrices_nonuniform",
    "assemble_mkc",
    "compute_local_element_matrices",
    "print_local_element_matrices",
    "build_global_mkc_from_config",
    "assemble_mass",
    "assemble_stiffness",
    "rayleigh_damping",
]

# ---------------------------------------------------------------------------
# Debug controls (can be overridden in digital_twin.back_end.config)
# ---------------------------------------------------------------------------
try:
    from .. import config as _cfg  # type: ignore
    _DEBUG_RAYLEIGH: bool = bool(getattr(_cfg, "DEBUG_RAYLEIGH", True))
    _DEBUG_RAYLEIGH_LEVEL: int = int(getattr(_cfg, "DEBUG_RAYLEIGH_LEVEL", 1))
except Exception:
    # Enabled by default per user request; set to False to silence
    _DEBUG_RAYLEIGH = True
    _DEBUG_RAYLEIGH_LEVEL = 1


def _matrix_stats(name: str, M: np.ndarray) -> None:
    try:
        nz = np.count_nonzero(np.abs(M) > 0)
        tot = M.size
        frac = (nz / tot * 100.0) if tot else 0.0
        print(f"[DEBUG] {name}: shape={M.shape}, nnz={nz} ({frac:.1f}%), fro={np.linalg.norm(M, 'fro'):.3e}, min={M.min():.3e}, max={M.max():.3e}")
    except Exception as e:
        print(f"[DEBUG] {name}: stats error: {e}")


def _print_blocks(name: str, M: np.ndarray, k: int = 4) -> None:
    try:
        k = int(max(1, k))
        tl = M[:k, :k]
        br = M[-k:, -k:]
        with np.printoptions(precision=3, suppress=True):
            print(f"[DEBUG] {name} top-left {k}x{k}:\n{tl}")
            print(f"[DEBUG] {name} bottom-right {k}x{k}:\n{br}")
    except Exception as e:
        print(f"[DEBUG] {name}: block print error: {e}")


def _print_bc_edges(name: str, M: np.ndarray) -> None:
    try:
        r0 = float(np.sum(np.abs(M[0, :])))
        rn = float(np.sum(np.abs(M[-1, :])))
        c0 = float(np.sum(np.abs(M[:, 0])))
        cn = float(np.sum(np.abs(M[:, -1])))
        print(f"[DEBUG] {name} edge sums: row0={r0:.3e}, rowN={rn:.3e}, col0={c0:.3e}, colN={cn:.3e}")
    except Exception as e:
        print(f"[DEBUG] {name}: edge sums error: {e}")

def _detect_constrained_dofs(M: np.ndarray, K: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """Détecte les DDL contraints (CL de type Dirichlet appliquées via diag=1, lignes/colonnes nulles).

    On considère qu'un DDL i est contraint si:
      - la ligne et la colonne i de M sont nulles en dehors de la diagonale
      - la ligne et la colonne i de K sont nulles en dehors de la diagonale
      - M[i,i] ≈ 1 et K[i,i] ≈ 1 (convention du module pour garder l'inversibilité)
    """
    n = M.shape[0]
    constrained = []
    for i in range(n):
        # Copie des lignes/colonnes sans la diagonale
        rowM = M[i, :].copy(); rowM[i] = 0.0
        colM = M[:, i].copy(); colM[i] = 0.0
        rowK = K[i, :].copy(); rowK[i] = 0.0
        colK = K[:, i].copy(); colK[i] = 0.0
        if (
            np.all(np.abs(rowM) <= atol) and np.all(np.abs(colM) <= atol)
            and np.all(np.abs(rowK) <= atol) and np.all(np.abs(colK) <= atol)
            and abs(M[i, i] - 1.0) <= 1e-9 and abs(K[i, i] - 1.0) <= 1e-9
        ):
            constrained.append(i)
    return np.asarray(constrained, dtype=int)

def _apply_fixed_bc_inplace(mat: np.ndarray) -> None:
        """
        Applique des conditions aux limites fixes (Dirichlet) sur place.

        Stratégie :
            - Annule les lignes/colonnes des nœuds extrêmes.
            - Met la diagonale à 1 sur ces nœuds pour rendre la matrice inversible
                (utile pour solveurs linéaires simples / élimination).
        """
        mat[0, :] = 0.0
        mat[-1, :] = 0.0
        mat[:, 0] = 0.0
        mat[:, -1] = 0.0
        mat[0, 0] = 1.0
        mat[-1, -1] = 1.0

def _apply_fixed_bc_inplace_damping(mat: np.ndarray) -> None:
    """
    Applique des conditions aux limites sur la matrice d'amortissement C.

    Différence clé par rapport à _apply_fixed_bc_inplace:
    - On ANNULE les lignes/colonnes des nœuds extrêmes, mais on NE met PAS
      la diagonale à 1. Pour C, il n'est pas nécessaire (ni souhaitable)
      d'introduire des termes artificiels sur les DDL contraints.
    """
    mat[0, :] = 0.0
    mat[-1, :] = 0.0
    mat[:, 0] = 0.0
    mat[:, -1] = 0.0

def rayleigh_damping(
    M: np.ndarray,
    K: np.ndarray,
    modes_ref: tuple[int, int],
    zetas_ref: tuple[float, float],
):
    """Calcule α et β de Rayleigh à partir de deux amortissements modaux et construit C = α M + β K.

    Hypothèse standard: ζ(ω) = 1/2 (α/ω + β ω). On résout sur deux modes de référence p et q.

    Paramètres
    ----------
    M, K : np.ndarray
        Matrices de masse et raideur (n x n), symétriques définies positives (avec CL appliquées si nécessaire).
    modes_ref : tuple[int, int]
        Indices des deux modes utilisés comme référence (0-based), ex: (0, 1).
    zetas_ref : tuple[float, float]
        Amortissements désirés pour ces deux modes, ex: (0.01, 0.02).

    Retour
    ------
    alpha : float, beta : float, C : np.ndarray, omegas : np.ndarray
    """
    # Si des CL fixes ont été appliquées via diag=1 et lignes/colonnes nulles,
    # retirer ces DDL pour le calcul propre des modes physiques.
    constrained = _detect_constrained_dofs(M, K)
    if constrained.size > 0:
        free = np.setdiff1d(np.arange(M.shape[0]), constrained)
        M_eval = M[np.ix_(free, free)]
        K_eval = K[np.ix_(free, free)]
    else:
        M_eval, K_eval = M, K

    # Évite l'inversion explicite: résoudre M X = K -> X = M^{-1} K
    A = np.linalg.solve(M_eval, K_eval)
    eigvals, _ = np.linalg.eig(A)
    eigvals = np.real(eigvals)
    eigvals[eigvals < 0] = 0.0  # coupe les petites négativités numériques
    omegas = np.sqrt(eigvals)
    omegas.sort()

    p, q = modes_ref
    if p < 0 or q < 0 or p >= omegas.size or q >= omegas.size or p == q:
        raise ValueError("modes_ref invalides par rapport au nombre de modes disponibles")
    zeta_p, zeta_q = zetas_ref
    omega_p, omega_q = float(omegas[p]), float(omegas[q])

    # Solve Rayleigh parameters from the 2x2 linear system derived from
    #   2 ζ(ω) = α/ω + β ω  evaluated at ω_p and ω_q
    # System form:
    #   [ 1/ω_p   ω_p ] [α] = [ 2 ζ_p ]
    #   [ 1/ω_q   ω_q ] [β]   [ 2 ζ_q ]
    A_sys = np.array([[1.0/omega_p, omega_p], [1.0/omega_q, omega_q]], dtype=float)
    b_sys = np.array([2.0*zeta_p, 2.0*zeta_q], dtype=float)
    try:
        alpha_lin, beta_lin = np.linalg.solve(A_sys, b_sys)
    except np.linalg.LinAlgError:
        # Fallback to closed form (equivalent but numerically different path)
        denom = (omega_p**2 - omega_q**2)
        if abs(denom) < 1e-16:
            raise ValueError("Fréquences de référence très proches: impossible de déterminer α et β de façon stable")
        beta_lin = 2.0 * (zeta_p * omega_p - zeta_q * omega_q) / denom
        alpha_lin = 2.0 * zeta_p * omega_p - beta_lin * (omega_p**2)

    # Keep names alpha, beta for downstream
    alpha = float(alpha_lin)
    beta = float(beta_lin)
    C = alpha * M + beta * K
    # Debug prints with levels
    if _DEBUG_RAYLEIGH:
        try:
            two_pi = 2.0 * np.pi
            f_hz = omegas / two_pi
            p, q = modes_ref
            zp, zq = zetas_ref
            # Back-substitution check
            zeta_p_chk = 0.5 * (alpha/omega_p + beta*omega_p)
            zeta_q_chk = 0.5 * (alpha/omega_q + beta*omega_q)

            lvl = int(_DEBUG_RAYLEIGH_LEVEL)
            if lvl <= 1:
                # Concise 1-2 line summary
                print(
                    f"[RAYLEIGH] p={p}, q={q} | f_p={f_hz[p]:.2f} Hz, f_q={f_hz[q]:.2f} Hz | "
                    f"alpha={alpha:.3e} [1/s], beta={beta:.3e} [s]"
                )
                print(
                    f"[RAYLEIGH] ζp tgt/ok Δ: {zp:.4f}/{zeta_p_chk:.4f} Δ={abs(zeta_p_chk-zp):.1e} | "
                    f"ζq tgt/ok Δ: {zq:.4f}/{zeta_q_chk:.4f} Δ={abs(zeta_q_chk-zq):.1e}"
                )
            if lvl >= 2:
                print("[DEBUG] Modelo: ζ(ω) = 0.5 (α/ω + β ω)")
                print(f"[DEBUG] modes_ref=(p={p}, q={q}), ζ_targets=({zp:.5f}, {zq:.5f})")
                print(f"[DEBUG] ω_p={omegas[p]:.6e} rad/s, ω_q={omegas[q]:.6e} rad/s | f_p={f_hz[p]:.3f} Hz, f_q={f_hz[q]:.3f} Hz")
                lhs_p = f"2 ζ_p = α/ω_p + β ω_p  ->  2*{zp:.6f} = α/{omega_p:.6e} + β*{omega_p:.6e}"
                lhs_q = f"2 ζ_q = α/ω_q + β ω_q  ->  2*{zq:.6f} = α/{omega_q:.6e} + β*{omega_q:.6e}"
                print(f"[DEBUG] Eq(p): {lhs_p}")
                print(f"[DEBUG] Eq(q): {lhs_q}")
                with np.printoptions(precision=6, suppress=False):
                    print("[DEBUG] Sistema linear A·[α β]^T = b:")
                    print(A_sys)
                    print(b_sys)
                print(f"[DEBUG] Solução -> α = {alpha:.6e} [1/s], β = {beta:.6e} [s]")
                print(f"[DEBUG] Checagem ζ_p: alvo={zp:.6f} | obtido={zeta_p_chk:.6f} | Δ={abs(zeta_p_chk - zp):.2e}")
                print(f"[DEBUG] Checagem ζ_q: alvo={zq:.6f} | obtido={zeta_q_chk:.6f} | Δ={abs(zeta_q_chk - zq):.2e}")
            if lvl >= 3:
                _matrix_stats("M", M)
                _matrix_stats("K", K)
                _matrix_stats("C=αM+βK", C)
                try:
                    nC = float(np.linalg.norm(C, 'fro'))
                    nAM = float(np.linalg.norm(alpha * M, 'fro'))
                    nBK = float(np.linalg.norm(beta * K, 'fro'))
                    sAM = (nAM / nC * 100.0) if nC > 0 else 0.0
                    sBK = (nBK / nC * 100.0) if nC > 0 else 0.0
                    print(f"[DEBUG] ||αM||_F={nAM:.3e} ({sAM:.1f}% de ||C||), ||βK||_F={nBK:.3e} ({sBK:.1f}% de ||C||)")
                except Exception:
                    pass
            if lvl >= 4:
                _print_blocks("C", C, k=4)
        except Exception as _e_dbg:
            print("[DEBUG] Rayleigh print error:", _e_dbg)
    #C = alpha * M * 0 + beta * K * 0                               # Pour désactiver l'amortissement global (test)
    return float(alpha), float(beta), C, omegas


def assemble_system_matrices_nonuniform(
    dx_vector: Sequence[float] | np.ndarray,
    tension: float,
    lin_density: float,
    *,
    damping_modes_ref: tuple[int, int] | None = None,
    damping_zetas_ref: tuple[float, float] | None = None,
    apply_fixed_bc: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble M, K puis calcule C = α M + β K pour une maille 1D NON uniforme.

    Paramètres
    ----------
    dx_vector : Sequence[float] | np.ndarray
        Longueurs élémentaires (m), taille = n_elems.
    tension : float
        Tension T (N).
    lin_density : float
        Densité linéique μ (kg/m).
    rayleigh_alpha : float
        Coefficient α.
    rayleigh_beta : float
        Coefficient β.
    apply_fixed_bc : bool, défaut False
        Si True applique CL fixes (nœuds extrêmes bloqués) en modifiant les matrices.

    Retour
    ------
    (M, K, C) : tuple de matrices carrées (n_nodes, n_nodes)
    """
    dx_arr = np.asarray(dx_vector, dtype=float)
    if dx_arr.ndim != 1 or dx_arr.size == 0:
        raise ValueError("dx_vector doit être un vecteur 1D non vide")
    if np.any(dx_arr <= 0):
        raise ValueError("Toutes les longueurs d'élément doivent être positives")
    n_elems = dx_arr.size
    n_nodes = n_elems + 1

    # Allocation des matrices globales (C sera calculée à la fin)
    M = np.zeros((n_nodes, n_nodes), dtype=float)
    K = np.zeros((n_nodes, n_nodes), dtype=float)

    # Boucle d'assemblage élément par élément (on ne construit pas C_local globalement)
    for i, dx in enumerate(dx_arr):
        mass_pref = lin_density * dx / 6.0
        M_local = mass_pref * np.array([[2.0, 1.0], [1.0, 2.0]])
        K_local = (tension / dx) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        sl = slice(i, i + 2)
        M[sl, sl] += M_local
        K[sl, sl] += K_local
    if apply_fixed_bc:
        _apply_fixed_bc_inplace(M)
        _apply_fixed_bc_inplace(K)
    # Amortissement global via cibles modales s'il y en a, sinon nul
    if damping_modes_ref is not None and damping_zetas_ref is not None:
        _, _, C, _ = rayleigh_damping(M, K, damping_modes_ref, damping_zetas_ref)
    else:
        C = np.zeros_like(M)
    if apply_fixed_bc:
        _apply_fixed_bc_inplace_damping(C)
        if _DEBUG_RAYLEIGH and _DEBUG_RAYLEIGH_LEVEL >= 2:
            print("[DEBUG] Applied damping BC to C (zeroed edge rows/cols)")
            _print_bc_edges("C after BC", C)
    return M, K, C


def assemble_system_matrices(
    n_nodes: int,
    length: float,
    tension: float,
    lin_density: float,
    *,
    damping_modes_ref: tuple[int, int] | None = None,
    damping_zetas_ref: tuple[float, float] | None = None,
    apply_fixed_bc: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemblage de M, K puis C = α M + β K pour un maillage UNIFORME.

    Paramètres identiques à la version précédente, avec ajout de `apply_fixed_bc`.
    """
    if n_nodes < 2:
        raise ValueError("n_nodes doit être >= 2")
    if length <= 0:
        raise ValueError("length doit être positif")
    if tension <= 0:
        raise ValueError("tension doit être positive")
    if lin_density <= 0:
        raise ValueError("lin_density doit être positive")

    n_elems = n_nodes - 1
    dx = length / n_elems

    M = np.zeros((n_nodes, n_nodes), dtype=float)
    K = np.zeros((n_nodes, n_nodes), dtype=float)

    mass_prefactor = lin_density * dx / 6.0
    M_template = mass_prefactor * np.array([[2.0, 1.0], [1.0, 2.0]])
    stiff_prefactor = tension / dx
    K_template = stiff_prefactor * np.array([[1.0, -1.0], [-1.0, 1.0]])

    for i in range(n_elems):
        sl = slice(i, i + 2)
        M[sl, sl] += M_template
        K[sl, sl] += K_template

    if apply_fixed_bc:
        _apply_fixed_bc_inplace(M)
        _apply_fixed_bc_inplace(K)
    if damping_modes_ref is not None and damping_zetas_ref is not None:
        _, _, C, _ = rayleigh_damping(M, K, damping_modes_ref, damping_zetas_ref)
    else:
        C = np.zeros_like(M)
    if apply_fixed_bc:
        _apply_fixed_bc_inplace_damping(C)
        if _DEBUG_RAYLEIGH and _DEBUG_RAYLEIGH_LEVEL >= 2:
            print("[DEBUG] Applied damping BC to C (zeroed edge rows/cols)")
            _print_bc_edges("C after BC", C)
    return M, K, C

# ---------------------------------------------------------------------------
#             Assemble uniquement la matrice de masse globale
# ---------------------------------------------------------------------------
def assemble_mass(
    *,
    lin_density: float,                                     # Densité linéique μ (kg/m)
    dx_vector: Sequence[float] | None = None,               # Longueurs élémentaires (mode non uniforme)
    n_nodes: int | None = None,                             # Nombre de nœuds (mode uniforme) 
    length: float | None = None,                            # Longueur totale (mode uniforme)
    apply_fixed_bc: bool = False,                           # Applique CL fixes si True
) -> np.ndarray:
    """Assemble uniquement la matrice de masse globale M.

    Deux modes:
      - Non uniforme : fournir dx_vector (longueurs élémentaires en m).
      - Uniforme     : fournir n_nodes et length.

    Formule locale (élément de longueur dx):
        M_e = (μ * dx / 6) [[2, 1], [1, 2]]

    Paramètres
    ----------
    lin_density : float
        Densité linéique μ (kg/m).
    dx_vector : sequence[float] | None
        Longueurs élémentaires (mode non uniforme).
    n_nodes : int | None
        Nombre de nœuds (mode uniforme).
    length : float | None
        Longueur totale (mode uniforme).
    apply_fixed_bc : bool
        Si True applique CL fixes (nœuds extrêmes bloqués) sur M.
    
    Retour
    ------
    M : np.ndarray
        Matrice de masse globale (n_nodes x n_nodes).
    """
    if dx_vector is not None:
        dx_arr = np.asarray(dx_vector, dtype=float)
        if dx_arr.ndim != 1 or dx_arr.size == 0 or np.any(dx_arr <= 0):
            raise ValueError("dx_vector invalide pour assemble_mass")
        n_elems = dx_arr.size
        n_nodes_eff = n_elems + 1
        M = np.zeros((n_nodes_eff, n_nodes_eff), dtype=float)
        for i, dx in enumerate(dx_arr):
            mass_pref = lin_density * dx / 6.0
            M_local = mass_pref * np.array([[2.0, 1.0], [1.0, 2.0]])
            sl = slice(i, i + 2)
            M[sl, sl] += M_local
    else:
        if n_nodes is None or length is None:
            raise ValueError("Fournir n_nodes et length pour le mode uniforme dans assemble_mass")
        if n_nodes < 2:
            raise ValueError("n_nodes doit être >= 2")
        if length <= 0 or lin_density <= 0:
            raise ValueError("length et lin_density doivent être positifs")
        n_elems = n_nodes - 1
        dx = length / n_elems
        M = np.zeros((n_nodes, n_nodes), dtype=float)
        mass_pref = lin_density * dx / 6.0
        M_local = mass_pref * np.array([[2.0, 1.0], [1.0, 2.0]])
        for i in range(n_elems):
            sl = slice(i, i + 2)
            M[sl, sl] += M_local

    if apply_fixed_bc:
        _apply_fixed_bc_inplace(M)
    return M


# ---------------------------------------------------------------------------
# Assemblage dédié uniquement à la matrice de raideur K
# ---------------------------------------------------------------------------
def assemble_stiffness(
    *,
    tension: float,
    dx_vector: Sequence[float] | None = None,
    n_nodes: int | None = None,
    length: float | None = None,
    apply_fixed_bc: bool = False,
) -> np.ndarray:
    """Assemble seulement la matrice de raideur globale K.

    Deux modes (comme assemble_mkc) :
      - Non uniforme : fournir dx_vector (longueurs élémentaires en m).
      - Uniforme     : fournir n_nodes et length.

    Paramètres
    ----------
    tension : float
        Tension T (N).
    dx_vector : sequence[float] | None
        Longueurs élémentaires (mode non uniforme).
    n_nodes : int | None
        Nombre de nœuds (mode uniforme) si dx_vector est None.
    length : float | None
        Longueur totale (mode uniforme).
    apply_fixed_bc : bool
        Si True, applique des CL fixes (nœuds extrêmes bloqués) en modifiant K.

    Retour
    ------
    K : np.ndarray
        Matrice de raideur globale (n_nodes x n_nodes).
    """
    if dx_vector is not None:
        dx_arr = np.asarray(dx_vector, dtype=float)
        if dx_arr.ndim != 1 or dx_arr.size == 0 or np.any(dx_arr <= 0):
            raise ValueError("dx_vector invalide pour assemble_stiffness")
        n_elems = dx_arr.size
        n_nodes_eff = n_elems + 1
        K = np.zeros((n_nodes_eff, n_nodes_eff), dtype=float)
        for i, dx in enumerate(dx_arr):
            K_local = (tension / dx) * np.array([[1.0, -1.0], [-1.0, 1.0]])
            sl = slice(i, i + 2)
            K[sl, sl] += K_local
    else:
        if n_nodes is None or length is None:
            raise ValueError("Fournir n_nodes et length pour le mode uniforme dans assemble_stiffness")
        if n_nodes < 2:
            raise ValueError("n_nodes doit être >= 2")
        if length <= 0 or tension <= 0:
            raise ValueError("length et tension doivent être positifs")
        n_elems = n_nodes - 1
        dx = length / n_elems
        K = np.zeros((n_nodes, n_nodes), dtype=float)
        K_local = (tension / dx) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        for i in range(n_elems):
            sl = slice(i, i + 2)
            K[sl, sl] += K_local

    if apply_fixed_bc:
        _apply_fixed_bc_inplace(K)
    return K

# ---------------------------------------------------------------------------
#              Utilitaires d'inspection des matrices locales (DEBUG)
# ---------------------------------------------------------------------------
# [AVERTISSEMENT] Fonction d'inspection/débogage uniquement — ne pas utiliser en production
def compute_local_element_matrices(
    tension: float,
    lin_density: float,
    rayleigh_alpha: float,
    rayleigh_beta: float,
    dx_vector: Sequence[float] | None = None,
    n_nodes: int | None = None,
    length: float | None = None,
) -> list[dict[str, Any]]:
    """
    Calcule et retourne la liste des matrices locales (par élément) — mode NON UNIFORME uniquement.

    NOTE: Fonction destinée à l'inspection et au débogage uniquement; pas d'usage en production.

    Fournir dx_vector. Les paramètres n_nodes/length ne sont pas supportés ici.

    Retour
    ------
        list[dict] : chaque entrée contient
                {
                    'index': i,
                    'dx': dx_i,
                    'mass_pref': μ * dx_i / 6,
                    'stiff_pref': T / dx_i,
                    'M_local': array(2x2),
                    'K_local': array(2x2)
                }
    """
    entries: list[dict[str, Any]] = []

    # Mode non uniforme requis
    if dx_vector is None:
        raise ValueError("compute_local_element_matrices: dx_vector est requis (mode non uniforme uniquement)")

    dx_arr = np.asarray(dx_vector, dtype=float)

    # Validation de dx_vector
    if dx_arr.ndim != 1 or dx_arr.size == 0 or np.any(dx_arr <= 0):
        raise ValueError("dx_vector invalide")

    # Boucle d'assemblage élément par élément
    for i, dx in enumerate(dx_arr):
        mass_pref = lin_density * dx / 6.0

        # Matrices locales
        # La matrice locale M est calculée en fonction de la densité linéique et de la longueur de l'élément
        # M_local = (μ * dx / 6) * [[2, 1], [1, 2]]
        M_local = mass_pref * np.array([[2.0, 1.0], [1.0, 2.0]])

        # La matrice locale K est calculée en fonction de la tension et de la longueur de l'élément
        # K_local = (T / dx) * [[1, -1], [-1, 1]]
        stiff_pref = tension / dx
        K_local = stiff_pref * np.array([[1.0, -1.0], [-1.0, 1.0]])

        entries.append({
            'index': i,
            'dx': float(dx),
            'mass_pref': float(mass_pref),
            'stiff_pref': float(stiff_pref),
            'M_local': M_local,
            'K_local': K_local,
        })
    return entries


    # [AVERTISSEMENT] Fonction d'inspection/débogage uniquement — ne pas utiliser en production
def print_local_element_matrices(
    tension: float,
    lin_density: float,
    rayleigh_alpha: float,
    rayleigh_beta: float,
    dx_vector: Sequence[float] | None = None,
    n_nodes: int | None = None,
    length: float | None = None,
    limit: int | None = None,
) -> None:
    """
    Imprime les matrices locales de chaque élément — mode NON UNIFORME uniquement.

    NOTE: Fonction destinée à l'inspection et au débogage uniquement; pas d'usage en production.

    Paramètres
    ----------
    limit : int | None
        Si défini, n'affiche que les `limit` premiers éléments.
    """
    if dx_vector is None:
        raise ValueError("print_local_element_matrices: dx_vector est requis (mode non uniforme uniquement)")
    data = compute_local_element_matrices(
        tension=tension,
        lin_density=lin_density,
        rayleigh_alpha=rayleigh_alpha,
        rayleigh_beta=rayleigh_beta,
        dx_vector=dx_vector,
        n_nodes=None,
        length=None,
    )
    total = len(data)
    show = data if limit is None else data[:limit]
    print(f"--- MATRICES LOCALES (total éléments = {total}) ---")
    print("Paramètres globaux: μ = {:.6g} kg/m | T = {:.6g} N".format(
        lin_density, tension
    ))
    for entry in show:
        i = entry['index']
        dx = entry['dx']
        mass_pref = entry.get('mass_pref')
        stiff_pref = entry.get('stiff_pref')
        M_loc = entry['M_local']
        K_loc = entry['K_local']
        print(f"\nÉlément {i} | dx = {(dx*1000):.6g} mm")
        if mass_pref is not None and stiff_pref is not None:
            print("  Facteurs: mass_pref = μ*dx/6 = {:.6g} | stiff_pref = T/dx = {:.6g}".format(mass_pref, stiff_pref))
        print("M_local =\n", M_loc)
        print("K_local =\n", K_loc)
    if limit is not None and total > limit:
        print(f"\n... ({total - limit} éléments supplémentaires non affichés) ...")


def assemble_mkc(
    tension: float,
    lin_density: float,
    n_nodes: int | None = None,
    length: float | None = None,
    dx_vector: Sequence[float] | None = None,
    apply_fixed_bc: bool = False,
    return_meta: bool = False,
    damping_modes_ref: tuple[int, int] | None = None,
    damping_zetas_ref: tuple[float, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
        Assemble M,K,C uniquement en mode NON UNIFORME (dx_vector requis).

        Utilisation :
            - Non uniforme : fournir dx_vector (liste/array). Les paramètres n_nodes/length
                sont ignorés et ne doivent pas être utilisés ici.

    Retour
    ------
    (M,K,C) ou (M,K,C, meta) si return_meta=True.
    meta contient : {'mode': 'uniform'|'nonuniform', 'n_nodes', 'n_elems', 'dx_min', 'dx_max', 'length_recon'}.
    """
    if dx_vector is None:
        raise ValueError("assemble_mkc: dx_vector est requis (mode non uniforme uniquement)")

    # Mode non uniforme
    M, K, C = assemble_system_matrices_nonuniform(
        dx_vector=dx_vector,
        tension=tension,
        lin_density=lin_density,
        damping_modes_ref=damping_modes_ref,
        damping_zetas_ref=damping_zetas_ref,
        apply_fixed_bc=apply_fixed_bc,
    )
    dx_arr = np.asarray(dx_vector, dtype=float)
    meta = {
        'mode': 'nonuniform',
        'n_nodes': dx_arr.size + 1,
        'n_elems': dx_arr.size,
        'dx_min': float(dx_arr.min()),
        'dx_max': float(dx_arr.max()),
        'length_recon': float(dx_arr.sum()),
        'mass_total': float(lin_density * dx_arr.sum())
    }

    if return_meta:
        # Ajouter α, β et fréquences si paramètres fournis
        if damping_modes_ref is not None and damping_zetas_ref is not None:
            try:
                alpha, beta, _, omegas = rayleigh_damping(M, K, damping_modes_ref, damping_zetas_ref)
                meta.update({'alpha': alpha, 'beta': beta, 'omegas': omegas})
            except Exception:
                pass
        return M, K, C, meta
    return M, K, C





def build_global_mkc_from_config(apply_fixed_bc: bool = False, return_meta: bool = True):
    """
    Assemble automatiquement M,K,C à partir du module `config`.

    Détection : si `config.FRET_DXS_MM` existe et non vide -> mode non uniforme.
    Sinon -> ERREUR (dans ce contexte, on exige FRET_DXS_MM).

    Il n'y a pas de excuse pour calculer les elements globaux sans un vecteur de dx non uniforme (ça c'est necessaire pour les bonnes notes).

    Paramètres
    ----------
    apply_fixed_bc : bool
        Applique (ou non) des CL fixes aux extrémités.
    return_meta : bool
        Si True retourne aussi le dictionnaire meta de `assemble_mkc`.

    Retour
    ------
    (M,K,C) ou (M,K,C, meta)
    """
    try:
        from .. import config  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Impossible d'importer le module config dans build_global_mkc_from_config") from exc

    dxs_mm = config.FRET_DXS_MM
    if dxs_mm and len(dxs_mm) > 0:
        # Récupérer aussi les cibles d'amortissement dans config (strict, pas de défauts)
        try:
            modes_ref = config.DAMPING_MODES_REF  # type: ignore[attr-defined]
            zetas_ref = config.DAMPING_ZETAS_REF  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("[ERROR] DAMPING_MODES_REF ou DAMPING_ZETAS_REF manquants dans config") from exc
        if modes_ref is None or zetas_ref is None:
            raise RuntimeError("[ERROR] DAMPING_MODES_REF/DAMPING_ZETAS_REF invalides dans config")
        dx_vector = [d/1000.0 for d in dxs_mm]
        result = assemble_mkc(
            tension=config.T,
            lin_density=config.MU,
            dx_vector=dx_vector,
            apply_fixed_bc=apply_fixed_bc,
            return_meta=return_meta,
            damping_modes_ref=modes_ref,
            damping_zetas_ref=zetas_ref,
        )
    else:
        # Config invalide si on attend une maille frettes
        raise RuntimeError("[ERROR] FRET_DXS_MM absent ou vide dans config")
    return result


# ---------------------------------------------------------------------------
#             Assemble uniquement la matrice de masse globale
# ---------------------------------------------------------------------------
def assemble_mass(
    *,
    lin_density: float,                                     # Densité linéique μ (kg/m)
    dx_vector: Sequence[float] | None = None,               # Longueurs élémentaires (mode non uniforme)
    n_nodes: int | None = None,                             # Nombre de nœuds (mode uniforme) 
    length: float | None = None,                            # Longueur totale (mode uniforme)
    apply_fixed_bc: bool = False,                           # Applique CL fixes si True
) -> np.ndarray:
    """Assemble uniquement la matrice de masse globale M.

    Deux modes:
      - Non uniforme : fournir dx_vector (longueurs élémentaires en m).
      - Uniforme     : fournir n_nodes et length.

    Formule locale (élément de longueur dx):
        M_e = (μ * dx / 6) [[2, 1], [1, 2]]

    Paramètres
    ----------
    lin_density : float
        Densité linéique μ (kg/m).
    dx_vector : sequence[float] | None
        Longueurs élémentaires (mode non uniforme).
    n_nodes : int | None
        Nombre de nœuds (mode uniforme).
    length : float | None
        Longueur totale (mode uniforme).
    apply_fixed_bc : bool
        Si True applique CL fixes (nœuds extrêmes bloqués) sur M.
    
    Retour
    ------
    M : np.ndarray
        Matrice de masse globale (n_nodes x n_nodes).
    """
    if dx_vector is not None:
        dx_arr = np.asarray(dx_vector, dtype=float)
        if dx_arr.ndim != 1 or dx_arr.size == 0 or np.any(dx_arr <= 0):
            raise ValueError("dx_vector invalide pour assemble_mass")
        n_elems = dx_arr.size
        n_nodes_eff = n_elems + 1
        M = np.zeros((n_nodes_eff, n_nodes_eff), dtype=float)
        for i, dx in enumerate(dx_arr):
            mass_pref = lin_density * dx / 6.0
            M_local = mass_pref * np.array([[2.0, 1.0], [1.0, 2.0]])
            sl = slice(i, i + 2)
            M[sl, sl] += M_local
    else:
        if n_nodes is None or length is None:
            raise ValueError("Fournir n_nodes et length pour le mode uniforme dans assemble_mass")
        if n_nodes < 2:
            raise ValueError("n_nodes doit être >= 2")
        if length <= 0 or lin_density <= 0:
            raise ValueError("length et lin_density doivent être positifs")
        n_elems = n_nodes - 1
        dx = length / n_elems
        M = np.zeros((n_nodes, n_nodes), dtype=float)
        mass_pref = lin_density * dx / 6.0
        M_local = mass_pref * np.array([[2.0, 1.0], [1.0, 2.0]])
        for i in range(n_elems):
            sl = slice(i, i + 2)
            M[sl, sl] += M_local

    if apply_fixed_bc:
        _apply_fixed_bc_inplace(M)
    return M


# ---------------------------------------------------------------------------
# Assemblage dédié uniquement à la matrice de raideur K
# ---------------------------------------------------------------------------
def assemble_stiffness(
    *,
    tension: float,
    dx_vector: Sequence[float] | None = None,
    n_nodes: int | None = None,
    length: float | None = None,
    apply_fixed_bc: bool = False,
) -> np.ndarray:
    """Assemble seulement la matrice de raideur globale K.

    Deux modes (comme assemble_mkc) :
      - Non uniforme : fournir dx_vector (longueurs élémentaires en m).
      - Uniforme     : fournir n_nodes et length.

    Paramètres
    ----------
    tension : float
        Tension T (N).
    dx_vector : sequence[float] | None
        Longueurs élémentaires (mode non uniforme).
    n_nodes : int | None
        Nombre de nœuds (mode uniforme) si dx_vector est None.
    length : float | None
        Longueur totale (mode uniforme).
    apply_fixed_bc : bool
        Si True, applique des CL fixes (nœuds extrêmes bloqués) en modifiant K.

    Retour
    ------
    K : np.ndarray
        Matrice de raideur globale (n_nodes x n_nodes).
    """
    if dx_vector is not None:
        dx_arr = np.asarray(dx_vector, dtype=float)
        if dx_arr.ndim != 1 or dx_arr.size == 0 or np.any(dx_arr <= 0):
            raise ValueError("dx_vector invalide pour assemble_stiffness")
        n_elems = dx_arr.size
        n_nodes_eff = n_elems + 1
        K = np.zeros((n_nodes_eff, n_nodes_eff), dtype=float)
        for i, dx in enumerate(dx_arr):
            K_local = (tension / dx) * np.array([[1.0, -1.0], [-1.0, 1.0]])
            sl = slice(i, i + 2)
            K[sl, sl] += K_local
    else:
        if n_nodes is None or length is None:
            raise ValueError("Fournir n_nodes et length pour le mode uniforme dans assemble_stiffness")
        if n_nodes < 2:
            raise ValueError("n_nodes doit être >= 2")
        if length <= 0 or tension <= 0:
            raise ValueError("length et tension doivent être positifs")
        n_elems = n_nodes - 1
        dx = length / n_elems
        K = np.zeros((n_nodes, n_nodes), dtype=float)
        K_local = (tension / dx) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        for i in range(n_elems):
            sl = slice(i, i + 2)
            K[sl, sl] += K_local

    if apply_fixed_bc:
        _apply_fixed_bc_inplace(K)
    return K

# --- Exécution directe pour inspection rapide ---
# [AVERTISSEMENT] Bloc principal pour banc d'essai/diagnostic uniquement — ne pas utiliser en production
if __name__ == "__main__":  # Petit banc d'essai
    import sys
    from pathlib import Path

    try:
        from .. import config  # type: ignore
    except Exception:
        this_file = Path(__file__).resolve()
        back_end_dir = this_file.parent.parent
        root = back_end_dir.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from digital_twin.back_end import config  # type: ignore

    # Détection du mode (présence FRET_DXS_MM) pour test
    # Ça c'est le vecteur de dx issu du module de frettes
    dxs_mm = config.FRET_DXS_MM
    modes_ref = config.DAMPING_MODES_REF
    zetas_ref = config.DAMPING_ZETAS_REF
    apply_bc = config.APPLY_FIXED_BC
    if not dxs_mm:
        raise RuntimeError("[ERROR] FRET_DXS_MM absent ou vide dans config — mode non uniforme requis")
    # Nous avons besoin de souvenir de convertir les dx que sont en mm pour etre en m
    M, K, C, meta = assemble_mkc(
        tension=config.T,
        lin_density=config.MU,
        dx_vector=[d/1000.0 for d in dxs_mm],
        apply_fixed_bc=apply_bc,
        return_meta=True,
        damping_modes_ref=modes_ref,
        damping_zetas_ref=zetas_ref,
    )
    print("Meta:", meta)
    print("Dimensions:", M.shape)
    print("Symétrie M,K,C:", np.allclose(M,M.T), np.allclose(K,K.T), np.allclose(C,C.T))

    # Affiche un extrait de diagonales
    import numpy as _np
    print("Diag M:", _np.round(_np.diag(M),6))
    print("Diag K:", _np.round(_np.diag(K),6))
    print("Diag C:", _np.round(_np.diag(C),6))

    # ---------------------------------------------------------------
    # Diagnostics supplémentaires pour éviter l'effet "tout zéro" dû
    #                  à l'arrondi d'impression.
    # ---------------------------------------------------------------
    print("\n[DIAGNOSTIC] Statistiques M:")
    # Extraction des entrées non nulles
    nonzero = M[np.abs(M) > 0]
    if nonzero.size:
        print(f" min = {nonzero.min():.3e}")                                                                                            # Valeur minimale non nulle
        print(f" max = {nonzero.max():.3e}")                                                                                            # Valeur maximale
        print(f" somme = {M.sum():.3e}")                                                                                                # Somme des éléments
        masse_teoricale = float(config.MU * meta.get('length_recon', 0))
        print(f" masse totale théorique = μ * L = {masse_teoricale:.3e} kg")                                                            # Masse totale théorique
        diference_masse = abs(masse_teoricale - M.sum())
        print(f" diference masse totale - somme de la matrice M = {diference_masse:.3e}")                                               # Vérification masse totale
        print(f" porcentage de erreur sur la masse totale = {100.0 * diference_masse / float(config.MU * meta.get('length_recon', 1e-10)):.3e} %")  # Pourcentage d'erreur
    else:
        print("  (aucune entrée non nulle détectée)")

    # Impression en notation scientifique
    with _np.printoptions(precision=3, suppress=False):
        print("\nM (notation scientifique) =\n", M)

    # Version mise à l'échelle (micro) pour lecture intuitive
    scale = 1e6
    with _np.printoptions(precision=3, suppress=True):
        print(f"\nM (x{scale:.0e}) =\n", M * scale, "\n")

    # ---------------------------------------------------------------
    #      Diagnostics similaires pour la matrice de raideur K
    # ---------------------------------------------------------------
    print("[DIAGNOSTIC] Statistiques K:")
    nonzero_K = K[np.abs(K) > 0]
    if nonzero_K.size:
        print(f" min = {nonzero_K.min():.3e}")                      # Valeur minimale non nulle
        print(f" max = {nonzero_K.max():.3e}")                      # Valeur maximale
        print(f" somme = {K.sum():.3e}")                            # Somme des éléments
    else:
        print("  (aucune entrée non nulle détectée)")

    with _np.printoptions(precision=3, suppress=False):
        print("\nK (notation scientifique) =\n", K)
    # Impression supplémentaire uniquement si valeurs TRÈS grandes
    if nonzero_K.size:
        max_abs_K = float(np.max(np.abs(nonzero_K)))
        if max_abs_K > 1e3:                                         # Seuil arbitraire pour considérer "grand"
            scale_K = 1e-3                                          # Échelle milli
            with _np.printoptions(precision=3, suppress=True):      # _np.printoptions nous permet de faire un contexte temporaire
                print(f"\nK (échelle réduite x{scale_K:.0e}) =\n", K * scale_K, "\n")

    # ---------------------------------------------------------------
    #      Diagnostics similaires pour la matrice d'amortissement C
    # ---------------------------------------------------------------
    print("[DIAGNOSTIC] Statistiques C:")
    nonzero_C = C[np.abs(C) > 0]
    if nonzero_C.size:
        print(f" min = {nonzero_C.min():.3e}")                      # Valeur minimale non nulle
        print(f" max = {nonzero_C.max():.3e}")                      # Valeur maximale
        print(f" somme = {C.sum():.3e}")                            # Somme des éléments
    else:
        print("  (aucune entrée non nulle détectée)")

    with _np.printoptions(precision=3, suppress=False):
        print("\nC (notation scientifique) =\n", C)
    # Comme pour K, on ne réduit l'échelle que si les valeurs sont très grandes
    if nonzero_C.size:
        max_abs_C = float(np.max(np.abs(nonzero_C)))
        if max_abs_C > 1e3:
            scale_C = 1e-3
            with _np.printoptions(precision=3, suppress=True):
                print(f"\nC (échelle réduite x{scale_C:.0e}) =\n", C * scale_C, "\n")

    # ---------------------------------------------------------------
    #    Diagnostics Rayleigh: α, β, normes et amortissements modaux
    # ---------------------------------------------------------------
    try:
        alpha = meta.get('alpha', None)
        beta = meta.get('beta', None)
        _omegas_meta = np.asarray(meta.get('omegas', []), dtype=float)
        if alpha is not None and beta is not None:
            print("\n[DIAGNOSTIC] Rayleigh (C = α M + β K):")
            print(f"  Alpha = {alpha:.6e} [1/s] | Beta = {beta:.6e} [s]")

            # Normes de Frobenius pour évaluer les ordres de grandeur
            nM = float(np.linalg.norm(M, 'fro'))
            nK = float(np.linalg.norm(K, 'fro'))
            nC = float(np.linalg.norm(C, 'fro'))
            nAM = float(np.linalg.norm(alpha * M, 'fro'))
            nBK = float(np.linalg.norm(beta * K, 'fro'))
            share_am = (nAM / nC * 100.0) if nC > 0 else 0.0
            share_bk = (nBK / nC * 100.0) if nC > 0 else 0.0
            print(f"  ||M||_F = {nM:.3e} | ||K||_F = {nK:.3e} | ||C||_F = {nC:.3e}")
            print(f"  ||αM||_F = {nAM:.3e} ({share_am:.1f}% de ||C||) | ||βK||_F = {nBK:.3e} ({share_bk:.1f}% de ||C||)")

            # Amortissements modaux réalisés: ζ_i = 1/2 (α/ω_i + β ω_i)
            if _omegas_meta.size > 0:
                zetas_real = 0.5 * (alpha / _omegas_meta + beta * _omegas_meta)
                k_show = int(min(10, zetas_real.size))
                with _np.printoptions(precision=4, suppress=True):
                    print(f"  ζ(modaux) réalisés — premiers {k_show}:", zetas_real[:k_show])
                # Mettre en évidence les modes de référence et l'écart
                if isinstance(modes_ref, tuple) and len(modes_ref) == 2:
                    p, q = modes_ref
                    if 0 <= p < zetas_real.size and 0 <= q < zetas_real.size:
                        zp, zq = zetas_real[p], zetas_real[q]
                        try:
                            zt_p, zt_q = zetas_ref
                        except Exception:
                            zt_p, zt_q = None, None
                        msg_p = f"p={p}: ζ_real = {zp:.5f}"
                        if zt_p is not None:
                            msg_p += f" | ζ_cible = {zt_p:.5f} | Δ = {abs(zp - zt_p):.2e}"
                        msg_q = f"q={q}: ζ_real = {zq:.5f}"
                        if zt_q is not None:
                            msg_q += f" | ζ_cible = {zt_q:.5f} | Δ = {abs(zq - zt_q):.2e}"
                        print("  Cibles:")
                        print("   ", msg_p)
                        print("   ", msg_q)
                        print("  (Des valeurs de C petites peuvent être NORMALES si les ζ réalisés correspondent aux cibles.)")
    except Exception as _e_diag:
        print("[WARN] Échec des diagnostics Rayleigh:", _e_diag)

    # Affiche maintenant les matrices locales (c'est possible de limiter en cas de grand nombre d'éléments)
    #print_local_element_matrices(
    #    tension=config.T,                                           # Paramètres globaux -> Tension                                        
    #    lin_density=config.MU,                                      # Paramètres globaux -> Densité linéique    
    #    rayleigh_alpha=0.0,                                         # Non utilisé ici (calcul α/β modal)
    #    rayleigh_beta=0.0,                                          # Non utilisé ici (calcul α/β modal)
    #    dx_vector=[d/1000.0 for d in dxs_mm] if dxs_mm else None,   # Mode non uniforme si dxs_mm existe et converti en m
    #    n_nodes=None if dxs_mm else config.N_NODES,                 # Le nombre de nœuds n'est utile que pour le mode uniforme
    #    length=None if dxs_mm else config.L,                        # La longueur totale n'est utile que pour le mode uniforme   
    #    limit=None,                                                 # Se juste placer ici le limite que vous voulez
    #)

    # ---------------------------------------------------------------
    #            Fréquences propres: affichage de contrôle
    # ---------------------------------------------------------------
    try:
        # 1) Source préférée des pulsations propres ω (rad/s):
        #    Si le calcul de Rayleigh (α, β) a déjà été effectué, la fonction
        #    rayleigh_damping(...) a résolu le problème aux valeurs propres
        #    généralisé K x = ω² M x (sur les DDL libres) et a stocké les ω
        #    dans meta['omegas'].
        _omegas = np.asarray(meta.get('omegas', []), dtype=float)

        # 2) Fallback si meta ne contient pas les ω:
        #    On recalcule rapidement à partir de M et K en formant la matrice
        #    A = M^{-1} K, puis en résolvant A v = λ v avec λ = ω².
        #    Détails des appels numpy:
        #      - np.linalg.solve(M, K): résout M · A = K pour A (évite l'inversion explicite de M).
        #      - np.linalg.eig(A): renvoie (valeurs propres λ, vecteurs propres).
        #      - np.real(...): on projette sur la partie réelle (petits imaginaires numériques -> 0).
        #      - np.sqrt(λ): donne ω = √λ (rad/s).
        if _omegas.size == 0:
            # NOTE sur CL (Dirichlet via diag=1 et lignes/colonnes nulles):
            # Si M et K contiennent déjà ces CL, A inclura des modes triviaux
            # contraints. C'est acceptable ici car ce bloc est purement
            # diagnostique et, en pratique, meta['omegas'] est préféré.
            A = np.linalg.solve(M, K)         # A ≡ M^{-1} K (sans inversion explicite)
            eigvals, _ = np.linalg.eig(A)     # valeurs propres λ ≈ ω²
            eigvals = np.real(eigvals)        # on ignore de minuscules parties imaginaires
            eigvals[eigvals < 0] = 0.0        # coupe les petites négativités dues aux erreurs d'arrondi
            _omegas = np.sqrt(eigvals)        # ω = √λ (rad/s)
            _omegas.sort()                    # tri croissant des pulsations

        # 3) Conversion en fréquences f (Hz): f = ω / (2π)
        f_hz = _omegas / (2 * np.pi)

        # 4) Impression formatée: np.printoptions crée un contexte temporaire
        #    de formatage (ici: 3 décimales, suppression des notations
        #    scientifiques pour les petites valeurs si approprié).
        with _np.printoptions(precision=3, suppress=True):
            print("\nFréquences (Hz) — premières 10 =", f_hz[:10])

        # 5) Rappel des modes de référence (si définis): on affiche les
        #    fréquences associées aux indices (p, q) et les ζ demandés.
        if isinstance(modes_ref, tuple) and len(modes_ref) == 2:
            p, q = modes_ref
            if 0 <= p < f_hz.size and 0 <= q < f_hz.size:
                print(f"Modes réf. (p={p}, q={q}) → f_p = {f_hz[p]:.3f} Hz | f_q = {f_hz[q]:.3f} Hz | ζ = {zetas_ref}")
    except Exception as e:
        # C'est juste un diagnostic, on ne veut pas planter le script
        print("[WARN] Échec lors de l'impression des fréquences:", e)