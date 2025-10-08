# Formulation FEM 1D (corde tendue) — maillage uniforme et non uniforme
#
# Objectifs
# ---------
# Fournir des routines d'assemblage des matrices globales :
#  - Masse   (M)
#  - Raideur (K)
#  - Amortissement de Rayleigh (C = α M + β K)
#
# Points clés
# -----------
# 1) Deux cas de discrétisation via une API unifiée :
#    - Uniforme: n_nodes + longueur totale -> dx constant.
#    - Non uniforme: vecteur dx_i explicite.
# 2) Matrices locales (élément de longueur dx):
#       M_e = (μ * dx / 6) [[2, 1], [1, 2]]
#       K_e = (T / dx)     [[1, -1], [-1, 1]]
#    La matrice d'amortissement globale N'EST PAS assemblée élément par élément; on fait
#    C = α M + β K APRÈS l'assemblage global.
# 3) Assemblage additif standard sur blocs (i,i), (i,i+1), (i+1,i), (i+1,i+1).
# 4) CL fixes (extrémités bloquées) appliquées directement sur M/K; C est ensuite construit
#    et reçoit bords annulés (sans diagonale =1) via _apply_fixed_bc_inplace_damping.
# 5) α, β de Rayleigh sont déduits de 2 amortissements modaux cibles.
# 6) Option de retourner des métadonnées (dx_min, dx_max, etc.).
#
# Conventions
# -----------
#  - n_nodes = nœuds totaux
#  - n_elems = n_nodes - 1
#  - l’élément e relie les nœuds e et e+1
#
# Sécurité & Validation
# ---------------------
#  - Tous les dx > 0
#  - n_nodes >= 2
#
# API publique
# ------------
#  - assemble_system_matrices_nonuniform(dx_vector, ...)
#  - assemble_system_matrices(n_nodes, length, ...)
#  - assemble_mkc(...)

from __future__ import annotations

import numpy as np
from typing import Sequence, Tuple, Dict, Any
try:
    # Preferred: package-relative import
    from ..utils import debug as dbg  # type: ignore
except Exception:
    # Try absolute import; if it fails, add project root to sys.path and retry
    try:
        from digital_twin.back_end.utils import debug as dbg  # type: ignore
    except Exception:
        try:
            import sys
            from pathlib import Path
            ROOT = Path(__file__).resolve().parents[3]  # repo root containing 'digital_twin'
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from digital_twin.back_end.utils import debug as dbg  # type: ignore
        except Exception:
            # Fallback no-op shim to avoid crashes during direct execution
            class _DbgNoOp:
                @staticmethod
                def is_enabled() -> bool: return False
                @staticmethod
                def dprint(*args, **kwargs): pass
                @staticmethod
                def section(*args, **kwargs): pass
                @staticmethod
                def matrix_stats(*args, **kwargs): pass
                @staticmethod
                def print_blocks(*args, **kwargs): pass
                @staticmethod
                def bc_edge_sums(*args, **kwargs): pass
            dbg = _DbgNoOp()  # type: ignore

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

# Debug is centrally managed in digital_twin.back_end.utils.debug (imported as dbg)

# ---------------------------------------------------------------------------
# Fonctions internes (non exportées)
# ---------------------------------------------------------------------------
# Calcul des matrices locales d'un élément 1D (corde tendue)
def _matrix_stats(name: str, M: np.ndarray) -> None:
    # Use centralized debug helper
    dbg.matrix_stats(name, M)


def _print_blocks(name: str, M: np.ndarray, k: int = 4) -> None:
    dbg.print_blocks(name, M, k=k)


def _print_bc_edges(name: str, M: np.ndarray) -> None:
    dbg.bc_edge_sums(name, M)

def _detect_constrained_dofs(M: np.ndarray, K: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    # Détecte les DDL contraints de type Dirichlet selon la convention du projet :
    # - lignes/colonnes nulles hors diagonale
    # - diagonale ≈ 1 dans M et K aux nœuds extrêmes (préserve l'inversibilité)
    n = M.shape[0]                                  # .shape renvoie un tuple (n,), on veut l'entier
    constrained = []
    for i in range(n):
        # Copie des lignes/colonnes sans la diagonale
        rowM = M[i, :].copy(); rowM[i] = 0.0        # Ignore diagonal
        colM = M[:, i].copy(); colM[i] = 0.0        # Ignore diagonal
        rowK = K[i, :].copy(); rowK[i] = 0.0        # Ignore diagonal
        colK = K[:, i].copy(); colK[i] = 0.0        # Ignore diagonal
        # Verifie la nullité hors diagonale et diag≈1 dans M et K
        # .all pour s'assurer que c'est vrai pour tous les éléments du vecteur
        # .abs pour tolérer des petites valeurs négatives numériques
        if (
            np.all(np.abs(rowM) <= atol) and np.all(np.abs(colM) <= atol)
            and np.all(np.abs(rowK) <= atol) and np.all(np.abs(colK) <= atol)
            and abs(M[i, i] - 1.0) <= 1e-9 and abs(K[i, i] - 1.0) <= 1e-9
        ):
            constrained.append(i)
    # Retourne les DDL contraints détectés
    # .asarray pour compatibilité avec np.setdiff1d en aval
    return np.asarray(constrained, dtype=int)

def _apply_fixed_bc_inplace(mat: np.ndarray) -> None:
    # Applique les CL fixes in-place (Dirichlet) :
    # - annule les lignes/colonnes des extrémités; impose diag=1 sur ces nœuds
    #   afin de garder la matrice inversible lors de la résolution linéaire.
        mat[0, :] = 0.0     # ligne 0 -> première ligne
        mat[-1, :] = 0.0    # ligne -1 -> dernière ligne
        mat[:, 0] = 0.0     # colonne 0 -> première colonne
        mat[:, -1] = 0.0    # colonne -1 -> dernière colonne
        mat[0, 0] = 1.0     # impose diag=1 -> premier nœud
        mat[-1, -1] = 1.0   # impose diag=1 -> dernier nœud

def _apply_fixed_bc_inplace_damping(mat: np.ndarray) -> None:
    # CL pour la matrice d'amortissement C :
    # - annule les lignes/colonnes aux extrémités
    # - NE PAS mettre diag=1 (pas de sens physique pour C)
    mat[0, :] = 0.0         # ligne 0 -> première ligne
    mat[-1, :] = 0.0        # ligne -1 -> dernière ligne
    mat[:, 0] = 0.0         # colonne 0 -> première colonne
    mat[:, -1] = 0.0        # colonne -1 -> dernière colonne

def rayleigh_damping(
    M: np.ndarray,
    K: np.ndarray,
    modes_ref: tuple[int, int],
    zetas_ref: tuple[float, float],
):
    # Calcul des paramètres d'amortissement de Rayleigh (α, β) à partir de
    # deux amortissements modaux cibles (ζ).
    # Formule: ζ(ω) = 1/2 (α/ω + β ω). Résoufre le système 2x2 nos modos p e q.
    # Retourne: alpha, beta, C (= αM + βK), omegas (rad/s) des modos libres.
    # Si des CL fixes ont été appliquées via diag=1 et lignes/colonnes nulles,
    # retirer ces DDL pour le calcul propre des modes physiques.
    constrained = _detect_constrained_dofs(M, K)
    
    # Vérifie si des DDL sont contraints
    # Si oui, on retire ces DDL pour le calcul des modes physiques
    # Si non, on utilise M et K tels quels
    if constrained.size > 0:    # .size returne le nombre d'éléments                
        # .setdiff1d pour obtenir les DDL libres
        # .arrange pour générer [0, 1, 2, ..., N-1]
        # .shape[0] pour obtenir le nombre de DDL (M est carré)
        free = np.setdiff1d(np.arange(M.shape[0]), constrained)
        # .ix_ pour indexer les sous-matrices
        M_eval = M[np.ix_(free, free)]
        K_eval = K[np.ix_(free, free)]
    else:
        M_eval, K_eval = M, K

    # Évite l'inversion explicite: résout M X = K -> X = M^{-1} K
    A = np.linalg.solve(M_eval, K_eval)     # .linalg.solve pour éviter l'inversion explicite
    eigvals, _ = np.linalg.eig(A)           # .eig pour obtenir les autovalues/vecteurs de A ~ ω^2
    eigvals = np.real(eigvals)              # .real pour éviter les petites imaginaire numériques
    eigvals[eigvals < 0] = 0.0              # coupe les petites négativités numériques
    omegas = np.sqrt(eigvals)               # ω = sqrt(λ) (rad/s) 
    omegas.sort()                           # trie croissant (utile pour les modes de référence)

    p, q = modes_ref                        # indices des modes de référence
    if p < 0 or q < 0 or p >= omegas.size or q >= omegas.size or p == q:    # .size returne le nombre d'éléments
        raise ValueError("modes_ref invalides par rapport au nombre de modes disponibles")
    zeta_p, zeta_q = zetas_ref              # amortissements modaux cibles
    omega_p, omega_q = float(omegas[p]), float(omegas[q])                   # Ça doit être float pour la suite

    # Validité des entrées de référence
    # Juste pour éviter des divisions par zéro ou des cas non physiques
    if omega_p <= 0 or omega_q <= 0:
        raise ValueError("Fréquences de référence doivent être strictement positives")
    if zeta_p < 0 or zeta_q < 0:
        raise ValueError("Amortissements modaux de référence doivent être positifs ou nuls")

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
    # Debug prints (single on/off)
    if dbg.is_enabled():
        try:
            two_pi = 2.0 * np.pi
            f_hz = omegas / two_pi
            p, q = modes_ref
            zp, zq = zetas_ref
            # Back-substitution check
            zeta_p_chk = 0.5 * (alpha/omega_p + beta*omega_p)
            zeta_q_chk = 0.5 * (alpha/omega_q + beta*omega_q)

            # Concise 1-2 line summary (lazy formatting)
            dbg.dlazy(lambda: (
                f"[RAYLEIGH] p={p}, q={q} | f_p={f_hz[p]:.2f} Hz, f_q={f_hz[q]:.2f} Hz | "
                f"alpha={alpha:.3e} [1/s], beta={beta:.3e} [s]"
            ), prefix="")
            dbg.dlazy(lambda: (
                f"[RAYLEIGH] ζp tgt/ok Δ: {zp:.4f}/{zeta_p_chk:.4f} Δ={abs(zeta_p_chk-zp):.1e} | "
                f"ζq tgt/ok Δ: {zq:.4f}/{zeta_q_chk:.4f} Δ={abs(zeta_q_chk-zq):.1e}"
            ), prefix="")
        except Exception as _e_dbg:
            dbg.dprint(f"[AVERTISSEMENT] Échec d'impression Rayleigh: {_e_dbg}")
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
    # Assemble M et K, puis calcule C = α M + β K pour une maille NON uniforme.
    # - dx_vector : longueurs élémentaires (m)
    # - apply_fixed_bc : applique des CL fixes (extrémités bloquées)
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
    # Amortissement global: calculer α, β à partir du système de référence
    # puis former C = α M + β K sur les matrices PRE-CL pour éviter que les
    # diagonales artificielles (=1) se propagent dans C.
    if damping_modes_ref is not None and damping_zetas_ref is not None:
        if apply_fixed_bc:
            # Déterminer α, β sur la structure CONTRAINTE (physiquement cohérent)
            M_eval = M.copy(); K_eval = K.copy()
            _apply_fixed_bc_inplace(M_eval)
            _apply_fixed_bc_inplace(K_eval)
            alpha, beta, _, _ = rayleigh_damping(M_eval, K_eval, damping_modes_ref, damping_zetas_ref)
        else:
            alpha, beta, _, _ = rayleigh_damping(M, K, damping_modes_ref, damping_zetas_ref)
        C = alpha * M + beta * K
    else:
        C = np.zeros_like(M)

    # Appliquer ensuite les CL: M/K avec diag=1, C avec bords annulés (sans 1)
    if apply_fixed_bc:
        _apply_fixed_bc_inplace(M)
        _apply_fixed_bc_inplace(K)
        _apply_fixed_bc_inplace_damping(C)
        if dbg.is_enabled():
            dbg.dprint("CL d'amortissement appliquées à C (lignes/colonnes de bord annulées)")
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
    # Assemble M et K, puis C = α M + β K, pour une maille UNIFORME (n_nodes, length -> dx constant).
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

    # Amortissement global: α, β depuis le système de référence
    if damping_modes_ref is not None and damping_zetas_ref is not None:
        if apply_fixed_bc:
            # Évaluer α, β sur la version contrainte (M/K avec CL)
            M_eval = M.copy(); K_eval = K.copy()
            _apply_fixed_bc_inplace(M_eval)
            _apply_fixed_bc_inplace(K_eval)
            alpha, beta, _, _ = rayleigh_damping(M_eval, K_eval, damping_modes_ref, damping_zetas_ref)
        else:
            alpha, beta, _, _ = rayleigh_damping(M, K, damping_modes_ref, damping_zetas_ref)
        # Former C sur les matrices PRE-CL pour ne pas propager les diag=1
        C = alpha * M + beta * K
    else:
        C = np.zeros_like(M)

    # Appliquer ensuite les CL: M/K (diag=1 sur bords), C (bords annulés)
    if apply_fixed_bc:
        _apply_fixed_bc_inplace(M)
        _apply_fixed_bc_inplace(K)
        _apply_fixed_bc_inplace_damping(C)
        if dbg.is_enabled():
            dbg.dprint("CL d'amortissement appliquées à C (lignes/colonnes de bord annulées)")
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
    # Assemble uniquement la matrice de masse M (uniforme ou non uniforme).
    # Formule locale : M_e = (μ * dx / 6) [[2, 1], [1, 2]]
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
    # Assemble uniquement la matrice de raideur K (uniforme ou non uniforme).
    # Formule locale : K_e = (T / dx) [[1, -1], [-1, 1]]
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
    # Calcule et retourne les matrices locales par élément (mode NON uniforme).
    # Usage : inspection/débogage. Requiert dx_vector.
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
    # Wrapper de débogage : redirige vers dbg.print_local_element_matrices (silencieux si debug OFF).
    # Conservé pour compatibilité ; utilise seulement le chemin centralisé utils.debug.
    if dx_vector is None:
        raise ValueError("print_local_element_matrices: dx_vector est requis (mode non uniforme uniquement)")
    # Déléguer vers l'aide centralisée (elle effectue elle-même les impressions protégées)
    try:
        dbg.print_local_element_matrices(
            tension=tension,
            lin_density=lin_density,
            dx_vector=list(dx_vector),
            limit=limit,
        )
    except Exception:
    # Ne jamais casser le flux à cause du débogage
        pass


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
if __name__ == "__main__":  # Petit banc d'essai (silencieux se debug desligado)
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
    # Diagnostics (silencieux quando debug desativado)
    if hasattr(dbg, "print_formulation_diagnostics"):
        dbg.print_formulation_diagnostics(M, K, C, meta=meta, modes_ref=modes_ref, zetas_ref=zetas_ref, config=config)