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
#  - assemble_mkc(...)
#    (le bas-niveau _assemble_system_matrices_nonuniform reste interne)

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
                @staticmethod
                def print_formulation_diagnostics(*args, **kwargs): pass
            dbg = _DbgNoOp()  # type: ignore

# ---------------------------------------------------------------------------
# __all__ : définit l'API publique du module
# ---------------------------------------------------------------------------
__all__ = [
    "assemble_mkc",
    "build_global_mkc_from_config",
    "rayleigh_damping",
]


def _detect_constrained_dofs(M: np.ndarray, K: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    # Détecte les DDL contraints de type Dirichlet selon la convention du projet :
    # - lignes/colonnes nulles hors diagonale
    # - diagonale ≈ 1 dans M et K aux nœuds extrêmes (préserve l'inversibilité)
    n = M.shape[0]
    constrained: list[int] = []
    for i in range(n):
        rowM = M[i, :].copy()
        rowM[i] = 0.0
        colM = M[:, i].copy()
        colM[i] = 0.0
        rowK = K[i, :].copy()
        rowK[i] = 0.0
        colK = K[:, i].copy()
        colK[i] = 0.0

        if (
            np.all(np.abs(rowM) <= atol) and np.all(np.abs(colM) <= atol)
            and np.all(np.abs(rowK) <= atol) and np.all(np.abs(colK) <= atol)
            and abs(M[i, i] - 1.0) <= 1e-9 and abs(K[i, i] - 1.0) <= 1e-9
        ):
            constrained.append(i)
    return np.asarray(constrained, dtype=int)


def _apply_fixed_bc_inplace(mat: np.ndarray) -> None:
    # Applique les CL fixes in-place (Dirichlet) :
    # - annule les lignes/colonnes des extrémités; impose diag=1 sur ces nœuds
    mat[0, :] = 0.0     # ligne 0
    mat[-1, :] = 0.0    # ligne -1
    mat[:, 0] = 0.0     # colonne 0
    mat[:, -1] = 0.0    # colonne -1
    mat[0, 0] = 1.0     # diag 0,0 = 1
    mat[-1, -1] = 1.0   # diag -1,-1 = 1


def _apply_fixed_bc_inplace_damping(mat: np.ndarray) -> None:
    # CL pour la matrice d'amortissement C :
    # - annule les lignes/colonnes aux extrémités
    # - NE PAS mettre diag=1 (pas de sens physique pour C)
    mat[0, :] = 0.0     # ligne 0
    mat[-1, :] = 0.0    # ligne -1
    mat[:, 0] = 0.0     # colonne 0
    mat[:, -1] = 0.0    # colonne -1


def rayleigh_damping(
    M: np.ndarray,
    K: np.ndarray,
    modes_ref: tuple[int, int],
    zetas_ref: tuple[float, float],
):
    # Calcul des paramètres d'amortissement de Rayleigh (α, β) à partir de deux amortissements modaux cibles (ζ).
    # Formule: ζ(ω) = 1/2 (α/ω + β ω). Retourne: alpha, beta, C (= αM + βK), omegas (rad/s) des modes libres.
    constrained = _detect_constrained_dofs(M, K)

    # .size returne le nombre d'éléments, si > 0 on extrait les DDL libres
    if constrained.size > 0:
        # .setdiff1d pour obtenir les DDL libres
        # .arrange pour générer un tableau d'indices
        # .shape[0] returne le nombre de lignes (ou colonnes)
        free = np.setdiff1d(np.arange(M.shape[0]), constrained)
        # .ix_ pour extraire les sous-matrices
        M_eval = M[np.ix_(free, free)]
        K_eval = K[np.ix_(free, free)]
    else:
        M_eval, K_eval = M, K

    # .linalg.solve pour résoudre le système d'équations
    # A = M^(-1) K
    A = np.linalg.solve(M_eval, K_eval)
    eigvals, _ = np.linalg.eig(A)       # .linalg.eig pour obtenir les valeurs propres/modes
    eigvals = np.real(eigvals)          # .real pour éviter les petites parties imaginaires numériques
    eigvals[eigvals < 0] = 0.0          # élimine les négatives par erreur numérique
    omegas = np.sqrt(eigvals)           # rad/s
    omegas.sort()                       # tri croissant

    p, q = modes_ref                    # indices des modes de référence
    # Validation des entrées de référence
    # - indices valides et distincts
    # - fréquences strictement positives
    if p < 0 or q < 0 or p >= omegas.size or q >= omegas.size or p == q:
        raise ValueError("modes_ref invalides par rapport au nombre de modes disponibles")
    zeta_p, zeta_q = zetas_ref          # amortissements modaux cibles
    omega_p, omega_q = float(omegas[p]), float(omegas[q])  # rad/s

    # Validation des entrées de référence
    # - indices valides et distincts
    # - fréquences strictement positives
    if omega_p <= 0 or omega_q <= 0:
        raise ValueError("Fréquences de référence doivent être strictement positives")
    if zeta_p < 0 or zeta_q < 0:
        raise ValueError("Amortissements modaux de référence doivent être positifs ou nuls")

    # Résolution du système linéaire pour α et β
    # [1/ω_p   ω_p ] [α] = [2 ζ_p]
    # [1/ω_q   ω_q ] [β]   [2 ζ_q]
    # .array pour construire les matrices/vecteurs
    A_sys = np.array([[1.0/omega_p, omega_p], [1.0/omega_q, omega_q]], dtype=float)
    b_sys = np.array([2.0*zeta_p, 2.0*zeta_q], dtype=float)
    try:
        # Résolution du système linéaire
        alpha_lin, beta_lin = np.linalg.solve(A_sys, b_sys)
    except np.linalg.LinAlgError:
        # Système singulier (modes trop proches)
        denom = (omega_p**2 - omega_q**2)
        if abs(denom) < 1e-16:
            raise ValueError("Fréquences de référence très proches: impossible de déterminer α et β de façon stable")
        beta_lin = 2.0 * (zeta_p * omega_p - zeta_q * omega_q) / denom
        alpha_lin = 2.0 * zeta_p * omega_p - beta_lin * (omega_p**2)

    # Utilise alpha_lin/beta_lin calculés (quelque soit le chemin)
    alpha = float(alpha_lin)            # [1/s]
    beta = float(beta_lin)              # [s]
    C = alpha * M + beta * K            # matrice d'amortissement globale
    if dbg.is_enabled():  # information de débogage (centrée dans debug.py)
        try:
            dbg.print_rayleigh_explained(alpha, beta, omegas, modes_ref, zetas_ref, print_once=True)
        except Exception as _e_dbg:
            dbg.dprint(f"[AVERTISSEMENT] Échec d'impression Rayleigh: {_e_dbg}")
    return float(alpha), float(beta), C, omegas


def _assemble_system_matrices_nonuniform(
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

    # .asarray pour accepter les listes et tuples
    # dtype=float pour s'assurer que c'est du float64 (par défaut) pour la stabilité numérique
    dx_arr = np.asarray(dx_vector, dtype=float)

    # .ndim returne le nombre de dimensions (1D, 2D, etc.)
    # .size returne le nombre d'éléments
    # On veut utiliser ces deux propriétés pour valider l'entrée
    # On veut un vecteur 1D non vide
    if dx_arr.ndim != 1 or dx_arr.size == 0:
        raise ValueError("dx_vector doit être un vecteur 1D non vide")
    
    # .any pour vérifier que tous les éléments sont > 0
    if np.any(dx_arr <= 0):
        raise ValueError("Toutes les longueurs d'élément doivent être positives")
    n_elems = dx_arr.size                 # nombre d'éléments
    n_nodes = n_elems + 1                 # nombre de nœuds

    # Allocation des matrices globales (C sera calculée à la fin)
    # .zeros pour initialiser à zéro
    # dtype=float pour s'assurer que c'est du float64 (par défaut) pour la stabilité numérique
    M = np.zeros((n_nodes, n_nodes), dtype=float)       # Matrice de masse globale
    K = np.zeros((n_nodes, n_nodes), dtype=float)       # Matrice de raideur globale 

    # Boucle d'assemblage élément par élément (on ne construit pas C_local globalement)
    for i, dx in enumerate(dx_arr):
        mass_pref = lin_density * dx / 6.0                                  # préfacteur masse
        M_local = mass_pref * np.array([[2.0, 1.0], [1.0, 2.0]])            # matrice locale masse
        K_local = (tension / dx) * np.array([[1.0, -1.0], [-1.0, 1.0]])     # matrice locale raideur
        # slice c'est utile pour indexer les blocs (i,i), (i,i+1), (i+1,i), (i+1,i+1)
        sl = slice(i, i + 2)                                                # slice pour les nœuds de l'élément i
        M[sl, sl] += M_local                                                # assemblage additif
        K[sl, sl] += K_local                                                # assemblage additif
        
    # Amortissement global: calculer α, β à partir du système de référence
    # puis former C = α M + β K sur les matrices PRE-CL pour éviter que les
    # diagonales artificielles (=1) se propagent dans C.
    if damping_modes_ref is not None and damping_zetas_ref is not None:
        if apply_fixed_bc:
            # Déterminer α, β sur la structure CONTRAINTE (physiquement cohérent)
            M_eval = M.copy()                   # copie pour ne pas modifier M/K originaux 
            K_eval = K.copy()                   # copie pour ne pas modifier M/K originaux
            _apply_fixed_bc_inplace(M_eval)     # applique CL fixes in-place
            _apply_fixed_bc_inplace(K_eval)    # calcule α, β sur la version contrainte
            alpha, beta, _, _ = rayleigh_damping(M_eval, K_eval, damping_modes_ref, damping_zetas_ref)
        else:
            alpha, beta, _, _ = rayleigh_damping(M, K, damping_modes_ref, damping_zetas_ref)
        C = alpha * M + beta * K
    else:
        # Pas d'amortissement de Rayleigh: C = 0
        # .zeros_like pour créer une matrice de même forme que M/K avec zéros
        C = np.zeros_like(M)

    # Appliquer ensuite les CL: M/K avec diag=1, C avec bords annulés (sans 1)
    if apply_fixed_bc:
        # Applique CL fixes in-place (Dirichlet) sur M et K
        _apply_fixed_bc_inplace(M)
        _apply_fixed_bc_inplace(K)
        # Applique CL d'amortissement in-place (bords annulés) sur C
        _apply_fixed_bc_inplace_damping(C)

        # Information de debug (single on/off)
        if dbg.is_enabled():
            dbg.dprint("CL d'amortissement appliquées à C (lignes/colonnes de bord annulées)")
    
    # Returne M, K, C globales
    return M, K, C    

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
    M, K, C = _assemble_system_matrices_nonuniform(
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
        dx_vector = [d/1000.0 for d in dxs_mm]    # Convert les dx en mm
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