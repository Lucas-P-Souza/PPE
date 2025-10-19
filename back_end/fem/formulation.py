# Formulation FEM 1D pour corde tendue (maillage par frettes — non-uniform)
#
# Résumé
# -------
# Ce module fournit routines fiables et testées pour assembler les matrices
# globales nécessaires à l'intégration temporelle de la corde :
#   - M : matrice de masse (global)
#   - K : matrice de raideur (global)
#   - C : matrice d'amortissement de Rayleigh (C = α M + β K)
#
# Portée et conventions
# ---------------------
# - Ce code attend une discrétisation « frettes » (NON-uniforme) fournie via
#   un vecteur de longueurs d'éléments (dx_vector) en mètres.
# - n_nodes = nombre total de nœuds
# - n_elems = n_nodes - 1
# - l'élément e relie les nœuds e et e+1
#
# Validation
# ----------
# - Tous les dx doivent être strictement positifs
# - n_nodes >= 2
#
# API publique principale
# -----------------------
# build_global_mkc_from_config(apply_fixed_bc=False, return_meta=True)

from __future__ import annotations
from typing import Sequence, Tuple, Dict, Any

import warnings
import numpy as np

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
                ENABLED = False

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
# Surface publique stable
# ----------------------
# Ici la liste (`__all__`) définit la surface publique stable du module.
# ---------------------------------------------------------------------------
__all__ = [
    "amortissement_rayleigh",
    "build_global_mkc_from_config",
    "build_node_positions_from_config",
]

# Détecte les DDL contraints de type Dirichlet selon la convention du projet :
# - lignes/colonnes nulles hors diagonale
# - diagonale ≈ 1 dans M et K aux nœuds extrêmes (préserve l'inversibilité)
# NOTE : on utilise ça pour eviter de recalculer les modes sur les DDL contraints
def _detecter_ddl_contraints(
    M: np.ndarray, 
    K: np.ndarray,         
    atol: float = 1e-12) -> np.ndarray:

    # .shape[0] retourne le nombre de lignes (ou colonnes)
    n = M.shape[0]

    # Liste des DDL contraints
    constrained: list[int] = []

    # Boucle sur tous les DDL pour détecter ceux qui sont contraints
    for i in range(n):

        # Extraire lignes/colonnes i de M et K, en mettant à zéro la diagonale
        # .copy pour éviter de modifier M/K originales
        rowM = M[i, :].copy()
        rowM[i] = 0.0
        colM = M[:, i].copy()
        colM[i] = 0.0
        rowK = K[i, :].copy()
        rowK[i] = 0.0
        colK = K[:, i].copy()
        colK[i] = 0.0

        # Vérification des conditions de contrainte
        # - lignes/colonnes nulles hors diagonale
        # - diagonale ≈ 1 dans M et K aux nœuds extrêmes (préserve l'inversibilité)
        # .all() pour vérifier que toutes les conditions sont remplies
        # .abs() pour valeurs absolues
        if (
            np.all(np.abs(rowM) <= atol) and np.all(np.abs(colM) <= atol)
            and np.all(np.abs(rowK) <= atol) and np.all(np.abs(colK) <= atol)
            and abs(M[i, i] - 1.0) <= 1e-9 and abs(K[i, i] - 1.0) <= 1e-9
        ):
            # DDL i est contraint
            constrained.append(i)

    # Conversion en tableau numpy pour la sortie
    return np.asarray(constrained, dtype=int)

# Applique les CL fixes in-place (Dirichlet) :
# - annule les lignes/colonnes des extrémités; impose diag=1 sur ces nœuds
# NOTE : ça function c'est utilise JUSTE pour M et K
def _appliquer_cl_fixes_en_place(mat: np.ndarray) -> None:

    # Manipulation des lignes/colonnes
    mat[0, :] = 0.0     # ligne 0 = 0
    mat[-1, :] = 0.0    # ligne -1 = 0
    mat[:, 0] = 0.0     # colonne 0 = 0
    mat[:, -1] = 0.0    # colonne -1 = 0

    # Manipulation des diagonales
    mat[0, 0] = 1.0     # diag 0,0 = 1
    mat[-1, -1] = 1.0   # diag -1,-1 = 1

# Applique les CL d'amortissement in-place (bords annulés) sur C :
# - annule les lignes/colonnes aux extrémités
# - NE PAS mettre diag=1 (pas de sens physique pour C)
# NOTE : ça function c'est utilise JUSTE pour C
def _appliquer_cl_amortissement_en_place(mat: np.ndarray) -> None:

    # Manipulation des lignes/colonnes
    mat[0, :] = 0.0     # ligne 0 = 0
    mat[-1, :] = 0.0    # ligne -1 = 0
    mat[:, 0] = 0.0     # colonne 0 = 0
    mat[:, -1] = 0.0    # colonne -1 = 0

# Calcul des paramètres d'amortissement de Rayleigh (α, β) à partir de deux 
#   amortissements modaux cibles (ζ).
def amortissement_rayleigh(
    M: np.ndarray,
    K: np.ndarray,
    modes_ref: tuple[int, int],
    zetas_ref: tuple[float, float],
):
    
    # Détecte les DDL contraints pour exclure ces DDL du calcul modal
    constrained = _detecter_ddl_contraints(M, K)

    # .size returne le nombre d'éléments, si > 0 on extrait les DDL libres
    if constrained.size > 0:

        # .setdiff1d pour obtenir les DDL libres
        # .arrange pour générer un tableau d'indices
        # .shape[0] returne le nombre de lignes (ou colonnes)
        free = np.setdiff1d(np.arange(M.shape[0]), constrained)

        # .ix_ pour extraire les sous-matrices
        M_eval = M[np.ix_(free, free)]
        K_eval = K[np.ix_(free, free)]
    
    # si pas de DDL contraints, utilise M/K complets
    else:
        M_eval, K_eval = M, K

    # .linalg.solve pour résoudre le système d'équations
    # A = M^(-1) K
    A = np.linalg.solve(M_eval, K_eval)

    # .linalg.eig pour obtenir les valeurs propres/modes
    eigvals, _ = np.linalg.eig(A)

    # .real pour éviter les petites parties imaginaires numériques
    eigvals = np.real(eigvals)

    # .abs pour eliminer les négatives par erreur numérique
    eigvals[eigvals < 0] = 0.0

    # Calcul des fréquences naturelles et tri croissant
    omegas = np.sqrt(eigvals)           # rad/s
    omegas.sort()                       

    p, q = modes_ref                    # indices des modes de référence

    # Validation des entrées de référence
    # - indices valides et distincts
    # - fréquences strictement positives
    if p < 0 or q < 0 or p >= omegas.size or q >= omegas.size or p == q:
        raise ValueError("modes_ref invalides par rapport au nombre de modes disponibles")
    
    # Extraction des amortissements modaux cibles
    zeta_p, zeta_q = zetas_ref                              # amortissements modaux cibles
    omega_p, omega_q = float(omegas[p]), float(omegas[q])   # rad/s

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

    # Tentative de résolution directe
    try:
        # Résolution du système linéaire
        # .linalg.solve pour résoudre [α, β] = A_sys^(-1) b_sys
        alpha_lin, beta_lin = np.linalg.solve(A_sys, b_sys)
    
    # Gestion des cas singuliers
    except np.linalg.LinAlgError:
        # Système singulier (modes trop proches)
        denom = (omega_p**2 - omega_q**2)

        # Verification de la stabilité numérique
        if abs(denom) < 1e-16:
            raise ValueError("Fréquences de référence très proches: impossible de déterminer α et β de façon stable")
        
        # Calcul alternatif de α et β
        beta_lin = 2.0 * (zeta_p * omega_p - zeta_q * omega_q) / denom  
        alpha_lin = 2.0 * zeta_p * omega_p - beta_lin * (omega_p**2)    

    # Utilise alpha_lin/beta_lin calculés (quelque soit le chemin)
    alpha = float(alpha_lin)            # [1/s]
    beta = float(beta_lin)              # [s]
    C = alpha * M + beta * K            # matrice d'amortissement globale

    # Information de debug (single on/off)
    if getattr(dbg, "rayleigh_summary_enabled", None) and dbg.rayleigh_summary_enabled():
        try:
            dbg.print_rayleigh_explained(alpha, beta, omegas, modes_ref, zetas_ref, print_once=True)
        except Exception as _e_dbg:
            dbg.dprint(f"[AVERTISSEMENT] Échec d'impression Rayleigh: {_e_dbg}")

    return float(alpha), float(beta), C, omegas

# Assemble M et K, puis calcule C = α M + β K pour une maille NON uniforme.
def _assemble_system_matrices_nonuniform(
    dx_vector: Sequence[float] | np.ndarray,
    tension: float,
    lin_density: float,
    *,
    damping_modes_ref: tuple[int, int] | None = None,
    damping_zetas_ref: tuple[float, float] | None = None,
    apply_fixed_bc: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # .asarray pour accepter les listes et tuples
    # dtype=float pour s'assurer que c'est du float64 (par défaut) pour la stabilité numérique
    dx_arr = np.asarray(dx_vector, dtype=float)

    # On veut utiliser ces deux propriétés pour valider l'entrée
    # On veut un vecteur 1D non vide
    # .ndim returne le nombre de dimensions (1D, 2D, etc.)
    # .size returne le nombre d'éléments
    if dx_arr.ndim != 1 or dx_arr.size == 0:
        raise ValueError("dx_vector doit être un vecteur 1D non vide")
    
    # Verification des longueurs d'éléments
    # .any pour vérifier que tous les éléments sont > 0
    if np.any(dx_arr <= 0):
        raise ValueError("Toutes les longueurs d'élément doivent être positives")
    
    # Obtention du nombre d'éléments et de nœuds
    # .size returne le nombre d'éléments
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
        
    # Vérification si les paramètres d'amortissement sont fournis
    if damping_modes_ref is not None and damping_zetas_ref is not None:
        
        # Verification de l'application des CL fixes
        if apply_fixed_bc:

            # .copy pour éviter de modifier M/K originales
            M_eval = M.copy()                       # copie pour ne pas modifier M/K originaux 
            K_eval = K.copy()                       # copie pour ne pas modifier M/K originaux
            
            # Appliquer les CL fixes in-place sur les copies
            _appliquer_cl_fixes_en_place(M_eval)    # applique CL fixes in-place
            _appliquer_cl_fixes_en_place(K_eval)    # calcule α, β sur la version contrainte

            # Appeler la fonction d'amortissement de Rayleigh pour obtenir α et β
            alpha, beta, _, _ = amortissement_rayleigh(M_eval, K_eval, damping_modes_ref, damping_zetas_ref)
        
        # Si pas de CL fixes, utilise M/K complets
        else:
            alpha, beta, _, _ = amortissement_rayleigh(M, K, damping_modes_ref, damping_zetas_ref)

        # Construire la matrice d'amortissement C à partir des coefficients α et β.
        C = alpha * M + beta * K
    
    # Si il n'y a pas de paramètres d'amortissement, C = 0
    else:
        # .zeros_like pour créer une matrice de même forme que M/K avec zéros
        C = np.zeros_like(M)

    # Si demandé, appliquer les conditions aux limites -> M/K avec diag=1, C avec bords annulés (sans 1)
    if apply_fixed_bc:

        # Applique CL fixes in-place (Dirichlet) sur M et K
        _appliquer_cl_fixes_en_place(M)
        _appliquer_cl_fixes_en_place(K)
        
        # Applique CL d'amortissement in-place (bords annulés) sur C
        _appliquer_cl_amortissement_en_place(C)

        # Information de debug (single on/off)
        if getattr(dbg, 'ENABLED', False):
            dbg.dprint("CL d'amortissement appliquées à C (lignes/colonnes de bord annulées)")
    
    # Returne M, K, C globales
    return M, K, C    

# Construire M, K, C globaux à partir de la configuration (config.py)
def build_global_mkc_from_config(apply_fixed_bc: bool = False, return_meta: bool = True):
    
    # Importer le module config (avec gestion des chemins)
    try:
        # Preferred: package-relative import
        from .. import config  # type: ignore
    except Exception:
        try:
            # Absolute import fallback
            from digital_twin.back_end import config  # type: ignore
        except Exception:
            # Fallback: add project root to sys.path and try again
            try:
                import sys
                from pathlib import Path
                ROOT = Path(__file__).resolve().parents[3]
                if str(ROOT) not in sys.path:
                    sys.path.insert(0, str(ROOT))
                from digital_twin.back_end import config  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("Impossible d'importer le module config dans build_global_mkc_from_config") from exc
            
    # Récupération du vecteur de dx (en m) depuis config
    dxs_m = getattr(config, "FRET_DXS_M", None)

    # Validation de l'existence du vecteur de dx
    if dxs_m is None:
        raise RuntimeError("[ERROR] FRET_DXS_M manquant dans config")

    # Conversion en tableau numpy pour validation si il est bien 1D non vide
    dxs_arr = np.asarray(dxs_m, dtype=float)
    if dxs_arr.ndim != 1 or dxs_arr.size == 0:
        raise RuntimeError("[ERROR] FRET_DXS_M doit être un vecteur 1D non-vide")

    # Vérification de la finitude et de la positivité stricte des dx
    if not np.all(np.isfinite(dxs_arr)) or np.any(dxs_arr <= 0.0):
        raise RuntimeError("[ERROR] FRET_DXS_M contient des valeurs non-finites ou non-positives")

    # Récupérer aussi les cibles d'amortissement dans config (strict, pas de défauts)
    try:
        modes_ref = config.DAMPING_MODES_REF  # type: ignore[attr-defined]
        zetas_ref = config.DAMPING_ZETAS_REF  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("[ERROR] DAMPING_MODES_REF ou DAMPING_ZETAS_REF manquants dans config") from exc
    if modes_ref is None or zetas_ref is None:
        raise RuntimeError("[ERROR] DAMPING_MODES_REF/DAMPING_ZETAS_REF invalides dans config")

    # Assembler M, K, C via la fonction interne
    M, K, C = _assemble_system_matrices_nonuniform(
        dx_vector=dxs_m,
        tension=config.T,
        lin_density=config.MU,
        damping_modes_ref=modes_ref,
        damping_zetas_ref=zetas_ref,
        apply_fixed_bc=apply_fixed_bc,
    )

    # Construire les métadonnées
    dx_arr = np.asarray(dxs_m, dtype=float)
    meta = {
        'mode': 'nonuniform',
        'n_nodes': dx_arr.size + 1,
        'n_elems': dx_arr.size,
        'dx_min': float(dx_arr.min()),
        'dx_max': float(dx_arr.max()),
        'length_recon': float(dx_arr.sum()),
        'mass_total': float(config.MU * dx_arr.sum()),
    }

    # Si demandé, calculer et ajouter α, β et fréquences aux métadonnées
    if return_meta:
        # Ajouter α, β et fréquences si paramètres fournis
        try:
            alpha, beta, _, omegas = amortissement_rayleigh(M, K, modes_ref, zetas_ref)
            meta.update({'alpha': alpha, 'beta': beta, 'omegas': omegas})
        except Exception:
            pass
    result = (M, K, C, meta) if return_meta else (M, K, C)

    return result
    
def build_node_positions_from_config(n_nodes: int):
    """Construire des coordonnées x à partir du module `config`.

    Comportement:
    - Si `FRET_DXS_MM` est présent et sa longueur == n_nodes-1, l'utiliser (converti en mètres).
    - Sinon, utiliser `L` (si présent) et construire un maillage uniforme [0, L].
    - En dernier recours, renvoyer un maillage uniforme sur [0, 1].
    """
    # Importer config de façon robuste et l'exposer localement comme _cfg
    try:
        from .. import config as _cfg  # type: ignore
    except Exception:
        try:
            from digital_twin.back_end import config as _cfg  # type: ignore
        except Exception:
            # repli : uniforme [0, 1]
            return np.linspace(0.0, 1.0, int(n_nodes))

    # essayer FRET_DXS_MM (mm -> m)
    dxs_mm = getattr(_cfg, "FRET_DXS_MM", None)
    try:
        if dxs_mm and len(dxs_mm) == n_nodes - 1:
            dxs_m = np.asarray(dxs_mm, dtype=float) / 1000.0
            x = np.concatenate([[0.0], np.cumsum(dxs_m)])
            return x
    except Exception:
        # en cas d'erreur de conversion, continuer vers le repli
        pass

    # repli: utiliser la longueur L si fournie, sinon unité [0,1]
    L = float(getattr(_cfg, "L", 1.0))
    return np.linspace(0.0, L, int(n_nodes))

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
    M, K, C, meta = build_global_mkc_from_config(apply_fixed_bc=apply_bc, return_meta=True)
    # Diagnostics (silencieux quando debug desativado)
    if hasattr(dbg, "print_formulation_diagnostics"):
        dbg.print_formulation_diagnostics(M, K, C, meta=meta, modes_ref=modes_ref, zetas_ref=zetas_ref, config=config)