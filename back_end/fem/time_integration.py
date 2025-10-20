# Intégration temporelle et conditions initiales pour la corde vibrante (FEM).
# Ce module regroupe UNIQUEMENT:
# - l'intégrateur de Newmark-beta (β=1/4, γ=1/2, stable inconditionnel pour systèmes linéaires),
# - les conditions initiales (pincement triangulaire) et un calcul du premier pas (diagnostic),
# - les paramètres de simulation et le calcul d'énergies au cours du temps,
# - un fournisseur de force nulle (F(t) ≡ 0).

from __future__ import annotations

import numpy as np
from typing import Any, Callable

# Debug import (safe fallback)
try:
    from ..utils import debug as dbg  # type: ignore
except Exception:
    try:
        from digital_twin.back_end.utils import debug as dbg  # type: ignore
    except Exception:
        class _DbgNoOp:
            # provide attribute-based API compatibility
            ENABLED = False
            # legacy per-print flags removed; rely on ENABLED

            @staticmethod
            def dprint(*args, **kwargs):
                pass

            @staticmethod
            def get_solver_sample_interval(n_steps: int) -> int:
                try:
                    return max(1, int(round(n_steps / 50)))
                except Exception:
                    return 100

            @staticmethod
            def print_solver_setup_summary(*args, **kwargs):
                pass

            @staticmethod
            def print_newmark_constants(*args, **kwargs):
                pass

            @staticmethod
            def print_step_snapshot(*args, **kwargs):
                pass

            @staticmethod
            def print_energy_start_end(*args, **kwargs):
                pass

        dbg = _DbgNoOp()  # type: ignore


# Détection des GL (DDLs) contraints — implémentation locale (maintenue ici)
def detecter_ddl_contraints_mk(M: np.ndarray, K: np.ndarray, atol: float = 1e-12) -> np.ndarray:

    # .shape[0] : nombre de DDLs
    n = M.shape[0]

    # Matrices auxiliaires pour les critères
    constrained = []

    # Itérer sur chaque DDL et vérifier les conditions de contrainte
    for i in range(n):

        # Lignes/colonnes hors diagonale ≈ 0
        # .copy() pour éviter de modifier M/K originales
        rowM = M[i, :].copy(); rowM[i] = 0.0
        colM = M[:, i].copy(); colM[i] = 0.0
        rowK = K[i, :].copy(); rowK[i] = 0.0
        colK = K[:, i].copy(); colK[i] = 0.0

        # Vérification des critères de contrainte
        # - lignes/colonnes hors diag ≈ 0
        # - diag ≈ 1.0
        # - M[i, i] ≈ 1.0
        # - K[i, i] ≈ 1.0
        # .isclose pour comparaisons avec tolérance
        if (
            np.all(np.abs(rowM) <= atol) and np.all(np.abs(colM) <= atol)
            and np.all(np.abs(rowK) <= atol) and np.all(np.abs(colK) <= atol)
            and np.isclose(M[i, i], 1.0, atol=1e-9) and np.isclose(K[i, i], 1.0, atol=1e-9)
        ):
            # DDL i est contraint
            constrained.append(i)

    # Retourner les indices contraints sous forme de numpy.ndarray
    # .asarray pour conversion de liste Python à ndarray
    return np.asarray(constrained, dtype=int)

# Définit les paramètres temporels de simulation.
def definir_parametres_simulation(delta_t: float, T_total: float):

    # Validations simples des entrées
    if delta_t <= 0.0:  # Temps de pas invalide
        raise ValueError("delta_t doit être > 0")
    if T_total <= 0.0:  # Durée totale invalide
        raise ValueError("T_total doit être > 0")

    # Calcul des paramètres dérivés
    # round() pour éviter les imprécisions flottantes
    n_pas = int(round(T_total / delta_t))       # nombre de pas de temps
    inv_dt_carre = 1.0 / (delta_t * delta_t)    # 1/Δt²
    inv_2dt = 1.0 / (2.0 * delta_t)             # 1/(2Δt)

    # Debug print
    if getattr(dbg, 'ENABLED', False):  
        dbg.dprint(f"n_pas = {n_pas}")

    return delta_t, T_total, n_pas, inv_dt_carre, inv_2dt

# Calcule le premier pas U1 par différences centrées (diagnostic).
def calculer_u1(M: np.ndarray, C: np.ndarray, K: np.ndarray,
                U_n: np.ndarray, U_nm1: np.ndarray, delta_t: float) -> np.ndarray:
    """
    Formulation (discrétisation classique):
    - A = M/Δt^2 + C/(2Δt) + K
    - rhs = (2M/Δt^2) U_n − (M/Δt^2 − C/(2Δt)) U_{n−1}
    - U_1 = A^{-1} rhs, résolu par numpy.linalg.solve
    """

    # Validations simples des entrées
    # Verifie que M, C, K sont 2D
    if M.ndim != 2 or C.ndim != 2 or K.ndim != 2:
        raise ValueError("M, C, K doivent être 2D (matrices)")
    
    # Vérifie que M, C, K sont carrées
    if M.shape[0] != M.shape[1] or C.shape[0] != C.shape[1] or K.shape[0] != K.shape[1]:
        raise ValueError("M, C, K doivent être carrées")
    
    # Vérifie que M, C, K ont la même forme
    if not (M.shape == C.shape == K.shape):
        raise ValueError(f"Dimensions différentes : M={M.shape}, C={C.shape}, K={K.shape}")
    
    # Vérifie la compatibilité des vecteurs U_n, U_nm1
    if U_n.shape[0] != M.shape[0] or U_nm1.shape[0] != M.shape[0]:
        raise ValueError(f"U_n / U_nm1 incompatibles avec n={M.shape[0]}")
    
    # Vérifie delta_t > 0
    if delta_t <= 0.0:
        raise ValueError("delta_t doit être > 0")

    # Calcul des constantes
    dt = float(delta_t)         # Assure que dt est float
    inv_dt2 = 1.0 / (dt * dt)   # 1/Δt^2
    inv_2dt = 1.0 / (2.0 * dt)  # 1/(2Δt)

    # Formulation matricielle
    # A = M/Δt^2 + C/(2Δt) + K
    A = (M * inv_dt2) + (C * inv_2dt) + K
    
    # Calcul du second membre
    # rhs = (2M/Δt^2) U_n − (M/Δt^2 − C/(2Δt)) U_{n−1}
    rhs = (2.0 * M * inv_dt2) @ U_n - ((M * inv_dt2) - (C * inv_2dt)) @ U_nm1
    
    # Résolution du système linéaire A U1 = rhs
    # np.linalg.solve: résout A·x = rhs (système linéaire)
    U1 = np.linalg.solve(A, rhs)

    # Debug print
    if getattr(dbg, 'ENABLED', False):
        dbg.dprint(f"U1: shape={U1.shape}, max={np.nanmax(U1):.3e}, min={np.nanmin(U1):.3e}")
    
    # Retour du résultat
    return U1

# Génère une force externe localisée (noeud unique) avec enveloppe trapézoïdale.
def fournisseur_force_localisee(
    N: int,
    i_force: int,
    F_max: float,
    t_rise: float,
    t_hold: float,
    t_decay: float,
    t0: float = 0.0,
):
    
    # Garantir les types et valeurs corrects
    i_force = int(i_force)
    t_rise = float(max(0.0, t_rise))
    t_hold = float(max(0.0, t_hold))
    t_decay = float(max(0.0, t_decay))
    F_max = float(F_max)
    t0 = float(t0)

    # Instants clés de l'enveloppe
    t1 = t_rise
    t2 = t_rise + t_hold
    t3 = t_rise + t_hold + t_decay

    # Définition de l'amplitude en fonction du temps
    def _amp(t: float) -> float:

        # Temps relatif à l'instant de début t0
        tr = t - t0
        
        # Calcul de l'amplitude selon l'enveloppe trapézoïdale
        if tr < 0.0:
            return 0.0
        if t1 > 0.0 and tr < t1:
            return F_max * (tr / t1)
        if t1 == 0.0 and tr < t2:
            # montée instantanée → palier direct
            return F_max
        if tr < t2:
            return F_max
        if tr < t3 and t_decay > 0.0:
            return F_max * (1.0 - (tr - t2) / t_decay)
        # après t3 (ou t_decay == 0): zéro
        return 0.0

    # Fournisseur de force F(t, k)
    def F(t: float, _k: int) -> np.ndarray:

        # Vecteur force initialisé à zéro
        f = np.zeros(N, dtype=float)

        # Appliquer l'amplitude au noeud i_force
        if 0 <= i_force < N:
            f[i_force] = _amp(float(t))
        return f

    return F

# Somme de plusieurs fournisseurs de force.
# La fonction externe fixe N et *Fs, et retourne une fonction F_sum(t, k).
def somme_de_forces(N: int, *Fs: Callable[[float, int], np.ndarray]):
    
    # Fonction interne F_sum(t, k)
    # Somme les contributions de chaque fournisseur dans Fs.
    def F_sum(t: float, k: int) -> np.ndarray:
        
        # Initialisation du vecteur de sortie avec zéros
        out = np.zeros(N, dtype=float)

        # Itération sur chaque fournisseur et sommation
        for f in Fs:

            # Récupération du vecteur force du fournisseur
            # .asarray pour assurer le type numpy.ndarray
            vec = np.asarray(f(t, k), dtype=float)

            # Vérification de la dimension avant sommation
            if vec.shape[0] == N:
                out += vec
            else:
                # Sécurité: ignorons les vecteurs mal dimensionnés
                pass

        # Retour du vecteur de sortie avec la somme des forces
        return out
    
    # Retour de la fonction somme
    return F_sum

# Intégrateur Newmark-beta (β=1/4, γ=1/2) pour M u¨ + C u˙ + K u = F(t).
def integrer_newmark_beta(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    F: np.ndarray | Callable[[float, int], np.ndarray],
    dt: float,
    t_max: float,
    U0: np.ndarray | None = None,
    V0: np.ndarray | None = None,
    A0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Étapes clés et formules:
    - Détection des DDLs contraints (Dirichlet) puis extraction des sous-matrices libres M_ff, C_ff, K_ff.
    - Si A0 non fournie, résolution A0_ff = M_ff^{-1}(F0_ff − C_ff V0_ff − K_ff U0_ff).
    - Coefficients (β=1/4, γ=1/2):
        a0 = 1/(β Δt^2), a1 = γ/(β Δt), a2 = 1/(β Δt), a3 = 1/(2β) − 1,
        a4 = γ/β − 1, a5 = Δt(γ/(2β) − 1)
    - Matrice efficace constante: K_eff = K_ff + a1 C_ff + a0 M_ff
    - Pour k→k+1: résoudre U_{k+1}^f dans
        K_eff U_{k+1}^f = F_{k+1}^f + M_ff(a0 U_k^f + a2 V_k^f + a3 A_k^f)
                                       + C_ff(a1 U_k^f + a4 V_k^f + a5 A_k^f)
      puis mettre à jour:
        A_{k+1}^f = a0(U_{k+1}^f − U_k^f) − a2 V_k^f − a3 A_k^f
        V_{k+1}^f = V_k^f + Δt[(1−γ)A_k^f + γ A_{k+1}^f]
    """

    # Vérifie que dt et t_max sont positifs
    if dt <= 0 or t_max <= 0:
        raise ValueError("dt et t_max doivent être > 0")
    
    # Vérifie que M, C, K sont carrées du même taille
    # .shape[0] pour nombre de DDLs
    # .ndim pour vérifier que ce sont des matrices 2D
    if M.shape != C.shape or M.shape != K.shape or M.ndim != 2 or C.ndim != 2 or K.ndim != 2:
        raise ValueError("M, C, K doivent être carrées du même taille et 2D")

    # Vérifie que F est soit une fonction, soit un tableau numpy
    # callable() pour fonctions
    # isinstance(..., np.ndarray) pour tableaux numpy
    if not (callable(F) or isinstance(F, np.ndarray)):
        raise ValueError("F doit être une fonction ou un tableau numpy")
    
    # Initialisations
    n = M.shape[0]                                      # Nombre de DDLs
    n_steps = int(np.floor(t_max / dt)) + 1             # Nombre de pas de temps
    t = np.linspace(0.0, dt * (n_steps - 1), n_steps)   # Vecteur temps

    # Détection des DDLs contraints et extraction des sous-matrices libres
    constrained = detecter_ddl_contraints_mk(M, K)      # Indices des DDLs contraints
    free = np.setdiff1d(np.arange(n), constrained)      # Indices des DDLs libres
    if free.size == 0:                                  # Vérification des DDLs libres
        raise ValueError("Aucun DDL libre détecté pour l'intégration Newmark")

    # Extraction des sous-matrices libres
    Mff = M[np.ix_(free, free)]                         # Sous-matrice libre de M
    Kff = K[np.ix_(free, free)]                         # Sous-matrice libre de K
    Cff = C[np.ix_(free, free)]                         # Sous-matrice libre de C

    # Conditions initiales pour DDLs libres
    U0f = np.zeros(free.size, dtype=float) if U0 is None else np.asarray(U0, dtype=float)[free]
    V0f = np.zeros(free.size, dtype=float) if V0 is None else np.asarray(V0, dtype=float)[free]
    if A0 is None:                                      # Calculer A0f si non fourni
        if callable(F):                                 # F est une fonction    
            F0_full = np.asarray(F(0.0, 0), dtype=float).reshape(n)
        else:
            F_arr = np.asarray(F, dtype=float)
            F0_full = F_arr[:, 0] if (F_arr.ndim == 2 and F_arr.shape[1] >= 1) else np.zeros(n)
        A0f = np.linalg.solve(Mff, F0_full[free] - Cff @ V0f - Kff @ U0f)
    else:
        A0f = np.asarray(A0, dtype=float)[free]

    beta = 1.0 / 4.0
    gamma = 1.0 / 2.0
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)

    if getattr(dbg, "ENABLED", False):
        try:
            dbg.print_newmark_constants(dt=dt, beta=beta, gamma=gamma, a0=a0, a1=a1, a2=a2, a3=a3, a4=a4, a5=a5)
        except Exception:
            pass

    K_eff = Kff + a1 * Cff + a0 * Mff

    U = np.zeros((n, n_steps), dtype=float)
    V = np.zeros((n, n_steps), dtype=float)
    A = np.zeros((n, n_steps), dtype=float)
    U[free, 0] = U0f
    V[free, 0] = V0f
    A[free, 0] = A0f
    if constrained.size:
        U[constrained, 0] = 0.0
        V[constrained, 0] = 0.0
        A[constrained, 0] = 0.0

    sample = dbg.get_solver_sample_interval(n_steps)
    fixed_every = 0
    try:
        fixed_every = int(getattr(dbg, 'get_fixed_step_interval', lambda: 0)() or 0)  # type: ignore[attr-defined]
    except Exception:
        fixed_every = 0
    for k in range(n_steps - 1):
        Uf_k = U[free, k]
        Vf_k = V[free, k]
        Af_k = A[free, k]

        if callable(F):
            Fk1_full = np.asarray(F(t[k + 1], k + 1), dtype=float).reshape(n)
            Fk1 = Fk1_full[free]
        else:
            F_arr = np.asarray(F, dtype=float)
            if F_arr.ndim == 2 and F_arr.shape[1] > k + 1:
                Fk1 = F_arr[free, k + 1]
            else:
                Fk1 = np.zeros(free.size, dtype=float)

        RHS = (
            Fk1
            + Mff @ (a0 * Uf_k + a2 * Vf_k + a3 * Af_k)
            + Cff @ (a1 * Uf_k + a4 * Vf_k + a5 * Af_k)
        )

        Uf_k1 = np.linalg.solve(K_eff, RHS)
        Af_k1 = a0 * (Uf_k1 - Uf_k) - a2 * Vf_k - a3 * Af_k
        Vf_k1 = Vf_k + dt * ((1.0 - gamma) * Af_k + gamma * Af_k1)

        U[free, k + 1] = Uf_k1
        V[free, k + 1] = Vf_k1
        A[free, k + 1] = Af_k1
        if constrained.size:
            U[constrained, k + 1] = 0.0
            V[constrained, k + 1] = 0.0
            A[constrained, k + 1] = 0.0

        should_print = False
        if getattr(dbg, 'ENABLED', False):
            if fixed_every and ((k + 1) % fixed_every == 0 or k == 0):
                should_print = True
            elif ((k + 1) % sample == 0 or k == 0):
                should_print = True
        if should_print:
            try:
                dbg.print_step_snapshot(k + 1, t[k + 1], U[:, k + 1], V[:, k + 1], A[:, k + 1], M=M, K=K)
            except Exception:
                pass

    if getattr(dbg, 'ENABLED', False):
        try:
            Ek, Ep, Et = calculer_energies_dans_le_temps(M, K, U, V)
            dbg.print_energy_start_end(t0=t[0], tn=t[-1], Ek0=Ek[0], Ep0=Ep[0], Et0=Et[0], Ekn=Ek[-1], Epn=Ep[-1], Etn=Et[-1])
        except Exception:
            pass

    return t, U, V, A


def calculer_energies_dans_le_temps(M: np.ndarray, K: np.ndarray, U: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcule énergie cinétique, potentielle et totale au cours du temps.

    Pour chaque pas k:
    - E_cin(k) = 1/2 · V_k^T (M V_k)
    - E_pot(k) = 1/2 · U_k^T (K U_k)
    - E_tot(k) = E_cin(k) + E_pot(k)
    """
    n_steps = U.shape[1]
    Ek = np.zeros(n_steps, dtype=float)
    Ep = np.zeros(n_steps, dtype=float)
    for k in range(n_steps):
        v = V[:, k]
        u = U[:, k]
        # produits matriciels/vecteurs via @ (NumPy: v.T @ (M @ v) = scalaire)
        Ek[k] = 0.5 * float(v.T @ (M @ v))
        Ep[k] = 0.5 * float(u.T @ (K @ u))
    Etot = Ek + Ep
    return Ek, Ep, Etot
