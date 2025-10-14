"""
Intégration temporelle et conditions initiales pour la corde vibrante (FEM).

Ce module regroupe UNIQUEMENT l'API canonique en français:
 - l'intégrateur de Newmark-beta (β=1/4, γ=1/2, stable inconditionnel pour systèmes linéaires),
 - les conditions initiales (pincement triangulaire) et un calcul du premier pas (diagnostic),
 - les paramètres de simulation et le calcul d'énergies au cours du temps,
 - un fournisseur de force nulle (F(t) ≡ 0).

Remarque: un utilitaire pour détecter les DDL contraints est importé depuis fem/modal.
"""
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
            @staticmethod
            def is_enabled() -> bool: return False
            @staticmethod
            def dprint(*args, **kwargs): pass
            @staticmethod
            def get_solver_sample_interval(n_steps: int) -> int:
                try:
                    return max(1, int(round(n_steps / 50)))
                except Exception:
                    return 100
            @staticmethod
            def print_solver_setup_summary(*args, **kwargs): pass
            @staticmethod
            def print_newmark_constants(*args, **kwargs): pass
            @staticmethod
            def print_step_snapshot(*args, **kwargs): pass
            @staticmethod
            def print_energy_start_end(*args, **kwargs): pass
        dbg = _DbgNoOp()  # type: ignore


# Détection des GL (DDLs) contraints (import depuis fem.modal)
try:
    from .modal import detect_constrained_dofs_mk  # type: ignore
except Exception:
    # Fallback local minimal si l'import échoue (devrait rarement arriver)
    def detect_constrained_dofs_mk(M: np.ndarray, K: np.ndarray, atol: float = 1e-12) -> np.ndarray:
        """Détecte les DDLs contraints de type Dirichlet selon la convention du projet.
        Critère utilisé: lignes/colonnes ≈ 0 hors diagonale et diag ≈ 1 aux extrémités.
        """
        n = M.shape[0]
        constrained = []
        for i in range(n):
            rowM = M[i, :].copy(); rowM[i] = 0.0
            colM = M[:, i].copy(); colM[i] = 0.0
            rowK = K[i, :].copy(); rowK[i] = 0.0
            colK = K[:, i].copy(); colK[i] = 0.0
            if (
                np.all(np.abs(rowM) <= atol) and np.all(np.abs(colM) <= atol)
                and np.all(np.abs(rowK) <= atol) and np.all(np.abs(colK) <= atol)
                and np.isclose(M[i, i], 1.0, atol=1e-9) and np.isclose(K[i, i], 1.0, atol=1e-9)
            ):
                constrained.append(i)
        return np.asarray(constrained, dtype=int)

def definir_parametres_simulation(delta_t: float, T_total: float):
    """Définit les paramètres temporels de simulation.

    Entrées
    - delta_t: pas de temps (s), doit être > 0
    - T_total: durée totale (s), doit être > 0

    Sorties
    - (delta_t, T_total, n_pas, inv_dt_carre, inv_2dt)

    Détails:
    - n_pas = arrondi(T_total / delta_t)
    - inv_dt_carre = 1 / (delta_t^2)  # accélère les calculs
    - inv_2dt = 1 / (2 delta_t)
    """
    if delta_t <= 0.0:
        raise ValueError("delta_t deve ser > 0")
    if T_total <= 0.0:
        raise ValueError("T_total deve ser > 0")
    n_pas = int(round(T_total / delta_t))
    inv_dt_carre = 1.0 / (delta_t * delta_t)
    inv_2dt = 1.0 / (2.0 * delta_t)
    if dbg.is_enabled():
        dbg.dprint(f"n_pas = {n_pas}")
    return delta_t, T_total, n_pas, inv_dt_carre, inv_2dt


def initialiser_u0_triangle(M: np.ndarray, *, L: float, h: float, x_p: float) -> np.ndarray:
    """Construit un déplacement initial U0 triangulaire (pincement).

    Paramètres
    - M: matrice de masse (N,N), utilisée uniquement pour N = M.shape[0]
    - L: longueur (m)
    - h: amplitude du pincement (m)
    - x_p: position du pincement (m)

    Retour
    - U0: vecteur (N,) des déplacements initiaux.

    Détails et formules:
    - Discrétisation uniforme implicite via N points: Δx = L/(N-1), x_i = i·Δx
    - U0(x) = h·(x/x_p) pour x ≤ x_p ; U0(x) = h·((L - x)/(L - x_p)) pour x > x_p
    """
    N = int(M.shape[0])
    if N < 2:
        raise ValueError("N deve ser >= 2 para inicializar U0")
    if h is None or h <= 0.0:
        return np.zeros(N, dtype=float)

    L = float(L)
    x_p = float(x_p)
    delta_x = L / float(N - 1)
    # np.arange: crée [0, 1, ..., N-1] et multiplie par Δx pour obtenir les abscisses
    x = np.arange(N, dtype=float) * delta_x

    # Vecteur gauche (croissant) et droit (décroissant) de la forme triangulaire
    left = h * (x / x_p) if x_p > 0.0 else np.zeros_like(x)
    denom_right = (L - x_p) if (L - x_p) > 0.0 else 1.0
    right = h * ((L - x) / denom_right)
    # np.where: choisit left si (x <= x_p) sinon right
    U0 = np.where(x <= x_p, left, right)
    return U0.astype(float, copy=False)


def initialiser_etats_initiaux(M: np.ndarray, *, L: float, h: float, x_p: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne le triplet de CI: (U0, U_nm1, U_n).

    - U0: forme triangulaire
    - U_nm1 = U0 (état à t=-Δt pour schémas à 2 pas)
    - U_n = U0 (état à t=0)
    """
    U0 = initialiser_u0_triangle(M, L=L, h=h, x_p=x_p)
    U_nm1 = U0.copy()
    U_n = U0.copy()
    return U0, U_nm1, U_n


def calculer_u1(M: np.ndarray, C: np.ndarray, K: np.ndarray,
                U_n: np.ndarray, U_nm1: np.ndarray, delta_t: float) -> np.ndarray:
    """Calcule le premier pas U1 par différences centrées (diagnostic).

    Formulation (discrétisation classique):
    - A = M/Δt^2 + C/(2Δt) + K
    - rhs = (2M/Δt^2) U_n − (M/Δt^2 − C/(2Δt)) U_{n−1}
    - U_1 = A^{-1} rhs, résolu par numpy.linalg.solve
    """
    if M.ndim != 2 or C.ndim != 2 or K.ndim != 2:
        raise ValueError("M, C, K devem ser 2D (matrizes)")
    if M.shape[0] != M.shape[1] or C.shape[0] != C.shape[1] or K.shape[0] != K.shape[1]:
        raise ValueError("M, C, K devem ser quadradas")
    if not (M.shape == C.shape == K.shape):
        raise ValueError(f"Dimensões diferentes: M={M.shape}, C={C.shape}, K={K.shape}")
    n = M.shape[0]
    if U_n.shape[0] != n or U_nm1.shape[0] != n:
        raise ValueError(f"U_n/U_nm1 incompatíveis com n={n}")
    if delta_t <= 0.0:
        raise ValueError("delta_t deve ser > 0")

    dt = float(delta_t)
    inv_dt2 = 1.0 / (dt * dt)   # 1/Δt^2
    inv_2dt = 1.0 / (2.0 * dt)  # 1/(2Δt)

    # A = M/Δt^2 + C/(2Δt) + K
    A = (M * inv_dt2) + (C * inv_2dt) + K
    # rhs = (2M/Δt^2) U_n − (M/Δt^2 − C/(2Δt)) U_{n−1}
    rhs = (2.0 * M * inv_dt2) @ U_n - ((M * inv_dt2) - (C * inv_2dt)) @ U_nm1
    # np.linalg.solve: résout A·x = rhs (système linéaire)
    U1 = np.linalg.solve(A, rhs)
    if dbg.is_enabled():
        dbg.dprint(f"U1: shape={U1.shape}, max={np.nanmax(U1):.3e}, min={np.nanmin(U1):.3e}")
    return U1


def fournisseur_force_nulle(n: int):
    """Retourne F(t,k) ≡ 0 (vecteur de taille n)."""
    def F_zero(_t: float, _k: int) -> np.ndarray:
        # np.zeros: crée un vecteur de zéros de taille n
        return np.zeros(n, dtype=float)
    return F_zero


def fournisseur_force_localisee(
    N: int,
    i_force: int,
    F_max: float,
    t_rise: float,
    t_hold: float,
    t_decay: float,
    t0: float = 0.0,
):
    """Génère une force externe localisée (noeud unique) avec enveloppe trapézoïdale.

    Paramètres
    - N: taille du système (nombre total de DDL)
    - i_force: indice du nœud où appliquer la force (0 ≤ i_force < N)
    - F_max: amplitude maximale (Newtons)
    - t_rise: durée de montée linéaire jusqu'à F_max (s)
    - t_hold: durée de maintien à F_max (s)
    - t_decay: durée de décroissance linéaire vers 0 (s)
    - t0: instant de début de l'enveloppe (décalage temporel, s)

    Retour
    - Une fonction F(t, k) → ndarray shape (N,), appliquant la force sur i_force.

    Enveloppe (t >= 0):
      1) 0 → F_max en t_rise (rampe linéaire)
      2) palier F_max pendant t_hold
      3) F_max → 0 en t_decay (rampe linéaire)
      sinon: 0

    Remarques
    - Les durées nulles sont gérées: si t_rise == 0, le signal saute directement à F_max; idem pour t_decay.
    - Si i_force est hors bornes, la force sera ignorée (vecteur nul) par sécurité.
    """
    i_force = int(i_force)
    t_rise = float(max(0.0, t_rise))
    t_hold = float(max(0.0, t_hold))
    t_decay = float(max(0.0, t_decay))
    F_max = float(F_max)

    t0 = float(t0)
    t1 = t_rise
    t2 = t_rise + t_hold
    t3 = t_rise + t_hold + t_decay

    def _amp(t: float) -> float:
        # Temps relatif à l'instant de début t0
        tr = t - t0
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

    def F(t: float, _k: int) -> np.ndarray:
        f = np.zeros(N, dtype=float)
        if 0 <= i_force < N:
            f[i_force] = _amp(float(t))
        return f

    return F


def somme_de_forces(N: int, *Fs: Callable[[float, int], np.ndarray]):
    """Combine plusieurs fournisseurs de force F_i(t,k) en une seule force somme.

    Paramètres
    - N: taille du système (longueur des vecteurs retournés)
    - *Fs: une ou plusieurs fonctions F_i(t,k) → ndarray shape (N,)

    Retour
    - F_sum(t,k) = Σ_i F_i(t,k)

    Remarques
    - Les sorties sont additionnées élément par élément. Aucune saturation n'est appliquée.
    - On s'attend à ce que chaque F_i retourne un vecteur de taille N.
    """
    def F_sum(t: float, k: int) -> np.ndarray:
        out = np.zeros(N, dtype=float)
        for f in Fs:
            vec = np.asarray(f(t, k), dtype=float)
            if vec.shape[0] == N:
                out += vec
            else:
                # Sécurité: ignorons les vecteurs mal dimensionnés
                pass
        return out
    return F_sum


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
    """Intègre M u¨ + C u˙ + K u = F(t) par Newmark-beta (β=1/4, γ=1/2) sur les DDL libres.

    Retourne (t, U, V, A) sur l'espace complet (DDLs contraints maintenus à 0).

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
    if dt <= 0 or t_max <= 0:
        raise ValueError("dt e t_max devem ser > 0")
    if M.shape != C.shape or M.shape != K.shape or M.ndim != 2:
        raise ValueError("M, C, K devem ser quadradas do mesmo tamanho")

    n = M.shape[0]
    n_steps = int(np.floor(t_max / dt)) + 1
    t = np.linspace(0.0, dt * (n_steps - 1), n_steps)

    constrained = detect_constrained_dofs_mk(M, K)
    free = np.setdiff1d(np.arange(n), constrained)
    if free.size == 0:
        raise ValueError("Nenhum GL livre detectado para integração Newmark")

    Mff = M[np.ix_(free, free)]
    Cff = C[np.ix_(free, free)]
    Kff = K[np.ix_(free, free)]

    U0f = np.zeros(free.size, dtype=float) if U0 is None else np.asarray(U0, dtype=float)[free]
    V0f = np.zeros(free.size, dtype=float) if V0 is None else np.asarray(V0, dtype=float)[free]
    if A0 is None:
        if callable(F):
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

    if dbg.is_enabled():
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

        if dbg.is_enabled() and ((k + 1) % sample == 0 or k == 0):
            try:
                dbg.print_step_snapshot(k + 1, t[k + 1], U[:, k + 1], V[:, k + 1], A[:, k + 1], M=M, K=K)
            except Exception:
                pass

    if dbg.is_enabled():
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
