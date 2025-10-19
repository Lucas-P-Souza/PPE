from __future__ import annotations
"""
Événements de pression pour corde vibrante (FEM): ajoute une rigidité locale (et amortisseur
optionnel) sur des nœuds internes pendant des fenêtres de temps, sans modifier la masse.

API (inchangée):
- class PressEvent
- build_press_segments(...)
- simulate_with_press(...)

Internel: s'appuie sur l'intégrateur Newmark-β.
"""
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple, Dict, Optional
import numpy as np

try:
    # Intégrateur (FR)
    from digital_twin.back_end.fem.time_integration import integrer_newmark_beta  # type: ignore
except Exception:  # pragma: no cover
    try:
        from ..fem.time_integration import integrer_newmark_beta  # type: ignore
    except Exception:
        # Fallback: exécution directe sans contexte de package
        try:
            import sys
            from pathlib import Path
            ROOT = Path(__file__).resolve().parents[3]  # repo root contenant 'digital_twin'
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from digital_twin.back_end.fem.time_integration import integrer_newmark_beta  # type: ignore
        except Exception as exc:
            raise

# Debug minimal (optionnel)
try:
    from digital_twin.back_end.utils import debug as dbg  # type: ignore
except Exception:  # pragma: no cover
    class _DbgNoOp:
        ENABLED = False
        @staticmethod
        def dprint(*args, **kwargs): pass
    dbg = _DbgNoOp()  # type: ignore

DEBUG_PRESS: bool = True

# -------------------------------
# 1) Outils: matrices locales
# -------------------------------

def _valider_noeud_interne(n_noeuds: int, indice_noeud: int) -> None:
    if not (0 <= indice_noeud < n_noeuds):
        raise ValueError(f"indice_noeud hors limites: {indice_noeud} (n_noeuds={n_noeuds})")
    if indice_noeud in (0, n_noeuds - 1):
        raise ValueError("Interdit d'appuyer sur les nœuds d'extrémité (Dirichlet)")


def fabriquer_ressort_local(n_noeuds: int, indice_noeud: int, ks: float) -> np.ndarray:
    """Crée K_press avec rigidité locale ks au nœud j (diag += ks)."""
    _valider_noeud_interne(n_noeuds, indice_noeud)
    ks = float(ks)
    if ks <= 0.0:
        raise ValueError("ks doit être > 0")
    Kp = np.zeros((n_noeuds, n_noeuds), dtype=float)
    Kp[indice_noeud, indice_noeud] += ks
    return Kp


def fabriquer_amortisseur_local(n_noeuds: int, indice_noeud: int, cs: float) -> np.ndarray:
    """Crée C_press local (diag += cs) — optionnel."""
    _valider_noeud_interne(n_noeuds, indice_noeud)
    cs = float(cs)
    if cs <= 0.0:
        raise ValueError("cs doit être > 0")
    Cp = np.zeros((n_noeuds, n_noeuds), dtype=float)
    Cp[indice_noeud, indice_noeud] += cs
    return Cp

# -------------------------------
# 2) Événements et segments
# -------------------------------

@dataclass
class PressEvent:
    node: int
    t_on: float
    t_off: float
    ks: float
    cs: float = 0.0
    # Paramètres du profil polynomial symétrique (profil unique)
    t_montee: Optional[float] = None
    t_plateau: Optional[float] = None
    n_sous_segments: int = 20  # subdivision pour approx. du profil pendant montée/descente

    def normalized(self) -> "PressEvent":
        if self.t_off <= self.t_on:
            raise ValueError("t_off doit être > t_on")
        if self.ks <= 0.0:
            raise ValueError("ks doit être > 0")
        if self.cs < 0.0:
            raise ValueError("cs doit être >= 0")
        return self


@dataclass
class Segment:
    t_start: float
    t_end: float
    # node -> (ks_max, cs, ref_event)
    active: Dict[int, Tuple[float, float, Optional[PressEvent]]]

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


def build_press_segments(
    n_nodes: int,
    press_events: Iterable[PressEvent],
    *,
    t_start: float = 0.0,
    t_end: Optional[float] = None,
) -> List[Segment]:
    events = [ev.normalized() for ev in press_events]
    for ev in events:
        _valider_noeud_interne(n_nodes, ev.node)
    boundary: List[float] = []
    if t_end is not None:
        boundary = [float(t_start), float(t_end)]
        if boundary[1] <= boundary[0]:
            raise ValueError("t_end doit être > t_start")
    cut_core = sorted({ev.t_on for ev in events} | {ev.t_off for ev in events})
    cut_times: List[float] = sorted(set(boundary + cut_core)) if cut_core or boundary else []
    if not cut_times:
        return []
    segments: List[Segment] = []
    for i in range(len(cut_times) - 1):
        t0, t1 = cut_times[i], cut_times[i + 1]
        if t1 <= t0: continue
        active: Dict[int, Tuple[float, float, Optional[PressEvent]]] = {}
        for ev in events:
            if ev.t_on <= t0 < ev.t_off:
                active[ev.node] = (float(ev.ks), float(ev.cs), ev)
        segments.append(Segment(t_start=t0, t_end=t1, active=active))
    return segments


def _profil_polynomial_symetrique(t: float, t_debut: float, t_montee: float, t_plateau: float, k_max: float) -> float:
    """Retourne k_virtuel(t) (raideur additionnelle) pour un profil cubique symétrique avec plateau.

    Phases:
      - t < t_debut: 0
      - montée sur [t_debut, t_debut+t_montee): k = k_max*(3s^2 - 2s^3), s∈[0,1]
      - plateau sur [t_debut+t_montee, t_debut+t_montee+t_plateau): k = k_max
      - descente symétrique sur [t_fin_plateau, t_fin_plateau+t_montee): k = k_max*(3s^2 - 2s^3) avec s=(t_fin_descente - t)/t_montee
      - après: 0
    """
    t_fin_montee = t_debut + t_montee
    t_fin_plateau = t_fin_montee + t_plateau
    t_fin_descente = t_fin_plateau + t_montee
    if t < t_debut:
        return 0.0
    elif t < t_fin_montee:
        s = (t - t_debut) / max(1e-12, t_montee)
        return k_max * (3.0 * s * s - 2.0 * s * s * s)
    elif t < t_fin_plateau:
        return k_max
    elif t < t_fin_descente:
        s = (t_fin_descente - t) / max(1e-12, t_montee)
        return k_max * (3.0 * s * s - 2.0 * s * s * s)
    else:
        return 0.0


def _build_segments_profiled(n_nodes: int, press_events: Iterable[PressEvent], T_total: float) -> List[Segment]:
    """Construit des segments plus fins si des profils temporels sont demandés.

    - Découpe l'intervalle [0, T_total] selon les événements.
    - Pour chaque événement avec profil "polynomial", subdivise la montée/descente
      en `n_sous_segments` morceaux.
    - Plateau reste en un seul morceau (ou deux si coïncide avec d'autres coupures).
    """
    events = [ev.normalized() for ev in press_events]
    for ev in events:
        _valider_noeud_interne(n_nodes, ev.node)

    cut_times: List[float] = [0.0, float(T_total)]
    for ev in events:
        cut_times.extend([ev.t_on, ev.t_off])
        # Profil polynomial uniquement: toujours subdiviser montée et descente
        dur = max(0.0, ev.t_off - ev.t_on)
        t_montee = float(ev.t_montee) if ev.t_montee is not None else 0.25 * dur
        t_plateau = float(ev.t_plateau) if ev.t_plateau is not None else max(0.0, dur - 2.0 * t_montee)
        t_rise_end = ev.t_on + t_montee
        t_fall_start = ev.t_off - t_montee
        # subdiviser montée et descente
        nseg = max(1, int(ev.n_sous_segments))
        cut_times.extend(list(np.linspace(ev.t_on, t_rise_end, nseg + 1)))
        cut_times.extend(list(np.linspace(t_fall_start, ev.t_off, nseg + 1)))

    # unique et trié
    cut_sorted = sorted(set(float(x) for x in cut_times if 0.0 <= float(x) <= float(T_total)))
    segments: List[Segment] = []
    for i in range(len(cut_sorted) - 1):
        t0, t1 = cut_sorted[i], cut_sorted[i + 1]
        if t1 <= t0: continue
        active: Dict[int, Tuple[float, float, Optional[PressEvent]]] = {}
        for ev in events:
            if ev.t_on <= t0 < ev.t_off:
                active[ev.node] = (float(ev.ks), float(ev.cs), ev)
        segments.append(Segment(t_start=t0, t_end=t1, active=active))
    return segments

# -------------------------------
# 3) Intégration segmentée
# -------------------------------

def simulate_with_press(
    M: np.ndarray,
    K_base: np.ndarray,
    alpha: float,
    beta: float,
    F_base: np.ndarray | Callable[[float, int], np.ndarray],
    dt: float,
    press_events: Iterable[PressEvent],
    T_total: float,
    U0: Optional[np.ndarray] = None,
    V0: Optional[np.ndarray] = None,
):
    n = int(M.shape[0])
    if K_base.shape != M.shape:
        raise ValueError("K_base et M doivent avoir la même shape")
    if dt <= 0.0 or T_total <= 0.0:
        raise ValueError("dt et T_total doivent être > 0")

    try:
        events_list = list(press_events)
    except TypeError:
        events_list = [ev for ev in press_events]

    # Profil polynomial uniquement: toujours utiliser le découpage raffiné
    segs = _build_segments_profiled(n, events_list, float(T_total))
    C_base = alpha * M + beta * K_base

    if not segs:
        C0 = alpha * M + beta * K_base
        return integrer_newmark_beta(M, C0, K_base, F_base, dt, t_max=float(T_total), U0=U0, V0=V0, A0=None)

    U_prev = np.zeros(n, dtype=float) if U0 is None else np.asarray(U0, dtype=float)
    V_prev = np.zeros(n, dtype=float) if V0 is None else np.asarray(V0, dtype=float)

    t_lists: List[np.ndarray] = []
    U_lists: List[np.ndarray] = []
    V_lists: List[np.ndarray] = []
    A_lists: List[np.ndarray] = []

    t_acc = 0.0
    k_acc = 0

    def decaler_force(F, t_offset: float, k_offset: int):
        if callable(F):
            def F_shift(t_local: float, k_local: int):
                return np.asarray(F(float(t_local) + float(t_offset), int(k_local) + int(k_offset)), dtype=float)
            return F_shift
        return F

    first = True
    for seg in segs:
        T_seg = float(seg.duration)
        if T_seg <= 0.0: continue
        K_seg = K_base.copy()
        C_local = np.zeros_like(M)
        if seg.active:
            # Choisir ks effectif selon le profil polynomial du segment
            t_mid = seg.t_start + 0.5 * T_seg
            for j, (ks_max, cs, ev_ref) in seg.active.items():
                dur = max(0.0, ev_ref.t_off - ev_ref.t_on) if ev_ref is not None else 0.0
                t_montee = float(ev_ref.t_montee) if (ev_ref is not None and ev_ref.t_montee is not None) else 0.25 * dur
                t_plateau = float(ev_ref.t_plateau) if (ev_ref is not None and ev_ref.t_plateau is not None) else max(0.0, dur - 2.0 * t_montee)
                ks_eff = _profil_polynomial_symetrique(t_mid, ev_ref.t_on if ev_ref is not None else 0.0, t_montee, t_plateau, ks_max)
                K_seg[j, j] += float(ks_eff)
                if cs and cs > 0.0:
                    C_local[j, j] += float(cs)
        C_seg = alpha * M + beta * K_seg + C_local
        F_seg = decaler_force(F_base, t_offset=seg.t_start, k_offset=k_acc)
        t_loc, U_loc, V_loc, A_loc = integrer_newmark_beta(
            M, C_seg, K_seg, F_seg, dt=dt, t_max=T_seg, U0=U_prev, V0=V_prev, A0=None
        )
        if first:
            t_lists.append(t_loc + t_acc)
            U_lists.append(U_loc)
            V_lists.append(V_loc)
            A_lists.append(A_loc)
            first = False
        else:
            t_lists.append(t_loc[1:] + t_acc)
            U_lists.append(U_loc[:, 1:])
            V_lists.append(V_loc[:, 1:])
            A_lists.append(A_loc[:, 1:])
        U_prev = U_loc[:, -1]
        V_prev = V_loc[:, -1]
        t_acc += t_loc[-1]
        k_acc += (U_loc.shape[1] - 1)

    t_vec = np.concatenate(t_lists, axis=0)
    U_hist = np.concatenate(U_lists, axis=1)
    V_hist = np.concatenate(V_lists, axis=1)
    A_hist = np.concatenate(A_lists, axis=1)
    return t_vec, U_hist, V_hist, A_hist


# ---------------------------------------------------------------
# Petit banc d'essai: tracer le profil de force de pression (f_doigt)
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Ce mini-main exécute une courte simulation avec un événement de pression
    # et une excitation localisée, puis trace f_doigt(t) = ks·u_j + cs·v_j
    # pendant la fenêtre de pression.
    import sys
    from pathlib import Path

    import numpy as np
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print("[ERREUR] matplotlib est requis pour ce banc d'essai:", exc)
        sys.exit(1)

    # Imports tardifs pour éviter des dépendances lourdes au chargement du module
    try:
        from ..fem.formulation import build_global_mkc_from_config, amortissement_rayleigh  # type: ignore
        from .. import config  # type: ignore
    except Exception:
        # Fallback absolu si exécution directe hors package
        try:
            from digital_twin.back_end.fem.formulation import build_global_mkc_from_config, amortissement_rayleigh  # type: ignore
            from digital_twin.back_end import config  # type: ignore
        except Exception as exc:
            print("[ERREUR] Impossible d'importer formulation/config:", exc)
            sys.exit(1)

    # Excitation (fournisseur FR)
    try:
        from .excitation import fournisseur_force_localisee
    except Exception:  # pragma: no cover
        try:
            from digital_twin.back_end.interactions.excitation import fournisseur_force_localisee  # type: ignore
        except Exception as exc:
            print("[ERREUR] Impossible d'importer interactions.excitation:", exc)
            sys.exit(1)

    # 1) Matrices globales (avec CL fixes) + paramètres Rayleigh
    M, K, C, meta = build_global_mkc_from_config(apply_fixed_bc=True, return_meta=True)
    alpha = float(meta.get('alpha')) if 'alpha' in meta else None  # type: ignore[assignment]
    beta = float(meta.get('beta')) if 'beta' in meta else None    # type: ignore[assignment]
    if alpha is None or beta is None:
        # Sécurité: recalcul sur M/K contraints
        alpha, beta, _, _ = amortissement_rayleigh(M, K, config.DAMPING_MODES_REF, config.DAMPING_ZETAS_REF)

    n = M.shape[0]

    # 2) Paramètres de pression (démo)
    j = int(config.PRESS_NODE_INDEX) if getattr(config, 'PRESS_NODE_INDEX', None) is not None else max(1, int(0.3 * (n - 1)))
    t_on = float(getattr(config, 'PRESS_T_ON', 0.6))
    t_off = float(getattr(config, 'PRESS_T_OFF', 1.0))
    ks = float(getattr(config, 'PRESS_KS', 5e4))
    cs = float(getattr(config, 'PRESS_CS', 0.0))
    # Activer le profil polynomial (cubique symétrique) avec plateau
    # Choix par défaut: montée/descente = 25% de la fenêtre, plateau = reste
    dur_ev = max(0.0, t_off - t_on)
    t_montee = 0.25 * dur_ev
    t_plateau = max(0.0, dur_ev - 2.0 * t_montee)
    ev = PressEvent(node=j, t_on=t_on, t_off=t_off, ks=ks, cs=cs, t_montee=t_montee, t_plateau=t_plateau)

    # 3) Excitation localisée (même nœud pour maximiser l'effet)
    Fmax = float(getattr(config, 'EXCITATION_F_MAX', 1.0))
    t_rise = float(getattr(config, 'EXCITATION_T_RISE', 0.01))
    t_hold = float(getattr(config, 'EXCITATION_T_HOLD', 0.03))
    t_decay = float(getattr(config, 'EXCITATION_T_DECAY', 0.005))
    t0 = 0.10
    F_base = fournisseur_force_localisee(n, j, Fmax, t_rise, t_hold, t_decay, t0=t0)

    # 4) Intégration segmentée avec pression
    dt = float(getattr(config, 'DT', 1.0e-4))
    T_total = max(float(getattr(config, 'T_SIM', 2.0)), t_off + 0.2)
    t, U, V, A = simulate_with_press(M, K, alpha, beta, F_base, dt, [ev], T_total)

    # 5) Profil de rigidité locale ΔK(t) selon le même profil polynomial utilisé pendant l'intégration
    k_incr = np.zeros_like(t)
    for idx, tt in enumerate(t):
        k_incr[idx] = _profil_polynomial_symetrique(tt, t_on, t_montee, t_plateau, ks)
    # 6) Tracé et sauvegarde (dans back_end/results)
    results_dir = Path(__file__).resolve().parents[1] / 'results/plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    out_png_k = results_dir / 'profil_rigidite_pression_demo.png'
    plt.figure(figsize=(8, 3.2))
    plt.plot(t, k_incr, color="#e45756", label="ΔK_local (t)")
    plt.axvline(t_on, color="red", linestyle="--", linewidth=1.0, label="t_on")
    plt.axvline(t_off, color="green", linestyle="--", linewidth=1.0, label="t_off")
    plt.xlabel("Temps (s)")
    plt.ylabel("ΔK_local (N/m)")
    plt.title("Profil de rigidité locale pendant la pression")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    try:
        plt.savefig(out_png_k, dpi=150)
        print(f"[OK] Figure sauvegardée: {out_png_k}")
    except Exception as exc:
        print("[AVERTISSEMENT] Échec de sauvegarde de la figure ΔK:", exc)
    finally:
        plt.close()
