from __future__ import annotations

"""
Script pour visualiser la force externe F(t) utilisée dans la simulation.

- Construit le fournisseur de force localisé selon la configuration actuelle.
- Échantillonne l'amplitude au nœud excité au cours du temps [0, T_SIM].
- Sauvegarde un PNG dans results/plots/force_over_time.png.

Peut être exécuté directement :
    python -X utf8 -u digital_twin/back_end/results/plot_force_over_time.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from digital_twin.back_end import config as _cfg  # type: ignore
from digital_twin.back_end.fem.formulation import build_node_positions_from_config  # type: ignore
from digital_twin.back_end.fem.time_integration import fournisseur_force_localisee  # type: ignore


def build_force_provider_and_index(n: int):
    """Retourne (F_total, i_force) selon la configuration actuelle.

    Note: le main utilise aujourd'hui une seule force localisée (t0=0). La seconde est désactivée.
    """
    # Mapear PLUCK_POS em [0,1] para índice de nó mais próximo
    L_eff = float(getattr(_cfg, "L", 1.0))
    x_coords = build_node_positions_from_config(n)
    x_p_rel = float(getattr(_cfg, "PLUCK_POS", 0.25))
    x_force = float(x_p_rel * L_eff)
    i_force = int(np.argmin(np.abs(x_coords - x_force)))

    # Paramètres de l'enveloppe temporelle
    F_max = float(getattr(_cfg, "EXCITATION_F_MAX", 1.0))
    t_rise = float(getattr(_cfg, "EXCITATION_T_RISE", 0.01))
    t_hold = float(getattr(_cfg, "EXCITATION_T_HOLD", 0.03))
    t_decay = float(getattr(_cfg, "EXCITATION_T_DECAY", 0.005))
    t0 = 0.0

    F1 = fournisseur_force_localisee(n, i_force, F_max, t_rise, t_hold, t_decay, t0=t0)
    return F1, i_force


def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]
    plots_dir = ROOT / "digital_twin" / "back_end" / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Dimension du système et temps
    # On utilise les positions nodales pour inférer n actuel de la maille de frettes
    # (build_node_positions_from_config requiert n, donc on le récupère depuis config)
    try:
        # FRET_N_NODES se existir; senão caímos em N_NODES
        n_nodes = int(getattr(_cfg, "FRET_N_NODES", getattr(_cfg, "N_NODES", 100)))
    except Exception:
        n_nodes = 100

    # Fournisseur et index du nœud excité
    F_total, i_force = build_force_provider_and_index(n_nodes)

    # Échantillonnage temporel — uniquement jusqu'à la fin de l'enveloppe (t3)
    dt = float(getattr(_cfg, "DT", 1e-4))
    F_max = float(getattr(_cfg, "EXCITATION_F_MAX", 1.0))
    t_rise = float(getattr(_cfg, "EXCITATION_T_RISE", 0.01))
    t_hold = float(getattr(_cfg, "EXCITATION_T_HOLD", 0.03))
    t_decay = float(getattr(_cfg, "EXCITATION_T_DECAY", 0.005))
    t0 = 0.0
    t1 = t0 + t_rise
    t2 = t1 + t_hold
    t3 = t2 + t_decay
    T_plot = max(t0, t3)
    n_steps = int(np.floor(T_plot / dt)) + 1 if T_plot > 0 else 2
    t = np.linspace(0.0, dt * (n_steps - 1), n_steps)

    # Amplitude au nœud excité au cours du temps
    amp = np.zeros_like(t)
    for k in range(n_steps):
        Ft = np.asarray(F_total(t[k], k), dtype=float)
        amp[k] = Ft[i_force] if 0 <= i_force < Ft.shape[0] else 0.0

    # Tracé
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, amp, lw=1.8)
    ax.set_title(f"Force localisée au nœud {i_force} — enveloppe trapézoïdale (0→t3={t3:.3f}s)")
    ax.set_xlabel("temps (s)")
    ax.set_ylabel("F (N)")
    ax.grid(True, alpha=0.3)

    # Anotar marcos do trapézio para referência
    for tx, lbl in [(t0, "t0"), (t1, "t1"), (t2, "t2"), (t3, "t3")]:
        ax.axvline(tx, color="k", lw=0.8, ls=":")
        ax.text(tx, 0.02 * F_max, lbl, rotation=90, va="bottom", ha="right", fontsize=8)

    out_path = plots_dir / "force_over_time.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Graphique enregistré dans : {out_path}")
    print(f"[INFO] Nœud excité : {i_force}")


if __name__ == "__main__":
    main()
