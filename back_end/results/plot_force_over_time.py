from __future__ import annotations

"""
Script para visualizar a força externa F(t) usada na simulação.

- Constrói o provedor de força localizado conforme o config atual.
- Amostra a amplitude no nó excitado ao longo do tempo [0, T_SIM].
- Salva um PNG em results/plots/force_over_time.png.

Pode ser executado diretamente:
  python -X utf8 -u digital_twin/back_end/results/plot_force_over_time.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Imports com fallback para execução direta
try:
    from digital_twin.back_end import config as _cfg  # type: ignore
    from digital_twin.back_end.fem.solver import build_node_positions_from_config  # type: ignore
    from digital_twin.back_end.fem.time_integration import fournisseur_force_localisee  # type: ignore
except ModuleNotFoundError:
    import sys as _sys
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    from digital_twin.back_end import config as _cfg  # type: ignore
    from digital_twin.back_end.fem.solver import build_node_positions_from_config  # type: ignore
    from digital_twin.back_end.fem.time_integration import fournisseur_force_localisee  # type: ignore


def build_force_provider_and_index(n: int):
    """Retorna (F_total, i_force) conforme config atual.

    Hoje o main usa apenas uma força localizada com t0=0. A segunda está desativada.
    """
    # Mapear PLUCK_POS em [0,1] para índice de nó mais próximo
    L_eff = float(getattr(_cfg, "L", 1.0))
    x_coords = build_node_positions_from_config(n)
    x_p_rel = float(getattr(_cfg, "PLUCK_POS", 0.25))
    x_force = float(x_p_rel * L_eff)
    i_force = int(np.argmin(np.abs(x_coords - x_force)))

    # Parâmetros de envelope
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

    # Dimensão do sistema e tempo
    # Usamos as posições de nó para inferir n atual da malha de frettes
    # (build_node_positions_from_config requer n, então pegamos do config)
    try:
        # FRET_N_NODES se existir; senão caímos em N_NODES
        n_nodes = int(getattr(_cfg, "FRET_N_NODES", getattr(_cfg, "N_NODES", 100)))
    except Exception:
        n_nodes = 100

    # Fornecedor e índice excitado
    F_total, i_force = build_force_provider_and_index(n_nodes)

    # Amostragem temporal — apenas até o fim do envelope (t3)
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

    # Amplitude no nó excitado ao longo do tempo
    amp = np.zeros_like(t)
    for k in range(n_steps):
        Ft = np.asarray(F_total(t[k], k), dtype=float)
        amp[k] = Ft[i_force] if 0 <= i_force < Ft.shape[0] else 0.0

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, amp, lw=1.8)
    ax.set_title(f"Força localizada no nó {i_force} — envelope trapezoidal (0→t3={t3:.3f}s)")
    ax.set_xlabel("tempo (s)")
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
    print(f"[INFO] Gráfico salvo em: {out_path}")
    print(f"[INFO] Nó excitado: {i_force}")


if __name__ == "__main__":
    main()
