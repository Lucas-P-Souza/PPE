"""
Script temporário: lê o CSV gerado pelo solver e monta gráficos de:
- Movimento da onda (mapa de calor: deslocamento por nó vs tempo)
- Amortecimento (série temporal de um nó + envelope RMS)

Uso:
  - Sem argumentos: busca o CSV mais recente em ./results/positions_simple_*.csv
  - Com argumento: caminho para um CSV específico

Saída:
  - Salva um PNG ao lado do CSV com sufixo _plot.png
  - Abre a janela do gráfico (se backend permitir)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec


def _find_latest_csv(results_dir: Path) -> Optional[Path]:
    candidates = sorted(results_dir.glob("positions_simple_*.csv"))
    return candidates[-1] if candidates else None


def _moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    if win <= 1:
        return np.abs(x)
    kernel = np.ones(win, dtype=float) / float(win)
    y2 = x * x
    rms2 = np.convolve(y2, kernel, mode="same")
    return np.sqrt(np.maximum(rms2, 0.0))


def _x_positions_from_config(n_nodes: int) -> Optional[np.ndarray]:
    """Tenta recuperar coordenadas físicas x (em metros) a partir do config.
    - Se FRET_DXS_MM existir: usa cumulativa desses dxs (em m).
    - Senão, se L e N_NODES existirem: usa linspace uniforme [0, L].
    Retorna None em caso de falha.
    """
    try:
        # Ajuste sys.path para importar o pacote
        ROOT = Path(__file__).resolve().parents[3]
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from digital_twin.back_end import config as _cfg  # type: ignore
    except Exception:
        return None

    try:
        dxs_mm = getattr(_cfg, "FRET_DXS_MM", None)
        if dxs_mm and len(dxs_mm) == n_nodes - 1:
            dxs_m = np.asarray([d/1000.0 for d in dxs_mm], dtype=float)
            x = np.concatenate([[0.0], np.cumsum(dxs_m)])
            return x
        # fallback uniforme
        L = float(getattr(_cfg, "L", None))
        N = int(getattr(_cfg, "N_NODES", n_nodes))
        if np.isfinite(L) and N == n_nodes and L > 0:
            return np.linspace(0.0, L, n_nodes)
    except Exception:
        return None
    return None


def plot_wave_and_damping(csv_path: Path, node_index: int | None = None, save_png: bool = True) -> Path:
    # Carrega dados
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t = data[:, 0]
    U = data[:, 1:]
    steps, n_nodes = U.shape
    if node_index is None:
        node_index = n_nodes // 2
    node_index = int(np.clip(node_index, 0, n_nodes - 1))

    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    # Janela ~10 ms para envelope (se dt muito pequeno, ainda funciona)
    win = max(1, int(round(0.010 / dt)))
    y = U[:, node_index]
    env = _moving_rms(y, win)

    # Eixo X físico (opcional)
    x = _x_positions_from_config(n_nodes)
    x_nodes = x if x is not None else np.arange(n_nodes)
    x_label = "Posição (m)" if x is not None else "Índice do nó"

    # Figura com dois painéis
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0])

    # Painel 1: movimento da onda (heatmap)
    ax1 = fig.add_subplot(gs[0, :])
    # Heatmap em coordenada de nó; se quisermos x físico no eixo vertical,
    # precisaríamos reamostrar. Aqui, mantemos índice para o heatmap.
    im = ax1.imshow(U.T, aspect="auto",
                    extent=[float(t[0]), float(t[-1]), 0, n_nodes - 1],
                    origin="lower", cmap="RdBu_r")
    ax1.set_title("Movimento da onda (deslocamento vs tempo)")
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Índice do nó")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)
    cbar.set_label("Deslocamento (m)")

    # Painel 2: perfis espaciais em 4 instantes
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Perfis espaciais em instantes amostrados")
    ax2.set_xlabel("Índice do nó")
    ax2.set_ylabel("Deslocamento (m)")
    node_ids = x_nodes
    sample_ids = np.linspace(0, steps - 1, num=4, dtype=int)
    for sid in sample_ids:
        y_plot = np.nan_to_num(U[sid, :], nan=0.0, posinf=0.0, neginf=0.0)
        ax2.plot(node_ids, y_plot, alpha=0.8, label=f"t={t[sid]:.4f}s")
    ax2.legend(loc="best", fontsize=8)
    ax2.set_xlim(x_nodes[0], x_nodes[-1])
    ax2.set_xlabel(x_label)

    # Painel 3: amortecimento (nó selecionado)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title(f"Amortecimento — nó {node_index}")
    ax3.set_xlabel("Tempo (s)")
    ax3.set_ylabel("Deslocamento (m)")
    ax3.plot(t, np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), label="Sinal")
    ax3.plot(t, np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0), "k", lw=2, label="Envelope (RMS)")
    ax3.legend(loc="best")

    fig.tight_layout()

    out_png = csv_path.with_name(csv_path.stem + "_plot.png")
    if save_png:
        try:
            fig.savefig(out_png, dpi=150)
        except Exception:
            # fallback de backend
            matplotlib.use("Agg", force=True)
            fig.savefig(out_png, dpi=150)
    return out_png


if __name__ == "__main__":
    # Detecta CSV
    this_file = Path(__file__).resolve()
    results_dir = this_file.parent
    csv_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if csv_arg is None:
        csv_path = _find_latest_csv(results_dir)
        if csv_path is None:
            print("[ERRO] Nenhum CSV 'positions_simple_*.csv' encontrado em:", results_dir)
            sys.exit(1)
    else:
        csv_path = csv_arg
        if not csv_path.exists():
            print("[ERRO] CSV não encontrado:", csv_path)
            sys.exit(1)

    out_png = plot_wave_and_damping(csv_path)
    print("[OK] Gráfico salvo em:", out_png)
    # Mostrar figura (se backend interativo disponível)
    try:
        plt.show()
    except Exception:
        pass
