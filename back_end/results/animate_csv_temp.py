"""
Script temporário: cria uma animação (GIF) do movimento da corda a partir do CSV.

Funcionalidades:
- Carrega o CSV mais recente (positions_simple_*.csv) ou um arquivo específico.
- Anima o perfil espacial U(x, t) ao longo do tempo.
- Faz decimação de frames para controlar duração.
- Ajuste robusto de limites Y (com clipping por percentil) para evitar estouros.

Saída: salva um GIF ao lado do CSV com sufixo _anim.gif
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)  # render sem janela
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


def _find_latest_csv(results_dir: Path) -> Optional[Path]:
    files = sorted(results_dir.glob("positions_simple_*.csv"))
    return files[-1] if files else None


def _robust_ylim(U: np.ndarray, q: float = 99.0) -> tuple[float, float]:
    # Limites por percentil para evitar escala dominada por outliers/instabilidade
    lo = float(np.nanpercentile(U, 100 - q))
    hi = float(np.nanpercentile(U, q))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(U)), float(np.nanmax(U))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = -1e-3, 1e-3
    pad = 0.05 * (hi - lo) if hi > lo else 1e-6
    return lo - pad, hi + pad


def _x_positions_from_config(n_nodes: int) -> Optional[np.ndarray]:
    try:
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
        L = float(getattr(_cfg, "L", None))
        N = int(getattr(_cfg, "N_NODES", n_nodes))
        if np.isfinite(L) and N == n_nodes and L > 0:
            return np.linspace(0.0, L, n_nodes)
    except Exception:
        return None
    return None


def animate_csv(csv_path: Path, every: int = 5, fps: int = 30) -> Path:
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t = data[:, 0]
    U = data[:, 1:]
    steps, n_nodes = U.shape
    x = _x_positions_from_config(n_nodes)
    node_ids = x if x is not None else np.arange(n_nodes)
    x_label = "Posição (m)" if x is not None else "Índice do nó"

    # Remoção de cauda instável: encontra último índice válido
    U_finite = np.isfinite(U)
    valid_mask = np.all(U_finite, axis=1)
    last_valid = np.where(valid_mask)[0].max() if np.any(valid_mask) else (steps - 1)
    U_clip = U[: last_valid + 1, :]
    t_clip = t[: last_valid + 1]

    # Threshold robusto por percentil para cortar explosões tardias
    U_abs = np.abs(np.nan_to_num(U_clip, nan=0.0, posinf=0.0, neginf=0.0))
    q = np.nanpercentile(U_abs, 99.5)
    if not np.isfinite(q) or q <= 0:
        q = 1e6  # fallback alto
    amp_mask = (np.max(U_abs, axis=1) <= 5.0 * q)
    last_amp_ok = np.where(amp_mask)[0].max() if np.any(amp_mask) else (U_clip.shape[0] - 1)
    last_idx = int(min(last_valid, last_amp_ok))
    U_clip = U[: last_idx + 1, :]
    t_clip = t[: last_idx + 1]

    # Seleção de frames (decimação) após limpeza
    idx = np.arange(0, U_clip.shape[0], max(1, int(every)))
    t_sel = t_clip[idx]
    U_sel = U_clip[idx, :]

    # Eixos e curva
    fig, ax = plt.subplots(figsize=(8, 4))
    y0 = np.nan_to_num(U_sel[0, :], nan=0.0, posinf=0.0, neginf=0.0)
    line, = ax.plot(node_ids, y0, lw=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Deslocamento (m)")
    ax.set_title("Movimento da corda ao longo do tempo")

    # Limites robustos
    ylo, yhi = _robust_ylim(np.nan_to_num(U_sel, nan=0.0, posinf=0.0, neginf=0.0))
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(node_ids[0], node_ids[-1])

    # Animação com PillowWriter
    out_gif = csv_path.with_name(csv_path.stem + "_anim.gif")
    writer = PillowWriter(fps=fps)

    with writer.saving(fig, str(out_gif), dpi=120):
        for k in range(len(idx)):
            yk = np.nan_to_num(U_sel[k, :], nan=0.0, posinf=0.0, neginf=0.0)
            line.set_ydata(yk)
            ax.set_title(f"Movimento da corda — t={t_sel[k]:.4f} s")
            writer.grab_frame()

    plt.close(fig)
    return out_gif


if __name__ == "__main__":
    res_dir = Path(__file__).resolve().parent
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else _find_latest_csv(res_dir)
    if csv is None or not csv.exists():
        print("[ERRO] CSV não encontrado. Gere a simulação primeiro.")
        sys.exit(1)
    gif_path = animate_csv(csv)
    print("[OK] GIF salvo em:", gif_path)
