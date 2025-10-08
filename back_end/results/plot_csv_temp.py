# Script temporaire: lit le CSV généré par le solveur et produit des graphiques de:
# - Mouvement de l'onde (carte thermique: déplacement par nœud vs temps)
# - Amortissement (série temporelle d'un nœud + enveloppe RMS)
#
# Utilisation:
#  - Sans arguments: cherche le CSV le plus récent dans ./results/positions_simple_*.csv
#  - Avec argument: chemin vers un CSV spécifique
#
# Sortie:
#  - Enregistre un PNG à côté du CSV avec le suffixe _plot.png
#  - Ouvre la fenêtre du graphique (si le backend le permet)

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
    # Tente de récupérer des coordonnées physiques x (en mètres) à partir du module config.
    # - Si FRET_DXS_MM existe: utilise la cumulée de ces dx (en m).
    # - Sinon, si L et N_NODES existent: utilise un linspace uniforme [0, L].
    # Retourne None en cas d'échec.
    try:
    # Ajuste sys.path pour importer le paquet
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
    # Charge les données
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t = data[:, 0]
    U = data[:, 1:]
    steps, n_nodes = U.shape
    if node_index is None:
        node_index = n_nodes // 2
    node_index = int(np.clip(node_index, 0, n_nodes - 1))

    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    # Fenêtre ~10 ms pour l'enveloppe (si dt est très petit, cela fonctionne toujours)
    win = max(1, int(round(0.010 / dt)))
    y = U[:, node_index]
    env = _moving_rms(y, win)

    # Axe X physique (optionnel)
    x = _x_positions_from_config(n_nodes)
    x_nodes = x if x is not None else np.arange(n_nodes)
    x_label = "Posição (m)" if x is not None else "Índice do nó"

    # Figure avec deux panneaux
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0])

    # Panneau 1: mouvement de l'onde (carte thermique)
    ax1 = fig.add_subplot(gs[0, :])
    # Carte thermique en coordonnées de nœud; si l'on veut x physique sur l'axe vertical,
    # il faudrait rééchantillonner. Ici, on garde l'indice pour la carte thermique.
    im = ax1.imshow(U.T, aspect="auto",
                    extent=[float(t[0]), float(t[-1]), 0, n_nodes - 1],
                    origin="lower", cmap="RdBu_r")
    ax1.set_title("Mouvement de l'onde (déplacement vs temps)")
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Indice du nœud")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)
    cbar.set_label("Déplacement (m)")

    # Panneau 2: profils spatiaux à 4 instants
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Profils spatiaux à des instants échantillonnés")
    ax2.set_xlabel("Indice du nœud")
    ax2.set_ylabel("Déplacement (m)")
    node_ids = x_nodes
    sample_ids = np.linspace(0, steps - 1, num=4, dtype=int)
    for sid in sample_ids:
        y_plot = np.nan_to_num(U[sid, :], nan=0.0, posinf=0.0, neginf=0.0)
        ax2.plot(node_ids, y_plot, alpha=0.8, label=f"t={t[sid]:.4f}s")
    ax2.legend(loc="best", fontsize=8)
    ax2.set_xlim(x_nodes[0], x_nodes[-1])
    ax2.set_xlabel(x_label)

    # Panneau 3: amortissement (nœud sélectionné)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title(f"Amortissement — nœud {node_index}")
    ax3.set_xlabel("Temps (s)")
    ax3.set_ylabel("Déplacement (m)")
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
    # Détecte le CSV
    this_file = Path(__file__).resolve()
    results_dir = this_file.parent
    csv_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if csv_arg is None:
        csv_path = _find_latest_csv(results_dir)
        if csv_path is None:
            print("[ERREUR] Aucun CSV 'positions_simple_*.csv' trouvé dans:", results_dir)
            sys.exit(1)
    else:
        csv_path = csv_arg
        if not csv_path.exists():
            print("[ERREUR] CSV introuvable:", csv_path)
            sys.exit(1)

    out_png = plot_wave_and_damping(csv_path)
    print("[OK] Graphique enregistré dans:", out_png)
    # Mostrar figura (se backend interativo disponível)
    try:
        plt.show()
    except Exception:
        pass
