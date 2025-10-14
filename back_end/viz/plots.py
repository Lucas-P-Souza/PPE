"""
Fonctions de visualisation (tracés statiques) pour la corde vibrante.
"""
from __future__ import annotations

import numpy as np


def plot_first_modes(x: np.ndarray, modes_full: np.ndarray, freqs_hz: np.ndarray, *, max_modes: int = 4, savepath: str | None = None) -> None:
    """Trace jusqu'à max_modes les formes modales vs x et enregistre éventuellement."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        print("[AVERTISSEMENT] matplotlib indisponible pour tracer les modes:", e)
        return

    m = int(min(max_modes, modes_full.shape[1]))
    if m == 0:
        print("[INFO] Aucun mode à tracer.")
        return
    plt.figure(figsize=(8, 6))
    for j in range(m):
        plt.plot(x, modes_full[:, j], label=f"Mode {j+1} — {freqs_hz[j]:.2f} Hz")
    plt.title("Premiers modes — formes et fréquences")
    plt.xlabel("x (m)")
    plt.ylabel("Forme modale (normalisée en masse)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if savepath:
        from pathlib import Path as _P
        out = _P(savepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150)
        print(f"[INFO] Modes enregistrés dans : {out}")
    else:
        plt.show()
    plt.close()


def save_string_frame_png(
    x: np.ndarray,
    U: np.ndarray,
    frame_idx: int,
    savepath: str,
    *,
    y_scale: float | None = None,
    y_pad_frac: float = 0.05,
) -> None:
    """Enregistre une image unique (profil) du déplacement en PNG."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        print("[AVERTISSEMENT] Matplotlib indisponible pour enregistrer l'image:", e)
        return
    from pathlib import Path as _P
    out = _P(savepath)
    out.parent.mkdir(parents=True, exist_ok=True)
    y = np.array(U[:, frame_idx], dtype=float)
    y[0] = 0.0
    y[-1] = 0.0
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x, y, lw=1.6)
    ax.set_title(f"Profil de la corde — frame {frame_idx}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("déplacement (m)")
    ax.grid(True, alpha=0.3)
    lo = float(np.nanpercentile(y, 1))
    hi = float(np.nanpercentile(y, 99))
    rng = max(1e-12, (hi - lo))
    if y_scale is None:
        pad = float(y_pad_frac) * rng
        ax.set_ylim(lo - pad, hi + pad)
    else:
        c = 0.5 * (hi + lo)
        factor = max(1e-6, float(y_scale))
        half = 0.5 * rng * factor
        pad = float(y_pad_frac) * rng
        ax.set_ylim(c - (half + pad), c + (half + pad))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] Image enregistrée dans : {out}")


def plot_snapshots_png(
    x: np.ndarray,
    U: np.ndarray,
    t: np.ndarray,
    *,
    n_snapshots: int = 8,
    decim: int = 1,
    savepath: str,
    title: str = "Profils de la corde à des temps sélectionnés",
    y_scale: float | None = None,
    y_pad_frac: float = 0.05,
    t_window: tuple[float, float] | None = None,
    use_colorbar: bool = True,
    cmap: str = "viridis",
    alpha: float = 0.9,
    linewidth: float = 1.2,
    show_legend: bool = False,
) -> None:
    """Enregistre un PNG statique avec plusieurs profils à des temps sélectionnés."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        print("[AVERTISSEMENT] Matplotlib indisponible pour les instantanés:", e)
        return
    from pathlib import Path as _P
    out = _P(savepath)
    out.parent.mkdir(parents=True, exist_ok=True)

    step = max(1, int(decim))
    U_dec = U[:, ::step]
    t_dec = t[::step]
    n_steps = U_dec.shape[1]
    if n_steps == 0:
        print("[AVERTISSEMENT] Aucun frame pour les instantanés.")
        return

    if t_window is not None and len(t_window) == 2:
        t0, t1 = float(min(t_window)), float(max(t_window))
        mask = (t_dec >= t0) & (t_dec <= t1)
        if np.any(mask):
            U_dec = U_dec[:, mask]
            t_dec = t_dec[mask]
            n_steps = U_dec.shape[1]
        else:
            print("[AVERTISSEMENT] Fenêtre temporelle sans intersection, utilisation de l'intervalle complet.")

    if n_steps < 2 and step > 1:
        U_dec = U
        t_dec = t
        if t_window is not None and len(t_window) == 2:
            t0, t1 = float(min(t_window)), float(max(t_window))
            mask = (t_dec >= t0) & (t_dec <= t1)
            if np.any(mask):
                U_dec = U_dec[:, mask]
                t_dec = t_dec[mask]
        n_steps = U_dec.shape[1]

    k_req = int(max(1, n_snapshots))
    k_eff = int(min(k_req, n_steps))
    if k_eff < k_req:
        print(f"[AVERTISSEMENT] Seulement {k_eff} frames disponibles pour les instantanés (demandé {k_req}).")
    if n_steps <= 1:
        idx = np.array([0], dtype=int)
    else:
        idx = np.linspace(0, n_steps - 1, k_eff)
        idx = np.unique(np.rint(idx).astype(int))
        if idx.size < min(2, n_steps):
            idx = np.arange(min(n_steps, 2), dtype=int)

    Y = U_dec[:, idx]
    lo = float(np.nanpercentile(Y, 1))
    hi = float(np.nanpercentile(Y, 99))
    rng = max(1e-12, hi - lo)
    if y_scale is not None:
        c = 0.5 * (hi + lo)
        factor = max(1e-6, float(y_scale))
        half = 0.5 * rng * factor
        pad = float(y_pad_frac) * rng
        ymin, ymax = c - (half + pad), c + (half + pad)
    else:
        pad = float(y_pad_frac) * rng
        ymin, ymax = lo - pad, hi + pad

    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    cmap_obj = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=float(t_dec[idx[0]]), vmax=float(t_dec[idx[-1]]))
    for j, jj in enumerate(idx):
        col = cmap_obj(norm(float(t_dec[jj])))
        lbl = f"t={t_dec[jj]:.3f}s" if show_legend else None
        ax.plot(x, U_dec[:, jj], color=col, lw=float(linewidth), alpha=float(alpha), label=lbl)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("déplacement (m)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(ymin, ymax)
    if show_legend:
        ax.legend(ncol=2, fontsize=8, framealpha=0.85)
    if use_colorbar:
        import matplotlib.cm as cm
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("temps (s)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[INFO] Instantanés enregistrés dans : {out}")


# -------------------------------
#   Alias en français (API)
# -------------------------------
def tracer_premiers_modes(x: np.ndarray, modes_full: np.ndarray, freqs_hz: np.ndarray, *, max_modes: int = 4, savepath: str | None = None) -> None:
    return plot_first_modes(x, modes_full, freqs_hz, max_modes=max_modes, savepath=savepath)


def enregistrer_image_corde(x: np.ndarray, U: np.ndarray, frame_idx: int, savepath: str, *, y_scale: float | None = None, y_pad_frac: float = 0.05) -> None:
    return save_string_frame_png(x, U, frame_idx, savepath, y_scale=y_scale, y_pad_frac=y_pad_frac)


def tracer_profils_temps(x: np.ndarray, U: np.ndarray, t: np.ndarray, *, n_snapshots: int = 8, decim: int = 1, savepath: str, title: str = "Profils de la corde à des temps sélectionnés", y_scale: float | None = None, y_pad_frac: float = 0.05, t_window: tuple[float, float] | None = None, use_colorbar: bool = True, cmap: str = "viridis", alpha: float = 0.9, linewidth: float = 1.2, show_legend: bool = False) -> None:
    return plot_snapshots_png(x, U, t, n_snapshots=n_snapshots, decim=decim, savepath=savepath, title=title, y_scale=y_scale, y_pad_frac=y_pad_frac, t_window=t_window, use_colorbar=use_colorbar, cmap=cmap, alpha=alpha, linewidth=linewidth, show_legend=show_legend)
