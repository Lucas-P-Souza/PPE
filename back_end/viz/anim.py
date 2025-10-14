"""
Animations (mouvement de la corde) avec Matplotlib.
"""
from __future__ import annotations

import numpy as np


def animate_string_motion(
    x: np.ndarray,
    U: np.ndarray,
    *,
    interval_ms: int = 30,
    decim: int = 1,
    savepath: str | None = None,
    show: bool = False,
    y_scale: float = 1.0,
    y_pad_frac: float = 0.05,
    y_limits: tuple[float, float] | None = None,
):
    """Crée une animation du déplacement dans le temps (Matplotlib FuncAnimation)."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.animation import FuncAnimation, PillowWriter  # type: ignore
    except Exception as e:  # pragma: no cover
        print("[AVERTISSEMENT] Matplotlib/animation indisponible:", e)
        return

    U_anim = U[:, ::max(1, int(decim))]
    n, n_frames = U_anim.shape
    U_anim[0, :] = 0.0
    U_anim[-1, :] = 0.0

    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot(x, U_anim[:, 0], lw=1.5)
    ax.set_title("Mouvement de la corde")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("déplacement (m)")
    ax.grid(True, alpha=0.3)

    if y_limits is not None and len(y_limits) == 2:
        ax.set_ylim(float(y_limits[0]), float(y_limits[1]))
    else:
        lo = float(np.nanpercentile(U_anim, 1))
        hi = float(np.nanpercentile(U_anim, 99))
        rng = max(1e-12, (hi - lo))
        c = 0.5 * (hi + lo)
        factor = max(1e-6, float(y_scale))
        half = 0.5 * rng * factor
        pad = float(y_pad_frac) * rng
        ax.set_ylim(c - (half + pad), c + (half + pad))

    def init():
        line.set_ydata(U_anim[:, 0])
        return (line,)

    def update(frame: int):
        line.set_ydata(U_anim[:, frame])
        return (line,)

    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=interval_ms, blit=True)

    if savepath:
        from pathlib import Path as _P
        out = _P(savepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".gif":
            try:
                ani.save(out, writer=PillowWriter(fps=max(1, int(1000/interval_ms))))
                print(f"[INFO] Animation sauvegardée dans : {out}")
            except Exception as _esave:
                print("[AVERTISSEMENT] Échec lors de l'enregistrement du GIF:", _esave)
        else:
            try:
                ani.save(out)
                print(f"[INFO] Animation sauvegardée dans : {out}")
            except Exception as _esave2:
                print("[AVERTISSEMENT] Échec lors de l'enregistrement de l'animation:", _esave2)

    if show:
        plt.show()
    else:
        plt.close(fig)


# Alias en français (API)
def animer_mouvement_corde(x, U, *, interval_ms: int = 30, decim: int = 1, savepath: str | None = None, show: bool = False, y_scale: float = 1.0, y_pad_frac: float = 0.05, y_limits=None):
    return animate_string_motion(x, U, interval_ms=interval_ms, decim=decim, savepath=savepath, show=show, y_scale=y_scale, y_pad_frac=y_pad_frac, y_limits=y_limits)
