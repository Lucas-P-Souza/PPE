"""Utilitaires FEM pour la simulation d'une corde vibrante.

Fonctions fournies :
 - generate_pluck_shape : génère le déplacement initial (pincement) triangulaire.
 - plot_waveform : enregistre la forme d'onde temporelle.
 - plot_fft : calcule et enregistre le spectre de magnitude (FFT).
 - create_animation : produit une animation spatio-temporelle.

Référence (forme triangulaire) :
 u(i,0) = h * (x_i / x_p) si x_i <= x_p, sinon h * ((L - x_i)/(L - x_p)).
 (h = amplitude maximale, x_p = position du pincement.)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import shutil
from pathlib import Path


def generate_pluck_shape(
    n_nodes: int,
    length: float,
    pluck_position_ratio: float,
    pluck_amplitude: float,
) -> np.ndarray:
    """Construit le vecteur de déplacement initial triangulaire (pincement).

    Paramètres
    ----------
    n_nodes : int
        Nombre total de nœuds.
    length : float
        Longueur totale L (m).
    pluck_position_ratio : float
        Position relative 0..1 du pincement.
    pluck_amplitude : float
        Amplitude maximale h (m).

    Retour
    ------
    np.ndarray
        Tableau (n_nodes,) des déplacements initiaux.
    """
    if not (0.0 < pluck_position_ratio < 1.0):
        raise ValueError("pluck_position_ratio deve estar em (0,1)")
    if n_nodes < 2:
        raise ValueError("n_nodes deve ser >= 2")
    if length <= 0:
        raise ValueError("length deve ser positivo")

    # Indice du point de pincement (arrondi au nœud le plus proche)
    pluck_point = int(round(pluck_position_ratio * (n_nodes - 1)))
    # Évite 0 ou n_nodes-1 pour prévenir division par zéro
    pluck_point = max(1, min(pluck_point, n_nodes - 2))

    x = np.linspace(0.0, length, n_nodes)
    x_p = x[pluck_point]

    u0 = np.zeros(n_nodes, dtype=float)

    # Segment croissant : x_i <= x_p  => h * (x_i / x_p)
    left_mask = x <= x_p
    u0[left_mask] = pluck_amplitude * (x[left_mask] / x_p)

    # Segment décroissant : x_i > x_p => h * ((L - x_i)/(L - x_p))
    right_mask = ~left_mask
    denom = (length - x_p)
    if denom <= 0:
    # Sécurité contre division par zéro en cas de point extrême
        denom = 1.0
    u0[right_mask] = pluck_amplitude * ((length - x[right_mask]) / denom)

    # Extrémités fixées = 0 (stabilité numérique)
    u0[0] = 0.0
    u0[-1] = 0.0
    return u0


def _ensure_parent(filepath: str | Path):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_waveform(
    signal: np.ndarray,
    sample_rate: float,
    filepath: str | Path,
    title: str = "Waveform",
    time_limit: float | None = None,
):
    """Trace et enregistre le signal temporel.

    time_limit : optionnel, tronque l'affichage jusqu'à ce temps (s).
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate deve ser positivo")
    n = len(signal)
    t = np.arange(n) / sample_rate
    if time_limit is not None:
        mask = t <= time_limit
        t = t[mask]
        signal = signal[mask]

    path = _ensure_parent(filepath)
    plt.figure(figsize=(8, 3))
    plt.plot(t, signal, lw=1.0)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Deslocamento (m)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_fft(
    signal: np.ndarray,
    sample_rate: float,
    filepath: str | Path,
    title: str = "FFT",
    n_fft: int | None = None,
    freq_limit: float | None = None,
):
    """Calcule la magnitude de la FFT d'un signal réel et enregistre le spectre.

    n_fft : si fourni, applique zero‑padding / découpe.
    freq_limit : limite supérieure de fréquences (Hz) pour l'affichage.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate deve ser positivo")
    x = np.asarray(signal, dtype=float)
    if n_fft is not None:
        if n_fft <= 0:
            raise ValueError("n_fft deve ser positivo")
        if len(x) < n_fft:
            x = np.pad(x, (0, n_fft - len(x)))
        else:
            x = x[:n_fft]

    N = len(x)
    # FFT real -> usamos rfft
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)
    magnitude = np.abs(spec) / N

    if freq_limit is not None:
        mask = freqs <= freq_limit
        freqs = freqs[mask]
        magnitude = magnitude[mask]

    path = _ensure_parent(filepath)
    plt.figure(figsize=(8, 3))
    plt.plot(freqs, magnitude, lw=1.0)
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def create_animation(
    full_history: np.ndarray,
    length: float,
    dt: float,
    filepath: str | Path,
    frame_step: int | None = None,
    fps: int = 30,
    max_frames: int = 120,
):
    """Génère une animation (MP4 ou GIF) de la vibration de la corde.

    Si `frame_step` est None il est choisi pour ne pas dépasser `max_frames` images.
    """
    if full_history.ndim != 2:
        raise ValueError("full_history deve ter dimensão 2 (num_steps, n_nodes)")
    if fps <= 0:
        raise ValueError("fps deve ser positivo")
    if max_frames <= 1:
        raise ValueError("max_frames deve ser > 1")

    num_steps, n_nodes = full_history.shape
    if num_steps < 2:
        raise ValueError("Histórico temporal insuficiente para animar")

    if frame_step is None or frame_step <= 0:
        frame_step = max(1, num_steps // max_frames)

    anim_data = full_history[::frame_step]
    # Inclusion forcée du dernier état si l'échantillonnage ne tombe pas juste :
    if (num_steps - 1) % frame_step != 0:
        anim_data = np.vstack([anim_data, full_history[-1]])
    num_frames = anim_data.shape[0]

    x_coords = np.linspace(0.0, length, n_nodes)
    max_amp = float(np.max(np.abs(full_history))) or 1e-9  # évite plage nulle

    fig, ax = plt.subplots(figsize=(8, 3))
    line, = ax.plot(x_coords, anim_data[0], lw=2.0)
    ax.set_xlim(0.0, length)
    ax.set_ylim(-1.2 * max_amp, 1.2 * max_amp)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Amplitude (m)")
    ax.set_title("Temps: 0.000 s")
    ax.grid(True, alpha=0.3)

    def update(frame: int):
        y = anim_data[frame]
        line.set_ydata(y)
        current_time = frame * dt * frame_step
        ax.set_title(f"Temps: {current_time:.3f} s")
        return (line,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        blit=True,
        interval=1000 / fps,
    )

    path = _ensure_parent(filepath)
    suffix = Path(filepath).suffix.lower()
    try:
        if suffix == ".gif":
            ani.save(path, writer="pillow", fps=fps)
            print(f"Animation GIF sauvegardée : {path}")
        elif suffix == ".mp4":
            if shutil.which("ffmpeg") is None:
                fallback = str(path.with_suffix(".gif"))
                print("ffmpeg introuvable. Génération GIF au lieu de MP4 ->", fallback)
                ani.save(fallback, writer="pillow", fps=fps)
                print(f"Animation GIF sauvegardée : {fallback}")
            else:
                ani.save(path, writer="ffmpeg", fps=fps)
                print(f"Animation MP4 sauvegardée : {path}")
        else:
            fallback = str(path.with_suffix(".gif"))
            print(f"Extension {suffix} non supportée, génération GIF : {fallback}")
            ani.save(fallback, writer="pillow", fps=fps)
            print(f"Animation GIF sauvegardée : {fallback}")
    except Exception as e:
        print("Échec de sauvegarde animation. Installez ffmpeg (mp4) ou pillow (gif).")
        print(f"Erreur: {e}")
    finally:
        plt.close(fig)


__all__ = [
    "generate_pluck_shape",
    "plot_waveform",
    "plot_fft",
    "create_animation",
]
