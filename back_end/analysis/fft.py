"""
Analyse FFT (spectre à un seul côté) et tracés associés.
Toutes les annotations et docstrings sont en français.
"""
from __future__ import annotations

import numpy as np
try:
    import scipy as sp  # type: ignore
except Exception:  # SciPy peut ne pas être disponible
    sp = None  # type: ignore

# --- FFT utilitaire (à un seul côté) ---
def calculer_fft_monocotee(
    y: np.ndarray,
    dt: float,
    *,
    window: str = "hann",
    zero_pad_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule le spectre d'amplitude à un seul côté avec fenêtre et échelle appropriée.
    Paramètres:
      - y: signal 1D
      - dt: pas d'échantillonnage (s)
      - window: type de fenêtre ("hann", "hamming", ...)
      - zero_pad_factor: facteur >=1.0 pour raffiner l'échelle fréquentielle
    Retourne (freqs, amp_lin).
    """
    y = np.asarray(y, dtype=float).ravel()
    N = y.size
    if N < 2:
        return np.array([]), np.array([])

    # Fenêtre
    win_name = (window or "hann").lower()
    try:
        if sp is not None and hasattr(sp, "signal"):
            w = sp.signal.get_window(win_name, N, fftbins=True)
        else:
            if win_name in ("hann", "hanning"):
                w = np.hanning(N)
            elif win_name == "hamming":
                w = np.hamming(N)
            elif win_name == "blackman":
                w = np.blackman(N)
            else:
                w = np.ones(N, dtype=float)
    except Exception:
        w = np.hanning(N)

    yw = y * w
    # Zero padding (améliore l'échantillonnage fréquentiel uniquement)
    try:
        factor = float(zero_pad_factor)
    except Exception:
        factor = 1.0
    factor = max(1.0, factor)
    target_len = int(np.ceil(N * factor))
    n_fft = 1 << (target_len - 1).bit_length()  # puissance de 2

    Y = np.fft.rfft(yw, n=n_fft)
    f = np.fft.rfftfreq(n_fft, d=dt)

    # Échelle d'amplitude via le gain cohérent de la fenêtre
    denom = float(np.sum(w))
    denom = denom if denom > 0 else float(N)
    amp = (2.0 / denom) * np.abs(Y)
    amp[0] *= 0.5
    if n_fft % 2 == 0 and amp.size > 1:
        amp[-1] *= 0.5
    return f, amp


# --- Tracé FFT en échelle linéaire + dB ---
def tracer_fft_png(
    signal: np.ndarray,
    dt: float,
    savepath: str,
    title: str = "FFT (déplacement)",
    fmax: float | None = None,
    *,
    fmin: float = 0.0,
    show_db: bool = True,
    smooth_window: int = 0,
    annotate_peaks: bool = True,
    n_peaks: int = 6,
    window: str = "hann",
    zero_pad_factor: float = 1.0,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    f, A = calculer_fft_monocotee(signal, dt, window=window, zero_pad_factor=zero_pad_factor)
    if f.size == 0:
        print("[AVERTISSEMENT] Signal trop court pour la FFT.")
        return

    Amax = float(np.max(A)) if A.size else 1.0
    A_lin = (A / Amax) if Amax > 0 else A

    # Bande
    mask = f >= float(fmin)
    f_plot = f[mask]
    A_plot = A_lin[mask]

    # Lissage optionnel
    A_smooth = None
    if int(smooth_window) and int(smooth_window) > 1 and A_plot.size:
        w = int(smooth_window)
        w = max(3, w + (w % 2 == 0))  # force impair >= 3
        kern = np.ones(w, dtype=float) / float(w)
        A_smooth = np.convolve(A_plot, kern, mode="same")

    # Détection de pics optionnelle
    peaks = []
    if annotate_peaks and A_plot.size:
        try:
            if sp is not None and hasattr(sp, "signal"):
                height_thr = max(0.05, 0.1 * float(np.max(A_plot)))
                idx, _ = sp.signal.find_peaks(A_plot, height=height_thr, distance=max(1, int(0.002 / max(dt, 1e-12))))
                cand = [(float(f_plot[i]), float(A_plot[i])) for i in idx]
            else:
                cand = []
        except Exception:
            cand = []
        if not cand:
            n_top = min(int(n_peaks) * 3, int(A_plot.size))
            if n_top > 0:
                top_idx = np.argpartition(A_plot, -n_top)[-n_top:]
                cand = [(float(f_plot[i]), float(A_plot[i])) for i in top_idx]
                cand.sort(key=lambda x: x[1], reverse=True)
        for f0, a0 in cand:
            if any(abs(f0 - fp) < 2.0 for fp, _ in peaks):
                continue
            peaks.append((f0, a0))
            if len(peaks) >= int(n_peaks):
                break

    from pathlib import Path as _P
    out = _P(savepath)
    out.parent.mkdir(parents=True, exist_ok=True)

    if show_db:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 4.2))
        ax2 = None  # type: ignore

    ax1.plot(f_plot, A_plot, lw=0.9, color="#1f77b4", alpha=0.85, label="FFT")
    if A_smooth is not None:
        ax1.plot(f_plot, A_smooth, lw=1.1, color="#ff7f0e", alpha=0.9, label="lissé")
    if peaks:
        for fp, ap in peaks:
            ax1.plot([fp], [ap], "ro", ms=3)
    ax1.set_ylabel("Amplitude (norm)")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    if A_smooth is not None:
        ax1.legend(loc="upper right", fontsize=8, framealpha=0.85)

    if ax2 is not None:
        eps = 1e-12
        A_db = 20.0 * np.log10(np.maximum(A_plot, eps))
        ax2.plot(f_plot, A_db, lw=0.9, color="#1f77b4")
        if A_smooth is not None:
            A_db_s = 20.0 * np.log10(np.maximum(A_smooth, eps))
            ax2.plot(f_plot, A_db_s, lw=1.1, color="#ff7f0e")
        for fp, _ in peaks:
            ax2.axvline(fp, color="red", lw=0.8, alpha=0.5)
        ax2.set_ylabel("Amplitude (dB rel)")
        ax2.grid(True, which="both", alpha=0.3)

    fs = 1.0 / float(dt)
    x_right = float(fmax) if fmax is not None else fs / 2.0
    if ax2 is not None:
        ax2.set_xlim(left=max(0.0, float(fmin)), right=x_right)
    ax1.set_xlim(left=max(0.0, float(fmin)), right=x_right)
    (ax2 or ax1).set_xlabel("fréquence (Hz)")

    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[INFO] FFT enregistrée dans : {out}")


# --- Tracé FFT log-fréquence rempli en dB ---
def tracer_fft_logdb_remplie(
    signal: np.ndarray,
    dt: float,
    savepath: str,
    *,
    fmin: float = 20.0,
    fmax: float | None = None,
    min_db: float = -90.0,
    smooth_window: int = 0,
    color: str = "#4c78a8",
    edgecolor: str | None = None,
    linewidth: float = 0.8,
    annotate_peaks: bool = True,
    n_peaks: int = 6,
    title: str = "FFT — échelle log en fréquence (dB)",
    db_offset: float = 0.0,
    log_bins_per_octave: int | None = None,
    octave_smoothing: float = 0.0,
    smooth_domain: str = "db",
    window: str = "hann",
    zero_pad_factor: float = 1.0,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    f, A = calculer_fft_monocotee(signal, dt, window=window, zero_pad_factor=zero_pad_factor)
    if f.size == 0:
        print("[AVERTISSEMENT] Signal trop court pour FFT (log dB).")
        return

    fmin = max(1e-3, float(fmin))
    fs = 1.0 / float(dt)
    x_right = float(fmax) if fmax is not None else fs / 2.0
    mask = (f >= fmin) & (f <= x_right)
    if not np.any(mask):
        print("[AVERTISSEMENT] Bande de fréquences invalide pour FFT log dB.")
        return
    fp = f[mask]
    Ap = A[mask]
    Amax = float(np.max(Ap)) if Ap.size else 1.0
    if Amax <= 0:
        Amax = 1.0
    Arel = Ap / Amax

    if int(smooth_window) and int(smooth_window) > 1 and Ap.size:
        w = int(smooth_window)
        w = max(3, w + (w % 2 == 0))
        kern = np.ones(w, dtype=float) / float(w)
        Arel = np.convolve(Arel, kern, mode="same")

    bpo = None if log_bins_per_octave is None else int(max(1, log_bins_per_octave))
    if bpo is not None and fp.size >= 5:
        fs = 1.0 / float(dt)
        x_right = float(fmax) if fmax is not None else fs / 2.0
        fmin_grid = max(fmin, float(fp[0]))
        if x_right <= fmin_grid * (1.0 + 1e-6):
            print("[AVERTISSEMENT] Grille log invalide pour FFT log dB.")
            return
        octaves = np.log2(x_right / fmin_grid)
        Nbins = int(np.floor(octaves * bpo)) + 1
        f_grid = np.geomspace(fmin_grid, x_right, Nbins)
        log_fp = np.log(fp)
        log_fg = np.log(f_grid)
        Arel_g = np.interp(log_fg, log_fp, Arel, left=Arel[0], right=Arel[-1])
        w_oct = float(max(0.0, octave_smoothing))
        if w_oct > 0.0 and Nbins >= 3:
            w_bins = int(round(w_oct * bpo))
            w_bins = max(3, w_bins + (w_bins % 2 == 0))
            if smooth_domain.lower() == "db":
                A_db_g = 20.0 * np.log10(np.maximum(Arel_g, 1e-12))
                kern = np.ones(w_bins, dtype=float) / float(w_bins)
                A_db_g = np.convolve(A_db_g, kern, mode="same")
                fp = f_grid
                Arel = np.power(10.0, A_db_g / 20.0)
            else:
                kern = np.ones(w_bins, dtype=float) / float(w_bins)
                Arel = np.convolve(Arel_g, kern, mode="same")
                fp = f_grid

    eps = 1e-12
    A_db = 20.0 * np.log10(np.maximum(Arel, eps)) + float(db_offset)
    A_db = np.maximum(A_db, float(min_db))

    peaks = []
    if annotate_peaks and Arel.size:
        try:
            if sp is not None and hasattr(sp, "signal"):
                thr = max(0.05, 0.1 * float(np.max(Arel)))
                idx, _ = sp.signal.find_peaks(Arel, height=thr, distance=max(1, int(0.002 / max(dt, 1e-12))))
                cand = [(float(fp[i]), float(Arel[i])) for i in idx]
            else:
                cand = []
        except Exception:
            cand = []
        if not cand:
            n_top = min(int(n_peaks) * 3, int(Arel.size))
            if n_top > 0:
                top_idx = np.argpartition(Arel, -n_top)[-n_top:]
                cand = [(float(fp[i]), float(Arel[i])) for i in top_idx]
                cand.sort(key=lambda x: x[1], reverse=True)
        for f0, a0 in cand:
            if any(abs(f0 - fx) < 2.0 for fx, _ in peaks):
                continue
            peaks.append((f0, a0))
            if len(peaks) >= int(n_peaks):
                break

    from pathlib import Path as _P
    out = _P(savepath)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.set_title(title)
    ax.set_xlabel("fréquence (Hz)")
    ax.set_ylabel("Amplitude (dB rel)")
    ax.set_xscale("log")
    ax.fill_between(fp, A_db, float(min_db), facecolor=color, edgecolor=edgecolor or color, linewidth=float(linewidth), alpha=0.85, step=None)
    ax.grid(True, which="both", axis="both", alpha=0.4)
    ax.set_xlim(left=fmin, right=x_right)
    ax.set_ylim(bottom=float(min_db), top=0.0)
    for fx, _ in peaks:
        ax.axvline(fx, color="#666666", lw=0.7, alpha=0.4)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[INFO] FFT (log f, dB, remplie) enregistrée dans : {out}")