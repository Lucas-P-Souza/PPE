"""
Spectrogramme temps-fréquence en dB relatif.
"""
from __future__ import annotations

import numpy as np
try:
    import scipy as sp  # type: ignore
except Exception:
    sp = None  # type: ignore


def plot_spectrogram(
    signal: np.ndarray,
    dt: float,
    savepath: str,
    *,
    fmin: float = 0.0,
    fmax: float | None = None,
    nperseg: int | None = None,
    overlap_frac: float = 0.5,
    cmap: str = "magma",
    title: str = "Spectrogramme (dB)",
) -> None:
    """Trace un spectrogramme temps-fréquence en dB rel.
    Utilise scipy.signal.spectrogram si dispo; sinon, fallback avec STFT naïf numpy.
    """
    import matplotlib.pyplot as plt  # type: ignore

    y = np.asarray(signal, dtype=float).ravel()
    N = y.size
    if N < 8:
        print("[AVERTISSEMENT] Signal trop court pour spectrogramme.")
        return

    fs = 1.0 / float(dt)
    wlen = nperseg if (nperseg and nperseg > 0) else max(32, int(round(N / 20)))
    overlap = int(max(0, min(0.95, float(overlap_frac))) * wlen)

    Sxx = None; f = None; t = None
    if sp is not None and hasattr(sp, "signal"):
        try:
            f, t, Sxx = sp.signal.spectrogram(y, fs=fs, window="hann", nperseg=wlen, noverlap=overlap, detrend=False, scaling="spectrum", mode="magnitude")
        except Exception:
            Sxx = None

    if Sxx is None:
        # Fallback STFT simple
        step = max(1, wlen - overlap)
        win = np.hanning(wlen)
        n_frames = 1 + max(0, (N - wlen) // step)
        if n_frames < 1:
            n_frames = 1
            y = np.pad(y, (0, max(0, wlen - N)))
            n_eff = y.size
            N = n_eff
            step = wlen
        frames = np.stack([y[i:i+wlen] * win for i in range(0, N - wlen + 1, step)], axis=1)
        Z = np.fft.rfft(frames, axis=0)
        Sxx = np.abs(Z)
        f = np.fft.rfftfreq(wlen, d=dt)
        t = np.arange(Sxx.shape[1]) * (step / fs)

    # Bande
    mask = (f >= max(0.0, float(fmin))) & (f <= (fmax if fmax is not None else fs / 2.0))
    fp = f[mask]
    S = Sxx[mask, :]
    # dB relatif
    eps = 1e-12
    S /= (np.max(S) + eps)
    S_db = 20.0 * np.log10(np.maximum(S, eps))

    from pathlib import Path as _P
    out = _P(savepath)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    im = ax.pcolormesh(t, fp, S_db, shading="gouraud", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("temps (s)")
    ax.set_ylabel("fréquence (Hz)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Amplitude (dB rel)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[INFO] Spectrogramme enregistré dans : {out}")
