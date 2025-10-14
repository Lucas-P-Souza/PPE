# Point d'entrée (historique) de la simulation backend.
#
# Lance la simulation FEM (maille non uniforme) avec amortissement de Rayleigh
# depuis config, intègre avec Newmark-beta et sauvegarde des graphiques/animations.
from __future__ import annotations

from pathlib import Path
import numpy as np

try:
    from digital_twin.back_end import config as _cfg  # type: ignore
    from digital_twin.back_end.fem.formulation import build_global_mkc_from_config  # type: ignore
    from digital_twin.back_end.utils.validators import valider_mck  # type: ignore
    from digital_twin.back_end.fem.time_integration import (  # type: ignore
        definir_parametres_simulation,
        initialiser_etats_initiaux,
        calculer_u1,
        fournisseur_force_nulle,
        fournisseur_force_localisee,
        somme_de_forces,
        integrer_newmark_beta,
        calculer_energies_dans_le_temps,
    )
    # Importer depuis les nouveaux modules organisés
    from digital_twin.back_end.fem.solver import build_node_positions_from_config  # type: ignore
    from digital_twin.back_end.fem.modal import compute_modal_frequencies_and_modes  # type: ignore
    from digital_twin.back_end.analysis.fft import tracer_fft_png, tracer_fft_logdb_remplie  # type: ignore
    from digital_twin.back_end.analysis.spectrogram import plot_spectrogram  # type: ignore
    from digital_twin.back_end.viz.plots import plot_first_modes, plot_snapshots_png, save_string_frame_png  # type: ignore
    from digital_twin.back_end.viz.anim import animate_string_motion  # type: ignore
    from digital_twin.back_end.io.exports import save_displacement_csv  # type: ignore
except ModuleNotFoundError:
    # Fallback when executed as a standalone script (ensure workspace root on sys.path)
    import sys as _sys
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    from digital_twin.back_end import config as _cfg  # type: ignore
    from digital_twin.back_end.fem.formulation import build_global_mkc_from_config  # type: ignore
    from digital_twin.back_end.utils.validators import valider_mck  # type: ignore
    from digital_twin.back_end.fem.time_integration import (  # type: ignore
        definir_parametres_simulation,
        initialiser_etats_initiaux,
        calculer_u1,
        fournisseur_force_nulle,
        fournisseur_force_localisee,
        somme_de_forces,
        integrer_newmark_beta,
        calculer_energies_dans_le_temps,
    )
    from digital_twin.back_end.fem.solver import build_node_positions_from_config  # type: ignore
    from digital_twin.back_end.fem.modal import compute_modal_frequencies_and_modes  # type: ignore
    from digital_twin.back_end.analysis.fft import tracer_fft_png, tracer_fft_logdb_remplie  # type: ignore
    from digital_twin.back_end.analysis.spectrogram import plot_spectrogram  # type: ignore
    from digital_twin.back_end.viz.plots import plot_first_modes, plot_snapshots_png, save_string_frame_png  # type: ignore
    from digital_twin.back_end.viz.anim import animate_string_motion  # type: ignore
    from digital_twin.back_end.io.exports import save_displacement_csv  # type: ignore


def main() -> None:
    ROOT = Path(__file__).resolve().parents[2]
    # Build M, K, C from config (frets mesh mandatory here)
    res = build_global_mkc_from_config(apply_fixed_bc=True, return_meta=True)
    if not (isinstance(res, tuple) and len(res) >= 3):
        raise RuntimeError("Retorno inesperado de build_global_mkc_from_config")
    if len(res) == 4:
        M, K, C, meta = res
    else:
        M, K, C = res[:3]
        meta = {}
    print("[INFO] Matrices assemblées à partir du config.")

    # Validate shapes, symmetry and boundary conditions
    valider_mck(M, C, K, verbose=True)

    # Simulation parameters
    delta_t = float(getattr(_cfg, "DT", 1e-5))
    T_total = float(getattr(_cfg, "T_SIM", 0.1))
    definir_parametres_simulation(delta_t, T_total)

    # Initial conditions: string initially flat and at rest (U0=0, V0=0)
    L_eff = float(getattr(_cfg, "L", 1.0))
    U0 = np.zeros(M.shape[0], dtype=float)
    U_nm1 = U0.copy()
    U_n = U0.copy()
    print(f"U0 initialisé: shape={U0.shape}, max={np.nanmax(U0):.3e}, min={np.nanmin(U0):.3e}")

    # First step via central differences (diagnostic)
    _ = calculer_u1(M, C, K, U_n=U_n, U_nm1=U_nm1, delta_t=delta_t)

    # Modal analysis and plot of first modes
    n = M.shape[0]
    x_coords = build_node_positions_from_config(n)
    freqs_hz, modes_full = compute_modal_frequencies_and_modes(M, K, num_modes=4)
    print("Premières fréquences (Hz):", np.round(freqs_hz, 3))
    plots_dir = ROOT / "digital_twin" / "back_end" / "results" / "plots"
    if bool(getattr(_cfg, "OUTPUT_ENABLE_IMAGES", True)):
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_first_modes(x_coords, modes_full, freqs_hz, max_modes=4, savepath=str(plots_dir / "modes_first4.png"))

    # Newmark-beta integration with external localized force(s)
    # External localized force (trapezoidal envelope) at node derived from PLUCK_POS
    V0_zero = np.zeros(n, dtype=float)
    # Map PLUCK_POS in [0,1] to a node index (closest)
    x_coords = build_node_positions_from_config(n)
    x_p_rel = float(getattr(_cfg, "PLUCK_POS", 0.25))
    x_force = float(x_p_rel * L_eff)
    i_force = int(np.argmin(np.abs(x_coords - x_force)))
    # Envelope parameters (fallback defaults)
    F_max = float(getattr(_cfg, "EXCITATION_F_MAX", 1.0))
    t_rise = float(getattr(_cfg, "EXCITATION_T_RISE", 0.01))
    t_hold = float(getattr(_cfg, "EXCITATION_T_HOLD", 0.03))
    t_decay = float(getattr(_cfg, "EXCITATION_T_DECAY", 0.005))
    # First excitation at t0 = 0
    F1 = fournisseur_force_localisee(n, i_force, F_max, t_rise, t_hold, t_decay, t0=0.0)
    # Second excitation at configurable time (ensure T_SIM > second_t0 + envelope)
    #second_t0 = float(getattr(_cfg, "EXCITATION_SECOND_T0", 0.6))
    second_t0 = None
    if second_t0 is not None and second_t0 > 0.0:
        F2 = fournisseur_force_localisee(n, i_force, F_max, t_rise, t_hold, t_decay, t0=second_t0)
        F_total = somme_de_forces(n, F1, F2)
    else:
        F_total = F1

    # Helper: shift a force provider by a global time offset (for chunked runs)
    def decaler_force(F_base, t_offset: float):
        def F_shift(t: float, k: int):
            return F_base(float(t) + float(t_offset), k)
        return F_shift

    # Option: extend simulation in chunks until motion decays (or max seconds)
    if bool(getattr(_cfg, "AUTO_EXTEND_SIM", False)):
        max_total = float(getattr(_cfg, "MAX_SIM_SECONDS", max(T_total, 2.0)))
        chunk_sec = float(getattr(_cfg, "CHUNK_SECONDS", T_total))
        stop_win = float(getattr(_cfg, "STOP_WINDOW_SEC", 0.2))
        th_u = float(getattr(_cfg, "STOP_THRESH_U", 1e-6))
        th_v = float(getattr(_cfg, "STOP_THRESH_V", 1e-4))

        t_lists: list[np.ndarray] = []
        U_lists: list[np.ndarray] = []
        V_lists: list[np.ndarray] = []
        A_lists: list[np.ndarray] = []

        t_acc = 0.0
        U_prev = U0.copy()
        V_prev = V0_zero.copy()
        A_prev = None  # let integrator compute from F at each chunk start
        first_chunk = True
        while t_acc < max_total - 1e-15:
            t_chunk = min(chunk_sec, max_total - t_acc)
            F_shift = decaler_force(F_total, t_acc)
            t_loc, U_loc, V_loc, A_loc = integrer_newmark_beta(
                M, C, K, F_shift, dt=delta_t, t_max=t_chunk, U0=U_prev, V0=V_prev, A0=A_prev
            )
            # Convert to global time and append (avoid duplicating the initial sample after the first chunk)
            if first_chunk:
                t_lists.append(t_loc + t_acc)
                U_lists.append(U_loc)
                V_lists.append(V_loc)
                A_lists.append(A_loc)
                first_chunk = False
            else:
                t_lists.append(t_loc[1:] + t_acc)
                U_lists.append(U_loc[:, 1:])
                V_lists.append(V_loc[:, 1:])
                A_lists.append(A_loc[:, 1:])

            # Prepare for next chunk
            U_prev = U_loc[:, -1]
            V_prev = V_loc[:, -1]
            A_prev = A_loc[:, -1]

            # Stop condition: check last window within this chunk
            if stop_win > 0.0:
                n_last = max(1, int(round(stop_win / delta_t)))
                U_tail = U_loc[:, -n_last:]
                V_tail = V_loc[:, -n_last:]
                if np.nanmax(np.abs(U_tail)) < th_u and np.nanmax(np.abs(V_tail)) < th_v:
                    t_acc += t_loc[-1]
                    break

            t_acc += t_loc[-1]

        # Concatenate histories
        t_vec = np.concatenate(t_lists, axis=0)
        U_hist = np.concatenate(U_lists, axis=1)
        V_hist = np.concatenate(V_lists, axis=1)
        A_hist = np.concatenate(A_lists, axis=1)
    else:
        # Single-shot integration over T_total
        t_vec, U_hist, V_hist, A_hist = integrer_newmark_beta(
            M, C, K, F_total, dt=delta_t, t_max=T_total, U0=U0, V0=V0_zero, A0=None
        )

    # Save CSV of string positions over time
    if bool(getattr(_cfg, "OUTPUT_ENABLE_CSV", True)):
        plots_dir.mkdir(parents=True, exist_ok=True)
        csv_path = plots_dir / "string_positions.csv"
        save_displacement_csv(t_vec, U_hist, str(csv_path))

    # Displacement over time for a representative node
    if bool(getattr(_cfg, "OUTPUT_ENABLE_IMAGES", True)):
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as _eplt:  # pragma: no cover
            print("[AVERTISSEMENT] matplotlib indisponible pour tracer x(t):", _eplt)
        else:
            node_idx = max(1, n // 3)
            plt.figure(figsize=(8, 4))
            plt.plot(t_vec, U_hist[node_idx, :], lw=1.2)
            plt.title(f"Déplacement au nœud {node_idx} — Newmark (β=1/4, γ=1/2), U0=V0=0; force localisée")
            plt.xlabel("temps (s)")
            plt.ylabel("x (m)")
            plt.grid(True, alpha=0.3)
            outp = plots_dir / "newmark_node_displacement.png"
            plt.tight_layout(); plt.savefig(outp, dpi=150); plt.close()
            print(f"[INFO] Tracé de x(t) enregistré dans: {outp}")

            # Énergies
            Ek, Ep, Et = calculer_energies_dans_le_temps(M, K, U_hist, V_hist)
            plt.figure(figsize=(9, 5))
            plt.plot(t_vec, Ek, label="E cinétique")
            plt.plot(t_vec, Ep, label="E potentielle")
            plt.plot(t_vec, Et, label="E totale", lw=1.8)
            plt.title("Énergies au cours du temps")
            plt.xlabel("temps (s)")
            plt.ylabel("énergie (J)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            outpE = plots_dir / "newmark_energies.png"
            plt.tight_layout(); plt.savefig(outpE, dpi=150); plt.close()
            print(f"[INFO] Tracé des énergies enregistré dans: {outpE}")

    # FFT at output node
    out_node = int(getattr(_cfg, "OUTPUT_NODE", max(1, n // 2)))
    out_node = max(0, min(n - 1, out_node))
    if bool(getattr(_cfg, "OUTPUT_ENABLE_IMAGES", True)):
        plots_dir.mkdir(parents=True, exist_ok=True)
        # Choose FFT style
        _style = str(getattr(_cfg, "FFT_STYLE", "logdb")).lower()
        if _style == "linear":
            fft_path = plots_dir / "newmark_output_fft.png"
            tracer_fft_png(U_hist[out_node, :], delta_t, str(fft_path), title=f"FFT — nœud {out_node}", fmax=None)
        else:
            fft_logdb_path = plots_dir / "newmark_output_fft_logdb.png"
            try:
                fmin_cfg = float(getattr(_cfg, "FFT_LOG_FMIN", 30.0))
                fmax_cfg = getattr(_cfg, "FFT_LOG_FMAX", None)
                fmax_cfg = None if fmax_cfg is None else float(fmax_cfg)
                min_db_cfg = float(getattr(_cfg, "FFT_LOG_MIN_DB", -90.0))
                smooth_w_cfg = int(getattr(_cfg, "FFT_LOG_SMOOTH", 0))
                use_velocity = bool(getattr(_cfg, "FFT_LOG_USE_VELOCITY", False))
                color_hex = str(getattr(_cfg, "FFT_LOG_COLOR", "#4c78a8"))
                db_offset_cfg = float(getattr(_cfg, "FFT_LOG_DB_OFFSET", 0.0))
                bpo_cfg = getattr(_cfg, "FFT_LOG_BINS_PER_OCTAVE", None)
                bpo_cfg = None if bpo_cfg is None else int(bpo_cfg)
                oct_smooth_cfg = float(getattr(_cfg, "FFT_LOG_OCTAVE_SMOOTH", 0.0))
                smooth_domain_cfg = str(getattr(_cfg, "FFT_LOG_SMOOTH_DOMAIN", "db"))
            except Exception:
                fmin_cfg, fmax_cfg, min_db_cfg, smooth_w_cfg = 30.0, None, -90.0, 0
                use_velocity, color_hex, db_offset_cfg = False, "#4c78a8", 0.0
                bpo_cfg, oct_smooth_cfg, smooth_domain_cfg = None, 0.0, "db"
            _sig = V_hist[out_node, :] if use_velocity else U_hist[out_node, :]
            tracer_fft_logdb_remplie(_sig, delta_t, str(fft_logdb_path), fmin=fmin_cfg, fmax=fmax_cfg, min_db=min_db_cfg, smooth_window=smooth_w_cfg, color=color_hex, annotate_peaks=True, n_peaks=8, title=f"FFT — nœud {out_node} (log f, dB)", db_offset=db_offset_cfg, log_bins_per_octave=bpo_cfg, octave_smoothing=oct_smooth_cfg, smooth_domain=smooth_domain_cfg)

        # Optional spectrogram
        try:
            if bool(getattr(_cfg, "ENABLE_SPECTROGRAM", False)):
                sp_fmin = float(getattr(_cfg, "SPECTROGRAM_FMIN", 0.0))
                sp_fmax = getattr(_cfg, "SPECTROGRAM_FMAX", None)
                sp_fmax = None if sp_fmax is None else float(sp_fmax)
                sp_nperseg = getattr(_cfg, "SPECTROGRAM_NPERSEG", None)
                sp_nperseg = None if sp_nperseg in (None, 0) else int(sp_nperseg)
                sp_ovlp = float(getattr(_cfg, "SPECTROGRAM_OVERLAP", 0.5))
                sp_cmap = str(getattr(_cfg, "SPECTROGRAM_CMAP", "magma"))
                sp_path = plots_dir / "newmark_output_spectrogram.png"
                plot_spectrogram(U_hist[out_node, :], delta_t, str(sp_path), fmin=sp_fmin, fmax=sp_fmax, nperseg=sp_nperseg, overlap_frac=sp_ovlp, cmap=sp_cmap)
        except Exception:
            pass

    # Animation and first-frame PNG
    if bool(getattr(_cfg, "OUTPUT_ENABLE_GIFS", True)):
        plots_dir.mkdir(parents=True, exist_ok=True)
        # -- Old single GIF (commented):
        # anim_path = plots_dir / "string_motion.gif"
        # animate_string_motion(
        #     x_coords,
        #     U_hist,
        #     interval_ms=30,
        #     decim=5,
        #     savepath=str(anim_path),
        #     show=False,
        #     y_scale=_y_scale,
        #     y_pad_frac=_y_pad,
        # )

        try:
            _y_scale = float(getattr(_cfg, "ANIM_Y_SCALE", 1.0))
            _y_pad = float(getattr(_cfg, "ANIM_Y_PAD_FRAC", 0.05))
            _fps_slow = int(getattr(_cfg, "ANIM_FPS_SLOW", 33))
            _fps_real = int(getattr(_cfg, "ANIM_FPS_REAL", 30))
            _decim_slow = int(getattr(_cfg, "ANIM_DECIM_SLOW", 5))
        except Exception:
            _y_scale, _y_pad = 1.0, 0.05
            _fps_slow, _fps_real = 33, 30
            _decim_slow = 5

        # Compute interval in ms from FPS
        _int_ms_slow = max(1, int(round(1000.0 / max(1, _fps_slow))))
        _int_ms_real = max(1, int(round(1000.0 / max(1, _fps_real))))

        # Compute decimation for true real-time playback:
        # decim_real ≈ 1 / (dt * fps)
        decim_real = max(1, int(round(1.0 / (delta_t * max(1, _fps_real)))))
        # Slow-motion options priority:
        # A) Absolute duration target: ANIM_SLOW_DURATION_S
        # B) Relative factor: ANIM_SLOW_FACTOR (duration_slow ≈ factor * duration_real)
        # C) Fallback: ANIM_DECIM_SLOW heuristic
        total_steps = int(U_hist.shape[1])
        slow_duration_target = getattr(_cfg, "ANIM_SLOW_DURATION_S", None)
        if slow_duration_target is not None:
            try:
                T_target = max(1.0, float(slow_duration_target))
                # frames_needed = T_target * fps_slow
                frames_needed = max(1.0, T_target * float(_fps_slow))
                # decim_slow ≈ total_steps / frames_needed
                decim_slow = max(1, int(round(float(total_steps) / frames_needed)))
            except Exception:
                decim_slow = max(1, decim_real // max(1, _decim_slow))
        else:
            _slow_factor = getattr(_cfg, "ANIM_SLOW_FACTOR", None)
            if _slow_factor is not None:
                try:
                    SF = max(1.0, float(_slow_factor))
                except Exception:
                    SF = float(max(1, _decim_slow))  # robust fallback
                decim_slow = max(1, int(round(decim_real * (float(_fps_real) / max(1.0, float(_fps_slow) * SF)))))
            else:
                # Fallback: smaller decimation than real-time → more frames → slower playback
                decim_slow = max(1, decim_real // max(1, _decim_slow))

        # Debug info: effective durations and ratio
        frames_real = max(1, total_steps // int(decim_real))
        frames_slow = max(1, total_steps // int(decim_slow))
        dur_real = frames_real / max(1, _fps_real)
        dur_slow = frames_slow / max(1, _fps_slow)
        ratio = dur_slow / max(1e-12, dur_real)
        print(f"[ANIM] real: decim={decim_real}, fps={_fps_real}, frames={frames_real}, duration≈{dur_real:.2f}s")
        print(f"[ANIM] slow: decim={decim_slow}, fps={_fps_slow}, frames={frames_slow}, duration≈{dur_slow:.2f}s (x{ratio:.2f} slower)")

        # 1) Real-time GIF
        anim_real = plots_dir / "string_motion_real.gif"
        animate_string_motion(
            x_coords,
            U_hist,
            interval_ms=_int_ms_real,
            decim=decim_real,
            savepath=str(anim_real),
            show=False,
            y_scale=_y_scale,
            y_pad_frac=_y_pad,
        )

        # 2) Slow GIF (decimated frames)
        anim_slow = plots_dir / "string_motion_slow.gif"
        animate_string_motion(
            x_coords,
            U_hist,
            interval_ms=_int_ms_slow,
            decim=decim_slow,
            savepath=str(anim_slow),
            show=False,
            y_scale=_y_scale,
            y_pad_frac=_y_pad,
        )

        if bool(getattr(_cfg, "OUTPUT_ENABLE_IMAGES", True)):
            png0 = plots_dir / "string_motion_t0.png"
            try:
                _frame_y_scale = getattr(_cfg, "ANIM_Y_SCALE", None)
            except Exception:
                _frame_y_scale = None
            save_string_frame_png(x_coords, U_hist, frame_idx=0, savepath=str(png0), y_scale=_frame_y_scale, y_pad_frac=_y_pad)

    # Static multi-time snapshots PNG
    try:
        n_snap = int(getattr(_cfg, "SNAPSHOTS_COUNT", 8))
        snap_decim = int(getattr(_cfg, "SNAPSHOTS_DECIM", 10))
    except Exception:
        n_snap, snap_decim = 8, 10
    try:
        _snap_y_scale = getattr(_cfg, "SNAPSHOTS_Y_SCALE", None)
        _snap_y_pad = float(getattr(_cfg, "SNAPSHOTS_Y_PAD_FRAC", 0.06))
        _snap_t_window = getattr(_cfg, "SNAPSHOTS_T_WINDOW", None)
        _snap_use_cbar = bool(getattr(_cfg, "SNAPSHOTS_USE_COLORBAR", True))
        _snap_cmap = str(getattr(_cfg, "SNAPSHOTS_CMAP", "viridis"))
        _snap_alpha = float(getattr(_cfg, "SNAPSHOTS_ALPHA", 0.9))
        _snap_lw = float(getattr(_cfg, "SNAPSHOTS_LINEWIDTH", 1.3))
        _snap_show_legend = bool(getattr(_cfg, "SNAPSHOTS_SHOW_LEGEND", False))
    except Exception:
        _snap_y_scale = None
        _snap_y_pad = 0.06
        _snap_t_window = None
        _snap_use_cbar = True
        _snap_cmap = "viridis"
        _snap_alpha = 0.9
        _snap_lw = 1.3
        _snap_show_legend = False
    if bool(getattr(_cfg, "OUTPUT_ENABLE_IMAGES", True)):
        plots_dir.mkdir(parents=True, exist_ok=True)
        snap_path = plots_dir / "string_snapshots.png"
        plot_snapshots_png(
            x_coords,
            U_hist,
            t_vec,
            n_snapshots=n_snap,
            decim=snap_decim,
            savepath=str(snap_path),
            title="Profils de la corde à des temps sélectionnés",
            y_scale=_snap_y_scale,
            y_pad_frac=_snap_y_pad,
            t_window=_snap_t_window,
            use_colorbar=_snap_use_cbar,
            cmap=_snap_cmap,
            alpha=_snap_alpha,
            linewidth=_snap_lw,
            show_legend=_snap_show_legend,
        )


if __name__ == "__main__":
    main()
