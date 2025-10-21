from __future__ import annotations
from pathlib import Path
import numpy as np

# Tentative d'importation normale (exécution via package)
try:
    from digital_twin.back_end import config as _cfg  
    from digital_twin.back_end.utils.validators import valider_mck  
    from digital_twin.back_end.io import enregistrer_deplacement_csv  
    from digital_twin.back_end.interactions.press import PressEvent, simulate_with_press
    from digital_twin.back_end.audio.generate_audio_from_positions import generate_multiple_positions_audio
    from digital_twin.back_end.fem.modal import (
        detecter_ddl_contraints_mk,
        calculer_frequences_et_modes,
    )
    from digital_twin.back_end.fem.formulation import (
        build_global_mkc_from_config,
        build_node_positions_from_config,
        amortissement_rayleigh,
    )
    from digital_twin.back_end.fem.time_integration import (
        definir_parametres_simulation,
        calculer_u1,
        fournisseur_force_localisee,
        somme_de_forces,
        integrer_newmark_beta,
        calculer_energies_dans_le_temps,
    )

# Si l'importation échoue (exécution directe), ajuster le sys.path
except ModuleNotFoundError:
    import sys as _sys
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    from digital_twin.back_end import config as _cfg
    from digital_twin.back_end.utils.validators import valider_mck  
    from digital_twin.back_end.io import enregistrer_deplacement_csv
    from digital_twin.back_end.interactions.press import PressEvent, simulate_with_press
    from digital_twin.back_end.audio.generate_audio_from_positions import generate_multiple_positions_audio
    from digital_twin.back_end.fem.modal import (
        detecter_ddl_contraints_mk,
        calculer_frequences_et_modes,
    )
    from digital_twin.back_end.fem.formulation import (
        build_global_mkc_from_config,
        build_node_positions_from_config,
        amortissement_rayleigh,
    )
    from digital_twin.back_end.fem.time_integration import (
        definir_parametres_simulation,
        calculer_u1,
        fournisseur_force_localisee,
        somme_de_forces,
        integrer_newmark_beta,
        calculer_energies_dans_le_temps,
    )

def main() -> None:

    # Définir le chemin racine du projet pour les sorties
    ROOT = Path(__file__).resolve().parents[2]

    # ====================================================================
    #       SECTION 1 — CALCUL : Assemblage de M, C, K et validation
    #       (Aucun tracé ici ; uniquement la construction numérique)
    # ====================================================================
    
    # Assemblage de M, K, C depuis la configuration (maillage des frettes requis)
    res = build_global_mkc_from_config(apply_fixed_bc=True, return_meta=True)
    
    # Vérifier le format de retour
    if not (isinstance(res, tuple) and len(res) >= 3):
        # Si le format est incorrect, lever une erreur explicite et casser l'exécution
        raise RuntimeError("Retour inattendu de build_global_mkc_from_config")
    
    # Déballer les matrices et les méta-données
    if len(res) == 4:
        M, K, C, meta = res
    # En cas d'absence de méta-données, initialiser un dictionnaire vide
    else:
        M, K, C = res[:3]
        meta = {}           # juste pour typer correctement

    # Debug: afficher un message si activé
    if getattr(_cfg, "DEBUG_ENABLED", False):
        print("[INFO] Matrices assemblées depuis la configuration.")

    # Validate shapes, symmetry and boundary conditions
    valider_mck(M, C, K, verbose=True)

    # =========================================================
    #       SECTION 2 — CALCUL : Paramètres de simulation
    # =========================================================

    # Paramètres pour la simulation temporelle
    delta_t = float(getattr(_cfg, "DT", 1e-4))                      # pas de temps
    nos_press = [6, 11, 17, 21, 25, 29, 34, 38, 41, 44, 48, 51]     # nœuds à presser successivement
    dur_press = 1                                                   # durée de la pression à chaque nœud
    t0 = 1.0                                                        # temps de début du premier press
    ti = 1.0                                                        # délai entre press et pincement
    tp = 1.0                                                        # temps de pause entre press

    # Initialiser les listes d'événements de pression et des temps d'excitation
    exc_times = []
    press_events = []

    # Construire les événements de pression et les temps d'excitation
    for idx, node_press in enumerate(nos_press, start=1):

        # Temps d'activation et de désactivation de la pression
        t_press_on = t0                         # début de la pression
        t_pinc = t_press_on + ti                # pincement juste après pression
        t_press_off = t_press_on + dur_press    # fin de la pression

        # Ajouter le temps de pincement à la liste des excitations
        exc_times.append(t_pinc)                # pincement à ce temps

        # Créer et ajouter l'événement de pression à la liste
        press_events.append(PressEvent(node=node_press, t_on=t_press_on, t_off=t_press_off, ks=5e4, cs=0.0))

        # Après le 6ème press (idx == 6) ajouter 1s extra de pause avant le prochain press
        if idx == 6:
            t0 = t_press_off + tp # pause supplémentaire avant le prochain press
        else:
            t0 = t_press_off  # prochain nœud pressé immédiatement après avoir relâché le précédent
    
    # Durée totale de la simulation (incluant tous les press et pincements)
    T_total = t0 + 0.5
    definir_parametres_simulation(delta_t, T_total)

    # ==============================================
    # SECTION 3 — CALCUL : Conditions initiales et premier pas
    # ==============================================
    # Conditions initiales : corde initialement à plat et au repos (U0=0, V0=0)
    L_eff = float(getattr(_cfg, "L", 1.0))
    U0 = np.zeros(M.shape[0], dtype=float)
    U_nm1 = U0.copy()
    U_n = U0.copy()
    if getattr(_cfg, "DEBUG_ENABLED", False):
        print(f"U0 initialisé: shape={U0.shape}, max={np.nanmax(U0):.3e}, min={np.nanmin(U0):.3e}")

    # Premier pas via différences centrales (diagnostic)
    _ = calculer_u1(M, C, K, U_n=U_n, U_nm1=U_nm1, delta_t=delta_t)

    # ==============================================
    # SECTION 4 — CALCUL : Analyse modale (calcul uniquement)
    # (Le tracé des modes sera fait en POST-TRAITEMENT)
    # ==============================================
    # Analyse modale
    n = M.shape[0]
    x_coords = build_node_positions_from_config(n)
    freqs_hz, modes_full = calculer_frequences_et_modes(M, K, num_modes=4)
    if (CONFIG := getattr(_cfg, "DEBUG_ENABLED", False)):
        print("Premières fréquences (Hz) :", np.round(freqs_hz, 3))

    # ==============================================
    # SECTION 5 — CALCUL : Définition des forces externes et intégration
    # (Aucun tracé ici ; uniquement l'intégration Newmark-β)
    # ==============================================
    # Intégration Newmark-β avec force(s) localisée(s)
    # Force localisée (enveloppe trapézoïdale) au nœud dérivé de PLUCK_POS
    V0_zero = np.zeros(n, dtype=float)
    # Projection de PLUCK_POS dans [0,1] vers l'indice de nœud le plus proche
    x_coords = build_node_positions_from_config(n)
    x_p_rel = float(getattr(_cfg, "PLUCK_POS", 0.25))
    x_force = float(x_p_rel * L_eff)
    i_force = int(np.argmin(np.abs(x_coords - x_force)))
    # Paramètres d'enveloppe (valeurs par défaut de repli)
    F_max = float(getattr(_cfg, "EXCITATION_F_MAX", 1.0))
    t_rise = float(getattr(_cfg, "EXCITATION_T_RISE", 0.01))
    t_hold = float(getattr(_cfg, "EXCITATION_T_HOLD", 0.03))
    t_decay = float(getattr(_cfg, "EXCITATION_T_DECAY", 0.005))
    # -------------------------------------------------------------------------
    # Scénario demandé (chronologie):
    #   1) Exciter la corde (pincement) à t = 0
    #   2) Attendre un peu (pas de force)
    #   3) Appuyer une note (pression locale) pendant une fenêtre [t_on, t_off]
    #   4) Toujours avec la note APPUYÉE, attendre encore un peu
    #   5) Toujours avec la note APPUYÉE, exciter la corde à nouveau (deuxième pincement)
    #
    # Mise en œuvre:
    #   - Les « attentes » sont des intervalles sans force externe.
    #   - La « pression » est modélisée par un PressEvent (raideur locale au nœud choisi).
    #   - Le 2e pincement se produit À L'INTÉRIEUR de la fenêtre de pression.
    # -------------------------------------------------------------------------

    # Gerar fornecedores de força para cada pincamento
    F_list = [fournisseur_force_localisee(n, i_force, F_max, t_rise, t_hold, t_decay, t0=tt) for tt in exc_times]
    F_total = F_list[0]
    for F in F_list[1:]:
        F_total = somme_de_forces(n, F_total, F)

    # Aide : décaler un fournisseur de force par un offset temporel global (exécution par segments)
    def decaler_force(F_base, t_offset: float):
        def F_shift(t: float, k: int):
            return F_base(float(t) + float(t_offset), k)
        return F_shift

    # Activer la PRESSION (appui de note) pendant [press_t_on, press_t_off]
    #  - Étendre press_t_off si nécessaire pour couvrir TOUTE l'enveloppe du 2e pincement
    #    (afin que le 2e pincement se passe encore avec la note appuyée)
    #  - Choix du nœud à appuyer: config.PRESS_NODE_INDEX (sinon ~30% de la longueur, en nœud interne)
    #  - Raideur locale et amortissement additionnel: PRESS_KS et PRESS_CS (cs=0 → amortissement Rayleigh seulement)
    # ...existing code...

    # Paramètres de Rayleigh pour les segments: (α, β) issus de meta si possible; sinon recalcul par sécurité
    try:
        alpha = float(meta.get('alpha')) if isinstance(meta, dict) and ('alpha' in meta) else None  # type: ignore[arg-type]
        beta = float(meta.get('beta')) if isinstance(meta, dict) and ('beta' in meta) else None  # type: ignore[arg-type]
    except Exception:
        alpha, beta = None, None
    if (alpha is None) or (beta is None):
        try:
            modes_ref = getattr(_cfg, "DAMPING_MODES_REF")
            zetas_ref = getattr(_cfg, "DAMPING_ZETAS_REF")
            alpha, beta, _, _ = amortissement_rayleigh(M, K, modes_ref, zetas_ref)
        except Exception as _e_rb:  # pragma: no cover
            raise RuntimeError("Impossible de déterminer α et β (Rayleigh) pour la simulation avec pression") from _e_rb

        # --- Sanity-check des PressEvent: garantir indices valides et pas de nœuds d'extrémité ---
        valid_press_events: list[PressEvent] = []
        for ev in press_events:
            if not isinstance(ev.node, int):
                print(f"[WARN] PressEvent ignoré: node non int ({ev.node})")
                continue
            if ev.node <= 0 or ev.node >= (M.shape[0] - 1):
                # interdit d'appuyer sur nœuds d'extrémité (Dirichlet)
                print(f"[WARN] PressEvent ignoré: node {ev.node} est extrême ou hors-limites (n={M.shape[0]})")
                continue
            # valider aussi t_on < t_off
            if ev.t_off <= ev.t_on:
                print(f"[WARN] PressEvent ignoré: t_off <= t_on pour node {ev.node}")
                continue
            valid_press_events.append(ev)

        if not valid_press_events and bool(getattr(_cfg, "PRESS_EVENTS_ENABLED", False)):
            print("[INFO] Aucun PressEvent valide — désactivation des événements de pression")
            # proceed without press events
            press_events = []
        else:
            press_events = valid_press_events

    # Choix du flux d'intégration selon la configuration: avec ou sans événements de pression
    if bool(getattr(_cfg, "PRESS_EVENTS_ENABLED", False)):
        if getattr(_cfg, "DEBUG_ENABLED", False):
            print("[INFO] PRESS_EVENTS_ENABLED=True → simulation with press events (simulate_with_press)")
        t_vec, U_hist, V_hist, A_hist = simulate_with_press(
            M, K, alpha, beta, F_total, delta_t, press_events, T_total, U0=U0, V0=V0_zero
        )
    else:
        # Flux normal: intégration unique avec la force résultante F_total (fonction F(t,k))
        if getattr(_cfg, "DEBUG_ENABLED", False):
            print("[INFO] PRESS_EVENTS_ENABLED=False → normal simulation (integrer_newmark_beta)")
        t_vec, U_hist, V_hist, A_hist = integrer_newmark_beta(
            M, C, K, F_total, delta_t, T_total, U0=U0, V0=V0_zero
        )

    # ==============================================
    # SECTION 6 — POST-TRAITEMENT : enregistrer CSV, générer PLOTS, FFT, GIFs
    # (UNIQUEMENT des sorties visuelles/fichiers — pas de nouveau calcul de dynamique)
    # ==============================================
    plots_dir = ROOT / "digital_twin" / "back_end" / "results" / "plots"

    # 6.1) CSV des déplacements en fonction du temps (sortie de données)
    if bool(getattr(_cfg, "OUTPUT_ENABLE_CSV", True)):
        plots_dir.mkdir(parents=True, exist_ok=True)
        csv_path = plots_dir / "string_positions.csv"
        # write CSV using the I/O package (lightweight)
        enregistrer_deplacement_csv(t_vec, U_hist, str(csv_path))
        # generate audio files from the CSV just produced
        try:
            generate_multiple_positions_audio(str(csv_path))
        except Exception as _ea:  # pragma: no cover
            print(f"[WARN] Unable to generate audio from positions: {_ea}")
        

    # 6.2) PLOTS — Modes and other heavy post-processing
    if bool(getattr(_cfg, "OUTPUT_ENABLE_IMAGES", True)):
        # import heavy plotting/fft modules lazily to avoid large imports during simulation
        from digital_twin.back_end.viz.plots import plot_first_modes, plot_snapshots_png, save_string_frame_png  # type: ignore
        from digital_twin.back_end.analysis.fft import tracer_fft_png, tracer_fft_logdb_remplie  # type: ignore
        from digital_twin.back_end.analysis.spectrogram import plot_spectrogram  # type: ignore
        from digital_twin.back_end.viz.anim import animate_string_motion  # type: ignore

        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_first_modes(x_coords, modes_full, freqs_hz, max_modes=4, savepath=str(plots_dir / "modes_first4.png"))

    # 6.3) PLOTS — Déplacement à un nœud représentatif et énergies
    if bool(getattr(_cfg, "OUTPUT_ENABLE_IMAGES", True)):
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as _eplt:  # pragma: no cover
            print("[AVERTISSEMENT] matplotlib indisponible pour tracer x(t):", _eplt)
        else:
            # Déplacement x(t) au nœud choisi
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

            # Énergies en fonction du temps
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

    # 6.4) FFT (et spectrogramme en option)
    out_node = int(getattr(_cfg, "OUTPUT_NODE", max(1, n // 2)))
    out_node = max(0, min(n - 1, out_node))
    if bool(getattr(_cfg, "OUTPUT_ENABLE_IMAGES", True)):
        plots_dir.mkdir(parents=True, exist_ok=True)
        # Choix du style de FFT
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
            tracer_fft_logdb_remplie(
                _sig,
                delta_t,
                str(fft_logdb_path),
                fmin=fmin_cfg,
                fmax=fmax_cfg,
                min_db=min_db_cfg,
                smooth_window=smooth_w_cfg,
                color=color_hex,
                annotate_peaks=True,
                n_peaks=8,
                title=f"FFT — nœud {out_node} (log f, dB)",
                db_offset=db_offset_cfg,
                log_bins_per_octave=bpo_cfg,
                octave_smoothing=oct_smooth_cfg,
                smooth_domain=smooth_domain_cfg,
            )

        # Spectrogramme (optionnel)
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
                plot_spectrogram(
                    U_hist[out_node, :],
                    delta_t,
                    str(sp_path),
                    fmin=sp_fmin,
                    fmax=sp_fmax,
                    nperseg=sp_nperseg,
                    overlap_frac=sp_ovlp,
                    cmap=sp_cmap,
                )
        except Exception:
            pass

    # 6.5) GIFs d'animation et PNG de la première image
    if bool(getattr(_cfg, "OUTPUT_ENABLE_GIFS", True)):
        plots_dir.mkdir(parents=True, exist_ok=True)

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

        # Intervalles (ms) à partir du FPS
        _int_ms_slow = max(1, int(round(1000.0 / max(1, _fps_slow))))
        _int_ms_real = max(1, int(round(1000.0 / max(1, _fps_real))))

        # Décimation pour une lecture en temps réel : decim_real ≈ 1 / (dt * fps)
        decim_real = max(1, int(round(1.0 / (delta_t * max(1, _fps_real)))))

        # Stratégie pour le GIF « lent »
        total_steps = int(U_hist.shape[1])
        slow_duration_target = getattr(_cfg, "ANIM_SLOW_DURATION_S", None)
        if slow_duration_target is not None:
            try:
                T_target = max(1.0, float(slow_duration_target))
                frames_needed = max(1.0, T_target * float(_fps_slow))
                decim_slow = max(1, int(round(float(total_steps) / frames_needed)))
            except Exception:
                decim_slow = max(1, decim_real // max(1, _decim_slow))
        else:
            _slow_factor = getattr(_cfg, "ANIM_SLOW_FACTOR", None)
            if _slow_factor is not None:
                try:
                    SF = max(1.0, float(_slow_factor))
                except Exception:
                    SF = float(max(1, _decim_slow))
                decim_slow = max(
                    1,
                    int(round(decim_real * (float(_fps_real) / max(1.0, float(_fps_slow) * SF)))),
                )
            else:
                decim_slow = max(1, decim_real // max(1, _decim_slow))

        # Informations de diagnostic des animations
        frames_real = max(1, total_steps // int(decim_real))
        frames_slow = max(1, total_steps // int(decim_slow))
        dur_real = frames_real / max(1, _fps_real)
        dur_slow = frames_slow / max(1, _fps_slow)
        ratio = dur_slow / max(1e-12, dur_real)
        print(
            f"[ANIM] real: decim={decim_real}, fps={_fps_real}, frames={frames_real}, duration≈{dur_real:.2f}s"
        )
        print(
            f"[ANIM] slow: decim={decim_slow}, fps={_fps_slow}, frames={frames_slow}, duration≈{dur_slow:.2f}s (x{ratio:.2f} slower)"
        )

        # GIF 1 — Temps réel
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

        # GIF 2 — Ralenti
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

        # PNG de la première image (utile pour l'aperçu)
        if bool(getattr(_cfg, "OUTPUT_ENABLE_IMAGES", True)):
            png0 = plots_dir / "string_motion_t0.png"
            try:
                _frame_y_scale = getattr(_cfg, "ANIM_Y_SCALE", None)
            except Exception:
                _frame_y_scale = None
            save_string_frame_png(
                x_coords,
                U_hist,
                frame_idx=0,
                savepath=str(png0),
                y_scale=_frame_y_scale,
                y_pad_frac=_y_pad,
            )

    # 6.6) PLOT — Profils à des temps sélectionnés (instantanés)
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
