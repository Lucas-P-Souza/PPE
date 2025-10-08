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
    from digital_twin.back_end.fem.solver import (
        validar_mck,
        definir_parametros_simulacao,
        inicializar_estados_iniciais,
        calcular_u1,
        compute_modal_frequencies_and_modes,
        build_node_positions_from_config,
        plot_first_modes,
        zero_force_provider,
        newmark_beta,
        compute_energies_over_time,
        plot_fft_png,
        animate_string_motion,
        save_string_frame_png,
        plot_snapshots_png,
        save_displacement_csv,
    )
except ModuleNotFoundError:
    # Fallback when executed as a standalone script (ensure workspace root on sys.path)
    import sys as _sys
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    from digital_twin.back_end import config as _cfg  # type: ignore
    from digital_twin.back_end.fem.formulation import build_global_mkc_from_config  # type: ignore
    from digital_twin.back_end.fem.solver import (
        validar_mck,
        definir_parametros_simulacao,
        inicializar_estados_iniciais,
        calcular_u1,
        compute_modal_frequencies_and_modes,
        build_node_positions_from_config,
        plot_first_modes,
        zero_force_provider,
        newmark_beta,
        compute_energies_over_time,
        plot_fft_png,
        animate_string_motion,
        save_string_frame_png,
        plot_snapshots_png,
        save_displacement_csv,
    )


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
    validar_mck(M, C, K, verbose=True)

    # Simulation parameters
    delta_t = float(getattr(_cfg, "DT", 1e-5))
    T_total = float(getattr(_cfg, "T_SIM", 0.1))
    definir_parametros_simulacao(delta_t, T_total)

    # Initial conditions: triangular pluck at position PLUCK_POS * L with amplitude PLUCK_AMP
    L_eff = float(getattr(_cfg, "L", 1.0))
    h_eff = float(getattr(_cfg, "PLUCK_AMP", 0.0))
    x_p_rel = float(getattr(_cfg, "PLUCK_POS", 0.25))
    x_p = x_p_rel * L_eff
    U0, U_nm1, U_n = inicializar_estados_iniciais(M, L=L_eff, h=h_eff, x_p=x_p)
    print(f"U0 inicializado: shape={U0.shape}, max={np.nanmax(U0):.3e}, min={np.nanmin(U0):.3e}")

    # First step via central differences (diagnostic)
    _ = calcular_u1(M, C, K, U_n=U_n, U_nm1=U_nm1, delta_t=delta_t)

    # Modal analysis and plot of first modes
    n = M.shape[0]
    x_coords = build_node_positions_from_config(n)
    freqs_hz, modes_full = compute_modal_frequencies_and_modes(M, K, num_modes=4)
    print("Premières fréquences (Hz):", np.round(freqs_hz, 3))
    plots_dir = ROOT / "digital_twin" / "back_end" / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_first_modes(x_coords, modes_full, freqs_hz, max_modes=4, savepath=str(plots_dir / "modes_first4.png"))

    # Newmark-beta integration with zero external force
    V0_zero = np.zeros(n, dtype=float)
    F_zero = zero_force_provider(n)
    t_vec, U_hist, V_hist, A_hist = newmark_beta(M, C, K, F_zero, dt=delta_t, t_max=T_total, U0=U0, V0=V0_zero, A0=None)

    # Save CSV of string positions over time
    csv_path = plots_dir / "string_positions.csv"
    save_displacement_csv(t_vec, U_hist, str(csv_path))

    # Displacement over time for a representative node
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as _eplt:  # pragma: no cover
        print("[AVERTISSEMENT] matplotlib indisponible pour tracer x(t):", _eplt)
    else:
        node_idx = max(1, n // 3)
        plt.figure(figsize=(8, 4))
        plt.plot(t_vec, U_hist[node_idx, :], lw=1.2)
        plt.title(f"Déplacement au nœud {node_idx} — Newmark (β=1/4, γ=1/2), U0 triangulaire (pincement), F=0")
        plt.xlabel("temps (s)")
        plt.ylabel("x (m)")
        plt.grid(True, alpha=0.3)
        outp = plots_dir / "newmark_node_displacement.png"
        plt.tight_layout(); plt.savefig(outp, dpi=150); plt.close()
        print(f"[INFO] Tracé de x(t) enregistré dans: {outp}")

        # Énergies
        Ek, Ep, Et = compute_energies_over_time(M, K, U_hist, V_hist)
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
    fft_path = plots_dir / "newmark_output_fft.png"
    plot_fft_png(U_hist[out_node, :], delta_t, str(fft_path), title=f"FFT — nœud {out_node}", fmax=None)

    # Animation and first-frame PNG
    anim_path = plots_dir / "string_motion.gif"
    try:
        _y_scale = float(getattr(_cfg, "ANIM_Y_SCALE", 1.0))
        _y_pad = float(getattr(_cfg, "ANIM_Y_PAD_FRAC", 0.05))
    except Exception:
        _y_scale, _y_pad = 1.0, 0.05
    animate_string_motion(
        x_coords,
        U_hist,
        interval_ms=30,
        decim=5,
        savepath=str(anim_path),
        show=False,
        y_scale=_y_scale,
        y_pad_frac=_y_pad,
    )
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
# Ce fichier a été retiré du pipeline à la demande de l'utilisateur.
#
# À conserver seulement:
#  - back_end/fem/formulation.py
#  - back_end/fem/solver.py
#  - back_end/mesh/calcul_trous_de_frette.py
#  - back_end/mesh/fret_mesh.py
#  - back_end/config.py
#  - back_end/__init__.py
#
# Tout usage de ce fichier doit être interrompu. Il reste uniquement comme stub
# pour éviter des imports cassés dans d'anciens environnements.

import sys

def _removed_entrypoint() -> None:  # pragma: no cover
    print("[INFO] main.py retiré lors du nettoyage. Utilisez seulement formulation/solver/mesh/config.")
    raise SystemExit(0)

if __name__ == "__main__":  # pragma: no cover
    _removed_entrypoint()
