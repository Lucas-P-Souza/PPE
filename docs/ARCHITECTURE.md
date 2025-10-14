# Architecture Overview

This document explains the current backend flow, where results are written, and how the main modules are organized. It reflects the present code under `digital_twin/back_end` (formulation/solver/config/debug) and the updated FFT/GIF features.

## End-to-end flow (backend)

1) Configuration: `back_end/config.py`
   - Centralizes physics, mesh, integration, output toggles and analysis settings.
   - Key toggles: `OUTPUT_ENABLE_IMAGES`, `OUTPUT_ENABLE_GIFS`, `OUTPUT_ENABLE_CSV`, `DEBUG_ENABLED`.
   - FFT style: `FFT_STYLE` = "linear" or "logdb" (default: "logdb"). Optional spectrogram switches provided.

2) Formulation (FEM): `back_end/fem/formulation.py`
   - Assembles global Mass (M) and Stiffness (K) from a non-uniform “frets” mesh defined in config.
   - Computes Rayleigh damping `C = α M + β K` from two modal targets via `rayleigh_damping`.
   - Applies boundary conditions as per project policy:
     - M/K: ends clamped with diagonal 1 to preserve invertibility.
     - C: edges zeroed (no diag=1).
   - Public API: `assemble_mkc`, `build_global_mkc_from_config`, `rayleigh_damping`.

3) Time integration and analysis: `back_end/fem/solver.py`
   - Validation helpers: shapes, symmetry and BC checks (`validar_mck`).
   - Initial conditions: triangular pluck at `PLUCK_POS * L` with amplitude `PLUCK_AMP`.
   - Newmark-β integration on free DOFs (`newmark_beta`, β=1/4, γ=1/2).
   - Modal analysis (`compute_modal_frequencies_and_modes`).
   - Analysis/plots:
     - FFT (linear panel + dB) and log-frequency filled dB plot (`tracer_fft_png`, `tracer_fft_logdb_remplie`).
     - Optional spectrogram (`plot_spectrogram`, SciPy if available; NumPy fallback).
   - Visualization: modes plot, single-frame PNG, multi-snapshot PNG, and GIF animations (slow motion + real-time).
   - Export: `save_displacement_csv` writes time and node displacements to CSV.

4) Orchestration: `back_end/main.py` (CLI)
   - Builds M, K, C from config; runs Newmark; saves CSV/plots/FFT/GIFs respecting output toggles.
   - The GUI (`front_end/`) is separate and can be wired to call the same primitives if desired.

## Results location

Current outputs live under:

- `digital_twin/back_end/results/plots/`
  - Time series and energies:
    - `newmark_node_displacement.png`
    - `newmark_energies.png`
  - Modal:
    - `modes_first4.png`
  - Spectral:
    - `newmark_output_fft.png` (linear) and/or `newmark_output_fft_logdb.png` (log-dB filled)
    - `newmark_output_spectrogram.png` (optional; if enabled)
  - Animation/frames:
    - `string_motion_slow.gif`, `string_motion_real.gif`, `string_motion_t0.png`, `string_snapshots.png`
  - Data:
    - `string_positions.csv`

These are controlled by `OUTPUT_ENABLE_IMAGES`, `OUTPUT_ENABLE_GIFS`, and `OUTPUT_ENABLE_CSV` in `config.py`.

## Current folder structure (relevant parts)

- `digital_twin/`
  - `back_end/`
    - `config.py` → all parameters and toggles (French annotations)
    - `fem/`
      - `formulation.py` → global assembly (M,K,C) and Rayleigh damping
      - `solver.py` → validations, ICs, Newmark, modal, FFT/spectrogram, plots/animations, CSV
    - `utils/`
      - `debug.py` → centralized debug helpers (lazy prints, Rayleigh and solver diagnostics)
    - `results/plots/` → generated images/GIFs/CSV
    - `main.py` → backend CLI orchestrator
  - `front_end/` → GUI (Qt) components (`main.py`, `gui.py`, `corde_widget.py`, etc.)
  - `docs/` → this document and future docs

## Notes on damping and BCs

- Rayleigh coefficients (α, β) are derived from two modal damping targets `(p,q)` and `(ζ_p, ζ_q)`.
- BC policy adopted:
  - M/K: clamp first and last nodes by zeroing rows/cols and setting the diagonal to 1.0 on those nodes.
  - C: zero rows/cols at ends (no artificial diagonal 1).
  - Rayleigh α/β are determined on the constrained evaluation matrices, then `C = αM + βK` is formed from pre-BC M/K and finally C gets its BCs.

## Debugging

`back_end/utils/debug.py` provides a single switch `DEBUG_ENABLED` to enable all debug prints and rich diagnostics:

- Rayleigh explanation block (once): computed modal frequencies, achieved ζ at references, α/β, crossover frequency.
- Solver instrumentation: setup summary, Newmark constants, sampled per-step snapshots with energy, start/end energy summary.

## Reorganization suggestion (optional, future work)

`fem/solver.py` currently contains integration, modal analysis, FFT/spectrogram, visualization, and CSV I/O. For clearer concerns separation, consider splitting into:

- `fem/time_integration.py` → `newmark_beta`, IC helpers, energy computations
- `fem/modal.py` → `compute_modal_frequencies_and_modes`, mode plotting
- `analysis/fft.py` → `calculer_fft_monocotee`, `tracer_fft_png`, `tracer_fft_logdb_remplie`, `plot_spectrogram`
- `viz/plots.py` and `viz/anim.py` → snapshot plots, modes plot, GIF generation
- `io/exports.py` → `save_displacement_csv`

Keep `formulation.py` as-is and expose a thin public API via `__all__`. This refactor is not required to run; it’s a suggestion to simplify maintenance as features grow.

## How to run (backend CLI)

- From VS Code tasks: "Run backend simulation" runs `back_end/main.py` with Python.
- Outputs are written to `digital_twin/back_end/results/plots/` based on `config.py` toggles.

