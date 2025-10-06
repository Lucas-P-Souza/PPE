# Architecture Overview

This document explains the end-to-end flow, the folder structure, and where results live.

## End-to-end flow

1) GUI: `front_end/gui.py` and `front_end/corde_widget.py`
   - User clicks the string or presses Space.
   - The GUI reads the two cursors (note and strike) in percent and converts them to [0..100].
   - Calls `back_end.integration.SimulationController.trigger()` with those percentages.

2) Bridge/controller: `back_end/integration.py`
   - Converts percent to meters: pos = percent/100 * length.
   - Adds an extra fixed node at the “note” position (if > 0%).
   - Applies a triangular pluck condition at the “strike” position.
   - Runs `run_simulation` in a background thread, keeping the UI responsive.
   - Best-effort post-processing: saves some PNGs and a short GIF.

3) Simulation pipeline: `back_end/simulation.py`
   - Builds a uniform mesh (1D elements) with fixed endpoints and optional extra fixed node.
   - Assembles global stiffness/mass matrices and Rayleigh damping.
   - Reduces to free DOFs, integrates in time (Newmark-β default).
   - Saves CSVs and metadata.

4) Visualization and GIF
   - Plots and animations in `back_end/visualize_corda.py`.
   - GIF tooling in `back_end/generate_gif.py` (direct from CSV or from frames).

## Results location

Top-level directory for results:

- `digital_twin/results/`
  - `figures/`
    - `png/`  → static plots
    - `gifs/` → animations (GIF/MP4)
  - `outputs/`
    - `csv/`   → numeric outputs (`results.csv`, `results_full_nodes.csv`)
    - `other/` → metadata (`meta.json`), logs, etc.

## Folder structure

- `digital_twin/`
  - `back_end/`
    - `fem/`
      - `formulation/` → `mesh.py`, `matrices.py`, `shape_functions.py`
      - `time_solver/` → `newmark.py`, `runge_kutta.py`
    - `integration.py` → bridge from GUI to simulation
    - `simulation.py` → orchestration and output
    - `visualize_corda.py` → plotting and animations
    - `generate_gif.py`, `frames_export.py` → media helpers
    - `utils/outputs.py` → result directories
    - `main.py` → CLI runner
  - `front_end/`
    - `gui.py` → MainWindow and event wiring
    - `corde_widget.py` → custom drawn string with cursors
    - `main.py` → GUI entry point
    - `styles/main_style.qss` → styling
  - `results/` → standardized outputs (created on demand)
  - `docs/` → this document and future docs
  - `README.md` → overview and how to run

## Integration mode

- Direct function calls only (no HTTP/API). The GUI uses `SimulationController` to run the backend safely in a thread, with Qt signals used to update the UI on completion.
