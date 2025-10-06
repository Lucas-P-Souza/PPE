"""Script principal d'exécution de la simulation de la corde.

Flux général :
 1. Charge les paramètres depuis `config`.
 2. Assemble les matrices globales M, K, C (FEM 1D uniforme).
 3. Effectue l'intégration temporelle et récupère l'historique de déplacement d'un nœud.
 4. Sauvegarde les graphiques (forme d'onde et spectre FFT) dans `back_end/results/plots`.

Remarque : le répertoire de sortie est interne à `digital_twin/back_end/results` pour éviter d'encombrer la racine.
Audio : génération WAV désactivée (placeholder) – possibilité d'ajouter plus tard un export.
"""

from __future__ import annotations

from pathlib import Path

from . import config  # Paramètres globaux (longueur, tension, maillage, etc.)
from .fem.formulation import (
    assemble_system_matrices,
    assemble_system_matrices_nonuniform,
)
from .fem.solver import run_time_simulation
from .utils.utils import plot_waveform, plot_fft, create_animation


BASE_DIR = Path(__file__).resolve().parent      # Répertoire back_end
RESULTS_ROOT = BASE_DIR / "results"            # Racine des sorties
PLOTS_DIR = RESULTS_ROOT / "plots"             # Dossier des figures
AUDIO_DIR = RESULTS_ROOT / "audio"             # Dossier de l'audio (placeholder)


def _ensure_dirs():
    """Crée les dossiers de sortie si nécessaire (idempotent)."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)  # Crée hiérarchie si absent
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    _ensure_dirs()

    print("[1/5] Assemblage des matrices M, K, C ...")  # Étape 1 : construction FEM
    # Détection d'une maille non uniforme provenant du module de frettes.
    fret_dx_mm = config.FRET_DXS_MM  # Liste de dx en mm si maillage frettes
    modes_ref = config.DAMPING_MODES_REF
    zetas_ref = config.DAMPING_ZETAS_REF
    apply_bc = config.APPLY_FIXED_BC
    if fret_dx_mm:  # Liste non vide => mode frettes actif
        dx_vector_m = [v / 1000.0 for v in fret_dx_mm]  # Conversion en mètres
        M, K, C = assemble_system_matrices_nonuniform(
            dx_vector=dx_vector_m,
            tension=config.T,
            lin_density=config.MU,
            damping_modes_ref=modes_ref,
            damping_zetas_ref=zetas_ref,
            apply_fixed_bc=apply_bc,
        )
        n_nodes_effectif = len(dx_vector_m) + 1
        length_effective = sum(dx_vector_m)
        print(
            f"    -> Maille frettes détectée: n_elems={len(dx_vector_m)}, n_nodes={n_nodes_effectif}, L≈{length_effective:.6f} m"
        )
    else:  # Fallback : maillage uniforme
        M, K, C = assemble_system_matrices(
            n_nodes=config.N_NODES,
            length=config.L,
            tension=config.T,
            lin_density=config.MU,
            damping_modes_ref=modes_ref,
            damping_zetas_ref=zetas_ref,
            apply_fixed_bc=apply_bc,
        )
        n_nodes_effectif = config.N_NODES
        length_effective = config.L
        print("    -> Maille uniforme utilisée.")

    print("[2/5] Intégration temporelle ...")  # Étape 2 : schéma explicite central
    sim_result = run_time_simulation(
        M=M,
        K=K,
        C=C,
        pluck_position_ratio=config.PLUCK_POS,
        pluck_amplitude=config.PLUCK_AMP,
        length=length_effective,
        n_nodes=n_nodes_effectif,
        sim_time=config.T_SIM,
        dt=config.DT,
        output_node=config.OUTPUT_NODE,
        return_full=True,
    )
    history, full_history = sim_result  # unpack
    print("    -> Intégration terminée.")

    print("[3/5] (Audio) - Étape sautée (placeholder)")  # Étape 3 : audio non implémenté
    sample_rate = int(1.0 / config.DT)

    print("[4/5] Génération des graphiques ...")  # Étape 4 : figures temporelles + spectre
    waveform_path = PLOTS_DIR / f"{config.FILENAME}_waveform.png"
    fft_path = PLOTS_DIR / f"{config.FILENAME}_fft.png"
    plot_waveform(history, sample_rate=sample_rate, filepath=waveform_path, title="Déplacement du nœud")
    plot_fft(history, sample_rate=sample_rate, filepath=fft_path, title="Spectre (FFT)", freq_limit=2000)
    print(f"    -> Graphiques sauvegardés dans {PLOTS_DIR}")

    # Animation (déplacement spatio-temporel)
    print("[5/5] Génération de l'animation MP4 ...")
    anim_path = PLOTS_DIR / f"{config.FILENAME}_animation.mp4"  # Nom en français
    create_animation(
        full_history,
        length=length_effective,
        dt=config.DT,
        filepath=anim_path,
        frame_step=50,  # Saut de frames pour limiter taille
        fps=30,
    )
    print(f"    -> Animation (si ffmpeg disponible) : {anim_path}")

    print("Simulation terminée avec succès.")  # Fin de pipeline


if __name__ == "__main__":
    main()
