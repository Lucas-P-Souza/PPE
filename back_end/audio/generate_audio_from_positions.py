
"""Utilities to generate WAV audio files from simulation CSVs.

This module provides two convenience functions used by the backend CLI and
by developers when inspecting timbral differences along the string.

The code is intentionally small and uses only numpy/pandas/scipy.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.io.wavfile import write
from scipy.interpolate import interp1d


def generate_audio_from_csv(csv_path: str | None = None, output_path: str | None = None) -> str:
    """Generate a WAV file from the midpoint displacement recorded in a CSV.

    The CSV must contain a 't' column and columns named 'u_<i>' for node
    displacements. This function picks the midpoint column and writes a mono
    16-bit WAV at 44.1 kHz.
    """
    if csv_path is None:
        csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results', 'plots', 'string_positions.csv'))

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Fichier CSV introuvable: {csv_path}")

    data = pd.read_csv(csv_path)
    time = data['t'].values

    pos_cols = [c for c in data.columns if c.startswith('u_')]
    if not pos_cols:
        raise ValueError("Aucune colonne 'u_<i>' trouvée dans le CSV")

    mid_index = len(pos_cols) // 2
    mid_col = pos_cols[mid_index]
    position = data[mid_col].values

    # normalize and resample
    position_norm = position / np.max(np.abs(position))
    sample_rate = 44100
    duration = float(time[-1] - time[0])
    if duration <= 0:
        raise ValueError('Durée de signal invalide (time[-1] <= time[0])')
    n_samples = int(max(1, duration * sample_rate))
    new_time = np.linspace(time[0], time[-1], n_samples)

    interpolator = interp1d(time, position_norm, kind='linear')
    wave_data = interpolator(new_time)
    wave_int16 = np.int16(wave_data * 32767)

    output_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results', 'plots'))
    os.makedirs(output_dir, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(output_dir, 'string_middle.wav')

    write(output_path, sample_rate, wave_int16)
    print(f'Audio généré (milieu de la corde) : {output_path}')
    return output_path


def generate_multiple_positions_audio(csv_path: str | None = None) -> None:
    """Generate WAVs for several listening positions along the string.

    Writes one WAV per selected position (debut, quart, milieu, trois_quarts, fin).
    """
    if csv_path is None:
        csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results', 'plots', 'string_positions.csv'))

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Fichier CSV introuvable: {csv_path}")

    data = pd.read_csv(csv_path)
    time = data['t'].values

    pos_cols = [c for c in data.columns if c.startswith('u_')]
    if not pos_cols:
        raise ValueError("Aucune colonne 'u_<i>' trouvée dans le CSV")

    total_points = len(pos_cols)
    positions_to_test = {
        'debut': 5,
        'quart': total_points // 4,
        'milieu': total_points // 2,
        'trois_quarts': 3 * total_points // 4,
        'fin': max(0, total_points - 6),
    }

    sample_rate = 44100
    duration = float(time[-1] - time[0])
    if duration <= 0:
        raise ValueError('Durée de signal invalide (time[-1] <= time[0])')
    n_samples = int(max(1, duration * sample_rate))
    new_time = np.linspace(time[0], time[-1], n_samples)

    output_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results', 'plots'))
    os.makedirs(output_dir, exist_ok=True)

    for position_name, point_index in positions_to_test.items():
        col_name = f'u_{point_index}'
        if col_name not in data.columns:
            print(f"Colonne {col_name} introuvable dans le CSV — saut de cette position.")
            continue

        position = data[col_name].values
        position_norm = position / np.max(np.abs(position))
        interpolator = interp1d(time, position_norm, kind='linear')
        wave_data = interpolator(new_time)
        wave_int16 = np.int16(wave_data * 32767)

        output_path = os.path.join(output_dir, f'string_{position_name}.wav')
        write(output_path, sample_rate, wave_int16)
        print(f'Audio généré pour la position {position_name} (point {point_index}) : {output_path}')

    print(f'\n{len(positions_to_test)} fichiers audio générés.')
    print('Écoutez-les pour comparer les timbres selon la position sur la corde.')


if __name__ == '__main__':
    print('Génération de fichiers audio pour différentes positions sur la corde...')
    generate_multiple_positions_audio()
