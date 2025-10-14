"""
Exports de données (CSV, etc.).
"""
from __future__ import annotations

import numpy as np


def save_displacement_csv(t: np.ndarray, U: np.ndarray, savepath: str) -> None:
    """Sauvegarde un CSV avec le temps et les positions de la corde (une colonne par nœud).
    t : (n_steps,), U : (n_nodes, n_steps) → CSV (n_steps x (1+n_nodes)) avec en-têtes.
    """
    try:
        from pathlib import Path as _P
        out = _P(savepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        n_nodes, n_steps = int(U.shape[0]), int(U.shape[1])
        if t.shape[0] != n_steps:
            raise ValueError("Dimensions incompatibles: t et U doivent avoir le même nombre d'instants")
        data = np.empty((n_steps, 1 + n_nodes), dtype=float)
        data[:, 0] = t
        data[:, 1:] = U.T
        header = "t," + ",".join(f"u_{i}" for i in range(n_nodes))
        np.savetxt(out, data, delimiter=",", header=header, comments="", fmt="%.10e")
        print(f"[INFO] Positions de la corde sauvegardées en CSV : {out}")
    except Exception as _ecsv:  # pragma: no cover
        print("[AVERTISSEMENT] Échec de la sauvegarde CSV des positions:", _ecsv)
