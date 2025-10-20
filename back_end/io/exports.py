from pathlib import Path
import numpy as np


def enregistrer_deplacement_csv(t: np.ndarray, U: np.ndarray, savepath: str) -> None:
    # Fonction utilitaire : enregistre t et U dans un CSV 'savepath' (format ASCII)
    """Minimal ASCII-only CSV writer for backend use.

    Writes a CSV with columns: t,u_0,u_1,...,u_{n_nodes-1}.
    """
    out = Path(savepath)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(t, np.ndarray) or t.ndim != 1:
        raise ValueError("t must be a 1D numpy.ndarray of time instants")
    if not isinstance(U, np.ndarray) or U.ndim != 2:
        raise ValueError("U must be a 2D numpy.ndarray with shape (n_nodes, n_steps)")

    n_nodes, n_steps = int(U.shape[0]), int(U.shape[1])
    if t.shape[0] != n_steps:
        raise ValueError("t and U must have matching time dimension")

    data = np.empty((n_steps, 1 + n_nodes), dtype=float)
    data[:, 0] = t
    data[:, 1:] = U.T

    header = "t," + ",".join(f"u_{i}" for i in range(n_nodes))
    np.savetxt(out, data, delimiter=",", header=header, comments="", fmt="%.10e")
