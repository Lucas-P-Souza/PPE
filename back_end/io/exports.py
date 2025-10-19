from pathlib import Path
import numpy as np


def enregistrer_deplacement_csv(t: np.ndarray, U: np.ndarray, savepath: str) -> None:
    """Write time vs. node displacements to CSV.

    Minimal ASCII-only implementation.
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
from pathlib import Path
import numpy as np


def enregistrer_deplacement_csv(t: np.ndarray, U: np.ndarray, savepath: str) -> None:
    # Minimal ASCII-only CSV writer used by the backend.
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
from pathlib import Path
import numpy as np


def enregistrer_deplacement_csv(t: np.ndarray, U: np.ndarray, savepath: str) -> None:
    """Write time vs. node displacements to CSV.

    Parameters
    - t: (n_steps,) array of time instants
    - U: (n_nodes, n_steps) matrix of displacements
    - savepath: path to output CSV
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
from pathlib import Path
import numpy as np


def enregistrer_deplacement_csv(t: np.ndarray, U: np.ndarray, savepath: str) -> None:
  """Write time vs. node displacements to CSV.

  Simple, ASCII-only implementation to avoid encoding issues.
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
"""CSV export utilities - French API only.
Exports de données (CSV) — unique French API.

This module exposes exactly one function used by the backend:
  - enregistrer_deplacement_csv(t, U, savepath)

Expected shapes:
  - t: numpy.ndarray with shape (n_steps,)
  - U: numpy.ndarray with shape (n_nodes, n_steps)

The CSV format produced has a leading column 't' followed by columns
u_0, u_1, ..., u_{n_nodes-1}.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def enregistrer_deplacement_csv(t: np.ndarray, U: np.ndarray, savepath: str) -> None:
    """Write time vs. node displacements to CSV.

    Parameters
    - t: (n_steps,) array of time instants
    - U: (n_nodes, n_steps) matrix of displacements
    - savepath: path to output CSV

    Raises
    - ValueError on shape/type mismatches
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

