from __future__ import annotations
"""
Excitations localisées (forces) pour corde vibrante.

But: concentrer la création et la composition de fournisseurs de force
(enveloppes trapézoïdales, décalage temporel, somme) dans un module
spécifique aux interactions, en gardant les noms en français.
"""
from typing import Callable
import numpy as np

try:
    from digital_twin.back_end.fem.time_integration import (
        fournisseur_force_localisee as _f_localisee,
        somme_de_forces as _somme,
    )  # type: ignore
except Exception:  # pragma: no cover
    from ..fem.time_integration import (
        fournisseur_force_localisee as _f_localisee,
        somme_de_forces as _somme,
    )  # type: ignore


def fournisseur_force_localisee(n: int, i_node: int, Fmax: float, t_rise: float, t_hold: float, t_decay: float, t0: float = 0.0) -> Callable[[float, int], np.ndarray]:
    """Alias direct vers l'implémentation FEM, exposé dans le paquet interactions."""
    return _f_localisee(n, i_node, Fmax, t_rise, t_hold, t_decay, t0=t0)


def somme_de_forces(n: int, *fornecedores: Callable[[float, int], np.ndarray]) -> Callable[[float, int], np.ndarray]:
    """Somme de plusieurs fournisseurs (même signature)."""
    return _somme(n, *fornecedores)


def decalage_temporel(F_base: Callable[[float, int], np.ndarray], t_offset: float):
    """Retourne un fournisseur F_shift(t,k) = F_base(t + t_offset, k)."""
    def F_shift(t: float, k: int):
        return F_base(float(t) + float(t_offset), int(k))
    return F_shift
