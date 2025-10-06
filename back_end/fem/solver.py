"""Solveur temporel pour la corde utilisant différences centrales + amortissement de Rayleigh.

Équation du mouvement discrétisée :
    M * U_ddot + C * U_dot + K * U = 0

Avec approximations en différences centrales :
    U_ddot ≈ (U_{n+1} - 2 U_n + U_{n-1}) / dt^2
    U_dot  ≈ (U_{n+1} - U_{n-1}) / (2 dt)

On réarrange vers un système linéaire en U_{n+1} :
    (M/dt^2 + C/(2 dt)) * U_{n+1} = (2M/dt^2 - K) * U_n + (C/(2 dt) - M/dt^2) * U_{n-1}

Ce module implémente cette itération.
"""

from __future__ import annotations

import numpy as np
try:  # tqdm est optionnel (barre de progression élégante)
    from tqdm import trange  # type: ignore
except Exception:  # Repli simple : utiliser range sans barre
    def trange(*args, **kwargs):  # type: ignore
        return range(*args)

from ..utils.utils import generate_pluck_shape


def run_time_simulation(
    M: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    pluck_position_ratio: float,
    pluck_amplitude: float,
    length: float,
    n_nodes: int,
    sim_time: float,
    dt: float,
    output_node: int | None = None,
    return_full: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Exécute la simulation temporelle et retourne l'historique d'un nœud.

    Paramètres
    ----------
    M, K, C : np.ndarray
        Matrices globales (masse, raideur, amortissement) de taille (n_nodes, n_nodes).
    pluck_position_ratio : float
        Position relative du pincement (0..1).
    pluck_amplitude : float
        Amplitude initiale maximale du pincement (m).
    length : float
        Longueur totale de la corde.
    n_nodes : int
        Nombre de nœuds.
    sim_time : float
        Durée totale de la simulation (s).
    dt : float
        Pas de temps (s).
    output_node : int | None
        Nœud dont on extrait l'historique. Si None, utilise le nœud central.

    Retour
    ------
    history : np.ndarray
        Vecteur (num_steps,) du déplacement au nœud d'intérêt à chaque pas.
    """
    # --- Validations de base ---
    if M.shape != (n_nodes, n_nodes) or K.shape != (n_nodes, n_nodes) or C.shape != (n_nodes, n_nodes):
        raise ValueError("M, K et C doivent avoir dimension (n_nodes, n_nodes)")
    if sim_time <= 0:
        raise ValueError("sim_time doit être positif")
    if dt <= 0:
        raise ValueError("dt doit être positif")

    # Nombre de pas de temps total (tronqué à l'entier)
    num_steps = int(sim_time / dt)
    if num_steps < 3:  # Besoin d'au moins 3 états pour le schéma à 2 niveaux précédents
        raise ValueError("Nombre de pas insuffisant (augmenter sim_time ou réduire dt)")

    # Si aucun nœud de sortie explicitement choisi -> milieu de la corde
    if output_node is None:
        output_node = n_nodes // 2
    if not (0 <= output_node < n_nodes):
        raise ValueError("output_node hors intervalle")

    # Historique du nœud choisi et, optionnellement, champ complet
    history = np.zeros(num_steps, dtype=float)
    full_history = None
    if return_full:
        full_history = np.zeros((num_steps, n_nodes), dtype=float)

    # c) Déplacement initial (U0)
    U0 = generate_pluck_shape(n_nodes, length, pluck_position_ratio, pluck_amplitude)  # Forme initiale pincée
    # d) Vitesse initiale nulle -> U1 = U0
    U1 = U0.copy()  # Vitesse initiale nulle -> duplication

    # Stocke les deux premiers états dans l'historique
    history[0] = U0[output_node]
    history[1] = U1[output_node]
    if return_full:
        full_history[0] = U0
        full_history[1] = U1

    inv_dt2 = 1.0 / (dt * dt)      # 1 / dt^2
    inv_2dt = 1.0 / (2.0 * dt)     # 1 / (2 dt)

    # f) Pré-calculs de matrices
    LHS_matrix = M * inv_dt2 + C * inv_2dt  # Matrice gauche (facteur de U_{n+1})
    try:
        inv_LHS = np.linalg.inv(LHS_matrix)
    except np.linalg.LinAlgError as e:
        raise RuntimeError("Échec inversion LHS_matrix (singulière). Vérifier conditions de bord.") from e

    RHS_matrix1 = 2.0 * M * inv_dt2 - K     # Coefficient appliqué à U_n
    RHS_matrix2 = C * inv_2dt - M * inv_dt2 # Coefficient appliqué à U_{n-1}

    U2 = np.zeros_like(U0)  # Tampon pour l'état futur U_{n+1}

    # g) Boucle temporelle
    for step in trange(2, num_steps, desc="Intégration temporelle", leave=False):
        # Calcul du second membre (combinaison linéaire des deux états précédents)
        RHS_vector = RHS_matrix1 @ U1 + RHS_matrix2 @ U0
        # Résolution explicite via l'inverse pré-calculé
        U2[:] = inv_LHS @ RHS_vector

        # Enregistre déplacement du nœud d'intérêt
        history[step] = U2[output_node]
        if return_full:
            full_history[step] = U2

        # Décale les pointeurs (U_{n-1} <- U_n, U_n <- U_{n+1})
        U0, U1 = U1, U2

    if return_full:
        return history, full_history  # type: ignore[return-value]
    return history


__all__ = ["run_time_simulation"]
