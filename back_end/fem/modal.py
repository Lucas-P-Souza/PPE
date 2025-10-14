"""
Analyse modale (FEM):
 - Détection des DDLs contraints (Dirichlet) selon la convention du projet (bords fixés)
 - Calcul des fréquences et formes modales via K v = λ M v sur les DDLs libres

Mathématiques:
 - Problème généralisé: K v = λ M v, avec M,K symétriques définis positifs sur les DDLs libres.
 - Valeurs propres: λ = ω^2 (ω en rad/s). Fréquences en Hz: f = ω / (2π).
 - Normalisation modale (massique): v^T M v = 1 (appliquée sur l'espace libre, puis étendue à l'espace complet).
"""
from __future__ import annotations

import numpy as np


def detect_constrained_dofs_mk(M: np.ndarray, K: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """Détecte des DDL contraints par CL de Dirichlet selon la convention du projet :
    - lignes/colonnes nulles hors diagonale ; diag=1.0 sur les DDL contraints.
    Retourne un tableau d'indices contraints.
    """
    n = M.shape[0]
    constrained = []
    for i in range(n):
        rowM = M[i, :].copy(); rowM[i] = 0.0
        colM = M[:, i].copy(); colM[i] = 0.0
        rowK = K[i, :].copy(); rowK[i] = 0.0
        colK = K[:, i].copy(); colK[i] = 0.0
        if (
            np.all(np.abs(rowM) <= atol) and np.all(np.abs(colM) <= atol)
            and np.all(np.abs(rowK) <= atol) and np.all(np.abs(colK) <= atol)
            and np.isclose(M[i, i], 1.0, atol=1e-9) and np.isclose(K[i, i], 1.0, atol=1e-9)
        ):
            constrained.append(i)
    return np.asarray(constrained, dtype=int)


def compute_modal_frequencies_and_modes(M: np.ndarray, K: np.ndarray, num_modes: int = 4):
    """Résout K v = λ M v sur les DDL libres et renvoie (freqs_hz, modes_full).

    Étapes:
    1) Détecter les DDLs contraints et extraire sous-matrices libres M_ff, K_ff.
    2) Résoudre A v = λ v avec A = M_ff^{-1} K_ff (NumPy: np.linalg.solve pour M_ff^{-1}K_ff, puis eig).
       - eig renvoie (eigvals, eigvecs) avec potentielle petite partie imaginaire -> on prend la partie réelle.
       - λ ≥ 0 pour stabilité numérique; ω = sqrt(λ) (rad/s); tri croissant.
    3) Garder les 'num_modes' premières valeurs/vecteurs.
    4) Normaliser chaque vecteur selon v^T M_ff v = 1, puis étendre au plein espace (contraints=0).
    5) Convertir ω → f (Hz) via f = ω/(2π).
    """
    if M.shape != K.shape or M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M et K doivent être carrées et de même taille")
    n = M.shape[0]
    constrained = detect_constrained_dofs_mk(M, K)
    free = np.setdiff1d(np.arange(n), constrained)
    if free.size == 0:
        raise ValueError("Aucun DDL libre détecté pour l'analyse modale")

    M_ff = M[np.ix_(free, free)]
    K_ff = K[np.ix_(free, free)]

    # np.linalg.solve: résout M_ff X = K_ff pour X = M_ff^{-1}K_ff (plus stable que inverser M_ff)
    A = np.linalg.solve(M_ff, K_ff)
    eigvals, eigvecs = np.linalg.eig(A)  # eig: valeurs/vecteurs propres du problème standard A v = λ v
    eigvals = np.real(eigvals)
    eigvals[eigvals < 0] = 0.0
    omegas = np.sqrt(eigvals)
    idx = np.argsort(omegas)
    omegas = omegas[idx]
    V = np.real(eigvecs[:, idx])

    k = int(min(num_modes, V.shape[1]))
    omegas_k = omegas[:k]
    V_k = V[:, :k]

    modes_full = np.zeros((n, k), dtype=float)
    for j in range(k):
        v = V_k[:, j]
        # Normalisation massique: ||v||_M = sqrt(v^T M_ff v)
        norm = float(np.sqrt(v.T @ (M_ff @ v)))
        vj = v if norm <= 0 else v / norm
        full = np.zeros(n, dtype=float)
        full[free] = vj
        modes_full[:, j] = full

    freqs_hz = omegas_k / (2.0 * np.pi)
    return freqs_hz, modes_full
