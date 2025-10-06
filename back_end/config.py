"""Centralisation des paramètres de la simulation FEM d'une corde vibrante.

Ce module permet de modifier facilement les paramètres physiques, numériques
et de sortie sans toucher au cœur du code.

Sections :
1. Paramètres physiques de la corde
2. Amortissement (Rayleigh)
3. Paramètres numériques de la simulation
4. Conditions initiales (pincement / excitation)
5. Paramètres de sortie

Notes :
- Fréquence fondamentale théorique d'une corde idéale : f1 = (1/(2L)) * sqrt(T/μ)
- Vitesse de propagation de l'onde : c = sqrt(T/μ)
- Ajuster N_NODES, DT et T_SIM selon le compromis précision / coût de calcul.
"""
from __future__ import annotations

from math import sqrt
from pathlib import Path

# =============================================================
# MODE MAILLE FRETTES STATIQUE (INLINE)
# -------------------------------------------------------------
# Nous figeons ici une discrétisation non uniforme (issue d'une génération
# antérieure). Plus aucune génération dynamique ni cache JSON.
# Si vous souhaitez régénérer : réintroduire temporairement le module
# fret_mesh et remplacer le bloc ci-dessous.
# =============================================================
_CONFIG_DIR = Path(__file__).resolve().parent  # conservé pour éventuelles sorties

# =============================================================
# 1. PARAMÈTRES PHYSIQUES DE LA CORDE
#    Exemple : corde nylon (Note ~ B3 ≈ 247 Hz)
# -------------------------------------------------------------
# L  : Longueur de la corde (m)
# T  : Tension (N)
# MU : Densité linéique μ (kg/m)
# -------------------------------------------------------------
# Valeurs de référence (adapter si besoin) :
L: float = 0.65          # 65 cm
T: float = 60.0          # Newtons
MU: float = 0.000582     # kg/m (0.5 g/m)

# Fréquence fondamentale théorique (idéale, sans rigidité ni amortissement additionnel)
# f1 = (1/(2L)) * sqrt(T/μ)
FUNDAMENTAL_FREQ_IDEAL: float = (1.0 / (2.0 * L)) * sqrt(T / MU)

# Vitesse de propagation de l'onde dans la corde (m/s)
# c = sqrt(T/μ)
WAVE_SPEED: float = sqrt(T / MU)

# =============================================================
# 2. AMORTISSEMENT (Rayleigh)
#    C = ALPHA * M + BETA * K
# -------------------------------------------------------------
# ALPHA agit surtout sur les basses fréquences
# BETA  agit surtout sur les hautes fréquences
# Commencer avec des valeurs faibles et affiner empiriquement.
#ALPHA: float = 0.045821 #2 * s * pi * 1/L * sqrt(T/μ) -> s < 1 (0,001 - 0,01)
#BETA: float = 0.00

# Nouvelles références modales pour calculer (alpha, beta) via Rayleigh
# ζ(ω) = 1/2 (α/ω + β ω) résolu sur deux modes de référence p, q (0-based)
DAMPING_MODES_REF: tuple[int, int] = (0, 1)
DAMPING_ZETAS_REF: tuple[float, float] = (0.01, 0.02)

# Conditions de bord (fixes aux extrémités)
APPLY_FIXED_BC: bool = True

# =============================================================
# 3. PARAMÈTRES NUMÉRIQUES DE LA SIMULATION
# -------------------------------------------------------------
# N_NODES : Nombre total de nœuds (>= 2). Nombre d'éléments = N_NODES - 1.
# T_SIM   : Durée totale de la simulation (s)
# DT      : Pas de temps (s) — choisir petit pour stabilité & précision.
N_NODES: int = 100          # Nombre de nœuds (>=2)
T_SIM: float = 2.0          # secondes
DT: float = 1.0e-5          # pas du temps

# -------------------------------------------------------------------------
# PARAMÈTRE GÉOMÉTRIQUE FRETTES
# -------------------------------------------------------------------------
# Exposant géométrique utilisé pour calculer les positions des divisions
# (frettes) le long de la corde : formule générique
#   pos_n = L * (1 - 1 / 2**(n / FRET_EXPOSANT_GEOMETRIQUE))
# Classique 12-TET -> 12 ; expérimental (ici) -> 8.
FRET_EXPOSANT_GEOMETRIQUE: float = 12.0

# -------------------------------------------------------------------------
# MAILLE FRETTES FIGÉE
# -------------------------------------------------------------------------
# Liste des longueurs élémentaires en millimètres (dx_i). Provenance :
# générée par l'ancien algorithme (date d'intégration statique).
# NOTE: Si vous modifiez L, vérifiez la cohérence avec la somme ci-dessous.
FRET_DXS_MM: list[float] = [
    # Liste régénérée (expoente=12.0, dx_alvo≈6.0 mm) le 2025-10-06.
    # Total éléments: 51 ; Somme ≈ 325.01 mm (couvre jusqu'à la 12e frette, pas la longueur totale L=650 mm).
    # NOTE: Maille partielle sur la zone des 12 premières frettes; adapter si besoin pour couvrir toute la corde.
    # Seg 1 (n=6, dx=6.08)
    6.08, 6.08, 6.08, 6.08, 6.08, 6.08,
    # Seg 2 (n=6, dx=5.74)
    5.74, 5.74, 5.74, 5.74, 5.74, 5.74,
    # Seg 3 (n=5, dx=6.50)
    6.50, 6.50, 6.50, 6.50, 6.50,
    # Seg 4 (n=4, dx=7.67)
    7.67, 7.67, 7.67, 7.67,
    # Seg 5 (n=4, dx=7.24)
    7.24, 7.24, 7.24, 7.24,
    # Seg 6 (n=4, dx=6.83)
    6.83, 6.83, 6.83, 6.83,
    # Seg 7 (n=5, dx=5.16)
    5.16, 5.16, 5.16, 5.16, 5.16,
    # Seg 8 (n=4, dx=6.09)
    6.09, 6.09, 6.09, 6.09,
    # Seg 9 (n=3, dx=7.66)
    7.66, 7.66, 7.66,
    # Seg 10 (n=3, dx=7.23)
    7.23, 7.23, 7.23,
    # Seg 11 (n=4, dx=5.12)
    5.12, 5.12, 5.12, 5.12,
    # Seg 12 (n=3, dx=6.44)
    6.44, 6.44, 6.44,
    # Ajout pour couvrir davantage la corde (total 49 éléments, somme ≈ 650 mm) 49 x 6,63
    6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63,
    6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63,
    6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63,
    6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63,
    6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63, 6.63     
]

FRET_N_ELEMS: int = len(FRET_DXS_MM)
FRET_N_NODES: int = FRET_N_ELEMS + 1

# Positions cumulées en millimètres (node_positions[0] = 0)
FRET_NODE_POSITIONS_MM: list[float] = [0.0]
_acc = 0.0
for _dx in FRET_DXS_MM:
    _acc += _dx
    FRET_NODE_POSITIONS_MM.append(_acc)
FRET_TOTAL_LENGTH_MM: float = FRET_NODE_POSITIONS_MM[-1]

# Vérification de cohérence avec L (tolérance relative 1%)
_length_m_recon = FRET_TOTAL_LENGTH_MM / 1000.0
if abs(_length_m_recon - L) / L > 0.01:
    # Avertissement simple (ne lève pas d'exception pour rester robuste)
    print(f"[AVERTISSEMENT] Longueur reconstruite {_length_m_recon:.6f} m ≠ L={L:.6f} m (>1%).")

# DX moyen (utile pour diagnostics uniquement)
DX: float = _length_m_recon / FRET_N_ELEMS if FRET_N_ELEMS else L

# Alias historiques pour compatibilité avec code existant
N_ELEMS: int = FRET_N_ELEMS
N_NODES: int = FRET_N_NODES

N_STEPS: int = int(T_SIM / DT)      # Nombre de pas de temps simulés

# (Optionnel) Critère de Courant (1D) pour maille uniforme : c * DT / DX <= ~1
COURANT_NUMBER: float = WAVE_SPEED * DT / DX

# ---------------------------------------------------------------------------
# 3.1 Utilitaires pour gérer deux types de maillage (uniforme vs frettes)
# ---------------------------------------------------------------------------
def has_fret_mesh() -> bool:
    """Retourne toujours True (maille frettes statique figée)."""
    return True


def fret_n_elems() -> int:
    """Nombre d'éléments de la maille frettes si présente sinon maille uniforme."""
    if 'FRET_N_ELEMS' in globals():
        return globals()['FRET_N_ELEMS']  # type: ignore[index]
    return N_ELEMS


def fret_n_nodes() -> int:
    """Nombre de nœuds de la maille frettes si présente sinon uniforme."""
    if 'FRET_N_NODES' in globals():
        return globals()['FRET_N_NODES']  # type: ignore[index]
    return N_NODES


def fret_dx_vector_m() -> list[float] | None:
    """Vecteur des dx (m) si maille frettes active, sinon None (maille uniforme)."""
    if has_fret_mesh():
        return [v / 1000.0 for v in globals()['FRET_DXS_MM']]  # type: ignore[index]
    return None


def effective_dx_uniform() -> float:
    """Retourne le DX uniforme de référence (m)."""
    return DX


def courant_number() -> float:
    """Retourne un nombre de Courant estimé.

    Si maille frettes : utilise le plus petit dx pour une estimation conservative.
    """
    fret_dxs = fret_dx_vector_m()
    if fret_dxs:
        dx_min = min(fret_dxs)
        return WAVE_SPEED * DT / dx_min
    return COURANT_NUMBER

# =============================================================
# 4. CONDITIONS INITIALES (PINÇAGE)
# -------------------------------------------------------------
# PLUCK_POS : Fraction de la longueur où la corde est pincée initialement (0 < x < 1)
# PLUCK_AMP : Amplitude initiale (m)
PLUCK_POS: float = 0.30    # 30% de L
PLUCK_AMP: float = 0.005   # 5 mm

# =============================================================
# 5. PARAMÈTRES DE SORTIE
# -------------------------------------------------------------
# FILENAME    : Nom de base des fichiers de sortie (sans extension)
# OUTPUT_NODE : Indice du nœud dont on extrait le "signal" (ex : milieu de la corde)
FILENAME: str = "simulacao_corda"
OUTPUT_NODE: int = N_NODES // 2

# =============================================================
# 6. FONCTIONS AUXILIAIRES (optionnel)
# -------------------------------------------------------------

def summary() -> str:
    """Retourne une chaîne formatée résumant les paramètres actuels."""
    fret_mode = 'OUI' if has_fret_mesh() else 'NON'
    fret_dxs_m = fret_dx_vector_m()
    if fret_dxs_m and len(fret_dxs_m) > 0:
        dx_info = f"Frettes: ~dx_min={min(fret_dxs_m):.6f} m, n_elems={fret_n_elems()}"
    else:
        dx_info = f"Uniforme DX={DX:.6f} m"
    # Valeurs effectives (si override core a été fait, N_NODES reflète déjà la maille frettes)
    n_nodes_effectif = fret_n_nodes() if has_fret_mesh() else N_NODES
    n_elems_effectif = fret_n_elems() if has_fret_mesh() else N_ELEMS
    dx_min_fret = (min(fret_dx_vector_m()) if fret_dx_vector_m() else DX)
    return (
        "Résumé de la simulation:\n"
        f"  Longueur L               = {L:.4f} m\n"
        f"  Tension T                = {T:.4f} N\n"
        f"  Densité linéique μ       = {MU:.6f} kg/m\n"
        f"  Vitesse d'onde c         = {WAVE_SPEED:.2f} m/s\n"
        f"  f1 idéale                = {FUNDAMENTAL_FREQ_IDEAL:.2f} Hz\n"
        #f"  Rayleigh alpha           = {ALPHA} \n"
        #f"  Rayleigh beta            = {BETA} \n"
        f"  Mode maille frettes      = {fret_mode}\n"
        f"  Info maille              = {dx_info}\n"
        f"  Nº nœuds effectif        = {n_nodes_effectif}\n"
        f"  Nº éléments effectif     = {n_elems_effectif}\n"
        f"  Δx min (effectif)        = {dx_min_fret:.6f} m\n"
        f"  Δt                       = {DT:.2e} s\n"
        f"  Pas (N_STEPS)            = {N_STEPS}\n"
        f"  Courant estimé           = {courant_number():.3f}\n"
        f"  Position pincement frac  = {PLUCK_POS:.2f}\n"
        f"  Amplitude pincement      = {PLUCK_AMP:.4f} m\n"
        f"  Nœud de sortie           = {OUTPUT_NODE}\n"
        f"  Nom base fichier         = {FILENAME}\n"
    )


def apply_fret_mesh_runtime(*args, **kwargs) -> bool:  # type: ignore[unused-argument]
    """(Obsolète) Conservée pour compatibilité — ne fait rien et retourne True."""
    return True

def export_fret_static(*args, **kwargs):  # type: ignore[unused-argument]
    """(Obsolète) La maille est déjà intégrée statiquement; rien à exporter."""
    return None

# --- Application automatique au moment de l'import du module ---
# Pour désactiver ce comportement, définir la variable d'environnement
# DT_DISABLE_AUTO_FRETS=1 avant d'importer ce module.
def ensure_fret_mesh(*args, **kwargs) -> bool:  # type: ignore[unused-argument]
    """Compatibilité : retourne toujours True (maille figée)."""
    return True

if __name__ == "__main__":
    # Uso manual: garantir malha e mostrar resumo.
    print("[INFO] Mode frettes statique (inline).")
    print(summary())
