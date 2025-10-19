###############################################################
# CONFIGURATION CENTRALE DE LA SIMULATION FEM DE CORDE VIBRANTE
# -------------------------------------------------------------
# Ce fichier regroupe tous les paramètres physiques, d'amortissement,
# de maillage, d'intégration, d'excitation, de pression, de sortie,
# d'analyse spectrale et d'animation. Organisation stricte par thème.
###############################################################
from __future__ import annotations
from pathlib import Path
from math import sqrt

_CONFIG_DIR = Path(__file__).resolve().parent  # conservé pour éventuelles sorties

# =============================================================
# 1. PARAMÈTRES PHYSIQUES DE LA CORDE
#    Exemple : corde nylon (note ~ Si3 ≈ 247 Hz)
# -------------------------------------------------------------

# L  : Longueur de la corde (m)
# T  : Tension (N)
# MU : Densité linéique μ (kg/m)
# WAVE_SPEED : Vitesse de l'onde (m/s)
# FUNDAMENTAL_FREQ_IDEAL : Fréquence fondamentale idéale (Hz) (sans rigidité ni amortissement)
L: float = 0.65                     # cm
T: float = 60.0                     # Newtons
MU: float = 0.000582                # kg/m (0.5 g/m)
WAVE_SPEED: float = sqrt(T / MU)                                    # c(m/s) = sqrt(T/μ)
FUNDAMENTAL_FREQ_IDEAL: float = (1.0 / (2.0 * L)) * WAVE_SPEED      # f1(Hz) = (1/(2L)) * c
# NOTE : Diagnostic uniquement — n'affecte pas M, K, C ni l'intégration

# =============================================================
# 2. AMORTISSEMENT (Rayleigh)
#    C = α M + β K (déduit automatiquement à partir de cibles modales)
# -------------------------------------------------------------
# Paramètres cibles pour calculer (α, β): ζ(ω) = 1/2 (α/ω + β ω)

# MODE_1 et MODE_4 : modes cibles (indices 0-based)
# ZETA_1 et ZETA_4 : amortissements cibles (facteurs de perte, ζ = c/(2mω))
# Booléen pour activer/désactiver l'amortissement Rayleigh
DAMPING_MODES_REF: tuple[int, int] = (0, 3)                         
DAMPING_ZETAS_REF: tuple[float, float] = (0.001, 0.001)             
APPLY_FIXED_BC: bool = True                                         

# =============================================================
# 3. MAILLE FRETTES (DISCRÉTISATION) + FONCTIONS UTILITAIRES
# -------------------------------------------------------------
# Discrétisation non uniforme figée. Les longueurs élémentaires (dx_i) sont définies en mm.
# Fonctions utilitaires pour accéder au nombre de nœuds, d'éléments et aux positions.

# Exposant géométrique utilisé pour calculer les positions des divisions 
# (frettes) le long de la corde : formule générique
#   pos_n = L * (1 - 1 / 2**(n / FRET_EXPOSANT_GEOMETRIQUE))
FRET_EXPOSANT_GEOMETRIQUE: float = 12.0 # 12-TET (demitons égaux)
# NOTE : Sans effet sur les calculs tant que la maille reste figée
# NOTE : C'est juste utilisé dans le fiche calcul_trous_frettes.py et 
# n'est pas intégré dans le reste du code. Donc, si vous modifiez cette valeur,
# cela n'affectera pas la simulation actuelle, il faut régénérer FRET_DXS_MM.

# Liste des longueurs élémentaires en millimètres (dx_i)
FRET_DXS_MM: list[float] = [
    # Liste régénérée (expoente=12.0, dx_alvo≈6.0 mm).
    # Total éléments: 100 -> 325.01 mm (juste les 12 premières frettes) + 324.99 mm (restants).
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
# NOTE : Pour déterminer touts le dx_i (Seg 1 - Seg 12), a été utilisé le script calcul_trous_frettes.py
# NOTE : Le restant des éléments (49 x 6.63 mm) a été ajouté pour couvrir toute la corde (~650 mm) et
# obtenir un nombre raisonnable d'éléments (~100) pour la simulation.

# FRET_N_ELEMS : nombre d'éléments dans la maille
# FRET_N_NODES : nombre de nœuds dans la maille
FRET_N_ELEMS: int = len(FRET_DXS_MM)            
FRET_N_NODES: int = FRET_N_ELEMS + 1           
# NOTE : Si vous modifiez FRET_DXS_MM, ces valeurs sont recalculées automatiquement.
# NOTE : Le nombre de noeuds sont nombre d'éléments + 1 parce que chaque élément a deux nœuds.

# Le nombre de nœuds et d'éléments est toujours calculé à partir de la maille frettes (voir FRET_DXS_MM, FRET_N_NODES, FRET_N_ELEMS).
# Les paramètres d'intégration temporelle sont gérés par les modules solver et formulation.
# Ici, on ne définit que la structure de la maille.

# Positions cumulées en millimètres (node_positions[0] = 0)
FRET_NODE_POSITIONS_MM: list[float] = [0.0]
_acc = 0.0
for _dx in FRET_DXS_MM:
    _acc += _dx
    FRET_NODE_POSITIONS_MM.append(_acc)
FRET_TOTAL_LENGTH_MM: float = FRET_NODE_POSITIONS_MM[-1]
# NOTE : Cette variable est utilisée juste dans le frontend.

# Vérification de cohérence avec L (tolérance relative 1%)
# _length_m_recon : longueur reconstruite (m)
_length_m_recon = FRET_TOTAL_LENGTH_MM / 1000.0
# Verification simple (ne lève pas d'exception)
if abs(_length_m_recon - L) / L > 0.01:
    print(f"[AVERTISSEMENT] Longueur reconstruite {_length_m_recon:.6f} m ≠ L={L:.6f} m (>1%).")
# NOTE : Ceci est juste un avertissement, la simulation continue.

# DX : valeur de référence Δx (m) pour diagnostics (valeur moyenne des frettes)
DX: float = _length_m_recon / FRET_N_ELEMS if FRET_N_ELEMS else L
# NOTE : DX n'affecte pas les calculs, il est juste utilisé pour des diagnostics.

# Pré-calculer la version en mètres des dx (mm -> m) pour éviter conversions répétées.
# Remplaçant les petites fonctions d'accès (plus simples et plus direct).
FRET_DXS_M: list[float] = [v / 1000.0 for v in FRET_DXS_MM] 

# =============================================================
# 4. PPINCAGE (CONDITIONS INITIALES)
# -------------------------------------------------------------
# PLUCK_POS : Fraction de la longueur où la corde est pincée (0 < x < 1)
# PLUCK_AMP : Amplitude initiale de pincement (m)
PLUCK_POS: float = 0.7    # 0 < PLUCK_POS ≤ 1
PLUCK_AMP: float = 0.003  # mm

# =============================================================
# 4.1. FORCE D'EXCITATION LOCALISÉE (corde pincée au temps)
# -------------------------------------------------------------
# Ces paramètres contrôlent l'enveloppe trapézoïdale utilisée par
# time_integration.fournisseur_force_localisee. Le nœud d'application est
# choisi à partir de PLUCK_POS (fraction de L), en prenant le nœud le plus proche.
# EXCITATION_F_MAX  : amplitude maximale (Newtons)
# EXCITATION_T_RISE : temps de montée (s)
# EXCITATION_T_HOLD : temps de palier (s)
# EXCITATION_T_DECAY: temps de décroissance (s)
EXCITATION_F_MAX: float = 1.0           # Newts
EXCITATION_T_RISE: float = 0.001        # s
EXCITATION_T_HOLD: float = 0.001        # s
EXCITATION_T_DECAY: float = 0.003       # s
# NOTE : Ces paramètres définissent une force temporelle localisée appliquée

# Deuxième excitation optionnelle (re-pincement) — temps de début (s)
# - Si <= 0, aucune deuxième excitation n'est ajoutée.
EXCITATION_SECOND_T0: float = 0

# =============================================================
# 5. PARAMÈTRES DE FORCE DE PRESSION (interaction externe)
# -------------------------------------------------------------
# Paramètres d'exemple/config par défaut (peuvent être substitués):
# PRESS_NODE_INDEX: indice du nœud à presser (si None, usa ~0.3·(n_nodes-1))
# PRESS_T_ON / PRESS_T_OFF: fenêtre de temps [s]
# PRESS_KS: rigueur locale (N/m)
# PRESS_CS: amortissement local supplémentaire (N·s/m). 0.0 → Rayleigh pur
PRESS_NODE_INDEX: int | None = None
PRESS_T_ON: float = 0.5                 # s
PRESS_T_OFF: float = 8.0                # s
PRESS_KS: float = 5e6                   # N/m
PRESS_CS: float = 0.0                   # N·s/m

# =============================================================
# 5. PARAMÈTRES DE SORTIE/TRACE
# -------------------------------------------------------------
# OUTPUT_NODE : Indice du nœud d'où est extrait le signal (ex.: milieu)
OUTPUT_NODE: int = FRET_N_NODES // 2
# NOTE : Ça c'est basé sur un vecteur 0-based, donc FRET_N_NODES//2 est le nœud central.

# Contrôles d'animation (GIF/MP4) — utilisés par solver.animate_string_motion
# ANIM_FPS: images/s du fichier final
# ANIM_FRAME_STEP: pas des trames échantillonnées (None -> auto)
# ANIM_VIDEO_DURATION_S: durée cible; si définie, rééchantillonne pour coller à cette durée
# ANIM_Y_SCALE > 1.0 = « zoom out »
# ANIM_Y_PAD_FRAC : marges verticales relatives (0.0–0.5)
ANIM_FPS: int = 30                              # images/s
ANIM_FRAME_STEP: int                            # frames
ANIM_VIDEO_DURATION_S: float | None = None      # s
ANIM_Y_SCALE: float = 2                         
ANIM_Y_PAD_FRAC: float = 0.08

# Captures statiques (PNG) — utilisées par solver.plot_snapshots_png
# SNAPSHOTS_COUNT : nombre de captures (uniformément réparties)
# SNAPSHOTS_DECIM : décimation temporelle avant sélection
# SNAPSHOTS_T_WINDOW : tuple (t_min, t_max) pour limiter la plage temporelle (None = tout)
# SNAPSHOTS_Y_SCALE > 1.0 = « zoom out »
# SNAPSHOTS_Y_PAD_FRAC : marge verticale relative (0.0–0.5)
# SNAPSHOTS_USE_COLORBAR : barre de couleur mappée au temps
# SNAPSHOTS_CMAP : colormap pour les profils (si pas de colorbar, sinon viridis)
# SNAPSHOTS_ALPHA : transparence des profils (0.0–1.0)
# SNAPSHOTS_LINEWIDTH : épaisseur des lignes
# SNAPSHOTS_SHOW_LEGEND : légende avec temps (utile si peu de profils)
SNAPSHOTS_COUNT: int = 50        # captures
SNAPSHOTS_DECIM: int = 10
SNAPSHOTS_T_WINDOW = None        # ex.: (0.0, 0.2) pour limiter la plage temporelle
SNAPSHOTS_Y_SCALE = 1.5      
SNAPSHOTS_Y_PAD_FRAC = 0.06
SNAPSHOTS_USE_COLORBAR = True    # True / False
SNAPSHOTS_CMAP = "viridis"
SNAPSHOTS_ALPHA = 0.9            # 0 < SNAPSHOTS_ALPHA ≤ 1.0
SNAPSHOTS_LINEWIDTH = 1.3        
SNAPSHOTS_SHOW_LEGEND = False    # True / False

# Interrupteur central de débogage (único ON/OFF)
# Agora todas as flags de debug dependem apenas de DEBUG_ENABLED.
DEBUG_ENABLED: bool = True

# Interrupteurs pour activer/désactiver les sorties de fichiers (pour débogage terminal uniquement)
# OUTPUT_ENABLE_IMAGES : contrôle tous les PNG (modes, x(t), énergies, FFT, snapshots, frame t0)
# OUTPUT_ENABLE_GIFS : contrôle l'animation GIF
# OUTPUT_ENABLE_CSV : contrôle l'export CSV des déplacements
OUTPUT_ENABLE_IMAGES: bool = True
OUTPUT_ENABLE_GIFS: bool = True
OUTPUT_ENABLE_CSV: bool = True
# NOTE : Ces interrupteurs n'affectent pas le calcul (Newmark, M/K/C), uniquement les E/S.

# Activer/désactiver l'utilisation d'événements de pression (simulate_with_press)
PRESS_EVENTS_ENABLED: bool = True

# =============================================================
# 5.1. PARAMÈTRES DE FFT (consommés par solver)
# -------------------------------------------------------------
# AVERTISSEMENT (FFT): Tous les paramètres de cette section influencent uniquement l'analyse/les graphiques
# et ne modifient pas l'intégration temporelle ni les matrices M/K/C.
# Style préféré pour le tracé FFT dans le backend/main
#   'linear'  → tracer_fft_png (amplitude lin. + panneau dB)
#   'logdb'   → tracer_fft_logdb_remplie (axe fréquence log, amplitude dB remplie)
FFT_STYLE: str = "logdb"

# Contrôles généraux (FFT linéaire et dB)
FFT_WINDOW: str = "hann"           # fenêtre: hann, hamming, blackman, flattop, etc.
FFT_ZERO_PAD_FACTOR: float = 1.0    # >=1.0; zero-padding pour raffiner l'échantillonnage en fréquence

# Tracé FFT à axe de fréquence logarithmique (zone remplie en dB)
FFT_LOG_FMIN: float = 30.0          # Hz (log nécessite fmin > 0)
FFT_LOG_FMAX: float | None = None   # Hz (None -> Nyquist)
FFT_LOG_MIN_DB: float = -90.0       # plancher dB pour l'affichage
FFT_LOG_SMOOTH: int = 0             # lissage (échantillons) avant dB; 0 = off
FFT_LOG_USE_VELOCITY: bool = False  # utiliser la vitesse au lieu du déplacement
FFT_LOG_COLOR: str = "#4c78a8"      # couleur de remplissage
FFT_LOG_DB_OFFSET: float = 0.0      # décalage vertical (dB) pour alignement visuel
FFT_LOG_BINS_PER_OCTAVE: int | None = None  # rééchantillonnage log (bins/octave); None = off
FFT_LOG_OCTAVE_SMOOTH: float = 0.0  # lissage fraction d'octave (en octaves)
FFT_LOG_SMOOTH_DOMAIN: str = "db"   # "db" (par défaut) ou "linear"

# Optionnel: spectrogramme (temps-fréquence) — utile pour visualiser l'évolution spectrale
ENABLE_SPECTROGRAM: bool = False
SPECTROGRAM_FMIN: float = 0.0       # Hz
SPECTROGRAM_FMAX: float | None = None
SPECTROGRAM_NPERSEG: int | None = None   # None → auto (~1/20 de la durée)
SPECTROGRAM_OVERLAP: float = 0.5         # fraction [0,1), ex.: 0.5 → 50% recouvrement
SPECTROGRAM_CMAP: str = "magma"

# =============================================================
# 5.2. PARAMÈTRES D'ANIMATION ADDITIONNELS
# -------------------------------------------------------------
# Contrôles spécifiques pour deux GIFs générés par solver
# AVERTISSEMENT: Visualisation uniquement — n'affecte pas les résultats physiques
ANIM_DECIM_SLOW: int = 5            # décimation pour le GIF au ralenti
ANIM_FPS_SLOW: int = 33             # FPS du GIF au ralenti
ANIM_FPS_REAL: int = 30             # FPS du GIF « temps réel »

# Durée alvo (optionnelle) para o GIF slow, em segundos. Se definida (>0), o código
# tentará ajustar a decimação temporal para que a duração do GIF seja ≈ ANIM_SLOW_DURATION_S.
# Caso não definida, usa-se ANIM_SLOW_FACTOR (se existir) ou ANIM_DECIM_SLOW (fallback antigo).
# Observação: a duração máxima atingível é (n_passos_totais / ANIM_FPS_SLOW).
ANIM_SLOW_DURATION_S: float | None = 60.0


# Extension automatique de la simulation « jusqu'à arrêt » (décroissance)
# - Si True, la simulation est exécutée par tranches (CHUNK_SECONDS) jusqu'à
#   ce que le déplacement et la vitesse restent sous des seuils pendant une
#   fenêtre STOP_WINDOW_SEC, ou que MAX_SIM_SECONDS soit atteint.
AUTO_EXTEND_SIM: bool = False
MAX_SIM_SECONDS: float = 3.0     # borne supérieure de sécurité
CHUNK_SECONDS: float = 0.5       # taille des tranches d'intégration
STOP_WINDOW_SEC: float = 0.2     # fenêtre d'observation de fin (secondes)
STOP_THRESH_U: float = 1e-6      # seuil déplacement (m)
STOP_THRESH_V: float = 1e-4      # seuil vitesse (m/s)

# =============================================================
# 6. FONCTIONS AUXILIAIRES (optionnel)
# -------------------------------------------------------------

def summary() -> str:
    # Retourne une chaîne formatée résumant les paramètres actuels (diagnostic/affichage).
    # Utile pour confirmer ce que formulation/solver consommerão.
    fret_mode = 'OUI'
    fret_dxs_m = FRET_DXS_M if len(FRET_DXS_M) > 0 else None
    if fret_dxs_m and len(fret_dxs_m) > 0:
        dx_info = f"Frettes: ~dx_min={min(fret_dxs_m):.6f} m, n_elems={FRET_N_ELEMS}"
    else:
        dx_info = f"DX moyen={DX:.6f} m (maille frettes)"
    # Valeurs effectives (si override core a été fait, N_NODES reflète déjà la maille frettes)
    n_nodes_effectif = FRET_N_NODES
    n_elems_effectif = FRET_N_ELEMS
    dx_min_fret = (min(FRET_DXS_M) if len(FRET_DXS_M) > 0 else DX)
    return (
        "Résumé de la simulation:\n"
        f"- Longueur corde L = {L:.6f} m\n"
        f"- Densité linéique MU = {MU:.6f} kg/m\n"
        f"- Tension T = {T:.2f} N\n"
        f"- Vitesse onde c = {WAVE_SPEED:.2f} m/s\n"
        f"- Fréquence fondamentale idéale f1 = {FUNDAMENTAL_FREQ_IDEAL:.2f} Hz\n"
        f"- Amortissement Rayleigh activé: {DAMPING_MODES_REF} avec ζ = {DAMPING_ZETAS_REF}\n"
        f"- Maille: {dx_info}\n"
        f"- Nœuds effectifs: {n_nodes_effectif}, Éléments effectifs: {n_elems_effectif}\n"
        f"- DX min frettes = {dx_min_fret:.6f} m\n"
        f"- Condition initiale: pincement à {PLUCK_POS*L:.3f} m avec amplitude {PLUCK_AMP:.6f} m\n"
        f"- Nœud de sortie: {OUTPUT_NODE}\n"
        f"- Debug activé: {DEBUG_ENABLED}\n"
    )

if __name__ == "__main__":
    # Usage manuel: imprimer un résumé lisible.
    print("[INFO] Mode frettes statique (inline).")
    print(summary())
