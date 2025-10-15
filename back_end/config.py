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
# -------------------------------------------------------------
# Valeurs de référence (adapter si besoin)
# - Utilisé par: formulation (M,K), diagnostics et résumés
L: float = 0.65          # 65 cm
T: float = 60.0          # Newtons
MU: float = 0.000582     # kg/m (0.5 g/m)

# Fréquence fondamentale idéale (sans rigidité ni amortissement additionnel)
# f1 = (1/(2L)) * sqrt(T/μ)
# - Usage informatif (logs/résumé)
# - AVERTISSEMENT: Diagnostic uniquement — n'affecte pas M, K, C ni l'intégration
FUNDAMENTAL_FREQ_IDEAL: float = (1.0 / (2.0 * L)) * sqrt(T / MU)

# Vitesse de propagation de l'onde dans la corde (m/s)
# c = sqrt(T/μ)
# - Utilisé pour diagnostics (nombre de Courant, etc.)
# - AVERTISSEMENT: Diagnostic uniquement — n'affecte pas les calculs physiques
WAVE_SPEED: float = sqrt(T / MU)

# =============================================================
# 2. AMORTISSEMENT (Rayleigh)
#    C = α M + β K (déduit automatiquement à partir de cibles modales)
# -------------------------------------------------------------
# Paramètres cibles pour calculer (α, β): ζ(ω) = 1/2 (α/ω + β ω)
# Résolution sur deux modes de référence p, q (indices 0-based)
# - Utilisé par: formulation.rayleigh_damping via build_global_mkc_from_config
DAMPING_MODES_REF: tuple[int, int] = (0, 3)
DAMPING_ZETAS_REF: tuple[float, float] = (0.001, 0.001)
APPLY_FIXED_BC: bool = True

###############################################################
# 3. MAILLE FRETTES (DISCRÉTISATION) + FONCTIONS UTILITAIRES
# -------------------------------------------------------------
# Discrétisation non uniforme figée. Les longueurs élémentaires (dx_i) sont définies en mm.
# Fonctions utilitaires pour accéder au nombre de nœuds, d'éléments et aux positions.
###############################################################
DAMPING_MODES_REF: tuple[int, int] = (0, 3)
DAMPING_ZETAS_REF: tuple[float, float] = (0.001, 0.001)
APPLY_FIXED_BC: bool = True

###############################################################
# 3. PARAMÈTRES DE MAILLE FRETTES (DISCRÉTISATION)
# -------------------------------------------------------------
# La maille frettes est une discrétisation non uniforme figée.
# Les longueurs élémentaires (dx_i) sont définies en millimètres.
# Les fonctions utilitaires permettent d'accéder au nombre de nœuds,
# d'éléments et aux positions des nœuds.


# Conditions de bord fixes aux extrémités (Dirichlet)
# - Utilisé par: formulation (application des CL sur M,K, puis C)
APPLY_FIXED_BC: bool = True


# Le nombre de nœuds et d'éléments est toujours calculé à partir de la maille frettes (voir FRET_DXS_MM, FRET_N_NODES, FRET_N_ELEMS).
# Les paramètres d'intégration temporelle sont gérés par les modules solver et formulation.
# Ici, on ne définit que la structure de la maille.


# -------------------------------------------------------------------------
# PARAMÈTRE GÉOMÉTRIQUE FRETTES (référence documentaire)
# -------------------------------------------------------------------------
# Exposant géométrique utilisé pour calculer les positions des divisions
# (frettes) le long de la corde : formule générique
#   pos_n = L * (1 - 1 / 2**(n / FRET_EXPOSANT_GEOMETRIQUE))
# Classique 12-TET -> 12 ; expérimental (ici) -> 8.
FRET_EXPOSANT_GEOMETRIQUE: float = 12.0  # indicatif; non utilisé directement (maille figée)
# AVERTISSEMENT: Sans effet sur les calculs tant que la maille reste figée

# -------------------------------------------------------------------------
# MAILLE FRETTES FIGÉE (dx_i en millimètres)
# -------------------------------------------------------------------------
# Liste des longueurs élémentaires en millimètres (dx_i)
# - Utilisé par: formulation.build_global_mkc_from_config (converti en mètres)
#                solver.build_node_positions_from_config (positions x)
# NOTE: si vous modifiez L, vérifiez la cohérence avec la somme.
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

# Informations sur la maille frettes
# - Calculées à partir de FRET_DXS_MM
#   pour connaître le nombre d'éléments/nœuds et les positions x
# NOTE: Si vous modifiez FRET_DXS_MM, ces valeurs sont recalculées automatiquement.
# NOTE: Le nombre de noeuds sont nombre d'éléments + 1 parce que chaque élément a deux nœuds.
FRET_N_ELEMS: int = len(FRET_DXS_MM)            # Nombre d'éléments (segments)
FRET_N_NODES: int = FRET_N_ELEMS + 1            # Nombre de nœuds dans la maille frettes

# Positions cumulées en millimètres (node_positions[0] = 0)
FRET_NODE_POSITIONS_MM: list[float] = [0.0]
_acc = 0.0
for _dx in FRET_DXS_MM:
    _acc += _dx
    FRET_NODE_POSITIONS_MM.append(_acc)
FRET_TOTAL_LENGTH_MM: float = FRET_NODE_POSITIONS_MM[-1]

# Vérification de cohérence avec L (tolérance relative 1%) — avertissement seulement
_length_m_recon = FRET_TOTAL_LENGTH_MM / 1000.0
if abs(_length_m_recon - L) / L > 0.01:
    # Avertissement simple (ne lève pas d'exception pour rester robuste)
    # AVERTISSEMENT: Affichage d'un message uniquement — n'interrompt pas les calculs
    print(f"[AVERTISSEMENT] Longueur reconstruite {_length_m_recon:.6f} m ≠ L={L:.6f} m (>1%).")

# DX moyen (diagnostic)
# AVERTISSEMENT: Diagnostic uniquement — n'affecte pas l'assemblage ni l'intégration
DX: float = _length_m_recon / FRET_N_ELEMS if FRET_N_ELEMS else L
# N_STEPS, COURANT_NUMBER e outros parâmetros dependentes de T_SIM/DT removidos

# ---------------------------------------------------------------------------
# 3.1 Utilitaires pour gérer deux types de maillage (uniforme vs frettes)
# ---------------------------------------------------------------------------
def has_fret_mesh() -> bool:
    # Retourne toujours True (maille frettes statique figée).
    # Utilisé par diagnostics/helpers; n'affecte pas M/K/C ni l'intégration.
    return True


def fret_n_elems() -> int:
    # Nombre d'éléments de la maille frettes.
    return FRET_N_ELEMS


def fret_n_nodes() -> int:
    # Nombre de nœuds de la maille frettes.
    return FRET_N_NODES


def fret_dx_vector_m() -> list[float] | None:
    # Vecteur des dx (m) si maille frettes active; sinon None.
    # Utilisé par: formulation (assemblage non uniforme), solver (positions x).
    if has_fret_mesh():
        return [v / 1000.0 for v in globals()['FRET_DXS_MM']]  # type: ignore[index]
    return None


def effective_dx_uniform() -> float:
    # Retourne un DX uniforme de référence (m) — usage diagnostique uniquement.
    return DX


def courant_number() -> float:
    # Retourne un nombre de Courant estimé (diagnostic).
    # Maille frettes: utilise dx_min pour estimativa conservadora; usado por summary().
    fret_dxs = fret_dx_vector_m()
    # Parâmetro removido: DT não está mais definido. Retorne 0 ou None para diagnóstico.
    return 0.0

# =============================================================
# 4. CONDITIONS INITIALES (PINÇAGE)
# -------------------------------------------------------------
# - Utilisé par: solver.initialiser_etats_initiaux / initialiser_u0_triangle
# PLUCK_POS : Fraction de la longueur où la corde est pincée (0 < x < 1)
# PLUCK_AMP : Amplitude initiale (m)
PLUCK_POS: float = 0.7    # 50% de L
PLUCK_AMP: float = 0.003  # 3 mm

# =============================================================
# 5. PARAMÈTRES DE SORTIE/TRACE
# -------------------------------------------------------------
# - Utilisé par: solver (séries temporelles, FFT, figures/animations)
# FILENAME    : Nom de base des fichiers de sortie (sans extension)
# OUTPUT_NODE : Indice du nœud d'où est extrait le signal (ex.: milieu)
# AVERTISSEMENT: Contrôles d'E/S uniquement — n'affectent pas les calculs physiques
FILENAME: str = "simulacao_corda"
OUTPUT_NODE: int = FRET_N_NODES // 2  # nœud central (indice 0-based)

# Contrôles d'animation (GIF/MP4) — utilisés par solver.animate_string_motion
# AVERTISSEMENT: Visualisation uniquement — n'affecte pas M/K/C ou Newmark
# - ANIM_FPS: images/s du fichier final
# - ANIM_FRAME_STEP: pas des trames échantillonnées (None -> auto)
# - ANIM_VIDEO_DURATION_S: durée cible; si définie, rééchantillonne pour coller à cette durée
ANIM_FPS: int = 30
ANIM_FRAME_STEP: int | None = 50
ANIM_VIDEO_DURATION_S: float | None = None  # Ex.: 5.0 pour une vidéo ~5 s

# Zoom/limites pour animation (ANIM_Y_SCALE > 1.0 = « zoom out »)
# Limites Y basées sur les percentiles 1–99 et marge ANIM_Y_PAD_FRAC
ANIM_Y_SCALE: float = 2          # facteur d'échelle vertical (None = auto)
ANIM_Y_PAD_FRAC: float = 0.08    # marge verticale relative

# Captures statiques (PNG) — utilisées par solver.plot_snapshots_png
# AVERTISSEMENT: Visualisation uniquement — n'affecte pas les calculs
SNAPSHOTS_COUNT: int = 50         # nombre de captures (uniformément réparties)
SNAPSHOTS_DECIM: int = 10        # décimation temporelle avant sélection
SNAPSHOTS_T_WINDOW = None        # ex.: (0.0, 0.2) pour limiter la plage temporelle
SNAPSHOTS_Y_SCALE = 1.5          # facteur d'échelle vertical (None = auto)
SNAPSHOTS_Y_PAD_FRAC = 0.06      # marge verticale relative
SNAPSHOTS_USE_COLORBAR = True    # barre de couleur mappée au temps
SNAPSHOTS_CMAP = "viridis"       # colormap pour les profils (si pas de colorbar, sinon viridis)
SNAPSHOTS_ALPHA = 0.9            # transparence des profils
SNAPSHOTS_LINEWIDTH = 1.3        # épaisseur des lignes
SNAPSHOTS_SHOW_LEGEND = False    # légende avec temps (utile si peu de profils)

# Interrupteur central de débogage (impressions contrôlées globalement)
# - Utilisé par: digital_twin.back_end.utils.debug (dbg.is_enabled())
# - Mettre True pour activer les impressions de debug de tous les modules qui utilisent dbg
# AVERTISSEMENT: Debug/prints uniquement — n'affecte pas les matrices ni l'intégration
DEBUG_ENABLED: bool = True

# Niveau de verbosité du debug: 'off' | 'concise' | 'verbose'
# - concise: imprime resumos curtos e poucas amostras por passo
# - verbose: imprime diagnósticos detalhados (blocos/matrizes/explicações longas)
DEBUG_LEVEL: str = "concise"

# Controles finos por categoria (lidos se DEBUG_ENABLED=True)
# Rayleigh
DEBUG_PRINT_RAYLEIGH_SUMMARY: bool = True
DEBUG_PRINT_RAYLEIGH_DETAILS: bool = False

# Validadores (M/C/K): em modo conciso imprime apenas um resumo breve
DEBUG_PRINT_VALIDATORS: bool = False

# Passos do solver: quantidade alvo de snapshots durante a simulação inteira
# get_solver_sample_interval calcula o intervalo a partir deste alvo
DEBUG_SOLVER_SAMPLES_TARGET: int = 12

# Informações de energia em cada passo de impressão (custoso)
# Em modo conciso, deixar False para acelerar
DEBUG_PRINT_STEP_ENERGY: bool = False

# Habilitar/desabilitar completamente as impressões de passos
DEBUG_PRINT_STEPS: bool = False

# Quando > 0, força a impressão de passos a cada N passos (substitui a amostragem automática)
# Ex.: 1000 → imprime k=1000, 2000, ...; 0 → desativa e usa a amostragem automática (~alvo DEBUG_SOLVER_SAMPLES_TARGET)
DEBUG_STEP_EVERY: int = 0

# Resumo final de energia (início x fim)
DEBUG_PRINT_FINAL_ENERGY_SUMMARY: bool = False

# Alias rétrocompatibilité: certains modules peuvent encore lire DEBUG_RAYLEIGH
# (le module debug le considère si DEBUG_ENABLED n'est pas défini)
DEBUG_RAYLEIGH: bool = DEBUG_ENABLED

# Interrupteurs pour activer/désactiver les sorties de fichiers (pour débogage terminal uniquement)
# - OUTPUT_ENABLE_IMAGES : contrôle tous les PNG (modes, x(t), énergies, FFT, snapshots, frame t0)
# - OUTPUT_ENABLE_GIFS   : contrôle l'animation GIF
# - OUTPUT_ENABLE_CSV    : contrôle l'export CSV des déplacements
# AVERTISSEMENT: Ces interrupteurs n'affectent pas le calcul (Newmark, M/K/C), uniquement les E/S.
OUTPUT_ENABLE_IMAGES: bool = True
OUTPUT_ENABLE_GIFS: bool = True
OUTPUT_ENABLE_CSV: bool = True

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


# =============================================================
# 5.3. FORCE D'EXCITATION LOCALISÉE (corde pincée au temps)
# -------------------------------------------------------------
# Ces paramètres contrôlent l'enveloppe trapézoïdale utilisée par
# time_integration.fournisseur_force_localisee. Le nœud d'application est
# choisi à partir de PLUCK_POS (fraction de L), en prenant le nœud le plus proche.
# - EXCITATION_F_MAX  : amplitude maximale (Newtons)
# - EXCITATION_T_RISE : montée (s)
# - EXCITATION_T_HOLD : palier (s)
# - EXCITATION_T_DECAY: décroissance (s)
EXCITATION_F_MAX: float = 1.0
EXCITATION_T_RISE: float = 0.001
EXCITATION_T_HOLD: float = 0.001
EXCITATION_T_DECAY: float = 0.003

# Deuxième excitation optionnelle (re-pincement) — temps de début (s)
# - Si <= 0, aucune deuxième excitation n'est ajoutée.
EXCITATION_SECOND_T0: float = 0

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
# 5.4. PRESS EVENTS (pression du doigt)
# -------------------------------------------------------------
# Activation des événements de pression de nœuds internes (ajoute K locale et C via Rayleigh)
# Quand True, le main utilise simulate_with_press; quand False, flux normal.
PRESS_EVENTS_ENABLED: bool = False

# Paramètres d'exemple/config par défaut (peuvent être substitués):
# - PRESS_NODE_INDEX: indice du nœud à presser (si None, usa ~0.3·(n_nodes-1))
# - PRESS_T_ON / PRESS_T_OFF: janela de tempo [s]
# - PRESS_KS: rigidez local (N/m)
# - PRESS_CS: amortecimento local adicional (N·s/m). 0.0 → Rayleigh puro
PRESS_NODE_INDEX: int | None = None
PRESS_T_ON: float = 0.5
PRESS_T_OFF: float = 8.0
PRESS_KS: float = 5e6
PRESS_CS: float = 0.0

# =============================================================
# 6. FONCTIONS AUXILIAIRES (optionnel)
# -------------------------------------------------------------

def summary() -> str:
    # Retourne une chaîne formatée résumant les paramètres actuels (diagnostic/affichage).
    # Utile pour confirmer ce que formulation/solver consommerão.
    fret_mode = 'OUI' if has_fret_mesh() else 'NON'
    fret_dxs_m = fret_dx_vector_m()
    if fret_dxs_m and len(fret_dxs_m) > 0:
        dx_info = f"Frettes: ~dx_min={min(fret_dxs_m):.6f} m, n_elems={fret_n_elems()}"
    else:
        dx_info = f"Uniforme DX={DX:.6f} m"
    # Valeurs effectives (si override core a été fait, N_NODES reflète déjà la maille frettes)
    n_nodes_effectif = fret_n_nodes()
    n_elems_effectif = fret_n_elems()
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
    # f"  Δt                       = {DT:.2e} s\n"  # Parâmetro removido
        f"  Courant estimé           = {courant_number():.3f}\n"
        f"  Position pincement frac  = {PLUCK_POS:.2f}\n"
        f"  Amplitude pincement      = {PLUCK_AMP:.4f} m\n"
        f"  Nœud de sortie           = {OUTPUT_NODE}\n"
        f"  Nom base fichier         = {FILENAME}\n"
    )

if __name__ == "__main__":
    # Usage manuel: imprimer un résumé lisible.
    print("[INFO] Mode frettes statique (inline).")
    print(summary())
