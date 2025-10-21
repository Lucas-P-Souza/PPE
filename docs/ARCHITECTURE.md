# Digital Twin — Architecture et Guide Technique

Ce document décrit l'architecture du projet `digital_twin` de manière fidèle au code présent dans l'arborescence du dépôt. Il est rédigé en français et contient : la structure du projet, le flux d'exécution, les choix d'architecture, les dépendances principales, les paramètres importants et des recommandations pour un rapport technique.

## 1. Type d'architecture

- Architecture modulaire : séparation claire par dossiers (fem, analysis, viz, io, etc.).
- Approche orientée-objet : le code majeur utilise des classes pour encapsuler la logique des composants (solveur, éléments modaux, widgets GUI).
- Séparation CLI / GUI : le backend est exécutable en ligne de commande via `back_end/main.py`, et une interface utilisateur Qt est disponible dans `front_end/`.
- Flux de données centralisé : sorties et artefacts produits dans `back_end/results/plots/` pour traçabilité.

## 2. Informations techniques clés

- Langage : Python 3.13
- Bibliothèques principales : NumPy, SciPy, Matplotlib, PyQt5
- Formats d'export : CSV, PNG, GIF, WAV
- Style FFT configurable : linéaire ou log-dB (`FFT_STYLE` dans `config.py`)
- Mode debug/diagnostic : activable via `DEBUG_ENABLED` dans `config.py`

Remarque : les chemins et noms de fichiers utilisés dans ce document correspondent exactement aux fichiers présents dans l'arborescence du dépôt au 21/10/2025.

## 3. Arborescence principale (extrait)

```
digital_twin/
├── back_end/
│   ├── analysis/
│   │   ├── fft.py
```
## Nettoyage et consolidation
Le document présentait des sections en double (répétition de l'introduction, des sections 1→6 et du titre). J'ai consolidé le contenu en une seule version cohérente en français. Ci-dessous figure la version unique et épurée (les doublons supprimés).

# Digital Twin — Architecture et Guide technique

Ce document décrit l'architecture du projet `digital_twin` et sert de guide technique aligné sur le code présent dans le dépôt. Il a été relu et consolidé en français. Il couvre la structure, le flux d'exécution, les choix d'architecture, les dépendances principales, les paramètres importants et des recommandations pour la maintenance et les tests.

Remarque : les chemins et noms de fichiers indiqués correspondent à l'état du dépôt au 21/10/2025.

## 1. Vue d'ensemble de l'architecture

- Architecture modulaire — séparation claire par dossiers (fem, analysis, viz, io, etc.).
- Conception orientée-objet pour les composants principaux (solveur, éléments modaux, widgets de l'interface).
- Séparation CLI / GUI — le backend s'exécute en ligne de commande via `back_end/main.py`, l'interface graphique est dans `front_end/`.
- Flux de données centralisé — les résultats et artefacts sont produits dans `back_end/results/plots/` pour assurer traçabilité.

## 2. Informations techniques principales

- Langage : Python 3.13
# Digital Twin — Architecture et Guide technique

Ce document décrit l'architecture du projet `digital_twin` et sert de guide technique aligné sur le code présent dans le dépôt. Il a été relu et consolidé en français. Il couvre la structure, le flux d'exécution, les choix d'architecture, les dépendances principales, les paramètres importants et des recommandations pour la maintenance et les tests.

Remarque : les chemins et noms de fichiers indiqués correspondent à l'état du dépôt au 21/10/2025.

## 1. Vue d'ensemble de l'architecture

- Architecture modulaire — séparation claire par dossiers (fem, analysis, viz, io, etc.).
- Conception orientée-objet pour les composants principaux (solveur, éléments modaux, widgets de l'interface).
- Séparation CLI / GUI — le backend s'exécute en ligne de commande via `back_end/main.py`, l'interface graphique est dans `front_end/`.
- Flux de données centralisé — les résultats et artefacts sont produits dans `back_end/results/plots/` pour assurer traçabilité.

## 2. Informations techniques principales

- Langage : Python 3.13
- Bibliothèques usuelles : NumPy, SciPy, Matplotlib, PyQt5
- Formats d'export : CSV, PNG, GIF, WAV
- FFT : style configurable (linéaire ou log-dB) via `back_end/config.py`
- Mode debug/diagnostic activable via `DEBUG_ENABLED` dans `back_end/config.py`

## 3. Arborescence (extrait)

```
digital_twin/
├── back_end/
│   ├── analysis/
│   │   ├── fft.py
│   │   ├── spectrogram.py
│   │   └── __init__.py
│   ├── audio/
│   │   └── generate_audio_from_positions.py
│   ├── fem/
│   │   ├── formulation.py
│   │   ├── modal.py
│   │   ├── time_integration.py
│   │   └── __init__.py
│   ├── interactions/
│   │   ├── excitation.py
│   │   ├── press.py
│   │   └── __init__.py
│   ├── io/
│   │   ├── exports.py
│   │   └── __init__.py
│   ├── mesh/
│   │   ├── calcul_trous_de_frette.py
│   │   ├── fret_mesh.py
│   │   └── __init__.py
│   ├── results/
│   │   ├── animate_csv_temp.py
│   │   ├── plot_csv_temp.py
│   │   ├── plot_force_over_time.py
│   │   ├── profils_symetriques_plateau.py
│   │   └── plots/
│   │       ├── string_positions.csv
│   │       ├── string_debut.wav
│   │       ├── string_fin.wav
│   │       ├── string_milieu.wav
│   │       ├── string_quart.wav
│   │       ├── string_trois_quarts.wav
│   │       └── ...
│   ├── utils/
│   │   ├── debug.py
│   │   ├── validators.py
│   │   └── __init__.py
│   ├── viz/
│   │   ├── anim.py
│   │   ├── plots.py
│   │   └── __init__.py
│   ├── config.py
│   ├── main.py
│   └── __init__.py
├── front_end/
│   ├── corde_widget.py
│   ├── gui.py
│   ├── main.py
│   ├── settings_dialog.py
│   ├── styles/
│   │   └── main_style.qss
│   ├── README.md
│   └── __init__.py
├── docs/
│   ├── ARCHITECTURE.md
│   └── BACKEND_TIME_INTEGRATION.md
├── tests/
├── README.md
└── __init__.py
```

## 4. Description des modules

- `back_end/config.py` : centralise tous les paramètres (physique, maillage, intégration, options de sortie, toggles de debug).
- `back_end/fem/formulation.py` : assemble les matrices globales (M, K) et fournit des utilitaires pour appliquer les conditions aux limites.
- `back_end/fem/modal.py` : calcul des fréquences propres et des modes; fonctions de tri et de normalisation.
- `back_end/fem/time_integration.py` : implémentation de Newmark-β (β=1/4, γ=1/2) et calcul des énergies.
- `back_end/analysis/fft.py` : calcul et tracé de la FFT (linéaire et log-dB).
- `back_end/analysis/spectrogram.py` : génération de spectrogrammes (SciPy si disponible, sinon solution de repli NumPy).
- `back_end/io/exports.py` : fonctions d'export (CSV, WAV, images) utilisées par le backend.
- `back_end/viz/plots.py` & `back_end/viz/anim.py` : génération de graphiques, snapshots et animations (GIF).
- `back_end/audio/generate_audio_from_positions.py` : conversion positions → fichiers audio `.wav`.
- `back_end/mesh/*` : génération et utilitaires pour la maille adaptée aux frettes.
- `back_end/utils/debug.py` : aides au débogage et sorties détaillées lorsque `DEBUG_ENABLED` est activé.

## 5. Flux d'exécution (résumé)

1. Charger la configuration (`back_end/config.py`).
2. Construire la maille et assembler M, K via `back_end/fem/formulation.py`.
3. (Optionnel) Calcul modal via `back_end/fem/modal.py`.
4. Initialiser les conditions initiales (ex. pizzicato/pression) et exécuter l'intégration temporelle (Newmark) via `back_end/fem/time_integration.py`.
5. Sauvegarder les déplacements et les instants de temps dans `back_end/results/plots/string_positions.csv` via `back_end/io/exports.py`.
6. Produire des analyses complémentaires : FFT (`back_end/analysis/fft.py`), spectrogramme (`back_end/analysis/spectrogram.py`), et visualisations (`back_end/viz/plots.py`, `back_end/viz/anim.py`).
7. Générer des fichiers audio si l'option est activée (`back_end/audio/generate_audio_from_positions.py`).

## 6. Emplacement des résultats

Par défaut, les sorties sont écrites dans `back_end/results/plots/`. Exemples de fichiers produits :

- `newmark_node_displacement.png`, `newmark_energies.png`
- `modes_first4.png`
- `newmark_output_fft.png`, `newmark_output_fft_logdb.png`
- `newmark_output_spectrogram.png` (si activé)
- `string_motion_slow.gif`, `string_motion_real.gif`, `string_snapshots.png`
- `string_positions.csv`, `string_debut.wav`, `string_fin.wav`, etc.

Ces sorties sont contrôlées par les flags `OUTPUT_ENABLE_IMAGES`, `OUTPUT_ENABLE_GIFS`, `OUTPUT_ENABLE_CSV` dans `back_end/config.py`.

## 7. Détails techniques importants

- Amortissement : calcul des coefficients de Rayleigh (α, β) à partir de cibles modales, puis construction de la matrice d'amortissement C = α M + β K.
- Conditions aux limites : politique appliquée — pour M et K, les nœuds d'extrémité sont encastrés (lignes/colonnes nulles, diag = 1) ; pour C, les lignes/colonnes d'extrémité sont mises à zéro.
- Intégration temporelle : Newmark-β (stable pour β=1/4, γ=1/2) appliqué aux degrés de liberté libres.
- Analyse spectrale : FFT (linéaire et dB) et représentation en fréquence logarithmique pour faciliter l'interprétation.
- Validation : utilitaires dans `back_end/utils/validators.py` vérifiant formats, symétrie et conditions aux limites.

## Formules et définitions

Cette section rassemble les formules mathématiques et notations utilisées dans les différentes étapes du projet. Les équations sont présentées de manière générale et correspondent aux implémentations typiques utilisées dans les modules `fem/`, `analysis/` et `fem/time_integration.py`.

### Assemblage des matrices M et K

Pour un élément linéaire 1D de longueur $L_e$, densité surfacique/massique $
ho A$ et tension/rigidité $T$ ou $E A$ selon le modèle, les matrices élémentaires usuelles sont (par exemple pour un élément de barre/corde) :

Consistent mass (2×2) :
$$
M_e = \frac{\rho A L_e}{6} \begin{bmatrix}2 & 1\\[4pt]1 & 2\end{bmatrix}
$$

Stiffness (2×2) pour une corde soumise à tension $T$ (ou une barre élastique avec $EA$) :
$$
K_e = \frac{T}{L_e} \begin{bmatrix}1 & -1\\[4pt]-1 & 1\end{bmatrix}
$$

L'assemblage global se fait par superposition des contributions élémentaires :
$$
M = \sum_{e} A_e^T M_e A_e, \qquad K = \sum_{e} A_e^T K_e A_e
$$
où $A_e$ est la matrice d'assemblage (connexion entre degrés de liberté élémentaires et globaux).

### Conditions aux limites (Dirichlet / encastrement)

Application pratique pour imposer un déplacement nul en un nœud $i$ : mettre les lignes et colonnes $i$ de $K$ et $M$ à zéro puis fixer la diagonale : $K_{ii}=1$ et $M_{ii}=1$ (ou utiliser l'élimination de DOF). Dans ce projet la politique adoptée est :

- Pour $M$ et $K$ (encastrement) : lignes/colonnes nulles, diagonale mise à $1$ pour préserver l'inversibilité.
- Pour $C$ (amortissement) : lignes/colonnes aux extrémités mises à zéro (pas de diag=1 artificiel).

### Analyse modale

Résolution du problème aux valeurs propres généralisé :
$$
K \Phi = M \Phi \Omega^2
$$
où les colonnes de $\Phi$ sont les vecteurs propres (modes) et $\Omega^2$ la matrice diagonale des $\omega_i^2$. Les fréquences propres sont $\omega_i = \sqrt{\lambda_i}$ si $\lambda_i$ sont les valeurs propres de $M^{-1}K$. Normalisation modale usuelle : $\Phi^T M \Phi = I$.

### Amortissement de Rayleigh

Modèle :
$$
C = \alpha M + \beta K
$$
Pour obtenir des rapports d'amortissement modaux ciblés $\zeta_p$ et $\zeta_q$ aux fréquences $\omega_p$ et $\omega_q$, on résout le système linéaire (pour $\alpha,\beta$) :
$$
\begin{bmatrix}
	frac{1}{2\omega_p} & \tfrac{\omega_p}{2} \\
	frac{1}{2\omega_q} & \tfrac{\omega_q}{2}
\end{bmatrix}
\begin{bmatrix}\alpha\\\beta\end{bmatrix}
=
\begin{bmatrix}\zeta_p\\\zeta_q\end{bmatrix}.
$$

### Schéma Newmark-β (intégration temporelle)

Paramètres usuels stables : $\beta=1/4$, $\gamma=1/2$. Les relations prédictives et correctives sont :

Prédicteur (valeurs préliminaires) :
$$
	ilde{u}_{n+1} = u_n + \Delta t\,v_n + \tfrac{\Delta t^2}{2}(1-2\beta) a_n
$$
$$
	ilde{v}_{n+1} = v_n + \Delta t(1-\gamma) a_n
$$

Équation d'équilibre implicite (résolution pour $a_{n+1}$) :
$$
\left(M + \gamma\Delta t\,C + \beta\Delta t^2\,K\right) a_{n+1} = F_{n+1} - C\left(v_n + (1-\gamma)\Delta t\,a_n\right) - K\left(u_n + \Delta t\,v_n + \tfrac{\Delta t^2}{2}(1-2\beta)a_n\right).
$$

Puis mise à jour :
$$
u_{n+1} = u_n + \Delta t\,v_n + \tfrac{\Delta t^2}{2}(a_n + a_{n+1})\quad\text{(pour }\beta=1/4\text{)}
$$
$$
v_{n+1} = v_n + \tfrac{\Delta t}{2}(a_n + a_{n+1})\quad\text{(pour }\gamma=1/2\text{)}
$$

Remarque : la forme générale des correcteurs dépend de $\beta,\gamma$; la version ci-dessus reprend la forme symétrique pour $\beta=1/4,\gamma=1/2$.

### Énergies

Énergie cinétique :
$$
E_k = \tfrac{1}{2} v^T M v
$$

Énergie potentielle (élastique) :
$$
E_p = \tfrac{1}{2} u^T K u
$$

Puissance dissipée instantanée par amortissement :
$$
P_d = v^T C v
$$

Énergie totale (instantanée) :
$$
E_{tot} = E_k + E_p
$$

### Conditions initiales : pluck (pizzicato)

Un pluck triangulaire centré en $x_p$ avec amplitude $A$ et largeur effective $w$ peut être exprimé :
$$
u(x) = \begin{cases}
A\left(1 - \dfrac{|x-x_p|}{w}\right), & |x-x_p| \le w,\\[6pt]
0, & \text{sinon.}
\end{cases}
$$
Les valeurs nodales $u_i$ sont obtenues en évaluant $u(x)$ aux positions nodales $x_i$. La vitesse initiale est souvent nulle ($v_0=0$).

### Transformée de Fourier discrète (FFT)

DFT d'un signal discret $x[n]$ de longueur $N$ :
$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-i 2\pi k n / N}, \qquad k=0,\dots,N-1.
$$
Fréquence correspondante : $f_k = k f_s / N$. Amplitude et conversion en décibels :
$$
|X[k]|, \qquad 20\log_{10}(|X[k]|)\;[\mathrm{dB}].
$$

### Spectrogramme (STFT)

Short-Time Fourier Transform (fenêtre $w[n]$, hop $R$) :
$$
\mathrm{STFT}(m,k) = \sum_{n} x[n] w[n-mR] e^{-i 2\pi k n / N}.
$$
Le spectrogramme est généralement $|\mathrm{STFT}(m,k)|^2$ ou 20·log10 de son module.

### Remarque sur l'implémentation numérique

- Toutes les opérations linéaires sont réalisées sur les degrés de liberté libres (après application des BCs) pour les solveurs implicites.
- Les normalisations (facteur $2/N$, fenêtrages) appliquées en FFT/STFT suivent l'option choisie dans `analysis/fft.py` et `analysis/spectrogram.py`.
