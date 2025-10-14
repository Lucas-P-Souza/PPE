# Backend FEM — Intégration temporelle (pas à pas)

Ce document décrit, étape par étape, comment le backend (FEM corde vibrante) fonctionne à partir du moment où les matrices globales sont formées (après `formulation.py`). Pour chaque étape, on indique:
- le fichier et la fonction qui interviennent,
- les équations appliquées,
- et les points d'attention pratiques.

> Portée: Intégration temporelle (Newmark-beta), CI triangulaires, premier pas (diagnostic), détection des DDLs contraints, énergies. La construction M,K,C (assemblage FEM + Rayleigh) est couverte dans `fem/formulation.py`.

---

## 0) Pré-requis — Matrices globales (déjà prêtes)
- Source (amont): `digital_twin/back_end/fem/formulation.py`
  - `assemble_mkc(...)` ou `build_global_mkc_from_config(...)` renvoient M (masse), K (raideur), C (amortissement Rayleigh = αM + βK), avec conditions aux limites (CL) aux extrémités.
  - Convention CL: lignes/colonnes ≈ 0 hors diagonale et diag ≈ 1 sur nœuds contraints (Dirichlet).

---

## 1) Paramètres de simulation
- Fichier: `digital_twin/back_end/fem/time_integration.py`
- Fonction: `definir_parametres_simulation(delta_t, T_total)`
- Rôle: définir la maille temporelle et pré-calculer facteurs utiles.
- Équations:
  - Nombre de pas: $n_{pas} = \operatorname{round}(T_{total}/\Delta t)$
  - Facteurs: $\text{inv\_dt\_carre} = 1/\Delta t^2$, $\text{inv\_2dt} = 1/(2\Delta t)$

---

## 2) Conditions initiales (pincement triangulaire)
- Fichier: `digital_twin/back_end/fem/time_integration.py`
- Fonctions:
  - `initialiser_u0_triangle(M, L, h, x_p)`
  - `initialiser_etats_initiaux(M, L, h, x_p) -> (U0, U_nm1, U_n)`
- Rôle: générer le déplacement initial U0 d'une corde pincée à la position $x_p$ (amplitude $h$), puis fournir $(U0, U_{n-1}, U_n)$.
- Discrétisation spatiale implicite pour CI: $x_i = i\,\Delta x$, avec $\Delta x = L/(N-1)$ et $N = M.shape[0]$.
- Forme triangulaire:
  $$
  U_0(x) = \begin{cases}
    h\,\dfrac{x}{x_p}, & x \le x_p,\\[6pt]
    h\,\dfrac{L - x}{L - x_p}, & x > x_p.
  \end{cases}
  $$
- Sorties: $U0$, $U_{n-1} = U0$, $U_n = U0$.

---

## 3) Premier pas (diagnostic) — différences centrées
- Fichier: `digital_twin/back_end/fem/time_integration.py`
- Fonction: `calculer_u1(M, C, K, U_n, U_nm1, delta_t) -> U1`
- Rôle: calculer $U_1$ via un schéma de 2 pas (utile pour vérifier CI/échelles).
- Équations:
  - $A = M/\Delta t^2 + C/(2\Delta t) + K$
  - $\text{rhs} = (2M/\Delta t^2)\,U_n - (M/\Delta t^2 - C/(2\Delta t))\,U_{n-1}$
  - $U_1 = A^{-1}\,\text{rhs}$ (résolu par `numpy.linalg.solve`).

---

## 4) Détection des DDLs contraints et sous-système libre
- Fichier: `digital_twin/back_end/fem/modal.py`
- Fonction: `detect_constrained_dofs_mk(M, K, atol)`
- Rôle: repérer les nœuds imposés (Dirichlet) selon la convention de CL.
- Critère: lignes/colonnes ≈ 0 hors diag sur M/K et diag ≈ 1 aux extrémités.
- Indices:
  - Contraints: $\Gamma$
  - Libres: $F = \{0,\dots,N-1\} \setminus \Gamma$
- Sous-matrices libres:
  - $M_{ff} = M[F,F]$, $C_{ff} = C[F,F]$, $K_{ff} = K[F,F]$.

---

## 5) Intégration Newmark-beta (β=1/4, γ=1/2)
- Fichier: `digital_twin/back_end/fem/time_integration.py`
- Fonction: `integrer_newmark_beta(M, C, K, F, dt, t_max, U0, V0, A0) -> (t, U, V, A)`
- Rôle: résoudre $M u¨ + C u˙ + K u = F(t)$ sur les DDLs libres avec Newmark classique.
- Maille temporelle: $t_k = k\,\Delta t$, $k = 0,\dots,n_{steps}-1$.
- Si $A_0$ absent: $A_{0,f} = M_{ff}^{-1}\,(F_0^f - C_{ff}V_{0,f} - K_{ff}U_{0,f})$.
- Coefficients (β=1/4, γ=1/2):
  $$
  a_0 = \frac{1}{\beta\,\Delta t^2},\qquad a_1 = \frac{\gamma}{\beta\,\Delta t},\qquad a_2 = \frac{1}{\beta\,\Delta t},\\
  a_3 = \frac{1}{2\beta} - 1,\qquad a_4 = \frac{\gamma}{\beta} - 1,\qquad a_5 = \Delta t\left(\frac{\gamma}{2\beta} - 1\right).
  $$
- Matrice efficace (constante): $K_{eff} = K_{ff} + a_1 C_{ff} + a_0 M_{ff}$.
- Pas $k\to k+1$:
  - Second membre (RHS):
    $$
    \mathrm{RHS} = F_{k+1}^f + M_{ff}(a_0 U_k^f + a_2 V_k^f + a_3 A_k^f) + C_{ff}(a_1 U_k^f + a_4 V_k^f + a_5 A_k^f)
    $$
  - Déplacement: résoudre $K_{eff}\,U_{k+1}^f = \mathrm{RHS}$.
  - Accélération:
    $$
    A_{k+1}^f = a_0\,(U_{k+1}^f - U_k^f) - a_2 V_k^f - a_3 A_k^f
    $$
  - Vitesse:
    $$
    V_{k+1}^f = V_k^f + \Delta t\,\big[(1-\gamma)A_k^f + \gamma\,A_{k+1}^f\big].
    $$
- Remontage: insérer U,V,A sur indices libres dans l'espace complet (contraints = 0).

---

## 6) Énergies au cours du temps
- Fichier: `digital_twin/back_end/fem/time_integration.py`
- Fonction: `calculer_energies_dans_le_temps(M, K, U, V) -> (Ek, Ep, Etot)`
- Formules (pour chaque pas k):
  - Énergie cinétique: $E_k = \tfrac{1}{2}\, V_k^T (M V_k)$
  - Énergie potentielle: $E_p = \tfrac{1}{2}\, U_k^T (K U_k)$
  - Énergie totale: $E_{tot} = E_k + E_p$
- Avec amortissement (C > 0), l'énergie totale décroit au cours du temps.

---

## 7) Analyse modale (support / validation)
- Fichier: `digital_twin/back_end/fem/modal.py`
- Fonction: `compute_modal_frequencies_and_modes(M, K, num_modes)`
- Problème généralisé: $K v = \lambda M v$ (sur DDLs libres)
  - Construction: $A = M_{ff}^{-1}K_{ff}$ (`np.linalg.solve` avec $M_{ff}$)
  - Valeurs propres: $\lambda = \omega^2 \ge 0$; tri croissant
  - Fréquences: $f = \omega/(2\pi)$ (Hz)
  - Normalisation: $v^T M_{ff} v = 1$, puis extension au plein espace (contraints=0)

---

## 8) Où tout est assemblé (exemple d'usage)
- Fichier: `digital_twin/back_end/main.py`
- Chaînage:
  1) Construire M,K,C via `build_global_mkc_from_config(apply_fixed_bc=True)`
  2) Valider M,C,K (`utils.validators.valider_mck`)
  3) Définir $(\Delta t, T_{total})` et appeler `definir_parametres_simulation`
  4) CI: `initialiser_etats_initiaux`
  5) Diagnostics: `calculer_u1` (optionnel)
  6) Intégrer: `integrer_newmark_beta`
  7) Énergies, FFT, spectrogramme, plots, animation (modules `analysis`/`viz`/`io`)

---

## 9) Notes pratiques
- GLs contraints: convention de CL (diag≈1, hors diag≈0) est essentielle pour que la détection fonctionne.
- Force F(t): peut être une fonction Python F(t,k)→(N,) ou un tableau (N, n_steps); manquants ⇒ zéros.
- Stabilité: Newmark avec (β=1/4, γ=1/2) est inconditionnellement stable pour systèmes linéaires.
- U0 triangulaire: amplitude/position contrôlées par (h, x_p). On peut plugguer d'autres CI.

---

## 10) Références rapides (API FR uniquement)
- Paramètres: `definir_parametres_simulation`
- CI triangulaire: `initialiser_u0_triangle`
- CI tuple: `initialiser_etats_initiaux`
- Premier pas: `calculer_u1`
- Force nulle: `fournisseur_force_nulle`
- Newmark: `integrer_newmark_beta`
- Énergies: `calculer_energies_dans_le_temps`

---

Fichier généré automatiquement pour documentation interne du backend.
