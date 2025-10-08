# Validateur basique pour M, C, K
# -------------------------------
# Ce module (NumPy/SciPy) fournit des utilitaires pour :
#  - vérifier que M, C et K ont la même taille carrée (n x n) ;
#  - contrôler que des conditions de bord fixes (Dirichlet) ont été appliquées
#    aux extrémités (lignes/colonnes 0 et n-1 : zéros hors diagonale et diag≈1) ;
#  - afficher la taille des matrices et le nombre de degrés de liberté (n).
#
# Usage : appelez les fonctions avec M, C, K déjà assemblées. Si vous exécutez
# ce fichier directement, il tente d'assembler M, K, C à partir de config/formulation
# et lance les vérifications.

from __future__ import annotations

import numpy as np
try:
	import scipy as sp  # importação solicitada
except Exception:  # SciPy peut ne pas être disponible dans cet environnement
	sp = None  # type: ignore
from typing import Tuple

# Débogage (centralisé)
try:
	from ..utils import debug as dbg  # type: ignore
except Exception:
	try:
		from digital_twin.back_end.utils import debug as dbg  # type: ignore
	except Exception:
		class _DbgNoOp:
			@staticmethod
			def is_enabled() -> bool: return False
			@staticmethod
			def dprint(*args, **kwargs): pass
		dbg = _DbgNoOp()  # type: ignore


def _is_square_same_shape(M: np.ndarray, C: np.ndarray, K: np.ndarray) -> Tuple[bool, str, int]:
	# Vérifie si M, C et K ont la même taille carrée. Retourne (ok, msg, n).
	for name, A in (('M', M), ('C', C), ('K', K)):
		if not isinstance(A, np.ndarray):
			return False, f"{name} n'est pas un numpy.ndarray", 0
		if A.ndim != 2:
			return False, f"{name} n'est pas 2D (ndim={A.ndim})", 0
		if A.shape[0] != A.shape[1]:
			return False, f"{name} n'est pas carrée (shape={A.shape})", 0
	if not (M.shape == C.shape == K.shape):
		return False, f"M, C, K ont des shapes différents: M={M.shape}, C={C.shape}, K={K.shape}", 0
	return True, "OK", M.shape[0]


def _bc_applied_at_ends(A: np.ndarray, atol: float = 1e-12, *, allow_diag_zero: bool = False) -> Tuple[bool, str]:
	# Vérifie si des CL fixes semblent appliquées aux extrémités de A.
	# Critère (convention) : lignes/colonnes 0 et n-1 nulles (hors diagonale ~ 0).
	# Diagonale : ≈1.0 pour M/K ; 0.0 (ou 1.0) accepté pour C si allow_diag_zero=True.
	n = A.shape[0]
	for i in (0, n - 1):
		row = A[i, :].copy(); row[i] = 0.0
		col = A[:, i].copy(); col[i] = 0.0
		if not (np.all(np.abs(row) <= atol) and np.all(np.abs(col) <= atol)):
			return False, f"Extrémité {i} : lignes/colonnes non nulles hors diagonale (max hors diag={max(np.max(np.abs(row)), np.max(np.abs(col))):.2e})"
		if allow_diag_zero:
			# accepte 0.0 ou 1.0 (robuste), mais n'impose pas 1.0
			if not (np.isclose(A[i, i], 0.0, atol=1e-9) or np.isclose(A[i, i], 1.0, atol=1e-9)):
				return False, f"Extrémité {i} : diagonale ni 0 ni 1 (A[{i},{i}]={A[i,i]:.3e})"
		else:
			if not np.isclose(A[i, i], 1.0, atol=1e-9):
				return False, f"Extrémité {i} : diagonale non ≈ 1.0 (A[{i},{i}]={A[i,i]:.3e})"
	return True, "OK"


def _is_symmetric(A: np.ndarray, rtol: float = 1e-8, atol: float = 1e-10) -> bool:
	return np.allclose(A, A.T, rtol=rtol, atol=atol)


def validar_mck(M: np.ndarray, C: np.ndarray, K: np.ndarray, *, verbose: bool = True) -> None:
	# Exécute les validations et imprime un résumé (taille, DDL, symétrie, CL).
	ok, msg, n = _is_square_same_shape(M, C, K)
	if not ok:
		raise ValueError(f"[ERREUR] Échec de la vérification des dimensions : {msg}")

	if verbose and dbg.is_enabled():
		dbg.dprint(f"Taille des matrices: {M.shape} (carrées)")
		dbg.dprint(f"Degrés de liberté (n): {n}")

	# Symétrie (informatif)
	if verbose and dbg.is_enabled():
		dbg.dprint(f"Symétrie: M={_is_symmetric(M)}, C={_is_symmetric(C)}, K={_is_symmetric(K)}")

	# CL aux extrémités
	for name, A in (("M", M), ("C", C), ("K", K)):
		# Pour C on accepte diagonale 0 aux extrémités (traitement spécifique de l'amortissement)
		allow_zero = True if name == "C" else False
		ok_bc, msg_bc = _bc_applied_at_ends(A, allow_diag_zero=allow_zero)
		if verbose and dbg.is_enabled():
			status = "OK" if ok_bc else "NOK"
			dbg.dprint(f"CL aux extrémités dans {name}: {status} — {msg_bc}")
		if not ok_bc:
			raise ValueError(f"[ERREUR] Les conditions aux limites ne semblent pas correctement appliquées dans {name}: {msg_bc}")


def definir_parametros_simulacao(delta_t: float, T_total: float):
	# Définit et retourne les paramètres de simulation et affiche num_steps.
	# Retourne (delta_t, T_total, num_steps, inv_dt2, inv_2dt).
	if delta_t <= 0.0:
		raise ValueError("delta_t doit être > 0")
	if T_total <= 0.0:
		raise ValueError("T_total doit être > 0")
	num_steps = int(round(T_total / delta_t))
	inv_dt2 = 1.0 / (delta_t * delta_t)
	inv_2dt = 1.0 / (2.0 * delta_t)
	if dbg.is_enabled():
		dbg.dprint(f"num_steps = {num_steps}")
	return delta_t, T_total, num_steps, inv_dt2, inv_2dt


def inicializar_u0_triangulo(M: np.ndarray, *, L: float, h: float, x_p: float) -> np.ndarray:
	# Initialise le vecteur U0 avec une forme triangulaire (pincement).
	# Paramètres : M (pour N), L (m), h (m), x_p (m). Retourne U0 de shape (N,).
	N = int(M.shape[0])
	if N < 2:
		raise ValueError("N doit être >= 2 pour initialiser U0")
	# Caso sem excitação (h <= 0) → vetor nulo
	if h is None or h <= 0.0:
		return np.zeros(N, dtype=float)

	L = float(L)
	x_p = float(x_p)
	delta_x = L / float(N - 1)
	x = np.arange(N, dtype=float) * delta_x

	# Évite division par zéro si x_p == 0 ou x_p == L
	left = h * (x / x_p) if x_p > 0.0 else np.zeros_like(x)
	denom_right = (L - x_p) if (L - x_p) > 0.0 else 1.0
	right = h * ((L - x) / denom_right)
	U0 = np.where(x <= x_p, left, right)
	return U0.astype(float, copy=False)


def inicializar_estados_iniciais(M: np.ndarray, *, L: float, h: float, x_p: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	# Commodité : retourne (U0, U_nm1, U_n) avec U_nm1 = U0 et U_n = U0.
	U0 = inicializar_u0_triangulo(M, L=L, h=h, x_p=x_p)
	U_nm1 = U0.copy()
	U_n = U0.copy()
	return U0, U_nm1, U_n


def calcular_u1(M: np.ndarray, C: np.ndarray, K: np.ndarray,
				U_n: np.ndarray, U_nm1: np.ndarray, delta_t: float) -> np.ndarray:
	# Calcule le premier vecteur U1 en utilisant des différences centrées.
	# M u¨ + C u˙ + K u = 0, discrétisé avec approximations centrées.
	# (M/Δt² + C/(2Δt) + K) U1 = (2M/Δt²) U0 - (M/Δt² - C/(2Δt)) U_-1
	# Vérifications basiques de dimensions
	if M.ndim != 2 or C.ndim != 2 or K.ndim != 2:
		raise ValueError("M, C, K doivent être 2D (matrices)")
	if M.shape[0] != M.shape[1] or C.shape[0] != C.shape[1] or K.shape[0] != K.shape[1]:
		raise ValueError("M, C, K doivent être carrées")
	if not (M.shape == C.shape == K.shape):
		raise ValueError(f"Dimensions différentes: M={M.shape}, C={C.shape}, K={K.shape}")
	n = M.shape[0]
	if U_n.shape[0] != n or U_nm1.shape[0] != n:
		raise ValueError(f"U_n/U_nm1 non compatibles avec n={n}")
	if delta_t <= 0.0:
		raise ValueError("delta_t doit être > 0")

	dt = float(delta_t)
	inv_dt2 = 1.0 / (dt * dt)
	inv_2dt = 1.0 / (2.0 * dt)

	# Montagem dos termos
	A = (M * inv_dt2) + (C * inv_2dt) + K
	rhs = (2.0 * M * inv_dt2) @ U_n - ((M * inv_dt2) - (C * inv_2dt)) @ U_nm1

	# Resolução do sistema linear
	U1 = np.linalg.solve(A, rhs)

	# Relato
	if dbg.is_enabled():
		dbg.dprint(f"U1: shape={U1.shape}, max={np.nanmax(U1):.3e}, min={np.nanmin(U1):.3e}")
	return U1


# -------------------------------
#    Utilitaires d'analyse modale
# -------------------------------
def _detect_constrained_dofs_mk(M: np.ndarray, K: np.ndarray, atol: float = 1e-12) -> np.ndarray:
	# Détecte des DDL contraints par CL de Dirichlet selon la convention du projet :
	# - lignes/colonnes nulles hors diagonale ; diag=1.0 sur les DDL contraints.
	# Retourne un tableau d'indices contraints.
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
	# Résout K v = λ M v sur les DDL libres :
	# - freqs_hz : les num_modes premières en Hz (croissant)
	# - modes_full : formes modales (n, num_modes) normalisées en masse (v^T M v = 1),
	#   avec DDL contraints remplis par 0.
	if M.shape != K.shape or M.ndim != 2 or M.shape[0] != M.shape[1]:
		raise ValueError("M et K doivent être carrées et de même taille")
	n = M.shape[0]
	constrained = _detect_constrained_dofs_mk(M, K)
	free = np.setdiff1d(np.arange(n), constrained)
	if free.size == 0:
		raise ValueError("Aucun DDL libre détecté pour l'analyse modale")

	M_ff = M[np.ix_(free, free)]
	K_ff = K[np.ix_(free, free)]

	# Résout M_ff A = K_ff -> A = M_ff^{-1} K_ff, puis eig(A) = lambdas ~ omega^2
	A = np.linalg.solve(M_ff, K_ff)
	eigvals, eigvecs = np.linalg.eig(A)
	eigvals = np.real(eigvals)
	eigvals[eigvals < 0] = 0.0
	omegas = np.sqrt(eigvals)
	# tri
	idx = np.argsort(omegas)
	omegas = omegas[idx]
	V = np.real(eigvecs[:, idx])

	# prendre les premiers num_modes
	k = int(min(num_modes, V.shape[1]))
	omegas_k = omegas[:k]
	V_k = V[:, :k]

	# Normaliser en masse sur les DDL libres, puis étendre au plein avec zéros aux contraints
	modes_full = np.zeros((n, k), dtype=float)
	for j in range(k):
		v = V_k[:, j]
		norm = float(np.sqrt(v.T @ (M_ff @ v)))
		if norm <= 0:
			vj = v.copy()
		else:
			vj = v / norm
		full = np.zeros(n, dtype=float)
		full[free] = vj
		modes_full[:, j] = full

	freqs_hz = omegas_k / (2.0 * np.pi)
	return freqs_hz, modes_full


def build_node_positions_from_config(n_nodes: int):
	# Construit des coordonnées x à partir du config :
	# - Préférer des dx non uniformes de FRET_DXS_MM (m)
	# - Sinon, uniformes basées sur L
	try:
		from digital_twin.back_end import config as _cfg  # type: ignore
	except Exception:
		# repli : uniforme [0, 1]
		return np.linspace(0.0, 1.0, int(n_nodes))
	try:
		dxs_mm = getattr(_cfg, "FRET_DXS_MM", None)
		if dxs_mm and len(dxs_mm) == n_nodes - 1:
			dxs_m = np.asarray(dxs_mm, dtype=float) / 1000.0
			x = np.concatenate([[0.0], np.cumsum(dxs_m)])
			return x
	except Exception:
		pass
	# repli vers uniforme basé sur la longueur L si fournie
	L = float(getattr(_cfg, "L", 1.0))
	return np.linspace(0.0, L, int(n_nodes))


def inicializar_u0_senoidal(M: np.ndarray, *, L: float, amplitude: float = 1.0) -> np.ndarray:
	# Initialise U0 comme A * sin(pi * x / L) en utilisant les positions de nœuds du config.
	# L : longueur totale (m), amplitude : A (m)
	n = M.shape[0]
	x = build_node_positions_from_config(n)
	if L <= 0:
		L = float(x[-1]) if x[-1] > 0 else 1.0
	U0 = amplitude * np.sin(np.pi * x / L)
	# Imposer les CL : les nœuds extrêmes doivent être 0 (c'est déjà le cas pour la sinusoïde, mais on force)
	U0[0] = 0.0
	U0[-1] = 0.0
	return U0.astype(float, copy=False)


def zero_force_provider(n: int):
	# Retourne F(t,k) qui fournit une force nulle (vecteur de zéros de taille n).
	def F_zero(_t: float, _k: int) -> np.ndarray:
		return np.zeros(n, dtype=float)
	return F_zero



def plot_first_modes(x: np.ndarray, modes_full: np.ndarray, freqs_hz: np.ndarray, *, max_modes: int = 4, savepath: str | None = None) -> None:
	# Trace jusqu'à max_modes les formes modales vs x et enregistre éventuellement.
	try:
		import matplotlib.pyplot as plt  # type: ignore
	except Exception as e:
		print("[AVERTISSEMENT] matplotlib indisponible pour tracer les modes:", e)
		return

	m = int(min(max_modes, modes_full.shape[1]))
	if m == 0:
		print("[INFO] Aucun mode à tracer.")
		return
	plt.figure(figsize=(8, 6))
	for j in range(m):
		plt.plot(x, modes_full[:, j], label=f"Mode {j+1} — {freqs_hz[j]:.2f} Hz")
	plt.title("Premiers modes — formes et fréquences")
	plt.xlabel("x (m)")
	plt.ylabel("Forme modale (normalisée en masse)")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	if savepath:
		from pathlib import Path as _P
		out = _P(savepath)
		out.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(out, dpi=150)
		print(f"[INFO] Modes enregistrés dans : {out}")
	else:
		plt.show()
	plt.close()


# -------------------------------
#    Newmark-beta (β=1/4, γ=1/2)
# -------------------------------
def newmark_beta(
	M: np.ndarray,
	C: np.ndarray,
	K: np.ndarray,
	F: np.ndarray | callable,
	dt: float,
	t_max: float,
	U0: np.ndarray | None = None,
	V0: np.ndarray | None = None,
	A0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	# Résout M u¨ + C u˙ + K u = F(t) avec Newmark-beta (β=1/4, γ=1/2) sur les DDL libres.
	# Détecte les DDL contraints via M/K et intègre sur l'ensemble libre ; réélargit à la taille totale avec des zéros.
	if dt <= 0 or t_max <= 0:
		raise ValueError("dt et t_max doivent être > 0")
	if M.shape != C.shape or M.shape != K.shape or M.ndim != 2:
		raise ValueError("M, C, K doivent être carrées de même taille")

	n = M.shape[0]
	n_steps = int(np.floor(t_max / dt)) + 1
	t = np.linspace(0.0, dt * (n_steps - 1), n_steps)

	# Detect constrained DOFs and build free set
	constrained = _detect_constrained_dofs_mk(M, K)
	free = np.setdiff1d(np.arange(n), constrained)
	if free.size == 0:
		raise ValueError("Aucun DDL libre détecté pour l'intégration Newmark")

	# Slice system to free DOFs
	Mff = M[np.ix_(free, free)]
	Cff = C[np.ix_(free, free)]
	Kff = K[np.ix_(free, free)]

	# Initial conditions on free DOFs
	U0f = np.zeros(free.size, dtype=float) if U0 is None else np.asarray(U0, dtype=float)[free]
	V0f = np.zeros(free.size, dtype=float) if V0 is None else np.asarray(V0, dtype=float)[free]
	if A0 is None:
		# Compute A0 from equilibrium at t=0
		if callable(F):
			F0_full = np.asarray(F(0.0, 0), dtype=float).reshape(n)
		else:
			F_arr = np.asarray(F, dtype=float)
			F0_full = F_arr[:, 0] if (F_arr.ndim == 2 and F_arr.shape[1] >= 1) else np.zeros(n)
		A0f = np.linalg.solve(Mff, F0_full[free] - Cff @ V0f - Kff @ U0f)
	else:
		A0f = np.asarray(A0, dtype=float)[free]

	# Newmark coefficients (average acceleration)
	beta = 1.0 / 4.0
	gamma = 1.0 / 2.0
	a0 = 1.0 / (beta * dt * dt)
	a1 = gamma / (beta * dt)
	a2 = 1.0 / (beta * dt)
	a3 = 1.0 / (2.0 * beta) - 1.0
	a4 = gamma / beta - 1.0
	a5 = dt * (gamma / (2.0 * beta) - 1.0)

	# Effective stiffness (constant)
	K_eff = Kff + a1 * Cff + a0 * Mff
	# Prefer solve over explicit inverse for stability

	# Allocate result arrays (full size)
	U = np.zeros((n, n_steps), dtype=float)
	V = np.zeros((n, n_steps), dtype=float)
	A = np.zeros((n, n_steps), dtype=float)
	# Initialize at k=0
	U[free, 0] = U0f
	V[free, 0] = V0f
	A[free, 0] = A0f
	if constrained.size:
		U[constrained, 0] = 0.0
		V[constrained, 0] = 0.0
		A[constrained, 0] = 0.0

	# Time stepping on free DOFs
	for k in range(n_steps - 1):
		Uf_k = U[free, k]
		Vf_k = V[free, k]
		Af_k = A[free, k]

		# Load at t_{k+1} sliced to free DOFs
		if callable(F):
			Fk1_full = np.asarray(F(t[k + 1], k + 1), dtype=float).reshape(n)
			Fk1 = Fk1_full[free]
		else:
			F_arr = np.asarray(F, dtype=float)
			if F_arr.ndim == 2 and F_arr.shape[1] > k + 1:
				Fk1 = F_arr[free, k + 1]
			else:
				Fk1 = np.zeros(free.size, dtype=float)

		# Canonical RHS using state at k
		RHS = (
			Fk1
			+ Mff @ (a0 * Uf_k + a2 * Vf_k + a3 * Af_k)
			+ Cff @ (a1 * Uf_k + a4 * Vf_k + a5 * Af_k)
		)

		# Solve for U_{k+1} (free)
		Uf_k1 = np.linalg.solve(K_eff, RHS)
		# Update accelerations and velocities
		Af_k1 = a0 * (Uf_k1 - Uf_k) - a2 * Vf_k - a3 * Af_k
		Vf_k1 = Vf_k + dt * ((1.0 - gamma) * Af_k + gamma * Af_k1)

		# Write back to full arrays, enforce constraints as zero
		U[free, k + 1] = Uf_k1
		V[free, k + 1] = Vf_k1
		A[free, k + 1] = Af_k1
		if constrained.size:
			U[constrained, k + 1] = 0.0
			V[constrained, k + 1] = 0.0
			A[constrained, k + 1] = 0.0

	return t, U, V, A


# -------------------------------
#    Energy computation helpers
# -------------------------------
def compute_energies_over_time(M: np.ndarray, K: np.ndarray, U: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	# Calcule l'énergie cinétique, potentielle et totale au cours du temps.
	# Entrées : M,K (n,n) ; U,V (n, n_steps). Sorties : Ek, Ep, Etot (n_steps,).
	n_steps = U.shape[1]
	Ek = np.zeros(n_steps, dtype=float)
	Ep = np.zeros(n_steps, dtype=float)
	for k in range(n_steps):
		v = V[:, k]
		u = U[:, k]
		Ek[k] = 0.5 * float(v.T @ (M @ v))
		Ep[k] = 0.5 * float(u.T @ (K @ u))
	Etot = Ek + Ep
	return Ek, Ep, Etot


# -------------------------------
#    Export CSV des positions
# -------------------------------
def save_displacement_csv(t: np.ndarray, U: np.ndarray, savepath: str) -> None:
	# Sauvegarde un CSV avec le temps et les positions de la corde (une colonne par nœud).
	# t : (n_steps,), U : (n_nodes, n_steps) → CSV (n_steps x (1+n_nodes)) avec en-têtes.
	try:
		from pathlib import Path as _P
		out = _P(savepath)
		out.parent.mkdir(parents=True, exist_ok=True)
		n_nodes, n_steps = int(U.shape[0]), int(U.shape[1])
		if t.shape[0] != n_steps:
			raise ValueError("Dimensions incompatibles: t et U doivent avoir le même nombre d'instants")
		data = np.empty((n_steps, 1 + n_nodes), dtype=float)
		data[:, 0] = t
		data[:, 1:] = U.T
		header = "t," + ",".join(f"u_{i}" for i in range(n_nodes))
		# Utiliser comments="" pour éviter le # au début de la ligne d'en-tête
		np.savetxt(out, data, delimiter=",", header=header, comments="", fmt="%.10e")
		print(f"[INFO] Positions de la corde sauvegardées en CSV : {out}")
	except Exception as _ecsv:
		print("[AVERTISSEMENT] Échec de la sauvegarde CSV des positions:", _ecsv)


# -------------------------------
#    Animation helper (FuncAnimation)
# -------------------------------
def animate_string_motion(
	x: np.ndarray,
	U: np.ndarray,
	*,
	interval_ms: int = 30,
	decim: int = 1,
	savepath: str | None = None,
	show: bool = False,
	y_scale: float = 1.0,
	y_pad_frac: float = 0.05,
	y_limits: tuple[float, float] | None = None,
):
	# Crée une animation du déplacement dans le temps (Matplotlib FuncAnimation).
	# x: (n,), U: (n, n_steps), interval_ms: délai entre images, decim: décimation, savepath: GIF/MP4.
	try:
		import matplotlib.pyplot as plt  # type: ignore
		from matplotlib.animation import FuncAnimation, PillowWriter  # type: ignore
	except Exception as e:
		print("[AVERTISSEMENT] Matplotlib/animation indisponible:", e)
		return

	U_anim = U[:, ::max(1, int(decim))]
	n, n_frames = U_anim.shape
	# imposer des extrémités fixes (robuste pour la visualisation)
	U_anim[0, :] = 0.0
	U_anim[-1, :] = 0.0

	fig, ax = plt.subplots(figsize=(8, 4))
	line, = ax.plot(x, U_anim[:, 0], lw=1.5)
	ax.set_title("Mouvement de la corde")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("déplacement (m)")
	ax.grid(True, alpha=0.3)

	# fixer les limites y : explicites, ou robustes sur percentiles + échelle/marge
	if y_limits is not None and len(y_limits) == 2:
		ax.set_ylim(float(y_limits[0]), float(y_limits[1]))
	else:
		lo = float(np.nanpercentile(U_anim, 1))
		hi = float(np.nanpercentile(U_anim, 99))
		rng = max(1e-12, (hi - lo))
		c = 0.5 * (hi + lo)
		# y_scale > 1.0 élargit la plage (zoom out) ; < 1.0 réduit (zoom in)
		factor = max(1e-6, float(y_scale))
		half = 0.5 * rng * factor
		pad = float(y_pad_frac) * rng
		ax.set_ylim(c - (half + pad), c + (half + pad))

	def init():
		line.set_ydata(U_anim[:, 0])
		return (line,)

	def update(frame: int):
		line.set_ydata(U_anim[:, frame])
		return (line,)

	ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=interval_ms, blit=True)

	if savepath:
		from pathlib import Path as _P
		out = _P(savepath)
		out.parent.mkdir(parents=True, exist_ok=True)
		# par défaut GIF via PillowWriter si extension .gif
		if out.suffix.lower() == ".gif":
			try:
				ani.save(out, writer=PillowWriter(fps=max(1, int(1000/interval_ms))))
				print(f"[INFO] Animation sauvegardée dans : {out}")
			except Exception as _esave:
				print("[AVERTISSEMENT] Échec lors de l'enregistrement du GIF:", _esave)
		else:
			try:
				ani.save(out)
				print(f"[INFO] Animation sauvegardée dans : {out}")
			except Exception as _esave2:
				print("[AVERTISSEMENT] Échec lors de l'enregistrement de l'animation:", _esave2)

	if show:
		plt.show()
	else:
		plt.close(fig)


def save_string_frame_png(
	x: np.ndarray,
	U: np.ndarray,
	frame_idx: int,
	savepath: str,
	*,
	y_scale: float | None = None,
	y_pad_frac: float = 0.05,
) -> None:
	# Enregistre une image unique (profil) du déplacement en PNG.
	# x: (n,), U: (n,n_steps), frame_idx: index de colonne, savepath: chemin PNG.
	try:
		import matplotlib.pyplot as plt  # type: ignore
	except Exception as e:
		print("[AVERTISSEMENT] Matplotlib indisponible pour enregistrer l'image:", e)
		return
	from pathlib import Path as _P
	out = _P(savepath)
	out.parent.mkdir(parents=True, exist_ok=True)
	y = np.array(U[:, frame_idx], dtype=float)
	# imposer visuellement les extrémités fixes
	y[0] = 0.0
	y[-1] = 0.0
	fig, ax = plt.subplots(figsize=(7, 3.5))
	ax.plot(x, y, lw=1.6)
	ax.set_title(f"Profil de la corde — frame {frame_idx}")
	ax.set_xlabel("x (m)")
	ax.set_ylabel("déplacement (m)")
	ax.grid(True, alpha=0.3)
	# limites y robustes avec échelle optionnelle
	lo = float(np.nanpercentile(y, 1))
	hi = float(np.nanpercentile(y, 99))
	rng = max(1e-12, (hi - lo))
	if y_scale is None:
		pad = float(y_pad_frac) * rng
		ax.set_ylim(lo - pad, hi + pad)
	else:
		c = 0.5 * (hi + lo)
		factor = max(1e-6, float(y_scale))
		half = 0.5 * rng * factor
		pad = float(y_pad_frac) * rng
		ax.set_ylim(c - (half + pad), c + (half + pad))
	fig.tight_layout()
	fig.savefig(out, dpi=150)
	plt.close(fig)
	print(f"[INFO] Image enregistrée dans : {out}")


def plot_snapshots_png(
	x: np.ndarray,
	U: np.ndarray,
	t: np.ndarray,
	*,
	n_snapshots: int = 8,
	decim: int = 1,
	savepath: str,
	title: str = "Profils de la corde à des temps sélectionnés",
	y_scale: float | None = None,
	y_pad_frac: float = 0.05,
	t_window: tuple[float, float] | None = None,
	use_colorbar: bool = True,
	cmap: str = "viridis",
	alpha: float = 0.9,
	linewidth: float = 1.2,
	show_legend: bool = False,
):
	# Enregistre un PNG statique avec plusieurs profils à des temps sélectionnés.
	# x: (n,), U: (n,n_steps), t: (n_steps,), n_snapshots: échantillons sur t, decim: décimation.
	try:
		import matplotlib.pyplot as plt  # type: ignore
	except Exception as e:
		print("[AVERTISSEMENT] Matplotlib indisponible pour les instantanés:", e)
		return
	from pathlib import Path as _P
	out = _P(savepath)
	out.parent.mkdir(parents=True, exist_ok=True)

	# Décimer pour l'efficacité si nécessaire
	step = max(1, int(decim))
	U_dec = U[:, ::step]
	t_dec = t[::step]
	n_steps = U_dec.shape[1]
	if n_steps == 0:
		print("[AVERTISSEMENT] Aucun frame pour les instantanés.")
		return

	# Appliquer une fenêtre temporelle si fournie
	if t_window is not None and len(t_window) == 2:
		t0, t1 = float(min(t_window)), float(max(t_window))
		mask = (t_dec >= t0) & (t_dec <= t1)
		if np.any(mask):
			U_dec = U_dec[:, mask]
			t_dec = t_dec[mask]
			n_steps = U_dec.shape[1]
		else:
			print("[AVERTISSEMENT] Fenêtre temporelle sans intersection, utilisation de l'intervalle complet.")

	# Si la décimation/fenêtre a abouti à trop peu d'images, essayer sans décimation
	if n_steps < 2 and step > 1:
		U_dec = U
		t_dec = t
		if t_window is not None and len(t_window) == 2:
			t0, t1 = float(min(t_window)), float(max(t_window))
			mask = (t_dec >= t0) & (t_dec <= t1)
			if np.any(mask):
				U_dec = U_dec[:, mask]
				t_dec = t_dec[mask]
		n_steps = U_dec.shape[1]

	# Choisir des indices uniformément sur les images disponibles en garantissant l'unicité
	k_req = int(max(1, n_snapshots))
	k_eff = int(min(k_req, n_steps))
	if k_eff < k_req:
		print(f"[AVERTISSEMENT] Seulement {k_eff} frames disponibles pour les instantanés (demandé {k_req}).")
	if n_steps <= 1:
		idx = np.array([0], dtype=int)
	else:
		idx = np.linspace(0, n_steps - 1, k_eff)
		idx = np.unique(np.rint(idx).astype(int))
		# Garantir au moins 2 indices si possible
		if idx.size < min(2, n_steps):
			idx = np.arange(min(n_steps, 2), dtype=int)

	# Déterminer les limites y à partir des images choisies
	Y = U_dec[:, idx]
	lo = float(np.nanpercentile(Y, 1))
	hi = float(np.nanpercentile(Y, 99))
	rng = max(1e-12, hi - lo)
	if y_scale is not None:
		c = 0.5 * (hi + lo)
		factor = max(1e-6, float(y_scale))
		half = 0.5 * rng * factor
		pad = float(y_pad_frac) * rng
		ymin, ymax = c - (half + pad), c + (half + pad)
	else:
		pad = float(y_pad_frac) * rng
		ymin, ymax = lo - pad, hi + pad

	import matplotlib.cm as cm
	import matplotlib.colors as mcolors

	fig, ax = plt.subplots(figsize=(9.5, 4.8))
	# Utiliser l'API non dépréciée pour récupérer la colormap
	cmap_obj = plt.get_cmap(cmap)
	norm = mcolors.Normalize(vmin=float(t_dec[idx[0]]), vmax=float(t_dec[idx[-1]]))
	for j, jj in enumerate(idx):
		col = cmap_obj(norm(float(t_dec[jj])))
		lbl = f"t={t_dec[jj]:.3f}s" if show_legend else None
		ax.plot(x, U_dec[:, jj], color=col, lw=float(linewidth), alpha=float(alpha), label=lbl)
	ax.set_title(title)
	ax.set_xlabel("x (m)")
	ax.set_ylabel("déplacement (m)")
	ax.grid(True, alpha=0.3)
	ax.set_ylim(ymin, ymax)
	if show_legend:
		ax.legend(ncol=2, fontsize=8, framealpha=0.85)
	if use_colorbar:
		sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
		sm.set_array([])
		cbar = fig.colorbar(sm, ax=ax)
		cbar.set_label("temps (s)")
	fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
	print(f"[INFO] Instantanés enregistrés dans : {out}")


# -------------------------------
#    FFT helper and plot
# -------------------------------
def compute_single_sided_fft(
	y: np.ndarray,
	dt: float,
	*,
	window: str = "hann",
	zero_pad_factor: float = 1.0,
):
	# Calcule le spectre d'amplitude à un seul côté avec fenêtre et échelle appropriée.
	# window : nom de la fenêtre ('hann','hamming','blackman', ...); zero_pad_factor : >=1.0 pour un échantillonnage fréquentiel plus fin.
	# Retourne (freqs, amp_lin).
	y = np.asarray(y, dtype=float).ravel()
	N = y.size
	if N < 2:
		return np.array([]), np.array([])
	# Window construction
	win_name = (window or "hann").lower()
	try:
		if sp is not None and hasattr(sp, "signal"):
			w = sp.signal.get_window(win_name, N, fftbins=True)
		else:
			# Fallback to numpy windows
			if win_name in ("hann", "hanning"):
				w = np.hanning(N)
			elif win_name == "hamming":
				w = np.hamming(N)
			elif win_name == "blackman":
				w = np.blackman(N)
			else:
				# rectangular if unknown
				w = np.ones(N, dtype=float)
	except Exception:
		w = np.hanning(N)

	yw = y * w
	# Zero padding pour améliorer l'échantillonnage fréquentiel (et non l'amplitude)
	try:
		factor = float(zero_pad_factor)
	except Exception:
		factor = 1.0
	factor = max(1.0, factor)
	target_len = int(np.ceil(N * factor))
	# pad to next power of two for efficiency
	n_fft = 1 << (target_len - 1).bit_length()
	Y = np.fft.rfft(yw, n=n_fft)
	f = np.fft.rfftfreq(n_fft, d=dt)
	# Correction d'amplitude via le gain cohérent (somme de la fenêtre) et échelle single-sided
	# Utiliser sum(w) au lieu de N pour garder l'amplitude invariante vis-à-vis du zero padding
	denom = float(np.sum(w))
	denom = denom if denom > 0 else float(N)
	amp = (2.0 / denom) * np.abs(Y)
	# DC and Nyquist (if present) should not be doubled
	amp[0] *= 0.5
	if n_fft % 2 == 0 and amp.size > 1:
		amp[-1] *= 0.5
	return f, amp


def plot_fft_png(
	signal: np.ndarray,
	dt: float,
	savepath: str,
	title: str = "FFT (déplacement)",
	fmax: float | None = None,
	*,
	fmin: float = 0.0,
	show_db: bool = True,
	smooth_window: int = 0,
	annotate_peaks: bool = True,
	n_peaks: int = 6,
	window: str = "hann",
	zero_pad_factor: float = 1.0,
) -> None:
	try:
		import matplotlib.pyplot as plt  # type: ignore
	except Exception as e:
		print("[AVERTISSEMENT] Matplotlib indisponible pour la FFT:", e)
		return
	f, A = compute_single_sided_fft(signal, dt, window=window, zero_pad_factor=zero_pad_factor)
	if f.size == 0:
		print("[AVERTISSEMENT] Signal trop court pour la FFT.")
		return
	# Normaliser dans [0,1] pour faciliter la visualisation relative
	Amax = float(np.max(A)) if A.size else 1.0
	if Amax > 0:
		A_lin = A / Amax
	else:
		A_lin = A

	# Filtre de bande (éviter DC/bruit faible)
	mask = f >= float(fmin)
	f_plot = f[mask]
	A_plot = A_lin[mask]

	# Lissage optionnel (moyenne glissante)
	A_smooth = None
	if int(smooth_window) and int(smooth_window) > 1 and A_plot.size:
		w = int(smooth_window)
		w = max(3, w + (w % 2 == 0))  # força ímpar >=3
		kern = np.ones(w, dtype=float) / float(w)
		A_smooth = np.convolve(A_plot, kern, mode="same")

	# Détection de pics (optionnel)
	peaks = []  # list[(freq, amp)]
	if annotate_peaks and A_plot.size:
		try:
			if sp is not None and hasattr(sp, "signal"):
				height_thr = max(0.05, 0.1 * float(np.max(A_plot)))
				idx, _ = sp.signal.find_peaks(A_plot, height=height_thr, distance=max(1, int(0.002 / max(dt, 1e-12))))
				cand = [(float(f_plot[i]), float(A_plot[i])) for i in idx]
			else:
				cand = []
		except Exception:
			cand = []
		if not cand:
			n_top = min(int(n_peaks) * 3, int(A_plot.size))
			if n_top > 0:
				top_idx = np.argpartition(A_plot, -n_top)[-n_top:]
				cand = [(float(f_plot[i]), float(A_plot[i])) for i in top_idx]
				cand.sort(key=lambda x: x[1], reverse=True)
	# retirer les pics trop proches (<= 2 Hz) et limiter la quantité
		for f0, a0 in cand:
			if any(abs(f0 - fp) < 2.0 for fp, _ in peaks):
				continue
			peaks.append((f0, a0))
			if len(peaks) >= int(n_peaks):
				break
	from pathlib import Path as _P
	out = _P(savepath)
	out.parent.mkdir(parents=True, exist_ok=True)

	if show_db:
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
	else:
		fig, ax1 = plt.subplots(1, 1, figsize=(10, 4.2))
		ax2 = None  # type: ignore

	# Haut : amplitude linéaire normalisée
	ax1.plot(f_plot, A_plot, lw=0.9, color="#1f77b4", alpha=0.85, label="FFT")
	if A_smooth is not None:
		ax1.plot(f_plot, A_smooth, lw=1.1, color="#ff7f0e", alpha=0.9, label="suavizado")
	if peaks:
		for fp, ap in peaks:
			ax1.plot([fp], [ap], "ro", ms=3)
	ax1.set_ylabel("Amplitude (norm)")
	ax1.set_title(title)
	ax1.grid(True, alpha=0.3)
	if A_smooth is not None:
		ax1.legend(loc="upper right", fontsize=8, framealpha=0.85)

	# Bas : échelle en dB relative
	if ax2 is not None:
		eps = 1e-12
		A_db = 20.0 * np.log10(np.maximum(A_plot, eps))
		ax2.plot(f_plot, A_db, lw=0.9, color="#1f77b4")
		if A_smooth is not None:
			A_db_s = 20.0 * np.log10(np.maximum(A_smooth, eps))
			ax2.plot(f_plot, A_db_s, lw=1.1, color="#ff7f0e")
		for fp, _ in peaks:
			ax2.axvline(fp, color="red", lw=0.8, alpha=0.5)
		ax2.set_ylabel("Amplitude (dB rel)")
		ax2.grid(True, which="both", alpha=0.3)

	# Limites de fréquence explicites (montre tout le spectre par défaut)
	fs = 1.0 / float(dt)
	x_right = float(fmax) if fmax is not None else fs / 2.0
	if ax2 is not None:
		ax2.set_xlim(left=max(0.0, float(fmin)), right=x_right)
	ax1.set_xlim(left=max(0.0, float(fmin)), right=x_right)
	(ax2 or ax1).set_xlabel("fréquence (Hz)")

	fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
	print(f"[INFO] FFT enregistrée dans : {out}")


def plot_fft_logdb_filled(
	signal: np.ndarray,
	dt: float,
	savepath: str,
	*,
	fmin: float = 20.0,
	fmax: float | None = None,
	min_db: float = -90.0,
	smooth_window: int = 0,
	color: str = "#4c78a8",
	edgecolor: str | None = None,
	linewidth: float = 0.8,
	annotate_peaks: bool = True,
	n_peaks: int = 6,
	title: str = "FFT — échelle log en fréquence (dB)",
	db_offset: float = 0.0,
	log_bins_per_octave: int | None = None,
	octave_smoothing: float = 0.0,
	smooth_domain: str = "db",
	window: str = "hann",
	zero_pad_factor: float = 1.0,
) -> None:
	# Trace une FFT (panneau unique) avec :
	# - axe de fréquence en log, amplitude en dB (relative), aire remplie.
	# Remarques : fmin > 0 pour l'échelle log ; min_db définit le plancher visuel.
	try:
		import matplotlib.pyplot as plt  # type: ignore
	except Exception as e:
		print("[AVERTISSEMENT] Matplotlib indisponible pour FFT log dB:", e)
		return

	f, A = compute_single_sided_fft(signal, dt, window=window, zero_pad_factor=zero_pad_factor)
	if f.size == 0:
		print("[AVERTISSEMENT] Signal trop court pour FFT (log dB).")
		return

	# Sélection de bande et normalisation
	fmin = max(1e-3, float(fmin))
	fs = 1.0 / float(dt)
	x_right = float(fmax) if fmax is not None else fs / 2.0
	mask = (f >= fmin) & (f <= x_right)
	if not np.any(mask):
		print("[AVERTISSEMENT] Bande de fréquences invalide pour FFT log dB.")
		return
	fp = f[mask]
	Ap = A[mask]
	Amax = float(np.max(Ap)) if Ap.size else 1.0
	if Amax <= 0:
		Amax = 1.0
	Arel = Ap / Amax

	# Lissage linéaire optionnel (indépendant du temps) de l'amplitude avant conversion dB
	if int(smooth_window) and int(smooth_window) > 1 and Ap.size:
		w = int(smooth_window)
		w = max(3, w + (w % 2 == 0))
		kern = np.ones(w, dtype=float) / float(w)
		Arel = np.convolve(Arel, kern, mode="same")

	# Regroupement en log-fréquence et lissage en fraction d'octave (optionnels)
	bpo = None if log_bins_per_octave is None else int(max(1, log_bins_per_octave))
	if bpo is not None and fp.size >= 5:
	# Construire une grille de fréquences espacée logarithmiquement
		fs = 1.0 / float(dt)
		x_right = float(fmax) if fmax is not None else fs / 2.0
		fmin_grid = max(fmin, float(fp[0]))
		if x_right <= fmin_grid * (1.0 + 1e-6):
			print("[AVERTISSEMENT] Grille log invalide pour FFT log dB.")
			return
	# Calculer le nombre de bacs en fonction des octaves
		octaves = np.log2(x_right / fmin_grid)
		Nbins = int(np.floor(octaves * bpo)) + 1
		f_grid = np.geomspace(fmin_grid, x_right, Nbins)
	# Interpoler l'amplitude sur la grille log (d'abord dans le domaine linéaire)
		log_fp = np.log(fp)
		log_fg = np.log(f_grid)
		Arel_g = np.interp(log_fg, log_fp, Arel, left=Arel[0], right=Arel[-1])
	# Largeur de lissage en fraction d'octave (en octaves)
		w_oct = float(max(0.0, octave_smoothing))
		if w_oct > 0.0 and Nbins >= 3:
			w_bins = int(round(w_oct * bpo))
			w_bins = max(3, w_bins + (w_bins % 2 == 0))
			if smooth_domain.lower() == "db":
				A_db_g = 20.0 * np.log10(np.maximum(Arel_g, 1e-12))
				kern = np.ones(w_bins, dtype=float) / float(w_bins)
				A_db_g = np.convolve(A_db_g, kern, mode="same")
				# Remplacer les tableaux de tracé
				fp = f_grid
				Arel = np.power(10.0, A_db_g / 20.0)
			else:
				kern = np.ones(w_bins, dtype=float) / float(w_bins)
				Arel = np.convolve(Arel_g, kern, mode="same")
				fp = f_grid

	# Conversion en dB (relative)
	eps = 1e-12
	A_db = 20.0 * np.log10(np.maximum(Arel, eps))
	# Optional vertical offset (visual alignment with external analyzers)
	A_db = A_db + float(db_offset)
	A_db = np.maximum(A_db, float(min_db))  # floor

	# Detect peaks for reference markers (optional)
	peaks = []
	if annotate_peaks and Arel.size:
		try:
			if sp is not None and hasattr(sp, "signal"):
				thr = max(0.05, 0.1 * float(np.max(Arel)))
				idx, _ = sp.signal.find_peaks(Arel, height=thr, distance=max(1, int(0.002 / max(dt, 1e-12))))
				cand = [(float(fp[i]), float(Arel[i])) for i in idx]
			else:
				cand = []
		except Exception:
			cand = []
		if not cand:
			n_top = min(int(n_peaks) * 3, int(Arel.size))
			if n_top > 0:
				top_idx = np.argpartition(Arel, -n_top)[-n_top:]
				cand = [(float(fp[i]), float(Arel[i])) for i in top_idx]
				cand.sort(key=lambda x: x[1], reverse=True)
		for f0, a0 in cand:
			if any(abs(f0 - fx) < 2.0 for fx, _ in peaks):
				continue
			peaks.append((f0, a0))
			if len(peaks) >= int(n_peaks):
				break

	from pathlib import Path as _P
	out = _P(savepath)
	out.parent.mkdir(parents=True, exist_ok=True)

	fig, ax = plt.subplots(figsize=(9.5, 6.0))
	ax.set_title(title)
	ax.set_xlabel("fréquence (Hz)")
	ax.set_ylabel("Amplitude (dB rel)")
	ax.set_xscale("log")
	# Filled spectrum area
	ax.fill_between(fp, A_db, float(min_db), facecolor=color, edgecolor=edgecolor or color, linewidth=float(linewidth), alpha=0.85, step=None)
	# Grid to resemble the reference (major+minor on log x)
	ax.grid(True, which="both", axis="both", alpha=0.4)
	ax.set_xlim(left=fmin, right=x_right)
	ax.set_ylim(bottom=float(min_db), top=0.0)

	# Marqueurs de pics optionnels sous forme de lignes verticales discrètes
	for fx, _ in peaks:
		ax.axvline(fx, color="#666666", lw=0.7, alpha=0.4)

	fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
	print(f"[INFO] FFT (log f, dB, remplie) enregistrée dans : {out}")


if __name__ == "__main__":
	# Tentative de monter M, K, C à partir du config/formulation
	import sys
	from pathlib import Path
	# Ajoute la racine du projet au sys.path pour importer le paquet 'digital_twin'
	ROOT = Path(__file__).resolve().parents[3]
	if str(ROOT) not in sys.path:
		sys.path.insert(0, str(ROOT))

	try:
		from digital_twin.back_end.fem.formulation import build_global_mkc_from_config  # type: ignore
		from digital_twin.back_end import config as _cfg  # type: ignore
		res = build_global_mkc_from_config(apply_fixed_bc=True, return_meta=True)
		if isinstance(res, tuple) and len(res) >= 3:
			if len(res) == 4:
				M, K, C, meta = res
			else:
				M, K, C = res[:3]
			print("[INFO] Matrices assemblées à partir du config.")
			# Ordre attendu par les validations : M, C, K
			validar_mck(M, C, K, verbose=True)
			# Paramètres de simulation à partir du config
			delta_t = float(getattr(_cfg, "DT", 1e-5))
			T_total = float(getattr(_cfg, "T_SIM", 0.1))
			definir_parametros_simulacao(delta_t, T_total)
			# Initialisation triangulaire (U0, U_{n-1}, U_n)
			L_eff = float(getattr(_cfg, "L", 1.0))
			h_eff = float(getattr(_cfg, "PLUCK_AMP", 0.0))
			x_p_rel = float(getattr(_cfg, "PLUCK_POS", 0.25))
			x_p = x_p_rel * L_eff
			U0, U_nm1, U_n = inicializar_estados_iniciais(M, L=L_eff, h=h_eff, x_p=x_p)
			print(f"U0 initialisé : shape={U0.shape}, max={np.nanmax(U0):.3e}, min={np.nanmin(U0):.3e}")
			# Démonstration : calcul de U1 par la forme classique
			_ = calcular_u1(M, C, K, U_n=U_n, U_nm1=U_nm1, delta_t=delta_t)

			# --- Analyse modale et tracé ---
			n = M.shape[0]
			x_coords = build_node_positions_from_config(n)
			freqs_hz, modes_full = compute_modal_frequencies_and_modes(M, K, num_modes=4)
			print("Premières fréquences (Hz) :", np.round(freqs_hz, 3))
			plots_dir = ROOT / "digital_twin" / "back_end" / "results" / "plots"
			plot_first_modes(x_coords, modes_full, freqs_hz, max_modes=4, savepath=str(plots_dir / "modes_first4.png"))

			# --- Démo Newmark-beta : pincement (triangulaire) en condition initiale, V0=0, F=0 ---
			n = M.shape[0]
			# U0 already initialized above with triangular shape at x_p using PLUCK_AMP
			V0_zero = np.zeros(n, dtype=float)
			F_zero = zero_force_provider(n)
			t_vec, U_hist, V_hist, A_hist = newmark_beta(M, C, K, F_zero, dt=delta_t, t_max=T_total, U0=U0, V0=V0_zero, A0=None)
			# Sauvegarder CSV des positions au cours du temps (t, u_0..u_{n-1})
			plots_dir = ROOT / "digital_twin" / "back_end" / "results" / "plots"
			plots_dir.mkdir(parents=True, exist_ok=True)
			csv_path = plots_dir / "string_positions.csv"
			save_displacement_csv(t_vec, U_hist, str(csv_path))
			# Plot x(t) for the chosen node
			try:
				import matplotlib.pyplot as plt  # type: ignore
			except Exception as _eplt:
				print("[AVERTISSEMENT] matplotlib indisponible pour tracer x(t):", _eplt)
			else:
				plots_dir = ROOT / "digital_twin" / "back_end" / "results" / "plots"
				plots_dir.mkdir(parents=True, exist_ok=True)
				plt.figure(figsize=(8,4))
				node_idx = max(1, n//3)
				plt.plot(t_vec, U_hist[node_idx, :], lw=1.2)
				plt.title(f"Déplacement au nœud {node_idx} — Newmark (β=1/4, γ=1/2), U0 triang (pincement), F=0")
				plt.xlabel("temps (s)")
				plt.ylabel("x (m)")
				plt.grid(True, alpha=0.3)
				outp = plots_dir / "newmark_node_displacement.png"
				plt.tight_layout(); plt.savefig(outp, dpi=150); plt.close()
				print(f"[INFO] Courbe x(t) enregistrée dans : {outp}")

				# Energies over time
				Ek, Ep, Et = compute_energies_over_time(M, K, U_hist, V_hist)
				plt.figure(figsize=(9,5))
				plt.plot(t_vec, Ek, label="Énergie cinétique")
				plt.plot(t_vec, Ep, label="Énergie potentielle")
				plt.plot(t_vec, Et, label="Énergie totale", lw=1.8)
				plt.title("Énergies au cours du temps")
				plt.xlabel("temps (s)")
				plt.ylabel("énergie (J)")
				plt.grid(True, alpha=0.3)
				plt.legend()
				outpE = plots_dir / "newmark_energies.png"
				plt.tight_layout(); plt.savefig(outpE, dpi=150); plt.close()
				print(f"[INFO] Courbes d'énergie enregistrées dans : {outpE}")

				# FFT of displacement at output node
				out_node = int(getattr(_cfg, "OUTPUT_NODE", max(1, n//2)))
				out_node = max(0, min(n-1, out_node))
				fft_path = plots_dir / "newmark_output_fft.png"
				try:
					fft_win = str(getattr(_cfg, "FFT_WINDOW", "hann"))
					fft_zpf = float(getattr(_cfg, "FFT_ZERO_PAD_FACTOR", 1.0))
				except Exception:
					fft_win, fft_zpf = "hann", 1.0
				plot_fft_png(
					U_hist[out_node, :],
					delta_t,
					str(fft_path),
					title=f"FFT — nœud {out_node}",
					fmax=None,
					window=fft_win,
					zero_pad_factor=fft_zpf,
				)
				# Additional FFT in log-frequency filled dB style
				fft_logdb_path = plots_dir / "newmark_output_fft_logdb.png"
				try:
					fmin_cfg = float(getattr(_cfg, "FFT_LOG_FMIN", 30.0))
					fmax_cfg = getattr(_cfg, "FFT_LOG_FMAX", None)
					fmax_cfg = None if fmax_cfg is None else float(fmax_cfg)
					min_db_cfg = float(getattr(_cfg, "FFT_LOG_MIN_DB", -90.0))
					smooth_w_cfg = int(getattr(_cfg, "FFT_LOG_SMOOTH", 0))
					use_velocity = bool(getattr(_cfg, "FFT_LOG_USE_VELOCITY", False))
					color_hex = str(getattr(_cfg, "FFT_LOG_COLOR", "#4c78a8"))
					db_offset_cfg = float(getattr(_cfg, "FFT_LOG_DB_OFFSET", 0.0))
					bpo_cfg = getattr(_cfg, "FFT_LOG_BINS_PER_OCTAVE", None)
					bpo_cfg = None if bpo_cfg is None else int(bpo_cfg)
					oct_smooth_cfg = float(getattr(_cfg, "FFT_LOG_OCTAVE_SMOOTH", 0.0))
					smooth_domain_cfg = str(getattr(_cfg, "FFT_LOG_SMOOTH_DOMAIN", "db"))
				except Exception:
					fmin_cfg, fmax_cfg, min_db_cfg, smooth_w_cfg = 30.0, None, -90.0, 0
					use_velocity, color_hex, db_offset_cfg = False, "#4c78a8", 0.0
					bpo_cfg, oct_smooth_cfg, smooth_domain_cfg = None, 0.0, "db"
				# Choose signal: displacement (default) or velocity for closer match to some analyzers
				_sig = V_hist[out_node, :] if use_velocity else U_hist[out_node, :]
				plot_fft_logdb_filled(
					_sig,
					delta_t,
					str(fft_logdb_path),
					fmin=fmin_cfg,
					fmax=fmax_cfg,
					min_db=min_db_cfg,
					smooth_window=smooth_w_cfg,
					color=color_hex,
					annotate_peaks=True,
					n_peaks=8,
					title=f"FFT — nœud {out_node} (log f, dB)",
					db_offset=db_offset_cfg,
					log_bins_per_octave=bpo_cfg,
					octave_smoothing=oct_smooth_cfg,
					smooth_domain=smooth_domain_cfg,
					window=fft_win,
					zero_pad_factor=fft_zpf,
				)

				# Animation du mouvement de la corde — générer des versions ralentie et temps réel
				x_coords = build_node_positions_from_config(n)
				try:
					_y_scale = float(getattr(_cfg, "ANIM_Y_SCALE", 1.0))
					_y_pad = float(getattr(_cfg, "ANIM_Y_PAD_FRAC", 0.05))
				except Exception:
					_y_scale, _y_pad = 1.0, 0.05

				# GIF au ralenti (par défaut : decim=5, fps~33)
				try:
					_decim_slow = int(getattr(_cfg, "ANIM_DECIM_SLOW", 5))
					_fps_slow = int(getattr(_cfg, "ANIM_FPS_SLOW", 33))
				except Exception:
					_decim_slow, _fps_slow = 5, 33
				_interval_slow_ms = max(1, int(round(1000.0 / max(1, _fps_slow))))
				# Statistiques : durée de simulation vs durée de lecture (ralenti)
				n_steps = int(U_hist.shape[1])
				sim_span = float((n_steps - 1) * delta_t)
				frames_slow = int((n_steps + max(1, _decim_slow) - 1) // max(1, _decim_slow))
				play_slow = float(frames_slow) / float(max(1, _fps_slow))
				speed_slow = sim_span / play_slow if play_slow > 0 else float('inf')
				print(f"[INFO] GIF ralenti :  sim={sim_span:.3f}s, images={frames_slow}, fps={_fps_slow}, lecture={play_slow:.3f}s, vitesse={speed_slow:.2f}x")
				anim_slow_path = plots_dir / "string_motion_slow.gif"
				animate_string_motion(
					x_coords,
					U_hist,
					interval_ms=_interval_slow_ms,
					decim=max(1, _decim_slow),
					savepath=str(anim_slow_path),
					show=False,
					y_scale=_y_scale,
					y_pad_frac=_y_pad,
				)

				# GIF temps réel : choisir la décimation de sorte que le temps simulé par image ≈ 1/fps
				try:
					_fps_real = int(getattr(_cfg, "ANIM_FPS_REAL", 30))
				except Exception:
					_fps_real = 30
				# decim_real ≈ 1 / (dt * fps)
				_decim_real = max(1, int(round(1.0 / (delta_t * float(max(1, _fps_real))))))
				_interval_real_ms = max(1, int(round(1000.0 / max(1, _fps_real))))
				# Statistiques : durée de simulation vs durée de lecture (temps réel)
				frames_real = int((n_steps + _decim_real - 1) // _decim_real)
				play_real = float(frames_real) / float(max(1, _fps_real))
				speed_real = sim_span / play_real if play_real > 0 else float('inf')
				print(f"[INFO] GIF temps réel :  sim={sim_span:.3f}s, images={frames_real}, fps={_fps_real}, lecture={play_real:.3f}s, vitesse={speed_real:.2f}x")
				anim_real_path = plots_dir / "string_motion_real.gif"
				animate_string_motion(
					x_coords,
					U_hist,
					interval_ms=_interval_real_ms,
					decim=_decim_real,
					savepath=str(anim_real_path),
					show=False,
					y_scale=_y_scale,
					y_pad_frac=_y_pad,
				)
				# Save first frame (t=0) as PNG
				png0 = plots_dir / "string_motion_t0.png"
				save_string_frame_png(x_coords, U_hist, frame_idx=0, savepath=str(png0))

				# Enregistrer également un PNG statique multi-instantanés (plus rapide que le GIF)
				try:
					n_snap = int(getattr(_cfg, "SNAPSHOTS_COUNT", 8))
					snap_decim = int(getattr(_cfg, "SNAPSHOTS_DECIM", 10))
				except Exception:
					n_snap, snap_decim = 8, 10
				snap_path = plots_dir / "string_snapshots.png"
				# Snapshot styling and controls from config
				try:
					_snap_y_scale = getattr(_cfg, "SNAPSHOTS_Y_SCALE", None)
					_snap_y_pad = float(getattr(_cfg, "SNAPSHOTS_Y_PAD_FRAC", 0.06))
					_snap_t_window = getattr(_cfg, "SNAPSHOTS_T_WINDOW", None)
					_snap_use_cbar = bool(getattr(_cfg, "SNAPSHOTS_USE_COLORBAR", True))
					_snap_cmap = str(getattr(_cfg, "SNAPSHOTS_CMAP", "viridis"))
					_snap_alpha = float(getattr(_cfg, "SNAPSHOTS_ALPHA", 0.9))
					_snap_lw = float(getattr(_cfg, "SNAPSHOTS_LINEWIDTH", 1.3))
					_snap_show_legend = bool(getattr(_cfg, "SNAPSHOTS_SHOW_LEGEND", False))
				except Exception:
					_snap_y_scale = None
					_snap_y_pad = 0.06
					_snap_t_window = None
					_snap_use_cbar = True
					_snap_cmap = "viridis"
					_snap_alpha = 0.9
					_snap_lw = 1.3
					_snap_show_legend = False
				plot_snapshots_png(
					x_coords,
					U_hist,
					t_vec,
					n_snapshots=n_snap,
					decim=snap_decim,
					savepath=str(snap_path),
					title="Profils de la corde à des temps sélectionnés",
					y_scale=_snap_y_scale,
					y_pad_frac=_snap_y_pad,
					t_window=_snap_t_window,
					use_colorbar=_snap_use_cbar,
					cmap=_snap_cmap,
					alpha=_snap_alpha,
					linewidth=_snap_lw,
					show_legend=_snap_show_legend,
				)

		else:
			print("[ERREUR] Retour inattendu de build_global_mkc_from_config")
	except Exception as e:
		print("[AVERTISSEMENT] Impossible d'assembler M,K,C à partir du config:", e)
		print("Définissez M, C, K et appelez validar_mck(M, C, K).")

