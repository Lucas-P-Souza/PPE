"""
Shim de compatibilité pour l'ancien module solver.

Objectif:
- Centraliser les imports et réexporter des fonctions depuis les modules
	organisés, sans modifier l'API historique (PT/EN). Les implémentations
	actives résident dans:
	- fem.time_integration (paramètres, CI, Newmark-beta, énergies, force nulle)
	- fem.modal (analyse modale: K v = λ M v)
	- utils.validators (validation M,C,K)
	- analysis.* et viz.* (tracés et analyses)

Remarques:
- Ce fichier ne contient pas d'algorithmes nouveaux; il délègue et documente.
- Les alias FR sont préférés et des alias PT/EN sont conservés pour compat.
"""

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
			# Provide safe fallbacks for advanced helpers used by the solver
			@staticmethod
			def get_solver_sample_interval(n_steps: int) -> int:
				try:
					return max(1, int(round(n_steps / 50)))
				except Exception:
					return 100
			@staticmethod
			def print_solver_setup_summary(*args, **kwargs): pass
			@staticmethod
			def print_newmark_constants(*args, **kwargs): pass
			@staticmethod
			def print_step_snapshot(*args, **kwargs): pass
			@staticmethod
			def print_energy_start_end(*args, **kwargs): pass
		dbg = _DbgNoOp()  # type: ignore

# Shims de compatibilité: déléguer vers les nouveaux modules (préférence FR)
try:
	from ..utils.validators import valider_mck as _valider_mck_impl  # type: ignore
except Exception:
	_valider_mck_impl = None  # type: ignore
try:
	from .time_integration import (
		# Importer directement les fonctions canoniques FR
		definir_parametres_simulation as _definir_parametres_simulation_impl,
		initialiser_u0_triangle as _initialiser_u0_triangle_impl,
		initialiser_etats_initiaux as _initialiser_etats_initiaux_impl,
		calculer_u1 as _calculer_u1_impl,
		fournisseur_force_nulle as _fournisseur_force_nulle_impl,
		integrer_newmark_beta as _integrer_newmark_beta_impl,
		calculer_energies_dans_le_temps as _calculer_energies_au_cours_du_temps_impl,
	)  # type: ignore
except Exception:
	_definir_parametres_simulation_impl = None  # type: ignore
	_initialiser_u0_triangle_impl = None  # type: ignore
	_initialiser_etats_initiaux_impl = None  # type: ignore
	_calculer_u1_impl = None  # type: ignore
	_fournisseur_force_nulle_impl = None  # type: ignore
	_integrer_newmark_beta_impl = None  # type: ignore
	_calculer_energies_au_cours_du_temps_impl = None  # type: ignore

# Analyse modale
try:
	from .modal import (
		compute_modal_frequencies_and_modes as _compute_modal_frequencies_and_modes_impl,
		detect_constrained_dofs_mk as _detect_constrained_dofs_mk_impl,
	)  # type: ignore
except Exception:
	_compute_modal_frequencies_and_modes_impl = None  # type: ignore
	_detect_constrained_dofs_mk_impl = None  # type: ignore

# Analyse (FFT / spectrogramme)
try:
	from ..analysis.fft import (
		calculer_fft_monocotee as _compute_single_sided_fft_impl,
		tracer_fft_png as _plot_fft_png_impl,
		tracer_fft_logdb_remplie as _plot_fft_logdb_filled_impl,
	)  # type: ignore
except Exception:
	_compute_single_sided_fft_impl = None  # type: ignore
	_plot_fft_png_impl = None  # type: ignore
	_plot_fft_logdb_filled_impl = None  # type: ignore
try:
	from ..analysis.spectrogram import plot_spectrogram as _plot_spectrogram_impl  # type: ignore
except Exception:
	_plot_spectrogram_impl = None  # type: ignore

# Visualisation (plots/animation)
try:
	from ..viz.plots import (
		plot_first_modes as _plot_first_modes_impl,
		plot_snapshots_png as _plot_snapshots_png_impl,
		save_string_frame_png as _save_string_frame_png_impl,
	)  # type: ignore
except Exception:
	_plot_first_modes_impl = None  # type: ignore
	_plot_snapshots_png_impl = None  # type: ignore
	_save_string_frame_png_impl = None  # type: ignore
try:
	from ..viz.anim import animate_string_motion as _animate_string_motion_impl  # type: ignore
except Exception:
	_animate_string_motion_impl = None  # type: ignore

# E/S
try:
	from ..io.exports import save_displacement_csv as _save_displacement_csv_impl  # type: ignore
except Exception:
	_save_displacement_csv_impl = None  # type: ignore


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


def valider_mck(M: np.ndarray, C: np.ndarray, K: np.ndarray, *, verbose: bool = True) -> None:
	"""Validation FR de M, C, K (délègue à utils.validators)."""
	if _valider_mck_impl is None:
		# Fallback local basique si l'import a échoué
		ok, msg, _ = _is_square_same_shape(M, C, K)
		if not ok:
			raise ValueError(f"[ERREUR] Échec de la vérification des dimensions : {msg}")
		# Effectuer au minimum le contrôle des CL
		for name, A in (("M", M), ("C", C), ("K", K)):
			allow_zero = True if name == "C" else False
			ok_bc, msg_bc = _bc_applied_at_ends(A, allow_diag_zero=allow_zero)
			if not ok_bc:
				raise ValueError(f"[ERREUR] CL non conformes dans {name}: {msg_bc}")
		return None
	return _valider_mck_impl(M, C, K, verbose=verbose)


def definir_parametres_simulation(delta_t: float, T_total: float):
	"""Renvoie les paramètres de simulation (FR)."""
	if _definir_parametres_simulation_impl is None:
		raise ImportError("fem.time_integration.definir_parametres_simulation indisponible")
	return _definir_parametres_simulation_impl(delta_t, T_total)

def initialiser_u0_triangle(M: np.ndarray, *, L: float, h: float, x_p: float) -> np.ndarray:
	"""Initialisation triangulaire (FR)."""
	if _initialiser_u0_triangle_impl is None:
		raise ImportError("fem.time_integration.initialiser_u0_triangle indisponible")
	return _initialiser_u0_triangle_impl(M, L=L, h=h, x_p=x_p)

def initialiser_etats_initiaux(M: np.ndarray, *, L: float, h: float, x_p: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Conditions initiales (FR)."""
	if _initialiser_etats_initiaux_impl is None:
		raise ImportError("fem.time_integration.initialiser_etats_initiaux indisponible")
	return _initialiser_etats_initiaux_impl(M, L=L, h=h, x_p=x_p)

def calculer_u1(M: np.ndarray, C: np.ndarray, K: np.ndarray,
				U_n: np.ndarray, U_nm1: np.ndarray, delta_t: float) -> np.ndarray:
	"""Premier pas (FR)."""
	if _calculer_u1_impl is None:
		raise ImportError("fem.time_integration.calculer_u1 indisponible")
	return _calculer_u1_impl(M, C, K, U_n=U_n, U_nm1=U_nm1, delta_t=delta_t)


def _detect_constrained_dofs_mk(M: np.ndarray, K: np.ndarray, atol: float = 1e-12) -> np.ndarray:
	"""Shim vers fem.modal.detect_constrained_dofs_mk."""
	if _detect_constrained_dofs_mk_impl is None:
		raise ImportError("fem.modal.detect_constrained_dofs_mk indisponible")
	return _detect_constrained_dofs_mk_impl(M, K, atol=atol)


def compute_modal_frequencies_and_modes(M: np.ndarray, K: np.ndarray, num_modes: int = 4):
	"""Shim → fem.modal (fréquences et formes modales)."""
	if _compute_modal_frequencies_and_modes_impl is None:
		raise ImportError("fem.modal.compute_modal_frequencies_and_modes indisponible")
	return _compute_modal_frequencies_and_modes_impl(M, K, num_modes=num_modes)


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
	"""Initialisation sinusoïdale simple (conservée pour compatibilité)."""
	n = M.shape[0]
	x = build_node_positions_from_config(n)
	if L <= 0:
		L = float(x[-1]) if x[-1] > 0 else 1.0
	U0 = amplitude * np.sin(np.pi * x / L)
	U0[0] = 0.0
	U0[-1] = 0.0
	return U0.astype(float, copy=False)


def fournisseur_force_nulle(n: int):
	"""Fournisseur de force nulle (FR)."""
	if _fournisseur_force_nulle_impl is None:
		raise ImportError("fem.time_integration.fournisseur_force_nulle indisponible")
	return _fournisseur_force_nulle_impl(n)



def plot_first_modes(x: np.ndarray, modes_full: np.ndarray, freqs_hz: np.ndarray, *, max_modes: int = 4, savepath: str | None = None) -> None:
	"""Shim vers viz.plots.plot_first_modes."""
	if _plot_first_modes_impl is None:
		raise ImportError("viz.plots.plot_first_modes indisponible")
	return _plot_first_modes_impl(x, modes_full, freqs_hz, max_modes=max_modes, savepath=savepath)


# -------------------------------
#    Newmark-beta (β=1/4, γ=1/2)
# -------------------------------
def integrer_newmark_beta(
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
	"""Intégrateur Newmark-beta (FR)."""
	if _integrer_newmark_beta_impl is None:
		raise ImportError("fem.time_integration.integrer_newmark_beta indisponible")
	return _integrer_newmark_beta_impl(M, C, K, F, dt, t_max, U0=U0, V0=V0, A0=A0)



# -------------------------------
#    Energy computation helpers
# -------------------------------
def calculer_energies_au_cours_du_temps(M: np.ndarray, K: np.ndarray, U: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Calcul des énergies au cours du temps (FR)."""
	if _calculer_energies_au_cours_du_temps_impl is None:
		raise ImportError("fem.time_integration.calculer_energies_dans_le_temps indisponible")
	return _calculer_energies_au_cours_du_temps_impl(M, K, U, V)


# -------------------------------
#    Export CSV des positions
# -------------------------------
def save_displacement_csv(t: np.ndarray, U: np.ndarray, savepath: str) -> None:
	"""Shim → io.exports.save_displacement_csv (FR)."""
	if _save_displacement_csv_impl is None:
		raise ImportError("io.exports.save_displacement_csv indisponible")
	return _save_displacement_csv_impl(t, U, savepath)


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
	"""Shim → viz.anim.animate_string_motion (FR)."""
	if _animate_string_motion_impl is None:
		raise ImportError("viz.anim.animate_string_motion indisponible")
	return _animate_string_motion_impl(
		x,
		U,
		interval_ms=interval_ms,
		decim=decim,
		savepath=savepath,
		show=show,
		y_scale=y_scale,
		y_pad_frac=y_pad_frac,
		y_limits=y_limits,
	)


def save_string_frame_png(
	x: np.ndarray,
	U: np.ndarray,
	frame_idx: int,
	savepath: str,
	*,
	y_scale: float | None = None,
	y_pad_frac: float = 0.05,
) -> None:
	"""Shim → viz.plots.save_string_frame_png (FR)."""
	if _save_string_frame_png_impl is None:
		raise ImportError("viz.plots.save_string_frame_png indisponible")
	return _save_string_frame_png_impl(x, U, frame_idx, savepath, y_scale=y_scale, y_pad_frac=y_pad_frac)


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
	"""Shim → viz.plots.plot_snapshots_png (FR)."""
	if _plot_snapshots_png_impl is None:
		raise ImportError("viz.plots.plot_snapshots_png indisponible")
	return _plot_snapshots_png_impl(
		x,
		U,
		t,
		n_snapshots=n_snapshots,
		decim=decim,
		savepath=savepath,
		title=title,
		y_scale=y_scale,
		y_pad_frac=y_pad_frac,
		t_window=t_window,
		use_colorbar=use_colorbar,
		cmap=cmap,
		alpha=alpha,
		linewidth=linewidth,
		show_legend=show_legend,
	)


def calculer_fft_monocotee(
	y: np.ndarray,
	dt: float,
	*,
	window: str = "hann",
	zero_pad_factor: float = 1.0,
):
	"""Shim → analysis.fft.calculer_fft_monocotee (FR)."""
	if _compute_single_sided_fft_impl is None:
		raise ImportError("analysis.fft.calculer_fft_monocotee indisponible")
	return _compute_single_sided_fft_impl(y, dt, window=window, zero_pad_factor=zero_pad_factor)


def tracer_fft_png(
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
	"""Shim → analysis.fft.tracer_fft_png (FR)."""
	if _plot_fft_png_impl is None:
		raise ImportError("analysis.fft.tracer_fft_png indisponible")
	return _plot_fft_png_impl(
		signal,
		dt,
		savepath,
		title=title,
		fmax=fmax,
		fmin=fmin,
		show_db=show_db,
		smooth_window=smooth_window,
		annotate_peaks=annotate_peaks,
		n_peaks=n_peaks,
		window=window,
		zero_pad_factor=zero_pad_factor,
	)


def tracer_fft_logdb_remplie(
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
	"""Shim → analysis.fft.tracer_fft_logdb_remplie (FR)."""
	if _plot_fft_logdb_filled_impl is None:
		raise ImportError("analysis.fft.tracer_fft_logdb_remplie indisponible")
	return _plot_fft_logdb_filled_impl(
		signal,
		dt,
		savepath,
		fmin=fmin,
		fmax=fmax,
		min_db=min_db,
		smooth_window=smooth_window,
		color=color,
		edgecolor=edgecolor,
		linewidth=linewidth,
		annotate_peaks=annotate_peaks,
		n_peaks=n_peaks,
		title=title,
		db_offset=db_offset,
		log_bins_per_octave=log_bins_per_octave,
		octave_smoothing=octave_smoothing,
		smooth_domain=smooth_domain,
		window=window,
		zero_pad_factor=zero_pad_factor,
	)


def plot_spectrogram(
	signal: np.ndarray,
	dt: float,
	savepath: str,
	*,
	fmin: float = 0.0,
	fmax: float | None = None,
	nperseg: int | None = None,
	overlap_frac: float = 0.5,
	cmap: str = "magma",
	title: str = "Spectrogramme (dB)",
) -> None:
	"""Shim → analysis.spectrogram.plot_spectrogram (FR)."""
	if _plot_spectrogram_impl is None:
		raise ImportError("analysis.spectrogram.plot_spectrogram indisponible")
	return _plot_spectrogram_impl(
		signal,
		dt,
		savepath,
		fmin=fmin,
		fmax=fmax,
		nperseg=nperseg,
		overlap_frac=overlap_frac,
		cmap=cmap,
		title=title,
	)


if __name__ == "__main__":  # pragma: no cover
	print("[INFO] Ce module est un shim de compatibilité et ne doit pas être exécuté directement.")

