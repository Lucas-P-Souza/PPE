# Utilitaires de débogage centralisés dans un seul module
#
# Fonctionnalités
# - Interrupteur unique ON/OFF pour tous les affichages de debug du backend
# - Aides légères :
#   - dprint : impression protégée avec préfixe optionnel
#   - section : séparateur visuel (en-tête)
#   - matrix_stats / print_blocks / bc_edge_sums : résumés rapides via NumPy
#
# Bascule par défaut issue de digital_twin.back_end.config :
# - DEBUG_ENABLED (préféré)
# - Rétrocompatibilité avec DEBUG_RAYLEIGH si présent
#
# Usage
# from digital_twin.back_end.utils import debug as dbg
# dbg.dprint("quelque chose…")
# if dbg.is_enabled():
#     dbg.matrix_stats("M", M)
#
# Formatage paresseux (aucun coût si le debug est OFF) :
#   dbg.dlazy(lambda: f"max ω = {np.max(omegas)}")
from __future__ import annotations

from typing import Any, Callable

_ENABLED: bool = False
_LEVEL: str = "off"  # 'off' | 'concise' | 'verbose'
_PRINT_STEPS: bool = True
_PRINT_STEP_ENERGY: bool = False
_SOLVER_SAMPLES_TARGET: int = 12
_PRINT_RAYLEIGH_SUMMARY: bool = True
_PRINT_RAYLEIGH_DETAILS: bool = False
_PRINT_VALIDATORS: bool = True
_PRINT_FINAL_ENERGY_SUMMARY: bool = True
_PRINT_ONCE_FLAGS: dict[str, bool] = {}
_FIXED_STEP_EVERY: int = 0

# Initialisation depuis le module config si disponible
try:  # pragma: no cover - best-effort import
    from .. import config as _cfg  # type: ignore
    if hasattr(_cfg, "DEBUG_ENABLED"):
        _ENABLED = bool(getattr(_cfg, "DEBUG_ENABLED"))
    elif hasattr(_cfg, "DEBUG_RAYLEIGH"):
        # Rétrocompatibilité : on honore l'ancien indicateur si le nouveau est absent
        _ENABLED = bool(getattr(_cfg, "DEBUG_RAYLEIGH"))
    # Niveau et flags adicionais (valeurs par défaut si absent)
    _LEVEL = str(getattr(_cfg, "DEBUG_LEVEL", "off") or "off")
    _PRINT_STEPS = bool(getattr(_cfg, "DEBUG_PRINT_STEPS", True))
    _PRINT_STEP_ENERGY = bool(getattr(_cfg, "DEBUG_PRINT_STEP_ENERGY", False))
    _SOLVER_SAMPLES_TARGET = int(getattr(_cfg, "DEBUG_SOLVER_SAMPLES_TARGET", 12) or 12)
    _PRINT_RAYLEIGH_SUMMARY = bool(getattr(_cfg, "DEBUG_PRINT_RAYLEIGH_SUMMARY", True))
    _PRINT_RAYLEIGH_DETAILS = bool(getattr(_cfg, "DEBUG_PRINT_RAYLEIGH_DETAILS", False))
    _PRINT_VALIDATORS = bool(getattr(_cfg, "DEBUG_PRINT_VALIDATORS", True))
    _PRINT_FINAL_ENERGY_SUMMARY = bool(getattr(_cfg, "DEBUG_PRINT_FINAL_ENERGY_SUMMARY", True))
    _FIXED_STEP_EVERY = int(getattr(_cfg, "DEBUG_STEP_EVERY", 0) or 0)
except Exception:
    # Conserver la valeur False par défaut si l'import de config échoue
    _ENABLED = False
    _LEVEL = "off"
    _FIXED_STEP_EVERY = 0


def set_enabled(value: bool) -> None:
    # Active/Désactive les impressions de débogage à l'exécution.
    global _ENABLED
    _ENABLED = bool(value)


def is_enabled() -> bool:
    # Retourne l'état actuel d'activation du débogage.
    return bool(_ENABLED)


def is_concise() -> bool:
    return _ENABLED and (_LEVEL.lower() == "concise")


def is_verbose() -> bool:
    return _ENABLED and (_LEVEL.lower() == "verbose")


def dprint(*args: Any, prefix: str = "[DEBUG] ", sep: str = " ", end: str = "\n") -> None:
    # Impression protégée : n'affiche que si le débogage est activé.
    # prefix : chaîne optionnelle à préfixer (utiliser "" pour supprimer)
    if not _ENABLED:
        return
    try:
        if prefix:
            print(prefix, end="")
        print(*args, sep=sep, end=end)
    except Exception:
        # Ne jamais casser l'appelant à cause du débogage
        pass


def dlazy(make_msg: Callable[[], Any], *, prefix: str = "[DEBUG] ") -> None:
    # Impression paresseuse : ne construit le message que si le débogage est activé.
    # Usage : dlazy(lambda: f"max ω = {np.max(omegas)}")
    # Remarque : la fonction n'est PAS exécutée si le débogage est désactivé ; si elle retourne None, on ignore.
    if not _ENABLED:
        return
    try:
        msg = make_msg()
        if msg is None:
            return
        if prefix:
            print(prefix, end="")
        print(msg)
    except Exception:
        # Ne pas laisser un échec de debug affecter le flot principal
        pass


def _should_print_once(key: str) -> bool:
    # Retourne True si l'impression associée à 'key' n'a pas encore été effectuée.
    # Marque l'impression comme faite (idempotent côté appelant).
    done = _PRINT_ONCE_FLAGS.get(key, False)
    if not done:
        _PRINT_ONCE_FLAGS[key] = True
        return True
    return False


def print_rayleigh_explained(
    alpha: float,
    beta: float,
    omegas: Any,
    modes_ref: tuple[int, int],
    zetas_ref: tuple[float, float],
    *,
    print_once: bool = True,
) -> None:
    # Bloc d'explication détaillé pour l'amortissement de Rayleigh (centré ici).
    # Imprime une seule fois par défaut (print_once=True). Protégé par is_enabled().
    if not is_enabled():
        return
    try:
        import numpy as np
        two_pi = 2.0 * np.pi
        omegas = np.asarray(omegas, dtype=float)
        f_hz = omegas / two_pi if omegas.size else np.array([])
        p, q = modes_ref
        zp, zq = zetas_ref
        # Sécurité indices
        if omegas.size == 0 or p < 0 or q < 0 or p >= omegas.size or q >= omegas.size or p == q:
            # Indices invalides ou pas de modes — on affiche juste α/β
            msg_basic = (
                "\n[RAILEIGH] Paramètres (indices non valides ou omegas vides)\n"
                f"  • Paramètres: α={alpha:.3e} [1/s], β={beta:.3e} [s]\n"
            )
            if print_once and not _should_print_once("rayleigh_explained"):
                return
            dlazy(lambda: msg_basic, prefix="")
            return

        omega_p = float(omegas[p])
        omega_q = float(omegas[q])
        # Amortissements atteints aux fréquences de référence
        zeta_p_chk = 0.5 * (alpha/omega_p + beta*omega_p)
        zeta_q_chk = 0.5 * (alpha/omega_q + beta*omega_q)
        d_abs_p = abs(zeta_p_chk - zp)
        d_abs_q = abs(zeta_q_chk - zq)
        d_rel_p = (d_abs_p / zp * 100.0) if zp > 0 else float('nan')
        d_rel_q = (d_abs_q / zq * 100.0) if zq > 0 else float('nan')
        # Fréquence de croisement (où α/ω = β ω)
        fx_txt = "n/a"
        if alpha > 0.0 and beta > 0.0:
            try:
                omega_x = float(np.sqrt(alpha / beta))
                fx_txt = f"{omega_x/two_pi:.2f} Hz"
            except Exception:
                fx_txt = "n/a"

        def _fmt_samples(max_modes: int = 3) -> str:
            k = min(max_modes, omegas.size)
            if k <= 0:
                return "(aucun mode libre détecté)"
            vals: list[str] = []
            for i in range(k):
                wi = float(omegas[i])
                zi = 0.5 * (alpha/wi + beta*wi)
                vals.append(f"f{i+1}={f_hz[i]:.2f} Hz → ζ={zi:.4f}")
            return ", ".join(vals)

        def _make_msg() -> str:
            return (
                "\n"  # bloc multi-lignes explicatif
                "[RAYLEIGH] Paramètres et vérifications (explications)\n"
                "  • Loi: ζ(ω) = 1/2 (α/ω + β ω) — C = α M + β K\n"
                f"  • Références: modes p={p}, q={q} | f_p={f_hz[p]:.2f} Hz, f_q={f_hz[q]:.2f} Hz\n"
                f"  • Cibles: ζ_p={zp:.4f}, ζ_q={zq:.4f}\n"
                f"  • Atteints: ζ_p={zeta_p_chk:.4f} (Δ={d_abs_p:.2e}, {d_rel_p:.2f}%) | "
                f"ζ_q={zeta_q_chk:.4f} (Δ={d_abs_q:.2e}, {d_rel_q:.2f}%)\n"
                f"  • Paramètres: α={alpha:.3e} [1/s], β={beta:.3e} [s]\n"
                f"  • Fréquence de croisement (α/ω = βω): {fx_txt}\n"
                f"  • Échantillon ζ aux premiers modes: {_fmt_samples()}\n"
                "  • Note: α,β déterminés sur matrices évaluées (libres si CL fixées);\n"
                "          puis C formée sur M/K avant CL; bord de C annulé (sans diag=1).\n"
            )

        if print_once and not _should_print_once("rayleigh_explained"):
            return
        # Print detailed only if explicitly requested
        if _PRINT_RAYLEIGH_DETAILS or is_verbose():
            dlazy(_make_msg, prefix="")
        elif _PRINT_RAYLEIGH_SUMMARY or is_concise():
            # concise one-liner
            try:
                two_pi = 2.0 * __import__("numpy").pi
                f_hz = __import__("numpy").asarray(omegas, dtype=float) / two_pi
                p, q = modes_ref
                omega_p = float(omegas[p]) if len(f_hz) > p else float("nan")
                omega_q = float(omegas[q]) if len(f_hz) > q else float("nan")
                zeta_p_chk = 0.5 * (alpha/omega_p + beta*omega_p) if omega_p > 0 else float("nan")
                zeta_q_chk = 0.5 * (alpha/omega_q + beta*omega_q) if omega_q > 0 else float("nan")
                dlazy(lambda: (
                    f"[RAYLEIGH] α={alpha:.2e}, β={beta:.2e} | f_p={f_hz[p]:.1f}Hz, f_q={f_hz[q]:.1f}Hz | "
                    f"ζ_p={zeta_p_chk:.4f} (target {zetas_ref[0]:.4f}), ζ_q={zeta_q_chk:.4f} (target {zetas_ref[1]:.4f})"
                ), prefix="")
            except Exception:
                dlazy(lambda: f"[RAYLEIGH] α={alpha:.2e}, β={beta:.2e}", prefix="")
    except Exception as _e_dbg:
        dprint(f"[AVERTISSEMENT] Échec d'impression Rayleigh: {_e_dbg}")


def section(title: str, char: str = "-", width: int = 60) -> None:
    # Affiche un en-tête de section formaté si le débogage est activé.
    if not _ENABLED:
        return
    line = char * max(0, width)
    try:
        print(line)
        print(f"{title}")
        print(line)
    except Exception:
        pass


def matrix_stats(name: str, M) -> None:
    # Affiche des statistiques rapides pour une matrice/array NumPy (shape, nnz%, norme de Frobenius, min, max).
    # Utilise NumPy : np.count_nonzero, np.linalg.norm, np.min/np.max, etc.
    if not _ENABLED:
        return
    try:
        import numpy as np  # local import to avoid hard dependency at module import
        A = np.asarray(M)
        nz = int(np.count_nonzero(np.abs(A) > 0))
        tot = int(A.size)
        frac = (nz / tot * 100.0) if tot else 0.0
        fro = float(np.linalg.norm(A, 'fro')) if A.ndim == 2 else float(np.linalg.norm(A))
        amin = float(np.min(A))
        amax = float(np.max(A))
        dprint(f"{name}: shape={A.shape}, nnz={nz} ({frac:.1f}%), fro={fro:.3e}, min={amin:.3e}, max={amax:.3e}")
    except Exception as e:
        dprint(f"{name}: erreur de statistiques: {e}")


def print_blocks(name: str, M, k: int = 4) -> None:
    # Affiche les blocs kxk en haut-gauche et bas-droite (avec numpy.printoptions).
    if not _ENABLED:
        return
    try:
        import numpy as np
        A = np.asarray(M)
        k = int(max(1, k))
        tl = A[:k, :k]
        br = A[-k:, -k:]
        with np.printoptions(precision=3, suppress=True):
            dprint(f"{name} top-left {k}x{k}:\n{tl}", prefix="")
            dprint(f"{name} bottom-right {k}x{k}:\n{br}", prefix="")
    except Exception as e:
        dprint(f"{name}: erreur d'affichage des blocs: {e}")


def bc_edge_sums(name: str, M) -> None:
    # Affiche les sommes des lignes/colonnes de bord (0 et N-1) pour inspecter l'application des CL.
    if not _ENABLED:
        return
    try:
        import numpy as np
        A = np.asarray(M)
        r0 = float(np.sum(np.abs(A[0, :])))
        rn = float(np.sum(np.abs(A[-1, :])))
        c0 = float(np.sum(np.abs(A[:, 0])))
        cn = float(np.sum(np.abs(A[:, -1])))
        dprint(f"{name} edge sums: row0={r0:.3e}, rowN={rn:.3e}, col0={c0:.3e}, colN={cn:.3e}")
    except Exception as e:
        dprint(f"{name}: erreur des sommes de bords: {e}")


# ---------------------------------------------------------------------------
# Debug-only helpers for FEM diagnostics (moved from formulation)
# ---------------------------------------------------------------------------
def compute_local_element_matrices(
    tension: float,
    lin_density: float,
    *,
    dx_vector: list[float] | Any | None = None,
) -> list[dict[str, Any]]:
    # Retourne une liste de matrices élémentaires locales (maillage non uniforme uniquement).
    # Remarque : uniquement pour diagnostic/inspection. Utilise NumPy pour les matrices 2x2 locales.
    import numpy as np
    if dx_vector is None:
        raise ValueError("compute_local_element_matrices: dx_vector est obligatoire pour le mode non uniforme")
    dx_arr = np.asarray(dx_vector, dtype=float)
    if dx_arr.ndim != 1 or dx_arr.size == 0 or np.any(dx_arr <= 0):
        raise ValueError("dx_vector invalide")
    out: list[dict[str, Any]] = []
    for i, dx in enumerate(dx_arr):
        mass_pref = lin_density * dx / 6.0
        M_local = mass_pref * np.array([[2.0, 1.0], [1.0, 2.0]])
        stiff_pref = tension / dx
        K_local = stiff_pref * np.array([[1.0, -1.0], [-1.0, 1.0]])
        out.append({
            'index': i,
            'dx': float(dx),
            'mass_pref': float(mass_pref),
            'stiff_pref': float(stiff_pref),
            'M_local': M_local,
            'K_local': K_local,
        })
    return out


def print_local_element_matrices(
    tension: float,
    lin_density: float,
    *,
    dx_vector: list[float] | Any | None = None,
    limit: int | None = None,
) -> None:
    # Impression protégée des matrices élémentaires locales (non uniforme uniquement).
    if not is_enabled():
        return
    import numpy as np
    data = compute_local_element_matrices(tension=tension, lin_density=lin_density, dx_vector=dx_vector)
    total = len(data)
    show = data if limit is None else data[:limit]
    dprint(f"--- MATRICES LOCALES (total éléments = {total}) ---", prefix="")
    dprint(f"Paramètres globaux: μ = {lin_density:.6g} kg/m | T = {tension:.6g} N", prefix="")
    for entry in show:
        i = entry['index']; dx = entry['dx']
        mass_pref = entry.get('mass_pref'); stiff_pref = entry.get('stiff_pref')
        M_loc = entry['M_local']; K_loc = entry['K_local']
        dprint(f"\nÉlément {i} | dx = {(dx*1000):.6g} mm", prefix="")
        if mass_pref is not None and stiff_pref is not None:
            dprint("  Facteurs: mass_pref = μ*dx/6 = {:.6g} | stiff_pref = T/dx = {:.6g}".format(mass_pref, stiff_pref), prefix="")
        with np.printoptions(precision=3, suppress=True):
            dprint("M_local =\n" + str(M_loc), prefix="")
            dprint("K_local =\n" + str(K_loc), prefix="")
    if limit is not None and total > limit:
        dprint(f"\n... ({total - limit} éléments supplémentaires non affichés) ...", prefix="")


def print_formulation_diagnostics(
    M, K, C,
    *,
    meta: dict[str, Any] | None = None,
    modes_ref: tuple[int, int] | None = None,
    zetas_ref: tuple[float, float] | None = None,
    config: Any | None = None,
) -> None:
    # Diagnostics complets auparavant imprimés dans formulation.__main__.
    # Toutes les impressions sont protégées par dbg.is_enabled(). Utilise NumPy pour normes/VP/formatage.
    if not is_enabled():
        return
    import numpy as np
    _np = np
    meta = meta or {}
    try:
        dprint(f"Meta: {meta}", prefix="")
        dprint(f"Dimensions: {M.shape}", prefix="")
        dprint(f"Symétrie M,K,C: {np.allclose(M,M.T)} {np.allclose(K,K.T)} {np.allclose(C,C.T)}", prefix="")
        # Diagonals excerpt
        with _np.printoptions(precision=6, suppress=True):
            dprint("Diag M: " + str(_np.round(_np.diag(M),6)), prefix="")
            dprint("Diag K: " + str(_np.round(_np.diag(K),6)), prefix="")
            dprint("Diag C: " + str(_np.round(_np.diag(C),6)), prefix="")

        # M stats
        dprint("\n[DIAGNOSTIC] Statistiques M:", prefix="")
        nonzero = M[np.abs(M) > 0]
        if nonzero.size:
            dprint(f" min = {nonzero.min():.3e}", prefix="")
            dprint(f" max = {nonzero.max():.3e}", prefix="")
            dprint(f" somme = {M.sum():.3e}", prefix="")
            if config is not None:
                masse_teoricale = float(getattr(config, 'MU', 0.0) * meta.get('length_recon', 0.0))
                dprint(f" masse totale théorique = μ * L = {masse_teoricale:.3e} kg", prefix="")
                diference_masse = abs(masse_teoricale - M.sum())
                dprint(f" diference masse totale - somme de la matrice M = {diference_masse:.3e}", prefix="")
                denom = float(getattr(config, 'MU', 1e-10) * meta.get('length_recon', 1e-10)) or 1e-10
                dprint(f" porcentage de erreur sur la masse totale = {100.0 * diference_masse / denom:.3e} %", prefix="")
        else:
            dprint("  (aucune entrée non nulle détectée)", prefix="")
        with _np.printoptions(precision=3, suppress=False):
            dprint("\nM (notation scientifique) =\n" + str(M), prefix="")
        scale = 1e6
        with _np.printoptions(precision=3, suppress=True):
            dprint(f"\nM (x{scale:.0e}) =\n" + str(M * scale) + "\n", prefix="")

        # K stats
        dprint("[DIAGNOSTIC] Statistiques K:", prefix="")
        nonzero_K = K[np.abs(K) > 0]
        if nonzero_K.size:
            dprint(f" min = {nonzero_K.min():.3e}", prefix="")
            dprint(f" max = {nonzero_K.max():.3e}", prefix="")
            dprint(f" somme = {K.sum():.3e}", prefix="")
        else:
            dprint("  (aucune entrée non nulle détectée)", prefix="")
        with _np.printoptions(precision=3, suppress=False):
            dprint("\nK (notation scientifique) =\n" + str(K), prefix="")
        if nonzero_K.size and float(np.max(np.abs(nonzero_K))) > 1e3:
            scale_K = 1e-3
            with _np.printoptions(precision=3, suppress=True):
                dprint(f"\nK (échelle réduite x{scale_K:.0e}) =\n" + str(K * scale_K) + "\n", prefix="")

        # C stats
        dprint("[DIAGNOSTIC] Statistiques C:", prefix="")
        nonzero_C = C[np.abs(C) > 0]
        if nonzero_C.size:
            dprint(f" min = {nonzero_C.min():.3e}", prefix="")
            dprint(f" max = {nonzero_C.max():.3e}", prefix="")
            dprint(f" somme = {C.sum():.3e}", prefix="")
        else:
            dprint("  (aucune entrée non nulle détectée)", prefix="")
        with _np.printoptions(precision=3, suppress=False):
            dprint("\nC (notation scientifique) =\n" + str(C), prefix="")
        if nonzero_C.size and float(np.max(np.abs(nonzero_C))) > 1e3:
            scale_C = 1e-3
            with _np.printoptions(precision=3, suppress=True):
                dprint(f"\nC (échelle réduite x{scale_C:.0e}) =\n" + str(C * scale_C) + "\n", prefix="")

        # Rayleigh diagnostics if available
        try:
            alpha = meta.get('alpha', None) if isinstance(meta, dict) else None
            beta = meta.get('beta', None) if isinstance(meta, dict) else None
            _omegas_meta = _np.asarray(meta.get('omegas', []), dtype=float) if isinstance(meta, dict) else _np.array([])
            if alpha is not None and beta is not None:
                dprint("\n[DIAGNOSTIC] Rayleigh (C = α M + β K):", prefix="")
                dprint(f"  Alpha = {alpha:.6e} [1/s] | Beta = {beta:.6e} [s]", prefix="")
                nM = float(_np.linalg.norm(M, 'fro'))
                nK = float(_np.linalg.norm(K, 'fro'))
                nC = float(_np.linalg.norm(C, 'fro'))
                nAM = float(_np.linalg.norm(alpha * M, 'fro'))
                nBK = float(_np.linalg.norm(beta * K, 'fro'))
                share_am = (nAM / nC * 100.0) if nC > 0 else 0.0
                share_bk = (nBK / nC * 100.0) if nC > 0 else 0.0
                dprint(f"  ||M||_F = {nM:.3e} | ||K||_F = {nK:.3e} | ||C||_F = {nC:.3e}", prefix="")
                dprint(f"  ||αM||_F = {nAM:.3e} ({share_am:.1f}% de ||C||) | ||βK||_F = {nBK:.3e} ({share_bk:.1f}% de ||C||)", prefix="")
                if _omegas_meta.size > 0:
                    zetas_real = 0.5 * (alpha / _omegas_meta + beta * _omegas_meta)
                    k_show = int(min(10, zetas_real.size))
                    with _np.printoptions(precision=4, suppress=True):
                        dprint(f"  ζ(modaux) réalisés — premiers {k_show}: {zetas_real[:k_show]}", prefix="")
                    if isinstance(modes_ref, tuple) and len(modes_ref) == 2:
                        p, q = modes_ref
                        if 0 <= p < zetas_real.size and 0 <= q < zetas_real.size:
                            zp, zq = zetas_real[p], zetas_real[q]
                            try:
                                zt_p, zt_q = zetas_ref if zetas_ref is not None else (None, None)
                            except Exception:
                                zt_p, zt_q = None, None
                            msg_p = f"p={p}: ζ_real = {zp:.5f}"
                            if zt_p is not None:
                                msg_p += f" | ζ_cible = {zt_p:.5f} | Δ = {abs(zp - zt_p):.2e}"
                            msg_q = f"q={q}: ζ_real = {zq:.5f}"
                            if zt_q is not None:
                                msg_q += f" | ζ_cible = {zt_q:.5f} | Δ = {abs(zq - zt_q):.2e}"
                            dprint("  Cibles:", prefix="")
                            dprint("   " + msg_p, prefix="")
                            dprint("   " + msg_q, prefix="")
                            dprint("  (Des valeurs de C petites peuvent être NORMALES si les ζ réalisés correspondent aux cibles.)", prefix="")
        except Exception as _e_diag:
                dprint(f"[AVERTISSEMENT] Échec des diagnostics Rayleigh: {_e_diag}")

        # Frequencies (Hz) control
        try:
            _omegas = _np.asarray(meta.get('omegas', []), dtype=float)
            if _omegas.size == 0:
                A = _np.linalg.solve(M, K)
                eigvals, _ = _np.linalg.eig(A)
                eigvals = _np.real(eigvals)
                eigvals[eigvals < 0] = 0.0
                _omegas = _np.sqrt(eigvals)
                _omegas.sort()
            f_hz = _omegas / (2 * _np.pi)
            with _np.printoptions(precision=3, suppress=True):
                dprint("\nFréquences (Hz) — premières 10 = " + str(f_hz[:10]), prefix="")
            if isinstance(modes_ref, tuple) and len(modes_ref) == 2:
                p, q = modes_ref
                if 0 <= p < f_hz.size and 0 <= q < f_hz.size:
                    dprint(f"Modes réf. (p={p}, q={q}) → f_p = {f_hz[p]:.3f} Hz | f_q = {f_hz[q]:.3f} Hz | ζ = {zetas_ref}", prefix="")
        except Exception as e:
            dprint(f"[AVERTISSEMENT] Échec lors de l'impression des fréquences: {e}")
    except Exception:
        # guard against any printing failure
        pass


# ---------------------------------------------------------------------------
# Solver-specific debug helpers
# ---------------------------------------------------------------------------
def get_solver_sample_interval(n_steps: int) -> int:
    # Retourne l'intervalle d'échantillonnage pour les impressions par pas.
    # Essaie config.DEBUG_SOLVER_SAMPLE, sinon ~50 échantillons sur la durée.
    try:
        from .. import config as _cfg  # type: ignore
        # Prefer explicit target if present
        target = int(getattr(_cfg, "DEBUG_SOLVER_SAMPLES_TARGET", _SOLVER_SAMPLES_TARGET))
        target = max(1, target)
        return max(1, int(round(n_steps / target)))
    except Exception:
        pass
    # Default heuristic: ~12 samples
    return max(1, int(round(n_steps / max(1, _SOLVER_SAMPLES_TARGET))))


def get_fixed_step_interval() -> int:
    # Returns fixed step interval for printing. When >0, overrides sampling cadence.
    try:
        from .. import config as _cfg  # type: ignore
        val = int(getattr(_cfg, "DEBUG_STEP_EVERY", _FIXED_STEP_EVERY))
        return int(max(0, val))
    except Exception:
        return int(max(0, _FIXED_STEP_EVERY))


def print_solver_setup_summary(M, C, K, *, dt: float, t_max: float, n_steps: int, constrained_idxs: list[int] | Any) -> None:
    # Résumé des tailles, DOF libres/contraints, pas de temps et normes.
    if not is_enabled():
        return
    try:
        import numpy as np
        n = int(M.shape[0])
        constrained = list(constrained_idxs) if constrained_idxs is not None else []
        free_n = n - len(constrained)
        dprint(f"[SOLVER] Setup: n={n} | free={free_n} | constrained={len(constrained)} | dt={dt:.6e}s | t_max={t_max:.6e}s | n_steps={n_steps}")
        # Normes de Frobenius (information simple de magnitude)
        try:
            nM = float(np.linalg.norm(np.asarray(M), 'fro'))
            nC = float(np.linalg.norm(np.asarray(C), 'fro'))
            nK = float(np.linalg.norm(np.asarray(K), 'fro'))
            dprint(f"[SOLVER] Normes: ||M||_F={nM:.3e} | ||C||_F={nC:.3e} | ||K||_F={nK:.3e}")
        except Exception:
            pass
        # Aperçu des contraintes
        if len(constrained) > 0:
            show = constrained[:10]
            suffix = "..." if len(constrained) > 10 else ""
            dprint(f"[SOLVER] Constrained idx (first 10): {show}{suffix}")
    except Exception:
        pass


def print_newmark_constants(*, dt: float, beta: float, gamma: float, a0: float, a1: float, a2: float, a3: float, a4: float, a5: float) -> None:
    # Affiche les coefficients Newmark utilisés.
    if not is_enabled():
        return
    try:
        dprint("[SOLVER] Newmark (β, γ) et constantes:", prefix="")
        dprint(f"  β={beta:.6g}, γ={gamma:.6g}, dt={dt:.6e}s", prefix="")
        dprint(f"  a0={a0:.3e}, a1={a1:.3e}, a2={a2:.3e}, a3={a3:.3e}, a4={a4:.3e}, a5={a5:.3e}", prefix="")
    except Exception:
        pass


def print_step_snapshot(k: int, t_k: float, U_col, V_col, A_col, *, M=None, K=None, compute_energy: bool = True) -> None:
    # Imprime un résumé pour un pas de temps: max/min des états et énergies (optionnel).
    if not is_enabled() or not _PRINT_STEPS:
        return
    try:
        import numpy as np
        U_col = np.asarray(U_col, dtype=float).reshape(-1)
        V_col = np.asarray(V_col, dtype=float).reshape(-1)
        A_col = np.asarray(A_col, dtype=float).reshape(-1)
        u_max = float(np.nanmax(np.abs(U_col)))
        v_max = float(np.nanmax(np.abs(V_col)))
        a_max = float(np.nanmax(np.abs(A_col)))
        line = f"[STEP] k={k:6d} t={t_k:10.6e} | |U|max={u_max:.3e}, |V|max={v_max:.3e}, |A|max={a_max:.3e}"
        if _PRINT_STEP_ENERGY and compute_energy and (M is not None) and (K is not None):
            try:
                Ek = 0.5 * float(V_col.T @ (np.asarray(M) @ V_col))
                Ep = 0.5 * float(U_col.T @ (np.asarray(K) @ U_col))
                Et = Ek + Ep
                line += f" | Ek={Ek:.3e}, Ep={Ep:.3e}, Et={Et:.3e}"
            except Exception:
                pass
        dprint(line)
    except Exception:
        pass


def print_energy_start_end(*, t0: float, tn: float, Ek0: float, Ep0: float, Et0: float, Ekn: float, Epn: float, Etn: float) -> None:
    # Résumé simple des énergies au début et à la fin (utile pour vérifier la décroissance avec amortissement).
    if not is_enabled() or not _PRINT_FINAL_ENERGY_SUMMARY:
        return
    try:
        drop = (Et0 - Etn) / Et0 * 100.0 if (Et0 is not None and Et0 > 0) else float('nan')
        dprint("[ENERGY] Début/Fin:", prefix="")
        dprint(f"  t0={t0:.6e}: Ek0={Ek0:.3e}, Ep0={Ep0:.3e}, Et0={Et0:.3e}", prefix="")
        dprint(f"  tn={tn:.6e}: Ekn={Ekn:.3e}, Epn={Epn:.3e}, Etn={Etn:.3e}", prefix="")
        dprint(f"  ΔEtot = {Et0 - Etn:.3e} ({drop:.2f}%)", prefix="")
    except Exception:
        pass


def validators_enabled() -> bool:
    return is_enabled() and _PRINT_VALIDATORS


def rayleigh_summary_enabled() -> bool:
    return is_enabled() and (_PRINT_RAYLEIGH_SUMMARY or is_concise())
