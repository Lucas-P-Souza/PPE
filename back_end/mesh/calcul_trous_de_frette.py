"""
Script utilitaire (inspection console) de la maille non uniforme basée sur les frettes.

Objectif
--------
Permettre :
 1. Génération ponctuelle d'une maille fret (sans lancer la simulation complète).
 2. Visualisation tabulaire des intervalles entre frettes et de leur discrétisation
        en sous-éléments (dx) quasi uniformes.
 3. Analyse rapide des erreurs de reconstruction (théorique vs somme des dx arrondis).
 4. Distribution statistique des longueurs élémentaires (si "afficher_detail=True").
 5. Injection optionnelle des variables FRET_* dans le module `config` (runtime) pour
        que le reste du backend utilise immédiatement ce maillage.

Usage typique
-------------
        python calcul_trous_de_frette.py

Le script essaye d'abord d'importer le package installé (mode projet). S'il est
exécuté directement depuis le dossier, un fallback ajoute la racine du projet à
`sys.path`.

Notes d'implémentation
----------------------
- On conserve les noms de fonctions d'origine (ex: gerar_malha_trastes) pour
    cohérence avec le reste du code déjà adapté.
- Les variables locales du script utilisent un style français clair.
- Aucune écriture disque ici (le cache JSON est géré côté `fret_mesh` si utilisé).
"""
from typing import Optional  # (Ici pas indispensable, conservé si extension future)

# ---------------------------------------------------------------------------
#                          Phase d'import robuste
# ---------------------------------------------------------------------------
# 1. Tentative d'import "propre" via l'espace de noms du package.
# 2. En cas d'échec (exécution directe hors contexte package), insertion de la
#    racine du projet dans sys.path puis nouvel import.
# ---------------------------------------------------------------------------
try:
    from digital_twin.back_end.mesh.fret_mesh import (
        gerar_malha_trastes,
        inject_into_config,
    )
    from digital_twin.back_end import config as dt_config
except Exception:
    import sys
    from pathlib import Path
    this_file = Path(__file__).resolve()
    mesh_dir = this_file.parent                # .../back_end/mesh
    back_end_dir = mesh_dir.parent             # .../back_end
    project_root = back_end_dir.parent.parent  # racine (contient digital_twin)
    if str(project_root) not in sys.path:      # Ajout sécurisé (idempotent)
        sys.path.insert(0, str(project_root))
    from digital_twin.back_end.mesh.fret_mesh import (  # type: ignore
        gerar_malha_trastes,
        inject_into_config,
    )
    from digital_twin.back_end import config as dt_config  # type: ignore

# ---------------------------------------------------------------------------
#                     Fonction d'affichage principal
# ---------------------------------------------------------------------------
# Paramètres
#   res : FretMeshResult (structure définie dans fret_mesh)
#   premiers_dx : int -> nombre de premières longueurs élémentaires à montrer
# ---------------------------------------------------------------------------
def _print_result(res, premiers_dx: int):
    # --- Tableau des frettes ---
    print("\n=== FRETTES ===")
    print(f"Nombre total de frettes considérées : {res.total_trastes}")
    print("n   | position (mm) | Δ depuis précédente (mm)")
    for ntr, pos_raw, inc_raw in res.trastes:
        print(f"{ntr:>3} | {pos_raw:>11.2f} | {inc_raw:>18.2f}")

    # --- Statistiques globales sur les dx ---
    dxs = res.dxs_mm
    total_elems = len(dxs)
    media = sum(dxs) / total_elems if total_elems else 0.0
    uniq = sorted(set(dxs))

    print("\n=== RÉSUMÉ GÉNÉRAL ===")
    print(f"Longueur totale (segments jusqu'à la frette {res.total_trastes}) : {res.soma_teorica_mm:.2f} mm")
    print(f"Cible dx ~ {res.dx_alvo_mm} mm")
    print(f"Nombre total d'éléments : {total_elems}")
    print(f"dx moyen : {media:.3f} mm")
    print(f"Valeurs uniques de dx ({len(uniq)}) : {', '.join(f'{u:.2f}' for u in uniq)}")
    if premiers_dx:
        print(f"Premiers {premiers_dx} dx : {[round(v,2) for v in dxs[:premiers_dx]]}")

    # --- Détails par segment (entre deux frettes consécutives) ---
    print("\n=== SEGMENTS (ENTRE FRETTES) ===")
    header_seg = "Seg | L_seg(mm) | n_elem | dx(mm) | L_recon (mm) |  Erreur(mm)  | Indices [début:fin)"
    print(header_seg)
    print('-' * len(header_seg))
    for seg in res.segmentos:
        s = seg.segmento
        Ls = seg.comprimento_mm
        n_el = seg.n_elementos
        dxv = round(seg.dx_mm, 2)
        L_recon = n_el * dxv
        err = seg.erro_mm
        ini, fim = seg.range_indices
        print(f"{s:>3} |  {Ls:>8.2f} | {n_el:>6} | {dxv:>6.2f} |  {L_recon:>11.2f} | {err:>12.8f} | [{ini}:{fim}]")

    # --- Cohérence globale (erreurs agrégées) ---
    print("\n=== COHÉRENCE ===")
    print(f"∑ Théorique (segments) : {res.soma_teorica_mm:.2f} mm")
    print(f"∑ Reconstruit (∑ dx) : {res.soma_recon_mm:.2f} mm")
    print(f"Différence (théorique - reconstruit) : {res.diferenca_total_mm:.4f} mm")
    print(f"Erreur maximale de segment : {res.erro_max_seg_mm:.4f} mm")
    print(f"Erreur moyenne absolue par segment : {res.erro_medio_abs_seg_mm:.4f} mm")
    # Calcul de l'erreur totale signée et de la somme absolue (pas stockées dans FretMeshResult)
    err_tot_sig = sum(seg.erro_mm for seg in res.segmentos)
    err_tot_abs = sum(abs(seg.erro_mm) for seg in res.segmentos)
    print(f"Erreur totale signée (∑ erreurs) : {err_tot_sig:.4f} mm")
    print(f"Erreur totale absolue (∑ |erreurs|) : {err_tot_abs:.4f} mm")

    print("\n=== FIN ===")


if __name__ == "__main__":
    # Paramètres ajustables (ligne de commande simple possible en extension future)
    total_trastes = 12              # Nombre de frettes considérées (intervals générés)
    escala_total_mm = 650.0         # Longueur de référence (mm) de l'échelle complète
    dx_alvo_mm = 6.0                # Pas cible moyen visé (mm)
    afficher_detail = True          # Imprimer distribution groupée des dx
    premiers_dx = 12                # Nombre de premières valeurs dx à montrer (aperçu)
    try:
        from digital_twin.back_end import config as _cfg  # type: ignore
        expoente = _cfg.FRET_EXPOSANT_GEOMETRIQUE
    except Exception as exc:               # pragma: no cover
        raise RuntimeError("[ERROR] FRET_EXPOSANT_GEOMETRIQUE manquant dans config ou import de config échoué") from exc
    # Exposant géométrique centralisé (config.FRET_EXPOSANT_GEOMETRIQUE)
    injecter_dans_config = True     # Si True : applique FRET_* et override core dans config

    # Génération du résultat de maillage (structure FretMeshResult)
    res = generer_maille_frettes(
        escala_total_mm,            # Longueur d'échelle totale mm
        total_trastes,              # Nombre de frettes
        dx_alvo_mm=dx_alvo_mm,      # Pas cible
        minimo_por_segmento=1,      # Au moins 1 élément par segment
        expoente=expoente,          # Exposant de division centralisé
    )

    if injecter_dans_config:
        # Injection runtime : modifie N_NODES, N_ELEMS, DX, L (override_core=True)
        inject_into_config(dt_config, res, override_core=True)
        print("[INFO] Attributs FRET_* injectés et noyau (N_NODES, DX, L, etc.) mis à jour.")

    _print_result(res, premiers_dx=premiers_dx)
