"""
Génération d'une maille non uniforme basée sur les frettes.

Objectif principal
------------------
Proposer une discrétisation (maillage) adaptée aux intervalles réels entre frettes
pour une corde instrumentale. Chaque segment entre deux frettes successives est
maillé par un nombre entier d'éléments de longueur (dx) quasi uniforme, en visant
un pas cible (dx_alvo_mm) sous contraintes minimales et maximales.

Fonctionnalités offertes
------------------------
1. Calcul analytique des positions de frettes selon une loi géométrique (12-TET
    standard ou exposant expérimental personnalisé).
2. Discrétisation indépendante de chaque segment : sélection du nombre d'éléments
    optimisant (a) la reconstruction de la longueur réelle et (b) la proximité du
    pas dx cible tout en respectant bornes dx_min / dx_max.
3. Agrégation : construction du vecteur global des dx et des positions nodales
    cumulées (incluant l'origine).
4. Injection optionnelle de variables dérivées dans le module de configuration
    afin de rendre la maille disponible ailleurs dans l'application sans modifier
    le fichier de config sur disque.

Conception / Décisions
----------------------
- Aucun arrondi n'est fait sur la somme finale des dx en dehors du choix local
  de dx arrondi à 2 décimales par segment, garantissant traçabilité.
- Une stratégie de recherche élargie est utilisée si l'erreur de reconstruction
  dépasse une tolérance locale (tolerancia_erro_seg).
- Les noms de variables d'origine (en portugais) sont conservés pour limiter
  l'impact sur d'autres modules déjà traduits partiellement.

NOTE : On évite toute écriture physique dans `config.py`; on injecte uniquement
         des attributs au runtime afin de préserver l'historique de version.
"""

from __future__ import annotations
from dataclasses import dataclass               # Pour créer des conteneurs de données immuables légers
from typing import List, Sequence, Tuple, Dict  # Types pour lisibilité et aide d'analyse statique
import math                                     # Fonctions mathématiques (puissances, arrondis auxiliaires)
from digital_twin.back_end import config as _dt_cfg  # type: ignore
_FRET_EXP_DEFAULT = _dt_cfg.FRET_EXPOSANT_GEOMETRIQUE
import json
from pathlib import Path

# ---------------------------------------------------------------------------
#                         Structures de données
# ---------------------------------------------------------------------------
@dataclass
class SegmentInfo:
    """
    Métadonnées d'un segment (entre deux frettes consécutives).

    Attributes
    ----------
    segmento : int
        Index (1-based) du segment (1 = entre sillet et 1ère frette, etc.).
    comprimento_mm : float
        Longueur réelle du segment en millimètres.
    n_elementos : int
        Nombre d'éléments choisis pour ce segment.
    dx_mm : float
        Longueur élémentaire retenue (arrondie à 2 décimales) pour ce segment.
    range_indices : tuple[int, int]
        Intervalle [début, fin) dans le vecteur global des dx où résident
        les éléments de ce segment (style slicing Python).
    erro_mm : float
        Erreur (longueur_réelle - n_elementos * dx_mm_arrondi).
    """
    segmento: int
    comprimento_mm: float
    n_elementos: int
    dx_mm: float
    range_indices: tuple[int, int]
    erro_mm: float

@dataclass
class FretMeshResult:
    # Conteneur de sortie agrégé pour la maille de frettes.
    escala_total_mm: float                   # Longueur d'échelle totale de la corde
    total_trastes: int                       # Nombre de frettes calculées
    dx_alvo_mm: float                        # Pas cible souhaité (approx.)
    trastes: List[tuple[int, float, float]]  # Liste (n, position_cumulative_mm, intervalle_mm)
    dxs_mm: List[float]                      # Vecteur global des pas élémentaires
    segmentos: List[SegmentInfo]             # Métadonnées par segment
    soma_teorica_mm: float                   # Somme des longueurs théoriques (somme des intervalles)
    soma_recon_mm: float                     # Somme reconstruite (somme des dx arrondis)
    diferenca_total_mm: float                # Différence (théorique - reconstruite)
    erro_max_seg_mm: float                   # Erreur absolue maximale segmentaire
    erro_medio_abs_seg_mm: float             # Erreur absolue moyenne segmentaire
    node_positions_mm: List[float]           # Positions nodales cumulées (0 inclus)
    
    # --- Méthodes utilitaires de sérialisation ---
    def to_dict(self) -> dict:
        return {
            "escala_total_mm": self.escala_total_mm,
            "total_trastes": self.total_trastes,
            "dx_alvo_mm": self.dx_alvo_mm,
            "trastes": self.trastes,
            "dxs_mm": self.dxs_mm,
            "segmentos": [
                {
                    "segmento": s.segmento,
                    "comprimento_mm": s.comprimento_mm,
                    "n_elementos": s.n_elementos,
                    "dx_mm": s.dx_mm,
                    "range_indices": list(s.range_indices),
                    "erro_mm": s.erro_mm,
                }
                for s in self.segmentos
            ],
            "soma_teorica_mm": self.soma_teorica_mm,
            "soma_recon_mm": self.soma_recon_mm,
            "diferenca_total_mm": self.diferenca_total_mm,
            "erro_max_seg_mm": self.erro_max_seg_mm,
            "erro_medio_abs_seg_mm": self.erro_medio_abs_seg_mm,
            "node_positions_mm": self.node_positions_mm,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FretMeshResult":
        segs = [
            SegmentInfo(
                segmento=sd["segmento"],
                comprimento_mm=sd["comprimento_mm"],
                n_elementos=sd["n_elementos"],
                dx_mm=sd["dx_mm"],
                range_indices=tuple(sd["range_indices"]),
                erro_mm=sd["erro_mm"],
            )
            for sd in data.get("segmentos", [])
        ]
        return cls(
            escala_total_mm=data["escala_total_mm"],
            total_trastes=data["total_trastes"],
            dx_alvo_mm=data["dx_alvo_mm"],
            trastes=[tuple(t) for t in data["trastes"]],
            dxs_mm=data["dxs_mm"],
            segmentos=segs,
            soma_teorica_mm=data["soma_teorica_mm"],
            soma_recon_mm=data["soma_recon_mm"],
            diferenca_total_mm=data["diferenca_total_mm"],
            erro_max_seg_mm=data["erro_max_seg_mm"],
            erro_medio_abs_seg_mm=data["erro_medio_abs_seg_mm"],
            node_positions_mm=data["node_positions_mm"],
        )


def save_fret_mesh_json(result: FretMeshResult, filepath: str | Path) -> Path:
    """Sauvegarde la maille de frettes en JSON."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    return p


def load_fret_mesh_json(filepath: str | Path) -> FretMeshResult:
    """Charge une maille de frettes depuis un JSON."""
    p = Path(filepath)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return FretMeshResult.from_dict(data)

# ---------------------------------------------------------------------------
#                           Calcul des frettes
# ---------------------------------------------------------------------------

def calcular_trastes(escala_mm: float, total_trastes: int, expoente: float = 12.0) -> List[Tuple[int, float, float]]:
    """
    Calcule les positions cumulées et les intervalles entre frettes.

    Paramètres
    ----------
    escala_mm : float
        Longueur totale de référence (échelle) en mm.
    total_trastes : int
        Nombre de frettes à générer.
    expoente : float, défaut 12.0
        Exposant du diviseur géométrique (12 => division en 12 sous-intervalles égaux de l'octave expérimentale).

    Retour
    ------
    List[Tuple[int, float, float]]
        Liste ordonnée de tuples (n, position_cumulative_mm, intervalle_mm_depuis_précédente).
    """
    # Liste des frettes (n, position_mm, intervalle_mm)
    trastes: List[Tuple[int, float, float]] = []
    anterior = 0.0
    # Calcul des positions et intervalles
    for n in range(1, total_trastes + 1):                       # Itère sur chaque frette
        posicao = escala_mm * (1 - 1 / (2 ** (n / expoente)))   # Position cumulative selon la formule géométrique
        intervalo = posicao - anterior                          # Intervalle local = différence avec la position précédente
        trastes.append((n, posicao, intervalo))                 # Ajout à la liste agrégée
        anterior = posicao                                      # Mise à jour pour prochain intervalle
    return trastes

# ---------------------------------------------------------------------------
#         Discrétisation par segment (adaptation du script original)
# ---------------------------------------------------------------------------

def _discretizar_segmento(
    L_seg: float,
    dx_alvo_mm: float,
    minimo_por_segmento: int,
    limite_busca_factor: int = 6,
    tolerancia_erro_seg: float = 0.01,
    dx_min: float = 5.0,
    dx_max: float = 8.0,
) -> Tuple[int, float]:
    if L_seg <= 0:                                                        # Segment vide / dégénéré
        return 0, 0.0                                                     # Aucun élément, dx nul
    # Estimation centrale du nombre d'éléments autour de la cible dx
    n_central = max(minimo_por_segmento, int(round(L_seg / dx_alvo_mm)))  # Estimation initiale du nombre d'éléments
    # Limites imposées par dx_max et dx_min
    n_min_dxmax = math.ceil(L_seg / dx_max) if dx_max > 0 else 1          # Min imposé par la borne supérieure dx_max
    n_max_dxmin = math.floor(L_seg / dx_min) if dx_min > 0 else 400       # Max imposé par la borne inférieure dx_min
    n_min = max(minimo_por_segmento, min(n_central, n_min_dxmax))         # Borne inférieure candidate
    if n_min > n_max_dxmin:                                               # Correction si les contraintes se croisent
        n_min = max(minimo_por_segmento, n_min_dxmax)                     # Ajuste n_min
        n_max_dxmin = max(n_min + 1, n_max_dxmin)                         # Protège contre bornes inversées
    n_max = min(400, max(n_central + 5, n_max_dxmin))                     # Borne supérieure brute
    n_max = max(n_max, n_central + 5)                                     # Garde une fenêtre de recherche minimale
    n_max = min(n_max, 400)                                               # Limite dure pour éviter explosion

    def avaliar(faixa):                                                   # Fonction interne de scoring
        # Critère d'ordre lexicographique : (erreur_longueur, proximité_dx_cible, -n, dx)
        melhor = (10 ** 9, 10 ** 9, -1, 0.0)                              # Initialisation avec valeurs sentinelles
        for n in faixa:
            # Ignore si n est en dessous du minimum requis
            if n < minimo_por_segmento:
                continue
            
            dx_r = round(L_seg / n, 2)                                    # dx candidate arrondie à 2 décimales

            # Ignore si dx_r est en dessous de 0
            if dx_r <= 0:
                continue
            
            erro_len = abs(L_seg - n * dx_r)                              # Erreur de reconstruction de longueur
            prox_alvo = abs(dx_r - dx_alvo_mm)                            # Distance au pas cible
            chave = (erro_len, prox_alvo, -n, dx_r)                       # Tuple de comparaison

            # Mise à jour si meilleur trouvé    
            if chave < melhor:
                melhor = chave
        
        return melhor                                                     # Retourne le meilleur tuple trouvé
    
    faixa1 = range(n_min, n_max + 1)                                      # Première fenêtre de recherche
    erro_len, prox_alvo, neg_n, dx_r = avaliar(faixa1)
    n_escolhido = -neg_n

    # Si l'erreur est trop grande, on élargit la recherche
    if erro_len > tolerancia_erro_seg:                                    
        faixa2 = range(max(1, n_min - 50), min(400, n_max + 50) + 1)
        erro2, prox2, neg_n2, dx_r2 = avaliar(faixa2)

        # Mise à jour si meilleur trouvé dans la seconde passe
        if (erro2, prox2, neg_n2) < (erro_len, prox_alvo, neg_n):
            erro_len, prox_alvo, neg_n, dx_r = erro2, prox2, neg_n2, dx_r2
            n_escolhido = -neg_n

    dx_final = dx_r                                                       # dx retenu (peut être réajusté ci-dessous)
    
    # Ajustements finaux pour respecter dx_min / dx_max
    if dx_final > dx_max:                                                 # Si dx dépasse la borne haute -> augmente n
        n_forcado = math.ceil(L_seg / dx_max)
        dx_forcado = round(L_seg / n_forcado, 2)

        # Ignore si n_forcado est en dessous du minimum requis
        if n_forcado >= minimo_por_segmento and dx_forcado >= dx_min * 0.5:
            n_escolhido, dx_final = n_forcado, dx_forcado

    elif dx_final < dx_min and n_escolhido > 1:                           # Si dx trop petit -> réduit n si possible
        n_forcado = max(minimo_por_segmento, math.floor(L_seg / dx_min))

        # Ignore si n_forcado est en dessous du minimum requis
        if n_forcado > 0:
            dx_forcado = round(L_seg / n_forcado, 2)

            # Ignore si dx_forcado est en dessous de la borne minimale
            if dx_min * 0.5 <= dx_forcado <= dx_max * 1.5:
                n_escolhido, dx_final = n_forcado, dx_forcado   

    return n_escolhido, dx_final                                          # Return la nouvelle valeur de n et dx

# ---------------------------------------------------------------------------
#                    Génération complète de la maille
# ---------------------------------------------------------------------------
def gerar_malha_trastes(
    escala_mm: float,
    total_trastes: int,
    dx_alvo_mm: float = 6.0,
    minimo_por_segmento: int = 1,
    expoente: float = _FRET_EXP_DEFAULT,  # Centralisé depuis config si présent
) -> FretMeshResult:
    trastes = calcular_trastes(escala_mm, total_trastes, expoente=expoente)  # Génère la géométrie des frettes
    dist_incr = [t[2] for t in trastes]                                   # Liste des intervalles successifs
    soma_teorica = sum(dist_incr)                                         # Longueur théorique couverte

    dxs: List[float] = []                                                 # Vecteur global des dx
    segmentos: List[SegmentInfo] = []                                     # Métadonnées segmentaires
    soma_segmentos_recon = 0.0                                            # Accumulateur reconstruction (diagnostic)
    max_abs_err = 0.0                                                     # Erreur absolue max rencontrée
    soma_abs_err = 0.0                                                    # Somme des erreurs absolues (pour moyenne)

    for idx_seg, L_seg in enumerate(dist_incr, start=1):                  # Parcours de chaque segment
        n_elems_seg, dx_real = _discretizar_segmento(
            L_seg, dx_alvo_mm, minimo_por_segmento
        )
        if n_elems_seg == 0:                                              # Skip si segment dégénéré
            continue
        start_index = len(dxs)                                            # Index de début dans le vecteur global
        dxs.extend([dx_real] * n_elems_seg)                               # Ajout répété de dx_real
        end_index = len(dxs)                                              # Index de fin (exclusif)
        L_recon = n_elems_seg * round(dx_real, 2)                         # Longueur reconstruite pour ce segment
        err = L_seg - L_recon                                             # Erreur signée
        abs_err = abs(err)                                                # Erreur absolue
        soma_segmentos_recon += L_recon                                   # Accumulation reconstruction globale
        soma_abs_err += abs_err                                           # Accumulation erreur absolue
        if abs_err > max_abs_err:                                         # Mise à jour du maximum global
            max_abs_err = abs_err
        segmentos.append(                                                 # Enregistrement des métadonnées du segment
            SegmentInfo(
                segmento=idx_seg,
                comprimento_mm=L_seg,
                n_elementos=n_elems_seg,
                dx_mm=dx_real,
                range_indices=(start_index, end_index),
                erro_mm=err,
            )
        )

    soma_dxs = sum(dxs)                                                   # Somme totale des dx
    diferenca_total = soma_teorica - soma_dxs                             # Différence vs théorie
    erro_medio_abs = soma_abs_err / len(segmentos) if segmentos else 0.0  # Erreur absolue moyenne

    # Construire les positions nodales cumulées (0 inclus)
    node_positions: List[float] = [0.0]                                   # Position du premier nœud (origine)
    acc = 0.0                                                             # Accumulateur de longueur
    for d in dxs:                                                         # Intégration discrète
        acc += d
        node_positions.append(acc)                                        # Ajout de la position nodale suivante

    return FretMeshResult(                                                # Emballage structuré du résultat
        escala_total_mm=escala_mm,
        total_trastes=total_trastes,
        dx_alvo_mm=dx_alvo_mm,
        trastes=trastes,
        dxs_mm=dxs,
        segmentos=segmentos,
        soma_teorica_mm=soma_teorica,
        soma_recon_mm=soma_dxs,
        diferenca_total_mm=diferenca_total,
        erro_max_seg_mm=max_abs_err,
        erro_medio_abs_seg_mm=erro_medio_abs,
        node_positions_mm=node_positions,
    )

# ---------------------------------------------------------------------------
# Intégration avec le module config au runtime
# ---------------------------------------------------------------------------

def inject_into_config(config_module, result: FretMeshResult, override_core: bool = True) -> None:
    """Injecte des attributs dérivés dans le module config.

    Paramètres
    ----------
    config_module : module
        Référence au module de configuration (ex: digital_twin.back_end.config)
    result : FretMeshResult
        Résultat de la génération de maille de frettes.
    override_core : bool (défaut True)
        Si True, met aussi à jour les constantes centrales (N_NODES, N_ELEMS, DX, L)
        pour que tout module qui importe *après* injection voie la maille frettes
        comme maille active par défaut. Si False, on garde coexistence (ancien
        comportement) et seules les variables préfixées FRET_* sont ajoutées.
    """
    # Attributs spécifiques FRET_* (toujours injectés)
    setattr(config_module, "FRETS_TABLE_MM", result.trastes)
    setattr(config_module, "FRET_NODE_POSITIONS_MM", result.node_positions_mm)
    setattr(config_module, "FRET_DXS_MM", result.dxs_mm)
    setattr(config_module, "FRET_N_ELEMS", len(result.dxs_mm))
    setattr(config_module, "FRET_N_NODES", len(result.dxs_mm) + 1)
    setattr(config_module, "FRET_TOTAL_LENGTH_MM", result.soma_recon_mm)
    setattr(config_module, "FRET_LENGTH_M", result.soma_recon_mm / 1000.0)

    if override_core:
        # Mise à jour des constantes de base pour refléter la nouvelle maille
        n_nodes = len(result.dxs_mm) + 1
        n_elems = len(result.dxs_mm)
        dx_moyen_m = (result.soma_recon_mm / 1000.0) / n_elems if n_elems else getattr(config_module, 'DX', 0.0)
        setattr(config_module, "N_NODES", n_nodes)
        setattr(config_module, "N_ELEMS", n_elems)
        setattr(config_module, "DX", dx_moyen_m)
        # Met à jour longueur L pour cohérence si différence significative (> 1e-6 m)
        L_fret_m = result.soma_recon_mm / 1000.0
        if abs(getattr(config_module, 'L', L_fret_m) - L_fret_m) > 1e-9:
            setattr(config_module, "L", L_fret_m)
        # Recalcule dérivés si présents
        if hasattr(config_module, 'COURANT_NUMBER') and hasattr(config_module, 'WAVE_SPEED') and hasattr(config_module, 'DT'):
            try:
                new_courant = getattr(config_module, 'WAVE_SPEED') * getattr(config_module, 'DT') / dx_moyen_m
                setattr(config_module, 'COURANT_NUMBER', new_courant)
            except Exception:  # pragma: no cover - protect if inconsistent values
                pass
        # Ajuste OUTPUT_NODE (ex milieu)
        setattr(config_module, 'OUTPUT_NODE', n_nodes // 2)

    # Indicateur explicite
    setattr(config_module, "FRET_OVERRIDE_CORE", override_core)


def persist_into_config_file(config_path: str | Path, result: FretMeshResult) -> Path:
    """Écrit physiquement dans config.py les valeurs N_NODES, N_ELEMS, DX et L.

    Recherche le bloc délimité par les commentaires:
      # --- FRET AUTO-MANAGED BLOCK (BEGIN) ---
      ...
      # --- FRET AUTO-MANAGED BLOCK (END) ---

    Et remplace son contenu interne par des lignes mises à jour. Conserve le reste intact.
    """
    p = Path(config_path)
    texte = p.read_text(encoding="utf-8").splitlines()
    try:
        i_begin = next(i for i, l in enumerate(texte) if "FRET AUTO-MANAGED BLOCK (BEGIN)" in l)
        i_end = next(i for i, l in enumerate(texte) if "FRET AUTO-MANAGED BLOCK (END)" in l)
    except StopIteration:
        raise RuntimeError("Blocs de gestion automatique non trouvés dans config.py")

    n_nodes = len(result.dxs_mm) + 1
    n_elems = len(result.dxs_mm)
    dx_moyen_m = (result.soma_recon_mm / 1000.0) / n_elems if n_elems else 0.0
    L_m = result.soma_recon_mm / 1000.0

    # Construir nouvelles lignes (préserve lignes begin/end)
    nouvelles = [
        texte[i_begin],
        f"N_NODES: int = {n_nodes}",
        f"N_ELEMS: int = {n_elems}",
        f"DX: float = {dx_moyen_m}",
        f"L: float = {L_m}  # Longueur mise à jour par fret mesh",  # Optionnel si on veut refléter L exact
        texte[i_end],
    ]

    # Remplacement dans le tableau original
    nouveau_contenu = texte[:i_begin] + nouvelles + texte[i_end + 1:]
    p.write_text("\n".join(nouveau_contenu) + "\n", encoding="utf-8")
    return p

__all__ = [
    "calcular_trastes",
    "gerar_malha_trastes",
    "inject_into_config",
    "save_fret_mesh_json",
    "load_fret_mesh_json",
    "persist_into_config_file",
    "FretMeshResult",
    "SegmentInfo",
]
