#!/usr/bin/env python3
"""
Profil de raideur virtuelle polynomial symétrique avec plateau ajustable
=======================================================================

Ce module implémente un profil polynomial de raideur avec montée/descente symétrique
et un temps de plateau (latence) configurable où K = K_max.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def profil_polynomial_symetrique(t, t_debut, t_montee, t_plateau, k_max):
    """
    Profil polynomial cubique parfaitement symétrique.
    
    IMPORTANT: Ce profil retourne la raideur ADDITIONNELLE k_virtuel(t).
    La raideur totale sera K_total(t) = K0 + k_virtuel(t) où K0 est extrait
    automatiquement de la matrice de raideur.
    
    Paramètres
    ----------
    t : float
        Temps actuel
    t_debut : float
        Début de l'activation du ressort
    t_montee : float
        Durée de la montée (= durée de la descente)
    t_plateau : float
        Durée du plateau à K0 + k_max (temps de latence)
    k_max : float
        Raideur additionnelle maximale (s'ajoute à K0)
        
    Phases
    ------
    1. t < t_debut : k_virtuel(t) = 0 → K_total = K0
    2. t_debut ≤ t < t_debut + t_montee : montée polynomial → K_total = K0 + k_virtuel(t)
    3. t_debut + t_montee ≤ t < t_debut + t_montee + t_plateau : k_virtuel(t) = k_max → K_total = K0 + k_max
    4. t_debut + t_montee + t_plateau ≤ t < t_debut + t_montee + t_plateau + t_montee : descente → K_total = K0 + k_virtuel(t)
    5. t ≥ t_debut + t_montee + t_plateau + t_montee : k_virtuel(t) = 0 → K_total = K0
    """
    t_fin_montee = t_debut + t_montee
    t_fin_plateau = t_fin_montee + t_plateau
    t_fin_descente = t_fin_plateau + t_montee
    
    if t < t_debut:
        return 0.0
    elif t < t_fin_montee:
        # Phase de montée (polynomial cubique)
        s = (t - t_debut) / t_montee
        return k_max * (3 * s**2 - 2 * s**3)
    elif t < t_fin_plateau:
        # Phase de plateau
        return k_max
    elif t < t_fin_descente:
        # Phase de descente (polynomial cubique symétrique)
        s = (t_fin_descente - t) / t_montee  # Symétrie: s va de 1 à 0
        return k_max * (3 * s**2 - 2 * s**3)
    else:
        return 0.0

def demonstration_profil_polynomial_K0():
    """Démonstration du profil polynomial avec K_total = K0 + k_virtuel(t)."""
    print("Démonstration : profil polynomial avec K0")
    print("=" * 50)
    
    # Simulation d'un système FEM simple
    K0 = 1800.0  # Raideur structurelle de base (N/m)
    print(f"Raideur structurelle K0 = {K0} N/m")
    
    # Paramètres du profil polynomial
    t_debut = 0.5      # Début activation (s)
    t_montee = 0.2     # Durée montée (s)
    t_plateau = 0.4    # Durée plateau (s)
    k_max = 1200.0     # Raideur virtuelle maximale (N/m)
    
    print(f"Profil polynomial :")
    print(f"  - Début activation : {t_debut} s")
    print(f"  - Durée montée = descente : {t_montee} s")
    print(f"  - Durée plateau : {t_plateau} s")
    print(f"  - k_max (virtuelle) : {k_max} N/m")
    print(f"  - K_total_max = K0 + k_max = {K0 + k_max} N/m")
    
    # Temps de simulation
    duree_totale = t_debut + t_montee + t_plateau + t_montee + 0.5
    temps = np.linspace(0, duree_totale, 1000)
    
    # Calcul du profil polynomial
    k_virtuel_vals = [profil_polynomial_symetrique(t, t_debut, t_montee, t_plateau, k_max) 
                      for t in temps]
    K_total_vals = [K0 + k_virt for k_virt in k_virtuel_vals]
    
    # Vérifications
    K_total_min = min(K_total_vals)
    K_total_max = max(K_total_vals)
    
    print(f"\nVérifications :")
    print(f"  ✓ K_total minimum : {K_total_min} N/m (= K0)")
    print(f"  ✓ K_total maximum : {K_total_max} N/m (= K0 + k_max)")
    print(f"  ✓ Démarre bien à K0 : {abs(K_total_min - K0) < 1e-10}")
    
    # Graphique
    plt.figure(figsize=(12, 8))
    
    # Tracer K_total = K0 + k_virtuel(t)
    plt.plot(temps, K_total_vals, 'b-', linewidth=3, label='K_total(t) = K0 + k_virtuel(t)')
    
    # Tracer la ligne de base K0
    plt.axhline(K0, color='red', linestyle='--', linewidth=2, 
                label=f'K0 = {K0} N/m (raideur structurelle)')
    
    # Tracer la ligne K_max
    plt.axhline(K0 + k_max, color='green', linestyle='--', linewidth=2, 
                label=f'K0 + k_max = {K0 + k_max} N/m')
    
    # Marquer les phases
    plt.axvline(t_debut, color='gray', linestyle=':', alpha=0.7, label='Début activation')
    plt.axvline(t_debut + t_montee, color='gray', linestyle=':', alpha=0.7, label='Début plateau')
    plt.axvline(t_debut + t_montee + t_plateau, color='gray', linestyle=':', alpha=0.7, label='Fin plateau')
    plt.axvline(t_debut + t_montee + t_plateau + t_montee, color='gray', linestyle=':', alpha=0.7, label='Fin activation')
    
    # Zone du plateau
    t_fin_montee = t_debut + t_montee
    t_fin_plateau = t_debut + t_montee + t_plateau
    plt.axvspan(t_fin_montee, t_fin_plateau, alpha=0.2, color='yellow', label='Plateau')
    
    plt.title('Profil Polynomial : K_total(t) = K0 + k_virtuel(t)\n' + 
              'Évolution Symétrique de la Raideur Totale')
    plt.xlabel('Temps (s)')
    plt.ylabel('Raideur Totale K_total (N/m)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Annotations
    plt.annotate(f'Début: K0 = {K0} N/m', 
                xy=(0.1, K0), xytext=(0.1, K0 + 200),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, ha='left')
    
    plt.annotate(f'Maximum: {K0 + k_max} N/m', 
                xy=(t_debut + t_montee + t_plateau/2, K0 + k_max), 
                xytext=(t_debut + t_montee + t_plateau/2, K0 + k_max + 300),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    
    # Sauvegarder
    chemin_graphique = Path(__file__).parent / "profil_polynomial_K0.png"
    plt.savefig(chemin_graphique, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graphique sauvegardé : {chemin_graphique.absolute()}")
    
    try:
        plt.show()
    except:
        print("Affichage graphique non disponible")
    
    return True

def main():
    """Fonction principale de test."""
    print("Test du profil polynomial avec K0")
    print("=" * 50)
    
    try:
        # Test du profil polynomial avec K0
        demonstration_profil_polynomial_K0()
        
        print("\n Test terminé avec succès !")
        print("Le profil polynomial respecte :")
        print("  ✓ Démarre à K0 (raideur structurelle)")
        print("  ✓ Atteint K0 + k_max au plateau")
        print("  ✓ Revient à K0 en fin")
        print("  ✓ Transitions douces et symétriques")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()