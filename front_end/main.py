import sys
import os
from pathlib import Path

from PyQt5.QtWidgets import QApplication
# Tentative d'import de la fenêtre principale via le package.
# Si l'exécution se fait en lançant directement ce fichier depuis
# le dossier `digital_twin`, Python ne trouve pas le package
# racine — on ajoute alors la racine du dépôt au sys.path.
try:
    from digital_twin.front_end.gui import MainWindow
except ModuleNotFoundError:
    # Exécution directe (cwd == digital_twin) — ajouter le parent du
    # dossier `digital_twin` au sys.path pour que l'import fonctionne.
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from digital_twin.front_end.gui import MainWindow

class AudioManager:
    def play_note(self, note_name):
        # pour l'instant ça ne fait rien (simulation)
        # cette fonctionnalité sera ajoutée plus tard
        print(f"AudioManager: jouer la note {note_name} (simulation)")

# c'est le main qui lance l'application
# il crée l'application et la fenêtre principale
if __name__ == '__main__':
    # Permite desativar fullscreen passando argumento --window
    fullscreen = True
    if '--window' in sys.argv:
        fullscreen = False
        sys.argv = [a for a in sys.argv if a != '--window']

    # cree l'application
    app = QApplication(sys.argv)

    # charge le stylesheet (QSS) s'il existe
    # c'est utilisé pour le thème sombre ou clair
    try:
        base_dir = os.path.dirname(__file__)
        qss_path = os.path.join(base_dir, 'styles', 'main_style.qss')
        if os.path.exists(qss_path):
            with open(qss_path, 'r', encoding='utf-8') as f:
                app.setStyleSheet(f.read())
    except Exception as e:
        # en cas d'erreur lors du chargement du QSS, continuer sans interrompre l'application
        print(f"avertissement : impossible de charger le stylesheet : {e}")
    
    # crée l'instance du gestionnaire audio (actuellement un substitut)
    audio_manager = AudioManager()
    
    # crée et affiche la fenêtre principale en plein écran
    main_window = MainWindow(audio_manager)
    if fullscreen:
        main_window.showFullScreen()
    else:
        main_window.show()
    
    # exécute la boucle d'événements de l'application
    sys.exit(app.exec_())

