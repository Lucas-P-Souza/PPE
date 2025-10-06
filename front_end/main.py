import sys
import os
# Ensure repository root is on sys.path so 'digital_twin' package can be imported
# We need the parent directory that contains the 'digital_twin' package on sys.path
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_here))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from PyQt5.QtWidgets import QApplication
from gui import MainWindow

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

