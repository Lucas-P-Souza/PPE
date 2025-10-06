from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QPushButton
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal
import math
from corde_widget import CordeWidget
try:
    # integração com o back-end (controlador de simulação)
    from digital_twin.back_end.integration import SimulationController, SimSettings  # type: ignore
except Exception:
    # fallback quando executado diretamente sem backend completo
    class SimSettings:  # minimal stub
        def __init__(self):
            pass

    class SimulationController:  # minimal stub que não faz nada
        def __init__(self, *_args, **_kw):
            self.on_started = None
            self.on_finished = None
            self._running = False
        def trigger(self, **_kw):  # simula rápido
            if self._running:
                return
            self._running = True
            if callable(self.on_started):
                try: self.on_started()
                except Exception: pass
            # encerra rapidamente
            if callable(self.on_finished):
                try: self.on_finished("(stub) sem saída")
                except Exception: pass
            self._running = False

from settings_dialog import SettingsDialog

# cette classe gère la fenêtre principale de l'application
class MainWindow(QMainWindow):

    # ça marche pour initialiser la fenêtre principale
    def __init__(self, audio_manager=None):
        super().__init__()
        self.setWindowTitle("Digital Twin – Guitare numérique")
        self.setGeometry(300, 100, 1200, 900)

        # ça dit où appliquer QSS (#MainWindow)
        self.setObjectName("MainWindow")
        self.audio_manager = audio_manager

        # ça dit quel est le titre (utilise QSS #TitleLabel)
        self.title_label = QLabel("Guitare numérique")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)

        # bouton de paramètres (coin supérieur droit)
        self.settings_button = QPushButton("⚙")
        self.settings_button.setObjectName("SettingsButton")
        self.settings_button.setCursor(Qt.PointingHandCursor)
        # ouvre le dialogue des paramètres (style carte)
        self.settings_button.clicked.connect(self._open_settings_dialog)

        # ça crée le widget de la corde à l'intérieur d'un conteneur centré
        self.corde_widget = CordeWidget()
        self.corde_widget.setMinimumHeight(200)  # hauteur minimum du widget
        self.corde_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # permet d'étendre horizontalement mais pas verticalement
        # clique/touche sur la corde -> déclenche simulation avec positions en pourcentage
        def _on_click(pos_norm: float):
            try:
                note_pct = float(self.corde_widget.cursor_pos) * 100.0
                pluck_pct = float(self.corde_widget.strike_cursor_pos) * 100.0
                # Marque excitation et démarre simulation
                self._last_pluck = "✔"
                self._update_cursor_status()
                self._pluck_timer.start(500)
                self.sim_ctrl.trigger(
                    note_cursor_pos_percent=note_pct,
                    pluck_cursor_pos_percent=pluck_pct,
                    excited=True,
                )
                # --- feedback terminal ---
                print(f"Clique détecté: Curseur Nota = {note_pct:.1f}%, Curseur Pincer = {pluck_pct:.1f}%")
            except Exception:
                pass
        try:
            self.corde_widget.clickedAt.connect(_on_click)  # type: ignore[attr-defined]
        except Exception:
            pass

        # ça crée le conteneur centré qui contient la corde
        self.string_container = QWidget()
        self.string_container.setObjectName("StringContainer")
        self.string_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.string_container.setMaximumWidth(1400)

        # création du layout interne pour le conteneur de la corde
        inner_v = QVBoxLayout()
        inner_v.setContentsMargins(16, 12, 16, 12)  # marges internes
        inner_v.addWidget(self.corde_widget)  # ajoute le widget de la corde au layout vertical
        self.string_container.setLayout(inner_v)  # définit le layout du conteneur de la corde

        # informations (utilise QSS #InfoLabel)
        self.info_label = QLabel("1..9,0,-,= → cases 1..12  |  Flèches : 1 % par pas  |  Shift + ←/→ : frettes 0..12  |  Ctrl + ←/→ : cases 0..12 (milieu)  |  ↑ gauche • ↓ droite  |  A/D ou Num4/6 : curseur d’attaque  |  Espace : pincer")
        self.info_label.setObjectName("InfoLabel")
        self.info_label.setAlignment(Qt.AlignCenter)
        # style via QSS (voir main_style.qss)

        # ici on définit le label de statut (utilise QSS #StatusLabel)
        # c'est ce qu'on utilise pour montrer la position des curseurs et l'état du pincement
        self.status_label = QLabel("Curseur: 0%  |  Attaque: 0%  |  Pincer: —")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        self._last_pluck = None  # garde l'état du dernier pincement
        self._pluck_timer = QTimer(self)
        self._pluck_timer.setSingleShot(True)
        self._pluck_timer.timeout.connect(self._clear_pluck)

        # ça crée le layout principal vertical avec centralisation horizontale du conteneur
        layout = QVBoxLayout()
        # barre supérieure avec titre centré et bouton à droite
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(12, 8, 12, 8)
        # espaceur gauche pour garder le titre parfaitement centré
        self._left_spacer = QWidget()
        self._left_spacer.setFixedSize(44, 44)
        self.settings_button.setFixedSize(44, 44)
        top_bar.addWidget(self._left_spacer, 0, Qt.AlignLeft)
        top_bar.addWidget(self.title_label, 1, Qt.AlignHCenter)
        top_bar.addWidget(self.settings_button, 0, Qt.AlignRight)
        layout.addLayout(top_bar)
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.string_container, 0, Qt.AlignHCenter)
        hbox.addStretch(1)
        layout.addLayout(hbox)
        layout.addWidget(self.info_label)
        layout.addWidget(self.status_label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        # ensure status bar is created in the GUI thread
        self.statusBar()

        # ça permet de capturer les événements clavier dans la fenêtre principale
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        # gestion du mouvement continu avec les flèches gauche/droite
        self._arrow_timer = QTimer(self)
        self._arrow_timer.setInterval(33)  # ~30 FPS
        self._arrow_timer.timeout.connect(self._on_arrow_tick)
        self._arrow_direction = 0  # -1 = gauche, +1 = droite
        self._pressed = set()  # ensemble des touches actuellement enfoncées (flèches)

        # --- pont de sinais para respeitar o thread do Qt ---
        class BackendBridge(QObject):
            simStarted = pyqtSignal()
            simFinished = pyqtSignal(str)

        self._bridge = BackendBridge(self)
        self._bridge.simStarted.connect(self._on_sim_started)
        self._bridge.simFinished.connect(self._on_sim_finished)

        # --- contrôleur de simulation du back-end ---
        # paramètres par défaut cohérents avec le back-end
        sim_settings = SimSettings()
        # (optionnel) on pourrait exposer un Dialogue pour changer ces valeurs
        self.sim_ctrl = SimulationController(sim_settings)
        # callbacks usam sinais (thread-safe)
        self.sim_ctrl.on_started = lambda: self._bridge.simStarted.emit()
        self.sim_ctrl.on_finished = lambda out: self._bridge.simFinished.emit(str(out))

    def _open_settings_dialog(self):
        # ouvre le dialogue des paramètres sous forme de "carte" centrée
        dlg = SettingsDialog(self)
        # centre par rapport à la fenêtre principale
        if self.isVisible():
            geo = self.geometry()
            dlg.move(geo.center() - dlg.rect().center())
        dlg.exec_()

    # ça gère les événements clavier pour interagir avec la corde
    #    1..9,0,-,= → cases 1..12
    #    flèches : 1 % par pas
    #    Shift + ←/→ : frettes 0..12
    #    Ctrl + ←/→ : cases 0..12 (milieu)
    #    ↑ vers la frette de gauche • ↓ vers la frette de droite

    def _raw_f_from_pos(self, pos: float) -> float:
        # retourne la position de la frette "brute" (0..12) à partir de la position normalisée (0..1)
        return 12 * math.log2(1.0 / max(1e-9, (1.0 - pos)))

    def _clamp_line(self, line: int) -> int:
        # retourne la frette dans la plage 0..12
        return max(0, min(12, int(line)))

    def _current_house(self) -> int:
        # retourne la maison (0..12) la plus proche de la position actuelle du curseur
        pos = self.corde_widget.cursor_pos
        raw_f = self._raw_f_from_pos(pos)
        if raw_f <= 0:
            return 0
        return max(1, min(12, int(round(raw_f))))

    def _step_fret_line(self, direction: int) -> None:
        # parcoure les lignes de frettes (0..12) le long de toute la corde
        pos = self.corde_widget.cursor_pos
        raw_f = self._raw_f_from_pos(pos)
        eps = 1e-6
        if direction > 0:
            # Floor avec tolérance : si légèrement en dessous d'un entier, traite comme entier
            target_line = int(math.floor(raw_f + eps)) + 1
        else:
            # Ceil avec tolérance : si légèrement au-dessus d'un entier, traite comme entier
            target_line = int(math.ceil(raw_f - eps)) - 1
        target_line = self._clamp_line(target_line)
        self.corde_widget.set_cursor_fret_line(target_line)
        self._update_cursor_status()

    def _step_house_center(self, direction: int) -> None:
        # parcourt les centres des maisons (0..12) le long de toute la corde
        pos = self.corde_widget.cursor_pos
        # construire liste de centres : maison 0 (corde à vide) + maisons 1..12
        centers = [0.0] + [self.corde_widget.house_center_frac(h) for h in range(1, 13)]
        eps = 1e-6

        if direction > 0:
            # prochain centre strictement à droite du curseur
            next_idx = None
            for i, c in enumerate(centers):
                if c > pos + eps:
                    next_idx = i
                    break
            if next_idx is None:
                next_idx = 12
            self.corde_widget.set_cursor_house_center(next_idx)
            self._update_cursor_status(house=next_idx)
        else:
            # précédent centre strictement à gauche du curseur
            prev_idx = None
            for i in range(12, -1, -1):
                if centers[i] < pos - eps:
                    prev_idx = i
                    break
            if prev_idx is None:
                prev_idx = 0
            self.corde_widget.set_cursor_house_center(prev_idx)
            self._update_cursor_status(house=prev_idx)

    def _snap_left_line(self) -> None:
        # snap à la frette de gauche
        pos = self.corde_widget.cursor_pos
        raw_f = self._raw_f_from_pos(pos)
        target_line = self._clamp_line(int(math.ceil(raw_f)) - 1)
        self.corde_widget.set_cursor_fret_line(target_line)
        self._update_cursor_status()

    def _snap_right_line(self) -> None:
        # snap à la frette de droite
        pos = self.corde_widget.cursor_pos
        raw_f = self._raw_f_from_pos(pos)
        target_line = self._clamp_line(int(math.floor(raw_f)) + 1)
        self.corde_widget.set_cursor_fret_line(target_line)
        self._update_cursor_status()

    def keyPressEvent(self, event):
        # gère les événements clavier pour interagir avec la corde
        key = event.text().upper()

        qt_key = event.key()

        # navigation par flèches : continu (timer) | Shift : sauter frette | Ctrl : sauter maison (centre)
        if qt_key in (Qt.Key_Right, Qt.Key_Left):
            mods = event.modifiers()
            if mods & Qt.ShiftModifier:
                self._step_fret_line(direction=1 if qt_key == Qt.Key_Right else -1)
                return
            if mods & Qt.ControlModifier:
                self._step_house_center(direction=1 if qt_key == Qt.Key_Right else -1)
                return
            else:
                # ignorer l'auto-répétition, on gère via un timer
                if event.isAutoRepeat():
                    return
                # enregistrer la touche enfoncée et démarrer/ajuster le timer
                self._pressed.add(qt_key)
                left = Qt.Key_Left in self._pressed
                right = Qt.Key_Right in self._pressed
                new_dir = 0
                if right and not left:
                    new_dir = 1
                elif left and not right:
                    new_dir = -1
                self._arrow_direction = new_dir
                if new_dir != 0 and not self._arrow_timer.isActive():
                    self._arrow_timer.start()
                if new_dir == 0 and self._arrow_timer.isActive():
                    self._arrow_timer.stop()
                return

    # magnétisation pour lignes de frette: ↑ frette de gauche, ↓ frette de droite
        if qt_key in (Qt.Key_Up, Qt.Key_Down):
            try:
                if qt_key == Qt.Key_Up:
                    self._snap_left_line()
                else:
                    self._snap_right_line()
                return
            except Exception:
                pass

    # ligne supérieure: 1..9,0,-,=  -> cases 1..12
        mapping = {
            "'": 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, '0': 10, '-': 11, '=': 12
        }
        # ignore les chiffres du pavé numérique pour le mapping des maisons
        if key in mapping and not (event.modifiers() & Qt.KeypadModifier):
            house = mapping[key]
            # ici nous positionnons le curseur exactement au milieu de la maison (entre les frettes N-1 et N)
            self.corde_widget.set_cursor_house_center(house)
            self._update_cursor_status(house=house)
            return

    # second curseur (attaque/strike): A/D ou pavé numérique 4/6
        if (qt_key in (Qt.Key_A, Qt.Key_D)) or ((qt_key in (Qt.Key_4, Qt.Key_6)) and (event.modifiers() & Qt.KeypadModifier)):
            # A/D toujours; 4/6 uniquement avec le modificateur du pavé numérique
            if qt_key == Qt.Key_A or qt_key == Qt.Key_4:
                self.corde_widget.move_strike_cursor(-0.01)
            elif qt_key == Qt.Key_D or qt_key == Qt.Key_6:
                self.corde_widget.move_strike_cursor(0.01)
            self._update_cursor_status()
            return

        # espace: signale "pincer/attaque"
        if qt_key == Qt.Key_Space:
            self._last_pluck = "✔"
            self._update_cursor_status()
            # démarrer un timer pour effacer l'état de pincer après 500 ms
            self._pluck_timer.start(500)
            # --- déclenche aussi la simulation back-end ---
            try:
                note_pct = float(self.corde_widget.cursor_pos) * 100.0
                pluck_pct = float(self.corde_widget.strike_cursor_pos) * 100.0
                # lança simulação em background; ignora se já houver uma em curso
                self.sim_ctrl.trigger(
                    note_cursor_pos_percent=note_pct,
                    pluck_cursor_pos_percent=pluck_pct,
                    excited=True,
                )
                # --- feedback terminal ---
                print(f"Espace détecté: Curseur Note = {note_pct:.1f}%, Curseur Pincer = {pluck_pct:.1f}%")
            except Exception:
                pass
            return

        if qt_key == Qt.Key_Escape:
            self.showNormal()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        # gère les événements de relâchement des touches
        qt_key = event.key()
        # ignorer l'auto-répétition
        if event.isAutoRepeat():
            return
        # mettre à jour l'état des flèches pour le timer de mouvement
        if qt_key in (Qt.Key_Left, Qt.Key_Right):
            if qt_key in self._pressed:
                self._pressed.remove(qt_key)
            left = Qt.Key_Left in self._pressed
            right = Qt.Key_Right in self._pressed
            new_dir = 0
            if right and not left:
                new_dir = 1
            elif left and not right:
                new_dir = -1
            self._arrow_direction = new_dir
            if new_dir == 0 and self._arrow_timer.isActive():
                self._arrow_timer.stop()
            elif new_dir != 0 and not self._arrow_timer.isActive():
                self._arrow_timer.start()
            return
        super().keyReleaseEvent(event)

    def _update_cursor_status(self, house: int | None = None):
        # met à jour le label de statut avec les deux positions de curseur et l'état de pincement
        pos = self.corde_widget.cursor_pos
        s_pos = self.corde_widget.strike_cursor_pos
        nearest = None
        try:
            if 0.0 <= pos < 1.0:
                raw_f = 12 * math.log2(1.0 / max(1e-6, (1.0 - pos)))
                nearest_line = max(0, min(12, int(round(raw_f))))
                nearest = 0 if nearest_line <= 0 else nearest_line
        except Exception:
            pass
        curseur_txt = None
        if house is not None:
            curseur_txt = f"Curseur {house}"
        elif nearest is not None:
            curseur_txt = f"Curseur: {int(pos*100)}% (≈ {nearest})"
        else:
            curseur_txt = f"Curseur: {int(pos*100)}%"

        attaque_txt = f"Attaque: {int(s_pos*100)}%"
        pluck_txt = f"Pincer: {self._last_pluck or '—'}"
        self.status_label.setText(f"{curseur_txt}  |  {attaque_txt}  |  {pluck_txt}")

    def _clear_pluck(self):
        # remet l'état de pincer à inactif
        self._last_pluck = None
        self._update_cursor_status()

    def _on_arrow_tick(self):
        # applique un déplacement de 1 % à chaque tick dans la direction en cours
        if self._arrow_direction == 0:
            return
        step = 0.01 * (1 if self._arrow_direction > 0 else -1)
        self.corde_widget.move_cursor(step)
        self._update_cursor_status()

    # --- slots Qt para atualizar UI no thread principal ---
    def _on_sim_started(self) -> None:
        try:
            self.statusBar().showMessage("Simulação iniciada…", 2000)
        except Exception:
            pass

    def _on_sim_finished(self, out_csv_path: str) -> None:
        try:
            self.statusBar().showMessage(f"Simulação concluída: {out_csv_path}", 5000)
        except Exception:
            pass

