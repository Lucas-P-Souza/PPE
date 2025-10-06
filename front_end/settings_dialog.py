from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QLabel, QFormLayout, QHBoxLayout,
    QPushButton, QSlider, QGraphicsDropShadowEffect, QCheckBox, QFrame, QTabWidget
)
from PyQt5.QtCore import Qt

# petit dialogue de configuration (style carte)
class SettingsDialog(QDialog):
    # initialise le dialogue de configuration
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paramètres")
        self.setModal(True)
        self.setObjectName("SettingsCard")
        self.setMinimumWidth(520)
    # style "carte" : sans cadre, fond translucide et ombre portée douce
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setColor(Qt.black)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 8)
        self.setGraphicsEffect(shadow)

        # layout externe du dialogue (translucide)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
    # corps de la carte pour le style via QSS (#SettingsCardBody)
        self._body = QWidget(self)
        self._body.setObjectName("SettingsCardBody")
        layout = QVBoxLayout(self._body)
        layout.setContentsMargins(20, 20, 20, 20)

        # titre et options globales
        header = QHBoxLayout()
        title = QLabel("Paramètres de la corde")
        title.setAlignment(Qt.AlignLeft)
        header.addWidget(title, 1)
        self.live_preview_chk = QCheckBox("aperçu en temps réel")
        self.live_preview_chk.setChecked(True)
        header.addWidget(self.live_preview_chk, 0, Qt.AlignRight)
        layout.addLayout(header)

        # séparateur visuel
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # tabs modernes: Géométrie | Frettes
        self.tabs = QTabWidget(self._body)
        layout.addWidget(self.tabs, 1)

        # helpers
        def make_slider_row(min_v:int, max_v:int, init:int):
            row = QHBoxLayout()
            s = QSlider(Qt.Horizontal)
            s.setMinimum(min_v)
            s.setMaximum(max_v)
            s.setValue(init)
            s.setSingleStep(1)
            s.setTickInterval(max(1, (max_v - min_v) // 10))
            s.setTickPosition(QSlider.TicksBelow)
            val = QLabel("")
            val.setMinimumWidth(72)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(s, 1)
            row.addWidget(val, 0)
            return s, val, row

        # onglet: géométrie
        self.geometry_tab = QWidget()
        g_layout = QVBoxLayout(self.geometry_tab)
        g_layout.setContentsMargins(8, 8, 8, 8)
        g_form = QFormLayout()
        g_form.setLabelAlignment(Qt.AlignLeft)
        g_form.setFormAlignment(Qt.AlignLeft)

        # décalage : -0.25..0.25 -> -250..250
        self.offset_slider, self.offset_val, off_row = make_slider_row(-250, 250, 0)
        g_form.addRow("Décalage global (fraction)", off_row)

        # fin de la corde (scaleRightFrac) : 0.70..1.00 -> 700..1000
        self.end_slider, self.end_val, end_row = make_slider_row(700, 1000, 820)
        g_form.addRow("Fin de la corde (0.70..1.00)", end_row)

        # rosace : position 0..1 -> 0..1000 ; rayon 4..200
        self.soundhole_slider, self.soundhole_val, sh_row = make_slider_row(0, 1000, 660)
        g_form.addRow("Position de la rosace (0..1)", sh_row)

        self.radius_slider, self.radius_val, r_row = make_slider_row(4, 200, 28)
        g_form.addRow("Rayon de la rosace (px)", r_row)
        g_layout.addLayout(g_form)
        self.tabs.addTab(self.geometry_tab, "Géométrie")

        # onglet: frettes (ajustements fins)
        self.frets_tab = QWidget()
        f_layout = QVBoxLayout(self.frets_tab)
        f_layout.setContentsMargins(8, 8, 8, 8)
        f_form = QFormLayout()
        self.fret_sliders = []
        self.fret_vals = []
        for i in range(1, 13):
            # ajustement fin: -0.02..0.02 -> -20..20
            s, v, row = make_slider_row(-20, 20, 0)
            self.fret_sliders.append(s)
            self.fret_vals.append(v)
            f_form.addRow(f"Frette {i}", row)
        f_layout.addLayout(f_form)
        self.tabs.addTab(self.frets_tab, "Frettes")

        # boutons d'action
        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Appliquer")
        self.reset_btn = QPushButton("Réinitialiser")
        self.cancel_btn = QPushButton("Annuler")
        btn_row.addStretch(1)
        btn_row.addWidget(self.reset_btn)
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        # valeurs initiales à partir du parent
        cw = getattr(parent, 'corde_widget', None)
        if cw is not None:
            try:
                # positionne les curseurs à partir des valeurs actuelles
                self.offset_slider.setValue(int(round(float(cw.getGeometryOffsetFrac()) * 1000)))
                self.end_slider.setValue(int(round(float(cw.getScaleRightFrac()) * 1000)))
                self.soundhole_slider.setValue(int(round(float(cw.getSoundholeFrac()) * 1000)))
                self.radius_slider.setValue(int(cw.getSoundholeRadiusPx()))
                adj = cw.getFretAdjustments()
                for i in range(1, 13):
                    self.fret_sliders[i-1].setValue(int(round(float(adj[i]) * 1000)))
            except Exception:
                pass
        else:
            self.offset_slider.setValue(0)
            self.end_slider.setValue(820)
            self.soundhole_slider.setValue(660)
            self.radius_slider.setValue(28)
            for s in self.fret_sliders:
                s.setValue(0)

        # connexions
        self.apply_btn.clicked.connect(self._apply_settings)
        self.reset_btn.clicked.connect(self._reset_settings)
        self.cancel_btn.clicked.connect(self.reject)

        # met à jour les étiquettes de valeur lors du déplacement des curseurs
        self.offset_slider.valueChanged.connect(self._update_value_labels)
        self.end_slider.valueChanged.connect(self._update_value_labels)
        self.soundhole_slider.valueChanged.connect(self._update_value_labels)
        self.radius_slider.valueChanged.connect(self._update_value_labels)
        for s in self.fret_sliders:
            s.valueChanged.connect(self._update_value_labels)

        # aperçu en temps réel: applique automatiquement pendant l'édition
        self.live_preview_chk.toggled.connect(self._on_live_toggle)
        for s in [self.offset_slider, self.end_slider, self.soundhole_slider, self.radius_slider, *self.fret_sliders]:
            s.valueChanged.connect(self._apply_live)

        # initialise les étiquettes avec les valeurs initiales
        self._update_value_labels()

        # ajoute le corps au layout externe
        outer.addWidget(self._body)

    # applique les valeurs à la corde et ferme
    def _apply_settings(self):
        parent = self.parent()
        cw = getattr(parent, 'corde_widget', None)
        if cw is None:
            self.accept()
            return
        cw.setGeometryOffsetFrac(self.offset_slider.value() / 1000.0)
        cw.setScaleRightFrac(self.end_slider.value() / 1000.0)
        cw.setSoundholeFrac(self.soundhole_slider.value() / 1000.0)
        cw.setSoundholeRadiusPx(self.radius_slider.value())
        values = [0.0] + [s.value() / 1000.0 for s in self.fret_sliders]
        cw.setFretAdjustments(values)
        cw.update()
        self.accept()

    # réinitialise vers les valeurs par défaut
    def _reset_settings(self):
        self.offset_slider.setValue(0)
        self.end_slider.setValue(820)
        self.soundhole_slider.setValue(660)
        self.radius_slider.setValue(28)
        for s in self.fret_sliders:
            s.setValue(0)

    def _update_value_labels(self):
        # met à jour les étiquettes de valeur en fonction des curseurs
        self.offset_val.setText(f"{self.offset_slider.value()/1000.0:.3f}")
        self.end_val.setText(f"{self.end_slider.value()/1000.0:.3f}")
        self.soundhole_val.setText(f"{self.soundhole_slider.value()/1000.0:.3f}")
        self.radius_val.setText(str(self.radius_slider.value()))
        for i, (s, v) in enumerate(zip(self.fret_sliders, self.fret_vals), start=1):
            v.setText(f"{s.value()/1000.0:.3f}")

    def _on_live_toggle(self, checked: bool):
        # applique immédiatement quand on active l'aperçu en temps réel
        if checked:
            self._apply_live()

    def _apply_live(self):
        # applique les changements sans fermer le dialogue si l'aperçu en temps réel est actif
        if not self.live_preview_chk.isChecked():
            return
        parent = self.parent()
        cw = getattr(parent, 'corde_widget', None)
        if cw is None:
            return
        cw.setGeometryOffsetFrac(self.offset_slider.value() / 1000.0)
        cw.setScaleRightFrac(self.end_slider.value() / 1000.0)
        cw.setSoundholeFrac(self.soundhole_slider.value() / 1000.0)
        cw.setSoundholeRadiusPx(self.radius_slider.value())
        values = [0.0] + [s.value() / 1000.0 for s in self.fret_sliders]
        cw.setFretAdjustments(values)
        cw.update()
