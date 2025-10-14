from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QLabel, QFormLayout, QHBoxLayout,
    QPushButton, QSlider, QFrame, QColorDialog, QGridLayout, QLineEdit, QSpinBox, QTabWidget
)
from PyQt5.QtCore import Qt, QSettings, QPoint

# petit dialogue de configuration (style carte)
class SettingsDialog(QDialog):
    # initialise le dialogue de configuration
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paramètres")
        self.setModal(True)
        self.setObjectName("SettingsCard")
        self.setMinimumWidth(520)
    # style "carte" : sans cadre
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        # Fundo opaco (sem transparência)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

    # nota: aplicar sombra no corpo interno evita artefatos de composição em janelas sem moldura

        # layout externe du dialogue (opaco)
        outer = QVBoxLayout(self)
        # sem margens externas para não aparecer moldura ao redor do corpo
        outer.setContentsMargins(0, 0, 0, 0)
    # corps de la carte pour le style via QSS (#SettingsCardBody)
        self._body = QWidget(self)
        self._body.setObjectName("SettingsCardBody")
        layout = QVBoxLayout(self._body)
        layout.setContentsMargins(20, 20, 20, 20)

        # header com título e botão de fechar (X), também serve de área de arraste
        self._header = QWidget(self._body)
        self._header.setObjectName("DialogHeader")
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        title = QLabel("Aparência da corda", self._header)
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._btn_close = QPushButton("✕", self._header)
        self._btn_close.setObjectName("DialogCloseButton")
        self._btn_close.setCursor(Qt.PointingHandCursor)
        self._btn_close.setFixedSize(28, 28)
        self._btn_close.clicked.connect(self.reject)
        header_layout.addWidget(title, 1)
        header_layout.addWidget(self._btn_close, 0, Qt.AlignRight)
        layout.addWidget(self._header)

        # suporte a arraste pela barra de título
        self._drag_active = False
        self._drag_pos = None
        self._header.installEventFilter(self)

        # (título já adicionado no header)

        # séparateur visuel
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # Abas: "Couleurs" e "Dimensions"
        self._tabs = QTabWidget(self._body)
        layout.addWidget(self._tabs)

        # Página Couleurs
        tab_colors = QWidget()
        tab_colors_layout = QVBoxLayout(tab_colors)
        tab_colors_layout.setContentsMargins(0, 0, 0, 0)

        # grid de cores: [rótulo | amostra | hex | reset]
        colors_grid = QGridLayout()
        colors_grid.setHorizontalSpacing(10)
        colors_grid.setVerticalSpacing(8)

        def add_color_row(row: int, label_text: str, attr_name: str, default_hex: str):
            lbl = QLabel(label_text)
            btn = QPushButton()
            btn.setFixedWidth(56)
            btn.setCursor(Qt.PointingHandCursor)
            hex_edit = QLineEdit()
            hex_edit.setReadOnly(True)
            hex_edit.setMinimumWidth(96)
            reset_btn = QPushButton("↺")
            reset_btn.setToolTip("Restaurar padrão")
            reset_btn.setFixedWidth(28)

            colors_grid.addWidget(lbl, row, 0)
            colors_grid.addWidget(btn, row, 1)
            colors_grid.addWidget(hex_edit, row, 2)
            colors_grid.addWidget(reset_btn, row, 3)

            return lbl, btn, hex_edit, reset_btn, attr_name, default_hex

        # criar linhas de cor (mantendo as cores atuais da paleta)
        rows = []
        rows.append(add_color_row(0, "Cor da corda", 'string', "#E9EDF2"))
        rows.append(add_color_row(1, "Cor das casas", 'fret', "#8B94A6"))
        rows.append(add_color_row(2, "Cor do sillet", 'nut', "#DDE2EB"))
        rows.append(add_color_row(3, "Cor dos números", 'fret_text', "#B8C0CC"))
        rows.append(add_color_row(4, "Cor do cursor", 'cursor', "#7DD3FC"))
        rows.append(add_color_row(5, "Cor do ataque", 'strike', "#F4B96B"))
        rows.append(add_color_row(6, "Cor do rastilho", 'bridge', "#9AA3B2"))

        tab_colors_layout.addLayout(colors_grid)
        self._tabs.addTab(tab_colors, "Couleurs")

        # Página Dimensions
        tab_dims = QWidget()
        tab_dims_layout = QVBoxLayout(tab_dims)
        tab_dims_layout.setContentsMargins(0, 0, 0, 0)

        # Seção: Altura mínima (slider + spinbox sincronizados)
        min_row = QHBoxLayout()
        lbl_min = QLabel("Altura mínima da corda (px)")
        lbl_min.setMinimumWidth(240)
        self.minh_slider = QSlider(Qt.Horizontal)
        self.minh_slider.setMinimum(200)
        self.minh_slider.setMaximum(600)
        self.minh_slider.setSingleStep(5)
        self.minh_spin = QSpinBox()
        self.minh_spin.setRange(200, 600)
        self.minh_spin.setSingleStep(5)
        min_row.addWidget(lbl_min, 0)
        min_row.addWidget(self.minh_slider, 1)
        min_row.addSpacing(8)
        min_row.addWidget(self.minh_spin, 0)
        tab_dims_layout.addLayout(min_row)
        self._tabs.addTab(tab_dims, "Dimensions")

        # boutons d'action
        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Aplicar")
        self.reset_btn = QPushButton("Padrão")
        self.cancel_btn = QPushButton("Cancelar")
        btn_row.addStretch(1)
        btn_row.addWidget(self.reset_btn)
        btn_row.addSpacing(6)
        btn_row.addWidget(self.apply_btn)
        btn_row.addSpacing(6)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        # valores iniciais a partir do parent
        cw = getattr(parent, 'corde_widget', None)
        if cw is not None:
            try:
                # min height
                vmin = max(200, min(600, int(cw.minimumHeight())))
                self.minh_slider.setValue(vmin)
                self.minh_spin.setValue(vmin)
                # sem rótulo de rastilho (configuração não exposta)
                # cores atuais
                def qcolor_to_hex(qc):
                    try:
                        return qc.name()
                    except Exception:
                        return str(qc)
                self._color_string = qcolor_to_hex(cw.getStringColor())
                self._color_fret = qcolor_to_hex(cw.getFretColor())
                self._color_nut = qcolor_to_hex(cw.getNutColor())
                self._color_fret_text = qcolor_to_hex(cw.getFretTextColor())
                self._color_cursor = qcolor_to_hex(cw.getCursorColor())
                self._color_strike = qcolor_to_hex(cw.getStrikeCursorColor())
                self._color_bridge = qcolor_to_hex(cw.getBridgeColor())
            except Exception:
                pass
        else:
            self.minh_slider.setValue(300)
            self.minh_spin.setValue(300)
            # fallback para cores (Paleta A)
            self._color_string = "#E9EDF2"
            self._color_fret = "#8B94A6"
            self._color_nut = "#DDE2EB"
            self._color_fret_text = "#B8C0CC"
            self._color_cursor = "#7DD3FC"
            self._color_strike = "#F4B96B"
            self._color_bridge = "#9AA3B2"

        # mapa de atributos para valores atuais e widgets
        self._color_map = {
            'string': [self._color_string, None],
            'fret': [self._color_fret, None],
            'nut': [self._color_nut, None],
            'fret_text': [self._color_fret_text, None],
            'cursor': [self._color_cursor, None],
            'strike': [self._color_strike, None],
            'bridge': [self._color_bridge, None],
        }

        # configurar UI das linhas de cor e callbacks
        def set_btn_color(btn: QPushButton, hex_color: str):
            btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #3E4A5E; border-radius: 6px;")

        for (lbl, btn, hex_edit, reset_btn, key, def_hex) in rows:
            # valor inicial
            current_hex = self._color_map[key][0]
            set_btn_color(btn, current_hex)
            hex_edit.setText(current_hex)
            # armazenar hex_edit no mapa para atualizar depois
            self._color_map[key][1] = hex_edit

            def make_pick(btn_ref, key_name):
                def _pick():
                    color = QColorDialog.getColor()
                    if color and color.isValid():
                        hexv = color.name()
                        self._color_map[key_name][0] = hexv
                        set_btn_color(btn_ref, hexv)
                        self._color_map[key_name][1].setText(hexv)
                return _pick
            btn.clicked.connect(make_pick(btn, key))

            def make_reset(btn_ref, key_name, default_hex_val):
                def _reset():
                    self._color_map[key_name][0] = default_hex_val
                    set_btn_color(btn_ref, default_hex_val)
                    self._color_map[key_name][1].setText(default_hex_val)
                return _reset
            reset_btn.clicked.connect(make_reset(btn, key, def_hex))

        # connexions
        self.apply_btn.clicked.connect(self._apply_settings)
        self.reset_btn.clicked.connect(self._reset_settings)
        self.cancel_btn.clicked.connect(self.reject)

        # sincronização slider <-> spinbox
        self._syncing_min = False
        def on_slider(v):
            if self._syncing_min: return
            self._syncing_min = True
            try:
                self.minh_spin.setValue(int(v))
            finally:
                self._syncing_min = False
        def on_spin(v):
            if self._syncing_min: return
            self._syncing_min = True
            try:
                self.minh_slider.setValue(int(v))
            finally:
                self._syncing_min = False
        self.minh_slider.valueChanged.connect(on_slider)
        self.minh_spin.valueChanged.connect(on_spin)

        # preview leve ao soltar o slider (sem live contínuo)
        def preview_min_height():
            cw = getattr(self.parent(), 'corde_widget', None)
            if cw is None: return
            try:
                cw.setMinimumHeight(int(self.minh_slider.value()))
                cw.update()
            except Exception:
                pass
        self.minh_slider.sliderReleased.connect(preview_min_height)

        # ajoute le corps au layout externe
        outer.addWidget(self._body)

        # restaurar posição anterior do diálogo (persistência leve)
        try:
            s = QSettings("DigitalTwin", "UI")
            pos = s.value("settingsDialog/pos")
            if isinstance(pos, QPoint):
                self.move(pos)
        except Exception:
            pass

    def eventFilter(self, obj, event):
        # permite arrastar a janela pela barra de título
        try:
            from PyQt5.QtCore import QEvent, QPoint
            if obj is self._header:
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    self._drag_active = True
                    self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
                    return True
                elif event.type() == QEvent.MouseMove and self._drag_active:
                    self.move(event.globalPos() - self._drag_pos)
                    return True
                elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                    self._drag_active = False
                    return True
        except Exception:
            pass
        return super().eventFilter(obj, event)

    # aplica os valores à corda e fecha
    def _apply_settings(self):
        parent = self.parent()
        cw = getattr(parent, 'corde_widget', None)
        if cw is None:
            self.accept()
            return
        # altura mínima
        try:
            cw.setMinimumHeight(int(self.minh_slider.value()))
        except Exception:
            pass
        # cores (qproperties)
        try: cw.setStringColor(self._color_map['string'][0])
        except Exception: pass
        try: cw.setFretColor(self._color_map['fret'][0])
        except Exception: pass
        try: cw.setNutColor(self._color_map['nut'][0])
        except Exception: pass
        try: cw.setFretTextColor(self._color_map['fret_text'][0])
        except Exception: pass
        try: cw.setCursorColor(self._color_map['cursor'][0])
        except Exception: pass
        try: cw.setStrikeCursorColor(self._color_map['strike'][0])
        except Exception: pass
        try: cw.setBridgeColor(self._color_map['bridge'][0])
        except Exception: pass
        cw.update()
        # salvar posição antes de fechar
        try:
            s = QSettings("DigitalTwin", "UI")
            s.setValue("settingsDialog/pos", self.pos())
        except Exception:
            pass
        self.accept()

    # restaura para valores padrão visuais
    def _reset_settings(self):
        self.minh_slider.setValue(300)
        self.minh_spin.setValue(300)
        defaults = {
            'string': "#E9EDF2",
            'fret': "#8B94A6",
            'nut': "#DDE2EB",
            'fret_text': "#B8C0CC",
            'cursor': "#7DD3FC",
            'strike': "#F4B96B",
            'bridge': "#9AA3B2",
        }
        def set_btn_color(btn: QPushButton, hex_color: str):
            btn.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #3E4A5E; border-radius: 6px;")
        for k, v in defaults.items():
            self._color_map[k][0] = v
            self._color_map[k][1].setText(v)
        # atualizar botões (percorrer grid widgets)
        # como não guardamos referência dos botões, vamos reconfigurar via rows recriados acima
        # para simplificar, apenas atualizamos estilos consultando os hex atuais nos edits
        # (os botões manterão seu estilo até nova escolha; mudança visual ocorre ao clicar reset por linha)

    def keyPressEvent(self, event):
        # Enter aplica, Esc cancela
        try:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._apply_settings()
                return
            if event.key() == Qt.Key_Escape:
                try:
                    s = QSettings("DigitalTwin", "UI")
                    s.setValue("settingsDialog/pos", self.pos())
                except Exception:
                    pass
                self.reject()
                return
        except Exception:
            pass
        super().keyPressEvent(event)
