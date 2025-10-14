from PyQt5.QtWidgets import QWidget, QSizePolicy, QToolTip
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PyQt5.QtCore import Qt, QPointF, QSize, pyqtProperty, pyqtSignal

class CordeWidget(QWidget):
    # sinal emitido quando o usuário clica/toca na corda (posição normalizada 0..1)
    clickedAt = pyqtSignal(float)
    """
    widget personnalisé qui dessine la corde musicale, les frettes et les curseurs.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # définit la politique de taille pour que le widget puisse s'étendre
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(300)
        self.setMinimumWidth(320)

        # état actuel de la note mise en évidence
        self.current_note = None
        # curseur déplaçable le long de la corde (0.0 = sillet, 1.0 = extrémité droite)
        self.cursor_pos = 0.0
        # réglages/calibration de géométrie (fraction de l'échelle)
        self._geometryOffsetFrac = 0.0  # décale toute la géométrie en X (en fraction de l'échelle), pour un alignement fin
        # position typique de la rosace le long de l'échelle (fraction 0..1 depuis le sillet)
        self._soundholeFrac = 0.66
        self._soundholeRadiusPx = 28
        # second curseur (attaque/strike), contrôlé séparément — commence à la position de la rosace
        self.strike_cursor_pos = self._soundholeFrac
        # ajustements fins par frette (fraction 0..1 à ajouter au modèle 12-TET), indices 0..12
        # 0 forcé à 0.0 ; autorise de petits ajustements (-0.02..0.02) pour 1..12
        self._fretAdjustFrac = [0.0] * 13
        # fraction de fin visible de la corde (0.6..1.0). Pour refléter exactement la longueur, utiliser 1.0.
        # Ajusté par défaut à 1.0 pour que la corde visible couvre 100% de la longueur normalisée.
        self._scaleRightFrac = 1.0
        
        # couleurs configurables via QSS (qproperty-*)
        self._stringColor = QColor("#c0c0c0")
        self._fretColor = QColor("#888888")
        self._nutColor = QColor("#bbbbbb")
        self._fretTextColor = QColor("#aaaaaa")
        self._cursorColor = QColor("#5db3f0")
        self._strikeCursorColor = QColor("#f59e0b")  # couleur distincte pour le curseur d'attaque
        # couleur du rastilho (bridge)
        self._bridgeColor = QColor("#9AA3B2")
        # exibir rótulo do rastilho ("R") — desativado por padrão
        self._showBridgeLabel = False

        # --- géométrie issue du back-end (config) ---
        # Nous construisons la discrétisation normalisée [0..1] directement à partir de
        # digital_twin.back_end.config.FRET_NODE_POSITIONS_MM (cumule) et FRET_DXS_MM.
        # Si indisponible, nous retombons sur une échelle uniforme simple.
        self._node_fracs = [0.0]
        self._have_backend_geom = False
        self._total_length_m = 1.0
        try:
            from digital_twin.back_end import config as _cfg  # type: ignore
            nodes_mm = list(getattr(_cfg, 'FRET_NODE_POSITIONS_MM', [0.0]))
            if nodes_mm and nodes_mm[-1] > 0:
                total_m = float(nodes_mm[-1]) / 1000.0
                # normalise en fractions 0..1 par rapport à la dernière position de nœud
                self._node_fracs = [float(v) / 1000.0 / total_m for v in nodes_mm]
                self._have_backend_geom = True
                self._total_length_m = total_m
        except Exception:
            self._node_fracs = [i / 100.0 for i in range(101)]  # fallback uniforme 100 divisions
            self._have_backend_geom = False
            self._total_length_m = 1.0

        # Politique de pas: aimanter les curseurs aux nœuds réels (discrétisation non uniforme)
        self._snapToNodes = True
        # activer le suivi de la souris pour mettre à jour l'infobulle dynamiquement
        self.setMouseTracking(True)

    def _build_tooltip_text(self) -> str:
        """Construit un texte d'infobulle avec les informations des deux curseurs (note et attaque)."""
        try:
            node_note = self.nearest_node_index(self.cursor_pos)
            elem_note = self.nearest_element_index(self.cursor_pos)
            x_note = self.frac_to_meters(self.cursor_pos)
        except Exception:
            node_note, elem_note, x_note = 0, 0, 0.0
        try:
            node_strike = self.nearest_node_index(self.strike_cursor_pos)
            elem_strike = self.nearest_element_index(self.strike_cursor_pos)
            x_strike = self.frac_to_meters(self.strike_cursor_pos)
        except Exception:
            node_strike, elem_strike, x_strike = 0, 0, 0.0
        return (
            f"Note: Nœud {node_note}, Élément {elem_note}, x={x_note:.3f} m\n"
            f"Pincer: Nœud {node_strike}, Élément {elem_strike}, x={x_strike:.3f} m"
        )

    # --- api d'interaction ---
    def set_current_note(self, note_name):
        """définit quelle note doit être mise en évidence."""
        self.current_note = note_name
        self.update() # planifie un repaint du widget

    def _nearest_node_frac(self, x: float) -> float:
        """retourne la fraction du nœud le plus proche dans self._node_fracs."""
        if not self._node_fracs:
            return max(0.0, min(1.0, float(x)))
        # recherche linéaire suffisante vu N ~ O(100); peut être optimisée si besoin
        xf = max(0.0, min(1.0, float(x)))
        fracs = self._node_fracs
        # bornes rapides
        if xf <= fracs[0]:
            return fracs[0]
        if xf >= fracs[-1]:
            return fracs[-1]
        # trouve intervalle
        for i in range(1, len(fracs)):
            if fracs[i] >= xf:
                # choisir le plus proche entre i-1 et i
                a, b = fracs[i - 1], fracs[i]
                return a if (xf - a) <= (b - xf) else b
        return fracs[-1]

    def _step_to_neighbor_node(self, current: float, direction: int) -> float:
        """retourne la fraction du nœud voisin à gauche/droite par rapport à current."""
        fracs = self._node_fracs
        if not fracs:
            return max(0.0, min(1.0, current + (0.01 if direction > 0 else -0.01)))
        # trouve l'indice du nœud le plus proche
        xf = self._nearest_node_frac(current)
        try:
            idx = fracs.index(xf)
        except ValueError:
            # si valeurs flottantes non exactes, approxime par balayage
            idx = 0
            best_d = 1e9
            for i, v in enumerate(fracs):
                d = abs(v - xf)
                if d < best_d:
                    best_d, idx = d, i
        # pas vers voisin
        if direction > 0:
            idx = min(idx + 1, len(fracs) - 1)
        elif direction < 0:
            idx = max(idx - 1, 0)
        return fracs[idx]

    def node_fracs(self):
        """expose la liste des fractions de nœuds [0..1] (lecture seule)."""
        return list(self._node_fracs)

    def total_length_m(self) -> float:
        """retourne la longueur totale en mètres (dérivée du backend ou 1.0 en fallback)."""
        return float(self._total_length_m)

    def frac_to_meters(self, frac: float) -> float:
        """convertit une fraction [0..1] en position en mètres sur la corde."""
        xf = max(0.0, min(1.0, float(frac)))
        return xf * self._total_length_m

    def nearest_node_index(self, x: float) -> int:
        """retourne l'indice du nœud le plus proche pour une fraction x [0..1]."""
        fracs = self._node_fracs
        if not fracs:
            return 0
        xf = max(0.0, min(1.0, float(x)))
        best_i, best_d = 0, 1e9
        for i, v in enumerate(fracs):
            d = abs(v - xf)
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def nearest_element_index(self, x: float) -> int:
        """retourne l'indice d'élément j tel que fracs[j] <= x <= fracs[j+1]. Clampe à [0, n_elems-1]."""
        fracs = self._node_fracs
        if not fracs:
            return 0
        xf = max(0.0, min(1.0, float(x)))
        # dernier cas: si sur le dernier nœud, renvoyer l'élément final
        if xf >= fracs[-1]:
            return max(0, len(fracs) - 2)
        # trouver intervalle
        for i in range(len(fracs) - 1):
            if fracs[i] <= xf <= fracs[i + 1]:
                return i
        return 0

    def set_cursor_pos(self, pos_norm: float):
        """définit la position du curseur (0..1), avec aimantation optionnelle aux nœuds."""
        v = max(0.0, min(1.0, float(pos_norm)))
        if self._snapToNodes:
            v = self._nearest_node_frac(v)
        self.cursor_pos = v
        self.update()

    def move_cursor(self, delta_norm: float):
        """déplace le curseur vers le nœud voisin (discret) si snapping actif; sinon incrément normalisé."""
        if self._snapToNodes:
            direction = 1 if float(delta_norm) > 0 else (-1 if float(delta_norm) < 0 else 0)
            if direction != 0:
                self.set_cursor_pos(self._step_to_neighbor_node(self.cursor_pos, direction))
        else:
            self.set_cursor_pos(self.cursor_pos + float(delta_norm))

    def set_strike_cursor_pos(self, pos_norm: float):
        """définit la position du curseur d'attaque (0..1); applique le même snapping par défaut."""
        v = max(0.0, min(1.0, float(pos_norm)))
        if self._snapToNodes:
            v = self._nearest_node_frac(v)
        self.strike_cursor_pos = v
        self.update()

    def move_strike_cursor(self, delta_norm: float):
        """déplace le curseur d'attaque vers le nœud voisin si snapping actif; sinon incrément normalisé."""
        if self._snapToNodes:
            direction = 1 if float(delta_norm) > 0 else (-1 if float(delta_norm) < 0 else 0)
            if direction != 0:
                self.set_strike_cursor_pos(self._step_to_neighbor_node(self.strike_cursor_pos, direction))
        else:
            self.set_strike_cursor_pos(self.strike_cursor_pos + float(delta_norm))

    def _fret_line_frac(self, f: int) -> float:
        """retourne la fraction normalisée de la frette (ligne métallique) f (0..12) via formule (12-TET-like).
        La visualisation des frettes utilise la formule géométrique; les curseurs peuvent être aimantés aux nœuds réels."""
        f = max(0, min(12, int(f)))
        # Récupère l'exposant géométrique centralisé (config) si disponible
        try:
            from digital_twin.back_end import config as _cfg  # type: ignore
            _exp = getattr(_cfg, 'FRET_EXPOSANT_GEOMETRIQUE', 12.0)
        except Exception:  # pragma: no cover
            _exp = 12.0
        base = 0.0 if f == 0 else 1.0 - (1.0 / (2 ** (float(f) / float(_exp))))
        adj = 0.0 if f == 0 else float(self._fretAdjustFrac[f])
        # applique l'ajustement et limite à [0,1]
        v = max(0.0, min(1.0, base + adj))
        return v

    # --- api d'ajustement des frettes ---
    def getFretAdjustments(self):
        """retourne une copie de la liste des ajustements fins par frette (fractions)."""
        return list(self._fretAdjustFrac)

    def setFretAdjustment(self, index: int, value: float):
        """définit un ajustement fin pour une frette spécifique (1..12), en fraction (-0.02..0.02)."""
        idx = max(0, min(12, int(index)))
        if idx == 0:
            self._fretAdjustFrac[0] = 0.0
        else:
            try:
                v = float(value)
            except Exception:
                v = 0.0
            v = max(-0.02, min(0.02, v))
            self._fretAdjustFrac[idx] = v
        self.update()

    def setFretAdjustments(self, values):
        """définit tous les ajustements (liste taille 13, indices 0..12) ; 0 est forcé à 0.0."""
        if not values:
            return
        arr = [0.0] * 13
        for i in range(13):
            try:
                v = float(values[i])
            except Exception:
                v = 0.0
            if i == 0:
                arr[0] = 0.0
            else:
                arr[i] = max(-0.02, min(0.02, v))
        self._fretAdjustFrac = arr
        self.update()

    def set_cursor_by_fret(self, fret_index: int, press_bias: float = 0.9):
        """place le curseur exactement sur la ligne de frette N (nœud N), 0..12.
        La bias n'est pas utilisée lorsque l'aimantation aux nœuds est active."""
        fret_index = max(0, min(12, int(fret_index)))
        self.set_cursor_pos(self._fret_line_frac(fret_index))

    def house_center_frac(self, house_index: int) -> float:
        """retourne la fraction normalisée du centre de la maison (entre frettes) house_index (1..12)."""
        house_index = max(1, min(12, int(house_index)))
        left = self._fret_line_frac(house_index - 1)
        right = self._fret_line_frac(house_index)
        return (left + right) / 2.0

    def set_cursor_house_center(self, house_index: int):
        """positionne le curseur exactement au centre de la maison (entre frettes). 1..12. Pour 0, utilise la corde à vide (0.0)."""
        if int(house_index) <= 0:
            self.set_cursor_pos(0.0)
            return
        self.set_cursor_pos(self.house_center_frac(int(house_index)))

    def set_cursor_fret_line(self, fret_index: int):
        """positionne le curseur exactement sur la ligne métallique de la frette (0..12)."""
        fret_index = max(0, min(12, int(fret_index)))
        self.set_cursor_pos(self._fret_line_frac(fret_index))

    # --- propriétés de couleur (pour QSS : qproperty-<name>) ---
    def getStringColor(self):
        return self._stringColor

    def setStringColor(self, color):
        self._stringColor = QColor(color)
        self.update()

    stringColor = pyqtProperty(QColor, fget=getStringColor, fset=setStringColor)

    def getFretColor(self):
        return self._fretColor

    def setFretColor(self, color):
        self._fretColor = QColor(color)
        self.update()

    fretColor = pyqtProperty(QColor, fget=getFretColor, fset=setFretColor)

    def getNutColor(self):
        return self._nutColor

    def setNutColor(self, color):
        self._nutColor = QColor(color)
        self.update()

    nutColor = pyqtProperty(QColor, fget=getNutColor, fset=setNutColor)

    def getFretTextColor(self):
        return self._fretTextColor

    def setFretTextColor(self, color):
        self._fretTextColor = QColor(color)
        self.update()

    fretTextColor = pyqtProperty(QColor, fget=getFretTextColor, fset=setFretTextColor)

    def getCursorColor(self):
        return self._cursorColor

    def setCursorColor(self, color):
        self._cursorColor = QColor(color)
        self.update()

    cursorColor = pyqtProperty(QColor, fget=getCursorColor, fset=setCursorColor)

    def getStrikeCursorColor(self):
        return self._strikeCursorColor

    def setStrikeCursorColor(self, color):
        self._strikeCursorColor = QColor(color)
        self.update()

    strikeCursorColor = pyqtProperty(QColor, fget=getStrikeCursorColor, fset=setStrikeCursorColor)

    def getBridgeColor(self):
        return self._bridgeColor

    def setBridgeColor(self, color):
        self._bridgeColor = QColor(color)
        self.update()

    bridgeColor = pyqtProperty(QColor, fget=getBridgeColor, fset=setBridgeColor)

    def getShowBridgeLabel(self) -> bool:
        return bool(self._showBridgeLabel)

    def setShowBridgeLabel(self, value: bool):
        try:
            self._showBridgeLabel = bool(value)
        except Exception:
            self._showBridgeLabel = True
        self.update()

    showBridgeLabel = pyqtProperty(bool, fget=getShowBridgeLabel, fset=setShowBridgeLabel)

    # --- propriétés d'ajustement/calibration ---
    def getGeometryOffsetFrac(self) -> float:
        return self._geometryOffsetFrac

    def setGeometryOffsetFrac(self, value: float):
        # limite le décalage entre -0.25 et 0.25 de l'échelle (25 %)
        try:
            v = float(value)
        except Exception:
            v = 0.0
        self._geometryOffsetFrac = max(-0.25, min(0.25, v))
        self.update()

    geometryOffsetFrac = pyqtProperty(float, fget=getGeometryOffsetFrac, fset=setGeometryOffsetFrac)

    def getSoundholeFrac(self) -> float:
        return self._soundholeFrac

    def setSoundholeFrac(self, value: float):
        try:
            v = float(value)
        except Exception:
            v = 0.66
        self._soundholeFrac = max(0.0, min(1.0, v))
        # maintient éventuellement le curseur d'attaque aligné à la rosace s'il est très proche
        if abs(self.strike_cursor_pos - self._soundholeFrac) < 1e-6:
            self.strike_cursor_pos = self._soundholeFrac
        self.update()

    soundholeFrac = pyqtProperty(float, fget=getSoundholeFrac, fset=setSoundholeFrac)

    def getSoundholeRadiusPx(self) -> int:
        return int(self._soundholeRadiusPx)

    def setSoundholeRadiusPx(self, value: int):
        try:
            r = int(value)
        except Exception:
            r = 28
        self._soundholeRadiusPx = max(4, min(200, r))
        self.update()

    soundholeRadiusPx = pyqtProperty(int, fget=getSoundholeRadiusPx, fset=setSoundholeRadiusPx)

    def getScaleRightFrac(self) -> float:
        return float(self._scaleRightFrac)

    def setScaleRightFrac(self, value: float):
        try:
            v = float(value)
        except Exception:
            v = 0.82
        # limite entre 0.6 et 1.0
        self._scaleRightFrac = max(0.6, min(1.0, v))
        self.update()

    scaleRightFrac = pyqtProperty(float, fget=getScaleRightFrac, fset=setScaleRightFrac)

    # --- tailles préférées ---
    def sizeHint(self) -> QSize:
        return QSize(1000, max(320, self.minimumHeight()))

    def clear_current_note(self):
        """efface la note mise en évidence."""
        self.current_note = None
        self.update()

    def paintEvent(self, event):
        """
        méthode appelée automatiquement pour dessiner le widget.
        dessine les 12 frettes de la guitare et distribue correctement les éléments.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = max(1, self.width())
        height = max(1, self.height())
        center_y = height // 2
        # ajuste la marge pour s'adapter aux petites largeurs
        margin = min(30, max(10, width // 12))
        escala = max(1, width - 2 * margin)

        # décalage global en pixels (comme fraction de l'échelle)
        dx = int(escala * self._geometryOffsetFrac)

        # calcule le décalage gauche pour centrer la corde raccourcie
        left_pad = int(escala * (1.0 - float(self._scaleRightFrac)) / 2.0)

        # 1. dessine la corde principale
        # facteur d'échelle UI pour agrandir légèrement l'objet en fonction de la hauteur
        ui_scale = max(1.0, min(1.5, float(height) / 240.0))

        corda_pen = QPen(self._stringColor, 3 * ui_scale, Qt.SolidLine)
        painter.setPen(corda_pen)
        start_x = margin + left_pad + dx
        end_x = start_x + int(escala * self._scaleRightFrac)
        painter.drawLine(start_x, center_y, end_x, center_y)

        # 2. dessine les 12 frettes basées sur les nœuds du backend
        num_frettes = 12
        for traste in range(num_frettes + 1):
            frac = self._fret_line_frac(traste)
            x = margin + left_pad + escala * frac
            # sillet (frette 0) plus épais
            if traste == 0:
                traste_pen = QPen(self._nutColor, max(4, int(4 * ui_scale)))
            else:
                traste_pen = QPen(self._fretColor, max(2, int(2 * ui_scale)))
            painter.setPen(traste_pen)
            h = int(28 * ui_scale)
            painter.drawLine(int(x + dx), center_y - h, int(x + dx), center_y + h)
            # ajoute la numérotation des frettes
            if traste > 0:
                painter.setPen(self._fretTextColor)
                font = QFont("Arial", 10)
                painter.setFont(font)
                painter.drawText(int(x + dx) - 8, center_y - 35, str(traste))

        # 2b. dessine le rastilho (bridge) à l'extrémité droite de la corde visible
        try:
            # utiliser les mêmes dimensions que le sillet (frette 0)
            nut_thickness = max(4, int(4 * ui_scale))
            nut_h = int(28 * ui_scale)
            bridge_pen = QPen(self._bridgeColor, nut_thickness, Qt.SolidLine)
            painter.setPen(bridge_pen)
            painter.drawLine(int(end_x), center_y - nut_h, int(end_x), center_y + nut_h)
            # pas d'étiquette "R"
        except Exception:
            pass

        # 3. (pas de marqueurs de note) — seulement frettes et numérotation

        # 4. dessine le curseur principal
        # le curseur ne doit pas dépasser la fin visible de la corde (rendu seulement)
        cursor_vis = min(max(0.0, self.cursor_pos), self._scaleRightFrac)
        cursor_x = margin + left_pad + escala * cursor_vis + dx
        cursor_pen = QPen(self._cursorColor, 4.4 * ui_scale, Qt.DashLine)
        painter.setPen(cursor_pen)
        c_h = int(36 * ui_scale)
        painter.drawLine(int(cursor_x), center_y - c_h, int(cursor_x), center_y + c_h)

        # 5. dessine le curseur d'attaque (strike) en ligne pleine
        strike_vis = min(max(0.0, self.strike_cursor_pos), self._scaleRightFrac)
        strike_x = margin + left_pad + escala * strike_vis + dx
        strike_pen = QPen(self._strikeCursorColor, 3.8 * ui_scale, Qt.SolidLine)
        painter.setPen(strike_pen)
        painter.drawLine(int(strike_x), center_y - c_h, int(strike_x), center_y + c_h)

        # 6. dessine un marqueur de rosace (cercle) à la position indiquée
        #    utilise la couleur de la corde (stringColor) avec un remplissage translucide
        soundhole_x = margin + left_pad + escala * self._soundholeFrac + dx
        c_fill = QColor(self._stringColor)
        c_fill.setAlpha(36)
        painter.setPen(QPen(self._stringColor, 2))
        painter.setBrush(QBrush(c_fill))
        r = int(self._soundholeRadiusPx * ui_scale)
        painter.drawEllipse(QPointF(float(soundhole_x), float(center_y)), float(r), float(r))

    def mousePressEvent(self, event):
        """Mappe un clic de souris en position normalisée le long de la corde et émet un signal.

        La position est calculée par rapport au segment visible de la corde, et clampée à [0, _scaleRightFrac].
        """
        try:
            width = max(1, self.width())
            margin = min(30, max(10, width // 12))
            escala = max(1, width - 2 * margin)
            dx = int(escala * self._geometryOffsetFrac)
            left_pad = int(escala * (1.0 - float(self._scaleRightFrac)) / 2.0)
            start_x = margin + left_pad + dx
            length_vis = escala * float(self._scaleRightFrac)
            x = float(event.pos().x())
            frac = (x - start_x) / max(1.0, float(length_vis))
            frac = max(0.0, min(1.0, frac))
            # converte para a escala completa [0..1] dentro do trecho visível
            pos_norm = float(frac * float(self._scaleRightFrac))
            try:
                self.clickedAt.emit(pos_norm)
            except Exception:
                pass
            # affiche une infobulle contextuelle avec les infos des curseurs
            try:
                QToolTip.showText(event.globalPos(), self._build_tooltip_text(), self)
            except Exception:
                pass
        except Exception:
            pass
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Met à jour l'infobulle pendant le survol pour refléter les infos de position."""
        try:
            self.setToolTip(self._build_tooltip_text())
        except Exception:
            pass
        super().mouseMoveEvent(event)
