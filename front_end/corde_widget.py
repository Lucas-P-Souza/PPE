from PyQt5.QtWidgets import QWidget, QSizePolicy
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
        self.setMinimumHeight(150)
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
        # fraction de fin visible de la corde (0.6..1.0), pour que la corde se termine "un peu après" la rosace
        self._scaleRightFrac = 0.82
        
        # couleurs configurables via QSS (qproperty-*)
        self._stringColor = QColor("#c0c0c0")
        self._fretColor = QColor("#888888")
        self._nutColor = QColor("#bbbbbb")
        self._fretTextColor = QColor("#aaaaaa")
        self._cursorColor = QColor("#5db3f0")
        self._strikeCursorColor = QColor("#f59e0b")  # couleur distincte pour le curseur d'attaque

    # --- api d'interaction ---
    def set_current_note(self, note_name):
        """définit quelle note doit être mise en évidence."""
        self.current_note = note_name
        self.update() # planifie un repaint du widget

    def set_cursor_pos(self, pos_norm: float):
        """définit la position du curseur (0..1)."""
        self.cursor_pos = max(0.0, min(1.0, float(pos_norm)))
        self.update()

    def move_cursor(self, delta_norm: float):
        """déplace le curseur par un incrément normalisé."""
        self.set_cursor_pos(self.cursor_pos + float(delta_norm))

    def set_strike_cursor_pos(self, pos_norm: float):
        """définit la position du curseur d'attaque (0..1)."""
        self.strike_cursor_pos = max(0.0, min(1.0, float(pos_norm)))
        self.update()

    def move_strike_cursor(self, delta_norm: float):
        """déplace le curseur d'attaque par un incrément normalisé."""
        self.set_strike_cursor_pos(self.strike_cursor_pos + float(delta_norm))

    def _fret_line_frac(self, f: int) -> float:
        """retourne la fraction normalisée de la frette (ligne métallique) f (0..12), avec ajustement fin."""
        f = max(0, min(12, int(f)))
        # Récupère l'exposant géométrique centralisé (config) si disponible
        try:
            from digital_twin.back_end import config as _cfg  # type: ignore
            _exp = getattr(_cfg, 'FRET_EXPOSANT_GEOMETRIQUE', 8.0)
        except Exception:  # pragma: no cover
            _exp = 8.0
        base = 0.0 if f == 0 else 1.0 - (1.0 / (2 ** (f / _exp)))
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
        """
        positionne le curseur dans la "maison" de la frette (entre deux frettes), proche de la frette cible.
        - fret_index : 0..12 (0 = corde à vide). Pour f>=1, place entre f-1 et f, plus près de f.
        - press_bias : proximité de la frette f (0.0 = collé à f-1, 1.0 = sur la ligne de f).
        """
        fret_index = max(0, min(12, int(fret_index)))
        if fret_index == 0:
            self.set_cursor_pos(0.0)
            return
        prev_frac = self._fret_line_frac(fret_index - 1)
        curr_frac = self._fret_line_frac(fret_index)
        # garantit l'intervalle [0,1]
        press_bias = max(0.0, min(1.0, float(press_bias)))
        frac = prev_frac + (curr_frac - prev_frac) * press_bias
        self.set_cursor_pos(frac)

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
        return QSize(1000, max(180, self.minimumHeight()))

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
        corda_pen = QPen(self._stringColor, 3, Qt.SolidLine)
        painter.setPen(corda_pen)
        start_x = margin + left_pad + dx
        end_x = start_x + int(escala * self._scaleRightFrac)
        painter.drawLine(start_x, center_y, end_x, center_y)

        # 2. dessine les 12 frettes
        num_frettes = 12
        for traste in range(num_frettes + 1):
            # utilise la fraction (modèle 12-TET + ajustements fins)
            frac = self._fret_line_frac(traste)
            x = margin + left_pad + escala * frac
            # sillet (frette 0) plus épais
            if traste == 0:
                traste_pen = QPen(self._nutColor, 4)
            else:
                traste_pen = QPen(self._fretColor, 2)
            painter.setPen(traste_pen)
            painter.drawLine(int(x + dx), center_y - 28, int(x + dx), center_y + 28)
            # ajoute la numérotation des frettes
            if traste > 0:
                painter.setPen(self._fretTextColor)
                font = QFont("Arial", 10)
                painter.setFont(font)
                painter.drawText(int(x + dx) - 8, center_y - 35, str(traste))

        # 3. (pas de marqueurs de note) — seulement frettes et numérotation

        # 4. dessine le curseur principal
        # le curseur ne doit pas dépasser la fin visible de la corde (rendu seulement)
        cursor_vis = min(max(0.0, self.cursor_pos), self._scaleRightFrac)
        cursor_x = margin + left_pad + escala * cursor_vis + dx
        cursor_pen = QPen(self._cursorColor, 4.4, Qt.DashLine)
        painter.setPen(cursor_pen)
        painter.drawLine(int(cursor_x), center_y - 36, int(cursor_x), center_y + 36)

        # 5. dessine le curseur d'attaque (strike) en ligne pleine
        strike_vis = min(max(0.0, self.strike_cursor_pos), self._scaleRightFrac)
        strike_x = margin + left_pad + escala * strike_vis + dx
        strike_pen = QPen(self._strikeCursorColor, 3.8, Qt.SolidLine)
        painter.setPen(strike_pen)
        painter.drawLine(int(strike_x), center_y - 36, int(strike_x), center_y + 36)

        # 6. dessine un marqueur de rosace (cercle) à la position indiquée
        #    utilise la couleur de la corde (stringColor) avec un remplissage translucide
        soundhole_x = margin + left_pad + escala * self._soundholeFrac + dx
        c_fill = QColor(self._stringColor)
        c_fill.setAlpha(36)
        painter.setPen(QPen(self._stringColor, 2))
        painter.setBrush(QBrush(c_fill))
        r = int(self._soundholeRadiusPx)
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
        except Exception:
            pass
        super().mousePressEvent(event)
