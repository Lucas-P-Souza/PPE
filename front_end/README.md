# Front-end (PyQt5) — Digital Twin

GUI para interagir com a simulação FEM 1D da corda.

## Entrar e executar

```powershell
cd .\digital_twin\front_end
python .\main.py
```

## Interação
- Clique na corda ou pressione Espaço para iniciar uma simulação.
- Setas movem o cursor principal (Shift: pular frette; Ctrl: pular centro de casa).
- A/D movem o cursor de ataque (ou Num 4/6 com keypad).

## Integração com o back-end
- A GUI usa `digital_twin.back_end.integration.SimulationController`.
- A integração é por chamada de função direta, em thread de background.
- As posições de “nota” e “ataque” são convertidas de % para metros (pos = percent/100 * length).
- Saídas são salvas em `digital_twin/results` (PNGs, GIFs, CSVs e meta.json).
