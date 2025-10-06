# Digital Twin — Corda 1D FEM

Projeto com GUI PyQt5 acionando uma simulação FEM 1D de uma corda vibrante. Integração é por chamada de função direta (sem HTTP), com resultados padronizados em `digital_twin/results`.

## Estrutura

digital_twin/
- back_end/ — simulação FEM, solvers, integração com GUI e visualização
- front_end/ — GUI PyQt (MainWindow, CordeWidget, estilos) e `main.py`
- results/ — saídas (figures/png, figures/gifs, outputs/csv, outputs/other)
- docs/ — documentação (ex.: `ARCHITECTURE.md`)
- README.md — este arquivo

Veja `docs/ARCHITECTURE.md` para o fluxo completo.

## Como executar a GUI (Windows / PowerShell)

No diretório do repositório (`digital_twin` no caminho), rode:

```powershell
python .\digital_twin\front_end\main.py
```

Ou entre na pasta da GUI e execute:

```powershell
cd .\digital_twin\front_end
python .\main.py
```

Interação:
- Clique na corda ou pressione Espaço para iniciar uma simulação.
- Setas movem o cursor principal (com Shift/Ctrl para pular frettes/centros).
- A/D move o cursor de ataque (Num 4/6 opcional).

## Como rodar a simulação via CLI

No diretório `digital_twin/back_end`:

```powershell
python .\main.py --t-end 1.0 --num-elements 200 --beta 1e-4 --pluck-amp 1e-3
```

Os resultados serão salvos em `digital_twin/results`.

## Onde ficam os resultados

`digital_twin/results/`
- `figures/png/`  → PNGs
- `figures/gifs/` → GIFs/MP4s
- `outputs/csv/`  → `results.csv` e `results_full_nodes.csv`
- `outputs/other/` → `meta.json`

## Gerar GIF a partir de CSV

```powershell
python .\digital_twin\back_end\generate_gif.py --full-nodes \
  .\digital_twin\results\outputs\csv\results_full_nodes.csv \
  --out .\digital_twin\results\figures\gifs\corda.gif --fps 30
```

Também é possível exportar frames com `frames_export.py`.

## Requisitos

Python 3.10+ e pacotes: numpy, PyQt5, pillow, matplotlib (opcional). Instale conforme necessidade do seu ambiente.
