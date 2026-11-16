# Resultados da avaliação

Gerado automaticamente por `python scripts/avaliar.py` com os parâmetros padrão (`src/config.py`). Acerto = IoU ≥ 0,90 com o gabarito.

## Desempenho por categoria (imagens)

| Categoria | Imagens | Acertos | Imprecisos | FN | FP | IoU médio | Erro de canto (% diag.) | Tempo médio (ms) |
|---|---|---|---|---|---|---|---|---|
| dificil | 6 | 4 | 1 | 1 | 0 | 0.787 | 1.22 | 37.1 |
| facil | 12 | 12 | 0 | 0 | 0 | 0.996 | 0.06 | 26.3 |
| medio | 8 | 8 | 0 | 0 | 0 | 0.998 | 0.03 | 30.7 |
| negativo | 4 | 0 | 0 | 0 | 0 | - | - | 41.8 |
| **TOTAL** | 30 | 24 | 1 | 1 | 0 | 0.948 | 0.29 | 31.7 |

* Taxa de acerto (recall @IoU 0,90): **92.3%**
* Precisão: **96.0%**
* Tempo p95: **58.1 ms** por imagem (pré-processamento + detecção)

## Comparação entre métodos

| Configuração | Acertos | Imprecisos | FN | FP | IoU médio | Erro de canto (% diag.) | Tempo médio (ms) |
|---|---|---|---|---|---|---|---|
| bordas | 24 | 0 | 2 | 0 | 0.912 | 0.17 | 20.4 |
| limiar | 23 | 2 | 1 | 0 | 0.935 | 0.64 | 13.8 |
| auto | 24 | 1 | 1 | 0 | 0.948 | 0.29 | 19.0 |
| auto sem CLAHE | 23 | 1 | 2 | 0 | 0.911 | 0.29 | 17.2 |
| auto sem refinamento subpixel | 24 | 1 | 1 | 0 | 0.931 | 0.54 | 19.9 |
| auto com área mínima 5% | 25 | 1 | 0 | 0 | 0.987 | 0.28 | 21.7 |

## Vídeo: efeito da suavização (EMA)

| Alfa | Taxa de detecção | Quadros mantidos em memória | Erro médio (px) | Tremor médio (px) | FPS de processamento |
|---|---|---|---|---|---|
| 1.0 | 100.0% | 9 | 3.32 | 2.41 | 36.0 |
| 0.7 | 100.0% | 9 | 5.52 | 2.05 | 34.0 |
| 0.5 | 100.0% | 9 | 8.54 | 1.91 | 34.8 |
| 0.3 | 100.0% | 9 | 14.69 | 1.93 | 35.5 |

## Resultado por imagem

| Imagem | Categoria | Classe | Confiança | IoU | Erro (px) | Tempo (ms) |
|---|---|---|---|---|---|---|
| facil_01.jpg | facil | VP | 0.94 | 0.999 | 0.3 | 29.8 |
| facil_02.jpg | facil | VP | 0.89 | 0.998 | 0.3 | 34.0 |
| facil_03.jpg | facil | VP | 0.89 | 0.999 | 0.2 | 47.2 |
| facil_04.jpg | facil | VP | 0.90 | 0.999 | 0.1 | 18.5 |
| facil_05.jpg | facil | VP | 0.98 | 0.999 | 0.3 | 25.1 |
| facil_06.jpg | facil | VP | 0.91 | 0.999 | 0.3 | 21.2 |
| facil_07.jpg | facil | VP | 0.95 | 0.999 | 0.2 | 17.5 |
| facil_08.jpg | facil | VP | 0.93 | 1.000 | 0.2 | 24.2 |
| facil_09.jpg | facil | VP | 0.91 | 0.999 | 0.2 | 18.8 |
| facil_10.jpg | facil | VP | 0.92 | 0.999 | 0.3 | 38.9 |
| facil_11.jpg | facil | VP | 0.87 | 0.958 | 9.7 | 14.4 |
| facil_12.jpg | facil | VP | 0.95 | 0.999 | 0.2 | 26.3 |
| medio_01.jpg | medio | VP | 0.96 | 0.999 | 0.2 | 38.1 |
| medio_02.jpg | medio | VP | 0.86 | 0.999 | 0.4 | 33.1 |
| medio_03.jpg | medio | VP | 0.89 | 0.999 | 0.3 | 29.0 |
| medio_04.jpg | medio | VP | 0.88 | 0.999 | 0.2 | 30.7 |
| medio_05.jpg | medio | VP | 0.91 | 0.998 | 0.4 | 30.2 |
| medio_06.jpg | medio | VP | 0.84 | 0.994 | 2.2 | 24.8 |
| medio_07.jpg | medio | VP | 0.88 | 0.998 | 0.3 | 33.4 |
| medio_08.jpg | medio | VP | 0.97 | 0.999 | 0.3 | 26.4 |
| dificil_01_baixo_contraste.jpg | dificil | VP | 0.88 | 0.999 | 0.4 | 25.3 |
| dificil_02_canto_ocluido.jpg | dificil | VP | 0.69 | 0.931 | 17.1 | 28.3 |
| dificil_03_tremida.jpg | dificil | VP | 0.84 | 0.956 | 10.9 | 26.3 |
| dificil_04_cortado.jpg | dificil | VP_impreciso | 0.63 | 0.835 | 68.7 | 21.0 |
| dificil_05_sombra_forte.jpg | dificil | VP | 0.94 | 0.999 | 0.3 | 79.5 |
| dificil_06_pequeno.jpg | dificil | FN | 0.00 | 0.000 | - | 42.3 |
| negativo_01.jpg | negativo | VN | 0.17 | - | - | 34.0 |
| negativo_02.jpg | negativo | VN | 0.00 | - | - | 50.9 |
| negativo_03.jpg | negativo | VN | 0.12 | - | - | 64.0 |
| negativo_04.jpg | negativo | VN | 0.00 | - | - | 18.4 |

Legenda: VP = acerto; VP_impreciso = detectou, mas IoU < 0,90; FN = não detectou; FP = detectou documento onde não há; VN = corretamente não detectou.
