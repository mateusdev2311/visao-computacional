# Pipeline técnica

Este documento detalha cada etapa, onde ela está no código e por que foi escolhida.
Para instalação e uso, veja o [README](../README.md).

```
 ┌───────────┐   ┌────────────┐   ┌──────────────────┐   ┌───────────┐   ┌────────────────┐
 │ Aquisição │──▶│ Validação  │──▶│ Pré-processamento│──▶│ Detecção  │──▶│ Interpretação  │
 │ imagem,   │   │ extensão,  │   │ resize, cinza,   │   │ contornos,│   │ ordenar cantos,│
 │ vídeo,    │   │ tamanho,   │   │ CLAHE, gauss,    │   │ approxPoly│   │ mapear p/ orig,│
 │ câmera    │   │ decodif.   │   │ Canny, morfol.,  │   │ pontuação │   │ subpixel, EMA  │
 └───────────┘   └────────────┘   │ Otsu             │   └───────────┘   └───────┬────────┘
                                  └──────────────────┘                           │
 ┌───────────┐   ┌───────────────────────────┐   ┌──────────────────────────────▼───────┐
 │ Métricas  │◀──│ Interface (Streamlit)     │◀──│ Representação gráfica + perspectiva   │
 │ IoU, erro,│   │ entrada, parâmetros,      │   │ camadas, máscara alfa, contorno,      │
 │ FP/FN, ms,│   │ resultados, etapas,       │   │ cantos, HUD, legenda, retificação,    │
 │ FPS,tremor│   │ comparação, métricas      │   │ mapa de calor, comparação lado a lado │
 └───────────┘   └───────────────────────────┘   └──────────────────────────────────────┘
```

## 1. Aquisição — `src/acquisition/loader.py`

| Fonte | Função | Observação |
|---|---|---|
| Imagem (arquivo/upload) | `carregar_imagem_bytes`, `carregar_imagem_arquivo` | `cv2.imdecode` (aceita caminhos com acento no Windows) |
| Foto do navegador | `st.camera_input` + `carregar_imagem_bytes` | câmera do navegador |
| Vídeo | `abrir_video` | upload é salvo num arquivo temporário apagado ao final |
| Webcam ao vivo | `abrir_camera` | backend DirectShow no Windows; confirma que entrega quadros |

## 2. Validação

Gera `ErroEntrada` (mensagem em português, mostrada pela interface) para: arquivo vazio, maior que
25 MB, extensão não suportada, conteúdo que não decodifica, imagem menor que 64 px ou maior que
8000 px, vídeo sem quadros legíveis, câmera inexistente ou ocupada. Imagens em tons de cinza e
com canal alfa são convertidas para BGR. Parâmetros fora da faixa são corrigidos por
`Parametros.validar()` (ex.: kernel par vira ímpar).

## 3. Pré-processamento — `src/preprocessing/ops.py`

| # | Operação | Parâmetro na interface | Justificativa |
|---|---|---|---|
| 1 | Redimensionamento para altura fixa, mantendo proporção (`INTER_AREA`) | Altura de processamento | Velocidade; reduz a textura do texto; escala guardada para voltar à resolução original |
| 2 | Conversão BGR → cinza | — | O contorno do papel é diferença de brilho; cor não ajuda |
| 3 | CLAHE (clip 2,0, grade 8×8) | Equalizar contraste | Sombras e iluminação irregular |
| 4 | Filtro gaussiano | Kernel do filtro | Remove ruído e "apaga" as letras antes do Canny |
| 5 | Canny (limiares automáticos = 0,67× e 1,33× a mediana, ou manuais) | Limiares / automático | Mapa de bordas |
| 6 | Fechamento + dilatação morfológica | Kernel morfológico | Liga bordas interrompidas → contornos fechados |
| 7 | Limiarização de Otsu + abertura/fechamento | Método "limiar" ou "auto" | Caminho alternativo quando a borda é fraca mas o papel é claro |

A comparação em `docs/resultados/RESULTADOS.md` mostra o efeito de desligar o CLAHE e de usar só
um dos caminhos (bordas ou limiar).

## 4. Detecção — `src/models/document_detector.py`

1. `cv2.findContours` nas máscaras de bordas e/ou limiar; os 10 maiores contornos são analisados.
2. `cv2.approxPolyDP` com tolerância = `epsilon × perímetro`. Quadrilátero convexo → candidato.
3. Senão, fecho convexo com tolerância crescente (até 0,08) → candidato **aproximado** (útil com
   canto ocluído).
4. Pontuação de confiança em [0, 1]:

   `confiança = (0,55·suporte_de_borda + 0,25·contraste + 0,20·área) × penalidades`

   * **suporte de borda**: fração de 240 pontos amostrados nos 4 lados (ignorando 10% perto dos
     cantos) que caem sobre bordas do Canny;
   * **contraste**: |média dentro − média num anel de 15 px fora| / 60, limitado a 1;
   * **área**: fração da imagem / 0,5, limitado a 1;
   * **penalidades**: ×0,5 se algum ângulo interno < 45° ou > 135°; ×0,3 se ocupa > 97% da imagem
     (é a moldura da foto); ×0,85 se aproximado.
5. Remoção de duplicados; o melhor candidato é aceito se `confiança ≥ confiança mínima`.

Por que não uma rede neural? Documento é um objeto de geometria simples (quadrilátero claro).
O método clássico é explicável etapa a etapa, roda em ~20 ms na CPU sem GPU nem download de pesos,
e permite mostrar ao usuário *por que* um candidato foi aceito ou descartado (tabela de candidatos).

## 5. Interpretação das saídas

* **Ordenação dos cantos** (`geometry.ordenar_cantos`): ordena por ângulo em torno do centróide e
  começa no ponto com menor x+y → [sup-esq, sup-dir, inf-dir, inf-esq]. Funciona mesmo com o
  documento girado 45°, caso em que a regra clássica "soma/diferença" falha.
* **Mapeamento de coordenadas** (`geometry.MapeadorCoordenadas`): processamento → original →
  exibição, por escalas uniformes.
* **Refinamento subpixel** (`refinar_cantos`, `cv2.cornerSubPix`) na imagem original, com janela
  proporcional à ampliação (1/escala). Reduziu o erro médio de canto de 0,54% para 0,29% da
  diagonal (≈ 5 px → 0,3 px na mediana).
* **Vídeo** (`models/smoothing.py`): média móvel exponencial dos cantos, reinício em saltos
  grandes e persistência por N quadros quando a detecção falha.

## 6. Representação gráfica — `src/visualization/draw.py`, `src/geometry.py`

Camadas da imagem de resultado (desenhadas no espaço de exibição, para traços com espessura
constante na tela):

0. imagem original redimensionada para a largura de exibição;
1. fundo fora do documento escurecido (destaca a região de interesse);
2. máscara colorida semitransparente (`addWeighted` + `copyTo` com máscara); a cor codifica a
   confiança: verde ≥ 0,7, amarelo ≥ 0,5, vermelho abaixo; amarelo também indica "memória" no vídeo;
3. candidatos descartados em cinza (opcional);
4. gabarito em azul tracejado (amostras anotadas);
5. contorno, cantos numerados (SE, SD, ID, IE) com coordenadas da imagem **original**;
6. painel de informações (HUD) e legenda semitransparentes (escurecimento só na ROI do painel).

Além disso: **transformação de perspectiva** (`getPerspectiveTransform` + `warpPerspective`), com
tamanho de saída igual aos maiores lados opostos; **rotação** de 90° em 90°; **mapa de calor** da
densidade de bordas (`applyColorMap`); **grade comparativa** original | bordas | detecção |
retificado, mantendo a proporção de cada painel (letterbox).

## 7. Métricas — `src/metrics/`

| Tipo | Métrica | Onde |
|---|---|---|
| Qualidade | IoU do quadrilátero (interseção exata de polígonos convexos) | `geometry.iou_poligonos` |
| Qualidade | Erro médio de canto (px e % da diagonal) | `geometry.erro_medio_cantos` |
| Qualidade | Acertos (IoU ≥ 0,90), imprecisos, FN, FP, VN, precisão, recall | `metrics/quality.py` |
| Qualidade | Nitidez do resultado (variância do Laplaciano) | `metrics/quality.py` |
| Qualidade (vídeo) | Tremor: deslocamento dos cantos não explicado pelo movimento real | `metrics/avaliacao.py` |
| Desempenho | Tempo por etapa (ms), média e p95 | `metrics/performance.py` |
| Desempenho | FPS (janela deslizante de 30 quadros) | `metrics/performance.py` |

## Itens de Computação Gráfica atendidos (exigidos: 4)

| Item do enunciado | Implementação |
|---|---|
| Mapeamento entre coordenadas do modelo, da imagem original e da interface | `MapeadorCoordenadas`; cantos exibidos com coordenadas originais |
| Redimensionamento e manutenção da proporção | `redimensionar_por_altura`, `redimensionar_para_caber`, letterbox da comparação |
| Translação, rotação, escala ou perspectiva | homografia de retificação, rotação da saída, escalas entre espaços |
| Composição por camadas | 7 camadas na imagem de resultado |
| Cores, transparência, máscaras ou mapas de calor | máscara alfa, cor por confiança, mapa de calor de bordas |
| Desenho de contornos, pontos, linhas | contorno, cantos, linhas tracejadas do gabarito, legenda |
| Atualização em tempo real | vídeo e webcam quadro a quadro com FPS |
| Suavização/estabilização de coordenadas | EMA + persistência (`SuavizadorCantos`) |
| Seleção e recorte de regiões de interesse | retificação recorta o documento; HUD escurece só a ROI |
| Comparação original / processada / resultado | aba "Comparação lado a lado" e aba "Etapas" |
