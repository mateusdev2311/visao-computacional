"""Etapa 6 da pipeline: conversão dos resultados em elementos gráficos.

A imagem final é composta por CAMADAS, desenhadas sempre no sistema de
coordenadas de EXIBIÇÃO (as coordenadas são convertidas pelo
``MapeadorCoordenadas``; os traços ficam com espessura constante na tela,
independentemente da resolução da foto):

  0. base: imagem original redimensionada para a largura de exibição;
  1. escurecimento do fundo fora do documento (foco visual / ROI);
  2. máscara colorida semitransparente sobre o documento (alfa configurável);
  3. candidatos descartados (cinza fino, opcional);
  4. gabarito/ground truth (azul tracejado, quando houver anotação);
  5. contorno escolhido + cantos numerados com coordenadas ORIGINAIS;
  6. painel de informações (HUD) semitransparente e legenda.

Observação: as fontes Hershey do OpenCV não têm acentos, por isso os textos
desenhados sobre a imagem são escritos sem acentuação.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.geometry import MapeadorCoordenadas

# Cores em BGR
VERDE = (60, 200, 60)
AMARELO = (0, 210, 255)
VERMELHO = (60, 60, 230)
AZUL = (230, 140, 30)
CINZA = (170, 170, 170)
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)

ROTULOS_CANTOS = ("SE", "SD", "ID", "IE")  # superior-esq., superior-dir., inferior-dir., inferior-esq.
FONTE = cv2.FONT_HERSHEY_SIMPLEX


def cor_por_confianca(conf: float) -> tuple[int, int, int]:
    if conf >= 0.7:
        return VERDE
    if conf >= 0.5:
        return AMARELO
    return VERMELHO


def misturar_alfa(base: np.ndarray, camada: np.ndarray, mascara: np.ndarray, alfa: float) -> np.ndarray:
    """Composição 'over': saida = alfa*camada + (1-alfa)*base, só onde mascara > 0."""
    misturado = cv2.addWeighted(camada, alfa, base, 1.0 - alfa, 0)
    saida = base.copy()
    cv2.copyTo(misturado, mascara, saida)
    return saida


def escurecer_retangulo(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, alfa: float) -> None:
    """Fundo preto semitransparente aplicado só na região de interesse (ROI) do retângulo."""
    h, w = img.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        roi[:] = cv2.convertScaleAbs(roi, alpha=1.0 - alfa)


def linha_tracejada(img, p1, p2, cor, espessura=2, traco=12):
    p1, p2 = np.array(p1, float), np.array(p2, float)
    comprimento = np.linalg.norm(p2 - p1)
    if comprimento < 1:
        return
    n = int(comprimento // traco)
    for i in range(0, n, 2):
        a = p1 + (p2 - p1) * (i / n)
        b = p1 + (p2 - p1) * (min(i + 1, n) / n)
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), cor, espessura, cv2.LINE_AA)


def texto_com_fundo(img, texto, origem, escala=0.5, cor=BRANCO, espessura=1, alfa=0.6):
    (tw, th), base = cv2.getTextSize(texto, FONTE, escala, espessura)
    x, y = int(origem[0]), int(origem[1])
    x = int(np.clip(x, 3, max(3, img.shape[1] - tw - 6)))
    y = int(np.clip(y, th + 6, img.shape[0] - base - 2))
    escurecer_retangulo(img, x - 3, y - th - 5, x + tw + 3, y + base + 1, alfa)
    cv2.putText(img, texto, (x, y), FONTE, escala, cor, espessura, cv2.LINE_AA)


def painel_info(img, linhas: list[str], origem=(10, 10), escala=0.5, alfa=0.55):
    if not linhas:
        return
    alturas = [cv2.getTextSize(l, FONTE, escala, 1)[0] for l in linhas]
    largura = max(w for w, _ in alturas) + 16
    passo = max(h for _, h in alturas) + 10
    altura = passo * len(linhas) + 8
    x, y = origem
    escurecer_retangulo(img, x, y, x + largura, y + altura, alfa)
    for i, linha in enumerate(linhas):
        cv2.putText(img, linha, (x + 8, y + passo * (i + 1)), FONTE, escala, BRANCO, 1, cv2.LINE_AA)


def legenda(img, itens: list[tuple[str, tuple[int, int, int]]], escala=0.45):
    """Legenda no canto inferior esquerdo: amostra de cor + descrição."""
    if not itens:
        return
    passo = 20
    largura = max(cv2.getTextSize(t, FONTE, escala, 1)[0][0] for t, _ in itens) + 44
    altura = passo * len(itens) + 10
    x, y = 10, img.shape[0] - altura - 10
    escurecer_retangulo(img, x, y, x + largura, y + altura, 0.55)
    for i, (texto, cor) in enumerate(itens):
        yy = y + 8 + passo * i
        cv2.rectangle(img, (x + 8, yy + 2), (x + 26, yy + 14), cor, -1)
        cv2.putText(img, texto, (x + 34, yy + 13), FONTE, escala, BRANCO, 1, cv2.LINE_AA)


def desenhar_resultado(
    original: np.ndarray,
    mapeador: MapeadorCoordenadas,
    cantos_original: np.ndarray | None,
    confianca: float,
    opacidade: float = 0.35,
    candidatos_original: list[np.ndarray] | None = None,
    gabarito_original: np.ndarray | None = None,
    linhas_hud: list[str] | None = None,
    persistido: bool = False,
) -> np.ndarray:
    h, w = original.shape[:2]
    tam = (max(1, int(round(w * mapeador.escala_exibicao))), max(1, int(round(h * mapeador.escala_exibicao))))
    base = cv2.resize(original, tam, interpolation=cv2.INTER_AREA)
    saida = base.copy()
    itens_legenda: list[tuple[str, tuple[int, int, int]]] = []

    if cantos_original is not None:
        cantos = mapeador.original_para_exibicao(cantos_original)
        poligono = np.round(cantos).astype(np.int32)
        cor = AMARELO if persistido else cor_por_confianca(confianca)

        mascara = np.zeros(saida.shape[:2], np.uint8)
        cv2.fillConvexPoly(mascara, poligono, 255)
        # Camada 1: escurece o fundo (fora do documento) -> destaca a região de interesse
        escuro = cv2.convertScaleAbs(saida, alpha=0.55)
        cv2.copyTo(escuro, cv2.bitwise_not(mascara), saida)
        # Camada 2: máscara colorida semitransparente sobre o documento
        saida = misturar_alfa(saida, np.full_like(saida, cor), mascara, opacidade)
        itens_legenda.append(("Documento detectado" + (" (memoria)" if persistido else ""), cor))

    if candidatos_original:
        for cand in candidatos_original:
            pts = np.round(mapeador.original_para_exibicao(cand)).astype(np.int32)
            cv2.polylines(saida, [pts], True, CINZA, 1, cv2.LINE_AA)
        itens_legenda.append(("Candidatos descartados", CINZA))

    if gabarito_original is not None:
        g = mapeador.original_para_exibicao(gabarito_original)
        for i in range(4):
            linha_tracejada(saida, g[i], g[(i + 1) % 4], AZUL, 2)
        itens_legenda.append(("Gabarito (anotacao)", AZUL))

    if cantos_original is not None:
        cv2.polylines(saida, [poligono], True, cor, 3, cv2.LINE_AA)
        for i, (pd, po) in enumerate(zip(cantos, cantos_original)):
            centro = tuple(np.round(pd).astype(int))
            cv2.circle(saida, centro, 8, cor, -1, cv2.LINE_AA)
            cv2.circle(saida, centro, 8, BRANCO, 2, cv2.LINE_AA)
            texto_com_fundo(saida, f"{ROTULOS_CANTOS[i]} ({int(po[0])},{int(po[1])})",
                            (centro[0] + 12, centro[1] - 10), escala=0.45)

    painel_info(saida, linhas_hud or [])
    legenda(saida, itens_legenda)
    return saida


def mapa_de_calor_bordas(base_bgr: np.ndarray, bordas: np.ndarray, alfa: float = 0.55) -> np.ndarray:
    """Densidade de bordas (Canny borrado) em pseudo-cor sobre a imagem."""
    densidade = cv2.GaussianBlur(bordas.astype(np.float32), (0, 0), sigmaX=6)
    densidade = cv2.normalize(densidade, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cor = cv2.applyColorMap(densidade, cv2.COLORMAP_JET)
    return cv2.addWeighted(base_bgr, 1 - alfa, cor, alfa, 0)


def para_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def montar_comparacao(paineis: list[tuple[str, np.ndarray]], altura: int = 320, colunas: int = 2) -> np.ndarray:
    """Grade lado a lado com título; cada painel mantém a proporção (letterbox)."""
    largura_celula = int(altura * 0.85)
    celulas = []
    for titulo, img in paineis:
        img = para_bgr(img)
        h, w = img.shape[:2]
        esc = min(largura_celula / w, (altura - 30) / h)
        red = cv2.resize(img, (max(1, int(w * esc)), max(1, int(h * esc))), interpolation=cv2.INTER_AREA)
        celula = np.full((altura, largura_celula, 3), 35, np.uint8)
        y0 = 30 + (altura - 30 - red.shape[0]) // 2
        x0 = (largura_celula - red.shape[1]) // 2
        celula[y0:y0 + red.shape[0], x0:x0 + red.shape[1]] = red
        cv2.putText(celula, titulo, (8, 21), FONTE, 0.55, BRANCO, 1, cv2.LINE_AA)
        celulas.append(celula)
    while len(celulas) % colunas:
        celulas.append(np.full_like(celulas[0], 35))
    linhas = [np.hstack(celulas[i:i + colunas]) for i in range(0, len(celulas), colunas)]
    grade = np.vstack(linhas)
    # linhas divisórias
    for c in range(1, colunas):
        cv2.line(grade, (c * largura_celula, 0), (c * largura_celula, grade.shape[0]), (80, 80, 80), 2)
    for l in range(1, len(linhas)):
        cv2.line(grade, (0, l * altura), (grade.shape[1], l * altura), (80, 80, 80), 2)
    return grade
