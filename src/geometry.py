"""Geometria e mapeamento de coordenadas.

Trabalhamos com três sistemas de coordenadas (todos em pixels, origem no
canto superior esquerdo, x para a direita e y para baixo):

  * PROCESSAMENTO: imagem reduzida usada pela detecção (ex.: altura 500 px);
  * ORIGINAL: resolução real da foto/quadro;
  * EXIBIÇÃO: tamanho em que a imagem aparece na interface.

Como as duas reduções preservam a proporção, cada mapeamento é uma escala
uniforme  p_destino = p_origem * (s_destino / s_origem), em que s é o fator
"escala em relação à original". O ``MapeadorCoordenadas`` concentra essas
conversões para que nenhum desenho seja feito no sistema errado.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MapeadorCoordenadas:
    escala_processamento: float   # reduzida / original
    escala_exibicao: float        # exibição / original

    def processamento_para_original(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, dtype=np.float32) / self.escala_processamento

    def original_para_exibicao(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, dtype=np.float32) * self.escala_exibicao

    def processamento_para_exibicao(self, pts: np.ndarray) -> np.ndarray:
        return self.original_para_exibicao(self.processamento_para_original(pts))

    def exibicao_para_original(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, dtype=np.float32) / self.escala_exibicao


def ordenar_cantos(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos como [sup-esq, sup-dir, inf-dir, inf-esq].

    Ordena pelo ângulo em torno do centróide (robusto mesmo com o documento
    girado ~45°, caso em que a regra clássica da soma/diferença x+y falha) e
    depois rotaciona a lista para começar no ponto com menor x+y.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    centro = pts.mean(axis=0)
    angulos = np.arctan2(pts[:, 1] - centro[1], pts[:, 0] - centro[0])
    pts = pts[np.argsort(angulos)]          # sentido horário na tela (y para baixo)
    inicio = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -inicio, axis=0)


def tamanho_retificado(cantos: np.ndarray) -> tuple[int, int]:
    """Largura e altura do documento retificado = maiores lados opostos."""
    tl, tr, br, bl = cantos
    largura = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    altura = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return max(1, int(round(largura))), max(1, int(round(altura)))


def matriz_perspectiva(cantos: np.ndarray, largura: int, altura: int) -> np.ndarray:
    """Homografia 3x3 que leva os 4 cantos ao retângulo [0,L]x[0,A]."""
    destino = np.array(
        [[0, 0], [largura - 1, 0], [largura - 1, altura - 1], [0, altura - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(cantos.astype(np.float32), destino)


def retificar(img: np.ndarray, cantos: np.ndarray, rotacao: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Aplica a transformação de perspectiva e, opcionalmente, uma rotação de 90° em 90°."""
    cantos = ordenar_cantos(cantos)
    largura, altura = tamanho_retificado(cantos)
    M = matriz_perspectiva(cantos, largura, altura)
    saida = cv2.warpPerspective(img, M, (largura, altura), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if rotacao in rot:
        saida = cv2.rotate(saida, rot[rotacao])
    return saida, M


def area_poligono(pts: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2))))


def iou_poligonos(a: np.ndarray, b: np.ndarray) -> float:
    """IoU exato entre dois quadriláteros convexos (interseção via OpenCV)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 2)
    area_a, area_b = area_poligono(a), area_poligono(b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    # intersectConvexConvex exige polígonos convexos; o fecho convexo garante isso.
    ha = cv2.convexHull(a)
    hb = cv2.convexHull(b)
    inter, _ = cv2.intersectConvexConvex(ha, hb)
    inter = max(float(inter), 0.0)
    uniao = area_a + area_b - inter
    return inter / uniao if uniao > 0 else 0.0


def erro_medio_cantos(previsto: np.ndarray, real: np.ndarray) -> float:
    """Distância euclidiana média (px) entre cantos correspondentes."""
    p = ordenar_cantos(previsto)
    r = ordenar_cantos(real)
    return float(np.linalg.norm(p - r, axis=1).mean())
