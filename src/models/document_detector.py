"""Etapa 4 e 5 da pipeline: detecção do documento e interpretação da saída.

Método (visão computacional clássica, sem rede neural):

1. Gerar máscaras binárias candidatas a partir do pré-processamento:
   - "bordas": mapa de Canny fechado por morfologia;
   - "limiar": segmentação de Otsu (papel claro x fundo).
2. Extrair contornos (cv2.findContours) e, para os maiores, aproximar por um
   polígono (Douglas-Peucker, cv2.approxPolyDP). Um documento visto em
   perspectiva é um quadrilátero convexo.
3. Se o contorno não virar 4 vértices (canto ocluído, borda curva), tentamos
   o fecho convexo com tolerância crescente; o candidato fica marcado como
   "aproximado" e recebe penalidade.
4. Pontuar cada quadrilátero com uma confiança em [0, 1]:
     confiança = (0,55·suporte_de_borda + 0,25·contraste + 0,20·área) · penalidades
   - suporte_de_borda: fração de pontos amostrados sobre os 4 lados que caem
     em pixels de borda do Canny (o lado "existe" na imagem?);
   - contraste: papel mais claro que a vizinhança imediata (anel externo);
   - área: documentos tendem a ocupar boa parte da foto;
   - penalidades: ângulos internos implausíveis, quadrilátero colado na moldura
     da imagem inteira, candidato aproximado.
5. O melhor candidato acima de ``confianca_minima`` é a resposta. Os cantos são
   ordenados (sup-esq, sup-dir, inf-dir, inf-esq) e mapeados para a resolução
   original.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.config import Parametros
from src.geometry import ordenar_cantos, area_poligono
from src.preprocessing import ResultadoPreprocessamento


@dataclass
class Candidato:
    cantos: np.ndarray            # 4x2, coordenadas de PROCESSAMENTO
    origem: str                   # "bordas" | "limiar"
    aproximado: bool
    confianca: float = 0.0
    componentes: dict = field(default_factory=dict)


@dataclass
class ResultadoDeteccao:
    encontrado: bool
    cantos_processamento: np.ndarray | None
    confianca: float
    candidato: Candidato | None
    candidatos: list[Candidato]
    mensagem: str


def _amostrar_lados(cantos: np.ndarray, pontos_por_lado: int = 60) -> np.ndarray:
    amostras = []
    for i in range(4):
        a, b = cantos[i], cantos[(i + 1) % 4]
        # Ignora 10% perto de cada canto: cantos costumam estar dobrados/sombreados.
        t = np.linspace(0.1, 0.9, pontos_por_lado)[:, None]
        amostras.append(a + (b - a) * t)
    return np.vstack(amostras)


def _angulos_internos(cantos: np.ndarray) -> np.ndarray:
    angs = []
    for i in range(4):
        p0, p1, p2 = cantos[i - 1], cantos[i], cantos[(i + 1) % 4]
        v1, v2 = p0 - p1, p2 - p1
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        angs.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    return np.array(angs)


def _pontuar(c: Candidato, prep: ResultadoPreprocessamento, bordas_dilatadas: np.ndarray) -> None:
    h, w = prep.cinza.shape[:2]
    area_img = float(h * w)
    cantos = c.cantos

    # Suporte de borda
    pts = np.round(_amostrar_lados(cantos)).astype(int)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    suporte = float((bordas_dilatadas[pts[:, 1], pts[:, 0]] > 0).mean())

    # Contraste: média dentro x média num anel logo fora do quadrilátero
    mascara = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mascara, cantos.astype(np.int32), 255)
    anel = cv2.dilate(mascara, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))) & ~mascara
    dentro = cv2.mean(prep.cinza, mask=mascara)[0]
    fora = cv2.mean(prep.cinza, mask=anel)[0] if anel.any() else dentro
    contraste = float(np.clip(abs(dentro - fora) / 60.0, 0.0, 1.0))

    frac_area = area_poligono(cantos) / area_img
    termo_area = float(min(1.0, frac_area / 0.5))

    penalidade = 1.0
    angulos = _angulos_internos(cantos)
    if angulos.min() < 45 or angulos.max() > 135:
        penalidade *= 0.5
    if frac_area > 0.97:                 # é a moldura da própria imagem
        penalidade *= 0.3
    if c.aproximado:
        penalidade *= 0.85

    c.confianca = float((0.55 * suporte + 0.25 * contraste + 0.20 * termo_area) * penalidade)
    c.componentes = {
        "suporte_borda": round(suporte, 3),
        "contraste": round(contraste, 3),
        "fracao_area": round(frac_area, 3),
        "angulo_min": round(float(angulos.min()), 1),
        "angulo_max": round(float(angulos.max()), 1),
        "penalidade": round(penalidade, 3),
    }


def _quadrilateros(mascara: np.ndarray, origem: str, p: Parametros) -> list[Candidato]:
    h, w = mascara.shape[:2]
    area_min = p.area_minima * h * w
    contornos, _ = cv2.findContours(mascara, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:10]

    candidatos: list[Candidato] = []
    for cnt in contornos:
        if cv2.contourArea(cnt) < area_min:
            break  # lista ordenada: os demais são menores
        perimetro = cv2.arcLength(cnt, True)
        aprox = cv2.approxPolyDP(cnt, p.epsilon_aprox * perimetro, True)
        if len(aprox) == 4 and cv2.isContourConvex(aprox):
            candidatos.append(Candidato(ordenar_cantos(aprox), origem, aproximado=False))
            continue
        # Plano B: fecho convexo com tolerância crescente
        fecho = cv2.convexHull(cnt)
        peri_fecho = cv2.arcLength(fecho, True)
        for eps in np.arange(p.epsilon_aprox, 0.081, 0.01):
            aprox = cv2.approxPolyDP(fecho, eps * peri_fecho, True)
            if len(aprox) == 4:
                if cv2.contourArea(aprox) >= area_min:
                    candidatos.append(Candidato(ordenar_cantos(aprox), origem, aproximado=True))
                break
            if len(aprox) < 4:
                break
    return candidatos


def _remover_duplicados(candidatos: list[Candidato], tol: float) -> list[Candidato]:
    unicos: list[Candidato] = []
    for c in sorted(candidatos, key=lambda c: c.confianca, reverse=True):
        if all(np.abs(c.cantos - u.cantos).max() > tol for u in unicos):
            unicos.append(c)
    return unicos


def refinar_cantos(cinza_original: np.ndarray, cantos: np.ndarray, escala_processamento: float) -> np.ndarray:
    """Refinamento subpixel dos cantos na resolução ORIGINAL (cv2.cornerSubPix).

    A detecção acontece na imagem reduzida e sobre bordas dilatadas, então o
    canto pode estar alguns pixels fora do lugar após voltar à escala original.
    O cornerSubPix busca, numa janela ao redor de cada canto, o ponto em que os
    gradientes da imagem convergem. A janela cresce com o fator de ampliação
    (1/escala). Se algum canto "fugir" mais que a janela, mantemos o original.
    """
    janela = int(np.clip(round(4.0 / max(escala_processamento, 1e-6)), 5, 25))
    pts = np.ascontiguousarray(cantos, dtype=np.float32).reshape(-1, 1, 2).copy()
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    refinados = cv2.cornerSubPix(cinza_original, pts, (janela, janela), (-1, -1), criterio).reshape(4, 2)
    deslocamento = np.linalg.norm(refinados - cantos, axis=1)
    return np.where((deslocamento <= janela)[:, None], refinados, cantos).astype(np.float32)


def detectar_documento(prep: ResultadoPreprocessamento, p: Parametros) -> ResultadoDeteccao:
    mascaras = []
    if p.metodo in ("auto", "bordas"):
        mascaras.append((prep.bordas_fechadas, "bordas"))
    if p.metodo in ("auto", "limiar"):
        mascaras.append((prep.limiar, "limiar"))

    candidatos: list[Candidato] = []
    for mascara, origem in mascaras:
        candidatos.extend(_quadrilateros(mascara, origem, p))

    if not candidatos:
        return ResultadoDeteccao(
            False, None, 0.0, None, [],
            "Nenhum contorno com 4 lados e área suficiente foi encontrado. "
            "Tente reduzir a 'área mínima', aumentar o fechamento morfológico "
            "ou fotografar o documento sobre um fundo que contraste com o papel.",
        )

    bordas_dilatadas = cv2.dilate(prep.bordas, np.ones((5, 5), np.uint8))
    for c in candidatos:
        _pontuar(c, prep, bordas_dilatadas)
    candidatos = _remover_duplicados(candidatos, tol=0.02 * prep.cinza.shape[0])
    melhor = candidatos[0]

    if melhor.confianca < p.confianca_minima:
        return ResultadoDeteccao(
            False, None, melhor.confianca, melhor, candidatos,
            f"Melhor candidato com confiança {melhor.confianca:.2f}, abaixo do mínimo "
            f"de {p.confianca_minima:.2f}. Resultado descartado para não exibir uma "
            "detecção pouco confiável.",
        )
    return ResultadoDeteccao(
        True, melhor.cantos, melhor.confianca, melhor, candidatos,
        f"Documento encontrado (método: {melhor.origem}"
        f"{', aproximado' if melhor.aproximado else ''}).",
    )
