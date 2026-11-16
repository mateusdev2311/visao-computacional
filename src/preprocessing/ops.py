"""Etapa 3 da pipeline: pré-processamento.

Operações usadas e por que cada uma existe neste problema:

1. Redimensionamento mantendo proporção: fotos de celular têm 3000-4000 px;
   detectar contornos nessa resolução é lento e realça textura/ruído do papel.
   Reduzimos para uma altura fixa (padrão 500 px) e guardamos o fator de
   escala para mapear os cantos de volta à imagem original.
2. Conversão para tons de cinza: o contorno do documento é uma diferença de
   brilho (papel claro x fundo); a cor não acrescenta informação útil e
   triplica o custo.
3. CLAHE (equalização adaptativa de histograma): melhora o contraste local em
   fotos com sombra ou iluminação irregular, sem estourar regiões claras.
4. Filtro gaussiano: remove o ruído do sensor e a textura das letras, para
   que o Canny responda à borda da folha e não ao texto.
5. Canny: produz o mapa binário de bordas. Com ``canny_auto`` os limiares são
   derivados da mediana dos pixels (regra de 0,66x e 1,33x a mediana).
6. Fechamento morfológico + dilatação: liga trechos de borda interrompidos
   (cantos dobrados, sombras) para formar contornos fechados.
7. Limiarização de Otsu (caminho alternativo): separa papel claro de fundo
   escuro quando a borda é fraca mas o contraste global é alto.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.config import Parametros


@dataclass
class ResultadoPreprocessamento:
    reduzida: np.ndarray        # BGR na escala de processamento
    escala: float               # fator reduzida/original
    cinza: np.ndarray
    contraste: np.ndarray       # após CLAHE (ou igual a cinza)
    suavizada: np.ndarray
    bordas: np.ndarray          # Canny bruto
    bordas_fechadas: np.ndarray # após morfologia
    limiar: np.ndarray          # Otsu + morfologia
    limiares_canny: tuple[int, int]


def redimensionar_por_altura(img: np.ndarray, altura: int) -> tuple[np.ndarray, float]:
    """Redimensiona mantendo a proporção. Retorna (imagem, escala)."""
    h, w = img.shape[:2]
    escala = altura / float(h)
    if abs(escala - 1.0) < 1e-6:
        return img.copy(), 1.0
    nova_w = max(1, int(round(w * escala)))
    # INTER_AREA é o mais indicado para reduzir (evita serrilhado/moiré).
    interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (nova_w, altura), interpolation=interp), escala


def redimensionar_para_caber(img: np.ndarray, largura_max: int, altura_max: int | None = None) -> tuple[np.ndarray, float]:
    """Encaixa a imagem numa caixa (largura_max x altura_max) sem distorcer."""
    h, w = img.shape[:2]
    escala = largura_max / float(w)
    if altura_max is not None:
        escala = min(escala, altura_max / float(h))
    nova = (max(1, int(round(w * escala))), max(1, int(round(h * escala))))
    interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, nova, interpolation=interp), escala


def limiares_canny_automaticos(img_cinza: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:
    mediana = float(np.median(img_cinza))
    baixo = int(max(0, (1.0 - sigma) * mediana))
    alto = int(min(255, (1.0 + sigma) * mediana))
    if alto <= baixo:
        alto = min(255, baixo + 1)
    return baixo, alto


def preprocessar(img_bgr: np.ndarray, p: Parametros) -> ResultadoPreprocessamento:
    reduzida, escala = redimensionar_por_altura(img_bgr, p.altura_processamento)

    cinza = cv2.cvtColor(reduzida, cv2.COLOR_BGR2GRAY)

    if p.usar_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contraste = clahe.apply(cinza)
    else:
        contraste = cinza

    k = p.kernel_blur
    suavizada = cv2.GaussianBlur(contraste, (k, k), 0) if k > 1 else contraste.copy()

    if p.canny_auto:
        baixo, alto = limiares_canny_automaticos(suavizada)
    else:
        baixo, alto = p.canny_baixo, p.canny_alto
    bordas = cv2.Canny(suavizada, baixo, alto)

    elem = cv2.getStructuringElement(cv2.MORPH_RECT, (p.kernel_morf, p.kernel_morf))
    bordas_fechadas = cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, elem)
    bordas_fechadas = cv2.dilate(bordas_fechadas, elem, iterations=1)

    _, limiar = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Abertura remove pontos claros isolados; fechamento preenche o texto escuro dentro da folha.
    limiar = cv2.morphologyEx(limiar, cv2.MORPH_OPEN, elem)
    limiar = cv2.morphologyEx(limiar, cv2.MORPH_CLOSE, elem, iterations=3)

    return ResultadoPreprocessamento(
        reduzida=reduzida,
        escala=escala,
        cinza=cinza,
        contraste=contraste,
        suavizada=suavizada,
        bordas=bordas,
        bordas_fechadas=bordas_fechadas,
        limiar=limiar,
        limiares_canny=(baixo, alto),
    )
