"""Orquestração da pipeline completa para uma imagem ou um quadro de vídeo.

aquisição -> validação -> pré-processamento -> detecção -> interpretação
-> retificação (perspectiva) -> representação gráfica -> métricas
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.config import Parametros
from src.geometry import MapeadorCoordenadas, retificar
from src.metrics import Cronometro, nitidez
from src.models import ResultadoDeteccao, SuavizadorCantos, detectar_documento, refinar_cantos
from src.preprocessing import ResultadoPreprocessamento, preprocessar
from src.visualization import desenhar_resultado


@dataclass
class ResultadoPipeline:
    original: np.ndarray
    prep: ResultadoPreprocessamento
    deteccao: ResultadoDeteccao
    mapeador: MapeadorCoordenadas
    cantos_original: np.ndarray | None          # após suavização (se houver)
    cantos_brutos_original: np.ndarray | None   # medidos neste quadro
    persistido: bool
    retificada: np.ndarray | None
    scan: np.ndarray | None
    sobreposicao: np.ndarray
    tempos_ms: dict = field(default_factory=dict)
    nitidez: float | None = None

    @property
    def encontrado(self) -> bool:
        return self.cantos_original is not None


def aplicar_modo_saida(img: np.ndarray, modo: str) -> np.ndarray:
    """Pós-processamento do documento retificado."""
    if modo == "cinza":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if modo == "pb":
        cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cinza = cv2.GaussianBlur(cinza, (3, 3), 0)
        bloco = max(15, (min(cinza.shape[:2]) // 40) | 1)  # janela ímpar proporcional ao tamanho
        return cv2.adaptiveThreshold(cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, bloco, 12)
    return img


def processar(
    img: np.ndarray,
    p: Parametros,
    suavizador: SuavizadorCantos | None = None,
    gabarito: np.ndarray | None = None,
    linhas_extra_hud: list[str] | None = None,
    retificar_saida: bool = True,
) -> ResultadoPipeline:
    p.validar()
    cron = Cronometro()

    with cron.etapa("pre_processamento"):
        prep = preprocessar(img, p)

    with cron.etapa("deteccao"):
        det = detectar_documento(prep, p)

    h, w = img.shape[:2]
    escala_exib = min(1.0, p.largura_exibicao / float(w)) if w > 0 else 1.0
    mapeador = MapeadorCoordenadas(escala_processamento=prep.escala, escala_exibicao=escala_exib)

    brutos = mapeador.processamento_para_original(det.cantos_processamento) if det.encontrado else None
    if brutos is not None and p.refinar_cantos:
        with cron.etapa("refinamento"):
            cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brutos = refinar_cantos(cinza, brutos, prep.escala)
    persistido = False
    cantos = brutos
    if suavizador is not None:
        with cron.etapa("suavizacao"):
            cantos, persistido = suavizador.atualizar(brutos, diagonal=float(np.hypot(w, h)))

    retificada = scan = None
    valor_nitidez = None
    if cantos is not None and retificar_saida:
        with cron.etapa("retificacao"):
            retificada, _ = retificar(img, cantos, rotacao=p.rotacao_saida)
            scan = aplicar_modo_saida(retificada, p.modo_saida)
            valor_nitidez = nitidez(retificada)

    tempo_ate_aqui = cron.total_ms
    hud = []
    if cantos is not None:
        status = "MEMORIA (deteccao falhou neste quadro)" if persistido else "DOCUMENTO ENCONTRADO"
        hud.append(status)
        hud.append(f"Confianca: {det.confianca:.2f}  Metodo: {det.candidato.origem if det.candidato else '-'}")
    else:
        hud.append("NENHUM DOCUMENTO")
        if det.candidato is not None:
            hud.append(f"Melhor candidato: {det.confianca:.2f} (< {p.confianca_minima:.2f})")
    hud.append(f"Tempo: {tempo_ate_aqui:.1f} ms  |  Original: {w}x{h}")
    hud.extend(linhas_extra_hud or [])

    with cron.etapa("visualizacao"):
        candidatos = None
        if p.mostrar_candidatos and det.candidatos:
            escolhidos = det.candidato
            candidatos = [mapeador.processamento_para_original(c.cantos)
                          for c in det.candidatos if c is not escolhidos]
        sobreposicao = desenhar_resultado(
            img, mapeador, cantos, det.confianca,
            opacidade=p.opacidade_mascara,
            candidatos_original=candidatos,
            gabarito_original=gabarito,
            linhas_hud=hud,
            persistido=persistido,
        )

    tempos = dict(cron.tempos_ms)
    tempos["total"] = cron.total_ms
    return ResultadoPipeline(
        original=img, prep=prep, deteccao=det, mapeador=mapeador,
        cantos_original=cantos, cantos_brutos_original=brutos, persistido=persistido,
        retificada=retificada, scan=scan, sobreposicao=sobreposicao,
        tempos_ms=tempos, nitidez=valor_nitidez,
    )
