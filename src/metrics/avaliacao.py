"""Execução do protocolo de avaliação sobre o conjunto anotado.

Reaproveitado pelo script ``scripts/avaliar.py`` e pela aba "Métricas" da
interface, garantindo que os números exibidos e os documentados venham do
mesmo código.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src.acquisition import carregar_imagem_arquivo, abrir_video
from src.config import DATA_DIR, Parametros
from src.geometry import erro_medio_cantos, ordenar_cantos
from src.metrics.quality import avaliar_deteccao, resumir
from src.models import SuavizadorCantos

PASTA_IMAGENS = DATA_DIR / "imagens"
PASTA_VIDEOS = DATA_DIR / "videos"


def carregar_anotacoes(pasta: Path = PASTA_IMAGENS) -> dict:
    arq = pasta / "anotacoes.json"
    if not arq.is_file():
        return {}
    return json.loads(arq.read_text(encoding="utf-8"))


def avaliar_imagens(
    p: Parametros,
    pasta: Path = PASTA_IMAGENS,
    ao_processar: Callable[[str, dict, object], None] | None = None,
) -> tuple[list[dict], dict, dict]:
    """Roda a pipeline em todas as imagens anotadas.

    Retorna (linhas, resumo_geral, resumo_por_categoria).
    ``ao_processar(nome, linha, resultado)`` permite salvar imagens ou
    atualizar uma barra de progresso.
    """
    from src.pipeline import processar  # import tardio evita ciclo

    anot = carregar_anotacoes(pasta)
    linhas = []
    for nome, info in anot.items():
        img = carregar_imagem_arquivo(pasta / nome)
        gab = None if info["cantos"] is None else np.array(info["cantos"], np.float32)
        r = processar(img, copy.deepcopy(p), gabarito=gab, retificar_saida=False)
        h, w = img.shape[:2]
        linha = {
            "imagem": nome,
            "categoria": info["categoria"],
            "descricao": info["descricao"],
            "confianca": round(r.deteccao.confianca, 3),
            "metodo": r.deteccao.candidato.origem if r.deteccao.candidato else "-",
            # tempo de análise (sem o desenho da sobreposição, que só existe para a interface)
            "tempo_ms": r.tempos_ms["total"] - r.tempos_ms.get("visualizacao", 0.0),
        }
        linha.update(avaliar_deteccao(r.cantos_original, gab, float(np.hypot(w, h))))
        linhas.append(linha)
        if ao_processar:
            ao_processar(nome, linha, r)

    por_categoria = {}
    for cat in sorted({l["categoria"] for l in linhas}):
        por_categoria[cat] = resumir([l for l in linhas if l["categoria"] == cat])
    return linhas, resumir(linhas), por_categoria


def avaliar_video(p: Parametros, alfa: float | None = None,
                  nome: str = "documento_movendo.mp4", pasta: Path = PASTA_VIDEOS) -> dict:
    """Mede erro, tremor (jitter) e FPS da detecção quadro a quadro.

    Tremor = média de || (c_t - c_{t-1}) - (g_t - g_{t-1}) || : o deslocamento
    dos cantos que NÃO é explicado pelo movimento real do documento.
    """
    from src.pipeline import processar

    anot_arq = pasta / "anotacoes_video.json"
    gab_quadros = json.loads(anot_arq.read_text(encoding="utf-8"))[nome]["cantos_por_quadro"]
    p = copy.deepcopy(p)
    if alfa is not None:
        p.suavizacao = alfa
    suav = SuavizadorCantos(p.suavizacao, p.quadros_persistencia)
    cap, _ = abrir_video(pasta / nome)

    erros, tremores, tempos = [], [], []
    detectados = persistidos = 0
    anterior = anterior_gab = None
    t = 0
    while True:
        ok, quadro = cap.read()
        if not ok or t >= len(gab_quadros):
            break
        gab = ordenar_cantos(np.array(gab_quadros[t], np.float32))
        r = processar(quadro, p, suavizador=suav, retificar_saida=False)
        tempos.append(r.tempos_ms["total"])
        if r.cantos_original is not None:
            detectados += 1
            persistidos += int(r.persistido)
            c = ordenar_cantos(r.cantos_original)
            erros.append(erro_medio_cantos(c, gab))
            if anterior is not None:
                tremores.append(float(np.linalg.norm((c - anterior) - (gab - anterior_gab), axis=1).mean()))
            anterior, anterior_gab = c, gab
        else:
            anterior = anterior_gab = None
        t += 1
    cap.release()
    tempo_medio = float(np.mean(tempos)) if tempos else 0.0
    return {
        "alfa": p.suavizacao,
        "quadros": t,
        "taxa_deteccao": detectados / t if t else 0.0,
        "quadros_persistidos": persistidos,
        "erro_canto_medio_px": float(np.mean(erros)) if erros else None,
        "tremor_medio_px": float(np.mean(tremores)) if tremores else None,
        "tempo_medio_ms": tempo_medio,
        "fps_processamento": 1000.0 / tempo_medio if tempo_medio else None,
    }


def salvar_csv(linhas: list[dict], caminho: Path) -> None:
    import csv
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        for l in linhas:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in l.items()})


def salvar_imagem(caminho: Path, img: np.ndarray, largura: int = 640) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    if w > largura:
        img = cv2.resize(img, (largura, int(h * largura / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(caminho.suffix, img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if ok:
        caminho.write_bytes(buf.tobytes())
