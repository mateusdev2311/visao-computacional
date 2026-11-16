"""Métricas de qualidade da detecção.

* IoU (Intersection over Union) entre o quadrilátero detectado e o gabarito:
  1,0 = sobreposição perfeita. Consideramos ACERTO quando IoU >= 0,90
  (limiar mais rígido que o 0,5 usual em detecção de objetos, porque um
  canto deslocado já distorce o documento retificado).
* Erro médio de canto: distância média (px) entre cada canto previsto e o
  canto real. Também reportado como % da diagonal da imagem, para comparar
  fotos de resoluções diferentes.
* Falso negativo: havia documento e nada foi detectado.
* Falso positivo: não havia documento (imagem negativa) e algo foi detectado.
* Nitidez: variância do Laplaciano do documento retificado (quanto maior,
  mais nítido; útil para avisar o usuário de fotos tremidas).
"""
from __future__ import annotations

import cv2
import numpy as np

from src.geometry import iou_poligonos, erro_medio_cantos

LIMIAR_ACERTO_IOU = 0.90


def avaliar_deteccao(previsto: np.ndarray | None, gabarito: np.ndarray | None, diagonal: float) -> dict:
    """Compara uma detecção com o gabarito e classifica o resultado."""
    if gabarito is None:
        return {
            "classe": "VN" if previsto is None else "FP",
            "iou": None, "erro_canto_px": None, "erro_canto_pct": None,
        }
    if previsto is None:
        return {"classe": "FN", "iou": 0.0, "erro_canto_px": None, "erro_canto_pct": None}
    iou = iou_poligonos(previsto, gabarito)
    erro = erro_medio_cantos(previsto, gabarito)
    return {
        "classe": "VP" if iou >= LIMIAR_ACERTO_IOU else "VP_impreciso",
        "iou": iou,
        "erro_canto_px": erro,
        "erro_canto_pct": 100.0 * erro / diagonal,
    }


def resumir(linhas: list[dict]) -> dict:
    """Agrega as avaliações individuais num resumo do conjunto de teste."""
    positivos = [l for l in linhas if l["classe"] in ("VP", "VP_impreciso", "FN")]
    negativos = [l for l in linhas if l["classe"] in ("VN", "FP")]
    acertos = sum(l["classe"] == "VP" for l in linhas)
    imprecisos = sum(l["classe"] == "VP_impreciso" for l in linhas)
    fn = sum(l["classe"] == "FN" for l in linhas)
    fp = sum(l["classe"] == "FP" for l in linhas)
    vn = sum(l["classe"] == "VN" for l in linhas)
    ious = [l["iou"] for l in positivos if l["iou"] is not None]
    erros = [l["erro_canto_pct"] for l in positivos if l.get("erro_canto_pct") is not None]
    tempos = [l["tempo_ms"] for l in linhas if l.get("tempo_ms") is not None]

    detectados_ok = acertos
    total_detectados = acertos + imprecisos + fp
    return {
        "imagens": len(linhas),
        "com_documento": len(positivos),
        "sem_documento": len(negativos),
        "acertos_iou90": acertos,
        "imprecisos": imprecisos,
        "falsos_negativos": fn,
        "falsos_positivos": fp,
        "verdadeiros_negativos": vn,
        "taxa_acerto": acertos / len(positivos) if positivos else None,
        "precisao": detectados_ok / total_detectados if total_detectados else None,
        "recall": acertos / len(positivos) if positivos else None,
        "iou_medio": float(np.mean(ious)) if ious else None,
        "iou_mediano": float(np.median(ious)) if ious else None,
        "erro_canto_medio_pct": float(np.mean(erros)) if erros else None,
        "tempo_medio_ms": float(np.mean(tempos)) if tempos else None,
        "tempo_p95_ms": float(np.percentile(tempos, 95)) if tempos else None,
    }


def nitidez(img: np.ndarray) -> float:
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(cinza, cv2.CV_64F).var())
