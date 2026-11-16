"""Etapa 1 e 2 da pipeline: aquisição e validação da entrada.

Fontes suportadas:
  * imagem (arquivo no disco ou bytes enviados pela interface);
  * vídeo (arquivo);
  * câmera (índice do dispositivo, normalmente 0).

Toda falha previsível gera ``ErroEntrada`` com uma mensagem em português,
que a interface mostra ao usuário sem encerrar a aplicação.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXTENSOES_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

TAMANHO_MAX_BYTES = 25 * 1024 * 1024   # 25 MB
LADO_MINIMO = 64                        # px
LADO_MAXIMO = 8000                      # px


class ErroEntrada(Exception):
    """Erro previsível de entrada (arquivo inválido, câmera ausente etc.)."""


def _validar_imagem(img: np.ndarray | None, nome: str) -> np.ndarray:
    if img is None:
        raise ErroEntrada(
            f"Não foi possível decodificar '{nome}'. O arquivo está corrompido "
            "ou não é uma imagem suportada (JPG, PNG, BMP, WEBP, TIFF)."
        )
    if img.ndim == 2:                       # tons de cinza -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                 # descarta canal alfa
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[:2]
    if min(h, w) < LADO_MINIMO:
        raise ErroEntrada(f"Imagem muito pequena ({w}x{h}). O mínimo é {LADO_MINIMO} px por lado.")
    if max(h, w) > LADO_MAXIMO:
        raise ErroEntrada(f"Imagem muito grande ({w}x{h}). O máximo é {LADO_MAXIMO} px por lado.")
    return img


def carregar_imagem_bytes(dados: bytes, nome: str = "arquivo") -> np.ndarray:
    """Decodifica bytes (ex.: upload do Streamlit) para uma imagem BGR."""
    if not dados:
        raise ErroEntrada("O arquivo enviado está vazio.")
    if len(dados) > TAMANHO_MAX_BYTES:
        raise ErroEntrada("Arquivo maior que 25 MB.")
    ext = Path(nome).suffix.lower()
    if ext and ext not in EXTENSOES_IMAGEM:
        raise ErroEntrada(f"Extensão '{ext}' não suportada para imagens.")
    buffer = np.frombuffer(dados, dtype=np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    return _validar_imagem(img, nome)


def carregar_imagem_arquivo(caminho: str | Path) -> np.ndarray:
    """Lê uma imagem do disco. Usa imdecode para aceitar caminhos com acentos no Windows."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise ErroEntrada(f"Arquivo não encontrado: {caminho.name}")
    return carregar_imagem_bytes(caminho.read_bytes(), caminho.name)


def abrir_video(origem: str | Path | bytes, nome: str = "video.mp4") -> tuple[cv2.VideoCapture, str | None]:
    """Abre um vídeo a partir de caminho ou bytes.

    Retorna a captura e, se foi necessário criar um arquivo temporário, o caminho
    dele (quem chamou deve apagá-lo depois de liberar a captura).
    """
    temporario = None
    if isinstance(origem, (bytes, bytearray)):
        if not origem:
            raise ErroEntrada("O vídeo enviado está vazio.")
        ext = Path(nome).suffix.lower() or ".mp4"
        if ext not in EXTENSOES_VIDEO:
            raise ErroEntrada(f"Extensão '{ext}' não suportada para vídeos.")
        fd, temporario = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, "wb") as f:
            f.write(origem)
        caminho = temporario
    else:
        caminho = str(origem)
        if not Path(caminho).is_file():
            raise ErroEntrada(f"Vídeo não encontrado: {Path(caminho).name}")

    cap = cv2.VideoCapture(caminho)
    ok, quadro = (cap.read() if cap.isOpened() else (False, None))
    if not ok or quadro is None:
        cap.release()
        if temporario:
            os.remove(temporario)
        raise ErroEntrada("Não foi possível ler quadros do vídeo (arquivo inválido ou codec não suportado).")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap, temporario


def abrir_camera(indice: int = 0) -> cv2.VideoCapture:
    """Abre a webcam e confirma que ela entrega quadros."""
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY  # DSHOW abre mais rápido no Windows
    cap = cv2.VideoCapture(indice, backend)
    if not cap.isOpened():
        cap.release()
        raise ErroEntrada(f"Nenhuma câmera encontrada no índice {indice}.")
    ok, quadro = cap.read()
    if not ok or quadro is None:
        cap.release()
        raise ErroEntrada(
            f"A câmera {indice} foi encontrada, mas não entregou imagens "
            "(pode estar em uso por outro programa ou sem permissão)."
        )
    return cap


def listar_amostras(pasta: str | Path, extensoes: set[str] = EXTENSOES_IMAGEM) -> list[Path]:
    pasta = Path(pasta)
    if not pasta.is_dir():
        return []
    return sorted(p for p in pasta.iterdir() if p.suffix.lower() in extensoes)
