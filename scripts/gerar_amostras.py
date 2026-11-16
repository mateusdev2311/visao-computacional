"""Gera o conjunto de demonstração/teste sintético com gabarito exato.

Por que sintético? Para medir IoU e erro de canto precisamos saber onde estão
os cantos reais. Ao "fotografar" um documento virtual com uma homografia
conhecida, o gabarito é exato e não há imagens de terceiros ou dados pessoais
(não há questões de autorização nem de privacidade).

Saídas (em data_samples/):
  imagens/*.jpg              fotos sintéticas (fácil, médio, difícil, negativas)
  imagens/anotacoes.json     cantos reais [SE, SD, ID, IE] por imagem
  videos/documento_movendo.mp4  vídeo com documento em movimento
  videos/anotacoes_video.json   cantos reais por quadro
  invalidos/                 arquivos para demonstrar o tratamento de erros

Uso:  python scripts/gerar_amostras.py  [--semente 42]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR  # noqa: E402

PALAVRAS = ("contrato prestacao servicos clausula valor prazo pagamento empresa cliente "
            "documento assinatura data entrega relatorio projeto computacao grafica visao "
            "imagem processamento resultado analise sistema camera pixel matriz transformacao "
            "perspectiva contorno borda filtro media nota fiscal total item quantidade unidade").split()

FONTE = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------
# Documento virtual
# --------------------------------------------------------------------------
def criar_documento(rng: np.random.Generator, tipo: str = "a4") -> np.ndarray:
    if tipo == "recibo":
        w, h = 420, 1100
    elif tipo == "cartao":
        w, h = 860, 540
    else:
        w, h = 850, 1200
    tom = rng.integers(232, 252)
    papel = np.full((h, w, 3), tom, np.uint8)
    papel[:] = papel * np.array([1.0, rng.uniform(0.97, 1.0), rng.uniform(0.95, 1.0)])
    papel = papel.astype(np.uint8)
    # Textura leve do papel
    ruido = rng.normal(0, 2.0, (h, w, 1))
    papel = np.clip(papel.astype(np.float32) + ruido, 0, 255).astype(np.uint8)

    margem = int(0.07 * w)
    y = margem + 30
    titulo = " ".join(rng.choice(PALAVRAS, 2)).upper()
    cv2.putText(papel, titulo, (margem, y), FONTE, 1.3 * w / 850, (30, 30, 30), 3, cv2.LINE_AA)
    y += 50
    cv2.line(papel, (margem, y), (w - margem, y), (60, 60, 60), 2)
    y += 40

    escala_txt = 0.6 * max(w, 500) / 850
    while y < h - margem - 160:
        if rng.random() < 0.12 and tipo != "cartao":
            # tabela
            linhas, cols = rng.integers(3, 6), rng.integers(2, 5)
            alt = 32
            x0, x1 = margem, w - margem
            for i in range(linhas + 1):
                cv2.line(papel, (x0, y + i * alt), (x1, y + i * alt), (70, 70, 70), 1)
            for j in range(cols + 1):
                xx = int(x0 + j * (x1 - x0) / cols)
                cv2.line(papel, (xx, y), (xx, y + linhas * alt), (70, 70, 70), 1)
            y += linhas * alt + 30
            continue
        n = rng.integers(3, 9)
        linha = " ".join(rng.choice(PALAVRAS, n))
        cv2.putText(papel, linha, (margem, y), FONTE, escala_txt, (45, 45, 45), 1, cv2.LINE_AA)
        y += int(28 * escala_txt / 0.6)
    # "assinatura"
    xs = np.linspace(margem, margem + 0.35 * w, 60)
    ys = h - margem - 60 + 15 * np.sin(xs / 9.0) * rng.uniform(0.5, 1.2)
    pts = np.stack([xs, ys], 1).astype(np.int32)
    cv2.polylines(papel, [pts], False, (120, 40, 20), 2, cv2.LINE_AA)
    return papel


# --------------------------------------------------------------------------
# Fundos
# --------------------------------------------------------------------------
def criar_fundo(rng: np.random.Generator, w: int, h: int, tipo: str) -> np.ndarray:
    if tipo == "madeira":
        base = np.array([40, 80, 130], np.float32) * rng.uniform(0.7, 1.1)
        xs = np.arange(w)[None, :]
        ys = np.arange(h)[:, None]
        veios = (np.sin(ys / rng.uniform(8, 14) + 3 * np.sin(xs / 120.0)) * 0.5 + 0.5)[..., None]
        img = base * (0.75 + 0.35 * veios)
    elif tipo == "escuro":
        img = np.full((h, w, 3), rng.uniform(25, 60), np.float32)
        img *= np.array([1.0, rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.2)])
    elif tipo == "claro":            # pouco contraste com o papel (caso difícil)
        img = np.full((h, w, 3), rng.uniform(185, 210), np.float32)
    elif tipo == "baguncado":
        img = np.full((h, w, 3), rng.uniform(70, 110), np.float32)
        for _ in range(rng.integers(8, 16)):
            cor = rng.uniform(30, 200, 3)
            if rng.random() < 0.5:
                c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
                cv2.circle(img, c, int(rng.integers(20, 90)), cor.tolist(), -1, cv2.LINE_AA)
            else:
                p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
                p2 = (p1[0] + int(rng.integers(-200, 200)), p1[1] + int(rng.integers(-200, 200)))
                cv2.rectangle(img, p1, p2, cor.tolist(), -1)
    else:
        raise ValueError(tipo)
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# "Câmera" virtual
# --------------------------------------------------------------------------
def quad_aleatorio(rng, w, h, ocupacao=(0.25, 0.6), rot_max=30, persp=0.08, aspecto=1.41, dentro=True):
    for _ in range(200):
        area = rng.uniform(*ocupacao) * w * h
        larg = np.sqrt(area / aspecto)
        alt = larg * aspecto
        ang = np.radians(rng.uniform(-rot_max, rot_max))
        cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
        base = np.array([[-larg / 2, -alt / 2], [larg / 2, -alt / 2], [larg / 2, alt / 2], [-larg / 2, alt / 2]])
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        q = base @ R.T + [cx, cy]
        q += rng.uniform(-persp, persp, (4, 2)) * [larg, alt]
        margem = 8
        if not dentro or (q[:, 0].min() > margem and q[:, 1].min() > margem and
                          q[:, 0].max() < w - margem and q[:, 1].max() < h - margem):
            return q.astype(np.float32)
    raise RuntimeError("não foi possível posicionar o documento")


def compor(fundo: np.ndarray, doc: np.ndarray, quad: np.ndarray, rng, sombra=True) -> np.ndarray:
    h, w = fundo.shape[:2]
    dh, dw = doc.shape[:2]
    origem = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]], np.float32)
    M = cv2.getPerspectiveTransform(origem, quad)
    warp = cv2.warpPerspective(doc, M, (w, h), flags=cv2.INTER_LINEAR)
    mascara = cv2.warpPerspective(np.full((dh, dw), 255, np.uint8), M, (w, h))
    saida = fundo.astype(np.float32)
    if sombra:   # sombra projetada suave, deslocada
        desloc = np.float32([[1, 0, rng.uniform(6, 16)], [0, 1, rng.uniform(6, 16)]])
        s = cv2.warpAffine(mascara, desloc, (w, h))
        s = cv2.GaussianBlur(s, (0, 0), 9).astype(np.float32) / 255.0
        saida *= (1 - 0.45 * s)[..., None]
    m = cv2.GaussianBlur(mascara, (3, 3), 0).astype(np.float32)[..., None] / 255.0  # borda antisserrilhada
    saida = saida * (1 - m) + warp.astype(np.float32) * m
    return saida


def iluminacao(img: np.ndarray, rng, intensidade=0.35) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    ys, xs = np.mgrid[0:h, 0:w]
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / np.hypot(w, h)
    fator = 1.0 + intensidade * (0.5 - d)
    return img * fator[..., None]


def finalizar(img: np.ndarray, rng, ruido=4.0, borrar=0, borrao_mov=0) -> np.ndarray:
    img = img + rng.normal(0, ruido, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if borrar:
        img = cv2.GaussianBlur(img, (0, 0), borrar)
    if borrao_mov:
        k = np.zeros((borrao_mov, borrao_mov), np.float32)
        k[borrao_mov // 2, :] = 1.0 / borrao_mov
        img = cv2.filter2D(img, -1, k)
    return img


def ocluir(img: np.ndarray, ponto, rng) -> None:
    """Objeto (ex.: caneca/mão) cobrindo um canto do documento."""
    cor = rng.uniform(40, 120, 3).tolist()
    cv2.ellipse(img, (int(ponto[0]), int(ponto[1])), (int(rng.uniform(60, 90)), int(rng.uniform(40, 70))),
                rng.uniform(0, 180), 0, 360, cor, -1, cv2.LINE_AA)


# --------------------------------------------------------------------------
def gerar_imagens(rng, pasta: Path) -> dict:
    pasta.mkdir(parents=True, exist_ok=True)
    anot = {}
    W, H = 1280, 960

    def salvar(nome, img, quad, categoria, descricao):
        cv2.imwrite(str(pasta / nome), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        anot[nome] = {
            "cantos": None if quad is None else np.round(quad, 2).tolist(),
            "categoria": categoria,
            "descricao": descricao,
        }

    # Fáceis: fundo escuro/madeira, documento inteiro, perspectiva moderada
    for i in range(12):
        tipo_fundo = ["escuro", "madeira"][i % 2]
        tipo_doc = ["a4", "a4", "recibo", "cartao"][i % 4]
        aspecto = {"a4": 1.41, "recibo": 2.6, "cartao": 0.63}[tipo_doc]
        retrato = rng.random() < 0.5
        w, h = (H, W) if retrato else (W, H)
        q = quad_aleatorio(rng, w, h, (0.2, 0.5), rot_max=25, aspecto=aspecto)
        img = compor(criar_fundo(rng, w, h, tipo_fundo), criar_documento(rng, tipo_doc), q, rng)
        img = finalizar(iluminacao(img, rng, 0.25), rng)
        salvar(f"facil_{i + 1:02d}.jpg", img, q, "facil", f"fundo {tipo_fundo}, documento {tipo_doc}")

    # Médios: fundo bagunçado, rotação forte, iluminação irregular, leve desfoque
    for i in range(8):
        tipo_doc = ["a4", "recibo", "a4", "cartao"][i % 4]
        aspecto = {"a4": 1.41, "recibo": 2.6, "cartao": 0.63}[tipo_doc]
        q = quad_aleatorio(rng, W, H, (0.2, 0.55), rot_max=45, persp=0.12, aspecto=aspecto)
        fundo = criar_fundo(rng, W, H, "baguncado" if i % 2 == 0 else "madeira")
        img = compor(fundo, criar_documento(rng, tipo_doc), q, rng)
        img = finalizar(iluminacao(img, rng, 0.6), rng, ruido=6, borrar=1.0 if i % 3 == 0 else 0)
        salvar(f"medio_{i + 1:02d}.jpg", img, q, "medio", f"fundo com objetos/iluminacao irregular, {tipo_doc}")

    # Difíceis: cada um com uma limitação conhecida
    q = quad_aleatorio(rng, W, H, (0.3, 0.5), rot_max=15)
    img = finalizar(compor(criar_fundo(rng, W, H, "claro"), criar_documento(rng), q, rng, sombra=False), rng)
    salvar("dificil_01_baixo_contraste.jpg", img, q, "dificil", "fundo claro quase da cor do papel, sem sombra")

    q = quad_aleatorio(rng, W, H, (0.3, 0.5), rot_max=20)
    img = compor(criar_fundo(rng, W, H, "escuro"), criar_documento(rng), q, rng)
    ocluir(img, q[1], rng)
    salvar("dificil_02_canto_ocluido.jpg", finalizar(img, rng), q, "dificil", "objeto cobrindo o canto superior direito")

    q = quad_aleatorio(rng, W, H, (0.3, 0.5), rot_max=20)
    img = compor(criar_fundo(rng, W, H, "madeira"), criar_documento(rng), q, rng)
    salvar("dificil_03_tremida.jpg", finalizar(img, rng, borrao_mov=25), q, "dificil", "borrao de movimento forte")

    q = quad_aleatorio(rng, W, H, (0.45, 0.6), rot_max=10, dentro=False)
    q[:, 0] += 0.25 * W   # empurra parte do documento para fora da foto
    img = compor(criar_fundo(rng, W, H, "escuro"), criar_documento(rng), q, rng)
    salvar("dificil_04_cortado.jpg", finalizar(img, rng), q, "dificil", "documento parcialmente fora da foto")

    q = quad_aleatorio(rng, W, H, (0.3, 0.5), rot_max=20)
    img = compor(criar_fundo(rng, W, H, "madeira"), criar_documento(rng), q, rng)
    img = iluminacao(img, rng, 1.2)
    salvar("dificil_05_sombra_forte.jpg", finalizar(img, rng, ruido=10), q, "dificil", "iluminacao muito irregular e ruido alto")

    q = quad_aleatorio(rng, W, H, (0.06, 0.09), rot_max=20)
    img = compor(criar_fundo(rng, W, H, "escuro"), criar_documento(rng), q, rng)
    salvar("dificil_06_pequeno.jpg", finalizar(img, rng), q, "dificil", "documento ocupa menos de 10% da foto")

    # Negativos: sem documento (mede falsos positivos)
    for i, tipo in enumerate(["baguncado", "madeira", "escuro", "baguncado"]):
        img = finalizar(iluminacao(criar_fundo(rng, W, H, tipo).astype(np.float32), rng, 0.4), rng)
        salvar(f"negativo_{i + 1:02d}.jpg", img, None, "negativo", f"sem documento, fundo {tipo}")

    (pasta / "anotacoes.json").write_text(json.dumps(anot, indent=2, ensure_ascii=False), encoding="utf-8")
    return anot


def gerar_video(rng, pasta: Path, quadros: int = 150, fps: int = 25) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    W, H = 800, 600
    fundo = criar_fundo(rng, W, H, "madeira")
    doc = criar_documento(rng)
    caminho = pasta / "documento_movendo.mp4"
    gravador = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not gravador.isOpened():
        raise RuntimeError("VideoWriter não abriu (codec mp4v indisponível)")
    larg, alt = 270, 270 * 1.41
    base = np.array([[-larg / 2, -alt / 2], [larg / 2, -alt / 2], [larg / 2, alt / 2], [-larg / 2, alt / 2]])
    gabaritos = []
    for t in range(quadros):
        s = t / quadros
        cx = W / 2 + 140 * np.sin(2 * np.pi * s)
        cy = H / 2 + 50 * np.sin(4 * np.pi * s)
        ang = np.radians(20 * np.sin(2 * np.pi * s))
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        incl = 0.10 * np.sin(2 * np.pi * s)       # inclinação da câmera (perspectiva)
        b = base * [[1 - incl, 1], [1 - incl, 1], [1 + incl, 1], [1 + incl, 1]]
        q = (b @ R.T + [cx, cy]).astype(np.float32)
        img = compor(fundo, doc, q, rng)
        img = iluminacao(img, np.random.default_rng(7), 0.3)
        img = finalizar(img, rng, ruido=3)
        if 60 <= t < 75:    # mão passa por cima do documento por 15 quadros
            ocluir(img, q.mean(axis=0) + [0, -40], np.random.default_rng(t))
            cv2.rectangle(img, (int(q[:, 0].min()) - 30, int(q[:, 1].mean()) - 30),
                          (int(q[:, 0].max()) + 30, int(q[:, 1].mean()) + 30), (70, 90, 120), -1)
        gravador.write(img)
        gabaritos.append(np.round(q, 2).tolist())
    gravador.release()
    (pasta / "anotacoes_video.json").write_text(json.dumps(
        {"documento_movendo.mp4": {"fps": fps, "quadros_ocluidos": [60, 75], "cantos_por_quadro": gabaritos}},
        ensure_ascii=False), encoding="utf-8")


def gerar_invalidos(pasta: Path) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "texto_renomeado.jpg").write_text("isto nao e uma imagem", encoding="utf-8")
    (pasta / "vazio.png").write_bytes(b"")
    (pasta / "imagem_minuscula.png").write_bytes(cv2.imencode(".png", np.zeros((20, 20, 3), np.uint8))[1].tobytes())
    (pasta / "video_corrompido.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semente", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.semente)
    anot = gerar_imagens(rng, DATA_DIR / "imagens")
    gerar_video(rng, DATA_DIR / "videos")
    gerar_invalidos(DATA_DIR / "invalidos")
    print(f"{len(anot)} imagens, 1 vídeo e exemplos inválidos gerados em {DATA_DIR}")


if __name__ == "__main__":
    main()
