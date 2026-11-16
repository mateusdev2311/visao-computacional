import cv2
import numpy as np
import pytest

from src.acquisition import ErroEntrada, abrir_video, carregar_imagem_arquivo, carregar_imagem_bytes
from src.config import DATA_DIR, Parametros
from src.geometry import iou_poligonos
from src.models import SuavizadorCantos
from src.pipeline import processar


def cena_sintetica(fundo=40, papel=235):
    img = np.full((600, 800, 3), fundo, np.uint8)
    cantos = np.array([[220, 90], [590, 130], [560, 520], [180, 470]], np.float32)
    cv2.fillConvexPoly(img, cantos.astype(np.int32), (papel, papel, papel))
    for y in range(160, 440, 25):   # "texto" dentro da folha
        cv2.line(img, (260, y), (520, y + 4), (60, 60, 60), 2)
    return img, cantos


def test_detecta_documento_sintetico():
    img, cantos = cena_sintetica()
    r = processar(img, Parametros())
    assert r.encontrado
    assert iou_poligonos(r.cantos_original, cantos) > 0.97
    assert r.scan is not None and r.scan.shape[0] > 300


@pytest.mark.parametrize("metodo", ["bordas", "limiar", "auto"])
def test_todos_os_metodos_funcionam(metodo):
    img, cantos = cena_sintetica()
    r = processar(img, Parametros(metodo=metodo))
    assert r.encontrado and iou_poligonos(r.cantos_original, cantos) > 0.95


def test_imagem_sem_documento_nao_gera_deteccao():
    img = np.full((480, 640, 3), 90, np.uint8)
    r = processar(img, Parametros())
    assert not r.encontrado
    assert r.scan is None
    assert "Nenhum" in r.deteccao.mensagem or "abaixo" in r.deteccao.mensagem
    assert r.sobreposicao.shape[2] == 3   # mesmo sem detecção há imagem para exibir


def test_parametros_fora_da_faixa_sao_corrigidos():
    p = Parametros(kernel_blur=4, kernel_morf=-3, metodo="inexistente", area_minima=5, rotacao_saida=45).validar()
    assert p.kernel_blur == 5 and p.kernel_morf == 1 and p.metodo == "auto"
    assert p.area_minima == 0.95 and p.rotacao_saida == 0


def test_entradas_invalidas_geram_erro_amigavel():
    with pytest.raises(ErroEntrada):
        carregar_imagem_bytes(b"", "vazio.png")
    with pytest.raises(ErroEntrada):
        carregar_imagem_bytes(b"nao sou uma imagem", "x.jpg")
    with pytest.raises(ErroEntrada):
        carregar_imagem_bytes(b"123", "planilha.xlsx")
    pequena = cv2.imencode(".png", np.zeros((10, 10, 3), np.uint8))[1].tobytes()
    with pytest.raises(ErroEntrada):
        carregar_imagem_bytes(pequena, "p.png")
    with pytest.raises(ErroEntrada):
        carregar_imagem_arquivo("nao/existe.jpg")
    with pytest.raises(ErroEntrada):
        abrir_video(b"\x00" * 100, "corrompido.mp4")


def test_suavizador_reduz_tremor_e_persiste():
    rng = np.random.default_rng(1)
    base = np.array([[0, 0], [100, 0], [100, 150], [0, 150]], np.float32) + 200
    s = SuavizadorCantos(alfa=0.3, quadros_persistencia=2)
    saidas = [s.atualizar(base + rng.normal(0, 3, base.shape), diagonal=800)[0] for _ in range(40)]
    tremor_bruto = 3 * np.sqrt(2)
    tremor_suave = np.mean([np.linalg.norm(a - b, axis=1).mean() for a, b in zip(saidas[10:], saidas[11:])])
    assert tremor_suave < tremor_bruto
    # persistência: 2 quadros mantidos, depois perde
    assert s.atualizar(None, 800)[1] is True
    assert s.atualizar(None, 800)[1] is True
    assert s.atualizar(None, 800)[0] is None


@pytest.mark.skipif(not (DATA_DIR / "imagens" / "anotacoes.json").is_file(), reason="amostras não geradas")
def test_conjunto_de_amostras_facil_e_medio():
    from src.metrics.avaliacao import avaliar_imagens
    linhas, geral, por_cat = avaliar_imagens(Parametros())
    assert por_cat["facil"]["acertos_iou90"] == por_cat["facil"]["imagens"]
    assert geral["falsos_positivos"] == 0
