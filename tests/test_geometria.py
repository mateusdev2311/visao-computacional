import numpy as np
import pytest

from src.geometry import (
    MapeadorCoordenadas, erro_medio_cantos, iou_poligonos, ordenar_cantos, retificar, tamanho_retificado,
)
from src.preprocessing import redimensionar_por_altura, redimensionar_para_caber

QUAD = np.array([[10, 10], [110, 10], [110, 60], [10, 60]], np.float32)


def test_ordenar_cantos_qualquer_permutacao():
    rng = np.random.default_rng(0)
    for _ in range(10):
        embaralhado = QUAD[rng.permutation(4)]
        assert np.allclose(ordenar_cantos(embaralhado), QUAD)


def test_ordenar_cantos_documento_girado_45_graus():
    losango = np.array([[50, 0], [100, 50], [50, 100], [0, 50]], np.float32)
    ordenado = ordenar_cantos(losango[::-1])
    # deve seguir o sentido horário a partir do canto superior
    assert np.allclose(ordenado[0], [50, 0]) and np.allclose(ordenado[1], [100, 50])


def test_iou_identico_disjunto_e_metade():
    assert iou_poligonos(QUAD, QUAD) == pytest.approx(1.0)
    longe = QUAD + 1000
    assert iou_poligonos(QUAD, longe) == pytest.approx(0.0)
    metade = np.array([[10, 10], [60, 10], [60, 60], [10, 60]], np.float32)
    assert iou_poligonos(QUAD, metade) == pytest.approx(0.5, abs=1e-3)


def test_erro_medio_cantos():
    assert erro_medio_cantos(QUAD + [3, 4], QUAD) == pytest.approx(5.0)


def test_mapeador_ida_e_volta():
    m = MapeadorCoordenadas(escala_processamento=0.25, escala_exibicao=0.5)
    pts_proc = np.array([[100, 50]], np.float32)
    original = m.processamento_para_original(pts_proc)
    assert np.allclose(original, [[400, 200]])
    assert np.allclose(m.processamento_para_exibicao(pts_proc), [[200, 100]])
    assert np.allclose(m.exibicao_para_original(m.original_para_exibicao(original)), original)


def test_redimensionamento_mantem_proporcao():
    img = np.zeros((1000, 1500, 3), np.uint8)
    red, esc = redimensionar_por_altura(img, 500)
    assert red.shape[:2] == (500, 750) and esc == pytest.approx(0.5)
    caixa, esc2 = redimensionar_para_caber(img, 300, 300)
    assert caixa.shape[1] == 300 and caixa.shape[0] == 200 and esc2 == pytest.approx(0.2)


def test_retificacao_e_rotacao():
    img = np.zeros((100, 200, 3), np.uint8)
    img[10:60, 10:110] = 255
    saida, _ = retificar(img, QUAD)
    assert saida.shape[:2] == (50, 100) and saida.mean() > 240
    girada, _ = retificar(img, QUAD, rotacao=90)
    assert girada.shape[:2] == (100, 50)
    assert tamanho_retificado(QUAD) == (100, 50)
