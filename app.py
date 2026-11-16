"""Interface da aplicação — Scanner de Documentos com Visão Computacional.

Execução:  streamlit run app.py

A interface só coleta a entrada e os parâmetros e exibe os resultados; todo o
processamento está em src/ (aquisição, pré-processamento, detecção,
visualização e métricas).
"""
from __future__ import annotations

import os
import time
from dataclasses import fields

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.acquisition import (
    ErroEntrada, abrir_camera, abrir_video, carregar_imagem_arquivo,
    carregar_imagem_bytes, listar_amostras,
)
from src.acquisition.loader import EXTENSOES_IMAGEM, EXTENSOES_VIDEO
from src.config import DATA_DIR, RESULTS_DIR, Parametros
from src.geometry import iou_poligonos, erro_medio_cantos
from src.metrics import MedidorFPS
from src.metrics.avaliacao import avaliar_imagens, avaliar_video, carregar_anotacoes
from src.models import SuavizadorCantos
from src.pipeline import ResultadoPipeline, processar
from src.visualization import mapa_de_calor_bordas, montar_comparacao

st.set_page_config(page_title="Scanner de Documentos — Visão Computacional", page_icon="📄", layout="wide")

PADRAO = Parametros()
FONTES = ("Imagem de exemplo", "Enviar imagem", "Foto da câmera (navegador)",
          "Vídeo de exemplo", "Enviar vídeo", "Câmera ao vivo")
NOMES_METODO = {"auto": "Automático (bordas + limiar)", "bordas": "Bordas (Canny)", "limiar": "Limiarização (Otsu)"}
NOMES_SAIDA = {"colorido": "Colorido", "cinza": "Tons de cinza", "pb": "Preto e branco (limiar adaptativo)"}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def mostrar(img: np.ndarray, legenda: str | None = None):
    st.image(img, caption=legenda, channels="BGR" if img.ndim == 3 else "RGB", width="stretch")


def png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def restaurar_padroes():
    for f in fields(Parametros):
        st.session_state.pop(f"p_{f.name}", None)


@st.cache_data(show_spinner=False)
def amostra_em_cache(caminho: str) -> np.ndarray:
    return carregar_imagem_arquivo(caminho)


# ---------------------------------------------------------------------------
# Barra lateral: entrada e parâmetros
# ---------------------------------------------------------------------------
def barra_lateral() -> tuple[str, Parametros]:
    sb = st.sidebar
    sb.title("📄 Scanner de Documentos")
    sb.caption("Projeto Final — Computação Gráfica")

    sb.header("1. Entrada")
    fonte = sb.radio("Fonte da imagem", FONTES, key="fonte")

    sb.header("2. Parâmetros")
    k = lambda nome: f"p_{nome}"  # noqa: E731
    with sb.expander("Pré-processamento", expanded=True):
        altura = st.slider("Altura de processamento (px)", 300, 1200, PADRAO.altura_processamento, 50, key=k("altura_processamento"),
                           help="A imagem é reduzida para esta altura antes da detecção. Menor = mais rápido; maior = mais detalhe.")
        clahe = st.checkbox("Equalizar contraste (CLAHE)", PADRAO.usar_clahe, key=k("usar_clahe"))
        blur = st.slider("Filtro gaussiano (kernel)", 1, 15, PADRAO.kernel_blur, 2, key=k("kernel_blur"),
                         help="Suaviza ruído e textura do texto antes do Canny.")
        canny_auto = st.checkbox("Limiares do Canny automáticos", PADRAO.canny_auto, key=k("canny_auto"))
        canny = st.slider("Limiares do Canny (baixo, alto)", 0, 255, (PADRAO.canny_baixo, PADRAO.canny_alto),
                          key="p_canny", disabled=canny_auto)
        morf = st.slider("Fechamento morfológico (kernel)", 1, 15, PADRAO.kernel_morf, 2, key=k("kernel_morf"),
                         help="Liga trechos de borda interrompidos.")
    with sb.expander("Detecção", expanded=True):
        metodo = st.selectbox("Método", list(NOMES_METODO), format_func=NOMES_METODO.get, key=k("metodo"))
        eps = st.slider("Tolerância da aproximação poligonal", 0.005, 0.08, PADRAO.epsilon_aprox, 0.005,
                        format="%.3f", key=k("epsilon_aprox"), help="Fração do perímetro usada pelo approxPolyDP.")
        area = st.slider("Área mínima do documento", 0.02, 0.9, PADRAO.area_minima, 0.01, format="%.2f",
                         key=k("area_minima"), help="Fração da imagem que o documento deve ocupar.")
        conf = st.slider("Confiança mínima", 0.0, 1.0, PADRAO.confianca_minima, 0.05, key=k("confianca_minima"))
        refinar = st.checkbox("Refinamento subpixel dos cantos", PADRAO.refinar_cantos, key=k("refinar_cantos"))
    with sb.expander("Saída e visualização"):
        modo = st.selectbox("Documento retificado", list(NOMES_SAIDA), format_func=NOMES_SAIDA.get, key=k("modo_saida"))
        rot = st.select_slider("Rotação da saída (graus)", [0, 90, 180, 270], PADRAO.rotacao_saida, key=k("rotacao_saida"))
        opac = st.slider("Opacidade da máscara", 0.0, 1.0, PADRAO.opacidade_mascara, 0.05, key=k("opacidade_mascara"))
        cand = st.checkbox("Mostrar candidatos descartados", PADRAO.mostrar_candidatos, key=k("mostrar_candidatos"))
        larg = st.slider("Largura de exibição (px)", 480, 1600, PADRAO.largura_exibicao, 20, key=k("largura_exibicao"))
    with sb.expander("Vídeo e câmera"):
        suav = st.slider("Suavização (alfa da EMA)", 0.05, 1.0, PADRAO.suavizacao, 0.05, key=k("suavizacao"),
                         help="1,0 = sem suavização. Valores menores estabilizam o contorno, mas atrasam o movimento.")
        persist = st.slider("Persistência (quadros)", 0, 30, PADRAO.quadros_persistencia, key=k("quadros_persistencia"),
                            help="Por quantos quadros manter o último contorno quando a detecção falha.")
    sb.button("Restaurar padrões", on_click=restaurar_padroes, width="stretch")

    p = Parametros(
        altura_processamento=altura, usar_clahe=clahe, kernel_blur=blur, canny_auto=canny_auto,
        canny_baixo=canny[0], canny_alto=canny[1], kernel_morf=morf, metodo=metodo,
        epsilon_aprox=eps, area_minima=area, confianca_minima=conf, refinar_cantos=refinar,
        modo_saida=modo, rotacao_saida=rot, opacidade_mascara=opac, mostrar_candidatos=cand,
        largura_exibicao=larg, suavizacao=suav, quadros_persistencia=persist,
    ).validar()
    return fonte, p


# ---------------------------------------------------------------------------
# Exibição de um resultado de imagem
# ---------------------------------------------------------------------------
def painel_resultado(r: ResultadoPipeline, p: Parametros, gabarito: np.ndarray | None, nome: str):
    det = r.deteccao
    if r.encontrado:
        st.success(det.mensagem)
    else:
        st.warning(det.mensagem)

    c = st.columns(5)
    c[0].metric("Confiança", f"{det.confianca:.2f}")
    c[1].metric("Tempo total", f"{r.tempos_ms['total']:.1f} ms")
    c[2].metric("Resolução original", f"{r.original.shape[1]}×{r.original.shape[0]}")
    c[3].metric("Nitidez do documento", f"{r.nitidez:.0f}" if r.nitidez is not None else "—",
                help="Variância do Laplaciano. Abaixo de ~100 a foto provavelmente está tremida.")
    if gabarito is not None and r.encontrado:
        iou = iou_poligonos(r.cantos_original, gabarito)
        c[4].metric("IoU com o gabarito", f"{iou:.3f}",
                    delta=f"erro de canto {erro_medio_cantos(r.cantos_original, gabarito):.1f} px",
                    delta_color="off")
    elif gabarito is not None:
        c[4].metric("IoU com o gabarito", "0.000")

    aba_res, aba_etapas, aba_comp = st.tabs(["Resultado", "Etapas da pipeline", "Comparação lado a lado"])
    with aba_res:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("**Detecção sobre a imagem original**")
            mostrar(r.sobreposicao)
        with col2:
            st.markdown("**Documento retificado (transformação de perspectiva)**")
            if r.scan is not None:
                mostrar(r.scan)
                st.download_button("Baixar documento (PNG)", png_bytes(r.scan),
                                   file_name=f"{os.path.splitext(nome)[0]}_scan.png", mime="image/png")
            else:
                st.info("Sem documento detectado, não há o que retificar.")
        with st.expander("Tempo por etapa (ms)"):
            tempos = pd.DataFrame({"etapa": [k for k in r.tempos_ms if k != "total"],
                                   "ms": [v for k, v in r.tempos_ms.items() if k != "total"]})
            st.bar_chart(tempos, x="etapa", y="ms", horizontal=True)
        if r.encontrado:
            with st.expander("Cantos detectados (coordenadas da imagem original)"):
                st.dataframe(
                    {"canto": ["sup. esquerdo", "sup. direito", "inf. direito", "inf. esquerdo"],
                     "x (px)": np.round(r.cantos_original[:, 0], 1),
                     "y (px)": np.round(r.cantos_original[:, 1], 1)},
                    hide_index=True)

    with aba_etapas:
        prep = r.prep
        st.caption(f"Imagem reduzida para {prep.reduzida.shape[1]}×{prep.reduzida.shape[0]} px "
                   f"(escala {prep.escala:.3f}). Limiares do Canny usados: {prep.limiares_canny}.")
        etapas = [
            ("1. Redimensionada (proporção mantida)", prep.reduzida),
            ("2. Tons de cinza", prep.cinza),
            ("3. Contraste (CLAHE)" if p.usar_clahe else "3. Contraste (CLAHE desligado)", prep.contraste),
            ("4. Filtro gaussiano", prep.suavizada),
            ("5. Bordas (Canny)", prep.bordas),
            ("6. Bordas após fechamento morfológico", prep.bordas_fechadas),
            ("7. Limiarização de Otsu + morfologia", prep.limiar),
            ("8. Mapa de calor da densidade de bordas", mapa_de_calor_bordas(prep.reduzida, prep.bordas)),
        ]
        for i in range(0, len(etapas), 4):
            cols = st.columns(4)
            for col, (titulo, img) in zip(cols, etapas[i:i + 4]):
                with col:
                    mostrar(img, titulo)
        if det.candidatos:
            st.markdown("**Candidatos avaliados** (o primeiro é o escolhido, se passar da confiança mínima)")
            st.dataframe([{"origem": cd.origem, "aproximado": cd.aproximado, "confiança": round(cd.confianca, 3),
                           **cd.componentes} for cd in det.candidatos], hide_index=True)

    with aba_comp:
        paineis = [("Original", r.original), ("Bordas (Canny + morfologia)", r.prep.bordas_fechadas),
                   ("Deteccao", r.sobreposicao)]
        if r.scan is not None:
            paineis.append(("Resultado retificado", r.scan))
        mostrar(montar_comparacao(paineis, altura=420, colunas=len(paineis)))


# ---------------------------------------------------------------------------
# Modos de entrada
# ---------------------------------------------------------------------------
def modo_imagem(fonte: str, p: Parametros):
    img, nome, gabarito = None, "documento", None
    try:
        if fonte == "Imagem de exemplo":
            anot = carregar_anotacoes()
            amostras = listar_amostras(DATA_DIR / "imagens")
            if not amostras:
                st.error("Nenhuma amostra encontrada. Gere com: python scripts/gerar_amostras.py")
                return
            rotulo = lambda c: f"{c.name} — {anot.get(c.name, {}).get('descricao', '')}"  # noqa: E731
            escolha = st.selectbox("Imagem de exemplo", amostras, format_func=rotulo)
            img, nome = amostra_em_cache(str(escolha)), escolha.name
            if anot.get(nome, {}).get("cantos") and st.checkbox("Sobrepor gabarito (azul tracejado) e calcular IoU", True):
                gabarito = np.array(anot[nome]["cantos"], np.float32)
        elif fonte == "Enviar imagem":
            arq = st.file_uploader("Selecione uma foto de documento",
                                   type=[e.strip(".") for e in EXTENSOES_IMAGEM])
            if arq is None:
                st.info("Envie uma imagem (JPG, PNG, BMP, WEBP ou TIFF) para começar.")
                return
            img, nome = carregar_imagem_bytes(arq.getvalue(), arq.name), arq.name
        else:  # Foto da câmera (navegador)
            foto = st.camera_input("Fotografe o documento")
            if foto is None:
                st.info("Permita o acesso à câmera no navegador e tire uma foto. "
                        "Sem câmera, use 'Imagem de exemplo' ou 'Enviar imagem'.")
                return
            img, nome = carregar_imagem_bytes(foto.getvalue(), "camera.jpg"), "camera.jpg"
    except ErroEntrada as e:
        st.error(f"Entrada inválida: {e}")
        return

    try:
        r = processar(img, p, gabarito=gabarito)
    except cv2.error as e:  # rede de segurança: nunca derrubar a interface
        st.error(f"Erro do OpenCV ao processar a imagem: {e}")
        return
    painel_resultado(r, p, gabarito, nome)


def laco_video(cap: cv2.VideoCapture, p: Parametros, total_quadros: int | None, ao_vivo: bool):
    """Processa quadro a quadro e atualiza os elementos gráficos em tempo real."""
    suav = SuavizadorCantos(p.suavizacao, p.quadros_persistencia)
    fps = MedidorFPS()
    col1, col2 = st.columns([3, 2])
    quadro_ph, scan_ph = col1.empty(), col2.empty()
    info_ph = st.empty()
    barra = st.progress(0.0) if total_quadros else None
    n = detectados = persistidos = 0
    while True:
        ok, quadro = cap.read()
        if not ok or quadro is None:
            if ao_vivo:
                st.error("A câmera parou de enviar imagens (desconectada ou em uso por outro programa).")
            break
        n += 1
        f = fps.marcar()
        r = processar(quadro, p, suavizador=suav,
                      linhas_extra_hud=[f"Quadro {n}  |  FPS: {f:.1f}  |  Alfa EMA: {p.suavizacao:.2f}"])
        detectados += int(r.encontrado)
        persistidos += int(r.persistido)
        quadro_ph.image(r.sobreposicao, channels="BGR", width="stretch")
        if r.scan is not None:
            scan_ph.image(r.scan, channels="BGR" if r.scan.ndim == 3 else "RGB", width="stretch",
                          caption="Documento retificado")
            st.session_state["ultimo_scan"] = r.scan
        info_ph.caption(f"Quadros: {n} · com documento: {detectados} ({100 * detectados / n:.0f}%) · "
                        f"mantidos em memória: {persistidos} · FPS: {f:.1f} · processamento: {r.tempos_ms['total']:.0f} ms")
        if barra is not None:
            barra.progress(min(1.0, n / total_quadros))
        if not ao_vivo:
            time.sleep(0.001)
    return n, detectados


def modo_video(fonte: str, p: Parametros):
    temporario = None
    try:
        if fonte == "Vídeo de exemplo":
            videos = listar_amostras(DATA_DIR / "videos", EXTENSOES_VIDEO)
            if not videos:
                st.error("Nenhum vídeo de exemplo encontrado. Gere com: python scripts/gerar_amostras.py")
                return
            escolha = st.selectbox("Vídeo de exemplo", videos, format_func=lambda c: c.name)
            origem, nome = escolha, escolha.name
        else:
            arq = st.file_uploader("Selecione um vídeo", type=[e.strip(".") for e in EXTENSOES_VIDEO])
            if arq is None:
                st.info("Envie um vídeo (MP4, AVI, MOV, MKV ou WEBM).")
                return
            origem, nome = arq.getvalue(), arq.name
        if not st.button("▶ Processar vídeo", type="primary"):
            st.caption("O processamento é feito quadro a quadro; qualquer alteração na barra lateral interrompe e reinicia.")
            return
        cap, temporario = abrir_video(origem, nome)
    except ErroEntrada as e:
        st.error(f"Entrada inválida: {e}")
        return
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        n, det = laco_video(cap, p, total, ao_vivo=False)
        st.success(f"Vídeo concluído: {n} quadros, documento em {det} ({100 * det / max(n, 1):.0f}%).")
    finally:
        cap.release()
        if temporario and os.path.exists(temporario):
            os.remove(temporario)


def modo_camera_ao_vivo(p: Parametros):
    st.caption("Usa a webcam conectada a este computador (OpenCV). Desligue para encerrar e ver o último documento capturado.")
    c1, c2 = st.columns([1, 3])
    indice = c1.number_input("Índice da câmera", 0, 10, 0)
    ligada = c2.toggle("Ligar câmera", False)
    if not ligada:
        if "ultimo_scan" in st.session_state:
            st.markdown("**Último documento capturado**")
            scan = st.session_state["ultimo_scan"]
            mostrar(scan)
            st.download_button("Baixar documento (PNG)", png_bytes(scan), file_name="camera_scan.png", mime="image/png")
        return
    try:
        cap = abrir_camera(int(indice))
    except ErroEntrada as e:
        st.error(f"{e} Use 'Imagem de exemplo', 'Enviar imagem' ou 'Vídeo de exemplo'.")
        return
    try:
        laco_video(cap, p, None, ao_vivo=True)
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Aba de avaliação do conjunto de teste
# ---------------------------------------------------------------------------
def pagina_metricas(p: Parametros):
    st.subheader("Avaliação no conjunto anotado (data_samples)")
    st.markdown(
        "Executa a pipeline, **com os parâmetros atuais da barra lateral**, nas imagens sintéticas com gabarito. "
        "Acerto = IoU ≥ 0,90. Mude um parâmetro e rode de novo para comparar.")
    if st.button("Avaliar imagens com os parâmetros atuais", type="primary"):
        anot = carregar_anotacoes()
        barra = st.progress(0.0, "Processando...")
        cont = {"n": 0}

        def progresso(nome, linha, r):
            cont["n"] += 1
            barra.progress(cont["n"] / max(len(anot), 1), f"{nome}: {linha['classe']}")

        linhas, geral, por_cat = avaliar_imagens(p, ao_processar=progresso)
        barra.empty()
        c = st.columns(5)
        c[0].metric("Taxa de acerto", f"{100 * (geral['taxa_acerto'] or 0):.1f}%")
        c[1].metric("Precisão", f"{100 * (geral['precisao'] or 0):.1f}%")
        c[2].metric("IoU médio", f"{geral['iou_medio'] or 0:.3f}")
        c[3].metric("Falsos positivos", geral["falsos_positivos"])
        c[4].metric("Tempo médio", f"{geral['tempo_medio_ms']:.1f} ms")
        st.markdown("**Por categoria**")
        st.dataframe([{"categoria": k, **{kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}}
                      for k, v in por_cat.items()], hide_index=True)
        st.markdown("**Por imagem**")
        st.dataframe([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in l.items()} for l in linhas],
                     hide_index=True)

    if st.button("Avaliar vídeo (erro, tremor e FPS) com o alfa atual"):
        with st.spinner("Processando 150 quadros..."):
            res = [avaliar_video(p, alfa=1.0), avaliar_video(p)]
        st.dataframe([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in res], hide_index=True)
        st.caption("Linha 1: sem suavização (alfa = 1). Linha 2: alfa escolhido. Tremor = movimento dos cantos "
                   "não explicado pelo movimento real do documento.")

    rel = RESULTS_DIR / "RESULTADOS.md"
    if rel.is_file():
        with st.expander("Resultados registrados com os parâmetros padrão (docs/resultados/RESULTADOS.md)"):
            st.markdown(rel.read_text(encoding="utf-8"))


def pagina_sobre():
    st.markdown("""
### Problema
Fotografar documentos com o celular gera imagens **inclinadas, em perspectiva e com fundo**. A aplicação
localiza a folha na cena, corrige a perspectiva e entrega uma imagem "escaneada" — útil para quem precisa
digitalizar contratos, recibos e formulários sem um scanner (pequenos escritórios, estudantes, setor de cadastro).

### Pipeline
1. **Aquisição**: imagem, vídeo, foto do navegador ou webcam ao vivo.
2. **Validação**: extensão, arquivo vazio/corrompido, tamanho mínimo/máximo, câmera ausente.
3. **Pré-processamento**: redimensionamento com proporção, tons de cinza, CLAHE, filtro gaussiano, Canny,
   fechamento morfológico e limiarização de Otsu.
4. **Detecção**: contornos → aproximação poligonal → quadriláteros convexos → pontuação de confiança.
5. **Interpretação**: ordenação dos cantos, mapeamento para a resolução original, refinamento subpixel.
6. **Representação**: camadas (fundo escurecido, máscara translúcida, contorno, cantos, HUD, legenda),
   transformação de perspectiva, comparação lado a lado e mapa de calor.
7. **Avaliação**: IoU, erro de canto, falsos positivos/negativos, tempo por etapa, FPS e tremor no vídeo.

### Limitações conhecidas
Fundo da mesma cor do papel, cantos cobertos, documento cortado pela foto, documento muito pequeno e
páginas curvas (livros) — veja exemplos na aba **Métricas** e no README.

Detalhes, créditos e declaração de uso de IA: `README.md`.
""")


# ---------------------------------------------------------------------------
def main():
    fonte, p = barra_lateral()
    st.title("Scanner de documentos com visão computacional")
    st.caption("Localiza a folha na cena, corrige a perspectiva e mostra cada etapa da análise.")
    aba_app, aba_metricas, aba_sobre = st.tabs(["🔍 Análise", "📊 Métricas", "ℹ️ Sobre"])
    # As abas estáticas são preenchidas antes: o laço de vídeo/câmera bloqueia o script até terminar.
    with aba_metricas:
        pagina_metricas(p)
    with aba_sobre:
        pagina_sobre()
    with aba_app:
        if fonte in ("Imagem de exemplo", "Enviar imagem", "Foto da câmera (navegador)"):
            modo_imagem(fonte, p)
        elif fonte in ("Vídeo de exemplo", "Enviar vídeo"):
            modo_video(fonte, p)
        else:
            modo_camera_ao_vivo(p)


main()
