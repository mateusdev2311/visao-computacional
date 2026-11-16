"""Avaliação reprodutível da pipeline sobre data_samples/.

Gera em docs/resultados/:
  metricas_imagens.csv     uma linha por imagem (IoU, erro de canto, classe, tempo)
  comparacao_metodos.csv   métricas agregadas de cada método de detecção
  video_suavizacao.csv     erro/tremor/FPS no vídeo para vários alfas da EMA
  resumo.json              todos os resumos
  RESULTADOS.md            tabelas prontas para README/apresentação
  exemplos/*.jpg           sobreposições de cada imagem (gabarito em azul tracejado)

Uso:  python scripts/avaliar.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Parametros, RESULTS_DIR  # noqa: E402
from src.metrics.avaliacao import avaliar_imagens, avaliar_video, salvar_csv, salvar_imagem  # noqa: E402


def fmt(v, casas=3, pct=False):
    if v is None:
        return "-"
    if pct:
        return f"{100 * v:.1f}%"
    return f"{v:.{casas}f}" if isinstance(v, float) else str(v)


def main():
    p = Parametros().validar()
    pasta_ex = RESULTS_DIR / "exemplos"

    def salvar_exemplo(nome, linha, r):
        salvar_imagem(pasta_ex / f"{Path(nome).stem}_{linha['classe']}.jpg", r.sobreposicao)

    for antigo in pasta_ex.glob("*.jpg"):
        antigo.unlink()

    print("Avaliando imagens (parâmetros padrão)...")
    linhas, geral, por_cat = avaliar_imagens(p, ao_processar=salvar_exemplo)
    salvar_csv(linhas, RESULTS_DIR / "metricas_imagens.csv")

    print("Comparando métodos de detecção...")
    comparacao = []
    for metodo in ("bordas", "limiar", "auto"):
        pm = Parametros(metodo=metodo).validar()
        _, g, _ = avaliar_imagens(pm)
        comparacao.append({"metodo": metodo, **g})
    for rotulo, pm in (("auto sem CLAHE", Parametros(usar_clahe=False)),
                       ("auto sem refinamento subpixel", Parametros(refinar_cantos=False)),
                       ("auto com área mínima 5%", Parametros(area_minima=0.05))):
        _, g, _ = avaliar_imagens(pm.validar())
        comparacao.append({"metodo": rotulo, **g})
    salvar_csv(comparacao, RESULTS_DIR / "comparacao_metodos.csv")

    print("Avaliando vídeo com diferentes suavizações...")
    video = [avaliar_video(p, alfa=a) for a in (1.0, 0.7, 0.5, 0.3)]
    salvar_csv(video, RESULTS_DIR / "video_suavizacao.csv")

    resumo = {"parametros": p.como_dict(), "geral": geral, "por_categoria": por_cat,
              "comparacao_metodos": comparacao, "video": video}
    (RESULTS_DIR / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- Markdown ----------------
    md = ["# Resultados da avaliação", "",
          "Gerado automaticamente por `python scripts/avaliar.py` com os parâmetros padrão "
          "(`src/config.py`). Acerto = IoU ≥ 0,90 com o gabarito.", "",
          "## Desempenho por categoria (imagens)", "",
          "| Categoria | Imagens | Acertos | Imprecisos | FN | FP | IoU médio | Erro de canto (% diag.) | Tempo médio (ms) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for cat, r in list(por_cat.items()) + [("**TOTAL**", geral)]:
        md.append(f"| {cat} | {r['imagens']} | {r['acertos_iou90']} | {r['imprecisos']} | "
                  f"{r['falsos_negativos']} | {r['falsos_positivos']} | {fmt(r['iou_medio'])} | "
                  f"{fmt(r['erro_canto_medio_pct'], 2)} | {fmt(r['tempo_medio_ms'], 1)} |")
    md += ["", f"* Taxa de acerto (recall @IoU 0,90): **{fmt(geral['taxa_acerto'], pct=True)}**",
           f"* Precisão: **{fmt(geral['precisao'], pct=True)}**",
           f"* Tempo p95: **{fmt(geral['tempo_p95_ms'], 1)} ms** por imagem (pré-processamento + detecção)", "",
           "## Comparação entre métodos", "",
           "| Configuração | Acertos | Imprecisos | FN | FP | IoU médio | Erro de canto (% diag.) | Tempo médio (ms) |",
           "|---|---|---|---|---|---|---|---|"]
    for c in comparacao:
        md.append(f"| {c['metodo']} | {c['acertos_iou90']} | {c['imprecisos']} | {c['falsos_negativos']} | "
                  f"{c['falsos_positivos']} | {fmt(c['iou_medio'])} | {fmt(c['erro_canto_medio_pct'], 2)} | "
                  f"{fmt(c['tempo_medio_ms'], 1)} |")
    md += ["", "## Vídeo: efeito da suavização (EMA)", "",
           "| Alfa | Taxa de detecção | Quadros mantidos em memória | Erro médio (px) | Tremor médio (px) | FPS de processamento |",
           "|---|---|---|---|---|---|"]
    for v in video:
        md.append(f"| {v['alfa']:.1f} | {fmt(v['taxa_deteccao'], pct=True)} | {v['quadros_persistidos']} | "
                  f"{fmt(v['erro_canto_medio_px'], 2)} | {fmt(v['tremor_medio_px'], 2)} | {fmt(v['fps_processamento'], 1)} |")
    md += ["", "## Resultado por imagem", "",
           "| Imagem | Categoria | Classe | Confiança | IoU | Erro (px) | Tempo (ms) |",
           "|---|---|---|---|---|---|---|"]
    for l in linhas:
        md.append(f"| {l['imagem']} | {l['categoria']} | {l['classe']} | {l['confianca']:.2f} | "
                  f"{fmt(l['iou'])} | {fmt(l['erro_canto_px'], 1)} | {l['tempo_ms']:.1f} |")
    md += ["", "Legenda: VP = acerto; VP_impreciso = detectou, mas IoU < 0,90; FN = não detectou; "
           "FP = detectou documento onde não há; VN = corretamente não detectou.", ""]
    (RESULTS_DIR / "RESULTADOS.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(geral, indent=2, ensure_ascii=False))
    for v in video:
        print(v)
    print(f"Arquivos salvos em {RESULTS_DIR}")


if __name__ == "__main__":
    main()
