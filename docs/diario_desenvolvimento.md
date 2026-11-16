# Diário de desenvolvimento

Registro das etapas pedidas no enunciado (Proposta → Protótipo → Intermediária → Final).
Cada entrada deve ter data, responsável e evidência (commit, arquivo ou print).

## Planejamento

| Etapa | Marco | Evidência esperada | Data | Responsável |
|---|---|---|---|---|
| 1 | Proposta: problema, usuário, entrada/saída, técnica, dados, divisão de tarefas | README seções 1 e 13 | | Mateus Costa Guimarães (todos) |
| 2 | Protótipo visual: leitura, pré-processamento, primeira saída visual | `src/acquisition`, `src/preprocessing`, `src/visualization` | | Hiago, Heithor, John Lucas |
| 3 | Versão intermediária: detector, interface, parâmetros, testes, métricas | `src/models`, `app.py`, `src/metrics`, `tests/` | | Daniel, Lucas Oliveira, Pedro Henrique |
| 4 | Versão final: estabilidade, documentação, avaliação, apresentação | `docs/`, apresentação, vídeo | | Mateus Costa Guimarães (todos) |

## Registro

### 16/11/2026: versão inicial gerada com apoio de IA

* Estrutura do repositório, pipeline completa, interface Streamlit, testes e gerador de dados
  sintéticos criados com o Claude Code (ver README, seção 12).
* Decisões e problemas encontrados nesta etapa:
  * **Escolha do método clássico** (contornos + approxPolyDP) em vez de rede neural: geometria
    simples, explicável, sem GPU e sem baixar pesos.
  * **Dados sintéticos** para ter gabarito exato e evitar imagens com dados pessoais.
  * **Precisão dos cantos**: a primeira versão tinha erro mediano de ~5 px (IoU ≈ 0,977) porque o
    contorno é extraído de bordas dilatadas na imagem reduzida. O refinamento com `cornerSubPix`
    na resolução original reduziu o erro para ~0,3 px (IoU mediano ≈ 0,999).
  * **Desempenho no vídeo**: a primeira versão processava ~5 FPS. Ao medir por etapa, o gargalo
    era o desenho (~120 ms): cada rótulo fazia uma mistura alfa em ponto flutuante sobre a imagem
    inteira. Com `addWeighted`/`copyTo` e escurecimento só na ROI dos painéis, caiu para ~11 ms
    (~35 FPS de processamento).
  * **Vídeo de demonstração** tinha 15 MB; a resolução e o ruído por quadro foram reduzidos para
    ~5 MB.

### _[data]_: _[próxima entrada da equipe]_

* O que foi feito:
* Quem fez:
* Problemas e soluções:
* Evidência (commit/print):
