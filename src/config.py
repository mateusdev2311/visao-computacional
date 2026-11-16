"""Parâmetros ajustáveis da pipeline.

Todos os valores que o usuário pode alterar pela interface ficam concentrados
nesta dataclass. Assim, a interface (app.py), o script de avaliação e os testes
usam exatamente a mesma configuração, sem números "mágicos" espalhados no código.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path

# Raiz do projeto calculada a partir deste arquivo -> caminhos sempre relativos.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_samples"
RESULTS_DIR = PROJECT_ROOT / "docs" / "resultados"

METODOS = ("auto", "bordas", "limiar")
MODOS_SAIDA = ("colorido", "cinza", "pb")


@dataclass
class Parametros:
    # --- Pré-processamento -------------------------------------------------
    altura_processamento: int = 500     # px; a imagem é reduzida para esta altura
    usar_clahe: bool = True             # equalização adaptativa de contraste
    kernel_blur: int = 5                # tamanho (ímpar) do filtro gaussiano
    canny_auto: bool = True             # limiares do Canny pela mediana da imagem
    canny_baixo: int = 50
    canny_alto: int = 150
    kernel_morf: int = 5                # fechamento morfológico das bordas

    # --- Detecção ----------------------------------------------------------
    metodo: str = "auto"                # "auto" | "bordas" | "limiar"
    epsilon_aprox: float = 0.02         # fração do perímetro no approxPolyDP
    area_minima: float = 0.10           # fração mínima da imagem ocupada pelo documento
    confianca_minima: float = 0.35      # abaixo disto o resultado é descartado
    refinar_cantos: bool = True         # ajuste subpixel dos cantos na resolução original

    # --- Saída / visualização ----------------------------------------------
    modo_saida: str = "colorido"        # "colorido" | "cinza" | "pb"
    rotacao_saida: int = 0              # 0, 90, 180 ou 270 graus
    opacidade_mascara: float = 0.35
    mostrar_candidatos: bool = False
    largura_exibicao: int = 900         # px da imagem mostrada na interface

    # --- Vídeo / câmera ----------------------------------------------------
    suavizacao: float = 0.5             # alfa da média móvel exponencial (1 = sem suavização)
    quadros_persistencia: int = 5       # quadros que o último resultado é mantido se a detecção falhar

    extras: dict = field(default_factory=dict)

    def validar(self) -> "Parametros":
        """Corrige valores fora da faixa em vez de deixar o OpenCV falhar."""
        self.altura_processamento = int(min(max(self.altura_processamento, 200), 1500))
        self.kernel_blur = _impar(self.kernel_blur, 1, 31)
        self.kernel_morf = _impar(self.kernel_morf, 1, 31)
        self.canny_baixo = int(min(max(self.canny_baixo, 0), 254))
        self.canny_alto = int(min(max(self.canny_alto, self.canny_baixo + 1), 255))
        self.epsilon_aprox = float(min(max(self.epsilon_aprox, 0.005), 0.1))
        self.area_minima = float(min(max(self.area_minima, 0.01), 0.95))
        self.confianca_minima = float(min(max(self.confianca_minima, 0.0), 1.0))
        self.opacidade_mascara = float(min(max(self.opacidade_mascara, 0.0), 1.0))
        self.suavizacao = float(min(max(self.suavizacao, 0.05), 1.0))
        self.quadros_persistencia = int(max(self.quadros_persistencia, 0))
        if self.metodo not in METODOS:
            self.metodo = "auto"
        if self.modo_saida not in MODOS_SAIDA:
            self.modo_saida = "colorido"
        if self.rotacao_saida not in (0, 90, 180, 270):
            self.rotacao_saida = 0
        return self

    def como_dict(self) -> dict:
        return asdict(self)


def _impar(valor: int, minimo: int, maximo: int) -> int:
    valor = int(min(max(valor, minimo), maximo))
    return valor if valor % 2 == 1 else valor + 1
