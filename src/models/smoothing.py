"""Estabilização temporal dos cantos em vídeo/câmera.

A detecção quadro a quadro "treme" alguns pixels porque o Canny e a
aproximação poligonal variam com ruído. Usamos uma média móvel exponencial
(EMA) em cada coordenada:

    c_t = alfa · medido_t + (1 - alfa) · c_{t-1}

* alfa = 1  -> sem suavização (resposta imediata, mais tremor);
* alfa pequeno -> mais estável, porém com atraso quando o documento se move.

Regras adicionais:
* se o documento "salta" (deslocamento médio > 15% da diagonal), reiniciamos
  a média para não arrastar o contorno pela tela;
* se a detecção falhar, mantemos o último resultado por até N quadros
  (persistência), evitando que o contorno pisque.
"""
from __future__ import annotations

import numpy as np

from src.geometry import ordenar_cantos


class SuavizadorCantos:
    def __init__(self, alfa: float = 0.5, quadros_persistencia: int = 5, limiar_salto: float = 0.15):
        self.alfa = float(alfa)
        self.quadros_persistencia = int(quadros_persistencia)
        self.limiar_salto = float(limiar_salto)
        self.estado: np.ndarray | None = None
        self.perdidos = 0

    def reiniciar(self) -> None:
        self.estado = None
        self.perdidos = 0

    def atualizar(self, cantos: np.ndarray | None, diagonal: float) -> tuple[np.ndarray | None, bool]:
        """Recebe os cantos medidos (coord. originais) ou None.

        Retorna (cantos_suavizados, eh_persistido). ``eh_persistido`` indica que
        o valor veio da memória porque a detecção do quadro atual falhou.
        """
        if cantos is None:
            self.perdidos += 1
            if self.estado is not None and self.perdidos <= self.quadros_persistencia:
                return self.estado.copy(), True
            self.estado = None
            return None, False

        self.perdidos = 0
        cantos = ordenar_cantos(cantos)
        if self.estado is None:
            self.estado = cantos.copy()
        else:
            deslocamento = float(np.linalg.norm(cantos - self.estado, axis=1).mean())
            if deslocamento > self.limiar_salto * diagonal:
                self.estado = cantos.copy()
            else:
                self.estado = self.alfa * cantos + (1.0 - self.alfa) * self.estado
        return self.estado.copy(), False
