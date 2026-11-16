"""Métricas de desempenho: tempo por etapa (ms) e quadros por segundo."""
from __future__ import annotations

import time
from collections import deque


class Cronometro:
    """Acumula o tempo de cada etapa nomeada.

    Uso:
        cron = Cronometro()
        with cron.etapa("deteccao"):
            ...
        cron.tempos_ms -> {"deteccao": 3.2}
    """

    def __init__(self):
        self.tempos_ms: dict[str, float] = {}

    def etapa(self, nome: str):
        return _Etapa(self, nome)

    @property
    def total_ms(self) -> float:
        return sum(self.tempos_ms.values())


class _Etapa:
    def __init__(self, cron: Cronometro, nome: str):
        self.cron, self.nome = cron, nome

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = (time.perf_counter() - self.t0) * 1000.0
        self.cron.tempos_ms[self.nome] = self.cron.tempos_ms.get(self.nome, 0.0) + dt
        return False


class MedidorFPS:
    """FPS médio numa janela deslizante dos últimos N quadros."""

    def __init__(self, janela: int = 30):
        self.instantes: deque[float] = deque(maxlen=janela)

    def marcar(self) -> float:
        self.instantes.append(time.perf_counter())
        return self.fps

    @property
    def fps(self) -> float:
        if len(self.instantes) < 2:
            return 0.0
        dt = self.instantes[-1] - self.instantes[0]
        return (len(self.instantes) - 1) / dt if dt > 0 else 0.0
