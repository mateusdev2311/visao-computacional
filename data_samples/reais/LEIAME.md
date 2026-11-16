# Fotos reais (opcional)

Coloque aqui fotos reais de documentos **tiradas pela própria equipe**, de documentos
sem dados pessoais (ex.: uma página impressa deste README, uma folha de exercícios).

* O conteúdo desta pasta é ignorado pelo Git (`.gitignore`), exceto este arquivo.
  Antes de versionar alguma foto, confirme que não há rostos, nomes, CPF, endereços
  ou assinaturas reais.
* Para medir IoU em fotos reais, anote os 4 cantos (ordem: superior-esquerdo,
  superior-direito, inferior-direito, inferior-esquerdo) num `anotacoes.json` no
  mesmo formato de `data_samples/imagens/anotacoes.json` e rode a avaliação
  apontando para esta pasta:

```python
from pathlib import Path
from src.config import Parametros
from src.metrics.avaliacao import avaliar_imagens
linhas, geral, por_categoria = avaliar_imagens(Parametros(), pasta=Path("data_samples/reais"))
print(geral)
```
