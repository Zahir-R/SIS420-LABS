# Stun & Dash

Agente vs. adversario en plataformas estocásticas.

- `python/`: entorno MDP, Bellman/Value Iteration y Q-Learning tabular.
- `web/`: vertical slice Phaser 3.
- `shared/`: nivel y políticas exportadas.
- `shared/assets/`: Pixel Adventure (CC0, sin atribución obligatoria).

## Entorno

El proyecto incluye su propio `shell.nix` con Python 3.11, `uv` y Node 22.

```sh
direnv allow
uv sync
```

## Entrenar

```sh
uv run python python/train.py --mode bellman
uv run python python/train.py --mode qlearning --episodes 4000 --seed 1
```

## Web

```sh
cd web
npm install
npm run dev
```

Abre `http://localhost:5173/`.

Controles del saboteador:

- **WASD**: moverse libremente, sin colisión.
- **E**: colocar trampa en la celda actual.
- **Q** o **F**: disparar horizontalmente hacia el cursor.
- **R**: reiniciar.
- **M**: menú.

Para entrar directo a una política:

```text
http://localhost:5173/?play=bellman
http://localhost:5173/?play=qlearning
http://localhost:5173/?play=random
```

## Niveles

No hay editor visual todavía. Los niveles se editan en `shared/levels/level01.json`.

- `rows` debe tener `height` filas de `width` caracteres.
- `S` define el inicio del agente y `G` la meta; `agent_start`/`goal` se derivan de esas letras.
- `human_start` indica dónde aparece el saboteador.
- `time_limit_s` define el cronómetro.
- `trap_slots` y `fire_lanes` son metadatos del nivel.

El planificador compara el retorno esperado de la ruta Q/Bellman contra el retorno esperado de la ruta de desvío. No elige solo por tiempo: también considera paso, stun, progreso, meta y timeout. La política se consulta en cada tick desde el estado real.

Si el agente pasa 4 ticks consecutivos sin cambiar de celda, entra en **FORCED DETOUR**: trata las celdas de proyectiles como bloqueadas y busca una ruta segura sin esperas. La ruta forzada se mantiene hasta completarla; si un peligro nuevo la invalida, se recalcula. Si no existe alternativa, acepta el impacto y avanza.

Después de cambiar el nivel o el tiempo:

```sh
make validate
make train
```

`make validate` comprueba también que la meta sea alcanzable con las reglas de movimiento actuales.
