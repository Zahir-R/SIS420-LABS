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
make validate
make train
```

`make train` entrena Bellman y Q-Learning para los cinco niveles. Las políticas quedan en `shared/policies/level01/` … `level05/`.

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

- Hay cinco niveles en `shared/levels/level01.json` … `level05.json`.
- `rows` debe tener `height` filas de `width` caracteres.
- `S` define el inicio del agente y `G` la meta; `agent_start`/`goal` se derivan de esas letras.
- `human_start` indica dónde aparece el saboteador.
- `time_limit_s` define el cronómetro.
- `trap_slots` y `fire_lanes` son metadatos del nivel.

La política Q-Learning/Bellman es el único decisor. El estado incluye posición, stun, tiempo y amenaza relativa: tipo, dirección y distancia. El modo aleatorio no recibe ayuda de ningún planificador.

Las trampas son persistentes: no desaparecen por tiempo, solo cuando el agente colisiona con ellas. Si el agente pasa 2 ticks sin cambiar de celda o su siguiente paso cruza una trampa, se calcula una ruta local de desvío y se mantiene hasta completarla o hasta que un peligro la invalide. Si no existe ruta, acepta el impacto.

Bellman resuelve un modelo nominal aproximado porque las trampas pueden colocarse en cualquier celda. Q-Learning aprende de episodios muestreados con amenazas persistentes.

Después de cambiar el nivel o el tiempo:

```sh
make validate
make train
```

`make validate` comprueba también que la meta sea alcanzable con las reglas de movimiento actuales.
