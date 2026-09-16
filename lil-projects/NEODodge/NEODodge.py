from model import HazardModel
import numpy as np
import pygame
import json, os, math

MODEL = "legacy"

_R = np.load(os.path.join(os.path.dirname(__file__), "game_data.npz"))
_SP = json.load(open(os.path.join(os.path.dirname(__file__), "stats_full.json")))
IDX = {c: i for i, c in enumerate(_SP["features"])}

if MODEL == "legacy":
    cols5 = ["magnitude", "diam_min_m", "diam_max_m", "vel_km_h", "miss_km"]
    X5 = _R["X"][:, [IDX[c] for c in cols5]]
    hs = HazardModel("hazard_model.pt", "stats.json")
    _X = X5.astype(np.float32)
else:
    hs = HazardModel("hazard_model_full.pt", "stats_full.json")
    _X = _R["X"].astype(np.float32)

_probs = hs.predict(_X).numpy()
_preds = _probs > 0.5
_truth = _R["label"].astype(bool)

# Predicts and Oracle
AU = 149597870.7
P_THRESH = 0.5
ORACLE_MAG_MAX = 22.0
ORACLE_MISS_AU_MAX = 0.05
# Game basics
# SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
BG_COLOR = (10, 10, 25)
FPS = 60
SPAWN_MS = 100
FONT_SIZE = 24
HUD_COLOR = (220, 220, 220)
# Player basics
PLAYER_COLOR = (120, 200, 255)
PLAYER_RADIUS = 12
PLAYER_VEL = 6
PLAYER_SIZE = 28
PLAYER_INIT_OFFSET = 80
PLAYER_LIVES = 3
# Asteroid basics
asteroids = []
last_spawn = 0
AST_MIN_R = 6
AST_R_PER_LOG = 2.5
AST_MIN_VY = 180
AST_VY_RANGE = 170
AST_SPAWN_MARGIN = 30
AST_TOP_OFFSET = 20
AST_REMOVE_MARGIN = 50
CAT = {(True, True): "TP", (False, False): "TN",
        (True, False): "FP", (False, True): "FN"}
CAT_COLOR = {"TP": (220, 60, 60), "TN": (150, 150, 160),
             "FP": (230, 200, 60), "FN": (250, 130, 40)}
FP_RING_COLOR = (255, 240, 150)
FN_BLINK_MS = 250
anomalies = {"FP": 0, "FN": 0}


pygame.init()
player_x, player_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT - PLAYER_INIT_OFFSET
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("NEODodge")
clock = pygame.time.Clock()
lives, score, game_over = PLAYER_LIVES, 0, False

def loop():
    running = True
    while running:
        clock.tick(FPS)
        running = handle_events()
        keys = pygame.key.get_pressed()
        if not game_over:
            update(pygame.time.get_ticks(), keys)
        elif keys[pygame.K_r]:
            reset()
        draw()
        pygame.display.flip()
    pygame.quit()


def spawn_asteroid():
    i = np.random.randint(len(_R["X"]))
    p = float(_probs[i])
    pred = _preds[i]
    truth = bool(_truth[i])
    diam_max = float(_R["diam_max"][i])
    vel = float(_R["vel"][i])
    vmin, vmax = _SP["min"][IDX["vel_km_h"]], _SP["max"][IDX["vel_km_h"]]
    rad = AST_MIN_R + AST_R_PER_LOG * np.log1p(diam_max)
    fall = (np.log1p(vel) - vmin) / (vmax - vmin)
    return {"x": np.random.uniform(AST_SPAWN_MARGIN, SCREEN_WIDTH - AST_SPAWN_MARGIN),
            "y": -AST_TOP_OFFSET,
            "vy": AST_MIN_VY + fall * AST_VY_RANGE,
            "rad": rad,
            "p": p,
            "pred": pred,
            "truth": truth,
            "cat": CAT[(pred, truth)],
            "kind": "rock"}


def hitbox(a):
    return a["x"], a["y"], a["rad"]


def move(keys):
    global player_x, player_y
    if keys[pygame.K_LEFT]:
        player_x -= PLAYER_VEL
    if keys[pygame.K_RIGHT]:
        player_x += PLAYER_VEL
    if keys[pygame.K_UP]:
        player_y -= PLAYER_VEL
    if keys[pygame.K_DOWN]:
        player_y += PLAYER_VEL

    player_x = max(PLAYER_RADIUS, min(SCREEN_WIDTH - PLAYER_RADIUS, player_x))
    player_y = max(PLAYER_RADIUS, min(SCREEN_HEIGHT - PLAYER_RADIUS, player_y))


def reset():
    global player_x, player_y, lives, score, asteroids, last_spawn, game_over, anomalies
    player_x, player_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT - PLAYER_INIT_OFFSET
    lives, score, game_over = PLAYER_LIVES, 0, False
    asteroids, last_spawn, anomalies = [], 0, {"FP": 0, "FN": 0}


def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True


def update(now, keys):
    global asteroids, last_spawn, lives, score, game_over
    move(keys)
    if now - last_spawn >= SPAWN_MS:
        asteroids.append(spawn_asteroid())
        last_spawn = now
    on_screen = []
    for a in asteroids:
        a["y"] += a["vy"] / FPS
        ax, ay, ar = hitbox(a)
        colliding = math.hypot(ax - player_x, ay - player_y) <= ar + PLAYER_RADIUS
        if a["cat"] == "FN" and a["kind"] == "rock" and ay >= player_y:
            anomalies["FN"] += 1
            on_screen.extend(explode(a))
            continue
        if colliding and deals_dmg(a):
            take_damage()
            if game_over:
                return
            continue
        if colliding:
            on_screen.append(a)
            continue
        if ay < SCREEN_HEIGHT + AST_REMOVE_MARGIN:
            on_screen.append(a)
        else:
            if a["kind"] != "shard":
                score += 1
                if a["cat"] == "FP":
                    anomalies["FP"] += 1
    asteroids = on_screen


def draw():
    font = pygame.font.SysFont(None, FONT_SIZE)
    screen.fill(BG_COLOR)
    hud1 = font.render(f"Vidas: {lives}  Score: {score}", True, HUD_COLOR)
    hud2 = font.render(f"Anomalías: FP {anomalies['FP']} | FN {anomalies['FN']}", True, HUD_COLOR)
    screen.blit(hud1, (12, 8))
    screen.blit(hud2, (12, 8 + FONT_SIZE))
 
    for a in asteroids:
        now_ms = pygame.time.get_ticks()
        x, y, r = int(a["x"]), int(a["y"]), int(a["rad"])
        if a["cat"] == "TP":
            pygame.draw.circle(screen, CAT_COLOR["TP"], (x, y), r)
        elif a["cat"] == "TN":
            pygame.draw.circle(screen, CAT_COLOR["TN"], (x, y), r)
        elif a["cat"] == "FP":
            pygame.draw.circle(screen, CAT_COLOR["FP"], (x, y), r)
            pulse = 3 + int(3 * math.sin(now_ms / 300))
            pygame.draw.circle(screen, FP_RING_COLOR, (x, y), r + 6, width=max(1, pulse))
        elif a["kind"] == "shard":
            pygame.draw.circle(screen, CAT_COLOR["TP"], (x, y), r)
        else:
            if (now_ms // FN_BLINK_MS) % 2 == 0:
                pygame.draw.circle(screen, CAT_COLOR["FN"], (x, y), r)

    puntos = [(player_x, player_y - PLAYER_SIZE // 2),
                  (player_x - PLAYER_SIZE // 2, player_y + PLAYER_SIZE // 2),
                  (player_x + PLAYER_SIZE // 2, player_y + PLAYER_SIZE // 2)]
    pygame.draw.polygon(screen, PLAYER_COLOR, puntos)
    if game_over:
        go = font.render("GAME OVER - Pulsa R para reiniciar", True, HUD_COLOR)
        screen.blit(go, go.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))

def take_damage():
    global lives, game_over
    lives -= 1
    if lives <= 0:
        game_over = True


def deals_dmg(a):
    return (a["truth"] and (a["kind"] == "shard" or a["cat"] != "FN"))


def explode(a):
    return [{
        "x": a["x"] + dx,
        "y": a["y"],
        "vy": a["vy"] * 1.4,
        "rad": a["rad"] * 0.5,
        "pred": a["pred"],
        "truth": True,
        "cat": "FN",
        "kind": "shard"
    } for dx in (-a["rad"] * 0.7, 0.0, a["rad"] * 0.7)]


if __name__ == "__main__":
    loop()


