"""One pygame window on the main monitor: the game on the left, the brain on the right, a score chart below the brain."""
from __future__ import annotations
import os
import numpy as np

os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "120,80")   # primary monitor starts at (0, 0)
import pygame  # noqa: E402
import game as G  # noqa: E402

WIN_W, WIN_H = 1040, 560
SKY, PIPE, GROUND, BIRD = (112, 197, 206), (84, 168, 52), (222, 216, 149), (250, 200, 40)
BG, FG, DIM = (18, 20, 26), (225, 228, 235), (120, 125, 140)


class View:
    def __init__(self, title):
        pygame.init()
        self.s = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption(title)
        self.f = pygame.font.SysFont("consolas", 15); self.big = pygame.font.SysFont("consolas", 34, bold=True)
        self.quit = False; self.paused = False; self.fast = False
        self.history = []   # (label, value) points for the chart
        self.chart_title = "median pipes on dev games per round"

    def pump(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE)):
                self.quit = True
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                self.paused = not self.paused
            if e.type == pygame.KEYDOWN and e.key == pygame.K_f:
                self.fast = not self.fast
        return not self.quit

    def text(self, t, x, y, c=FG, font=None):
        self.s.blit((font or self.f).render(t, True, c), (x, y))

    def draw_game(self, g: G.Flappy, row: int, flap: int):
        s = self.s
        pygame.draw.rect(s, SKY, (0, 0, G.W, G.H))
        for k in range(3):
            x, c = g.px[row, k], g.pg[row, k]
            top, bot = c - G.GAP / 2, c + G.GAP / 2
            pygame.draw.rect(s, PIPE, (x, 0, G.PIPE_W, top)); pygame.draw.rect(s, PIPE, (x, bot, G.PIPE_W, G.GROUND - bot))
            pygame.draw.rect(s, (40, 90, 30), (x - 3, top - 18, G.PIPE_W + 6, 18)); pygame.draw.rect(s, (40, 90, 30), (x - 3, bot, G.PIPE_W + 6, 18))
        pygame.draw.rect(s, GROUND, (0, G.GROUND, G.W, G.H - G.GROUND))
        y = g.y[row]
        pygame.draw.circle(s, BIRD, (G.BIRD_X, int(y)), G.BIRD_R)
        pygame.draw.circle(s, (0, 0, 0), (G.BIRD_X + 5, int(y) - 4), 3)
        if flap:
            pygame.draw.polygon(s, (255, 255, 255), [(G.BIRD_X - 14, int(y)), (G.BIRD_X - 4, int(y) - 12), (G.BIRD_X - 2, int(y))])
        self.text(str(int(g.score[row])), G.W // 2 - 12, 30, (255, 255, 255), self.big)

    def draw_brain(self, senses, act_vec, assoc_idx, motor_idx, p_flap, label, x0=300):
        s = self.s
        pygame.draw.rect(s, BG, (x0, 0, WIN_W - x0, WIN_H))
        self.text(label, x0 + 10, 8)
        # senses: 4 rows of 12 bumps
        names = ["height vs gap", "speed", "distance to pipe", "next gap shift"]
        sv = senses.reshape(4, G.BUMPS)
        for r in range(4):
            self.text(names[r], x0 + 10, 38 + r * 26, DIM)
            for k in range(G.BUMPS):
                v = float(sv[r, k]); c = (int(40 + 200 * v), int(90 + 140 * v), 255)
                pygame.draw.rect(s, c if v > 0.05 else (35, 38, 48), (x0 + 160 + k * 18, 38 + r * 26, 15, 18))
        # association neurons as a grid of firing rates
        self.text(f"association ({len(assoc_idx)} neurons)", x0 + 10, 150, DIM)
        a = np.asarray(act_vec)[assoc_idx] if act_vec is not None else np.zeros(len(assoc_idx))
        a = a / (np.abs(a).max() + 1e-9)
        cols = 16
        for i, v in enumerate(a):
            v = float(v); c = (int(255 * max(v, 0)), int(120 * max(v, 0)), int(255 * max(-v, 0)))
            pygame.draw.rect(s, c if abs(v) > 0.03 else (35, 38, 48), (x0 + 10 + (i % cols) * 22, 172 + (i // cols) * 22, 19, 19))
        # motor neurons + flap probability
        ytop = 172 + ((len(a) + cols - 1) // cols) * 22 + 10
        self.text("motor", x0 + 10, ytop, DIM)
        m = np.asarray(act_vec)[motor_idx] if act_vec is not None else np.zeros(2)
        for i, (nm, v) in enumerate(zip(["glide", "FLAP"], m)):
            v = float(v); w = int(200 * min(1.0, abs(v)))
            pygame.draw.rect(s, (35, 38, 48), (x0 + 80, ytop + 2 + i * 22, 200, 16))
            pygame.draw.rect(s, (255, 170, 40) if i else (90, 160, 255), (x0 + 80, ytop + 2 + i * 22, w, 16))
            self.text(nm, x0 + 290, ytop + i * 22)
        if p_flap is not None:
            self.text(f"P(flap) {p_flap:.2f}", x0 + 380, ytop + 10)
        self.chart(x0 + 10, ytop + 60, WIN_W - x0 - 30, WIN_H - ytop - 70)

    def chart(self, x, y, w, h):
        s = self.s
        pygame.draw.rect(s, (28, 31, 40), (x, y, w, h))
        if not self.history:
            self.text("score chart fills in as rounds finish", x + 8, y + 8, DIM); return
        vals = [v for _, v in self.history]; top = max(100.0, max(vals))
        n = len(vals); bw = max(4, min(40, w // max(n, 1) - 4))
        for i, (lab, v) in enumerate(self.history):
            bh = int((h - 30) * v / top)
            pygame.draw.rect(s, (255, 170, 40) if lab.startswith("teach") else (0, 200, 160), (x + 6 + i * (bw + 4), y + h - 18 - bh, bw, bh))
        self.text(f"{self.chart_title} (max {top:.0f})", x + 8, y + 4, DIM)

    def flip(self):
        pygame.display.flip()
