"""3D OpenGL visualisation of the SmallholderFarmEnv (pygame-ce + PyOpenGL).

Scene: a terraced maize plot whose plants grow / yellow / get diseased according to the
state, soil colour reflecting moisture, rain particles, the farmer's house with sacks of
harvested produce (home stock) and the cooperative warehouse with a fill gauge
(cooperative stock). A 2D HUD shows the week, the advisory action sent, reward and the
market price curve.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pygame
from OpenGL.GL import (
    GL_BLEND, GL_COLOR_BUFFER_BIT, GL_COLOR_MATERIAL, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
    GL_DIFFUSE, GL_AMBIENT_AND_DIFFUSE, GL_FRONT_AND_BACK, GL_LIGHT0, GL_LIGHTING, GL_LINES,
    GL_MODELVIEW, GL_NORMALIZE, GL_ONE_MINUS_SRC_ALPHA, GL_POSITION, GL_PROJECTION, GL_QUADS,
    GL_RGB, GL_RGBA, GL_SRC_ALPHA, GL_TRIANGLES, GL_UNSIGNED_BYTE, GL_AMBIENT, GL_PACK_ALIGNMENT,
    GL_UNPACK_ALIGNMENT, GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE,
    glBegin, glBlendFunc, glClear, glClearColor, glColor3f, glColor4f, glColorMaterial,
    glDisable, glDrawPixels, glEnable, glEnd, glLightfv, glLineWidth, glLoadIdentity,
    glMatrixMode, glNormal3f, glOrtho, glPixelStorei, glPopMatrix, glPushMatrix, glReadPixels,
    glRasterPos2i, glRotatef, glLightModeli, glScalef, glTranslatef, glVertex3f, glViewport,
)
from OpenGL.GLU import gluLookAt, gluPerspective

from environment.custom_env import ACTIONS, MAX_WEEKS

W, H = 960, 600
GRID = 8  # 8x8 maize plants


def _box(sx, sy, sz):
    """Axis-aligned box centred at origin, base on y=0."""
    x, z = sx / 2, sz / 2
    faces = [
        ((0, 1, 0), [(-x, sy, -z), (x, sy, -z), (x, sy, z), (-x, sy, z)]),
        ((0, -1, 0), [(-x, 0, -z), (-x, 0, z), (x, 0, z), (x, 0, -z)]),
        ((0, 0, 1), [(-x, 0, z), (-x, sy, z), (x, sy, z), (x, 0, z)]),
        ((0, 0, -1), [(-x, 0, -z), (x, 0, -z), (x, sy, -z), (-x, sy, -z)]),
        ((1, 0, 0), [(x, 0, -z), (x, 0, z), (x, sy, z), (x, sy, -z)]),
        ((-1, 0, 0), [(-x, 0, -z), (-x, sy, -z), (-x, sy, z), (-x, 0, z)]),
    ]
    glBegin(GL_QUADS)
    for n, vs in faces:
        glNormal3f(*n)
        for v in vs:
            glVertex3f(*v)
    glEnd()


def _pyramid(sx, sy, sz):
    x, z = sx / 2, sz / 2
    apex = (0, sy, 0)
    base = [(-x, 0, -z), (x, 0, -z), (x, 0, z), (-x, 0, z)]
    glBegin(GL_TRIANGLES)
    for i in range(4):
        a, b = base[i], base[(i + 1) % 4]
        nx, nz = (a[0] + b[0]) / 2, (a[2] + b[2]) / 2
        glNormal3f(nx, 0.6, nz)
        glVertex3f(*a)
        glVertex3f(*apex)
        glVertex3f(*b)
    glEnd()


def _leaf(length, angle_deg, height):
    glPushMatrix()
    glTranslatef(0, height, 0)
    glRotatef(angle_deg, 0, 1, 0)
    glBegin(GL_TRIANGLES)
    glNormal3f(0, 1, 0)
    glVertex3f(0, 0, -0.07)
    glVertex3f(length, -0.25 * length, 0)
    glVertex3f(0, 0, 0.07)
    glEnd()
    glPopMatrix()


class FarmRenderer:
    def __init__(self, visible: bool = True):
        if not visible:
            os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "-2000,-2000")
        pygame.init()
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if not visible:
            flags |= pygame.HIDDEN
        self.screen = pygame.display.set_mode((W, H), flags)
        pygame.display.set_caption("Smallholder Farm Advisory - RL agent")
        self.font = pygame.font.SysFont("segoeui,arial", 18, bold=True)
        self.small = pygame.font.SysFont("segoeui,arial", 14)
        self.big = pygame.font.SysFont("segoeui,arial", 24, bold=True)
        self.visible = visible
        self.frame_idx = 0
        rng = np.random.default_rng(7)
        self.jitter = rng.uniform(-0.12, 0.12, size=(GRID, GRID, 2))
        self.leaf_angles = rng.uniform(0, 360, size=(GRID, GRID))
        self.sick_order = rng.permutation(GRID * GRID).reshape(GRID, GRID) / (GRID * GRID)
        self.rain = rng.uniform([-6, 0, -6], [6, 8, 6], size=(250, 3))

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_POSITION, (4.0, 10.0, 6.0, 0.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.9, 0.9, 0.85, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.35, 0.35, 0.38, 1.0))

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # ------------------------------------------------------------------ scene parts
    def _ground(self, moisture):
        dry, wet = np.array([0.72, 0.56, 0.36]), np.array([0.33, 0.22, 0.13])
        c = dry + (wet - dry) * moisture
        # surrounding grass
        glColor3f(0.36, 0.55, 0.25)
        glPushMatrix()
        glTranslatef(0, -0.05, 0)
        _box(20, 0.05, 20)
        glPopMatrix()
        # plot soil
        glColor3f(*c)
        glPushMatrix()
        glTranslatef(0, -0.02, 0)
        _box(5.4, 0.06, 5.4)
        glPopMatrix()
        # furrows
        glColor3f(*(c * 0.8))
        for i in range(GRID):
            glPushMatrix()
            glTranslatef(0, 0.0, -2.4 + i * 0.68)
            _box(5.2, 0.05, 0.12)
            glPopMatrix()

    def _crops(self, s):
        if s.harvested:
            glColor3f(0.75, 0.65, 0.35)  # stubble
            for i in range(GRID):
                for j in range(GRID):
                    glPushMatrix()
                    glTranslatef(-2.4 + i * 0.68, 0, -2.4 + j * 0.68)
                    _box(0.05, 0.12, 0.05)
                    glPopMatrix()
            return
        height = 0.15 + 1.6 * s.growth
        healthy = np.array([0.18, 0.62, 0.16])
        stressed = np.array([0.78, 0.72, 0.22])
        diseased = np.array([0.45, 0.25, 0.10])
        base = stressed + (healthy - stressed) * s.health
        sick_frac = min(1.0, s.infection * 1.3)
        for i in range(GRID):
            for j in range(GRID):
                x = -2.4 + i * 0.68 + self.jitter[i, j, 0]
                z = -2.4 + j * 0.68 + self.jitter[i, j, 1]
                sick = self.sick_order[i, j] < sick_frac
                col = diseased if sick else base
                h = height * (0.75 if sick else 1.0)
                glPushMatrix()
                glTranslatef(x, 0, z)
                glColor3f(*(col * 0.8))
                _box(0.06, h, 0.06)
                glColor3f(*col)
                n_leaves = 2 + int(4 * s.growth)
                for k in range(n_leaves):
                    _leaf(0.2 + 0.3 * s.growth, self.leaf_angles[i, j] + k * 137, h * (0.3 + 0.6 * k / max(1, n_leaves)))
                if s.growth > 0.85:  # maize cob / tassel
                    glColor3f(0.95, 0.82, 0.25)
                    glPushMatrix()
                    glTranslatef(0.05, h * 0.6, 0)
                    _box(0.07, 0.18, 0.07)
                    glPopMatrix()
                glPopMatrix()

    def _house(self, s):
        glPushMatrix()
        glTranslatef(-4.6, 0, 1.6)
        glColor3f(0.82, 0.62, 0.45)
        _box(1.8, 1.1, 1.6)
        glTranslatef(0, 1.1, 0)
        glColor3f(0.55, 0.30, 0.18)
        _pyramid(2.1, 0.8, 1.9)
        glPopMatrix()
        # sacks at home
        n = int(round(s.home_stock * 8))
        for k in range(n):
            glPushMatrix()
            glTranslatef(-5.2 + (k % 4) * 0.36, 0.0 + (k // 4) * 0.3, 2.9)
            glColor3f(0.88, 0.84, 0.7)
            _box(0.3, 0.3, 0.3)
            glPopMatrix()

    def _coop(self, s):
        glPushMatrix()
        glTranslatef(5.2, 0, -1.0)
        glColor3f(0.75, 0.78, 0.82)
        _box(2.2, 1.6, 2.0)
        glTranslatef(0, 1.6, 0)
        glColor3f(0.25, 0.42, 0.65)
        _pyramid(2.5, 0.7, 2.3)
        glPopMatrix()
        # fill gauge
        fill = min(1.0, s.coop_stock / 1.5)
        glPushMatrix()
        glTranslatef(5.2, 0, 0.15)
        glColor3f(0.2, 0.2, 0.2)
        _box(0.35, 2.0, 0.05)
        glTranslatef(0, 0, 0.04)
        glColor3f(0.95, 0.75, 0.15)
        _box(0.28, 2.0 * fill + 0.01, 0.05)
        glPopMatrix()

    def _rain(self, s):
        if not s.rained:
            return
        glDisable(GL_LIGHTING)
        glColor4f(0.6, 0.7, 1.0, 0.7)
        glLineWidth(1.5)
        glBegin(GL_LINES)
        off = (self.frame_idx * 0.6) % 8
        for x, y, z in self.rain:
            yy = (y - off) % 8
            glVertex3f(x, yy, z)
            glVertex3f(x, yy + 0.35, z)
        glEnd()
        glEnable(GL_LIGHTING)

    # ------------------------------------------------------------------ HUD
    def _hud_surface(self, env):
        s = env.s
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        panel = pygame.Surface((330, 250), pygame.SRCALPHA)
        panel.fill((15, 25, 20, 190))
        surf.blit(panel, (10, 10))
        act = ACTIONS[env.last_action] if env.last_action is not None else "-"
        lines = [
            (self.big, f"Week {s.week}/{MAX_WEEKS}", (255, 255, 255)),
            (self.font, f"Advisory (SMS): {act}", (255, 220, 120)),
            (self.small, f"Step reward {env.last_reward:+.2f}   Return {env.episode_return:+.2f}", (220, 220, 220)),
            (self.small, f"Growth {s.growth:.2f}   Health {s.health:.2f}   Moisture {s.moisture:.2f}", (200, 240, 200)),
            (self.small, f"Infection {s.infection:.2f}   Diagnosis {s.diag_level:.2f} ({min(s.weeks_since_diag, 10)} wk ago)", (240, 200, 200)),
            (self.small, f"Home stock {s.home_stock:.2f} t   Coop stock {s.coop_stock:.2f} t", (230, 230, 200)),
            (self.small, f"Price index {env.price():.2f}   Sold {s.sold_tonnes:.2f} t   Lost {s.lost_tonnes:.2f} t", (200, 220, 255)),
            (self.small, f"Profit {s.revenue - s.costs:+.2f} (x25k RWF)", (180, 255, 180)),
        ]
        y = 18
        for f, t, c in lines:
            surf.blit(f.render(t, True, c), (22, y))
            y += 30 if f is self.big else 26
        # price sparkline
        px, py, pw, ph = W - 290, 12, 275, 130
        box = pygame.Surface((pw, ph), pygame.SRCALPHA)
        box.fill((15, 20, 35, 190))
        surf.blit(box, (px, py))
        surf.blit(self.small.render("Market price index (eSoko-like API)", True, (220, 220, 255)), (px + 8, py + 4))
        hist = s.price_history
        if len(hist) > 1:
            pts = [
                (px + 10 + (pw - 20) * k / MAX_WEEKS, py + ph - 10 - (ph - 40) * (v - 0.4) / 1.4)
                for k, v in enumerate(hist)
            ]
            pygame.draw.lines(surf, (255, 200, 80), False, pts, 2)
        legend = "Brown plants = diseased   Sacks = home stock   Yellow gauge = coop storage"
        surf.blit(self.small.render(legend, True, (255, 255, 255)), (12, H - 26))
        return surf

    def _blit_hud(self, surf):
        data = pygame.image.tobytes(surf, "RGBA", True)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, W, 0, H, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glRasterPos2i(0, 0)
        glDrawPixels(W, H, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    # ------------------------------------------------------------------ main draw
    def draw(self, env):
        if self.visible:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit
        s = env.s
        self.frame_idx += 1
        sky = (0.55, 0.62, 0.72) if s.rained else (0.53, 0.78, 0.95)
        glViewport(0, 0, W, H)
        glClearColor(*sky, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, W / H, 0.1, 100)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        ang = math.radians(-12 + 0.6 * s.week + 0.15 * (self.frame_idx % 40))
        gluLookAt(11.5 * math.sin(ang), 6.0, 11.5 * math.cos(ang), 0, 0.4, -0.5, 0, 1, 0)

        glLightfv(GL_LIGHT0, GL_POSITION, (4.0, 10.0, 6.0, 0.0))
        self._ground(s.moisture)
        self._crops(s)
        self._house(s)
        self._coop(s)
        self._rain(s)
        self._blit_hud(self._hud_surface(env))

        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        raw = glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)[::-1].copy()
        pygame.display.flip()
        return frame

    def close(self):
        pygame.display.quit()
        pygame.quit()
