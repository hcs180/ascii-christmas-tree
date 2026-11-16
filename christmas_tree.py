#!/usr/bin/env python3
"""
ASCII Christmas Tree - a rotating 3D point cloud drawn with coloured letters.

    pip install numpy
    python christmas_tree.py                 # Ctrl+C to quit
    python christmas_tree.py --speed 0.6 --fps 24

How it works
    1. Build the scene once as ~60k 3D points (foliage, garland, baubles,
       fairy lights, presents, ground), each with a normal and a colour.
    2. Every frame: rotate the points, project them to terminal cells,
       keep only the nearest point per cell (a z-buffer), shade it with a
       light direction, and pick a character by brightness.
    3. Print the whole frame with 24-bit ANSI colour codes.

Needs a terminal with 24-bit colour (Windows Terminal, VS Code, iTerm2,
macOS Terminal, most Linux terminals). Make the window big (120x50 or more)
and the font small for the best look.
"""

import argparse
import math
import os
import shutil
import sys
import time

import numpy as np

# ------------------------------------------------------------------ palette
GREEN_DARK = (0.05, 0.38, 0.12)
GREEN = (0.25, 0.88, 0.32)
GOLD = (1.00, 0.76, 0.20)
RED = (0.95, 0.14, 0.14)
SILVER = (0.80, 0.85, 0.95)
BLUE = (0.30, 0.50, 1.00)
WARM = (1.00, 0.86, 0.45)
WHITE = (1.00, 1.00, 1.00)
BROWN = (0.45, 0.26, 0.10)
GREEN_BOX = (0.10, 0.60, 0.35)
GROUND = (0.40, 0.45, 0.60)
HUD = (0.50, 0.55, 0.66)

RAMP = np.array(list(".:-~=+*^vxo%&#W@"))  # dark -> bright

# point kinds
SHADED, LIGHT, GROUND_K, STAR, TINSEL = 0, 1, 2, 3, 4
N_LIGHTS = 140

# ------------------------------------------------------------------ tree shape
Y_BASE, Y_TOP, R_BASE, TIERS = 0.25, 1.75, 0.80, 5
TARGET = np.array([0.0, 0.85, 0.0])  # point the camera looks at
DIST = 4.2                           # camera distance
PITCH = 0.30                         # camera looks down a little (radians)


def fib_sphere(n):
    """n points spread evenly on a unit sphere."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi),
                     np.sin(theta) * np.sin(phi)], axis=1)


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def envelope(u):
    """Tree radius at height fraction u (0 = bottom, 1 = tip), with tiers."""
    f = (u * TIERS) % 1.0                 # position inside the current tier
    return R_BASE * (1 - u) * (0.70 + 0.30 * (1 - f)), f


def tree_y(u, f):
    # each tier's skirt droops a little at its lower edge
    return Y_BASE + u * (Y_TOP - Y_BASE) - 0.06 * (1 - f) * (1 - u)


class Scene:
    """Collects point groups, then freezes them into flat numpy arrays."""

    def __init__(self):
        self.groups = []

    def add(self, pos, nrm, col, kind=SHADED, shine=0.0, phase=0.0, freq=0.0):
        n = len(pos)
        full = lambda v, shape: np.broadcast_to(np.asarray(v, float), shape).copy()
        self.groups.append(dict(
            pos=np.asarray(pos, float), nrm=full(nrm, (n, 3)), col=full(col, (n, 3)),
            kind=np.full(n, kind), shine=full(shine, (n,)),
            phase=full(phase, (n,)), freq=full(freq, (n,))))

    def freeze(self, rng):
        for key in self.groups[0]:
            setattr(self, key, np.concatenate([g[key] for g in self.groups]))
        self.grain = rng.random(len(self.pos))  # per-point character jitter
        self.star = self.kind == STAR
        self.n_lights = int(getattr(self, "_n_lights", 0))
        del self.groups
        return self


def build_scene(seed=7):
    rng = np.random.default_rng(seed)
    sc = Scene()

    # --- foliage: a fuzzy tiered cone shell
    n = 22000
    u = 1 - np.sqrt(rng.random(n))          # more points near the wide bottom
    a = rng.random(n) * 2 * np.pi
    env, f = envelope(u)
    depth = rng.random(n)
    r = env * (0.80 + 0.20 * depth)
    pos = np.stack([r * np.cos(a), tree_y(u, f), r * np.sin(a)], 1)
    nrm = normalize(np.stack([np.cos(a), np.full(n, 0.55), np.sin(a)], 1)
                    + 0.35 * rng.normal(size=(n, 3)))
    mix = (rng.random(n) ** 0.7 * (0.55 + 0.45 * depth))[:, None]
    col = np.array(GREEN_DARK) * (1 - mix) + np.array(GREEN) * mix
    col *= (1 - 0.35 * f)[:, None]          # tucked under the next tier = darker
    sc.add(pos, nrm, col, shine=0.15)

    # --- fairy lights on the branch tips
    tips = pos[depth > 0.9]
    tips = tips[rng.choice(len(tips), N_LIGHTS, replace=False)] * [1.05, 1.0, 1.05]
    bulb = fib_sphere(10) * 0.014
    lcol = np.array([WARM, WARM, WARM, WARM, (1.0, 0.97, 0.9), RED])[rng.integers(0, 6, N_LIGHTS)]
    sc.add((tips[:, None] + bulb[None]).reshape(-1, 3), (0, 1, 0),
           np.repeat(lcol, 10, 0), kind=LIGHT,
           phase=np.repeat(rng.random(N_LIGHTS) * 2 * np.pi, 10),
           freq=np.repeat(rng.uniform(1.5, 4.0, N_LIGHTS), 10))
    sc._n_lights = N_LIGHTS

    # --- gold tinsel garland: a spiral hugging the tiers
    m = 10000
    s = rng.random(m)
    u = 0.03 + 0.85 * s
    a = s * 3.25 * 2 * np.pi
    env, f = envelope(u)
    r = env * 1.04 + 0.015
    pos = np.stack([r * np.cos(a), tree_y(u, f), r * np.sin(a)], 1)
    pos += 0.028 * rng.normal(size=(m, 3))
    nrm = normalize(np.stack([np.cos(a), np.full(m, 0.3), np.sin(a)], 1)
                    + 0.8 * rng.normal(size=(m, 3)))
    col = np.array(GOLD) * rng.uniform(0.7, 1.0, (m, 1))
    col[rng.random(m) < 0.15] = (1.0, 0.95, 0.8)
    sc.add(pos, nrm, col, kind=TINSEL, shine=1.0)

    # --- baubles: mostly red, some gold, silver and blue
    sphere = fib_sphere(160)
    tier_p = np.arange(TIERS, 0, -1) / np.arange(TIERS, 0, -1).sum()
    for k in range(34):
        u = (rng.choice(TIERS, p=tier_p) + rng.uniform(0.08, 0.5)) / TIERS
        u = min(u, 0.88)
        a = k * 2.39996 + rng.random() * 0.4          # golden angle
        env, f = envelope(u)
        rad = 0.040 + 0.025 * (1 - u)
        c = np.array([env * np.cos(a), tree_y(u, f) - 0.03, env * np.sin(a)])
        colour = RED if k % 2 == 0 else [GOLD, SILVER, BLUE, RED][rng.integers(4)]
        sc.add(c + sphere * rad, sphere, colour, shine=0.9)

    # --- trunk
    n = 800
    a = rng.random(n) * 2 * np.pi
    y = rng.random(n) * (Y_BASE + 0.05)
    sc.add(np.stack([0.08 * np.cos(a), y, 0.08 * np.sin(a)], 1),
           np.stack([np.cos(a), np.zeros(n), np.sin(a)], 1), BROWN)

    # --- presents: boxes with a crossed ribbon and a bow
    gifts = [(0.30, 0.86, 0.30, 0.22, 0.26, 0.4, RED, GOLD),
             (1.50, 0.88, 0.24, 0.26, 0.24, 1.0, GOLD, RED),
             (2.60, 0.86, 0.34, 0.16, 0.28, 0.2, BLUE, SILVER),
             (3.70, 0.90, 0.24, 0.24, 0.24, 0.7, SILVER, RED),
             (4.70, 0.86, 0.28, 0.20, 0.30, 0.1, GREEN_BOX, GOLD),
             (5.60, 0.92, 0.22, 0.15, 0.22, 0.5, RED, WHITE)]
    loop = fib_sphere(80)
    face_n = np.array([[0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]], float)
    for ang, dist, w, h, d, rot, body, rib in gifts:
        areas = np.array([w * d, h * d, h * d, h * w, h * w])
        m = int(7000 * areas.sum())
        face = rng.choice(5, m, p=areas / areas.sum())
        fa, fb = rng.random(m) - 0.5, rng.random(m) - 0.5
        lx = np.select([face == 1, face == 2], [w / 2, -w / 2], fa * w)
        lz = np.select([face == 3, face == 4, face == 0], [d / 2, -d / 2, fb * d], fa * d)
        ly = np.where(face == 0, h, (fb + 0.5) * h)
        on_x, on_z = np.abs(lx) < 0.022, np.abs(lz) < 0.022
        ribbon = np.select([face == 0, face <= 2], [on_x | on_z, on_z], on_x)
        col = np.where(ribbon[:, None], rib, body)
        p, nn = np.stack([lx, ly, lz], 1), face_n[face]
        # bow: two little loops on the lid
        bow = np.concatenate([loop * [0.03, 0.022, 0.012] + [sx * 0.03, h + 0.02, 0]
                              for sx in (-1, 1)])
        p = np.concatenate([p, bow])
        nn = np.concatenate([nn, np.concatenate([loop, loop])])
        col = np.concatenate([col, np.tile(rib, (len(bow), 1))])
        c, s_ = math.cos(rot), math.sin(rot)
        yaw = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
        centre = np.array([dist * math.cos(ang), 0, dist * math.sin(ang)])
        sc.add(p @ yaw.T + centre, nn @ yaw.T, col, shine=0.35)

    # --- snowy ground disc, with a warm glow under the tree
    n = 16000
    r = 2.4 * np.sqrt(rng.random(n))
    a = rng.random(n) * 2 * np.pi
    glow = np.exp(-r ** 2 / 0.35)[:, None]
    col = np.array(GROUND) * rng.uniform(0.45, 0.8, (n, 1)) + np.array([0.5, 0.35, 0.1]) * glow * 0.5
    sc.add(np.stack([r * np.cos(a), np.zeros(n), r * np.sin(a)], 1), (0, 1, 0),
           col, kind=GROUND_K)

    # --- star on top (always faces the camera; pos = offset from the tip)
    outer, inner = 0.085, 0.036
    ang = np.pi / 2 + np.arange(10) * np.pi / 5
    rad = np.where(np.arange(10) % 2 == 0, outer, inner)
    verts = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
    k = 1500
    tri = rng.integers(0, 10, k)
    b1, b2 = rng.random(k), rng.random(k)
    flip = b1 + b2 > 1
    b1[flip], b2[flip] = 1 - b1[flip], 1 - b2[flip]
    xy = verts[tri] * b1[:, None] + verts[(tri + 1) % 10] * b2[:, None]
    sc.add(np.stack([xy[:, 0], xy[:, 1], np.zeros(k)], 1), (0, 0, -1),
           np.where((np.hypot(*xy.T) < 0.03)[:, None], (1.0, 1.0, 0.85), GOLD), kind=STAR)

    return sc.freeze(rng)


class Sky:
    """Twinkling background stars and falling snow (screen space)."""

    def __init__(self, seed=3, n_stars=220, n_snow=160):
        rng = np.random.default_rng(seed)
        self.sx, self.sy = rng.random(n_stars), rng.random(n_stars) * 0.62
        self.sph, self.sf = rng.random(n_stars) * 6.28, rng.uniform(0.5, 2.5, n_stars)
        self.fx, self.fy = rng.random(n_snow), rng.random(n_snow)
        self.fv, self.fph = rng.uniform(0.03, 0.10, n_snow), rng.random(n_snow) * 6.28
        self.front = rng.random(n_snow) < 0.25   # some flakes fall in front of the tree

    def step(self, dt, t):
        self.fy = (self.fy + self.fv * dt) % 1.0
        self.fx = (self.fx + 0.015 * np.sin(t * 0.7 + self.fph) * dt) % 1.0


LIGHT_DIR = normalize(np.array([-0.55, 0.65, -0.55]))  # camera space, -z = towards viewer
HALF_VEC = normalize(LIGHT_DIR + np.array([0, 0, -1.0]))


def render(sc, sky, W, H, t, theta, fps=0.0):
    """Return (chars, colours) for one frame: arrays of size W*H."""
    c, s = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(PITCH), math.sin(PITCH)
    R = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]]) @ \
        np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    P = (sc.pos - TARGET) @ R.T
    tip = (np.array([0, Y_TOP + 0.05, 0]) - TARGET) @ R.T
    P[sc.star] = tip + sc.pos[sc.star] - [0, 0, 0.05]   # billboard star
    Z = P[:, 2] + DIST

    # fit the tree to the window; terminal cells are ~2x taller than wide
    fy = min(0.66 * H * DIST / 1.95, 0.92 * W * DIST / 2.2 / 2.0)
    fx = 2.0 * fy
    xs = np.floor(W * 0.5 + P[:, 0] * fx / Z).astype(np.int64)
    ys = np.floor(H * 0.45 - (P[:, 1] + 0.02) * fy / Z).astype(np.int64)
    ok = np.flatnonzero((Z > 0.2) & (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H))

    # z-buffer: nearest point per cell wins
    order = ok[np.argsort(Z[ok], kind="stable")]
    cell = ys[order] * W + xs[order]
    _, first = np.unique(cell, return_index=True)
    sel, cells = order[first], cell[first]

    kind = sc.kind[sel]
    base = sc.col[sel]
    grain = sc.grain[sel]
    N = sc.nrm[sel] @ R.T
    diff = np.clip(N @ LIGHT_DIR, 0, 1)
    spec = np.clip(N @ HALF_VEC, 0, 1) ** 20 * sc.shine[sel]
    lum = 0.38 + 0.62 * diff
    col = base * lum[:, None] + spec[:, None]
    val = lum * 0.85 + spec * 0.6 + (grain - 0.5) * 0.16

    ts = kind == TINSEL                    # tinsel stays bright and glittery
    col[ts] = base[ts] * (0.75 + 0.35 * lum[ts, None]) + spec[ts, None]
    val[ts] = np.maximum(val[ts], 0.6 + 0.3 * grain[ts])

    tw = 0.5 + 0.5 * np.sin(t * sc.freq[sel] + sc.phase[sel])       # fairy lights
    lt = kind == LIGHT
    col[lt] = base[lt] * (0.35 + 0.8 * tw[lt, None])
    val[lt] = 0.55 + 0.45 * tw[lt]

    st = kind == STAR
    pulse = 0.85 + 0.15 * math.sin(t * 2.2)
    col[st] = base[st] * pulse
    val[st] = 0.80 + 0.2 * pulse + (grain[st] - 0.5) * 0.1

    chars = np.full(W * H, " ", dtype="<U1")
    cols = np.zeros((W * H, 3))
    idx = np.clip((val * (len(RAMP) - 1)).round(), 0, len(RAMP) - 1).astype(int)
    chars[cells] = RAMP[idx]
    gd = kind == GROUND_K
    chars[cells[gd]] = np.where(grain[gd] < 0.7, ".", "-")
    col[gd] = base[gd]
    cols[cells] = col

    # background stars and snow only where nothing was drawn
    def put(x01, y01, ch, rgb, front=None):
        cx = np.clip((x01 * W).astype(int), 0, W - 1)
        cy = np.clip((y01 * H).astype(int), 0, H - 1)
        ci = cy * W + cx
        free = chars[ci] == " " if front is None else (chars[ci] == " ") | front
        chars[ci[free]] = ch[free]
        cols[ci[free]] = rgb[free]

    b = 0.5 + 0.5 * np.sin(t * sky.sf + sky.sph)
    put(sky.sx, sky.sy, np.where(b > 0.8, "+", "."), np.outer(0.3 + 0.7 * b, WHITE))
    put(sky.fx, sky.fy, np.where(sky.front, "*", "."),
        np.outer(np.where(sky.front, 0.95, 0.6), WHITE), front=sky.front)

    # HUD
    letters = int((chars != " ").sum())
    hud = ["christmas_tree.py",
           f"rotation {math.degrees(theta) % 360:6.1f} deg   lights {sc.n_lights}",
           f"letters {letters:,}   {fps:4.1f} fps"]
    for i, line in enumerate(hud):
        row = H - len(hud) + i
        if row >= 0:
            line = line[:W]
            chars[row * W: row * W + len(line)] = list(line)
            cols[row * W: row * W + len(line)] = HUD
    clock = f"t {t:6.2f}s"[:W]
    chars[W - len(clock): W] = list(clock)
    cols[W - len(clock): W] = GOLD
    return chars, cols


def to_ansi(chars, cols, W, H):
    q = (np.clip(cols, 0, 1) * 255).astype(np.int32) & ~7     # fewer colour changes
    keys = ((q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]).tolist()
    ch = chars.tolist()
    lines, last = [], -1
    for y in range(H):
        row = []
        for i in range(y * W, y * W + W):
            c = ch[i]
            if c != " ":
                k = keys[i]
                if k != last:
                    row.append(f"\x1b[38;2;{k >> 16};{(k >> 8) & 255};{k & 255}m")
                    last = k
            row.append(c)
        lines.append("".join(row))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Rotating 3D ASCII Christmas tree")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--speed", type=float, default=0.35, help="rotation, rad/s")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = forever)")
    args = ap.parse_args()

    if os.name == "nt":
        os.system("")  # turn on ANSI escape codes in the Windows console
    sc, sky = build_scene(args.seed), Sky()
    out = sys.stdout
    out.write("\x1b[?25l\x1b[2J")
    t0 = prev = time.perf_counter()
    fps, frame = 0.0, 0
    try:
        while True:
            now = time.perf_counter()
            t, dt = now - t0, now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if frame else 1 / dt
            cols_, rows_ = shutil.get_terminal_size((100, 40))
            W, H = max(cols_ - 1, 20), max(rows_ - 1, 10)
            sky.step(dt, t)
            chars, cols = render(sc, sky, W, H, t, t * args.speed, fps)
            out.write("\x1b[H" + to_ansi(chars, cols, W, H) + "\x1b[0m")
            out.flush()
            frame += 1
            if args.frames and frame >= args.frames:
                break
            wait = 1 / args.fps - (time.perf_counter() - now)
            if wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\x1b[0m\x1b[?25h\n")
        out.flush()


if __name__ == "__main__":
    main()
