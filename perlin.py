import math
import random
import time

import numpy as np

def grad_np(h, x, y):
    h = h & 3

    return np.where(
        h == 0, x + y,
        np.where(
            h == 1, -x + y,
            np.where(
                h == 2, x - y,
                -x - y
            )
        )
    )

def fade_np(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp_np(a, b, t):
    return a + t * (b - a)
    
class PerlinNoise2D:
    def __init__(self, seed=time.time()):
        self.p = list(range(256))
        random.shuffle(self.p)
        self.p += self.p

    def noise_np(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        xi = (x.astype(int)) & 255
        yi = (y.astype(int)) & 255

        xf = x - x.astype(int)
        yf = y - y.astype(int)

        u = fade_np(xf)
        v = fade_np(yf)

        p = np.array(self.p)

        aa = p[p[xi] + yi]
        ab = p[p[xi] + yi + 1]
        ba = p[p[xi + 1] + yi]
        bb = p[p[xi + 1] + yi + 1]

        x1 = lerp_np(
            grad_np(aa, xf, yf),
            grad_np(ba, xf - 1, yf),
            u
        )

        x2 = lerp_np(
            grad_np(ab, xf, yf - 1),
            grad_np(bb, xf - 1, yf - 1),
            u
        )

        return lerp_np(x1, x2, v)
    
    def fbm(self, x, y, octaves=4, lacunarity=2.0, gain=0.5):
        total = np.zeros_like(x, dtype=np.float32)
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0

        for _ in range(octaves):
            total += self.noise_np(x * frequency, y * frequency) * amplitude
            max_value += amplitude

            frequency *= lacunarity
            amplitude *= gain

        return total / max_value