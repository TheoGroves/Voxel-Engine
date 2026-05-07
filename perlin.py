import numpy as np
import random
import time
from numba import njit, float64, int32


@njit(float64[:](int32[:], float64[:], float64[:]), cache=True)
def grad_np(h, x, y):
    out = np.empty(x.shape[0], dtype=np.float64)

    for i in range(x.shape[0]):
        hh = h[i] & 3

        if hh == 0:
            out[i] = x[i] + y[i]
        elif hh == 1:
            out[i] = -x[i] + y[i]
        elif hh == 2:
            out[i] = x[i] - y[i]
        else:
            out[i] = -x[i] - y[i]

    return out


@njit(float64[:](float64[:]), cache=True)
def fade_np(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(float64[:](float64[:], float64[:], float64[:]), cache=True)
def lerp_np(a, b, t):
    return a + t * (b - a)


@njit(float64[:](int32[:], float64[:], float64[:]), cache=True)
def _noise_1d(p, x, y):
    n = x.shape[0]

    xi = np.empty(n, dtype=np.int32)
    yi = np.empty(n, dtype=np.int32)

    xf = np.empty(n, dtype=np.float64)
    yf = np.empty(n, dtype=np.float64)

    for i in range(n):
        x_int = int(x[i])
        y_int = int(y[i])

        xi[i] = x_int & 255
        yi[i] = y_int & 255

        xf[i] = x[i] - x_int
        yf[i] = y[i] - y_int

    u = fade_np(xf)
    v = fade_np(yf)

    aa = np.empty(n, dtype=np.int32)
    ab = np.empty(n, dtype=np.int32)
    ba = np.empty(n, dtype=np.int32)
    bb = np.empty(n, dtype=np.int32)

    for i in range(n):
        aa[i] = p[p[xi[i]] + yi[i]]
        ab[i] = p[p[xi[i]] + yi[i] + 1]
        ba[i] = p[p[xi[i] + 1] + yi[i]]
        bb[i] = p[p[xi[i] + 1] + yi[i] + 1]

    x1 = lerp_np(
        grad_np(aa, xf, yf),
        grad_np(ba, xf - 1.0, yf),
        u
    )

    x2 = lerp_np(
        grad_np(ab, xf, yf - 1.0),
        grad_np(bb, xf - 1.0, yf - 1.0),
        u
    )

    return lerp_np(x1, x2, v)


@njit(float64[:](int32[:], float64[:], float64[:], int32, float64, float64), cache=True)
def _fbm_1d(p, x, y, octaves, lacunarity, gain):
    n = x.shape[0]
    total = np.zeros(n, dtype=np.float64)

    frequency = 1.0
    amplitude = 1.0
    max_value = 0.0

    for _ in range(octaves):
        total += _noise_1d(p, x * frequency, y * frequency) * amplitude
        max_value += amplitude
        frequency *= lacunarity
        amplitude *= gain

    return total / max_value


class PerlinNoise2D:
    def __init__(self, seed=time.time()):
        random.seed(seed)

        self.p = np.arange(256, dtype=np.int32)
        np.random.shuffle(self.p)

        self.p = np.concatenate([self.p, self.p]).astype(np.int32)

    def noise_np(self, x, y):
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        return _noise_1d(self.p, x, y)

    def fbm(self, x, y, octaves=4, lacunarity=2.0, gain=0.5):
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        return _fbm_1d(self.p, x, y, octaves, lacunarity, gain)