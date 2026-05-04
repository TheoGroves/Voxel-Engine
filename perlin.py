import math
import random
import time

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(a, b, t):
    return a + t * (b-a)

def grad(hash, x, y):
    h = hash & 3
    if h == 0:
        return x + y
    elif h == 1:
        return -x + y
    elif h == 2:
        return x - y
    else:
        return -x - y
    
class PerlinNoise2D:
    def __init__(self, seed=time.time()):
        self.p = list(range(256))
        random.shuffle(self.p)
        self.p += self.p

    def noise(self, x, y):
        xi = int(x) & 255
        yi = int(y) & 255

        xf = x - int(x)
        yf = y - int(y)

        u = fade(xf)
        v = fade(yf)

        aa = self.p[self.p[xi] + yi]
        ab = self.p[self.p[xi] + yi + 1]
        ba = self.p[self.p[xi + 1] + yi]
        bb = self.p[self.p[xi + 1] + yi + 1]

        x1 = lerp(grad(aa, xf, yf),
                  grad(ba, xf - 1, yf), u)

        x2 = lerp(grad(ab, xf, yf - 1),
                  grad(bb, xf - 1, yf - 1), u)
        
        return lerp(x1, x2, v)