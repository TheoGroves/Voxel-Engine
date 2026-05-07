from perlin import PerlinNoise2D
import numpy as np
from numba import njit, uint8, float32, int32, void

CHUNK_SIZE = 16
RENDER_DIST = 4

AIR = 0
DIRT = 1
GRASS = 2
ROCK = 3
COBBLE = 4

@njit(void(uint8[:, :, :], float32[:, :], int32), cache=True, fastmath=True)
def fill_chunk(blocks, heightmap, base_y):
    size = blocks.shape[0]

    for lx in range(size):
        for lz in range(size):

            h = heightmap[lx, lz]
            height = int(h * (size * 32))

            if lx < size - 1:
                hx = heightmap[lx + 1, lz]
            else:
                hx = h

            if lz < size - 1:
                hz = heightmap[lx, lz + 1]
            else:
                hz = h

            dx = hx - h
            dz = hz - h
            steepness = (dx * dx + dz * dz) ** 0.5 * 200.0

            for ly in range(size):
                world_y = base_y + ly

                block = 0

                if world_y == height:
                    # deterministic pseudo-random instead of np.random
                    r = (lx * 928371 + lz * 1237 + base_y * 17) & 255
                    if steepness > 0.5:
                        if r > 180:
                            block = 3
                        else:
                            block = 4
                    elif steepness > 0.25:
                        block = 1
                    else:
                        block = 2

                elif world_y <= height - 4:
                    block = 3

                elif world_y <= height - 1:
                    if steepness > 0.5:
                        block = 3
                    else:
                        block = 1

                else:
                    block = 0

                blocks[lx, ly, lz] = block

class World:
    def __init__(self):
        self.chunks = {}

    def get_chunk(self, cx, cy, cz):
        key = (cx, cy, cz)
        chunk = self.chunks.get(key)
        if chunk is None:
            chunk = Chunk()
            self.chunks[key] = chunk
        return chunk
    
    def world_to_chunk(self, x, y, z):
        return (x // CHUNK_SIZE, y // CHUNK_SIZE, z // CHUNK_SIZE)

    def local_pos(self, x, y, z):
        return (x % CHUNK_SIZE, y % CHUNK_SIZE, z % CHUNK_SIZE)
    
    def set_block(self, x, y, z, value):
        cx, cy, cz = self.world_to_chunk(x, y, z)
        lx, ly, lz = self.local_pos(x, y, z)

        chunk = self.get_chunk(cx, cy, cz)
        chunk.set(lx, ly, lz, value)

    def update_chunk_flags(self, chunk):
        arr = chunk.blocks

        chunk.is_empty_cache = np.all(arr == 0)
        chunk.is_full_cache = np.all(arr != 0)

    def generate_chunk(self, cx, cy, cz, noise: PerlinNoise2D):
        chunk = self.get_chunk(cx, cy, cz)

        base_x = cx * CHUNK_SIZE
        base_y = cy * CHUNK_SIZE
        base_z = cz * CHUNK_SIZE

        heightmap = np.zeros((CHUNK_SIZE+1, CHUNK_SIZE+1))

        xs = np.arange(base_x, base_x + CHUNK_SIZE + 1)
        zs = np.arange(base_z, base_z + CHUNK_SIZE + 1)

        wx, wz = np.meshgrid(xs, zs, indexing='ij')

        heightmap = (((noise.fbm(wx * 0.002 + 10000,
                                wz * 0.002 + 10000,
                                6) + 1) * 0.5) ** 3).reshape(CHUNK_SIZE + 1, CHUNK_SIZE + 1).astype(np.float32)

        fill_chunk(chunk.blocks, heightmap, base_y)

        chunk.generated = True
        self.update_chunk_flags(chunk)

    def get_stream_chunks(self, player_pos, render_dist, y_range=10):
        px, py, pz = player_pos
        pcx, pcy, pcz = self.world_to_chunk(px, py, pz)
        pcx = int(pcx)
        pcy = int(pcy)
        pcz = int(pcz)

        needed = []

        for cx in range(pcx - render_dist, pcx + render_dist + 1):
            for cz in range(pcz - render_dist, pcz + render_dist + 1):
                for cy in range(pcy - y_range, pcy + y_range + 1):
                    dx = cx - pcx
                    dy = cy - pcy
                    dz = cz - pcz

                    dist2 = dx*dx + dy*dy + dz*dz

                    needed.append((dist2, (cx, cy, cz)))

        needed = sorted(needed)
        return [pos for _, pos in needed]

class Chunk:
    def __init__(self):
        self.blocks = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
        self.generated = False

        self.is_empty_cache = True
        self.is_full_cache = False

    def get(self, x, y, z):
        return self.blocks[x,y,z]
    
    def set(self, x, y, z, value):
        self.blocks[x,y,z] = value

    def is_empty(self):
        return np.all(self.blocks == 0)