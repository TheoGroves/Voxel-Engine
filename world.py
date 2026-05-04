from perlin import PerlinNoise2D
import numpy as np

CHUNK_SIZE = 16
RENDER_DIST = 4

AIR = 0
DIRT = 1
GRASS = 2
ROCK = 3

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

    def generate_chunk(self, cx, cy, cz, noise: PerlinNoise2D):
        chunk = self.get_chunk(cx, cy, cz)

        base_x = cx * CHUNK_SIZE
        base_y = cy * CHUNK_SIZE
        base_z = cz * CHUNK_SIZE

        for lx in range(CHUNK_SIZE):
            for lz in range(CHUNK_SIZE):

                world_x = base_x + lx
                world_z = base_z + lz

                n = noise.fbm((world_x * 0.005) + 10000, (world_z * 0.005) + 10000)

                height = int((n + 1) * 0.5 * (CHUNK_SIZE * 8))

                for ly in range(CHUNK_SIZE):
                    world_y = base_y + ly

                    if world_y <= height:
                        eps = 1.0
                        h  = noise.fbm(world_x * 0.005 + 10000, world_z * 0.005 + 10000)
                        hx = noise.fbm((world_x + eps) * 0.005 + 10000, world_z * 0.005 + 10000)
                        hz = noise.fbm(world_x * 0.005 + 10000, (world_z + eps) * 0.005 + 10000)
                        dx = hx - h
                        dz = hz - h
                        steepness = (dx*dx + dz*dz) ** 0.5
                        steepness *= 100                        
                        block = GRASS
                        if steepness > 0.6:
                            block = DIRT
                        if steepness > 1:
                            block = ROCK

                        chunk.blocks[lx, ly, lz] = block
                    else:
                        chunk.blocks[lx, ly, lz] = AIR

        chunk.generated = True

    def get_stream_chunks(self, player_pos, render_dist, y_range=3):
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

    def get(self, x, y, z):
        return self.blocks[x,y,z]
    
    def set(self, x, y, z, value):
        self.blocks[x,y,z] = value