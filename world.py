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

        for lx in range(CHUNK_SIZE+1):
            for lz in range(CHUNK_SIZE+1):
                wx = base_x + lx
                wz = base_z + lz
                heightmap[lx, lz] = noise.fbm(wx * 0.005 + 10000, wz * 0.005 + 10000)

        for lx in range(CHUNK_SIZE):
            for lz in range(CHUNK_SIZE):
                h  = heightmap[lx, lz]
                height = int((h + 1) * 0.5 * (CHUNK_SIZE * 8))

                for ly in range(CHUNK_SIZE):
                    world_y = base_y + ly

                    if world_y <= height:
                        hx = heightmap[lx+1, lz]
                        hz = heightmap[lx, lz+1]
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
        self.update_chunk_flags(chunk)

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

        self.is_empty_cache = True
        self.is_full_cache = False

    def get(self, x, y, z):
        return self.blocks[x,y,z]
    
    def set(self, x, y, z, value):
        self.blocks[x,y,z] = value

    def is_empty(self):
        return np.all(self.blocks == 0)