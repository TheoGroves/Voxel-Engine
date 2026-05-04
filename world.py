from perlin import PerlinNoise2D
import numpy as np

CHUNK_SIZE = 16
RENDER_DIST = 4

AIR = 0
DIRT = 1

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
        if chunk.generated:
            return
        chunk.generated = True

        base_x = cx * CHUNK_SIZE
        base_y = cy * CHUNK_SIZE
        base_z = cz * CHUNK_SIZE

        heights = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.int32)

        for x in range(CHUNK_SIZE):
            wx = base_x + x
            for z in range(CHUNK_SIZE):
                wz = base_z + z

                h = noise.noise(wx * 0.005, wz * 0.005)
                heights[x, z] = int((h + 1) * 0.5 * 7)

        for x in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):
                height = heights[x, z]

                local_top = height - base_y

                if local_top < 0:
                    continue

                if local_top >= CHUNK_SIZE:
                    local_top = CHUNK_SIZE - 1

                chunk.blocks[x, :local_top + 1, z] = DIRT

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