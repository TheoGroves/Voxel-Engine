import numpy as np
import random

CHUNK_SIZE = 16
RENDER_DIST = 4

AIR = 0
DIRT = 1

class World:
    def __init__(self):
        self.chunks = {}

    def get_chunk(self, cx, cy, cz):
        key = (cx, cy, cz)
        if key not in self.chunks:
            self.chunks[key] = Chunk()
        return self.chunks[key]
    
    def world_to_chunk(self, x, y, z):
        return (x // CHUNK_SIZE, y // CHUNK_SIZE, z // CHUNK_SIZE)

    def local_pos(self, x, y, z):
        return (x % CHUNK_SIZE, y % CHUNK_SIZE, z % CHUNK_SIZE)
    
    def set_block(self, x, y, z, value):
        cx, cy, cz = self.world_to_chunk(x, y, z)
        lx, ly, lz = self.local_pos(x, y, z)

        chunk = self.get_chunk(cx, cy, cz)
        chunk.set(lx, ly, lz, value)

    def generate_chunk(self, cx, cy, cz):
        chunk = self.get_chunk(cx, cy, cz)
        if chunk.generated:
            return
        chunk.generated = True
        for x in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):
                for y in range(random.randint(1, 5)):
                    chunk.set(x, y, z, DIRT)

    def get_stream_chunks(self, player_pos, render_dist):
        px, py, pz = player_pos
        pcx, pcy, pcz = self.world_to_chunk(px, py, pz)
        pcx = int(pcx)
        pcy = int(pcy)
        pcz = int(pcz)

        needed = set()

        for cx in range(pcx - render_dist, pcx + render_dist + 1):
            for cy in range(pcy - render_dist, pcy + render_dist + 1):
                for cz in range(pcz - render_dist, pcz + render_dist + 1):
                    needed.add((cx, cy, cz))

        return needed

class Chunk:
    def __init__(self):
        self.blocks = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
        self.generated = False

    def get(self, x, y, z):
        return self.blocks[x,y,z]
    
    def set(self, x, y, z, value):
        self.blocks[x,y,z] = value