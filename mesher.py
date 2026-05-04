import numpy as np

BLOCK_COLORS = {
    1: (0.49, 0.416, 0.369),
    2: (0.561, 0.71, 0.518),
    3: (0.5, 0.5, 0.5),
}

FACES = [
    (0, 0, 1),
    (0, 0, -1),
    (0, 1, 0),
    (0, -1, 0),
    (1, 0, 0),
    (-1, 0, 0),
]

FACE_VERTS = {
    (0, 0, 1):  [(0,0,1),(1,0,1),(1,1,1),(0,1,1)],
    (0, 0,-1):  [(1,0,0),(0,0,0),(0,1,0),(1,1,0)],
    (0, 1, 0):  [(0,1,1),(1,1,1),(1,1,0),(0,1,0)],
    (0,-1, 0):  [(0,0,0),(1,0,0),(1,0,1),(0,0,1)],
    (1, 0, 0):  [(1,0,1),(1,0,0),(1,1,0),(1,1,1)],
    (-1,0, 0):  [(0,0,0),(0,0,1),(0,1,1),(0,1,0)],
}

CHUNK_SIZE = 16

def is_solid(world, wx, wy, wz):
    cx, cy, cz = world.world_to_chunk(wx, wy, wz)
    lx, ly, lz = world.local_pos(wx, wy, wz)

    chunk = world.get_chunk(cx, cy, cz)
    return chunk.get(lx, ly, lz) != 0


def build_chunk_mesh(world, chunk_pos):
    cx, cy, cz = chunk_pos
    chunk = world.get_chunk(cx, cy, cz)

    verts = []
    indices = []
    index_offset = 0

    size = CHUNK_SIZE

    for x in range(size):
        for y in range(size):
            for z in range(size):

                block_type = chunk.get(x, y, z)
                if block_type == 0:
                    continue

                wx = cx * size + x
                wy = cy * size + y
                wz = cz * size + z

                block_color = BLOCK_COLORS.get(block_type, (1.0, 0.0, 1.0))

                for fx, fy, fz in FACES:
                    nx, ny, nz = wx + fx, wy + fy, wz + fz

                    if is_solid(world, nx, ny, nz):
                        continue

                    corners = FACE_VERTS[(fx, fy, fz)]
                    normal = np.array([fx, fy, fz], dtype=np.float32)

                    for corner in corners:
                        pos = np.array(corner, dtype=np.float32) + np.array([wx, wy, wz], dtype=np.float32)
                        verts.append((
                            pos[0], pos[1], pos[2],
                            normal[0], normal[1], normal[2],
                            block_color[0], block_color[1], block_color[2]
                        ))

                    indices.extend([
                        index_offset, index_offset + 1, index_offset + 2,
                        index_offset + 2, index_offset + 3, index_offset
                    ])

                    index_offset += 4

    return (
        np.array(verts, dtype=np.float32),
        np.array(indices, dtype=np.uint32)
    )