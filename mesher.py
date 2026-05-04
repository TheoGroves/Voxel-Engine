import numpy as np

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

                if chunk.get(x, y, z) == 0:
                    continue

                wx = cx * size + x
                wy = cy * size + y
                wz = cz * size + z

                for fx, fy, fz in FACES:
                    nx, ny, nz = wx + fx, wy + fy, wz + fz

                    if is_solid(world, nx, ny, nz):
                        continue

                    corners = FACE_VERTS[(fx, fy, fz)]

                    base = [
                        np.array(corners[0]) + [wx, wy, wz],
                        np.array(corners[1]) + [wx, wy, wz],
                        np.array(corners[2]) + [wx, wy, wz],
                        np.array(corners[3]) + [wx, wy, wz],
                    ]

                    verts.extend(base)

                    indices.extend([
                        index_offset, index_offset + 1, index_offset + 2,
                        index_offset + 2, index_offset + 3, index_offset
                    ])

                    index_offset += 4

    return (
        np.array(verts, dtype=np.float32),
        np.array(indices, dtype=np.uint32)
    )