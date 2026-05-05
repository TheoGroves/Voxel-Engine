import numpy as np
import time

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

t_sum = 0
t_num = 0

def build_chunk_mesh(world, chunk_pos):
    global t_sum, t_num
    start = time.perf_counter()
    cx, cy, cz = chunk_pos
    chunk = world.get_chunk(cx, cy, cz)

    if chunk.is_empty_cache:
        return (
            np.empty((0, 9), dtype=np.float32),
            np.empty((0,), dtype=np.uint32)
        )

    size = CHUNK_SIZE

    chunk_get = chunk.get
    world_get_chunk = world.get_chunk
    world_to_chunk = world.world_to_chunk
    local_pos = world.local_pos

    faces = FACES
    face_verts = FACE_VERTS
    colors = BLOCK_COLORS

    chunk_world_x = cx * size
    chunk_world_y = cy * size
    chunk_world_z = cz * size

    MAX_VERTS = size * size * size * 24
    MAX_INDICES = size * size * size * 36

    verts = np.empty((MAX_VERTS, 9), dtype=np.float32)
    indices = np.empty((MAX_INDICES,), dtype=np.uint32)

    v_i = 0
    i_i = 0
    index_offset = 0

    chunk_cache = {(cx, cy, cz): chunk}

    def get_chunk_cached(ccx, ccy, ccz):
        key = (ccx, ccy, ccz)
        if key not in chunk_cache:
            chunk_cache[key] = world_get_chunk(ccx, ccy, ccz)
        return chunk_cache[key]

    for x in range(size):
        wx = chunk_world_x + x

        for y in range(size):
            wy = chunk_world_y + y

            for z in range(size):
                block_type = chunk_get(x, y, z)
                if block_type == 0:
                    continue

                wz = chunk_world_z + z

                color = colors[block_type]
                r, g, b = color

                for fx, fy, fz in faces:

                    nlx = x + fx
                    nly = y + fy
                    nlz = z + fz

                    if 0 <= nlx < size and 0 <= nly < size and 0 <= nlz < size:
                        if chunk_get(nlx, nly, nlz) != 0:
                            continue
                    else:
                        nx = wx + fx
                        ny = wy + fy
                        nz = wz + fz

                        ncx, ncy, ncz = world_to_chunk(nx, ny, nz)
                        neighbor_chunk = get_chunk_cached(ncx, ncy, ncz)
                        nlx, nly, nlz = local_pos(nx, ny, nz)

                        if neighbor_chunk.get(nlx, nly, nlz) != 0:
                            continue

                    corners = face_verts[(fx, fy, fz)]
                    nxn, nyn, nzn = fx, fy, fz

                    base = v_i

                    for cx_, cy_, cz_ in corners:
                        verts[v_i] = (
                            cx_ + wx,
                            cy_ + wy,
                            cz_ + wz,
                            nxn, nyn, nzn,
                            r, g, b
                        )
                        v_i += 1

                    indices[i_i + 0] = base + 0
                    indices[i_i + 1] = base + 1
                    indices[i_i + 2] = base + 2
                    indices[i_i + 3] = base + 2
                    indices[i_i + 4] = base + 3
                    indices[i_i + 5] = base + 0

                    i_i += 6
                    index_offset += 4

    t_sum += (time.perf_counter()-start)*1000
    t_num += 1
    print(f"[DEBUG] - Meshing: {t_sum/t_num:.1f}ms/chunk. Estimating {((t_sum/t_num) / 1000) * 2023 / 60:.1f} minutes to complete.")
    return (
        verts[:v_i].copy(),
        indices[:i_i].copy()
    )