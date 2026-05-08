import numpy as np
from numba import njit
import time

CHUNK_SIZE = 16

FACE_UVS = np.array([
    [[0,0],[1,0],[1,1],[0,1]],
    [[0,0],[1,0],[1,1],[0,1]],
    [[0,0],[1,0],[1,1],[0,1]],
    [[0,0],[1,0],[1,1],[0,1]],
    [[0,0],[1,0],[1,1],[0,1]],
    [[0,0],[1,0],[1,1],[0,1]],
], dtype=np.float32)

FACES = np.array([
    [0,0,1],
    [0,0,-1],
    [0,1,0],
    [0,-1,0],
    [1,0,0],
    [-1,0,0],
], dtype=np.int32)

FACE_VERTS = np.array([
    [[0,0,1],[1,0,1],[1,1,1],[0,1,1]],
    [[1,0,0],[0,0,0],[0,1,0],[1,1,0]],
    [[0,1,1],[1,1,1],[1,1,0],[0,1,0]],
    [[0,0,0],[1,0,0],[1,0,1],[0,0,1]],
    [[1,0,1],[1,0,0],[1,1,0],[1,1,1]],
    [[0,0,0],[0,0,1],[0,1,1],[0,1,0]],
], dtype=np.float32)

AO_OFFSETS = np.array([
    [
        [[-1, 0, 0], [0, -1, 0], [-1, -1, 0]],
        [[ 1, 0, 0], [0, -1, 0], [ 1, -1, 0]],
        [[ 1, 0, 0], [0,  1, 0], [ 1,  1, 0]],
        [[-1, 0, 0], [0,  1, 0], [-1,  1, 0]],
    ],

    [
        [[ 1, 0, 0], [0, -1, 0], [ 1, -1, 0]],
        [[-1, 0, 0], [0, -1, 0], [-1, -1, 0]],
        [[-1, 0, 0], [0,  1, 0], [-1,  1, 0]],
        [[ 1, 0, 0], [0,  1, 0], [ 1,  1, 0]],
    ],

    [
        [[-1, 0, 0], [0, 0,  1], [-1, 0,  1]],
        [[ 1, 0, 0], [0, 0,  1], [ 1, 0,  1]],
        [[ 1, 0, 0], [0, 0, -1], [ 1, 0, -1]],
        [[-1, 0, 0], [0, 0, -1], [-1, 0, -1]],
    ],

    [
        [[-1, 0, 0], [0, 0, -1], [-1, 0, -1]],
        [[ 1, 0, 0], [0, 0, -1], [ 1, 0, -1]],
        [[ 1, 0, 0], [0, 0,  1], [ 1, 0,  1]],
        [[-1, 0, 0], [0, 0,  1], [-1, 0,  1]],
    ],

    [
        [[0, 0,  1], [0, -1, 0], [0, -1,  1]],
        [[0, 0, -1], [0, -1, 0], [0, -1, -1]],
        [[0, 0, -1], [0,  1, 0], [0,  1, -1]],
        [[0, 0,  1], [0,  1, 0], [0,  1,  1]],
    ],

    [
        [[0, 0, -1], [0, -1, 0], [0, -1, -1]],
        [[0, 0,  1], [0, -1, 0], [0, -1,  1]],
        [[0, 0,  1], [0,  1, 0], [0,  1,  1]],
        [[0, 0, -1], [0,  1, 0], [0,  1, -1]],
    ],

], dtype=np.int32)

t_sum = 0
t_num = 0

def build_padded(world, chunk_pos):
    cx, cy, cz = chunk_pos
    size = CHUNK_SIZE

    padded = np.zeros((size+2, size+2, size+2), dtype=np.uint8)

    chunk = world.get_chunk(cx, cy, cz)

    for x in range(size):
        for y in range(size):
            for z in range(size):
                padded[x+1, y+1, z+1] = chunk.get(x, y, z)

    directions = [
        (1,0,0), (-1,0,0),
        (0,1,0), (0,-1,0),
        (0,0,1), (0,0,-1),
    ]

    for fx, fy, fz in directions:
        neighbor = world.get_chunk(cx+fx, cy+fy, cz+fz)
        if neighbor is None or not neighbor.terrain_generated:
            continue

        for i in range(size):
            for j in range(size):
                if fx == 1:
                    padded[size+1, i+1, j+1] = neighbor.get(0, i, j)
                elif fx == -1:
                    padded[0, i+1, j+1] = neighbor.get(size-1, i, j)

                elif fy == 1:
                    padded[i+1, size+1, j+1] = neighbor.get(i, 0, j)
                elif fy == -1:
                    padded[i+1, 0, j+1] = neighbor.get(i, size-1, j)

                elif fz == 1:
                    padded[i+1, j+1, size+1] = neighbor.get(i, j, 0)
                elif fz == -1:
                    padded[i+1, j+1, 0] = neighbor.get(i, j, size-1)

    return padded

@njit("Tuple((f4[:, :], u4[:]))(u1[:, :, :], i4[:, :], f4[:, :, :], f4[:, :], f4[:, :, :], i4, i4, i4)", cache=True, fastmath=True)
def mesh_core(padded, faces, face_verts, uv_table, face_uvs, base_x, base_y, base_z):
    size = 16

    max_verts = size*size*size*24
    max_indices = size*size*size*36

    verts = np.empty((max_verts, 9), dtype=np.float32)
    indices = np.empty((max_indices,), dtype=np.uint32)

    v_i = 0
    i_i = 0
    index_offset = 0

    for x in range(size):
        px = x + 1
        wx = base_x + x

        for y in range(size):
            py = y + 1
            wy = base_y + y

            for z in range(size):
                pz = z + 1
                block_type = int(padded[px, py, pz])
                if block_type == 0:
                    continue

                wz = base_z + z

                u0, v0, u1, v1 = uv_table[block_type]

                for f in range(6):
                    fx = int(faces[f, 0])
                    fy = int(faces[f, 1])
                    fz = int(faces[f, 2])

                    if padded[px+fx, py+fy, pz+fz] != 0:
                        continue

                    base = index_offset

                    for k in range(4):
                        cx_ = face_verts[f, k, 0]
                        cy_ = face_verts[f, k, 1]
                        cz_ = face_verts[f, k, 2]

                        fu = face_uvs[f, k, 0]
                        fv = face_uvs[f, k, 1]

                        u = u0 + fu * (u1 - u0)
                        v = v0 + fv * (v1 - v0)

                        s1x, s1y, s1z = AO_OFFSETS[f, k, 0]
                        s2x, s2y, s2z = AO_OFFSETS[f, k, 1]
                        cx2, cy2, cz2 = AO_OFFSETS[f, k, 2]

                        sx = px + fx
                        sy = py + fy
                        sz = pz + fz

                        side1 = padded[sx + s1x, sy + s1y, sz + s1z] != 0
                        side2 = padded[sx + s2x, sy + s2y, sz + s2z] != 0
                        corner = padded[sx + cx2, sy + cy2, sz + cz2] != 0

                        if side1 and side2:
                            ao = 0.0
                        else:
                            ao = float(3 - (side1 + side2 + corner)) / 3.0

                        verts[v_i, 0] = cx_ + wx
                        verts[v_i, 1] = cy_ + wy
                        verts[v_i, 2] = cz_ + wz
                        verts[v_i, 3] = fx
                        verts[v_i, 4] = fy
                        verts[v_i, 5] = fz
                        verts[v_i, 6] = u
                        verts[v_i, 7] = v
                        verts[v_i, 8] = ao

                        v_i += 1

                    indices[i_i+0] = base+0
                    indices[i_i+1] = base+1
                    indices[i_i+2] = base+2
                    indices[i_i+3] = base+2
                    indices[i_i+4] = base+3
                    indices[i_i+5] = base+0

                    i_i += 6
                    index_offset += 4

    return verts[:v_i], indices[:i_i]

def build_chunk_mesh(world, chunk_pos, suppress, suppress_m, uv_table):
    cx, cy, cz = chunk_pos
    size = CHUNK_SIZE

    chunk = world.get_chunk(cx, cy, cz)
    if chunk.is_empty_cache:
        return (
            np.empty((0, 9), dtype=np.float32),
            np.empty((0,), dtype=np.uint32)
        )

    padded = build_padded(world, chunk_pos)

    base_x = cx * size
    base_y = cy * size
    base_z = cz * size

    s = time.perf_counter()
    verts, indices = mesh_core(
        padded,
        FACES,
        FACE_VERTS,
        uv_table,
        FACE_UVS,
        base_x, base_y, base_z
    )
    chunk.dirty = False
    if not suppress and not suppress_m:
        print(f"[DEBUG] - Meshing: {(time.perf_counter()-s)*1000:.2f}ms/chunk")

    return verts, indices