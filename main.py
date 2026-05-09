import pygame
import moderngl
import time
import heapq
from queue import Queue
import threading
from renderer import Renderer
from camera import Camera
from world import World
from mesher import build_chunk_mesh
from perlin import PerlinNoise2D
from texture_handler import TextureHandler
from collections import defaultdict
import time

prof_data = defaultdict(float)
prof_count = defaultdict(int)

class Prof:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.start
        prof_data[self.name] += dt
        prof_count[self.name] += 1

SUPPRESS_WARNINGS = True
SUPPRESS_GEN = True
SUPPRESS_MESHING = True
SUPPRESS_TRIS = True
SUPPRESS_PROFILER = True

RENDER_DIST = 8
streamed_chunks = set()
needed_now = set()
needed_snapshot = set()

gen_queue = []
gen_lock = threading.Lock()
mesh_queue = []
mesh_lock = threading.Lock()
MAX_MESH_PER_FRAME = 64

MAX_STREAM_PER_FRAME = 16

upload_queue = Queue()

last_generated_time = time.perf_counter()
last_meshed_time = time.perf_counter()

generated_chunks = set()
meshed_chunks = set()

cam_pos = (0,0,0)
cam_last_pos = (0,0,0)

STRUCTURE_RADIUS = 1
structure_queue = []
structure_lock = threading.Lock()

th = TextureHandler(1024, 1024)
th.pack(["textures/Empty.png", "textures/Dirt.png", "textures/Grass.png", "textures/Rock.png", "textures/Cobblestone.png", "textures/Wood.png", "textures/Leaves.png"])
th.save_atlas("textures/atlas.png")
uv_table = th.build_uv_table()

def gen_worker():
    global last_generated_time
    pn = PerlinNoise2D()
    while True:
        with Prof("gen_total"):
            s = time.perf_counter()
            pos_s = time.perf_counter()
            with gen_lock:
                if len(gen_queue) == 0:
                    time.sleep(0.1)
                    continue
                _, pos = heapq.heappop(gen_queue)

            if pos not in needed_snapshot:
                continue
            if pos is None:
                break

            last_generated_time = time.perf_counter()

            pos_t = time.perf_counter()-pos_s

            get_s = time.perf_counter()

            with Prof("gen_fetch"):
                cx, cy, cz = pos
                chunk = world.get_chunk(cx, cy, cz)
            
            if chunk.terrain_generated or chunk.structures_generated:
                continue
            get_t = time.perf_counter()-get_s
            gen_s = time.perf_counter()
            with Prof("gen_noise"):
                world.generate_chunk(cx, cy, cz, pn)
                chunk.terrain_generated = True
                generated_chunks.add(pos)

                for dx in range(-1, 2):
                    for dz in range(-1, 2):
                        check_pos = (cx + dx, cy, cz + dz)

                        if should_generate_structures(world, check_pos):
                            chunk2 = world.get_chunk(*check_pos)

                            if not chunk2.structures_generated:
                                cx2, cy2, cz2 = cam_last_pos
                                px, py, pz = check_pos

                                priority2 = (px - cx2)**2 + (py - cy2)**2 + (pz - cz2)**2

                                with structure_lock:
                                    heapq.heappush(structure_queue, (priority2, check_pos))

                cx, cy, cz = pos
                cx2, cy2, cz2 = cam_last_pos

                priority = (cx - cx2)**2 + (cy - cy2)**2 + (cz - cz2)**2

                if should_generate_structures(world, pos):
                    with structure_lock:
                        heapq.heappush(structure_queue, (priority, pos))
                gen_t = time.perf_counter()-gen_s

                fin_time = time.perf_counter()-s
                if not SUPPRESS_WARNINGS and not SUPPRESS_GEN:
                    if fin_time*1000 > 5:
                        error = "SLOW GENERATION"
                        if fin_time*1000 > 20:
                            error = "VERY SLOW GENERATION"
                        print(f"[WARNING] - {error}: Chunk generated in {(fin_time)*1000:.3f}ms:\n- Position: {pos_t*1000:.3f}ms\n- Chunk Fetch: {get_t*1000:.3f}ms\n- Chunk Generation: {gen_t*1000:.3f}ms\n")

threading.Thread(target=gen_worker, daemon=True).start()

def should_generate_structures(world, chunk_pos):
    cx, cy, cz = chunk_pos

    for dx in range(-1, 2):
        for dz in range(-1, 2):
            n = world.get_chunk(cx + dx, cy, cz + dz)

            if n is None or not n.terrain_generated:
                return False

    return True

def structure_worker():
    while True:
        with Prof("struct_total"):
            with structure_lock:
                if len(structure_queue) == 0:
                    time.sleep(0.1)
                    continue

                priority, pos = heapq.heappop(structure_queue)

            if not should_generate_structures(world, pos):
                with structure_lock:
                    heapq.heappush(structure_queue, (priority + 1, pos))
                continue

            chunk = world.get_chunk(*pos)

            if chunk.structures_generated:
                continue

            world.generate_structures(*pos)

            chunk.structures_generated = True

            directions = [
                (0, 0, 0),
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
            ]

            cx, cy, cz = pos
            cx2, cy2, cz2 = cam_last_pos

            for dx, dy, dz in directions:
                check_pos = (cx + dx, cy + dy, cz + dz)

                if should_mesh(world, check_pos):
                    px, py, pz = check_pos
                    priority2 = (px - cx2)**2 + (py - cy2)**2 + (pz - cz2)**2

                    with mesh_lock:
                        heapq.heappush(mesh_queue, (priority2, check_pos))

threading.Thread(target=structure_worker, daemon=True).start()

def should_mesh(world, chunk_pos):
    directions = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1)
    ]

    cx, cy, cz = chunk_pos
    for dx, dy, dz in directions:
        n = world.get_chunk(cx+dx, cy+dy, cz+dz)
        if n is None or not n.structures_generated:
            return False
    return True

def mesh_worker():
    global last_meshed_time
    while True:
        with Prof("mesh_total"):
            with mesh_lock:
                if len(mesh_queue) == 0:
                    time.sleep(0.1)
                    continue
                priority, pos = heapq.heappop(mesh_queue)

            if pos not in needed_snapshot:
                continue

            if not should_mesh(world, pos):
                continue

            v, i = build_chunk_mesh(world, pos, SUPPRESS_WARNINGS, SUPPRESS_MESHING, uv_table)
            meshed_chunks.add(pos)
            last_meshed_time = time.perf_counter()

            if len(i) > 0:
                upload_queue.put((pos, v, i))

threading.Thread(target=mesh_worker, daemon=True).start()

def monitor():
    while True:
        if time.perf_counter() - last_generated_time > 1.0 and not SUPPRESS_WARNINGS:
            print(f"[WARNING] No chunks have been generated in the last second, last generated {(time.perf_counter() - last_generated_time)*1000:.1f}ms ago")
        if time.perf_counter() - last_meshed_time > 1.0 and not SUPPRESS_WARNINGS:
            print(f"[WARNING] No chunks have been meshed in the last second, last meshed {(time.perf_counter() - last_meshed_time)*1000:.1f}ms ago")
        if not SUPPRESS_WARNINGS:
            print(
                "[DEBUG] "
                f"Streamed: {len(streamed_chunks)}/{len(needed_now)} | "
                f"Generated: {len(generated_chunks)}/{len(needed_now)} | "
                f"Meshed: {len(meshed_chunks)}/{len(needed_now)}"
            )

        time.sleep(0.5)

if not SUPPRESS_WARNINGS:
    threading.Thread(target=monitor, daemon=True).start()

def print_prof():
    while True:
        time.sleep(1)

        if not SUPPRESS_PROFILER:
            print("\n--- PROFILER ---")
            for k in sorted(prof_data.keys()):
                total = prof_data[k]
                count = prof_count[k]
                avg = (total / count * 1000) if count else 0
                print(f"{k:15} | total: {total:.3f}s | avg: {avg:.3f}ms | calls: {count}")

            prof_data.clear()
            prof_count.clear()

if not SUPPRESS_PROFILER:
    threading.Thread(target=print_prof, daemon=True).start()

pygame.init()
screen_width, screen_height = 1280, 720
screen = pygame.display.set_mode((screen_width, screen_height), pygame.OPENGL | pygame.DOUBLEBUF)

ctx = moderngl.create_context()

renderer = Renderer(ctx, screen_width, screen_height)
renderer.load_atlas(th.atlas)

cam = Camera((0, 50, 0))

world = World()

clock = pygame.time.Clock()

dt = 0

while True:
    moving = False
    with Prof("input"):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
    cam_old_pos = cam.position
    keys = pygame.key.get_pressed()
    with Prof("cam"):
        cam.process_inputs(keys, dt)

    with Prof("block_placing"):
        if pygame.mouse.get_pressed()[0]:
            bx, by, bz = int(cam_pos[0]), int(cam_pos[1]), int(cam_pos[2])
            world.set_block(bx, by, bz, 2)

            cx = bx // 16
            cy = by // 16
            cz = bz // 16

            to_remesh = {(cx, cy, cz)}

            if bx % 16 == 0:
                to_remesh.add((cx - 1, cy, cz))
            if bx % 16 == 15:
                to_remesh.add((cx + 1, cy, cz))

            if by % 16 == 0:
                to_remesh.add((cx, cy - 1, cz))
            if by % 16 == 15:
                to_remesh.add((cx, cy + 1, cz))

            if bz % 16 == 0:
                to_remesh.add((cx, cy, cz - 1))
            if bz % 16 == 15:
                to_remesh.add((cx, cy, cz + 1))

            with mesh_lock:
                for pos in to_remesh:
                    heapq.heappush(mesh_queue, (0, pos))

    cam_pos = cam.position
    cam_last_pos = cam_old_pos
    with Prof("streaming_total"):

        with Prof("streaming_get_chunks"):
            needed = world.get_stream_chunks(cam_pos, RENDER_DIST)

        with Prof("streaming_set_build"):
            needed_now = set(needed)
            needed_snapshot = needed_now.copy()

        with Prof("streaming_remove_old"):
            for pos in list(streamed_chunks):
                if pos not in needed_snapshot:
                    streamed_chunks.remove(pos)
                    renderer.remove_chunk(pos)

        count = 0

        with Prof("streaming_iterate_chunks"):
            for pos in needed_now:

                with Prof("streaming_skip_check"):
                    if pos in streamed_chunks:
                        continue

                    if pos in generated_chunks:
                        continue

                    if pos in meshed_chunks:
                        continue

                with Prof("streaming_priority_calc"):
                    cx, cy, cz = pos
                    cx2, cy2, cz2 = cam_pos
                    priority = (cx - cx2)**2 + (cy - cy2)**2 + (cz - cz2)**2

                with Prof("streaming_get_chunk"):
                    c = world.get_chunk(cx, cy, cz)

                with Prof("streaming_queue_push"):
                    if not c.terrain_generated:
                        with gen_lock:
                            heapq.heappush(gen_queue, (priority, pos))
                    else:
                        with mesh_lock:
                            heapq.heappush(mesh_queue, (priority, pos))

                with Prof("streaming_mark"):
                    streamed_chunks.add(pos)
                    c.dirty = True

                count += 1
                if count >= MAX_STREAM_PER_FRAME:
                    break

    with Prof("upload"):
        for _ in range(min(MAX_MESH_PER_FRAME, upload_queue.qsize())):
            pos, v, i = upload_queue.get()
            renderer.upload_chunk_mesh(pos, v, i)

    with Prof("rendering"):
        tot_tris = renderer.render(cam)

    if not SUPPRESS_WARNINGS and not SUPPRESS_TRIS:
        print(f"[DEBUG] {tot_tris} triangles are being rendered")
    pygame.display.set_caption(f"Voxel Engine | FPS: {clock.get_fps():.1f} | x: {cam_pos[0]:.1f} y: {cam_pos[1]:.1f} z: {cam_pos[2]:.1f} | Tris: {tot_tris}")
    pygame.display.flip()
    dt = clock.tick(60) / 1000.0