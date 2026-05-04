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

SUPPRESS_WARNINGS = True

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

cam_pos = (0,0,0)
cam_last_pos = (0,0,0)

def gen_worker():
    global last_generated_time
    pn = PerlinNoise2D()
    while True:
        s = time.perf_counter()
        pos_s = time.perf_counter()
        with gen_lock:
            if len(gen_queue) == 0:
                continue
            _, pos = heapq.heappop(gen_queue)

        if pos not in needed_snapshot:
            continue
        if pos is None:
            break

        last_generated_time = time.perf_counter()

        pos_t = time.perf_counter()-pos_s

        get_s = time.perf_counter()
        cx, cy, cz = pos

        world.get_chunk(cx, cy, cz)
        get_t = time.perf_counter()-get_s
        gen_s = time.perf_counter()
        world.generate_chunk(cx, cy, cz, pn)
        gen_t = time.perf_counter()-gen_s

        cx, cy, cz = pos
        cx2, cy2, cz2 = cam_last_pos

        priority = (cx - cx2)**2 + (cy - cy2)**2 + (cz - cz2)**2

        with mesh_lock:
            heapq.heappush(mesh_queue, (priority, pos))
        fin_time = time.perf_counter()-s
        if not SUPPRESS_WARNINGS:
            if fin_time*1000 > 5:
                error = "SLOW GENERATION"
                if fin_time*1000 > 20:
                    error = "VERY SLOW GENERATION"
                print(f"[WARNING] - {error}: Chunk generated in {(fin_time)*1000:.3f}ms:\n- Position: {pos_t*1000:.3f}ms\n- Chunk Fetch: {get_t*1000:.3f}ms\n- Chunk Generation: {gen_t*1000:.3f}ms\n")

threading.Thread(target=gen_worker, daemon=True).start()

def mesh_worker():
    global last_meshed_time
    while True:
        with mesh_lock:
            if len(mesh_queue) == 0:
                continue
            priority, pos = heapq.heappop(mesh_queue)

        if pos not in needed_snapshot:
            continue

        v, i = build_chunk_mesh(world, pos)
        last_meshed_time = time.perf_counter()

        if len(i) > 0:
            upload_queue.put((pos, v, i))

threading.Thread(target=mesh_worker, daemon=True).start()

def monitor():
    while True:
        if time.perf_counter() - last_generated_time > 1.0 and not SUPPRESS_WARNINGS:
            print("[WARNING] No chunks are being generated")
        if time.perf_counter() - last_meshed_time > 1.0 and not SUPPRESS_WARNINGS:
            print("[WARNING] No chunks are being meshed")

        time.sleep(0.5)

if not SUPPRESS_WARNINGS:
    threading.Thread(target=monitor, daemon=True).start()

pygame.init()
screen_width, screen_height = 1280, 720
screen = pygame.display.set_mode((screen_width, screen_height), pygame.OPENGL | pygame.DOUBLEBUF)

ctx = moderngl.create_context()

renderer = Renderer(ctx, screen_width, screen_height)

cam = Camera()

world = World()

clock = pygame.time.Clock()

dt = 0

while True:
    moving = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
    
    cam_old_pos = cam.position
    cam.process_inputs(pygame.key.get_pressed(), dt)
    cam_pos = cam.position
    cam_last_pos = cam_old_pos
    needed = world.get_stream_chunks(cam_pos, RENDER_DIST)
    needed_now = set(needed)
    needed_snapshot = needed_now.copy()

    for pos in list(streamed_chunks):
        if pos not in needed_snapshot:
            streamed_chunks.remove(pos)
            renderer.remove_chunk(pos)

    count = 0

    for pos in needed_now:
        if pos in streamed_chunks:
            continue

        cx, cy, cz = pos
        cx2, cy2, cz2 = cam_pos

        priority = (cx - cx2)**2 + (cy - cy2)**2 + (cz - cz2)**2

        with gen_lock:
            heapq.heappush(gen_queue, (priority, pos))

        streamed_chunks.add(pos)

        count += 1
        if count >= MAX_STREAM_PER_FRAME:
            break

    for _ in range(min(MAX_MESH_PER_FRAME, upload_queue.qsize())):
        pos, v, i = upload_queue.get()
        renderer.upload_chunk_mesh(pos, v, i)

    renderer.render(cam)
    pygame.display.set_caption(f"Voxel Engine | FPS: {clock.get_fps():.1f}")
    pygame.display.flip()
    dt = clock.tick(60) / 1000.0