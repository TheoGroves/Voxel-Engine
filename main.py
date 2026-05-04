import pygame
import moderngl
import time
from queue import Queue
import threading
from renderer import Renderer
from camera import Camera
from world import World
from mesher import build_chunk_mesh
from perlin import PerlinNoise2D

RENDER_DIST = 8
streamed_chunks = set()
SUPPRESS_WARNINGS = True

gen_queue = Queue()
mesh_queue = Queue()
MAX_MESH_PER_FRAME = 64

MAX_STREAM_PER_FRAME = 16

mesh_queue = Queue()
upload_queue = Queue()

def gen_worker():
    pn = PerlinNoise2D()
    while True:
        s = time.perf_counter()
        pos_s = time.perf_counter()
        pos = gen_queue.get()
        if pos is None:
            break
        pos_t = time.perf_counter()-pos_s

        get_s = time.perf_counter()
        cx, cy, cz = pos

        world.get_chunk(cx, cy, cz)
        get_t = time.perf_counter()-get_s
        gen_s = time.perf_counter()
        world.generate_chunk(cx, cy, cz, pn)
        gen_t = time.perf_counter()-gen_s

        mesh_queue.put(pos)
        fin_time = time.perf_counter()-s
        if not SUPPRESS_WARNINGS:
            if fin_time*1000 > 5:
                error = "[SLOW GENERATION]"
                if fin_time*1000 > 20:
                    error = "[VERY SLOW GENERATION]"
                print(f"{error} Generated in {(fin_time)*1000:.3f}ms:\n- Position: {pos_t*1000:.3f}ms\n- Chunk Fetch: {get_t*1000:.3f}ms\n- Chunk Gen: {gen_t*1000:.3f}ms\n")

threading.Thread(target=gen_worker, daemon=True).start()

def mesh_worker():
    while True:
        item = mesh_queue.get()
        if item is None:
            break

        pos = item
        v, i = build_chunk_mesh(world, pos)

        if len(i) > 0:
            upload_queue.put((pos, v, i))

threading.Thread(target=mesh_worker, daemon=True).start()

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
        
    cam.process_inputs(pygame.key.get_pressed(), dt)
    cam_pos = cam.position
    needed = world.get_stream_chunks(cam_pos, RENDER_DIST)

    count = 0
    for pos in needed:
        if pos in streamed_chunks:
            continue

        gen_queue.put(pos)
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