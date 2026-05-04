import pygame
import moderngl
from queue import Queue
import threading
from renderer import Renderer
from camera import Camera
from world import World
from mesher import build_chunk_mesh

RENDER_DIST = 4
streamed_chunks = set()

gen_queue = Queue()
mesh_queue = Queue()
MAX_MESH_PER_FRAME = 1

def gen_worker():
    while True:
        pos = gen_queue.get()
        if pos is None:
            break

        cx, cy, cz = pos

        world.get_chunk(cx, cy, cz)
        world.generate_chunk(cx, cy, cz)

        mesh_queue.put(pos)

threading.Thread(target=gen_worker, daemon=True).start()

pygame.init()
screen_width, screen_height = 1280, 720
screen = pygame.display.set_mode((screen_width, screen_height), pygame.OPENGL | pygame.DOUBLEBUF)

ctx = moderngl.create_context()

renderer = Renderer(ctx, screen_width, screen_height)

cam = Camera()

world = World()
world.generate_chunk(0, 0, 0)

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

    for pos in needed - streamed_chunks:
        if pos not in streamed_chunks:
            gen_queue.put(pos)
            streamed_chunks.add(pos)


    for _ in range(min(MAX_MESH_PER_FRAME, mesh_queue.qsize())):
        pos = mesh_queue.get()

        v, i = build_chunk_mesh(world, pos)

        if len(i) == 0:
            continue

        renderer.upload_chunk_mesh(pos, v, i)

    for pos in list(streamed_chunks - needed):
        if pos in renderer.chunks:
            vao, _ = renderer.chunks[pos]
            vao.release()
            del renderer.chunks[pos]

        streamed_chunks.remove(pos)

    renderer.render(cam)
    pygame.display.set_caption(f"Voxel Engine | FPS: {clock.get_fps():.1f}")
    pygame.display.flip()
    dt = clock.tick(60) / 1000.0