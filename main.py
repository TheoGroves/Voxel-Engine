import pygame
import moderngl
from renderer import Renderer
from camera import Camera
from world import World
from mesher import build_chunk_mesh

pygame.init()
screen_width, screen_height = 1280, 720
screen = pygame.display.set_mode((screen_width, screen_height), pygame.OPENGL | pygame.DOUBLEBUF)

ctx = moderngl.create_context()

renderer = Renderer(ctx, screen_width, screen_height)

cam = Camera()

world = World()
for x in range(16):
    for z in range(16):
        for y in range(2):
            world.set_block(x, y, z, 1)

chunk_pos = (0, 0, 0)

verts, indices = build_chunk_mesh(world, chunk_pos)
renderer.upload_chunk_mesh((0, 0, 0), verts, indices)

clock = pygame.time.Clock()

dt = 0

while True:
    moving = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
        
    cam.process_inputs(pygame.key.get_pressed(), dt)

    renderer.render(cam)
    pygame.display.set_caption(f"Voxel Engine | FPS: {clock.get_fps():.1f}")
    pygame.display.flip()
    dt = clock.tick(60) / 1000.0