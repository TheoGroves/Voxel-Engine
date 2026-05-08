import pygame
from lattice_manager import LatticeManager
from scroll_bar import ScrollBar

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Structure Editor")

clock = pygame.time.Clock()

block_paths = ["textures/Empty.png", "textures/Dirt.png", "textures/Grass.png", "textures/Rock.png", "textures/Cobblestone.png", "textures/Wood.png", "textures/Leaves.png"]
lattice = LatticeManager(screen, 16, 16, 16, 32)
lattice.load_blocks(block_paths)

selected_block = 0

sb = ScrollBar(730, 50)

layer = 0
last_layer = 1

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            key = event.unicode
            if key.isdigit() and 0 <= int(key) < len(block_paths):
                selected_block = int(key)
                print(f"Selected block {block_paths[selected_block].split('/')[-1].split('.')[0]}")

    mouse = pygame.mouse.get_pressed()
    if mouse[0] and not sb.block_input:
        pos = lattice.screen_to_grid(pygame.mouse.get_pos())
        lattice.place_block(pos, selected_block)

    screen.fill((0,0,0))
    lattice.render_grid()

    sb.update()
    layer = int((1-sb.get_pos()) * (lattice.depth-1))
    if not layer == last_layer:
        print(f"Selected layer {layer}")
        lattice.layer = layer

    sb.render(screen)

    pygame.display.flip()
    last_layer = layer
    clock.tick(60)

pygame.quit()

