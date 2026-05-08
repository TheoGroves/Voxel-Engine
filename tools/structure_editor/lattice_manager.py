import pygame

class LatticeManager:
    def __init__(self, screen, width, height, depth, grid_size=16):
        self.screen = screen
        self.blocks = []
        self.grid_size = grid_size
        self.width = width
        self.height = height
        self.depth = depth
        self.middle = (self.width//2, self.height//2)

        # shape: [depth][height][width]
        self.lattice = [[[0] * width for _ in range(height)] for _ in range(depth)]
        self.layer = 0

    def load_blocks(self, paths):
        for path in paths:
            self.blocks.append(pygame.image.load(path).convert_alpha())

    def screen_to_grid(self, screen_pos):
        return (max(0, min(self.width-1, screen_pos[0] // self.grid_size)), max(0, min(self.height-1, screen_pos[1] // self.grid_size + 1)))

    def place_block(self, pos, block_type):
        self.lattice[self.layer][pos[1]][pos[0]] = block_type

    def render_grid(self):
        plane = self.lattice[self.layer]
        for y in range(len(plane)):
            for x in range(len(plane[y])):
                block = plane[y][x]
                tex = pygame.transform.scale(self.blocks[block], (self.grid_size, self.grid_size))
                screen_x = x * self.grid_size
                screen_y = y * self.grid_size
                self.screen.blit(tex, (screen_x, screen_y-self.grid_size))
                pygame.draw.circle(self.screen, (255, 255, 255), (self.middle[0]*self.grid_size - self.grid_size/2, self.middle[1]*self.grid_size - self.grid_size/2), 2)