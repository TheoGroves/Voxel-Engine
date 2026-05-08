import pygame

class ScrollBar:
    def __init__(self, x, y, width=20, height=500):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.thumb_position = 1
        self.thumb_height = 50

        self.block_input = False
        self.started_dragging = False
        self.drag_offset = 0
        self.thumb_rect = None

    def update(self):
        mx,my = pygame.mouse.get_pos()
        mouse_on_sb = self.x <= mx <= self.x + self.width and self.y <= my <= self.y + self.height

        mouse = pygame.mouse.get_pressed()

        if mouse[0] and self.thumb_rect and self.thumb_rect.collidepoint(mx, my):
            self.started_dragging = True
            self.drag_offset = my - self.thumb_rect.y
        
        if not mouse[0]:
            self.started_dragging = False

        if self.started_dragging:
            track_height = self.height - self.thumb_height
            new_pos = my - self.y - self.drag_offset
            new_pos = max(0, min(track_height, new_pos))

            self.thumb_position = new_pos / track_height

        self.block_input = mouse_on_sb or self.started_dragging

    def get_pos(self):
        return self.thumb_position

    def render(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(self.x, self.y, self.width, self.height), 2, 3)
        thumb_pos = self.y + (self.height - self.thumb_height - 2) * self.thumb_position
        self.thumb_rect = pygame.Rect(self.x+2, thumb_pos+2, self.width-4, self.thumb_height)
        pygame.draw.rect(screen, (245, 245, 245), self.thumb_rect, 2, 2)