from PIL import Image
import numpy as np

class TextureHandler:
    def __init__(self, atlas_width, atlas_height):
        self.atlas_width = atlas_width
        self.atlas_height = atlas_height
        self.atlas = None
        self.metadata = None

    def pack(self, paths):
        x, y = 0, 0
        row_height = 0
        atlas = Image.new("RGBA", (self.atlas_width, self.atlas_height))
        self.metadata = {}

        textures = []
        for path in paths:
            img = Image.open(path).convert("RGBA")
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            textures.append(img)

        for i, tex in enumerate(textures):
            if x + tex.width > self.atlas_width:
                x = 0
                y += row_height
                row_height = 0
            
            atlas.paste(tex, (x,y))
            self.metadata[i] = {"x": x, "y": y, "w": img.width, "h": img.height}

            x += img.width
            row_height = max(row_height, img.height)

        self.atlas = atlas

    def build_uv_table(self):
        uv_table = np.zeros((256, 4), dtype=np.float32)

        for block_id, m in self.metadata.items():
            u0 = m["x"] / self.atlas_width
            v0 = m["y"] / self.atlas_height
            u1 = (m["x"] + m["w"]) / self.atlas_width
            v1 = (m["y"] + m["h"]) / self.atlas_height

            uv_table[block_id] = (u0, v0, u1, v1)

        return uv_table

    def save_atlas(self, path):
        self.atlas.save(path)