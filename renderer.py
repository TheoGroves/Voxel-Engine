import os
import moderngl
from matrices import perspective, get_model_matrix

shader_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")

with open(os.path.join(shader_dir, "block.vert")) as f:
    VERT_SHADER = f.read()

with open(os.path.join(shader_dir, "block.frag")) as f:
    FRAG_SHADER = f.read()

class Renderer:
    def __init__(self, ctx: moderngl.Context, width: int, height:int):
        self.ctx = ctx
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.front_face = "ccw"
        self.ctx.cull_face = "back"
        self.ctx.wireframe = False
        self.width = width
        self.height = height

        self.fov = 90.0
        self.near = 0.1
        self.far = 1000.0
        self.aspect = width / height

        self.program = self.ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=FRAG_SHADER,
        )

        self.chunks = {}

    def get_projection_matrix(self):
        return perspective(self.fov, self.aspect, self.near, self.far)
    
    def upload_chunk_mesh(self, chunk_pos, vertices, indices):
        vbo = self.ctx.buffer(vertices.astype("f4").tobytes())
        ibo = self.ctx.buffer(indices.astype("u4").tobytes())

        vao = self.ctx.vertex_array(
            self.program,
            [(vbo, "3f 3f 2f 1f", "aPos", "aNormal", "uvCoords", "aAO")],
            index_buffer=ibo
        )

        self.chunks[chunk_pos] = {
            "vao": vao,
            "vbo": vbo,
            "ibo": ibo,
            "count": len(indices),
            "triangles": len(indices) // 3
        }

    def remove_chunk(self, chunk_pos):
        chunk = self.chunks.pop(chunk_pos, None)
        if chunk is None:
            return

        chunk["vao"].release()
        chunk["vbo"].release()
        chunk["ibo"].release()

    def load_atlas(self, image):
        tex = self.ctx.texture(image.size, 4, image.tobytes())
        tex.build_mipmaps()
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.use(location=0)

        self.program["atlas"] = 0
        self.atlas = tex
        
    def render(self, camera):
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.8, 0.949, 1)

        proj = self.get_projection_matrix()
        view = camera.get_view_matrix()
        model = get_model_matrix((0, 0, 0))

        self.program["projection"].write(proj.T.tobytes())
        self.program["view"].write(view.T.tobytes())
        self.program["model"].write(model.T.tobytes())
        self.atlas.use(location=0)

        for chunk in self.chunks.values():
            chunk["vao"].render()

        return sum(chunk["triangles"] for chunk in self.chunks.values())