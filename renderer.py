import os
import moderngl
import numpy as np
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
        self.ctx.wireframe = True
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
            [(vbo, "3f", "aPos")],
            index_buffer=ibo
        )

        self.chunks[chunk_pos] = (vao, len(indices))

    def render(self, camera):
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.1, 0.1, 0.1)

        proj = self.get_projection_matrix()
        view = camera.get_view_matrix()
        model = get_model_matrix((0, 0, 0))

        self.program["projection"].write(proj.T.tobytes())
        self.program["view"].write(view.T.tobytes())
        self.program["model"].write(model.T.tobytes())

        for vao, count in self.chunks.values():
            vao.render()