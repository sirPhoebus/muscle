from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import moderngl as mgl


@dataclass
class CircleInstance:
    cx: float
    cy: float
    sx: float
    sy: float
    r: float
    g: float
    b: float


class GLRenderer:
    def __init__(self, ctx: mgl.Context, win_size: Tuple[int, int]):
        self.ctx = ctx
        self.win_size = win_size
        self.ctx.enable(flags=mgl.BLEND)
        self.ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_pos;               // base quad vertex in [-1,1]
                in vec2 i_center;             // world center in px
                in vec2 i_scale;              // half-size (rx, ry) in px
                in vec3 i_color;
                uniform vec2 u_cam;           // camera top-left in px
                uniform vec2 u_win;           // window size in px
                out vec2 v_local;             // local coordinates for fragment-circle test
                out vec3 v_color;
                void main(){
                    v_local = in_pos;        // for circle test in fragment
                    v_color = i_color;
                    vec2 world = i_center + in_pos * i_scale;
                    vec2 screen = world - u_cam;
                    vec2 ndc = (screen / u_win) * 2.0 - 1.0;
                    gl_Position = vec4(ndc, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                in vec2 v_local;              // local quad coords
                in vec3 v_color;
                out vec4 f_color;
                void main(){
                    // discard outside unit circle to get circle/ellipse
                    float r2 = dot(v_local, v_local);
                    if(r2 > 1.0){ discard; }
                    f_color = vec4(v_color, 1.0);
                }
            ''',
        )

        # Quad covering [-1,1]^2 as two triangles
        quad = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0,
        ], dtype='f4')
        self.vbo_quad = self.ctx.buffer(quad.tobytes())

        # Empty instance buffer (will be updated each frame)
        self.instance_stride = (2 + 2 + 3) * 4  # 2f center + 2f scale + 3f color
        self.instance_buf = self.ctx.buffer(reserve=0)

        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo_quad, '2f', 'in_pos'),
                (self.instance_buf, '2f 2f 3f/i', 'i_center', 'i_scale', 'i_color'),
            ],
        )

        # Uniforms
        self.u_cam = self.prog['u_cam']
        self.u_win = self.prog['u_win']

    def begin(self, cam: Tuple[float, float]):
        self.ctx.viewport = (0, 0, self.win_size[0], self.win_size[1])
        self.ctx.clear(9/255.0, 12/255.0, 24/255.0, 1.0)
        self.u_cam.value = (float(cam[0]), float(cam[1]))
        self.u_win.value = (float(self.win_size[0]), float(self.win_size[1]))

    def draw_instances(self, instances: List[CircleInstance]):
        if not instances:
            return
        # Build interleaved float32 array [cx,cy, sx,sy, r,g,b] per instance
        arr = np.fromiter(
            (v for inst in instances for v in (inst.cx, inst.cy, inst.sx, inst.sy, inst.r, inst.g, inst.b)),
            dtype='f4',
        )
        # Update instance buffer
        self.instance_buf.orphan(size=arr.nbytes)
        self.instance_buf.write(arr.tobytes())
        # Draw quad instanced
        self.vao.render(mode=mgl.TRIANGLES, instances=len(instances))

    def end(self):
        # With pygame backend, swap handled via pg.display.flip()
        pass

