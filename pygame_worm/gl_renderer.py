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
        # Reserve a small non-zero buffer; will be orphaned per frame
        self.instance_buf = self.ctx.buffer(reserve=4)

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

        # UI textured quad pipeline
        self.ui_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_uv;              // unit quad (0..1)
                in vec2 in_tc;              // texcoords (0..1)
                uniform vec4 u_rect;        // x,y,w,h in pixels
                uniform vec2 u_win;         // window size in px
                out vec2 v_tc;
                void main(){
                    v_tc = in_tc;
                    vec2 px = u_rect.xy + in_uv * u_rect.zw;
                    vec2 ndc = (px / u_win) * 2.0 - 1.0;
                    gl_Position = vec4(ndc, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                in vec2 v_tc;
                out vec4 f_color;
                uniform sampler2D u_tex;
                void main(){
                    f_color = texture(u_tex, v_tc);
                }
            ''',
        )
        # Unit quad for UI (two triangles) with UVs
        ui_vertices = np.array([
            # in_uv   in_tc
            0.0, 0.0,  0.0, 1.0,
            1.0, 0.0,  1.0, 1.0,
            1.0, 1.0,  1.0, 0.0,
            0.0, 0.0,  0.0, 1.0,
            1.0, 1.0,  1.0, 0.0,
            0.0, 1.0,  0.0, 0.0,
        ], dtype='f4')
        self.vbo_ui = self.ctx.buffer(ui_vertices.tobytes())
        self.vao_ui = self.ctx.vertex_array(
            self.ui_prog,
            [
                (self.vbo_ui, '2f 2f', 'in_uv', 'in_tc'),
            ],
        )
        self.u_rect = self.ui_prog['u_rect']
        self.u_win_ui = self.ui_prog['u_win']
        self._ui_tex = None  # type: ignore

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

    def blit_ui_rgba(self, rgba_bytes: bytes, width: int, height: int, x: int, y: int) -> None:
        """Blit an RGBA image as a UI panel at pixel rect (x,y,width,height)."""
        # Create or resize texture
        if self._ui_tex is None or self._ui_tex.size != (width, height):
            self._ui_tex = self.ctx.texture((width, height), 4)
            self._ui_tex.filter = (mgl.NEAREST, mgl.NEAREST)
        self._ui_tex.write(rgba_bytes)
        self._ui_tex.use(location=0)
        self.u_rect.value = (float(x), float(y), float(width), float(height))
        self.u_win_ui.value = (float(self.win_size[0]), float(self.win_size[1]))
        self.vao_ui.render(mode=mgl.TRIANGLES)
