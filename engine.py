from alias_py import ecs
from alias_py import transform_2d as t

import time
import numpy as np
import glfw
import OpenGL
from OpenGL.GL import *


@ecs.component(required=(t.LocalToWorld,))
class Camera:
  clear_color: (np.float32, 4)
  left: np.float32
  top: np.float32
  right: np.float32
  bottom: np.float32


@ecs.component
class DrawBox:
  width: np.float32
  height: np.float32
  color: (np.float32, 4)


def draw2d(vertexes, texcoords, colors, indexes):
  glBegin(GL_TRIANGLES)
  for index in indexes:
    glColor4f(colors[index][0], colors[index][1], colors[index][2], colors[index][3])
    glTexCoord2f(texcoords[index][0], texcoords[index][1])
    glVertex2f(vertexes[index][0], vertexes[index][1])
  glEnd()


def draw_boxes():
  for e, [local_to_world, draw_box] in ecs.query(t.LocalToWorld, DrawBox):
    x, y = local_to_world.position.e01, local_to_world.position.e02
    draw2d(
      vertexes=[
        [x - draw_box.width / 2, y - draw_box.height / 2],
        [x - draw_box.width / 2, y + draw_box.height / 2],
        [x + draw_box.width / 2, y + draw_box.height / 2],
        [x + draw_box.width / 2, y - draw_box.height / 2]
      ],
      texcoords=[[0, 0], [0, 1], [1, 1], [1, 0]],
      colors=[draw_box.color] * 4,
      indexes=[0, 1, 2, 0, 2, 3]
    )


class ViewState:
  def on_begin(self, engine: 'Engine'):
    pass

  def on_pause(self, engine: 'Engine'):
    pass

  def draw(self, engine: 'Engine', active: bool):
    pass

  def on_resume(self, engine: 'Engine'):
    pass

  def on_stop(self, engine: 'Engine'):
    pass


class Engine:
  def __init__(self, first_view_state: type):
    self.first_view_state = first_view_state
    self.view_states = []


  def setup(self):
    assert len(self.view_states) == 0
    self.view_states.append(self.first_view_state())
    self.view_states[-1].on_begin(self)


  def push_view_state(self, view_state: ViewState):
    if len(self.view_states) > 0:
      self.view_states[-1].on_pause(self)
    self.view_states.append(view_state)
    self.view_states[-1].on_begin(self)


  def pop_view_state(self, count: int = 1):
    while count > 0 and len(self.view_states) > 0:
      count -= 1
      self.view_states[-1].on_stop(self)
      self.view_states.pop()
    if len(self.view_states) == 0:
      glfw.set_window_should_close(self.window, 1)
      return
    self.view_states[-1].on_resume(self)


  def run(self):
    if not glfw.init():
      return

    self.window = glfw.create_window(800, 600, "test.py", None, None)
    if not self.window:
      glfw.terminate()
      return

    glfw.make_context_current(self.window)

    window_width, window_height = glfw.get_window_size(self.window)

    while not glfw.window_should_close(self.window):
      t.update()

      for _, [local_to_world, camera] in ecs.query(t.LocalToWorld, Camera):
        x1 = window_width * camera.left
        y1 = window_height * camera.top
        x2 = window_width * camera.right
        y2 = window_height * camera.bottom
        width = x2 - x1
        height = y2 - y1

        glViewport(int(x1), int(y1), int(width), int(height))
        glClearColor(camera.clear_color[0], camera.clear_color[1], camera.clear_color[2], camera.clear_color[3])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(x1, -width, height, y1, -99999, 99999)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslate(-local_to_world.position.e01, -local_to_world.position.e02, 0)

        draw_boxes()

      for view_state in self.view_states[:-1]:
        view_state.draw(self, False)
      self.view_states[-1].draw(self, True)
      
      glfw.swap_buffers(self.window)
      glfw.poll_events()

    glfw.terminate()

