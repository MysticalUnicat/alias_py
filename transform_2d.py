from alias_py.pga2d import *
from alias_py.ecs import component, query

import numpy as np


@component
class LocalToWorld:
  motor: Motor
  position: Point
  orientation: float


@component(required=[LocalToWorld])
class Transform:
  motor: Motor


@component(required=[Transform])
class Translation:
  direction: Direction


@component(required=[Transform])
class Rotation:
  rotation: float


@component
class Parent:
  handle: int


def _update_translation():
  for entity, [transform, translation] in query(Transform, Translation, modified=[Translation], exclude=[Rotation]):
    transform.motor = Motor(1, translation.direction.e01 * 0.5, translation.direction.e02 * 0.5, 0)

def _update_rotation():
  for entity, [transform, rotation] in query(Transform, Rotation, modified=[Rotation], exclude=[Translation]):
    pass

def _update_translation_rotation():
  for entity, [transform, translation, rotation] in query(Transform, Translation, Rotation, modified=[Translation, Rotation]):
    pass

def _update_parent_world():
  for entity, [local_to_world, transform] in query(LocalToWorld, Transform, modified=[Transform], exclude=[Parent]):
    local_to_world.motor = transform.motor
    local_to_world.position = local_to_world.motor >> BiVector(0, 0, 1)
    local_to_world.orientation = local_to_world.motor.e12 * 2

def _update_child_world():
  for entity, [local_to_world, transform] in query(LocalToWorld, Transform, modified=[Transform, Parent]):
    pass

def update():
  _update_translation()
  _update_rotation()
  _update_translation_rotation()
  _update_parent_world()
  _update_child_world()

