from typing import Iterable, Optional
from collections import defaultdict
import sys
import inspect
import abc
from operator import itemgetter

import numpy as np
from numpy.typing import ArrayLike

from alias_py.numpy_dataclass import record


COMPONENT_BY_INDEX: list[type] = []
LAYER_BY_INDEX: list['Layer'] = []
ARCHETYPE_BY_COMPONENT_SET: dict[tuple, 'Archetype'] = {}
ARCHETYPE_BY_INDEX: list['Archetype'] = []
ENTITY_UNUSED_INDEXES: list[int] = []
ENTITY_GENERATION: list[int] = [0]
ENTITY_LAYER: list[int] = [0]
ENTITY_ARCHETYPE_INDEX: list[int] = [0]
ENTITY_ARCHETYPE_CODE: list[int] = [0]


def mk_component_set(components: list[type]) -> tuple[tuple, list[type]]:
  components = sorted(components, key=lambda x: x.__alias__component_index__)
  component_set = tuple(component.__alias__component_index__ for component in components)
  assert len(component_set) == len(set(component_set))
  return component_set, components


class PagedSOA:
  PAGE_SIZE: int = 1 << 16


  def __init__(self, *types: Iterable[np.dtype], prefix_type: Optional[np.dtype] = None):
    prefix_size = np.dtype(prefix_type).itemsize if prefix_type else 0
    self.sizes = [np.dtype(t).itemsize for t in types]

    self.size = 0
    self.itemsize = sum(self.sizes)
    self.types = (prefix_type, tuple(types))
    self.items_per_page = (self.PAGE_SIZE - prefix_size) // self.itemsize
    self.offsets = ((np.cumsum(np.array([0] + self.sizes))[:-1] * self.items_per_page) + prefix_size).tolist()
    self.pages = []


  def set_capacity(self, new_capacity: int):
    new_num_pages = (new_capacity + self.items_per_page - 1) // self.items_per_page
    while new_num_pages > len(self.pages):
      self.pages.append(np.zeros((self.PAGE_SIZE,), dtype=np.uint8))


  def space_for(self, count: int):
    new_num_pages = (self.size + count + self.items_per_page - 1) // self.items_per_page
    if new_num_pages > len(self.pages):
      self.set_capacity(self.size + count)


  def push(self) -> int:
    row = self.size
    self.size += 1
    page = row // self.items_per_page
    index = row % self.items_per_page
    return (page << 16) | index


  def page(self, code: int) -> np.array:
    page = code >> 16
    return self.pages[page]


  @staticmethod
  def decode_code(code: int) -> tuple[int, int]:
    return code >> 16, code & 0xFFFF


  def decode_column(self, column: int) -> tuple[int, int]:
    return self.sizes[column], self.offsets[column]


  def raw_access(self, page: int, index: int, size: int, offset: int) -> ArrayLike:
    start = offset + size * index
    return self.pages[page][start:start + size]


  def access(self, code: int, column: int) -> ArrayLike:
    page, index = self.decode_code(code)
    size, offset = self.decode_column(column)
    r = self.raw_access(page, index, size, offset)
    return np.record(r, dtype=self.types[1][column])


  def prefix(self, page: int):
    return np.ndarray((1,), dtype=self.types[0], buffer=self.pages[page], offset=0)
  

  def column(self, page: int, column: int) -> ArrayLike:
    size, offset = self.decode_column(column)
    return np.ndarray((self.items_per_page,), dtype=self.types[1][column], buffer=self.pages[page], offset=offset)

#

def require_components(cs: list[type]) -> list[type]:
  s = cs[:]
  r = set([])
  while s:
    c = s.pop()
    if c not in r:
      r.add(c)
      s.extend(c.__alias__component_required__)
  return list(r)


class Archetype:
  def __init__(self, components: list[type]):
    components = require_components(components)
    
    self.component_set, components = mk_component_set(components)
    self.types = components
    self.dtypes = [component.dtype for component in components]
    self.index = len(ARCHETYPE_BY_INDEX)
    self.free_codes = []
    self.paged_soa = PagedSOA(*([np.int32] + self.dtypes), prefix_type=np.dtype([('count', np.int32)]))

    ARCHETYPE_BY_INDEX.append(self)
    ARCHETYPE_BY_COMPONENT_SET[self.component_set] = self


  def allocate_code(self) -> int:
    if self.free_codes:
      return self.free_codes.pop()
    self.paged_soa.space_for(1)
    return self.paged_soa.push()


  def free_code(self, code: int):
    self.free_codes.append(code)


  def access(self, component_index: int, page: int, index: int):
    size, offset = self.paged_soa.decode_column(component_index + 1)
    r = self.paged_soa.raw_access(page, index, size, offset)
    return np.record(r, dtype=self.dtypes[component_index])[0]


class Entity:
  def __init__(self, index: int):
    self.index = index


  @classmethod
  def spawn(self, *components: list[any], layer: Optional[int] = None):
    components = sorted(components, key=lambda x: x.__alias__component_index__)
    
    component_set = tuple(component.__alias__component_index__ for component in components)

    if len(component_set) != len(set(component_set)):
      raise TypeError("Same component used more than once")

    archetype = ARCHETYPE_BY_COMPONENT_SET.get(component_set)
    if archetype == None:
      archetype = Archetype([type(c) for c in components])

    if ENTITY_UNUSED_INDEXES:
      entity_index = ENTITY_UNUSED_INDEXES.pop()
    else:
      entity_index = len(ENTITY_GENERATION)
      ENTITY_GENERATION.append(0)
      ENTITY_LAYER.append(0)
      ENTITY_ARCHETYPE_INDEX.append(0)
      ENTITY_ARCHETYPE_CODE.append(0)

    generation = ENTITY_GENERATION[entity_index]

    entity = Entity((generation << 32) | entity_index)

    entity.set_archetype(archetype)

    code = ENTITY_ARCHETYPE_CODE[entity_index & 0xFFFFFFFF]

    for component in components:
      index = archetype.component_set.index(component.__alias__component_index__)
      column = archetype.paged_soa.column(code >> 16, index + 1)
      column.put(code & 0xFFFF, component)


  def set_archetype(self, archetype: Archetype):
    old_archetype_index = ENTITY_ARCHETYPE_INDEX[self.index & 0xFFFFFFFF]
    old_archetype_code = ENTITY_ARCHETYPE_CODE[self.index & 0xFFFFFFFF]

    new_archetype_index = archetype.index
    new_archetype_code = archetype.allocate_code()

    ENTITY_ARCHETYPE_INDEX[self.index & 0xFFFFFFFF] = new_archetype_index
    ENTITY_ARCHETYPE_CODE[self.index & 0xFFFFFFFF] = new_archetype_code
    archetype.paged_soa.column(new_archetype_code >> 16, 0)[new_archetype_code & 0xFFFF] = self.index & 0xFFFFFFFF
    archetype.paged_soa.prefix(new_archetype_code >> 16)["count"] += 1

    if old_archetype_index != 0:
      raise NotImplementedError()

  
  
  # API
  def add_component(self, component: type):
    pass



class Query:
  class State:
    def __init__(self, components: tuple[int]):
      self.components = components
      self.true_component_set = set(components)
      self.archetypes: list[tuple[Archetype, list[tuple[type, int, int]]]] = []
      self.external_archetype_index = 0


    def find_archetype(self):
      while self.external_archetype_index < len(ARCHETYPE_BY_INDEX):
        archetype = ARCHETYPE_BY_INDEX[self.external_archetype_index]
        self.external_archetype_index += 1
        if self.true_component_set.issubset(archetype.component_set):
          columns = []
          for a in self.components:
            index = archetype.component_set.index(a)
            columns.append((archetype.types[index], index + 1))
          self.archetypes.append((archetype, columns))
          return True
      return False

  STATES: dict[tuple[tuple[int], tuple[int]], State] = {}

  def __init__(self, components: Iterable[type]):
    self.components, _ = mk_component_set(components)
    key = self.components

    if key in self.STATES:
      self.state = self.STATES[key]
    else:
      self.state = Query.State(self.components)
      self.STATES[key] = self.state


  def pages_internal(self) -> Iterable[tuple[int, np.ndarray, list[np.ndarray]]]:
    ai = 0
    while ai < len(self.state.archetypes) or self.state.find_archetype():
      archetype, columns = self.state.archetypes[ai]
      ai += 1

      for pi in range(len(archetype.paged_soa.pages)):
        entities = archetype.paged_soa.column(pi, 0)
        data = [archetype.paged_soa.column(pi, i) for _t, i in columns]

        yield ai - 1, entities, data


  # API
  def pages(self) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for ai, entities, data in self.pages_internal():
      yield entities, data


  def entities(self) -> Iterable[tuple[int, list[any]]]:
    for ai, entities, data in self.pages_internal():
      archetype, columns = self.state.archetypes[ai]

      for i in range(archetype.paged_soa.items_per_page):
        e = entities[i]
        if e:
          yield e, [d[i].view(t) for d, (t, _) in zip(data, columns)]


  def __iter__(self) -> Iterable[tuple[int, list[any]]]:
    return self.entities()


# component class decorator - a lot used from dataclasses
# 
# installs a __init__ that is simular to that, but uses numpy.record to store the data.
# types must by able to be used as a numpy record.
# 
# @component
# class Position:
#   dirty: bool = False
#   x: int
#   y: int
# 
# # becomes
# class Position(np.record):
#   __alias__component_index__ = 0
#   __alias__component_required__ = (,)
#   _COMPONENT_DTYPE = np.dtype([('dirty', bool), ('x', int), ('y', int)])
# 
#   def __init__(self, x: int, y: int, /, *, dirty: bool = False):
#     self.record = _nprecord((dirty, x, y), dtype=Position._COMPONENT_DTYPE)
#
#   def __init_unitialized__(self):
#     self.dirty = False

#def _component_init_field_value(f, globals, self_name):
#  default_name = f"_dflt_{f.name}"
#  if f.default_factory is not dataclasses.MISSING:
#    if f.init:
#      globals[default_name] = f.default_factory
#      value = f"{default_name}() if {f.name} is _HAS_DEFAULT_FACTORY else {f.name}"
#    else:
#      globals[default_name] = f.default_factory
#      value = f"{default_name}()"
#  else:
#    if f.init:
#      if f.default is dataclasses.MISSING:
#        value = f.name
#      else:
#        globals[default_name] = f.default
#        value = f.name
#    else:
#      return None
#  if f._field_type is dataclasses._FIELD_INITVAR:
#    return None
#  return value
#
#def _component_init_fn(fields, std_fields, kw_only_fields, dtype, self_name, globals):
#  seen_default = False
#
#  for f in std_fields:
#    if f.init:
#      if not (f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING):
#        seen_default = True
#      elif seen_default:
#        raise TypeError(f"non-default argument {f.name!r} follows default argument")
#
#  locals = {f"_type_{f.name}": f.type for f in fields}
#  locals.update({
#    "MISSING": dataclasses.MISSING,
#    "_HAS_DEFAULT_FACTORY": dataclasses._HAS_DEFAULT_FACTORY,
#    "__dataclass_builtins_object__": object,
#    "_nprecord": np.record,
#    "_dtype": dtype
#  })
#
#  values = []
#  for f in fields:
#    value = _component_init_field_value(f, locals, self_name)
#    if value:
#      values.append(value)
#
#  body_lines = [f"self._record = _nprecord((" + ",".join(values) + "), dtype=_dtype)"]
#
#  init_params = [dataclasses._init_param(f) for f in std_fields]
#  if kw_only_fields:
#    init_params += ["*"]
#    init_params += [dataclasses._init_param(f) for f in kw_only_fields]
#
#  return dataclasses._create_fn("__init__",
#    [self_name] + init_params,
#    body_lines,
#    locals=locals,
#    globals=globals,
#    return_type=None
#  )
#
#
#def _setup_component(cls, required: Optional[Iterable[type]]):
#  component_index = len(COMPONENT_BY_INDEX)
#  COMPONENT_BY_INDEX.append(cls)
#
#  fields = {}
#
#  if cls.__module__ in sys.modules:
#    globals = sys.modules[cls.__module__].__dict__
#  else:
#    globals = {}
#
#  setattr(cls, "__alias__component_index__", component_index)
#  setattr(cls, "__alias__component_required__", tuple(required) if required else tuple())
#
#  cls_annotations = cls.__annotations__
#
#  cls_fields = []
#  for name, type in cls_annotations.items():
#    f = dataclasses._get_field(cls, name, type, False)
#    if '_FIELDS' in type.__dict__:
#      print(f.type.__dict__)
#      raise NotImplementedError()
#    cls_fields.append(f)
#
#  for f in cls_fields:
#    fields[f.name] = f
#
#    if isinstance(getattr(cls, f.name, None), dataclasses.Field):
#      if f.default is dataclasses.MISSING:
#        delattr(cls, f.name)
#      else:
#        setattr(cls, f.name, f.default)
#
#  for name, value in cls.__dict__.items():
#    if isinstance(value, dataclasses.Field) and not name in cls_annotations:
#      raise TypeError(f"{name!r} is a field but has no type annotation")
#
#  setattr(cls, "_FIELDS", fields)
#
#  all_init_fields = [f for f in fields.values() if f._field_type in (dataclasses._FIELD, dataclasses._FIELD_INITVAR)]
#  (std_init_fields, kw_only_init_fields) = dataclasses._fields_in_init_order(all_init_fields)
#
#  field_list = [f for f in fields.values() if f._field_type is dataclasses._FIELD]
#
#  dtype_names = []
#  dtype_formats = []
#
#  for f in field_list:
#    dtype_names.append(f.name)
#    dtype_formats.append(f.type)
#
#  dtype = np.dtype({"names": dtype_names, "formats": dtype_formats})
#
#  setattr(cls, "_COMPONENT_DTYPE", dtype)
#
#  setattr(cls, "__init__", _component_init_fn(
#    all_init_fields,
#    std_init_fields,
#    kw_only_init_fields,
#    dtype,
#    '__dataclass_self_' if 'self' in fields else 'self',
#    globals
#  ))
#
#  locals = {f"_type_{f.name}": f.type for f in field_list}
#  locals["_ndrecord"] = np.record
#
#  setattr(cls, "__from_record__", classmethod(dataclasses._create_fn(
#    "__from_record__",
#    ("cls", "record:_ndrecord"),
#    [f"self = {cls.__name__}.__new__({cls.__name__})", "self._record = record", "return self"],
#    globals=globals,
#    locals=locals
#  )))
#
#  repr_fields = [f for f in field_list if f.repr]
#  repr_fn = dataclasses._create_fn(
#    "__repr__",
#    ("self",),
#    ['return self.__class__.__qualname__ + f"(' + ", ".join([f"{f.name}={{self.{f.name}!r}}" for f in field_list if f.repr]) + ')"'],
#    globals=globals
#  )
#  setattr(cls, "__repr__", dataclasses._recursive_repr(repr_fn))
#
#  for f in field_list:
#    get_fn = dataclasses._create_fn(
#      f"get_{f.name}",
#      ("self",),
#      [f"return self._record[\"{f.name}\"]"],
#      locals=locals,
#      globals=globals
#    )
#    set_fn = dataclasses._create_fn(
#      f"set_{f.name}",
#      ("self", f"{f.name}:_type_{f.name}"),
#      [f"self._record[\"{f.name}\"] = {f.name}"],
#      locals=locals,
#      globals=globals
#    )
#    setattr(cls, f.name, property(get_fn, set_fn))
#
#  if not getattr(cls, "__doc__"):
#    try:
#      text_sig = str(inspect.signature(cls)).replace(" -> None", "")
#    except:
#      text_sig = ""
#    cls.__doc__ = cls.__name__ + text_sig
#
#  abc.update_abstractmethods(cls)
#
#  return cls

def _remove_entity_index(entity: int):
  ENTITY_GENERATION[entity] += 1


def _remove_entity(entity: int):
  assert (entity >> 32) == ENTITY_GENERATION[entity & 0xFFFFFFFF]
  _remove_entity_index(entity & 0xFFFFFFFF)


# API
class Layer:
  def __init__(self, name: str, *, cleanup_entities: bool = False):
    self.index = len(LAYER_BY_INDEX)
    self.name = name
    self.cleanup_entities = cleanup_entities
    self.entities = []
    LAYER_BY_INDEX.append(self)


  def __del__(self):
    if self.cleanup_entities:
      self.remove_all_entities()
      

  def remove_all_entities(self):
    for entity in self.entities:
      _remove_entity_index(entity)
      

def component(cls=None, /, *, required: Optional[Iterable[type]] = None):
  def wrap(cls):
    cls = record(cls)

    component_index = len(COMPONENT_BY_INDEX)
    COMPONENT_BY_INDEX.append(cls)

    setattr(cls, "__alias__component_index__", component_index)
    setattr(cls, "__alias__component_required__", tuple(required) if required else tuple())

    return cls

  if cls is None:
    return wrap
  return wrap(cls)


def spawn(*components: list[type], layer: Optional[int] = None):
  return Entity.spawn(*components, layer=layer)


def query(*component_types: list[type], modified=[], exclude=[]) -> Query:
  return Query(component_types)


