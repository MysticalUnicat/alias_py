import numpy as np

# @record
# class Foo:
#   a: int
#
# @record
# class Bar:
#   b: float 
#   foo: Foo
#
# # becomes
# class Foo(np.void):
#   dtype = np.dtype([('a', np.int32)])
# 
#   def __new__(cls, a: np.int32):
#     r = np.void((1,), dtype=cls.dtype)
#     return r.view(cls)
# 
#   def __repr__(self):
#     return f"{self.__class__.__name__}(a={self.a})"
# 
#   def __str__(self):
#     return self.__repr__()
# 
#   @property
#   def a(self):
#     return self["a"]
# 
# class Bar(np.void):
#   dtype = np.dtype({'names': ['b', 'foo'], 'formats': [np.int32, Foo]})
# 
#   def __new__(cls, b: float, foo: Foo):
#     return np.void((b, foo), dtype=cls.dtype).view(cls)
# 
#   def __repr__(self):
#     return f"{self.__class__.__name__}(b={self.b}, foo={self.foo})"
# 
#   def __str__(self):
#     return self.__repr__()
# 
#   @property
#   def b(self):
#     return self["b"]
# 
#   @property
#   def foo(self):
#     return self["foo"]

class Missing:
  pass
MISSING = Missing()

TYPE_REMAP = {
  int: np.int32,
  float: np.float32
}

class Field:
  __slots__ = ("name", "type", "default")

  def __init__(self, name, type, default):
    self.name = name
    self.type = type
    self.default = default


def _create_dtype(fields):
  names = []
  formats = []
  for field in fields:
    names.append(field.name)
    formats.append(field.type)
  return np.dtype({'names': names, 'formats': formats})


def _create_fn(name, args, body, *, globals=None, locals=None, return_type=MISSING):
  if locals is None:
    locals = {}
  return_annotation = ''
  if return_type is not MISSING:
    locals['__alias__return_type__'] = return_type
    return_annotation = '-> __alias__return_type__'
  args = ",".join(args)
  body = "\n".join(f"  {b}" for b in body)
  txt = f" def {name}({args}){return_annotation}:\n{body}"
  local_vars = ", ".join(locals.keys())
  txt = f"def __create_fn__({local_vars}):\n{txt}\n return {name}"
  ns = {}
  exec(txt, globals, ns)
  return ns["__create_fn__"](**locals)


def _create_new_fn(fields, dtype):
  locals = {
    '_npvoid': np.void,
    '__alias__dtype__': dtype
  }

  args = ["cls"]
  for field in fields:
    type_key = f"__alias__{field.name}_type__"
    default_key = f"__alias__{field.name}_default__"
    locals[type_key] = field.type
    name_and_type = f"{field.name}:{type_key}"
    if field.default is not MISSING:
      locals[default_key] = self.default
      args.append(f"{name_and_type} = {default_key}")
    else:
      args.append(name_and_type)

  comma = "," if len(fields) == 1 else ""

  return _create_fn("__new__", args, [
    f"return _npvoid((" + ",".join(field.name for field in fields) + f"{comma}), dtype=__alias__dtype__).view(cls)"
  ], locals=locals)


def _create_repr_fn(fields):
  locals = {}
  return _create_fn("__repr__", ("self",), [
    "return f'{self.__class__.__name__}(" + ",".join(f"{field.name}={{self.{field.name}}}" for field in fields) + ")'"
  ], locals=locals)


def _create_str_fn(fields):
  locals = {}
  return _create_fn("__repr__", ("self",), [
    "return f'{self.__class__.__name__}(" + ",".join(f"{field.name}={{self.{field.name}}}" for field in fields) + ")'"
  ], locals=locals)


def _construct_record_class(name, fields, bases=None, namespace=None):
  cls_dict = {}

  dtype = _create_dtype(fields)

  cls_dict["__repr__"] = _create_repr_fn(fields)
  cls_dict["__str__"] = _create_str_fn(fields)

  for field in fields:
    get_key = f"get_{field.name}"
    set_key = f"set_{field.name}"
    cls_dict[get_key] = _create_fn(get_key, ("self",), [f"return self['{field.name}']"], return_type=field.type)
    cls_dict[set_key] = _create_fn(set_key, ("self", "value"), [f"self['{field.name}'] = value"])
    cls_dict[field.name] = property(cls_dict[get_key], cls_dict[set_key])

  if namespace is not None:
    cls_dict.update(namespace)

  cls_dict["dtype"] = dtype
  cls_dict["__new__"] = _create_new_fn(fields, dtype)

  if bases is not None:
    bases = list(bases)
    bases.append(np.void)
  else:
    bases = [np.void]

  return type(name, tuple(bases), cls_dict)


def _extract_fields_from_class(cls):
  fields = []
  
  for key in cls.__annotations__.keys():
    default = MISSING
    if key in cls.__dict__:
      default = cls.__dict__[key]
    type = cls.__annotations__[key]
    if type in TYPE_REMAP:
      type = TYPE_REMAP[type]
    fields.append(Field(key, type, default))

  return fields


def _extract_namespace_from_class(cls, fields):
  namespace = dict(cls.__dict__)

  for field in fields:
    namespace.pop(field.name, None)

  namespace.pop("__dict__")
  namespace.pop("__weakref__")

  return namespace


def record(cls=None):
  def wrap(cls):
    fields = _extract_fields_from_class(cls)
    namespace = _extract_namespace_from_class(cls, fields)
    return _construct_record_class(cls.__name__, fields, namespace=namespace)
  if cls is None:
    return wrap
  return wrap(cls)


def make_record(name, fields, bases=None, namespace=None):
  new_fields = []
  
  for field in fields:
    new_fields.append(Field(field[0], field[1], MISSING))

  return _construct_record_class(name, new_fields, bases=bases, namespace=namespace)

