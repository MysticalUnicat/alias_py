from typing import Iterable
from dataclasses import dataclass, field, _create_fn, KW_ONLY
from enum import Enum

import numpy as np

from alias_py.numpy_dataclass import make_record

ONE = 0
NEGATIVE = 1
DEGENERATE = 2
POSITIVE = 3

metric_sign = [1, -1, 0, 1]

@dataclass(slots=True)
class Basis:
  name: str
  bits: int
  grade: int
  metric: int


@dataclass(slots=True)
class Layout:
  p: int
  q: int
  r: int
  _: KW_ONLY
  first_basis: int = field(init=False)
  num_dimensions: int = field(init=False)
  num_grades: int = field(init=False)
  num_basis: int = field(init=False)
  basis: list[int] = field(init=False)
  basis_by_bits: list[int] = field(init=False)
  cayley_table_basis: list[int] = field(init=False)
  cayley_table_sign: list[int] = field(init=False)
  graded_types: list[type] = field(init=False)

  def __post_init__(self):
    self.first_basis = 0 if self.r else 1
    self.num_dimensions = self.p + self.q + self.r
    self.num_grades = 1 + self.num_dimensions
    self.num_basis = 1 << self.num_dimensions
    self.graded_types = []

    basis = []
    for i in range(self.num_basis):
      name = 'e'
      for j in range(self.num_dimensions):
        if i & (1 << j):
          name += chr(ord('0') + self.first_basis + j)

      basis.append(Basis(name if i else 'one', i, len(name) - 1, 0))

    # sort by name (one at front)
    self.basis = sorted(basis, key=lambda x: (len(x.name) if x.name != 'one' else 0, x.name))

    # now that basis is sorted, index basis by the bits field for computed basis index
    self.basis_by_bits = [self.basis[i].bits for i in range(self.num_basis)]

    # build basis metric, how it behaves when multiplied by itself
    for i in range(self.num_basis):
      if self.basis[i].grade == 0:
        self.basis[i].metric = ONE
      elif self.basis[i].grade == 1:
        if i - 1 < self.r:
          self.basis[i].metric = DEGENERATE
        elif i - 1 - self.r < self.p:
          self.basis[i].metric = POSITIVE
        else:
          self.basis[i].metric = NEGATIVE
      else:
        sign = -1

        for j in range(self.num_dimensions):
          if (self.basis[i].bits & (1 << j)) == 0:
            continue
          other = self.basis_by_bits[1 << j]
          if self.basis[other].metric == DEGENERATE:
            sign = 0
            break
          if self.basis[other].metric == NEGATIVE:
            sign *= -1

        self.basis[i].metric = DEGENERATE + sign

    # cayley basis and sign tables
    self.cayley_table_basis = [0] * (self.num_basis * self.num_basis)
    self.cayley_table_sign = [0] * (self.num_basis * self.num_basis)

    for a in range(self.num_basis):
      for b in range(self.num_basis):
        c_basis, c_sign = self.mul_basis_sign(a, b)
        self.cayley_table_basis[a * self.num_basis + b] = c_basis
        self.cayley_table_sign[a * self.num_basis + b] = c_sign


  def mul_basis_sign(self, a: int, b: int) -> tuple[int, int]:
    if a == b:
      return (0, metric_sign[self.basis[a].metric])
    if a == 0 or b == 0:
      return (a | b, 1)
    ab = self.basis[a].bits
    bb = self.basis[b].bits
    sign = 1
    nums = [ord(x) for x in (self.basis[a].name[1:] + self.basis[b].name[1:])]
    length = len(nums)
    modified = True
    while modified:
      modified = False
      i = 1
      while i < length:
        if nums[i - 1] == nums[i]:
          i -= 1
          sign *= metric_sign[self.basis[(nums[i] - (ord('0') + self.first_basis)) + 1].metric]
          nums = nums[:i] + nums[i+2:]
          length -= 2
          modified = True
        elif nums[i - 1] > nums[i]:
          t = nums[i - 1]
          nums[i - 1] = nums[i]
          nums[i] = t
          sign *= -1
          modified = True
        i += 1
    return (self.basis_by_bits[ab ^ bb], sign)


  def dual_basis_sign(self, a: int) -> tuple[int, int]:
   if self.r:
     b = self.num_basis - 1 - a
     return (b, self.mul_basis_sign(a, b)[1])
   else:
     return self.mul_basis_ref(a, self.num_basis - 1)[0]


  def key(self) -> int:
    return self.p * 3 + self.q * 5 + self.r * 7 + self.first_basis * 11


class MultiVector:
  def negate(self):
    return Negate(self)


  def grades(self, indexes: Iterable[int]):
    return Grades(self, set(indexes))


  def grade(self, index: int):
    return Grades(self, set([index]))


  def dual(self):
    return Dual(self)


  def undual(self):
    return Undual(self)


  def polar(self):
    return Polar(self)


  def reverse(self):
    return Reverse(self)


  def involute(self):
    return Involute(self)


  def conjugate(self):
    return Conjugate(self)


  def add(self, other):
    return Add(self, other)


  def subtract(self, other):
    return Subtract(self, other)


  def geometric_product(self, other):
    return GeometricProduct(self, other)


  def outer_product(self, other):
    return OuterProduct(self, other)


  def regressive_product(self, other):
    return RegressiveProduct(self, other)


  def inner_product(self, other):
    return InnerProduct(self, other)


  def __neg__(a):
    return Negate(a)


  def __add__(a, b):
    return Add(a, b)


  def __sub__(a, b):
    return Subtract(a, b)


  def __mul__(a, b):
    return GeometricProduct(a, b)


  def __xor__(a, b):
    return OuterProduct(a, b)


  def __and__(a, b):
    return RegressiveProduct(a, b)


  def __or__(a, b):
    return InnerProduct(a, b)


  def __multivector_layout__(self) -> Layout:
    raise NotImplementedError()


  def __multivector_mask__(self) -> Iterable[int]:
    raise NotImplementedError()


  def __multivector_code__(self, args: list[str], locals: dict, lines: list[str], prefix: str):
    raise NotImplementedError()


  def __multivector_make_function__(self, layout=None):
    if layout is None:
      layout = self.__multivector_layout__()

    if layout is None:
      return MultiVectorFunction(self)

    locals = {}
    lines = []
    args = []
    mask = self.__multivector_mask__(layout)
    self.__multivector_code__(layout, args, locals, lines, "")
    gradebits = 0
    for b in mask:
      gradebits |= 1 << layout.basis[b].grade
    locals["__result_type__"] = layout.graded_types[gradebits]
    result_args = []
    for b in range(layout.num_basis):
      if (gradebits & (1 << layout.basis[b].grade) != 0):
        if b in mask:
          result_args.append(f"_{layout.basis[b].name}")
        else:
          result_args.append("0")
    lines.append("return __result_type__(" + ",".join(result_args) + ")")
    txt = "def code_fn(" + ",".join(args) + ") -> __result_type__:\n" + "\n".join(f"  {line}" for line in lines)
    ns = {}
    exec(txt, locals, ns)
    return ns['code_fn']


  def __call__(self, *args, **kwargs):
    f = self.__multivector_make_function__()
    return f(*args, **kwargs)


@dataclass
class MultiVectorFunction: 
  expression: MultiVector
  layout_cache: dict = field(default_factory=dict)


  def __post_init__(self):
    assert self.expression.__multivector_layout__() == None


  def __call__(self, *args, **kwargs):
    # get at least one layout from arguments
    layout = None
    for a in list(args) + list(kwargs.values()):
      l = getattr(a, '__multivector_layout__', lambda: None)()
      if layout is not None and l != layout:
        raise TypeError("mixed layouts in arguments")
      layout = l
    if layout is None:
      raise TypeError("could not determine layout")

    key = layout.key()

    if key not in self.layout_cache:
      self.layout_cache[key] = (layout, self.expression.__multivector_make_function__(layout))
    return self.layout_cache[key][1](*args, **kwargs)
  

@dataclass
class Argument(MultiVector):
  name: str | int
  gradebits: int = -1


  def __post_init__(self):
    if not isinstance(self.name, str):
      self.name = f"_{self.name}"


  def __multivector_layout__(self):
    return None


  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    if self.gradebits < 0:
      return range(layout.num_basis)
    else:
      for i in range(layout.num_basis):
        if (self.gradebits & (1 << i)) != 0:
          yield i


  def __multivector_code__(self, layout: Layout, args: list[str], locals: dict, lines: list[str], prefix: str) -> str:
    if self.name not in args:
      args.append(self.name)
    mask = self.__multivector_mask__(layout)
    for b in mask:
      lines.append(f"{prefix}_{layout.basis[b].name} = {self.name}.{layout.basis[b].name}") 


@dataclass
class Scalar(Argument):
  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    for b in range(layout.num_basis):
      if layout.basis[b].grade == 0:
        yield b


@dataclass
class Vector(Argument):
  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    for b in range(layout.num_basis):
      if layout.basis[b].grade == 1:
        yield b


@dataclass
class BiVector(Argument):
  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    for b in range(layout.num_basis):
      if layout.basis[b].grade == 2:
        yield b


@dataclass
class TriVector(Argument):
  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    for b in range(layout.num_basis):
      if layout.basis[b].grade == 2:
        yield b


@dataclass
class AntiScalar(Argument):
  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    for b in range(layout.num_basis):
      if layout.basis[b].grade == layout.num_grades - 1:
        yield b


@dataclass
class AntiVector(Argument):
  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    for b in range(layout.num_basis):
      if layout.basis[b].grade == layout.num_grades - 2:
        yield b


@dataclass
class AntiBiVector(Argument):
  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    for b in range(layout.num_basis):
      if layout.basis[b].grade == layout.num_grades - 3:
        yield b


@dataclass
class AntiTriVector(Argument):
  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    for b in range(layout.num_basis):
      if layout.basis[b].grade == layout.num_grades - 4:
        yield b


@dataclass
class Unary(MultiVector):
  a: MultiVector


  def __multivector_layout__(self):
    return self.a.__multivector_layout__()


@dataclass
class Binary(MultiVector):
  a: MultiVector
  b: MultiVector


  def __multivector_layout__(self):
    a = self.a.__multivector_layout__()
    b = self.b.__multivector_layout__()
    if a is not None and b is not None and a != b:
      raise TypeError("incompatible geometric spaces")
    return a if a is not None else b


@dataclass
class Negate(Unary):
  a: MultiVector


  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    return self.a.__multivector_mask__(layout)


  def __multivector_code__(self, layout: Layout, args: list[str], locals: dict, lines: list[str], prefix: str) -> str:
    self.a.__multivector_code__(layout, args, locals, lines, prefix + "a")
    for b in self.__multivector_mask__(layout):
      lines.append(f"{prefix}_{basis[b].name} = -{prefix}a_{basis[b].name}")

@dataclass
class Grades(MultiVector):
  a: MultiVector
  indexes: set[int]


  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    for ab in self.a.__multivector_mask__(layout):
      if basis[ab].grade in self.indexes:
        yield ab


  def __multivector_code__(self, layout: Layout, args: list[str], locals: dict, lines: list[str], prefix: str):
    self.a.__multivector_code__(layout, args, locals, lines, prefix + "a")
    for b in self.__multivector_mask__(layout):
      lines.append(f"{prefix}_{basis[b].name} = {prefix}a_{basis[b].name}")

@dataclass
class Dual(MultiVector):
  a: MultiVector


  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    for ab in self.a.__multivector_mask__(layout):
      yield dual(BasisRef(ab, 1)).basis


  def __multivector_code__(self, layout: Layout, args: list[str], locals: dict, lines: list[str], prefix: str) -> str:
    self.a.__multivector_code__(layout, args, locals, lines, prefix + "a")
    for b in self.__multivector_mask__(layout):
      lines.append(f"{prefix}_{basis[b].name} = {prefix}a_{basis[dual(BasisRef(b, a)).basis].name}")


@dataclass
class Undual(MultiVector):
  a: MultiVector


  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    for ab in self.a.__multivector_mask__(layout):
      yield dual(BasisRef(ab, 1)).basis


  def __multivector_code__(self, layout: Layout, args: list[str], locals: dict, lines: list[str], prefix: str) -> str:
    self.a.__multivector_code__(layout, args, locals, lines, prefix + "a")
    for b in self.__multivector_mask__(layout):
      d1 = dual(BasisRef(b, 1))
      d2 = dual(BasisRef(num_basis - b - 1, 1))
      if d2.sign < 0:
        lines.append(f"{prefix}_{basis[b].name} = -{prefix}a_{basis[d1.basis].name}")
      else:
        lines.append(f"{prefix}_{basis[b].name} = {prefix}a_{basis[d1.basis].name}")


@dataclass
class Polar(MultiVector):
  a: MultiVector


  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    for ab in self.a.__multivector_mask__(layout):
      for jb in range(num_basis):
        polar = mul_basis_ref(BasisRef(jb, 1), BasisRef(num_basis - 1, 1))
        if polar.basis != jb:
          continue
        if polar.sign != 0:
          yield polar.basis


  def __multivector_code__(self, layout: Layout, args: list[str], locals: dict, lines: list[str], prefix: str) -> str:
    self.a.__multivector_code__(layout, args, locals, lines, prefix + "a")
    for ab in self.a.__multivector_mask__(layout):
      for jb in range(num_basis):
        polar = mul_basis_ref(BasisRef(jb, 1), BasisRef(num_basis - 1, 1))
        if polar.basis != ab:
          continue
        if polar.sign < 0:
          lines.append(f"{prefix}_{basis[polar.basis].name} = -{prefix}a_{basis[ab].name}")
        else:
          lines.append(f"{prefix}_{basis[polar.basis].name} = {prefix}a_{basis[ab].name}")

@dataclass
class Reverse(MultiVector):
  a: MultiVector

  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    return self.a.__multivector_mask__(layout)


  def __multivector_code__(self, layout: Layout, args: list[str], locals: dict, lines: list[str], prefix: str) -> str:
    self.a.__multivector_code__(layout, args, locals, lines, prefix + "a")
    for ab in self.a.__multivector_mask__(layout):
      if basis[ab].grade % 4 < 2:
        lines.append(f"{prefix}_{basis[ab].name} = {prefix}a_{basis[ab].name}")
      else:
        lines.append(f"{prefix}_{basis[ab].name} = -{prefix}a_{basis[ab].name}")


@dataclass
class Involute(MultiVector):
  a: MultiVector


  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    return self.a.__multivector_mask__(layout)


  def __multivector_code__(self, layout: Layout, args: list[str], locals: dict, lines: list[str], prefix: str) -> str:
    self.a.__multivector_code__(layout, args, locals, lines, prefix + "a")
    for ab in self.a.__multivector_mask__(layout):
      if basis[ab].grade % 2 < 1:
        lines.append(f"{prefix}_{basis[ab].name} = {prefix}a_{basis[ab].name}")
      else:
        lines.append(f"{prefix}_{basis[ab].name} = -{prefix}a_{basis[ab].name}")

@dataclass
class Conjugate(MultiVector):
  a: MultiVector


  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    return self.a.__multivector_mask__(layout)


  def __multivector_code__(self, layout: Layout, args: list[str], locals: dict, lines: list[str], prefix: str) -> str:
    self.a.__multivector_code__(layout, args, locals, lines, prefix + "a")
    for ab in self.a.__multivector_mask__(layout):
      if (basis[ab].grade + 1) % 4 < 2:
        lines.append(f"{prefix}_{basis[ab].name} = {prefix}a_{basis[ab].name}")
      else:
        lines.append(f"{prefix}_{basis[ab].name} = -{prefix}a_{basis[ab].name}")

@dataclass
class Add(MultiVector):
  a: MultiVector
  b: MultiVector


  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    a = set(self.a.__multivector_mask__(layout))
    b = set(self.b.__multivector_mask__(layout))
    return a.union(b)


  def __multivector_code__(self, layout: Layout, args: list[str], locals: dict, lines: list[str], prefix: str) -> str:
    self.a.__multivector_code__(layout, args, locals, lines, prefix + "a")
    self.b.__multivector_code__(layout, args, locals, lines, prefix + "b")
    for ab in self.__multivector_mask__():
      lines.append(f"{prefix}_{basis[ab].name} = {prefix}a_{basis[ab].name} + {prefix}b_{basis[ab].name}")


@dataclass
class Subtract(MultiVector):
  a: MultiVector
  b: MultiVector


  def __multivector_mask__(self, layout: Layout) -> Iterable[int]:
    a = set(self.a.__multivector_mask__(layout))
    b = set(self.b.__multivector_mask__(layout))
    return a.union(b)


  def __multivector_code__(self, layout: Layout, args: list[str], locals: dict, lines: list[str], prefix: str) -> str:
    self.a.__multivector_code__(layout, args, locals, lines, prefix + "a")
    self.b.__multivector_code__(layout, args, locals, lines, prefix + "b")
    for ab in self.__multivector_mask__():
      lines.append(f"{prefix}_{basis[ab].name} = {prefix}a_{basis[ab].name} - {prefix}b_{basis[ab].name}")


@dataclass
class GeometricProduct(Binary):
  def __multivector_mask__(self, layout: Layout) -> set[int]:
    a_mask = set(self.a.__multivector_mask__(layout))
    b_mask = set(self.b.__multivector_mask__(layout))
    r = set([])
    for cb in range(layout.num_basis):
      for ab in a_mask:
        for bb in b_mask:
          m_basis, m_sign = layout.mul_basis_sign(ab, bb)
          if m_basis == cb and m_sign != 0:
            r.add(cb)
    return r


  def __multivector_code__(self, layout: Layout, args: list[str], locals: dict, lines: list[str], prefix: str) -> str:
    a_mask = set(self.a.__multivector_mask__(layout))
    b_mask = set(self.b.__multivector_mask__(layout))
    self.a.__multivector_code__(layout, args, locals, lines, prefix + "a")
    self.b.__multivector_code__(layout, args, locals, lines, prefix + "b")
    for cb in range(layout.num_basis):
      line = f"{prefix}_{layout.basis[cb].name} ="
      plus = " "
      for ab in a_mask:
        for bb in b_mask:
          r_basis, r_sign = layout.mul_basis_sign(ab, bb)
          if r_basis == cb:
            if r_sign < 0:
              line = line + f"-({prefix}a_{layout.basis[ab].name}*{prefix}b_{layout.basis[bb].name})"
              plus = "+"
            elif r_sign > 0:
              line = line + f"{plus}({prefix}a_{layout.basis[ab].name}*{prefix}b_{layout.basis[bb].name})"
              plus = "+"
      if not line.endswith("="):
        lines.append(line)


@dataclass
class OuterProduct(MultiVector):
  a: MultiVector
  b: MultiVector


  def __multivector_mask__(self, layout: Layout) -> set[int]:
    a_mask = set(self.a.__multivector_mask__(layout))
    b_mask = set(self.b.__multivector_mask__(layout))
    r = set([])
    for cb in range(num_basis):
      for ab in a_mask:
        for bb in b_mask:
          if basis[ab].grade + basis[bb].grade != basis[cb].grade:
            continue
          if mul_basis_ref(BasisRef(ab, 1), BasisRef(bb, 1)).basis == cb:
            r.add(cb)
    return r


  def __multivector_code__(self, layout: Layout, args: list[str], locals: dict, lines: list[str], prefix: str) -> str:
    a_mask = set(self.a.__multivector_mask__(layout))
    b_mask = set(self.b.__multivector_mask__(layout))
    self.a.__multivector_code__(layout, args, locals, lines, prefix + "a")
    self.b.__multivector_code__(layout, args, locals, lines, prefix + "b")
    for cb in range(num_basis):
      line = f"{prefix}_{basis[cb].name} ="
      plus = " "
      for ab in a_mask:
        for bb in b_mask:
          if basis[ab].grade + basis[bb].grade != basis[cb].grade:
            continue
          r = mul_basis_ref(BasisRef(ab, 1), BasisRef(bb, 1))
          if r.basis == cb:
            if r.sign < 0:
              line = line + f"-({prefix}a_{basis[ab].name}*{prefix}b_{basis[bb].name})"
              plus = "+"
            elif r.sign > 0:
              line = line + f"{plus}({prefix}a_{basis[ab].name}*{prefix}b_{basis[bb].name})"
              plus = "+"
      if not line.endswith("="):
        lines.append(line)


@dataclass
class RegressiveProduct(MultiVector):
  a: MultiVector
  b: MultiVector


  def __multivector_mask__(self, layout: Layout) -> set[int]:
    a_mask = set(self.a.__multivector_mask__(layout))
    b_mask = set(self.b.__multivector_mask__(layout))
    r = set([])
    for cb in range(num_basis):
      for ab in a_mask:
        for bb in b_mask:
          if basis[ab].grade + basis[bb].grade != basis[cb].grade:
            continue
          if mul_basis_ref(BasisRef(ab, 1), BasisRef(bb, 1)).basis == cb:
            r.add(cb)
    return r


  def __multivector_code__(self, layout: Layout, args: list[str], locals: dict, lines: list[str], prefix: str) -> str:
    a_mask = set(self.a.__multivector_mask__(layout))
    b_mask = set(self.b.__multivector_mask__(layout))
    self.a.__multivector_code__(layout, args, locals, lines, prefix + "a")
    self.b.__multivector_code__(layout, args, locals, lines, prefix + "b")
    for cb in range(num_basis):
      line = f"{prefix}_{basis[cb].name} ="
      plus = " "
      for ab in a_mask:
        for bb in b_mask:
          if basis[ab].grade + basis[bb].grade != basis[cb].grade:
            continue
          r = mul_basis_ref(BasisRef(ab, 1), BasisRef(bb, 1))
          if r.basis == cb:
            if r.sign < 0:
              line = line + f"-({prefix}a_{basis[ab].name}*{prefix}b_{basis[bb].name})"
              plus = "+"
            elif r.sign > 0:
              line = line + f"{plus}({prefix}a_{basis[ab].name}*{prefix}b_{basis[bb].name})"
              plus = "+"
      if not line.endswith("="):
        lines.append(line)


@dataclass
class InnerProduct(MultiVector):
  a: MultiVector
  b: MultiVector


  def __multivector_mask__(self, layout: Layout) -> set[int]:
    a_mask = set(self.a.__multivector_mask__(layout))
    b_mask = set(self.b.__multivector_mask__(layout))
    r = set([])
    for cb in range(num_basis):
      for ab in a_mask:
        for bb in b_mask:
          if abs(basis[ab].grade - basis[bb].grade) != basis[cb].grade:
            continue
          if mul_basis_ref(BasisRef(ab, 1), BasisRef(bb, 1)).basis == cb:
            r.add(cb)
    return r


  def __multivector_code__(self, layout: Layout, args: list[str], locals: dict, lines: list[str], prefix: str) -> str:
    a_mask = set(self.a.__multivector_mask__(layout))
    b_mask = set(self.b.__multivector_mask__(layout))
    self.a.__multivector_code__(layout, args, locals, lines, prefix + "a")
    self.b.__multivector_code__(layout, args, locals, lines, prefix + "b")
    for cb in range(num_basis):
      line = f"{prefix}_{basis[cb].name} ="
      plus = " "
      for ab in a_mask:
        for bb in b_mask:
          if abs(basis[ab].grade - basis[bb].grade) != basis[cb].grade:
            continue
          r = mul_basis_ref(BasisRef(ab, 1), BasisRef(bb, 1))
          if r.basis == cb:
            if r.sign < 0:
              line = line + f"-({prefix}a_{basis[ab].name}*{prefix}b_{basis[bb].name})"
              plus = "+"
            elif r.sign > 0:
              line = line + f"{plus}({prefix}a_{basis[ab].name}*{prefix}b_{basis[bb].name})"
              plus = "+"
      if not line.endswith("="):
        lines.append(line)


def generate(p: int = 0, q: int = 0, r: int = 0, prefix: str = ""):
  layout = Layout(p, q, r)

  ns = {}

  fn_locals = {
    '_nparray': np.array,
    '_npdtype': np.float32,
    '_npmatmul': np.matmul,
    '__layout__': layout
  }

  for i in range(1 << layout.num_grades):
    s = f"{prefix}MultiVector"
    for j in range(layout.num_grades):
      s += '1' if (i & (1 << j)) != 0 else '0'

    grade_ns = {}

    fields = []
    mask_lines = []
    code_lines = []

    for j in range(layout.num_basis):
      if (i & (1 << layout.basis[j].grade)) != 0:
        fields.append((layout.basis[j].name, np.float32))
        mask_lines.append(f"if self.{layout.basis[j].name} != 0:")
        mask_lines.append(f"  yield {j}")
        code_lines.append(f"lines.append(f'{{prefix}}_{layout.basis[j].name} = {{mykey}}.{layout.basis[j].name}')")

    if len(mask_lines) == 0:
      mask_lines = ["pass"]
    if len(code_lines) == 0:
      code_lines = ["pass"]
    else:
      code_lines = ["mykey = f'__MultiVector{id(self)}__'", "locals[mykey] = self"] + code_lines

    grade_ns["__multivector_layout__"] = _create_fn("__multivector_layout__", ("self",), ["return __layout__"], locals=fn_locals) 

    grade_ns["__multivector_mask__"] = _create_fn("__multivector_mask__", ("self", "layout"), mask_lines, locals=fn_locals)
    
    grade_ns["__multivector_code__"] = _create_fn("__multivector_code__", ("self", "layout", "args", "locals", "lines", "prefix:str"), code_lines, locals=fn_locals)

    grade_ns["__multivector__"] = _create_fn("__multivector__", ("self",), [
      "return self"
    ])

    ns[s] = make_record(s, fields, bases=(MultiVector,), namespace=grade_ns)
    layout.graded_types.append(ns[s])

  ## for basis_i in range(num_basis):
  ##   x = np.zeros(num_basis)
  ##   x[basis_i] = 1
  ##   ns[basis[basis_i].name] = MultiVector.__ga_classify__(x)

  x = ns[f"MultiVector0100"](0, 1, 2)

  self_geometric_product = (Vector(0) * Vector(0))

  print((x * x)())
  print(self_geometric_product(x))
  # should be the same, but self_geometric_product does not need to be compiled again to work

  # optimize a function that returns a MultiVector and takes one or more MultiVectors or numbers
  # (x >> y).execute()

  return ns

