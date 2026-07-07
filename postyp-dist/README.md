# postyp

The [Post-Py](https://post-py.org/) type vocabulary: scalar dtypes
(`Float64`, `Int64`, `Bool`, …), `Array` with `Shape` and layout
qualifiers, and `DataFrame`/`Series` annotations.

`postyp` is the canonical, compiler-independent source of Post-Py
type metadata (spec §10). Conforming compilers introspect this module
rather than duplicating dtype definitions; POST source files import
their annotations from it:

```python
from postyp import Array, Float64, Shape

def det3(m: Array[Float64, Shape[3, 3]]) -> Float64: ...
```

The reference compiler is distributed separately as
[`post-py`](https://pypi.org/project/post-py/) (import name
`post_py`), which depends on this package. Development happens in
[openteams-ai/postpython](https://github.com/openteams-ai/postpython).
