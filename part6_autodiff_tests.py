"""Small deterministic tests for Part 6 - Automatic Differentiation.

Operation tests capture the forward/VJP callbacks passed to Tensor.build, so
individual Tensor methods can be tested while the class is being defined.
LLM tests use the student's completed Tensor implementation. Expected results
are literals; no reference autodiff engine or LLM is distributed here.
"""

from contextlib import contextmanager
from types import SimpleNamespace, MethodType
from unittest.mock import Mock, patch

import mugrade
import numpy as np


@contextmanager
def _globals(function, **updates):
    namespace = function.__globals__
    missing = object()
    previous = {key: namespace.get(key, missing) for key in updates}
    namespace.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                namespace.pop(key, None)
            else:
                namespace[key] = value


def _array(value, dtype="float64"):
    return np.asarray(value, dtype=dtype)


def _host(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


def _close(actual, expected):
    return np.allclose(_host(actual), expected, atol=2e-6, rtol=2e-5)


def _operation(function, values, upstream, *extra):
    """Record the student's callbacks, then exercise them on concrete arrays."""
    marker = object()
    build = Mock(return_value=marker)
    with _globals(function, np=np):
        with _globals(function, Tensor=SimpleNamespace(build=build)):
            result = function(*values, *extra)
        assert result is marker, "Return the result of Tensor.build."
        assert build.call_count == 1
        forward, vjp, *parents = build.call_args.args
        out = forward(*parents)
        gradients = vjp(upstream, out, *parents)
    return np.asarray(out), tuple(np.asarray(g) for g in gradients)


def test_Tensor_init(init):
    value = _array([[1., -2.], [3., 4.]])
    obj = SimpleNamespace()
    assert init(obj, value) is None
    assert obj.value is value and obj.grad is None
    assert obj.parents is None and obj.vjp is None
    assert obj.num_children == 0 and obj.requires_grad is False
    parents, vjp = (object(),), Mock()
    other = SimpleNamespace()
    init(other, value, vjp=vjp, parents=parents, requires_grad=True)
    assert other.parents is parents and other.vjp is vjp
    assert other.requires_grad is True and other.num_children == 0


def submit_Tensor_init(init):
    obj = SimpleNamespace()
    value = _array([3., -4., 0.])
    parents, vjp = (object(), object()), Mock()
    init(obj, value, vjp=vjp, parents=parents, requires_grad=True)
    mugrade.submit(obj.value)
    mugrade.submit([obj.grad is None, obj.num_children, obj.requires_grad])
    mugrade.submit([obj.value is value, obj.parents is parents, obj.vjp is vjp])
    default = SimpleNamespace()
    init(default, _array(7.))
    mugrade.submit([default.parents is None, default.vjp is None, default.requires_grad])


def _build_case(build, tracked, repeated=False, offset=0):
    # This empty type supplies isinstance identity only. Its constructor is a
    # recorder, not an implementation of Tensor.__init__.
    class Token:
        pass
    a, b = object.__new__(Token), object.__new__(Token)
    a.value, b.value = _array([1.+offset, -2.]), _array([3., 4.+offset])
    a.requires_grad, b.requires_grad = tracked
    a.num_children, b.num_children = 2, 5
    inputs = (a, a) if repeated else (a, b)
    forward = Mock(return_value=_array([8.+offset, -6.]))
    vjp = Mock()
    with patch.object(Token, "__init__", Mock(return_value=None)) as constructor:
        with _globals(build, Tensor=Token):
            result = build(forward, vjp, *inputs)
        args, kwargs = constructor.call_args
    return result, Token, inputs, forward, vjp, args, kwargs


def test_build(build):
    r, cls, inputs, f, vjp, args, kw = _build_case(build, (True, False))
    assert isinstance(r, cls) and _close(args[0], [8., -6.])
    assert f.call_args.args[0] is inputs[0].value and f.call_args.args[1] is inputs[1].value
    assert kw["vjp"] is vjp and kw["parents"] == inputs and kw["requires_grad"]
    assert [p.num_children for p in inputs] == [3, 6]
    _, _, inputs, _, _, _, kw = _build_case(build, (False, False))
    assert not kw.get("requires_grad", False) and kw.get("parents") is None
    assert [p.num_children for p in inputs] == [2, 5]
    _, _, inputs, _, _, _, _ = _build_case(build, (True, False), repeated=True)
    assert inputs[0].num_children == 4
    forward = Mock(return_value=_array([7., 8.]))
    with _globals(build, Tensor=SimpleNamespace):
        assert _close(build(forward, Mock(), _array([2., 3.])), [7., 8.])
    assert forward.call_count == 1


def submit_build(build):
    r, cls, inputs, f, vjp, args, kw = _build_case(build, (False, True), offset=3)
    mugrade.submit(args[0])
    mugrade.submit([isinstance(r, cls), kw["vjp"] is vjp, kw["parents"] == inputs, kw["requires_grad"]])
    mugrade.submit([p.num_children for p in inputs])
    _, _, inputs, _, _, _, kw = _build_case(build, (False, False), offset=-2)
    mugrade.submit([*[p.num_children for p in inputs], kw.get("parents") is None, kw.get("requires_grad", False)])
    with _globals(build, Tensor=SimpleNamespace):
        mugrade.submit(build(lambda x: x, Mock(), _array([5., -3., 2.])))
    _, _, inputs, _, _, _, _ = _build_case(build, (True, True), repeated=True, offset=2)
    mugrade.submit(inputs[0].num_children)


def test___setitem__(setitem):
    obj = SimpleNamespace(value=_array([[1., 2.], [3., 4.]]))
    source = SimpleNamespace(value=_array([-5., 6.]))
    assert setitem(obj, 1, source) is None
    assert _close(obj.value, [[1., 2.], [-5., 6.]])
    source.value[0] = 99
    assert obj.value[1, 0] == -5
    setitem(obj, (slice(None), 0), SimpleNamespace(value=_array([7., 8.])))
    assert _close(obj.value, [[7., 2.], [8., 6.]])


def submit___setitem__(setitem):
    obj = SimpleNamespace(value=np.arange(12.).reshape(3, 4))
    result = setitem(obj, (slice(1, 3), slice(0, 2)), SimpleNamespace(value=_array([[-1., 2.], [4., -3.]])))
    mugrade.submit(obj.value.copy())
    mugrade.submit(result is None)
    setitem(obj, (0, 3), SimpleNamespace(value=_array(8.5)))
    mugrade.submit(obj.value.copy())


def test_shape(shape):
    assert shape(SimpleNamespace(value=np.zeros((2, 3, 4)))) == (2, 3, 4)
    assert shape(SimpleNamespace(value=np.zeros(()))) == ()
    assert shape(SimpleNamespace(value=np.zeros((0, 5)))) == (0, 5)


def submit_shape(shape):
    mugrade.submit(list(shape(SimpleNamespace(value=np.zeros((1, 4, 2))))))
    mugrade.submit(list(shape(SimpleNamespace(value=np.zeros((7,))))))
    mugrade.submit(list(shape(SimpleNamespace(value=np.zeros((3, 0, 2))))))


def test_dtype(dtype):
    assert dtype(SimpleNamespace(value=np.zeros(2, dtype="float32"))) == np.dtype("float32")
    assert dtype(SimpleNamespace(value=np.zeros((), dtype="float64"))) == np.dtype("float64")
    assert dtype(SimpleNamespace(value=np.zeros((0, 3), dtype="int64"))) == np.dtype("int64")


def submit_dtype(dtype):
    mugrade.submit(str(dtype(SimpleNamespace(value=np.zeros(3, dtype="float16")))))
    mugrade.submit(str(dtype(SimpleNamespace(value=np.zeros((2, 2), dtype="int32")))))
    mugrade.submit(str(dtype(SimpleNamespace(value=np.zeros((), dtype="bool")))))


def _backward_case(backward, scale=1.):
    def node(value, parents=None, gradients=None, children=0, tracked=True):
        obj = SimpleNamespace(value=_array(value), parents=parents, grad=None,
                              requires_grad=tracked, num_children=children,
                              vjp=Mock(return_value=gradients))
        obj.backward = MethodType(backward, obj)
        return obj
    leaf = node([2., -1.], children=1)
    shared = node([4., 1.], (leaf,), (_array([14., -6.])*scale,), children=2)
    constant = node([5., 3.], children=1, tracked=False)
    left = node([9., 4.], (shared, constant), (_array([2., 1.])*scale, _array([9., 9.])), children=1)
    right = node([8., 2.], (shared,), (_array([5., 2.])*scale,), children=1)
    root = node(12., (left, right), (_array([1., 2.])*scale, _array([3., 4.])*scale))
    root.grad = None if scale == 1 else _array(scale)
    backward(root)
    return root, left, right, shared, leaf, constant


def test_backward(backward):
    root, left, right, shared, leaf, constant = _backward_case(backward)
    assert _close(root.grad, 1.) and _close(leaf.grad, [14., -6.])
    assert _close(shared.grad, [7., 3.])
    assert shared.vjp.call_count == 1 and _close(shared.vjp.call_args.args[0], [7., 3.])
    assert constant.grad is None and constant.vjp.call_count == 0
    assert [n.num_children for n in (left, right, shared, leaf)] == [0, 0, 0, 0]
    _, _, _, shared, leaf, _ = _backward_case(backward, -2.)
    assert _close(shared.grad, [-14., -6.]) and _close(leaf.grad, [-28., 12.])
    # Repeated parent edges must each contribute, then recurse exactly once.
    parent = SimpleNamespace(value=_array([2., 3.]), grad=_array([1., -1.]),
                             requires_grad=True, num_children=2, backward=Mock())
    root = SimpleNamespace(value=_array(1.), grad=None, parents=(parent, parent),
                           vjp=Mock(return_value=(_array([2., 4.]), _array([3., 5.]))))
    backward(root)
    assert _close(parent.grad, [6., 8.]) and parent.backward.call_count == 1


def submit_backward(backward):
    root, left, right, shared, leaf, constant = _backward_case(backward, 3.)
    mugrade.submit([root.grad, left.grad, right.grad])
    mugrade.submit([shared.grad, leaf.grad])
    mugrade.submit([shared.vjp.call_count, constant.grad is None, *[n.num_children for n in (left, right, shared, leaf)]])
    mugrade.submit(shared.vjp.call_args.args[0])


def test___add__(function):
    out, gradients = _operation(function, [_array([[1., -2.], [0., 4.]]), _array([[3., 5.], [-1., 2.]])], _array([[2., -3.], [4., 1.]]))
    assert _close(out, [[4.0, 3.0], [-1.0, 6.0]])
    assert out.shape == (2, 2) and str(out.dtype) == 'float64'
    assert len(gradients) == 2
    assert _close(gradients[0], [[2.0, -3.0], [4.0, 1.0]])
    assert gradients[0].shape == (2, 2) and str(gradients[0].dtype) == 'float64'
    assert _close(gradients[1], [[2.0, -3.0], [4.0, 1.0]])
    assert gradients[1].shape == (2, 2) and str(gradients[1].dtype) == 'float64'
    out, gradients = _operation(function, [_array(-2.), _array(0.5)], _array(-3.))
    assert _close(out, -1.5)
    assert out.shape == () and str(out.dtype) == 'float64'
    assert len(gradients) == 2
    assert _close(gradients[0], -3.0)
    assert gradients[0].shape == () and str(gradients[0].dtype) == 'float64'
    assert _close(gradients[1], -3.0)
    assert gradients[1].shape == () and str(gradients[1].dtype) == 'float64'


def submit___add__(function):
    out, gradients = _operation(function, [_array([2., -3., 4.]), _array([-1., 5., 0.5])], _array([-2., 1., 3.]))
    mugrade.submit(out)
    mugrade.submit(list(gradients))
    mugrade.submit([list(out.shape), str(out.dtype), [[list(g.shape), str(g.dtype)] for g in gradients]])


def test___mul__(function):
    out, gradients = _operation(function, [_array([[1., -2.], [0., 4.]]), _array([[3., 5.], [-1., 2.]])], _array([[2., -3.], [4., 1.]]))
    assert _close(out, [[3.0, -10.0], [-0.0, 8.0]])
    assert out.shape == (2, 2) and str(out.dtype) == 'float64'
    assert len(gradients) == 2
    assert _close(gradients[0], [[6.0, -15.0], [-4.0, 2.0]])
    assert gradients[0].shape == (2, 2) and str(gradients[0].dtype) == 'float64'
    assert _close(gradients[1], [[2.0, 6.0], [0.0, 4.0]])
    assert gradients[1].shape == (2, 2) and str(gradients[1].dtype) == 'float64'
    out, gradients = _operation(function, [_array([-2., 0., 3.]), _array([0., 5., -4.])], _array([2., -1., 0.5]))
    assert _close(out, [-0.0, 0.0, -12.0])
    assert out.shape == (3,) and str(out.dtype) == 'float64'
    assert len(gradients) == 2
    assert _close(gradients[0], [0.0, -5.0, -2.0])
    assert gradients[0].shape == (3,) and str(gradients[0].dtype) == 'float64'
    assert _close(gradients[1], [-4.0, -0.0, 1.5])
    assert gradients[1].shape == (3,) and str(gradients[1].dtype) == 'float64'


def submit___mul__(function):
    out, gradients = _operation(function, [_array([2., -3., 4.]), _array([-1., 5., 0.5])], _array([-2., 1., 3.]))
    mugrade.submit(out)
    mugrade.submit(list(gradients))
    mugrade.submit([list(out.shape), str(out.dtype), [[list(g.shape), str(g.dtype)] for g in gradients]])


def test___matmul__(function):
    out, gradients = _operation(function, [_array([[1., -2., 3.], [0., 4., 1.]]), _array([[2., 1.], [-1., 3.], [4., -2.]])], _array([[2., -1.], [3., 4.]]))
    assert _close(out, [[16.0, -11.0], [0.0, 10.0]])
    assert out.shape == (2, 2) and str(out.dtype) == 'float64'
    assert len(gradients) == 2
    assert _close(gradients[0], [[3.0, -5.0, 10.0], [10.0, 9.0, 4.0]])
    assert gradients[0].shape == (2, 3) and str(gradients[0].dtype) == 'float64'
    assert _close(gradients[1], [[2.0, -1.0], [8.0, 18.0], [9.0, 1.0]])
    assert gradients[1].shape == (3, 2) and str(gradients[1].dtype) == 'float64'
    out, gradients = _operation(function, [np.arange(12.).reshape(2, 2, 3) / 4, _array([[1., -1.], [2., 0.], [-2., 3.]])], _array([[[1., 2.], [3., -1.]], [[2., -3.], [1., 4.]]]))
    assert _close(out, [[[-0.5, 1.5], [0.25, 3.0]], [[1.0, 4.5], [1.75, 6.0]]])
    assert out.shape == (2, 2, 2) and str(out.dtype) == 'float64'
    assert len(gradients) == 2
    assert _close(gradients[0], [[[-1.0, 2.0, 4.0], [4.0, 6.0, -9.0]], [[5.0, 4.0, -13.0], [-3.0, 2.0, 10.0]]])
    assert gradients[0].shape == (2, 2, 3) and str(gradients[0].dtype) == 'float64'
    assert _close(gradients[1], [[7.5, 3.75], [9.25, 4.25], [11.0, 4.75]])
    assert gradients[1].shape == (3, 2) and str(gradients[1].dtype) == 'float64'
    out, gradients = _operation(function, [_array([[1., 2., -1.], [-2., 0., 3.]]), np.arange(12.).reshape(2, 3, 2) / 3], _array([[[1., -2.], [3., 1.]], [[4., 2.], [-1., 3.]]]))
    assert _close(out, [[[0.0, 0.6666666666666667], [4.0, 4.333333333333334]], [[3.9999999999999996, 4.666666666666668], [6.0, 6.333333333333332]]])
    assert out.shape == (2, 2, 2) and str(out.dtype) == 'float64'
    assert len(gradients) == 2
    assert _close(gradients[0], [[12.000000000000002, 15.33333333333333, 18.666666666666668], [5.333333333333333, 9.333333333333334, 13.333333333333332]])
    assert gradients[0].shape == (2, 3) and str(gradients[0].dtype) == 'float64'
    assert _close(gradients[1], [[[-5.0, -4.0], [2.0, -4.0], [8.0, 5.0]], [[6.0, -4.0], [8.0, 4.0], [-7.0, 7.0]]])
    assert gradients[1].shape == (2, 3, 2) and str(gradients[1].dtype) == 'float64'


def submit___matmul__(function):
    out, gradients = _operation(function, [np.arange(18.).reshape(3, 2, 3) / 5 - 1, _array([[2., -1.], [0., 3.], [-2., 1.]])], np.arange(12.).reshape(3, 2, 2) / 3 - 2)
    mugrade.submit(out)
    mugrade.submit(list(gradients))
    mugrade.submit([list(out.shape), str(out.dtype), [[list(g.shape), str(g.dtype)] for g in gradients]])


def test___getitem__(function):
    out, gradients = _operation(function, [np.arange(12.).reshape(4, 3)], _array([[1., 2., 3.], [4., -1., 2.], [3., 0., -2.]]), np.array([2, 0, 2]))
    assert _close(out, [[6.0, 7.0, 8.0], [0.0, 1.0, 2.0], [6.0, 7.0, 8.0]])
    assert out.shape == (3, 3) and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [[4.0, -1.0, 2.0], [0.0, 0.0, 0.0], [4.0, 2.0, 1.0], [0.0, 0.0, 0.0]])
    assert gradients[0].shape == (4, 3) and str(gradients[0].dtype) == 'float64'
    out, gradients = _operation(function, [np.arange(12.).reshape(3, 4)], _array([[1., -2.], [3., 4.]]), (slice(1, 3), slice(0, 4, 2)))
    assert _close(out, [[4.0, 6.0], [8.0, 10.0]])
    assert out.shape == (2, 2) and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, -2.0, 0.0], [3.0, 0.0, 4.0, 0.0]])
    assert gradients[0].shape == (3, 4) and str(gradients[0].dtype) == 'float64'
    out, gradients = _operation(function, [np.arange(6.).reshape(2, 3)], _array([2., -1., 4.]), -1)
    assert _close(out, [3.0, 4.0, 5.0])
    assert out.shape == (3,) and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [[0.0, 0.0, 0.0], [2.0, -1.0, 4.0]])
    assert gradients[0].shape == (2, 3) and str(gradients[0].dtype) == 'float64'


def submit___getitem__(function):
    out, gradients = _operation(function, [np.arange(10.).reshape(5, 2) - 3], _array([[2., -1.], [3., 4.], [-2., 5.], [1., 2.]]), np.array([4, 1, 4, 0]))
    mugrade.submit(out)
    mugrade.submit(list(gradients))
    mugrade.submit([list(out.shape), str(out.dtype), [[list(g.shape), str(g.dtype)] for g in gradients]])


def test_transpose(function):
    out, gradients = _operation(function, [np.arange(12.).reshape(2, 3, 2)], np.arange(12.).reshape(3, 2, 2) - 4, 0, 1)
    assert _close(out, [[[0.0, 1.0], [6.0, 7.0]], [[2.0, 3.0], [8.0, 9.0]], [[4.0, 5.0], [10.0, 11.0]]])
    assert out.shape == (3, 2, 2) and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [[[-4.0, -3.0], [0.0, 1.0], [4.0, 5.0]], [[-2.0, -1.0], [2.0, 3.0], [6.0, 7.0]]])
    assert gradients[0].shape == (2, 3, 2) and str(gradients[0].dtype) == 'float64'
    out, gradients = _operation(function, [np.arange(6.).reshape(2, 3)], _array([[2., 3.], [4., -1.], [-2., 0.]]), -1, -2)
    assert _close(out, [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])
    assert out.shape == (3, 2) and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [[2.0, 4.0, -2.0], [3.0, -1.0, 0.0]])
    assert gradients[0].shape == (2, 3) and str(gradients[0].dtype) == 'float64'


def submit_transpose(function):
    out, gradients = _operation(function, [np.arange(24.).reshape(2, 3, 4) / 3], np.arange(24.).reshape(4, 3, 2) / 4 - 2, 0, 2)
    mugrade.submit(out)
    mugrade.submit(list(gradients))
    mugrade.submit([list(out.shape), str(out.dtype), [[list(g.shape), str(g.dtype)] for g in gradients]])


def test_reshape(function):
    out, gradients = _operation(function, [np.arange(12.).reshape(2, 2, 3)], np.arange(12.).reshape(3, 4) - 5, 3, -1)
    assert _close(out, [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]])
    assert out.shape == (3, 4) and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [[[-5.0, -4.0, -3.0], [-2.0, -1.0, 0.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    assert gradients[0].shape == (2, 2, 3) and str(gradients[0].dtype) == 'float64'
    out, gradients = _operation(function, [np.arange(6.).reshape(2, 3)], _array([2., -1., 0., 4., 5., -2.]), -1)
    assert _close(out, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert out.shape == (6,) and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [[2.0, -1.0, 0.0], [4.0, 5.0, -2.0]])
    assert gradients[0].shape == (2, 3) and str(gradients[0].dtype) == 'float64'


def submit_reshape(function):
    out, gradients = _operation(function, [np.arange(8.).reshape(2, 4) - 3], np.arange(8.).reshape(2, 2, 2) / 3, 2, 2, -1)
    mugrade.submit(out)
    mugrade.submit(list(gradients))
    mugrade.submit([list(out.shape), str(out.dtype), [[list(g.shape), str(g.dtype)] for g in gradients]])


def test_flip(function):
    out, gradients = _operation(function, [np.arange(12.).reshape(2, 3, 2)], np.arange(12.).reshape(2, 3, 2) - 3, 1)
    assert _close(out, [[[4.0, 5.0], [2.0, 3.0], [0.0, 1.0]], [[10.0, 11.0], [8.0, 9.0], [6.0, 7.0]]])
    assert out.shape == (2, 3, 2) and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [[[1.0, 2.0], [-1.0, 0.0], [-3.0, -2.0]], [[7.0, 8.0], [5.0, 6.0], [3.0, 4.0]]])
    assert gradients[0].shape == (2, 3, 2) and str(gradients[0].dtype) == 'float64'
    out, gradients = _operation(function, [_array([[1., 2., 3.], [-1., 0., 4.]])], _array([[2., 3., -1.], [4., 0., -2.]]), -1)
    assert _close(out, [[3.0, 2.0, 1.0], [4.0, 0.0, -1.0]])
    assert out.shape == (2, 3) and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [[-1.0, 3.0, 2.0], [-2.0, 0.0, 4.0]])
    assert gradients[0].shape == (2, 3) and str(gradients[0].dtype) == 'float64'


def submit_flip(function):
    out, gradients = _operation(function, [np.arange(8.).reshape(2, 2, 2) - 2], np.arange(8.).reshape(2, 2, 2) / 3, 0)
    mugrade.submit(out)
    mugrade.submit(list(gradients))
    mugrade.submit([list(out.shape), str(out.dtype), [[list(g.shape), str(g.dtype)] for g in gradients]])


def test_to(function):
    out, gradients = _operation(function, [_array([[1.25, -2.5], [0.5, 4.]], "float64")], _array([[2., -1.], [3., 4.]], "float32"), "float32")
    assert _close(out, [[1.25, -2.5], [0.5, 4.0]])
    assert out.shape == (2, 2) and str(out.dtype) == 'float32'
    assert len(gradients) == 1
    assert _close(gradients[0], [[2.0, -1.0], [3.0, 4.0]])
    assert gradients[0].shape == (2, 2) and str(gradients[0].dtype) == 'float64'
    out, gradients = _operation(function, [_array([2., -3., 0.], "float32")], _array([0.5, -2., 1.25], "float64"), "float64")
    assert _close(out, [2.0, -3.0, 0.0])
    assert out.shape == (3,) and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [0.5, -2.0, 1.25])
    assert gradients[0].shape == (3,) and str(gradients[0].dtype) == 'float32'


def submit_to(function):
    out, gradients = _operation(function, [_array([3.5, -1.25, 0.], "float32")], _array([2., -3., 1.], "float16"), "float16")
    mugrade.submit(out)
    mugrade.submit(list(gradients))
    mugrade.submit([list(out.shape), str(out.dtype), [[list(g.shape), str(g.dtype)] for g in gradients]])


def test_softmax(function):
    out, gradients = _operation(function, [_array([[1000., 1001., 999.], [-2., 0., 1.]])], _array([[1., -2., 3.], [4., -1., 2.]]))
    assert _close(out, [[0.24472847105479764, 0.6652409557748218, 0.09003057317038046], [0.03511902695933972, 0.2594964603424191, 0.7053845126982411]])
    assert out.shape == (2, 3) and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [[0.4443441169290646, -0.787870328693508, 0.3435262117644432], [0.09511115136939374, -0.5947006686097603, 0.4995895172403668]])
    assert gradients[0].shape == (2, 3) and str(gradients[0].dtype) == 'float64'
    out, gradients = _operation(function, [_array([[[0., 0., 0.]], [[2., -1., 0.5]]])], _array([[[2., -1., 3.]], [[-2., 4., 1.]]]))
    assert _close(out, [[[0.3333333333333333, 0.3333333333333333, 0.3333333333333333]], [[0.7855970345892759, 0.039112573270687456, 0.1752903921400367]]])
    assert out.shape == (2, 1, 3) and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [[[0.22222222222222224, -0.7777777777777777, 0.5555555555555556]], [[-0.5974831662312593, 0.2049285043783212, 0.39255466185293825]]])
    assert gradients[0].shape == (2, 1, 3) and str(gradients[0].dtype) == 'float64'


def submit_softmax(function):
    out, gradients = _operation(function, [_array([[[8., 7., 9., 6.], [-1., 2., 0., 1.]]])], _array([[[2., -1., 3., 0.], [4., 1., -2., 3.]]]))
    mugrade.submit(out)
    mugrade.submit(list(gradients))
    mugrade.submit([list(out.shape), str(out.dtype), [[list(g.shape), str(g.dtype)] for g in gradients]])


def test_cross_entropy_loss(function):
    out, gradients = _operation(function, [_array([[1000., 1001., 999.], [-2., 0., 1.]])], _array(2.5), np.array([1, 0]))
    assert _close(out, 1.8783090906062836)
    assert out.shape == () and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [[0.30591058881849703, -0.41844880528147277, 0.11253821646297557], [-1.2061012163008253, 0.32437057542802383, 0.8817306408728014]])
    assert gradients[0].shape == (2, 3) and str(gradients[0].dtype) == 'float64'
    out, gradients = _operation(function, [_array([[[0., 0., 0.], [2., -1., 0.5]], [[-2., 1., 0.], [4., 2., -1.]]])], _array(-1.5), np.array([[2, 1], [0, 0]]))
    assert _close(out, 1.9554452589552573)
    assert out.shape == () and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [[[-0.125, -0.125, 0.25], [-0.29459888797097844, 0.3603327850234922, -0.06573389705251376]], [[0.3618303648902476, -0.26451919226184045, -0.09731117262840717], [0.04664977685134214, -0.044437370450628594, -0.0022124064007135427]]])
    assert gradients[0].shape == (2, 2, 3) and str(gradients[0].dtype) == 'float64'


def submit_cross_entropy_loss(function):
    out, gradients = _operation(function, [_array([[[8., 7., 9., 6.], [-1., 2., 0., 1.]]])], _array(3.), np.array([[2, 0]]))
    mugrade.submit(out)
    mugrade.submit(list(gradients))
    mugrade.submit([list(out.shape), str(out.dtype), [[list(g.shape), str(g.dtype)] for g in gradients]])


def test_silu(function):
    out, gradients = _operation(function, [_array([[-4., -1., 0.], [0.5, 2., 5.]])], _array([[2., -1., 3.], [4., 0.5, -2.]]))
    assert _close(out, [[-0.07194483984836623, -0.2689414213699951, 0.0], [0.3112296656009273, 1.7615941559557646, 4.966535745378576]])
    assert out.shape == (2, 3) and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [[-0.10532922978214582, -0.07232948812851327, 1.5], [2.9598447492106073, 0.5453921243924478, -2.0530948648593315]])
    assert gradients[0].shape == (2, 3) and str(gradients[0].dtype) == 'float64'
    out, gradients = _operation(function, [_array([0., 1., -2.], "float32")], _array([-3., 2., 0.5], "float32"))
    assert _close(out, [0.0, 0.7310585975646973, -0.23840586841106415])
    assert out.shape == (3,) and str(out.dtype) == 'float32'
    assert len(gradients) == 1
    assert _close(gradients[0], [-1.5, 1.8553410768508911, -0.045392125844955444])
    assert gradients[0].shape == (3,) and str(gradients[0].dtype) == 'float32'


def submit_silu(function):
    out, gradients = _operation(function, [_array([[-3., 0.25, 1.5, 4.]])], _array([[3., -2., 1., 0.5]]))
    mugrade.submit(out)
    mugrade.submit(list(gradients))
    mugrade.submit([list(out.shape), str(out.dtype), [[list(g.shape), str(g.dtype)] for g in gradients]])


def test_rms_norm(function):
    out, gradients = _operation(function, [_array([[1., 2., -2.], [-3., 1., 2.]], "float32")], _array([[2., -1., 3.], [4., 1., -2.]], "float32"))
    assert _close(out, [[0.5773502588272095, 1.154700517654419, -1.154700517654419], [-1.3887301683425903, 0.4629100561141968, 0.9258201122283936]])
    assert out.shape == (2, 3) and str(out.dtype) == 'float32'
    assert len(gradients) == 1
    assert _close(gradients[0], [[1.5396006107330322, 0.19245004653930664, 0.9622505307197571], [0.3637150824069977, 0.958885133266449, 0.06612997502088547]])
    assert gradients[0].shape == (2, 3) and str(gradients[0].dtype) == 'float32'
    out, gradients = _operation(function, [_array([[[2., -1.], [-2., 4.]]], "float64")], _array([[[1., 3.], [-2., 1.]]], "float64"))
    assert _close(out, [[[1.2649110555648804, -0.6324555277824402], [-0.6324555277824402, 1.2649110555648804]]])
    assert out.shape == (1, 2, 2) and str(out.dtype) == 'float64'
    assert len(gradients) == 1
    assert _close(gradients[0], [[[0.8854377269744873, 1.7708754539489746], [-0.37947332859039307, -0.18973666429519653]]])
    assert gradients[0].shape == (1, 2, 2) and str(gradients[0].dtype) == 'float64'


def submit_rms_norm(function):
    out, gradients = _operation(function, [_array([[[2., -3., 1., 4.], [-1., 2., 3., 1.]]], "float32")], _array([[[1., -2., 3., 0.5], [-3., 1., 2., 4.]]], "float32"))
    mugrade.submit(out)
    mugrade.submit(list(gradients))
    mugrade.submit([list(out.shape), str(out.dtype), [[list(g.shape), str(g.dtype)] for g in gradients]])


def test_self_attn(function):
    out, gradients = _operation(function, [_array([[[[1., 0.], [0.5, -1.]]]]), _array([[[[0., 2.], [1., -1.]]]]), _array([[[[2., -1.], [0., 3.]]]]), _array([[0., -np.inf], [0., 0.]])], _array([[[[2., -1.], [3., 1.]]]]))
    assert _close(out, [[[[2.0, -1.0], [0.15527700813683734, 2.6894459837263254]]]])
    assert out.shape == (1, 1, 2, 2) and str(out.dtype) == 'float64'
    assert len(gradients) == 3
    assert _close(gradients[0], [[[[0.0, 0.0], [-0.10127291755606085, 0.3038187526681826]]]])
    assert gradients[0].shape == (1, 1, 2, 2) and str(gradients[0].dtype) == 'float64'
    assert _close(gradients[1], [[[[0.05063645877803044, -0.10127291755606088], [-0.05063645877803043, 0.10127291755606085]]]])
    assert gradients[1].shape == (1, 1, 2, 2) and str(gradients[1].dtype) == 'float64'
    assert _close(gradients[2], [[[[2.232915512205256, -0.9223614959315813], [2.767084487794744, 0.9223614959315813]]]])
    assert gradients[2].shape == (1, 1, 2, 2) and str(gradients[2].dtype) == 'float64'
    out, gradients = _operation(function, [_array([[[[1., -1.], [2., 0.]], [[0., 1.], [-1., 2.]]]]), _array([[[[0., 2.], [1., -1.]], [[2., 1.], [0., -1.]]]]), _array([[[[2., -1.], [0., 3.]], [[1., 2.], [-2., 0.]]]]), _array([[0., -0.5], [0.25, 0.]])], _array([[[[2., -1.], [3., 1.]], [[-1., 4.], [2., -2.]]]]))
    assert _close(out, [[[[0.17759168820935503, 2.6448166235812898], [0.47580500435619044, 2.048389991287619]], [[0.6144753402124141, 1.7429835601416095], [0.5224084393408917, 1.6816056262272612]]]])
    assert out.shape == (1, 2, 2, 2) and str(out.dtype) == 'float64'
    assert len(gradients) == 3
    assert _close(gradients[0], [[[[-0.45770253844062364, 1.3731076153218704], [-0.25640385081763917, 0.7692115524529177]], [[0.7919161597622895, 0.7919161597622896], [0.3785947077164939, 0.3785947077164938]]]])
    assert gradients[0].shape == (1, 2, 2, 2) and str(gradients[0].dtype) == 'float64'
    assert _close(gradients[1], [[[[0.970510240075902, -0.4577025384406234], [-0.970510240075902, 0.45770253844062364]], [[-0.18929735385824695, 0.7745527875976386], [0.1892973538582468, -0.7745527875976386]]]])
    assert gradients[1].shape == (1, 2, 2, 2) and str(gradients[1].dtype) == 'float64'
    assert _close(gradients[2], [[[[0.8912991947436407, 0.14910665807341772], [4.108700805256359, -0.14910665807341772]], [[0.8101138461564564, 1.8043614940559578], [0.18988615384354354, 0.19563850594404247]]]])
    assert gradients[2].shape == (1, 2, 2, 2) and str(gradients[2].dtype) == 'float64'


def submit_self_attn(function):
    out, gradients = _operation(function, [_array([[[[2., -1.], [-0.5, 1.]]]]), _array([[[[1., 0.], [-1., 2.]]]]), _array([[[[-2., 1.], [3., 0.5]]]]), _array([[0., -np.inf], [0., 0.]])], _array([[[[-1., 2.], [0.5, 3.]]]]))
    mugrade.submit(out)
    mugrade.submit(list(gradients))
    mugrade.submit([list(out.shape), str(out.dtype), [[list(g.shape), str(g.dtype)] for g in gradients]])


@contextmanager
def _numpy_backend(function):
    """Use CPU arrays for tiny integration fixtures with the student's Tensor."""
    with _globals(function, np=np):
        yield function.__globals__["Tensor"]


def _integration(function, name, variant=0):
    """Exercise the student's implementation with literal state and recorders."""
    with _numpy_backend(function) as Tensor:
        def tensor(value, grad=False, dtype="float64"):
            return Tensor(_array(value, dtype), requires_grad=grad)
        step = variant * 0.25
        if name == "embedding":
            w = tensor([[1.+step, -2.], [3., 0.5], [-1., 4.-step], [2., -3.]], True)
            indices = np.array([[2, 0], [2, 1]]) if not variant else np.array([[3, 1, 3]])
            out = function(indices, w, "float32")
            inputs = [w]
        elif name == "linear":
            x = tensor([[[1.+step, -2.], [0.5, 3.]]], True, "float32")
            w = tensor([[2., -1., 0.5], [-3., 2.+step, 1.]], True)
            out = function(x, w)
            inputs = [x, w]
        elif name == "rope":
            x = tensor(np.arange(16.).reshape(1, 2, 2, 4) / 4 + step, True, "float32")
            # Hand-selected rotations: identity, quarter-turn, half-turn.
            obj = SimpleNamespace(buffers={
                "rope1": tensor([[1., 1., 1., 1.], [0., 0., 1., 1.], [-1., -1., 0., 0.], [1., 1., -1., -1.]]),
                "rope2": tensor([[0., 0., 0., 0.], [-1., 1., 0., 0.], [0., 0., -1., 1.], [0., 0., 0., 0.]])})
            out = function(obj, x, pos=1 if not variant else 2)
            inputs = [x]
        elif name == "multihead_attn":
            x = tensor([[[1.+step, 2., -1., 0.5], [-2., 1., 3., 1.-step]]], True, "float32")
            params = {"wq_1": tensor(np.eye(4), True),
                      "wk_1": tensor(np.diag([2., -1., 1., 0.5]), True),
                      "wv_1": tensor(np.diag([1., 2., -1., 3.]), True),
                      "wp_1": tensor(_array([[1., 0., 1., 0.], [0., 1., 0., -1.], [0.5, 0., 2., 0.], [0., 2., 0., 1.]]), True)}
            obj = SimpleNamespace(params=params, num_heads=2, head_dim=2,
                                  rope=Mock(side_effect=lambda value, pos=0: value))
            mask = tensor([[0., -np.inf], [0., 0.]])
            out = function(obj, x, 1, mask, pos=variant)
            assert obj.rope.call_count == 2
            assert all(call.kwargs.get("pos") == variant for call in obj.rope.call_args_list)
            inputs = [x] + list(params.values())
        elif name == "mlp":
            x = tensor([[[1.+step, -2.], [0.5, 3.-step]]], True, "float32")
            params = {"w1_2": tensor([[1., -1., 0.5], [2., 0.5, -1.]], True),
                      "w2_2": tensor([[1., -2.], [0.5, 1.+step], [-1., 3.]], True)}
            obj = SimpleNamespace(params=params)
            out = function(obj, x, 2)
            inputs = [x] + list(params.values())
        elif name == "transformer_block":
            x = tensor([[[1.+step, -2.], [3., 1.-step]]], True, "float32")
            attn = tensor([[[0.5, 1.], [-1., 2.]]], True, "float32")
            mlp = tensor([[[2., -1.], [0.25, 0.5]]], True, "float32")
            obj = SimpleNamespace(multihead_attn=Mock(return_value=attn), mlp=Mock(return_value=mlp))
            mask, cache = object(), object()
            out = function(obj, x, 2, mask, pos=3+variant, cache=cache)
            assert obj.multihead_attn.call_count == obj.mlp.call_count == 1
            args, kwargs = obj.multihead_attn.call_args
            assert args[1:] == (2, mask) and kwargs == {"pos": 3+variant, "cache": cache}
            assert obj.mlp.call_args.args[1] == 2
            recorded = [args[0].value.copy(), obj.mlp.call_args.args[0].value.copy()]
            # The scripted branch outputs do not depend on these recorded
            # normalization nodes; inspect residual gradients directly below.
            inputs = [x, attn, mlp]
        elif name == "__call__":
            embedding = tensor([[1., 2.], [-1., 3.], [2.+step, -2.], [0.5, 1.]], True)
            output = tensor([[1., -1., 2.], [0.5, 2., -1.]], True)
            mask = tensor(np.triu(np.full((6, 6), -np.inf), 1))
            blocks = [tensor([[[1., 3.], [-2., 1.]]], True, "float32"),
                      tensor([[[2.+step, -1.], [1., 2.-step]]], True, "float32")]
            obj = SimpleNamespace(params={"embedding": embedding, "output": output},
                                  buffers={"mask": mask}, dtype="float32", num_layers=2,
                                  transformer_block=Mock(side_effect=blocks))
            cache = object()
            out = function(obj, np.array([[2, 0]]), pos=1+variant, cache=cache)
            calls = obj.transformer_block.call_args_list
            assert len(calls) == 2 and [call.args[1] for call in calls] == [0, 1]
            assert all(call.kwargs == {"pos": 1+variant, "cache": cache} for call in calls)
            assert calls[1].args[0] is blocks[0]
            recorded = [calls[0].args[0].value.copy(), calls[0].args[2].value.copy()]
            inputs = [blocks[-1], output]
        else:
            raise AssertionError(name)
        if name == "transformer_block":
            return out.value.copy(), recorded, str(out.dtype)
        upstream = (np.arange(out.value.size).reshape(out.shape) / 3 - 1 + step).astype(out.dtype)
        out.grad = upstream
        out.backward()
        result = (out.value.copy(), [value.grad.copy() for value in inputs], str(out.dtype))
        if name in ("transformer_block", "__call__"):
            result += (recorded,)
        return result


def _llm_init(init, variant=0):
    config = dict(depth=1+variant, aspect_ratio=4, mlp_multiple=2,
                  head_dim=4, dtype="float32" if not variant else "float64",
                  vocab_size=5+variant, seq_len=3+variant, rope_theta=100.)
    obj = SimpleNamespace()
    # Seed and restore the CPU RNG; do not grade particular random draws.
    state = np.random.get_state()
    try:
        np.random.seed(17+variant)
        with _numpy_backend(init):
            result = init(obj, config)
    finally:
        np.random.set_state(state)
    assert result is None
    return obj


def test_LLM_init(init):
    obj = _llm_init(init)
    assert (obj.num_layers, obj.num_heads, obj.head_dim, obj.dtype) == (1, 1, 4, "float32")
    assert {k: v.shape for k, v in obj.params.items()} == {"embedding": (5, 4), "output": (4, 5), "wq_0": (4, 4), "wk_0": (4, 4), "wv_0": (4, 4), "wp_0": (4, 4), "w1_0": (4, 8), "w2_0": (8, 4)}
    assert all(np.isfinite(v.value).all() and np.any(v.value != 0) for v in obj.params.values())
    assert set(obj.buffers) == {"mask", "rope1", "rope2"}
    assert _close(obj.buffers["mask"].value, [[0., -np.inf, -np.inf], [0., 0., -np.inf], [0., 0., 0.]])
    assert _close(obj.buffers["rope1"].value, [[1., 1., 1., 1.], [0.5403023, 0.5403023, 0.9950042, 0.9950042], [-0.41614684, -0.41614684, 0.9800666, 0.9800666]])
    assert _close(obj.buffers["rope2"].value, [[0., 0., 0., 0.], [-0.841471, 0.841471, -0.09983342, 0.09983342], [-0.9092974, 0.9092974, -0.1986693, 0.1986693]])
    assert all(str(v.dtype) == "float32" for v in obj.buffers.values())
    other = _llm_init(init, 1)
    assert (other.num_layers, other.num_heads) == (2, 2) and len(other.params) == 14
    assert other.params["w1_1"].shape == (8, 16) and other.params["w2_1"].shape == (16, 8)
    assert obj.params["embedding"] is not other.params["embedding"]


def submit_LLM_init(init):
    obj = _llm_init(init, 2)
    mugrade.submit([obj.num_layers, obj.num_heads, obj.head_dim, str(obj.dtype)])
    mugrade.submit([[key, list(value.shape)] for key, value in sorted(obj.params.items())])
    mugrade.submit([[key, np.where(np.isneginf(value.value), 0., value.value),
                    np.isneginf(value.value), str(value.dtype)]
                   for key, value in sorted(obj.buffers.items())])
    mugrade.submit([bool(np.isfinite(v.value).all() and np.any(v.value != 0)) for _, v in sorted(obj.params.items())])


def test_embedding(function):
    result = _integration(function, 'embedding', 0)
    assert _close(result[0], [[[-1.0, 4.0], [1.0, -2.0]], [[-1.0, 4.0], [3.0, 0.5]]])
    assert result[0].shape == (2, 2, 2) and result[2] == 'float32'
    assert len(result[1]) == 1
    assert _close(result[1][0], [[-0.3333333432674408, 0.0], [1.0, 1.3333333730697632], [-0.6666666567325592, 0.0], [0.0, 0.0]])
    result = _integration(function, 'embedding', 1)
    assert _close(result[0], [[[2.0, -3.0], [3.0, 0.5], [2.0, -3.0]]])
    assert result[0].shape == (1, 3, 2) and result[2] == 'float32'
    assert len(result[1]) == 1
    assert _close(result[1][0], [[0.0, 0.0], [-0.0833333358168602, 0.25], [0.0, 0.0], [-0.1666666865348816, 0.5000000298023224]])


def submit_embedding(function):
    result = _integration(function, 'embedding', 2)
    mugrade.submit(result[0])
    mugrade.submit(result[1])
    mugrade.submit([list(result[0].shape), result[2]])


def test_linear(function):
    result = _integration(function, 'linear', 0)
    assert _close(result[0], [[[8.0, -5.0, -1.5], [-8.0, 5.5, 3.25]]])
    assert result[0].shape == (1, 2, 3) and result[2] == 'float32'
    assert len(result[1]) == 2
    assert _close(result[1][0], [[[-1.4999998807907104, 1.3333332538604736], [0.0, 1.3333333730697632]]])
    assert _close(result[1][1], [[-1.0, -0.5, 0.0], [2.0, 2.3333334922790527, 2.6666667461395264]])
    result = _integration(function, 'linear', 1)
    assert _close(result[0], [[[8.5, -5.75, -1.375], [-8.0, 6.25, 3.25]]])
    assert result[0].shape == (1, 2, 3) and result[2] == 'float32'
    assert len(result[1]) == 2
    assert _close(result[1][0], [[[-1.125, 1.2291666269302368], [0.3750000298023224, 1.4791666269302368]]])
    assert _close(result[1][1], [[-0.8125, -0.2291666567325592, 0.3541666865348816], [2.25, 2.5833332538604736, 2.9166667461395264]])


def submit_linear(function):
    result = _integration(function, 'linear', 2)
    mugrade.submit(result[0])
    mugrade.submit(result[1])
    mugrade.submit([list(result[0].shape), result[2]])


def test_rope(function):
    result = _integration(function, 'rope', 0)
    assert _close(result[0], [[[[-0.25, 0.0, 0.5, 0.75], [-1.0, -1.25, -1.75, 1.5]], [[-2.25, 2.0, 2.5, 2.75], [-3.0, -3.25, -3.75, 3.5]]]])
    assert result[0].shape == (1, 2, 2, 4) and result[2] == 'float32'
    assert len(result[1]) == 1
    assert _close(result[1][0], [[[[-0.6666666865348816, 1.0, -0.3333333432674408, 0.0], [-0.3333333432674408, -0.6666666865348816, 1.3333333730697632, -1.0]], [[2.0, -1.6666666269302368, 2.3333332538604736, 2.6666667461395264], [-3.0, -3.3333332538604736, 4.0, -3.6666667461395264]]]])
    result = _integration(function, 'rope', 1)
    assert _close(result[0], [[[[-0.25, -0.5, -1.0, 0.75], [1.25, 1.5, -1.75, -2.0]], [[-2.25, -2.5, -3.0, 2.75], [3.25, 3.5, -3.75, -4.0]]]])
    assert result[0].shape == (1, 2, 2, 4) and result[2] == 'float32'
    assert len(result[1]) == 1
    assert _close(result[1][0], [[[[0.75, 0.4166666567325592, 0.25, 0.0833333358168602], [0.5833333134651184, 0.9166666865348816, -1.25, -1.5833333730697632]], [[-1.9166666269302368, -2.25, 2.9166667461395264, -2.5833332538604736], [3.25, 3.5833332538604736, -3.9166667461395264, -4.25]]]])


def submit_rope(function):
    result = _integration(function, 'rope', 2)
    mugrade.submit(result[0])
    mugrade.submit(result[1])
    mugrade.submit([list(result[0].shape), result[2]])


def test_multihead_attn(function):
    result = _integration(function, 'multihead_attn', 0)
    assert _close(result[0], [[[1.5, 7.0, 3.0, -2.5], [-3.1144141784286843, 7.954663573422431, -7.195715290947759, 0.7313122609668747]]])
    assert result[0].shape == (1, 2, 4) and result[2] == 'float64'
    assert len(result[1]) == 5
    assert _close(result[1][0], [[[-1.268880503743745, -1.4202307575743827, 1.0695877343996478, -3.333498950860726], [1.2647248455582627, -1.1425242298542033, -2.041539915709109, 7.519944865069615]]])
    assert _close(result[1][1], [[-0.03799358755350113, -0.07598719745874405, 0.05936340615153313, -0.17808997631072998], [0.018996793776750565, 0.037993598729372025, -0.029681703075766563, 0.08904498815536499], [0.056990381330251694, 0.11398079991340637, -0.08904510736465454, 0.26713496446609497], [0.018996793776750565, 0.037993598729372025, -0.029681703075766563, 0.08904498815536499]])
    assert _close(result[1][2], [[-0.04418132081627846, 0.06450240314006805, 0.046894621104002, 0.25981149077415466], [-0.03402081876993179, -0.0883626639842987, 0.1118474081158638, 0.4112735986709595], [0.05504968762397766, -0.10797589272260666, -0.04328298941254616, -0.28148138523101807], [-0.0003539305180311203, -0.05469578504562378, 0.03067057766020298, 0.08656598627567291]])
    assert _close(result[1][3], [[-3.671973965674212, 0.5026536495037729, -5.046409538562331, -6.108401483358767], [-1.2239913218914045, -2.054671005720965, -0.01546984618744407, 0.18608839443596636], [4.895965287565616, -1.114649310449475, 7.0618793847497745, 8.588979755589467], [0.6119956609457019, -0.9726644971395177, 1.5077349230937216, 1.9069558027820168]])
    assert _close(result[1][4], [[-1.5846601580852195, -1.8359869828371065, -2.0873138075889925, -2.3386406323408795], [-3.278662327612369, -1.2239913218914042, 0.8306796838295605, 2.8853506895505254], [-1.9069558027820164, -2.4805782722307, -3.0542007416793835, -3.6278232111280673], [-0.534891573956744, 0.9302168520865127, 2.3953252781297687, 3.8604337041730252]])
    result = _integration(function, 'multihead_attn', 1)
    assert _close(result[0], [[[1.75, 7.0, 3.25, -2.5], [-3.111622855372662, 6.559098842393043, -7.216344172683794, 0.042671651598630955]]])
    assert result[0].shape == (1, 2, 4) and result[2] == 'float64'
    assert len(result[1]) == 5
    assert _close(result[1][0], [[[-0.7274693124242539, -1.414303245185602, 0.44066678101086665, -0.9089030836365348], [1.7669732816763632, -1.0712570794783156, -2.638760901260019, 9.698767094745412]]])
    assert _close(result[1][1], [[-0.08395864069461823, -0.16791731119155884, 0.06941065192222595, -0.27764320373535156], [0.04197932034730911, 0.08395865559577942, -0.034705325961112976, 0.13882160186767578], [0.12593796849250793, 0.25187596678733826, -0.10411597788333893, 0.41646480560302734], [0.031484492123126984, 0.06296899169683456, -0.026028994470834732, 0.10411620140075684]])
    assert _close(result[1][2], [[-0.060728397220373154, 0.1152908056974411, 0.09624972194433212, 0.4737199544906616], [-0.0206849854439497, -0.12145678699016571, 0.16952571272850037, 0.633743166923523], [0.07407619804143906, -0.19420666992664337, -0.07182438671588898, -0.4203788936138153], [0.003933568950742483, -0.06678348779678345, 0.04422977939248085, 0.14364902675151825]])
    assert _close(result[1][3], [[-4.237867763159744, 0.3289216108459676, -5.6626949086423535, -6.830959937940393], [0.31142530364315535, -2.052639504355087, 1.8922477204177377, 2.475089249864495], [5.754298785427377, -1.1227753159129858, 8.181009118329051, 9.932976333875356], [0.9221436740892109, -0.8201734572445617, 1.7769380698955657, 2.21456102086721]])
    assert _close(result[1][4], [[-1.954473076156888, -2.1189338815798724, -2.283394687002856, -2.4478554924258398], [-1.7412142007119316, 0.31142530364315535, 2.364064807998242, 4.416704312353329], [-2.346280512287662, -2.9251074716901364, -3.5039344310926097, -4.082761390495084], [0.15867759605393653, 1.3922076509419006, 2.6257377058298643, 3.8592677607178283]])


def submit_multihead_attn(function):
    result = _integration(function, 'multihead_attn', 2)
    mugrade.submit(result[0])
    mugrade.submit(result[1])
    mugrade.submit([list(result[0].shape), result[2]])


def test_mlp(function):
    result = _integration(function, 'mlp', 0)
    assert _close(result[0], [[[-2.5718350410461426, 6.977212905883789], [7.02100944519043, -12.74514102935791]]])
    assert result[0].shape == (1, 2, 2) and result[2] == 'float32'
    assert len(result[1]) == 3
    assert _close(result[1][0], [[[-0.6849835515022278, 1.0936225652694702], [-0.19733944535255432, -0.7177262306213379]]])
    assert _close(result[1][1], [[-0.19740839302539825, 0.02860908955335617, -1.1152715682983398], [-0.9495060443878174, -0.6756652593612671, 2.10357928276062]])
    assert _close(result[1][2], [[-2.021136522293091, 0.09485174715518951], [-0.005280338227748871, 0.1589372456073761], [-2.255275011062622, -1.5402363538742065]])
    result = _integration(function, 'mlp', 1)
    assert _close(result[0], [[[-2.72019624710083, 7.4053754806518555], [6.483591079711914, -11.767313003540039]]])
    assert result[0].shape == (1, 2, 2) and result[2] == 'float32'
    assert len(result[1]) == 3
    assert _close(result[1][0], [[[-0.3708283603191376, 0.5772460699081421], [-0.8723000288009644, -0.9780329465866089]]])
    assert _close(result[1][1], [[-0.30518078804016113, 0.23073777556419373, -0.7275338172912598], [-1.6080690622329712, 0.48409563302993774, 0.8699924349784851]])
    assert _close(result[1][2], [[-0.3748350143432617, 1.5651404857635498], [0.10943872481584549, 0.24378058314323425], [-1.8199634552001953, -1.0672820806503296]])


def submit_mlp(function):
    result = _integration(function, 'mlp', 2)
    mugrade.submit(result[0])
    mugrade.submit(result[1])
    mugrade.submit([list(result[0].shape), result[2]])


def test_transformer_block(function):
    result = _integration(function, 'transformer_block', 0)
    assert _close(result[0], [[[3.5, -2.0], [2.25, 3.5]]])
    assert result[0].shape == (1, 2, 2) and result[2] == 'float32'
    assert len(result[1]) == 2
    assert _close(result[1][0], [[[0.6324555277824402, -1.2649110555648804], [1.3416407108306885, 0.4472135901451111]]])
    assert _close(result[1][1], [[[1.17669677734375, -0.7844645380973816], [0.7844645380973816, 1.17669677734375]]])
    result = _integration(function, 'transformer_block', 1)
    assert _close(result[0], [[[3.75, -2.0], [2.25, 3.25]]])
    assert result[0].shape == (1, 2, 2) and result[2] == 'float32'
    assert len(result[1]) == 2
    assert _close(result[1][0], [[[0.7495316863059998, -1.1992506980895996], [1.3719887733459473, 0.3429971933364868]]])
    assert _close(result[1][1], [[[1.2278811931610107, -0.7016463875770569], [0.8318004012107849, 1.1437255144119263]]])


def submit_transformer_block(function):
    result = _integration(function, 'transformer_block', 2)
    mugrade.submit(result[0])
    mugrade.submit(result[1])
    mugrade.submit([list(result[0].shape), result[2]])


def test___call__(function):
    result = _integration(function, '__call__', 0)
    assert _close(result[0], [[[0.9486832618713379, -2.5298221111297607, 3.1622776985168457], [1.2649110555648804, 1.8973665237426758, 0.0]]])
    assert result[0].shape == (1, 2, 3) and result[2] == 'float32'
    assert len(result[1]) == 2
    assert _close(result[1][0], [[[-0.5059643983840942, -1.0119287967681885], [0.5059643983840942, -0.2529821991920471]]])
    assert _close(result[1][1], [[-1.2649110555648804, -0.6324555277824402, 2.631654894003077e-09], [0.6324555277824402, 0.8432740569114685, 1.054092526435852]])
    assert _close(result[3][0], [[[1.0, -1.0], [0.6324555277824402, 1.2649110555648804]]])
    assert _close(result[3][1], [[0.0, 0.0, -np.inf], [0.0, 0.0, 0.0]])
    result = _integration(function, '__call__', 1)
    assert _close(result[0], [[[1.0051413774490356, -2.4410576820373535, 3.1590158939361572], [1.315587043762207, 1.7541160583496094, 0.17541158199310303]]])
    assert result[0].shape == (1, 2, 3) and result[2] == 'float32'
    assert len(result[1]) == 2
    assert _close(result[1][0], [[[-0.2871832251548767, -0.6461622714996338], [0.6800573468208313, -0.38860419392585754]]])
    assert _close(result[1][1], [[-0.7938319444656372, -0.12917488813400269, 0.5354821681976318], [0.7377451658248901, 0.955583393573761, 1.1734216213226318]])
    assert _close(result[3][0], [[[1.0569963455200195, -0.939552366733551], [0.6324555277824402, 1.2649110555648804]]])
    assert _close(result[3][1], [[0.0, 0.0, 0.0, -np.inf], [0.0, 0.0, 0.0, 0.0]])


def submit___call__(function):
    result = _integration(function, '__call__', 2)
    mugrade.submit(result[0])
    mugrade.submit(result[1])
    mugrade.submit([list(result[0].shape), result[2]])
    normalized, mask = result[3]
    mugrade.submit([normalized, np.where(np.isneginf(mask), 0., mask), np.isneginf(mask)])
