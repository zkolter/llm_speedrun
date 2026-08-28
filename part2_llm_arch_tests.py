from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import mugrade
import torch


_MISSING = object()


@contextmanager
def _patched_globals(function, **updates):
    """Temporarily replace a graded callable's global collaborators."""
    namespace = function.__globals__
    previous = {name: namespace.get(name, _MISSING) for name in updates}
    namespace.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is _MISSING:
                namespace.pop(name, None)
            else:
                namespace[name] = value


def _causal_mask(length):
    return torch.triu(
        torch.full((length, length), float("-inf")),
        diagonal=1,
    )


def test_embedding(embedding):
    weights = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0], [2.5, -3.0, 1.0]],
        dtype=torch.float32,
    )
    indices = torch.tensor([[2, 0], [1, 2]])
    out = embedding(indices, weights, torch.float64)

    assert out.shape == (2, 2, 3)
    assert out.dtype == torch.float64
    assert torch.equal(
        out,
        torch.tensor(
            [
                [[2.5, -3.0, 1.0], [1.0, 2.0, 3.0]],
                [[-1.0, 0.5, 4.0], [2.5, -3.0, 1.0]],
            ],
            dtype=torch.float64,
        ),
    )

    scalar = embedding(torch.tensor(1), weights, torch.float16)
    assert scalar.shape == (3,)
    assert scalar.dtype == torch.float16
    assert torch.equal(scalar, torch.tensor([-1.0, 0.5, 4.0], dtype=torch.float16))


def submit_embedding(embedding):
    weights = torch.tensor(
        [[0.0, 1.0], [2.0, 3.0], [-1.0, 4.0], [5.0, -2.0]],
        dtype=torch.float32,
    )
    mugrade.submit(
        embedding(torch.tensor([[3, 1], [0, 2]]), weights, torch.float64)
        .detach()
        .cpu()
        .numpy()
    )
    mugrade.submit(
        embedding(torch.tensor([2, 2, 1]), weights, torch.float32)
        .detach()
        .cpu()
        .numpy()
    )
    mugrade.submit(str(embedding(torch.tensor(0), weights, torch.float16).dtype))


def test_linear(linear):
    x = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32)
    weights = torch.tensor(
        [[1.0, -1.0, 0.5], [2.0, 0.0, 3.0]],
        dtype=torch.float64,
    )
    out = linear(x, weights)

    assert out.shape == (2, 3)
    assert out.dtype == torch.float32
    assert torch.allclose(
        out,
        torch.tensor([[5.0, -1.0, 6.5], [0.0, 1.0, 1.0]]),
        atol=1e-6,
    )

    batched = linear(torch.stack([x, -x]), weights)
    assert batched.shape == (2, 2, 3)
    assert torch.allclose(batched[1], -out, atol=1e-6)


def submit_linear(linear):
    weights = torch.tensor(
        [[2.0, -1.0], [0.5, 3.0], [-2.0, 1.5]],
        dtype=torch.float64,
    )
    x = torch.tensor([[1.0, 0.0, -1.0], [2.0, -0.5, 1.0]])
    mugrade.submit(linear(x, weights).detach().cpu().numpy())
    mugrade.submit(linear(torch.stack([x, x + 1.0]), weights).detach().cpu().numpy())
    mugrade.submit(str(linear(x.to(torch.float16), weights).dtype))


def test_silu(silu):
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = torch.tensor(
        [-0.23840584, -0.26894143, 0.0, 0.73105860, 1.76159406]
    )
    out = silu(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.allclose(out, expected, atol=1e-6)
    assert silu(torch.tensor(0.0)).item() == 0.0
    assert torch.all(silu(torch.tensor([1.0, 3.0])) > 0)


def submit_silu(silu):
    values = torch.tensor([[-3.0, -0.25, 0.5], [1.5, 2.5, -1.5]])
    mugrade.submit(silu(values).detach().cpu().numpy())
    mugrade.submit(silu(torch.tensor([0.125, -4.0, 4.0])).detach().cpu().numpy())
    mugrade.submit(str(silu(torch.tensor([1.0], dtype=torch.float64)).dtype))


def test_rms_norm(rms_norm):
    x = torch.tensor([[3.0, 4.0], [1.0, -1.0]])
    expected = torch.tensor(
        [[0.84852815, 1.13137090], [1.0, -1.0]]
    )
    out = rms_norm(x)

    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(rms_norm(7.0 * x), out, atol=1e-6)

    half = rms_norm(torch.tensor([[2.0, -2.0, 1.0, -1.0]], dtype=torch.float16))
    assert half.dtype == torch.float16
    assert torch.allclose(
        half.float(),
        torch.tensor([[1.2649111, -1.2649111, 0.6324555, -0.6324555]]),
        atol=1e-3,
    )


def submit_rms_norm(rms_norm):
    x = torch.tensor([[2.0, -1.0, 2.0], [-3.0, 1.0, 0.5]])
    mugrade.submit(rms_norm(x).detach().cpu().numpy())
    mugrade.submit(rms_norm(x.reshape(1, 2, 3) * 0.25).detach().cpu().numpy())
    mugrade.submit(str(rms_norm(x.to(torch.float64)).dtype))


def test_softmax(softmax):
    x = torch.tensor([[0.0, 0.0], [1.0, 2.0], [1000.0, 999.0]])
    expected = torch.tensor(
        [
            [0.5, 0.5],
            [0.26894143, 0.73105860],
            [0.73105860, 0.26894143],
        ]
    )
    out = softmax(x)

    assert out.shape == x.shape
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(out.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert torch.isfinite(out).all()
    assert out.dtype == x.dtype


def submit_softmax(softmax):
    x = torch.tensor([[3.0, 1.0, -2.0], [-4.0, -4.0, -4.0]])
    mugrade.submit(softmax(x).detach().cpu().numpy())
    mugrade.submit(
        softmax(torch.tensor([[[0.0, 1.0], [2.0, -1.0]]]))
        .detach()
        .cpu()
        .numpy()
    )
    mugrade.submit(softmax(x).sum(dim=-1).detach().cpu().numpy())


def test_self_attn(self_attn):
    q = torch.tensor([[1.0], [0.0]])
    k = torch.tensor([[1.0], [0.0]])
    v = torch.tensor([[2.0, -1.0], [4.0, 1.0]])

    causal = self_attn(q, k, v, _causal_mask(2))
    assert causal.shape == (2, 2)
    assert torch.allclose(
        causal,
        torch.tensor([[2.0, -1.0], [3.0, 0.0]]),
        atol=1e-6,
    )

    unmasked = self_attn(q, k, v, torch.zeros(2, 2))
    assert torch.allclose(
        unmasked,
        torch.tensor([[2.53788280, -0.46211717], [3.0, 0.0]]),
        atol=1e-6,
    )

    q_batch = torch.zeros(1, 2, 2, 1)
    k_batch = torch.zeros(1, 2, 2, 1)
    v_batch = torch.tensor(
        [[[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]]]
    )
    batched = self_attn(q_batch, k_batch, v_batch, torch.zeros(2, 2))
    assert batched.shape == (1, 2, 2, 2)
    assert torch.allclose(
        batched,
        torch.tensor([[[[3.0, 5.0], [3.0, 5.0]], [[4.0, 6.0], [4.0, 6.0]]]]),
        atol=1e-6,
    )


def submit_self_attn(self_attn):
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    k = torch.tensor([[0.5, 1.0], [1.0, 0.0], [-1.0, 1.0]])
    v = torch.tensor([[1.0, -1.0], [0.5, 2.0], [-2.0, 0.25]])
    mugrade.submit(self_attn(q, k, v, _causal_mask(3)).detach().cpu().numpy())
    mugrade.submit(self_attn(q, k, v, torch.zeros(3, 3)).detach().cpu().numpy())
    mugrade.submit(list(self_attn(q[:1], k[:1], v[:1], torch.zeros(1, 1)).shape))


def test_rope(rope):
    x = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]
    )
    identity = SimpleNamespace(
        buffers={
            "rope_w1": torch.ones(2, 4),
            "rope_w2": torch.zeros(2, 4),
        }
    )
    assert torch.equal(rope(identity, x), x)

    rotation = SimpleNamespace(
        buffers={
            "rope_w1": torch.zeros(2, 4),
            "rope_w2": torch.tensor(
                [[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]]
            ),
        }
    )
    expected = torch.tensor(
        [[[[2.0, -1.0, 4.0, -3.0], [-6.0, 5.0, -8.0, 7.0]]]]
    )
    out = rope(rotation, x)
    assert out.shape == x.shape
    assert torch.equal(out, expected)

    half = rope(rotation, x.to(torch.float16))
    assert half.dtype == torch.float16
    assert torch.equal(half.float(), expected)


def submit_rope(rope):
    fixture = SimpleNamespace(
        buffers={
            "rope_w1": torch.tensor(
                [[0.5, 0.5, -1.0, -1.0], [1.0, 1.0, 0.25, 0.25]]
            ),
            "rope_w2": torch.tensor(
                [[1.0, -1.0, 0.5, -0.5], [-0.5, 0.5, 2.0, -2.0]]
            ),
        }
    )
    x = torch.tensor(
        [[[[2.0, -1.0, 0.5, 3.0], [-2.0, 4.0, 1.0, -3.0]]]]
    )
    mugrade.submit(rope(fixture, x).detach().cpu().numpy())
    mugrade.submit(rope(fixture, -x).detach().cpu().numpy())
    mugrade.submit(str(rope(fixture, x.to(torch.float64)).dtype))


def test_multihead_attn(multihead_attn):
    fixture = SimpleNamespace(
        params={
            "wq_0": torch.eye(4),
            "wk_0": torch.eye(4),
            "wv_0": torch.eye(4),
            "wp_0": torch.eye(4),
        },
        num_heads=2,
        head_dim=2,
        rope=lambda value: value,
    )
    x = torch.tensor(
        [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]]
    )

    causal = multihead_attn(fixture, x, 0, _causal_mask(2))
    assert causal.shape == (1, 2, 4)
    assert torch.allclose(
        causal,
        torch.tensor(
            [[
                [1.0, 0.0, 0.0, 1.0],
                [0.19557032, 0.80442971, 0.80442971, 0.19557032],
            ]]
        ),
        atol=1e-6,
    )

    unmasked = multihead_attn(fixture, x, 0, torch.zeros(2, 2))
    assert unmasked.dtype == x.dtype
    assert torch.allclose(
        unmasked,
        torch.tensor(
            [[
                [0.80442971, 0.19557032, 0.19557032, 0.80442971],
                [0.19557032, 0.80442971, 0.80442971, 0.19557032],
            ]]
        ),
        atol=1e-6,
    )


def submit_multihead_attn(multihead_attn):
    fixture = SimpleNamespace(
        params={
            "wq_1": torch.eye(4),
            "wk_1": torch.tensor(
                [[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5],
                 [0.5, 0.0, 1.0, 0.0], [0.0, 0.5, 0.0, 1.0]]
            ),
            "wv_1": torch.diag(torch.tensor([1.0, -1.0, 0.5, 2.0])),
            "wp_1": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 2.0]]
            ),
        },
        num_heads=2,
        head_dim=2,
        rope=lambda value: value,
    )
    x = torch.tensor(
        [[
            [1.0, 0.0, -1.0, 0.5],
            [0.0, 1.0, 0.5, -0.5],
            [1.0, 1.0, 0.0, 1.0],
        ]]
    )
    causal = multihead_attn(fixture, x, 1, _causal_mask(3))
    open_attention = multihead_attn(fixture, x, 1, torch.zeros(3, 3))
    mugrade.submit(causal.detach().cpu().numpy())
    mugrade.submit(open_attention.detach().cpu().numpy())
    mugrade.submit([list(causal.shape), str(causal.dtype)])


def test_mlp(mlp):
    fixture = SimpleNamespace(
        params={
            "w1_0": torch.tensor(
                [[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]]
            ),
            "w2_0": torch.tensor(
                [[1.0, -1.0], [0.5, 2.0], [-2.0, 1.0]]
            ),
        }
    )
    x = torch.tensor([[1.0, -1.0], [0.0, 2.0]])
    out = mlp(fixture, x, 0)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.allclose(
        out,
        torch.tensor(
            [[1.08866739, -1.44627607], [-1.55916059, 9.61770439]]
        ),
        atol=1e-6,
    )
    assert torch.equal(mlp(fixture, torch.zeros_like(x), 0), torch.zeros_like(x))


def submit_mlp(mlp):
    fixture = SimpleNamespace(
        params={
            "w1_2": torch.tensor(
                [[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]]
            ),
            "w2_2": torch.tensor(
                [[2.0, 0.0], [-1.0, 1.0], [0.5, -2.0]]
            ),
        }
    )
    x = torch.tensor([[2.0, -1.0], [-0.5, 3.0], [1.0, 1.0]])
    mugrade.submit(mlp(fixture, x, 2).detach().cpu().numpy())
    mugrade.submit(mlp(fixture, -x, 2).detach().cpu().numpy())
    mugrade.submit(str(mlp(fixture, x.to(torch.float64), 2).dtype))


def test_transformer_block(transformer_block):
    norm_inputs = []
    attention_calls = []
    mlp_calls = []

    def scripted_norm(value):
        norm_inputs.append(value.clone())
        return value + 10.0

    def scripted_attention(value, layer, mask):
        attention_calls.append((value.clone(), layer, mask.clone()))
        return torch.full_like(value, 2.0)

    def scripted_mlp(value, layer):
        mlp_calls.append((value.clone(), layer))
        return torch.full_like(value, -1.0)

    fixture = SimpleNamespace(
        multihead_attn=scripted_attention,
        mlp=scripted_mlp,
    )
    x = torch.tensor([[[1.0, -2.0], [3.0, 0.5]]])
    mask = _causal_mask(2)
    with _patched_globals(transformer_block, rms_norm=scripted_norm):
        out = transformer_block(fixture, x, 3, mask)

    assert torch.equal(out, x + 1.0)
    assert len(norm_inputs) == 2
    assert torch.equal(norm_inputs[0], x)
    assert torch.equal(norm_inputs[1], x + 2.0)
    assert attention_calls[0][1] == 3 and torch.equal(attention_calls[0][2], mask)
    assert torch.equal(attention_calls[0][0], x + 10.0)
    assert mlp_calls[0][1] == 3 and torch.equal(mlp_calls[0][0], x + 12.0)


def submit_transformer_block(transformer_block):
    norm_inputs = []
    trace = []

    def scripted_norm(value):
        norm_inputs.append(value.clone())
        return -value

    def scripted_attention(value, layer, mask):
        trace.append(["attention", layer, list(mask.shape)])
        return torch.full_like(value, 0.5)

    def scripted_mlp(value, layer):
        trace.append(["mlp", layer, list(value.shape)])
        return torch.full_like(value, 2.0)

    fixture = SimpleNamespace(
        multihead_attn=scripted_attention,
        mlp=scripted_mlp,
    )
    x = torch.tensor([[[2.0, -1.0], [0.0, 3.0], [-2.0, 0.5]]])
    with _patched_globals(transformer_block, rms_norm=scripted_norm):
        out = transformer_block(fixture, x, 4, _causal_mask(3))

    mugrade.submit(out.detach().cpu().numpy())
    mugrade.submit(torch.stack(norm_inputs).detach().cpu().numpy())
    mugrade.submit(trace)


def test___init__(student_init):
    config = {
        "depth": 2,
        "aspect_ratio": 2,
        "mlp_multiple": 2,
        "head_dim": 2,
        "dtype": torch.float16,
        "vocab_size": 7,
        "seq_len": 5,
        "rope_theta": 100.0,
    }
    model = SimpleNamespace()
    torch.manual_seed(11)
    result = student_init(model, config)

    assert result is None
    assert (model.num_layers, model.head_dim, model.num_heads, model.dtype) == (
        2,
        2,
        2,
        torch.float16,
    )
    assert sorted(model.params) == sorted(
        ["embedding", "output"]
        + [f"{name}_{layer}" for layer in range(2) for name in ("wq", "wk", "wv", "wp", "w1", "w2")]
    )
    assert {name: tuple(value.shape) for name, value in model.params.items()} == {
        "embedding": (7, 4),
        "output": (4, 7),
        "wq_0": (4, 4), "wk_0": (4, 4), "wv_0": (4, 4), "wp_0": (4, 4),
        "w1_0": (4, 8), "w2_0": (8, 4),
        "wq_1": (4, 4), "wk_1": (4, 4), "wv_1": (4, 4), "wp_1": (4, 4),
        "w1_1": (4, 8), "w2_1": (8, 4),
    }
    assert model.buffers["mask"].shape == (5, 5)
    assert torch.equal(
        torch.isneginf(model.buffers["mask"]),
        torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1),
    )
    assert torch.equal(model.buffers["rope_w1"][0], torch.ones(2))
    assert torch.equal(model.buffers["rope_w2"][0], torch.zeros(2))


def submit___init__(student_init):
    config = {
        "depth": 1,
        "aspect_ratio": 4,
        "mlp_multiple": 3,
        "head_dim": 2,
        "dtype": torch.float32,
        "vocab_size": 6,
        "seq_len": 4,
        "rope_theta": 1000.0,
    }
    model = SimpleNamespace()
    torch.manual_seed(23)
    student_init(model, config)

    mugrade.submit([model.num_layers, model.head_dim, model.num_heads, str(model.dtype)])
    mugrade.submit([[name, list(model.params[name].shape)] for name in sorted(model.params)])
    mugrade.submit(torch.isneginf(model.buffers["mask"]).detach().cpu().numpy())
    mugrade.submit(model.buffers["rope_w2"].detach().cpu().numpy())
    mugrade.submit(model.params["embedding"][0].detach().cpu().numpy())


def test___call__(student_call):
    base = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    helper_calls = []
    block_calls = []

    def scripted_embedding(tokens, weights, dtype):
        helper_calls.append(("embedding", tokens.clone(), weights, dtype))
        return base.clone()

    def scripted_norm(value):
        helper_calls.append(("norm", value.clone()))
        return value + 1.0

    def scripted_linear(value, weights):
        helper_calls.append(("linear", value.clone(), weights))
        return value * 2.0

    mask = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    def scripted_block(value, layer, used_mask):
        block_calls.append((value.clone(), layer, used_mask.clone()))
        return value + (layer + 1)

    embedding_weights = object()
    output_weights = object()
    fixture = SimpleNamespace(
        buffers={"mask": mask},
        params={"embedding": embedding_weights, "output": output_weights},
        dtype=torch.float16,
        num_layers=2,
        transformer_block=scripted_block,
    )
    tokens = torch.tensor([[2, 1]])
    with _patched_globals(
        student_call,
        embedding=scripted_embedding,
        rms_norm=scripted_norm,
        linear=scripted_linear,
    ):
        out = student_call(fixture, tokens)

    assert torch.equal(out, 2.0 * (base + 5.0))
    assert [call[1] for call in block_calls] == [0, 1]
    assert all(torch.equal(call[2], mask[:2, :2].to(torch.float16)) for call in block_calls)
    assert helper_calls[0][0] == "embedding" and helper_calls[0][2] is embedding_weights
    assert helper_calls[0][3] == torch.float16
    assert [call[0] for call in helper_calls] == ["embedding", "norm", "norm", "linear"]
    assert helper_calls[-1][2] is output_weights


def submit___call__(student_call):
    base = torch.tensor(
        [[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]
    )
    helper_trace = []
    block_trace = []

    def scripted_embedding(tokens, weights, dtype):
        helper_trace.append(["embedding", list(tokens.shape), str(dtype)])
        return base.clone()

    def scripted_norm(value):
        helper_trace.append(["norm", list(value.shape)])
        return value - 0.5

    def scripted_linear(value, weights):
        helper_trace.append(["linear", list(value.shape)])
        return value - 10.0

    def scripted_block(value, layer, mask):
        block_trace.append([layer, list(mask.shape), str(mask.dtype)])
        return value + (layer + 1)

    fixture = SimpleNamespace(
        buffers={"mask": _causal_mask(5)},
        params={"embedding": object(), "output": object()},
        dtype=torch.float32,
        num_layers=3,
        transformer_block=scripted_block,
    )
    tokens = torch.tensor([[5, 4, 3]])
    with _patched_globals(
        student_call,
        embedding=scripted_embedding,
        rms_norm=scripted_norm,
        linear=scripted_linear,
    ):
        out = student_call(fixture, tokens)

    mugrade.submit(out.detach().cpu().numpy())
    mugrade.submit(block_trace)
    mugrade.submit(helper_trace)


def test_save(save):
    fixture = SimpleNamespace(
        params={
            "vector": torch.tensor([1.0, -2.0, 3.5]),
            "matrix": torch.tensor([[1, 2], [3, 4]]),
        }
    )
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.pt"
        result = save(fixture, str(path))
        assert result is None
        assert path.is_file()
        loaded = torch.load(path, weights_only=True)

    assert sorted(loaded) == ["matrix", "vector"]
    assert torch.equal(loaded["vector"], fixture.params["vector"])
    assert torch.equal(loaded["matrix"], fixture.params["matrix"])


def submit_save(save):
    fixture = SimpleNamespace(
        params={
            "alpha": torch.tensor([-1.5, 2.25]),
            "beta": torch.tensor([[5.0], [6.0], [7.0]]),
        }
    )
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "submission.pt"
        save(fixture, str(path))
        loaded = torch.load(path, weights_only=True)

    mugrade.submit(sorted(loaded))
    mugrade.submit(loaded["alpha"].detach().cpu().numpy())
    mugrade.submit(loaded["beta"].detach().cpu().numpy())


def test_load(load):
    payload = {
        "left": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "right": torch.tensor([-5, 6, 7]),
    }
    fixture = SimpleNamespace(params={"stale": torch.tensor(0)})
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.pt"
        torch.save(payload, path)
        result = load(fixture, str(path))

    assert result is None
    assert sorted(fixture.params) == ["left", "right"]
    assert "stale" not in fixture.params
    assert torch.equal(fixture.params["left"], payload["left"])
    assert torch.equal(fixture.params["right"], payload["right"])


def submit_load(load):
    payload = {
        "embedding": torch.tensor([[0.5, -0.5], [1.5, 2.5]]),
        "output": torch.tensor([3.0, -4.0]),
    }
    fixture = SimpleNamespace(params={})
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "submission.pt"
        torch.save(payload, path)
        load(fixture, str(path))

    mugrade.submit(sorted(fixture.params))
    mugrade.submit(fixture.params["embedding"].detach().cpu().numpy())
    mugrade.submit(fixture.params["output"].detach().cpu().numpy())
