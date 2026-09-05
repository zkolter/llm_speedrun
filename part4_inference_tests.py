"""Small, deterministic tests for the new Part 4 inference methods.

Fixtures provide literal state and narrow collaborators, never a reference LLM.
No test compares randomly initialized parameters or seeded sampling sequences.
Only the five methods that add inference behavior are graded; parameter-only
renames of Part 2's embedding/linear helpers are treated as reused functions.
"""

from types import SimpleNamespace

import mugrade
import torch


def _array(x):
    return x.detach().cpu().numpy()


def _mask(length):
    return torch.triu(torch.full((length, length), -torch.inf), diagonal=1)


def _rope_state(dtype=torch.float32):
    # Literal rotations: two independent coordinate pairs at four positions.
    return SimpleNamespace(buffers={
        "rope1": torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1],
                               [-1, -1, 0, 0], [0, 0, -1, -1]], dtype=dtype),
        "rope2": torch.tensor([[0, 0, 0, 0], [-1, 1, 0, 0],
                               [0, 0, -1, 1], [1, -1, 0, 0]], dtype=dtype),
    })


def test_rope(rope):
    state = _rope_state()
    x = torch.tensor([[[[1., 2., 3., 4.], [5., 6., 7., 8.]]]])
    out = rope(state, x, pos=1)
    assert torch.equal(out, torch.tensor([[[[-2., 1., 3., 4.], [-5., -6., -8., 7.]]]]))
    assert torch.equal(rope(state, x[:, :, :1]), x[:, :, :1])
    assert torch.equal(rope(state, x[:, :, 1:], pos=2), out[:, :, 1:])
    batched = x.expand(2, 3, 2, 4).transpose(0, 1)
    assert torch.equal(rope(state, batched, pos=1), out.expand(3, 2, 2, 4))
    half = rope(_rope_state(torch.float16), x.half(), pos=2)
    assert half.dtype == torch.float16
    assert torch.equal(half, torch.tensor([[[[-1., -2., -4., 3.], [6., -5., -7., -8.]]]], dtype=torch.float16))


def submit_rope(rope):
    state = _rope_state()
    x = torch.tensor([[[[-3., 1., 2., -4.], [2., -5., 1., 3.]]],
                      [[[4., -2., -1., 5.], [-1., 3., -2., 4.]]]])
    mugrade.submit(_array(rope(state, x, pos=2)))
    mugrade.submit(_array(rope(state, x[:, :, :1], pos=1)))
    mugrade.submit(_array(rope(state, x, pos=0)))


def _attention_state():
    # Identity RoPE isolates attention from the separately graded rotation.
    # The recorder checks forwarding without imposing a Q/K call order.
    positions = []
    def identity_rope(x, pos=0):
        positions.append(pos)
        return x
    params = {
        "wq_1": torch.tensor([[1., .5, 0., 0.], [0., 1., 0., 0.],
                              [0., 0., 1., -.5], [0., 0., 0., 1.]]),
        "wk_1": torch.tensor([[1., 0., 0., .5], [0., 1., 0., 0.],
                              [0., .5, 1., 0.], [0., 0., 0., 1.]]),
        "wv_1": torch.diag(torch.tensor([2., -1., .5, 3.])),
        "wp_1": torch.tensor([[0., 1., 0., 0.], [1., 0., 0., 0.],
                              [0., 0., 1., 1.], [0., 0., 0., 2.]]),
    }
    return SimpleNamespace(params=params, num_heads=2, head_dim=2,
                           rope=identity_rope, positions=positions)


def _cache(batch, length=6):
    # Nonzero future entries must not leak into attention.
    return [{"k": torch.full((batch, 2, length, 2), 19.),
             "v": torch.full((batch, 2, length, 2), -23.)} for _ in range(2)]


def _attention_input():
    return torch.tensor([[[1., 2., 3., 1.], [2., -1., 1., 2.], [-1., 3., 2., -2.]],
                         [[2., 1., -1., 3.], [1., 3., 2., 1.], [3., -2., 1., -1.]]])


def test_multihead_attn(multihead_attn):
    state, x = _attention_state(), _attention_input()
    mask = _mask(3)
    out = multihead_attn(state, x, 1, mask)
    expected = torch.tensor([[[-2.0, 2.0, 1.5, 7.5], [0.183391273021698, 3.455594301223755, 0.9659277200698853, 10.17036247253418], [-2.3638501167297363, -0.03359014168381691, 1.0676895380020142, -5.404600143432617]], [[-1.0, 4.0, -0.5, 17.5], [-2.3393492698669434, 2.6606504917144775, 0.7205806374549866, 8.955936431884766], [0.08234630525112152, 4.6352057456970215, 0.5833012461662292, 1.9061710834503174]]])
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.allclose(out, expected, atol=2e-5, rtol=2e-5)
    cache = _cache(2)
    prefix = multihead_attn(state, x[:, :1], 1, mask[:1, :1], cache=cache)
    old_k, old_v = cache[1]["k"][:, :, :1].clone(), cache[1]["v"][:, :, :1].clone()
    state.positions.clear()
    tail = multihead_attn(state, x[:, 1:], 1, mask[1:, :3], pos=1, cache=cache)
    assert torch.allclose(torch.cat([prefix, tail], dim=1), expected, atol=2e-5, rtol=2e-5)
    assert state.positions and all(pos == 1 for pos in state.positions)
    assert torch.equal(cache[1]["k"][:, :, :1], old_k) and torch.equal(cache[1]["v"][:, :, :1], old_v)
    assert torch.all(cache[0]["k"] == 19.) and torch.all(cache[0]["v"] == -23.)
    # An arbitrary additive mask, rather than an implicitly causal kernel.
    only_first = torch.tensor([[0., -torch.inf, -torch.inf]]).expand(3, 3)
    selected = multihead_attn(_attention_state(), x, 1, only_first)
    assert torch.allclose(selected, expected[:, :1].expand_as(x), atol=2e-5, rtol=2e-5)


def submit_multihead_attn(multihead_attn):
    state = _attention_state()
    x = _attention_input().flip(0).flip(1) * .5
    mask = _mask(3)
    mugrade.submit(_array(multihead_attn(state, x, 1, mask)))
    cache = _cache(2, length=5)
    multihead_attn(state, x[:, :2], 1, mask[:2, :2], cache=cache)
    mugrade.submit(_array(multihead_attn(state, x[:, 2:], 1, mask[2:, :], pos=2, cache=cache)))
    selected = torch.tensor([[-torch.inf, 0., -torch.inf]]).expand(3, 3)
    mugrade.submit(_array(multihead_attn(state, x, 1, selected)))


def _block_state():
    calls = []
    def attention(x, layer, mask, pos=0, cache=None):
        calls.append((layer, mask.clone(), pos, cache))
        return x.flip(-1)
    def mlp(x, layer):
        return .5 * x
    return SimpleNamespace(multihead_attn=attention, mlp=mlp, calls=calls)


def test_transformer_block(transformer_block):
    state = _block_state()
    x = torch.tensor([[[3., 4.], [1., -2.]]])
    mask, cache = torch.tensor([[0., 0., -torch.inf], [0., 0., 0.]]), [{"marker": 7}]
    out = transformer_block(state, x, 2, mask, pos=1, cache=cache)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.allclose(out, torch.tensor([[[4.589979648590088, 5.386745452880859], [-0.3993869423866272, -2.061746120452881]]]), atol=2e-6, rtol=2e-6)
    assert state.calls and all(layer == 2 and pos == 1 and torch.equal(m, mask)
                               and c == cache for layer, m, pos, c in state.calls)
    zero_state = SimpleNamespace(multihead_attn=lambda x, *a, **kw: torch.zeros_like(x),
                                 mlp=lambda x, *a, **kw: torch.zeros_like(x))
    assert torch.equal(transformer_block(zero_state, x, 0, _mask(2)), x)
    batched = torch.tensor([[[2., 2.]], [[-2., -2.]]])
    assert torch.allclose(transformer_block(_block_state(), batched, 0, _mask(1)),
                          torch.tensor([[[3.5, 3.5]], [[-3.5, -3.5]]]), atol=2e-6)


def submit_transformer_block(transformer_block):
    state = _block_state()
    x = torch.tensor([[[4., 3.], [-2., -2.]]])
    mask = _mask(4)[2:, :]
    mugrade.submit(_array(transformer_block(state, x, 3, mask, pos=2, cache=[{}])))
    mugrade.submit(_array(transformer_block(_block_state(), torch.tensor([[[1., 2.]], [[-3., -1.]]]), 0, _mask(1))))
    mugrade.submit(bool(state.calls and all(layer == 3 and pos == 2 and torch.equal(m, mask)
                                           and c == [{}] for layer, m, pos, c in state.calls)))


def _call_state(dtype=torch.float32, num_layers=2):
    calls = []
    def block(x, layer, mask, pos=0, cache=None):
        calls.append((layer, mask.clone(), pos, cache))
        # Noncommuting affine collaborators make omitted/reordered layers visible.
        if layer == 0:
            return x + x.new_tensor([1., -1.])
        return 2 * x + x.new_tensor([-1., 2.])
    return SimpleNamespace(
        params={"embedding": torch.tensor([[3., 4.], [1., -1.], [-4., 3.], [2., 2.]]),
                "output": torch.tensor([[1., 0., 2.], [0., 1., -1.]])},
        dtype=dtype, num_layers=num_layers, buffers={"mask": _mask(7)},
        transformer_block=block, calls=calls,
    )


def test___call__(call):
    state = _call_state()
    tokens = torch.tensor([[0, 1], [2, 3]])
    cache = [{"marker": 4}, {"marker": 9}]
    out = call(state, tokens, pos=2, cache=cache)
    assert out.shape == (2, 2, 3) and out.dtype == torch.float32
    assert torch.allclose(out, torch.tensor([[[1.0834211111068726, 0.9089547395706177, 1.2578874826431274], [1.17669677734375, -0.7844645380973816, 3.1378581523895264]], [[-0.8442215919494629, 1.1345878839492798, -2.823030948638916], [1.17669677734375, 0.7844645380973816, 1.5689289569854736]]]), atol=2e-6, rtol=2e-6)
    assert state.calls and all(pos == 2 and c == cache and torch.equal(m, _mask(7)[2:4, :4])
                               for _, m, pos, c in state.calls)
    state = _call_state(num_layers=0)
    assert torch.allclose(call(state, torch.tensor([[3]])), torch.tensor([[[1., 1., 1.]]]), atol=2e-6)
    half = call(_call_state(torch.float16, num_layers=0), torch.tensor([[1]]))
    assert half.dtype == torch.float16 and torch.equal(half, torch.tensor([[[1., -1., 3.]]], dtype=torch.float16))


def submit___call__(call):
    state = _call_state()
    mugrade.submit(_array(call(state, torch.tensor([[3, 2, 0]]), pos=1, cache=[{}, {}])))
    mugrade.submit(_array(call(_call_state(num_layers=0), torch.tensor([[2], [0]]))))
    mugrade.submit(bool(state.calls and all(pos == 1 and c == [{}, {}] and torch.equal(m, _mask(7)[1:4, :4])
                                           for _, m, pos, c in state.calls)))


class _ScriptedModel:
    """Position-indexed logits only; no attention, cache, or generation algorithm."""
    num_layers = 2
    num_heads = 2
    head_dim = 2
    dtype = torch.float32

    def __init__(self, logits):
        self.logits = torch.as_tensor(logits, dtype=torch.float32)

    def __call__(self, tokens, pos=0, cache=None):
        return self.logits[:, pos:pos + tokens.shape[1]].to(tokens.device).clone()


def _forced_model(sequence):
    # Each row gives the sole possible output token at each absolute input position.
    indices = torch.tensor(sequence)
    logits = torch.full((*indices.shape, 5), -torch.inf)
    logits.scatter_(-1, indices.unsqueeze(-1), 0.)
    return _ScriptedModel(logits)


def _active_tokens(tokens, prompt_length, eos):
    # Only semantically active output is compared. Padding after EOS and a
    # shared batch's early-stop return length are intentionally unspecified.
    result = []
    for row in tokens[:, prompt_length:].tolist():
        if eos is not None and eos in row:
            row = row[:row.index(eos) + 1]
        result.append(row)
    return result


def _sampling_frequencies(generate, temp):
    # Statistical properties permit row-wise, batched, or alternative categorical
    # samplers. They do not prescribe a random-number consumption order.
    batch = 4096
    logits = torch.tensor([0., 1.38629436112, -torch.inf, -torch.inf, -torch.inf])
    model = _ScriptedModel(logits.reshape(1, 1, 5).expand(batch, 1, 5))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(812)
        out = generate(model, torch.full((batch, 1), 3, dtype=torch.long), max_tokens=1, temp=temp, eos=4)
    generated = out[:, 1]
    return float((generated == 1).float().mean()), bool(((generated == 0) | (generated == 1)).all())


def test_generate(generate):
    prompts = torch.tensor([[1, 2], [3, 1]])
    original = prompts.clone()
    out = generate(_forced_model([[0, 3, 2, 4, 1, 0], [0, 4, 1, 2, 0, 3]]),
                   prompts, max_tokens=4, temp=.6, eos=4)
    assert out.dtype == prompts.dtype and out.device == prompts.device and out.shape[0] == 2
    assert torch.equal(out[:, :2], prompts) and torch.equal(prompts, original)
    assert _active_tokens(out, 2, 4) == [[3, 2, 4], [4]]
    empty = generate(_forced_model([[1, 2], [2, 1]]), prompts, max_tokens=0, eos=4)
    assert torch.equal(empty, prompts)
    # An EOS token absent from the continuation must not shorten the budget.
    bounded = generate(_forced_model([[0, 3, 2, 4], [0, 1, 4, 2]]), prompts, max_tokens=3, eos=0)
    assert bounded.tolist() == [[1, 2, 3, 2, 4], [3, 1, 1, 4, 2]]
    other_eos = generate(_forced_model([[2, 0, 1], [1, 3, 0]]),
                         torch.tensor([[1], [2]]), max_tokens=3, eos=torch.tensor([0]))
    assert _active_tokens(other_eos, 1, 0) == [[2, 0], [1, 3, 0]]
    warm, valid_warm = _sampling_frequencies(generate, 1.)
    cool, valid_cool = _sampling_frequencies(generate, .5)
    assert valid_warm and abs(warm - .8) < .04
    assert valid_cool and abs(cool - 16 / 17) < .035


def submit_generate(generate):
    prompts = torch.tensor([[2], [1]])
    out = generate(_forced_model([[1, 4, 3, 0, 2], [2, 1, 4, 3, 0]]), prompts, max_tokens=5, temp=1.3, eos=3)
    mugrade.submit(_active_tokens(out, 1, 3))
    mugrade.submit(generate(_forced_model([[3, 1], [2, 4]]), prompts, max_tokens=2, eos=0).tolist())
    warm, valid_warm = _sampling_frequencies(generate, 2.)
    cool, valid_cool = _sampling_frequencies(generate, .25)
    mugrade.submit(bool(valid_warm and abs(warm - 2 / 3) < .04 and valid_cool and abs(cool - 256 / 257) < .015))
