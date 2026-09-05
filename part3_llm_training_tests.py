"""Student-facing tests for Part 3 - LLM Training.

The training fixtures below are deliberately tiny recorders and scripted
collaborators.  They exercise the public control flow without containing a
tokenizer, model, optimizer, scheduler, or distributed-training solution.
"""

from array import array
from contextlib import contextmanager
import math
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


@contextmanager
def _patched_attributes(obj, **updates):
    """Temporarily replace attributes used by a narrow runtime fixture."""
    previous = {name: getattr(obj, name, _MISSING) for name in updates}
    for name, value in updates.items():
        setattr(obj, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, value)


@contextmanager
def _cpu_cuda(call_shapes=None):
    """Record ``Tensor.cuda`` calls while keeping tiny tests CPU-only."""
    original = torch.Tensor.cuda

    def stay_on_cpu(tensor, *args, **kwargs):
        if call_shapes is not None:
            call_shapes.append(list(tensor.shape))
        return tensor

    torch.Tensor.cuda = stay_on_cpu
    try:
        yield
    finally:
        torch.Tensor.cuda = original


def _write_uint16(path, values):
    with Path(path).open("wb") as handle:
        array("H", values).tofile(handle)


def test_cross_entropy_loss(cross_entropy_loss):
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    labels = torch.tensor([0, 2])
    loss = cross_entropy_loss(logits, labels)

    assert loss.ndim == 0
    assert loss.dtype == torch.float32
    assert torch.isclose(loss, torch.tensor(0.90760601), atol=1e-6)

    shifted = cross_entropy_loss(logits + 10_000.0, labels)
    assert torch.isfinite(shifted)
    assert torch.isclose(shifted, loss, atol=1e-6)

    logits64 = torch.tensor([[0.0, 0.0], [-2.0, 1.0]], dtype=torch.float64)
    loss64 = cross_entropy_loss(logits64, torch.tensor([1, 0]))
    assert loss64.dtype == torch.float64
    assert torch.isclose(
        loss64,
        torch.tensor(1.870867260143105, dtype=torch.float64),
        atol=1e-12,
    )


def submit_cross_entropy_loss(cross_entropy_loss):
    logits = torch.tensor(
        [[1.5, -0.5, 0.25, 2.0], [-1.0, 0.0, 3.0, 1.0], [0.5, 0.5, 0.5, 0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor([3, 1, 2])
    loss = cross_entropy_loss(logits, labels)
    loss.backward()

    mugrade.submit(loss.item())
    mugrade.submit(logits.grad.detach().cpu().numpy())
    mugrade.submit(str(loss.dtype))


def test_Adam_init(Adam_init):
    params = {
        "vector": torch.tensor([1.0, -2.0]),
        "matrix": torch.tensor([[3.0, 4.0], [-1.0, 0.5]]),
    }
    schedule = SimpleNamespace(name="local-schedule")
    opt = SimpleNamespace()
    result = Adam_init(
        opt,
        params,
        schedule,
        betas=(0.8, 0.9),
        eps=1e-7,
        weight_decay=0.25,
    )

    assert result is None
    assert opt.params is params
    assert opt.schedule is schedule
    assert opt.t == 1
    assert opt.betas == (0.8, 0.9)
    assert opt.eps == 1e-7 and opt.weight_decay == 0.25
    assert sorted(opt.u) == ["matrix", "vector"]
    assert sorted(opt.v) == ["matrix", "vector"]
    assert all(torch.equal(opt.u[name], torch.zeros_like(value)) for name, value in params.items())
    assert all(torch.equal(opt.v[name], torch.zeros_like(value)) for name, value in params.items())
    assert all(value.requires_grad for value in params.values())


def submit_Adam_init(Adam_init):
    params = {
        "a": torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64),
        "b": torch.tensor([[4.0], [-3.0]], dtype=torch.float64),
    }
    schedule = SimpleNamespace(tag="submission-schedule")
    opt = SimpleNamespace()
    Adam_init(opt, params, schedule, betas=(0.7, 0.85), eps=2e-6, weight_decay=0.125)

    mugrade.submit([opt.t, list(opt.betas), opt.eps, opt.weight_decay])
    mugrade.submit(
        [[name, list(opt.params[name].shape), opt.params[name].requires_grad]
         for name in sorted(opt.params)]
    )
    mugrade.submit(
        [opt.u[name].detach().cpu().numpy() for name in sorted(opt.u)]
        + [opt.v[name].detach().cpu().numpy() for name in sorted(opt.v)]
    )


def test_step(step):
    class ScheduleRecorder:
        def get_lr(self, current_step):
            return 0.1

    params = {
        "a": torch.tensor([1.0, -2.0], requires_grad=True),
        "b": torch.tensor([0.5], requires_grad=True),
    }
    params["a"].grad = torch.tensor([2.0, -4.0])
    params["b"].grad = torch.tensor([-3.0])
    fixture = SimpleNamespace(
        schedule=ScheduleRecorder(),
        betas=(0.5, 0.75),
        params=params,
        t=1,
        weight_decay=0.2,
        u={name: torch.zeros_like(value) for name, value in params.items()},
        v={name: torch.zeros_like(value) for name, value in params.items()},
        eps=0.0,
    )

    result = step(fixture)

    assert result is None
    assert torch.allclose(params["a"], torch.tensor([0.88, -1.86]), atol=1e-7)
    assert torch.allclose(params["b"], torch.tensor([0.59]), atol=1e-7)
    assert torch.equal(fixture.u["a"], torch.tensor([1.0, -2.0]))
    assert torch.equal(fixture.u["b"], torch.tensor([-1.5]))
    assert torch.equal(fixture.v["a"], torch.tensor([1.0, 4.0]))
    assert torch.equal(fixture.v["b"], torch.tensor([2.25]))
    assert torch.count_nonzero(params["a"].grad) == 0
    assert torch.count_nonzero(params["b"].grad) == 0
    assert fixture.t == 2


def submit_step(step):
    class ScheduleRecorder:
        def get_lr(self, current_step):
            return {1: 0.05, 2: 0.02}[current_step]

    params = {
        "left": torch.tensor([2.0, -1.0], requires_grad=True),
        "right": torch.tensor([-0.5, 3.0], requires_grad=True),
    }
    fixture = SimpleNamespace(
        schedule=ScheduleRecorder(),
        betas=(0.8, 0.9),
        params=params,
        t=1,
        weight_decay=0.1,
        u={name: torch.zeros_like(value) for name, value in params.items()},
        v={name: torch.zeros_like(value) for name, value in params.items()},
        eps=1e-3,
    )

    params["left"].grad = torch.tensor([-1.0, 2.0])
    params["right"].grad = torch.tensor([3.0, -4.0])
    step(fixture)
    params["left"].grad = torch.tensor([0.5, -1.5])
    params["right"].grad = torch.tensor([-2.0, 1.0])
    step(fixture)

    mugrade.submit(torch.stack([params["left"], params["right"]]).detach().cpu().numpy())
    mugrade.submit(
        torch.stack([fixture.u["left"], fixture.u["right"],
                     fixture.v["left"], fixture.v["right"]])
        .detach().cpu().numpy()
    )
    mugrade.submit(
        [fixture.t, params["left"].grad.detach().cpu().tolist(),
         params["right"].grad.detach().cpu().tolist()]
    )


def test_LRSchedule_init(LRSchedule_init):
    schedule = SimpleNamespace()
    result = LRSchedule_init(
        schedule,
        100,
        lr=0.01,
        warmup_steps=10,
        decay_ratio=0.25,
        min_frac=0.2,
    )
    assert result is None
    assert schedule.total_steps == 100
    assert schedule.lr == 0.01
    assert schedule.warmup_steps == 10
    assert schedule.decay_steps == 25
    assert schedule.min_frac == 0.2

    rounded = SimpleNamespace()
    LRSchedule_init(rounded, 7, lr=0.5, warmup_steps=2, decay_ratio=0.4, min_frac=0.05)
    assert rounded.total_steps == 7
    assert rounded.decay_steps == 3
    assert rounded.lr == 0.5 and rounded.min_frac == 0.05


def submit_LRSchedule_init(LRSchedule_init):
    schedule = SimpleNamespace()
    LRSchedule_init(
        schedule,
        37,
        lr=0.003,
        warmup_steps=4,
        decay_ratio=0.35,
        min_frac=0.15,
    )
    defaults = SimpleNamespace()
    LRSchedule_init(defaults, 12)

    mugrade.submit(
        [schedule.total_steps, schedule.lr, schedule.warmup_steps,
         schedule.decay_steps, schedule.min_frac]
    )
    mugrade.submit(
        [defaults.total_steps, defaults.lr, defaults.warmup_steps,
         defaults.decay_steps, defaults.min_frac]
    )
    mugrade.submit([type(schedule.total_steps), type(schedule.decay_steps), type(schedule.lr)])


def test_get_lr(get_lr):
    schedule = SimpleNamespace(
        total_steps=20,
        lr=0.01,
        warmup_steps=4,
        decay_steps=6,
        min_frac=0.2,
    )

    assert math.isclose(get_lr(schedule, 0), 0.002, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(get_lr(schedule, 2), 0.006, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(get_lr(schedule, 4), 0.01, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(get_lr(schedule, 14), 0.01, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(get_lr(schedule, 17), 0.006, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(get_lr(schedule, 20), 0.002, rel_tol=0.0, abs_tol=1e-12)


def submit_get_lr(get_lr):
    schedule = SimpleNamespace(
        total_steps=30,
        lr=0.003,
        warmup_steps=3,
        decay_steps=9,
        min_frac=0.1,
    )

    mugrade.submit([get_lr(schedule, step) for step in (0, 1, 2)])
    mugrade.submit([get_lr(schedule, step) for step in (3, 12, 21)])
    mugrade.submit([get_lr(schedule, step) for step in (22, 26, 30)])


def test_train_llm(train_llm):
    models = []
    tokenizers = []
    schedules = []
    optimizers = []
    wandb_runs = []
    cuda_shapes = []

    class TinyModel:
        def __init__(self, config):
            self.config = config
            self.params = {"weight": torch.tensor([2.0], requires_grad=True)}
            self.buffers = {"marker": torch.tensor([7.0, 8.0])}
            self.calls = []
            models.append(self)

        def __call__(self, tokens):
            self.calls.append(tokens.detach().clone())
            return self.params["weight"].reshape(1, 1, 1).expand(
                tokens.shape[0], tokens.shape[1], 1
            )

    class TinyTokenizer:
        def __init__(self, filename):
            self.filename = filename
            self.decode_calls = []
            tokenizers.append(self)

        def decode(self, tokens):
            self.decode_calls.append(list(tokens))
            return "abcd"

    class ScheduleRecorder:
        def __init__(self, total_steps, lr):
            self.total_steps = total_steps
            self.lr = lr
            schedules.append(self)

        def get_lr(self, step):
            return self.lr

    class OptimizerRecorder:
        def __init__(self, params, schedule, weight_decay=0.0):
            self.params = params
            self.schedule = schedule
            self.weight_decay = weight_decay
            self.t = 1
            self.grads = []
            optimizers.append(self)

        def step(self):
            self.grads.append(self.params["weight"].grad.detach().clone())
            self.t += 1

    class RunRecorder:
        def __init__(self, project, config):
            self.project = project
            self.config = config
            self.logs = []
            wandb_runs.append(self)

        def log(self, values):
            self.logs.append(values)

    wandb = SimpleNamespace(init=lambda project, config: RunRecorder(project, config))

    def scripted_loss(logits, labels):
        return logits.mean() + labels.float().mean() * 0.0

    with TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "tokens.bin"
        _write_uint16(data_path, [11, 12, 13])
        config = {
            "tokenizer": "tiny.bpe",
            "token_multiple": 4,
            "batch_size": 1,
            "seq_len": 2,
            "lr": 0.05,
            "weight_decay": 0.125,
            "data_mix": {str(data_path): 1.0},
        }
        with (
            _patched_globals(
                train_llm,
                LLM=TinyModel,
                BPE=TinyTokenizer,
                LRSchedule=ScheduleRecorder,
                Adam=OptimizerRecorder,
                cross_entropy_loss=scripted_loss,
                wandb=wandb,
            ),
            _cpu_cuda(cuda_shapes),
        ):
            result = train_llm(config, log=True)

    assert result is models[0]
    assert config["total_params"] == 1
    assert config["total_tokens"] == 4
    assert config["total_steps"] == 2
    assert tokenizers[0].filename == "tiny.bpe"
    assert tokenizers[0].decode_calls == [[12, 13]]
    assert len(models[0].calls) == 1
    assert torch.equal(models[0].calls[0], torch.tensor([[11, 12]]))
    assert optimizers[0].weight_decay == 0.125 and optimizers[0].t == 2
    assert torch.equal(optimizers[0].grads[0], torch.tensor([1.0]))
    assert (schedules[0].total_steps, schedules[0].lr) == (2, 0.05)
    assert cuda_shapes == [[1], [2], [1, 3]]
    assert len(wandb_runs) == 1 and wandb_runs[0].project == "llm_speedrun"
    assert len(wandb_runs[0].logs) == 1
    assert set(wandb_runs[0].logs[0]) == {"step", "tokens", "loss", "bpb", "lr", "tok_per_sec"}


def submit_train_llm(train_llm):
    models = []
    tokenizers = []
    schedules = []
    optimizers = []
    cuda_shapes = []

    class TinyModel:
        def __init__(self, config):
            self.params = {"weight": torch.tensor([1.25], requires_grad=True)}
            self.buffers = {"marker": torch.tensor([4.0])}
            self.calls = []
            models.append(self)

        def __call__(self, tokens):
            self.calls.append(tokens.detach().clone())
            return self.params["weight"].reshape(1, 1, 1).expand(
                tokens.shape[0], tokens.shape[1], 1
            )

    class TinyTokenizer:
        def __init__(self, filename):
            self.filename = filename
            self.decode_calls = []
            tokenizers.append(self)

        def decode(self, tokens):
            self.decode_calls.append(list(tokens))
            return "xyzxyz"

    class ScheduleRecorder:
        def __init__(self, total_steps, lr):
            self.total_steps = total_steps
            self.lr = lr
            schedules.append(self)

        def get_lr(self, step):
            return self.lr

    class OptimizerRecorder:
        def __init__(self, params, schedule, weight_decay=0.0):
            self.params = params
            self.schedule = schedule
            self.weight_decay = weight_decay
            self.t = 1
            optimizers.append(self)

        def step(self):
            self.t += 1

    def scripted_loss(logits, labels):
        return logits.mean() + labels.float().mean() * 0.0

    with TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "tokens.bin"
        _write_uint16(data_path, [21, 22, 23, 24])
        config = {
            "tokenizer": "submit.bpe",
            "token_multiple": 6,
            "batch_size": 1,
            "seq_len": 3,
            "lr": 0.02,
            "weight_decay": 0.05,
            "data_mix": {str(data_path): 1.0},
        }
        with (
            _patched_globals(
                train_llm,
                LLM=TinyModel,
                BPE=TinyTokenizer,
                LRSchedule=ScheduleRecorder,
                Adam=OptimizerRecorder,
                cross_entropy_loss=scripted_loss,
            ),
            _cpu_cuda(cuda_shapes),
        ):
            result = train_llm(config, log=False)

    mugrade.submit(result.calls[0].detach().cpu().numpy())
    mugrade.submit(
        [config["total_params"], config["total_tokens"], config["total_steps"],
         optimizers[0].t, tokenizers[0].decode_calls]
    )
    mugrade.submit(
        [tokenizers[0].filename, schedules[0].total_steps, schedules[0].lr,
         optimizers[0].weight_decay, cuda_shapes]
    )


def test_train_llm_distributed(train_llm_distributed):
    models = []
    tokenizers = []
    schedules = []
    optimizers = []
    set_devices = []
    cuda_shapes = []
    wandb_runs = []

    class TinyModel:
        def __init__(self, config):
            self.params = {"weight": torch.tensor([1.5], requires_grad=True)}
            self.buffers = {"marker": torch.tensor([9.0, 10.0])}
            self.calls = []
            models.append(self)

        def __call__(self, tokens):
            self.calls.append(tokens.detach().clone())
            return self.params["weight"].reshape(1, 1, 1).expand(
                tokens.shape[0], tokens.shape[1], 1
            )

    class TinyTokenizer:
        def __init__(self, filename):
            self.filename = filename
            self.decode_calls = []
            tokenizers.append(self)

        def decode(self, tokens):
            self.decode_calls.append(list(tokens))
            return "wxyz"

    class ScheduleRecorder:
        def __init__(self, total_steps, lr):
            self.total_steps = total_steps
            self.lr = lr
            schedules.append(self)

        def get_lr(self, step):
            return self.lr

    class OptimizerRecorder:
        def __init__(self, params, schedule, weight_decay=0.0):
            self.params = params
            self.schedule = schedule
            self.weight_decay = weight_decay
            self.t = 1
            self.grads = []
            optimizers.append(self)

        def step(self):
            self.grads.append(self.params["weight"].grad.detach().clone())
            self.t += 1

    class NCCLRecorder:
        def __init__(self):
            self.init_calls = []
            self.broadcasts = []
            self.reductions = []

        def init_rank(self, world_size, uid, rank):
            self.init_calls.append([world_size, uid, rank])
            return "tiny-comm"

        def broadcast(self, tensors, root, comms):
            self.broadcasts.append([len(tensors), root, list(comms)])

        def all_reduce(self, tensors, comms):
            self.reductions.append([list(tensors[0].shape), list(comms)])

    class RunRecorder:
        def __init__(self, project, config):
            self.logs = []
            wandb_runs.append(self)

        def log(self, values):
            self.logs.append(values)

    nccl = NCCLRecorder()
    wandb = SimpleNamespace(init=lambda project, config: RunRecorder(project, config))

    def scripted_loss(logits, labels):
        return logits.mean() + labels.float().mean() * 0.0

    with TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "tokens.bin"
        _write_uint16(data_path, [1, 2, 3, 4, 5, 6])
        config = {
            "tokenizer": "distributed.bpe",
            "token_multiple": 8,
            "batch_size": 2,
            "seq_len": 2,
            "lr": 0.04,
            "weight_decay": 0.2,
            "data_mix": {str(data_path): 1.0},
            "num_gpus": 2,
        }
        with (
            _patched_globals(
                train_llm_distributed,
                LLM=TinyModel,
                BPE=TinyTokenizer,
                LRSchedule=ScheduleRecorder,
                Adam=OptimizerRecorder,
                cross_entropy_loss=scripted_loss,
                wandb=wandb,
            ),
            _patched_attributes(
                torch.cuda,
                set_device=lambda rank: set_devices.append(rank),
                nccl=nccl,
            ),
            _cpu_cuda(cuda_shapes),
        ):
            result = train_llm_distributed(0, "uid-local", config, log=True)

    assert result is models[0]
    assert set_devices == [0]
    assert nccl.init_calls == [[2, "uid-local", 0]]
    assert nccl.broadcasts == [[1, 0, ["tiny-comm"]]]
    assert nccl.reductions == [
        [[1], ["tiny-comm"]],
        [[], ["tiny-comm"]],
        [[], ["tiny-comm"]],
    ]
    assert torch.equal(models[0].calls[0], torch.tensor([[1, 2]]))
    assert tokenizers[0].decode_calls == [[2, 3]]
    assert torch.equal(optimizers[0].grads[0], torch.tensor([0.5]))
    assert optimizers[0].t == 2 and optimizers[0].weight_decay == 0.2
    assert (config["total_params"], config["total_tokens"], config["total_steps"]) == (1, 8, 2)
    assert cuda_shapes == [[1], [2], [], [1, 3]]
    assert len(wandb_runs) == 1 and len(wandb_runs[0].logs) == 1
    assert set(wandb_runs[0].logs[0]) == {"step", "tokens", "loss", "bpb", "lr", "tok_per_sec"}


def submit_train_llm_distributed(train_llm_distributed):
    models = []
    tokenizers = []
    schedules = []
    optimizers = []
    set_devices = []
    cuda_shapes = []

    class TinyModel:
        def __init__(self, config):
            self.params = {"weight": torch.tensor([0.75], requires_grad=True)}
            self.buffers = {"marker": torch.tensor([6.0])}
            self.calls = []
            models.append(self)

        def __call__(self, tokens):
            self.calls.append(tokens.detach().clone())
            return self.params["weight"].reshape(1, 1, 1).expand(
                tokens.shape[0], tokens.shape[1], 1
            )

    class TinyTokenizer:
        def __init__(self, filename):
            self.filename = filename
            self.decode_calls = []
            tokenizers.append(self)

        def decode(self, tokens):
            self.decode_calls.append(list(tokens))
            return "rank-one"

    class ScheduleRecorder:
        def __init__(self, total_steps, lr):
            self.total_steps = total_steps
            self.lr = lr
            schedules.append(self)

        def get_lr(self, step):
            return self.lr

    class OptimizerRecorder:
        def __init__(self, params, schedule, weight_decay=0.0):
            self.params = params
            self.schedule = schedule
            self.weight_decay = weight_decay
            self.t = 1
            optimizers.append(self)

        def step(self):
            self.t += 1

    class NCCLRecorder:
        def __init__(self):
            self.trace = []

        def init_rank(self, world_size, uid, rank):
            self.trace.append(["init", world_size, uid, rank])
            return "submit-comm"

        def broadcast(self, tensors, root, comms):
            self.trace.append(["broadcast", len(tensors), root, list(comms)])

        def all_reduce(self, tensors, comms):
            self.trace.append(["reduce", list(tensors[0].shape), list(comms)])

    nccl = NCCLRecorder()

    def scripted_loss(logits, labels):
        return logits.mean() + labels.float().mean() * 0.0

    with TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "tokens.bin"
        _write_uint16(data_path, [31, 32, 33, 41, 42, 43])
        config = {
            "tokenizer": "rank-one.bpe",
            "token_multiple": 8,
            "batch_size": 2,
            "seq_len": 2,
            "lr": 0.03,
            "weight_decay": 0.15,
            "data_mix": {str(data_path): 1.0},
            "num_gpus": 2,
        }
        with (
            _patched_globals(
                train_llm_distributed,
                LLM=TinyModel,
                BPE=TinyTokenizer,
                LRSchedule=ScheduleRecorder,
                Adam=OptimizerRecorder,
                cross_entropy_loss=scripted_loss,
            ),
            _patched_attributes(
                torch.cuda,
                set_device=lambda rank: set_devices.append(rank),
                nccl=nccl,
            ),
            _cpu_cuda(cuda_shapes),
        ):
            result = train_llm_distributed(1, "uid-submit", config, log=True)

    mugrade.submit([result is None, set_devices, models[0].calls[0].tolist()])
    mugrade.submit(nccl.trace)
    mugrade.submit(
        [config["total_params"], config["total_tokens"], config["total_steps"],
         optimizers[0].t, tokenizers[0].decode_calls, cuda_shapes]
    )
