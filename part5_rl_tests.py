"""Small, deterministic CPU tests for Part 5.

Fixtures below are call recorders and scripted collaborators, not implementations
of the graded tokenizer, model, optimizer, loss, generation, or training code.
No downloads, CUDA hardware, or pretrained weights are needed for these tests.
"""
import contextlib
import io
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mugrade
import numpy as np
import torch


def _array(x):
    return x.detach().cpu().numpy().copy()


@contextlib.contextmanager
def _environment(fn, **replacements):
    # Restore the notebook namespace and CUDA methods even when an assertion fails.
    with patch.dict(fn.__globals__, replacements), \
         patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self), \
         patch.object(torch.cuda, 'set_device'), contextlib.redirect_stdout(io.StringIO()):
        yield


def test_cross_entropy_loss(cross_entropy_loss):
    x = torch.tensor([[[0., 0.], [math.log(3), 0.], [0., 0.]]], requires_grad=True)
    y = torch.tensor([[0, 1, 0]])
    loss = cross_entropy_loss(x, y, mask_token=0)
    assert torch.allclose(loss, torch.tensor(math.log(4)), atol=1e-6)
    loss.backward()
    assert torch.allclose(x.grad, torch.tensor([[[0., 0.], [.75, -.75], [0., 0.]]]), atol=1e-6)
    x2 = torch.zeros(2, 2, 3, requires_grad=True)
    y2 = torch.tensor([[1, 0], [2, 1]])
    w = torch.tensor([[[2.]], [[-1.]]])
    assert abs(cross_entropy_loss(x2, y2, mask_token=0, weights=w).item()) < 1e-6
    assert torch.allclose(cross_entropy_loss(x2 + 10000, y2), torch.tensor(math.log(3)), atol=1e-6)
    assert torch.allclose(cross_entropy_loss(x2, y2, weights=0.), torch.tensor(0.))
    assert torch.allclose(cross_entropy_loss(x2, y2, weights=2.), torch.tensor(2*math.log(3)), atol=1e-6)


def submit_cross_entropy_loss(cross_entropy_loss):
    x = torch.tensor([[[2., -1., 0.], [0., 2., 1.]], [[1., 0., -2.], [3., 1., 2.]]], requires_grad=True)
    y = torch.tensor([[2, 1], [0, 2]])
    loss = cross_entropy_loss(x, y, mask_token=2)
    mugrade.submit(loss.item())
    loss.backward()
    mugrade.submit(_array(x.grad))
    mugrade.submit(cross_entropy_loss(x.detach(), y, weights=torch.tensor([[[-.5]], [[1.5]]])).item())
    mugrade.submit(cross_entropy_loss(x.detach()+500, y, mask_token=1, weights=2.).item())


def test___init__(init):
    p = {'a': torch.tensor([1., -2.]), 'b': torch.tensor([[3.]], dtype=torch.float64)}
    s = SimpleNamespace()
    assert init(s, p, lr=.02, betas=(.5, .75), eps=.03, weight_decay=.1) is None
    assert s.params is p and all(v.requires_grad for v in p.values())
    assert (s.lr, s.betas, s.eps, s.weight_decay, s.t) == (.02, (.5, .75), .03, .1, 1)
    assert all(torch.equal(s.u[k], torch.zeros_like(v)) and torch.equal(s.v[k], torch.zeros_like(v)) for k,v in p.items())
    assert s.u['b'].dtype == torch.float64 and s.u['a'].data_ptr() != s.v['a'].data_ptr()
    d = SimpleNamespace(); init(d, {'x': torch.tensor([4.])})
    assert (d.lr, d.betas, d.eps, d.weight_decay) == (.001, (.9, .95), 1e-5, 0.)


def submit___init__(init):
    s = SimpleNamespace(); p = {'w': torch.tensor([[2., 3.], [-1., 0.]])}
    mugrade.submit(init(s, p, lr=.04, betas=(.2, .6), eps=.01, weight_decay=.2) is None)
    mugrade.submit([s.lr, *s.betas, s.eps, s.weight_decay, s.t])
    mugrade.submit(_array(s.u['w']))
    mugrade.submit(_array(s.v['w']))
    mugrade.submit(p['w'].requires_grad and s.params is p)


def _adam_state(submit=False):
    p = torch.tensor([1., -2.] if not submit else [3., -.5], requires_grad=True)
    return SimpleNamespace(params={'w':p}, u={'w':torch.zeros(2)}, v={'w':torch.zeros(2)},
                           t=1, lr=.1 if not submit else .03, betas=(.5, .75), eps=.1, weight_decay=.2)


def test_step(step):
    s = _adam_state(); s.params['w'].grad = torch.tensor([2., -1.])
    assert step(s) is None
    assert torch.allclose(s.params['w'], torch.tensor([.8847619, -1.8690909]), atol=1e-6)
    assert torch.equal(s.u['w'], torch.tensor([1., -.5])) and torch.equal(s.v['w'], torch.tensor([1., .25]))
    assert s.t == 2 and torch.equal(s.params['w'].grad, torch.zeros(2))
    s.params['w'].grad = torch.tensor([-1., 3.]); step(s)
    assert torch.allclose(s.params['w'], torch.tensor([.8670667, -1.8994491]), atol=1e-6)
    assert torch.equal(s.u['w'], torch.tensor([0., 1.25])) and torch.equal(s.v['w'], torch.tensor([1., 2.4375]))
    assert s.t == 3 and torch.equal(s.params['w'].grad, torch.zeros(2))


def submit_step(step):
    s = _adam_state(True); s.params['w'].grad = torch.tensor([-.5, 2.]); step(s)
    mugrade.submit(_array(s.params['w']))
    mugrade.submit(_array(s.u['w']))
    mugrade.submit(_array(s.v['w']))
    s.params['w'].grad = torch.tensor([0., -1.]); step(s)
    mugrade.submit(_array(s.params['w']))
    mugrade.submit([s.t, s.params['w'].grad.abs().sum().item()])


def test_tool_eval(tool_eval):
    assert tool_eval('2 + 3 * 4') == 14
    assert tool_eval('10 / 3') == 3.33
    assert tool_eval('-9 / 2') == -4.5
    assert tool_eval('1 / 0') == ''
    assert tool_eval('not valid arithmetic') == ''
    assert isinstance(tool_eval('8 / 2'), int)


def submit_tool_eval(tool_eval):
    mugrade.submit(tool_eval('(7 - 11) ** 2'))
    mugrade.submit(tool_eval('2 / 7'))
    mugrade.submit(tool_eval('3 +'))
    mugrade.submit(type(tool_eval('15 / 3')))


def test_extract_answer(extract_answer):
    assert extract_answer('work <ANSWER>42</ANSWER> tail') == 42
    assert extract_answer('<ANSWER> -17 </ANSWER>') == -17
    assert extract_answer('<ANSWER>3.5</ANSWER>') is None
    assert extract_answer('<ANSWER>8') is None
    assert extract_answer('missing') is None
    assert extract_answer('<ANSWER>0</ANSWER><ANSWER>9</ANSWER>') == 0


def submit_extract_answer(extract_answer):
    mugrade.submit(extract_answer('<THINK>x</THINK><ANSWER>81</ANSWER>'))
    mugrade.submit(extract_answer('<ANSWER>-203</ANSWER>'))
    mugrade.submit(extract_answer('<ANSWER>hello</ANSWER>') is None)
    mugrade.submit(extract_answer('</ANSWER>') is None)


def _answer_stub(text):
    # Exact-case collaborator; answer extraction is tested separately.
    return {'one':1, 'two':2, 'bad':None, 'zero':0}[text]


def test_grade(grade):
    with _environment(grade, extract_answer=_answer_stub):
        assert grade(['one', 'two', 'bad'], ['one', 'one', 'two']) == ([True, False, False], [True, True, False])
        assert grade(['zero'], ['zero']) == ([True], [True])
        assert grade([], []) == ([], [])


def submit_grade(grade):
    with _environment(grade, extract_answer=_answer_stub):
        mugrade.submit(grade(['two', 'bad', 'zero'], ['one', 'two', 'zero'])[0])
        mugrade.submit(grade(['bad', 'one', 'zero'], ['one', 'two', 'one'])[1])
        mugrade.submit(grade(['two', 'one'], ['two', 'one']))


class _ScriptedModel:
    """Forced next-token logits, plus a record of cache and token slices."""
    num_heads = 1
    head_dim = 2
    num_layers = 2
    dtype = torch.float32

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.cache = None

    def __call__(self, tokens, pos=0, cache=None):
        self.calls.append((tokens.tolist(), pos, torch.is_grad_enabled()))
        if self.cache is None:
            self.cache = cache
            self.initial_cache = [(tuple(c['k'].shape), c['k'].dtype, c['k'].count_nonzero().item()) for c in cache]
        else:
            assert cache is self.cache, 'Generation must reuse its cache.'
        out = torch.full((*tokens.shape, 12), -1000.)
        for i,tok in enumerate(self.script[len(self.calls)-1]):
            out[i, :, tok] = 1000.
        return out


def _generation(generate, variant=0):
    script = [[4, 9], [9, 6], [7, 9], [9, 9]] if variant == 0 else [[5 if variant==1 else 4], [3], [9], [9], [9], [9]]
    m = _ScriptedModel(script)
    prompts = torch.tensor([[1, 2], [2, 1]]) if variant == 0 else torch.tensor([[1, 2, 4] if variant==1 else [8,2,7]])
    decoded, encoded, evaluated = [], [], []
    def decode(t): decoded.append(t); return '2+2'
    def encode(t): encoded.append(t); return [6, 7]
    def tool(t): evaluated.append(t); return 4
    bpe = SimpleNamespace(vocab=['<MASK/>', 'a', '<TOOL>', '</TOOL>', 'b', 'c', 'd', 'e', 'f', 'eos', 'g', 'h'], decode=decode, encode=encode)
    out = generate(m, prompts, max_tokens=4 if variant==0 else 6, eos=9,
                   bpe=None if variant==0 else bpe, tool_eval=None if variant==0 else tool)
    return out, m, decoded, encoded, evaluated


def test_generate(generate):
    out, m, *_ = _generation(generate)
    assert out.tolist() == [[1,2,4,9,0,0], [2,1,9,0,0,0]]
    assert [c[1] for c in m.calls] == [0,2] and [len(c[0][0]) for c in m.calls] == [2,1]
    assert all(not c[2] for c in m.calls) and len(m.initial_cache) == 2
    assert all(c == ((2,1,6,2), torch.float32, 0) for c in m.initial_cache)
    out, m, dec, enc, ev = _generation(generate, 1)
    assert out.tolist() == [[1,2,4,5,3,6,7,9,0]]
    assert dec == [[4,5]] and enc == ['<RESPONSE>4</RESPONSE>'] and ev == ['2+2']
    assert [c[1] for c in m.calls] == [0,3,4,5,6]
    z = _ScriptedModel([])
    assert generate(z, torch.tensor([[2]]), max_tokens=0, eos=9).tolist() == [[2]] and not z.calls


def submit_generate(generate):
    m = _ScriptedModel([[7], [4], [9]])
    out = generate(m, torch.tensor([[2,3,1]]), max_tokens=5, eos=9)
    mugrade.submit(_array(out))
    mugrade.submit([c[1] for c in m.calls])
    mugrade.submit([list(c[0]) for c in m.initial_cache])
    out, m, dec, enc, ev = _generation(generate, 2)
    mugrade.submit(_array(out))
    mugrade.submit([dec, enc, ev])


def _eval_examples_case(fn, variant=0):
    calls = {}
    examples = [{'tokens':[4,1,7,0,0], 'text':'first'}, {'tokens':[5,6,1,8,0], 'text':'second'}]
    if variant: examples.reverse()
    k = 2 if not variant else 3
    def generate(prompts, **kw):
        calls['prompts'] = prompts.tolist(); calls['kwargs'] = kw
        return torch.tensor([[i+10,0] for i in range(len(prompts))])
    def decode(tokens): return f'<MASK/>response{tokens[0]}<MASK/>'
    def grading(responses, answers):
        calls['responses'] = responses; calls['answers'] = answers
        return [True,False]*k, [True,True]*k
    bpe = SimpleNamespace(vocab=['<MASK/>','<THINK>','</ANSWER>'], decode=decode)
    tool = object()
    with _environment(fn, grade=grading):
        result = fn(SimpleNamespace(generate=generate), examples, bpe, tool, k, 9 if not variant else 11)
    calls['forwarded'] = calls['kwargs']['tool_eval'] is tool and calls['kwargs']['bpe'] is bpe
    return result, calls


def test_eval_examples(eval_examples):
    result, c = _eval_examples_case(eval_examples)
    assert c['prompts'] == [[0,4,1],[5,6,1]]*2
    assert c['answers'] == ['first','second']*2
    assert c['responses'] == ['response10','response11','response12','response13']
    assert c['kwargs']['max_tokens'] == 6 and c['kwargs']['eos'] == 2 and c['forwarded']
    assert result[0].tolist() == [[10,0],[11,0],[12,0],[13,0]]
    assert result[1] == ([True,False,True,False], [True]*4)


def submit_eval_examples(eval_examples):
    result, c = _eval_examples_case(eval_examples, 1)
    mugrade.submit(c['prompts'])
    mugrade.submit(c['answers'])
    mugrade.submit(c['responses'])
    mugrade.submit([c['kwargs']['max_tokens'], c['kwargs']['eos'], c['forwarded']])
    mugrade.submit(_array(result[0]))
    mugrade.submit(result[1])


@contextlib.contextmanager
def _files(size=5):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = [{'tokens':[i+1,2,0,0,0], 'text':str(i)} for i in range(size)]
        (root/'data.json').write_text(json.dumps(rows))
        (root/'model.json').write_text(json.dumps({'fixture':True}))
        yield dict(llm_config=str(root/'model.json'), datafile=str(root/'data.json'),
                   tokenizer='fixture.bpe', infilename='input.pt', filename='output.pt',
                   seq_len=4, batch_size=4, lr=.02, num_gpus=1, epochs=1, k=2,
                   sft_loss_weight=.25)


def _training(fn, *, rl=False, rank=0, gpus=1, epochs=1, shift=0,
              k=2, size=5, batch=4, sft=.25, equal=False, offset=0, uneven=False):
    """Trace orchestration with a differentiable toy and a scripted loss oracle.

    The loss oracle is deliberately NOT cross entropy. This isolates training's
    weighting/backward/communication behavior from the separately graded loss.
    """
    r = {key:[] for key in ['inputs','losses','grads','reduce','samples','shifts','save','load','init','opt']}
    class Model:
        def __init__(self, config):
            r['model_config'] = config
            self.params = {'p':torch.tensor(2.), 'q':torch.tensor(3.)}
            self.buffers = {'buffer':torch.tensor([1.])}
        def load(self, path): r['load'].append(path)
        def save(self, path): r['save'].append(path)
        def __call__(self, tokens):
            r['inputs'].append(tokens.tolist())
            return tokens.float().unsqueeze(-1)*self.params['p'] + self.params['q']
    class Optimizer:
        def __init__(self, params, lr):
            self.params = params; r['opt'].append(lr)
            for p in params.values(): p.requires_grad_()
        def step(self):
            r['grads'].append([p.grad.item() for p in self.params.values()])
            for p in self.params.values(): p.grad.zero_()
    def loss(logits, y, mask_token=-1, weights=1):
        w = weights.detach().flatten().tolist() if isinstance(weights, torch.Tensor) else weights
        r['losses'].append({'targets':y.tolist(), 'mask':mask_token, 'weights':w})
        scale = weights.flatten()[0] if isinstance(weights, torch.Tensor) else weights
        return logits.sum()*scale
    def samples(llm, data, bpe, tool, count, seq_len):
        ids = [int(e['text']) for e in data]
        r['samples'].append([ids, count, seq_len])
        generated = torch.tensor([[j+1+offset, sample+1, 0, 0, 0] for sample in range(count) for j in ids])
        correct = [False if equal else (sample+j)%count==0 and (not uneven or j%2==0) for sample in range(count) for j in ids]
        formatted = [True if equal else (sample+2*j)%3!=0 for sample in range(count) for j in ids]
        return generated, (correct, formatted)
    def randint(a,b): r['shifts'].append([a,b]); return shift
    def init_rank(n, uid, worker): r['init'].append([n,uid,worker]); return 'comm'
    def reduce(xs, comms):
        assert comms == ['comm']
        r['reduce'].append([x.detach().flatten().tolist() for x in xs])
        with torch.no_grad():
            for x in xs: x.mul_(gpus)
    with _files(size) as config:
        config.update(num_gpus=gpus, epochs=epochs, k=k, batch_size=batch, sft_loss_weight=sft)
        with _environment(fn, LLM=Model, BPE=lambda p:SimpleNamespace(vocab=['<MASK/>']),
                          Adam=Optimizer, cross_entropy_loss=loss, eval_examples=samples, tool_eval=object()), \
             patch.object(fn.__globals__['random'], 'randint', randint), \
             patch.object(torch.cuda.nccl, 'init_rank', init_rank), \
             patch.object(torch.cuda.nccl, 'all_reduce', reduce):
            r['return'] = fn(rank, 'uid', config)
    return r


def test_finetune_llm_sft(finetune_llm_sft):
    r = _training(finetune_llm_sft)
    assert r['return'] is None
    assert r['inputs'] == [[[1,2,0,0],[2,2,0,0],[3,2,0,0],[4,2,0,0]], [[5,2,0,0]]]
    assert [l['targets'] for l in r['losses']] == [[[2,0,0,0]]*4, [[2,0,0,0]]]
    assert all(l['mask']==0 and l['weights']==1 for l in r['losses'])
    assert r['grads'] == [[18.,16.],[7.,4.]]
    assert r['save'] == ['output.pt'] and r['load'] == ['input.pt']
    assert r['init'] == [[1,'uid',0]] and r['opt'] == [.02]
    assert r['shifts'] == [[0,1]]*5
    assert len(r['reduce']) == 6
    r = _training(finetune_llm_sft, rank=1, gpus=2, epochs=2, shift=1, size=6)
    assert r['inputs'] == [[[0,3,2,0],[0,4,2,0]], [[0,6,2,0]]]*2
    assert [l['targets'] for l in r['losses']] == [[[3,2,0,0],[4,2,0,0]], [[6,2,0,0]]]*2
    assert r['grads'] == [[11.,8.],[8.,4.]]*2
    assert r['save'] == [] and r['init'] == [[2,'uid',1]]
    assert len(r['reduce']) == 12 and len(r['shifts']) == 6
    r = _training(finetune_llm_sft, epochs=0)
    assert not r['inputs'] and not r['grads'] and r['save'] == ['output.pt']


def submit_finetune_llm_sft(finetune_llm_sft):
    r = _training(finetune_llm_sft, shift=1, size=3, batch=2, epochs=2)
    mugrade.submit(r['inputs'])
    mugrade.submit([l['targets'] for l in r['losses']])
    mugrade.submit([l['mask'] for l in r['losses']])
    mugrade.submit(r['grads'])
    mugrade.submit(r['shifts'])
    mugrade.submit(r['reduce'])
    mugrade.submit([r['load'],r['save'],r['opt']])
    r = _training(finetune_llm_sft, rank=1, gpus=3, size=9, batch=6)
    mugrade.submit(r['inputs'])
    mugrade.submit(r['grads'])
    mugrade.submit(r['init'])
    mugrade.submit(r['save'])
    r = _training(finetune_llm_sft, epochs=0)
    mugrade.submit([len(r['grads']),r['save']])


def test_finetune_llm_rl(finetune_llm_rl):
    r = _training(finetune_llm_rl, rl=True, size=3, batch=4)
    assert len(r['inputs']) == 4 and len(r['grads']) == 2
    assert r['samples'] == [[[0,1],2,5], [[2],2,5]]
    assert r['inputs'][0] == [[1,2,0,0],[2,2,0,0]]
    assert r['inputs'][1] == [[1,1,0,0],[2,1,0,0],[1,2,0,0],[2,2,0,0]]
    assert [l['mask'] for l in r['losses']] == [0]*4
    assert r['losses'][0]['weights'] == 1 and r['losses'][2]['weights'] == 1
    assert np.allclose(r['losses'][1]['weights'], [.7070963,-.7070968,-.7070963,.7070968], atol=2e-6)
    assert np.allclose(r['grads'][0], [10.2351556,13.3135414], atol=3e-5)
    assert np.allclose(r['grads'][1], [7.6138759,6.6567783], atol=3e-5)
    assert len(r['reduce']) == 6 and r['save'] == ['output.pt']
    assert r['load'] == ['input.pt'] and r['opt'] == [.02] and r['init'] == [[1,'uid',0]]
    r = _training(finetune_llm_rl, rl=True, rank=1, gpus=2, size=4, batch=8, k=2, epochs=2, shift=1, equal=True, sft=.5)
    assert r['samples'] == [[[2,3],2,5]]*2
    assert r['losses'][1]['weights'] == [0.]*4 and r['losses'][3]['weights'] == [0.]*4
    assert r['grads'] == [[5.5,4.]]*2
    assert r['inputs'][0] == [[0,3,2,0],[0,4,2,0]] and r['save'] == []
    assert len(r['reduce']) == 6 and r['init'] == [[2,'uid',1]]
    r = _training(finetune_llm_rl, rl=True, size=2, batch=6, k=3, sft=0.)
    assert r['samples'] == [[[0,1],3,5]]
    assert np.allclose(r['losses'][1]['weights'], [1.1546795,-.5346192,-.5773397,-.6190328,-.5773397,1.1536521], atol=3e-5)
    assert np.allclose(r['grads'][0], [24.2482700,27.7123089], atol=5e-5)
    r = _training(finetune_llm_rl, rl=True, size=2, batch=6, k=3, uneven=True)
    assert np.allclose(r['losses'][1]['weights'], [1.1546794,.5771503,-.5773397,-1.1543006,-.5773397,.5771503], atol=3e-5)
    r = _training(finetune_llm_rl, rl=True, epochs=0)
    assert not r['grads'] and not r['samples'] and r['save'] == ['output.pt']


def submit_finetune_llm_rl(finetune_llm_rl):
    r = _training(finetune_llm_rl, rl=True, size=4, batch=9, k=3, sft=.4, shift=1, offset=2, uneven=True)
    mugrade.submit(r['inputs'])
    mugrade.submit(r['samples'])
    mugrade.submit([l['targets'] for l in r['losses']])
    mugrade.submit([l['mask'] for l in r['losses']])
    mugrade.submit(np.array(r['losses'][1]['weights']))
    mugrade.submit(np.array(r['losses'][3]['weights']))
    mugrade.submit(np.array(r['grads']))
    mugrade.submit(r['shifts'])
    mugrade.submit([r['load'],r['save'],r['opt']])
    r = _training(finetune_llm_rl, rl=True, rank=1, gpus=2, size=6, batch=8, k=2, epochs=2, sft=0., equal=True)
    mugrade.submit(r['samples'])
    mugrade.submit(r['grads'])
    mugrade.submit(r['save'])
    mugrade.submit(r['init'])
    mugrade.submit(r['reduce'])


def _dataset_eval(fn, rank=0, gpus=1, size=5, k=2, batch=4):
    calls = []
    class Model:
        def __init__(self, cfg): self.params={'p':torch.tensor(1.)}; self.buffers={'b':torch.tensor(0.)}
        def load(self, path): calls.append(['load',path])
    def examples(llm, data, bpe, tool, count, seq_len):
        ids = [int(e['text']) for e in data]; calls.append([ids,count,seq_len])
        flags = [j%2==sample%2 for sample in range(count) for j in ids]
        valid = [j%3!=0 for sample in range(count) for j in ids]
        return None, (flags, valid)
    with _files(size) as config:
        config.update(num_gpus=gpus,k=k,batch_size=batch)
        with _environment(fn, LLM=Model, BPE=lambda p:object(), eval_examples=examples, tool_eval=object()):
            out = fn(rank, config)
    return out, calls


def test_eval_dataset(eval_dataset):
    out, calls = _dataset_eval(eval_dataset)
    assert np.allclose(out, [.5,1.,.6])
    assert calls == [['load','output.pt'], [[0,1],2,4], [[2,3],2,4], [[4],2,4]]
    a, ca = _dataset_eval(eval_dataset, rank=0, gpus=2, size=6, batch=8)
    b, cb = _dataset_eval(eval_dataset, rank=1, gpus=2, size=6, batch=8)
    assert np.allclose(np.array(a)+np.array(b), [.5,1.,2/3])
    assert ca[1:] == [[[0,1],2,4], [[4],2,4]] and cb[1:] == [[[2,3],2,4], [[5],2,4]]
    out, calls = _dataset_eval(eval_dataset, size=3, k=3, batch=6)
    assert np.allclose(out, [5/9,1.,2/3])


def submit_eval_dataset(eval_dataset):
    out, calls = _dataset_eval(eval_dataset, size=7, k=3, batch=9)
    mugrade.submit(list(out))
    mugrade.submit(calls)
    a, ca = _dataset_eval(eval_dataset, rank=0, gpus=2, size=8, k=3, batch=12)
    b, cb = _dataset_eval(eval_dataset, rank=1, gpus=2, size=8, k=3, batch=12)
    mugrade.submit(list(a))
    mugrade.submit(list(b))
    mugrade.submit(cb)
