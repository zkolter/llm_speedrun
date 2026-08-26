import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import mugrade


def _normalized_merges(merges):
    """Convert learned state to a stable, serializable observation."""
    return [[a, b, rank] for (a, b), rank in merges.items()]


def _training_fixture(characters, special_id, special_text):
    """Supply state plus no-op/recording collaborators for one train iteration."""
    vocab = [""] * 256
    for character in characters:
        vocab[ord(character)] = character

    merge_calls = []

    def record_merge(word, a, b, merged):
        merge_calls.append([list(word), a, b, merged])

    fixture = SimpleNamespace(
        vocab=vocab,
        merges={},
        special_tokens={special_id: special_text},
        replace_special_tokens=lambda text: text,
        merge_pair=record_merge,
    )
    return fixture, merge_calls


def _encoding_fixture(merges, replacements=None, scripted_merges=None):
    """Supply recorders and exact-case scripts for encode's collaborators."""
    merge_calls = []
    replacement_calls = []
    replacements = {} if replacements is None else replacements
    scripted_merges = {} if scripted_merges is None else scripted_merges

    def scripted_replace(text):
        replacement_calls.append(text)
        return replacements.get(text, text)

    def scripted_merge(word, a, b, merged):
        merge_calls.append([list(word), a, b, merged])
        key = (tuple(word), a, b, merged)
        if key not in scripted_merges:
            raise AssertionError(f"Unexpected merge collaborator call: {key}")
        word[:] = scripted_merges[key]

    fixture = SimpleNamespace(
        merges=merges,
        replace_special_tokens=scripted_replace,
        merge_pair=scripted_merge,
    )
    return fixture, replacement_calls, merge_calls


def _decoding_fixture():
    vocab = [""] * 301
    for token, value in {
        32: " ",
        97: "a",
        98: "b",
        120: "x",
        121: "y",
        122: "z",
        256: "ab",
        257: "abc",
        300: "<S>",
    }.items():
        vocab[token] = value
    return SimpleNamespace(vocab=vocab)


def test___init__(student_init):
    default = SimpleNamespace()
    assert student_init(default) is None
    assert len(default.vocab) == 256
    assert [ord(default.vocab[i]) for i in (0, 65, 255)] == [0, 65, 255]
    assert default.merges == {}
    assert default.special_tokens == {}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tokenizer.json"
        path.write_text('[["a", "b", "ab"], [[0, 1]], [2], {"5": "<S>"}]')
        loaded = SimpleNamespace()
        student_init(loaded, str(path))
        assert loaded.vocab == ["a", "b", "ab"]
        assert loaded.merges == {(0, 1): 2}
        assert loaded.special_tokens == {5: "<S>"}


def submit___init__(student_init):
    default = SimpleNamespace()
    student_init(default)
    mugrade.submit(len(default.vocab))
    mugrade.submit([ord(default.vocab[i]) for i in (1, 127, 254)])
    mugrade.submit([len(default.merges), len(default.special_tokens)])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "loaded.json"
        path.write_text(
            '[["x", "y", "xy", "xyz"], [[10, 11], [12, 13]], '
            '[12, 14], {"310": "<SPECIAL>"}]'
        )
        loaded = SimpleNamespace()
        student_init(loaded, str(path))
        mugrade.submit(loaded.vocab)
        mugrade.submit(_normalized_merges(loaded.merges))
        mugrade.submit([[key, value] for key, value in loaded.special_tokens.items()])


def test_merge_pair(merge_pair):
    word = [1, 2, 1, 2]
    assert merge_pair(word, 1, 2, 9) is None
    assert word == [9, 9]

    overlapping = [1, 1, 1]
    merge_pair(overlapping, 1, 1, 7)
    assert overlapping == [7, 1]

    absent = [3, 4, 5]
    merge_pair(absent, 1, 2, 8)
    assert absent == [3, 4, 5]

    boundaries = [4, 5, 0, 4, 5]
    merge_pair(boundaries, 4, 5, 6)
    assert boundaries == [6, 0, 6]


def submit_merge_pair(merge_pair):
    word = [2, 3, 2, 3, 4]
    merge_pair(word, 2, 3, 10)
    mugrade.submit(word)

    overlapping = [8, 8, 8, 8, 8]
    merge_pair(overlapping, 8, 8, 11)
    mugrade.submit(overlapping)

    mixed = [0, 6, 7, 6, 7, 6]
    merge_pair(mixed, 6, 7, 12)
    mugrade.submit(mixed)


def test_replace_special_tokens(replace_special_tokens):
    fixture = SimpleNamespace(special_tokens={300: "<S>", 301: "</S>"})
    assert [ord(c) for c in replace_special_tokens(fixture, "a<S>b")] == [97, 300, 98]
    assert [ord(c) for c in replace_special_tokens(fixture, "<S>x</S>")] == [
        300,
        120,
        301,
    ]
    assert replace_special_tokens(fixture, "ordinary text") == "ordinary text"
    assert replace_special_tokens(SimpleNamespace(special_tokens={}), "<S>") == "<S>"


def submit_replace_special_tokens(replace_special_tokens):
    fixture = SimpleNamespace(
        special_tokens={410: "<BOS>", 411: "<EOS>", 412: "<PAD>"}
    )
    mugrade.submit([ord(c) for c in replace_special_tokens(fixture, "<BOS>hi")])
    mugrade.submit([ord(c) for c in replace_special_tokens(fixture, "x<EOS><EOS>")])
    mugrade.submit([ord(c) for c in replace_special_tokens(fixture, "<PAD>")])


def test_train(train):
    tokenizer, merge_calls = _training_fixture(" ab", 300, "<S>")
    assert train(tokenizer, " ab ab ab", 257) is None
    assert tokenizer.merges == {(32, 97): 256}
    assert tokenizer.vocab[256] == " a"
    assert len(tokenizer.vocab) == 301
    assert tokenizer.vocab[300] == "<S>"
    assert merge_calls == [[[32, 97, 98], 32, 97, 256]]

    # The one-off word is filtered, so it cannot change the learned first pair.
    filtered, _ = _training_fixture(" xysingleton", 300, "<S>")
    train(filtered, " xy xy xy singleton", 257)
    assert filtered.merges == {(32, 120): 256}
    assert filtered.vocab[256] == " x"


def submit_train(train):
    tokenizer, merge_calls = _training_fixture(" pq", 310, "<SPECIAL>")
    train(tokenizer, " pq pq pq", 257)

    mugrade.submit(_normalized_merges(tokenizer.merges))
    mugrade.submit(tokenizer.vocab[256])
    mugrade.submit(len(tokenizer.vocab))
    mugrade.submit(tokenizer.vocab[310])
    mugrade.submit(merge_calls)


def test_encode(encode):
    tokenizer, replace_calls, merge_calls = _encoding_fixture(
        {(97, 98): 256, (256, 99): 257},
        scripted_merges={
            ((97, 98, 99), 97, 98, 256): [256, 99],
            ((256, 99), 256, 99, 257): [257],
        },
    )
    assert encode(tokenizer, "abc") == [257]
    assert replace_calls == ["abc"]
    assert merge_calls == [
        [[97, 98, 99], 97, 98, 256],
        [[256, 99], 256, 99, 257],
    ]

    special, _, special_merge_calls = _encoding_fixture(
        {}, {"<SPECIAL>": chr(300)}
    )
    assert encode(special, "<SPECIAL>") == [300]
    assert special_merge_calls == []

    empty, empty_replace_calls, _ = _encoding_fixture({})
    assert encode(empty, "") == []
    assert empty_replace_calls == []


def submit_encode(encode):
    tokenizer, replace_calls, merge_calls = _encoding_fixture(
        {(112, 113): 260, (260, 114): 261},
        scripted_merges={
            ((112, 113, 114), 112, 113, 260): [260, 114],
            ((260, 114), 260, 114, 261): [261],
        },
    )
    mugrade.submit(encode(tokenizer, "pqr"))
    mugrade.submit(replace_calls)
    mugrade.submit(merge_calls)

    special, _, _ = _encoding_fixture({}, {"<TAG>": chr(420)})
    mugrade.submit(encode(special, "<TAG>"))
    mugrade.submit(type(encode(special, "plain")).__name__)


def test_decode(decode):
    fixture = _decoding_fixture()
    assert decode(fixture, [97, 98]) == "ab"
    assert decode(fixture, [257, 32, 256]) == "abc ab"
    assert decode(fixture, [300, 257]) == "<S>abc"
    assert decode(fixture, []) == ""


def submit_decode(decode):
    fixture = _decoding_fixture()
    mugrade.submit(decode(fixture, [256, 99]))
    mugrade.submit(decode(fixture, [300, 32, 257, 300]))
    mugrade.submit(decode(fixture, [120, 121, 122]))
    mugrade.submit(type(decode(fixture, [97])).__name__)


def test_save(save):
    fixture = SimpleNamespace(
        vocab=["a", "b", "ab"],
        merges={(0, 1): 2, (2, 1): 3},
        special_tokens={300: "<S>"},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "saved.json"
        assert save(fixture, str(path)) is None
        payload = json.loads(path.read_text())

    assert payload[0] == ["a", "b", "ab"]
    assert payload[1:3] == [[[0, 1], [2, 1]], [2, 3]]
    assert payload[3] == {"300": "<S>"}


def submit_save(save):
    fixture = SimpleNamespace(
        vocab=["u", "v", "uv", "uvw"],
        merges={(20, 21): 22, (22, 23): 24},
        special_tokens={410: "<BOS>", 411: "<EOS>"},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "submission.json"
        save(fixture, str(path))
        payload = json.loads(path.read_text())

    mugrade.submit(payload[0])
    mugrade.submit(payload[1])
    mugrade.submit(payload[2])
    mugrade.submit([[key, value] for key, value in payload[3].items()])
