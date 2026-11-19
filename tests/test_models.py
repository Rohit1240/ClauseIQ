"""Model architectures, training steps, checkpoint round-trips and real-checkpoint inference."""
import numpy as np
import pytest
import torch

from clauseiq.inference.windows import predict_token_probs, window_starts
from clauseiq.models.neural import build_model, count_parameters
from clauseiq.models.tfidf_lr import TfidfLRModel
from clauseiq.text.tokenize import tokenize

from .conftest import have_checkpoints


@pytest.mark.parametrize("arch", ["cnn", "bilstm", "bilstm_attn"])
def test_neural_forward_backward(arch):
    torch.manual_seed(0)
    m = build_model(arch, vocab_size=50, n_labels=41, word_dim=16, hidden=8, cnn_dim=16)
    B, T = 3, 40
    ids = torch.randint(2, 50, (B, T))
    ids[2, 30:] = 0  # padding
    feats = [torch.randint(1, 5, (B, T)) for _ in range(3)]
    logits = m(ids, *feats)
    assert logits.shape == (B, T, 41)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad and p.dim() > 1)
    assert count_parameters(m) > 0


def test_attention_weights_exposed():
    m = build_model("bilstm_attn", vocab_size=30, n_labels=41, word_dim=16, hidden=8).eval()
    ids = torch.randint(2, 30, (1, 12))
    f = [torch.ones_like(ids)] * 3
    _, w = m(ids, *f, return_attention=True)
    assert w.shape == (1, 12, 12)
    assert torch.allclose(w.sum(-1), torch.ones(1, 12), atol=1e-4)


def test_neural_learns_toy_task():
    """A tiny BiLSTM must fit 'token id 7 is category 0' — checks the training loop end to end."""
    torch.manual_seed(0)
    m = build_model("bilstm", vocab_size=20, n_labels=2, word_dim=16, hidden=16, dropout=0.0)
    m.inp.word_dropout = 0.0
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    ids = torch.randint(2, 20, (16, 30))
    y = torch.zeros(16, 30, 2)
    y[..., 0] = (ids == 7).float()
    f = [torch.ones_like(ids)] * 3
    for _ in range(150):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(ids, *f), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    pred = torch.sigmoid(m(ids, *f))[..., 0] > 0.5
    assert (pred == (ids == 7)).float().mean() > 0.98


def test_sliding_windows_cover_long_documents():
    assert window_starts(10, 256, 128) == [0]
    st = window_starts(1000, 256, 128)
    assert st[0] == 0 and st[-1] == 1000 - 256
    covered = np.zeros(1000, bool)
    for s in st:
        covered[s:s + 256] = True
    assert covered.all()
    m = build_model("cnn", vocab_size=30, n_labels=41, word_dim=8, cnn_dim=8)
    n = 700
    P = predict_token_probs(m, np.random.randint(2, 30, n), np.ones(n, int), np.ones(n, int), np.ones(n, int))
    assert P.shape == (n, 41) and np.all((P >= 0) & (P <= 1))


def test_tfidf_fit_predict_save_load(tmp_path):
    docs = [["The laws of New York govern this agreement.", "Payment is due in 30 days."],
            ["This agreement is governed by Delaware law.", "Invoices are payable monthly."]] * 5
    Y = np.array([[1, 0], [0, 0]] * 10, dtype=np.uint8)
    m = TfidfLRModel(C=1.0, min_df=1)
    m.fit_vectorizers(docs)
    import scipy.sparse as sp
    X = sp.vstack([m.featurize(d) for d in docs]).tocsr()
    m.fit(X, Y, ["Governing Law", "Unused"])
    m.thresholds = np.array([0.5, 0.5], dtype=np.float32)
    text = "Governed by the laws of Texas. Fees are payable monthly."
    segs = [(0, 30), (31, len(text))]
    P = m.predict_proba(text, segs)
    assert P[0, 0] > P[1, 0]
    assert P[:, 1].max() < 1e-6  # no positives -> intercept -20
    m.save(tmp_path / "m.pkl")
    m2 = TfidfLRModel.load(tmp_path / "m.pkl")
    assert np.allclose(m2.predict_proba(text, segs), P)


@pytest.mark.skipif(not have_checkpoints(), reason="trained checkpoints not available")
@pytest.mark.parametrize("name", ["tfidf_lr", "cnn", "bilstm", "bilstm_attn"])
def test_trained_checkpoint_inference(name, sample_text):
    from clauseiq.inference.predictor import load_predictor
    p = load_predictor(name)
    assert len(p.cat_keys) == 41 and p.thresholds.shape == (41,)
    long_text = sample_text * 30  # longer than one 256-token window
    out = p.predict(long_text)
    for key, spans in out.spans.items():
        assert key in p.cat_keys
        for s, e, score in spans:
            assert 0 <= s < e <= len(long_text)
            assert 0.0 <= score <= 1.0
    assert out.n_units > 0
    if name == "bilstm_attn":
        out2 = p.predict(sample_text, explain=True)
        assert set(out2.extras["attention"]) == set(out2.spans)


@pytest.mark.skipif(not have_checkpoints(), reason="trained checkpoints not available")
def test_tokenizer_matches_checkpoint_vocab(sample_text):
    from clauseiq.inference.predictor import load_predictor
    p = load_predictor("bilstm")
    ids = p.vocab.encode(tokenize(sample_text).tokens)
    assert (ids == p.vocab.unk_id).mean() < 0.2
