import { ErrorBox, Loading, ModelTag, PageHead } from "../components/ui";
import { api } from "../lib/api";
import { MODEL_ORDER, duration, int, pct } from "../lib/format";
import type { ModelName } from "../lib/types";
import { useAsync } from "../lib/useAsync";

const ARCH: Record<ModelName, string> = {
  tfidf_lr: `contract ─► sentence-like segments (abbreviation-aware splitter, offsets kept)
segment i ─► [ TF-IDF word 1–3-grams of segment i             (80,000 features)
             ⊕ 0.5 × TF-IDF word 1–2-grams of segments i±1     (30,000 features)
             ⊕ position / heading / length one-hots            (21 features) ]
          ─► 41 × logistic regression (L2, class_weight=balanced, liblinear)
          ─► P(category | segment) ≥ tuned threshold
          ─► consecutive positive segments merged into one span`,
  cnn: `tokens ─► [ word emb 128 ⊕ shape emb 16 ⊕ rel-position emb 16 ⊕ log-position emb 16 ]
       ─► linear 176 → 192
       ─► 6 × residual block: Conv1d(k=3, dilation 1,2,4,8,16,1) → GLU → dropout → LayerNorm
          (receptive field ≈ 65 tokens)
       ─► linear 192 → 41 ─► sigmoid per token and category`,
  bilstm: `tokens ─► [ word emb 128 ⊕ shape emb 16 ⊕ rel-position emb 16 ⊕ log-position emb 16 ]
       ─► 2-layer bidirectional LSTM, 160 units per direction (dropout 0.3)
       ─► linear 320 → 41 ─► sigmoid per token and category`,
  bilstm_attn: `tokens ─► [ word emb 128 ⊕ shape emb 16 ⊕ rel-position emb 16 ⊕ log-position emb 16 ]
       ─► 2-layer bidirectional LSTM, 160 units per direction
       ─► 4-head self-attention over the 256-token window (+ residual, LayerNorm)
       ─► position-wise feed-forward 320 → 640 → 320 (+ residual, LayerNorm)
       ─► linear 320 → 41 ─► sigmoid per token and category
       (attention weights are exposed as "attention focus" words in the reader)`,
};

export default function DocsPage() {
  const ds = useAsync(() => api.dataset());
  const models = useAsync(() => api.models());
  const s = ds.data?.stats;

  return (
    <div className="page">
      <PageHead title="Documentation" sub="How ClauseIQ turns a contract into extracted clauses, how each model works, and how the numbers on this site were measured." />
      <div style={{ display: "grid", gridTemplateColumns: "200px minmax(0, 1fr)", gap: 36 }}>
        <nav className="toc" aria-label="On this page">
          <a href="#pipeline">NLP pipeline</a>
          <a href="#models">The four models</a>
          <a href="#dataset">Dataset</a>
          <a href="#training">Training configuration</a>
          <a href="#metrics">Evaluation metrics</a>
          <a href="#limits">Limitations</a>
          <a href="#api">API</a>
        </nav>
        <div className="prose">
          <h2 id="pipeline" style={{ marginTop: 0 }}>NLP pipeline</h2>
          <ol>
            <li><b>Text extraction.</b> PDFs are read page by page with pypdf; the character range of every page in the combined text is recorded, so each clause can be reported with its page. Plain-text files are decoded as UTF-8, falling back to Windows-1252 and Latin-1. Scanned PDFs without a text layer are rejected: ClauseIQ does not perform OCR.</li>
            <li><b>Tokenisation.</b> A regular-expression tokenizer splits words, numbers and punctuation and stores each token's exact start and end character offset. Words are lower-cased and digits are replaced by a digit pattern (1999 → 0000). Each token also gets a shape feature (lower/Title/UPPER/digit/punctuation, and whether it starts a new line or paragraph) and two position features (relative position in the document and a log-scaled absolute index).</li>
            <li><b>Vocabulary.</b> Built only from training contracts (words seen at least twice), {ds.data?.features?.vocab_size ? `${int(ds.data.features.vocab_size as number)} entries` : "see the dataset section"}. There are no pretrained embeddings: every embedding vector starts random and is learned from CUAD.</li>
            <li><b>Segmentation.</b> For the TF-IDF model and for segment-level metrics, the text is cut into sentence-like segments at paragraph breaks, numbered headings and sentence ends (with abbreviations such as “Inc.” and “Sec.” handled); segments longer than 900 characters are split at semicolons.</li>
            <li><b>Long documents.</b> Neural models read overlapping windows of 256 tokens with a stride of 128. Each token's probability is a weighted average over every window that contains it, with more weight near window centres. There is no length limit beyond the 2 million character upload cap.</li>
            <li><b>Decoding.</b> Model scores become spans with settings chosen on the 61 validation contracts only. For the neural models the search covers a smoothing radius and a gap-merge distance (shared by all categories) and, per category, a threshold, a minimum span length and the extraction mode. In <i>token mode</i>, consecutive tokens above the threshold form a span; in <i>segment mode</i>, a sentence segment is selected when the mean of the model's token probabilities inside it reaches the threshold, and consecutive selected segments merge. Token mode suits short answers (parties, dates); segment mode suits long clauses, where token-level probabilities otherwise fragment a clause into pieces. The TF-IDF model always works on segments. Every span keeps exact character offsets for highlighting.</li>
          </ol>

          <h2 id="models">The four models</h2>
          <p>All four models extract spans and assign them a category. They were trained independently on the same contracts and evaluated on the same test contracts.</p>
          {MODEL_ORDER.map((m) => (
            <div key={m}>
              <h3><ModelTag model={m} /></h3>
              <p>{
                {
                  tfidf_lr: "A traditional baseline. It classifies whole sentence-like segments, so its spans can never be shorter than a sentence. Short answers such as party names or dates are returned with their surrounding sentence, which limits its exact-match and IoU scores.",
                  cnn: "Learns local textual patterns. Dilated convolutions widen the field of view exponentially, so each token sees about 65 neighbouring tokens while the model stays fast.",
                  bilstm: "Reads the window left-to-right and right-to-left, so every token's representation reflects the whole sequential context of the window.",
                  bilstm_attn: "Adds multi-head self-attention on top of the BiLSTM so that each token can draw directly on the most relevant words anywhere in the window. In the results reader the words that received the most attention from an extracted span are shown as its attention focus.",
                }[m]
              }</p>
              <div className="arch">{ARCH[m]}</div>
            </div>
          ))}

          <h2 id="dataset">Dataset</h2>
          {ds.error && <ErrorBox message={ds.error} />}
          {!s ? <Loading /> : (
            <>
              <p>The Contract Understanding Atticus Dataset (CUAD) v1: {int(s.n_contracts)} commercial contracts from SEC EDGAR filings, annotated by legal experts with {int(s.n_spans)} clause spans across {s.n_categories} categories. ClauseIQ uses the original annotations and the official category names and definitions; no labels were generated.</p>
              <dl className="kv">
                <dt>Split</dt><dd>{s.split.train} train / {s.split.val} validation / {s.split.test} test contracts. {s.split_source}. No contract appears in two splits.</dd>
                <dt>Annotations per split</dt><dd>train {int(s.spans_by_split.train)}, validation {int(s.spans_by_split.val)}, test {int(s.spans_by_split.test)}</dd>
                <dt>Contract length</dt><dd>median {int(s.contract_chars.median)} characters (min {int(s.contract_chars.min)}, max {int(s.contract_chars.max)})</dd>
                <dt>Text inside a clause</dt><dd>{pct(s.annotated_char_fraction)} of all characters</dd>
                <dt>Absent clauses</dt><dd>{pct(s.missing_annotations.pct_pairs_without_clause / 100)} of (contract, category) pairs have no clause. {s.missing_annotations.note}</dd>
                <dt>Class imbalance</dt><dd>{int(s.class_imbalance.max_spans)} spans for the most common category vs {int(s.class_imbalance.min_spans)} for the rarest (ratio {s.class_imbalance.imbalance_ratio}×, Gini {s.class_imbalance.gini}). Most common: {s.class_imbalance.most_common.map(([n, c]) => `${n} (${c})`).join(", ")}. Rarest: {s.class_imbalance.rarest.map(([n, c]) => `${n} (${c})`).join(", ")}.</dd>
                <dt>Attribution</dt><dd>{ds.data!.attribution}</dd>
              </dl>
            </>
          )}

          <h2 id="training">Training configuration</h2>
          <ul>
            <li>Neural models: AdamW (weight decay 0.01), gradient clipping at 1.0, batch of 32 windows × 256 tokens, learning rate halved when validation span F1 stops improving, early stopping after 3 epochs without improvement, best epoch restored.</li>
            <li>Class imbalance: all windows containing a clause are used each epoch; clause-free windows are subsampled to 25%. The loss is binary cross-entropy with per-category positive weights √(negatives/positives), capped at 30.</li>
            <li>Regularisation: dropout on embeddings and encoder outputs, 5% word dropout (random tokens replaced by the unknown token), random window offsets every epoch.</li>
            <li>TF-IDF + LR: regularisation strength C chosen from {"{1, 4}"} by validation span F1.</li>
            <li>Each epoch is scored on validation span F1 at the best of four global thresholds (0.5, 0.7, 0.85, 0.95), because positive-class weighting shifts probabilities upwards and a fixed 0.5 would misjudge progress.</li>
            <li>Seed 13 for data split, sampling and initialisation. All training ran on CPU (2 threads).</li>
          </ul>
          {models.data && (
            <table className="table" style={{ border: "1px solid var(--rule)", marginTop: 8 }}>
              <thead><tr><th>Model</th><th className="num">Best epoch</th><th className="num">Parameters</th><th className="num">Training time</th></tr></thead>
              <tbody>
                {models.data.map((m) => (
                  <tr key={m.name}>
                    <td><ModelTag model={m.name} /></td>
                    <td className="num">{m.training.best_epoch ?? "—"}</td>
                    <td className="num">{m.training.n_parameters ? int(m.training.n_parameters) : m.training.n_features ? `${int(m.training.n_features)} features × 41` : "—"}</td>
                    <td className="num">{duration(m.training.train_seconds ?? null)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <h2 id="metrics">Evaluation metrics</h2>
          <p>Three families of metrics are reported separately, because labelling text correctly is not the same as extracting the right text.</p>
          <h3>Span extraction (the main result)</h3>
          <ul>
            <li><b>Span precision / recall / F1.</b> Predicted spans are matched one-to-one with gold spans of the same category in the same contract, highest overlap first. A pair counts as a match when the intersection over union (IoU) of their character ranges is at least 0.5.</li>
            <li><b>Exact match.</b> The predicted span has exactly the gold boundaries after trimming whitespace and edge punctuation. The exact match rate is the share of gold spans reproduced exactly.</li>
            <li><b>Character IoU.</b> Characters covered by predictions ∩ characters covered by gold spans, divided by their union, summed per category and contract.</li>
            <li><b>Mean best IoU.</b> For each gold span, the best IoU of any same-category prediction (0 when none), averaged.</li>
          </ul>
          <h3>Segment classification</h3>
          <p>Each test segment gets the set of categories whose gold spans cover it (at least half of the shorter of the two). The predicted set is computed from the predicted spans by the same rule. Accuracy is the share of segments whose predicted set matches exactly; because about 92% of segments contain no clause, accuracy is dominated by negatives, so precision, recall, macro F1 (unweighted mean over categories) and weighted F1 (weighted by support) are reported too. The confusion matrix uses one primary label per segment.</p>
          <h3>Clause detection</h3>
          <p>For every (contract, category) pair: does the model extract at least one clause where the contract has one? This is the question “does this contract contain a non-compete?”.</p>
          <h3>Time</h3>
          <p>Training time is wall-clock time including validation and threshold tuning. Inference time is the time to process each test contract, on the same 2-thread CPU.</p>

          <h2 id="limits">Limitations</h2>
          <ul>
            <li>The models are small and trained on 347 contracts, so they are well below fine-tuned transformer results reported for CUAD. Rare categories with only a few dozen training examples perform poorly.</li>
            <li>The TF-IDF model works at sentence granularity and cannot isolate short answers such as party names or dates.</li>
            <li>Scores are model probabilities, not calibrated confidence. Thresholds were tuned on 61 validation contracts and can be noisy for rare categories.</li>
            <li>No OCR: scanned PDFs are rejected. PDF text extraction can merge columns or split words in unusual layouts, which affects predictions.</li>
            <li>CUAD contracts are US SEC filings in English. Other jurisdictions, languages or contract styles are out of distribution.</li>
            <li>ClauseIQ is a review aid, not legal advice. Always check extracted clauses against the original document.</li>
          </ul>

          <h2 id="api">API</h2>
          <p>The FastAPI backend serves interactive OpenAPI documentation at <a href="/api/docs" target="_blank" rel="noreferrer">/api/docs</a>. Main endpoints: <code>POST /api/documents</code> (upload), <code>POST /api/analyses</code> (run models), <code>GET /api/analyses/&#123;id&#125;</code>, <code>GET /api/analyses/&#123;id&#125;/report?format=html|csv|json</code>, <code>GET /api/models</code>, <code>GET /api/categories</code>, <code>GET /api/dataset</code>, <code>GET /api/health</code>.</p>
        </div>
      </div>
    </div>
  );
}
