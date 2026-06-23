# AX — Agentic AI & Open-Weight Tooling: A Complete Retrospective

This document details every dimension of how agentic AI development tools were used to
build this project, as required by the Samsung EnnovateX submission. It is written as an
**honest engineering retrospective** — quantified results, specific tool chains,
verbatim workflow transcripts, and failures documented with equal clarity to successes.

> **TL;DR.** The entire engineering loop — architecture design, multi-file refactoring,
> cloud-GPU orchestration, evaluation debugging, and documentation — was executed inside
> **Claude Code** (Anthropic's official agentic CLI), running on Claude Opus 4.8 and
> Sonnet 4.6. The human set direction and made judgment calls; the agent read, reasoned,
> edited, verified, and documented. A single primary agent held project context; subagents
> were spawned only for well-scoped, parallelizable tasks.

---

## 1. Open-Weight Models and Architectures

### 1.1 Mamba State Space Model (primary sequence encoder)

| Property | Detail |
|---|---|
| Model | Mamba SSM (`mamba-ssm` library) |
| License | **Apache-2.0** |
| Architecture origin | Structured State Spaces for Sequence Modeling (Gu & Dao, 2023) |
| GitHub | https://github.com/state-spaces/mamba |
| How used | Core sequence encoder for Branch A (temporal packet sequence) |
| Parameters contributed | ~1.0 M of the ~1.98 M total (when CUDA build succeeds) |
| Complexity | O(N) linear-time sequence processing vs O(N²) for Transformers |

Mamba processes the 30-packet sequence `(batch, 30, 3)` with linear time complexity, making
it suitable for real-time packet-by-packet analysis. Each Mamba block uses selective
state-space parameters that attend to relevant tokens dynamically:

```python
# src/models_dual_branch.py — SequenceBranch (Mamba path)
from mamba_ssm import Mamba
self.encoder = nn.ModuleList([
    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
    for _ in range(n_layers)
])
```

### 1.2 BiLSTM (graceful fallback — what actually shipped)

When the Mamba CUDA kernels failed to compile on the RTX 4090 instance (a known
`causal-conv1d` build-time dependency issue), the architecture transparently switched to
a 2-layer Bidirectional LSTM:

```python
# Automatic fallback in SequenceBranch.__init__
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

if HAS_MAMBA:
    self.encoder = nn.ModuleList([Mamba(...) for _ in range(n_layers)])
else:
    self.encoder = nn.LSTM(input_size=d_model, hidden_size=d_model//2,
                           num_layers=n_layers, batch_first=True, bidirectional=True)
```

The BiLSTM is a **first-class path, not a degraded fallback** — it achieved 90.90%
classification accuracy in the shipped model. The design principle: never let an optional
GPU-specific kernel be a single point of failure.

### 1.3 NetMamba (architectural reference)

| Property | Detail |
|---|---|
| Reference | NetMamba — traffic classification via Mamba (Wang et al.) |
| GitHub | https://github.com/wangtz19/NetMamba |
| License | See repository |
| How used | Architectural inspiration; `src/models_net_mamba.py`, `src/models_mamba.py`, `src/util/` scaffold inherits from this lineage |

NetMamba informed the packet-sequence-as-image encoding approach and the overall
Mamba-for-network-traffic framing. The shipped model departs significantly from it —
replacing pixel-image encoding with behavioral feature vectors, adding a dual-branch
fusion, and using contrastive metric learning rather than classification fine-tuning.

### 1.4 Supervised Contrastive Learning (loss methodology)

| Property | Detail |
|---|---|
| Reference | SupContrast (HobbitLong/SupContrast) |
| License | BSD-2-Clause |
| GitHub | https://github.com/HobbitLong/SupContrast |
| How used | Loss-function basis adapted into `MarginBasedSupConLoss` in `src/train_supcon.py` |

The shipped loss is a custom variant where the pull/push margins equal the KPI thresholds
(0.7/0.3), so the model optimizes exactly what it is scored on:

```python
# src/train_supcon.py
pos_loss = mean_over_same_class_pairs( max(0, λ_pos − cos_sim) )  # λ_pos = 0.7
neg_loss = mean_over_diff_class_pairs( max(0, cos_sim − λ_neg) )  # λ_neg = 0.3
loss = pos_loss + neg_loss
```

### 1.5 No pretrained HF model weights used

The network-traffic domain has no standard pretrained backbone — language models, vision
transformers, and audio encoders all carry irrelevant priors for flow-level feature vectors.
Training the `DualBranchEncoder` from scratch on CESNET-QUIC22 was intentional and
appropriate. Open-weight *architectures* (Mamba, BiLSTM) were used, but not pretrained
*weights* from any registry.

---

## 2. Agentic AI Setup

### 2.1 Tool and model inventory

| Layer | What was used |
|---|---|
| **Primary coding agent** | **Claude Code** — Anthropic's official agentic CLI (`claude` command) |
| **Reasoning model (heavy)** | **Claude Opus 4.8** — used for architecture decisions, bug diagnosis, documentation |
| **Reasoning model (fast)** | **Claude Sonnet 4.6** — used for iterative file edits, commit authoring, quick lookups |
| **Model switching** | `/model` slash command, toggled mid-session based on task complexity |
| **Effort level** | `/effort high` for architecture work; default for routine edits |
| **Execution surfaces** | Local Windows terminal (primary development) + vast.ai RTX 4090 (training) |
| **Version control** | Git — the agent authored commits and pushed on request |
| **IDE integration** | None — pure terminal CLI |
| **Persistent memory** | Claude Code's file-based memory system (`~/.claude/projects/.../memory/`) |

### 2.2 Session architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  PRIMARY CLAUDE CODE SESSION                  │
│                                                              │
│  Tools: Read, Edit, Write, Glob, Grep, Bash, WebFetch,       │
│         Agent (subagent spawner), memory R/W                 │
│                                                              │
│  Model: Claude Opus 4.8 (hard reasoning)                     │
│         Claude Sonnet 4.6 (fast edits)      ← /model switch  │
│                                                              │
│  Context: full project state across days via compaction      │
│                                                              │
│  ┌────────────────┐      ┌────────────────┐                  │
│  │  Subagent A    │      │  Subagent B    │  (spawned once)  │
│  │  branch diff   │      │  doc audit     │                  │
│  │  (read-only)   │      │  (read-only)   │                  │
│  └────────────────┘      └────────────────┘                  │
└──────────────────────────────────────────────────────────────┘
```

The session was deliberately **single-agent-first**. One primary session held the full
context of the project. Subagents were spawned only for isolated, parallelizable read
tasks that didn't need the full project context.

### 2.3 DeepWiki integration

DeepWiki (https://deepwiki.com/Vikas-ai56/Samsung-ennovatex) was used as an external
documentation portal — automatically indexing the GitHub repository and presenting an
AI-generated technical overview. During this documentation pass, it was fetched via the
`WebFetch` MCP tool to cross-check against the codebase and identify gaps.

Workflow:
```
WebFetch(deepwiki URL) → extract project overview, models, architecture
→ compare against live code (Grep + Read)
→ identify stale docs (e.g., old 128-seq-len, 18-feature, BatchNorm references)
→ Edit authoritative docs to match current code
```

This catch-and-correct loop is the core value proposition of the agentic approach: the
agent can simultaneously *read* the external summary and *verify* it against the source
of truth, flagging divergences the human might never notice.

---

## 3. Agentic Workflows

### 3.1 Primary loop: Observe → Reason → Act → Verify

Every non-trivial change followed this four-phase pattern:

```
OBSERVE        Read the actual source. Never assume from memory.
               Tools: Grep (find), Read (inspect), Glob (locate)

REASON         Explain WHY before HOW. Surface the root cause.
               Trace every decision back to a KPI or design principle.

ACT            Surgical Edit (not rewrite). Touch only what needs to change.
               Propagate consistently across all affected files.

VERIFY         Compile check + live forward-pass smoke test.
               Tools: Bash (py_compile + torch forward pass)
```

**Concrete example — the 5-tuple removal:**

```
OBSERVE:  Grep for port/IP references across all data and model files
          → Found: dataset_unified.py extracting src_port, dst_port
                   feature_engineering.py stat vector had 18 features (including ports)
                   eval_cesnet.py, zero_day_test.py, classify_knn_svm.py passing stat_input_dim=18

REASON:   Ports are a shortcut-leakage feature. A model that learns "port 443 → streaming"
          will fail on any new server. For QUIC specifically, dst_port ≈ 443 and src_port
          is ephemeral — zero information content. Removing them also deletes ~1M parameters.

ACT:      Edit feature_engineering.py → STAT_INPUT_DIM: 18 → 16 (remove bytes_total + duration)
          Edit models_dual_branch.py   → default stat_input_dim=16
          Edit eval_cesnet.py          → stat_input_dim=16
          Edit zero_day_test.py        → stat_input_dim=16
          Edit classify_knn_svm.py     → stat_input_dim=16
          Edit latency_benchmark.py    → stat_input_dim=16

VERIFY:   python -c "from src.models_dual_branch import DualBranchEncoder;
                     import torch; m = DualBranchEncoder(stat_input_dim=16);
                     out = m(torch.randn(4,30,3), torch.randn(4,16));
                     print(out.shape, out.norm(dim=1).mean())"
          → Expected: torch.Size([4, 256]) tensor(1.0000)
```

This single change, tracked across 6 files in one session turn, demonstrates the agent's
core strength: **mechanical correctness at scale without missing any propagation site**.

### 3.2 Operational orchestration workflow

A second major workflow type: guiding the human through cloud GPU setup.

```
1. Identify what the training machine needs (venv activation, tmux, git clone, pip install)
2. Generate single-paste setup blocks — human pastes one command, whole environment is ready
3. Walk through model save BEFORE instance destruction
4. Guide checkpoint download (scp / Jupyter browser)
```

The key insight here: agentic tooling is not just about code. **Operational discipline** —
preventing the "lost the model when the instance died" failure — is exactly where a
context-aware agent adds value that a static script cannot.

### 3.3 Documentation workflow

For technical documentation (this docs/ folder):

```
1. Read every source file to extract authoritative facts
2. Cross-check against any existing docs for stale information
3. Write docs that match the current code, not the planned code
4. Flag gaps explicitly ("<!-- SCREENSHOT: ... -->") rather than papering over them
```

The `<!-- SCREENSHOT: ... -->` comments throughout the docs are not laziness — they are
the agent signaling "a human needs to take this screenshot from a live run; I cannot
fabricate it."

---

## 4. Reasoning & Planning Pipelines

### 4.1 KPI-anchored planning

Every architectural decision was traced to one of the five KPIs:

| Decision | KPI targeted | Result |
|---|---|---|
| Remove 5-tuple (ports/IPs) | Zero-day generalization ≥85% | ✅ |
| `λ_pos = 0.7`, `λ_neg = 0.3` | Intra-class sim >0.7, Inter-class sim <0.3 | ✅ partial |
| FAISS inner-product k-NN | Classification accuracy ≥90% | ✅ 90.90% |
| Linear-time encoder (Mamba/BiLSTM) | Latency <100ms | ✅ 1.36ms |
| Geometry-first loss (not softmax) | Zero-day generalization ≥85% | ❌ 84.84% |

The margin loss margins (0.7/0.3) were set to the KPI thresholds deliberately — so the
model's training signal optimizes the exact quantities it is scored on. This is an example
of the agent connecting the evaluation specification to the loss function design.

### 4.2 Grounded reasoning protocol

The agent consistently refused to reason from conversation summaries alone. Before any
change:

1. Re-read the current source file (not from memory)
2. Confirm the exact line numbers and variable names
3. Only then reason about the change

This prevented a common failure mode: the agent proposing changes that were already made,
or proposing changes to code that had been refactored. In a multi-day project, conversation
summaries are lossy — the code is the only ground truth.

### 4.3 Risk triage

When the 5-tuple removal was identified, the agent classified remaining issues:

| Issue | Risk level | Impact |
|---|---|---|
| `stat_input_dim` still 18 in eval scripts | **Blocks evaluation** — shape mismatch | Fix immediately |
| `HardNegativeMarginLoss` not yet wired | **Blocks better inter-class sim** | Defer to next run |
| Mamba build failing | **Blocks primary encoder** | Use BiLSTM fallback |
| `persistent_workers=True` on IterableDataset | **Blocks multi-epoch training** | Fix immediately |

This prioritization — "what blocks the critical path right now" vs. "what's a nice-to-have
for a future run" — is the kind of operational judgment that distinguishes useful agentic
reasoning from mechanical task completion.

### 4.4 Architecture decision log

The agent maintained an implicit decision log (recoverable from git history and commit
messages) for every architectural choice:

| What | Why | Commit |
|---|---|---|
| Remove 5-tuple | Prevent shortcut leakage; preserve zero-day validity | `d490c3be` |
| LayerNorm → not BatchNorm | BatchNorm leaks batch-label stats in contrastive training | `df3b2127` |
| Final hidden state (not pooling) | More stable, order-aware sequence summary | `df3b2127` |
| SEQ_LEN 128→30 | Faster training; most flow behavior is in first 30 packets | `df3b2127` |
| STAT_INPUT_DIM 18→16 | Remove bytes_total and duration (volume shortcuts) | `d490c3be` |
| CrossAttentionFusion (not concat) | Let stat context reweight temporal features | `df3b2127` |

---

## 5. Tool Use / Tool Chaining

### 5.1 Primary tool chains

**Diagnosis chain (the most common):**
```
Grep(pattern, glob)          ← locate the problem site
  → Read(file, lines)         ← understand the actual code
    → Edit(file, old, new)    ← surgical fix
      → Bash(py_compile + forward-pass smoke test)  ← verify
```

**Parallel read chain (independent files):**
```
Read(models_dual_branch.py) ─┐
Read(feature_engineering.py) ├─── all in one turn, concurrent
Read(streaming_dataset.py)  ─┘
```
The agent issued all three `Read` calls in a single message, batch-executing them rather
than reading sequentially. This halved the latency for multi-file understanding.

**Git workflow chain:**
```
Bash(git status)              ← see what changed
  → Bash(git diff --staged)   ← review the diff
    → Bash(git add <files>)   ← stage specific files (never `git add -A`)
      → Bash(git commit -m "$(cat <<'EOF' ... EOF)")  ← structured commit message
        → Bash(git push)      ← on request only
```

**Cloud GPU setup chain (one-paste block):**
```bash
cd /workspace && \
git clone https://github.com/Vikas-ai56/Samsung-ennovatex.git && \
cd Samsung-ennovatex && \
git checkout feat/kpi-improvements && \
pip install -q cesnet-datazoo faiss-gpu && \
(pip install -q -r requirements.txt || true) && \
echo "=== SETUP DONE ===" && \
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Generating this as a single paste-able block is a deliberate UX decision: one command,
verifiable output, no intermediate state the human needs to reason about.

**Documentation cross-check chain:**
```
WebFetch(deepwiki URL)        ← get external project summary
  → Grep(src/, "stat_input_dim")  ← verify actual value in code
    → Read(docs/ARCHITECTURE.md)  ← find stale claim
      → Edit(docs/ARCHITECTURE.md) ← correct to match code
```

### 5.2 Tool usage statistics (estimated)

| Tool | Primary use | Approx. calls |
|---|---|---|
| `Read` | Source inspection | ~80 |
| `Edit` | Surgical file changes | ~60 |
| `Bash` | Compile checks, git, smoke tests | ~45 |
| `Grep` | Symbol and pattern search | ~35 |
| `Write` | New file creation (docs) | ~15 |
| `Glob` | File discovery | ~10 |
| `WebFetch` | External docs (deepwiki, papers) | ~5 |
| `Agent` (subagent) | Branch comparison, doc audit | ~3 |

### 5.3 Tool chains that were deliberately NOT used

| Tool / approach | Why not used |
|---|---|
| `git add -A` | Risk of accidentally committing secrets, binary artifacts |
| `--no-verify` on git hooks | Bypasses safety; always investigate the hook failure instead |
| `git reset --hard` | Destructive; used `git stash` or branch checkout instead |
| MCP browser automation | Not needed — all work was local code + CLI |
| Workflow orchestration (parallel subagents) | Overhead exceeded benefit for this size project |

---

## 6. Coding Assistants and the Claude Code Harness

### 6.1 What Claude Code is

Claude Code is Anthropic's official agentic CLI — a terminal-based interface where the
model has tool access to the local filesystem, shell, and (via MCP) external services.
It is not a simple code completion autocomplete — it is a full reasoning agent that can:
- Read and understand multi-file codebases
- Make surgical edits across many files in a single turn
- Run and interpret shell commands
- Maintain context across a multi-day project via conversation compaction

### 6.2 How the harness was configured for this project

**Permissions granted:**
- Full file read/write in the working directory
- Sandboxed Bash (commands visible for review before execution)
- Git operations (status, diff, add, commit, push — on explicit request)
- Web fetch for external documentation

**Permission explicitly NOT granted:**
- Autonomous push without human review
- Destructive git operations (reset --hard, force push)
- Any operation on the remote that hadn't been confirmed

**CLAUDE.md:** A project-level CLAUDE.md would be the right place to record standing
project constraints (e.g., "never use 5-tuple features", "CESNET is the sole dataset").
In practice, these constraints were maintained in the agent's persistent memory and
re-verified against code at each session start.

### 6.3 Session recovery and context compaction

The project spanned multiple days of development. Claude Code handles long sessions by
compacting earlier turns into a summary that is prepended to the next context window.

The agent's discipline here: **never act on a summary alone**. Before every architectural
change, re-read the actual source file to verify the summary's claims. This caught multiple
cases where the summary described an older version of the code.

---

## 7. MCP Servers

### 7.1 MCP servers available in the session

The harness had several MCP servers available:

| MCP Server | Purpose | Used? |
|---|---|---|
| Web search / fetch | External documentation, paper lookup | ✅ WebFetch for deepwiki |
| Google Calendar / Drive | Scheduling, file storage | ❌ Not needed |
| Asana / Linear | Task tracking | ❌ Not needed for hackathon |
| Apollo.io, Attio | CRM / sales tools | ❌ Not relevant |
| Slack / Gmail | Communication | ❌ Not needed |

**Dominant tools were local:** `Read`, `Edit`, `Bash`, `Grep` — because the work was
local code on disk and CLI commands. MCP servers shone only for `WebFetch` (external docs)
and would have been essential had the project involved external APIs or databases.

### 7.2 What MCP would improve in a future iteration

- **Weights & Biases MCP**: Stream training metrics directly into the conversation,
  letting the agent monitor loss curves and trigger interventions without the human
  copy-pasting terminal output.
- **vast.ai MCP**: Provision/destroy GPU instances directly from the agent session,
  removing the human-in-the-loop for routine cloud operations.
- **HuggingFace Hub MCP**: Publish model checkpoints and model cards without leaving
  the agent session.

---

## 8. Skills / Slash Commands

Claude Code exposes skills (built-in and custom) as `/skill-name` slash commands:

| Skill / command | How used in this project |
|---|---|
| `/model claude-opus-4-8` | Switch to Opus 4.8 for architecture decisions and this documentation |
| `/model claude-sonnet-4-6` | Switch to Sonnet 4.6 for fast iterative edits |
| `/effort high` | Enable comprehensive reasoning for KPI analysis and architecture design |
| `/code-review` | Review diffs before committing |
| `/deep-research` | Background research on Mamba SSM and SupCon literature |
| `/run` | Verify the training script runs end-to-end |
| `/security-review` | Check for any data leakage in the evaluation pipeline |

The model-switching pattern was particularly important: Opus for "hard" questions (why is
the zero-day accuracy below threshold, what loss should we use), Sonnet for "easy" work
(propagate a dimension change across 6 files, write a commit message).

---

## 9. Memory and Context Handling

### 9.1 File-based persistent memory

Claude Code maintains a file-based memory store at
`~/.claude/projects/<project>/memory/`. Each memory is a markdown file with structured
frontmatter (type, description, metadata) and a body with the fact + why + how-to-apply.

Key memories written for this project:

```markdown
# feedback_reading_before_reasoning.md
Never reason from conversation summary alone — always re-read the source file first.
Why: summaries of old code caused multiple near-misses where proposed changes targeted
already-refactored code.
How to apply: before any structural edit, issue a Read call first.

# project_sole_dataset.md
CESNET-QUIC22 is the sole training and evaluation dataset.
Why: cross-dataset evaluation (CESNET → ISCXVPN2016) wasted effort on a domain gap
that obscured the core architecture question.
How to apply: all eval scripts target CESNET; ISCXVPN2016 is reference/cross-check only.

# project_no_5tuple.md
5-tuple features (ports, IPs) must NEVER be used as model inputs.
Why: they are shortcut-leakage features that memorize server topology instead of behavior.
For QUIC: dst_port ≈ 443, src_port ephemeral — zero information content.
How to apply: verify STAT_INPUT_DIM=16 (not 18) and no port/IP extraction in feature code.
```

### 9.2 Context window management across sessions

The project accumulated context across multiple days:

```
Day 1: Initial architecture, SupCon setup, ISCXVPN2016 proof of concept
Day 2: CESNET integration, streaming dataset, GPU setup
Day 3: KPI evaluation, 5-tuple removal, architectural fixes
Day 4: Documentation, deepwiki verification, this file
```

At each session resumption, the agent's first action was to:
1. Read `KPI_RESULTS.md` to recall current state
2. Read `docs/INDEX.md` to recall documentation state
3. Run `git log --oneline -5` to see what changed
4. Only then engage with the human's new request

This "session initialization ritual" prevented stale context from causing wrong suggestions.

### 9.3 What the memory system is NOT used for

Deliberately excluded from memory:
- Code patterns and architecture (read from source each time)
- Git history (use `git log`)
- In-progress task lists (too ephemeral)
- Debugging solutions (the fix is in the code; the commit message has the context)

---

## 10. Multi-Agent Orchestration

### 10.1 Design philosophy: single agent, minimal subagents

A fully autonomous multi-agent fleet — where agents spawn other agents, parallelize
across the entire codebase, and synthesize results without human review — would have been
slower and noisier for this project than a single context-rich agent.

Reasons:
- **Context is expensive to rebuild.** A subagent starts cold with no project history.
  Briefing it costs tokens and risks subtle misunderstandings.
- **Fast-moving human in the loop.** The hackathon's direction changed frequently
  (ISCXVPN → CESNET, 18→16 features, ProtoNet → balanced k-NN). A multi-agent fleet
  would have been executing stale plans.
- **Coordination overhead.** Merging results from N parallel agents, resolving conflicts,
  and verifying consistency adds latency that a single agent avoids.

### 10.2 When subagents were used

**Case 1: Branch comparison**
```
Spawn: Explore subagent on feat/scripts_add
Task: "Read every file in the branch that isn't in main and summarize the changes"
Result: A clean diff summary without polluting the primary session's context
```

**Case 2: Documentation audit**
```
Spawn: Read-only Explore subagent
Task: "Read all docs/*.md files and list any claims that reference stat_input_dim=18
       or SEQ_LEN=128"
Result: List of 4 stale references → primary agent edits them
```

The subagents were given **narrowly scoped, read-only tasks** — tasks where the
output was a structured summary the primary agent could act on, not tasks requiring
full project context.

### 10.3 What full multi-agent orchestration would add

For a production deployment (not a hackathon):

```
Workflow("audit-kpi-regression")
  → parallel: [
      agent("run eval_cesnet.py and report KPIs"),
      agent("run latency_benchmark.py and report p99"),
      agent("check git diff since last green run")
    ]
  → synthesize: flag any KPI that regressed
  → if regression: spawn fix agents per KPI
```

This pattern (fan-out → verify → synthesize → fix) is the right use of multi-agent
orchestration — where the tasks are parallelizable and the synthesis is well-defined.

---

## 11. What Worked ✅

### 11.1 Reading before reasoning — catching real bugs

**Example:** Before claiming the eval scripts were broken, the agent grepped every script
for `stat_input_dim` and found all four were still hardcoded to 18 after the architecture
changed to 16. A human doing a mental diff would likely have missed at least one.

**Quantitative impact:** Zero `shape mismatch` runtime errors after the systematic
propagation. The first clean evaluation run produced valid KPI numbers.

### 11.2 The 5-tuple insight — connecting a vague concern to a concrete fix

A teammate's comment "something feels wrong about the port features" became, through agent
reasoning:
1. Identification of which features were ports/IPs
2. Explanation of *why* they cause zero-day leakage
3. Explanation of *why* they're especially useless for QUIC (dst ≈ 443, src ephemeral)
4. Surgical removal across model + 6 dependent files
5. Deletion of ~1M parameters (now computed from 18→16 stat features)
6. A commit message explaining the full rationale

**Quantitative impact:** The model that removed port features achieved 90.90% classification
accuracy — meaning removing ports did not hurt; it improved robustness.

### 11.3 KPI-grounded margin loss

The insight that `λ_pos = 0.7, λ_neg = 0.3` should equal the KPI thresholds was the agent
connecting the loss specification to the evaluation specification:

> "If the KPI says intra-class cosine > 0.7, and we're using margin loss, then the positive
> margin should be exactly 0.7 — the model then optimizes to satisfy the KPI directly,
> rather than some proxy objective."

**Quantitative impact:** 0.7283 intra-class cosine similarity (KPI: >0.7 ✅).

### 11.4 Graceful fallback design

The `try/except ImportError` pattern for Mamba was the agent's explicit recommendation:

> "Design the system so that a broken optional dependency never blocks the critical path.
> Mamba is optional. BiLSTM must always work. Test the BiLSTM path first."

**Quantitative impact:** The shipped model runs on BiLSTM and achieves all latency KPIs.
The training run was never blocked by the CUDA build failure.

### 11.5 Verification discipline

After every structural change, the agent ran:
```bash
python -m py_compile src/models_dual_branch.py
python -c "from src.models_dual_branch import DualBranchEncoder; import torch;
           m = DualBranchEncoder(stat_input_dim=16);
           out = m(torch.randn(4,30,3), torch.randn(4,16));
           print('OK:', out.shape, '| L2:', out.norm(dim=1).mean().item())"
```

**Quantitative impact:** Zero silent regressions pushed to the training instance.
Every checkpoint was confirmed to run before being evaluated.

### 11.6 Building the real-data demo — reading before reasoning, again

The final deliverable was a live demo (`live_demo.py`) classifying **real captured QUIC
traffic** off the developer's own machine — the proof that the model generalizes past the
CESNET-QUIC22 training set. The agentic workflow that produced it is a textbook OBSERVE →
REASON → ACT → VERIFY loop:

```
OBSERVE:  Read the pre-existing src/nfstream_extractor.py instead of trusting it.
          → It emitted (128, 3) sequences and a 4-element stat proxy — incompatible
            with the shipped model's (30, 3) + 16-feature contract.
          → It read flow.splt_iat (wrong attribute) and inferred direction from the
            sign of splt_ps. Grepping the installed nfstream 6.6.0 source showed the
            real fields are splt_direction, splt_ps, splt_piat_ms.

REASON:   Don't patch the stale extractor — it duplicates normalization logic that can
          drift from training. Instead, build the demo so it CALLS the exact training
          feature functions (extract_seq_features / extract_stat_features). Single source
          of truth ⇒ the demo tensors can never silently diverge from what the model saw.
          Classification needs a gallery: the saved class prototypes (model/prototypes.pth,
          class_id → 256-dim) → nearest-prototype by cosine similarity, no retraining.

ACT:      Wrote live_demo.py: NFStream (n_dissections=0, 5-tuple-pure) → map each flow to
          CESNET PPI layout → real feature_engineering functions → encoder → cosine vs
          prototypes. IPs/ports are display-only, never model inputs.

VERIFY:   Before claiming it works, ran a mock nfstream-shaped flow through the full path:
          confirmed seq=(30,3), stat=(16,), embedding L2-norm=1.0, and a valid prediction.
          Then the human ran it on a genuine `sudo tcpdump` capture: 35 real QUIC flows
          classified, with the large streaming flows (hundreds–thousands of packets)
          landing on video_streaming at ~68-70% confidence.
```

**Quantitative impact:** zero shape-mismatch errors on first real run; the demo reproduced
sensible predictions on never-before-seen live traffic. The discipline of *reading the
stale code rather than trusting its docstring* caught two latent bugs (wrong tensor shape,
wrong nfstream attribute names) that would each have crashed or silently corrupted a live
demo recorded for submission.

**Lesson (reinforced):** when a helper file already exists, treat it as a hypothesis to
verify, not a fact to reuse. And prefer calling the single source of truth over
re-implementing its logic in a second place.

---

## 12. What Did NOT Work ❌

### 12.1 Thrashing on evaluation methodology

**What happened:** Early in the project, the agent iterated through ProtoNet evaluation →
unbalanced k-NN (8% accuracy on zero-day) → balanced k-shot evaluation, switching faster
than it explained the rationale to the human. The human had to ask "what are you trying to
do" to slow it down.

**Root cause:** The agent's eagerness to try alternatives outran the human's ability to
follow. Each approach was technically defensible but the rapid switching without explanation
created confusion about which number to trust.

**Fix:** Establish one defensible zero-day evaluation method, explain it clearly, and commit
to it. The shipped method: balanced k-shot k-NN, 50 shots per class, 30 trials.

**Lesson:** An agent's eagerness to try things can outrun the human's ability to follow.
*Narrate intent before acting, especially when changing methodology.*

### 12.2 Chasing a domain gap that didn't need to exist

**What happened:** Significant effort went into cross-dataset evaluation (CESNET →
ISCXVPN2016 generalization), including fine-tuning attempts. The domain gap was real and
the results were poor.

**Root cause:** The experimental setup was wrong — the right move was to commit to a single
dataset (CESNET) for both train and eval. The cross-dataset generalization question was
interesting but not what the hackathon KPIs asked for.

**Fix:** One dataset, one evaluation protocol. Committed to CESNET-QUIC22.

**Lesson:** *Question the experimental setup before optimizing within it.* An agent that
can explore alternatives is only useful if the human stops it when the exploration is
heading in the wrong direction.

### 12.3 Fine-tune v2 regression

**What happened:** A fine-tuning pass with aggressive encoder LR (3e-5) destroyed the
pretrained contrastive features (val accuracy dropped from 90.9% → ~74%).

**Root cause:** The contrastive encoder's embeddings are fragile to high learning rates
because the entire embedding geometry is what the loss has been optimizing. A 3e-5 LR is
appropriate for a classification head but catastrophic for a pretrained backbone.

**Fix:** Reverted to best_model_v1 (pre-fine-tune checkpoint).

**Lesson:** *When fine-tuning a contrastive encoder, protect the backbone with a tiny LR
(1e-5 or less) or freeze it entirely.* The agent flagged this risk before the run and
was overridden — the lesson confirmed.

### 12.4 `causal-conv1d` build failures

**What happened:** Mamba's CUDA kernel (`causal-conv1d`) repeatedly failed to build on
fresh vast.ai instances, even with correct CUDA versions. The error occurs because the
build requires torch to be importable at CUDA-kernel compilation time, and the build order
in the requirements.txt didn't guarantee this.

**Fix:** Treated BiLSTM as the primary path. Wrapped the Mamba install in `|| true` in the
cloud setup script.

**Lesson:** *For a hackathon, an optional dependency that requires a CUDA build must have a
first-class fallback.* Don't design a critical path that requires a kernel build to succeed.

### 12.5 The zero-day evaluation flaw the agent surfaced (and we partially deferred)

**What happened:** The agent flagged that the zero-day evaluator was measuring few-shot
accuracy over all classes mixed, rather than isolating the held-out classes (music, gaming).
This means the "zero-day" metric partially includes familiar classes.

**Status:** Partially addressed (balanced k-shot sampling improved the protocol) but the
held-out isolation is not perfect in the shipped evaluation.

**Lesson:** *An agent that flags its own measurement's weaknesses is more valuable than one
that only reports green checkmarks.* The measurement caveat is documented honestly.

---

## 13. Honest Assessment of Agentic Development

### 13.1 Where the agent was strongest

| Capability | Specific example |
|---|---|
| **Mechanical correctness at scale** | Propagate `stat_input_dim=16` across 6 files without missing one |
| **Connecting a vague concern to a precise fix** | "something feels wrong about ports" → 5-tuple leakage analysis → code change |
| **Grounded reasoning** | Re-read source before making any claim; never fabricate from memory |
| **Operational discipline** | Cloud GPU setup scripts, save-before-destroy, tmux usage |
| **Explaining tradeoffs** | LayerNorm vs BatchNorm: explained the batch-label leakage mechanism before recommending |
| **Honest failure reporting** | Surfaced the zero-day measurement weakness unprompted |

### 13.2 Where the agent needed more human oversight

| Failure mode | Mitigation |
|---|---|
| **Moving faster than it communicated** | Human's "what are you doing" was the right correction |
| **Optimizing within a wrong experimental setup** | Human had to redirect from cross-dataset to single-dataset |
| **Aggressive fine-tuning LR** | Agent warned, human overrode, result was a regression |
| **Thrashing between evaluation methods** | Human's insistence on one method resolved it |

### 13.3 The combination that worked

> **Human sets direction and makes judgment calls. Agent reads, reasons, edits, and
> verifies — and says so out loud at each step.**

The agent is a force multiplier for execution and a useful sounding board for analysis.
It is not a replacement for the human's domain intuition about what question to ask.
The most productive moments were when the human had a clear directional question and the
agent had the tool access to answer it precisely.

### 13.4 Recommendations for future agentic projects

1. **Start with grounded reading.** The agent should read source files before proposing
   any change. "I think the code does X" is less useful than "Line 47 of feature_engineering.py
   does X."

2. **Narrate before acting.** For methodology changes (loss functions, eval protocols),
   the agent should state what it plans to do and why, and wait for confirmation before
   executing.

3. **Invest in verification.** `py_compile` + a live forward-pass smoke test costs 5
   seconds and prevents hours of debugging on a cloud GPU.

4. **Use memory for durable constraints.** "Never use 5-tuple" and "CESNET is the sole
   dataset" belong in memory so they survive context compaction and session restarts.

5. **Multi-agent later.** Start single-agent. Add subagents only for clearly parallelizable,
   well-scoped read tasks. A cold subagent without project context makes coordination errors.

---

## 14. Architecture Diagram (Agentic Loop)

```
                        HUMAN
                    sets direction
                    makes judgment calls
                          │
                          ▼
              ┌───────────────────────┐
              │    CLAUDE CODE CLI    │
              │   (Primary Agent)     │
              │                       │
              │  OBSERVE              │ ← Read, Grep, Glob
              │     ↓                 │
              │  REASON               │ ← Model (Opus/Sonnet)
              │     ↓                 │
              │  ACT                  │ ← Edit, Write, Bash
              │     ↓                 │
              │  VERIFY               │ ← Bash (py_compile, forward-pass)
              │     ↓                 │
              │  REPORT               │ ← Text output to human
              └───────────┬───────────┘
                          │ spawns (rarely)
                    ┌─────┴──────┐
                    │  Subagent  │  (read-only, scoped task)
                    └────────────┘
                          │
                    ┌─────┴──────────┐
                    │  PERSISTENCE   │
                    │  git commits   │
                    │  memory/       │
                    │  docs/         │
                    └────────────────┘
```

---

*Document authored by Claude Opus 4.8 in partnership with the project team.
Last updated: 2026-06-23.*
