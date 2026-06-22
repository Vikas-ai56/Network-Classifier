# AX — Agentic AI & Open-Weight Tooling

This document details how agentic AI development tools were used to build this project,
as required by the submission. It is written as an **honest retrospective** — including
what worked, what didn't, and the mistakes we corrected — because that is the most useful
record for the judges and for ourselves.

> **TL;DR.** Essentially the entire engineering loop — architecture changes, bug fixing,
> feature engineering decisions, cloud-GPU orchestration, documentation — was done in
> partnership with **Claude Code** (Anthropic's agentic CLI), running on the Claude
> Opus 4.x / Sonnet 4.x models. The human drove direction and made the calls; the agent
> did the reading, reasoning, editing, and verification. No autonomous multi-agent fleet —
> a single primary agent with occasional task-scoped subagents.

---

## 1. Agentic AI setup

| Layer | What we used |
|---|---|
| **Coding agent / harness** | **Claude Code** (Anthropic official CLI) |
| **Models** | Claude **Opus 4.8** (heavy reasoning, architecture, this documentation) and **Sonnet 4.6** (faster iterative edits) — switched via `/model` during the session |
| **Execution surface** | Local macOS terminal (the repo on Desktop) + a remote **vast.ai** RTX 4090 driven through its Jupyter terminal |
| **Version control** | Git, with the agent authoring commits and pushing to GitHub on request |
| **Tools available to the agent** | File read/write/edit, ripgrep-style search, sandboxed Bash, sub-agent spawning, web/doc fetch (MCP), persistent memory |

The setup was deliberately **single-agent-first**. One primary Claude Code session held
the full context of the project and acted as the engineer. Specialized sub-agents were
spawned only for well-scoped, parallelizable read tasks.

## 2. Agentic workflows

The dominant workflow was a tight **observe → reason → act → verify** loop:

1. **Observe** — the agent read the actual source (not assumptions) with `Read`/`Grep`
   before proposing anything. Example: before claiming the eval scripts were broken, it
   grepped every script for tuple-unpacking patterns and model-dimension args.
2. **Reason** — it explained the *why* (e.g. why ports are a leakage shortcut) before the
   *how*, and surfaced a recommendation rather than a menu of options.
3. **Act** — surgical `Edit`s across the codebase (model, dataset, training loop, four
   eval scripts) rather than wholesale rewrites.
4. **Verify** — after edits it ran `py_compile` on all touched files and a live forward
   pass (`model(seq, stat) → (8,256), L2=1.0`) to confirm the change actually worked,
   not just that it parsed.

A second recurring workflow was **operational orchestration**: walking the human through
renting a GPU, SSH/Jupyter connection, `tmux` to survive disconnects, and the critical
"save the model **before** you destroy the instance" discipline.

## 3. Reasoning & planning pipelines

- **Grounded reasoning.** The agent consistently refused to reason from the conversation
  summary alone — it re-read files (`feature_engineering.py`, `models_dual_branch.py`,
  `eval_cesnet.py`, etc.) to ensure claims matched the *current* code. This caught that
  the existing `docs/` described a stale architecture.
- **KPI-anchored planning.** Every architectural decision was traced back to one of the
  five KPIs. The margin loss margins were set to the KPI thresholds (0.7 / 0.3) precisely
  so the model optimizes what it is scored on.
- **Risk triage.** When fixing the port removal, the agent classified the remaining work
  as "blocks training" vs "blocks evaluation only," so the human could prioritize.

## 4. Tool use / tool chaining

Representative chains from this project:

- **Diagnosis chain:** `Grep (find tuple/dim mismatches)` → `Read (the offending files)`
  → `Edit (fix)` → `Bash (py_compile + forward-pass smoke test)`.
- **Parallel reads:** multiple `Read` calls issued in a single turn to pull several source
  files at once (independent calls batched for speed).
- **Git chain:** `Bash (git status/diff)` → `Bash (git add + commit)` →
  `Bash (git push)`, with the agent writing descriptive commit messages.
- **Cloud chain:** the agent generated single-paste setup blocks (`clone && checkout &&
  pip install && verify`) so the human could run an entire instance bootstrap in one step.

## 5. Memory & context handling

- **Conversation compaction.** The project spanned more context than a single window. The
  harness summarized earlier turns and carried the summary forward, so multi-day work
  (training, evaluation, debugging, docs) stayed coherent. The agent was careful to
  re-verify summarized facts against the live code before acting on them.
- **Persistent file memory.** Claude Code maintains a file-based memory store for durable
  facts (user preferences, project constraints). This is where decisions like "use CESNET
  only, not ISCXVPN2016" and "the 5-tuple must never be a feature" would be recorded so
  they survive across sessions.

## 6. Multi-agent orchestration

Used **sparingly and on purpose**:

- A **sub-agent** was spawned once to compare the `feat/scripts_add` branch against
  `feat/kpi-improvements` — a read-only, parallelizable analysis task that didn't need the
  primary session's full context.
- We deliberately **avoided** a large autonomous multi-agent fleet. For a hackathon with a
  fast-moving, opinionated human in the loop, a single context-rich agent was faster and
  produced fewer coordination errors than spawning cold agents that re-derive context.

## 7. MCP servers, skills, agents.md

- **MCP servers** were available in the harness (e.g. a docs-fetch server and a browser
  automation server) but were **not central** to this project — the work was local code
  and CLI, so file/Bash tools dominated.
- **Skills / slash-commands** (e.g. `/model` to switch models, `/login`) were used for
  session control.
- We did **not** rely on a committed `agents.md`; direction came from the live human in
  the loop plus the agent's persistent memory.

## 8. What worked ✅

- **Reading before reasoning.** Forcing the agent to read actual source caught real bugs
  (e.g. `_apply_min_samples_filter` unpacking a 3-tuple while the dataset yielded 4-tuples;
  every eval script still on `stat_input_dim=18` after the architecture changed to 16).
- **The 5-tuple insight.** The agent connected a teammate's vague "5-tuple issue" comment
  to a concrete leakage problem in our own code, explained *why* ports break zero-day
  generalization, and removed them cleanly across model + dataset + 4 eval scripts — which
  also deleted ~1M dead parameters and *strengthened* the design.
- **Verification discipline.** `py_compile` + live forward-pass after every structural
  change meant we never pushed code that didn't at least run.
- **Operational hand-holding.** Clear, ordered cloud-GPU instructions (tmux, venv
  activation, save-before-destroy) prevented the classic "lost the model when the instance
  died" disaster.
- **KPI-grounded architecture.** Tying margins to KPI thresholds got us to 90.9%
  classification and 0.728 intra-class similarity on the first clean run.

## 9. What did NOT work ❌ (and the lessons)

- **Thrashing on evaluation scripts.** Early on the agent over-iterated on zero-day
  evaluation methodology (ProtoNet → unbalanced k-NN at 8% → balanced k-shot), churning
  through approaches faster than it explained them. The human's blunt "what are you trying
  to do" was the right correction; the fix was to **slow down, pick one defensible method,
  and explain it.** *Lesson: an agent's eagerness to try things can outrun the human's
  ability to follow — narrate intent first.*
- **Chasing a domain gap that shouldn't have existed.** Significant effort went into
  fine-tuning across CESNET→ISCXVPN2016 before realizing the right move was to commit to a
  **single dataset** (CESNET) for both train and eval. *Lesson: question the experimental
  setup before optimizing within it.*
- **Fine-tune v2 regressed.** An aggressive encoder learning rate (3e-5) destroyed the
  pretrained features (val acc dropped to ~74%). We reverted to v1. *Lesson: when
  fine-tuning a contrastive encoder, protect the backbone with a tiny LR or freeze it.*
- **A latent evaluation flaw the agent flagged but we deferred.** The zero-day evaluator
  measures few-shot accuracy over *all* classes mixed, rather than isolating the held-out
  classes — so the reported number doesn't perfectly match the "unseen classes" story. The
  agent surfaced this honestly rather than hiding it. *Lesson: an agent that flags its own
  measurement's weaknesses is more valuable than one that only reports green checkmarks.*
- **`causal-conv1d` build failures.** Mamba's CUDA dependency repeatedly failed to build
  on fresh instances. Rather than fight it, the BiLSTM fallback was treated as a
  first-class path — and the shipped model in fact uses BiLSTM. *Lesson: design graceful
  fallbacks so a broken optional dependency never blocks the critical path.*

## 10. Honest assessment of agentic development here

The agent was strongest at **mechanical correctness at scale** (propagating a single
design decision across ~10 files without missing one), at **explaining tradeoffs**, and at
**operational discipline**. It was weakest when it **moved faster than it communicated** —
the moments that went wrong were moments where it should have stated a plan and waited.
The combination that worked was: **human sets direction and makes judgment calls; agent
reads, reasons, edits, and verifies — and says so out loud at each step.**

<!-- SCREENSHOT: a Claude Code session showing a tool-chain (Grep → Read → Edit → Bash verify) -->
<!-- SCREENSHOT: the commit history on feat/kpi-improvements authored during the session -->
