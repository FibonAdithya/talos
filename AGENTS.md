# Agent guide

Read this first, then follow the links. This file is a router: it says which
document is authoritative, what must not be broken, and what needs a human.
It restates as little as possible, because a second copy of a fact is a copy
that goes stale.

## What this project is

<<FILL: Two to four sentences. What Talos does, who or what consumes
its output, and what success means for it. Say what it is not, if that is a
common misreading. Guidance: docs/adopting.md#what-this-project-is in the
agentify repo.>>

## Source of truth, in order

When two documents disagree, the one higher in this list wins.

1. **The code** and its configuration. If a document describes behaviour the
   code does not have, the document is wrong.
2. **`README.md`** — setup and the commands you run day to day.
3. **`docs/ai/`** — *not authoritative*. Design specs and plans written by
   agents during development, kept for the reasoning behind decisions. They
   are not updated as the code changes. See `docs/ai/README.md`. Agents
   writing a new design spec put it in `docs/ai/specs/`; implementation
   plans go in `docs/ai/plans/`; nowhere else.

<<FILL: If the project has a technical reference, a data contract, or a
config schema document, insert it between the code and README.md and say
what it covers. Otherwise delete this marker.>>

## Invariants

These are silent until violated. Nothing in the test suite catches them, and
each is easy to break while believing you are making progress.

<<FILL: A numbered list. One invariant per item: the rule, why it exists, and
where the truth lives. Find them by reading the tests for what is covered and
the configs and data contracts for what is not; an invariant is the mistake an
agent would make while believing it was making progress. Guidance:
docs/adopting.md#finding-invariants in the agentify repo.>>

## What "done" means

Run from the repo root:

    make check

That is the same command CI runs (`.github/workflows/ci.yml`). No target uses
`|| true`; a red suite is a failure, not a warning. The `Makefile` is the
executable definition of a valid change.

## What requires a human

Do not decide these yourself. Raise them and stop.

<<FILL: A bulleted list. Typical entries: changing what a version or variant
number means, tightening a threshold or gate band, changing a pinned
dependency, anything that redefines what an existing result meant. Guidance:
docs/adopting.md#what-requires-a-human in the agentify repo.>>

To report a bug in this project, file an issue with the `agent-reported`
label:

    gh issue create --label agent-reported --title "..." --body "..."

The label is what routes the issue to a person. An issue without it notifies
nobody.

## Where to look

| Task | Start here |
|---|---|
| Set up and run day-to-day commands | `README.md` |
| Run the gate | `Makefile` |
| Why a decision was made (non-authoritative) | `docs/ai/specs/` |
<<FILL: One row per recurring task: the entry point, the data contract, the
configs, where the tests for a subsystem live. Cite documents by anchor
(doc.md#heading) and code by symbol (src/x.py::name), never by line number;
tests/test_docs_references.py rejects line numbers.>>
