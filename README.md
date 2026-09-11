# Talos

Single-user autoresearch for [TIG](https://tig.foundation). Clone, run one setup
wizard, run one run wizard, and an LLM agent iterates on the current
state-of-the-art algorithm for a TIG challenge until it beats it on TIG's own
benchmark harness, or runs out of budget. You get back a submit-ready package.

Status: design accepted, implementation not started. Read
[docs/superpowers/specs/2026-09-11-talos-design.md](docs/superpowers/specs/2026-09-11-talos-design.md).

Talos lifts its research-loop pieces from
[prometheus-swarm](https://github.com/tig-foundation/prometheus-swarm) and its
scoring from a TIG pentesting harness; it is licensed under the GPLv3 like the
former.
