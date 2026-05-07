# VisionTrace Agent Guide

This repository uses an 8-agent operating model for larger Codex tasks. The
main agent remains responsible for scoping, conflict control, integration,
verification, and final delivery. Subagents should only be used when their work
can be bounded clearly and can run in parallel without blocking the next local
step.

For the full playbook, see `docs/subagents.md`. The reusable prompt pack lives
under `.agents/subagents/`; use `manifest.json`, `dispatcher.md`, and the
`roles/*.md` files as the source of truth when spawning subagents.

## Default Rules

- Do small, single-module fixes in the main agent.
- Use subagents for cross-module features, performance work, broad reviews, or
  documentation synchronization.
- Assign each implementation subagent an explicit ownership boundary.
- Tell implementation subagents that they are not alone in the codebase and
  must not revert or overwrite unrelated work.
- Keep supervision agents read-only by default. They should report risks,
  recommendations, and acceptance criteria unless explicitly asked to patch.
- The main agent makes the final tradeoff call, especially between latency,
  simplicity, feature completeness, and maintainability.

## Role Catalog

1. Capture Agent: screen capture, monitor selection, ROI, capture rate, dxcam.
2. Inference Agent: YOLO, TensorRT, Ultralytics, model args, class filtering.
3. Runtime Agent: process pipeline, shared buffers, overlay, smoothing, saving,
   metrics.
4. Web/API Agent: FastAPI, process manager, config forms, frontend controls,
   logs, charts.
5. Dataset/AutoLabel Agent: runtime image capture, batch annotation, shuffle,
   merge, YOLO labels.
6. QA/Review Agent: correctness review, regression risks, test strategy,
   uncommitted-change awareness.
7. Latency/Simplicity Agent: hot-path latency, blocking work, over-abstraction,
   avoidable state copying, logging/rendering overhead.
8. Docs Agent: README, architecture notes, run commands, config docs, API/UI
   behavior, change summaries.

## Common Dispatches

- Performance issue: Capture + Inference + Runtime + Latency/Simplicity.
- Web control-panel feature: Web/API + Runtime, add Docs when behavior changes.
- Auto-label feature: Dataset/AutoLabel + Web/API + QA, add Docs when the user
  flow changes.
- Large refactor: owning implementation agent + Latency/Simplicity + QA.
- Release cleanup: QA + Docs.
- Current dirty-worktree review: QA + Latency/Simplicity.

## Baseline Subagent Prompt

```text
You are the VisionTrace <role name>. You only own <responsibility boundary>.
Other people or agents may already have changes in this repository. Do not
revert, overwrite, or format unrelated files. First inspect the relevant files
and summarize the current shape. If code changes are requested, edit only your
owned file/module boundary. Final output must include findings, change summary,
files touched, verification, and remaining risks.
```

## Prompt Pack

- `.agents/subagents/manifest.json`: role catalog and common dispatches.
- `.agents/subagents/dispatcher.md`: main-agent dispatch procedure.
- `.agents/subagents/roles/*.md`: concrete prompt spec for each subagent.
