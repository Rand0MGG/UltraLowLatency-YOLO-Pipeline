# VisionTrace Subagent Dispatcher

Use this dispatcher when the user explicitly asks for subagents, delegation, or
parallel agent work.

## Dispatch Procedure

1. Define the task outcome in one sentence.
2. Decide whether subagents are useful. Do not delegate small single-module
   fixes.
3. Choose the minimum role set from `manifest.json`.
4. Assign ownership:
   - Implementation agents get specific files or modules.
   - Read-only agents get specific questions and expected output.
5. Tell every implementation agent:
   - The repository may already contain user or agent changes.
   - Do not revert, overwrite, or format unrelated files.
   - Edit only the assigned ownership boundary.
6. Do immediate critical-path work locally instead of waiting by default.
7. Integrate subagent results, review patches if any, and run final
   verification.

## Common Dispatch Blocks

### Performance Issue

```text
Use four read-only VisionTrace agents:

Capture Agent: inspect capture rate, dxcam, monitor selection, and ROI behavior.
Inference Agent: inspect model loading, inference args, class filtering, and
detection output costs.
Runtime Agent: inspect process orchestration, shared buffer, overlay, saving,
and metrics/backpressure.
Latency/Simplicity Agent: review the others' findings for hot-path overhead,
blocking work, extra copies, and unnecessary complexity.

Return findings, concrete file references, likely bottlenecks, and suggested
verification steps. Do not edit files.
```

### Web Control-Panel Feature

```text
Use two VisionTrace implementation agents:

Web/API Agent owns FastAPI routes, form metadata, process-manager behavior, and
static frontend files.
Runtime Agent owns runtime argument plumbing and runtime behavior.

Both agents must avoid reverting unrelated changes and must only edit their
assigned files. Final output must list files touched, verification, and residual
risks.
```

### Auto-Label Workflow

```text
Use three VisionTrace agents:

Dataset/AutoLabel Agent owns runtime auto-label image and label workflow.
Web/API Agent owns auto-label API routes and UI actions.
QA/Review Agent is read-only and checks overwrite, merge, validation, and test
risks.

Implementation agents must keep write scopes disjoint and avoid unrelated
formatting.
```

### Release Cleanup

```text
Use two read-only VisionTrace agents:

QA/Review Agent checks correctness, regression risk, test coverage, and dirty
worktree concerns.
Docs Agent checks README, run commands, config behavior, API/UI descriptions,
and architecture notes against the actual code.

Return findings and recommended edits. Do not edit files unless asked.
```
