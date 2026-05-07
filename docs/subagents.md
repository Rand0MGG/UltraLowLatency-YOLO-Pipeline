# VisionTrace 8-Agent Subagent Playbook

## Summary

VisionTrace is a low-latency Windows screen vision toolkit with separate
capture, inference, runtime, Web/API, and dataset workflows. Use this playbook
when a task is too broad for a single focused pass, needs parallel exploration,
or needs dedicated review for latency, simplicity, or documentation drift.

The main agent is always the integrator. It decides whether to delegate, keeps
write scopes disjoint, reviews results, resolves tradeoffs, and owns final
verification. The implemented prompt pack is stored in `.agents/subagents/`.

## Operating Rules

- Do not use subagents for small, single-file fixes.
- Use only the agents needed for the current task, usually 1 to 4.
- Give every implementation agent a concrete module or file boundary.
- Tell every implementation agent that other changes may exist and must not be
  reverted.
- Keep QA/Review, Latency/Simplicity, and Docs read-only unless the main agent
  explicitly asks for edits.
- Prefer behavior-level handoffs over vague exploration. Each subagent should
  return findings, changed files if any, verification, and residual risk.

## Implemented Prompt Pack

- `.agents/subagents/manifest.json`: machine-readable role catalog and dispatch
  matrix.
- `.agents/subagents/dispatcher.md`: reusable dispatch procedure and common
  task blocks.
- `.agents/subagents/roles/`: one prompt specification per agent role.

## Agent Roles

### 1. Capture Agent

Purpose: screen capture and capture-region behavior.

Primary areas:

- `src/cvcap/adapters/capture/`
- Capture setup and ROI path in `src/cvcap/app/runtime.py`
- `monitor_index`, `capture_hz`, `roi_square`, `roi_radius_px`

Use for:

- dxcam compatibility issues.
- Monitor selection bugs.
- Capture-rate or ROI behavior.
- Capture-side performance bottlenecks.

### 2. Inference Agent

Purpose: model execution and detection output.

Primary areas:

- `src/cvcap/adapters/inference/`
- `src/cvcap/core/detections.py`
- Model-related fields in `RunnerArgs`

Use for:

- YOLO, TensorRT, or Ultralytics behavior.
- `conf`, `iou`, `imgsz`, `half`, `end2end`, `yolo_classes`, `yolo_max_det`.
- Detection box or keypoint output shape.
- Model-load and inference latency questions.

### 3. Runtime Agent

Purpose: real-time pipeline orchestration.

Primary areas:

- `src/cvcap/app/runtime.py`
- `src/cvcap/runtime/`

Use for:

- Multiprocess capture/inference/render coordination.
- Shared-memory buffer behavior.
- Overlay, smoothing, saving, metrics, and shutdown.
- Dropped frames, flow metrics, and runtime backpressure.

### 4. Web/API Agent

Purpose: local control panel and backend API.

Primary areas:

- `src/cvcap/web/server.py`
- `src/cvcap/web/process_manager.py`
- `src/cvcap/web/forms.py`
- `src/cvcap/web/static/`

Use for:

- FastAPI routes and payload conversion.
- Config save/load and CLI preview behavior.
- Start/stop/status/logs in the Web UI.
- Frontend controls, charts, language strings, and layout.

### 5. Dataset/AutoLabel Agent

Purpose: runtime dataset capture and label preparation.

Primary areas:

- `src/cvcap/runtime/auto_label.py`
- `/api/auto-label/*` routes
- `datasets/` folder conventions

Use for:

- Auto image capture triggers.
- Shuffle, annotate, and merge workflows.
- YOLO label generation and image/label pairing.
- Data safety checks before moving, renaming, or merging datasets.

### 6. QA/Review Agent

Purpose: correctness, regression, and risk review.

Default mode: read-only.

Use for:

- Dirty-worktree awareness before broad edits.
- Bug and regression review.
- Test-case design.
- API, config, and UI consistency checks.
- Release or handoff readiness.

Expected output:

- Findings ordered by severity.
- Test gaps and residual risks.
- Suggested verification commands.
- Specific files or behaviors to recheck.

### 7. Latency/Simplicity Agent

Purpose: protect low latency and keep the codebase simple.

Default mode: read-only.

Review priorities:

- Hot-path overhead in capture, inference, runtime, and overlay.
- Blocking I/O, synchronous waits, avoidable locks, and unbounded queues.
- Extra frame copies, repeated conversions, or avoidable state duplication.
- Excessive logging, charting, saving, or rendering work on runtime paths.
- Over-abstracted code that hides simple flow or makes tuning harder.

Expected output categories:

- Must fix.
- Should consider.
- Keep as-is.

### 8. Docs Agent

Purpose: keep project documentation aligned with real behavior.

Default mode: read-only until code behavior is stable.

Primary areas:

- `readme.md`
- `AGENTS.md`
- `docs/`
- Run commands, config notes, API/UI descriptions, architecture diagrams

Use for:

- README and architecture drift.
- New feature usage docs.
- Release notes or change summaries.
- Clarifying config fields and workflows.

Docs Agent must not describe features that are not implemented.

## Dispatch Matrix

| Task type | Recommended agents |
| --- | --- |
| Capture or ROI issue | Capture + Runtime |
| Model or detection issue | Inference + Runtime |
| End-to-end performance issue | Capture + Inference + Runtime + Latency/Simplicity |
| Web control-panel feature | Web/API + Runtime |
| New runtime config option | Runtime + Web/API + QA |
| Auto-label workflow change | Dataset/AutoLabel + Web/API + QA |
| Dataset safety review | Dataset/AutoLabel + QA |
| Large refactor | Owning agent + Latency/Simplicity + QA |
| Release cleanup | QA + Docs |
| Documentation update | Docs, plus owning implementation agent if behavior is unclear |

## Prompt Templates

### Baseline Implementation Agent

```text
You are the VisionTrace <Agent Name>. You only own <responsibility boundary>.
Other people or agents may already have changes in this repository. Do not
revert, overwrite, or format unrelated files. First inspect the relevant files
and summarize the current shape. If code changes are requested, edit only your
owned file/module boundary. Final output must include findings, change summary,
files touched, verification, and remaining risks.
```

### Read-Only Analysis Agent

```text
You are the VisionTrace <Agent Name>. This is a read-only task. Inspect only the
files relevant to <question>. Do not edit files. Return concise findings,
specific file references, suggested next steps, verification ideas, and open
risks.
```

### Latency/Simplicity Agent

```text
You are the VisionTrace Latency/Simplicity Agent. Review this plan or change for
low-latency and simplicity risks only. Focus on hot-path overhead, blocking I/O,
locks, queues, frame copies, repeated conversions, excess logging/rendering, and
over-abstraction. Output exactly three sections: Must fix, Should consider, and
Keep as-is. Do not edit files unless the main agent explicitly asks.
```

### Docs Agent

```text
You are the VisionTrace Docs Agent. Review or update documentation according to
the code's real behavior. Do not document features that are not implemented.
Focus on README, run commands, config fields, API/UI behavior, architecture
diagrams, and change summaries. Output needed doc updates, proposed wording or
patch summary, touched files if any, and whether any docs depend on pending code
changes.
```

## Verification Scenarios

- Low-latency optimization: Capture, Inference, and Runtime agents identify
  bottlenecks; Latency/Simplicity confirms whether the final plan reduces
  latency without adding unnecessary complexity.
- New Web config option: Runtime owns the runtime parameter, Web/API owns form
  and route behavior, QA checks config persistence and CLI preview, Docs updates
  usage notes if the user-facing workflow changes.
- Auto-label enhancement: Dataset/AutoLabel owns image and label workflow,
  Web/API owns UI/API calls, QA checks overwrite and merge risks, Docs updates
  the workflow only after behavior is implemented.
- Release readiness: QA reviews correctness and residual risk, Docs checks that
  README and architecture notes match current behavior.

## Success Criteria

- Each subagent has a clear reason to exist for the task.
- Implementation agents do not compete for the same files.
- Supervision agents catch latency, complexity, and documentation drift early.
- The main agent can integrate results without redoing the delegated work.
- Final delivery includes verification and known residual risks.
