# VisionTrace Subagent Prompt Pack

This directory contains the persistent prompt definitions for the VisionTrace
8-agent operating model. These files are intentionally plain Markdown and JSON
so they can be read by Codex, copied into spawned subagent prompts, or reviewed
by humans without special tooling.

This prompt pack does not run agents by itself. The main agent uses it as the
source of truth when the user asks to use subagents.

## Files

- `manifest.json`: machine-readable role catalog and dispatch matrix.
- `dispatcher.md`: main-agent dispatch procedure and ready-to-copy task blocks.
- `roles/*.md`: one reusable prompt specification per subagent.

## Usage

When the user asks for subagents, the main agent should:

1. Read `manifest.json` and the relevant role files.
2. Choose only the agents needed for the task.
3. Give each implementation agent a disjoint ownership boundary.
4. Keep QA/Review, Latency/Simplicity, and Docs read-only unless edits are
   explicitly requested.
5. Integrate results and own final verification.

Example user request:

```text
Use Capture, Inference, Runtime, and Latency/Simplicity agents to investigate
the current dropped-frame problem.
```

Example main-agent action:

```text
Spawn Capture, Inference, and Runtime as read-only explorers for independent
pipeline analysis. Spawn Latency/Simplicity as a read-only reviewer of their
proposed optimizations. Continue locally on the critical path while they run.
```
