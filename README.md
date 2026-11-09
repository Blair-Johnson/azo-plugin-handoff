# azo-plugin-handoff

An Agent Zoo plugin that adds a `/handoff` command. It starts a fresh live
session whose only inherited context is the current session's most recent
handoff document, and then attaches the TUI to that session.

`/fork` copies the entire transcript into the new session. `/handoff` instead
carries over the working memory that context compaction has already distilled,
so the new agent starts with a small, focused context.

## Usage

```text
/handoff                       # open a fresh session seeded with the handoff
/handoff finish the parser     # also send a first message to the new agent
```

The new session receives one system message that names the source session and
the transcript range that the handoff covers, followed by the handoff text. If
the source session continued after the handoff was written, the message and the
command notice both say how many transcript entries were not summarized.

Handoff documents are produced by harness context compaction. When no handoff
exists yet, `/handoff` reports an error and launches nothing. Run `/compact`,
let the agent take one more turn so that compaction runs, and then retry.
Provider-side ("server") compaction does not produce a handoff document.

The new session is independent of the source session. It uses the source's
project, working directory, configuration, and current model, has the session
kind `handoff`, and records the source as its parent session. The source
session keeps running.

## Install

```bash
pixi run azo-plugin install ~/Documents/GitHub/azo-plugin-handoff
```

Then run `/reload` in the AZO TUI. The plugin loads only in the interactive
(`default`) pipeline, because RLM sub-agents have no TUI to attach.

## Test

```bash
python scripts/test.py                               # unit tests
AZO_HANDOFF_INTEGRATION=1 python scripts/test.py     # also launch a real child runtime
```

The test runner uses the installed Agent Zoo Pixi environment. Set
`AZO_HOST_MANIFEST` or `AZO_HOST_ENVIRONMENT` to target another one. The
integration test uses an isolated temporary `AGENT_ZOO_HOME` and makes no model
call.
