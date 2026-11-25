"""/handoff: start a fresh live session seeded with this session's latest handoff.

Unlike /fork, the new session does not inherit the transcript. It starts empty
except for one system message carrying the most recent context-compaction
handoff document, so it keeps the working memory without the token cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_utils.core.events import Event, IoCompleted, IoFailed, SessionRestored
from agent_utils.core.pipeline import EmitFn, Feature
from agent_utils.core.state import State
from agent_utils.features.buffers.tools import resolve_session_path
from agent_utils.features.core import IoFence
from agent_zoo.durability import session_context, session_state
from agent_zoo.events import (
    EmitNotice,
    ProjectionRequested,
    PublishView,
    SessionArtifact,
    SessionSource,
    SessionSpawnResult,
    SessionSpec,
    SpawnSession,
)
from agent_zoo.features.commands import (
    Arg,
    Command,
    CommandContext,
    CommandError,
    CommandResult,
)
from agent_zoo.features.compaction import latest_applicable_handoff

HANDOFF_KIND = "handoff"
MAX_TITLE_CHARS = 80
_RUNTIME_KEY = "handoff_command"


@dataclass
class _HandoffRuntime:
    pending_ops: dict[str, dict[str, Any]] = field(default_factory=dict)


NO_HANDOFF = (
    "This session has no handoff document yet. Handoffs are written by harness "
    "context compaction: run /compact, let the agent take one more turn so the "
    "compaction runs, then retry /handoff."
)


def _entry_index(state: Any) -> int:
    entries = getattr(state, "entries", None) or []
    if not entries:
        return -1
    index = getattr(entries[-1], "index", None)
    return int(index) if index is not None else len(entries) - 1


def _parent_title(state: Any) -> str:
    smeta = session_state(state)
    for raw in (smeta.title, smeta.name):
        title = str(raw or "").strip()
        if title:
            return title
    return ""


def _session_workdir(state: Any) -> str:
    has_cwd = any(
        str(getattr(state, attr, "") or "").strip()
        for attr in ("cwd", "session_cwd", "terminal_cwd")
    )
    return resolve_session_path("", state) if has_cwd else ""


def unsummarized_entries(state: Any, doc: Any) -> int:
    """Count transcript entries written after the handoff's source range."""
    return max(0, _entry_index(state) - int(doc.source_end_entry_index))


def handoff_message(state: Any, doc: Any) -> str:
    """Frame the handoff document as the new session's opening system message."""
    session_id = str(session_state(state).session_id or "")
    title = _parent_title(state)
    source = f'"{title}" (session {session_id})' if title else f"session {session_id}"
    intro = (
        f"[System]: The user initialized this fresh conversation with a handoff "
        f"document summarizing a previous conversation, {source}."
    )
    later = unsummarized_entries(state, doc)
    if later:
        intro += (
            f" The previous conversation had {later} more transcript entries after the "
            f"handoff was written, so the document may not reflect the most recent work."
        )
    closing = "[System]: Please stand by for new instructions from the user."
    return (
        f"{intro}\n\n[Handoff document]\n{doc.text.strip()}\n"
        f"[End of handoff document]\n\n{closing}"
    )


def _unavailable(ctx: CommandContext) -> str | None:
    smeta = session_state(ctx.state)
    has_live = bool(
        str(smeta.session_id or "").strip()
        and str(smeta.instance_id or "").strip()
    )
    return None if has_live else "/handoff requires a live agent backend."


def _pending_ops(state: Any) -> dict[str, dict[str, Any]]:
    runtime_fn = getattr(state, "runtime", None)
    if callable(runtime_fn):
        return runtime_fn(_RUNTIME_KEY, _HandoffRuntime).pending_ops
    ops = getattr(state, "_handoff_ops_fallback", None)
    if not isinstance(ops, dict):
        ops = {}
        setattr(state, "_handoff_ops_fallback", ops)
    return ops


class HandoffCompletionHandler:
    """Handle ``SpawnSession`` completion for ``/handoff`` and emit attach + notice."""

    handles = (SessionRestored, IoCompleted, IoFailed)

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        if isinstance(event, SessionRestored):
            _pending_ops(state).clear()
            return
        if not isinstance(event, (IoCompleted, IoFailed)):
            return
        ops = _pending_ops(state)
        if event.op_id not in ops:
            return
        meta = ops.pop(event.op_id)

        if isinstance(event, IoFailed):
            emit(EmitNotice(f"Failed to spawn handoff session: {event.error}", level="error"))
            emit(ProjectionRequested())
            return

        res = event.value
        if not isinstance(res, SessionSpawnResult) or not res.session_id:
            emit(EmitNotice("Failed to spawn handoff session: unexpected spawn result.", level="error"))
            emit(ProjectionRequested())
            return

        attach: dict[str, Any] = {"session_id": res.session_id, "kind": HANDOFF_KIND}
        if res.instance_id:
            attach["instance_id"] = res.instance_id
        if res.websocket_url:
            attach["websocket_url"] = res.websocket_url
        state.io.submit(PublishView(attach=attach), tag="frontend")

        notice = (
            f"Spawned handoff session · {res.session_id[:8]} · seeded with handoff of entries "
            f"{meta['doc_start']}-{meta['doc_end']}.{meta['note']}"
        )
        if res.stderr_path:
            notice += f" Logs: {res.stderr_path}"
        emit(EmitNotice(notice))
        emit(ProjectionRequested())


def make_handoff_command(config: dict[str, Any] | None) -> Command:
    # config is the runtime's live dict; /model updates llm.default in place.
    def handoff(ctx: CommandContext, message: str = "") -> CommandResult:
        state = ctx.state
        doc = latest_applicable_handoff(state)
        if doc is None:
            return CommandResult.error(NO_HANDOFF)

        smeta = session_state(state)
        sctx = session_context(state)
        cfg = (config if config is not None else ctx.config) or {}
        model = str((cfg.get("llm", {}) or {}).get("default") or getattr(state, "active_model", "") or "")
        project = str(smeta.project_name or sctx.project_name or "")
        if not project or not model:
            raise CommandError("/handoff needs a live backend with a known project and model.")

        parent_session_id = str(smeta.session_id or sctx.session_id or "")
        parent_title = _parent_title(state)
        title = f"handoff: {parent_title}"[:MAX_TITLE_CHARS] if parent_title else "handoff"
        later = unsummarized_entries(state, doc)
        note = (
            f" The handoff predates the last {later} transcript entries; run /compact "
            f"first if you need them carried over."
            if later
            else ""
        )
        handoff_doc = doc.text.strip()
        op_id = ctx.submit_io(
            SpawnSession(
                source=SessionSource(kind="fresh"),
                spec=SessionSpec(
                    project=project,
                    workdir=_session_workdir(state),
                    model=model,
                    env_name=str((cfg.get("pilot", {}) or {}).get("_env_name") or ""),
                    config_path=str(sctx.config_path or ""),
                    config_sets=tuple(sctx.config_sets or ()),
                    state_root=str(sctx.state_root or ""),
                    store_root=str(sctx.store_root or ""),
                    local_state_root=str(sctx.local_state_root or ""),
                    kind=HANDOFF_KIND,
                    title=title,
                    parent_session_id=parent_session_id,
                    startup_system_message=handoff_message(state, doc),
                    first_user_message=message.strip(),
                    spawn_policy="root",
                    lifetime="independent",
                ),
                inputs=(
                    SessionArtifact(
                        name="handoff_doc",
                        kind="text",
                        data=handoff_doc,
                        content_type="text/markdown",
                        metadata={
                            "source_session_id": parent_session_id,
                            "source_start_entry_index": doc.source_start_entry_index,
                            "source_end_entry_index": doc.source_end_entry_index,
                            "unsummarized_entries": later,
                        },
                    ),
                ),
                wait_ready=False,
            ),
            tag="handoff",
        )
        _pending_ops(state)[op_id] = {
            "doc_start": doc.source_start_entry_index,
            "doc_end": doc.source_end_entry_index,
            "note": note,
        }
        return CommandResult(outputs=(), operation_id=op_id, refresh=False)

    return Command(
        "/handoff",
        "Start a fresh session seeded with this session's latest handoff document.",
        handoff,
        args=(Arg("message", kind="text", default=""),),
        unavailable_reason=_unavailable,
        section="Sessions",
    )


def register_features(builder: Any, *, config: Any) -> None:
    builder.add(
        Feature(
            name="handoff_command",
            components=[make_handoff_command(config), HandoffCompletionHandler()],
            order=[IoFence, HandoffCompletionHandler, Command],
        )
    )
