"""/handoff: start a fresh live session seeded with this session's latest handoff.

Unlike /fork, the new session does not inherit the transcript. It starts empty
except for one system message carrying the most recent context-compaction
handoff document, so it keeps the working memory without the token cost.
"""

from __future__ import annotations

from typing import Any

from agent_utils.core.events import Event, IoCompleted, IoFailed
from agent_utils.core.pipeline import EmitFn, Feature
from agent_utils.core.state import State
from agent_utils.features.buffers.tools import resolve_session_path
from agent_utils.features.core import IoFence
from agent_zoo.events import (
    EmitNotice,
    ProjectionRequested,
    ReplyFrontendRequest,
    RunPluginTask,
)
from agent_zoo.features.commands import (
    Arg,
    Command,
    CommandContext,
    CommandError,
    CommandResult,
)
from agent_zoo.features.compaction import latest_applicable_handoff
from agent_zoo.host.launch import (
    Fresh,
    InstanceRef,
    RuntimeLaunch,
    RuntimeLauncher,
    RuntimeSettings,
)
from agent_zoo.storage.projects import resolve_state_root
from agent_zoo.storage.store import resolve_local_root

HANDOFF_KIND = "handoff"
MAX_TITLE_CHARS = 80
_HANDOFF_OPS_ATTR = "_handoff_ops"

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
    for name in ("_session_title", "_session_name"):
        title = str(getattr(state, name, "") or "").strip()
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
    session_id = str(getattr(state, "_session_id", "") or "")
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


def launch_settings(state: Any, config: dict[str, Any] | None) -> RuntimeSettings:
    """Mirror the current runtime's project, roots, config and model."""
    cfg = config or {}
    context = dict(getattr(state, "_agent_zoo_context", None) or {})
    provenance = cfg.get("_runtime_launch", {}) or {}
    model = str(
        (cfg.get("llm", {}) or {}).get("default")
        or getattr(state, "active_model", "")
        or ""
    )
    project = str(
        context.get("project_name")
        or getattr(state, "_project_name", "")
        or ""
    )
    if not project or not model:
        raise CommandError(
            "/handoff needs a live backend with a known project and model."
        )
    return RuntimeSettings(
        project=project,
        workdir=_session_workdir(state),
        state_root=str(resolve_state_root()),
        store_root=str(context.get("project_store_root") or ""),
        local_state_root=str(
            getattr(state, "_persistence_local_root", "") or resolve_local_root(cfg)
        ),
        config_path=str(provenance.get("config_path") or ""),
        config_sets=tuple(provenance.get("config_sets") or ()),
        model=model,
        env_name=str((cfg.get("pilot", {}) or {}).get("_env_name") or ""),
    )


def build_request(
    state: Any,
    config: dict[str, Any] | None,
    doc: Any,
    message: str = "",
) -> RuntimeLaunch:
    target = InstanceRef.new_session()
    title = _parent_title(state)
    title = (
        f"handoff: {title}"[:MAX_TITLE_CHARS]
        if title
        else f"handoff {target.session_id[:8]}"
    )
    return RuntimeLaunch(
        target=target,
        source=Fresh(),
        settings=launch_settings(state, config),
        parent_session_id=str(getattr(state, "_session_id", "") or ""),
        kind=HANDOFF_KIND,
        title=title,
        startup_system_message=handoff_message(state, doc),
        first_user_message=message.strip(),
        spawn_policy="root",
        lifetime="independent",
    )


def _launch_handoff(launcher: Any, request: RuntimeLaunch) -> Any:
    return launcher.start(launcher.prepare(request))


def _has_live_backend(state: Any) -> bool:
    if hasattr(state, "session_launcher"):
        return getattr(state, "session_launcher", None) is not None
    return bool(
        str(getattr(state, "_session_id", "") or "").strip()
        and str(getattr(state, "_instance_id", "") or "").strip()
    )


def _unavailable(ctx: CommandContext) -> str | None:
    if not _has_live_backend(ctx.state):
        return "/handoff requires a live agent backend."
    return None


def _pending_ops(state: Any) -> dict[str, dict[str, Any]]:
    ops = getattr(state, _HANDOFF_OPS_ATTR, None)
    if not isinstance(ops, dict):
        ops = {}
        setattr(state, _HANDOFF_OPS_ATTR, ops)
    return ops


class HandoffCompletionHandler:
    """Handle off-loop ``/handoff`` launch completion and emit attach + notice."""

    handles = (IoCompleted, IoFailed)
    init = {_HANDOFF_OPS_ATTR: dict}
    optional_reads = {_HANDOFF_OPS_ATTR}
    writes = {_HANDOFF_OPS_ATTR}

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        if not isinstance(event, (IoCompleted, IoFailed)):
            return
        ops = getattr(state, _HANDOFF_OPS_ATTR, None)
        if not isinstance(ops, dict):
            return
        meta = ops.pop(event.op_id, None)
        if meta is None:
            return

        if isinstance(event, IoFailed):
            emit(
                EmitNotice(
                    f"Failed to spawn handoff session: {event.error}",
                    level="error",
                )
            )
            emit(ProjectionRequested())
            return

        handle = event.value
        child = str(meta["child"])
        state.io.submit(
            ReplyFrontendRequest(
                channel="core",
                event_type="attach",
                payload={"session_id": child, "kind": HANDOFF_KIND},
            ),
            tag="frontend",
        )
        stderr_path = getattr(handle, "stderr_path", "")
        emit(
            EmitNotice(
                f"Spawned handoff session · {child[:8]} · seeded with handoff of entries "
                f"{meta['doc_start']}-{meta['doc_end']}.{meta['note']} "
                f"Logs: {stderr_path}"
            )
        )
        emit(ProjectionRequested())


def make_handoff_command(config: dict[str, Any] | None) -> Command:
    # config is the runtime's live dict; /model updates llm.default in place.
    def handoff(ctx: CommandContext, message: str = "") -> CommandResult:
        state = ctx.state
        doc = latest_applicable_handoff(state)
        if doc is None:
            return CommandResult.error(NO_HANDOFF)

        effective_config = config if config is not None else ctx.config
        request = build_request(state, effective_config, doc, message)
        launcher = getattr(state, "session_launcher", None) or RuntimeLauncher()
        child = request.target.session_id
        later = unsummarized_entries(state, doc)
        note = (
            f" The handoff predates the last {later} transcript entries; run /compact "
            f"first if you need them carried over."
            if later
            else ""
        )

        op_id = ctx.submit_io(
            RunPluginTask(fn=_launch_handoff, args=(launcher, request)),
            tag="handoff",
        )
        _pending_ops(state)[op_id] = {
            "child": child,
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


def register_features(builder: Any, *, session: Any, config: Any) -> None:
    del session
    builder.add(
        Feature(
            name="handoff_command",
            components=[
                make_handoff_command(config),
                HandoffCompletionHandler(),
            ],
            order=[IoFence, HandoffCompletionHandler, Command],
        )
    )
