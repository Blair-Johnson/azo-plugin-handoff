"""/handoff: start a fresh live session seeded with this session's latest handoff.

Unlike /fork, the new session does not inherit the transcript. It starts empty
except for one system message carrying the most recent context-compaction
handoff document, so it keeps the working memory without the token cost.
"""
from __future__ import annotations

from agent_utils import Feature
from agent_utils.session_cwd import get_session_cwd
from agent_zoo.context_compaction import latest_applicable_handoff
from agent_zoo.projects import resolve_state_root
from agent_zoo.runtime.commands import Arg, Command, CommandError, CommandResult
from agent_zoo.runtime_launch import Fresh, InstanceRef, RuntimeLaunch, RuntimeSettings
from agent_zoo.session_store import resolve_local_root

HANDOFF_KIND = "handoff"
MAX_TITLE_CHARS = 80

NO_HANDOFF = (
    "This session has no handoff document yet. Handoffs are written by harness "
    "context compaction: run /compact, let the agent take one more turn so the "
    "compaction runs, then retry /handoff."
)


def _entry_index(state) -> int:
    entries = getattr(state, "entries", None) or []
    if not entries:
        return -1
    index = getattr(entries[-1], "index", None)
    return int(index) if index is not None else len(entries) - 1


def _parent_title(state) -> str:
    for name in ("_session_title", "_session_name"):
        title = str(getattr(state, name, "") or "").strip()
        if title:
            return title
    return ""


def unsummarized_entries(state, doc) -> int:
    """Count transcript entries written after the handoff's source range."""
    return max(0, _entry_index(state) - int(doc.source_end_entry_index))


def handoff_message(state, doc) -> str:
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
    return f"{intro}\n\n[Handoff document]\n{doc.text.strip()}\n[End of handoff document]\n\n{closing}"


def launch_settings(state, config) -> RuntimeSettings:
    """Mirror the current runtime's project, roots, config and model."""
    config = config or {}
    context = dict(getattr(state, "_agent_zoo_context", {}) or {})
    provenance = config.get("_runtime_launch", {}) or {}
    model = str((config.get("llm", {}) or {}).get("default") or "")
    project = str(context.get("project_name") or "")
    if not project or not model:
        raise CommandError("/handoff needs a live backend with a known project and model.")
    return RuntimeSettings(
        project=project,
        workdir=get_session_cwd(state),
        state_root=str(resolve_state_root()),
        store_root=str(context.get("project_store_root") or ""),
        local_state_root=str(
            getattr(state, "_persistence_local_root", "") or resolve_local_root(config)
        ),
        config_path=str(provenance.get("config_path") or ""),
        config_sets=tuple(provenance.get("config_sets") or ()),
        model=model,
        env_name=str((config.get("pilot", {}) or {}).get("_env_name") or ""),
    )


def build_request(state, config, doc, message: str = "") -> RuntimeLaunch:
    target = InstanceRef.new_session()
    title = _parent_title(state)
    title = f"handoff: {title}"[:MAX_TITLE_CHARS] if title else f"handoff {target.session_id[:8]}"
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


def _unavailable(ctx) -> str | None:
    state = ctx.state
    if getattr(state, "session_launcher", None) is None or getattr(state, "session_io", None) is None:
        return "/handoff requires a live agent backend."
    return None


def make_handoff_command(config) -> Command:
    # config is the runtime's live dict; /model updates llm.default in place.
    def handoff(ctx, message=""):
        state = ctx.state
        doc = latest_applicable_handoff(state)
        if doc is None:
            return CommandResult.error(NO_HANDOFF)
        request = build_request(state, config, doc, message)
        launcher = state.session_launcher
        handle = launcher.start(launcher.prepare(request))
        child = request.target.session_id
        state.session_io.set("core.attach", {"session_id": child, "kind": HANDOFF_KIND})
        later = unsummarized_entries(state, doc)
        note = (
            f" The handoff predates the last {later} transcript entries; run /compact "
            f"first if you need them carried over." if later else ""
        )
        return CommandResult.notice(
            f"Spawned handoff session · {child[:8]} · seeded with handoff of entries "
            f"{doc.source_start_entry_index}-{doc.source_end_entry_index}.{note} "
            f"Logs: {handle.stderr_path}"
        )

    return Command(
        "/handoff",
        "Start a fresh session seeded with this session's latest handoff document.",
        handoff,
        args=(Arg("message", kind="text", default=""),),
        unavailable_reason=_unavailable,
        section="Sessions",
    )


def register_features(builder, *, session, config):
    builder.add(Feature(
        name="handoff_command",
        components=[make_handoff_command(config)],
        order=[Command],
    ))
