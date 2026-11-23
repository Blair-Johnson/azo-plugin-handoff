"""Dispatch /handoff through a real pipeline and command dispatcher with a fake launcher."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from agent_utils.core.events import IoCompleted, IoFailed
from agent_utils.core.io import SubmitOp
from agent_utils.core.pipeline import Feature, PipelineBuilder
from agent_utils.core.session import Session
from agent_utils.core.state import State
from agent_utils.features.core import core_feature
from agent_zoo.events import (
    EmitNotice,
    ReplyFrontendRequest,
    RunPluginTask,
    SlashCommand,
)
from agent_zoo.features.commands import (
    CommandContext,
    CommandError,
    CommandRegistry,
    SlashCommandDispatcher,
    commands_feature,
)
from agent_zoo.features.compaction import ContextCompactionState, ContextHandoffDoc
from agent_zoo.host.launch import Fresh, RuntimeLaunch
from agent_zoo.pipelines.default import build_default_builder

import handoff


class FakeLauncher:
    def __init__(self):
        self.requests = []

    def prepare(self, request):
        self.requests.append(request)
        return request

    def start(self, plan):
        return SimpleNamespace(stderr_path="/tmp/handoff/stderr.log")


class NoticeRecorder:
    handles = (EmitNotice,)

    def __init__(self):
        self.notices = []

    def handle(self, event, state, emit):
        del state, emit
        self.notices.append(event)


def make_doc(end, text="We were refactoring the parser."):
    return ContextHandoffDoc(
        text=text,
        created_at_turn=4,
        source_start_turn=0,
        source_end_turn=3,
        source_start_entry_index=0,
        source_end_entry_index=end,
        preserved_tail_start_turn=4,
        preserved_tail_start_entry_index=end + 1,
        token_estimate=10,
        model_config="m",
        profile="p",
        prompt_key="default",
    )


def make_state(tmp_path, docs=(), entries=12) -> State:
    cc = ContextCompactionState()
    cc.handoff_docs = list(docs)
    state = State(cwd=str(tmp_path))
    state.context_compaction = cc
    state.entries = [SimpleNamespace(index=i, step=0) for i in range(entries)]
    state.step = 6
    state._session_id = "parent-session"
    state._instance_id = "parent-instance"
    state._session_title = "Parser work"
    state._agent_zoo_context = {
        "project_name": "demo",
        "project_store_root": str(tmp_path / "store"),
    }
    state._persistence_local_root = str(tmp_path / "local")
    state.session_launcher = FakeLauncher()
    return state


CONFIG = {
    "llm": {"default": "opus"},
    "_runtime_launch": {"config_path": "", "config_sets": ("a.b=1",)},
}


def run_pipeline(state: State, raw="/handoff", config=CONFIG):
    recorder = NoticeRecorder()
    session = Session(state=state)
    builder = (
        PipelineBuilder()
        .add(core_feature())
        .add(commands_feature(config=config))
        .add(
            Feature(
                name="test_notice_recorder",
                components=[recorder],
                order=[SlashCommandDispatcher, NoticeRecorder],
            )
        )
    )
    handoff.register_features(builder, session=session, config=config)
    session.pipeline = builder.build(suppress_warnings=True)

    effects = session.step(SlashCommand(raw=raw))
    plugin_ops = [
        eff
        for eff in effects
        if isinstance(eff, SubmitOp) and isinstance(eff.request, RunPluginTask)
    ]
    attach_effects = []
    for op in plugin_ops:
        try:
            value = op.request.fn(*op.request.args, **op.request.kwargs)
            follow_effects = session.step(IoCompleted(op_id=op.op_id, value=value))
        except Exception as exc:
            follow_effects = session.step(IoFailed(op_id=op.op_id, error=exc))
        attach_effects.extend(
            eff.request
            for eff in follow_effects
            if isinstance(eff, SubmitOp) and isinstance(eff.request, ReplyFrontendRequest)
        )
    return recorder.notices, attach_effects


def notice_text(notices):
    return " ".join(notice.text for notice in notices)


def test_spawns_fresh_session_with_latest_handoff_and_attaches(tmp_path):
    state = make_state(tmp_path, [make_doc(5, "old memory"), make_doc(9, "newest memory")])
    notices, replies = run_pipeline(state)
    (request,) = state.session_launcher.requests
    assert isinstance(request, RuntimeLaunch) and isinstance(request.source, Fresh)
    assert request.kind == "handoff" and request.parent_session_id == "parent-session"
    assert request.title == "handoff: Parser work"
    assert request.first_user_message == ""
    assert "newest memory" in request.startup_system_message
    assert "old memory" not in request.startup_system_message
    assert request.settings.project == "demo" and request.settings.model == "opus"
    assert request.settings.config_sets == ("a.b=1",)
    assert request.target.session_id != "parent-session"
    assert len(replies) == 1
    assert replies[0].channel == "core"
    assert replies[0].event_type == "attach"
    assert replies[0].payload == {
        "session_id": request.target.session_id,
        "kind": "handoff",
    }
    assert request.target.session_id[:8] in notice_text(notices)


def test_message_becomes_first_user_turn(tmp_path):
    state = make_state(tmp_path, [make_doc(11)])
    run_pipeline(state, "/handoff  finish the parser tests ")
    (request,) = state.session_launcher.requests
    assert request.first_user_message == "finish the parser tests"


def test_reports_entries_newer_than_the_handoff(tmp_path):
    state = make_state(tmp_path, [make_doc(5)], entries=12)
    notices, _ = run_pipeline(state)
    (request,) = state.session_launcher.requests
    assert "6 more transcript entries" in request.startup_system_message
    assert "last 6 transcript entries" in notice_text(notices)


def test_current_handoff_has_no_staleness_note(tmp_path):
    state = make_state(tmp_path, [make_doc(11)], entries=12)
    notices, _ = run_pipeline(state)
    assert (
        "more transcript entries"
        not in state.session_launcher.requests[0].startup_system_message
    )
    assert "/compact" not in notice_text(notices)


def test_without_handoff_reports_error_and_launches_nothing(tmp_path):
    state = make_state(tmp_path)
    notices, replies = run_pipeline(state)
    assert notices[0].level == "error" and "/compact" in notices[0].text
    assert not state.session_launcher.requests and not replies


def test_skips_handoffs_beyond_a_rewound_transcript(tmp_path):
    state = make_state(
        tmp_path, [make_doc(5, "kept"), make_doc(30, "rewound away")], entries=12
    )
    run_pipeline(state)
    assert "kept" in state.session_launcher.requests[0].startup_system_message


def test_unavailable_without_live_backend(tmp_path):
    state = make_state(tmp_path, [make_doc(5)])
    state.session_launcher = None
    registry = CommandRegistry([handoff.make_handoff_command(CONFIG)])
    ctx = CommandContext(
        state=state,
        emit=lambda _ev: None,
        registry=registry,
        config=CONFIG,
    )
    with pytest.raises(CommandError, match="live agent backend"):
        registry.execute("/handoff", ctx)


def test_model_follows_live_config(tmp_path):
    config = {"llm": {"default": "opus"}}
    state = make_state(tmp_path, [make_doc(5)])
    command = handoff.make_handoff_command(config)
    config["llm"]["default"] = "sonnet"  # /model mutates the runtime config in place
    registry = CommandRegistry([command])
    ctx = CommandContext(
        state=state,
        emit=lambda _ev: None,
        registry=registry,
        config=config,
    )
    result = registry.execute("/handoff", ctx)
    assert result.operation_id is not None
    effects = state.io.take_outbox()
    (op,) = [eff for eff in effects if isinstance(eff, SubmitOp)]
    op.request.fn(*op.request.args, **op.request.kwargs)
    assert state.session_launcher.requests[0].settings.model == "sonnet"


def test_registers_one_feature():
    added = []
    handoff.register_features(SimpleNamespace(add=added.append), session=None, config={})
    (feature,) = added
    assert feature.name == "handoff_command"
    paths = [getattr(c, "path", None) for c in feature.components if hasattr(c, "path")]
    assert paths == ["/handoff"]
    assert any(
        isinstance(c, handoff.HandoffCompletionHandler) for c in feature.components
    )


def test_message_frames_handoff_as_user_action_and_ends_with_standby(tmp_path):
    state = make_state(tmp_path, [make_doc(11, "MEMORY")])
    run_pipeline(state)
    message = state.session_launcher.requests[0].startup_system_message
    assert message.startswith(
        "[System]: The user initialized this fresh conversation with a handoff document"
    )
    assert '"Parser work" (session parent-session)' in message
    memory = message.index("MEMORY")
    standby = message.index("Please stand by for new instructions")
    assert memory < standby and message.rstrip().endswith(
        "new instructions from the user."
    )


def test_composes_onto_default_builder():
    session = Session()
    builder = build_default_builder(
        session,
        config={},
        include_plugins=False,
        include_llm_drivers=False,
    )
    handoff.register_features(builder, session=session, config={})
    pipeline = builder.build()
    assert any(
        isinstance(comp, handoff.HandoffCompletionHandler) for comp in pipeline
    )
