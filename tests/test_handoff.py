"""Dispatch /handoff through a real pipeline and SimSessionDriver (100% Sans-I/O)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from agent_utils.core.events import IoFailed
from agent_utils.core.io import SubmitOp
from agent_utils.core.pipeline import Feature, PipelineBuilder
from agent_utils.core.session import Session
from agent_utils.core.state import State
from agent_utils.features.core import core_feature
from agent_utils.runtime.simulation import SimOutcome, SimulationRuntime
from agent_zoo.drivers.sessions import SimSessionDriver
from agent_zoo.durability import session_context, session_state
from agent_zoo.events import (
    EmitNotice,
    PublishView,
    SessionArtifact,
    SessionSource,
    SessionSpawnResult,
    SessionSpec,
    SlashCommand,
    SpawnSession,
)
from agent_zoo.features.commands import (
    CommandContext,
    CommandError,
    CommandRegistry,
    SlashCommandDispatcher,
    commands_feature,
)
from agent_zoo.features.compaction import ContextCompactionState, ContextHandoffDoc
from agent_zoo.pipelines.default import build_default_builder

import handoff


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
    smeta = session_state(state)
    smeta.session_id = "parent-session"
    smeta.instance_id = "parent-instance"
    smeta.title = "Parser work"
    smeta.project_name = "demo"
    sctx = session_context(state)
    sctx.project_name = "demo"
    sctx.store_root = str(tmp_path / "store")
    sctx.state_root = str(tmp_path / "state")
    sctx.local_state_root = str(tmp_path / "local")
    sctx.config_path = ""
    sctx.config_sets = ("a.b=1",)
    return state


CONFIG = {
    "llm": {"default": "opus"},
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
    handoff.register_features(builder, config=config)
    session.pipeline = builder.build(suppress_warnings=True)

    sim_sessions = SimSessionDriver()
    spawn_requests: list[SpawnSession] = []
    publish_views: list[PublishView] = []

    sim = SimulationRuntime(session)

    def _on_spawn(op_id, req, runtime):
        spawn_requests.append(req)
        outcome = sim_sessions.handle(op_id, req, runtime)
        if isinstance(outcome, SimOutcome) and isinstance(outcome.value, SessionSpawnResult):
            outcome.value.stderr_path = "/tmp/handoff/stderr.log"
        return outcome

    def _on_publish_view(_op_id, req, _runtime):
        publish_views.append(req)
        return SimOutcome(value=None, delay=0.0)

    sim.register_handler(SpawnSession, _on_spawn)
    sim.register_handler(PublishView, _on_publish_view)
    sim.inject(SlashCommand(raw=raw))
    sim.run_until_idle()
    return recorder.notices, spawn_requests, publish_views, sim_sessions


def notice_text(notices):
    return " ".join(notice.text for notice in notices)


def test_spawns_fresh_session_with_latest_handoff_and_attaches(tmp_path):
    state = make_state(tmp_path, [make_doc(5, "old memory"), make_doc(9, "newest memory")])
    notices, requests, views, sim_sessions = run_pipeline(state)
    (request,) = requests
    assert isinstance(request, SpawnSession)
    assert request.source == SessionSource(kind="fresh")
    assert request.wait_ready is False
    assert isinstance(request.spec, SessionSpec)
    assert request.spec.kind == "handoff"
    assert request.spec.spawn_policy == "root"
    assert request.spec.lifetime == "independent"
    assert request.spec.parent_session_id == "parent-session"
    assert request.spec.title == "handoff: Parser work"
    assert request.spec.first_user_message == ""
    assert "newest memory" in request.spec.startup_system_message
    assert "old memory" not in request.spec.startup_system_message
    assert request.spec.project == "demo" and request.spec.model == "opus"
    assert request.spec.config_sets == ("a.b=1",)
    assert request.spec.state_root == str(tmp_path / "state")
    assert request.spec.store_root == str(tmp_path / "store")
    assert request.spec.local_state_root == str(tmp_path / "local")

    assert len(request.inputs) == 1
    art = request.inputs[0]
    assert isinstance(art, SessionArtifact)
    assert art.name == "handoff_doc"
    assert art.kind == "text"
    assert art.content_type == "text/markdown"
    assert art.data == "newest memory"
    assert art.metadata["source_session_id"] == "parent-session"
    assert art.metadata["source_end_entry_index"] == 9

    (spawned,) = sim_sessions.sessions.values()
    assert spawned.session_id != "parent-session"
    assert len(views) == 1
    assert views[0].attach == {
        "session_id": spawned.session_id,
        "kind": "handoff",
        "instance_id": spawned.instance_id,
        "websocket_url": spawned.websocket_url,
    }
    assert spawned.session_id[:8] in notice_text(notices)
    assert "/tmp/handoff/stderr.log" in notice_text(notices)


def test_message_becomes_first_user_turn(tmp_path):
    state = make_state(tmp_path, [make_doc(11)])
    _, requests, _, _ = run_pipeline(state, "/handoff  finish the parser tests ")
    (request,) = requests
    assert request.spec.first_user_message == "finish the parser tests"


def test_reports_entries_newer_than_the_handoff(tmp_path):
    state = make_state(tmp_path, [make_doc(5)], entries=12)
    notices, requests, _, _ = run_pipeline(state)
    (request,) = requests
    assert "6 more transcript entries" in request.spec.startup_system_message
    assert "last 6 transcript entries" in notice_text(notices)
    assert request.inputs[0].metadata["unsummarized_entries"] == 6


def test_current_handoff_has_no_staleness_note(tmp_path):
    state = make_state(tmp_path, [make_doc(11)], entries=12)
    notices, requests, _, _ = run_pipeline(state)
    assert "more transcript entries" not in requests[0].spec.startup_system_message
    assert "/compact" not in notice_text(notices)


def test_without_handoff_reports_error_and_launches_nothing(tmp_path):
    state = make_state(tmp_path)
    notices, requests, views, _ = run_pipeline(state)
    assert notices[0].level == "error" and "/compact" in notices[0].text
    assert not requests and not views


def test_skips_handoffs_beyond_a_rewound_transcript(tmp_path):
    state = make_state(
        tmp_path, [make_doc(5, "kept"), make_doc(30, "rewound away")], entries=12
    )
    _, requests, _, _ = run_pipeline(state)
    assert "kept" in requests[0].spec.startup_system_message
    assert requests[0].inputs[0].data == "kept"


def test_unavailable_without_live_backend(tmp_path):
    state = make_state(tmp_path, [make_doc(5)])
    session_state(state).session_id = ""
    registry = CommandRegistry([handoff.make_handoff_command(CONFIG)])
    ctx = CommandContext(
        state=state,
        emit=lambda _ev: None,
        registry=registry,
        config=CONFIG,
    )
    with pytest.raises(CommandError, match="live agent backend"):
        registry.execute("/handoff", ctx)


def test_handoff_pipeline_is_strictly_sans_io():
    from agent_utils.core.session import Session
    from agent_zoo.pipelines.default import build_default_builder
    from agent_zoo.testing import assert_pipeline_sans_io

    session = Session()
    builder = build_default_builder(
        session, config={}, include_plugins=False, include_llm_drivers=False
    )
    handoff.register_features(builder, config={})
    assert_pipeline_sans_io(builder.build())


def test_missing_project_or_model_raises_command_error(tmp_path):
    state = make_state(tmp_path, [make_doc(5)])
    session_state(state).project_name = ""
    session_context(state).project_name = ""
    registry = CommandRegistry([handoff.make_handoff_command({"llm": {"default": "opus"}})])
    ctx = CommandContext(
        state=state,
        emit=lambda _ev: None,
        registry=registry,
        config={"llm": {"default": "opus"}},
    )
    with pytest.raises(CommandError, match="known project and model"):
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
    assert isinstance(op.request, SpawnSession)
    assert op.request.spec.model == "sonnet"


def test_spawn_failure_emits_error_notice_and_no_attach(tmp_path):
    state = make_state(tmp_path, [make_doc(5)])
    recorder = NoticeRecorder()
    session = Session(state=state)
    builder = (
        PipelineBuilder()
        .add(core_feature())
        .add(commands_feature(config=CONFIG))
        .add(
            Feature(
                name="test_notice_recorder",
                components=[recorder],
                order=[SlashCommandDispatcher, NoticeRecorder],
            )
        )
    )
    handoff.register_features(builder, config=CONFIG)
    session.pipeline = builder.build(suppress_warnings=True)

    effects = session.step(SlashCommand(raw="/handoff"))
    (spawn_op,) = [
        eff for eff in effects if isinstance(eff, SubmitOp) and isinstance(eff.request, SpawnSession)
    ]
    follow = session.step(IoFailed(op_id=spawn_op.op_id, error=RuntimeError("boom")))
    assert not [
        eff for eff in follow if isinstance(eff, SubmitOp) and isinstance(eff.request, PublishView)
    ]
    assert recorder.notices[-1].level == "error"
    assert "boom" in recorder.notices[-1].text


def test_registers_one_feature():
    added = []
    handoff.register_features(SimpleNamespace(add=added.append), config={})
    (feature,) = added
    assert feature.name == "handoff_command"
    paths = [getattr(c, "path", None) for c in feature.components if hasattr(c, "path")]
    assert paths == ["/handoff"]
    assert any(
        isinstance(c, handoff.HandoffCompletionHandler) for c in feature.components
    )


def test_message_frames_handoff_as_user_action_and_ends_with_standby(tmp_path):
    state = make_state(tmp_path, [make_doc(11, "MEMORY")])
    _, requests, _, _ = run_pipeline(state)
    message = requests[0].spec.startup_system_message
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
    handoff.register_features(builder, config={})
    pipeline = builder.build()
    assert any(
        isinstance(comp, handoff.HandoffCompletionHandler) for comp in pipeline
    )


def test_module_is_pure_sans_io():
    for forbidden in (
        "RuntimeLauncher",
        "RuntimeLaunch",
        "RuntimeSettings",
        "RunPluginTask",
        "ReplyFrontendRequest",
        "launch_settings",
        "build_request",
        "resolve_state_root",
        "resolve_local_root",
    ):
        assert not hasattr(handoff, forbidden), f"unexpected legacy symbol: {forbidden}"


def test_spawn_session_passes_real_session_driver_contract(tmp_path):
    """Verify SpawnSession + SessionArtifact flows cleanly through AgentZooSessionDriver."""
    import json
    import queue
    from agent_utils.core.events import IoCompleted
    from agent_zoo.drivers.sessions import AgentZooSessionDriver

    captured = {}

    class FakeDriverLauncher:
        def prepare(self, request, *, cwd=None, env=None):
            captured["request"] = request
            captured["cwd"] = cwd
            captured["env"] = dict(env or {})
            return SimpleNamespace(request=request, cwd=cwd, environment=captured["env"])

        def start(self, plan):
            return SimpleNamespace(
                request=plan.request,
                process=SimpleNamespace(pid=4242, poll=lambda: None),
                ready_file=str(tmp_path / "ready.json"),
                stdout_path=str(tmp_path / "stdout.log"),
                stderr_path=str(tmp_path / "stderr.log"),
            )

    state = make_state(tmp_path, [make_doc(9, "handoff body")])
    session = Session(state=state)
    builder = (
        PipelineBuilder()
        .add(core_feature())
        .add(commands_feature(config=CONFIG))
    )
    handoff.register_features(builder, config=CONFIG)
    session.pipeline = builder.build(suppress_warnings=True)

    effects = session.step(SlashCommand(raw="/handoff continue work"))
    (spawn_op,) = [
        eff for eff in effects if isinstance(eff, SubmitOp) and isinstance(eff.request, SpawnSession)
    ]

    inbox: queue.Queue = queue.Queue()
    driver = AgentZooSessionDriver(
        launcher=FakeDriverLauncher(),
        store_root=str(tmp_path / "store"),
        local_state_root=str(tmp_path / "local"),
    )
    driver.start(inbox.put)
    try:
        driver.submit(spawn_op.op_id, spawn_op.request)
        event = inbox.get(timeout=5)
    finally:
        driver.close(leave_children=True)

    assert isinstance(event, IoCompleted)
    assert isinstance(event.value, SessionSpawnResult)
    assert event.value.stderr_path == str(tmp_path / "stderr.log")

    launch = captured["request"]
    assert launch.kind == "handoff"
    assert launch.spawn_policy == "root"
    assert launch.lifetime == "independent"
    assert launch.first_user_message == "continue work"
    assert "handoff body" in launch.startup_system_message

    encoded_inputs = json.loads(captured["env"]["AZO_SESSION_INPUTS"])
    assert len(encoded_inputs) == 1
    assert encoded_inputs[0]["name"] == "handoff_doc"
    assert encoded_inputs[0]["kind"] == "text"
    assert encoded_inputs[0]["content_type"] == "text/markdown"
    assert encoded_inputs[0]["data"] == "handoff body"

    follow = session.step(event)
    (view_op,) = [
        eff for eff in follow if isinstance(eff, SubmitOp) and isinstance(eff.request, PublishView)
    ]
    assert view_op.request.attach == {
        "session_id": event.value.session_id,
        "kind": "handoff",
        "instance_id": event.value.instance_id,
    }
