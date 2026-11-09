"""Dispatch /handoff through a real command registry with a fake launcher."""
from types import SimpleNamespace

import pytest
from agent_zoo.context_compaction import ContextCompactionState, ContextHandoffDoc
from agent_zoo.runtime.commands import CommandContext, CommandError, CommandRegistry
from agent_zoo.runtime_launch import Fresh, RuntimeLaunch

import handoff


class FakeLauncher:
    def __init__(self):
        self.requests = []

    def prepare(self, request):
        self.requests.append(request)
        return request

    def start(self, plan):
        return SimpleNamespace(stderr_path="/tmp/handoff/stderr.log")


class FakeIO:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value


def make_doc(end, text="We were refactoring the parser."):
    return ContextHandoffDoc(
        text=text, created_at_turn=4, source_start_turn=0, source_end_turn=3,
        source_start_entry_index=0, source_end_entry_index=end,
        preserved_tail_start_turn=4, preserved_tail_start_entry_index=end + 1,
        token_estimate=10, model_config="m", profile="p", prompt_key="default",
    )


def make_state(tmp_path, docs=(), entries=12):
    cc = ContextCompactionState()
    cc.handoff_docs = list(docs)
    return SimpleNamespace(
        context_compaction=cc,
        entries=[SimpleNamespace(index=i, step=0) for i in range(entries)],
        step=6,
        _session_id="parent-session",
        _session_title="Parser work",
        _agent_zoo_context={"project_name": "demo", "project_store_root": str(tmp_path / "store")},
        _persistence_local_root=str(tmp_path / "local"),
        session_cwd=str(tmp_path),
        session_launcher=FakeLauncher(),
        session_io=FakeIO(),
    )


CONFIG = {"llm": {"default": "opus"}, "_runtime_launch": {"config_path": "", "config_sets": ("a.b=1",)}}


def run(state, raw="/handoff", config=CONFIG):
    registry = CommandRegistry([handoff.make_handoff_command(config)])
    ctx = CommandContext(runtime=None, session=SimpleNamespace(state=state))
    return registry.execute(raw, ctx)


def notice_text(result):
    return " ".join(output.text for output in result.outputs)


def test_spawns_fresh_session_with_latest_handoff_and_attaches(tmp_path):
    state = make_state(tmp_path, [make_doc(5, "old memory"), make_doc(9, "newest memory")])
    result = run(state)
    (request,) = state.session_launcher.requests
    assert isinstance(request, RuntimeLaunch) and isinstance(request.source, Fresh)
    assert request.kind == "handoff" and request.parent_session_id == "parent-session"
    assert request.title == "handoff: Parser work"
    assert request.first_user_message == ""
    assert "newest memory" in request.startup_system_message
    assert "old memory" not in request.startup_system_message
    assert "entries 0-9" in request.startup_system_message
    assert request.settings.project == "demo" and request.settings.model == "opus"
    assert request.settings.config_sets == ("a.b=1",)
    assert request.target.session_id != "parent-session"
    assert state.session_io.values["core.attach"] == {
        "session_id": request.target.session_id, "kind": "handoff",
    }
    assert request.target.session_id[:8] in notice_text(result)


def test_message_becomes_first_user_turn(tmp_path):
    state = make_state(tmp_path, [make_doc(11)])
    run(state, "/handoff  finish the parser tests ")
    (request,) = state.session_launcher.requests
    assert request.first_user_message == "finish the parser tests"


def test_reports_entries_newer_than_the_handoff(tmp_path):
    state = make_state(tmp_path, [make_doc(5)], entries=12)
    result = run(state)
    (request,) = state.session_launcher.requests
    assert "6 more transcript entries" in request.startup_system_message
    assert "last 6 transcript entries" in notice_text(result)


def test_current_handoff_has_no_staleness_note(tmp_path):
    state = make_state(tmp_path, [make_doc(11)], entries=12)
    result = run(state)
    assert "more transcript entries" not in state.session_launcher.requests[0].startup_system_message
    assert "/compact" not in notice_text(result)


def test_without_handoff_reports_error_and_launches_nothing(tmp_path):
    state = make_state(tmp_path)
    result = run(state)
    assert result.outputs[0].level == "error" and "/compact" in result.outputs[0].text
    assert not state.session_launcher.requests and not state.session_io.values


def test_skips_handoffs_beyond_a_rewound_transcript(tmp_path):
    state = make_state(tmp_path, [make_doc(5, "kept"), make_doc(30, "rewound away")], entries=12)
    run(state)
    assert "kept" in state.session_launcher.requests[0].startup_system_message


def test_unavailable_without_live_backend(tmp_path):
    state = make_state(tmp_path, [make_doc(5)])
    state.session_launcher = None
    with pytest.raises(CommandError, match="live agent backend"):
        run(state)


def test_model_follows_live_config(tmp_path):
    config = {"llm": {"default": "opus"}}
    state = make_state(tmp_path, [make_doc(5)])
    command = handoff.make_handoff_command(config)
    config["llm"]["default"] = "sonnet"  # /model mutates the runtime config in place
    registry = CommandRegistry([command])
    registry.execute("/handoff", CommandContext(runtime=None, session=SimpleNamespace(state=state)))
    assert state.session_launcher.requests[0].settings.model == "sonnet"


def test_registers_one_feature():
    added = []
    handoff.register_features(SimpleNamespace(add=added.append), session=None, config={})
    (feature,) = added
    assert feature.name == "handoff_command"
    assert [c.path for c in feature.components] == ["/handoff"]
