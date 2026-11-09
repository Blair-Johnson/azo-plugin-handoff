"""Launch a genuine isolated child runtime; deliberately make no model call.

Opt in with AZO_HANDOFF_INTEGRATION=1 because it starts a backend process.
"""
import asyncio
import os
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("AZO_HANDOFF_INTEGRATION") != "1",
    reason="set AZO_HANDOFF_INTEGRATION=1 to launch a real child runtime",
)


def test_child_runtime_starts_with_handoff_in_transcript(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("AGENT_ZOO_HOME", str(home))
    from agent_zoo.config_loader import load_layered_config
    from agent_zoo.runtime.commands import CommandContext, CommandRegistry
    from agent_zoo.runtime_launch import RuntimeLauncher
    from agent_zoo.wsctl import AzoWs

    import handoff
    from test_handoff import FakeIO, make_doc, make_state

    config = load_layered_config(state_root=home)
    config["_runtime_launch"] = {"config_path": "", "config_sets": ()}
    if not (config.get("llm") or {}).get("default"):
        pytest.skip("bundled config has no default model")

    handles = []

    class RecordingLauncher(RuntimeLauncher):
        def start(self, plan, **kwargs):
            handles.append(super().start(plan, **kwargs))
            return handles[-1]

    workdir = tmp_path / "work"
    workdir.mkdir()
    state = make_state(tmp_path, [make_doc(5, "MEMORY-MARKER: parser refactor is half done.")])
    state.session_cwd = str(workdir)
    state.session_launcher = RecordingLauncher()
    state.session_io = FakeIO()

    registry = CommandRegistry([handoff.make_handoff_command(config)])
    registry.execute("/handoff", CommandContext(runtime=None, session=SimpleNamespace(state=state)))
    (handle,) = handles
    try:
        ready = handle.wait_ready(timeout_s=90)
        assert ready["session_id"] == state.session_io.values["core.attach"]["session_id"]

        async def transcript():
            client = AzoWs.attach(ready["websocket_url"], modern=False,
                                  terminate_on_close=False, connect_timeout_s=10)
            await client.connect()
            try:
                await client.subscribe_core()
                return await client.wait_until(
                    lambda obs: "MEMORY-MARKER" in obs.transcript_text() and obs.transcript_text(),
                    timeout=30,
                )
            finally:
                await client.close()

        text = asyncio.run(transcript())
        assert "[Handoff document]" in text and "Please stand by" in text and "parent-session" in text
    finally:
        handle.process.terminate()
        handle.process.wait(timeout=30)
