"""Launch a genuine isolated child runtime via AgentZooSessionDriver; no model call.

Opt in with AZO_HANDOFF_INTEGRATION=1 because it starts a backend process.
"""

from __future__ import annotations

import asyncio
import os
import queue

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("AZO_HANDOFF_INTEGRATION") != "1",
    reason="set AZO_HANDOFF_INTEGRATION=1 to launch a real child runtime",
)


def test_child_runtime_starts_with_handoff_in_transcript(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("AGENT_ZOO_HOME", str(home))
    from agent_utils.core.events import IoCompleted
    from agent_utils.core.io import SubmitOp
    from agent_utils.core.pipeline import PipelineBuilder
    from agent_utils.core.session import Session
    from agent_utils.features.core import core_feature
    from agent_zoo.cli.wsctl import AzoWs
    from agent_zoo.config.loader import load_layered_config
    from agent_zoo.drivers.sessions import AgentZooSessionDriver
    from agent_zoo.durability import session_context
    from agent_zoo.events import PublishView, SlashCommand, SpawnSession
    from agent_zoo.features.commands import commands_feature
    from agent_zoo.host.launch import RuntimeLauncher

    import handoff
    from test_handoff import make_doc, make_state

    config = load_layered_config(state_root=home)
    if not (config.get("llm") or {}).get("default"):
        pytest.skip("bundled config has no default model")

    handles = []

    class RecordingLauncher(RuntimeLauncher):
        def start(self, plan, **kwargs):
            handles.append(super().start(plan, **kwargs))
            return handles[-1]

    workdir = tmp_path / "work"
    workdir.mkdir()
    state = make_state(
        tmp_path, [make_doc(5, "MEMORY-MARKER: parser refactor is half done.")]
    )
    state.cwd = str(workdir)
    sctx = session_context(state)
    sctx.state_root = str(home)
    sctx.config_path = ""
    sctx.config_sets = ()

    session = Session(state=state)
    builder = (
        PipelineBuilder()
        .add(core_feature())
        .add(commands_feature(config=config))
    )
    handoff.register_features(builder, config=config)
    session.pipeline = builder.build(suppress_warnings=True)

    inbox: queue.Queue = queue.Queue()
    driver = AgentZooSessionDriver(
        launcher=RecordingLauncher(),
        store_root=sctx.store_root,
        local_state_root=sctx.local_state_root,
    )
    driver.start(inbox.put)
    try:
        effects = session.step(SlashCommand(raw="/handoff"))
        (spawn_op,) = [
            eff
            for eff in effects
            if isinstance(eff, SubmitOp) and isinstance(eff.request, SpawnSession)
        ]
        driver.submit(spawn_op.op_id, spawn_op.request)
        completion = inbox.get(timeout=30)
        assert isinstance(completion, IoCompleted), f"unexpected event: {completion!r}"
        follow = session.step(completion)
        attach_views = [
            eff.request
            for eff in follow
            if isinstance(eff, SubmitOp) and isinstance(eff.request, PublishView)
        ]
        assert len(attach_views) == 1 and attach_views[0].attach is not None

        (handle,) = handles
        try:
            ready = handle.wait_ready(timeout_s=90)
            assert ready["session_id"] == attach_views[0].attach["session_id"]

            async def transcript():
                client = AzoWs.attach(
                    ready["websocket_url"],
                    modern=False,
                    terminate_on_close=False,
                    connect_timeout_s=10,
                )
                await client.connect()
                try:
                    await client.subscribe_core()
                    return await client.wait_until(
                        lambda obs: "MEMORY-MARKER" in obs.transcript_text()
                        and obs.transcript_text(),
                        timeout=30,
                    )
                finally:
                    await client.close()

            text = asyncio.run(transcript())
            assert (
                "[Handoff document]" in text
                and "Please stand by" in text
                and "parent-session" in text
            )
        finally:
            handle.process.terminate()
            handle.process.wait(timeout=30)
    finally:
        driver.close(leave_children=True)
