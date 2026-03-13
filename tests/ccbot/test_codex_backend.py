"""Tests for Codex backend session discovery and transcript parsing."""

import json
from pathlib import Path

import pytest

from ccbot.agent_backend import CodexBackend
from ccbot.codex_transcript_parser import CodexTranscriptParser


def _write_codex_session(file_path: Path, cwd: str, session_id: str) -> None:
    entries = [
        {
            "timestamp": "2026-03-13T10:00:00Z",
            "type": "session_meta",
            "payload": {"id": session_id, "cwd": cwd},
        },
        {
            "timestamp": "2026-03-13T10:00:01Z",
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "hello from user"},
        },
        {
            "timestamp": "2026-03-13T10:00:02Z",
            "type": "event_msg",
            "payload": {"type": "agent_reasoning", "text": "thinking"},
        },
        {
            "timestamp": "2026-03-13T10:00:03Z",
            "type": "event_msg",
            "payload": {"type": "agent_message", "message": "final answer"},
        },
    ]
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


class TestCodexTranscriptParser:
    def test_parse_entries(self) -> None:
        entries = [
            {
                "timestamp": "2026-03-13T10:00:01Z",
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "hello"},
            },
            {
                "timestamp": "2026-03-13T10:00:02Z",
                "type": "event_msg",
                "payload": {"type": "agent_reasoning", "text": "thinking"},
            },
            {
                "timestamp": "2026-03-13T10:00:03Z",
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": "done"},
            },
        ]
        parsed, pending = CodexTranscriptParser.parse_entries(entries)
        assert pending == {}
        assert [entry.role for entry in parsed] == ["user", "assistant", "assistant"]
        assert [entry.content_type for entry in parsed] == ["text", "thinking", "text"]
        assert parsed[2].text == "done"


class TestCodexBackend:
    @pytest.mark.asyncio
    async def test_list_sessions_for_directory(self, monkeypatch, tmp_path) -> None:
        root = tmp_path / "codex-home" / "sessions"
        target_cwd = "/tmp/work"
        _write_codex_session(
            root / "2026" / "03" / "13" / "rollout-test-session-a.jsonl",
            target_cwd,
            "test-session-a",
        )
        _write_codex_session(
            root / "2026" / "03" / "13" / "rollout-test-session-b.jsonl",
            "/tmp/other",
            "test-session-b",
        )

        from ccbot import agent_backend as backend_module

        monkeypatch.setattr(backend_module.config, "agent_sessions_path", root)
        backend = CodexBackend()
        sessions = await backend.list_sessions_for_directory(target_cwd)
        assert len(sessions) == 1
        assert sessions[0].session_id == "test-session-a"
        assert sessions[0].cwd == target_cwd
        assert sessions[0].summary == "final answer"

    @pytest.mark.asyncio
    async def test_discover_new_session_without_messages(
        self, monkeypatch, tmp_path
    ) -> None:
        root = tmp_path / "codex-home" / "sessions"
        target_cwd = "/tmp/work"
        file_path = root / "2026" / "03" / "13" / "rollout-test-session-a.jsonl"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(
                {
                    "timestamp": "2026-03-13T10:00:00Z",
                    "type": "session_meta",
                    "payload": {"id": "test-session-a", "cwd": target_cwd},
                }
            )
            + "\n"
        )

        from ccbot import agent_backend as backend_module

        monkeypatch.setattr(backend_module.config, "agent_sessions_path", root)
        backend = CodexBackend()
        session = await backend.discover_new_session(target_cwd, after_timestamp=0.0)
        assert session is not None
        assert session.session_id == "test-session-a"

    @pytest.mark.asyncio
    async def test_discover_new_session_falls_back_to_latest_matching_cwd(
        self, monkeypatch, tmp_path
    ) -> None:
        root = tmp_path / "codex-home" / "sessions"
        target_cwd = "/tmp/work"
        file_path = root / "2026" / "03" / "13" / "rollout-test-session-a.jsonl"
        _write_codex_session(file_path, target_cwd, "test-session-a")
        stale_mtime = 1000
        file_path.touch()
        file_path.chmod(0o644)
        import os

        os.utime(file_path, (stale_mtime, stale_mtime))

        from ccbot import agent_backend as backend_module

        monkeypatch.setattr(backend_module.config, "agent_sessions_path", root)
        backend = CodexBackend()
        session = await backend.discover_new_session(
            target_cwd, after_timestamp=stale_mtime + 10
        )
        assert session is not None
        assert session.session_id == "test-session-a"
        assert session.summary == "final answer"
