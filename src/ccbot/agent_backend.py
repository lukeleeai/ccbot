"""Agent backend selection and provider-specific session helpers."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles

from .codex_transcript_parser import CodexTranscriptParser
from .config import config
from .transcript_parser import ParsedEntry, PendingToolInfo, TranscriptParser

logger = logging.getLogger(__name__)


@dataclass
class AgentSession:
    """Provider-neutral session metadata used by the UI and monitor."""

    session_id: str
    summary: str
    message_count: int
    file_path: str
    cwd: str = ""


class AgentBackend(ABC):
    """Provider-specific logic for launch, discovery, and transcript parsing."""

    name: str
    display_name: str
    supports_hook: bool

    @abstractmethod
    def build_resume_command(self, command: str, session_id: str) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def parser(self) -> type[TranscriptParser] | type[CodexTranscriptParser]:
        raise NotImplementedError

    @abstractmethod
    async def list_sessions_for_directory(self, cwd: str) -> list[AgentSession]:
        raise NotImplementedError

    @abstractmethod
    async def get_session(self, session_id: str, cwd: str) -> AgentSession | None:
        raise NotImplementedError

    @abstractmethod
    async def discover_new_session(
        self, cwd: str, *, after_timestamp: float
    ) -> AgentSession | None:
        raise NotImplementedError

    @abstractmethod
    async def scan_active_sessions(self, active_cwds: set[str]) -> list[AgentSession]:
        raise NotImplementedError


class ClaudeBackend(AgentBackend):
    name = "claude"
    display_name = "Claude Code"
    supports_hook = True
    parser = TranscriptParser

    @staticmethod
    def _encode_cwd(cwd: str) -> str:
        import re

        return re.sub(r"[^a-zA-Z0-9-]", "-", cwd)

    def build_resume_command(self, command: str, session_id: str) -> str:
        return f"{command} --resume {session_id}"

    def _build_session_file_path(self, session_id: str, cwd: str) -> Path | None:
        if not session_id or not cwd:
            return None
        encoded_cwd = self._encode_cwd(cwd)
        return config.agent_sessions_path / encoded_cwd / f"{session_id}.jsonl"

    async def get_session(self, session_id: str, cwd: str) -> AgentSession | None:
        file_path = self._build_session_file_path(session_id, cwd)

        if not file_path or not file_path.exists():
            pattern = f"*/{session_id}.jsonl"
            matches = list(config.agent_sessions_path.glob(pattern))
            if matches:
                file_path = matches[0]
            else:
                return None

        summary = ""
        last_user_msg = ""
        message_count = 0
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    message_count += 1
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "summary":
                        s = data.get("summary", "")
                        if s:
                            summary = s
                    elif self.parser.is_user_message(data):
                        parsed = self.parser.parse_message(data)
                        if parsed and parsed.text.strip():
                            last_user_msg = parsed.text.strip()
        except OSError:
            return None

        if not summary:
            summary = last_user_msg[:50] if last_user_msg else "Untitled"

        return AgentSession(
            session_id=session_id,
            summary=summary,
            message_count=message_count,
            file_path=str(file_path),
            cwd=cwd,
        )

    async def list_sessions_for_directory(self, cwd: str) -> list[AgentSession]:
        encoded_cwd = self._encode_cwd(cwd)
        project_dir = config.agent_sessions_path / encoded_cwd
        if not project_dir.is_dir():
            return []

        jsonl_files = sorted(
            project_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        sessions: list[AgentSession] = []
        for f in jsonl_files:
            if f.stem == "sessions-index":
                continue
            if len(sessions) >= 10:
                break
            session = await self.get_session(f.stem, cwd)
            if session and session.message_count > 0:
                sessions.append(session)
        return sessions

    async def discover_new_session(
        self, cwd: str, *, after_timestamp: float
    ) -> AgentSession | None:
        fallback: AgentSession | None = None
        files = sorted(
            config.agent_sessions_path.glob("**/*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for file_path in files:
            meta = await self._read_session_meta(file_path)
            if not meta:
                continue
            session_id, file_cwd, _ = meta
            if file_cwd != cwd:
                continue
            session = await self.get_session(session_id, file_cwd)
            candidate = session or AgentSession(
                session_id=session_id,
                summary="Untitled",
                message_count=0,
                file_path=str(file_path),
                cwd=file_cwd,
            )
            if fallback is None:
                fallback = candidate
            try:
                if file_path.stat().st_mtime >= after_timestamp:
                    return candidate
            except OSError:
                continue
        return fallback

    async def scan_active_sessions(self, active_cwds: set[str]) -> list[AgentSession]:
        sessions: list[AgentSession] = []
        if not config.agent_sessions_path.exists():
            return sessions

        for project_dir in config.agent_sessions_path.iterdir():
            if not project_dir.is_dir():
                continue

            index_file = project_dir / "sessions-index.json"
            original_path = ""
            indexed_ids: set[str] = set()

            if index_file.exists():
                try:
                    async with aiofiles.open(index_file, "r") as f:
                        content = await f.read()
                    index_data = json.loads(content)
                    entries = index_data.get("entries", [])
                    original_path = index_data.get("originalPath", "")

                    for entry in entries:
                        session_id = entry.get("sessionId", "")
                        full_path = entry.get("fullPath", "")
                        project_path = entry.get("projectPath", original_path)
                        if not session_id or not full_path:
                            continue
                        if project_path not in active_cwds:
                            continue
                        indexed_ids.add(session_id)
                        file_path = Path(full_path)
                        if file_path.exists():
                            session = await self.get_session(session_id, project_path)
                            if session:
                                sessions.append(session)
                except (json.JSONDecodeError, OSError) as e:
                    logger.debug("Error reading Claude index %s: %s", index_file, e)

            try:
                for jsonl_file in project_dir.glob("*.jsonl"):
                    session_id = jsonl_file.stem
                    if session_id in indexed_ids:
                        continue
                    session = await self.get_session(session_id, "")
                    if session and session.cwd in active_cwds:
                        sessions.append(session)
            except OSError as e:
                logger.debug("Error scanning Claude dir %s: %s", project_dir, e)

        return sessions


class CodexBackend(AgentBackend):
    name = "codex"
    display_name = "Codex"
    supports_hook = False
    parser = CodexTranscriptParser

    def build_resume_command(self, command: str, session_id: str) -> str:
        return f"{command} resume {session_id}"

    @staticmethod
    async def _read_session_meta(file_path: Path) -> tuple[str, str, str] | None:
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                line = await f.readline()
            if not line:
                return None
            data = json.loads(line)
        except (OSError, json.JSONDecodeError):
            return None

        if data.get("type") != "session_meta":
            return None
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            return None
        return (
            str(payload.get("id", "")),
            str(payload.get("cwd", "")),
            str(payload.get("timestamp", "")),
        )

    async def get_session(self, session_id: str, cwd: str) -> AgentSession | None:
        matches = sorted(
            config.agent_sessions_path.glob(f"**/*{session_id}.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            return None
        file_path = matches[0]
        resolved_cwd = cwd
        summary = ""
        last_user_msg = ""
        message_count = 0

        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if data.get("type") == "session_meta":
                        payload = data.get("payload", {})
                        if isinstance(payload, dict) and payload.get("cwd"):
                            resolved_cwd = str(payload["cwd"])
                        continue

                    parsed_entries, _ = self.parser.parse_entries([data])
                    for entry in parsed_entries:
                        if entry.role == "user":
                            last_user_msg = entry.text.strip()
                        elif (
                            entry.role == "assistant"
                            and entry.content_type == "text"
                            and not summary
                        ):
                            summary = entry.text.strip()
                        message_count += 1
        except OSError:
            return None

        if not summary:
            summary = last_user_msg[:50] if last_user_msg else "Untitled"

        return AgentSession(
            session_id=session_id,
            summary=summary[:80],
            message_count=message_count,
            file_path=str(file_path),
            cwd=resolved_cwd,
        )

    async def list_sessions_for_directory(self, cwd: str) -> list[AgentSession]:
        sessions: list[AgentSession] = []
        if not config.agent_sessions_path.exists():
            return sessions

        files = sorted(
            config.agent_sessions_path.glob("**/*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for file_path in files:
            if len(sessions) >= 10:
                break
            meta = await self._read_session_meta(file_path)
            if not meta:
                continue
            session_id, file_cwd, _ = meta
            if file_cwd != cwd:
                continue
            session = await self.get_session(session_id, file_cwd)
            if session and session.message_count > 0:
                sessions.append(session)
        return sessions

    async def discover_new_session(
        self, cwd: str, *, after_timestamp: float
    ) -> AgentSession | None:
        fallback: AgentSession | None = None
        files = sorted(
            config.agent_sessions_path.glob("**/*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for file_path in files:
            meta = await self._read_session_meta(file_path)
            if not meta:
                continue
            session_id, file_cwd, _ = meta
            if file_cwd != cwd:
                continue
            session = await self.get_session(session_id, file_cwd)
            candidate = session or AgentSession(
                session_id=session_id,
                summary="Untitled",
                message_count=0,
                file_path=str(file_path),
                cwd=file_cwd,
            )
            if fallback is None:
                fallback = candidate
            try:
                if file_path.stat().st_mtime >= after_timestamp:
                    return candidate
            except OSError:
                continue
        return fallback

    async def scan_active_sessions(self, active_cwds: set[str]) -> list[AgentSession]:
        sessions: list[AgentSession] = []
        if not config.agent_sessions_path.exists():
            return sessions

        for file_path in sorted(
            config.agent_sessions_path.glob("**/*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            meta = await self._read_session_meta(file_path)
            if not meta:
                continue
            session_id, cwd, _ = meta
            if cwd not in active_cwds:
                continue
            session = await self.get_session(session_id, cwd)
            if session:
                sessions.append(session)
        return sessions


def get_backend() -> AgentBackend:
    if config.agent_backend == "codex":
        return CodexBackend()
    return ClaudeBackend()


backend = get_backend()
