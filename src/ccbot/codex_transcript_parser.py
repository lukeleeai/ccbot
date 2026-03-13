"""Transcript parsing for Codex session JSONL files."""

from __future__ import annotations

from typing import Any

from .transcript_parser import ParsedEntry, ParsedMessage, TranscriptParser


class CodexTranscriptParser:
    """Parser for Codex CLI session transcripts."""

    EXPANDABLE_QUOTE_START = TranscriptParser.EXPANDABLE_QUOTE_START
    EXPANDABLE_QUOTE_END = TranscriptParser.EXPANDABLE_QUOTE_END

    @staticmethod
    def parse_line(line: str) -> dict | None:
        return TranscriptParser.parse_line(line)

    @staticmethod
    def get_message_type(data: dict) -> str | None:
        if data.get("type") == "event_msg":
            payload = data.get("payload", {})
            if isinstance(payload, dict):
                return str(payload.get("type"))
        return data.get("type")

    @classmethod
    def is_user_message(cls, data: dict) -> bool:
        return cls.get_message_type(data) == "user_message"

    @staticmethod
    def get_timestamp(data: dict) -> str | None:
        return data.get("timestamp")

    @classmethod
    def parse_message(cls, data: dict) -> ParsedMessage | None:
        msg_type = cls.get_message_type(data)
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            return None

        if msg_type == "user_message":
            return ParsedMessage(message_type="user", text=str(payload.get("message", "")))
        if msg_type == "agent_message":
            return ParsedMessage(
                message_type="assistant", text=str(payload.get("message", ""))
            )
        if msg_type == "agent_reasoning":
            return ParsedMessage(
                message_type="thinking", text=str(payload.get("text", ""))
            )
        return None

    @classmethod
    def parse_entries(
        cls,
        entries: list[dict],
        pending_tools: dict[str, Any] | None = None,
    ) -> tuple[list[ParsedEntry], dict[str, Any]]:
        result: list[ParsedEntry] = []

        for data in entries:
            parsed = cls.parse_message(data)
            if not parsed or not parsed.text.strip():
                continue
            timestamp = cls.get_timestamp(data)
            if parsed.message_type == "user":
                result.append(
                    ParsedEntry(
                        role="user",
                        text=parsed.text.strip(),
                        content_type="text",
                        timestamp=timestamp,
                    )
                )
            elif parsed.message_type == "assistant":
                result.append(
                    ParsedEntry(
                        role="assistant",
                        text=parsed.text.strip(),
                        content_type="text",
                        timestamp=timestamp,
                    )
                )
            elif parsed.message_type == "thinking":
                result.append(
                    ParsedEntry(
                        role="assistant",
                        text=TranscriptParser._format_expandable_quote(
                            parsed.text.strip()
                        ),
                        content_type="thinking",
                        timestamp=timestamp,
                    )
                )

        return result, pending_tools or {}

