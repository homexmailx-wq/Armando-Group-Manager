"""Mention every member of a group (the «تگ همه» command).

Telegram never exposes the full member list to a bot, so the bot mentions the
members it has already seen in that chat: everybody who joined or wrote a
message while the bot was present (``chat_members`` joined with ``users``).
"""

from __future__ import annotations

import logging
import time
from html import escape
from typing import Sequence

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.normalization import to_persian_digits
from ..db.models import ChatMemberState, User

logger = logging.getLogger("armando.mentionall")

# Telegram refuses anything longer than this in one message.
TELEGRAM_MESSAGE_LIMIT = 4096
# Room kept for the header line of every chunk.
HEADER_RESERVE = 220
# Hard safety cap: a bot should never try to ping a whole city at once.
MAX_MEMBERS = 500
# Members who left or were removed are never mentioned.
EXCLUDED_STATUSES = ("left", "kicked")

HEADER = "📣 <b>فراخوانی اعضا</b>"

# Simple per-chat cooldown (process local, resets on restart).
COOLDOWN_SECONDS = 60
_last_call: dict[int, float] = {}


def mention_link(user_id: int, name: str) -> str:
    """An inline ``tg://user`` link - Telegram treats it as a real mention."""
    safe = escape(" ".join((name or "").split())[:64] or "عضو")
    return f'<a href="tg://user?id={int(user_id)}">{safe}</a>'


def display_name(first_name: str | None, last_name: str | None,
                 username: str | None) -> str:
    parts = [p for p in ((first_name or "").strip(), (last_name or "").strip()) if p]
    if parts:
        return " ".join(parts)
    if username:
        return f"@{username}"
    return "عضو"


async def collect(session: AsyncSession, chat_id: int, *,
                  caller: tuple[int, str] | None = None,
                  limit: int = MAX_MEMBERS) -> list[tuple[int, str]]:
    """Known members of ``chat_id`` as ``(user_id, display_name)`` pairs.

    ``caller`` is the admin who ran the command: the bot must not forget to
    mention them just because they never typed anything in the group.
    """
    query = (
        select(ChatMemberState.user_id, User.first_name, User.last_name, User.username)
        .join(User, User.id == ChatMemberState.user_id)
        .where(and_(
            ChatMemberState.chat_id == chat_id,
            ChatMemberState.status.notin_(EXCLUDED_STATUSES),
            ChatMemberState.left_at.is_(None),
            User.is_bot.is_(False),
        ))
        .order_by(ChatMemberState.last_message_at.desc().nulls_last(),
                  ChatMemberState.user_id)
        .limit(limit)
    )
    result = await session.execute(query)
    members = [(int(row[0]), display_name(row[1], row[2], row[3])) for row in result.all()]
    if caller is not None:
        user_id, name = int(caller[0]), caller[1]
        if user_id > 0 and not any(uid == user_id for uid, _ in members):
            members.insert(0, (user_id, display_name(name, None, None)))
    return members


def build_messages(members: Sequence[tuple[int, str]], *,
                   limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split the mentions into messages Telegram will actually accept."""
    budget = max(200, limit - HEADER_RESERVE)
    chunks: list[list[str]] = [[]]
    for user_id, name in members:
        link = mention_link(user_id, name)
        current = chunks[-1]
        used = sum(len(item) + 1 for item in current)
        if current and used + len(link) > budget:
            chunks.append([link])
        else:
            current.append(link)

    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        return []

    total = len(chunks)
    messages: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        head = HEADER
        if total > 1:
            head = f"{HEADER} ({to_persian_digits(f'{index}/{total}')})"
        messages.append(f"{head}\n\n" + " ".join(chunk))
    return messages


def on_cooldown(chat_id: int, *, now: float | None = None) -> int:
    """Seconds left before the command may be used again (0 = ready)."""
    stamp = _last_call.get(chat_id)
    if stamp is None:
        return 0
    remaining = int(COOLDOWN_SECONDS - ((now if now is not None else time.monotonic()) - stamp))
    return remaining if remaining > 0 else 0


def mark_called(chat_id: int, *, now: float | None = None) -> None:
    _last_call[chat_id] = now if now is not None else time.monotonic()
