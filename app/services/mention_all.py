"""Mention every member of a group (the «تگ همه» command).

Telegram never exposes the full member list to a bot, so the bot mentions the
members it has already seen in that chat: everybody who joined or wrote a
message while the bot was present (``chat_members`` joined with ``users``).
"""

from __future__ import annotations

import logging
from html import escape
from typing import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from ..core.normalization import to_persian_digits
from . import roster

logger = logging.getLogger("armando.mentionall")

# Telegram refuses anything longer than this in one message.
TELEGRAM_MESSAGE_LIMIT = 4096
# Room kept for the header line of every chunk.
HEADER_RESERVE = 220
# Hard safety cap: a bot should never try to ping a whole city at once.
MAX_MEMBERS = 1000

HEADER = "📣 <b>فراخوانی اعضا</b>"


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
                  bot=None, limit: int = MAX_MEMBERS) -> list[tuple[int, str]]:
    """Everybody the bot knows in ``chat_id`` as ``(user_id, display_name)``.

    ``caller`` is the admin who ran the command: the bot must not forget to
    mention them just because they never typed anything in the group.
    ``bot`` lets the roster also read the real administrator list.
    """
    members = await roster.known_members(session, chat_id, bot=bot, limit=limit)
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
