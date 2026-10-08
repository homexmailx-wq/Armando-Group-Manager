"""«تگ همه» / «منشن همه» - mention every member of the group in one message."""

from __future__ import annotations

import logging

from ..services import mention_all as mention_service
from .common import CommandContext, require
from .registry import command

logger = logging.getLogger("armando.handlers.mention_all")


@command("تگ همه", "منشن همه", "صدا زدن همه", "فراخوانی همه", "تگ اعضا", "منشن اعضا",
         "@all", role="admin", category="members", group_only=True,
         description="منشن کردن همهٔ اعضای گروه", usage="تگ همه")
async def cmd_mention_all(ctx: CommandContext) -> None:
    if not await require(ctx, role="admin"):
        return

    caller = ctx.message.from_user
    caller_pair = None
    if caller is not None and not caller.is_bot:
        caller_pair = (caller.id, mention_service.display_name(
            caller.first_name, caller.last_name, caller.username))

    members = await mention_service.collect(ctx.session, ctx.chat_id, bot=ctx.bot,
                                            caller=caller_pair)
    if not members:
        await ctx.reply(
            "📣 هنوز عضوی در این گروه ثبت نشده است.\n"
            "ربات فقط اعضایی را می‌شناسد که بعد از حضورش در گروه پیام فرستاده یا "
            "وارد شده باشند.")
        return

    messages = mention_service.build_messages(members)
    if not messages:
        return

    for text in messages:
        await ctx.reply(text, reply_to=False)

    if len(messages) > 1:
        logger.info("mention-all split into %s messages chat=%s", len(messages), ctx.chat_id)
