"""End-to-end dispatcher tests with a mocked Telegram bot.

These tests push real ``Update`` objects through the real dispatcher, so they
catch wiring mistakes that unit tests cannot (wrong handler signatures, missing
imports, broken permission flow).  Telegram is never contacted - the bot object
is a mock.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.types import (
    CallbackQuery,
    Chat,
    ChatMemberAdministrator,
    ChatMemberMember,
    ChatMemberOwner,
    Message,
    Update,
    User,
)

from app.keyboards.factory import cb

CHAT_ID = -1001234567890
OWNER_ID = 1000
MEMBER_ID = 2000
ADMIN_ID = 3000
BOT_ID = 999999
PROMOTED_ADMIN_ID = 4000  # an administrator the bot itself promoted (editable)


# --------------------------------------------------------------------------- #
# Fake Telegram
# --------------------------------------------------------------------------- #
def _msg(**kwargs) -> Message:
    payload: dict = dict(
        message_id=7,
        date=int(time.time()),
        chat=Chat(id=CHAT_ID, type="supergroup", title="گروه تست"),
        from_user=User(id=BOT_ID, is_bot=True, first_name="Armando", username="armando_bot"),
        text="ok",
    )
    payload.update(kwargs)
    return Message(**payload)


def _bot_member(**rights) -> ChatMemberAdministrator:
    """The bot's own administrator record - the rights decide what may work."""
    payload = dict(
        user=User(id=BOT_ID, is_bot=True, first_name="Armando"),
        status="administrator", can_be_edited=False, can_manage_chat=True,
        can_delete_messages=True, can_restrict_members=True, can_promote_members=False,
        can_change_info=True, can_invite_users=True, can_pin_messages=True,
        can_manage_video_chats=True, can_post_messages=True, can_edit_messages=True,
        is_anonymous=False, can_post_stories=True, can_edit_stories=True,
        can_delete_stories=True, can_send_welcome_messages=True)
    payload.update(rights)
    return ChatMemberAdministrator(**payload)


def build_fake_bot(**bot_rights) -> AsyncMock:
    """An ``AsyncMock`` bot: every Telegram call is awaitable and recorded."""
    bot = AsyncMock(spec=Bot)
    bot.id = BOT_ID
    bot_member = _bot_member(**bot_rights)

    async def get_chat_member(chat_id: int, user_id: int):
        user = User(id=user_id, is_bot=user_id == BOT_ID, first_name="User")
        if user_id == OWNER_ID:
            return ChatMemberOwner(user=user, is_anonymous=False, custom_title="مالک",
                                   status="creator")
        if user_id == ADMIN_ID:
            return ChatMemberAdministrator(
                user=user, status="administrator", can_be_edited=False,
                can_manage_chat=True, can_delete_messages=True, can_restrict_members=True,
                can_promote_members=True, can_change_info=True, can_invite_users=True,
                can_pin_messages=True, can_manage_video_chats=True, can_post_messages=True,
                can_edit_messages=True, is_anonymous=False, can_post_stories=True,
                can_edit_stories=True, can_delete_stories=True,
                can_send_welcome_messages=True, custom_title="مدیر")
        if user_id == PROMOTED_ADMIN_ID:
            return ChatMemberAdministrator(
                user=user, status="administrator", can_be_edited=True,
                can_manage_chat=True, can_delete_messages=True, can_restrict_members=True,
                can_promote_members=False, can_change_info=False, can_invite_users=True,
                can_pin_messages=False, can_manage_video_chats=False, can_post_messages=True,
                can_edit_messages=True, is_anonymous=False, can_post_stories=True,
                can_edit_stories=True, can_delete_stories=True,
                can_send_welcome_messages=True)
        if user_id == BOT_ID:
            return bot_member
        return ChatMemberMember(user=user, status="member")

    bot.get_chat_member = AsyncMock(side_effect=get_chat_member)
    bot.get_me = AsyncMock(return_value=User(id=BOT_ID, is_bot=True,
                                             first_name="Armando", username="armando_bot"))
    bot.get_chat = AsyncMock(return_value=Chat(id=CHAT_ID, type="supergroup", title="گروه تست"))
    bot.get_chat_administrators = AsyncMock(return_value=[])
    bot.send_message = AsyncMock(return_value=_msg())
    bot.send_document = AsyncMock(return_value=_msg())
    bot.delete_message = AsyncMock(return_value=True)
    bot.restrict_chat_member = AsyncMock(return_value=True)
    bot.ban_chat_member = AsyncMock(return_value=True)
    bot.unban_chat_member = AsyncMock(return_value=True)
    bot.pin_chat_message = AsyncMock(return_value=True)
    bot.unpin_chat_message = AsyncMock(return_value=True)
    bot.unpin_all_chat_messages = AsyncMock(return_value=True)
    bot.answer_callback_query = AsyncMock(return_value=True)
    bot.promote_chat_member = AsyncMock(return_value=True)
    bot.set_chat_permissions = AsyncMock(return_value=True)
    bot.get_file = AsyncMock(return_value=None)
    bot.create_chat_invite_link = AsyncMock(
        return_value=SimpleNamespace(invite_link="https://t.me/joinchat/TESTLINK"))
    bot.export_chat_invite_link = AsyncMock(return_value="https://t.me/joinchat/TESTLINK")
    bot.set_chat_title = AsyncMock(return_value=True)
    bot.set_chat_description = AsyncMock(return_value=True)
    # aiogram routes `message.answer()` / `edit_text()` through Bot.__call__
    calls: list = []

    async def _dispatch(method, *args, **kwargs):  # records and "succeeds"
        calls.append(method)
        return True

    bot.side_effect = _dispatch
    bot.telegram_calls = calls
    bot.return_value = True
    return bot


_counter = {"id": 0}


def _next_update_id() -> int:
    _counter["id"] += 1
    return _counter["id"]


async def feed(dispatcher: Dispatcher, bot, message: Message) -> None:
    await dispatcher.feed_update(bot, Update(update_id=_next_update_id(), message=message))


def _raw_api_recorder(monkeypatch) -> list:
    """Record raw Bot API calls (used for tags/demote) instead of posting them."""
    calls: list = []

    async def _call(bot, method_name, payload, *, timeout=30):
        calls.append((method_name, dict(payload)))
        return True

    monkeypatch.setattr("app.core.telegram_extra.call_api_raw", _call)
    return calls


def _edit_count(bot) -> int:
    """How many EditMessageText calls went through the mocked Bot."""
    from aiogram.methods import EditMessageText

    return sum(1 for method in bot.telegram_calls if isinstance(method, EditMessageText))


def group_message(text: str, *, user_id: int = OWNER_ID, reply_to: Message | None = None,
                  **kwargs) -> Message:
    return Message(
        message_id=kwargs.pop("message_id", _next_update_id()),
        date=int(time.time()),
        chat=Chat(id=CHAT_ID, type="supergroup", title="گروه تست"),
        from_user=User(id=user_id, is_bot=False, first_name="Ali"),
        text=text,
        reply_to_message=reply_to,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_private_start_answers_in_persian(dispatcher, monkeypatch):
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    message = Message(
        message_id=1, date=int(time.time()),
        chat=Chat(id=OWNER_ID, type="private"),
        from_user=User(id=OWNER_ID, is_bot=False, first_name="Ali"),
        text="/start",
    )
    await feed(dispatcher, bot, message)
    assert bot.send_message.await_count >= 1
    text = bot.send_message.await_args.kwargs.get("text", "")
    assert "Armando" in text
    keyboard = bot.send_message.await_args.kwargs.get("reply_markup")
    buttons = [btn for row in keyboard.inline_keyboard for btn in row]
    texts = [btn.text for btn in buttons]
    assert any("سازنده بات" in item for item in texts), texts
    assert any("اضافه کردن به گروه" in item for item in texts), texts
    assert any((btn.url or "").lower() == "https://t.me/rvivl" for btn in buttons)
    assert not reported


@pytest.mark.asyncio
async def test_persian_ban_by_reply(dispatcher, monkeypatch):
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    target = group_message("سلام", user_id=MEMBER_ID)
    await feed(dispatcher, bot, group_message("بن ۲روز تبلیغ", reply_to=target))
    assert not reported, reported
    assert bot.ban_chat_member.await_count >= 1
    assert bot.ban_chat_member.await_args.kwargs.get("user_id") == MEMBER_ID


@pytest.mark.asyncio
async def test_profanity_message_is_deleted(dispatcher, monkeypatch):
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    await feed(dispatcher, bot, group_message("این پیام شامل کلمه کیر است",
                                              user_id=MEMBER_ID))
    assert not reported, reported
    assert bot.delete_message.await_count >= 1


@pytest.mark.asyncio
async def test_ordinary_message_is_untouched(dispatcher, monkeypatch):
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    await feed(dispatcher, bot, group_message("بنظرم این فیلم خوب بود", user_id=MEMBER_ID))
    assert not reported, reported
    assert bot.delete_message.await_count == 0
    assert bot.ban_chat_member.await_count == 0
    assert bot.restrict_chat_member.await_count == 0


@pytest.mark.asyncio
async def test_panel_command_and_callback(dispatcher, monkeypatch):
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    await feed(dispatcher, bot, group_message("پنل"))
    assert not reported, reported
    assert bot.send_message.await_count >= 1

    callback = CallbackQuery(
        id="1", from_user=User(id=OWNER_ID, is_bot=False, first_name="Ali"),
        chat_instance="1", data=cb("panel", "main"),
        message=_msg(text="پنل"),
    )
    await dispatcher.feed_update(bot, Update(update_id=_next_update_id(), callback_query=callback))
    assert not reported, reported
    assert _edit_count(bot) >= 1, "panel callback must refresh the message"


@pytest.mark.asyncio
async def test_lock_panel_callback(dispatcher, monkeypatch):
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    callback = CallbackQuery(
        id="2", from_user=User(id=OWNER_ID, is_bot=False, first_name="Ali"),
        chat_instance="1", data=cb("panel", "locks"),
        message=_msg(text="پنل"),
    )
    await dispatcher.feed_update(bot, Update(update_id=_next_update_id(), callback_query=callback))
    assert not reported, reported
    assert _edit_count(bot) >= 1, "panel callback must refresh the message"


@pytest.mark.asyncio
async def test_member_cannot_ban(dispatcher, monkeypatch):
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    target = group_message("سلام", user_id=MEMBER_ID)
    await feed(dispatcher, bot, group_message("بن", user_id=MEMBER_ID, reply_to=target))
    assert not reported, reported
    assert bot.ban_chat_member.await_count == 0


@pytest.mark.asyncio
async def test_member_gets_a_personal_panel_only(dispatcher, monkeypatch):
    """Group management is never offered to ordinary members."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    await feed(dispatcher, bot, group_message("پنل", user_id=MEMBER_ID))
    assert not reported, reported
    texts = [call.kwargs.get("text", "") for call in bot.send_message.await_args_list]
    assert any("فقط برای مدیران" in text for text in texts), texts


@pytest.mark.asyncio
async def test_admin_gets_the_management_panel(dispatcher, monkeypatch):
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    await feed(dispatcher, bot, group_message("پنل", user_id=ADMIN_ID))
    assert not reported, reported
    texts = [call.kwargs.get("text", "") for call in bot.send_message.await_args_list]
    assert not any("فقط برای مدیران" in text for text in texts)
    assert any("Armando" in text for text in texts)
    keyboard = bot.send_message.await_args.kwargs.get("reply_markup")
    buttons = [btn.text for row in keyboard.inline_keyboard for btn in row]
    assert not any("سازنده" in item for item in buttons), buttons


@pytest.mark.asyncio
async def test_welcome_is_sent_once(dispatcher, monkeypatch):
    """A single join must produce exactly one welcome message."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    join = group_message("", user_id=MEMBER_ID,
                         new_chat_members=[User(id=MEMBER_ID, is_bot=False,
                                                first_name="تازه‌وارد")])
    await feed(dispatcher, bot, join)
    assert not reported, reported
    texts = [call.kwargs.get("text", "") for call in bot.send_message.await_args_list]
    welcomes = [text for text in texts if "خوش" in text]
    assert len(welcomes) == 1, f"expected exactly one welcome, got {welcomes}"


@pytest.mark.asyncio
async def test_telegram_admin_can_moderate_without_internal_role(dispatcher, monkeypatch):
    """A real Telegram admin must be able to moderate, role assignment or not."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    target = group_message("سلام", user_id=MEMBER_ID)
    await feed(dispatcher, bot, group_message("بن ۲روز تبلیغ", user_id=ADMIN_ID,
                                              reply_to=target))
    assert not reported, reported
    assert bot.ban_chat_member.await_count >= 1


@pytest.mark.asyncio
async def test_anonymous_admin_keeps_full_rights(dispatcher, monkeypatch):
    """Anonymous group admins (Telegram hides their id) stay in charge."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    target = group_message("سلام", user_id=MEMBER_ID)
    anonymous = group_message("بن تبلیغ", user_id=1087968824, reply_to=target)
    await feed(dispatcher, bot, anonymous)
    assert not reported, reported
    assert bot.ban_chat_member.await_count >= 1


@pytest.mark.asyncio
async def test_admin_fallback_to_administrators_list(dispatcher, monkeypatch):
    """If getChatMember fails, the administrators list still proves the rank."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    bot.get_chat_administrators = AsyncMock(return_value=[
        ChatMemberAdministrator(
            user=User(id=ADMIN_ID, is_bot=False, first_name="مدیر"),
            status="administrator", can_be_edited=False, can_manage_chat=True,
            can_delete_messages=True, can_restrict_members=True, can_promote_members=False,
            can_change_info=True, can_invite_users=True, can_pin_messages=True,
            can_manage_video_chats=True, can_post_messages=True, can_edit_messages=True,
            is_anonymous=False, can_post_stories=True, can_edit_stories=True,
            can_delete_stories=True, can_send_welcome_messages=True)])

    original = bot.get_chat_member

    async def flaky(chat_id: int, user_id: int):
        if user_id == ADMIN_ID:  # simulate a failing getChatMember
            raise RuntimeError("member not found")
        return await original(chat_id=chat_id, user_id=user_id)

    bot.get_chat_member = AsyncMock(side_effect=flaky)

    target = group_message("سلام", user_id=MEMBER_ID)
    await feed(dispatcher, bot, group_message("بن تبلیغ", user_id=ADMIN_ID, reply_to=target))
    assert not reported, reported
    assert bot.ban_chat_member.await_count >= 1


@pytest.mark.asyncio
async def test_help_topics_answer(dispatcher, monkeypatch):
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    for phrase, expected in [("راهنما مدیریت", "مدیریت کاربران"),
                             ("راهنما قفل‌ها", "قفل‌ها"),
                             ("راهنما ارز", "ارز")]:
        await feed(dispatcher, bot, group_message(phrase, user_id=MEMBER_ID))
        texts = [call.kwargs.get("text", "") for call in bot.send_message.await_args_list]
        assert any(expected in text for text in texts), (phrase, texts)
    assert not reported, reported


@pytest.mark.asyncio
async def test_create_invite_link(dispatcher, monkeypatch):
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    await feed(dispatcher, bot, group_message("ساخت لینک", user_id=ADMIN_ID))
    assert not reported, reported
    assert bot.create_chat_invite_link.await_count >= 1
    texts = [call.kwargs.get("text", "") for call in bot.send_message.await_args_list]
    assert any("https://t.me/joinchat/TESTLINK" in text for text in texts)


# --------------------------------------------------------------------------- #
# Member tags and promote/demote (the two reported bugs)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_tag_is_mirrored_into_the_telegram_member_list(dispatcher, monkeypatch):
    """`تگ` must call setChatMemberTag so the tag shows up in Telegram itself."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))
    raw_calls = _raw_api_recorder(monkeypatch)

    bot = build_fake_bot(can_manage_tags=True)
    target = group_message("سلام", user_id=MEMBER_ID)
    await feed(dispatcher, bot, group_message("تگ مدیر فروش", user_id=ADMIN_ID,
                                              reply_to=target))
    assert not reported, reported
    assert ("setChatMemberTag", {"chat_id": CHAT_ID, "user_id": MEMBER_ID,
                                 "tag": "مدیر فروش"}) in [(name, payload)
                                                          for name, payload in raw_calls]
    texts = [call.kwargs.get("text", "") for call in bot.send_message.await_args_list]
    assert any("در لیست اعضای تلگرام" in text for text in texts), texts


@pytest.mark.asyncio
async def test_tag_without_the_manage_tags_right_names_it(dispatcher, monkeypatch):
    """No `can_manage_tags` -> the bot says exactly which right is missing."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))
    raw_calls = _raw_api_recorder(monkeypatch)

    bot = build_fake_bot(can_manage_tags=False)
    target = group_message("سلام", user_id=MEMBER_ID)
    await feed(dispatcher, bot, group_message("تگ مدیر فروش", user_id=ADMIN_ID,
                                              reply_to=target))
    assert not reported, reported
    assert raw_calls == []
    texts = [call.kwargs.get("text", "") for call in bot.send_message.await_args_list]
    assert any("مدیریت تگ اعضا" in text for text in texts), texts


@pytest.mark.asyncio
async def test_promote_reports_the_missing_bot_right(dispatcher, monkeypatch):
    """`ارتقا` explains itself instead of silently doing nothing."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))
    raw_calls = _raw_api_recorder(monkeypatch)

    bot = build_fake_bot(can_promote_members=False)
    target = group_message("سلام", user_id=MEMBER_ID)
    await feed(dispatcher, bot, group_message("ارتقا", user_id=ADMIN_ID, reply_to=target))
    assert not reported, reported
    assert raw_calls == []
    texts = [call.kwargs.get("text", "") for call in bot.send_message.await_args_list]
    assert any("افزودن مدیر جدید" in text for text in texts), texts


@pytest.mark.asyncio
async def test_promote_works_when_the_bot_may_promote(dispatcher, monkeypatch):
    """With `can_promote_members` the bot really promotes the member."""
    import app.handlers.errors as errors_module
    from aiogram.methods import PromoteChatMember

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot(can_promote_members=True)
    target = group_message("سلام", user_id=MEMBER_ID)
    await feed(dispatcher, bot, group_message("ارتقا", user_id=ADMIN_ID, reply_to=target))
    assert not reported, reported
    promoted = [m for m in bot.telegram_calls
                if type(m).__name__ == "PromoteChatMemberWithTitle"]
    assert promoted, [type(m).__name__ for m in bot.telegram_calls]
    assert promoted[0].user_id == MEMBER_ID
    texts = [call.kwargs.get("text", "") for call in bot.send_message.await_args_list]
    assert any("ارتقا یافت" in text for text in texts), texts


@pytest.mark.asyncio
async def test_demote_sends_every_right_as_false(dispatcher, monkeypatch):
    """`تنزل` must transmit the False values, not drop them from the request."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))
    raw_calls = _raw_api_recorder(monkeypatch)

    bot = build_fake_bot(can_promote_members=True)
    target = group_message("سلام", user_id=MEMBER_ID)
    await feed(dispatcher, bot, group_message("تنزل", user_id=ADMIN_ID, reply_to=target))
    assert not reported, reported
    assert raw_calls and raw_calls[0][0] == "promoteChatMember"
    payload = raw_calls[0][1]
    assert payload["user_id"] == MEMBER_ID
    assert payload["can_delete_messages"] is False
    assert payload["can_restrict_members"] is False
    texts = [call.kwargs.get("text", "") for call in bot.send_message.await_args_list]
    assert any("برکنار شد" in text for text in texts), texts


@pytest.mark.asyncio
async def test_bot_rights_command_lists_what_is_missing(dispatcher, monkeypatch):
    """`دسترسی ربات` is the diagnostic an admin needs after a failure."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot(can_manage_tags=False, can_promote_members=True)
    await feed(dispatcher, bot, group_message("دسترسی ربات", user_id=ADMIN_ID))
    assert not reported, reported
    texts = [call.kwargs.get("text", "") for call in bot.send_message.await_args_list]
    assert any("مدیریت تگ اعضا" in text for text in texts), texts


# --------------------------------------------------------------------------- #
# Editing the Telegram rights of an administrator
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_admin_rights_command_renders_the_toggles(dispatcher, monkeypatch):
    """`دسترسی مدیر` shows a live editor for an admin promoted by the bot."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot(can_promote_members=True)
    target = group_message("سلام", user_id=PROMOTED_ADMIN_ID)
    await feed(dispatcher, bot, group_message("دسترسی مدیر", user_id=OWNER_ID,
                                              reply_to=target))
    assert not reported, reported
    keyboard = bot.send_message.await_args.kwargs.get("reply_markup")
    assert keyboard is not None
    labels = [btn.text for row in keyboard.inline_keyboard for btn in row]
    assert any("سنجاق" in label for label in labels), labels
    assert any("عزل مدیر" in label for label in labels), labels


@pytest.mark.asyncio
async def test_admin_rights_command_on_a_plain_member(dispatcher, monkeypatch):
    """A non-admin has no rights to edit - say how to promote instead."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot(can_promote_members=True)
    target = group_message("سلام", user_id=MEMBER_ID)
    await feed(dispatcher, bot, group_message("دسترسی مدیر", user_id=OWNER_ID,
                                              reply_to=target))
    assert not reported, reported
    texts = [call.kwargs.get("text", "") for call in bot.send_message.await_args_list]
    assert any("مدیر گروه نیست" in text for text in texts), texts


@pytest.mark.asyncio
async def test_admin_rights_toggle_promotes_with_the_new_value(dispatcher, monkeypatch):
    """Clicking a right flips exactly that right through promoteChatMember."""
    raw_calls = _raw_api_recorder(monkeypatch)
    bot = build_fake_bot(can_promote_members=True)
    callback = CallbackQuery(
        id="11", from_user=User(id=OWNER_ID, is_bot=False, first_name="Ali"),
        chat_instance="1",
        data=cb("ap", "t", PROMOTED_ADMIN_ID, "can_pin_messages"),
        message=_msg(text="دسترسی مدیر"),
    )
    await dispatcher.feed_update(bot, Update(update_id=_next_update_id(),
                                             callback_query=callback))
    assert raw_calls and raw_calls[0][0] == "promoteChatMember"
    payload = raw_calls[0][1]
    assert payload["user_id"] == PROMOTED_ADMIN_ID
    assert payload["can_pin_messages"] is True       # was False -> flipped on
    assert payload["can_delete_messages"] is True    # unchanged
    assert payload["can_change_info"] is False       # unchanged
    assert _edit_count(bot) >= 1, "the editor must refresh after a toggle"


@pytest.mark.asyncio
async def test_admin_rights_demote_preset_clears_every_right(dispatcher, monkeypatch):
    """The «عزل مدیر» button revokes every administrator right."""
    raw_calls = _raw_api_recorder(monkeypatch)
    bot = build_fake_bot(can_promote_members=True)
    callback = CallbackQuery(
        id="12", from_user=User(id=OWNER_ID, is_bot=False, first_name="Ali"),
        chat_instance="1", data=cb("ap", "demote", PROMOTED_ADMIN_ID),
        message=_msg(text="دسترسی مدیر"),
    )
    await dispatcher.feed_update(bot, Update(update_id=_next_update_id(),
                                             callback_query=callback))
    assert raw_calls and raw_calls[0][0] == "promoteChatMember"
    payload = raw_calls[0][1]
    assert payload["user_id"] == PROMOTED_ADMIN_ID
    for flag in ("can_delete_messages", "can_restrict_members", "can_invite_users",
                 "can_pin_messages", "can_change_info", "can_promote_members"):
        assert payload[flag] is False, flag


@pytest.mark.asyncio
async def test_demote_alias_works_with_azl(dispatcher, monkeypatch):
    """`عزل` is accepted as a plain-Persian alias of `تنزل`."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))
    raw_calls = _raw_api_recorder(monkeypatch)

    bot = build_fake_bot(can_promote_members=True)
    target = group_message("سلام", user_id=PROMOTED_ADMIN_ID)
    await feed(dispatcher, bot, group_message("عزل", user_id=OWNER_ID, reply_to=target))
    assert not reported, reported
    assert raw_calls and raw_calls[0][0] == "promoteChatMember"
    texts = [call.kwargs.get("text", "") for call in bot.send_message.await_args_list]
    assert any("برکنار شد" in text for text in texts), texts


@pytest.mark.asyncio
async def test_azl_in_a_sentence_is_not_a_command(dispatcher, monkeypatch):
    """«عزل شد» is ordinary talk - the bot must not demote anybody for it."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))
    raw_calls = _raw_api_recorder(monkeypatch)

    bot = build_fake_bot(can_promote_members=True)
    target = group_message("سلام", user_id=PROMOTED_ADMIN_ID)
    await feed(dispatcher, bot, group_message("عزل شد", user_id=OWNER_ID, reply_to=target))
    assert not reported, reported
    assert raw_calls == []


# --------------------------------------------------------------------------- #
# Undo buttons and group-wide lock
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_mute_reply_carries_an_undo_button(dispatcher, monkeypatch):
    """After «سکوت» the reply itself offers «لغو سکوت»."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    target = group_message("سلام", user_id=MEMBER_ID)
    await feed(dispatcher, bot, group_message("سکوت", user_id=OWNER_ID, reply_to=target))
    assert not reported, reported
    keyboard = bot.send_message.await_args.kwargs.get("reply_markup")
    assert keyboard is not None, "the mute reply must carry the undo button"
    callbacks = [btn.callback_data for row in keyboard.inline_keyboard for btn in row]
    assert f"undo:unmute:{MEMBER_ID}" in callbacks, callbacks


@pytest.mark.asyncio
async def test_undo_button_unmutes_the_member(dispatcher, monkeypatch):
    """Clicking «لغو سکوت» really lifts the restriction."""
    bot = build_fake_bot()
    callback = CallbackQuery(
        id="21", from_user=User(id=OWNER_ID, is_bot=False, first_name="Ali"),
        chat_instance="1", data=cb("undo", "unmute", MEMBER_ID),
        message=_msg(text="سکوت"),
    )
    await dispatcher.feed_update(bot, Update(update_id=_next_update_id(),
                                             callback_query=callback))
    assert bot.restrict_chat_member.await_count >= 1
    permissions = bot.restrict_chat_member.await_args.kwargs.get("permissions")
    assert permissions is not None and permissions.can_send_messages is True


@pytest.mark.asyncio
async def test_lock_group_command_sets_chat_permissions(dispatcher, monkeypatch):
    """«قفل گروه» locks the whole chat for ordinary members."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    await feed(dispatcher, bot, group_message("قفل گروه", user_id=OWNER_ID))
    assert not reported, reported
    assert bot.set_chat_permissions.await_count == 1
    permissions = bot.set_chat_permissions.await_args.kwargs.get("permissions")
    assert permissions.can_send_messages is False
    assert permissions.can_send_photos is False
    assert permissions.can_send_videos is False


@pytest.mark.asyncio
async def test_unlock_group_command_restores_permissions(dispatcher, monkeypatch):
    """«باز کردن گروه» gives the members their voice back."""
    import app.handlers.errors as errors_module

    reported = []
    monkeypatch.setattr(errors_module, "notify_error_chat",
                        AsyncMock(side_effect=lambda *a, **k: reported.append(a)))

    bot = build_fake_bot()
    await feed(dispatcher, bot, group_message("باز کردن گروه", user_id=OWNER_ID))
    assert not reported, reported
    assert bot.set_chat_permissions.await_count == 1
    permissions = bot.set_chat_permissions.await_args.kwargs.get("permissions")
    assert permissions.can_send_messages is True
    assert permissions.can_send_photos is True


@pytest.mark.asyncio
async def test_group_lock_button_in_the_panel(dispatcher, monkeypatch):
    """The locks panel can lock and unlock the group with one tap."""
    bot = build_fake_bot()
    callback = CallbackQuery(
        id="22", from_user=User(id=OWNER_ID, is_bot=False, first_name="Ali"),
        chat_instance="1", data=cb("cl", "media"),
        message=_msg(text="قفل‌ها"),
    )
    await dispatcher.feed_update(bot, Update(update_id=_next_update_id(),
                                             callback_query=callback))
    assert bot.set_chat_permissions.await_count == 1
    permissions = bot.set_chat_permissions.await_args.kwargs.get("permissions")
    assert permissions.can_send_messages is True    # text stays allowed
    assert permissions.can_send_photos is False     # media is blocked
