"""Download public Instagram post and reel media without account cookies."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from aiogram import Bot
from aiogram.types import (FSInputFile, InputMediaDocument, InputMediaPhoto,
                           InputMediaVideo, Message)

logger = logging.getLogger("armando.instagram")

_PUBLIC_POST_RE = re.compile(
    r"^https?://(?:www\.|m\.)?instagram\.com/(?:p|reel|reels|tv)/"
    r"[A-Za-z0-9_-]+/?(?:\?[^\s]*)?$",
    re.IGNORECASE,
)
_MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".m4v", ".mov"}
_MAX_FILES = 10
_MAX_TOTAL_BYTES = 48 * 1024 * 1024
_DOWNLOAD_TIMEOUT = 55
_DOWNLOAD_COOLDOWN = 10.0
_last_download: dict[int, float] = {}


def extract_public_instagram_url(text: str) -> str | None:
    """Accept only a standalone direct public post/reel URL."""
    candidate = (text or "").strip().strip("<> \t\r\n.,،؛!?؟")
    if _PUBLIC_POST_RE.fullmatch(candidate):
        return candidate
    return None


def _media_file_list(root: Path) -> list[Path]:
    files = [path for path in root.rglob("*")
             if path.is_file() and path.suffix.lower() in _MEDIA_EXTENSIONS]
    return sorted(files, key=lambda path: path.as_posix())[:_MAX_FILES]


def _input_media(path: Path, caption: str = ""):
    media = FSInputFile(path)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg", ".png"}:
        return InputMediaPhoto(media=media, caption=caption or None, parse_mode=None)
    if ext == ".mp4":
        return InputMediaVideo(media=media, caption=caption or None, parse_mode=None)
    return InputMediaDocument(media=media, caption=caption or None, parse_mode=None)


async def _send_notice(bot: Bot, message: Message, text: str) -> None:
    try:
        await bot.send_message(chat_id=message.chat.id, text=text,
                               reply_to_message_id=message.message_id,
                               parse_mode=None, disable_web_page_preview=True)
    except Exception as exc:  # noqa: BLE001
        logger.info("ارسال پیام نتیجهٔ اینستاگرام ممکن نشد: %s", exc.__class__.__name__)


async def handle_instagram_message(message: Message, bot: Bot) -> bool:
    """Handle one recognized URL; return False for ordinary non-Instagram text."""
    url = extract_public_instagram_url(message.text or message.caption or "")
    if url is None:
        return False

    actor = message.from_user
    actor_id = actor.id if actor is not None else message.chat.id
    now = time.monotonic()
    previous = _last_download.get(actor_id)
    if previous is not None and now - previous < _DOWNLOAD_COOLDOWN:
        await _send_notice(
            bot, message,
            "⏳ برای هر کاربر، درخواست دریافت بعدی ۱۰ ثانیه پس از درخواست قبلی ممکن است.")
        return True
    _last_download[actor_id] = now
    for old_actor, last_time in list(_last_download.items()):
        if now - last_time > 600:
            _last_download.pop(old_actor, None)

    try:
        with tempfile.TemporaryDirectory(prefix="armando-instagram-") as tmp:
            root = Path(tmp)
            # Pass only non-secret runtime essentials. The isolated temporary
            # home prevents gallery-dl from reading stored cookies/configuration.
            env = {
                "HOME": tmp,
                "XDG_CONFIG_HOME": tmp,
                "XDG_CACHE_HOME": tmp,
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "LANG": "C.UTF-8",
                "PYTHONUTF8": "1",
            }
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "gallery_dl", "--quiet", "--config-ignore",
                "--destination", tmp, url,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=env,
            )
            try:
                await asyncio.wait_for(process.communicate(), timeout=_DOWNLOAD_TIMEOUT)
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                await _send_notice(
                    bot, message,
                    "⏱ دریافت رسانه بیش از حد طول کشید. لطفاً بعداً دوباره امتحان کنید.")
                return True

            if process.returncode != 0:
                logger.info("دریافت لینک عمومی اینستاگرام با کد %s پایان یافت.",
                            process.returncode)
                await _send_notice(
                    bot, message,
                    "❌ دریافت ناموفق بود. فقط پیوند مستقیمِ پست یا ریلِ عمومی پشتیبانی می‌شود؛ "
                    "محتوای خصوصی، حذف‌شده، محدود یا نیازمند ورود قابل دریافت نیست.")
                return True

            files = _media_file_list(root)
            if not files:
                await _send_notice(
                    bot, message,
                    "❌ رسانه‌ای پیدا نشد؛ ممکن است پست خصوصی، حذف‌شده یا برای دریافت محدود باشد.")
                return True

            total_size = sum(path.stat().st_size for path in files)
            if total_size > _MAX_TOTAL_BYTES:
                await _send_notice(
                    bot, message,
                    "❌ حجم رسانه از سقف ارسال ربات در تلگرام بیشتر است.")
                return True

            caption = "📥 دریافت از پیوند عمومی اینستاگرام"
            media = [_input_media(path, caption if index == 0 else "")
                     for index, path in enumerate(files)]
            if len(media) > 1 and any(isinstance(item, InputMediaDocument) for item in media):
                # Telegram albums accept documents only when every item is a
                # document; keep mixed/unusual Instagram formats transferable.
                media = [InputMediaDocument(
                    media=FSInputFile(path), caption=caption if index == 0 else None,
                    parse_mode=None)
                    for index, path in enumerate(files)]
            if len(media) == 1:
                item = media[0]
                if isinstance(item, InputMediaPhoto):
                    await bot.send_photo(
                        chat_id=message.chat.id, photo=item.media, caption=caption,
                        reply_to_message_id=message.message_id, parse_mode=None)
                elif isinstance(item, InputMediaVideo):
                    await bot.send_video(
                        chat_id=message.chat.id, video=item.media, caption=caption,
                        reply_to_message_id=message.message_id, parse_mode=None)
                else:
                    await bot.send_document(
                        chat_id=message.chat.id, document=item.media, caption=caption,
                        reply_to_message_id=message.message_id, parse_mode=None)
            else:
                await bot.send_media_group(
                    chat_id=message.chat.id, media=media,
                    reply_to_message_id=message.message_id)
            return True
    except FileNotFoundError:
        await _send_notice(
            bot, message,
            "❌ ابزار دریافت رسانه روی سرور نصب نیست؛ مدیر ربات باید وابستگی‌ها را به‌روز کند.")
        return True
    except Exception as exc:  # noqa: BLE001 - external site and Telegram failures are expected
        logger.warning("دریافت یا ارسال رسانهٔ اینستاگرام ناموفق بود: %s",
                       exc.__class__.__name__)
        await _send_notice(
            bot, message,
            "❌ دریافت یا ارسال رسانه ناموفق شد. مطمئن شوید پیوند عمومی و رسانه در دسترس است.")
        return True
