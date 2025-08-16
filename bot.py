import os
import io
import math
import asyncio
import logging
import tempfile
import shutil
from datetime import datetime

import discord
from discord.commands import Option
from faster_whisper import WhisperModel
from langchain_gigachat import GigaChat
from dotenv import load_dotenv

# -------------------------
# ЛОГИРОВАНИЕ
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# -------------------------
# ENV / КОНСТАНТЫ
# -------------------------
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]

# ASR (Whisper / faster-whisper)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")      # "cpu" | "cuda"
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")   # например: "float16" на GPU

# GigaChat
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_B2B")
GIGACHAT_CERT_PATH = os.getenv("GIGACHAT_CERT_PATH")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Pro")

# Discord вложения
MAX_DISCORD_FILE_BYTES = int(os.getenv("MAX_DISCORD_FILE_BYTES", 7_500_000))  # < 8MB, с запасом

# Кодек/контейнер для записи (нужен ffmpeg для ogg/mp3)
AUDIO_SINK = os.getenv("AUDIO_SINK", "ogg").lower()  # "ogg" | "mp3" | "wav"

# Бюджет контекста для входа в GigaChat-2-Max (128k window)
# Оставляем запас под инструкции и системку:
GIGACHAT_INPUT_TOKENS_BUDGET = int(os.getenv("GIGACHAT_INPUT_TOKENS_BUDGET", 110_000))
# Грубая оценка: ~3.2 символа на токен (работает норм и для кириллицы)
CHAR_PER_TOKEN = float(os.getenv("CHAR_PER_TOKEN", "3.2"))

# -------------------------
# Discord intents
# -------------------------
intents = discord.Intents.default()
intents.guilds = True
intents.voice_states = True
intents.members = True

bot = discord.Bot(intents=intents)

# Активные записи по гильдии
sessions: dict[int, dict] = {}

# Ленивая загрузка Whisper
whisper_model = None


def get_whisper_model():
    """Ленивая инициализация Whisper."""
    global whisper_model
    if whisper_model is None:
        logger.info("Инициализирую Whisper модель.")
        whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE
        )
        logger.info("Whisper модель готова!")
    return whisper_model


async def _stop_after(guild_id: int, seconds: int):
    await asyncio.sleep(seconds)
    data = sessions.get(guild_id)
    if data and data["vc"].is_connected():
        try:
            data["vc"].stop_recording()
        except Exception as e:
            logger.error(f"Ошибка остановки записи: {e}")


def _asr_one(path: str) -> str:
    """Синхронная транскрипция одного файла (для asyncio.to_thread)."""
    model = get_whisper_model()
    try:
        segments, info = model.transcribe(path, vad_filter=True, language="ru")
        text = " ".join(seg.text for seg in segments)
        return text.strip()
    except Exception as e:
        logger.error(f"Ошибка ASR: {e}")
        return f"Ошибка распознавания: {e}"


def _limit_text_to_token_budget(s: str, budget_tokens: int, char_per_token: float) -> str:
    """Обрезает текст с конца под заданный токенный бюджет (грубая оценка)."""
    if budget_tokens <= 0 or char_per_token <= 0:
        return s
    char_budget = int(budget_tokens * char_per_token)
    if len(s) <= char_budget:
        return s
    # Берём "хвост" (как правило, последняя часть диалога полезнее для саммари)
    return s[-char_budget:]


def _summarize_with_gigachat(transcript: str) -> str:
    """Синхронная суммаризация (для asyncio.to_thread)."""
    if not GIGACHAT_CREDENTIALS:
        return "GigaChat не настроен (отсутствуют учетные данные)"

    prompt = f"""
<role>
Ты — ассистент-аналитик с опытом фасилитации групповых дискуссий, в роли опытного психолога.
</role>
<task>
Дан транскрипт группового разговора из Discord.
1) Кратко перечисли ключевые позиции участников (по именам/никам).
2) Выдели ключевые тезисы каждого.
3) Итог: один грамотный вывод (кто прав, какой совет/рекоммендацию по ситуации можно дать)
</task>
Транскрипт:
{transcript}
""".strip()

    try:
        giga = GigaChat(
            credentials=GIGACHAT_CREDENTIALS,
            scope=GIGACHAT_SCOPE,
            ca_bundle_file=GIGACHAT_CERT_PATH,
            model=GIGACHAT_MODEL,
        )
        resp = giga.invoke(prompt)
        return resp.content
    except Exception as e:
        logger.error(f"Ошибка GigaChat: {e}")
        return f"Ошибка суммаризации: {e}"


def make_sink():
    """Выбираем компактный аудио-контейнер (OGG/MP3), fallback на WAV."""
    try:
        # py-cord sinks доступны как discord.sinks.OGGSink / MP3Sink / WaveSink
        if AUDIO_SINK in ("ogg", "opus") and hasattr(discord.sinks, "OGGSink"):
            return discord.sinks.OGGSink()
        if AUDIO_SINK == "mp3" and hasattr(discord.sinks, "MP3Sink"):
            return discord.sinks.MP3Sink()
    except Exception as e:
        logger.warning(f"Не удалось создать {AUDIO_SINK} sink, откатываюсь на WAV: {e}")
    return discord.sinks.WaveSink()


def sink_ext(sink) -> str:
    n = type(sink).__name__.lower()
    if "ogg" in n or "opus" in n:
        return ".ogg"
    if "mp3" in n:
        return ".mp3"
    if "m4a" in n or "aac" in n:
        return ".m4a"
    return ".wav"


async def _on_recording_finished(sink: discord.sinks.Sink, channel: discord.TextChannel):
    """Callback после окончания записи."""
    guild_id = channel.guild.id
    logger.info(f"Завершение записи для гильдии {guild_id}")

    # 1) Отключение и очистка сессии
    data = sessions.get(guild_id)
    if data:
        try:
            await data["vc"].disconnect(force=True)
        except Exception as e:
            logger.error(f"Ошибка отключения: {e}")
        if (task := data.get("timer")) and not task.done():
            task.cancel()
        sessions.pop(guild_id, None)

    # 2) Обработка аудио
    parts: list[str] = []
    files_to_send: list[discord.File] = []

    tmp_dir = tempfile.mkdtemp(prefix=f"rec_{guild_id}_")
    ext = sink_ext(sink)

    try:
        for user_id, audio in sink.audio_data.items():
            member = channel.guild.get_member(user_id)
            speaker = member.display_name if member else f"User_{user_id}"
            tmp_path = os.path.join(tmp_dir, f"{speaker}_{guild_id}{ext}")

            try:
                # Сохраняем байты sink'а во временный файл
                audio.file.seek(0)
                with open(tmp_path, "wb") as f:
                    shutil.copyfileobj(audio.file, f)

                # ASR — выносим в поток, чтобы не блокировать event loop
                text = await asyncio.to_thread(_asr_one, tmp_path)
                if text and text.strip():
                    parts.append(f"## {speaker}\n{text}")

                # Вложение только если влезает в лимит Discord
                try:
                    size = os.path.getsize(tmp_path)
                    if size <= MAX_DISCORD_FILE_BYTES:
                        with open(tmp_path, "rb") as f:
                            buf = io.BytesIO(f.read())
                        buf.seek(0)
                        files_to_send.append(discord.File(buf, filename=f"{speaker}{ext}"))
                    else:
                        logger.warning(f"Файл {tmp_path} {size/1_000_000:.2f}MB > лимита, не прикладываю.")
                except Exception as e:
                    logger.warning(f"Не удалось прикрепить {tmp_path}: {e}")

            except Exception as e:
                logger.error(f"Ошибка обработки аудио для {speaker}: {e}")
                parts.append(f"## {speaker}\n_Ошибка обработки: {e}_")

        transcript_text = "\n\n".join(parts) if parts else "_Тишина или ошибки распознавания_"

        # 3) Режем транскрипт под бюджет контекста GigaChat-2-Max (128k window)
        transcript_for_llm = _limit_text_to_token_budget(
            transcript_text,
            GIGACHAT_INPUT_TOKENS_BUDGET,
            CHAR_PER_TOKEN
        )

        # 4) Суммаризация — также в отдельном потоке
        summary = await asyncio.to_thread(_summarize_with_gigachat, transcript_for_llm)

        # 5) Отправка результатов
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        # Отправим короткое служебное сообщение
        await channel.send(content=f"📊 Запись завершена ({timestamp})")

        # Вложения аудио — по одному, чтобы не упереться в общий payload
        for f in files_to_send:
            try:
                await channel.send(file=f)
            except Exception as e:
                logger.error(f"Не удалось отправить вложение: {e}")

        # Транскрипт / Анализ — файлами из памяти
        transcript_file = discord.File(
            io.BytesIO(transcript_text.encode("utf-8")),
            filename="transcript.md"
        )
        await channel.send("📝 **Транскрипт:**", file=transcript_file)

        summary_file = discord.File(
            io.BytesIO(summary.encode("utf-8")),
            filename="analysis.md"
        )
        await channel.send("🧠 **Анализ:**", file=summary_file)

    except Exception as e:
        logger.error(f"Ошибка отправки результатов: {e}")
        await channel.send(f"❌ Ошибка при отправке: {e}")
    finally:
        # 6) Гарантированная очистка временных файлов (Windows friendly)
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Не удалось удалить {tmp_dir}: {e}")


@bot.slash_command(name="listen", description="Записать голосовой канал с транскрипцией и анализом")
async def listen(
    ctx: discord.ApplicationContext,
    duration: Option(int, "Длительность записи в секундах", required=False, default=300, min_value=10, max_value=3600)
):
    # Поиск голосового канала
    voice_state = ctx.author.voice
    target_vc = voice_state.channel if voice_state else None

    if not target_vc and isinstance(ctx.channel, discord.TextChannel) and ctx.channel.category:
        for ch in ctx.channel.category.channels:
            if isinstance(ch, discord.VoiceChannel):
                target_vc = ch
                break

    if not target_vc:
        return await ctx.respond(
            "❌ Не найден голосовой канал. Войдите в голосовой канал или используйте команду в категории с голосовым каналом.",
            ephemeral=True
        )

    if ctx.guild.id in sessions:
        return await ctx.respond(
            "⚠️ Запись уже идёт на этом сервере. Остановите её командой `/stop_listening`",
            ephemeral=True
        )

    try:
        await ctx.respond(f"🎙️ Подключаюсь к **{target_vc.name}** и начинаю запись на {duration} сек.")

        vc = await target_vc.connect()
        timer_task = asyncio.create_task(_stop_after(ctx.guild.id, duration))
        sessions[ctx.guild.id] = {"vc": vc, "timer": timer_task}

        # Выбираем компактный аудио sink (OGG/MP3), fallback на WAV
        sink = make_sink()
        vc.start_recording(
            sink,
            _on_recording_finished,
            ctx.channel,  # будет передан вторым аргументом в callback
        )

        logger.info(f"Запись начата в {target_vc.name}")

    except Exception as e:
        logger.error(f"Ошибка начала записи: {e}")
        await ctx.respond(f"❌ Ошибка подключения: {e}", ephemeral=True)
        sessions.pop(ctx.guild.id, None)


@bot.slash_command(name="stop_listening", description="Остановить текущую запись")
async def stop_listening(ctx: discord.ApplicationContext):
    data = sessions.get(ctx.guild.id)
    if not data:
        return await ctx.respond("ℹ️ Активных записей нет", ephemeral=True)

    try:
        data["vc"].stop_recording()
        await ctx.respond("⏹️ Останавливаю запись.")
    except Exception as e:
        logger.error(f"Ошибка остановки: {e}")
        await ctx.respond(f"❌ Ошибка остановки: {e}")


@bot.slash_command(name="ping", description="Проверка работы бота")
async def ping(ctx: discord.ApplicationContext):
    whisper_status = "✅ Готов" if whisper_model else "⚠️ Не загружен"
    await ctx.respond(f"🏓 Понг! Whisper: {whisper_status}")


@bot.event
async def on_ready():
    logger.info(f"🤖 Бот запущен: {bot.user} (ID: {bot.user.id})")
    logger.info(f"📡 Подключен к {len(bot.guilds)} серверам")

    commands = [cmd.name for cmd in bot.pending_application_commands]
    logger.info(f"📝 Зарегистрированные команды: {commands}")

    # Принудительная синхронизация
    try:
        await bot.sync_commands()
        logger.info("✅ Синхронизация команд завершена")
        for guild in bot.guilds:
            logger.info(f"  └─ Сервер: {guild.name} (ID: {guild.id})")
        logger.info("⏳ Команды будут доступны через 1–2 минуты")
        logger.info("💡 Попробуй набрать '/' в текстовом канале")
    except Exception as e:
        logger.error(f"❌ Ошибка синхронизации команд: {e}")
        logger.info("🔄 Команды могут синхронизироваться автоматически")


@bot.event
async def on_application_command_error(ctx: discord.ApplicationContext, error):
    logger.error(f"Ошибка команды {ctx.command}: {error}")
    if not ctx.response.is_done():
        await ctx.respond(f"❌ Ошибка выполнения команды: {error}", ephemeral=True)


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        logger.error("❌ DISCORD_TOKEN не найден в переменных окружения!")
        raise SystemExit(1)

    logger.info("🚀 Запуск Discord бота...")
    try:
        bot.run(DISCORD_TOKEN)
    except Exception as e:
        logger.error(f"💥 Критическая ошибка запуска: {e}")
