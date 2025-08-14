import os
import io
import asyncio
import logging
from datetime import datetime
import discord
from discord.commands import Option
from faster_whisper import WhisperModel
from langchain_gigachat import GigaChat
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]

# Настройки ASR
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")

# GigaChat (опционально)
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE")
GIGACHAT_CERT_PATH = os.getenv("GIGACHAT_CERT_PATH")

# Интенты
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
    """Ленивая инициализация Whisper"""
    global whisper_model
    if whisper_model is None:
        logger.info("Инициализирую Whisper модель...")
        whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
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
    model = get_whisper_model()
    try:
        segments, info = model.transcribe(path, vad_filter=True, language="ru")
        text = " ".join(seg.text for seg in segments)
        return text.strip()
    except Exception as e:
        logger.error(f"Ошибка ASR: {e}")
        return f"Ошибка распознавания: {e}"

def _summarize_with_gigachat(transcript: str) -> str:
    if not GIGACHAT_CREDENTIALS:
        return "GigaChat не настроен (отсутствуют учетные данные)"
    
    prompt = f"""
Ты — ассистент-аналитик. Дан транскрипт группового разговора из Discord.
Проанализируй: кто прав и почему? Кто какую позицию транслирует?

Транскрипт:
{transcript}
"""
    
    try:
        # ИСПРАВЛЕНО: без контекстного менеджера
        giga = GigaChat(
            credentials=GIGACHAT_CREDENTIALS, 
            scope=GIGACHAT_SCOPE,
            ca_bundle_file=GIGACHAT_CERT_PATH,
            model='GigaChat-Pro'
        )
        resp = giga.invoke(prompt)
        return resp.content
    except Exception as e:
        logger.error(f"Ошибка GigaChat: {e}")
        return f"Ошибка суммаризации: {e}"

async def _on_recording_finished(sink: discord.sinks.Sink, channel: discord.TextChannel):
    """ИСПРАВЛЕННЫЙ callback - только sink и channel"""
    guild_id = channel.guild.id
    logger.info(f"Завершение записи для гильдии {guild_id}")
    
    # Отключение и очистка сессии
    data = sessions.get(guild_id)
    if data:
        try:
            await data["vc"].disconnect(force=True)
        except Exception as e:
            logger.error(f"Ошибка отключения: {e}")
        
        if (task := data.get("timer")) and not task.done():
            task.cancel()
        
        sessions.pop(guild_id, None)
    
    # Обработка аудио
    parts = []
    files_to_send = []
    
    for user_id, audio in sink.audio_data.items():
        member = channel.guild.get_member(user_id)
        speaker = member.display_name if member else f"User_{user_id}"
        filename = f"{speaker}_{guild_id}.wav"
        
        # Сохранение аудиофайла
        try:
            audio_data = audio.file
            audio_data.seek(0)
            with open(filename, "wb") as f:
                f.write(audio_data.read())
            
            # Транскрипция
            text = _asr_one(filename)
            if text and text.strip():
                parts.append(f"## {speaker}\n{text}")
            
            # Добавляем файл для отправки
            files_to_send.append(discord.File(filename, filename=f"{speaker}.wav"))
            
        except Exception as e:
            logger.error(f"Ошибка обработки аудио для {speaker}: {e}")
            parts.append(f"## {speaker}\n_Ошибка обработки: {e}_")
        finally:
            # Удаляем временный файл
            try:
                os.unlink(filename)
            except:
                pass
    
    # Формирование результатов
    transcript_text = "\n\n".join(parts) if parts else "_Тишина или ошибки распознавания_"
    summary = _summarize_with_gigachat(transcript_text)
    
    # Отправка результатов
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    
    try:
        if files_to_send:
            await channel.send(
                content=f"📊 Запись завершена ({timestamp})",
                files=files_to_send[:10]  # Ограничение Discord
            )
        
        # Транскрипт
        transcript_file = discord.File(
            io.BytesIO(transcript_text.encode("utf-8")), 
            filename="transcript.md"
        )
        await channel.send("📝 **Транскрипт:**", file=transcript_file)
        
        # Анализ
        summary_file = discord.File(
            io.BytesIO(summary.encode("utf-8")), 
            filename="analysis.md"
        )
        await channel.send("🧠 **Анализ:**", file=summary_file)
        
    except Exception as e:
        logger.error(f"Ошибка отправки результатов: {e}")
        await channel.send(f"❌ Ошибка при отправке: {e}")

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
        await ctx.respond(f"🎙️ Подключаюсь к **{target_vc.name}** и начинаю запись на {duration} сек...")
        
        vc = await target_vc.connect()
        timer_task = asyncio.create_task(_stop_after(ctx.guild.id, duration))
        sessions[ctx.guild.id] = {"vc": vc, "timer": timer_task}
        
        # ИСПРАВЛЕННЫЙ вызов - только sink и callback
        vc.start_recording(
            discord.sinks.WaveSink(), 
            _on_recording_finished,
            ctx.channel
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
        await ctx.respond("⏹️ Останавливаю запись...")
    except Exception as e:
        logger.error(f"Ошибка остановки: {e}")
        await ctx.respond(f"❌ Ошибка остановки: {e}")

@bot.slash_command(name="ping", description="Проверка работы бота")
async def ping(ctx: discord.ApplicationContext):
    """Простая команда для проверки"""
    whisper_status = "✅ Готов" if whisper_model else "⚠️ Не загружен"
    await ctx.respond(f"🏓 Понг! Whisper: {whisper_status}")

@bot.event
async def on_ready():
    logger.info(f"🤖 Бот запущен: {bot.user} (ID: {bot.user.id})")
    logger.info(f"📡 Подключен к {len(bot.guilds)} серверам")
    
    # Список команд
    commands = [cmd.name for cmd in bot.pending_application_commands]
    logger.info(f"📝 Зарегистрированные команды: {commands}")
    
    # ПРИНУДИТЕЛЬНАЯ синхронизация команд
    try:
        await bot.sync_commands()
        logger.info(f"✅ Синхронизация команд завершена")
        
        for guild in bot.guilds:
            logger.info(f"  └─ Сервер: {guild.name} (ID: {guild.id})")
            
        logger.info("⏳ Команды будут доступны через 1-2 минуты")
        logger.info("💡 Попробуйте набрать '/' в текстовом канале")
            
    except Exception as e:
        logger.error(f"❌ Ошибка синхронизации команд: {e}")
        logger.info("🔄 Команды могут синхронизироваться автоматически")

@bot.event
async def on_application_command_error(ctx: discord.ApplicationContext, error):
    """Обработка ошибок команд"""
    logger.error(f"Ошибка команды {ctx.command}: {error}")
    
    if not ctx.response.is_done():
        await ctx.respond(f"❌ Ошибка выполнения команды: {error}", ephemeral=True)

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        logger.error("❌ DISCORD_TOKEN не найден в переменных окружения!")
        exit(1)
    
    logger.info("🚀 Запуск Discord бота...")
    
    try:
        bot.run(DISCORD_TOKEN)
    except Exception as e:
        logger.error(f"💥 Критическая ошибка запуска: {e}")