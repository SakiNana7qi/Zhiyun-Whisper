"""
Zhiyun-Whisper CLI — 智云课堂语音转录工具

Usage:
    python main.py transcribe <URL> [--mode local|api] [--model MODEL]
    python main.py list --course-id COURSE_ID
"""

import os
import sys

import click
from dotenv import load_dotenv

load_dotenv()


def _get_credentials() -> tuple[str, str]:
    username = os.getenv("ZJU_USERNAME", "").strip().strip('"')
    password = os.getenv("ZJU_PASSWORD", "").strip().strip('"')
    if not username or not password:
        click.echo("Error: ZJU_USERNAME and ZJU_PASSWORD must be set in .env", err=True)
        sys.exit(1)
    return username, password


def _get_session(require_auth: bool = False) -> "requests.Session":
    """
    Create an HTTP session. The Zhiyun Classroom catalogue API works
    without CAS authentication for most courses. If auth is needed,
    set require_auth=True.
    """
    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })
    if require_auth:
        from src.auth import login
        username, password = _get_credentials()
        session = login(username, password)
    return session


@click.group()
def cli():
    """智云课堂语音转录工具 (Zhiyun-Whisper)"""
    pass


@cli.command()
@click.argument("url")
@click.option(
    "--mode", "-m",
    type=click.Choice(["local", "api"]),
    default="local",
    help="Transcription backend: local (faster-whisper) or api (OpenAI)",
)
@click.option(
    "--model",
    default="large-v3",
    help="Whisper model size for local mode (tiny/base/small/medium/large-v3)",
)
@click.option(
    "--language", "-l",
    default="zh",
    help="Language code for transcription",
)
@click.option(
    "--output-dir", "-o",
    default="output",
    help="Output directory for audio and transcription files",
)
def transcribe(url: str, mode: str, model: str, language: str, output_dir: str):
    """Transcribe a Zhiyun Classroom lesson from URL."""
    from src.crawler import parse_url, fetch_lessons, download_audio
    from src.transcriber import transcribe_local, transcribe_api
    from src.formatter import save_txt, save_srt

    # 1. Parse URL
    click.echo("[1/5] Parsing URL...")
    params = parse_url(url)
    course_id = params["course_id"]
    sub_id = params.get("sub_id")
    click.echo(f"  course_id={course_id}, sub_id={sub_id}")

    # 2. Create session (API works without CAS auth for most courses)
    click.echo("[2/5] Connecting to Zhiyun Classroom...")
    session = _get_session()
    click.echo("  Connected!")

    # 3. Fetch video URL
    click.echo("[3/5] Fetching course catalogue...")
    lessons = fetch_lessons(session, course_id)

    target = None
    if sub_id:
        for lesson in lessons:
            if lesson.sub_id == sub_id:
                target = lesson
                break
        if not target:
            click.echo(f"  Error: sub_id={sub_id} not found in course catalogue", err=True)
            click.echo(f"  Available lessons:")
            for l in lessons:
                status = "available" if l.video_url else "no playback"
                click.echo(f"    [{l.sub_id}] {l.title} ({status})")
            sys.exit(1)
    else:
        available = [l for l in lessons if l.video_url]
        if not available:
            click.echo("  Error: No lessons with available playback found", err=True)
            sys.exit(1)
        click.echo(f"  No sub_id specified, using the latest available lesson")
        target = available[-1]

    if not target.video_url:
        click.echo(f"  Error: Lesson '{target.title}' has no playback URL", err=True)
        sys.exit(1)

    click.echo(f"  Target: {target.title}")

    # 4. Download audio
    click.echo("[4/5] Downloading audio...")
    audio_path = download_audio(
        video_url=target.video_url,
        title=target.title,
        output_dir=output_dir,
    )

    # 5. Transcribe
    click.echo(f"[5/5] Transcribing ({mode} mode)...")
    if mode == "local":
        segments = transcribe_local(
            audio_path=audio_path,
            model_size=model,
            language=language,
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
        if not api_key:
            click.echo("Error: OPENAI_API_KEY must be set in .env for API mode", err=True)
            sys.exit(1)
        segments = transcribe_api(
            audio_path=audio_path,
            api_key=api_key,
            language=language,
        )

    # 6. Save output
    txt_path = save_txt(segments, target.title, output_dir)
    srt_path = save_srt(segments, target.title, output_dir)
    click.echo(f"\nDone! Output files:")
    click.echo(f"  TXT: {txt_path}")
    click.echo(f"  SRT: {srt_path}")


@cli.command("list")
@click.option(
    "--course-id", "-c",
    required=True,
    help="Course ID from the classroom URL",
)
def list_lessons(course_id: str):
    """List all lessons for a course."""
    from src.crawler import fetch_lessons

    click.echo("Connecting to Zhiyun Classroom...")
    session = _get_session()

    click.echo(f"Fetching lessons for course_id={course_id}...")
    lessons = fetch_lessons(session, course_id)

    click.echo(f"\nFound {len(lessons)} lesson(s):\n")
    for i, lesson in enumerate(lessons, start=1):
        status = "available" if lesson.video_url else "no playback"
        click.echo(f"  {i:3d}. [{lesson.sub_id}] {lesson.title} ({status})")

    available_count = sum(1 for l in lessons if l.video_url)
    click.echo(f"\n  Total: {len(lessons)} | Available for download: {available_count}")


if __name__ == "__main__":
    cli()
