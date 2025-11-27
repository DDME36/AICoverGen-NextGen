"""
AICoverGen NextGen - Downloader
YouTube and audio file downloading
"""

import hashlib
import os
import shlex
import subprocess
from contextlib import suppress
from urllib.parse import urlparse, parse_qs

import librosa
import numpy as np
import yt_dlp


def get_youtube_video_id(url: str, ignore_playlist: bool = True) -> str | None:
    """Extract video ID from YouTube URL"""
    query = urlparse(url)
    
    if query.hostname == 'youtu.be':
        if query.path[1:] == 'watch':
            return query.query[2:]
        return query.path[1:]

    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if not ignore_playlist:
            with suppress(KeyError):
                return parse_qs(query.query)['list'][0]
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/watch/':
            return query.path.split('/')[1]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]

    return None


def download_from_youtube(url: str) -> str:
    """Download audio from YouTube URL"""
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': '%(title)s',
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
        'extractaudio': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        if result is None:
            raise Exception("Failed to download from YouTube. Try uploading a local file instead.")
        download_path = ydl.prepare_filename(result, outtmpl='%(title)s.mp3')

    return download_path


def convert_to_stereo(audio_path: str) -> str:
    """Convert mono audio to stereo if needed"""
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)

    if type(wave[0]) != np.ndarray:
        stereo_path = f'{os.path.splitext(audio_path)[0]}_stereo.wav'
        command = shlex.split(f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"')
        subprocess.run(command)
        return stereo_path
    else:
        return audio_path


def get_file_hash(filepath: str) -> str:
    """Generate hash from file content for caching"""
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()[:11]
