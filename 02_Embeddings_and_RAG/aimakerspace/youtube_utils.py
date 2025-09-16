import yt_dlp
from typing import List, Dict, Any
from datetime import datetime
import re

class YouTubeTranscriptLoader:
    def __init__(self, language: str = "en"):
        self.language = language

    def extract_video_id(self, url: str) -> str | None:
        """Extract video ID from common YouTube URL formats."""
        patterns = [
            r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)",
            r"youtube\.com/watch\?.*v=([^&\n?#]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def get_video_info(self, video_url: str) -> Dict[str, Any]:
        """Get video info (and whether transcript is available)."""
        video_id = self.extract_video_id(video_url)
        if not video_id:
            return {"valid": False, "error": "Invalid YouTube URL"}

        ydl_opts = {
            "skip_download": True, 
            "quiet": True,
            "outtmpl": "/Users/home/Documents/aie8/02_Embeddings_and_RAG/aimakerspace/data/%(title)s.%(ext)s"
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                subtitles = info.get("subtitles", {})
                auto_subs = info.get("automatic_captions", {})
                available_langs = list(subtitles.keys()) + list(auto_subs.keys())

                return {
                    "valid": True,
                    "video_id": video_id,
                    "video_url": video_url,
                    "language": self.language,
                    "available_languages": available_langs,
                }
        except Exception as e:
            return {"valid": False, "video_id": video_id, "video_url": video_url, "error": str(e)}

    def get_transcript(self, video_url: str, chunk_by_time: bool = True, chunk_duration: int = 60) -> List[Dict[str, Any]]:
        """Fetch transcript using yt_dlp (falls back to auto captions)."""
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {video_url}")

        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [self.language],
            "subtitlesformat": "vtt",
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            subs = info.get("requested_subtitles", {})
            if not subs:
                raise Exception(f"No subtitles available for {video_id} in {self.language}")

            url = subs[self.language]["url"]

        # Fetch and parse VTT manually
        import requests
        resp = requests.get(url)
        resp.raise_for_status()
        vtt_text = resp.text

        # Very simple WebVTT parser
        transcript = []
        for block in vtt_text.split("\n\n"):
            if "-->" in block:
                lines = block.splitlines()
                if len(lines) >= 2:
                    time_line, text_lines = lines[0], lines[1:]
                    start_str, end_str = time_line.split(" --> ")
                    transcript.append({
                        "start": start_str,
                        "end": end_str,
                        "text": self._clean_vtt_text(" ".join(text_lines)).strip()
                    })

        if not chunk_by_time:
            full_text = " ".join([item["text"] for item in transcript])
            return [{
                "text": full_text,
                "metadata": {
                    "source_type": "youtube",
                    "source_name": f"video_{video_id}",
                    "chunk_index": 0,
                    "chunk_length": len(full_text),
                    "timestamp": datetime.now().isoformat(),
                    "total_chunks": 1,
                    "source_specific": {
                        "video_id": video_id,
                        "video_url": video_url,
                        "language": self.language,
                    },
                },
            }]

        # Simple chunking by count (could refine by timestamps)
        chunks, current_chunk, chunk_index = [], [], 0
        for i, item in enumerate(transcript):
            current_chunk.append(item["text"])
            if (i + 1) % 20 == 0 or i == len(transcript) - 1:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source_type": "youtube",
                        "source_name": f"video_{video_id}",
                        "chunk_index": chunk_index,
                        "chunk_length": len(chunk_text),
                        "timestamp": datetime.now().isoformat(),
                        "total_chunks": 0,
                        "source_specific": {
                            "video_id": video_id,
                            "video_url": video_url,
                            "language": self.language,
                        },
                    },
                })
                chunk_index += 1
                current_chunk = []

        for c in chunks:
            c["metadata"]["total_chunks"] = len(chunks)
        return chunks
    
    def _clean_vtt_text(self, text):
        """Remove VTT formatting from transcript text."""
        import re
        # Remove time codes like <00:00:30.920>
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


if __name__ == "__main__":
    loader = YouTubeTranscriptLoader()
    print(loader.get_video_info('https://www.youtube.com/watch?v=d-lZH6TJq2U'))