import speech_recognition as sr
from io import BytesIO


def speech_to_text(audio_file):
    """
    Convert audio bytes or a file-like object to text.
    """
    if audio_file is None:
        return ""

    try:
        if hasattr(audio_file, "getvalue"):
            audio_bytes = audio_file.getvalue()
        elif isinstance(audio_file, (bytes, bytearray)):
            audio_bytes = audio_file
        else:
            return ""

        if not audio_bytes:
            return ""

        recognizer = sr.Recognizer()
        audio_data = BytesIO(audio_bytes)

        with sr.AudioFile(audio_data) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        return text.strip()

    except sr.UnknownValueError:
        return ""

    except sr.RequestError as e:
        print("Speech recognition service error:", e)
        return ""

    except Exception as e:
        print("Voice processing error:", e)
        return ""


__all__ = ["speech_to_text"]
