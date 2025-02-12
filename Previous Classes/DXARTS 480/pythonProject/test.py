from pathlib import Path
import openai

openai.api_key="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"

text = "Hello world!"
speech_file_path = Path(__file__).parent / "speech.mp3"
response = openai.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=text
)
response.stream_to_file(speech_file_path)