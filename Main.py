import ffmpeg
import os
from openai import OpenAI

input_file = "1.mp4"
output_audio_file = "output_audio.mp3"
output_text_file = "transcript.txt"
summary_file = "summary.txt"

# Extract audio
ffmpeg.input(input_file).output(output_audio_file, format='mp3').run(cmd='C:/ffmpeg/bin/ffmpeg.exe')
print("Audio extracted successfully!")

# Initialize OpenAI client for Whisper
whisper_client = OpenAI(
  base_url="https://whisper-large-v3.lepton.run/api/v1",
  api_key="usRA4vRzh7cfhZSwfFnE2a0JUMkRJfBg"
)

# Open the audio file
with open(output_audio_file, "rb") as audio_file:
    # Get the transcript
    transcript = whisper_client.audio.transcriptions.create(
      model="whisper-1",
      file=audio_file
    )

# Print the transcript for debugging
print("Transcript response:", transcript)

# Check if the transcript was successfully created
if 'text' in transcript:
    # Save the transcript to a text file
    with open(output_text_file, "w") as text_file:
        text_file.write(transcript['text'])
    print("Transcript saved to", output_text_file)
else:
    print("Failed to get transcript:", transcript)
    exit()

# Initialize OpenAI client for Llama 3.3 70B
llama_client = OpenAI(
  base_url="https://llama3.3-70b.api.openai.com/v1",
  api_key="usRA4vRzh7cfhZSwfFnE2a0JUMkRJfBg"
)

# Summarize the transcript
summary = llama_client.completions.create(
    model="llama3.3-70b",
    messages=[
        {"role": "system", "content": "Summarize the following text."},
        {"role": "user", "content": transcript['text']}
    ],
    max_tokens=128
)

# Save the summary to a text file
with open(summary_file, "w") as file:
    file.write(summary['choices'][0]['message']['content'])
print("Summary saved to", summary_file)