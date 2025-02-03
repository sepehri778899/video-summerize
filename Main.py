import os
from openai import OpenAI
import tempfile
import subprocess

def extract_audio(video_path):
    """Extract audio from video file and save it temporarily."""
    print("Extracting audio from video...")
    
    # Create a temporary file for the audio
    temp_audio = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    
    # Use ffmpeg through subprocess to extract audio
    subprocess.run([
        'ffmpeg', '-i', video_path,
        '-vn',  # No video
        '-acodec', 'libmp3lame',  # MP3 codec
        '-y',  # Overwrite output file
        temp_audio.name
    ], check=True)
    
    return temp_audio.name

def transcribe_audio(audio_path):
    """Transcribe audio using Lepton's Whisper API."""
    print("Transcribing audio...")
    client = OpenAI(
        base_url="https://whisper-large-v3.lepton.run/api/v1",
        api_key=os.environ.get('LEPTON_API_TOKEN')
    )
    
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    
    return transcript.text

def summarize_text(text, language="en"):
    """Summarize text using Lepton's LLaMA API."""
    print("Generating summary...")
    client = OpenAI(
        base_url="https://llama3-3-70b.lepton.run/api/v1/",
        api_key=os.environ.get('LEPTON_API_TOKEN')
    )
    
    prompt = f"""Please provide a concise summary of the following text in the same language. Focus on the main points and key takeaways , and do not add any extra information , we only want the summery no further information or text is needed:

{text}

Summary:"""
    
    completion = client.chat.completions.create(
        model="llama3.3-70b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    
    return completion.choices[0].message.content

def main():
    # Default video path
    video_path = "default_video.mp4"
    
    # Hardcoded Lepton API token
    os.environ['LEPTON_API_TOKEN'] = "YdlHM8SL0wJK9yhu4cg7sMwS4phAwdIM"
    
    try:
        # Extract audio
        audio_path = extract_audio(video_path)
        
        # Get transcript
        transcript = transcribe_audio(audio_path)
        print("\nTranscript:")
        print("-" * 80)
        print(transcript)
        print("-" * 80)
        
        # Save transcript to file
        with open("transcript.txt", "w", encoding="utf-8") as transcript_file:
            transcript_file.write(transcript)
        
        # Get summary
        summary = summarize_text(transcript)
        print("\nSummary:")
        print("-" * 80)
        print(summary)
        print("-" * 80)
        
        # Save summary to file
        with open("summary.txt", "w", encoding="utf-8") as summary_file:
            summary_file.write(summary)
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        # Clean up temporary audio file
        if 'audio_path' in locals():
            os.unlink(audio_path)

if __name__ == "__main__":
    main()