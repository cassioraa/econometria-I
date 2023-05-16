from pydub import AudioSegment
from googletrans import Translator
from gtts import gTTS

import speech_recognition as sr
import speech_recognition as sr

def translate_audio(fname):
    # Load the audio file
    audio_file = AudioSegment.from_file(fname+".ogg", format="ogg")

    # Convert to WAV format
    audio_file.export(fname+".wav", format="wav")

    # Load the audio file
    audio_file = fname+".wav"

    # Initialize the recognizer
    r = sr.Recognizer()

    # Load audio file to memory
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)

    # Transcribe the audio to Portuguese
    transcription = r.recognize_google(audio, language="pt-BR")

    print("Transcription (Portuguese):", transcription)

    # Initialize the translator
    translator = Translator()

    # Translate the transcription to English
    translation = translator.translate(transcription, src='pt', dest='en')

    print("Translation (English):", translation.text)


    # Convert translated transcription to audio
    tts = gTTS(text=translation.text, lang='en')
    tts.save(fname + "translated_audio.mp3")
