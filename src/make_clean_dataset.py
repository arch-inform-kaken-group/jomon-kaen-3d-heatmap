import os
import shutil
import json

from dotenv import load_dotenv
from google import genai

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dataset.processing.voice import process_voice_data

# Import Google Cloud Speech-to-Text libraries
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

# kotoba-whisper-v2.0
import torch
from transformers import pipeline

model_id = "kotoba-tech/kotoba-whisper-v2.0"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_kwargs = {
    "attn_implementation": "sdpa"
} if torch.cuda.is_available() else {}
generate_kwargs = {"language": "ja", "task": "transcribe"}

# load model
pipe = pipeline("automatic-speech-recognition",
                model=model_id,
                torch_dtype=torch_dtype,
                device=device,
                model_kwargs=model_kwargs)

# PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
PROJECT_ID = "jomon-kaen"

# Add a boolean flag to control transcription
TRANSCRIBE_VOICE_DATA = False
TRANSCRIBE_VOICE_DATA_KOTOBA = False

REFINE_TRANSCRIPT = False
REFINE_TRANSCRIPT_KOTOBA = True

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variables
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable not found. Please set it in your .env file."
    )

client = genai.Client(api_key=API_KEY)


def refine_transcript_with_gemini_my_gstt(audio_file, raw_transcript_json):
    """Refine a transcript using Gemini API with a specific prompt.

    Args:
        audio_file (str): Path to the 45-second audio file.
        raw_transcript_json (str): Path to the raw transcript JSON file.

    Returns:
        str: The refined transcript or an error message.
    """
    try:
        # Read the raw transcript JSON
        with open(raw_transcript_json, "r", encoding='utf-8') as f:
            raw_data = json.load(f)

        imperfect_transcript = raw_data.get('highest_confidence_transcript',
                                            '')

        # Construct the prompt
        prompt = (
            "This is a comment on pottery viewing, please align the audio data with the imperfect transcript. "
            "Please try to make the transcript make logical sense. "
            "Give the complete and accurate transcript only (do not give other information, warning or text than the transcript), "
            "please make it as close to the audio as possible. Listen to the audio and give the transcript. "
            "Some of the words in the transcript may be wrong, example words such as \"person\" may actually be \"pottery\", "
            "or \"jock\" may be \"jug\". ",
            "There may also be background noise, please ignore the background noise."
        )

        # STEP 1: Upload the audio file to the Gemini API
        audio_file_part = client.files.upload(
            file=audio_file, config={"mime_type": "audio/mp3"})

        # STEP 2: Use the uploaded file object in the contents list
        contents = [
            prompt, audio_file_part,
            f"Imperfect Transcript: {imperfect_transcript}"
        ]

        # Use the client to generate content
        response = client.models.generate_content(model='gemini-2.5-flash',
                                                  contents=contents)

        return response.text.strip()

    except Exception as e:
        return f"Error refining transcript with Gemini: {e}"


def transcribe_audio_v2(audio_file: str) -> dict:
    """Transcribe an audio file using Google Cloud Speech-to-Text V2 and return all data. 

    Args: 
        audio_file (str): Path to the local audio file to be transcribed. 

    Returns: 
        dict: A dictionary containing all transcription results, confidence, and metadata. 
    """
    with open(audio_file, "rb") as f:
        audio_content = f.read()

    # Instantiates a client
    client = SpeechClient()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        # language_codes=["en-US", "ms-MY", "ta-IN"], # Can use this with "long" model, but the other languages get detected with telephony as well
        language_codes=["en-SG"],
        model="telephony",
        features=cloud_speech.RecognitionFeatures(
            max_alternatives=3,
            enable_word_time_offsets=True,
        ),
        adaptation=cloud_speech.SpeechAdaptation(phrase_sets=[
            cloud_speech.SpeechAdaptation.AdaptationPhraseSet(
                phrase_set=
                f'projects/{PROJECT_ID}/locations/global/phraseSets/malaysia')
        ]))

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/global/recognizers/_",
        config=config,
        content=audio_content,
    )

    # Transcribes the audio into text
    try:
        response = client.recognize(request=request)

        # Process the response to extract all data
        results = []
        highest_confidence_transcript_list = []

        for result in response.results:
            alternatives = []

            # Get the highest confidence transcript for the top-level key
            if result.alternatives and len(result.alternatives) > 0:
                highest_confidence_transcript_list.append(
                    result.alternatives[0].transcript)

            # Now, process all alternatives (up to 3)
            for alternative in result.alternatives:
                words = []
                for word_info in alternative.words:
                    words.append({
                        "word":
                        word_info.word,
                        "confidence":
                        word_info.confidence if hasattr(
                            word_info, 'confidence') else None,
                        "start_time":
                        word_info.start_offset.total_seconds(),
                        "end_time":
                        word_info.end_offset.total_seconds(),
                    })

                alternatives.append({
                    "transcript":
                    alternative.transcript,
                    "confidence":
                    alternative.confidence
                    if hasattr(alternative, 'confidence') else None,
                    "words":
                    words,
                })

            results.append({
                "language_code":
                result.language_code,
                "alternatives":
                alternatives,
                "result_end_offset":
                result.result_end_offset.total_seconds()
                if result.result_end_offset else None,
            })

        combined_highest_confidence_transcript = " ".join(
            highest_confidence_transcript_list)

        # Add the new key and metadata to the overall response
        return {
            "highest_confidence_transcript":
            combined_highest_confidence_transcript,
            "metadata": {
                "total_billed_duration":
                response.metadata.total_billed_duration.total_seconds()
                if response.metadata.total_billed_duration else None,
                "request_id":
                response.metadata.request_id,
            },
            "results": results
        }

    except Exception as e:
        print(f"Error transcribing {audio_file}: {e}")
        return {"error": str(e)}


def refine_transcript_with_gemini_jp_kotoba(audio_file: str, raw_transcript_json: str):
    """
    Refines a Japanese transcript using the Gemini API with a specific Japanese prompt.
    This version is updated to work with the JSON format from the kotoba-whisper-v2 pipeline.

    Args:
        audio_file (str): Path to the 45-second audio file (e.g., in MP3 format).
        raw_transcript_json (str): Path to the raw transcript JSON file from Whisper.

    Returns:
        str: The refined transcript or an error message.
    """
    try:
        # Read the raw transcript JSON generated by the Whisper pipeline
        with open(raw_transcript_json, "r", encoding='utf-8') as f:
            raw_data = json.load(f)

        # MODIFICATION 1: Get the full transcript from the 'text' key
        # The previous key was 'highest_confidence_transcript'. The new key is 'text'.
        imperfect_transcript = raw_data.get('text', '')
        
        if not imperfect_transcript:
            return "Error: Could not find 'text' key or transcript is empty in the JSON file."

        # MODIFICATION 2: Construct the prompt in Japanese
        prompt = (
            "これは土器鑑賞に関するコメントです。音声データと以下の不完全な書き起こしテキストを照合し、修正してください。\n"
            "最終的な書き起こしが論理的に意味の通る内容になるようにしてください。\n"
            "完全かつ正確な書き起こしテキストのみを返答してください（書き起こし以外の情報、警告、余分なテキストは含めないでください）。\n"
            "できる限り音声に忠実な内容にしてください。音声を注意深く聴き、正確な書き起こしをお願いします。\n"
            "提供された書き起こしには、似た発音の単語の誤りが含まれている可能性があります。\n"
            "特に、この音声は「土器（どき）」や「土偶（どぐう）」に関するコメントです。\n"
            "これらの単語が、例えば「道具（どうぐ） / ドグ / 時 / どき / どぐう / ドキ」のような、音の似た別の単語として誤って書き起こされている可能性が高いです。\n"
            "文脈を注意深く判断し、正確な単語に修正してください。\n"
            "背景に雑音が含まれる場合がありますが、それは無視して話者の言葉だけを書き起こしてください。\n"
        )

        # STEP 1: Upload the audio file. Make sure your client library supports this method.
        audio_file_part = client.files.upload(file=audio_file, config={"mime_type": "audio/mp3"})

        # STEP 2: Construct the contents for the API call
        contents = [
            prompt,
            f"不完全な書き起こし: {imperfect_transcript}",
            audio_file_part
        ]
        
        # Use your model client to generate content
        response = client.models.generate_content(model='gemini-2.5-flash',
                                                  contents=contents)

        
        # STEP 3: Clean and return the response text
        return response.text.strip()

    except FileNotFoundError:
        return f"Error: The file was not found at {raw_transcript_json}"
    except Exception as e:
        return f"Error refining transcript with Gemini: {e}"


def transcribe_audio_kotoba_whisper_v2(audio_file: str):
    """
    Transcribes an audio file using the kotoba-whisper-v2.0 model,
    saves the output to a JSON file, and returns the result.

    Args:
        audio_file (str): The path to the audio file to be transcribed.

    Returns:
        dict: A dictionary containing the transcription text and chunk-level timestamps.
    """
    print(f"Starting transcription for: {audio_file}")

    # Run the inference pipeline
    result = pipe(
        audio_file,
        return_timestamps=True,
        generate_kwargs=generate_kwargs
    )

    return result


CORRECTION_MAP = {
    "面白い・気になる形だ": "面白い・気になる形だ",
    "不思議・意味不明": "美しい・芸術的だ",
    "何も感じない": "不思議・意味不明",
    "不気味・不安・怖い": "不気味・不安・怖い",
    "美しい・芸術的だ": "何も感じない",
    "Interesting and attentional shape": "Interesting and attentional shape",
    "Beautiful and artistic": "Feel nothing",
    "Strange and incomprehensible": "Beautiful and artistic",
    "Creepy / unsettling / scary": "Creepy / unsettling / scary",
    "Feel nothing": "Strange and incomprehensible",
}


def increment_error(key, path, errors: dict):
    if errors.get(key) is None:
        errors[key] = {'count': 1, 'paths': set([path])}
    else:
        errors[key]['count'] += 1
        errors[key]['paths'].add(path)
    return errors


def main(root="", output_dir=""):
    errors = {}

    # Check if each data instance / file path exists
    data = []
    if not Path(root).exists():
        raise (ValueError(f"Root directory not found: {root}"))

    os.makedirs(output_dir, exist_ok=True)

    # Filter based on group, session, model
    print(f"\nCHECKING RAW DATA PATHS")

    group_keys = os.listdir(root)
    for g in group_keys:
        group_path = Path(root) / g

        session_keys = os.listdir(group_path)
        for s in tqdm(session_keys, desc=g):
            session_path = group_path / s

            pottery_keys = os.listdir(session_path)
            for p in pottery_keys:

                data_paths = {}
                pottery_path = session_path / p

                save_path = Path(output_dir) / g / s / p

                if p == 'language.txt':
                    if not save_path.exists():
                        shutil.copy(pottery_path, save_path)
                    continue

                pointcloud_path = pottery_path / "pointcloud.csv"
                qa_path = pottery_path / "qa.csv"
                model_path = pottery_path / "model.obj"
                voice_path = pottery_path / "session_audio_0.wav"

                pointcloud_save_path = save_path / "pointcloud.csv"
                qa_save_path = save_path / "qa_corrected.csv"
                model_save_path = save_path / "model.obj"
                voice_save_path = save_path / "session_audio_45s.mp3"
                transcript_save_path = save_path / "raw_transcript.json"
                refined_transcript_save_path = save_path / "refined_transcript.txt"

                # Check if paths exist and increment error
                if model_path.exists():
                    data_paths['model'] = str(model_path)
                    data_paths['model_save'] = str(model_save_path)
                    if pointcloud_path.exists():
                        data_paths['pointcloud'] = str(pointcloud_path)
                        data_paths['pointcloud_save'] = str(
                            pointcloud_save_path)
                    else:
                        errors = increment_error(
                            'Point cloud path does not exist',
                            str(pointcloud_path), errors)

                    if qa_path.exists():
                        data_paths['qa'] = str(qa_path)
                        data_paths['qa_save'] = str(qa_save_path)
                    else:
                        errors = increment_error('QNA path does not exist',
                                                 str(qa_path), errors)
                else:
                    errors = increment_error('Model path does not exist',
                                             str(model_path), errors)

                if voice_path.exists():
                    data_paths['voice'] = str(voice_path)
                    data_paths['voice_save'] = str(voice_save_path)
                    data_paths['transcript_save'] = str(transcript_save_path)
                    data_paths['refined_transcript_save'] = str(
                        refined_transcript_save_path)
                else:
                    errors = increment_error('Voice path does not exist',
                                             str(voice_path), errors)

                data_paths['SAVE'] = str(save_path)
                data_paths['GROUP'] = g
                data_paths['SESSION_ID'] = s
                data_paths['ID'] = p

                data.append(data_paths)

    n_valid_data = len(data)
    print(n_valid_data)

    model_count = 0
    pc_count = 0
    qa_count = 0
    voice_count = 0
    transcript_count = 0

    for data_paths in tqdm(data, desc="Making Clean Dataset"):
        os.makedirs(data_paths['SAVE'], exist_ok=True)

        if data_paths.get('model'):
            model_count += 1
            shutil.copy(data_paths['model'], data_paths['model_save'])

        if data_paths.get('pointcloud'):
            pc_count += 1
            shutil.copy(data_paths['pointcloud'],
                        data_paths['pointcloud_save'])

        if data_paths.get('qa'):
            qa_count += 1
            df = pd.read_csv(data_paths['qa'])
            df['answer'] = df['answer'].str.strip()
            df['answer'] = df['answer'].map(CORRECTION_MAP).fillna(
                df['answer'])
            df.to_csv(data_paths['qa_save'], index=False, encoding='utf-8-sig')

        if data_paths.get('voice'):
            voice_count += 1
            audio_segment = process_voice_data(data_paths['voice'])
            audio_segment.export(data_paths['voice_save'],
                                 format="mp3",
                                 bitrate="16k")

            if TRANSCRIBE_VOICE_DATA:
                print(f"Transcribing {data_paths['voice_save']}")
                transcript_data = transcribe_audio_v2(data_paths['voice_save'])

                with open(data_paths['transcript_save'], "w",
                          encoding='utf-8') as f:
                    json.dump(transcript_data, f, indent=4, ensure_ascii=False)

                transcript_count += 1

            if TRANSCRIBE_VOICE_DATA_KOTOBA:
                print(f"Transcribing {data_paths['voice_save']}")
                transcript_data = transcribe_audio_kotoba_whisper_v2(
                    data_paths['voice_save'])

                with open(data_paths['transcript_save'], "w",
                          encoding='utf-8') as f:
                    json.dump(transcript_data, f, indent=4, ensure_ascii=False)

                transcript_count += 1

            if REFINE_TRANSCRIPT or REFINE_TRANSCRIPT_KOTOBA:
                if Path(data_paths['transcript_save']).exists():
                    print(f"Refining transcript for {data_paths['ID']}")
                    if REFINE_TRANSCRIPT:
                        refined_transcript = refine_transcript_with_gemini_my_gstt(
                            data_paths['voice_save'],
                            data_paths['transcript_save'])
                    elif REFINE_TRANSCRIPT_KOTOBA:
                        refined_transcript = refine_transcript_with_gemini_jp_kotoba(
                            data_paths['voice_save'],
                            data_paths['transcript_save'])
                    with open(data_paths['refined_transcript_save'],
                              "w",
                              encoding='utf-8') as f:
                        f.write(refined_transcript)
                else:
                    errors = increment_error(
                        'Raw transcript not found for refinement',
                        str(data_paths['transcript_save']), errors)

    print('MODEL\t|\tPC\t|\tQA\t|\tVOICE\t|\tTRANSCRIPT')
    print(model_count, '\t|\t', pc_count, '\t|\t', qa_count, '\t|\t',
          voice_count, '\t|\t', transcript_count)


if __name__ == "__main__":
    main(
        root=r"D:\storage\jomon_kaen\data",
        output_dir=r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan",
    )

    # main(
    #     root="./src/data_my",
    #     output_dir="./src/jomon_kaen_dataset/malaysia",
    # )
