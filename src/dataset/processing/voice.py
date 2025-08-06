from pathlib import Path

import numpy as np #

import torchaudio #
from pydub import AudioSegment #

def process_voice_data(input_file):
    if not Path(input_file).exists():
        return

    # 2646000 / 60 * 45 = 1992000
    waveform, sample_rate = torchaudio.load(input_file, num_frames=1992000)

    # # torchaudio.save(data_paths[processed_voice_filename], waveform, sample_rate, format="mp3")
    
    # 1. Ensure tensor is on the CPU, convert to NumPy array, and remove the channel dimension
    #    The .squeeze(0) assumes tensor shape is [1, num_samples] (mono)
    waveform_numpy = waveform.squeeze(0).cpu().numpy()

    # 2. Convert the float audio from range [-1.0, 1.0] to 16-bit integer format,
    #    which is what pydub works with best
    waveform_int16 = (waveform_numpy * 32767).astype(np.int16)

    # 3. Create the AudioSegment object from the raw audio data
    audio_segment = AudioSegment(
        data=waveform_int16.tobytes(),
        sample_width=2,  # 2 bytes = 16-bit audio
        frame_rate=sample_rate,
        channels=1      # 1 for mono, 2 for stereo
    )

    return audio_segment