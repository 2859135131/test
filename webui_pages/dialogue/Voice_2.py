import os
import numpy as np
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components
import soundfile as sf
import struct
from VoiceChat.flash import Tranlate
duration = 5  # duration of each recording in seconds
fs = 44100  # sample rate
channels = 1  # number of channels
def st_audiorec():
    print("in")
    # get parent directory relative to current directory
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    # Custom REACT-based component for recording client audio in browser
    build_dir = os.path.join(parent_dir, "frontend/build")
    # specify directory and initialize st_audiorec object functionality
    st_audiorec = components.declare_component('st_audiorec',path="./frontend/build")
    # Create an instance of the component: STREAMLIT AUDIO RECORDER
    print("yes")
    raw_audio_data = st_audiorec()  # raw_audio_data: stores all the data returned from the streamlit frontend
    print("no")
    wav_bytes = None                # wav_bytes: contains the recorded audio in .WAV format after conversion
    # the frontend returns raw audio data in the form of arraybuffer
    # (this arraybuffer is derived from web-media API WAV-blob data)
    text =""
    if isinstance(raw_audio_data, dict):  # retrieve audio data
        with st.spinner('retrieving audio-recording...'):
            ind, raw_audio_data = zip(*raw_audio_data['arr'].items())
            ind = np.array(ind, dtype=int)  # convert to np array
            raw_audio_data = np.array(raw_audio_data)  # convert to np array
            sorted_ints = raw_audio_data[ind]
            stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
            # wav_bytes contains audio data in byte format, ready to be processed further
            wav_bytes = stream.read()
            #sf.write("../VoiceChat/Audio111.wav", wav_bytes,fs,channels=channels)
            # 步骤2和3：创建WAV文件头并合并数据
            # 这里假设您使用的是标准的PCM编码和采样率为44100Hz

            sample_rate = 44100
            num_channels = 1
            bits_per_sample = 16

            # 创建WAV文件头
            header = struct.pack('<4sI4s4sIHHIIHH4sI',
                                b'RIFF', 36 + len(wav_bytes), b'WAVE', b'fmt ', 16, 1, num_channels,
                                sample_rate, sample_rate * num_channels * bits_per_sample // 8,
                                num_channels * bits_per_sample // 8, bits_per_sample, b'data', len(wav_bytes))

            # 合并文件头和音频数据
            wav_data = header + wav_bytes

            # 步骤4：写入新的WAV文件
            with open('output.wav', 'wb') as file:
                file.write(wav_bytes)   
            text = Tranlate()
            os.remove('output.wav')  # 删除文件
            if st.session_state.PROMPT != text:
                st.toast('录音结束!')
    return text
    
