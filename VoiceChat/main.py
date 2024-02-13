
import os
import tempfile
import streamlit as st
class DemoError(Exception):
    pass
import sys
import wave
import soundfile as sf
import json
import base64
import time
from VoiceChat.flash import Tranlate
from webui_pages.dialogue.Voice_2 import st_audiorec
API_KEY = 'kVcnfD9iW2XVZSMaLXXXXXX'
SECRET_KEY = 'O9o1O213UgG5LFn0bDGNtoRN3xXXXXXX'

# 需要识别的文件
AUDIO_FILE = './audio/16k.pcm'  # 只支持 pcm/wav/amr 格式，极速版额外支持m4a 格式
# 文件格式
FORMAT = AUDIO_FILE[-3:]  # 文件后缀只支持 pcm/wav/amr 格式，极速版额外支持m4a 格式

CUID = '123456PYTHON'
# 采样率
RATE = 16000  # 固定值

# 普通版

DEV_PID = 1537  # 1537 表示识别普通话，使用输入法模型。根据文档填写PID，选择语言及识别模型
ASR_URL = 'http://vop.baidu.com/server_api'
SCOPE = 'audio_voice_assistant_get'  # 有此scope表示有asr能力，没有请在网页里勾选，非常旧的应用可能没有
IS_PY3 = sys.version_info.major == 3

if IS_PY3:
    from urllib.request import urlopen
    from urllib.request import Request
    from urllib.error import URLError
    from urllib.parse import urlencode
    timer = time.perf_counter
else:
    from urllib.request import urlopen
    from urllib.request import Request
    from urllib.request import URLError
    from urllib import urlencode
    if sys.platform == "win32":
        timer = time.clock
    else:
        # On most other platforms the best timer is time.time()
        timer = time.time
import numpy as np
import openai
import sounddevice as sd
import soundfile as sf
import tweepy
from elevenlabs import generate, play, set_api_key
from langchain.agents import initialize_agent, load_tools
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.utilities.zapier import ZapierNLAWrapper


# set_api_key("abee050fb40f05cbc4c020b1809e23d4")
# openai.api_key = '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'
                            # Set recording parameters
duration = 5  # duration of each recording in seconds
fs = 44100  # sample rate
channels = 1  # number of channels

# 记录音频信息
def record_audio():
    st.toast('正在记录语音......')
    wav_audio_data = st_audiorec()
    st.toast('录音结束!')
    print("Finished recording.")
    # sf.write("./VoiceChat/Audio111.wav", wav_audio_data,fs)
    # if wav_audio_data:
    #     with open("./VoiceChat/Audio111.wav",'wb') as file:
    #         file.write(wav_audio_data)
    
def VoiceChat():
          record_audio()
          text = Tranlate()
          return text

