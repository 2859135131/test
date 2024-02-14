import streamlit as st
def EdgeVoice():
    html_code = '''
    <!DOCTYPE html>
<html>
<head>
    <title>调用麦克风权限</title>
</head>
<body>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            startRecording();
        });
        function startRecording() {
            navigator.mediaDevices.getUserMedia({ audio: {
                sampleRate: 44100, // 采样率
                channelCount: 1,   // 声道
                } })
                .then(function(stream) {
                    // 获取到音频流
                    console.log("成功获取麦克风访问权限！");
                    var audioTracks = stream.getAudioTracks();
                    console.log(audioTracks[0].label);
                   
                    console.log("开始录音");
                   
                })
                .catch(function(err) {
                    console.error("无法获取麦克风访问权限：", err);
                });
        }
    </script>
</body>
</html>
'''
    st.components.v1.html(html_code, height=300)