import streamlit as st
from webui_pages.utils import *
from streamlit_chatbox import *
from streamlit_modal import Modal
from datetime import datetime
import os
import re
import time
import random
from configs import (
    TEMPERATURE,
    HISTORY_LEN,
    PROMPT_TEMPLATES,
    DEFAULT_KNOWLEDGE_BASE,
    DEFAULT_SEARCH_ENGINE,
    SUPPORT_AGENT_MODEL,
)
from server.knowledge_base.utils import LOADER_DICT
import uuid
from typing import List, Dict
from webui_pages.dialogue.Voice_2 import st_audiorec





chat_box = ChatBox(assistant_avatar=os.path.join("img", "20190712035319458.jpg"))
def get_messages_history(
    history_len: int, content_in_expander: bool = False
) -> List[Dict]:
    """
    返回消息历史。
    content_in_expander控制是否返回expander元素中的内容，一般导出的时候可以选上，传入LLM的history不需要
    """

    def filter(msg):
        content = [
            x for x in msg["elements"] if x._output_method in ["markdown", "text"]
        ]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    return chat_box.filter_history(history_len=history_len, filter=filter)


@st.cache_data
def Voice(prompt) -> str:
    return prompt

def parse_command(text: str, modal: Modal) -> bool:
    """
    检查用户是否输入了自定义命令，当前支持：
    /new {session_name}。如果未提供名称，默认为“会话X”
    /del {session_name}。如果未提供名称，在会话数量>1的情况下，删除当前会话。
    /clear {session_name}。如果未提供名称，默认清除当前会话
    /help。查看命令帮助
    返回值：输入的是命令返回True，否则返回False
    """
    if m := re.match(r"/([^\s]+)\s*(.*)", text):
        cmd, name = m.groups()
        name = name.strip()
        conv_names = chat_box.get_chat_names()
        if cmd == "help":
            modal.open()
        elif cmd == "new":
            if not name:
                i = 1
                while True:
                    name = f"会话{i}"
                    if name not in conv_names:
                        break
                    i += 1
            if name in st.session_state["conversation_ids"]:
                st.error(f"该会话名称 “{name}” 已存在")
                time.sleep(1)
            else:
                st.session_state["conversation_ids"][name] = uuid.uuid4().hex
                st.session_state["cur_conv_name"] = name
        elif cmd == "del":
            name = name or st.session_state.get("cur_conv_name")
            if len(conv_names) == 1:
                st.error("这是最后一个会话，无法删除")
                time.sleep(1)
            elif not name or name not in st.session_state["conversation_ids"]:
                st.error(f"无效的会话名称：“{name}”")
                time.sleep(1)
            else:
                st.session_state["conversation_ids"].pop(name, None)
                chat_box.del_chat_name(name)
                st.session_state["cur_conv_name"] = ""
        elif cmd == "clear":
            chat_box.reset_history(name=name or None)
        return True
    return False


def dialogue_page(api: ApiRequest, is_lite: bool = False):
    st.session_state.setdefault("conversation_ids", {})
    st.session_state["conversation_ids"].setdefault(
        chat_box.cur_chat_name, uuid.uuid4().hex
    )
    st.session_state.setdefault("file_chat_id", None)
    default_model = api.get_default_llm_model()[0]

    if not chat_box.chat_inited:
        st.toast(
            f"欢迎使用 氢聊 ! \n\n"
            f" 您可以开始提问了."
        )
        chat_box.init_session()

    # 弹出自定义命令帮助信息
    modal = Modal("自定义命令", key="cmd_help", max_width="500")
    if modal.is_open():
        with modal.container():
            cmds = [
                x
                for x in parse_command.__doc__.split("\n")
                if x.strip().startswith("/")
            ]
            st.write("\n\n".join(cmds))

    with st.sidebar:
        # 多会话
        conv_names = list(st.session_state["conversation_ids"].keys())
        index = 0
        if st.session_state.get("cur_conv_name") in conv_names:
            index = conv_names.index(st.session_state.get("cur_conv_name"))
        conversation_name = "default"
        chat_box.use_chat_name(conversation_name)
        conversation_id = st.session_state["conversation_ids"][conversation_name]

        # TODO: 对话模型与会话绑定
        def on_mode_change():
            mode = st.session_state.dialogue_mode
            text = f"已切换到 {mode} 模式。"
            if mode == "知识库问答":
                cur_kb = st.session_state.get("selected_kb")
                if cur_kb:
                    text = f"{text} 当前知识库： `{cur_kb}`。"
            st.toast(text)

        dialogue_mode = "知识库问答"

        def on_llm_change():
            if llm_model:
                config = api.get_model_config(llm_model)
                if not config.get("online_api"):  # 只有本地model_worker可以切换模型
                    st.session_state["prev_llm_model"] = llm_model
                st.session_state["cur_llm_model"] = st.session_state.llm_model

        def llm_model_format_func(x):
            if x in running_models:
                return f"{x} (Running)"
            return x

        def VoiChat(selected_kb, kb_top_k, score_threshold, dialogue_mode, Voiceprompt):
            Voiceprompt = Voiceprompt
            history = get_messages_history(5)
            chat_box.user_say(Voiceprompt)
            if dialogue_mode == "LLM 对话":
                chat_box.ai_say("正在思考...")
                text = ""
                message_id = ""
                r = api.chat_chat(
                    Voiceprompt,
                    history=history,
                    conversation_id=conversation_id,
                    model=llm_model,
                    prompt_name=prompt_template_name,
                    temperature=0.7,
                )
                for t in r:
                    if error_msg := check_error_msg(t):  # check whether error occured
                        st.error(error_msg)
                        break
                    text += t.get("text", "")
                    chat_box.update_msg(text)
                    message_id = t.get("message_id", "")

                metadata = {
                    "message_id": message_id,
                }
                chat_box.update_msg(
                    text, streaming=False, metadata=metadata
                )  # 更新最终的字符串，去除光标
            elif dialogue_mode == "知识库问答":
                chat_box.ai_say(
                    [
                        f"正在查询知识库 `{selected_kb}` ...",
                        Markdown(
                            "...", in_expander=True, title="知识库匹配结果", state="complete"
                        ),
                    ]
                )
                text = ""
                for d in api.knowledge_base_chat(
                    Voiceprompt,
                    knowledge_base_name=selected_kb,
                    top_k=kb_top_k,
                    score_threshold=score_threshold,
                    history=history,
                    model=llm_model,
                    prompt_name=prompt_template_name,
                    temperature=0.7,
                ):
                    if error_msg := check_error_msg(d):  # check whether error occured
                        st.error(error_msg)
                    elif chunk := d.get("answer"):
                        text += chunk
                        chat_box.update_msg(text, element_index=0)
                chat_box.update_msg(text, element_index=0, streaming=False)
                chat_box.update_msg(
                    "\n\n".join(d.get("docs", [])), element_index=1, streaming=False
                )
        running_models = list(api.list_running_models())
        available_models = []
        config_models = api.list_config_models()
        if not is_lite:
            for k, v in config_models.get("local", {}).items():  # 列出配置了有效本地路径的模型
                if v.get("model_path_exists") and k not in running_models:
                    available_models.append(k)
        for k, v in config_models.get("online", {}).items():  # 列出ONLINE_MODELS中直接访问的模型
            if not v.get("provider") and k not in running_models:
                available_models.append(k)
        llm_models = running_models + available_models
        cur_llm_model = st.session_state.get("cur_llm_model", default_model)
        if cur_llm_model in llm_models:
            index = llm_models.index(cur_llm_model)
        else:
            index = 0
        llm_model = "qwen-api"
        if (
            st.session_state.get("prev_llm_model") != llm_model
            and not is_lite
            and not llm_model in config_models.get("online", {})
            and not llm_model in config_models.get("langchain", {})
            and llm_model not in running_models
        ):
            with st.spinner(f"正在加载模型： {llm_model}，请勿进行操作或刷新页面"):
                prev_model = st.session_state.get("prev_llm_model")
                r = api.change_llm_model(prev_model, llm_model)
                if msg := check_error_msg(r):
                    st.error(msg)
                elif msg := check_success_msg(r):
                    st.success(msg)
                    st.session_state["prev_llm_model"] = llm_model

        index_prompt = {
            "LLM 对话": "llm_chat",
            "知识库问答": "knowledge_base_chat",
        }
        prompt_templates_kb_list = list(
            PROMPT_TEMPLATES[index_prompt[dialogue_mode]].keys()
        )
        prompt_template_name = prompt_templates_kb_list[0]
        if "prompt_template_select" not in st.session_state:
            st.session_state.prompt_template_select = prompt_templates_kb_list[0]

        def prompt_change():
            text = f"已切换为 {prompt_template_name} 模板。"
            st.toast(text)

        prompt_template_select = "with_history"
        prompt_template_name = st.session_state.prompt_template_select
        temperature = 0.7
        history_len = 5
        col1,col2,col3 = st.columns([0.2,0.75,0.1])
        a,b,c = st.columns([0.15,0.8,0.1])
        e,d,f = st.columns([0.2,0.75,0.1])
        def on_kb_change():
            st.toast("模型正在加载......")
        with col2:
           SelectProject = st.radio('',options=[ ':sunny:氢能领域分析'],on_change=on_kb_change,captions = [ "Analysis of hydrogen energy"])
        #with b:
            #st.markdown('<br>' * 1, unsafe_allow_html=True)
            #st.write(f"<h2 style='color: white;'>当前选择:{SelectProject[6::]}</h1>", unsafe_allow_html=True)':fire:火电领域分析', ':star:储能领域分析',,':heart:大语言对话模型'"Analysis of thermal power ", "Analysis of energy storage",,"Large language model"
            #st.subheader(f"当前选择:{SelectProject}")
        if SelectProject ==":heart:大语言对话模型":
            dialogue_mode = "LLM 对话" 
        else :
            dialogue_mode == "知识库问答"
        if dialogue_mode == "知识库问答":
            #kb_list = api.list_knowledge_bases(no_remote_api=True)
            selected_kb = "Hydrogen economy"
            if SelectProject == ":fire:火电领域分析":
                selected_kb = "Energy"
            elif SelectProject == ":sunny:氢能领域分析":
                selected_kb = "Hydrogen economy"
            elif SelectProject == ":star:储能领域分析":
                selected_kb = "accumulati"
            kb_top_k =5 #st.number_input("匹配知识条数：", 1, 20, VECTOR_SEARCH_TOP_K)
            score_threshold =0.9#st.number_input("知识匹配分数阈值：", 0.0, 1.0, float(SCORE_THRESHOLD), 0.01)
                # chunk_content = st.checkbox("关联上下文", False, disabled=True)
                # chunk_size = st.slider("关联长度：", 0, 500, 250, disabled=True)
        with d:
           #if dialogue_mode == "知识库问答" :  st.button(':balloon:语音对话模式',on_click=lambda: VoiChat(selected_kb,kb_top_k,score_threshold,dialogue_mode),help='单击开始录音')
           #else : st.button(':balloon:语音对话模式',on_click=lambda: VoiChat(" ",5,1.0,dialogue_mode),help='单击开始录音')
           #st.caption("Voice Conversation mode")
           st.markdown('<br>' * 10, unsafe_allow_html=True)
           st.write(':balloon:语音对话模式')
           st.caption('Voice Conversation mode')
           Voiceprompt = Voice(st_audiorec())
           if st.session_state.PROMPT == Voiceprompt:
               Voiceprompt =""
    #if Voiceprompt:
        #if dialogue_mode == "知识库问答" :
          #  VoiChat(selected_kb,kb_top_k,score_threshold,dialogue_mode,Voiceprompt)
        #else:
            #VoiChat(" ",5,1.0,dialogue_mode,Voiceprompt)
        def on_kb_change():
            st.toast(f"已加载知识库： {st.session_state.selected_kb}")
        print(dialogue_mode)
        if dialogue_mode == "知识库问答":
                kb_top_k = 5 
                ## Bge 模型会超过1
                score_threshold = 0.9
    # Display chat messages from history on app rerun
    chat_box.output_messages()

    chat_input_placeholder = "请输入对话内容，若使用语音对话请在侧边栏使用。 "

    def on_feedback(
        feedback,
        message_id: str = "",
        history_index: int = -1,
    ):
        reason = feedback["text"]
        score_int = chat_box.set_feedback(
            feedback=feedback, history_index=history_index
        )
        api.chat_feedback(message_id=message_id, score=score_int, reason=reason)
        st.session_state["need_rerun"] = True

    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "欢迎反馈您打分的理由",
    }
    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        if parse_command(text=prompt, modal=modal):  # 用户输入自定义命令
            st.rerun()
        else:
            if st.session_state.Select == None:
               st.toast("请先在左侧菜单栏选择您的身份，再进行提问。期待您对问题回答满意度进行评价，这将有助于我们完善模型！")
            else:
                history = get_messages_history(history_len)
                chat_box.user_say(prompt)
                if dialogue_mode == "LLM 对话":
                    chat_box.ai_say("正在思考...")
                    text = ""
                    message_id = str(random.randint(1, 100000))
                    r = api.chat_chat(
                        prompt,
                        history=history,
                        conversation_id=conversation_id,
                        model=llm_model,
                        prompt_name=prompt_template_name,
                        temperature=0.7,
                    )
                    for t in r:
                        if error_msg := check_error_msg(t):  # check whether error occured
                            st.error(error_msg)
                            break
                        text += t.get("text", "")
                        chat_box.update_msg(text)
                        message_id = t.get("message_id", str(random.randint(1, 100000)))

                    metadata = {
                        "message_id": message_id,
                    }
                    chat_box.update_msg(
                        text, streaming=False, metadata=metadata
                    )  # 更新最终的字符串，去除光标
                    chat_box.show_feedback(
                        **feedback_kwargs,
                        key=message_id,
                        on_submit=on_feedback,
                        kwargs={
                            "message_id": message_id,
                            "history_index": len(chat_box.history) - 1,
                        },
                    )
                elif dialogue_mode == "知识库问答":
                    chat_box.ai_say(
                        [
                            f"正在查询知识库 `{selected_kb}` ...",
                            Markdown(
                                "...", in_expander=True, title="知识库匹配结果", state="complete"
                            ),
                        ]
                    )
                    message_id = str(random.randint(1, 100000))
                    text = ""
                    for d in api.knowledge_base_chat(
                        prompt,
                        knowledge_base_name=selected_kb,
                        top_k=kb_top_k,
                        score_threshold=score_threshold,
                        history=history,
                        model=llm_model,
                        prompt_name=prompt_template_name,
                        temperature=0.7,
                    ):
                        if error_msg := check_error_msg(d):  # check whether error occured
                            st.error(error_msg)
                        elif chunk := d.get("answer"):
                            text += chunk
                            chat_box.update_msg(text, element_index=0)
                            message_id = d.get("message_id", str(random.randint(1, 100000)))
                    chat_box.update_msg(text, element_index=0, streaming=False)
                    chat_box.update_msg(
                        "\n\n".join(d.get("docs", [])), element_index=1, streaming=False
                    )
                    chat_box.show_feedback(
                        **feedback_kwargs,
                        key=message_id,
                        on_submit=on_feedback,
                        kwargs={
                            "message_id": message_id,
                            "history_index": len(chat_box.history) - 1,
                        },
                    ) # 更新最终的字符串，去除光标
    elif Voiceprompt:
        if st.session_state.Select == None:
            st.toast("请先在左侧菜单栏选择您的身份，再进行提问。期待您对问题回答满意度进行评价，这将有助于我们完善模型！")
        else:
            st.session_state.PROMPT = Voiceprompt
            print(st.session_state.PROMPT+'2')
            history = get_messages_history(5)
            chat_box.user_say(Voiceprompt)
            if dialogue_mode == "LLM 对话":
                chat_box.ai_say("正在思考...")
                text = ""
                message_id = str(random.randint(1, 100000))
                r = api.chat_chat(
                    Voiceprompt,
                    history=history,
                    conversation_id=conversation_id,
                    model=llm_model,
                    prompt_name=prompt_template_name,
                    temperature=0.7,
                )
                for t in r:
                    if error_msg := check_error_msg(t):  # check whether error occured
                        st.error(error_msg)
                        break
                    text += t.get("text", "")
                    chat_box.update_msg(text)
                    message_id = t.get("message_id", "")

                metadata = {
                    "message_id": message_id,
                }
                chat_box.update_msg(
                    text, streaming=False, metadata=metadata
                ) 
                chat_box.show_feedback(
                    **feedback_kwargs,
                    key=message_id,
                    on_submit=on_feedback,
                    kwargs={
                        "message_id": message_id,
                        "history_index": len(chat_box.history) - 1,
                    },
                ) # 更新最终的字符串，去除光标
            elif dialogue_mode == "知识库问答":
                chat_box.ai_say(
                    [
                        f"正在查询知识库 `{selected_kb}` ...",
                        Markdown(
                            "...", in_expander=True, title="知识库匹配结果", state="complete"
                        ),
                    ]
                )
                text = ""
                message_id = str(random.randint(1, 100000))
                for d in api.knowledge_base_chat(
                    Voiceprompt,
                    knowledge_base_name=selected_kb,
                    top_k=kb_top_k,
                    score_threshold=score_threshold,
                    history=history,
                    model=llm_model,
                    prompt_name=prompt_template_name,
                    temperature=0.7,
                ):
                    if error_msg := check_error_msg(d):  # check whether error occured
                        st.error(error_msg)
                    elif chunk := d.get("answer"):
                        text += chunk
                        chat_box.update_msg(text, element_index=0)
                        message_id = d.get("message_id", "")
                chat_box.update_msg(text, element_index=0, streaming=False)
                chat_box.update_msg(
                    "\n\n".join(d.get("docs", [])), element_index=1, streaming=False
                )
                chat_box.show_feedback(
                    **feedback_kwargs,
                    key=message_id,
                    on_submit=on_feedback,
                    kwargs={
                        "message_id": message_id,
                        "history_index": len(chat_box.history) - 1,
                    },
                ) # 更新最终的字符串，去除光标
    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()
