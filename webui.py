import streamlit as st
from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages.dialogue.dialogue import dialogue_page, chat_box
from webui_pages.knowledge_base.knowledge_base import knowledge_base_page
import os
import sys
from configs import VERSION
from server.utils import api_address



api = ApiRequest(base_url=api_address())

if __name__ == "__main__":
    st.set_page_config(
        "华电氢小智",
        os.path.join("img", "20190712035319458.jpg"),
        initial_sidebar_state="auto",
        layout="wide",
        
    )
    
    
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit=True
    else:
        st.session_state.first_visit=False
# 初始化全局配置
    if st.session_state.first_visit:
        st.balloons()  #第一次访问时才会放气球
        st.session_state.PROMPT=""
        st.session_state.Select = None
    if not chat_box.chat_inited:
        st.subheader(
            f":blue[Welcome to Hydrogen Chat (氢聊) , 氢聊是一个基于知识库问答的聊天助手，开展氢能科普] :sunglasses: ", divider='rainbow'
        )
    pages = {
        "能源知识库问答": {
            "icon": "chat",
            "func": dialogue_page,
        },
        "知识库管理": {
            "icon": "hdd-stack",
            "func": knowledge_base_page,
        },
    }
    with st.sidebar:
        st.image(
            os.path.join(
                "img",
                "R-C-1.png"
            ),
            use_column_width=True
            )
        st.caption(
            f"""<p align="right">当前版本：{VERSION}</p>""",
            unsafe_allow_html=True,
        )
        st.markdown('<br>' * 3, unsafe_allow_html=True)
        # cols = st.columns(2)
        # if cols[0].button(
        #      "新增会话",
        #     use_container_width=True,
        #      type="primary"
        # ):
        #   pages.setdefault("新会话",{
        #     "icon": "chat",
        #     "func": dialogue_page,
        # })
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]
        default_index = 0
        selected_page =     option_menu(      
          "",
            options=options,
            icons=icons,
             #menu_icon="chat-quote",
           default_index=default_index,
        )  
       
        #"能源知识库问答"
        
        c1,c2,c3 = st.columns([0.15,0.80,0.1])
        with c2:
            option = st.selectbox(
            '选择您的身份',
            options=('学校', '政府', '企业','科研机构'),placeholder="请选择",index=None)
            st.session_state.Select = option
            print(st.session_state.Select)

    if selected_page in pages:
        pages[selected_page]["func"](api)
