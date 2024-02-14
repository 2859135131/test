import os

# 可以指定一个绝对路径，统一存放所有的Embedding和LLM模型。
# 每个模型可以是一个单独的目录，也可以是某个目录下的二级子目录。
# 如果模型目录名称和 MODEL_PATH 中的 key 或 value 相同，程序会自动检测加载，无需修改 MODEL_PATH 中的路径。 "/root/HydrogenChat/Langchain-Chatchat/models"
MODEL_ROOT_PATH ="F://BaiduNetdiskDownload//Langchain-Chatchat//models"

# 选用的 Embedding 名称
EMBEDDING_MODEL = "m3e-base"

# Embedding 模型运行设备。设为"auto"会自动检测，也可手动设定为"cuda","mps","cpu"其中之一。
EMBEDDING_DEVICE = "cpu"

# 如果需要在 EMBEDDING_MODEL 中增加自定义的关键字时配置
EMBEDDING_KEYWORD_FILE = "keywords.txt"
EMBEDDING_MODEL_OUTPUT_PATH = "output"

# 要运行的 LLM 名称，可以包括本地模型和在线模型。列表中本地模型将在启动项目时全部加载。
# 列表中第一个模型将作为 API 和 WEBUI 的默认模型。
# 在这里，我们使用目前主流的两个离线模型，其中，chatglm3-6b 为默认加载模型。
# 如果你的显存不足，可使用 Qwen-1_8B-Chat, 该模型 FP16 仅需 3.8G显存。
LLM_MODELS = ["qwen-api"] # "Qwen-1_8B-Chat",

# AgentLM模型的名称 (可以不指定，指定之后就锁定进入Agent之后的Chain的模型，不指定就是LLM_MODELS[0])
Agent_MODEL = None

# LLM 运行设备。设为"auto"会自动检测，也可手动设定为"cuda","mps","cpu"其中之一。
LLM_DEVICE = "cpu"

# 历史对话轮数
HISTORY_LEN = 3

# 大模型最长支持的长度，如果不填写，则使用模型默认的最大长度，如果填写，则为用户设定的最大长度
MAX_TOKENS = None

# LLM通用对话参数
TEMPERATURE = 0.7
# TOP_P = 0.95 # ChatOpenAI暂不支持该参数

ONLINE_LLM_MODEL = {
    # 线上模型。请在server_config中为每个在线API设置不同的端口

    "openai-api": {
        "model_name": "gpt-3.5-turbo",
        "api_base_url": "https://api.openai.com/v1",
        "api_key": "",
        "openai_proxy": "",
    },

    # 具体注册及api key获取请前往 http://open.bigmodel.cn
    "zhipu-api": {
        "api_key": "",
        "version": "chatglm_turbo",  # 可选包括 "chatglm_turbo"
        "provider": "ChatGLMWorker",
    },


    # 具体注册及api key获取请前往 https://api.minimax.chat/
    "minimax-api": {
        "group_id": "",
        "api_key": "",
        "is_pro": False,
        "provider": "MiniMaxWorker",
    },


    # 具体注册及api key获取请前往 https://xinghuo.xfyun.cn/
    "xinghuo-api": {
        "APPID": "",
        "APISecret": "",
        "api_key": "",
        "version": "v1.5",  # 你使用的讯飞星火大模型版本，可选包括 "v3.0", "v1.5", "v2.0"
        "provider": "XingHuoWorker",
    },

    # 百度千帆 API，申请方式请参考 https://cloud.baidu.com/doc/WENXINWORKSHOP/s/4lilb2lpf
    "qianfan-api": {
        "version": "ERNIE-Bot",  # 注意大小写。当前支持 "ERNIE-Bot" 或 "ERNIE-Bot-turbo"， 更多的见官方文档。
        "version_url": "",  # 也可以不填写version，直接填写在千帆申请模型发布的API地址
        "api_key": "",
        "secret_key": "",
        "provider": "QianFanWorker",
    },

    # 火山方舟 API，文档参考 https://www.volcengine.com/docs/82379
    "fangzhou-api": {
        "version": "chatglm-6b-model",  # 当前支持 "chatglm-6b-model"， 更多的见文档模型支持列表中方舟部分。
        "version_url": "",  # 可以不填写version，直接填写在方舟申请模型发布的API地址
        "api_key": "",
        "secret_key": "",
        "provider": "FangZhouWorker",
    },

    # 阿里云通义千问 API，文档参考 https://help.aliyun.com/zh/dashscope/developer-reference/api-details
    "qwen-api": {
        "version": "qwen-max",  # 可选包括 "qwen-max"
        "api_key": "sk-403a728c6ae746dab1f7fd9d289ef8ff",  # 请在阿里云控制台模型服务灵积API-KEY管理页面创建
        "provider": "QwenWorker",
    },

    # 百川 API，申请方式请参考 https://www.baichuan-ai.com/home#api-enter
    "baichuan-api": {
        "version": "Baichuan2-53B",  # 当前支持 "Baichuan2-53B"， 见官方文档。
        "api_key": "",
        "secret_key": "",
        "provider": "BaiChuanWorker",
    },

    # Azure API
    "azure-api": {
        "deployment_name": "",  # 部署容器的名字
        "resource_name": "",  # https://{resource_name}.openai.azure.com/openai/ 填写resource_name的部分，其他部分不要填写
        "api_version": "",  # API的版本，不是模型版本
        "api_key": "",
        "provider": "AzureWorker",
    },

    # 昆仑万维天工 API https://model-platform.tiangong.cn/
    "tiangong-api": {
        "version": "SkyChat-MegaVerse",
        "api_key": "",
        "secret_key": "",
        "provider": "TianGongWorker",
    },

}

# 在以下字典中修改属性值，以指定本地embedding模型存储位置。支持3种设置方法：
# 1、将对应的值修改为模型绝对路径
# 2、不修改此处的值（以 text2vec 为例）：
#       2.1 如果{MODEL_ROOT_PATH}下存在如下任一子目录：
#           - text2vec
#           - GanymedeNil/text2vec-large-chinese
#           - text2vec-large-chinese
#       2.2 如果以上本地路径不存在，则使用huggingface模型
MODEL_PATH = {
    "embed_model": {
        "m3e-base": "Embedding/m3e-base",
    },

    "llm_model": {
        # 以下部分模型并未完全测试，仅根据fastchat和vllm模型的模型列表推定支持
        "chatglm2-6b": "LLM/chatglm2-6b",
        "chatglm2-6b-32k": "LLM/chatglm2-6b-32k",

        "chatglm3-6b": "LLM/chatglm3-6b",
        "chatglm3-6b-int4": "LLM/chatglm3-6b-int4",
        "chatglm3-6b-32k": "LLM/chatglm3-6b-32k",
        "chatglm3-6b-base": "LLM/chatglm3-6b-base",

        "Qwen-1_8B": "LLM/Qwen-1_8B",
        "Qwen-1_8B-Chat": "LLM/Qwen-1_8B-Chat",
        "Qwen-1_8B-Chat-Int8": "LLM/Qwen-1_8B-Chat-Int8",
        "Qwen-1_8B-Chat-Int4": "LLM/Qwen-1_8B-Chat-Int4",

        "Qwen-7B": "LLM/Qwen-7B",
        "Qwen-7B-Chat": "LLM/Qwen-7B-Chat",

        "Qwen-14B": "LLM/Qwen-14B",
        "Qwen-14B-Chat": "LLM/Qwen-14B-Chat",
        "Qwen-14B-Chat-Int8": "LLM/Qwen-14B-Chat-Int8",
        "Qwen-14B-Chat-Int4": "LLM/Qwen-14B-Chat-Int4",

        "Qwen-72B": "LLM/Qwen-72B",
        "Qwen-72B-Chat": "LLM/Qwen-72B-Chat",
        "Qwen-72B-Chat-Int8": "LLM/Qwen-72B-Chat-Int8",
        "Qwen-72B-Chat-Int4": "LLM/Qwen-72B-Chat-Int4",

        "baichuan2-13b": "LLM/Baichuan2-13B-Chat",
        "baichuan2-7b": "LLM/Baichuan2-7B-Chat",

        "baichuan-7b": "LLM/Baichuan-7B",
        "baichuan-13b": "LLM/Baichuan-13B",
        "baichuan-13b-chat": "LLM/Baichuan-13B-Chat",

        "aquila-7b": "LLM/Aquila-7B",
        "aquilachat-7b": "LLM/AquilaChat-7B",

        "internlm-7b": "LLM/internlm-7b",
        "internlm-chat-7b": "LLM/internlm-chat-7b",

        "falcon-7b": "LLM/falcon-7b",
        "falcon-40b": "LLM/falcon-40b",
        "falcon-rw-7b": "LLM/falcon-rw-7b",

        "gpt2": "gpt2",
        "gpt2-xl": "gpt2-xl",

        "gpt-j-6b": "EleutherAI/gpt-j-6b",
        "gpt4all-j": "nomic-ai/gpt4all-j",
        "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
        "pythia-12b": "EleutherAI/pythia-12b",
        "oasst-sft-4-pythia-12b-epoch-3.5": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
        "dolly-v2-12b": "databricks/dolly-v2-12b",
        "stablelm-tuned-alpha-7b": "stabilityai/stablelm-tuned-alpha-7b",

        "Llama-2-13b-hf": "LLM/Llama-2-13b-hf",
        "Llama-2-70b-hf": "LLM/Llama-2-70b-hf",
        "open_llama_13b": "LLM/open_llama_13b",
        "vicuna-13b-v1.3": "LLM/vicuna-13b-v1.3",
        "koala": "young-geng/koala",

        "mpt-7b": "LLM/mpt-7b",
        "mpt-7b-storywriter": "LLM/mpt-7b-storywriter",
        "mpt-30b": "LLM/mpt-30b",
        "opt-66b": "LLM/opt-66b",
        "opt-iml-max-30b": "LLM/opt-iml-max-30b",

        "agentlm-7b": "LLM/agentlm-7b",
        "agentlm-13b": "LLM/agentlm-13b",
        "agentlm-70b": "LLM/agentlm-70b",

        "Yi-34B-Chat": "LLM/Yi-34B-Chat",
    },
}


# 通常情况下不需要更改以下内容

# nltk 模型存储路径
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

VLLM_MODEL_DICT = {
    "aquila-7b": "BAAI/Aquila-7B",
    "aquilachat-7b": "BAAI/AquilaChat-7B",

    "baichuan-7b": "baichuan-inc/Baichuan-7B",
    "baichuan-13b": "baichuan-inc/Baichuan-13B",
    "baichuan-13b-chat": "baichuan-inc/Baichuan-13B-Chat",

    "chatglm2-6b": "THUDM/chatglm2-6b",
    "chatglm2-6b-32k": "THUDM/chatglm2-6b-32k",
    "chatglm3-6b": "THUDM/chatglm3-6b",
    "chatglm3-6b-32k": "THUDM/chatglm3-6b-32k",

    "BlueLM-7B-Chat": "vivo-ai/BlueLM-7B-Chat",
    "BlueLM-7B-Chat-32k": "vivo-ai/BlueLM-7B-Chat-32k",

    # 注意：bloom系列的tokenizer与model是分离的，因此虽然vllm支持，但与fschat框架不兼容
    # "bloom": "bigscience/bloom",
    # "bloomz": "bigscience/bloomz",
    # "bloomz-560m": "bigscience/bloomz-560m",
    # "bloomz-7b1": "bigscience/bloomz-7b1",
    # "bloomz-1b7": "bigscience/bloomz-1b7",

    "internlm-7b": "internlm/internlm-7b",
    "internlm-chat-7b": "internlm/internlm-chat-7b",
    "falcon-7b": "tiiuae/falcon-7b",
    "falcon-40b": "tiiuae/falcon-40b",
    "falcon-rw-7b": "tiiuae/falcon-rw-7b",
    "gpt2": "gpt2",
    "gpt2-xl": "gpt2-xl",
    "gpt-j-6b": "EleutherAI/gpt-j-6b",
    "gpt4all-j": "nomic-ai/gpt4all-j",
    "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
    "pythia-12b": "EleutherAI/pythia-12b",
    "oasst-sft-4-pythia-12b-epoch-3.5": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    "dolly-v2-12b": "databricks/dolly-v2-12b",
    "stablelm-tuned-alpha-7b": "stabilityai/stablelm-tuned-alpha-7b",
    "Llama-2-13b-hf": "meta-llama/Llama-2-13b-hf",
    "Llama-2-70b-hf": "meta-llama/Llama-2-70b-hf",
    "open_llama_13b": "openlm-research/open_llama_13b",
    "vicuna-13b-v1.3": "lmsys/vicuna-13b-v1.3",
    "koala": "young-geng/koala",
    "mpt-7b": "mosaicml/mpt-7b",
    "mpt-7b-storywriter": "mosaicml/mpt-7b-storywriter",
    "mpt-30b": "mosaicml/mpt-30b",
    "opt-66b": "facebook/opt-66b",
    "opt-iml-max-30b": "facebook/opt-iml-max-30b",

    "Qwen-1_8B": "Qwen/Qwen-1_8B",
    "Qwen-1_8B-Chat": "Qwen/Qwen-1_8B-Chat",
    "Qwen-1_8B-Chat-Int8": "Qwen/Qwen-1_8B-Chat-Int8",
    "Qwen-1_8B-Chat-Int4": "Qwen/Qwen-1_8B-Chat-Int4",

    "Qwen-7B": "Qwen/Qwen-7B",
    "Qwen-7B-Chat": "Qwen/Qwen-7B-Chat",

    "Qwen-14B": "Qwen/Qwen-14B",
    "Qwen-14B-Chat": "Qwen/Qwen-14B-Chat",
    "Qwen-14B-Chat-Int8": "Qwen/Qwen-14B-Chat-Int8",
    "Qwen-14B-Chat-Int4": "Qwen/Qwen-14B-Chat-Int4",

    "Qwen-72B": "Qwen/Qwen-72B",
    "Qwen-72B-Chat": "Qwen/Qwen-72B-Chat",
    "Qwen-72B-Chat-Int8": "Qwen/Qwen-72B-Chat-Int8",
    "Qwen-72B-Chat-Int4": "Qwen/Qwen-72B-Chat-Int4",

    "agentlm-7b": "THUDM/agentlm-7b",
    "agentlm-13b": "THUDM/agentlm-13b",
    "agentlm-70b": "THUDM/agentlm-70b",

}

# 你认为支持Agent能力的模型，可以在这里添加，添加后不会出现可视化界面的警告
# 经过我们测试，原生支持Agent的模型仅有以下几个
SUPPORT_AGENT_MODEL = [
    "azure-api",
    "openai-api",
    "qwen-api",
    "Qwen",
    "chatglm3",
    "xinghuo-api",
]