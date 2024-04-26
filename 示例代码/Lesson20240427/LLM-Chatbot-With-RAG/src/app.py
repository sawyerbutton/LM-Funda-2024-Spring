# 导入必要的库和模块
import os
import streamlit as st
from model import ChatModel
import rag_util

# 设置文件目录
FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)


# Streamlit界面设置
st.title("LLM Chatbot RAG Assistant")

# 装饰器用于缓存加载的模型
@st.cache_resource
def load_model():
    model = ChatModel(model_id="google/gemma-2b-it", device="cuda")
    return model


@st.cache_resource
def load_encoder():
    encoder = rag_util.Encoder(
        model_name="sentence-transformers/all-MiniLM-L12-v2", device="cpu"
    )
    return encoder


model = load_model()  # load our models once and then cache it
encoder = load_encoder()

# 文件上传与保存
def save_file(uploaded_file):
    """helper function to save documents to disk"""
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# 边栏输入和文件上传处理
with st.sidebar:
    max_new_tokens = st.number_input("max_new_tokens", 128, 4096, 512)
    # 选择最相似的文档数量
    k = st.number_input("k", 1, 10, 3)
    uploaded_files = st.file_uploader(
        "Upload PDFs for context", type=["PDF", "pdf"], accept_multiple_files=True
    )
    file_paths = []
    for uploaded_file in uploaded_files:
        file_paths.append(save_file(uploaded_file))
    if uploaded_files != []:
        docs = rag_util.load_and_split_pdfs(file_paths)
        DB = rag_util.FaissDb(docs=docs, embedding_function=encoder.embedding_function)

# 聊天消息处理与显示
# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 重新运行应用时显示聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接受用户输入信息
if prompt := st.chat_input("Ask me anything!"):
    # 将用户信息添加到聊天记录中
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 在聊天信息容器中展示用户信息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 将助手的回复信息添加到聊天记录中
    with st.chat_message("assistant"):
        user_prompt = st.session_state.messages[-1]["content"]
        context = (
            None if uploaded_files == [] else DB.similarity_search(user_prompt, k=k)
        )
        answer = model.generate(
            user_prompt, context=context, max_new_tokens=max_new_tokens
        )
        response = st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
