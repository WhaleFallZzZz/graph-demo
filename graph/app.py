import streamlit as st
import time

# 引入后端引擎
try:
    from rag_pro import Neo4jGraphRAG
except ImportError:
    st.error("❌ 找不到 rag_pro.py！请确保该文件存在。")
    st.stop()

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="眼科视光 AI 专家 Pro",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 核心：整容级 CSS 修复 ---
# 重点解决了你截图中的“文字重叠”和“字体乱码”问题
st.markdown("""
<style>
    /* 1. 隐藏加载失败的 Material Icons 文字 (解决 keyboard_arrow_right 乱码) */
    .st-emotion-cache-1wbqy5l, .material-icons, .icon-button {
        font-family: sans-serif !important; 
        font-size: 0px !important; /* 字体加载失败时，把乱码文字缩放到0看不到 */
    }
    /* 重新定义 expander 的箭头，防止重叠 */
    div[data-testid="stExpander"] summary span {
        font-size: 1rem !important;
    }

    /* 2. 聊天气泡样式优化 */
    .stChatMessage {
        background-color: transparent;
        border-radius: 10px;
        padding: 10px;
    }

    /* 3. 隐藏右上角默认菜单和红线 */
    header {visibility: hidden;}
    .stDeployButton {display:none;}

    /* 4. 调整主标题颜色 */
    h1 {
        color: #0083B8; /* 医疗蓝 */
    }
</style>
""", unsafe_allow_html=True)


# --- 3. 核心引擎加载 ---
@st.cache_resource
def get_rag_engine():
    try:
        return Neo4jGraphRAG()
    except Exception as e:
        return None


rag_engine = get_rag_engine()

# --- 4. 侧边栏 ---
with st.sidebar:
    st.title("👁️ 控制面板")
    st.caption("Ver 2.0 ")

    st.markdown("---")

    # 这里的图标我们换成 Emoji，防止再次出现乱码
    with st.expander("🕸️ 查看图谱结构 (Schema)"):
        if rag_engine:
            st.code(rag_engine.schema_str, language="text")
        else:
            st.error("数据库未连接")

    st.markdown("### 🛠️ 调试选项")
    show_cypher = st.toggle("显示 Cypher 语句", value=True)
    show_raw_data = st.toggle("显示原始数据", value=False)

    st.markdown("---")
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 5. 主界面 ---
# 使用 Columns 让标题布局更紧凑
col1, col2 = st.columns([1, 12])
with col1:
    st.image("https://img.icons8.com/color/96/ophthalmology.png", width=60)
with col2:
    st.title("眼科视光 AI 专家助手")

st.divider()

# --- 6. 初始化历史 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 7. 渲染历史消息 (关键修改：修复头像) ---
for message in st.session_state.messages:
    # ⚠️ 关键点：这里绝对不能用 "face" 这种字符串，必须用 Emoji
    if message["role"] == "assistant":
        avatar_icon = "🩺"  # 医生听诊器 Emoji
    else:
        avatar_icon = "👤"  # 用户人像 Emoji

    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

# --- 8. 交互区域 ---
if prompt := st.chat_input("请描述您的眼科问题..."):

    # A. 用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    # ⚠️ 同样修复这里的头像
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # B. AI 回复
    if rag_engine:
        with st.chat_message("assistant", avatar="🩺"):
            message_placeholder = st.empty()

            with st.status("🧠 正在分析病例与检索知识库...", expanded=True) as status:

                # 1. 生成 Cypher
                st.write("🔍 分析意图...")
                cypher_query = rag_engine.text_to_cypher(prompt)

                if show_cypher and cypher_query:
                    st.info("生成的查询语句:")
                    st.code(cypher_query, language="cypher")

                # 2. 执行查询
                st.write("💾 查询数据库...")
                db_results = rag_engine.execute_cypher(cypher_query)

                if show_raw_data:
                    with st.expander("查看原始数据"):
                        st.write(db_results)

                # 更新状态
                if db_results:
                    status.update(label="✅ 检索成功", state="complete", expanded=False)
                else:
                    status.update(label="⚠️ 未找到关联数据", state="complete", expanded=False)

            # 3. 生成回答
            final_answer = rag_engine.generate_answer(prompt, db_results)
            message_placeholder.markdown(final_answer)

            st.session_state.messages.append({"role": "assistant", "content": final_answer})
    else:
        st.error("数据库连接失败，无法回答。")