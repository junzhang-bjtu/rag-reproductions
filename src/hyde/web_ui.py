from pathlib import Path
from typing import cast
from llama_index.core import Settings
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from config import get_config_from_yaml
from index_builder import get_index_from_nodes
from node_loader import get_nodes_from_markdowns
from rag_system_builder import get_rag_system_from_index

import gradio as gr


WORK_DIR = Path(__file__).parent
CONFIG_PATH = WORK_DIR / "config.yaml"

# 获取配置
config = get_config_from_yaml(config_path=CONFIG_PATH)

# 设置全局 LLM
llm = Ollama(
    base_url=config["ollama"]["base_url"],
    model=config["ollama"]["llm"]["model"],
    request_timeout=config["ollama"]["llm"]["request_timeout"],
    context_window=config["ollama"]["llm"]["context_window"],
    keep_alive=config["ollama"]["llm"]["keep_alive"],
)
Settings.llm = llm

# 设置全局 Embed Model
embed_model = OllamaEmbedding(
    base_url=config["ollama"]["base_url"],
    model_name=config["ollama"]["embedding"]["model_name"],
)
Settings.embed_model = embed_model

# 获取结点
nodes = get_nodes_from_markdowns(
    input_dir=config["paths"]["data_dir"],
    chunk_size=config["text_splitter"]["chunk_size"],
    chunk_overlap=config["text_splitter"]["chunk_overlap"],
)

# 获取索引
index = get_index_from_nodes(nodes, persist_dir=Path(config["paths"]["storage_dir"]))

# 获取 RAG 系统
rag_system = get_rag_system_from_index(
    index=index,
    similarity_top_k=config["rag"]["similarity_top_k"],
    streaming=True,
)


def get_answer(question: str):
    answer = ""
    response = cast(StreamingResponse, rag_system.query(question))
    for part in response.response_gen:
        answer += part
        yield answer


def get_related_docs(question: str):
    query_bundle = QueryBundle(query_str=question)
    related_nodes = rag_system.retrieve(query_bundle)
    related_docs = [
        f"# 文档片段 {idx+1}\n\n" + node.node.get_content()
        for idx, node in enumerate(related_nodes)
    ]
    return "\n\n---\n\n".join(related_docs)


def update_submit_button_state(text: str):
    """根据输入文本更新提交按钮的状态"""
    return gr.Button(interactive=bool(text and text.strip()))


RELATED_DOCS_PLACEHOLDER = """<div style="color: #888; font-style: italic;">
相关文档将在此处显示...
</div>
"""

INTRO = """# 🤖 基于 RAG 的智能问答系统

## 📋 系统介绍

本系统基于 Retrieval-Augmented Generation (RAG) 技术构建，能够：

- 🔍 **智能检索**：从知识库中快速找到相关信息；
- 💬 **精准回答**：结合检索到的信息生成准确答案；
- 📚 **溯源展示**：显示回答所依据的相关文档片段。

---
"""

OUTPUT_PLACEHOLDER = """<div style="color: #888; font-style: italic;">
请在下方输入问题并点击"提交"按钮获取答案...
</div>
"""

INPUT_PLACEHOLDER = "请在此处输入问题..."


if __name__ == "__main__":
    with gr.Blocks(title="智能问答系统") as demo:
        with gr.Sidebar():
            related_docs = gr.Markdown(RELATED_DOCS_PLACEHOLDER)

        with gr.Column():
            title = gr.Markdown(INTRO)
            output = gr.Markdown(OUTPUT_PLACEHOLDER)
            input = gr.Textbox(label="", placeholder=INPUT_PLACEHOLDER, lines=5)

            with gr.Row():
                submit = gr.Button("🚀 提交", interactive=False)  # 默认禁用
                clear = gr.Button("🗑️ 清空")

        # 当输入框内容变化时，更新提交按钮状态
        input.change(fn=update_submit_button_state, inputs=[input], outputs=[submit])

        submit.click(
            fn=get_related_docs,
            inputs=[input],
            outputs=[related_docs],
        ).then(
            fn=get_answer,
            inputs=[input],
            outputs=[output],
        )

        clear.click(
            fn=lambda: (None, OUTPUT_PLACEHOLDER, RELATED_DOCS_PLACEHOLDER),
            inputs=[],
            outputs=[input, output, related_docs],
        )

    demo.launch()
