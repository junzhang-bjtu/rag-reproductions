from pathlib import Path
from typing import List, Union
from tqdm import tqdm
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode


def get_nodes_from_markdowns(
    input_dir: Union[str, Path], chunk_size: int = 512, chunk_overlap: int = 50
) -> List[BaseNode]:
    """
    从指定目录下的 Markdown 文件中加载文档，并将其解析为结构化的结点列表。

    该函数首先使用 SimpleDirectoryReader 加载所有 Markdown 文件，
    然后通过 MarkdownNodeParser 按照标题层级进行初步切分，
    最后再利用 SentenceSplitter 进一步将大段内容按句子边界分割成较小的文本块。

    参数:
        input_dir (Union[str, Path]): 包含 Markdown 文件的文件夹路径。
        chunk_size (int): 文本分块的目标大小（单位：字符），默认为 512。
        chunk_overlap (int): 分块之间的重叠长度（单位：字符），默认为 50。

    返回:
        List[BaseNode]: 解析并分割后的结点对象列表，每个结点代表一个文本片段及其元数据。

    异常:
        FileNotFoundError: 当提供的路径不存在时抛出。
        NotADirectoryError: 当提供的路径不是一个有效目录时抛出。
        ValueError: 当 chunk_size 不是正数或 chunk_overlap 是负数时抛出。
    """

    # 将输入路径统一转换为 Path 对象，便于后续操作
    input_dir = Path(input_dir)

    # 验证文件夹是否存在，提前暴露路径错误
    if not input_dir.exists():
        raise FileNotFoundError(f"存放 Markdown 文件的文件夹未找到：{input_dir}")

    # 验证路径是否为文件夹
    if not input_dir.is_dir():
        raise NotADirectoryError(f"存放 Markdown 文件的文件夹路径无效：{input_dir}")

    # 检查文件夹是否为空
    if not any(input_dir.iterdir()):
        raise ValueError(f"存放 Markdown 文件的文件夹为空：{input_dir}")

    # 参数验证
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正数，当前值: {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap 必须为非负数，当前值: {chunk_overlap}")

    # 1. 加载目录下所有 Markdown 文件为 Document 对象
    # SimpleDirectoryReader 会自动读取文件内容并提取基础元数据（如文件名）
    documents: List[Document] = SimpleDirectoryReader(
        input_dir=input_dir, required_exts=[".md"]
    ).load_data()

    print(f"[INFO] 已加载 {len(documents)} 个 Markdown 文件")

    # 2. 按 Markdown 标题层级进行初步分割
    # MarkdownNodeParser 会识别 #、## 等标题，创建以标题为边界的逻辑块
    print("[INFO] 正在按标题进行初步分割...")

    markdown_parser: MarkdownNodeParser = MarkdownNodeParser()
    initial_nodes: List[BaseNode] = markdown_parser.get_nodes_from_documents(
        documents, show_progress=True
    )

    print(f"[INFO] 获取 {len(initial_nodes)} 个初始结点")

    # 3. 初始化句子分割器，用于进一步细分大文本块
    # SentenceSplitter 会优先在句子边界分割，保持语义完整性
    sentence_splitter: SentenceSplitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # 4. 存储最终结果的列表
    final_nodes: List[BaseNode] = []

    # 遍历每个初始结点，进行递归分割
    print("[INFO] 正在递归分割文本块...")

    for node in tqdm(initial_nodes, desc="递归分割文本块"):
        # 4.1 将结点内容包装为 Document 对象，复制元数据避免污染原始数据
        sub_doc: Document = Document(
            text=node.get_content(), metadata=node.metadata.copy()
        )

        # 4.2 对当前结点进行递归分割，生成更小的子结点
        sub_nodes: List[BaseNode] = sentence_splitter.get_nodes_from_documents(
            [sub_doc]
        )

        # 4.3 将原始结点的元数据合并到子结点
        # 合并策略：原始结点元数据为基础，子结点元数据优先（如 chunk_id）
        for sub_node in sub_nodes:
            merged_metadata = {**node.metadata, **sub_node.metadata}
            sub_node.metadata = merged_metadata

        # 4.4 将处理好的子结点添加到最终结果
        final_nodes.extend(sub_nodes)

    print(f"[INFO] 获取 {len(final_nodes)} 个最终结点")

    return final_nodes
