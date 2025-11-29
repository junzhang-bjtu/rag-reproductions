from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine


def get_rag_system_from_index(
    index: VectorStoreIndex, similarity_top_k: int, streaming: bool
) -> BaseQueryEngine:
    """
    根据向量索引创建 RAG 问答系统查询引擎

    参数:
        index (VectorStoreIndex): 向量存储索引对象，用于文档检索
        similarity_top_k (int): 相似度检索返回的 top-k 文档数量，必须为正整数
        streaming (bool): 是否启用流式响应模式

    返回:
        BaseQueryEngine: 配置好的查询引擎对象，可用于 RAG 问答

    异常:
        ValueError: 当 similarity_top_k 不是正整数时抛出
    """
    # 验证检索参数合理性
    if not isinstance(similarity_top_k, int) or similarity_top_k <= 0:
        raise ValueError(f"similarity_top_k 必须为正整数，当前值: {similarity_top_k}")

    # 将索引转换为查询引擎，配置检索参数和语言模型
    print(f"[INFO] 正在创建 RAG 系统...")

    rag_system = index.as_query_engine(
        similarity_top_k=similarity_top_k,
        streaming=streaming,
    )
    hyde = HyDEQueryTransform(include_original=False)
    rag_system = TransformQueryEngine(rag_system, hyde)
    
    print(f"[INFO] 创建 RAG 系统成功，similarity_top_k: {similarity_top_k}")

    return rag_system
