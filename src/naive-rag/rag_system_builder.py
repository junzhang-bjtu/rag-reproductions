from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine


def get_rag_system_from_index(
    index: VectorStoreIndex, similarity_top_k: int, streaming: bool
) -> BaseQueryEngine:
    """
    根据向量索引创建RAG问答系统查询引擎

    参数:
        index (VectorStoreIndex): 向量存储索引对象，用于文档检索
        similarity_top_k (int): 相似度检索返回的top-k文档数量，必须为正整数
        streaming (bool): 是否启用流式响应模式

    返回:
        BaseQueryEngine: 配置好的查询引擎对象，可用于RAG问答

    异常:
        ValueError: 当similarity_top_k不是正整数时抛出
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
    
    print(f"[INFO] 创建 RAG 系统成功，similarity_top_k: {similarity_top_k}")

    return rag_system
