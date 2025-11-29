from pathlib import Path
from typing import List, Union, cast
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import BaseNode


def get_index_from_nodes(
    nodes: List[BaseNode],
    persist_dir: Union[str, Path],
) -> VectorStoreIndex:
    """
    从结点列表创建或加载向量存储索引

    该函数会检查指定的持久化目录，如果存在已保存的索引则加载它，
    否则基于提供的结点创建新索引并保存到目录中。

    参数:
        nodes: BaseNode 对象列表，用于创建索引的源数据
        persist_dir: 持久化存储目录路径，可以是字符串或 Path 对象

    返回:
        VectorStoreIndex: 向量存储索引对象

    异常:
        ValueError: 当需要创建新索引但结点列表为空时抛出
    """
    # 统一路径格式并验证
    persist_dir = Path(persist_dir)

    # 目录不存在时创建，确保持久化成功
    if not persist_dir.exists():
        persist_dir.mkdir(parents=True, exist_ok=True)

    # 检查持久化目录是否包含有效索引
    if (persist_dir / "docstore.json").exists():
        # 如果 persist_dir 为 Path 实例，则转化为 str 路径
        if isinstance(persist_dir, Path):
            persist_dir = persist_dir.as_posix()

        # 从现有存储加载索引，复用已生成的向量
        print(f"[INFO] 已找到已保存的索引，从 {persist_dir} 加载中...")

        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        
        index = cast(
            VectorStoreIndex,
            load_index_from_storage(storage_context=storage_context),
        )
        
        print(
            f"[INFO] 索引已加载完成，已找到 {len(index.docstore.get_all_document_hashes())} 个结点"
        )
    else:
        # 创建新索引，需要验证输入数据有效性
        if not nodes:
            raise ValueError("创建索引时结点列表不能为空")
        
        print(f"[INFO] 正在创建索引，请稍候...")

        index = VectorStoreIndex(nodes=nodes, show_progress=True)
        
        print(
            f"[INFO] 已创建索引，已找到 {len(index.docstore.get_all_document_hashes())} 个结点"
        )

        # 持久化索引，避免重复计算嵌入向量
        index.storage_context.persist(persist_dir=persist_dir)
        
        print(f"[INFO] 索引已保存到 {persist_dir}")

    return index
