from pathlib import Path
from typing import Dict, List, Union, cast
from tqdm import tqdm
from llama_index.core import Settings
from llama_index.core.base.response.schema import Response
from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    FaithfulnessEvaluator,
    ContextRelevancyEvaluator,
    AnswerRelevancyEvaluator,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from config import get_config_from_yaml
from index_builder import get_index_from_nodes
from node_loader import get_nodes_from_markdowns
from rag_system_builder import get_rag_system_from_index

import csv


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

# 设置评估 LLM
eval_llm = Ollama(
    base_url=config["ollama"]["base_url"],
    model=config["ollama"]["eval_llm"]["model"],
    request_timeout=config["ollama"]["eval_llm"]["request_timeout"],
    context_window=config["ollama"]["eval_llm"]["context_window"],
    keep_alive=config["ollama"]["eval_llm"]["keep_alive"],
)

# 设置全局 Embed Model
embed_model = OllamaEmbedding(
    base_url=config["ollama"]["base_url"],
    model_name=config["ollama"]["embedding"]["model_name"],
)
Settings.embed_model = embed_model

# 设置评估器
correctness_evaluator = CorrectnessEvaluator(llm=eval_llm)
semantic_similarity_evaluator = SemanticSimilarityEvaluator(embed_model=embed_model)
faithfulness_evaluator = FaithfulnessEvaluator(llm=eval_llm)
context_relevancy_evaluator = ContextRelevancyEvaluator(llm=eval_llm)
answer_relevancy_evaluator = AnswerRelevancyEvaluator(llm=eval_llm)


SYSTEM_PROMPT_WITHOUT_RAG = """You are a precise and honest AI assistant. Your task is to answer questions based solely on your internal knowledge.

- If you know the answer, provide a concise and factual response.
- If you are uncertain or do not know the answer, respond only with: "I don't know."
- Do not speculate, make up information, or generate content beyond what is necessary to answer the question.
- Stop generating immediately after giving your answer. Do not add explanations, apologies, or extra text.
"""


def evaluation_without_rag(
    dataset_path: Union[str, Path], result_path: Union[str, Path]
) -> None:
    """
    在不使用检索增强生成（RAG）的情况下对问答模型进行评估。

    参数:
        dataset_path (Union[str, Path]): 包含评估所需资源的数据集根目录路径。该目录应包含以下子结构：
            - <dataset_name>.csv: 包含评估问题和参考答案的 CSV 文件，需包含 'input' 和 'answers' 字段。
        result_path (Union[str, Path]): 用于保存评估结果的目录路径。如果该路径不存在，将自动创建。

    返回:
        None: 本函数无返回值，仅将评估结果写入指定目录下的 CSV 文件中。
    """

    # 验证并标准化输入路径
    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise ValueError(f"路径不存在或不是目录: {dataset_path}")

    # 确保输出目录存在
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    # 构建 QA 文件路径
    qa_file: Path = dataset_path / f"{dataset_path.name}.csv"

    # 验证 QA 文件存在
    if not qa_file.exists():
        raise ValueError(f"QA 评估文件不存在: {qa_file}")

    # 确保输出目录存在
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    # 加载 QA 评估集（CSV 格式：input 列存储问题，answers 列存储参考答案）
    print(f"[INFO] 正在加载 QA 文件：{qa_file}")

    questions: List[str] = []
    answers: List[str] = []

    try:
        with open(qa_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"QA 文件格式错误，没有发现表头：{qa_file} ")
            elif "input" not in reader.fieldnames or "answers" not in reader.fieldnames:
                raise ValueError(
                    f"QA 文件格式错误，必须包含 'input' 和 'answers' 列: {qa_file}"
                )
            for row in reader:
                questions.append(row["input"])
                answers.append(row["answers"])
    except Exception as e:
        raise ValueError(f"读取 QA 文件失败 {qa_file}: {str(e)}")

    # 空的评估集触发异常
    if not questions:
        raise ValueError(f"{qa_file} 没有评估问题")

    print(f"[INFO] QA 文件加载成功，读取 {len(questions)} 个问题")

    # 初始化变量以收集查询、响应及参考答案
    queries: List[str] = []
    responses: List[str] = []
    references: List[str] = []

    # 初始化各项评估指标的分数列表
    correctness_scores: List[float] = []
    semantic_similarity_scores: List[float] = []
    answer_relevancy_scores: List[float] = []

    # 对每个问题调用 LLM 模型获取响应，并计算三个维度的评分
    print(f"[INFO] 无需构建索引与 RAG 系统，直接调用 LLM 模型进行评估")
    print(f"[INFO] 开始在 {qa_file} 上进行评估...")

    pbar = tqdm(total=len(questions), desc="评估进度")

    system_prompt = ChatMessage(role="system", content=SYSTEM_PROMPT_WITHOUT_RAG)

    for query, reference in zip(questions, answers):
        messages = [system_prompt, ChatMessage(role="user", content=query)]
        response = llm.chat(messages=messages).message.content
        if response is None:
            response = ""
        queries.append(query)
        responses.append(response)
        references.append(reference)

        # 计算正确性得分
        correctness_score = correctness_evaluator.evaluate(
            query=query,
            response=response,
            reference=reference,
        ).score
        if correctness_score is None:
            correctness_score = 0.0
        correctness_score = max(0.0, min(5.0, correctness_score))
        correctness_scores.append(correctness_score)

        # 计算语义相似度得分
        semantic_similarity_score = semantic_similarity_evaluator.evaluate(
            query=query,
            response=response,
            reference=reference,
        ).score
        if semantic_similarity_score is None:
            semantic_similarity_score = 0.0
        semantic_similarity_score = max(0.0, min(1.0, semantic_similarity_score))
        semantic_similarity_scores.append(semantic_similarity_score)

        # 计算回答相关性得分
        answer_relevancy_score = answer_relevancy_evaluator.evaluate(
            query=query,
            response=response,
        ).score
        if answer_relevancy_score is None:
            answer_relevancy_score = 0.0
        answer_relevancy_score = max(0.0, min(1.0, answer_relevancy_score))
        answer_relevancy_scores.append(answer_relevancy_score)

        pbar.update(1)

    pbar.close()

    # 保存评估详细结果（六列：问题、响应、参考答案、正确性分数、语义相似度分数、回答相关性分数）
    print(f"[INFO] 正在保存评估详细结果...")

    details_path: Path = result_path / "eval_details.csv"
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "query",
                "response",
                "reference",
                "correctness_score",
                "semantic_similarity_score",
                "answer_relevancy_score",
            ]
        )
        writer.writerows(
            zip(
                queries,
                responses,
                references,
                correctness_scores,
                semantic_similarity_scores,
                answer_relevancy_scores,
            )
        )

    print(f"[INFO] 评估详细结果已保存至 {details_path}")

    # 计算最终分数（每项分数都缩放到 0-100）
    print(f"[INFO] 正在计算最终分数...")

    final_correctness_score: float = (
        sum(correctness_scores) / (5.0 * len(correctness_scores))
    ) * 100
    final_semantic_similarity_score: float = (
        sum(semantic_similarity_scores) / len(semantic_similarity_scores)
    ) * 100
    final_answer_relevancy_score: float = (
        sum(answer_relevancy_scores) / len(answer_relevancy_scores)
    ) * 100
    avg_score: float = (
        final_correctness_score
        + final_semantic_similarity_score
        + final_answer_relevancy_score
    ) / 3.0

    print(
        f"[INFO] 平均得分为：{avg_score:.2f}，各项具体得分为：\n"
        f"- correctness_score = {final_correctness_score:.2f}\n"
        f"- semantic_similarity_score = {final_semantic_similarity_score:.2f}\n"
        f"- answer_relevancy_score = {final_answer_relevancy_score:.2f}"
    )

    # 准备汇总结果
    results: List[Dict[str, Union[str, float]]] = [
        {
            "dataset": qa_file.name,
            "correctness_score": final_correctness_score,
            "semantic_similarity_score": final_semantic_similarity_score,
            "answer_relevancy_score": final_answer_relevancy_score,
            "avg_score": avg_score,
        }
    ]

    # 将最终评估结果保存至 CSV 文件
    print(f"[INFO] 正在保存最终评估结果...")

    summary_path: Path = result_path / "eval_scores.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "correctness_score",
                "semantic_similarity_score",
                "answer_relevancy_score",
                "avg_score",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(
        f"[INFO] 在 {qa_file.name} 上的评估完成，最终评估结果保存在 {summary_path} 中"
    )


def evaluation_with_rag(
    dataset_path: Union[str, Path], result_path: Union[str, Path]
) -> None:
    """
    在指定数据集上运行完整的 RAG 评估流程，包括加载 QA 数据、构建索引、执行查询并评估结果。

    参数:
        dataset_path (Union[str, Path]): 包含评估所需资源的数据集根目录路径。该目录应包含以下子结构：
            - data/: 存放用于构建知识库的 Markdown 文档；
            - storage/: 用于持久化向量索引的目录；
            - <dataset_name>.csv: 包含评估问题和参考答案的 CSV 文件，需包含 'input' 和 'answers' 字段。
        result_path (Union[str, Path]): 输出评估结果的目录路径。若目录不存在将自动创建。

    返回值:
        None: 本函数无返回值，所有评估结果将以 CSV 文件形式保存至 `result_path` 目录中。
    """

    # 验证并标准化输入路径
    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise ValueError(f"路径不存在或不是目录: {dataset_path}")

    # 确保输出目录存在
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    # 构建三项核心路径
    data_dir: Path = dataset_path / "data"
    persist_dir: Path = dataset_path / "storage"
    qa_file: Path = dataset_path / f"{dataset_path.name}.csv"

    # 验证 QA 文件存在
    if not qa_file.exists():
        raise ValueError(f"QA 评估文件不存在: {qa_file}")

    # 加载 QA 评估集（CSV 格式：input 列存储问题，answers 列存储参考答案）
    print(f"[INFO] 正在加载 QA 文件：{qa_file}")

    questions: List[str] = []
    answers: List[str] = []

    try:
        with open(qa_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"QA 文件格式错误，没有发现表头：{qa_file} ")
            elif "input" not in reader.fieldnames or "answers" not in reader.fieldnames:
                raise ValueError(
                    f"QA 文件格式错误，必须包含 'input' 和 'answers' 列: {qa_file}"
                )
            for row in reader:
                questions.append(row["input"])
                answers.append(row["answers"])
    except Exception as e:
        raise ValueError(f"读取 QA 文件失败 {qa_file}: {str(e)}")

    # 空的评估集触发异常
    if not questions:
        raise ValueError(f"{qa_file} 没有评估问题")

    print(f"[INFO] QA 文件加载成功，读取 {len(questions)} 个问题")

    # 构建向量索引与 RAG 系统
    print(f"[INFO] 构建索引与 RAG 系统...")

    nodes = get_nodes_from_markdowns(
        input_dir=data_dir,
        chunk_size=config["text_splitter"]["chunk_size"],
        chunk_overlap=config["text_splitter"]["chunk_overlap"],
    )

    index = get_index_from_nodes(nodes, persist_dir=persist_dir)

    rag_system = get_rag_system_from_index(
        index=index,
        similarity_top_k=config["rag"]["similarity_top_k"],
        streaming=False,
    )

    print(f"[INFO] 构建索引与 RAG 系统成功")

    # 批量评估并收集各项指标
    queries: List[str] = []
    responses: List[str] = []
    references: List[str] = []

    # 初始化各项评估指标的分数列表
    correctness_scores: List[float] = []
    semantic_similarity_scores: List[float] = []
    faithfulness_scores: List[float] = []
    context_relevancy_scores: List[float] = []
    answer_relevancy_scores: List[float] = []

    # 遍历问题和参考答案，调用 RAG 系统进行查询，并使用多个评估器计算各项指标得分
    print(f"[INFO] 开始在 {dataset_path.name} 上进行评估")

    pbar = tqdm(total=len(questions), desc="评估进度")
    for query, reference in zip(questions, answers):
        response = cast(Response, rag_system.query(query))
        queries.append(query)
        responses.append(str(response))
        references.append(reference)

        # 正确性评估
        correctness_score = correctness_evaluator.evaluate_response(
            query=query, response=response, reference=reference
        ).score
        if correctness_score is None:
            correctness_score = 0.0
        correctness_score = max(0.0, min(5.0, correctness_score))
        correctness_scores.append(correctness_score)

        # 语义相似度评估
        semantic_similarity_score = semantic_similarity_evaluator.evaluate_response(
            query=query, response=response, reference=reference
        ).score
        if semantic_similarity_score is None:
            semantic_similarity_score = 0.0
        semantic_similarity_score = max(0.0, min(1.0, semantic_similarity_score))
        semantic_similarity_scores.append(semantic_similarity_score)

        # 忠实度评估（回答是否忠实于上下文）
        faithfulness_score = faithfulness_evaluator.evaluate_response(
            query=query, response=response
        ).score
        if faithfulness_score is None:
            faithfulness_score = 0.0
        faithfulness_score = max(0.0, min(1.0, faithfulness_score))
        faithfulness_scores.append(faithfulness_score)

        # 上下文相关性评估
        context_relevancy_score = context_relevancy_evaluator.evaluate_response(
            query=query, response=response
        ).score
        if context_relevancy_score is None:
            context_relevancy_score = 0.0
        context_relevancy_score = max(0.0, min(1.0, context_relevancy_score))
        context_relevancy_scores.append(context_relevancy_score)

        # 回答相关性评估
        answer_relevancy_score = answer_relevancy_evaluator.evaluate_response(
            query=query, response=response
        ).score
        if answer_relevancy_score is None:
            answer_relevancy_score = 0.0
        answer_relevancy_score = max(0.0, min(1.0, answer_relevancy_score))
        answer_relevancy_scores.append(answer_relevancy_score)

        pbar.update(1)

    pbar.close()

    # 保存评估详细结果（八列：问题、响应、参考答案、正确性分数、语义相似度分数、忠实度分数、上下文相关性分数、回答相关性分数）
    print(f"[INFO] 正在保存评估详细结果...")

    details_path: Path = result_path / "eval_details.csv"
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "query",
                "response",
                "reference",
                "correctness_score",
                "semantic_similarity_score",
                "faithfulness_score",
                "context_relevancy_score",
                "answer_relevancy_score",
            ]
        )
        writer.writerows(
            zip(
                queries,
                responses,
                references,
                correctness_scores,
                semantic_similarity_scores,
                faithfulness_scores,
                context_relevancy_scores,
                answer_relevancy_scores,
            )
        )

    print(f"[INFO] 评估详细结果已保存至 {details_path}")

    # 计算最终分数（每项分数都缩放到 0-100）
    print(f"[INFO] 正在计算最终分数...")

    final_correctness_score: float = (
        sum(correctness_scores) / (5.0 * len(correctness_scores))
    ) * 100
    final_semantic_similarity_score: float = (
        sum(semantic_similarity_scores) / len(semantic_similarity_scores)
    ) * 100
    final_faithfulness_score: float = (
        sum(faithfulness_scores) / len(faithfulness_scores)
    ) * 100
    final_context_relevancy_score: float = (
        sum(context_relevancy_scores) / len(context_relevancy_scores)
    ) * 100
    final_answer_relevancy_score: float = (
        sum(answer_relevancy_scores) / len(answer_relevancy_scores)
    ) * 100
    avg_score: float = (
        final_correctness_score
        + final_semantic_similarity_score
        + final_faithfulness_score
        + final_context_relevancy_score
        + final_answer_relevancy_score
    ) / 5.0

    print(
        f"[INFO] 平均得分为：{avg_score:.2f}，具体得分为：\n"
        f"- correctness_score = {final_correctness_score:.2f}\n"
        f"- semantic_similarity_score = {final_semantic_similarity_score:.2f}\n"
        f"- faithfulness_score = {final_faithfulness_score:.2f}\n"
        f"- context_relevancy_score = {final_context_relevancy_score:.2f}\n"
        f"- answer_relevancy_score = {final_answer_relevancy_score:.2f}"
    )

    # 准备汇总结果
    results: List[Dict[str, Union[str, float]]] = [
        {
            "dataset": dataset_path.name,
            "correctness_score": final_correctness_score,
            "semantic_similarity_score": final_semantic_similarity_score,
            "faithfulness_score": final_faithfulness_score,
            "context_relevancy_score": final_context_relevancy_score,
            "answer_relevancy_score": final_answer_relevancy_score,
            "avg_score": avg_score,
        }
    ]

    # 保存评估分数到 CSV 文件
    print(f"[INFO] 正在保存最终评估结果...")

    summary_path: Path = result_path / "eval_scores.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "correctness_score",
                "semantic_similarity_score",
                "faithfulness_score",
                "context_relevancy_score",
                "answer_relevancy_score",
                "avg_score",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(
        f"[INFO] 在 {dataset_path.name} 上的评估完成，最终评估结果保存在 {summary_path} 中"
    )


# 定义颜色代码
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"  # 重置为默认颜色


if __name__ == "__main__":
    DATASETS_DIR = Path(config["paths"]["datasets_dir"])
    RESULTS_DIR = Path(config["paths"]["results_dir"])

    print(RED + f"[INFO] 正在 {DATASETS_DIR} 内的各个数据集上进行评估..." + RESET)

    for dataset_path in sorted(DATASETS_DIR.iterdir()):
        print(RED + f"[INFO] 正在 {dataset_path.name} 数据集上进行评估..." + RESET)

        if dataset_path.is_dir():
            evaluation_with_rag(
                dataset_path=dataset_path,
                result_path=RESULTS_DIR / dataset_path.name / "with_rag",
            )
            evaluation_without_rag(
                dataset_path=dataset_path,
                result_path=RESULTS_DIR / dataset_path.name / "without_rag",
            )

        print(GREEN + f"[INFO] {dataset_path.name} 数据集上的评估完成" + RESET)

    print(GREEN + f"[INFO] 所有数据集评估完成，结果保存在 {RESULTS_DIR} 中" + RESET)

