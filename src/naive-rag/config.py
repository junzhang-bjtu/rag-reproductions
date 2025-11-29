from pathlib import Path
from typing import Any, Dict, Union

import yaml


def get_config_from_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载并解析 YAML 配置文件。

    功能流程：
    1. 验证配置文件路径的有效性和格式
    2. 以 UTF-8 编码读取文件内容
    3. 安全加载 YAML 数据（防止代码注入）
    4. 返回字典类型的配置数据

    Args:
        config_path: YAML 配置文件的路径，支持字符串或 Path 对象

    Returns:
        Dict[str, Any]: 解析后的配置字典

    Raises:
        FileNotFoundError: 配置文件不存在时抛出
        ValueError: 路径不是文件或非 YAML 格式时抛出
        yaml.YAMLError: YAML 语法错误时抛出
        UnicodeDecodeError: 文件编码非 UTF-8 时抛出
    """
    # 将输入路径统一转换为 Path 对象，便于后续操作
    config_path = Path(config_path)

    # 验证文件是否存在，提前暴露路径错误
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到：{config_path}")

    # 验证路径是否为文件（而非文件夹）且扩展名为 YAML 格式
    if not config_path.is_file() or config_path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"无效的配置文件路径：{config_path}")

    # 使用 UTF-8 编码打开文件，确保跨平台兼容性
    # yaml.safe_load 提供安全的解析方式，避免执行恶意代码
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding,
            e.object,
            e.start,
            e.end,
            f"配置文件编码错误，请确保使用 UTF-8 编码：{config_path}",
        ) from e

    # 处理空文件情况（safe_load 返回 None），确保始终返回字典
    if config_data is None:
        return {}

    # 验证加载结果是否为字典类型
    if not isinstance(config_data, dict):
        raise TypeError(
            f"配置文件必须解析为字典，实际类型：{type(config_data).__name__}"
        )

    return config_data
