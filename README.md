# RAG 方案复现

## 项目情况

本项目目前复现了：  

* Naive RAG
* HyDE

## 快速开始评测

首先，配置好 Ollama 服务，请参考[官方教程](https://ollama.com/)。

然后，拉取需要使用的模型：

```bash
$ ollama pull qwen3-embedding:4b
$ ollama pull qwen3-vl:30b-a3b
$ ollama pull gemma3:27b
```

接着，配置环境：

```bash
pip install -r requirements.txt
```

> 本项目采用 Python 3.12。

最后，开始评测：

```bash
cd scripts/
chmod +x run.sh
./run.sh
```

> 评测使用的数据集为 [UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain)

评测的结果会保存在 `src/<RAG 方案>/results` 目录中。

> 本项目运行时使用了 4 卡 RTX 3090ti。如果您的配置不一致，请务必先检查 `src/<RAG 方案>/config.yaml` 中的设置，必要时请修改，以防评测过程出现问题。

## 快速运行带有前端界面的 demo

同样，需要先配置好 Ollama 服务、拉取需要使用的模型并配置环境。

> 可以不拉取 `gemma3:27b`，因为其仅用于评估。

然后，把您的私有文档转化为 `Markdown` 后，再放入 `src/<RAG 方案>/knowledge_base/data` 中。

接着，运行命令：

```bash
cd src/<RAG 方案>/
python web_ui.py
```

最后，根据终端的提示，用浏览器访问指定的链接即可。

> 索引需要花一点时间，请耐心等待。
