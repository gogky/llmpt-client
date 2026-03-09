# llmpt-client

HuggingFace Hub 模型/数据集的 P2P 加速下载客户端。

## 特性

- **无缝集成**：只需一行代码即可为 HuggingFace 下载启用 P2P 加速。
- **P2P 加速**：通过 BitTorrent 从其他用户处高速下载模型和数据集
- **WebSeed**：无 P2P peer 时自动从 HuggingFace CDN 获取数据，实现冷启动
- **自动降级**：P2P 失败时自动回退到标准 HTTP 下载
- **自动做种**：下载完成后自动创建 torrent 并持续做种，回馈社区
- **后台守护进程**：做种运行在独立的后台进程中，不影响正常使用
- **全缓存做种**：守护进程会扫描 HF 缓存中的所有模型/数据集并自动做种
- **自定义目录支持**：支持 `snapshot_download(..., cache_dir=...)` 和 `snapshot_download(..., local_dir=...)`
- **冷启动恢复**：守护进程重启后可恢复已登记的自定义目录做种任务

## 安装

```bash
git clone https://github.com/gogky/llmpt-client.git
cd llmpt-client
pip install -e ".[dev]"
```

### 依赖要求

- Python >= 3.8
- libtorrent >= 2.0.0

## 快速开始

### 方式一：环境变量配置 + 显式启用

```bash
# 设置 tracker 地址（enable_p2p() 会自动读取）
export HF_P2P_TRACKER=http://your-tracker-server
```

```python
from llmpt import enable_p2p
from huggingface_hub import snapshot_download

enable_p2p()  # 自动读取 HF_P2P_TRACKER 等环境变量
snapshot_download("meta-llama/Llama-2-7b")  # 使用 P2P
```

### 方式二：代码中直接指定

```python
from llmpt import enable_p2p
enable_p2p(tracker_url="http://your-tracker-server")

from huggingface_hub import snapshot_download

snapshot_download("meta-llama/Llama-2-7b")
```

当调用 `enable_p2p()` 时，系统会自动：
1. Monkey Patch HuggingFace 的下载函数，拦截文件请求到 P2P 网络
2. 在后台启动**做种守护进程**，扫描本地 HF 缓存，为所有已有的模型/数据集做种
3. 下载完成后通知守护进程，自动为新下载的模型/数据集创建 torrent 并注册到 tracker

推荐在`snapshot_download("meta-llama/Llama-2-7b")`前执行 `enable_p2p()` ，但不强制要求。

### 自定义下载目录

`llmpt` 现在支持 Hugging Face 的两种自定义存储方式：

- `cache_dir=...`：使用自定义 Hub cache 根目录
- `local_dir=...`：直接把文件展开到目标目录

这两种方式都会把存储位置透传给 P2P 下载端和后台守护进程。下载完成后，daemon 会基于对应目录创建 torrent 并开始做种；daemon 重启后，也会恢复已知的自定义目录做种任务。

```python
from llmpt import enable_p2p
enable_p2p(tracker_url="http://your-tracker-server")

from huggingface_hub import snapshot_download

# 自定义 Hub cache
snapshot_download(
    repo_id="gpt2",
    cache_dir="/data/hf-cache",
)

# 直接展开到目标目录
snapshot_download(
    repo_id="gpt2",
    local_dir="/data/models/gpt2",
)
```

## CLI 工具

安装后可使用 `llmpt-cli` 命令行工具：

```bash
# ─── 下载 ───
llmpt-cli download meta-llama/Llama-2-7b                     # 下载模型
llmpt-cli download fka/prompts.chat --repo-type dataset       # 下载数据集
llmpt-cli download gpt2 --tracker http://your-tracker-server  # 指定 tracker

# ─── 守护进程管理 ───
llmpt-cli start                                               # 启动后台做种守护进程
llmpt-cli start --tracker http://your-tracker-server          # 指定 tracker
llmpt-cli status                                              # 查看做种状态
llmpt-cli stop                                                # 停止守护进程
llmpt-cli restart                                             # 重启

# ─── 手动做种（高级用法）───
llmpt-cli seed --repo-id gpt2 --revision main                 # 为指定模型创建种子并做种
llmpt-cli seed --repo-id gpt2 --revision main --repo-type dataset  # 指定类型

```

当对应 CLI 参数未显式传入时，`llmpt-cli` 会继续读取环境变量作为默认值：
`HF_P2P_TRACKER`、`HF_P2P_TIMEOUT`、`HF_P2P_PORT`、`HF_P2P_WEBSEED`、`HF_P2P_VERBOSE`、`HF_TOKEN`。

### 守护进程

做种运行在独立的后台守护进程中，**不需要保持终端打开**：

```bash
$ llmpt-cli start
✓ Daemon started (PID: 12345)
  Tracker: http://your-tracker-server
  Logs: ~/.cache/llmpt/daemon.log
$  # ← 立刻返回，终端可以关掉
```

守护进程的核心功能：
- **自动扫描** `~/.cache/huggingface/hub/` 中所有完整的模型/数据集快照
- **扫描自定义 cache_dir**：恢复已登记的自定义 Hub cache 根目录
- **恢复 local_dir 做种**：基于本地 registry 和 `local_dir/.cache/huggingface/download/*.metadata` 重建做种任务
- **自动创建 torrent**：如果 tracker 上没有某模型的 torrent，自动创建并注册
- **持续做种**：为所有已发现的模型/数据集提供 P2P 上传
- **接收通知**：下载端下载完成后会通过 IPC 立即通知守护进程

当使用 `enable_p2p()` 时，守护进程会 **自动启动**，用户无需手动操作。如果不想做种，可以停止守护进程：

```bash
llmpt-cli stop
```

## 配置

### 环境变量

```bash
# HuggingFace 镜像（如果无法访问 huggingface.co）
HF_ENDPOINT=https://hf-mirror.com

# Tracker 服务器地址
HF_P2P_TRACKER=http://your-tracker-server

# libtorrent 监听端口（可选，默认 6881，被占用时自动尝试 6881-6999）
HF_P2P_PORT=6881

# P2P 下载超时时间（秒，默认：300）
HF_P2P_TIMEOUT=300

# 启用/禁用 WebSeed（默认：1）
HF_P2P_WEBSEED=1

# HuggingFace 认证 Token（用于 WebSeed 访问私有/受限仓库）
HF_TOKEN=hf_xxxxx

# 下载完成后显示是否显示统计信息（默认：False）
HF_P2P_VERBOSE=False
```

> **注意**：llmpt 在做种和下载时需要调用 HuggingFace API 解析仓库版本信息。如果未设置镜像且无法访问 `huggingface.co`，版本解析将超时失败。

### Python API

```python
from llmpt import enable_p2p, disable_p2p, is_enabled, stop_seeding, get_config, shutdown

# 使用自定义设置启用
enable_p2p(
    tracker_url="http://my-tracker",
    timeout=600,          # 10分钟超时
    port=6881,            # 指定端口
    hf_token="hf_xxxxx",  # 访问私有仓库
    webseed=True,         # 启用 WebSeed（默认）
)

# 检查 P2P 是否启用
is_enabled()  # -> True

# 获取当前配置
get_config()

# 禁用 P2P（恢复标准 HTTP 下载）
disable_p2p()

# 停止所有做种任务
stop_seeding()

# 完全关闭（停止做种、清理临时文件、释放 libtorrent session）
# 进程退出时会自动调用
shutdown()
```

## 兼容性

- ✅ Linux
