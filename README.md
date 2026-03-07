# llmpt-client

HuggingFace Hub 模型的 P2P 加速下载客户端

## 特性

- 🚀 **无缝集成**：只需一行 `import llmpt` 即可为 HuggingFace 下载启用 P2P 加速
- 🌐 **P2P 加速**：通过 BitTorrent 从其他用户处高速下载模型
- 🌍 **WebSeed**：无 P2P peer 时自动从 HuggingFace CDN 获取数据，实现冷启动
- 📦 **自动降级**：P2P 失败时自动回退到标准 HTTP 下载
- 🔄 **自动做种**：下载完成后自动创建 torrent 并持续做种，回馈社区
- 🖥️ **后台守护进程**：做种运行在独立的后台进程中，不影响正常使用
- � **全缓存做种**：守护进程会扫描 HF 缓存中的所有模型并自动做种

## 安装

```bash
pip install llmpt-client
```

### 依赖要求

- Python >= 3.8
- libtorrent >= 2.0.0

## 快速开始

### 方式一：环境变量（最简单）

```bash
# 设置环境变量
export HF_USE_P2P=1
export HF_P2P_TRACKER=http://118.195.159.242
```

```python
# 只需添加一行 import，其余代码无需修改
import llmpt
from huggingface_hub import snapshot_download

snapshot_download("meta-llama/Llama-2-7b")  # 自动使用 P2P
```

### 方式二：显式启用

```python
from llmpt import enable_p2p
from huggingface_hub import snapshot_download

enable_p2p(tracker_url="http://118.195.159.242")
snapshot_download("meta-llama/Llama-2-7b")
```

当调用 `enable_p2p()` 时，系统会自动：
1. Monkey Patch HuggingFace 的下载函数，拦截文件请求到 P2P 网络
2. 在后台启动**做种守护进程**，扫描本地 HF 缓存，为所有已有的模型做种
3. 下载完成后通知守护进程，自动为新下载的模型创建 torrent 并注册到 tracker

## CLI 工具

安装后可使用 `llmpt-cli` 命令行工具：

```bash
# ─── 下载 ───
llmpt-cli download meta-llama/Llama-2-7b
llmpt-cli download fka/prompts.chat --repo-type dataset
llmpt-cli download gpt2 --tracker http://118.195.159.242

# ─── 守护进程管理 ───
llmpt-cli daemon start                           # 启动后台做种守护进程
llmpt-cli daemon start --tracker http://118.195.159.242     # 指定 tracker
llmpt-cli daemon status                           # 查看做种状态
llmpt-cli daemon stop                             # 停止守护进程
llmpt-cli daemon restart                          # 重启

# ─── 手动做种（高级用法）───
llmpt-cli seed --repo-id gpt2 --revision main    # 为指定模型创建种子并做种

# ─── 状态管理 ───
llmpt-cli status                                  # 查看做种状态
llmpt-cli stop                                    # 停止所有做种
llmpt-cli stop gpt2@main                          # 停止指定做种
llmpt-cli stop fka/prompts.chat@main --repo-type dataset  # 停止指定类型仓库的做种
```

### 守护进程

做种运行在独立的后台守护进程中，**不需要保持终端打开**：

```bash
$ llmpt-cli daemon start
✓ Daemon started (PID: 12345)
  Tracker: http://118.195.159.242
  Logs: ~/.cache/llmpt/daemon.log
$  # ← 立刻返回，terminal 可以关掉
```

守护进程的核心功能：
- **自动扫描** `~/.cache/huggingface/hub/` 中所有完整的模型快照
- **自动创建 torrent**：如果 tracker 上没有某模型的 torrent，自动创建并注册
- **持续做种**：为所有已发现的模型提供 P2P 上传
- **接收通知**：下载端下载完成后会通过 IPC 立即通知守护进程

当使用 `enable_p2p()` 时，守护进程会 **自动启动**，用户无需手动操作。如果不想做种，可以：

```python
enable_p2p(auto_seed=False)  # 只下载，不做种
```

## 工作原理

```
┌─────────────────────────────────────────────────────────────────┐
│  下载端（随 HuggingFace 进程运行）                                │
│                                                                 │
│  enable_p2p() → Monkey Patch hf_hub_download + http_get         │
│       ↓                      + snapshot_download                │
│  snapshot_download("gpt2")                                      │
│       ↓                                                         │
│  _patched_http_get → 尝试 P2P                                   │
│       ├─ 成功 → 跳过 HTTP ✓                                     │
│       └─ 失败 → 回退到标准 HTTP                                  │
│       ↓                                                         │
│  下载完成 → 通知守护进程（IPC）                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↕ Unix Socket IPC
┌─────────────────────────────────────────────────────────────────┐
│  守护进程 llmptd（独立后台进程，长期运行）                          │
│                                                                 │
│  ┌─ 定时扫描 HF cache → 发现新模型                               │
│  ├─ 接收 IPC 通知    → 新下载完成                                │
│  ↓                                                              │
│  create_and_register_torrent() → 注册到 Tracker                  │
│  ↓                                                              │
│  P2PBatchManager → libtorrent → 持续做种 📡                     │
└─────────────────────────────────────────────────────────────────┘
```

## 配置

### 环境变量

```bash
# 启用 P2P
HF_USE_P2P=1

# HuggingFace 镜像（如果无法访问 huggingface.co）
HF_ENDPOINT=https://hf-mirror.com

# Tracker 服务器地址
HF_P2P_TRACKER=http://118.195.159.242

# libtorrent 监听端口（可选，默认自动选择 6881-6999）
HF_P2P_PORT=6881

# 下载后自动做种（默认：1）
HF_P2P_AUTO_SEED=1

# 做种时长（秒，0 = 永久，默认：3600）
HF_P2P_SEED_TIME=3600

# P2P 下载超时时间（秒，默认：300）
HF_P2P_TIMEOUT=300

# 启用/禁用 WebSeed（默认：1）
HF_P2P_WEBSEED=1

# HuggingFace 认证 Token（用于 WebSeed 访问私有仓库）
HF_TOKEN=hf_xxxxx
```

> **注意**：llmpt 在做种和下载时需要调用 HuggingFace API 解析仓库版本信息。如果未设置镜像且无法访问 `huggingface.co`，版本解析将超时失败。

### Python API

```python
from llmpt import enable_p2p, disable_p2p, is_enabled, stop_seeding, get_config

# 使用自定义设置启用
enable_p2p(
    tracker_url="http://118.195.159.242",
    auto_seed=True,       # 自动启动守护进程做种（默认 True）
    seed_duration=3600,
    timeout=300,
    port=6881,
    webseed=True,
    hf_token="hf_xxx",
)

# 检查 P2P 是否启用
is_enabled()  # -> True

# 获取当前配置
get_config()

# 禁用 P2P
disable_p2p()

# 停止所有做种任务
stop_seeding()
```

## 开发

### 搭建开发环境

```bash
git clone https://github.com/gogky/llmpt-client.git
cd llmpt-client

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

### 运行测试

```bash
# 单元测试
pytest tests/unit/

# E2E 测试（Docker 多容器）

# 场景 1：纯 P2P 下载（WebSeed 禁用）
docker compose -f docker-compose.test.yml up --build

# 场景 2：纯 WebSeed 下载（无 P2P peer，冷启动）
docker compose -f docker-compose.test-webseed.yml up --build

# 场景 3：P2P + WebSeed 混合模式
docker compose -f docker-compose.test-hybrid.yml up --build

# 场景 4：守护进程自动做种（冷启动 → 自动创建 torrent → P2P 下载）
docker compose -f docker-compose.test-daemon.yml up --build
```

### 项目结构

```
llmpt-client/
├── llmpt/                            # 主包
│   ├── __init__.py                   # 入口点，enable_p2p() API
│   ├── patch.py                      # Monkey Patch（hf_hub_download + http_get + snapshot_download）
│   ├── p2p_batch.py                  # P2P 批量管理器（单例，调度核心）
│   ├── session_context.py            # 会话上下文（单个 torrent 生命周期管理）
│   ├── monitor.py                    # 后台监控循环（进度跟踪 & 文件交付）
│   ├── daemon.py                     # 做种守护进程（后台运行，自动做种）
│   ├── cache_scanner.py              # HF 缓存扫描器（发现可做种模型）
│   ├── ipc.py                        # 进程间通信（Unix Socket）
│   ├── webseed_proxy.py              # WebSeed 本地反向代理
│   ├── tracker_client.py             # Tracker API 客户端
│   ├── seeder.py                     # 做种管理器
│   ├── torrent_creator.py            # 种子创建（v1_only 模式）
│   ├── torrent_cache.py              # 本地 .torrent 缓存
│   ├── seeding_mapper.py             # 做种文件映射（hardlink）
│   ├── cli.py                        # CLI 工具 (llmpt-cli)
│   └── utils.py                      # 工具函数
├── tests/
│   ├── unit/                         # 单元测试
│   └── e2e/                          # E2E 测试（Docker 容器运行）
├── docker-compose.test.yml           # E2E：纯 P2P
├── docker-compose.test-webseed.yml   # E2E：纯 WebSeed
├── docker-compose.test-hybrid.yml    # E2E：P2P + WebSeed 混合
├── docker-compose.test-daemon.yml    # E2E：守护进程自动做种
└── setup.py
```

## 兼容性

- ✅ Linux
