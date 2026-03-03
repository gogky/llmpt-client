# llmpt-client

HuggingFace Hub 模型的 P2P 加速下载客户端

## 特性

- 🚀 **无缝集成**：无需修改代码，直接替换 `huggingface_hub` 下载
- 🌐 **P2P 加速**：通过 BitTorrent 从其他用户处下载模型
- 📦 **自动降级**：P2P 失败时自动降级到 HTTP
-  **断点续传**：支持 fastresume，恢复中断的下载
- 🎯 **最小代码修改**：只需添加一行 `import llmpt` 即可启用 P2P
- 🖥️ **CLI 工具**：提供 `llmpt-cli` 命令行工具，支持下载、做种、状态查看

## 安装

```bash
pip install llmpt-client
```

### 依赖要求

- Python >= 3.8
- libtorrent >= 2.0.0（自动安装）

## 快速开始

### 方式一：环境变量（推荐）

```bash
# 全局启用 P2P
export HF_USE_P2P=1
export HF_P2P_TRACKER=http://your-tracker.com  # 可选，不设置则使用默认值
```

```python
# 只需添加一行 import（会自动检测环境变量并启用 P2P）
import llmpt

# 其余代码无需修改
from huggingface_hub import snapshot_download

# 自动使用 P2P 下载
snapshot_download("meta-llama/Llama-2-7b")
```

### 方式二：显式启用

```python
from llmpt import enable_p2p
from huggingface_hub import snapshot_download

# 显式启用 P2P
enable_p2p(tracker_url="http://your-tracker.com")

# 后续所有下载都使用 P2P
snapshot_download("meta-llama/Llama-2-7b")
```

## 工作原理

llmpt 通过 Monkey Patch 拦截 `huggingface_hub` 的下载流程，将文件请求路由到 BitTorrent 网络：

1. **Monkey Patch 拦截**：`enable_p2p()` 对 `hf_hub_download` 和 `http_get` 进行 Monkey Patch
2. **上下文注入**：当 `hf_hub_download` 被调用时，注入 P2P 上下文（repo_id、filename、tracker 信息）
3. **HTTP 请求拦截**：每个文件的 `http_get` 调用被拦截，转交给 `P2PBatchManager`
4. **BitTorrent 下载**：`P2PBatchManager` 为每个仓库维护一个 `SessionContext`，通过 magnet link 加入 BitTorrent swarm 下载文件
5. **自动降级**：如果 P2P 下载超时或失败，自动回退到原始 HTTP 下载

> **注意**：做种需要通过 CLI 工具 `llmpt-cli seed` 手动触发，下载时不会自动创建种子。

## CLI 工具

安装后可使用 `llmpt-cli` 命令行工具：

```bash
# 通过 P2P 下载模型
llmpt-cli download meta-llama/Llama-2-7b

# 指定 Tracker 下载
llmpt-cli download gpt2 --tracker http://tracker.example.com

# 为已缓存的仓库创建种子并做种
llmpt-cli seed --repo-id gpt2 --revision main

# 查看做种状态
llmpt-cli status

# 停止指定仓库的做种
llmpt-cli stop gpt2@main

# 停止所有做种
llmpt-cli stop
```

## 配置

### 环境变量

```bash
# 启用 P2P
HF_USE_P2P=1

# Tracker 服务器地址（可选）
HF_P2P_TRACKER=http://your-tracker.com

# 下载后自动做种（默认：1）
HF_P2P_AUTO_SEED=1

# 做种时长（秒，0 表示永久，默认：3600）
HF_P2P_SEED_TIME=3600

# P2P 下载超时时间（秒，默认：300）
HF_P2P_TIMEOUT=300
```

### Python API

```python
from llmpt import enable_p2p, disable_p2p, is_enabled, stop_seeding, get_config

# 使用自定义设置启用
enable_p2p(
    tracker_url="http://your-tracker.com",
    auto_seed=True,
    seed_duration=3600,  # 做种 1 小时
    timeout=300
)

# 检查 P2P 是否启用
is_enabled()  # -> True

# 获取当前配置
get_config()  # -> {'tracker_url': '...', 'auto_seed': True, ...}

# 禁用 P2P
disable_p2p()

# 停止所有做种任务
stop_seeding()
```

## 架构

```
用户代码
    ↓
huggingface_hub.snapshot_download()
    ↓
llmpt Monkey Patch (hf_hub_download + http_get)
    ↓
    ├─→ _patched_hf_hub_download: 注入 P2P 上下文
    │     ↓
    │   _patched_http_get: 拦截 HTTP 请求
    │     ↓
    │   P2PBatchManager.register_request()
    │     ↓
    │   SessionContext.download_file()
    │     ├─→ 查询 Tracker (magnet link)
    │     ├─→ 加入 BitTorrent swarm (libtorrent)
    │     ├─→ monitor 后台监控 & 交付文件
    │     ├─→ 成功 → 跳过 HTTP ✓
    │     └─→ 超时/失败 → Fallback 到原始 HTTP
    ↓
返回文件路径
```

## 开发

### 搭建开发环境

```bash
git clone https://github.com/yourusername/llmpt-client.git
cd llmpt-client

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -e ".[dev]"
```

### 运行测试

```bash
# 单元测试
pytest tests/unit/

# 集成测试（宿主机直接运行，需要网络连接）
pytest tests/integration/ --run-integration

# E2E 测试（Docker 多容器，验证完整 P2P 下载链路）
docker compose -f docker-compose.test.yml up --build
```

### 项目结构

```
llmpt-client/
├── llmpt/                    # 主包
│   ├── __init__.py           # 入口点，自动启用逻辑
│   ├── patch.py              # Monkey Patch 实现（hf_hub_download + http_get）
│   ├── p2p_batch.py          # P2P 批量管理器（单例，调度核心）
│   ├── session_context.py    # 会话上下文（管理单个 torrent 生命周期）
│   ├── monitor.py            # 后台监控循环（进度跟踪 & 文件交付）
│   ├── tracker_client.py     # Tracker API 客户端
│   ├── seeder.py             # 做种管理器
│   ├── torrent_creator.py    # 种子创建（v1_only 模式）
│   ├── cli.py                # CLI 工具 (llmpt-cli)
│   └── utils.py              # 工具函数
├── tests/                    # 测试
│   ├── unit/                 # 单元测试
│   ├── integration/          # 集成测试（宿主机运行）
│   └── e2e/                  # E2E 测试（Docker 容器运行）
├── examples/                 # 示例脚本
├── docs/                     # 文档
├── setup.py                  # 安装配置
├── setup.cfg                 # pytest / flake8 / mypy 配置
├── requirements.txt          # 运行依赖
├── requirements-dev.txt      # 开发依赖
└── docker-compose.test.yml   # E2E 测试 Docker 编排
```

## 兼容性

- ✅ Linux
