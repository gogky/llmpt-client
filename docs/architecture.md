# llmpt-client 架构设计

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

## 核心流程

### 下载端

1. 用户调用 `enable_p2p()` 或设置 `HF_USE_P2P=1`
2. 系统 Monkey Patch HuggingFace 的三个核心函数：
   - `hf_hub_download` — 注入 P2P 上下文（repo_id, filename, revision 等）
   - `http_get` — 拦截 HTTP 请求，尝试通过 P2P 下载
   - `snapshot_download` — 下载完成后通知守护进程
3. 同时禁用 HuggingFace 的 Xet Storage（否则 LFS 文件会绕过 http_get patch）
4. 自动启动 WebSeed 本地反向代理（作为无 peer 时的 fallback）
5. 自动启动做种守护进程（如果尚未运行）

### 守护进程 (llmptd)

1. 作为独立后台进程运行，独立于 HuggingFace 进程
2. 定时扫描 `~/.cache/huggingface/hub/` 中所有完整的快照
3. 为发现的模型/数据集创建 torrent 并注册到 Tracker
4. 通过 libtorrent 持续做种
5. 通过 Unix Socket IPC 接收来自下载端的通知
6. 支持 seed 失败指数退避重试

### Import Order 无关性

`enable_p2p()` 使用栈帧回溯（stack frame inspection）作为后备方案，
即使用户在 `enable_p2p()` 之前 import 了 `hf_hub_download`，P2P 仍然能正常工作。
详见 [monkey_patch_import_order.md](./monkey_patch_import_order.md)。

## 项目结构

```
llmpt-client/
├── llmpt/                            # 主包
│   ├── __init__.py                   # 入口点，enable_p2p() / disable_p2p() 等公开 API
│   ├── patch.py                      # Monkey Patch（hf_hub_download + http_get + snapshot_download）
│   ├── p2p_batch.py                  # P2P 批量管理器（单例，调度核心）
│   ├── session_context.py            # 会话上下文（单个 torrent 生命周期管理）
│   ├── monitor.py                    # 后台监控循环（进度跟踪 & 文件交付）
│   ├── daemon.py                     # 做种守护进程（后台运行，自动做种）
│   ├── cache_scanner.py              # HF 缓存扫描器（发现可做种模型/数据集）
│   ├── ipc.py                        # 进程间通信（Unix Socket）
│   ├── webseed_proxy.py              # WebSeed 本地反向代理
│   ├── tracker_client.py             # Tracker API 客户端
│   ├── seeder.py                     # 做种管理器
│   ├── torrent_creator.py            # 种子创建（v1_only 模式）
│   ├── torrent_cache.py              # 本地 .torrent 缓存
│   ├── torrent_init.py               # Torrent 初始化辅助函数
│   ├── seeding_mapper.py             # 做种文件映射（hardlink）
│   ├── cli.py                        # CLI 工具 (llmpt-cli)
│   └── utils.py                      # 工具函数
├── tests/
│   ├── unit/                         # 单元测试
│   ├── integration/                  # 集成测试
│   └── e2e/                          # E2E 测试（Docker 容器运行）
├── docs/                             # 文档
├── examples/                         # 使用示例
├── docker-compose.test.yml           # E2E：纯 P2P
├── docker-compose.test-webseed.yml   # E2E：纯 WebSeed
├── docker-compose.test-hybrid.yml    # E2E：P2P + WebSeed 混合
├── docker-compose.test-daemon.yml    # E2E：守护进程自动做种
└── setup.py
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
