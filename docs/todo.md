# llmpt-client 开发路线图

注意本项目使用uv管理python环境，本地存在.venv目录

## 依赖关系总览

```mermaid
graph TD
    A["✅ P1: revision 统一为 commit hash"] --> B["P3: 跨版本 swarm 共享"]
    A --> C["P2: 服务端存储 .torrent"]
    C --> D["P2: 服务端自动做种"]
    E["P2: WebSeed - HF HTTP 作为虚拟 seed"]
    F["✅ P1: 端口动态分配"] --> G["✅ P2: monitor 守护进程解耦"]
    G --> H["P3: 进度条"]
    I["✅ P1: timeout/metadata 等待可配置"] --> J["P2: seeder.py 重构"]
    K["✅ P1: E2E 测试覆盖真实路径"] --> J
    K --> L["P2: fastresume 兼容旧版 libtorrent"]
    C --> M["P2: magnet 携带扩展元数据"]
    N["P3: 服务端校验与安全"]
    O["P3: 支持 huggingface_xet"]
    P[\"✅ P2: 分片大小自动选择\"]
    Q["✅ P2: fastresume 跳过验证"]
    R["P3: 混合下载 - P2P 中断后 HTTP 续传"]
    S["✅ P1: logging.basicConfig 反模式"]
    U["✅ P1: auto_seed/seed_duration"]
    U --> T["P2: 进程退出清理"]
    G --> T
    V["P2: 返回值类型修正"]
    W["P3: 死代码清理"]
    G --> X["P2: P2P 超时改为卡顿检测"]
```


---

## P0 — 基础保障（不需要大改动，但影响全局正确性）

### ~~0.1 · 服务端 `/api/v1/health` 端点~~ ✅
- **已完成**：服务端 `web-server` 和 `tracker` 均已注册 `/api/v1/health`（复用 `/health` handler）。

---

## P1 — 高优先级（核心正确性 & 阻塞其他工作）

### ~~1.1 · revision 统一为 commit hash~~ ✅
- **已完成**：
  - `utils.py` 新增 `resolve_commit_hash()` —— 通过 `HfApi.repo_info()` 将分支名/标签解析为 40 字符 commit hash，带进程内缓存
  - `patch.py`：`_patched_hf_hub_download` 在存储 P2P 上下文前解析 revision（失败时 gracefully fallback 为原始值）
  - `cli.py`：`cmd_seed` 在创建 torrent 和注册前解析 revision
  - `torrent_creator.py`：`create_and_register_torrent` 使用 snapshot 路径中的实际 commit hash 进行注册（defense-in-depth）
  - `session_context.py`：移除了 "hash lookup failed → retry with main" 的 workaround
  - **服务端** `publish.go`：新增 commit hash 格式校验（拒绝非 40 字符 hex 的 revision）
  - **服务端** `torrents.go`：`ListTorrents` 支持 `revision` 查询参数（之前仅支持 `repo_id`）
  - **客户端** `tracker_client.py`：`get_torrent_info` 将 revision 作为服务端查询参数传递（之前为客户端本地过滤）
- **阻塞**：跨版本 swarm 共享 (3.1)、服务端存储 .torrent (2.1) 都依赖 revision 语义的明确性

### ~~1.2 · 端口动态分配~~ ✅
- **现状**：`p2p_batch.py` 第 46 行硬编码 `listen_interfaces = '0.0.0.0:6881'`。
- **问题**：
  - 同一台机器无法运行多个 llmpt 实例（测试、多用户场景）
  - 6881 是知名 BT 端口，某些 ISP/防火墙会封锁
- **方案**：
  1. 默认使用 `0.0.0.0:6881,[::]:6881`，若 bind 失败则自动尝试 6882-6999
  2. 或直接使用 `0.0.0.0:0` 让 OS 分配随机端口
  3. 支持通过环境变量 `HF_P2P_PORT` 或 `enable_p2p(port=...)` 覆盖

### ~~1.3 · timeout / metadata 等待时间可配置化~~ ✅
- **已完成**：
  - metadata 等待 8s 提取为 `METADATA_TIMEOUT` 常量（支持环境变量 `LLMPT_METADATA_TIMEOUT` 覆盖）
  - recheck 硬超时 120s **移除**（做种场景不应有硬超时，400GB 模型在 HDD 上需要 40+ 分钟，改为无限等待 + 进度日志）
  - seed init timeout 30s **移除**（发现是死代码：`self.timeout` 只在 `download_file()` 中使用，做种路径从不调用）
  - P2P 整体 per-file timeout 300s 保留当前设计，但识别出需要改进 → 见 2.11
- ~~**现状**~~：
  - ~~P2P 整体 timeout 已可配置（`enable_p2p(timeout=300)`），✅~~
  - ~~但 metadata 等待时间硬编码为 8 秒（`session_context.py` L143）~~
  - ~~seeding 的 register timeout 硬编码为 30 秒（`p2p_batch.py` L67）~~
  - ~~recheck timeout 硬编码为 120 秒（`session_context.py` L345）~~
- ~~**方案**：将这些超时统一纳入 `_config` 字典或作为 `SessionContext` 的构造参数，允许通过环境变量或 API 覆盖~~

### ~~1.4 · E2E 测试应覆盖真实用户路径，而非绕过公共接口调用内部函数~~ ✅
- **已完成**：
  - **做种端**：`run_seeder.py` 重构为直接调用 `cmd_seed()` — `llmpt-cli seed` 背后的同一个函数。覆盖完整 CLI 路径：参数解析 → revision 解析 → torrent 创建 → tracker 注册 → 做种引擎启动。通过 `SIGALRM` 模拟 Ctrl+C 退出。
  - **下载端**：`test_docker_p2p.py` 保持 `enable_p2p()` → `snapshot_download()` 路径（API 路径，已正确）
  - 断言改为通过 `get_download_stats()` 公共 API 验证，移除了对 `P2PBatchManager().sessions` 内部状态的直接访问
  - 修复了 `create_torrent` 使用 `local_files_only=True`（不再偷偷触发网络下载，避免 P2P 自拦截）
  - 修复了 `start_seeding` / `cmd_seed` 缺少 `torrent_data` 透传的 bug
- ~~**现状**~~：~~用户使用 llmpt 有两条入口路径，但 E2E 测试都没有完整覆盖~~
- ~~**问题**~~：~~这样测试的是内部组件的拼装，而不是用户实际使用时的行为~~


### ~~1.5 - logging.basicConfig 不应在库中调用~~ ✅
- **现状**：`__init__.py` L28-31 调用了 `logging.basicConfig(level=logging.INFO, ...)`。
- **问题**：这是 Python 库的严重反模式。`basicConfig()` 会覆盖用户应用程序自己的日志配置。用户 `import llmpt` 后，整个应用的日志格式和级别都被静默修改。
- **方案**：
  1. 移除 `logging.basicConfig()` 调用
  2. 改为 `logger.addHandler(logging.NullHandler())`（Python 官方推荐的库日志实践）
  3. 让用户在自己的应用中决定日志的级别和格式

### ~~1.6 - auto_seed 和 seed_duration 是死配置~~ ✅
- **已完成**：采用「进程内后台做种 + atexit 优雅退出 + CLI 阻塞」方案：
  - `SessionContext` 新增 `auto_seed` / `seed_duration` / `seed_start_time` 字段
  - `P2PBatchManager.register_request()` 从全局 `_config` 读取并传入
  - `_deliver_file()`：当 `auto_seed=True` 时保留 p2p_root 中的源文件（hardlink，零额外空间），使 libtorrent 可以继续向其他 peer 供种
  - `monitor.py`：所有文件交付后设置 `seed_start_time`，在主循环中检测 `seed_duration` 到期后清理
  - `cli.py`：`cmd_download()` 下载完成后阻塞做种（显示时长提示，Ctrl+C 退出）；`--no-seed` 跳过
  - `__init__.py`：注册 `atexit` 回调，进程退出时调用 `P2PBatchManager.shutdown()` 清理硬链接、源文件、关闭 lt_session（顺带部分解决 2.9）
  - `download_file()` 注册新文件时重置 `seed_start_time`，避免 snapshot 下载中途误启做种计时器
- ~~**现状**~~：~~`enable_p2p(auto_seed=True, seed_duration=3600)` 接受这两个参数并存入 `_config`，但整个代码库没有任何地方读取 `_config['auto_seed']` 或 `_config['seed_duration']` 来执行实际逻辑。~~
- ~~**问题**~~：~~用户以为设置了自动做种1小时，实际上完全无效，造成误导。~~
---

## P2 — 中优先级（功能增强 & 架构改善）

### ~~2.1 · 服务端存储 .torrent 文件 / 添加缓存~~ ✅
- **已完成**：实现了三层 .torrent 解析机制：
  1. **本地磁盘缓存**（`~/.cache/llmpt/torrents/`）— 0 延迟
  2. **Tracker 服务端下载** — 下载后自动缓存到本地
  3. **本地生成**（仅 seeder 路径）— 生成后自动缓存
  - `torrent_cache.py`：新模块，提供 `load_cached_torrent()` / `save_torrent_to_cache()` / `resolve_torrent_data()` 三层解析函数
  - `session_context._init_torrent()`：下载者路径集成三层缓存，重复下载跳过网络请求
  - `torrent_creator.create_torrent()`：做种者路径命中缓存时跳过 `set_piece_hashes()`（大模型可节省 30+ 分钟）
  - 缓存 key = `(repo_id, revision)`，永不失效（commit hash 确定性）
  - Atomic write（write .tmp → os.replace）防止半写入
  - `_torrent_data_to_result()`：从缓存字节解析完整结果字典，避免重复 hash 计算
  - `cli.py`：添加 SIGTERM handler，`kill <pid>` 也能触发优雅清理

### ~~2.2 · magnet link 携带扩展元数据~~ 去掉
- **现状**：magnet link 只含 info_hash 和 tracker announce URL。
- **方案**：在 tracker 返回的 JSON 中增加：
  - `total_size`（已有字段，但客户端未使用）→ 用于磁盘预分配和用户提示
  - `file_list` → 包含每个文件的大小，用于客户端预判磁盘空间
  - `piece_length` → 客户端可决定是否适合当前网络环境
- **注意**：这不是修改 magnet link 本身（magnet URI 格式有限），而是丰富 tracker API 返回的元数据


### 2.3 · 服务端自动做种
- **现状**：做种完全依赖已有用户的客户端持续运行。如果没人在线做种，新用户的 P2P 请求 100% 失败。（客户关闭huggingface_hub后仍可以做种的解决方案）
- **方案**：
  1. Tracker 服务端收到 torrent 注册后，自动启动一个做种进程
  2. 利用原始 HF Hub HTTP URL 作为 webseed（BEP 17/19），让 libtorrent 在没有 peer 时从 HF 官方 HTTP 拉取
  3. 服务端做种可以保证 swarm 永远有至少一个 seed
- **依赖**：需要 revision = commit hash (1.1) 以及 .torrent 存储 (2.1)



### ~~2.4 · 分片大小 (piece_length) 自动选择~~ ✅
- **已完成**：
  - `torrent_creator.py`：`create_torrent()` 和 `create_and_register_torrent()` 不再接受 `piece_length` 参数
  - piece_length 由 `get_optimal_piece_length(total_size)` 确定性计算，**不允许调用方自定义**——否则同一 repo@revision 会因 piece_length 不同产生不同 info_hash，导致用户被分散到不同 swarm
  - `utils.py`：`get_optimal_piece_length()` 参数名从 `file_size` 改为 `total_size`，明确语义为 torrent 所有文件总大小
  - 分档策略：<100MB → 256KB, 100MB-1GB → 1MB, 1-10GB → 4MB, 10-100GB → 16MB, 100GB-1TB → 32MB, ≥1TB → 64MB
- ~~**现状**~~：~~`torrent_creator.py` 默认 16MB piece_length。`utils.py` 有 `get_optimal_piece_length()` 函数但未被使用！~~
- ~~**问题**~~：
  - ~~16MB 对小文件（<100MB）过大，导致单 piece 包含多个文件，无法精细地按文件优先级下载~~
  - ~~对 100GB+ 的超大模型可能需要更大 piece~~

### ~~2.5 · seeder.py 重构~~ 已完全重构
- **现状**：`seeder.py` 直接操作 `P2PBatchManager` 的内部状态（`manager.sessions`、`manager._lock`、`manager.lt_session`），严重违反封装。
- **方案**：
  1. 将 `stop_seeding()`、`stop_all_seeding()`、`get_seeding_status()` 的逻辑迁移到 `P2PBatchManager` 内部作为方法
  2. `seeder.py` 变成薄封装层，只做 API 转发
  3. 或者直接取消 `seeder.py`，将其功能合并到 `P2PBatchManager`
- **依赖**：测试重构 (1.4) 作为安全网

### ~~2.6 · monitor 守护进程与 session 解耦~~ ✅（核心问题已修复）
- **已完成**：
  - **P0 — 全局 Alert 竞态**：`P2PBatchManager.dispatch_alerts()` 集中弹出 alert 并按 handle 路由到各 `SessionContext.pending_alerts` 收件箱，`_process_alerts()` 从收件箱消费。彻底消除跨 session alert 丢失。
  - **P1 — 无锁 handle 访问**：monitor 所有函数（`_log_diagnostics`、`_save_resume_data`、`_retry_test_peer_connection`、主循环条件）均在 `ctx.lock` 下快照 handle 引用；`stop_seeding()` / `shutdown()` 也在 `ctx.lock` 下置 `handle=None`，消除 TOCTOU 竞态。
  - **P1 — Monitor 退出无通知**：`run_monitor_loop` 添加 `finally: ctx.is_valid = False`，`download_file()` 改为轮询式等待（每秒检查 `is_valid`），monitor 死亡时 ≤1s 内感知并 fallback HTTP。
  - **P2 — shutdown 不等待线程退出**：`shutdown()` / `stop_seeding()` / `stop_all_seeding()` 均在锁外 `join(timeout=3)` worker 线程，防止 atexit 期间 GC 竞态。
  - **P2 — `_check_pending_files` 职责过多**：拆分为 `_check_session_health` / `_resolve_pending_metadata` / `_collect_ready_files` / `_update_seed_timer` 四个子函数，`_deliver_file()` I/O 移到锁外执行。
  - **P2 — `seed_start_time` 无锁读**：`seed_start_time` / `seed_duration` 的读取合并到 `ctx.lock` 快照块中。
- **未完成（可选优化）**：
  - 当前仍为每个 SessionContext 一个 monitor 线程。可考虑合并为单个全局 monitor 线程（由 `P2PBatchManager` 管理），减少线程数。但当前架构在线程安全性上已无问题，此优化为**性能向**而非**正确性向**，优先级降低。
- **新增测试**：`test_alert_race.py`（5 个）、`test_monitor.py` 新增 13 个子函数测试（含 `_deliver_file` 锁外执行验证）

### 2.7 · ~~fastresume 兼容旧版 libtorrent~~ --跳过此需求
- **现状**：`session_context.py` L111 有版本判断 `hasattr(lt.add_torrent_params, "parse_resume_data")`，但实现不完整——旧版 API 分支没有实际加载 resume data 的代码。
- **方案**：
  1. 对 lt < 1.2：使用 `params.resume_data = resume_data` 的旧接口
  2. 对 lt >= 2.0：使用 `lt.read_resume_data()`
  3. 添加单元测试 mock 两种版本

### ~~2.8 · 通过缓存 fastresume 或后台预生成跳过验证阶段~~ ✅
- **已完成**：采用比原方案更优的 **hardlink + seed_mode** 方案：
  - 在 libtorrent 期望的路径创建硬链接指向 HF blob，然后启用 `seed_mode`
  - 0 秒启动（每次都是，不只是第 2 次），libtorrent 在 peer 请求时按需验证 SHA1
  - 跨文件系统时自动 fallback 到 legacy `rename_file()` + `force_recheck()`
  - 做种停止后自动清理硬链接（`_cleanup_seeding_hardlinks()`）
  - 下载路径的 `_deliver_file()` 也增加了源文件清理，避免 p2p_root 残留
- ~~**原方案**~~：
  - ~~注册做种后，保存 fastresume data（包含已校验的 piece 状态）~~
  - ~~下次启动时加载 fastresume 跳过 recheck~~
  - ~~或后台预先计算 piece hash 并缓存~~

### ~~2.9 - 进程退出时优雅清理~~ ✅（基础版）
- **已完成**（随 1.6 auto_seed 一并实现）：
  - `enable_p2p()` 注册 `atexit.register(_cleanup_on_exit)` 回调
  - `P2PBatchManager.shutdown()` 清理所有 session 的做种硬链接、下载源文件、移除 torrent handle
  - 公共 API `llmpt.shutdown()` 可供手动调用
- **未完成**：
  - 捕获 SIGTERM/SIGINT 信号做同样的清理（atexit 在某些信号场景下不会触发）
  - 保存最终 fastresume 数据（当前 monitor 线程周期性保存，但退出时可能遗漏最新状态）

### ~~2.10 - create_and_register_torrent 返回值类型不一致~~
- **现状**：类型标注为 `-> bool`，但实际返回 torrent_info (dict) 或 None。`run_seeder.py` 依赖它返回 dict（`torrent_info['torrent_data']`）。
- **方案**：修正类型标注为 `-> Optional[dict]`，或统一返回值语义

### 2.11 - P2P 下载超时改为基于进度的卡顿检测
- **现状**：`download_file()` 使用固定 `timeout=300s` 的 `event.wait()`。300s 对 50GB 文件太短（50MB/s 需要 1000s），对卡住的下载又太长（白等 5 分钟）。超时后 fallback HTTP 会丢弃所有已下载数据（`_truncate_temp_file`）。
- **问题**：固定超时无法区分"P2P 在努力下载但文件太大"和"P2P 完全卡住没有进展"
- **方案**：改为"如果连续 N 秒没有新数据下载，才放弃 P2P"（stall detection）
  1. monitor 线程追踪 per-file 的字节进度变化
  2. 如果 `STALL_TIMEOUT`（如 60s）内 `file_progress[i]` 没有增长，视为卡住
  3. 只要有进展，就不超时——大文件可以安心下载
- **收益**：大文件不再被错误中断；卡住的下载更快 fallback；WebSeed (3.2) 实现后此机制自然兼容

---

## P3 — 低优先级（长期规划 & 高复杂度）

### 3.1 · 跨版本 swarm 共享
- **问题**：同一模型的两个版本（如 v1.0 和 v1.1）可能 90% 的文件完全相同，但处于不同 swarm，无法互相提供 piece。
- **可能方案**：
  - **方案 A**：Tracker 返回相关 swarm 信息，客户端同时加入多个 torrent
  - **方案 B**：利用 BT v2 的 per-file Merkle hash 实现文件级去重。但 libtorrent Python bindings 对 v2 的支持不成熟（pad 文件问题已验证）
  - **方案 C**：Tracker 维护文件级索引（file hash → 哪些 torrent 包含该文件），客户端从多个 torrent 拼凑
- **依赖**：revision = commit hash (1.1)
- **复杂度**：极高，需要 tracker 和客户端深度配合

### 2.12 · WebSeed — HF HTTP 作为 P2P 兜底（BEP 19）
- **优先级提升理由**：WebSeed 从根本上解决「做种人数不够」的问题。有了它，P2P 变成纯粹的加速层——有 peer 就更快，没 peer 也完全不影响（libtorrent 自动从 HTTP 拉取）。
- **当前架构问题**：
  - 没有 peer 在线 → P2P 超时 → fallback HTTP（白等 5 分钟，所有 P2P 开销浪费）
  - 做种人数完全依赖客户端自觉，不可控
- **WebSeed 架构**：
  - 0 个做种者：libtorrent 直接从 HF HTTP 下载 piece（速度 = 原始 HTTP 速度）
  - 1 个做种者：P2P + HTTP 混合，速度 ≥ 纯 HTTP
  - 多个做种者：P2P 为主，HTTP 补缺，速度最大化
  - 做种者中途退出：libtorrent 自动切到 HTTP 补完剩余 piece
- **实现**：
  1. `torrent_creator.py`：创建 torrent 时添加 `t.add_url_seed(base_url)`
     - 公有仓库：`https://huggingface.co/{repo_id}/resolve/{revision}/`
     - 私有仓库：需要在 URL 中携带 token 或使用 `t.add_http_seed()` (BEP 17，支持自定义 header)
  2. libtorrent 内置 BEP 19 支持，自动在 peer 不足时从 web seed 拉取 piece
  3. 下载的 piece 同样参与 SHA1 校验，安全性不变
- **挑战**：
  - HF URL 中可能包含 token（私有仓库），需要确保 token 不被泄漏到 .torrent 文件中
  - per-file URL 映射需要与 torrent 内部文件路径一致（BEP 19 要求 URL 按 torrent 文件结构拼接）
  - HF CDN 可能对高频 Range 请求有限流
- **依赖**：无硬依赖（公有仓库可立即实现）；私有仓库需要 token 透传机制
- **实现复杂度**：低（核心只需一行 `add_url_seed`），收益极高

### ~~3.2 · 混合下载 — WebSeed~~（已合并入 2.12）

### 3.3 · 混合下载 — P2P 中断后 HTTP 续传
- **概念**：P2P 超时后启动 HTTP fallback 时，不从头下载，而是从 P2P 已下载的 offset 继续。
- **现状**：`patch.py` 在 fallback 时 `_truncate_temp_file()` 把已下载数据清零，传 `resume_size=0` 重新开始。
- **挑战**：
  - BT 是 piece-based 乱序下载，HTTP 是顺序流式。已下载的 piece 可能不连续，无法简单用 `Range: bytes=offset-` 续传
  - 需要将已完成的 piece 数据按顺序重组写入 temp_file
- **复杂度**：高，ROI 可能不如 webseed

### 3.4 · 进度条
- **现状**：下载进度只通过 logger 输出（每 5 秒一次 STATUS 日志）。
- **方案**：
  1. 使用 `tqdm`（已在依赖中）显示每文件下载进度
  2. 需要 monitor 线程将进度数据暴露给主线程
  3. 可以通过回调函数或共享状态实现
- **依赖**：monitor 解耦 (2.6)，因为进度数据需要从全局 monitor 中获取

### 3.5 · 服务端校验与安全
- **范围**：
  - 注册 torrent 时校验：info_hash 是否与 torrent 数据匹配、文件哈希是否与 HF Hub 一致
  - 防止恶意 torrent 注册（伪造 magnet link 分发恶意文件）
  - 客户端下载完成后校验文件 hash 是否与 HF Hub metadata 一致
- **方案**：Tracker 可调用 HF Hub API 验证 repo_id + revision 的真实性，或要求注册时提供签名

### 3.6 · 支持 huggingface_xet
- **现状**：`enable_p2p()` 直接禁用 Xet 引擎（`HF_HUB_DISABLE_XET=1`），强制所有文件走 `http_get` 以被 monkey patch 拦截。
- **长期方案**：
  - 不禁用 Xet，而是也对 `xet_get` 做 monkey patch
  - 或在 Xet 引擎上层拦截
- **复杂度**：Xet 是 HuggingFace 的内容寻址存储引擎，其下载路径与 HTTP 完全不同，需要深入分析

### ~~3.7 - 死代码清理~~
- `TrackerClient.announce()` 方法已标注 unused，libtorrent 内部自动处理 announce，可以移除
- 如果确认不会使用，清理掉以减少维护负担

---

## 版本规划建议

| 版本 | 包含内容 | 目标 |
|---|---|---|
| **v0.2** | P0 全部 + P1 全部 (1.1~1.6) | 核心正确性，日志/API 规范化 |
| **v0.3** | P2.1 ~ P2.5, P2.12 (WebSeed) | 服务端增强，架构改善，P2P 可靠性兜底 |
| **v0.4** | P2.6 ~ P2.11 | 架构解耦，性能优化，代码质量 |
| **v1.0** | P3 选择性实现 | 生产可用 |

---

## 已解决 / 存档的设计决策

### torrent v1 vs v2
- **结论**：使用 **v1_only** (`create_torrent.v1_only` flag = 64)
- **原因**：
  - v2/hybrid 模式下 libtorrent Python bindings 依然会生成 `.pad/` 虚拟文件
  - pad 文件不在 HF cache 中，导致做种时 piece hash check 全部失败
  - 已在 `torrent_creator.py` L80 实现并经过 E2E 验证
- **v2 的理论优势**（跨 swarm 文件共享）暂时无法在 Python bindings 中利用
