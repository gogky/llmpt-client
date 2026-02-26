# LLMPT-Client 测试覆盖总结及 P2P 真实环境测试指南

本文档总结了当前 `llmpt-client` 测试集的覆盖范围，指出了尚未涉足的领域，并给出了一个设计和搭建“真实点对点（P2P）”继承测试的详细步骤指南。

---

## 1. 已完成的测试总结

目前测试架构分为**隔离单元测试（Unit）**和**真实集成测试（Integration）**两部分，共计 21 个核心测试用例，覆盖率从之前的 <5% 提升至近 40% 的业务主干逻辑。

### 1.1 单元测试（Unit Tests - 纯 Mock，无外部依赖，极速）
这些测试不会真正发起网络请求，而是通过注入（Mocking）行为来测试系统的每个逻辑分支是否按设计运作。

*   **`test_basic.py`**:
    *   **核心测试点**: `llmpt` 模块导入成功；`enable_p2p()` 和 `disable_p2p()` 是否正确启停功能；全局 Config 字典是否被正确修改；环境变量（如 `HF_USE_P2P`）是否被正确读取。
*   **`test_tracker.py`**:
    *   **核心测试点**: TrackerClient 的初始化与 URL 规范化（去除末尾斜杠）。
    *   **Mock 测试点**: 模拟了 Tracker 服务器 `http://118.195.159.242` 的各种返回。成功解析含有种子的 JSON 数组；在 `404 Not Found` 时的安全返回（`return None`）；在 Request 发生 Timeout 时的安全异常捕获返回；模拟 `register_torrent` (`/api/v1/publish`) 接口的 200/500 分发。
*   **`test_patch.py`**:
    *   **核心测试点**: 能够安全地替换和恢复 `huggingface_hub` 原生的下载函数，不破坏原始环境。
    *   **Mock 测试点**: 强行模拟 Hugging Face 调用了被劫持的 `patched_http_get` 函数，验证 `P2PBatchManager` 是否收到了准确的被拦截参数（`repo_id`, `filename` 等等），并在 P2P 失败的场合降级回执行原始 `http_get`。
*   **`test_batch.py`**:
    *   **核心测试点**: 验证 `P2PBatchManager` 是线程安全的单例（Singleton）模式。
    *   **Mock 测试点**: 验证当 P2P 触发时，`SessionContext` 能否正确地将 Tracker 拿到的磁力链塞入 `libtorrent.session.add_torrent()` 函数，且成功创建本地临时缓存路径供后续硬链接（Hardlink）。

### 1.2 集成测试（Integration Tests - 真实网络调用）
要求能够连通互联网。目前只用作兜底保障（Fallback）。

*   **`test_e2e_flow.py`**:
    *   **连通性测试**: 向真实的 Tracker 服务器 (`http://118.195.159.242`) 查询种子。
    *   **降级容错性测试**: 由于目前世界上没有其他机器在共享 `hf-internal-testing/tiny-random-GPTJForCausalLM`，测试验证了整个 llmpt 系统在请求 Tracker 扑空（返回 404/Empty）时，**不会导致用户请求失败**，而是静默地、稳健地通过源站 HTTP 完成下载，并且生成了最终能够被识别的 `config.json` 磁盘文件。

---

## 2. 还能进行哪些进一步测试？

目前我们尚未测试的领域，主要集中在底层环境的多样性以及一些独立的功能模块：

1.  **文件拼装与磁盘行为测试**: 
    在 Mock `libtorrent` 下载完成（进度 `100%`）的一瞬间，`_deliver_file()` 内部调用的 `os.link`（硬链接）或者回退拷贝 `shutil.copy2` 的动作。这部分与底层操作系统（Linux vs Windows 挂载机制）息息相关，目前的单元测试尚属空白。
2.  **种子生成模块（Torrent Creator）**:
    `llmpt/torrent_creator.py` 用于打包本地 HuggingFace 缓存为 `.torrent` 文件。我们还没有测试其路径遍历算法和 `libtorrent.create_torrent` 打包行为是否跨平台吻合。
3.  **做种者模块（Seeder）**:
    `llmpt/seeder.py` 作为服务端守护进程（Daemon）。没有对它的自动重试、心跳发包和异常关闭机制写具体的守护挂载测试。

---

## 3. 如何实施真实的 P2P 端到端（E2E）测试？

正如我们之前提到的，要在一个自动化脚本里验证真正的 P2P 传输（而不是 Fallback 回 HTTP），我们需要在同一台机器上或者 Docker Compose 容器群里“复刻”一整个局域网 P2P 宇宙。

以下是实现“真实 P2P 传输集成测试”的具体建议步骤与架构思路：

### 阶段一：搭建双节点沙盒基建

你要在 `tests/integration/` 下借助 pytest 这个框架，在一个测试 Session 中**同时启动两个客户端行为**。

1.  **准备环境目录**: 创建两个分别独立的根目录作为假象的电脑。
    *   `Downloader Node`: 只有空壳的 HF Cache。
    *   `Seeder Node`: 强行在磁盘里放入一个手动制作的 `dummy_model.bin`（例如 5MB 大小，全是随机字节），并仿照 HuggingFace 的目录结构把它假装缓存好。
2.  **构建真实的 Tracker 环境**:
    不使用线上服务器（避免网络波动）。建议在您的 CI pipeline 中用 Python 内置的 `http.server` 或者小型 FastAPI 甚至轻量级的开源 BT Tracker 单独在测试脚本里运行在一颗随机未被占用（如 `http://localhost:8899`）的接口上。

### 阶段二：脚本步骤流（以 `pytest` 用例设计）

1.  **启动 Dummy Tracker**:
    使用 `subprocess.Popen` 跑一个极简版的 Tracker，只包含 `/api/v1/publish` 和 `/api/v1/torrents` 路由。
2.  **启动做种 (Seeding Node)**:
    在 `Seeder Node` 的独立线程里，调用 `llmpt.seeder` 启动挂机模式。
    *   程序会遍历并扫描 `Seeder Node` 里的假模型，调用 `torrent_creator`。
    *   将生成的 `info_hash` 发布给刚跑起来的 `Dummy Tracker`。
    *   此时，Tracker 数据库里有了一条真实记录。
3.  **启动下载 (Downloading Node)**:
    设置 `HF_HOME=Downloader Node` 等隔离变量。在主线程里调用被 `llmpt.patch` 劫持过得原版 `snapshot_download()`。
    *   程序向 `Dummy Tracker` 询问。
    *   Tracker 不返回 404，返回了 `Seeder Node` 挂在的那个哈希！
4.  **监测传输**:
    `P2PBatchManager` 开始工作。因为两者都在 localhost，所以 `libtorrent` 会在本地端口发现对方！开始交换块（Pieces）。
5.  **严格断言 (Assertions)**:
    *   拦截器是否捕获到了网卡进站流量而拒绝了外部 HTTP 的 443 流出。（可以通过禁用 Python `requests.get` 来强迫证明只要通过了就是纯纯的 P2P）。
    *   最后，`Downloader Node` 中的缓存路径下出现了被硬链接好的 5MB `dummy_model.bin`。
6.  **清除进程**:
    关闭 Seeder，终结 Tracker，清理全套文件。

### 阶段三：为什么建议引入 Docker (可选)

由于 P2P 本身对 IP 端口极其敏感。在同一个宿主操作系统下开两个 `libtorrent` session 极容易出现端口互相抢占（哪怕一个 `listen_on(6881)` 另一个 `listen_on(6891)`，也容易打架）。
最好的解法是将这套“双端”逻辑放入一个 CI 控制的 `docker-compose.yml` 中：一个 Tracker 容器，一个 Seeder 容器，一个 Downloader 容器。用 Pytest 控制 Docker API 进行观测，这代表了绝对最健壮的网络环境模拟。

---

*只要您能完成上面这份流程的拼凑，您的 P2P 测试架构就完全从实验室级别到达了生产级别！*
