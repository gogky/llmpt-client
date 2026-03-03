# llmpt-client

HuggingFace Hub 模型的 P2P 加速下载客户端

## 特性

- 🚀 **无缝集成**：无需修改代码，直接替换 `huggingface_hub` 下载
- 🌐 **P2P 加速**：通过 BitTorrent 从其他用户处下载模型
- 📦 **自动降级**：P2P 失败时自动降级到 HTTP
- 🔄 **自动做种**：下载完成后自动做种，帮助其他用户
- 💾 **断点续传**：支持恢复中断的下载
- 🎯 **最小代码修改**：只需添加一行 `import llmpt` 即可启用 P2P

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

1. **首次下载**：当你第一次下载某个模型时：
   - 检查 tracker 是否有现有种子
   - 如果没有，通过 HTTP 下载
   - 自动创建种子并开始做种
   - 上传种子元数据到 tracker

2. **后续下载**：当其他人下载相同模型时：
   - 从 tracker 获取种子信息
   - 从多个节点下载（P2P）
   - 如果 P2P 太慢或失败，降级到 HTTP
   - 下载完成后自动做种

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
from llmpt import enable_p2p, disable_p2p, stop_seeding

# 使用自定义设置启用
enable_p2p(
    tracker_url="http://your-tracker.com",
    auto_seed=True,
    seed_duration=3600,  # 做种 1 小时
    timeout=300
)

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
llmpt (Monkey Patch)
    ↓
    ├─→ 查询 Tracker：种子是否存在？
    │   ├─→ 是：P2P 下载（libtorrent）
    │   └─→ 否：HTTP 下载 → 创建种子 → 做种
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
pytest tests/
```

```bash
docker compose -f docker-compose.test.yml down
docker compose -f docker-compose.test.yml up --build
```

### 项目结构

```
llmpt-client/
├── llmpt/                    # 主包
│   ├── __init__.py           # 入口点，Monkey Patch
│   ├── patch.py              # Monkey Patch 实现
│   ├── tracker_client.py     # Tracker API 客户端
│   ├── downloader.py         # P2P 下载器（libtorrent）
│   ├── seeder.py             # 做种管理器
│   ├── torrent_creator.py    # 种子创建
│   └── utils.py              # 工具函数
├── tests/                    # 测试
├── examples/                 # 示例脚本
├── docs/                     # 文档
└── setup.py                  # 安装配置
```

## 兼容性

- ✅ Windows
- ✅ Linux
- ✅ macOS
- ✅ Python 3.8+

## 许可证

MIT License

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 故障排除

### libtorrent 安装失败

如果 `python-libtorrent` 安装失败：

```bash
# Ubuntu/Debian
sudo apt-get install python3-libtorrent

# macOS
brew install libtorrent-rasterbar
pip install python-libtorrent

# Windows
# 从 https://github.com/arvidn/libtorrent/releases 下载预编译的 wheel
pip install python_libtorrent-2.0.9-cp311-cp311-win_amd64.whl
```

### P2P 不工作

1. 检查 P2P 是否已启用：
```python
import llmpt
print(llmpt.is_enabled())
```

2. 检查 libtorrent 是否可用：
```python
try:
    import libtorrent
    print("libtorrent 可用")
except ImportError:
    print("libtorrent 不可用")
```

3. 检查 tracker 连接：
```bash
curl http://your-tracker.com/api/v1/torrents
```

### P2P 下载失效（退化为纯 HTTP 下载）

如果你在 Python 脚本中调用，但发现 P2P 完全没起作用（甚至没有尝试），请确认：

**`llmpt.enable_p2p()` 必须在任何下载调用（`snapshot_download` / `hf_hub_download`）之前执行。**

`import` 的顺序不影响补丁效果——无论 `huggingface_hub` 在 `llmpt` 之前还是之后导入，补丁都会在 `enable_p2p()` 调用时正确注入到对应的子模块命名空间中。


## 路线图

- [x] 基础 P2P 下载
- [x] 自动创建种子
- [x] 自动做种
- [ ] DHT 支持
- [ ] 多 tracker 支持
- [ ] 做种管理 Web UI
- [ ] 下载统计

## 致谢

- [HuggingFace Hub](https://github.com/huggingface/huggingface_hub) - 官方 Python 客户端
- [libtorrent](https://www.libtorrent.org/) - BitTorrent 库
