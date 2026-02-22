# 快速开始指南

## 安装

```bash
pip install llmpt-client
```

## 5 分钟教程

### 步骤 1：启用 P2P（环境变量）

最简单的启用 P2P 的方式是通过环境变量：

```bash
export HF_USE_P2P=1
export HF_P2P_TRACKER=http://your-tracker.com  # 可选
```

### 步骤 2：使用 HuggingFace Hub

```python
# 添加这一行（会自动检测环境变量并启用 P2P）
import llmpt

from huggingface_hub import snapshot_download

# 这将自动使用 P2P 下载
path = snapshot_download("gpt2")
print(f"下载到：{path}")
```

就这么简单！只需添加一行 `import llmpt`，你的下载现在已经是 P2P 加速的了。

## 替代方案：显式启用

如果你更喜欢在代码中启用 P2P：

```python
from llmpt import enable_p2p
from huggingface_hub import snapshot_download

# 启用 P2P
enable_p2p(tracker_url="http://your-tracker.com")

# 下载
path = snapshot_download("gpt2")
```

## 工作原理

1. **首次下载**：当你第一次下载某个模型时：
   - 通过 HTTP 下载（正常行为）
   - 自动创建种子
   - 在后台开始做种

2. **后续下载**：当其他人下载相同模型时：
   - 从节点通过 P2P 下载
   - 如果 P2P 太慢则降级到 HTTP
   - 下载后自动做种

## 配置

### 环境变量

```bash
# 启用 P2P
HF_USE_P2P=1

# Tracker 地址（可选，不设置则使用默认值）
HF_P2P_TRACKER=http://tracker.example.com

# 下载后自动做种（默认：1）
HF_P2P_AUTO_SEED=1

# 做种时长（秒，0 表示永久，默认：3600）
HF_P2P_SEED_TIME=3600

# P2P 超时时间（秒，默认：300）
HF_P2P_TIMEOUT=300
```

### Python API

```python
from llmpt import enable_p2p

enable_p2p(
    tracker_url="http://tracker.example.com",
    auto_seed=True,
    seed_duration=3600,  # 1 小时
    timeout=300  # 5 分钟
)
```

## CLI 使用

### 下载模型

```bash
llmpt-cli download meta-llama/Llama-2-7b
```

### 查看做种状态

```bash
llmpt-cli status
```

### 停止做种

```bash
# 停止所有做种
llmpt-cli stop

# 停止特定种子
llmpt-cli stop <info_hash>
```

## 故障排除

### P2P 不工作

检查 P2P 是否已启用：

```python
import llmpt
print(llmpt.is_enabled())  # 应该输出 True
```

### libtorrent 不可用

如果你看到 "libtorrent not available"：

```bash
# Windows
pip install python-libtorrent

# Ubuntu/Debian
sudo apt-get install python3-libtorrent

# macOS
brew install libtorrent-rasterbar
pip install python-libtorrent
```

### Tracker 无法访问

检查 tracker 连接：

```bash
curl http://your-tracker.com/api/v1/torrents
```

## 下一步

- 阅读[架构文档](ARCHITECTURE.md)了解工作原理
- 查看[开发指南](DEVELOPMENT.md)参与贡献
- 查看[示例代码](../examples/)了解更多用法

## 获取帮助

- GitHub Issues: https://github.com/yourusername/llmpt-client/issues
- 文档: https://github.com/yourusername/llmpt-client/docs
