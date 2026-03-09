"""
示例：启用 P2P 加速下载

调用 enable_p2p() 启用 P2P，环境变量 HF_P2P_TRACKER 等会被用作默认参数值。
"""

from llmpt import enable_p2p
from huggingface_hub import snapshot_download

# 启用 P2P（未指定的参数从 HF_P2P_TRACKER 等环境变量读取）
enable_p2p()

print("下载 gpt2 模型...")
path = snapshot_download("gpt2")
print(f"下载到：{path}")

# 检查做种状态
from llmpt.seeder import get_seeding_status
status = get_seeding_status()
print(f"\n活跃的做种任务：{len(status)}")
