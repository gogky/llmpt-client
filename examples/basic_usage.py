"""
示例：使用环境变量启用 P2P

注意：即使使用环境变量，也必须 import llmpt 才能触发 Monkey Patch
"""

import os

# 启用 P2P（通过环境变量）
os.environ['HF_USE_P2P'] = '1'
os.environ['HF_P2P_TRACKER'] = 'http://localhost:8080'

# 导入 llmpt（会自动检测环境变量并启用 P2P）
import llmpt

# 使用 huggingface_hub（无需其他修改）
from huggingface_hub import snapshot_download

print("下载 gpt2 模型...")
path = snapshot_download("gpt2")
print(f"下载到：{path}")

# 检查做种状态
from llmpt.seeder import get_seeding_status
status = get_seeding_status()
print(f"\n活跃的做种任务：{len(status)}")
