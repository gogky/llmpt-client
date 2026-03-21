# llmpt-client 架构设计

## 概览

`llmpt-client` 是 Hugging Face Hub 下载流程上的一个加速层。

它的核心目标不是替代 `huggingface_hub`，而是：

- 在不改变用户下载接口的前提下，为文件下载增加 P2P 能力
- 在 P2P 不可用时保持 WebSeed 和原始 HTTP 的可回退性
- 在下载完成后把本地已验证的数据持续转化为可做种资源

从整体上看，系统由两个长期存在的角色组成：

- **下载端**：跟随用户进程运行，负责拦截文件下载、规划来源、执行传输
- **守护进程**：独立后台运行，负责扫描本地缓存、注册 torrent、持续做种

## 架构原则

当前实现围绕几个原则展开：

1. **保持 Hugging Face 原生行为为主**
   `huggingface_hub` 仍然负责 revision 解析、缓存组织、文件元数据获取和最终文件交付语义。

2. **按文件做传输决策**
   下载路径的最小调度单位是“单个文件”，而不是整个仓库。

3. **按 swarm 复用连接资源**
   文件可以各自选择来源，但底层连接和 `torrent_handle` 以 source swarm 为单位复用，而不是一个文件一个 handle。

4. **失败可回退**
   P2P、旧 swarm 复用、WebSeed 都是增强路径；失败时必须能自然回退到标准 HTTP。

5. **做种与下载解耦**
   下载端追求快速交付，长期做种交给 daemon，避免把用户前台进程变成复杂的长期运行节点。

## 主要运行时组件

### 1. 用户入口层

入口由 `enable_p2p()` / `disable_p2p()` 提供。

这一层负责：

- 初始化全局配置
- 启动可选的 WebSeed proxy
- 启动或连接后台 daemon
- 对 `huggingface_hub` 的关键下载入口打补丁

这一层的职责是“把 P2P 能力接到现有下载体验上”，而不是自己实现下载协议。

### 2. Patch 与请求接入层

patch 层把 `huggingface_hub` 的文件下载请求引流到 `llmpt`。

它负责：

- 记录当前文件下载的上下文
- 在真正执行 HTTP 下载前尝试 P2P 交付
- 统计本次下载中哪些文件走了 P2P、哪些文件走了 HTTP
- 在下载结束后通知 daemon 接手做种

这里的关键设计点是：

- patch 层负责“接入”和“回退”
- 不直接承担完整的调度和传输逻辑

### 3. 传输规划层

传输规划层负责回答一个问题：

**“当前这个目标文件，应该从哪里下载？”**

这一层使用显式的数据模型来区分：

- 目标文件是谁
- 来源 torrent 是谁
- 实际来源文件路径是谁
- 最终执行计划是什么

当前的来源选择策略以 exact revision 为主，并在必要时允许“同仓库旧 revision 的同内容文件”作为候选来源。

这里的一个重要边界是：

- 当前是**单文件、单 source** 的规划
- 不是多 swarm 协同下载

### 4. 执行层

执行层负责真正驱动 `libtorrent`：

- 维护全局 `lt.session`
- 为不同 source swarm 创建或复用 `torrent_handle`
- 对目标文件设置优先级
- 可选挂接 WebSeed
- 监控 piece / file 完成并完成本地交付

这里最重要的抽象是：

- **文件是调度单位**
- **swarm 是连接单位**
- **session / handle 是执行单位**

因此系统不是“一个文件一个 handle”，而是“多个文件可映射到少量 source swarms，再由这些 swarms 复用 handle”。

### 5. 守护进程与做种层

daemon 是系统的长期运行角色。

它负责：

- 扫描默认 HF cache、自定义 `cache_dir`、`local_dir`
- 验证本地数据是否完整可做种
- 创建并注册 torrent
- 长期保持 seeding 会话
- 在进程重启后恢复已知的做种任务

这样下载端可以专注于“本次下载是否成功交付”，而不需要长期保留大量做种状态。

### 6. Tracker / 元数据层

客户端会与 tracker 交互两类信息：

- **torrent 元数据**
  - torrent 注册
  - `.torrent` 获取
- **来源发现信息**
  - 某个目标文件有哪些可用来源

这层的职责不是“替代 BitTorrent peer discovery”，而是提供更高层的内容发现和调度信息。

## 端到端数据流

### 下载流

一个典型下载请求的路径可以概括为：

```text
snapshot_download / hf_hub_download
  -> patch 层拦截单文件请求
  -> 传输规划层选择来源
  -> 执行层通过 libtorrent 拉取数据
  -> 成功则直接交付文件
  -> 失败则回退到 WebSeed 或原始 HTTP
```

这里有两个特点：

- 选择来源发生在“文件级”
- 数据传输仍然尽量复用标准 BT / WebSeed / HTTP 机制

### 做种流

下载完成后的数据会通过 daemon 转化为长期可做种资源：

```text
文件下载完成
  -> 下载端通知 daemon
  -> daemon 验证本地数据
  -> 创建并注册 torrent
  -> 启动长期做种
```

这个流程保证了“消费数据”和“回馈网络”是连续的，但不要求用户前台进程一直存在。

## 存储与状态

系统需要同时处理三类状态：

- **Hugging Face 原生缓存状态**
  - revision、blobs、metadata、local_dir 展开目录
- **P2P 运行时状态**
  - session、handle、peer、piece 进度、resume data
- **做种恢复状态**
  - 哪些目录已知可恢复
  - 哪些 torrent 已注册

因此架构上专门区分了：

- 逻辑身份：仓库 / revision 的身份
- 存储身份：文件实际位于哪个缓存根或本地目录
- 传输身份：当前执行这次下载的是哪个 source swarm

这种区分的意义在于：

- 同一个仓库 revision 可以对应不同存储位置
- 同一个目标文件可以从不同的 source swarm 获取
- 但这些关系不能混成一个 tuple 到处传

## 当前能力边界

当前架构已经支持：

- exact swarm 下载
- WebSeed 冷启动
- 自动 HTTP 回退
- 后台 daemon 做种
- 自定义 `cache_dir` / `local_dir`
- 同仓库旧 revision 的单文件复用

当前还**没有**支持：

- 多 swarm 协同下载同一个文件
- repo 级全局 swarm 规划


## 测试策略

当前测试分成三层：

- **单元测试**
  验证 patch、传输模型、session 身份、tracker client、协调逻辑等局部行为
- **Docker E2E**
  验证纯 P2P、WebSeed、混合模式、daemon 路径，以及旧 swarm 单文件复用
- **真实 tracker / 仓库验证**
  用真实 Hugging Face 仓库和真实 tracker 验证实际可行性

其中旧 swarm 场景的 E2E 重点验证的是：

- exact target candidate 存在但没有 seeder
- 旧 revision candidate 可用
- 下载端最终选择旧 swarm
- WebSeed 关闭且运行时统计不显示 URL seed 流量

## 目录导览

如果只从宏观理解代码结构，可以把主目录理解成下面几类：

- `llmpt/__init__.py`
  对外入口和全局初始化
- `llmpt/patch*.py`
  下载接入、上下文、统计和展示
- `llmpt/transfer_*.py`
  传输规划和数据模型
- `llmpt/p2p_batch.py`、`llmpt/session_context.py`、`llmpt/monitor.py`
  执行层与 libtorrent 集成
- `llmpt/daemon.py`、`llmpt/cache_*`、`llmpt/seeding_mapper.py`
  后台做种与恢复
- `llmpt/tracker_client.py`、`llmpt/torrent_*`
  tracker / torrent 元数据交互

