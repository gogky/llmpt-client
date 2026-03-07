- 混合下载
    - 层次2： p2p中断后，http能否接上
- 分片大小的选择
- 服务端存储.torrent/为.torrent添加缓存/自动做种
- 服务端的校验与安全问题

- timeout时间与meta data等待时间变为可配置
- monitor守护进程与session不一一对应
- 进度条
- 同一模型的多个版本。跨swarm问题？或许通过tracker服务器返回旧版本peer并注明相同文件来实现。
- 6881端口需要动态
- 支持huggingface_xet
- 通过缓存.fastresume 或后台生成跳过验证阶段


- 构造更多端到端测试场景
- 是否支持同时做种和下载

- seeding_mapper去掉pad逻辑



目前monitor+session_context的设计合理吗？职责分配合理吗（比如monitor要负责检查是否有.torrent错误），代码中这么多轮询有没有性能问题？

当前代码库中，使用huggingface的api调用，如果下载完成后没有退出，会自动制作.torrent并上传到tracker服务器吗？

- 第一个制作.torrent的人是个问题


- 用户存储模型的顺序不一样的问题。
- 系统集成


# 场景 4：守护进程自动做种（冷启动 → 自动创建 torrent → P2P 下载）
docker compose -f docker-compose.test-daemon.yml up --build

auto_seed是否可以取消了，这是下载端与做种端没有解耦前的设计。因为目前有了做种端可以单独存在，所以其实强制auto_seed的，下载端下载完成后直接退出就好，让做种端接管，对吗？你认为是否可以取消auto_seed和seed_duration变量。
-  ipv6
- 下载端和做种端可能端口冲突