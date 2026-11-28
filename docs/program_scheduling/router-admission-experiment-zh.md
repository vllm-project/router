# Router 准入与独立 Acting TTL 实验

基线为上游 `d226cf43`。本次在 macOS arm64 上运行实际 Rust 调度代码和两个监听本机回环端口的模拟推理后端，验证目标是分工、派发位置和容量行为。HTTP 用例从聊天请求处理入口进入，读取返回体中的后端地址核对实际发送位置。性能、显存占用和真实 KV 命中率需要另行测量。

## 这版代码怎样运行

`ProgramRouter` 持有 `ProgramAdmission` 和绑定 policy。普通 HTTP 的 Program 请求和 worker 注册同步都交给同一个编排器。原 `ProgramScheduler` 名称作为兼容入口。准入持有请求队列、Rank 容量、运行位置及按模型池和 Program ID 区分的绑定表。

| 事件 | 实际顺序 |
|---|---|
| 首请求 | 准入登记 → Router 向配置的策略取建议 → 准入检查目标在池内并保存绑定 → 容量及队首允许时签发凭证 → HTTP 向凭证目标发送。 |
| 正常续轮 | 准入登记并复用当前放置 → 取得凭证 → HTTP 发送。选点调用增加零次。 |
| 当前 Rank 下线 | 注册层同步 → 准入找出受影响 Program → Router 通知策略清除旧目标亲和记录并以原策略重选 → 准入更新位置和待办。即使没有新的 HTTP 请求也会推进。 |
| 一轮结束 | 按原尝试结算。普通混布开启 Acting TTL 时沿用拟合或请求提示；关闭时，已空闲且还会续轮的 Program 立即让出活跃名额，诊断原因为 `acting_retention_disabled`。排队、容量控制和后续恢复继续工作。 |

实验接入上游已有的五种同步绑定策略。它们在一个短的同步调度事务中进行选择和提交，等待容量和网络发送均发生在事务外。RFC 里为慢速 `kv_aware` 设计的锁外评分、认领号复核和 Program 专用 KV 首绑适配属于后续实现。本次使用模拟后端验证的分发策略为 `program_round_robin`。P/D 继续使用其专用的 TTL 设置。

## 如何开启

总开关沿用 `--enable-program-scheduling`。关闭 Program 功能时客户端走普通路由。开启后默认 `enable_acting_ttl: true`，已有配置保留原语义。以下参数可追加到普通 HTTP Router 的启动命令，得到“Program 绑定、容量准入和恢复开启，工具间隔不留座”的组合。

```sh
--enable-program-scheduling --policy '{"default":"cache_aware","program":"round_robin"}' \
--program-scheduling-config-json '{"binding_only":false,"enable_acting_ttl":false}'
```

## 实验与结果

| 验证项 | 观察结果 |
|---|---|
| 同一会话内两个 Program、同一个 Program 的两个并发首请求 | 两个线程同时提交 A。A 的两个尝试都到 R0，合计只选点一次；B 首次独立选到 R1。删除 R0 的拓扑通知单独触发 A 重绑；A 随后的请求到 R1，选点次数不再增加。该用例另行重复 30 次，30 次通过。 |
| 关闭 TTL 后能否绕过容量 | 单 Rank、一个活跃名额。A 在途时 B 等待。即使 A 的请求显式要求保留 60 秒，A 完成后立即暂停，计时器为空，B 获得同一 Rank。开启默认设置时，原回归验证请求级 TTL 仍然生效。 |
| `binding_only` 与完整准入 | 对同一组请求，`binding_only: true` 让 B 在 A 完成前取得凭证；`false` 让 B 等到 A 释放容量。 |
| 两个本地 HTTP 后端 | Program 关闭时普通请求正常发送。开启后，A 首轮和续轮的响应由同一后端返回，B 的响应由另一后端返回。移除 A 的后端后，A 的下一条响应由 B 所在后端返回。 |

| 相同环境的库回归 | 通过 | 失败 | 忽略 |
|---|---:|---:|---:|
| 上游基线 | 1493 | 1 | 3 |
| 实验分支 | 1497 | 1 | 3 |

两次完整回归都在 P/D 用例 `dispatch_failures_release_only_owned_reservation` 的 `/v1/chat/completions decode_transport` 分支得到同一差异，预期负载为 1、实际为 2。Program 专项从基线的 171 项增加到 174 项，均通过。第 4 个新增用例在 HTTP 模块中。格式检查和 Clippy 严格检查通过。

## 复现

使用 Rust 1.95.0，完整库测试需要本地回环端口和示例 WASM 组件的 `wasm32-wasip2` 构建目标。在仓库根目录运行：

```sh
cargo fmt --all -- --check
cargo clippy --locked --features cpu,ready --lib --tests -- -D warnings
cargo test --locked --features cpu,ready --lib program_scheduling
cargo test --locked --features cpu,ready --lib mixed_pool_http_routes_first
cargo test --locked --features cpu,ready --lib -- --test-threads 6
```
