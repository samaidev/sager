# 2026-11-07 15:30 +08 cron（Job 510772）拉取失败记录

## 结果：三份日志均未能拉取 —— 隧道 1LCGODPP 离线

- 07:30 UTC（15:30 +08）执行拉取：
  - `/content/ebat3_waiter.log`（全文）→ HTTP 502 Bad Gateway
  - `/content/ebat.log` 尾 6000B、`/content/ebat2.log` 尾 4000B → 未尝试（通路已死）
- `/status` 明确返回：`{"error":"Tunnel offline","message":"tunnel 1LCGODPP is currently offline"}`
- 间隔 8s 重试 4 次，全部 offline —— 非瞬时抖动。
- 按任务协议：**停止，等待用户提供新隧道码，不自行猜测**。
- 无法确认 `/content/stage_ebat_done` 是否已写；ebat3 waiter 与
  转换/S6 流水线在 VM 侧的状态未知（最后已知快照 02:04-02:06 UTC：
  convert shard 4/14，预计 04:40 UTC 转换完成）。

## 恢复通路后的待办（不变）

1. 用新码拉 waiter.log 全文（判排队：pool-wait / aligned gate / POOL_GATE_OK_ALIGNED）
2. ebat.log 尾 6000B（E1/E2/E3 文本 + step_logits）
3. ebat2.log 尾 4000B（E4/E5 文本）
4. 按判定矩阵下结论（见 2026-11-07-1LCGODPP-ebat3-queue.md），E3 参考：
   raw prompt [760,6511,314,9338,369]，Paris id 11751 logit ≈ 18.572
5. 三份日志存入本目录，WORKLOG 追加结论
