# 2026-11-07 VM(1LCGODPP) — v7 电池改道排队记录

## 会话起点状态（10:00 +08:00 cron 检查的结果）

- 任务要求：拉 `/content/rebuild7.log` + `/content/ebat2.log` 判定 E1-E5。
- **实际情况：两份日志都不存在。** 隧道 UYJ23XUZ 已失效；用户提供的现役隧道
  **1LCGODPP** 连接的 VM 上跑的是 **rebuild6.sh（v6 流水线）**，v7 从未在此 VM 启动过。
- VM 状态（02:04 UTC 快照）：
  - Tesla T4 15GB，磁盘 57G/113G used
  - stage 标记：s1 ✅ s3 ✅ s4 ✅（S2 下载并入 s4 快照门）；**s5 无**（转换中）、**s6 无**
  - S5 转换器运行中：`convert_hf_to_sager.py --workers 8 --fast`，GGUF 7.25GB/约44.5GB，
    正在处理 shard 4/14，速率约 0.25 GB/min → 转换预计 04:40 UTC 完成
  - `hf_sage/tokenizer.json` 存在（ebat 脚本需要的 /content/sage_tokenizer 尚不存在）

## 关键风险（为何不能等 stage_s6_done）

本 VM 的 rebuild6.sh S6 池门是**旧版未对齐采样**（`f.seek(size*frac)`，无 root-28 的
144 字节对齐）→ 按根因 28 的结论必然产生 2.1-2.5% 假 NaN → `assert worst==0.0` 失败 →
**假报 POOL_GATE_FAILED 并 exit 1，stage_s6_done 永远不会写**。

## 已采取的动作（02:05 UTC）

`scripts/launch_ebat3_waiter.py` 已投递并启动 `/content/ebat3_waiter.sh`
（日志 `/content/ebat3_waiter.log`，02:04:55 UTC 启动）：

1. 等 `pool.bin ≥ 17GB` 且无 `v12_quality|sager_engine|convert_hf` 进程（6h 超时）
2. **自行跑 root-28 对齐版池门**（不信任 rebuild6 的旧门）
3. 从 `hf_sage/tokenizer.json` 引导 `/content/sage_tokenizer/`
4. 跑 `/content/ebat.py`（E1/E2/E3，内容从 launch_rebuild_v7.py 原样提取）
   → `/content/ebat.log`：E1 graphs=False+设备解码采样；E2 全主机；E3 主机+DUMP_LOGITS
   raw prompt [760,6511,314,9338,369]，参考 Paris id 11751 logit ≈ **18.572**
5. 跑 `/content/ebat2.py`（E4/E5，从 launch_ebat2.py 原样提取）→ `/content/ebat2.log`
   E4=最小运行时（主机+无suffix草稿）；E5=设备胶水去验证窗
6. `touch /content/stage_ebat_done`

## 判定矩阵（不变）

| 结果 | 结论 |
|---|---|
| E1 净 & E2 净 | CUDA graph 重放胶水是病灶 |
| E1 汤 & E2 净 | 设备解码/设备采样胶水 |
| E1 汤 & E2 汤 & E3 坏 | 前向内核 / 权重加载路径 |
| E1 汤 & E2 汤 & E3 好 | 主机路径采样/logits 管道 |
| E4 净 | 运行时胶水；E2 vs E4 切分 suffix/verify 窗 |
| E4 汤 | 前向内核 / 权重（E3 定 prefill vs decode 侧） |
| （S6 quality graphs=1 dev=1 suffix=1） | 已知词汤基线 |

## 预计时间线（UTC）

- ~04:40 转换完成 → S6 池冷填+quality ~40min → ~05:20 对齐门+E1-E3 → ~05:50 E4/E5
- 全部结果约 **06:15-06:30 UTC（14:15-14:30 +08）**
- 已建一次性 cron **510772**：15:30 +08:00 拉三份日志、按矩阵判定、留档到本目录、更新 WORKLOG。

## rebuild6.log 尾部摘录（02:06 UTC，转换进行中）

```
[convert]   released /content/hf_sage/model-00001-of-00014.safetensors
[convert]   released /content/hf_sage/model-00002-of-00014.safetensors
[convert]   released /content/hf_sage/model-00003-of-00014.safetensors
[dbg] ffn_down_exps: mats=(256, 2048, 512) bytes=150994944
[dbg] ffn_gate_exps: mats=(256, 512, 2048) bytes=150994944
[dbg] ffn_up_exps: mats=(256, 512, 2048) bytes=150994944
gguf_now= 7247797952
```
