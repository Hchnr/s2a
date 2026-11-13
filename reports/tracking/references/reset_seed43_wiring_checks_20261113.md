# seed 43 复位闭环接线的阶段检查

2026-11-13。本阶段接入固定 seed 43 的历史复位准入、当前图像请求、模型
worker、producer 和独立 checker；进程监督与真实闭环尚未接通。

- 64 并发相关 CPU 回归 **117 项通过、0 skip、0 collection errors**，所有
  worker 已 wait/reap，测试前后来源一致。结果
  `outputs/qa/reset_seed43_wiring_pr_20261113/result.json` SHA-256
  `f42701a5cf600586e711f313e91938941234175ebc9117a672d825c3cd32017a`。
- 全仓 `ruff check src tests scripts` 和 `ruff format --check src tests scripts`
  通过。本阶段没有重跑全仓 pytest。
- 首次手工调用登记的模型解释器时未设置 `PYTHONPATH=src:.`，在导入 S2A 前
  以 `ModuleNotFoundError` 退出；补齐正式模型环境后，导入新 worker 成功，
  且准入前 `cv2`、`torch` 和 `cosmos_framework` 均未加载。

以上是 CPU 软件、保存输入及解释器导入证据；没有真实模型初始化、请求、
新物理步或最终留出案例，不证明任务完成、实时性或真机性能。
