# 预览 WAM 序列化修复后的独立开发入口

原运行已在第一个 raw 案例完成 600 步/75 次请求后因结果序列化失败而退出；原始模型/物理数据、残缺 worker_result 和退出凭据保持原样。新入口尚未启动模型，完整开发资格和 fresh 资格仍未获得。

## 实现与保持项

新 `serialized_preview_wam` 文件族显式派生原冻结实现，使用 `outputs/serialized_preview_wam_development_20261026`、独立 preparation 目录及 `s2a.serialized-preview-wam-development` schema。采集器只将最终 worker_result 发布改为 `preview_result_publication.publish_result`，其余控制和模型执行逻辑不变。复用原只读协议、独立测量与物理前置资格检查，不覆盖原文件。

`tests/fixtures/serialized_preview_derivation.json` 记录 19 份原文件 SHA、派生文件及 namespace 替换映射；AST 检查核对全部派生文件仅存在这些声明变化、发布函数接入及 QA 清单调整。Ruff 重排连续 import 组时允许组内排序差异，但不得跨越任何执行语句或 guard preload。原 prepared 的 1572 份源码 SHA 再核查全部未变。

固定开发仍是 12 案例/900 请求/31500 节点/7200 物理周期，同一控制器、checkpoint、052/058 参数、seed 和成功门槛。由于服务要求一次初始化固定完整矩阵及 raw/terminal 成对首图，不将原失败运行的第一案例拼接成成功执行；后续新运行将重新执行 12 条，原失败额外消耗的 75 请求/2625 节点/600 周期单独记账。此为显式故障修复后的新实验，禁止自动重试和依据前缀调参。

## 已执行

- 序列化实际数据检查退出 0，0/16/56 三帧位姿逐值往返一致，见原故障报告。
- 新派生组件及发布函数相关 47 项测试通过；派生全文件 AST 检查在处理 Ruff 连续导入重排后单独通过（1 项）。此前 AST 失败仅为导入组排序差异，未放宽可执行语句比较。
- 新增与派生共 24 个 Python 文件 Ruff check 和 format --check 通过。
- 原生 Python 3.11 实际执行新 12 案例输入 preflight 退出 0；证据 `outputs/serialized_preview_wam_preparation_20261026/input_preflight_py311/result.json`，0 模型调用、0 物理步、0 array decode。

## 尚待完成

新准备清单仍需显式绑定原失败的来源/退出/GPU 释放证据、修复派生映射和实际数据验证结果；完成新完整 QA、源码冻结、独立 metadata、注册及 14 个实际 CLI dry 后才可启动。旧运行不会恢复或覆盖。最终开发仍须 terminal 至少 5/6 稳定且每任务至少 1/2，随后才可登记最终 24 条留出新初态验收（至少 20/24 且每任务至少 4/8）。

## 新准入完成及实际启动

上述尚待项中的准备、QA、注册与启动现已执行：

- 失败来源审查绑定原 prepared SHA、失败 owner、三个实际 wait 1/reap/进程消失记录、原 GPU 释放及实际序列化修复证明，共 18 份引用；新 prepared 输入表与独立 metadata 均重新核对。原失败预算单列。
- 完整 84 项 CPU 测试通过（20.29s），记录 `outputs/serialized_preview_wam_preparation_20261026/component_qa.json`。
- 新准备冻结 1577 份源码。`outputs/serialized_preview_wam_development_20261026/prepared.json` SHA `b1f2b793bc5818994a5de2c33a78c8561830469c0db1bcf789634eb9bfcd6c43`；原生 Python 3.11 独立 metadata 实际退出 0。
- 新 registration SHA `77dc9c6c66138a17326a0708a4496b699221ea12e0d2eac8924e4db7da6505b9`；owner、模型解释器 service 与 12 采集入口共 14 个实际 CLI dry 全部退出 0。
- 启动前查询允许 UUID `GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f` 为容器 index2、PCI BA:00.0、0 MiB 且无计算进程。查询 minor_number 字段不受当前 nvidia-smi 支持，随后以支持字段重查；未更改既有 UUID/minor6 的绑定。
- 外层 `durable_stage_supervisor.owned_stage` 持有实际 owner，监督目录为新根目录的 supervisor，负责最终 wait/reap。owner PID1154746/start_ticks595504098、service PID1154841/start_ticks595504917 经 /proc 确认同身份存活，owner ready 握手已完成。模型服务初始化尚未据此宣称成功。

从此保持新 1577 份注册源码冻结。新完整 WAM 结果、900 请求/31500 节点独立验收和 fresh 均仍未完成；启动不是任务成功。

## 模型 ready 与首案例启动

新 `development/orchestration/service_readiness.json` 已记录 ready=true、model_initializations=1、model_initialization_attempts=1；实际服务身份 PID1154841/start_ticks595504917，CUDA_VISIBLE_DEVICES 为允许 UUID，LD 为空。ready 时计数为 0 请求/0 节点，仅代表服务准入时点。

首个 raw 采集器 PID1155434/start_ticks595510420 已启动，和 owner/service 均经 /proc 核实同身份存活；服务日志显示实际 35 节点采样正在推进。尚无完整案例 worker_result，不以 ready 或采样前缀判定任务成功。

## 2026-10-27 提交前核实：运行已停止

原 12 案例开发未完成。前 8 个案例均完成 600 步/75 请求并成功写出完整 worker_result，说明原 NumPy 序列化故障已在实际采集中解决；这些 passed 字段仅表示采集流程完成。8 条物理 result 均 ever_success=false、stable_success=false，不能视为闭环成功，也尚未完成正式全量独立审查。

第 9 条 task2_init0_s42_raw 在 533 步、67 个 chunk 后失败，错误为 `Dependent QP blocking constraint; original state retained`，来自 `src/s2a/strict_convex_qp.py:60`。新发布函数成功保存了该失败的 JSON 结果。已记录物理周期合计 5333、查询 chunk 合计 667；这些是案例记录统计，完整服务账本与逐节点审查尚未完成。其余 3 条未执行。

监督、模型服务及已启动采集进程的原 PID/start_ticks 身份均已不存活。服务与失败采集器的实际退出和回收记录存在，gpu_after 显示允许 UUID 显存 0 且无 compute_apps。但外层 supervisor/exit.json 与 release.json 缺失，不能声称外层完整 wait/reap 生命周期闭合；此前聊天中断后外层工具会话的最终状态尚未恢复核实。失败 owner 记录保留，未自动重启。

提交前重核 1577 份冻结源码哈希全部相符。开发门槛未通过，fresh 未运行，总目标未完成。后续需定位 QP 约束退化并评估参考质量，不能只修复输出格式后宣称任务完成。
