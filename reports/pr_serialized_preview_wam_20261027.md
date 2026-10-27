# PR 标题
fix: 修复 WAM 结果序列化并增加可追溯恢复流程

## 变更
WAM 采集完成后，图像锚点中的 NumPy 位姿数组导致 worker_result.json 序列化失败。本次先将结果完整转换为严格 JSON，再独占创建文件，支持 NumPy 数值，拒绝 NaN/Inf 和未知类型，并保留已有失败凭据。

- 增加独立 serialized_preview_wam 准备、注册、服务、采集、监督和验证入口；保持控制算法、模型参数和固定 12 案例门槛。
- 绑定原失败运行的源码、退出、GPU 释放和实际数据修复证据；原 75 请求/600 周期单独记账。
- 增加实际位姿往返审查、发布边界测试、派生代码 AST 对照及执行报告。运行产物保留在 outputs/，不纳入提交。

## 验证与限制
- 84 项相关 CPU 测试通过；24 个新增 Python 文件 Ruff 检查和格式检查通过。
- 三帧实际保存位姿 JSON 往返逐值一致；12 案例输入预检、独立 metadata 和 14 个实际 CLI dry 均通过。
- 实际运行前 8 个案例完成采集并成功发布结果，证明序列化故障已修复；8 条均未成功完成任务。
- 第 9 条在 533 步后因 Dependent QP blocking constraint 停止，其余 3 条未执行。全量独立验收未完成、开发门槛未通过、fresh 未运行。
- 子进程退出/回收和 GPU 释放凭据存在；外层 supervisor 的 exit/release 凭据缺失，生命周期闭合仍待核实。
- 本次未重跑全仓 pytest，不宣称全仓检查通过。原有全仓 Ruff 问题不在本次修改范围。

---

# PR title
fix: serialize WAM results safely and add traceable recovery execution

## Changes
A NumPy pose array in an image anchor caused worker_result.json serialization to fail after physics collection. Serialize the complete result to strict JSON before exclusively creating the destination, support NumPy values, reject nonfinite and unsupported values, and preserve existing failure evidence.

- Add separate serialized_preview_wam preparation, registration, serving, collection, supervision, and verification entry points while retaining controller math, model parameters, and the fixed 12-case gate.
- Bind the failed run's source, exit, GPU-release, and actual-data repair evidence. Account separately for its 75 requests and 600 physics steps.
- Add measured-pose roundtrip auditing, publication boundary tests, AST derivation checks, and execution reports. Generated artifacts remain under outputs/ and are excluded from the commit.

## Validation and limitations
- All 84 related CPU tests pass. Ruff lint and formatting checks pass for the 24 added Python files.
- Three saved measured poses roundtrip through JSON with exact value equality. All 12 input preflights, independent metadata verification, and 14 actual CLI dry runs pass.
- The first eight runtime cases completed collection and successfully published results, confirming the serialization repair. None achieved task success.
- Case nine stopped after 533 steps with Dependent QP blocking constraint; the remaining three cases were not executed. Full independent acceptance is incomplete, the development gate has not passed, and fresh evaluation has not run.
- Child exit/reap and GPU-release receipts exist; outer supervisor exit/release receipts are missing, leaving its lifecycle closure unverified.
- The full repository pytest suite was not rerun. Existing repository-wide Ruff issues remain outside this change.
