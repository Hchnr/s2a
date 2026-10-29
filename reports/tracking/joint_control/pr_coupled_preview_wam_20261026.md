# PR 标题
feat: 接入四步预览 WAM 开发流程与独立验收

## 变更
正确参考控制已达到 18/18 准入门槛；本次将同一预览控制器接入真实 WAM 当前图像请求，支持固定 raw/terminal058 开发对照。

- 增加控制资格检查、请求身份与 Action128 协议校验、配置和执行注册。
- 完成模型服务、物理采集、进程监督，以及准备、metadata 和全量独立验证入口；绑定源码、输入和退出凭据。
- 固定 12 案例、900 请求、31500 去噪节点和 7200 物理周期；显式限制 GPU UUID，保留失败证据，禁止自动重试。
- 补充协议样本、组件测试及控制资格和实际执行报告。

## 验证与证据边界
- 正确参考独立物理仿真：18/18 稳定且通过位置/旋转误差门槛，共 4907 周期；不能计作 WAM 或真机成功。
- 相关 77 项 CPU 测试通过；新增 27 个 Python 文件 Ruff 检查和格式检查通过。
- 12 案例输入预检、独立 metadata、14 个实际 CLI dry 均通过；首次准备因解释器缺少 h5py 失败，已改用已有原生 Python 3.11 完成，失败日志保留。
- 前 3 个真实 WAM 请求的独立图像/协议/归一化检查通过。完整 WAM 开发和最终留出初态验收尚无完成结论；开发门槛仍为 terminal 稳定至少 5/6、每任务至少 1/2。
- 全仓库 Ruff 仍有既有问题：validate_temporal_response.py 导入排序、audit_rotation_goal_precision.py 格式。本次未重跑全量 pytest，不宣称全仓测试通过。

运行产物保留于 outputs/，不纳入提交。实现和运行配置保持已注册的冻结字节。

---

# PR title
feat: integrate four-step preview WAM development and independent acceptance

## Changes
Correct-reference control has passed the 18/18 admission gate. This change connects the same preview controller to live WAM image queries for a fixed raw versus terminal058 development comparison.

- Add physical qualification, request identity and Action128 validation, configuration, and execution registration.
- Implement model serving, physics collection, process supervision, preparation, metadata checks, and full independent verification with source, input, and process-exit evidence.
- Fix the budget at 12 cases, 900 requests, 31,500 denoising nodes, and 7,200 physics steps; enforce the allowed GPU UUID, preserve failures, and prohibit automatic retries.
- Add protocol fixtures, component tests, and qualification and execution reports.

## Validation and limitations
- Correct-reference independent simulation: 18/18 stable and within position/rotation error thresholds over 4,907 steps. This is not WAM or real-robot success.
- All 77 related CPU tests pass. Ruff lint and formatting checks pass for all 27 added Python files.
- Input preflight for all 12 cases, independent metadata verification, and 14 actual CLI dry runs pass. Initial preparation failed because its interpreter lacked h5py; preparation subsequently completed with the existing native Python 3.11 interpreter, preserving the failure log.
- Independent image, protocol, and normalization checks pass for the first three live WAM requests. Full WAM development and held-out initial-state acceptance remain unqualified; the development gate remains at least 5/6 stable terminal cases and at least 1/2 per task.
- Repository-wide Ruff retains existing import-order and formatting issues in validate_temporal_response.py and audit_rotation_goal_precision.py. The full pytest suite was not rerun; no repository-wide passing claim is made.

Generated artifacts remain under outputs/ and are excluded from this commit. Registered implementation and runtime configuration bytes remain frozen.
