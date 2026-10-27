# WAM 预览开发首次执行：结果序列化失败

首个 `task0_init0_s42_raw` 已完成 600 物理步和 75 次请求，progress 的 ever_success 为 false。随后采集器写入 worker_result.json 失败；整个开发未完成，不能授予开发或 fresh 资格。

## 已定位原因

`collect_coupled_preview_wam_development.py:174` 建立 images 列表，observe 将 render_anchor 返回值直接加入其中，随后在第 228 行附近赋予 result。继承的 `collect_wam_native_training.py:166` 将 NumPy source_pose 数组原样放入 current_source_pose。最终第 241 行 publish 使用严格 json.dump，抛出 `TypeError: Object of type ndarray is not JSON serializable`。残缺 JSON 停在 images[0].current_source_pose；不能把该文件当作成功凭据，也不能把早先读取失败继续解释为正在写入。

## 原始证据

根目录 `outputs/current_coupled_wam_development_20261026` 保持原样：

- `development/orchestration/task0_init0_s42_raw/command.log`：完整 600/75 进度及序列化 traceback。
- `development/orchestration/task0_init0_s42_raw/exit.json` 与 `ownership/release.json`：实际 wait、returncode 1、reaped true。
- `development/orchestration/service_process/exit.json` 与 `ownership/release.json`：服务因未完成 12/900 预算退出 1，实际回收。
- `supervisor/exit.json` 与 `release.json`：外层 owner 实际 wait 1/reap，无自动重试。三个原 PID/start_ticks 身份已不存活。
- `development/orchestration/result.json`：complete=false、passed=false，记录 gpu_after_ref。
- 原始物理 result、actual_trajectory.npz、端点、HTTP、请求和图像保留；尚未完成全量独立机械/数值审查。

## 后续处理边界

修复仅应将图像锚点中的数值数组无损转换为 JSON 类型，并在写入前验证整个结果可序列化。先用实际留存数组及相同 render_anchor 结构复现并验证修复，避免再次消耗模型/物理预算才发现输出错误。保留本次冻结源码、残缺文件、失败退出及全部实际请求；恢复需要显式新来源登记和独立验证，不覆盖失败凭据、不重启原 namespace，也不改变控制器、参考策略或性能门槛。完整开发仍要求 terminal 稳定至少 5/6 且每任务至少 1/2，最终 24 条新初态要求至少 20/24 且每任务至少 4/8。

## 已验证的序列化修复

新增 `src/s2a/preview_result_publication.py`，在创建目标文件前严格序列化整个对象，显式支持 NumPy ndarray/scalar，拒绝 NaN/Inf 和未知对象；以独占创建保留已有失败凭据。此措施针对序列化异常，不声称能保证磁盘写入故障时的原子性。

`tests/test_preview_result_publication.py` 的 6 项测试通过（2.35s），覆盖位姿数值往返、不合法测量不产生残缺文件、失败文件不覆盖和输出路径限制。首次路径限制测试错误地假定 pytest 临时目录在 outputs 外，实际仓库将其放在 outputs/pytest；修正测试中的 REPO 隔离后通过，未改生产边界。3 个新增 Python 文件 Ruff 检查与格式检查通过。

实际审查 `scripts/audit_preview_result_publication.py` 退出 0，使用本次 actual_trajectory.npz 的 0/16/56 帧 source_pose/opening 重现原 ndarray TypeError，再逐值核验 JSON 往返一致；原失败文件 SHA 保持。证据 `outputs/preview_result_publication_20261026/verified.json`，SHA `039a24af65290b10d750b228022cd63eb33ba637878a78ba98f5a4cd66ac2afa`。本次新增模型请求和物理步均 0。

重新核对原 prepared 的 1572 份源码 SHA，全部未变。修复函数尚未接入新注册采集入口；完整 worker 凭据恢复、独立机械/数值验证与剩余开发对照尚待完成，原失败运行未重启。
