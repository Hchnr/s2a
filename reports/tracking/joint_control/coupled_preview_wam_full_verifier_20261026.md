# WAM预览全量独立审查接入

新增`verify_coupled_preview_wam_development.py`、`coupled_preview_wam_audit_admission.py`和`coupled_preview_wam_measured_audit.py`。独立审查链接入新schema/namespace/服务端口、完整18条控制前置资格、060模型state资格及新预览机械checker；不导入生产控制gate、服务或owner来代替独立判定。

保留原12案例/900请求/31500去噪节点完整数值审查、同首图与采样初态配对、实际wait/reap/GPU释放、固定checkpoint/参数和来源核验。新worker的实测数组、601端点receipt、控制输入与raw-state绑定、同步标签和三张锚点图也进入全量审查。新控制参数逐项与通过18条资格的配置核对，不能沿用旧raw8控制器字段比较而忽略新算法。

验证结果：

- 新verifier和独立checker共12项CPU测试通过（0.83秒），覆盖原5/6开发门槛、拒绝部分/伪造任务结果、旧schema拒绝和全部每请求数值/配对核验语句保持。
- 新实测端点审查对已完成的preview正确参考case15执行成功：600周期、601端点、实测标签/控制输入和3张图像全部核对；实际退出0。证据为`outputs/coupled_preview_wam_measured_audit_20261026.json`，不是新WAM运行。
- 实际verifier `--help`退出0，日志`outputs/commit_preview_20261026/wam_verifier_help.txt`；新增文件Ruff/格式检查通过。

尚未产生新WAM完整运行数据，因此整个新verifier尚未执行真实12案例端到端审查。下一步仍需总owner、独立metadata与prepare/register流程，之后进行真实服务/采集CLI准入和固定12条开发。正确参考18/18控制门槛已通过，但WAM执行尚未获完整登记，fresh为0，总目标未完成。
