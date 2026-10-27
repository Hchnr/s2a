# W2 完整数值资格与 STATE 实际启动（2026-10-24）

W2 producer 和独立 QR 均完整通过 D0＋D1 的840000行、150标签锚点；float64/ridge=0.1/active10语义保持原合同，独立参数比较使用原rtol=1e-9、atol=1e-11。两角色均真实wait0/reap、无signal，537源及537快照无漂移。独立QR约468秒，监督器True且已退出。没有新增模型推理、物理任务或fresh评测。

权威W2结果位于`outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/runtime_w2_fit_20261024/`：

- `fit/W2/result.json` SHA256 `dd7ee351fad0b1a183b7ea490bc8c89e62d78f09ad2565fc0886e36177efcd47`。
- `review/W2_verified.json` SHA256 `434c27694c81806bc0a85de7a4629b35cb211a3e8f084db9f9ceacdebdeeb3a1`。
- 完整收尾复核`qa/terminal_audit.json` SHA256 `dd217884584b574bebea59dec0dde0443bc3f783531d66e0c8e0ceb140e50326`，复用实际048全行/来源/退出checker，无重拟合。

048已绑定真实W2、冻结551源/551快照，独立metadata审查和实际wrapper/supervisor CLI dry均通过。dry验证原owner和sampler两处真实来源查询、导入恢复和源guard，0模型/数组decode/fit。

`outputs/clean_training_state_continuation_20261024/execution/admitted.json` SHA256 `c8389757bbdc832b0195a86e1f4b273be493790a96f1c59fae6adf04a1636b7d`。实际dry收据`qa/actual_admitted_dry.json` SHA256 `e5d671e012006996fd423751c4257c551e2a8535f92c30037ec164e4c38e5099`。

资源检查确认允许UUID GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f显存0且31338空闲后，STATE已一次启动：supervisor PID2164481/birth575527117，wrapper PID2164516/birth575527280，原model sampler PID2164619。通过/proc核验模型只使用该UUID、空LD、PYTHONSAFEPATH=1；551源/快照无漂移，见`qa/actual_model_launch_observation.json`（SHA256 `6266f6a25b349e86544fa4823386f929fb552d31e2e3b108fb2b7cd0d786a542`）。模型尚未完成99请求/3465节点，不宣称数值或生命周期资格。

总目标未达到。W2是生成模型active10 readout残差，不能等同固定控制器action逆模型；拟合/QR通过不证明闭环任务成功。后续先完整STATE数值及来源/生命周期审查，再固定12条raw/W2开发对照；开发门通过后才执行原预留24条新初态及native/oracle正控。最终≥20/24且各任务≥4/8标准保持不变，实时和真机另验。

STATE采样随后完整99请求/3465 denoise和head完成，1次模型初始化，0未来标签/fit/物理/fresh。模型PID2164619真实wait0/reap、无signal，允许GPU显存0且compute_apps为空（释放观察的利用率44为残留统计，不替代显存/进程收据）。独立CPU审查PID2205987仍存活，尚无完整审查或048联合终态。551源/快照不变；`outputs/clean_training_state_continuation_20261024/qa/model_terminal_observation.json`，SHA256 `9fc40c7aba5a55f000be7ecc234863cb6beedac36545967a9d9807569f6734c6`。不据采样True单独准入开发。
