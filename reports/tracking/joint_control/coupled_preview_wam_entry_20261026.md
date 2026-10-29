# 四步预览 WAM 执行入口准备

新增`src/s2a/coupled_preview_wam_simulation.py`，从冻结oracle预览入口显式派生。当前只是实现与CPU准入检查，尚无新WAM服务、真实CLI准入、模型请求或物理成绩。

入口仅接受WAM、当前实测端点observer和空oracle_reference。首先检查完整18/4907控制closure，再绑定已获资格的arm/夹爪参数、SOURCE权重、预览配置。另要求新的`s2a.preview-wam-development-registration.v1`登记，明确通过、授权开发、不授权fresh、不自动重试，并绑定入口、控制gate和逐响应协议的来源SHA；旧061登记不能直接复用。

每案例固定600步、40端点chunk、每8步重观测、75请求及末40步稳定判据。服务身份核对登记的prepared/registration/source/readout引用、当前案例、raw/terminal分支、checkpoint和唯一计划GPU UUID `GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`、HTTP31365。每次保存的响应经新的当前请求/分支身份检查后，才交给原四步预览控制。raw_checkpoint_reference标签按实际服务分支区分，terminal058不冒称原checkpoint输出。

## 验证与限制

- 实际CPU导入通过，新增入口和测试文件Ruff/格式检查通过。
- 入口/控制gate/响应协议20项组合测试通过；随后增强固定控制参数绑定，入口3项测试重验通过。
- AST逐语句比较确认，除生成参考查询块外，每物理周期的控制逻辑与原冻结oracle入口一致，包含实测状态读取、原生goal/SOURCE最终提交和双历史更新。初检发现派生时文本替换误改了内部governor条件，修正后通过。
- 缺少控制资格会在创建输出、环境或访问网络前停止；显式expert参考在资格检查前即被拒绝。

该入口仍需实际服务/owner/collector和独立WAM机械审查接入、新登记与真实CLI dry。当前没有该版本的执行登记，不能据这些测试称为WAM执行资格。HTTP身份字段核对仍须服务端真实GPU/来源和完整退出证据支持。推理期间物理暂停，不声明实时控制性能。

当前正确参考回归最近确认前13条独立核验全部达标，续跑及关闭审查尚未结束。保留完整18/18准入、237固定12条开发和最终留出24条验收门槛；没有新fresh证据，总目标未完成。
