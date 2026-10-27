# 四步预览的独立 WAM 服务准备

新增`src/s2a/coupled_preview_wam_execution.py`与`scripts/serve_coupled_preview_wam_development.py`，为已准备的WAM预览控制入口提供独立服务/执行登记合同。未创建实际登记、未加载模型、未执行新开发矩阵。

新namespace采用237登记的`outputs/current_coupled_wam_development_20261026`，HTTP31365与MASTER31366分离。准入要求新的prepared/registration schema、固定12案例与900请求/31500去噪节点/7200物理周期预算、允许GPU UUID、完整物理18/4907资格、既有060 state资格、独立新metadata审查、完整角色及来源冻结、输入/快照SHA与163个权重文件stat。旧061执行登记不能直接授权该版本。

服务显式派生原061实现，仅更换准入模块、来源自标识和服务schema。固定案例ledger、同首图检查、当前图像输入约束、raw/terminal058数学、真实GPU UUID检查、compact采集和异常清理逻辑保持。只允许`GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`；没有数字GPU标签执行授权。

8项CPU检查通过：新服务的完整900请求合成ledger、三个真实loopback HTTP拒绝场景、实际run/finally清理失败在受监督CPU子进程中保留、实际dry路径在注入已准入fixture下不导入模型、旧schema拒绝，以及全部服务函数/类与原冻结版本除来源字符串外AST一致。Ruff与格式检查通过。

这些fixture与CPU子进程证据不代表真实服务已准入。新owner/client/verifier/metadata与prepare流程尚未完成，因此当前完整角色检查会拒绝执行；不能创建一个仅写passed的登记跳过这些要求。下一步补齐新控制器下的采集与独立机械/请求/退出审查，然后做完整真实CLI准入。实际模型和fresh仍为0。

物理回归最新确认case14通过（位置RMSE0.0006845687391446761m、SO3 RMSE0.0020226920549281137rad、末尾连续85步成功），前15条全部达标。原SQP case09/14失败在新候选对应案例均已修复，但完整18条及所有退出/来源关闭尚未完成，不进入WAM执行。
