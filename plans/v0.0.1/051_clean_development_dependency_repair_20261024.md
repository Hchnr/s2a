# 051：049首次开发失败后的有限依赖接续

049实际启动后首个task0/init0/seed42/raw物理client在execution_inputs收集输入指纹时失败：旧policy template的tool_contract指向不存在的`/share/mixed/projset/.../eef_tool_contract.json`。0实际模型请求/denoise、0物理步、0完成case；唯一model service已初始化1次，按abort退出1并实际wait/reap，无signal，GPU释放。原owner/supervisor/service/worker False与562源码/快照保留，禁止覆盖或原地重跑。

实际合同在`/share-mixed/projset/eai_pwm/datasets/action_processed_r2/libero/eef/meta/eef_tool_contract.json`存在，其SHA `6fcaed3fe499aacb15d03170d002b1c897091481a0022db5f2922c06320a3a0c`与历史native正控manifest tool_sha256完全一致。已在仓库`outputs/clean_training_development_dependency_repair_20261024/inputs/`建立同字节安全副本和仅修改tool_contract路径的template副本。原声明与SHA通过path_relocation.json保留，外部只读，不在旧目录恢复文件。

实际sim解释器调用原stability_position_simulation.execution_inputs对首case配置的新template执行存在性及指纹预检通过，0model/physics/array decode。此前手工预检误用--config而旧入口使用--case-config，未调用输入检查、无产物；正确flag后通过。此预检不证明完整仿真或新执行资格。

接续必须：

1. 重用完整048联合终态和原W2参数，不重采、不重拟合、不重跑STATE；原state gate False与初态同步未认证均保留。
2. 新开发输出namespace和新registration/source/snapshot；原049所有失败文件不覆盖。只允许显式路径迁移及输出/metadata来源适配，世界模型、固定控制器、同原12案例/seed顺序、900请求/31500 denoise/7200物理步、原score及独立审查数学保持不变。
3. 原049消耗1 model initialization、0推理/物理；新051最多再1次初始化，累计开发尝试2、模型初始化上限2，禁止自动重试。新namespace全空且旧service/client已reap/GPU释放后才可准入。
4. 固定安全副本/template的来源链和唯一字段diff；验证全部原12 case在实际sim环境的输入闭包、控制依赖、cv2 guard和模型命令，确保修复不是只跳过缺失输入。
5. 独立metadata准入、所有实际CLI dry/来源桥核验、源闭包及UUID/端口检查后，才能一次启动新开发owner；服务/12client/独立审查的正常退出及GPU释放逐项核验。

当前只有依赖定位及CPU输入预检，051执行实现、独立准入和新模型/物理均未执行。最终原预留24条保持关闭，20/24与各任务4/8、实时/真机分级不变，总目标未达到。

组件推进：四入口有限AST副本与可逆整树检查已实现，主7通过/实际sim6通过1跳过；实际Python3.11生成新source_py311，全部12case输入闭包预检通过，562来源未变。初版Python3.12输出语法失败副本保留。runtime角色来源桥、独立准入和新执行仍待完成，详见reports/wam_clean_development_dependency_repair_20261024.md。

来源适配核心admit_runtime/extend_sources已实现并通过组合16项检查；原注册不改、原证明先执行、新闭包不得覆盖原字节。接下来必须将同一闭包接入owner/service/client以及独立audit的源码期望检查；仅有输入路径副本尚不可执行。

运行接入准备已实现：五入口role-aware副本、不可变051准入、实际角色CUDA/PID收据、审查source ledger扩展、新监督器和独立verifier；实际四角色help/全12输入闭包、整树AST独立重建以及真实CPU子进程退出/源漂移等组合108项通过。Ruff通过，原562来源不改。下一步为最终冻结与独立metadata准入、实际原owner dry及资源检查；此快照尚未启动新模型/物理。
