# 最终验收接入的隔离实现 — 2026-10-23

**新增七项接入组件在隔离副本完成339项组合CPU检查，尚未提升到主源码；最终性能仍未运行。** 041新D0已完整通过，随后W1前置失败并停止；原484份冻结源码与隔离副本473份原来源逐SHA不变。本项不选择候选、不拟合、不启动模型或真实物理环境，也不读取现有fresh/训练future数组。总目标及20/24、各任务4/8保持不变。

[042设计草稿](../plans/v0.0.1/042_fresh_acceptance_implementation_20261023.md)与[source basis](../outputs/wam_fresh_acceptance_implementation_20261023/source_basis.json)保存边界。隔离副本为本仓库`outputs/wam_fresh_acceptance_implementation_20261023/repo`；算法、脚本和测试仍按src/scripts/tests结构维护，当前运行的041不导入这些新文件。完成当前冻结协议、实际候选及最终专用准入之前，不在主源码启用。

| 新组件 | 本次完成内容 | 证据范围 |
| --- | --- | --- |
| [固定矩阵及统计](../outputs/wam_fresh_acceptance_implementation_20261023/repo/src/s2a/fresh_matrix.py) | native24/oracle24/WAM24身份与顺序、24固定分母、600原生boolean、末40连续与12初态双seed分组；不完整拒评分 | 纯统计工具，不证明日志来源、实际跟踪或执行资格 |
| [首相机见证](../outputs/wam_fresh_acceptance_implementation_20261023/repo/src/s2a/fresh_camera_witness.py) | 原renderer/generator各一次、两张当前observable RGB、四份完整physics、fixture initial/time不变、实际首PNG/request指纹、回调/函数恢复与异常拒绝；退出时再核证据SHA | fake physics/HTTP配合实际Torch/PIL及原generate，0真实sim/model；不改冻结executor字节 |
| [独立相机检查器](../outputs/wam_fresh_acceptance_implementation_20261023/repo/scripts/verify_fresh_camera_witness.py) | 不导入生产helper/canvas；自行重建双flip、bilinear cover-crop、4px seam，逐像素核PNG、独立核四physics及request/source；正常入口拒注入渲染约定 | 证明已保存当前见证的一致性，不声称独立环境rerender、任务成功或fresh准入 |
| [严格oracle读取](../outputs/wam_fresh_acceptance_implementation_20261023/repo/src/s2a/fresh_oracle_reference.py) | 唯一带seed目录、拒无seed歧义与fallback；4份native产物＋独立qualification及实际wait/reap/birth/noSignal；600行/20Hz/tail40通过后才decode并逐端点核reference | 全部合成native数据；真实native owner及qualification尚未存在，不补写旧原始结果字段 |
| [native包装器](../outputs/wam_fresh_acceptance_implementation_20261023/repo/src/s2a/fresh_native_capture.py) | 原native executor委托一次，另存完整首physics/时间、真实worker PID/birth/argv/cwd/CPU、实际委托源码ref；旧expert声明原样保留，仅模块私有重定向到同SHA安全副本 | 原产物不补字段，不伪造wait/reap，额外见证与失败均保留；0真实physics |
| [独立native检查器](../outputs/wam_fresh_acceptance_implementation_20261023/repo/scripts/verify_fresh_native_reference.py) | 不import producer/oracle；先核同worker与owner birth、委托源码、配置模板SHA及实际wait/reap，再核600端点/20Hz/tail40、初态完整字节、原expert及零arm末夹爪保持、全部reference端点 | 正常入口固定真实委托源码；合成入口显式绑定整source map，不把假环境当真仿真 |
| [W2请求账本](../outputs/wam_fresh_acceptance_implementation_20261023/repo/src/s2a/fresh_request_policy.py) | 固定24个W2 case、当前PNG/15-key输入、唯一readout、35节点实测计数声明；1800请求/63000节点；首相机独立核验允许本条75请求结束后绑定，下一条必须已核 | 纯metadata状态机，尚无模型HTTP服务；外层仍须W2独立拟合、开发门与执行准入 |

主root完整联跑四个新增测试模块，**205 passed、0 skipped，38.15秒**，Ruff check及format --check通过，8个新增文件前后字节一致。[组合QA](../outputs/wam_fresh_acceptance_implementation_20261023/combined_qa_v1/qa.json)，SHA `726204a9b87b758e84684c0fbbca6008ca48259f92a3aeb0da26263532714612`。这不是主仓库全套QA或最终实际执行资格。单项为matrix18、oracle103、witness42、独立camera42；各收据、初轮失败及修订日志完整保留。[真实producer与独立checker联调](../outputs/wam_fresh_acceptance_implementation_20261023/qa_camera_verifier_v1.integration.json)仅使用fake physics/HTTP、实际原generate，不调用真实世界模型。

新增native包装35、独立native55和W2账本44项后，root完整联跑七个模块，**339 passed、0 skipped，78.13秒**；Ruff check/format通过，14份新源码、主484来源/484快照与隔离473来源前后逐SHA不变。[v2组合QA](../outputs/wam_fresh_acceptance_implementation_20261023/combined_qa_v2/qa.json)，SHA `15c0d4d6356a0e2732605b36bbe621b9cb46328373a87f86c719883798cd0952`。原v1收据保留。测试工具句柄失效后，已实际核结果文件与日志、确认对应进程消失，没有重跑测试。

另做[真实CPU Popen联调](../outputs/wam_fresh_acceptance_implementation_20261023/native_owned_interop_20261023/result.json)，SHA `e90688c63c28e24a2253f3a1722b133a27d661fa310e502c8b7b6356c9abfd3d`：seed42/43各600个新合成端点，原`checkpoint_simulation.run_case`字节不变，模块局部fake环境/观测，实际producer见证→独立native检查器（显式合成来源）→严格同seed oracle读取全部精确接通。两worker均实际wait0、同birth reap、无signal，原native JSON未改。sim env311/NumPy1.26写入、main/NumPy2.1读取也通过；1200次fake step，0真实物理/模型/既有future decode。这证明接口及真实生命周期互通，不是两条native任务成绩。

实际跨环境比较已得到**False**：main CPU为NumPy2.1/Torch2.12NV/Pillow12.2，实际sim env311为NumPy1.26.4/Torch2.6CPU/Pillow12.3。固定seed42共10组相同合成RGB输入，真实256随机图有184个颜色标量相差1；另3组尺寸也有差异，最大均为1。各自PNG解码与自身pixels一致，跨环境pixels与解码PNG不同，因此不能认证main3.12作为该实际画布的逐位核验环境。[原失败收据](../outputs/wam_fresh_acceptance_implementation_20261023/canvas_cross_runtime_20261023/result.json)，SHA `c53a2d367820d68e3b74aea618ae73d8310eee684f4a444347f9622352809af8`。

保留上述False、原输入和零像素容差后，**实际sim env311中的独立checker与原production输出10/10精确一致**，没有重抽样或重生成production图像。[同环境独立收据](../outputs/wam_fresh_acceptance_implementation_20261023/canvas_cross_runtime_20261023/result_actual_sim_verifier.json)，SHA `549367919c2f0685bc321dbfd22d07d9d7608353cca31eb3299856841ff504fd`。另在实际env311运行真实helper、原generate及完整独立checker，两个新合成当前相机fixture（256/257）均通过，覆盖NumPy1.26 NPZ、四physics、PNG/request及hook恢复。[完整联调收据](../outputs/wam_fresh_acceptance_implementation_20261023/camera_actual_sim_interop_20261023/result.json)，SHA `f3b2ba30065ea8eadf71e704c6241464033b44d0d63b4a32f23e115e62e14b05`。这些仍使用fake physics/HTTP和显式合成来源，不证明真实renderer、任务或fresh执行资格。后续正常相机核验指定实际sim env311解释器；不改变模型/控制或当前D0。

[原矩阵metadata草稿](../outputs/wam_fresh_acceptance_implementation_20261023/draft_matrix.json)保留，SHA `3e4f7624f8a392815e531beb814a94fc31a732c931b9cbaef5c616b4e8a33b6e`。[v2草稿](../outputs/wam_fresh_acceptance_implementation_20261023/draft_matrix_v2.json)增加实际sim解释器、包版本和上述两项原10输入比较收据，SHA `725a277e3362356ba8c4f21a9bab76e13858a03ca9379db26066cd6f4fd555d8`；原12初态及72条顺序不变，candidate/执行/fresh授权仍均False。条件性完整预算算术为72条/43200步、WAM1800请求/63000 denoise/1模型初始化、0fit及0自动重试；未授权或消耗此预算。

仍缺实际W2及其全量独立重算、state99数值资格、开发稳定门结果、最终真实native/oracle owner与资格、W2-only24case模型私有服务、全63000节点与全72条执行审查，以及最终跟踪准入/源码/资源/admitted闭合。账本不能直接替换旧开发服务：其public case索引48–71与service索引0–23分开，`requests`返回副本，旧12case/raw分支/31500计数和HTTP方法均须明确适配。因此当前不能打开fresh，也不能报告最终成功率或总目标已完成。

本次训练等待也有实际进展：07:07:04 UTC，supervisor1390098/birth567832941、owner1390101/birth567832997及模型1390170/birth567833132均实际存活且birth匹配，D0已保存94/300请求、3290/10500节点；无整轮result、无fit/新物理/fresh。[真实等待观察](../outputs/wam_fresh_acceptance_implementation_20261023/training_wait_observation_v1.json)，SHA `68c8bfaad6d7d30e16272ed068be6de33b89ed3be04ddf7a833ac61de047ea2e`。这是在具体存活句柄上的验证等待，同时完成本次隔离实现；不把partial采集记为完整资格。

07:52:08 UTC再次核验同三个PID/birth存活，D0已保存259/300完整请求、9065/10500节点，仍无整轮result；主484源码、484快照及隔离473来源逐SHA无漂移。[后续观察](../outputs/wam_fresh_acceptance_implementation_20261023/training_wait_observation_v2.json)，SHA `82f672ddb96916953c3ce7ca8688c118e51642bbbe405a758eaac0bf1ecd610b`。原启动未重试，原中断数据不进入fit。

随后D0完整300/10500和独立审查/正常退出通过，W1前置失败，supervisor保留False并停止；[041当前结果](wam_clean_training_continuation_20261023.md)。故上列存活段仅是对应时刻的真实等待证据。D0来源已[独立只读复核](../outputs/wam_fresh_acceptance_implementation_20261023/d0_data_readonly_audit_20261023_v1.json)，保留原资格；后续[043](../plans/v0.0.1/043_clean_training_postcapture_20261023.md)准备有限CPU修复，尚未执行fit。
