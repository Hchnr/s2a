# Fresh验收接入可行性只读审查 — 2026-10-21

**12个新初态已完成metadata预留，尚未进行performance；目前不能启动fresh验收。** 现有算法和seed-aware检查器具备复用基础，缺少冻结候选协议、fresh专用首请求来源证明和完整矩阵编排/计分入口。本次仅核metadata、源码及保存指纹，0模型请求、0物理步；没有从新初态结果选择候选。

## 已有证据及其边界

预留配置固定seed42选择每任务4个初态，共12；sampling seeds为42/43，构成24次重复试验。12个initial-state SHA各异，与此前18个物理开发初态无交集，并保守排除了已链接或疑似历史离线来源demo。已有独立metadata审计不是任务成功证据，也不证明预训练未见或历史图像/pose同步。见[预留配置](../configs/wam_stability_fresh_reservation_20261021.json#L116)、[metadata审计](../scripts/audit_wam_fresh_reservation_metadata.py#L75)。

本次重新核配置、fixture、既有metadata审计小文件及12份expert NPY指纹/shape；三份大HDF5沿用已有完整SHA，本次仅核size/mtime/ctime，没有重复读取大数据全量字节。12个expert序列长87–170步，均短于600；这只说明输入长度，不保证native执行后稳定。预留root下没有任何`result.json`。

可复核artifact：`outputs/wam_stability_20261021/fresh_preparation_review/metadata_and_code_scope.json`，保留本次source/metadata SHA及逐初态身份；完整fixture SHA为`6e279c4db07a06f7399f53eb298cb77b671305accfe5b520598f1ee1c8e134c6`。初态名称仍为`task*_init*`，必须结合fixture SHA和独立输出root辨认，不能因名字相同当成旧003/006场景。

## 可复用与待接入项

| 部分 | 现有能力 | Fresh最小待完成项 |
| --- | --- | --- |
| Native正控 | [checkpoint_simulation.py:99](../src/s2a/checkpoint_simulation.py#L99)核expert SHA并回放；:139在expert结束后保持最后gripper、arm为零；:224保存actual after-state参考 | 新只读配置/CPU编排器显式选择fresh fixture与24个initial/seed对；保存初始物理和相机来源；完整600步计分及释放审计。旧矩阵默认从output root找fixture，不能仅传reservation配置直接当完整执行配置 |
| Oracle正控 | [stability_recovery_simulation.py:166](../src/s2a/stability_recovery_simulation.py#L166)读取实际native参考并要求其完整稳定；position/tracking执行器支持oracle | 指向新native输出、按同record/seed绑定实际600端点；禁止补齐/伪造参考或用旧初态reference。禁止无seed旧目录优先匹配误选42/43来源 |
| WAM执行 | position/tracking新环境均使用当前观测、原始生成参考，模型等待期间物理不推进；[tracking执行器:253](../src/s2a/stability_tracking_simulation.py#L253)只在oracle分支读取特权参考 | 候选身份、checkpoint/backend/采样参数/控制器/重观测规则先整体冻结；仅一套算法覆盖所有fresh任务，不按task或fresh结果挑策略 |
| Seed-aware机械审计 | [014 checker:131](../scripts/wam_position_acceptance_checker.py#L131)显式验证registered sampling seed、请求/响应/原生motor等；first-query callback为必需依赖 | 新fresh callback及24case主审计；014旧42回归和015正确参考43资格不能宣称真实43 WAM已被验证 |
| 016因果时序审计 | literal014机械检查保留，新增独立query决策重放 | 仅当最终候选确实为016时复用；[016 wrapper:40](../scripts/audit_wam_tracking_replan.py#L40)仍硬编码seed42和旧006首请求，[其:69](../scripts/audit_wam_tracking_replan.py#L69)仅允许旧三场景，不能修改live文件后用于fresh |
| 启动/收尾 | 已有owned PID/startticks、GPU4–7限制、CPU空CUDA、拒覆盖、source SHA和服务预算sidecar模式 | 新条件编排器绑定冻结候选+native/oracle资格+新预算；准确核24case、全部HTTP/model生成目录及正常释放，未完成不能计失败或挑胜者 |

Reservation配置缺完整controller/model/runtime字段，`qualification_and_wam_evaluation_not_started=true`是明确边界；不是可直接执行的acceptance配置。旧native入口环境CUDA继承上层，新编排器须显式为空，不能沿用历史8卡映射。[旧入口](../scripts/run_wam_simulation_matrix.py#L92)、[新tracking入口](../scripts/run_wam_tracking_matrix.py#L104)。

## Fresh首请求callback合同

建议保留014接口：`fresh_first_query(config, record, variant, case, chunk, evidence, checkpoint, *, sampling_seed)`，但**不调用旧006精确回归函数**。Fresh不存在旧003缓存PNG或预期raw输出，强行与旧demo比较会拒绝真实新输入或冒认来源。

1. 验证record属于封存fresh fixture，initial-state/model-XML/expert来源SHA不变，seed为显式42/43；首chunk必须step0，controller记录的actual initial physics、query前后physics SHA/time和fixture初态一致。当前measured source、TCP变换及anchor/opening均从当时实测重建，不用dataset未来state作为在线锚点。
2. 未来获准运行时，在同一初始物理端点保存相机raw RGB witness及渲染参数，按冻结[cameras.py:4](../src/s2a/cameras.py#L4)和[canvas约定](../src/s2a/libero_simulation.py#L116)独立重建首PNG。渲染前后核物理state/time不变，不添加控制步或随机采样；该计划当前未执行。每个seed保存自己的witness，不假定两个环境seed的所有隐状态相同。
3. 核当前task条件、相机role/layout、Action128 domain、有效十槽mask、20Hz/H40、归一化及完整request；WAM允许使用任务语言，S2A不得据任务或物体标签重新规划。
4. 返回值与对应服务本地`sample_args/sample_outputs`精确一致，raw/normalized full128及反归一化可重算，checkpoint、EMA、backend、seed、实际source可追溯。没有fresh预期输出时，只证明响应来自正确当前输入和冻结模型，不能凭来源认证宣称生成正确。无需为制造“exact”额外调用模型；如要重复采样须另登记预算。
5. 原生predicate、object/contact和专家未来仅离线计分/正控使用，不进入query停止、候选选择或控制。Callback应有错误fixture/seed/相机/source/响应目录的有意义拒绝测试。

## 后续有限协议及未授权事项

若未来采用reservation既有600步/末40门槛，24WAM需要14400物理步；同initial×同seed的native24和oracle24再需28800步，总计72条/43200步。这里只做预算算术，**尚未登记或运行**；是否需要重复全部正控应在看fresh结果之前固定，不能事后省去失败项。若任何native/oracle失败，保留原24WAM分母和控制失败说明，不能删初态、换demo、重抽样或将所有失败归于checkpoint。

已预留可靠性阈值为overall≥0.8、每任务≥0.5：若使用24次/每任务8次口径，对应至少20/24和每任务4/8；同时按12个独特初态报告重复seed相关性和不确定性，不能当24个独立场景。最终候选、预算、首请求来源、控制资格和缺失项处理均需先冻结并独立review。

当前没有完整三任务稳定候选；010/016与后续017仍是旧场景开发。017的1200步旧场景持续效果诊断不会自动改写fresh原600步协议，更不授权开启fresh performance。本审查未新增factory、训练、模型、物理或fresh执行命令。
