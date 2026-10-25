# D1 完整资格与生命周期补充核验（2026-10-24）

D1 已完成300请求、10500个denoise节点及10500次head调用，原独立数值审查通过。原046额外监督器仍为False：在worker已实际wait0/reap后，错误地从旧格式release收据读取`actual_wait`，产生`KeyError`。旧格式的wait在exit收据中，release只记录reap等七个字段；不得补写字段或修改原失败。

050独立补充核验已通过，重新核对完整输入指纹、原数值资格、模型及审查进程的真实退出和GPU释放，并确认冻结源/快照不变。仅准许将该D1用于W2，不新增模型调用、数组decode、拟合、QR或物理步骤。

权威结果：`outputs/clean_training_d1_lifecycle_addendum_20261024/result.json`，SHA256 `0c4cc4ceab18060747714b60c3f169250fc0f91bcda506854368cceead918910`。原监督器False保留。

未执行的047/048准备代码改为显式选择legacy/durable收据格式；047必须绑定050结果。相关收据及接续检查68项通过，完整W2相关检查150项通过，Ruff检查与格式检查通过。首次测试因未创建basetemp父目录产生26项fixture错误，创建目录后完整通过，错误不涉及算法；首次bind因缺少线程环境限制被拒绝，未发布配置，补全三项线程限制8后成功。

047现已冻结537份源及537份快照，独立元数据审查通过。真实producer、independent及owner入口dry-run均通过，均0实际fit/模型/物理。实际W2 CPU拟合已启动，监督器PID2142698/birth575436577、producer PID2142742/birth575436904，经/proc核验均存活且空CUDA/LD、三项线程限制8；独立QR待producer正常完成后串行执行。尚无W2完成资格。总目标未达成，原预留24条性能验收保持关闭。

后续STATE接续在本次修复后通过149项组合检查（含真实生产入口help与CPU子进程退出）。新QA保存在`outputs/clean_training_state_continuation_20261024/qa/qa_receipts_v2.json`。首次组合命令有三个测试文件名拼写错误，未运行测试；改用实际文件名后149项通过。048尚未绑定实际W2或启动模型。只读资源检查显示允许UUID的显存为0，31338端口空闲；正式启动前须重新检查。

W2 producer随后完成：`runtime_w2_fit_20261024/fit/W2/result.json`为True，840000行、150标签锚点、1次fit，0新模型/物理/fresh。实际durable exit wait0、release reap/no-signal与537源/快照已复核，见同root下`qa/producer_terminal_audit.json`（SHA256 `ec0cfc410c67d42abbe46f57d00475b7945b6be5af1353a10e50885224e41d47`）。监督器已串行持有独立QR进程；QR尚无完整结果，暂不准入STATE。此前“拟合运行中”为该结果前快照。
