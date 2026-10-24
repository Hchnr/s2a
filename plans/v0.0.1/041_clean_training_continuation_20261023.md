# 038中断后的单次完整接续 — 2026-10-23

本项继续[036总目标](036_project_goal_20261023.md)，不降低实际跟踪/多任务/留出初态的完成标准。最终24条稳定≥20、各task≥4/8仍未运行；本项不打开fresh性能。它是对原038/040中断的一项新有限协议，原协议结果及1599/55965/4预算不回写为成功或扩大。

## 已证事实与未证部分

原D0 owner1292033/birth567473266及worker1292382/birth567478090已消失，会话4754不可再访问。20条完整请求/700节点与第21条25个持久化节点保留，原调用完成下界725，精确调用总数未知。原source462及snapshot462字节未变，capture文件稳定，GPU已释放；但原actual wait/reap、worker exit/release、capture result和owner result均缺失，不能补造正常退出。turn_aborted与消失仅有时间关联，终止signal/原因未证。[独立中断审计](../../outputs/clean_training_interruption_audit_20261023/audit.json)、[原文件清单](../../outputs/clean_training_recovery_design_20261023/interrupted_artifact_inventory.json)

原20条与partial25不进入fit、不拼成新完整D0，不覆盖、移动或重占原capture/D0及orchestration/D0路径。以独立新输出根`outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023`执行一次新的完整D0，仍同一300输入/31位seed及固定35节点。没有根据future标签、loss或任务结果换样本、选seed或参数；原生/validation future仍未解码。

## 新运行与跨运行累计预算

新一次完整条件流程保持D0 raw300→W1一次fit/独立全量QR→D1同300输入/初噪声与固定W1→union W2一次fit/独立QR→99完整生成→raw6/W2六开发任务。单次新运行预算仍1599请求、55965 denoise、4次模型初始化、2生产fit＋2独立QR、7200物理步；所有样本、35节点、40端点、权重、ridge.1、1290特征、FP32/FP64/BF16数学和容差不改。

旧调用范围保守记为725..735：已证700+25节点；原35-call scope未完成，第21请求尚未产生ledger或进入下一请求。上界按该请求完整35保留余量，不声称精确旧计数。旧21次请求尝试中20条完整，旧模型初始化1次。合计本项与中断原运行：**请求尝试≤1620、denoise≤56700、模型初始化≤5、生产fit2＋独立QR2、物理7200**。额外消耗仅为旧中断采集；不将旧20次完整请求伪装成未曾执行。037资格等更早历史维持各自原预算，不混入此两运行累计。

单次接续没有自动重试。任一新阶段失败就停，记录已消费与未执行项，不再次启动D0、缩样本、选W1或增加训练轮次。STATE原诊断门False时是否进入固定开发仍沿038：完整数值/来源/因果审查通过后允许执行原开发矩阵。开发稳定≥5/6且各task≥1/2、W2严格优于new raw的收益门分别报告；它们不能替代fresh20/24。

## 隔离源码与metadata派生

旧462份源码和原冻结计划/配置保持字节不变。新builder从040 SHA绑定的10个执行脚本与runtime生成独立bundle，basename保留，独立审查仅允许有限metadata变更：runtime import改为bundle内continuation_runtime、入口/required-source路径改为实际bundle路径、两处local REPO guard绑定本仓库、原输出root guard改为新root。数学/采样/控制/评分语句逐字保留；每个变更token/count、原/新SHA均登记，独立checker自行核允许变更，不靠producer True。真实argv必须指实际bundle文件，不能把新worker冒称旧入口。

新prepared是原b55 JSON的deepcopy，**仅root值**变为新绝对根。数据、全部当前PNG/opaque标签SHA、source身份、时序划分、150/300/24/96、模型配置、权重及原metadata-only字段不变；另一个独立metadata审查明确证明root-only派生。新31位seed-plan引用新prepared SHA，但300实际seed逐条等040已冻结值，不重新选择种子。原canonical/identity继续指原source身份。原b55及原独立metadata审查不改写。

新的proposal/独立execution-verification/admitted沿无循环SHA证明链；新增config/计划、bundle及supervisor/测试全部在新source closure与逐文件snapshot内。新准入独立核原SHA、失效/消失事实、数据与新root派生、seed、有限源码变更、实际资源、最终QA及累计预算，不能复用原040 admitted直接执行新root。CPU准备、QA、审查均空CUDA/LD、线程8、无future数组解码、模型或物理。

## 脱离交互会话的实际所有权

短launch在本仓库创建新的session supervisor，将stdin/stdout/stderr显式指向DEVNULL/本仓库日志并返回。supervisor在父launch退出后仍存活，持有每个stage owner的真实Popen，记录PID/birth/PGID/实际argv、正常wait/reap/timeout；各stage owner继续持原模型/独立checker的真实Popen与GPU锁。CPU小实验已证明正常父退出后的存活、exit0与timeout owned-child reap；不能声称免疫容器整体销毁或SIGKILL。[持久化设计与实验](../../outputs/clean_training_recovery_design_20261023/review.json)

六阶段顺序固定D0/W1/D1/W2/STATE/DEVELOPMENT，不并发模型、不自动恢复失败。supervisor单阶段主流程时限分别88400/15000/88400/15000/45500/175000秒，只覆盖既有内部生产/审查/释放时限及启动/指纹核验余量，内部86400/7200/1800/43200/172800等不放宽。首次metadata准备前，已把初稿STATE45100、DEVELOPMENT174700修正为上述值：初稿小于原生产＋审查＋120秒释放的45120/174720之和，不是执行后追加时限。异常或外层超时先只向birth匹配的自有stage owner发送SIGTERM，再留至多130秒让其回收自有模型/审查并观察释放；这段是失败清理，不新增采样/下一stage，避免两秒就强杀owner而留下模型孤儿。清理仍失败则如实保留False，不认证正常释放或继续。实际prepared与admitted SHA在存在后组装argv并保存，不在自引用registration中填假SHA。每stage必须真实exit0且对应owner result完整passed后才能继续；最后开发任务成绩另列。

模型仍只允许`GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f`，UUID/PCI BA/minor6实际交叉核验、空闲观测及同UUID锁不变；不使用另一允许卡来并行扩大预算。CPU/OSMesa空CUDA/LD；不改外部依赖或Cosmos。资源最低200GiB磁盘、128GiB有效RAM、8 affinity核、线程8及31338/31339端口，启动前再次实际核验。

实际顺序为新metadata/bundle准备→独立metadata审查→最终相关CPU QA/Ruff/源码冻结→未授权proposal→独立执行审查→admitted→唯一detached launch。任一资格未通过，后续模型保持0调用。以上是运行前登记，未声称新接续、fit、开发或fresh已经完成。
