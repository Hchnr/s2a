# 021：020 task2 零物理步启动故障的显式接续

本项只修复进程启动与身份检查，保持020已登记的效果方案。登记时020 task0/1仍执行原600步任务，禁止修改其任何已绑定源码、配置或产物。task2原matrix在Popen后读取worker的`/proc/environ`时异常，原代码把缺失值和非空值合并判错，且未记录被拒绝的具体值。精确根因未知，不能声称已证明exec竞态或实际使用了错误GPU。

原task2日志已通过两个Python自身空CUDA/LD检查，到GET /info时因服务退出而失败；尚未写case manifest、调用环境factory或POST /generate。case目录为空，服务生成目录和POST均为0。因此原task2为基础设施故障，0真实物理端点、0模型生成，不计任务失败。原error/release/日志完整保留，不覆盖、不重写为成功。独立诊断SHA `8588d03ea99c403bd155663788fbad01215750b3447e26f7469ccd2a9698b63f`，源码交叉核验SHA `0006f3da0694cadbdb537483a792009bb90b7dc169db27bfe8136a2b279b993b`。

仅允许一次显式接续task2_init0/sampling42，使用新配置`configs/wam_temporal_equal_task2_continuation_20261021.json`、新输出`outputs/wam_stability_20261021/temporal_equal_development/task2_continuation`、物理GPU6/端口9946。除输出位置、端口与说明外配置必须与020 task2相同。WAM8400非EMA、g1/35/shift5/SDPA/mask10、H40/requery8/20Hz、600步/75请求、v3夹爪/noGov/response .5/rate5以及同端点等权集成均保持。冻结的49778dd物理函数`run_temporal_case`原字节复用，不加入任务规划或性能调参。效果预算仍为三任务各一次、1800步/225请求；实际模型服务启动由三次增为四次，额外一次只有初始化、0生成。0新增oracle资格、0fresh、0额外seed，不自动重试。

新增隔离matrix/coordinator/握手入口；不得编辑旧020或被其source map绑定的任何文件。child自身Python检查CUDA_VISIBLE_DEVICES与LD_LIBRARY_PATH显式空、OSMesa，保存PID/birth ticks/父进程/命令/cwd/环境，再等待parent-go，之前禁止进入物理函数或访问模型。parent先登记Popen拥有关系，独立核验同一PID/birth/parent/cmd/cwd与环境，保存过程观察及正式worker注册后才能go。仅缺失/空读取的`/proc`观察允许有界重读并留证；实际非空环境必须立即拒绝，不把未知归一化为空。任意检查、超时或中断都在try/finally中精确清理并wait自有child，不留下孤儿。CPU-only测试验证无go不执行、环境拒绝和失败清理，不声称这些测试复现了原未知根因。

启动前要求原task2 coordinator/model/matrix均已退出、无同配置worker、原0步证据不变，实际8400全163文件stat与合并输入SHA一致，GPU6严格空闲、端口/锁空闲。新配置/源/启动证据预先绑定并独立审查。CPU/OSMesa显式空CUDA/LD，任何模型仅GPU6，禁止0–3。

最终审计使用新增显式接续合同：原root launch SHA `d702b78dcd3f781b7326417ab404986f34d2d6b877cfe41fdd1344731ddb179e`、原task0/1的8身份partial ready SHA `869d4b4732457ca89da54a2eae8b2988b23956b1aa6396cfc07ec88b5446f9a2`、原task2故障和新接续4身份ready分别保存。不得伪称12进程曾同时ready，不让原89c38d3全组审计接受partial快照。逐case复用独立参考/控制/评分检查器，核三条各600端点/75查询、全部225 HTTP/唯一生成、原失败0生成、所有自有进程正常释放与来源。首query分别对010同初态42精确核验。全部三条完成才报告统一ever/terminal/stable率；原task2故障单独列，不将四次服务启动当四个任务。

接续仅补足020原定未执行部分。若接续仍发生基础设施失败，保留错误，不进行第二次自动接续；若效果不佳，不用本组结果调整权重/旋转/夹爪/horizon后拼组。稳定成功的fresh验收门槛和预算保持原协议，本项不打开fresh。
