# 保存3543的有界末周期修复

执行前登记：以已关闭的完整H12 SQP的 `answer.phases[50].ray_attempts[3].entry`
（response3543）作为唯一输入。前项全部3657响应已独立复算；其因果分析按
prefix9保持、cursor9严格下降、仅terminal失败筛出40条，再按cursor9归一化最大平方误差、
响应序号排序，3543为首项。选择不使用本轮新响应。

该点cursor9归一化最大平方误差为1.314526976519977，terminal为
-1.20453558423459e-6，非terminal最小slack为6.13467374677e-8。检验仅改变最后
第12周期平移命令是否可在原全部安全约束下保留这段改善。仍远未达到完整轨迹门槛；
本轮成功不授权重新启动4000响应SQP或物理pilot。

固定actual368的实测q/v、completed367命令与旋转latch、原已生成12参考、零历史误差
profile、固定finger/控制器、2ms/25内步、原原生旋转语义与状态更新动力学。各响应
均重新复制同一初态，禁止使用后续实际反馈。保留原位置.01m/旋转.05rad caps、
command box/rate .25、joint margin .05、braking .1s及e7 terminal normal。

唯一调用序列为首anchor；末周期x/y/z各依次+.001、-.001、+.0005、-.0005；末anchor
repeat。共14完整响应、13唯一命令，每次300数值内步/312查询，另当前查询1；producer
与独立checker各4200内步/4369查询，合计28响应/8400内步/8738查询。所有命令在模型
构建前完整准入；任一不合格则0响应停止，不裁剪、不替换、不追加预算。技术失败保留
实际已发调用/完成/查询与异常，不自动重试。

每次保存全部rollout、原九组audit、五组非线性slack、12点误差与原witness标签。
首anchor与历史3543的完整数值字段exact；前11完整period、前275内步/11端点及12个
旋转goal exact；首末完整响应exact。候选必须满足原audit、所有slack>=1e-8、prefix9
与上述exact门。按terminal降序、实际扰动Linf升序、请求序号升序选择，且terminal必须
严格改善。旧E17已不能界定当前误差，本轮不输入或输出经验充分余量资格。

独立checker复用已经验证的独立数值路线，完整重算14次，独立生成网格、重算strict
标签/因果性及选择。保存精度严格排序与独立浮点路线的strict分类分别记录；浮点比较
沿既有容差，布尔/整数采用exact。历史来源和当前执行源码绑定SHA；独占outputs目录，
实际wait/reap保存，失败不覆盖。新核心和入口的针对性CPU QA通过后才运行。

全部命令从当前仓库执行，空CUDA/LD、单线程、seed42；不访问旧清理根，不改变冻结
前项源码。新增物理/world-model/fresh/参考消费均0，实际cursor8/task0/grasp0与原
18正确参考/12WAM开发/24最终留出门槛保持。总目标仍未达成。

## 首次零响应失败与输入引用修复

提交aa74655的首次producer在读取controller.json的`cv2_guard_registration`时KeyError退出，
owner实际wait/reap1（`outputs/terminal_only_tail_repair_producer_20261029/result.json`），
没有模型构建、响应或查询。153项组件QA未覆盖实际环境记录结构，该失败不追认为成功。
正确guard来自`terminal_development_execution.CONTROL_REF`，与控制器contract记录分开；
修订入口在dry阶段也读取并绑定其probe/runtime refs，再运行真实初始化。

原7来源以aa74655及`outputs/terminal_only_tail_repair_v1_sources_20261029/manifest.json`
保留原路径、SHA和字节一致的安全副本；不改写旧QA/result/log。当前producer、执行测试及
本计划的修订SHA在新QA记录，原110数学来源保持。修复执行写入新的
`outputs/tracking_v5_terminal_only_tail_repair_v2_20261029`，仍是同3543、同14请求、
同独立checker与全部科学门槛；这是一项明确修订后的执行，不是自动重试或追加搜索。
