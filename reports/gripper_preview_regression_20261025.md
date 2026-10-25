# 067 更广正确参考回归：执行审查通过，候选性能失败

固定六条辨识留出native轨迹各baseline/preview一对，加三原开发init0 preview oracle，共15条4430物理步，0模型/fit/fresh。沿用066六步单调打开预览及固定控制，无任务特例。候选必须9/9末40稳定且原10mm/.05rad门槛均通过，不以native来源成功替代S2A结果。

实际Python3.11十五条CLI dry及来源/精确预算检查通过，Ruff通过。持久化owner3428754/birth582710485、首客户端3428799/birth582710518已实际启动；源码、配置、所有15个profile的case/trajectory及066pilot绑定在source_sha256.json，固定顺序、无自动重试。CPU/OSMesa空CUDA/LD。

独立入口`verify_gripper_preview_regression_case.py`已实现并实际执行，明确区分native含初始端点与world未来参考两种数组格式；重用通过066的独立控制/预览、标签、图像与退出检查。

首组task0/demo_40 baseline和preview均完整170步，独立机械/时间/标签/三图像/正常wait-reap及source检查通过，performance_qualified=True。baseline成功97步/尾89，preview成功102步/尾95，两者末40稳定。只是首条候选，不能写9/9。原owner继续，第二组baseline客户端3430377已按实际/proc原birth确认存活，初始真实端点已落盘。

尚无完整067或新WAM收益；原061失败、总验收与fresh关闭不变。

## 第二组bowl通过，首组wine baseline失败

case02/03 task0/demo_48各170步完整独立控制/标签/图像/正常收据/source核验通过，baseline稳定True成功98/尾86，preview稳定True成功99/尾99。候选目前已独立完成两条均通过，不能写9/9。

case04 task1/demo_39 baseline完整244步机械/标签/图像/收据核验通过，但独立native ever/terminal/stable均False，成功0步，performance_qualified=False。保持原失败，不将原native来源成功当S2A成功。原owner3428754及case05 preview客户端3432394按/proc确认存活，固定候选继续；还无此wine候选完整结果。

## 首组wine留出候选修复稳定性，第二组继续

case05 task1/demo_39 preview完整244步，实际完整独立控制/预览、端点标签、图像、退出及source核验通过，原生ever/terminal/stable=True、成功/尾连续90步。对应baseline case04成功0步且稳定False；这是第二个wine正确参考的配对改善，不是WAM收益。候选当前已完整独立3条均通过，9/9仍未完成。

case06 task1/demo_46 baseline完整334步并完整独立核验通过，成功158步/尾87、稳定True。原owner3428754及其preview case07客户端3434173按/proc确认原birth存活，固定队列继续，无重启或调参。

## 第二组wine发生真实退化：原9/9门槛已失败

case07 task1/demo_46 preview完整334步，独立控制/预览/标签/图像/正常收据/source核验通过，但原生ever=True、terminal/stable=False，仅成功2步、尾连续0。对应case06 baseline稳定True、成功158/尾87。因此存在baseline成功而候选失败的实际回归，原要求已不满足，禁止将066三条成功或case05收益作为跳过该失败的理由，也不接入WAM或fresh。

case08 task2/demo_40 baseline完整215步独立核验通过，成功85步/尾82，稳定True。原owner3428754及case09 preview客户端3435945按/proc实际存活，继续原固定剩余队列，取得完整失效范围，不调整预览步数或门槛。统一提前开度虽修复部分释放滞后，也可能破坏与机械臂阶段的同步；这只是待检验解释，不使用对象真值作在线触发。

## 首组cream也发生退化；释放预览与位姿阶段描述

case09 task2/demo_40 preview完整215步，独立控制/预览/端点标签/图像/normal收据及来源核验通过，但ever/terminal/stable均False、0成功步；对应case08 baseline稳定True。固定六步开度预览已出现至少两条baseline成功/candidate失败，不能作为统一生产候选。

只读release_pose_synchronization_description.json检查三个已完整wine案例的实际position_tracking_release事件：训练demo_1及留出demo_39改善案例，六步预览跨越的目标平移最大约3.12/1.49mm，旋转约0.0722/0.0726rad；退化demo_46约14.94mm、旋转0.0188rad。输入前后SHA一致，0新模型/物理。提示提前开度可能与仍在移动的机械臂阶段错位，但只是有限描述，不能从两个改善与一个退化样本直接学习/宣称通用门控。更不能忽略旋转或未覆盖的cream失败。

原owner3428754及case10客户端3436524实际按/proc存活，原固定剩余对照继续；不改在运行源码、窗口或验收，fresh关闭。

## 六条留出对照完整：候选4/6、baseline5/6

case10/11 task2/demo_49各182步完整独立核验通过，baseline稳定True成功/尾86，preview稳定True成功/尾91。至此六条留出候选稳定4/6、baseline5/6：修复wine/demo_39，但破坏wine/demo_46和cream/demo_40，净收益不成立。三开发oracle仍由原owner继续，不能用其潜在成功修饰留出失败。

新增six_holdout_release_pose_description.json对全部六候选实际release事件做输入前后SHA核对。两个bowl成功事件参考平移跨度约21.5/32.0mm，wine失败约14.9mm，cream失败两个事件约30.8/35.1mm；另cream成功约8.7mm。因此“移动跨度大必失败”不成立，不能只用三个wine样本宣称通用门控。任何后续位姿耦合保护都须作为新显式假设另行固定，并在未参与该诊断的数据上验证，不对旧结果事后改判。

## 首个开发oracle完成

case12 task0/init0 preview完整600步，实际独立控制/预览/601端点标签/三当前图像/正常收据/source检查通过，原生ever/terminal/stable=True、成功及尾连续532步。共13/15执行完成且独立核验，候选已完成7条稳定5条；原9/9门槛失败保持。原owner3428754和task1/init0 case13客户端3438930按/proc确认存活，最后两个开发oracle继续，不重新初始化或更改候选。

## 第二个开发oracle独立完成

case13 task1/init0 preview完整600步，独立控制/预览/601端点标签/图像/正常收据/source审查通过，stable=True、成功465步/尾连续461。当前14/15执行并独立核验；候选8条已完成6条稳定，原9/9失败仍保留。最后case14 task2/init0客户端3439799/birth582764371实际/proc同birth存活，原owner3428754继续，无参数或预算变化。

## 全15条完整结论

case14 task2/init0 preview完整600步独立核验通过，成功及尾连续508步，stable=True。全15条4430步、控制/预览/标签/图像/实际正常wait-reap/source检查均通过，源闭包再次无漂移，原owner及全部client均退出。regression_summary.json绑定逐case审查SHA并区分执行通过与性能失败。

原9条候选稳定7/9：辨识留出4/6（baseline5/6）、三开发oracle3/3。两条明确baseline成功/candidate失败，原门槛不满足，不采用无条件六步预览，不进入WAM/fresh。原066局部3/3通过保留其有限范围，不覆盖本失败。下一步应将任何预览补偿作为位姿/开度联合参考的控制问题处理；新假设需明确因果输入、耦合保护及未参与本诊断的回归集，不能事后挑任务或窗口。
