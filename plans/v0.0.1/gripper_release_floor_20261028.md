# 明确张开参考才释放：独立物理对照

总目标、原18/WAM/最终留出门槛保持。v4完整矩阵12条任务成功均0；task0/init0/seed43/terminal058曾有50个grasp_contact步、抬高约.1m，但没有完成“put the bowl on the plate”。原step73在reference从min.041升至.107时触发release、command=-1，随后接触丢失；这只是机制线索，不据日志时序直接认定因果。

候选在原闭合latch内增加显式release_reference_minimum=.35，保持原reference excursion、measured clearance、持续时间条件；小幅reference回升不能单独解除闭合保持。策略只读取已生成opening、当前测量opening、已知motor target和因果历史，不读取object/contact/task成功或未来观察。source object/contact仅用于离线评价。该候选有意改变开度参考的意图解释，不能宣称连续开度精确重建或普遍安全。

完整600基线+600候选=1200物理周期，seed43，原task0/terminal058固定75生成块与600arm命令。基线夹爪重算及全部601端点必须复现原记录；候选首次改变命令前的端点也须一致。独立初始化所有夹爪状态，比较完整实际grasp/task、最后40步稳定、最大抬升、开度和动作改变，同时保存全部601原生state与600条动作，支持之后独立命令回放。不得把短暂接触替代任务完成，也不在本例反复调.35制造通过。

执行前权威登记 `outputs/gripper_release_floor_pair_execution_20261028/registration.json` 和 `outputs/gripper_release_floor_pair_20261028/registration.json`。入口 `scripts/execute_gripper_release_floor_pair.py`；原生产与latch源码保持冻结，候选在独立gripper_release_floor/loaded_velocity_release_floor模块。CPU CUDA/LD显式空、沿用原OSMesa/线程环境，无自动重试，模型调用0、fresh0。固定录制参考没有对候选当前图像重生成，结果仅为局部机制物理对照，若有收益仍需完整控制资格及响应式WAM开发矩阵后再考虑留出。
