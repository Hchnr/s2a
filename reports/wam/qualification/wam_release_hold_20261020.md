# 固定WAM参考的首次接触后释放 hold 诊断 — 2026-10-20

4条事后选择的保存world参考×baseline/hold20/hold40，共12条CPU物理重放，各600步。触发使用sim privileged的任意单个物体双指contact，不读取目标身份。

原OpeningIntent每步持续更新；首次正常open且命令前当前/上一端点已有contact时，仅将实际夹爪强制close20或40控制步（含触发步），之后恢复正常命令，不重触发。

| source | mode | trigger command-start s | hold steps | changed commands | ever | terminal | stable | TCP RMSE mm | SO(3) rad |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8800_bowl_init0_s43 | baseline | none | 0 | 0 | 1 | 1 | 1 | 13.903 | 0.05333 |
| 8800_bowl_init0_s43 | hold20 | 3.30 | 20 | 16 | 0 | 0 | 0 | 22.314 | 0.06749 |
| 8800_bowl_init0_s43 | hold40 | 3.30 | 40 | 27 | 0 | 0 | 0 | 23.667 | 0.05912 |
| 8800_bowl_init1_s43 | baseline | none | 0 | 0 | 0 | 0 | 0 | 8.110 | 0.04417 |
| 8800_bowl_init1_s43 | hold20 | 2.90 | 20 | 17 | 1 | 1 | 1 | 8.174 | 0.04353 |
| 8800_bowl_init1_s43 | hold40 | 2.90 | 40 | 28 | 1 | 1 | 1 | 8.203 | 0.04377 |
| 8800_wine_init0_s43 | baseline | none | 0 | 0 | 0 | 0 | 0 | 4.690 | 0.02203 |
| 8800_wine_init0_s43 | hold20 | 2.10 | 20 | 8 | 0 | 0 | 0 | 4.816 | 0.02196 |
| 8800_wine_init0_s43 | hold40 | 2.10 | 40 | 20 | 0 | 0 | 0 | 4.686 | 0.02206 |
| 8800_wine_init2_s43 | baseline | none | 0 | 0 | 0 | 0 | 0 | 10.914 | 0.03659 |
| 8800_wine_init2_s43 | hold20 | none | 0 | 0 | 0 | 0 | 0 | 10.914 | 0.03659 |
| 8800_wine_init2_s43 | hold40 | none | 0 | 0 | 0 | 0 | 0 | 10.914 | 0.03659 |

全部baseline须精确复现原命令、pose、开度、物体、接触、任务谓词与chunk物理/像素SHA；干预前缀含触发步命令前状态但不含被覆盖命令。无触发全轨迹精确一致；hold20/40触发及最初20个干预步一致。

固定原desired_source/opening_target及arm算法/参数/限速，每个反事实使用自己的当前反馈，所以干预后arm命令数值允许变化。原source seed43不被study seed42覆盖。没有WAM查询、重规划、重新锚定、途中state恢复或额外控制步。

这是局部sim privileged执行反事实，不是新WAM闭环或独立成功率。后续参考仍来自原轨迹，不能证明反事实重新观测后模型会生成这些参考；只检验有限时长强制保持的总影响，不能推广至所有失败、部署传感器或唯一权重/释放因果。flattened physics SHA也不认证所有未保存的controller隐藏状态。

完整normal/actual command、命令前contact、触发时刻、开度、canonical TCP误差、原始对象/任务日志、源及产物SHA见outputs/wam_release_hold_20261020/。
