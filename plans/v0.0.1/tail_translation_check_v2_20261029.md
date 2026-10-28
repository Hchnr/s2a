# 末周期平移恢复：独立检查器v2的类型比较修复

这是执行前登记的check-only修复。原生产实验14调用已成功关闭，原独立checker实际完成14次完整复算后在元数据审计失败；不覆盖旧产物、不重跑producer、不新增或选择其他候选、不改变科学门槛。原[14调用计划](tail_translation_reserve_20261029.md)保持，生产科学结果尚未取得完整独立资格。

## 已执行失败及固定来源

所有路径相对仓库根目录。

|来源|路径|SHA256|
|---|---|---|
|原95源QA|`outputs/tail_translation_reserve_native_qa_20261029_lkpp55kq/result.json`|`e421b3860faa93407194fad3cc57eabd56d16d6eea58bb46998a90dc62dde663`|
|producer|`outputs/tracking_v5_tail_translation_reserve_20261029/verified.json`|`0305dc39480abf4512205f8628729fd789595d40aeca189343bc19779f029d70`|
|producer owner|`outputs/tracking_v5_tail_translation_reserve_execution_20261029/result.json`|`f604b9e31b1a4aa00a576b099065334fd04aae8722ac5710165b080c2c76b6cb`|
|失败checker|`outputs/tracking_v5_tail_translation_reserve_check_20261029/verified.json`|`f7582560bc9dce92f56632b8c7688c9b4403b50ce23be4dc1d68b818d3600751`|
|失败owner|`outputs/tracking_v5_tail_translation_reserve_check_execution_20261029/result.json`|`22a0be80635c2023c2cc95b2a088eb0c9b51aa5e2c362c7b4a52c97eae88da8f`|
|失败日志|`outputs/tracking_v5_tail_translation_reserve_check_execution_20261029/capture/command.log`|`876e35928e2245bf60f093cf22cd2e1abd24af0e23506bafb1ad234b6d3df7a9`|

原句柄58216实际exit1，child569716/start616909863实际wait1/reap，无信号、无自动重试。失败前已核14完整响应、4200数值内步/4369查询，原严格分类差异为空；随后 `_tree` 将historical完整entry中的布尔`clipped`数组交给仅做数值相减的比较器，引发NumPy boolean subtract TypeError。故完整ledger及fresh因果/恢复检查未完成，不能据前半段复算取得完整资格。旧合成审计默认比较器会将数组转float，未覆盖actual checker传入的比较函数，此次需补真实接线回归。

## 唯一修复范围

新增隔离checker、evidence、check-only owner、测试和本计划共5源；旧95源和全部已执行产物保持原SHA。新100源/30模块QA保留旧95同字节子集。

布尔数组要求双方均为bool且shape/值exact；整数标签数组要求双方均为整数且shape/值exact，禁止与float或bool混用。标签比较返回差异0，禁止布尔相减或放宽标签容差。浮点比较继续调用同一冻结比较器与原绝对容差。候选、原九组审计、E17与排序、原buffer/caps、原生响应模型、初态、计账、所有科学规则均不改变。

QA需覆盖原冻结比较函数的实际布尔失败、新比较器相同标签通过/单bit或shape或类型变化拒绝、整数标签及原浮点边界，并在无模型/0query下将原真实producer完整answer经过新比较函数完成原ledger审计。不能只测试独立audit默认比较器。

## 固定check-only预算与关闭

新入口仍从同一actual368和completed367 latch完整复算原14条含末repeat，固定13唯一命令、4200内步/4369查询、0优化迭代。新输出独占 `outputs/tracking_v5_tail_translation_reserve_check_v2_20261029` 与对应 `_execution_20261029`；禁止覆盖、跳过已知点、复用失败进程内未保存的rollout或自动重试。CPU环境与seed42保持，GPU禁用。新checker技术失败保留原因、计数和退出状态。

执行前完成100源QA、Ruff/format、checker与check-only owner两条实际dry。owner仅运行新checker，绑定原生产owner正常关闭和原95源QA；不要求原producer伪装使用新QA。新checker的原plan_ref仍绑定生产14调用计划，另用repair_plan_ref绑定本计划和失败refs。所有14完整复算及metadata/fresh严格检查通过，且正常wait0/reap、所有新旧来源闭合，方可确认原生产恢复结论。

计入失败成本：原producer14 + 失败checker14 + 新checker14 = 42次完整数值响应、12600内步/13107查询。新checker本身只新增14；新物理/world-model/fresh/参考消费均0，实际仍cursor8/task0/grasp0。完整H12 witness与原18/12/24任务门槛不变；恢复成功也必须另登记后续全72维规划，不能据本修复进入物理pilot。
