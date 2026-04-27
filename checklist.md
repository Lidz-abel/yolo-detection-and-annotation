# Experiment Checklist

本文件用于统一规范 `YOLO` 项目后续每一轮**正式实验**的执行流程。  
后续只要进入正式训练、正式对比、正式结论，就按这份清单逐项完成。

---

## 1. 实验开始前

- [ ] 明确本轮实验名称
- [ ] 明确本轮主改动点
- [ ] 明确本轮保持不变的变量
- [ ] 确认使用的是正式 `config`
- [ ] 确认不是 smoke test

说明：
- 这一阶段要保证实验是**受控实验**
- 必须说清楚“改了什么、没改什么”

---

## 2. 正式训练

- [ ] 运行 full-run 训练
- [ ] train split 使用完整训练集
- [ ] val split 使用完整验证集
- [ ] 记录训练日志
- [ ] 保存 checkpoint

说明：
- smoke test 只能用于：
  - 语法检查
  - shape 检查
  - loss sanity check
- smoke 结果不能进入正式结论

---

## 3. 正式结果落盘

每轮正式实验结束后，必须确认以下文件存在：

- [ ] `config.toml`
- [ ] `result.txt`
- [ ] `metadata.json`

同时要确认以下信息可追溯：

- [ ] `run_id`
- [ ] `output_dir`
- [ ] `tensorboard_dir`
- [ ] `best.pth`
- [ ] `last.pth`

---

## 4. 评估

每轮正式实验结束后，必须补齐：

- [ ] internal eval
- [ ] COCO subset eval

至少关注以下指标：

- [ ] `precision`
- [ ] `recall`
- [ ] `mAP@0.5`
- [ ] `COCO AP50`
- [ ] `COCO AR@100`
- [ ] `num_predictions`

---

## 5. 可视化

- [ ] 生成 GT vs Pred 可视化图
- [ ] 固定保存若干张可比较样本
- [ ] 观察是否存在：
  - [ ] 套娃框
  - [ ] 误检过多
  - [ ] 漏检
  - [ ] 小目标失败
  - [ ] 排序质量差

说明：
- 可视化不能省略
- 很多问题只看数值不够，必须结合图像解释

---

## 6. 实验总结

每轮正式实验结束后，必须写对应总结 `.md`，至少包含：

- [ ] 本轮改了什么
- [ ] 本轮保持了什么不变
- [ ] 本轮关键结果
- [ ] 与上一轮相比的变化
- [ ] 本轮是否保留
- [ ] 下一轮为什么这样走

---

## 7. 更新索引与总表

每轮正式实验结束后，必须更新：

- [ ] `experiment_result_index.md`
- [ ] `detection_evaluation_summary.md`

要求：
- 每个实验都能对应到：
  - [ ] 配置文件
  - [ ] `run_id`
  - [ ] `result.txt`
  - [ ] `metadata.json`
  - [ ] `eval.json`
  - [ ] `coco_eval.json`

---

## 8. Git 提交

每轮正式实验结束后，必须做一次**专用 Git commit**：

- [ ] 提交训练代码
- [ ] 提交正式配置
- [ ] 提交结果记录
- [ ] 提交阶段总结
- [ ] commit message 能说明本轮实验内容

说明：
- 大型权重、TensorBoard、批量图片可不入库
- 但它们的路径必须能在结果文件中追到

---

## 9. 进入下一轮前

只有满足下面所有条件，才算本轮正式结束：

- [ ] full-run 完成
- [ ] `config.toml / result.txt / metadata.json` 落盘
- [ ] internal eval 完成
- [ ] COCO subset eval 完成
- [ ] visualization 完成
- [ ] summary `.md` 写完
- [ ] 实验索引更新
- [ ] 专用 Git commit 完成

---

## 10. 一句话总流程

后续每轮正式实验统一按下面这条链路执行：

**配置确定 -> full-run -> 结果落盘 -> 评估 -> 可视化 -> 总结 -> 索引更新 -> Git commit**
