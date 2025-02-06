# Training an LLM-as-a-Judge Model: Pipeline, Insights, and Practical Lessons 

**Title (ZH)**: 训练作为裁判的大型语言模型：流程、见解与实践经验 

**Authors**: Renjun Hu, Yi Cheng, Libin Meng, Jiaxin Xia, Yi Zong, Xing Shi, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.02988)  

**Abstract**: The rapid advancement of large language models (LLMs) has opened new possibilities for their adoption as evaluative judges. This paper introduces Themis, a fine-tuned LLM judge that delivers sophisticated context-aware evaluations. We provide a comprehensive overview of the development pipeline for Themis, highlighting its scenario-dependent evaluation prompts and two novel methods for controlled instruction generation. These designs enable Themis to effectively distill evaluative skills from teacher models, while retaining flexibility for continuous development. We introduce two human-labeled benchmarks for meta-evaluation, demonstrating that Themis can achieve high alignment with human preferences in an economical manner. Additionally, we explore insights into the LLM-as-a-judge paradigm, revealing nuances in performance and the varied effects of reference answers. Notably, we observe that pure knowledge distillation from strong LLMs, though common, does not guarantee performance improvement through scaling. We propose a mitigation strategy based on instruction-following difficulty. Furthermore, we provide practical guidelines covering data balancing, prompt customization, multi-objective training, and metric aggregation. We aim for our method and findings, along with the fine-tuning data, benchmarks, and model checkpoints, to support future research and development in this area. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展为将其作为评估法官的应用开启了新的可能性。本文介绍了Themis，这是一种细调的LLM法官，能够提供复杂的上下文感知评估。我们详细介绍了Themis的开发流程，强调了其场景依赖的评估提示，并介绍了两种新的控制指令生成方法。这些设计使得Themis能够有效地从教师模型中提炼评估技能，同时保留持续开发的灵活性。我们介绍了两个元评估的人工标注基准，展示了Themis能够在经济有效的方式下实现对人类偏好的高度一致。此外，我们探讨了LLM作为法官的范式，揭示了其性能中的复杂性以及参考答案的多样效用。值得注意的是，我们观察到，尽管从强大的LLM中提取纯粹的知识是一种常见做法，但通过扩展并不能保证性能的提升。我们提出了基于指令跟随难度的缓解策略。此外，我们还提供了关于数据平衡、提示定制、多目标训练和指标聚合的实用指南。我们希望我们的方法和发现，包括细调数据、基准和模型检查点，能够支持该领域未来的研究和发展。 

---
