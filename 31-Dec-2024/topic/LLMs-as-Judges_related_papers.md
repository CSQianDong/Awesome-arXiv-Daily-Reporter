# Right vs. Right: Can LLMs Make Tough Choices? 

**Title (ZH)**: “正确” vs. “正确”：LLM能够作出艰难选择吗？ 

**Authors**: Jiaqing Yuan, Pradeep K. Murukannaiah, Munindar P. Singh  

**Link**: [PDF](https://arxiv.org/pdf/2412.19926)  

**Abstract**: An ethical dilemma describes a choice between two "right" options involving conflicting moral values. We present a comprehensive evaluation of how LLMs navigate ethical dilemmas. Specifically, we investigate LLMs on their (1) sensitivity in comprehending ethical dilemmas, (2) consistency in moral value choice, (3) consideration of consequences, and (4) ability to align their responses to a moral value preference explicitly or implicitly specified in a prompt. Drawing inspiration from a leading ethical framework, we construct a dataset comprising 1,730 ethical dilemmas involving four pairs of conflicting values. We evaluate 20 well-known LLMs from six families. Our experiments reveal that: (1) LLMs exhibit pronounced preferences between major value pairs, and prioritize truth over loyalty, community over individual, and long-term over short-term considerations. (2) The larger LLMs tend to support a deontological perspective, maintaining their choices of actions even when negative consequences are specified. (3) Explicit guidelines are more effective in guiding LLMs' moral choice than in-context examples. Lastly, our experiments highlight the limitation of LLMs in comprehending different formulations of ethical dilemmas. 

**Abstract (ZH)**: 伦理困境是指在涉及冲突道德价值观的情况下，需要在两个“正确”的选择之间做出的抉择。本文对大型语言模型（LLM）在处理伦理困境时的路径进行了全方位评估。具体来说，我们考察了LLM在以下方面的表现：（1）对伦理困境的理解敏感度，（2）在道德价值选择上的一致性，（3）对后果的考虑，以及（4）将回应与明示或隐含在提示中指定的道德价值观偏好相一致的能力。借鉴领先伦理框架的灵感，我们构建了一个包含1730个伦理困境的数据集，这些困境涉及四种冲突价值的两两组合。我们评估了来自六个家庭的20个知名LLM。实验结果显示：（1）LLM在主要价值对之间表现出明显的偏好，并倾向于优先选择真实性而非忠诚度，集体利益而非个体利益，长期利益而非短期利益。（2）较大的LLM倾向于坚持原则性视角，在负面后果被指明的情况下仍保持其行为选择。（3）明确的指导方针比上下文中的例子在引导LLM的道德选择方面更为有效。最后，我们的实验揭示了LLM在理解不同表述形式的伦理困境方面的局限性。 

---
# Planning, Living and Judging: A Multi-agent LLM-based Framework for Cyclical Urban Planning 

**Title (ZH)**: 规划、居住与评判：基于多智能体LLM框架的循环城市规划系统 

**Authors**: Hang Ni, Yuzhi Wang, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20505)  

**Abstract**: Urban regeneration presents significant challenges within the context of urbanization, requiring adaptive approaches to tackle evolving needs. Leveraging advancements in large language models (LLMs), we propose Cyclical Urban Planning (CUP), a new paradigm that continuously generates, evaluates, and refines urban plans in a closed-loop. Specifically, our multi-agent LLM-based framework consists of three key components: (1) Planning, where LLM agents generate and refine urban plans based on contextual data; (2) Living, where agents simulate the behaviors and interactions of residents, modeling life in the urban environment; and (3) Judging, which involves evaluating plan effectiveness and providing iterative feedback for improvement. The cyclical process enables a dynamic and responsive planning approach. Experiments on the real-world dataset demonstrate the effectiveness of our framework as a continuous and adaptive planning process. 

**Abstract (ZH)**: 城市再开发在城市化进程中面临显著挑战，要求采用适应性策略来应对不断变化的需求。借助大型语言模型（LLMs）的进步，我们提出了一种新的范式——循环城市规划（Cyclical Urban Planning，CUP），该范式通过闭环不断生成、评估和优化城市规划。具体而言，我们的基于多智能体的大型语言模型框架包括三个关键组成部分：（1）规划阶段，LLM智能体基于上下文数据生成和优化城市规划；（2）生活阶段，智能体模拟居民的行为和互动，模型化城市环境中的人类生活；（3）评估阶段，涉及评估规划的有效性并提供迭代反馈以进行改进。循环过程使得规划方法具有动态和响应性。实验结果表明，我们的框架作为连续且适应性强的规划过程是有效的。 

---
# Multi-Objective Large Language Model Unlearning 

**Title (ZH)**: 多目标大型语言模型去学习 

**Authors**: Zibin Pan, Shuwen Zhang, Yuesheng Zheng, Chi Li, Yuheng Cheng, Junhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.20412)  

**Abstract**: Machine unlearning in the domain of large language models (LLMs) has attracted great attention recently, which aims to effectively eliminate undesirable behaviors from LLMs without full retraining from scratch. In this paper, we explore the Gradient Ascent (GA) approach in LLM unlearning, which is a proactive way to decrease the prediction probability of the model on the target data in order to remove their influence. We analyze two challenges that render the process impractical: gradient explosion and catastrophic forgetting. To address these issues, we propose Multi-Objective Large Language Model Unlearning (MOLLM) algorithm. We first formulate LLM unlearning as a multi-objective optimization problem, in which the cross-entropy loss is modified to the unlearning version to overcome the gradient explosion issue. A common descent update direction is then calculated, which enables the model to forget the target data while preserving the utility of the LLM. Our empirical results verify that MoLLM outperforms the SOTA GA-based LLM unlearning methods in terms of unlearning effect and model utility preservation. 

**Abstract (ZH)**: 在大型语言模型（LLMs）领域的机器遗忘问题最近引起了广泛的关注，其目标是在不完全从头开始重新训练的情况下，有效消除大型语言模型中的不良行为。本文探讨了在LLMs遗忘中应用梯度上升（GA）方法，这是一种主动的手段，旨在通过降低模型对目标数据的预测概率来减少其影响。我们分析了导致这一过程不可行的两个挑战：梯度爆炸和灾难性遗忘。为了解决这些问题，我们提出了多目标大型语言模型遗忘算法（MOLLM）。我们首先将LLMs遗忘问题形式化为一个多目标优化问题，在其中通过修改交叉熵损失为遗忘版本来克服梯度爆炸问题。然后计算了一个共同的下降更新方向，使模型能够忘记目标数据同时保留大型语言模型的有用性。我们的实证结果验证了MOLLM在遗忘效果和模型有用性保留方面优于现有的基于GA的大型语言模型遗忘方法。 

---
