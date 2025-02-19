# MCTS-Judge: Test-Time Scaling in LLM-as-a-Judge for Code Correctness Evaluation 

**Title (ZH)**: MCTS-Judge：作为代码正确性评估的LLM法官的测试时缩放 

**Authors**: Yutong Wang, Pengliang Ji, Chaoqun Yang, Kaixin Li, Ming Hu, Jiaoyang Li, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2502.12468)  

**Abstract**: The LLM-as-a-Judge paradigm shows promise for evaluating generative content but lacks reliability in reasoning-intensive scenarios, such as programming. Inspired by recent advances in reasoning models and shifts in scaling laws, we pioneer bringing test-time computation into LLM-as-a-Judge, proposing MCTS-Judge, a resource-efficient, System-2 thinking framework for code correctness evaluation. MCTS-Judge leverages Monte Carlo Tree Search (MCTS) to decompose problems into simpler, multi-perspective evaluations. Through a node-selection strategy that combines self-assessment based on historical actions in the current trajectory and the Upper Confidence Bound for Trees based on prior rollouts, MCTS-Judge balances global optimization and refinement of the current trajectory. We further designed a high-precision, unit-test-level reward mechanism to encourage the Large Language Model (LLM) to perform line-by-line analysis. Extensive experiments on three benchmarks and five LLMs demonstrate the effectiveness of MCTS-Judge, which improves the base model's accuracy from 41% to 80%, surpassing the o1-series models with 3x fewer tokens. Further evaluations validate the superiority of its reasoning trajectory in logic, analytics, thoroughness, and overall quality, while revealing the test-time scaling law of the LLM-as-a-Judge paradigm. 

**Abstract (ZH)**: LLM作为法官的范式在评估生成内容方面展现出潜力，但在编程等需要推理的场景中缺乏可靠性。借鉴最近推理模型的进步和规模法则的转变，我们率先将推理时的计算引入LLM作为法官的情景，提出了MCTS-Judge，这是一种资源高效的、适用于代码正确性评估的系统-2思维框架。MCTS-Judge利用蒙特卡洛树搜索（MCTS）将问题分解为更简单的多视角评估。通过结合基于当前轨迹历史行为的自我评估和基于先验展开的树的信心上限（UCB）的节点选择策略，MCTS-Judge平衡了全局优化和当前轨迹的细化。我们进一步设计了一种高精度、单元测试级别的奖励机制，鼓励大型语言模型（LLM）进行逐行分析。在三个基准和五种LLM的广泛实验中，MCTS-Judge展示了其有效性，将基线模型的准确率从41%提高到80%，并且在使用三分之一更少的令牌时超越了o1系列模型。进一步的评估验证了其推理轨迹在逻辑性、分析能力、覆盖面和整体质量上的优越性，同时揭示了LLM作为法官范式的推理时的规模法则。 

---
