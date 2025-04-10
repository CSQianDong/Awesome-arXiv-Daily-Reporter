# RbFT: Robust Fine-tuning for Retrieval-Augmented Generation against Retrieval Defects 

**Title (ZH)**: RbFT：对抗检索缺陷的鲁棒微调以增强生成检索方法 

**Authors**: Yiteng Tu, Weihang Su, Yujia Zhou, Yiqun Liu, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2501.18365)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by integrating external knowledge retrieved from a knowledge base. However, its effectiveness is fundamentally constrained by the reliability of both the retriever and the knowledge base. In real-world scenarios, imperfections in these components often lead to the retrieval of noisy, irrelevant, or misleading counterfactual information, ultimately undermining the trustworthiness of RAG systems. To address this challenge, we propose Robust Fine-Tuning (RbFT), a method designed to enhance the resilience of LLMs against retrieval defects through two targeted fine-tuning tasks. Experimental results demonstrate that RbFT significantly improves the robustness of RAG systems across diverse retrieval conditions, surpassing existing methods while maintaining high inference efficiency and compatibility with other robustness techniques. 

**Abstract (ZH)**: 检索增强生成（RAG）通过将知识库中检索到的外部知识整合到大型语言模型（LLMs）中来增强其性能。然而，其效果从根本上受到检索器和知识库可靠性的限制。在实际场景中，这些组件的缺陷往往会导致检索到嘈杂的、不相关信息或误导性的反事实信息，从而削弱RAG系统的可信度。为了解决这一挑战，我们提出了一种名为鲁棒微调（RbFT）的方法，该方法通过两个针对特定问题的微调任务来增强LLMs对检索缺陷的鲁棒性。实验结果表明，RbFT显著提高了RAG系统在各种检索条件下的鲁棒性，超越了现有方法，同时保持了高效性和与其他鲁棒性技术的良好兼容性。 

---
# R.I.P.: Better Models by Survival of the Fittest Prompts 

**Title (ZH)**: R.I.P.: 通过适者生存的提示生成更好的模型 

**Authors**: Ping Yu, Weizhe Yuan, Olga Golovneva, Tianhao Wu, Sainbayar Sukhbaatar, Jason Weston, Jing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.18578)  

**Abstract**: Training data quality is one of the most important drivers of final model quality. In this work, we introduce a method for evaluating data integrity based on the assumption that low-quality input prompts result in high variance and low quality responses. This is achieved by measuring the rejected response quality and the reward gap between the chosen and rejected preference pair. Our method, Rejecting Instruction Preferences (RIP) can be used to filter prompts from existing training sets, or to make high quality synthetic datasets, yielding large performance gains across various benchmarks compared to unfiltered data. Using Llama 3.1-8B-Instruct, RIP improves AlpacaEval2 LC Win Rate by 9.4%, Arena-Hard by 8.7%, and WildBench by 9.9%. Using Llama 3.3-70B-Instruct, RIP improves Arena-Hard from 67.5 to 82.9, which is from 18th place to 6th overall in the leaderboard. 

**Abstract (ZH)**: 训练数据质量是最终模型质量的重要驱动因素之一。在本研究中，我们提出了一种基于以下假设的方法：低质量的输入提示会导致高方差和低质量的响应。这种方法通过测量被拒绝的响应质量以及所选和被拒绝的偏好对之间的奖励差距来实现。我们提出的方法，名为拒绝指令偏好（RIP），可以用于从现有训练集中过滤提示，或用于创建高质量的合成数据集，在各种基准测试中，与未过滤的数据相比，其性能显著提高。使用Llama 3.1-8B-Instruct，RIP在AlpacaEval2 LC Win Rate 上提高了9.4%，在Arena-Hard 上提高了8.7%，在WildBench 上提高了9.9%。使用Llama 3.3-70B-Instruct，RIP将Arena-Hard 的得分从67.5提高到82.9，在排行榜上的排名从第18位上升到第6位。 

---
# Leveraging LLM Agents for Automated Optimization Modeling for SASP Problems: A Graph-RAG based Approach 

**Title (ZH)**: 利用大语言模型代理进行SASP问题的自动化优化建模：基于Graph-RAG的方法 

**Authors**: Tianpeng Pan, Wenqiang Pu, Licheng Zhao, Rui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.18320)  

**Abstract**: Automated optimization modeling (AOM) has evoked considerable interest with the rapid evolution of large language models (LLMs). Existing approaches predominantly rely on prompt engineering, utilizing meticulously designed expert response chains or structured guidance. However, prompt-based techniques have failed to perform well in the sensor array signal processing (SASP) area due the lack of specific domain knowledge. To address this issue, we propose an automated modeling approach based on retrieval-augmented generation (RAG) technique, which consists of two principal components: a multi-agent (MA) structure and a graph-based RAG (Graph-RAG) process. The MA structure is tailored for the architectural AOM process, with each agent being designed based on principles of human modeling procedure. The Graph-RAG process serves to match user query with specific SASP modeling knowledge, thereby enhancing the modeling result. Results on ten classical signal processing problems demonstrate that the proposed approach (termed as MAG-RAG) outperforms several AOM benchmarks. 

**Abstract (ZH)**: 自动化优化建模（AOM）随着大型语言模型（LLMs）的迅速发展引起了广泛的关注。现有的方法主要依赖于提示工程，利用精心设计的专家响应链或结构化的指导。然而，在传感器阵列信号处理（SASP）领域，提示基础的方法由于缺乏特定领域的知识而表现不佳。为了解决这一问题，我们提出了一种基于检索增强生成（RAG）技术的自动化建模方法，该方法主要包括两个主要组成部分：一个多代理（MA）结构和基于图的RAG（Graph-RAG）过程。多代理结构专门针对架构化的AOM流程，其中每个代理都是基于人类建模过程的原则设计的。基于图的RAG过程用于匹配用户查询与特定的SASP建模知识，从而提高建模结果。在十个经典信号处理问题上的实验结果显示，所提出的方法（称为MAG-RAG）在几个AOM基准中表现更优。 

---
