# GameTalk: Training LLMs for Strategic Conversation 

**Authors**: Victor Conchello Vendrell, Max Ruiz Luyten, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2601.16276)  

**Abstract**: Strategic decision-making in multi-agent settings is a key challenge for large language models (LLMs), particularly when coordination and negotiation must unfold over extended conversations. While recent work has explored the use of LLMs in isolated decision tasks, little attention has been given to optimizing long-term objectives through dialogue. We introduce \textbf{GameTalk}, a framework for training LLMs to make strategic decisions via multi-turn interactions. Unlike prior work that focuses on single-turn objectives or static action prediction, we train LLMs to optimize a global objective across full conversations. We achieve this by adapting fine-tuning methods like GRPO, DPO, and STaR to incorporate reward signals that depend on the entire interaction. We evaluate this approach on a suite of increasingly complex games, designed to stress different aspects of reasoning, coordination, and opponent modeling. Our results show that GameTalk significantly outperforms untrained models, especially under reward shaping, with DPO consistently yielding the strongest gains. These findings position conversational fine-tuning as a promising path for LLMs to reason, negotiate, and act in interactive environments. 

---
# TL-GRPO: Turn-Level RL for Reasoning-Guided Iterative Optimization 

**Authors**: Peiji Li, Linyang Li, Handa Sun, Wenjin Mai, Yongkang Chen, Xiaozhe Li, Yue Shen, Yichuan Ma, Yiliu Sun, Jiaxi Cao, Zhishu He, Bo Wang, Xiaoqing Zheng, Zhaori Bi, Xipeng Qiu, Qipeng Guo, Kai Chen, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.16480)  

**Abstract**: Large language models have demonstrated strong reasoning capabilities in complex tasks through tool integration, which is typically framed as a Markov Decision Process and optimized with trajectory-level RL algorithms such as GRPO. However, a common class of reasoning tasks, iterative optimization, presents distinct challenges: the agent interacts with the same underlying environment state across turns, and the value of a trajectory is determined by the best turn-level reward rather than cumulative returns. Existing GRPO-based methods cannot perform fine-grained, turn-level optimization in such settings, while black-box optimization methods discard prior knowledge and reasoning capabilities. To address this gap, we propose Turn-Level GRPO (TL-GRPO), a lightweight RL algorithm that performs turn-level group sampling for fine-grained optimization. We evaluate TL-GRPO on analog circuit sizing (ACS), a challenging scientific optimization task requiring multiple simulations and domain expertise. Results show that TL-GRPO outperforms standard GRPO and Bayesian optimization methods across various specifications. Furthermore, our 30B model trained with TL-GRPO achieves state-of-the-art performance on ACS tasks under same simulation budget, demonstrating both strong generalization and practical utility. 

---
# Mixing Expert Knowledge: Bring Human Thoughts Back To the Game of Go 

**Authors**: Yichuan Ma, Linyang Li, Yongkang Chen, Peiji Li, Jiasheng Ye, Qipeng Guo, Dahua Lin, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.16447)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional performance in reasoning tasks such as mathematics and coding, matching or surpassing human capabilities. However, these impressive reasoning abilities face significant challenges in specialized domains. Taking Go as an example, although AlphaGo has established the high performance ceiling of AI systems in Go, mainstream LLMs still struggle to reach even beginner-level proficiency, let alone perform natural language reasoning. This performance gap between general-purpose LLMs and domain experts is significantly limiting the application of LLMs on a wider range of domain-specific tasks. In this work, we aim to bridge the divide between LLMs' general reasoning capabilities and expert knowledge in domain-specific tasks. We perform mixed fine-tuning with structured Go expertise and general long Chain-of-Thought (CoT) reasoning data as a cold start, followed by reinforcement learning to integrate expert knowledge in Go with general reasoning capabilities. Through this methodology, we present \textbf{LoGos}, a powerful LLM that not only maintains outstanding general reasoning abilities, but also conducts Go gameplay in natural language, demonstrating effective strategic reasoning and accurate next-move prediction. LoGos achieves performance comparable to human professional players, substantially surpassing all existing LLMs. Through this work, we aim to contribute insights on applying general LLM reasoning capabilities to specialized domains. We will release the first large-scale Go dataset for LLM training, the first LLM Go evaluation benchmark, and the first general LLM that reaches human professional-level performance in Go at: this https URL. 

---
# Learning Domain Knowledge in Multimodal Large Language Models through Reinforcement Fine-Tuning 

**Authors**: Qinglong Cao, Yuntian Chen, Chao Ma, Xiaokang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.16419)  

**Abstract**: Multimodal large language models (MLLMs) have shown remarkable capabilities in multimodal perception and understanding tasks. However, their effectiveness in specialized domains, such as remote sensing and medical imaging, remains limited. A natural approach to domain adaptation is to inject domain knowledge through textual instructions, prompts, or auxiliary captions. Surprisingly, we find that such input-level domain knowledge injection yields little to no improvement on scientific multimodal tasks, even when the domain knowledge is explicitly provided. This observation suggests that current MLLMs fail to internalize domain-specific priors through language alone, and that domain knowledge must be integrated at the optimization level. Motivated by this insight, we propose a reinforcement fine-tuning framework that incorporates domain knowledge directly into the learning objective. Instead of treating domain knowledge as descriptive information, we encode it as domain-informed constraints and reward signals, shaping the model's behavior in the output space. Extensive experiments across multiple datasets in remote sensing and medical domains consistently demonstrate good performance gains, achieving state-of-the-art results on multimodal domain tasks. Our results highlight the necessity of optimization-level domain knowledge integration and reveal a fundamental limitation of textual domain conditioning in current MLLMs. 

---
