# Motif-2-12.7B-Reasoning: A Practitioner's Guide to RL Training Recipes 

**Authors**: Junghwan Lim, Sungmin Lee, Dongseok Kim, Taehyun Kim, Eunhwan Park, Jeesoo Lee, Jeongdoo Lee, Junhyeok Lee, Wai Ting Cheung, Dahye Choi, Minsu Ha, Jaeheui Her, Jaeyeon Huh, Hanbin Jung, Changjin Kang, Beomgyu Kim, Minjae Kim, Taewhan Kim, Youngrok Kim, Hyukjin Kweon, Haesol Lee, Kungyu Lee, Dongpin Oh, Yeongjae Park, Bokki Ryu, Dongjoo Weon  

**Link**: [PDF](https://arxiv.org/pdf/2512.11463)  

**Abstract**: We introduce Motif-2-12.7B-Reasoning, a 12.7B parameter language model designed to bridge the gap between open-weight systems and proprietary frontier models in complex reasoning and long-context understanding. Addressing the common challenges of model collapse and training instability in reasoning adaptation, we propose a comprehensive, reproducible training recipe spanning system, data, and algorithmic optimizations. Our approach combines memory-efficient infrastructure for 64K-token contexts using hybrid parallelism and kernel-level optimizations with a two-stage Supervised Fine-Tuning (SFT) curriculum that mitigates distribution mismatch through verified, aligned synthetic data. Furthermore, we detail a robust Reinforcement Learning Fine-Tuning (RLFT) pipeline that stabilizes training via difficulty-aware data filtering and mixed-policy trajectory reuse. Empirical results demonstrate that Motif-2-12.7B-Reasoning achieves performance comparable to models with significantly larger parameter counts across mathematics, coding, and agentic benchmarks, offering the community a competitive open model and a practical blueprint for scaling reasoning capabilities under realistic compute constraints. 

---
# Towards Trustworthy Multi-Turn LLM Agents via Behavioral Guidance 

**Authors**: Gonca Gürsun  

**Link**: [PDF](https://arxiv.org/pdf/2512.11421)  

**Abstract**: Large Language Models demonstrate strong reasoning and generation abilities, yet their behavior in multi-turn tasks often lacks reliability and verifiability. We present a task completion framework that enables LLM-based agents to act under explicit behavioral guidance in environments described by reinforcement learning formalisms with defined observation, action, and reward signals.
The framework integrates three components: a lightweight task profiler that selects reasoning and generation strategies, a reasoning module that learns verifiable observation - action mappings, and a generation module that enforces constraint-compliant outputs through validation or deterministic synthesis. We show that as the agent interacts with the environment, these components co-evolve, yielding trustworthy behavior. 

---
# SUMFORU: An LLM-Based Review Summarization Framework for Personalized Purchase Decision Support 

**Authors**: Yuming Feng, Xinrui Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2512.11755)  

**Abstract**: Online product reviews contain rich but noisy signals that overwhelm users and hinder effective decision-making. Existing LLM-based summarizers remain generic and fail to account for individual preferences, limiting their practical utility. We propose SUMFORU, a steerable review summarization framework that aligns outputs with explicit user personas to support personalized purchase decisions. Our approach integrates a high-quality data pipeline built from the Amazon 2023 Review Dataset with a two-stage alignment procedure: (1) persona-aware Supervised Fine-Tuning (SFT) via asymmetric knowledge distillation, and (2) Reinforcement Learning with AI Feedback (RLAIF) using a preference estimator to capture fine-grained, persona-relevant signals. We evaluate the model across rule-based, LLM-based, and human-centered metrics, demonstrating consistent improvements in consistency, grounding, and preference alignment. Our framework achieves the highest performance across all evaluation settings and generalizes effectively to unseen product categories. Our results highlight the promise of steerable pluralistic alignment for building next-generation personalized decision-support systems. 

---
# When Actions Teach You to Think: Reasoning-Action Synergy via Reinforcement Learning in Conversational Agents 

**Authors**: Mrinal Rawat, Arkajyoti Chakraborty, Neha Gupta, Roberto Pieraccini  

**Link**: [PDF](https://arxiv.org/pdf/2512.11277)  

**Abstract**: Supervised fine-tuning (SFT) has emerged as one of the most effective ways to improve the performance of large language models (LLMs) in downstream tasks. However, SFT can have difficulty generalizing when the underlying data distribution changes, even when the new data does not fall completely outside the training domain. Recent reasoning-focused models such as o1 and R1 have demonstrated consistent gains over their non-reasoning counterparts, highlighting the importance of reasoning for improved generalization and reliability. However, collecting high-quality reasoning traces for SFT remains challenging -- annotations are costly, subjective, and difficult to scale. To address this limitation, we leverage Reinforcement Learning (RL) to enable models to learn reasoning strategies directly from task outcomes. We propose a pipeline in which LLMs generate reasoning steps that guide both the invocation of tools (e.g., function calls) and the final answer generation for conversational agents. Our method employs Group Relative Policy Optimization (GRPO) with rewards designed around tool accuracy and answer correctness, allowing the model to iteratively refine its reasoning and actions. Experimental results demonstrate that our approach improves both the quality of reasoning and the precision of tool invocations, achieving a 1.5% relative improvement over the SFT model (trained without explicit thinking) and a 40% gain compared to the base of the vanilla Qwen3-1.7B model. These findings demonstrate the promise of unifying reasoning and action learning through RL to build more capable and generalizable conversational agents. 

---
# KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering 

**Authors**: Xin Sun, Zhongqi Chen, Xing Zheng, Qiang Liu, Shu Wu, Bowen Song, Zilei Wang, Weiqiang Wang, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.10999)  

**Abstract**: Knowledge Base Question Answering (KBQA) challenges models to bridge the gap between natural language and strict knowledge graph schemas by generating executable logical forms. While Large Language Models (LLMs) have advanced this field, current approaches often struggle with a dichotomy of failure: they either generate hallucinated queries without verifying schema existence or exhibit rigid, template-based reasoning that mimics synthesized traces without true comprehension of the environment. To address these limitations, we present \textbf{KBQA-R1}, a framework that shifts the paradigm from text imitation to interaction optimization via Reinforcement Learning. Treating KBQA as a multi-turn decision process, our model learns to navigate the knowledge base using a list of actions, leveraging Group Relative Policy Optimization (GRPO) to refine its strategies based on concrete execution feedback rather than static supervision. Furthermore, we introduce \textbf{Referenced Rejection Sampling (RRS)}, a data synthesis method that resolves cold-start challenges by strictly aligning reasoning traces with ground-truth action sequences. Extensive experiments on WebQSP, GrailQA, and GraphQuestions demonstrate that KBQA-R1 achieves state-of-the-art performance, effectively grounding LLM reasoning in verifiable execution. 

---
