# CCrepairBench: A High-Fidelity Benchmark and Reinforcement Learning Framework for C++ Compilation Repair 

**Authors**: Weixuan Sun, Jucai Zhai, Dengfeng Liu, Xin Zhang, Xiaojun Wu, Qiaobo Hao, AIMgroup, Yang Fang, Jiuyang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15690)  

**Abstract**: The automated repair of C++ compilation errors presents a significant challenge, the resolution of which is critical for developer productivity. Progress in this domain is constrained by two primary factors: the scarcity of large-scale, high-fidelity datasets and the limitations of conventional supervised methods, which often fail to generate semantically correct this http URL paper addresses these gaps by introducing a comprehensive framework with three core contributions. First, we present CCrepair, a novel, large-scale C++ compilation error dataset constructed through a sophisticated generate-and-verify pipeline. Second, we propose a Reinforcement Learning (RL) paradigm guided by a hybrid reward signal, shifting the focus from mere compilability to the semantic quality of the fix. Finally, we establish the robust, two-stage evaluation system providing this signal, centered on an LLM-as-a-Judge whose reliability has been rigorously validated against the collective judgments of a panel of human experts. This integrated approach aligns the training objective with generating high-quality, non-trivial patches that are both syntactically and semantically correct. The effectiveness of our approach was demonstrated experimentally. Our RL-trained Qwen2.5-1.5B-Instruct model achieved performance comparable to a Qwen2.5-14B-Instruct model, validating the efficiency of our training paradigm. Our work provides the research community with a valuable new dataset and a more effective paradigm for training and evaluating robust compilation repair models, paving the way for more practical and reliable automated programming assistants. 

---
# Best-of-L: Cross-Lingual Reward Modeling for Mathematical Reasoning 

**Authors**: Sara Rajaee, Rochelle Choenni, Ekaterina Shutova, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2509.15811)  

**Abstract**: While the reasoning abilities of large language models (LLMs) continue to advance, it remains unclear how such ability varies across languages in multilingual LLMs and whether different languages produce reasoning paths that complement each other. To investigate this question, we train a reward model to rank generated responses for a given question across languages. Our results show that our cross-lingual reward model substantially improves mathematical reasoning performance compared to using reward modeling within a single language, benefiting even high-resource languages. While English often exhibits the highest performance in multilingual models, we find that cross-lingual sampling particularly benefits English under low sampling budgets. Our findings reveal new opportunities to improve multilingual reasoning by leveraging the complementary strengths of diverse languages. 

---
# Reward Hacking Mitigation using Verifiable Composite Rewards 

**Authors**: Mirza Farhan Bin Tarek, Rahmatollah Beheshti  

**Link**: [PDF](https://arxiv.org/pdf/2509.15557)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has recently shown that large language models (LLMs) can develop their own reasoning without direct supervision. However, applications in the medical domain, specifically for question answering, are susceptible to significant reward hacking during the reasoning phase. Our work addresses two primary forms of this behavior: i) providing a final answer without preceding reasoning, and ii) employing non-standard reasoning formats to exploit the reward mechanism. To mitigate these, we introduce a composite reward function with specific penalties for these behaviors. Our experiments show that extending RLVR with our proposed reward model leads to better-formatted reasoning with less reward hacking and good accuracy compared to the baselines. This approach marks a step toward reducing reward hacking and enhancing the reliability of models utilizing RLVR. 

---
# It Depends: Resolving Referential Ambiguity in Minimal Contexts with Commonsense Knowledge 

**Authors**: Lukas Ellinger, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2509.16107)  

**Abstract**: Ambiguous words or underspecified references require interlocutors to resolve them, often by relying on shared context and commonsense knowledge. Therefore, we systematically investigate whether Large Language Models (LLMs) can leverage commonsense to resolve referential ambiguity in multi-turn conversations and analyze their behavior when ambiguity persists. Further, we study how requests for simplified language affect this capacity. Using a novel multilingual evaluation dataset, we test DeepSeek v3, GPT-4o, Qwen3-32B, GPT-4o-mini, and Llama-3.1-8B via LLM-as-Judge and human annotations. Our findings indicate that current LLMs struggle to resolve ambiguity effectively: they tend to commit to a single interpretation or cover all possible references, rather than hedging or seeking clarification. This limitation becomes more pronounced under simplification prompts, which drastically reduce the use of commonsense reasoning and diverse response strategies. Fine-tuning Llama-3.1-8B with Direct Preference Optimization substantially improves ambiguity resolution across all request types. These results underscore the need for advanced fine-tuning to improve LLMs' handling of ambiguity and to ensure robust performance across diverse communication styles. 

---
# Fleming-R1: Toward Expert-Level Medical Reasoning via Reinforcement Learning 

**Authors**: Chi Liu, Derek Li, Yan Shu, Robin Chen, Derek Duan, Teng Fang, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2509.15279)  

**Abstract**: While large language models show promise in medical applications, achieving expert-level clinical reasoning remains challenging due to the need for both accurate answers and transparent reasoning processes. To address this challenge, we introduce Fleming-R1, a model designed for verifiable medical reasoning through three complementary innovations. First, our Reasoning-Oriented Data Strategy (RODS) combines curated medical QA datasets with knowledge-graph-guided synthesis to improve coverage of underrepresented diseases, drugs, and multi-hop reasoning chains. Second, we employ Chain-of-Thought (CoT) cold start to distill high-quality reasoning trajectories from teacher models, establishing robust inference priors. Third, we implement a two-stage Reinforcement Learning from Verifiable Rewards (RLVR) framework using Group Relative Policy Optimization, which consolidates core reasoning skills while targeting persistent failure modes through adaptive hard-sample mining. Across diverse medical benchmarks, Fleming-R1 delivers substantial parameter-efficient improvements: the 7B variant surpasses much larger baselines, while the 32B model achieves near-parity with GPT-4o and consistently outperforms strong open-source alternatives. These results demonstrate that structured data design, reasoning-oriented initialization, and verifiable reinforcement learning can advance clinical reasoning beyond simple accuracy optimization. We release Fleming-R1 publicly to promote transparent, reproducible, and auditable progress in medical AI, enabling safer deployment in high-stakes clinical environments. 

---
