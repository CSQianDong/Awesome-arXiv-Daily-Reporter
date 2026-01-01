# Figure It Out: Improving the Frontier of Reasoning with Active Visual Thinking 

**Authors**: Meiqi Chen, Fandong Meng, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.24297)  

**Abstract**: Complex reasoning problems often involve implicit spatial, geometric, and structural relationships that are not explicitly encoded in text. While recent reasoning models have achieved strong performance across many domains, purely text-based reasoning struggles to represent global structural constraints in complex settings. In this paper, we introduce FIGR, which integrates active visual thinking into multi-turn reasoning via end-to-end reinforcement learning. FIGR externalizes intermediate structural hypotheses by constructing visual representations during problem solving. By adaptively regulating when and how visual reasoning should be invoked, FIGR enables more stable and coherent reasoning over global structural properties that are difficult to capture from text alone. Experiments on challenging mathematical reasoning benchmarks demonstrate that FIGR outperforms strong text-only chain-of-thought baselines. In particular, FIGR improves the base model by 13.12% on AIME 2025 and 11.00% on BeyondAIME, highlighting the effectiveness of figure-guided multimodal reasoning in enhancing the stability and reliability of complex reasoning. 

---
# CEC-Zero: Zero-Supervision Character Error Correction with Self-Generated Rewards 

**Authors**: Zhiming Lin, Kai Zhao, Sophie Zhang, Peilai Yu, Canran Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2512.23971)  

**Abstract**: Large-scale Chinese spelling correction (CSC) remains critical for real-world text processing, yet existing LLMs and supervised methods lack robustness to novel errors and rely on costly annotations. We introduce CEC-Zero, a zero-supervision reinforcement learning framework that addresses this by enabling LLMs to correct their own mistakes. CEC-Zero synthesizes errorful inputs from clean text, computes cluster-consensus rewards via semantic similarity and candidate agreement, and optimizes the policy with PPO. It outperforms supervised baselines by 10--13 F$_1$ points and strong LLM fine-tunes by 5--8 points across 9 benchmarks, with theoretical guarantees of unbiased rewards and convergence. CEC-Zero establishes a label-free paradigm for robust, scalable CSC, unlocking LLM potential in noisy text pipelines. 

---
# QianfanHuijin Technical Report: A Novel Multi-Stage Training Paradigm for Finance Industrial LLMs 

**Authors**: Shupeng Li, Weipeng Lu, Linyun Liu, Chen Lin, Shaofei Li, Zhendong Tan, Hanjun Zhong, Yucheng Zeng, Chenghao Zhu, Mengyue Liu, Daxiang Dong, Jianmin Wu, Yunting Xiao, Annan Li, Danyu Liu, Jingnan Zhang, Licen Liu, Dawei Yin, Dou Shen  

**Link**: [PDF](https://arxiv.org/pdf/2512.24314)  

**Abstract**: Domain-specific enhancement of Large Language Models (LLMs) within the financial context has long been a focal point of industrial application. While previous models such as BloombergGPT and Baichuan-Finance primarily focused on knowledge enhancement, the deepening complexity of financial services has driven a growing demand for models that possess not only domain knowledge but also robust financial reasoning and agentic capabilities. In this paper, we present QianfanHuijin, a financial domain LLM, and propose a generalizable multi-stage training paradigm for industrial model enhancement.
Our approach begins with Continual Pre-training (CPT) on financial corpora to consolidate the knowledge base. This is followed by a fine-grained Post-training pipeline designed with increasing specificity: starting with Financial SFT, progressing to Finance Reasoning RL and Finance Agentic RL, and culminating in General RL aligned with real-world business scenarios. Empirical results demonstrate that QianfanHuijin achieves superior performance across various authoritative financial benchmarks. Furthermore, ablation studies confirm that the targeted Reasoning RL and Agentic RL stages yield significant gains in their respective capabilities. These findings validate our motivation and suggest that this fine-grained, progressive post-training methodology is poised to become a mainstream paradigm for various industrial-enhanced LLMs. 

---
# Iterative Deployment Improves Planning Skills in LLMs 

**Authors**: Augusto B. Corrêa, Yoav Gelberg, Luckeciano C. Melo, Ilia Shumailov, André G. Pereira, Yarin Gal  

**Link**: [PDF](https://arxiv.org/pdf/2512.24940)  

**Abstract**: We show that iterative deployment of large language models (LLMs), each fine-tuned on data carefully curated by users from the previous models' deployment, can significantly change the properties of the resultant models. By testing this mechanism on various planning domains, we observe substantial improvements in planning skills, with later models displaying emergent generalization by discovering much longer plans than the initial models. We then provide theoretical analysis showing that iterative deployment effectively implements reinforcement learning (RL) training in the outer-loop (i.e. not as part of intentional model training), with an implicit reward function. The connection to RL has two important implications: first, for the field of AI safety, as the reward function entailed by repeated deployment is not defined explicitly, and could have unexpected implications to the properties of future model deployments. Second, the mechanism highlighted here can be viewed as an alternative training regime to explicit RL, relying on data curation rather than explicit rewards. 

---
# From Building Blocks to Planning: Multi-Step Spatial Reasoning in LLMs with Reinforcement Learning 

**Authors**: Amir Tahmasbi, Sadegh Majidi, Kazem Taram, Aniket Bera  

**Link**: [PDF](https://arxiv.org/pdf/2512.24532)  

**Abstract**: Spatial reasoning in large language models (LLMs) has gained increasing attention due to applications in navigation and planning. Despite strong general language capabilities, LLMs still struggle with spatial transformations and multi-step planning in structured environments. We propose a two-stage approach that decomposes spatial reasoning into atomic building blocks and their composition. First, we apply supervised fine-tuning on elementary spatial transformations, such as rotation, translation, and scaling, to equip the model with basic spatial physics. We then freeze this physics-aware model and train lightweight LoRA adapters within the GRPO framework to learn policies that compose these building blocks for multi-step planning in puzzle-based environments, in a closed-loop manner. To support this pipeline, we synthesize an ASCII-art dataset and construct a corresponding ASCII-based reinforcement learning environment. Our method consistently outperforms baselines, including the generic backbone, physics-aware model, and end-to-end RL models, under both Dynamic environments with explicit state updates and Static environments where the model must rely on its internal state across steps. In addition, the proposed approach converges faster and exhibits more stable training compared to end-to-end reinforcement learning from scratch. Finally, we analyze attention patterns to assess whether fine-tuning induces meaningful improvements in spatial understanding. 

---
# AHA: Aligning Large Audio-Language Models for Reasoning Hallucinations via Counterfactual Hard Negatives 

**Authors**: Yanxi Chen, Wenhui Zhu, Xiwen Chen, Zhipeng Wang, Xin Li, Peijie Qiu, Hao Wang, Xuanzhao Dong, Yujian Xiong, Anderson Schneider, Yuriy Nevmyvaka, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.24052)  

**Abstract**: Although Large Audio-Language Models (LALMs) deliver state-of-the-art (SOTA) performance, they frequently suffer from hallucinations, e.g. generating text not grounded in the audio input. We analyze these grounding failures and identify a distinct taxonomy: Event Omission, False Event Identity, Temporal Relation Error, and Quantitative Temporal Error. To address this, we introduce the AHA (Audio Hallucination Alignment) framework. By leveraging counterfactual hard negative mining, our pipeline constructs a high-quality preference dataset that forces models to distinguish strict acoustic evidence from linguistically plausible fabrications. Additionally, we establish AHA-Eval, a diagnostic benchmark designed to rigorously test these fine-grained temporal reasoning capabilities. We apply this data to align Qwen2.5-Omni. The resulting model, Qwen-Audio-AHA, achieves a 13.7% improvement on AHA-Eval. Crucially, this benefit generalizes beyond our diagnostic set. Our model shows substantial gains on public benchmarks, including 1.3% on MMAU-Test and 1.6% on MMAR, outperforming latest SOTA methods. 

---
# Youtu-Agent: Scaling Agent Productivity with Automated Generation and Hybrid Policy Optimization 

**Authors**: Yuchen Shi, Yuzheng Cai, Siqi Cai, Zihan Xu, Lichao Chen, Yulei Qin, Zhijian Zhou, Xiang Fei, Chaofan Qiu, Xiaoyu Tan, Gang Li, Zongyi Li, Haojia Lin, Guocan Cai, Yong Mao, Yunsheng Wu, Ke Li, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.24615)  

**Abstract**: Existing Large Language Model (LLM) agent frameworks face two significant challenges: high configuration costs and static capabilities. Building a high-quality agent often requires extensive manual effort in tool integration and prompt engineering, while deployed agents struggle to adapt to dynamic environments without expensive fine-tuning. To address these issues, we propose \textbf{Youtu-Agent}, a modular framework designed for the automated generation and continuous evolution of LLM agents. Youtu-Agent features a structured configuration system that decouples execution environments, toolkits, and context management, enabling flexible reuse and automated synthesis. We introduce two generation paradigms: a \textbf{Workflow} mode for standard tasks and a \textbf{Meta-Agent} mode for complex, non-standard requirements, capable of automatically generating tool code, prompts, and configurations. Furthermore, Youtu-Agent establishes a hybrid policy optimization system: (1) an \textbf{Agent Practice} module that enables agents to accumulate experience and improve performance through in-context optimization without parameter updates; and (2) an \textbf{Agent RL} module that integrates with distributed training frameworks to enable scalable and stable reinforcement learning of any Youtu-Agents in an end-to-end, large-scale manner. Experiments demonstrate that Youtu-Agent achieves state-of-the-art performance on WebWalkerQA (71.47\%) and GAIA (72.8\%) using open-weight models. Our automated generation pipeline achieves over 81\% tool synthesis success rate, while the Practice module improves performance on AIME 2024/2025 by +2.7\% and +5.4\% respectively. Moreover, our Agent RL training achieves 40\% speedup with steady performance improvement on 7B LLMs, enhancing coding/reasoning and searching capabilities respectively up to 35\% and 21\% on Maths and general/multi-hop QA benchmarks. 

---
# Group Deliberation Oriented Multi-Agent Conversational Model for Complex Reasoning 

**Authors**: Zheyu Shi, Dong Qiu, Shanlong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.24613)  

**Abstract**: This paper proposes a group deliberation oriented multi-agent conversational model to address the limitations of single large language models in complex reasoning tasks. The model adopts a three-level role division architecture consisting of generation, verification, and integration. An opinion generation agent produces diverse reasoning perspectives, an evidence verification agent retrieves external knowledge and quantifies factual support, and a consistency arbitration agent integrates logically coherent conclusions. A self-game mechanism is introduced to expand multi-path reasoning trajectories, while a retrieval enhancement module dynamically supplements external knowledge. A composite reward function combining factual consistency and logical coherence is designed, and an improved proximal policy optimization strategy is applied for collaborative training. Experimental results show that the proposed model improves multi-hop reasoning accuracy by 16.8 percent on HotpotQA, 14.3 percent on 2WikiMultihopQA, and 19.2 percent on MeetingBank, while improving consistency by 21.5 percent. The model achieves higher reasoning efficiency than mainstream multi-agent approaches, providing an effective and stable solution for complex reasoning tasks. 

---
# Reinforcement Learning-Augmented LLM Agents for Collaborative Decision Making and Performance Optimization 

**Authors**: Dong Qiu, Duo Xu, Limengxi Yue  

**Link**: [PDF](https://arxiv.org/pdf/2512.24609)  

**Abstract**: Large Language Models (LLMs) perform well in language tasks but often lack collaborative awareness and struggle to optimize global performance in multi-agent settings. We present a reinforcement learning-augmented LLM agent framework that formulates cooperation as a decentralized partially observable Markov decision process (Dec-POMDP) and adopts centralized training with decentralized execution (CTDE). We introduce Group Relative Policy Optimization (GRPO) to jointly optimize agent policies with access to global signals during training, together with a simplified joint reward that balances task quality, speed, and coordination cost. On collaborative writing and coding benchmarks, our framework delivers a 3x increase in task processing speed over single-agent baselines, 98.7% structural/style consistency in writing, and a 74.6% test pass rate in coding. The approach consistently outperforms strong multi-agent LLM baselines and provides a practical path toward reliable collaboration in complex workflows. 

---
# Constrained Language Model Policy Optimization via Risk-aware Stepwise Alignment 

**Authors**: Lijun Zhang, Lin Li, Wei Wei, Yajie Qi, Huizhong Song, Jun Wang, Yaodong Yang, Jiye Liang  

**Link**: [PDF](https://arxiv.org/pdf/2512.24263)  

**Abstract**: When fine-tuning pre-trained Language Models (LMs) to exhibit desired behaviors, maintaining control over risk is critical for ensuring both safety and trustworthiness. Most existing safety alignment methods, such as Safe RLHF and SACPO, typically operate under a risk-neutral paradigm that is insufficient to address the risks arising from deviations from the reference policy and offers limited robustness against rare but potentially catastrophic harmful behaviors. To address this limitation, we propose Risk-aware Stepwise Alignment (RSA), a novel alignment method that explicitly incorporates risk awareness into the policy optimization process by leveraging a class of nested risk measures. Specifically, RSA formulates safety alignment as a token-level risk-aware constrained policy optimization problem and solves it through a stepwise alignment procedure that yields token-level policy updates derived from the nested risk measures. This design offers two key benefits: (1) it mitigates risks induced by excessive model shift away from a reference policy, and (2) it explicitly suppresses low-probability yet high-impact harmful behaviors. Moreover, we provide theoretical analysis on policy optimality under mild assumptions. Experimental results demonstrate that our method achieves high levels of helpfulness while ensuring strong safety and significantly suppresses tail risks, namely low-probability yet high-impact unsafe responses. 

---
# RSAgent: Learning to Reason and Act for Text-Guided Segmentation via Multi-Turn Tool Invocations 

**Authors**: Xingqi He, Yujie Zhang, Shuyong Gao, Wenjie Li, Lingyi Hong, Mingxi Chen, Kaixun Jiang, Jiyuan Fu, Wenqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.24023)  

**Abstract**: Text-guided object segmentation requires both cross-modal reasoning and pixel grounding abilities. Most recent methods treat text-guided segmentation as one-shot grounding, where the model predicts pixel prompts in a single forward pass to drive an external segmentor, which limits verification, refocusing and refinement when initial localization is wrong. To address this limitation, we propose RSAgent, an agentic Multimodal Large Language Model (MLLM) which interleaves reasoning and action for segmentation via multi-turn tool invocations. RSAgent queries a segmentation toolbox, observes visual feedback, and revises its spatial hypothesis using historical observations to re-localize targets and iteratively refine masks. We further build a data pipeline to synthesize multi-turn reasoning segmentation trajectories, and train RSAgent with a two-stage framework: cold-start supervised fine-tuning followed by agentic reinforcement learning with fine-grained, task-specific rewards. Extensive experiments show that RSAgent achieves a zero-shot performance of 66.5% gIoU on ReasonSeg test, improving over Seg-Zero-7B by 9%, and reaches 81.5% cIoU on RefCOCOg, demonstrating state-of-the-art performance on both in-domain and out-of-domain benchmarks. 

---
# Audited Skill-Graph Self-Improvement for Agentic LLMs via Verifiable Rewards, Experience Synthesis, and Continual Memory 

**Authors**: Ken Huang, Jerry Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.23760)  

**Abstract**: Reinforcement learning is increasingly used to transform large language models into agentic systems that act over long horizons, invoke tools, and manage memory under partial observability. While recent work has demonstrated performance gains through tool learning, verifiable rewards, and continual training, deployed self-improving agents raise unresolved security and governance challenges: optimization pressure can incentivize reward hacking, behavioral drift is difficult to audit or reproduce, and improvements are often entangled in opaque parameter updates rather than reusable, verifiable artifacts.
This paper proposes Audited Skill-Graph Self-Improvement (ASG-SI), a framework that treats self-improvement as iterative compilation of an agent into a growing, auditable skill graph. Each candidate improvement is extracted from successful trajectories, normalized into a skill with an explicit interface, and promoted only after passing verifier-backed replay and contract checks. Rewards are decomposed into reconstructible components derived from replayable evidence, enabling independent audit of promotion decisions and learning signals. ASG-SI further integrates experience synthesis for scalable stress testing and continual memory control to preserve long-horizon performance under bounded context.
We present a complete system architecture, threat model, and security analysis, and provide a fully runnable reference implementation that demonstrates verifier-backed reward construction, skill compilation, audit logging, and measurable improvement under continual task streams. ASG-SI reframes agentic self-improvement as accumulation of verifiable, reusable capabilities, offering a practical path toward reproducible evaluation and operational governance of self-improving AI agents. 

---
# HarmTransform: Transforming Explicit Harmful Queries into Stealthy via Multi-Agent Debate 

**Authors**: Shenzhe Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.23717)  

**Abstract**: Large language models (LLMs) are equipped with safety mechanisms to detect and block harmful queries, yet current alignment approaches primarily focus on overtly dangerous content and overlook more subtle threats. However, users can often disguise harmful intent through covert rephrasing that preserves malicious objectives while appearing benign, which creates a significant gap in existing safety training data. To address this limitation, we introduce HarmTransform, a multi-agent debate framework for systematically transforming harmful queries into stealthier forms while preserving their underlying harmful intent. Our framework leverages iterative critique and refinement among multiple agents to generate high-quality, covert harmful query transformations that can be used to improve future LLM safety alignment. Experiments demonstrate that HarmTransform significantly outperforms standard baselines in producing effective query transformations. At the same time, our analysis reveals that debate acts as a double-edged sword: while it can sharpen transformations and improve stealth, it may also introduce topic shifts and unnecessary complexity. These insights highlight both the promise and the limitations of multi-agent debate for generating comprehensive safety training data. 

---
