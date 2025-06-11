# Learning to Reason Across Parallel Samples for LLM Reasoning 

**Authors**: Jianing Qi, Xi Ye, Hao Tang, Zhigang Zhu, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.09014)  

**Abstract**: Scaling test-time compute brings substantial performance gains for large language models (LLMs). By sampling multiple answers and heuristically aggregate their answers (e.g., either through majority voting or using verifiers to rank the answers), one can achieve consistent performance gains in math domains. In this paper, we propose a new way to leverage such multiple sample set. We train a compact LLM, called Sample Set Aggregator (SSA), that takes a concatenated sequence of multiple samples and output the final answer, optimizing it for the answer accuracy with reinforcement learning. Experiments on multiple reasoning datasets show that SSA outperforms other test-time scaling methods such as reward model-based re-ranking. Our approach also shows a promising generalization ability, across sample set sizes, base model families and scales, and tasks. By separating LLMs to generate answers and LLMs to analyze and aggregate sampled answers, our approach can work with the outputs from premier black box models easily and efficiently. 

---
# Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning 

**Authors**: Haozhen Zhang, Tao Feng, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2506.09033)  

**Abstract**: The rapid emergence of diverse large language models (LLMs) has spurred the development of LLM routers that assign user queries to the most suitable model. However, existing LLM routers typically perform a single-round, one-to-one mapping (\textit{i.e.}, assigning each query to a single model in isolation), which limits their capability to tackle complex tasks that demand the complementary strengths of multiple LLMs. In this paper, we present \textbf{Router-R1}, a reinforcement learning (RL)-based framework that formulates multi-LLM routing and aggregation as a sequential decision process. Router-R1 instantiates the router itself as a capable LLM, leveraging its reasoning ability to interleave "think" actions (internal deliberation) with "route" actions (dynamic model invocation), and integrates each response into its evolving context. To guide learning, we employ a lightweight rule-based reward comprising format rewards, final outcome rewards, and a novel cost reward for performance and cost trade-off optimization, opening a pathway toward optimizing performance-cost tradeoffs via RL. Router-R1 also conditions only on simple model descriptors such as pricing, latency, and example performance, enabling strong generalization to unseen model selection. Experiments on seven general and multi-hop QA benchmarks show that Router-R1 outperforms over several strong baselines, achieving superior performance while maintaining robust generalization and cost this http URL is available at this https URL. 

---
# AdversariaL attacK sAfety aLIgnment(ALKALI): Safeguarding LLMs through GRACE: Geometric Representation-Aware Contrastive Enhancement- Introducing Adversarial Vulnerability Quality Index (AVQI) 

**Authors**: Danush Khanna, Krishna Kumar, Basab Ghosh, Vinija Jain, Vasu Sharma, Aman Chadha, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2506.08885)  

**Abstract**: Adversarial threats against LLMs are escalating faster than current defenses can adapt. We expose a critical geometric blind spot in alignment: adversarial prompts exploit latent camouflage, embedding perilously close to the safe representation manifold while encoding unsafe intent thereby evading surface level defenses like Direct Preference Optimization (DPO), which remain blind to the latent geometry. We introduce ALKALI, the first rigorously curated adversarial benchmark and the most comprehensive to date spanning 9,000 prompts across three macro categories, six subtypes, and fifteen attack families. Evaluation of 21 leading LLMs reveals alarmingly high Attack Success Rates (ASRs) across both open and closed source models, exposing an underlying vulnerability we term latent camouflage, a structural blind spot where adversarial completions mimic the latent geometry of safe ones. To mitigate this vulnerability, we introduce GRACE - Geometric Representation Aware Contrastive Enhancement, an alignment framework coupling preference learning with latent space regularization. GRACE enforces two constraints: latent separation between safe and adversarial completions, and adversarial cohesion among unsafe and jailbreak behaviors. These operate over layerwise pooled embeddings guided by a learned attention profile, reshaping internal geometry without modifying the base model, and achieve up to 39% ASR reduction. Moreover, we introduce AVQI, a geometry aware metric that quantifies latent alignment failure via cluster separation and compactness. AVQI reveals when unsafe completions mimic the geometry of safe ones, offering a principled lens into how models internally encode safety. We make the code publicly available at this https URL. 

---
# ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Large Language Model Preference Optimization 

**Authors**: Hee Suk Yoon, Eunseop Yoon, Mark A. Hasegawa-Johnson, Sungwoong Kim, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2506.08712)  

**Abstract**: We introduce ConfPO, a method for preference learning in Large Language Models (LLMs) that identifies and optimizes preference-critical tokens based solely on the training policy's confidence, without requiring any auxiliary models or compute. Unlike prior Direct Alignment Algorithms (DAAs) such as Direct Preference Optimization (DPO), which uniformly adjust all token probabilities regardless of their relevance to preference, ConfPO focuses optimization on the most impactful tokens. This targeted approach improves alignment quality while mitigating overoptimization (i.e., reward hacking) by using the KL divergence budget more efficiently. In contrast to recent token-level methods that rely on credit-assignment models or AI annotators, raising concerns about scalability and reliability, ConfPO is simple, lightweight, and model-free. Experimental results on challenging alignment benchmarks, including AlpacaEval 2 and Arena-Hard, demonstrate that ConfPO consistently outperforms uniform DAAs across various LLMs, delivering better alignment with zero additional computational overhead. 

---
# MEMETRON: Metaheuristic Mechanisms for Test-time Response Optimization of Large Language Models 

**Authors**: Son The Nguyen, Theja Tulabandhula  

**Link**: [PDF](https://arxiv.org/pdf/2506.08643)  

**Abstract**: Large language models (LLMs) are increasingly used for both open-ended and structured tasks, yet their inference-time behavior is still largely dictated by heuristic decoding strategies such as greedy search, sampling, or reranking. These methods provide limited control and do not explicitly optimize for task-specific objectives. We introduce MEMETRON, a task-agnostic framework that formulates LLM decoding as a discrete black-box optimization problem. MEMETRON leverages hybrid metaheuristic algorithms, GENETRON and ANNETRON, to search the response space, guided by reward models and contextual operations performed by the LLM itself. This approach enables efficient discovery of high-reward responses without requiring model retraining or gradient access. The framework is modular and generalizes across diverse tasks, requiring only a reward function and lightweight prompt templates. We evaluate our framework on the critical human preference alignment task and demonstrate that it significantly outperforms standard decoding and reranking methods, highlighting its potential to improve alignment without model retraining. 

---
# RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling 

**Authors**: Yang Liu, Jiaqi Li, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.08672)  

**Abstract**: Rule-based reasoning has been acknowledged as one of the fundamental problems in reasoning, while deviations in rule formats, types, and complexity in real-world applications pose severe challenges. Recent studies have shown that large reasoning models (LRMs) have remarkable reasoning capabilities, and their performance is substantially enhanced by reinforcement learning (RL). However, it remains an open question whether small reasoning models (SRMs) can learn rule-based reasoning effectively with robust generalization across diverse tasks and domains. To address this, we introduce Reinforced Rule-based Reasoning, a.k.a. RuleReasoner, a simple yet effective method to conduct rule-based reasoning via a wide collection of curated tasks and a novel domain-aware dynamic sampling approach. Specifically, RuleReasoner resamples each training batch by updating the sampling weights of different domains based on historical rewards. This facilitates domain augmentation and flexible online learning schedules for RL, obviating the need for pre-hoc human-engineered mix-training recipes used in existing methods. Empirical evaluations on in-distribution (ID) and out-of-distribution (OOD) benchmarks reveal that RuleReasoner outperforms frontier LRMs by a significant margin ($\Delta$4.1% average points on eight ID tasks and $\Delta$10.4% average points on three OOD tasks over OpenAI-o1). Notably, our approach also exhibits higher computational efficiency compared to prior dynamic sampling methods for RL. 

---
# Compound AI Systems Optimization: A Survey of Methods, Challenges, and Future Directions 

**Authors**: Yu-Ang Lee, Guan-Ting Yi, Mei-Yi Liu, Jui-Chao Lu, Guan-Bo Yang, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.08234)  

**Abstract**: Recent advancements in large language models (LLMs) and AI systems have led to a paradigm shift in the design and optimization of complex AI workflows. By integrating multiple components, compound AI systems have become increasingly adept at performing sophisticated tasks. However, as these systems grow in complexity, new challenges arise in optimizing not only individual components but also their interactions. While traditional optimization methods such as supervised fine-tuning (SFT) and reinforcement learning (RL) remain foundational, the rise of natural language feedback introduces promising new approaches, especially for optimizing non-differentiable systems. This paper provides a systematic review of recent progress in optimizing compound AI systems, encompassing both numerical and language-based techniques. We formalize the notion of compound AI system optimization, classify existing methods along several key dimensions, and highlight open research challenges and future directions in this rapidly evolving field. A list of surveyed papers is publicly available at this https URL. 

---
# Step-Audio-AQAA: a Fully End-to-End Expressive Large Audio Language Model 

**Authors**: Ailin Huang, Bingxin Li, Bruce Wang, Boyong Wu, Chao Yan, Chengli Feng, Heng Wang, Hongyu Zhou, Hongyuan Wang, Jingbei Li, Jianjian Sun, Joanna Wang, Mingrui Chen, Peng Liu, Ruihang Miao, Shilei Jiang, Tian Fei, Wang You, Xi Chen, Xuerui Yang, Yechang Huang, Yuxiang Zhang, Zheng Ge, Zheng Gong, Zhewei Huang, Zixin Zhang, Bin Wang, Bo Li, Buyun Ma, Changxin Miao, Changyi Wan, Chen Xu, Dapeng Shi, Dingyuan Hu, Enle Liu, Guanzhe Huang, Gulin Yan, Hanpeng Hu, Haonan Jia, Jiahao Gong, Jiaoren Wu, Jie Wu, Jie Yang, Junzhe Lin, Kaixiang Li, Lei Xia, Longlong Gu, Ming Li, Nie Hao, Ranchen Ming, Shaoliang Pang, Siqi Liu, Song Yuan, Tiancheng Cao, Wen Li, Wenqing He, Xu Zhao, Xuelin Zhang, Yanbo Yu, Yinmin Zhong, Yu Zhou, Yuanwei Liang, Yuanwei Lu, Yuxiang Yang, Zidong Yang, Zili Zhang, Binxing Jiao, Heung-Yeung Shum, Jiansheng Chen, Jing Li, Xiangyu Zhang, Xinhao Zhang, Yibo Zhu, Daxin Jiang, Shuchang Zhou, Chen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08967)  

**Abstract**: Large Audio-Language Models (LALMs) have significantly advanced intelligent human-computer interaction, yet their reliance on text-based outputs limits their ability to generate natural speech responses directly, hindering seamless audio interactions. To address this, we introduce Step-Audio-AQAA, a fully end-to-end LALM designed for Audio Query-Audio Answer (AQAA) tasks. The model integrates a dual-codebook audio tokenizer for linguistic and semantic feature extraction, a 130-billion-parameter backbone LLM and a neural vocoder for high-fidelity speech synthesis. Our post-training approach employs interleaved token-output of text and audio to enhance semantic coherence and combines Direct Preference Optimization (DPO) with model merge to improve performance. Evaluations on the StepEval-Audio-360 benchmark demonstrate that Step-Audio-AQAA excels especially in speech control, outperforming the state-of-art LALMs in key areas. This work contributes a promising solution for end-to-end LALMs and highlights the critical role of token-based vocoder in enhancing overall performance for AQAA tasks. 

---
# QA-LIGN: Aligning LLMs through Constitutionally Decomposed QA 

**Authors**: Jacob Dineen, Aswin RRV, Qin Liu, Zhikun Xu, Xiao Ye, Ming Shen, Zhaonan Li, Shijie Lu, Chitta Baral, Muhao Chen, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.08123)  

**Abstract**: Alignment of large language models with explicit principles (such as helpfulness, honesty, and harmlessness) is crucial for ensuring safe and reliable AI systems. However, standard reward-based alignment methods typically collapse diverse feedback into a single scalar reward, entangling multiple objectives into one opaque training signal, which hinders interpretability. In this work, we introduce QA-LIGN, an automatic symbolic reward decomposition approach that preserves the structure of each constitutional principle within the reward mechanism. Instead of training a black-box reward model that outputs a monolithic score, QA-LIGN formulates principle-specific evaluation questions and derives separate reward components for each principle, making it a drop-in reward model replacement. Experiments aligning an uncensored large language model with a set of constitutional principles demonstrate that QA-LIGN offers greater transparency and adaptability in the alignment process. At the same time, our approach achieves performance on par with or better than a DPO baseline. Overall, these results represent a step toward more interpretable and controllable alignment of language models, achieved without sacrificing end-task performance. 

---
# Consistent Paths Lead to Truth: Self-Rewarding Reinforcement Learning for LLM Reasoning 

**Authors**: Kongcheng Zhang, Qi Yao, Shunyu Liu, Yingjie Wang, Baisheng Lai, Jieping Ye, Mingli Song, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.08745)  

**Abstract**: Recent advances of Reinforcement Learning (RL) have highlighted its potential in complex reasoning tasks, yet effective training often relies on external supervision, which limits the broader applicability. In this work, we propose a novel self-rewarding reinforcement learning framework to enhance Large Language Model (LLM) reasoning by leveraging the consistency of intermediate reasoning states across different reasoning trajectories. Our key insight is that correct responses often exhibit consistent trajectory patterns in terms of model likelihood: their intermediate reasoning states tend to converge toward their own final answers (high consistency) with minimal deviation toward other candidates (low volatility). Inspired by this observation, we introduce CoVo, an intrinsic reward mechanism that integrates Consistency and Volatility via a robust vector-space aggregation strategy, complemented by a curiosity bonus to promote diverse exploration. CoVo enables LLMs to perform RL in a self-rewarding manner, offering a scalable pathway for learning to reason without external supervision. Extensive experiments on diverse reasoning benchmarks show that CoVo achieves performance comparable to or even surpassing supervised RL. Our code is available at this https URL. 

---
# Reinforce LLM Reasoning through Multi-Agent Reflection 

**Authors**: Yurun Yuan, Tengyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.08379)  

**Abstract**: Leveraging more test-time computation has proven to be an effective way to boost the reasoning capabilities of large language models (LLMs). Among various methods, the verify-and-improve paradigm stands out for enabling dynamic solution exploration and feedback incorporation. However, existing approaches often suffer from restricted feedback spaces and lack of coordinated training of different parties, leading to suboptimal performance. To address this, we model this multi-turn refinement process as a Markov Decision Process and introduce DPSDP (Direct Policy Search by Dynamic Programming), a reinforcement learning algorithm that trains an actor-critic LLM system to iteratively refine answers via direct preference learning on self-generated data. Theoretically, DPSDP can match the performance of any policy within the training distribution. Empirically, we instantiate DPSDP with various base models and show improvements on both in- and out-of-distribution benchmarks. For example, on benchmark MATH 500, majority voting over five refinement steps increases first-turn accuracy from 58.2% to 63.2% with Ministral-based models. An ablation study further confirms the benefits of multi-agent collaboration and out-of-distribution generalization. 

---
# Reinforcement Learning from Human Feedback with High-Confidence Safety Constraints 

**Authors**: Yaswanth Chittepu, Blossom Metevier, Will Schwarzer, Austin Hoag, Scott Niekum, Philip S. Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2506.08266)  

**Abstract**: Existing approaches to language model alignment often treat safety as a tradeoff against helpfulness, which can lead to unacceptable responses in sensitive domains. To ensure reliable performance in such settings, we propose High-Confidence Safe Reinforcement Learning from Human Feedback (HC-RLHF), a method that provides high-confidence safety guarantees while maximizing helpfulness. Similar to previous methods, HC-RLHF explicitly decouples human preferences into helpfulness and harmlessness (safety), which are learned by training a reward model and a cost model, respectively. It then employs a two-step process to find safe solutions. In the first step, it optimizes the reward function under an intentionally pessimistic version of the cost constraint. In the second step, the trained model undergoes a safety test to verify whether its performance stays within an upper-confidence bound of the actual cost constraint. We provide a theoretical analysis of HC-RLHF, including proof that it will not return an unsafe solution with a probability greater than a user-specified threshold. For our empirical analysis, we apply HC-RLHF to align three different language models (Qwen2-1.5B, Qwen2.5-3B, and LLaMa3.2-3B) with human preferences. Our results demonstrate that HC-RLHF produces safe models with high probability and can improve harmlessness and helpfulness compared to previous methods. 

---
# Reinforcement Learning Teachers of Test Time Scaling 

**Authors**: Edoardo Cetin, Tianyu Zhao, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08388)  

**Abstract**: Training reasoning language models (LMs) with reinforcement learning (RL) for one-hot correctness inherently relies on the LM being able to explore and solve its task with some chance at initialization. Furthermore, a key use case of reasoning LMs is to act as teachers for distilling new students and cold-starting future RL iterations rather than being deployed themselves. From these considerations, we introduce a new framework that avoids RL's exploration challenge by training a new class of Reinforcement-Learned Teachers (RLTs) focused on yielding the most effective downstream distillation. RLTs are prompted with both the question and solution to each problem, and tasked to simply "connect-the-dots" with detailed explanations tailored for their students. We train RLTs with dense rewards obtained by feeding each explanation to the student and testing its understanding of the problem's solution. In practice, the raw outputs of a 7B RLT provide higher final performance on competition and graduate-level tasks than existing distillation and cold-starting pipelines that collect and postprocess the reasoning traces of orders of magnitude larger LMs. Furthermore, RLTs maintain their effectiveness when training larger students and when applied zero-shot to out-of-distribution tasks, unlocking new levels of efficiency and re-usability for the RL reasoning framework. 

---
# From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium 

**Authors**: Xie Yi, Zhanke Zhou, Chentao Cao, Qiyu Niu, Tongliang Liu, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.08292)  

**Abstract**: Multi-agent frameworks can substantially boost the reasoning power of large language models (LLMs), but they typically incur heavy computational costs and lack convergence guarantees. To overcome these challenges, we recast multi-LLM coordination as an incomplete-information game and seek a Bayesian Nash equilibrium (BNE), in which each agent optimally responds to its probabilistic beliefs about the strategies of others. We introduce Efficient Coordination via Nash Equilibrium (ECON), a hierarchical reinforcement-learning paradigm that marries distributed reasoning with centralized final output. Under ECON, each LLM independently selects responses that maximize its expected reward, conditioned on its beliefs about co-agents, without requiring costly inter-agent exchanges. We mathematically prove that ECON attains a markedly tighter regret bound than non-equilibrium multi-agent schemes. Empirically, ECON outperforms existing multi-LLM approaches by 11.2% on average across six benchmarks spanning complex reasoning and planning tasks. Further experiments demonstrate ECON's ability to flexibly incorporate additional models, confirming its scalability and paving the way toward larger, more powerful multi-LLM ensembles. The code is publicly available at: this https URL. 

---
# Bingo: Boosting Efficient Reasoning of LLMs via Dynamic and Significance-based Reinforcement Learning 

**Authors**: Hanbing Liu, Lang Cao, Yuanyi Ren, Mengyu Zhou, Haoyu Dong, Xiaojun Ma, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08125)  

**Abstract**: Large language models have demonstrated impressive reasoning capabilities, yet they often suffer from inefficiencies due to unnecessarily verbose or redundant outputs. While many works have explored reinforcement learning (RL) to enhance reasoning abilities, most primarily focus on improving accuracy, with limited attention to reasoning efficiency. Some existing approaches introduce direct length-based rewards to encourage brevity, but this often leads to noticeable drops in accuracy. In this paper, we propose Bingo, an RL framework that advances length-based reward design to boost efficient reasoning. Bingo incorporates two key mechanisms: a significance-aware length reward, which gradually guides the model to reduce only insignificant tokens, and a dynamic length reward, which initially encourages elaborate reasoning for hard questions but decays over time to improve overall efficiency. Experiments across multiple reasoning benchmarks show that Bingo improves both accuracy and efficiency. It outperforms the vanilla reward and several other length-based reward baselines in RL, achieving a favorable trade-off between accuracy and efficiency. These results underscore the potential of training LLMs explicitly for efficient reasoning. 

---
# Reinforcement Fine-Tuning for Reasoning towards Multi-Step Multi-Source Search in Large Language Models 

**Authors**: Wentao Shi, Yiqing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.08352)  

**Abstract**: Large language models (LLMs) can face factual limitations when responding to time-sensitive queries about recent events that arise after their knowledge thresholds in the training corpus. Existing search-augmented approaches fall into two categories, each with distinct limitations: multi-agent search frameworks incur substantial computational overhead by separating search planning and response synthesis across multiple LLMs, while single-LLM tool-calling methods restrict themselves to sequential planned, single-query searches from sole search sources. We present Reasoning-Search (R-Search), a single-LLM search framework that unifies multi-step planning, multi-source search execution, and answer synthesis within one coherent inference process. Innovatively, it structure the output into four explicitly defined components, including reasoning steps that guide the search process (<think>), a natural-language directed acyclic graph that represents the search plans with respect to diverse sources (<search>), retrieved results from executing the search plans (<result>), and synthesized final answers (<answer>). To enable effective generation of these structured outputs, we propose a specialized Reinforcement Fine-Tuning (ReFT) method based on GRPO, together with a multi-component reward function that optimizes LLM's answer correctness, structural validity of the generated DAG, and adherence to the defined output format. Experimental evaluation on FinSearchBench-24, SearchExpertBench-25, and seven Q and A benchmarks demonstrates that R-Search outperforms state-of-the-art methods, while achieving substantial efficiency gains through 70% reduction in context token usage and approximately 50% decrease in execution latency. Code is available at this https URL. 

---
# Can A Gamer Train A Mathematical Reasoning Model? 

**Authors**: Andrew Shin  

**Link**: [PDF](https://arxiv.org/pdf/2506.08935)  

**Abstract**: While large language models (LLMs) have achieved remarkable performance in various tasks including mathematical reasoning, their development typically demands prohibitive computational resources. Recent advancements have reduced costs for training capable models, yet even these approaches rely on high-end hardware clusters. In this paper, we demonstrate that a single average gaming GPU can train a solid mathematical reasoning model, by integrating reinforcement learning and memory optimization techniques. Specifically, we train a 1.5B parameter mathematical reasoning model on RTX 3080 Ti of 16GB memory that achieves comparable or better performance on mathematical reasoning benchmarks than models several times larger, in resource-constrained environments. Our results challenge the paradigm that state-of-the-art mathematical reasoning necessitates massive infrastructure, democratizing access to high-performance AI research. this https URL. 

---
# GFRIEND: Generative Few-shot Reward Inference through EfficieNt DPO 

**Authors**: Yiyang Zhao, Huiyu Bai, Xuejiao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.08965)  

**Abstract**: The ability to train high-performing reward models with few-shot data is critical for enhancing the efficiency and scalability of Reinforcement Learning from Human Feedback (RLHF). We propose a data augmentation and expansion framework that enables generative reward models trained on small datasets to achieve comparable performance to those trained on large-scale datasets. Traditional methods to train a generative reward model, such as Direct Preference Optimization (DPO), are constrained by inefficiencies in sample pairing and limited data diversity. This work introduces preference refinement, which employs Chain-of-Thought (CoT) sampling to uncover diverse and high-quality preference relationships. It also incorporates a perplexity-based scoring mechanism to assign nuanced preference levels and utilizes Multi-level Direct Preference Optimization (M-DPO) to enable the model to capture finer-grained preference differences between samples. Experimental results demonstrate that the proposed method significantly enhances data efficiency and model performance, enabling reward models trained in a few-shot setting to achieve results on par with those trained on large-scale datasets. This study underscores the potential of data-efficient strategies in advancing reward model optimization, offering a robust solution for low-resource RLHF applications. 

---
# Worst-Case Symbolic Constraints Analysis and Generalisation with Large Language Models 

**Authors**: Daniel Koh, Yannic Noller, Corina S. Pasareanu, Adrians Skapars, Youcheng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.08171)  

**Abstract**: Large language models (LLMs) have been successfully applied to a variety of coding tasks, including code generation, completion, and repair. However, more complex symbolic reasoning tasks remain largely unexplored by LLMs. This paper investigates the capacity of LLMs to reason about worst-case executions in programs through symbolic constraints analysis, aiming to connect LLMs and symbolic reasoning approaches. Specifically, we define and address the problem of worst-case symbolic constraints analysis as a measure to assess the comprehension of LLMs. We evaluate the performance of existing LLMs on this novel task and further improve their capabilities through symbolic reasoning-guided fine-tuning, grounded in SMT (Satisfiability Modulo Theories) constraint solving and supported by a specially designed dataset of symbolic constraints. Experimental results show that our solver-aligned model, WARP-1.0-3B, consistently surpasses size-matched and even much larger baselines, demonstrating that a 3B LLM can recover the very constraints that pin down an algorithm's worst-case behaviour through reinforcement learning methods. These findings suggest that LLMs are capable of engaging in deeper symbolic reasoning, supporting a closer integration between neural network-based learning and formal methods for rigorous program analysis. 

---
