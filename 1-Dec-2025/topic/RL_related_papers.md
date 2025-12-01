# Towards Continuous Intelligence Growth: Self-Training, Continual Learning, and Dual-Scale Memory in SuperIntelliAgent 

**Authors**: Jianzhe Lin, Zeyu Pan, Yun Zhu, Ruiqi Song, Jining Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.23436)  

**Abstract**: We introduce SuperIntelliAgent, an agentic learning framework that couples a trainable small diffusion model (the learner) with a frozen large language model (the verifier) to enable continual intelligence growth through self-supervised interaction. Unlike conventional supervised fine-tuning, SuperIntelliAgent learns autonomously without annotation: the learner generates candidate outputs, the verifier evaluates them through step-by-step reasoning, and their interaction produces chosen/rejected pairs for Direct Preference Optimization (DPO). This converts each input into a pseudo-training signal for continual improvement. The framework integrates dual-scale memory: short-term in-context memory that preserves reasoning traces across refinement cycles, and long-term memory that consolidates acquired knowledge through lightweight on-the-fly fine-tuning. A replay buffer retains samples that show verifiable progress and replays them as auxiliary supervision, reinforcing recent learning while forming adaptive curricula. SuperIntelliAgent is infrastructure-agnostic and can be plugged into existing agentic frameworks while turning ordinary inference loops into a lifelong optimization process. We posit that pairing a trainable learner with a reasoning-capable verifier forms a minimal reliable unit of growing intelligence, as paired feedback and partial-history replay yield richer learning curricula and stronger preference alignment. With a small number of automatically generated DPO pairs, the learner improves across all benchmarks, indicating that this mechanism provides a promising direction for continual intelligence accumulation and real-world deployment. 

---
# ORION: Teaching Language Models to Reason Efficiently in the Language of Thought 

**Authors**: Kumar Tanmay, Kriti Aggarwal, Paul Pu Liang, Subhabrata Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2511.22891)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance in mathematics, code generation, and task planning, but their reliance on long chains of verbose "thinking" tokens leads to high latency, redundancy, and incoherent reasoning paths. Inspired by the Language of Thought Hypothesis, which posits that human reasoning operates over a symbolic, compositional mental language called Mentalese, we introduce a framework that trains models to reason in a similarly compact style. Mentalese encodes abstract reasoning as ultra-compressed, structured tokens, enabling models to solve complex problems with far fewer steps. To improve both efficiency and accuracy, we propose SHORTER LENGTH PREFERENCE OPTIMIZATION (SLPO), a reinforcement learning method that rewards concise solutions that stay correct, while still allowing longer reasoning when needed. Applied to Mentalese-aligned models, SLPO yields significantly higher compression rates by enabling concise reasoning that preserves the benefits of detailed thinking without the computational overhead. Across benchmarks including AIME 2024 and 2025, MinervaMath, OlympiadBench, Math500, and AMC, our ORION models produce reasoning traces with 4-16x fewer tokens, achieve up to 5x lower inference latency, and reduce training costs by 7-9x relative to the DeepSeek R1 Distilled model, while maintaining 90-98% of its accuracy. ORION also surpasses Claude and ChatGPT-4o by up to 5% in accuracy while maintaining 2x compression. These results show that Mentalese-style compressed reasoning offers a step toward human-like cognitive efficiency, enabling real-time, cost-effective reasoning without sacrificing accuracy. 

---
# Thinking by Doing: Building Efficient World Model Reasoning in LLMs via Multi-turn Interaction 

**Authors**: Bao Shu, Yan Cai, Jianjian Sun, Chunrui Han, En Yu, Liang Zhao, Jingcheng Hu, Yinmin Zhang, Haoran Lv, Yuang Peng, Zheng Ge, Xiangyu Zhang, Daxin Jiang, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2511.23476)  

**Abstract**: Developing robust world model reasoning is crucial for large language model (LLM) agents to plan and interact in complex environments. While multi-turn interaction offers a superior understanding of environmental dynamics via authentic feedback, current approaches often impose a rigid reasoning process, which constrains the model's active learning, ultimately hindering efficient world model reasoning. To address these issues, we explore world-model internalization through efficient interaction and active reasoning (WMAct), which liberates the model from structured reasoning, allowing the model to shape thinking directly through its doing, and achieves effective and efficient world model reasoning with two key mechanisms: (1) a reward rescaling mechanism adjusting outcome reward based on action efficacy to incentivize redundancy reduction and purposeful interaction; (2) an interaction frequency annealing strategy to progressively reduce the maximum allowed interaction turns, which compels the model to condense its learning and internalize environmental dynamics rather than over-relying on environmental cues. Our experiments on Sokoban, Maze, and Taxi show that WMAct yields effective world model reasoning capable of resolving tasks in a single turn that previously required multiple interactions and fosters strong transferability to complex environments, improving performance on a suite of reasoning benchmarks. 

---
# DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning 

**Authors**: Zhihong Shao, Yuxiang Luo, Chengda Lu, Z.Z. Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, Xiaokang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.22570)  

**Abstract**: Large language models have made significant progress in mathematical reasoning, which serves as an important testbed for AI and could impact scientific research if further advanced. By scaling reasoning with reinforcement learning that rewards correct final answers, LLMs have improved from poor performance to saturating quantitative reasoning competitions like AIME and HMMT in one year. However, this approach faces fundamental limitations. Pursuing higher final answer accuracy doesn't address a key issue: correct answers don't guarantee correct reasoning. Moreover, many mathematical tasks like theorem proving require rigorous step-by-step derivation rather than numerical answers, making final answer rewards inapplicable. To push the limits of deep reasoning, we believe it is necessary to verify the comprehensiveness and rigor of mathematical reasoning. Self-verification is particularly important for scaling test-time compute, especially for open problems without known solutions. Towards self-verifiable mathematical reasoning, we investigate how to train an accurate and faithful LLM-based verifier for theorem proving. We then train a proof generator using the verifier as the reward model, and incentivize the generator to identify and resolve as many issues as possible in their own proofs before finalizing them. To maintain the generation-verification gap as the generator becomes stronger, we propose to scale verification compute to automatically label new hard-to-verify proofs, creating training data to further improve the verifier. Our resulting model, DeepSeekMath-V2, demonstrates strong theorem-proving capabilities, achieving gold-level scores on IMO 2025 and CMO 2024 and a near-perfect 118/120 on Putnam 2024 with scaled test-time compute. 

---
# Co-Evolving Agents: Learning from Failures as Hard Negatives 

**Authors**: Yeonsung Jung, Trilok Padhi, Sina Shaham, Dipika Khullar, Joonhyun Jeong, Ninareh Mehrabi, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.22254)  

**Abstract**: The rapid progress of large foundation models has accelerated the development of task-specialized agents across diverse domains. However, the effectiveness of agents remains tightly coupled with the quality of training data, while curating task-specific datasets remains costly and often infeasible in real-world scenarios. Recent work has explored self-improving agents that autonomously generate, refine, and re-train on their own trajectories. A prominent line of approaches further leverages preference optimization by pairing predicted trajectories with scarce ground-truth trajectories, enabling agents to learn directly from their own failures. While these methods outperform supervised fine-tuning, their heavy reliance on predicted trajectories under limited ground-truth supervision leaves them prone to overfitting. To address this, we propose a co-evolving agents framework in which a target agent improves jointly with an auxiliary failure agent. The failure agent learns through preference optimization over failure trajectories from both the target and itself, thereby generating hard negatives that are close to success yet remain failures. Incorporating these informative hard negatives into the target agent's optimization sharpens decision boundaries and enhances generalization. Our comprehensive analysis and experiments across benchmark datasets show that our method not only shows improved performance but also demonstrates that failures, instead of being used as-is, can be systematically transformed into structured and valuable learning signals in self-improving agents. 

---
# REVEAL: Reasoning-enhanced Forensic Evidence Analysis for Explainable AI-generated Image Detection 

**Authors**: Huangsen Cao, Qin Mei, Zhiheng Li, Yuxi Li, Ying Zhang, Chen Li, Zhimeng Zhang, Xin Ding, Yongwei Wang, Jing Lyu, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.23158)  

**Abstract**: With the rapid advancement of generative models, visually realistic AI-generated images have become increasingly difficult to distinguish from authentic ones, posing severe threats to social trust and information integrity. Consequently, there is an urgent need for efficient and truly explainable image forensic methods. Recent detection paradigms have shifted towards explainable forensics. However, state-of-the-art approaches primarily rely on post-hoc rationalizations or visual discrimination, lacking a verifiable chain of evidence. This reliance on surface-level pattern matching limits the generation of causally grounded explanations and often results in poor generalization. To bridge this critical gap, we introduce \textbf{REVEAL-Bench}, the first reasoning-enhanced multimodal benchmark for AI-generated image detection that is explicitly structured around a chain-of-evidence derived from multiple lightweight expert models, then records step-by-step reasoning traces and evidential justifications. Building upon this dataset, we propose \textbf{REVEAL} (\underline{R}easoning-\underline{e}nhanced Forensic E\underline{v}id\underline{e}nce \underline{A}na\underline{l}ysis), an effective and explainable forensic framework that integrates detection with a novel expert-grounded reinforcement learning. Our reward mechanism is specially tailored to jointly optimize detection accuracy, explanation fidelity, and logical coherence grounded in explicit forensic evidence, enabling REVEAL to produce fine-grained, interpretable, and verifiable reasoning chains alongside its detection outcomes. Extensive experimental results demonstrate that REVEAL significantly enhances detection accuracy, explanation fidelity, and robust cross-model generalization, benchmarking a new state of the art for explainable image forensics. 

---
# Commanding Humanoid by Free-form Language: A Large Language Action Model with Unified Motion Vocabulary 

**Authors**: Zhirui Liu, Kaiyang Ji, Ke Yang, Jingyi Yu, Ye Shi, Jingya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.22963)  

**Abstract**: Enabling humanoid robots to follow free-form language commands is critical for seamless human-robot interaction, collaborative task execution, and general-purpose embodied intelligence. While recent advances have improved low-level humanoid locomotion and robot manipulation, language-conditioned whole-body control remains a significant challenge. Existing methods are often limited to simple instructions and sacrifice either motion diversity or physical plausibility. To address this, we introduce Humanoid-LLA, a Large Language Action Model that maps expressive language commands to physically executable whole-body actions for humanoid robots. Our approach integrates three core components: a unified motion vocabulary that aligns human and humanoid motion primitives into a shared discrete space; a vocabulary-directed controller distilled from a privileged policy to ensure physical feasibility; and a physics-informed fine-tuning stage using reinforcement learning with dynamics-aware rewards to enhance robustness and stability. Extensive evaluations in simulation and on a real-world Unitree G1 humanoid show that Humanoid-LLA delivers strong language generalization while maintaining high physical fidelity, outperforming existing language-conditioned controllers in motion naturalness, stability, and execution success rate. 

---
# ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering 

**Authors**: Alberto Compagnoni, Marco Morini, Sara Sarto, Federico Cocchi, Davide Caffagni, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2511.22715)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown impressive capabilities in jointly understanding text, images, and videos, often evaluated via Visual Question Answering (VQA). However, even state-of-the-art MLLMs struggle with domain-specific or knowledge-intensive queries, where relevant information is underrepresented in pre-training data. Knowledge-based VQA (KB-VQA) addresses this by retrieving external documents to condition answer generation, but current retrieval-augmented approaches suffer from low precision, noisy passages, and limited reasoning. To address this, we propose ReAG, a novel Reasoning-Augmented Multimodal RAG approach that combines coarse- and fine-grained retrieval with a critic model that filters irrelevant passages, ensuring high-quality additional context. The model follows a multi-stage training strategy leveraging reinforcement learning to enhance reasoning over retrieved content, while supervised fine-tuning serves only as a cold start. Extensive experiments on Encyclopedic-VQA and InfoSeek demonstrate that ReAG significantly outperforms prior methods, improving answer accuracy and providing interpretable reasoning grounded in retrieved evidence. Our source code is publicly available at: this https URL. 

---
# Adversarial Training for Process Reward Models 

**Authors**: Gurusha Juneja, Deepak Nathani, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.22888)  

**Abstract**: Process Reward Models (PRMs) enhance reasoning ability of LLMs by providing step-level supervision. However, their widespread adoption is limited due to expensive manual step-level annotation and poor generalization of static training data to novel errors. We introduce Adversarially Trained PRMs (\texttt{APRM}), where a Generator ($G$) learns to produce reasoning errors to deceive a PRM ($R$), while $R$ concurrently learns to detect them. This interaction yields progressively harder negatives for $R$, improving its robustness and generalization to novel errors without requiring manual step-level labels. Averaged across diverse mathematical reasoning benchmarks, \texttt{APRM} improves solver accuracy by $+3.4$ percentage points (pp) over the strongest PRM baseline. \texttt{APRM} achieves gains of $+5.3$ pp on out-of-distribution tasks. 

---
# Prompted Policy Search: Reinforcement Learning through Linguistic and Numerical Reasoning in LLMs 

**Authors**: Yifan Zhou, Sachin Grover, Mohamed El Mistiri, Kamalesh Kalirathnam, Pratyush Kerhalkar, Swaroop Mishra, Neelesh Kumar, Sanket Gaurav, Oya Aran, Heni Ben Amor  

**Link**: [PDF](https://arxiv.org/pdf/2511.21928)  

**Abstract**: Reinforcement Learning (RL) traditionally relies on scalar reward signals, limiting its ability to leverage the rich semantic knowledge often available in real-world tasks. In contrast, humans learn efficiently by combining numerical feedback with language, prior knowledge, and common sense. We introduce Prompted Policy Search (ProPS), a novel RL method that unifies numerical and linguistic reasoning within a single framework. Unlike prior work that augment existing RL components with language, ProPS places a large language model (LLM) at the center of the policy optimization loop-directly proposing policy updates based on both reward feedback and natural language input. We show that LLMs can perform numerical optimization in-context, and that incorporating semantic signals, such as goals, domain knowledge, and strategy hints can lead to more informed exploration and sample-efficient learning. ProPS is evaluated across fifteen Gymnasium tasks, spanning classic control, Atari games, and MuJoCo environments, and compared to seven widely-adopted RL algorithms (e.g., PPO, SAC, TRPO). It outperforms all baselines on eight out of fifteen tasks and demonstrates substantial gains when provided with domain knowledge. These results highlight the potential of unifying semantics and numerics for transparent, generalizable, and human-aligned RL. 

---
# Building Domain-Specific Small Language Models via Guided Data Generation 

**Authors**: Aman Kumar, Ekant Muljibhai Amin, Xian Yeow Lee, Lasitha Vidyaratne, Ahmed K. Farahat, Dipanjan D. Ghosh, Yuta Koreeda, Chetan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.21748)  

**Abstract**: Large Language Models (LLMs) have shown remarkable success in supporting a wide range of knowledge-intensive tasks. In specialized domains, there is growing interest in leveraging LLMs to assist subject matter experts with domain-specific challenges. However, deploying LLMs as SaaS solutions raises data privacy concerns, while many open-source models demand significant computational resources for effective domain adaptation and deployment. A promising alternative is to develop smaller, domain-specialized LLMs, though this approach is often constrained by the lack of high-quality domain-specific training data. In this work, we address these limitations by presenting a cost-efficient and scalable training pipeline that combines guided synthetic data generation from a small seed corpus with bottom-up domain data curation. Our pipeline integrates Domain-Adaptive Pretraining (DAPT), Domain-specific Supervised Fine-tuning (DSFT), and Direct Preference Optimization (DPO) to train effective small-scale models for specialized use cases. We demonstrate this approach through DiagnosticSLM, a 3B-parameter domain-specific model tailored for fault diagnosis, root cause analysis, and repair recommendation in industrial settings. To evaluate model performance, we introduce four domain-specific benchmarks: multiple-choice questions (DiagnosticMCQ), question answering (DiagnosticQA), sentence completion (DiagnosticComp), and summarization (DiagnosticSum). DiagnosticSLM achieves up to 25% accuracy improvement over open-source models of comparable or larger size (2B-9B) on the MCQ task, while also outperforming or matching them in other tasks, demonstrating effective domain-specific reasoning and generalization capabilities. 

---
# Affective Multimodal Agents with Proactive Knowledge Grounding for Emotionally Aligned Marketing Dialogue 

**Authors**: Lin Yu, Xiaofei Han, Yifei Kang, Chiung-Yi Tseng, Danyang Zhang, Ziqian Bi, Zhimo Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.21728)  

**Abstract**: Recent advances in large language models (LLMs) have enabled fluent dialogue systems, but most remain reactive and struggle in emotionally rich, goal-oriented settings such as marketing conversations. To address this limitation, we propose AffectMind, a multimodal affective dialogue agent that performs proactive reasoning and dynamic knowledge grounding to sustain emotionally aligned and persuasive interactions. AffectMind combines three components: a Proactive Knowledge Grounding Network (PKGN) that continuously updates factual and affective context from text, vision, and prosody; an Emotion--Intent Alignment Model (EIAM) that jointly models user emotion and purchase intent to adapt persuasion strategies; and a Reinforced Discourse Loop (RDL) that optimizes emotional coherence and engagement via reinforcement signals from user responses. Experiments on two newly curated marketing dialogue datasets, MM-ConvMarket and AffectPromo, show that AffectMind outperforms strong LLM-based baselines in emotional consistency (+26\%), persuasive success rate (+19\%), and long-term user engagement (+23\%), highlighting emotion-grounded proactivity as a key capability for commercial multimodal agents. 

---
# On the Role of Preference Variance in Preference Optimization 

**Authors**: Jiacheng Guo, Zihao Li, Jiahao Qiu, Yue Wu, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13022)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as an important approach for learning from human preferences in aligning large language models (LLMs). However, collecting human preference data is costly and inefficient, motivating methods to reduce the required annotations. In this work, we investigate the impact of \emph{preference variance} (PVar), which measures the variance in model preferences when comparing pairs of responses, on the effectiveness of DPO training. We provide a theoretical insight by establishing an upper bound on the DPO gradient norm for any given prompt, showing it is controlled by the PVar of that prompt. This implies that prompts with low PVar can only produce small gradient updates, making them less valuable for learning. We validate this finding by fine-tuning LLMs with preferences generated by a reward model, evaluating on two benchmarks (AlpacaEval 2.0 and Arena-Hard). Experimental results demonstrate that prompts with higher PVar outperform randomly selected prompts or those with lower PVar. We also show that our PVar-based selection method is robust, when using smaller reward models (1B, 3B) for selection. Notably, in a separate experiment using the original human annotations from the UltraFeedback dataset, we found that training on only the top 10\% of prompts with the highest PVar yields better evaluation performance than training on the full dataset, highlighting the importance of preference variance in identifying informative examples for efficient LLM alignment. 

---
# Ambiguity Awareness Optimization: Towards Semantic Disambiguation for Direct Preference Optimization 

**Authors**: Jian Li, Shenglin Yin, Yujia Zhang, Alan Zhao, Xi Chen, Xiaohui Zhou, Pengfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.23391)  

**Abstract**: Direct Preference Optimization (DPO) is a widely used reinforcement learning from human feedback (RLHF) method across various domains. Recent research has increasingly focused on the role of token importance in improving DPO effectiveness. It is observed that identical or semantically similar content (defined as ambiguous content) frequently appears within the preference pairs. We hypothesize that the presence of ambiguous content during DPO training may introduce ambiguity, thereby limiting further improvements in alignment. Through mathematical analysis and proof-of-concept experiments, we reveal that ambiguous content may potentially introduce ambiguities, thereby degrading performance. To address this issue, we introduce Ambiguity Awareness Optimization (AAO), a simple yet effective approach that automatically re-weights ambiguous content to reduce ambiguities by calculating semantic similarity from preference pairs. Through extensive experiments, we demonstrate that AAO consistently and significantly surpasses state-of-the-art approaches in performance, without markedly increasing response length, across multiple model scales and widely adopted benchmark datasets, including AlpacaEval 2, MT-Bench, and Arena-Hard. Specifically, AAO outperforms DPO by up to 8.9 points on AlpacaEval 2 and achieves an improvement of by up to 15.0 points on Arena-Hard. 

---
# ThetaEvolve: Test-time Learning on Open Problems 

**Authors**: Yiping Wang, Shao-Rong Su, Zhiyuan Zeng, Eva Xu, Liliang Ren, Xinyu Yang, Zeyi Huang, Xuehai He, Luyao Ma, Baolin Peng, Hao Cheng, Pengcheng He, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, Yelong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2511.23473)  

**Abstract**: Recent advances in large language models (LLMs) have enabled breakthroughs in mathematical discovery, exemplified by AlphaEvolve, a closed-source system that evolves programs to improve bounds on open problems. However, it relies on ensembles of frontier LLMs to achieve new bounds and is a pure inference system that models cannot internalize the evolving strategies. We introduce ThetaEvolve, an open-source framework that simplifies and extends AlphaEvolve to efficiently scale both in-context learning and Reinforcement Learning (RL) at test time, allowing models to continually learn from their experiences in improving open optimization problems. ThetaEvolve features a single LLM, a large program database for enhanced exploration, batch sampling for higher throughput, lazy penalties to discourage stagnant outputs, and optional reward shaping for stable training signals, etc. ThetaEvolve is the first evolving framework that enable a small open-source model, like DeepSeek-R1-0528-Qwen3-8B, to achieve new best-known bounds on open problems (circle packing and first auto-correlation inequality) mentioned in AlphaEvolve. Besides, across two models and four open tasks, we find that ThetaEvolve with RL at test-time consistently outperforms inference-only baselines, and the model indeed learns evolving capabilities, as the RL-trained checkpoints demonstrate faster progress and better final performance on both trained target task and other unseen tasks. We release our code publicly: this https URL 

---
