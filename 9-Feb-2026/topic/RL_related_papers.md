# R-Align: Enhancing Generative Reward Models through Rationale-Centric Meta-Judging 

**Authors**: Yanlin Lai, Mitt Huang, Hangyu Guo, Xiangfeng Wang, Haodong Li, Shaoxiong Zhan, Liang Zhao, Chengyuan Yao, Yinmin Zhang, Qi Han, Chun Yuan, Zheng Ge, Xiangyu Zhang, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06763)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) remains indispensable for aligning large language models (LLMs) in subjective domains. To enhance robustness, recent work shifts toward Generative Reward Models (GenRMs) that generate rationales before predicting preferences. Yet in GenRM training and evaluation, practice remains outcome-label-only, leaving reasoning quality unchecked. We show that reasoning fidelity-the consistency between a GenRM's preference decision and reference decision rationales-is highly predictive of downstream RLHF outcomes, beyond standard label accuracy. Specifically, we repurpose existing reward-model benchmarks to compute Spurious Correctness (S-Corr)-the fraction of label-correct decisions with rationales misaligned with golden judgments. Our empirical evaluation reveals substantial S-Corr even for competitive GenRMs, and higher S-Corr is associated with policy degeneration under optimization. To improve fidelity, we propose Rationale-Centric Alignment, R-Align, which augments training with gold judgments and explicitly supervises rationale alignment. R-Align reduces S-Corr on RM benchmarks and yields consistent gains in actor performance across STEM, coding, instruction following, and general tasks. 

---
# Generating Data-Driven Reasoning Rubrics for Domain-Adaptive Reward Modeling 

**Authors**: Kate Sanders, Nathaniel Weir, Sapana Chaudhary, Kaj Bostrom, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2602.06795)  

**Abstract**: An impediment to using Large Language Models (LLMs) for reasoning output verification is that LLMs struggle to reliably identify errors in thinking traces, particularly in long outputs, domains requiring expert knowledge, and problems without verifiable rewards. We propose a data-driven approach to automatically construct highly granular reasoning error taxonomies to enhance LLM-driven error detection on unseen reasoning traces. Our findings indicate that classification approaches that leverage these error taxonomies, or "rubrics", demonstrate strong error identification compared to baseline methods in technical domains like coding, math, and chemical engineering. These rubrics can be used to build stronger LLM-as-judge reward functions for reasoning model training via reinforcement learning. Experimental results show that these rewards have the potential to improve models' task accuracy on difficult domains over models trained by general LLMs-as-judges by +45%, and approach performance of models trained by verifiable rewards while using as little as 20% as many gold labels. Through our approach, we extend the usage of reward rubrics from assessing qualitative model behavior to assessing quantitative model correctness on tasks typically learned via RLVR rewards. This extension opens the door for teaching models to solve complex technical problems without a full dataset of gold labels, which are often highly costly to procure. 

---
# FairJudge: An Adaptive, Debiased, and Consistent LLM-as-a-Judge 

**Authors**: Bo Yang, Lanfei Feng, Yunkui Chen, Yu Zhang, Xiao Xu, Shijian Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.06625)  

**Abstract**: Existing LLM-as-a-Judge systems suffer from three fundamental limitations: limited adaptivity to task- and domain-specific evaluation criteria, systematic biases driven by non-semantic cues such as position, length, format, and model provenance, and evaluation inconsistency that leads to contradictory judgments across different evaluation modes (e.g., pointwise versus pairwise). To address these issues, we propose FairJudge, an adaptive, debiased, and consistent LLM-as-a-Judge. Unlike prior approaches that treat the judge as a static evaluator, FairJudge models judging behavior itself as a learnable and regularized policy. From a data-centric perspective, we construct a high-information-density judging dataset that explicitly injects supervision signals aligned with evaluation behavior. Building on this dataset, we adopt a curriculum-style SFT-DPO-GRPO training paradigm that progressively aligns rubric adherence, bias mitigation, and cross-mode consistency, while avoiding catastrophic forgetting. Experimental results on multiple internal and public benchmarks show that FairJudge consistently improves agreement and F1, reduces non-semantic biases, and outperforms substantially larger instruction-tuned LLMs. All resources will be publicly released after acceptance to facilitate future research. 

---
# InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning 

**Authors**: Yuchen Yan, Liang Jiang, Jin Jiang, Shuaicheng Li, Zujie Wen, Zhiqiang Zhang, Jun Zhou, Jian Shao, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.06960)  

**Abstract**: Large reasoning models achieve strong performance by scaling inference-time chain-of-thought, but this paradigm suffers from quadratic cost, context length limits, and degraded reasoning due to lost-in-the-middle effects. Iterative reasoning mitigates these issues by periodically summarizing intermediate thoughts, yet existing methods rely on supervised learning or fixed heuristics and fail to optimize when to summarize, what to preserve, and how to resume reasoning. We propose InftyThink+, an end-to-end reinforcement learning framework that optimizes the entire iterative reasoning trajectory, building on model-controlled iteration boundaries and explicit summarization. InftyThink+ adopts a two-stage training scheme with supervised cold-start followed by trajectory-level reinforcement learning, enabling the model to learn strategic summarization and continuation decisions. Experiments on DeepSeek-R1-Distill-Qwen-1.5B show that InftyThink+ improves accuracy by 21% on AIME24 and outperforms conventional long chain-of-thought reinforcement learning by a clear margin, while also generalizing better to out-of-distribution benchmarks. Moreover, InftyThink+ significantly reduces inference latency and accelerates reinforcement learning training, demonstrating improved reasoning efficiency alongside stronger performance. 

---
# Improve Large Language Model Systems with User Logs 

**Authors**: Changyue Wang, Weihang Su, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06470)  

**Abstract**: Scaling training data and model parameters has long driven progress in large language models (LLMs), but this paradigm is increasingly constrained by the scarcity of high-quality data and diminishing returns from rising computational costs. As a result, recent work is increasing the focus on continual learning from real-world deployment, where user interaction logs provide a rich source of authentic human feedback and procedural knowledge. However, learning from user logs is challenging due to their unstructured and noisy nature. Vanilla LLM systems often struggle to distinguish useful feedback signals from noisy user behavior, and the disparity between user log collection and model optimization (e.g., the off-policy optimization problem) further strengthens the problem. To this end, we propose UNO (User log-driveN Optimization), a unified framework for improving LLM systems (LLMsys) with user logs. UNO first distills logs into semi-structured rules and preference pairs, then employs query-and-feedback-driven clustering to manage data heterogeneity, and finally quantifies the cognitive gap between the model's prior knowledge and the log data. This assessment guides the LLMsys to adaptively filter out noisy feedback and construct different modules for primary and reflective experiences extracted from user logs, thereby improving future responses. Extensive experiments show that UNO achieves state-of-the-art effectiveness and efficiency, significantly outperforming Retrieval Augmented Generation (RAG) and memory-based baselines. We have open-sourced our code at this https URL . 

---
# Evaluating an evidence-guided reinforcement learning framework in aligning light-parameter large language models with decision-making cognition in psychiatric clinical reasoning 

**Authors**: Xinxin Lin, Guangxin Dai, Yi Zhong, Xiang Li, Xue Xiao, Yixin Zhang, Zhengdong Wu, Yongbo Zheng, Runchuan Zhu, Ming Zhao, Huizi Yu, Shuo Wu, Jun Zhao, Lingming Hu, Yumei Wang, Ping Yin, Joey W.Y. Chan, Ngan Yin Chan, Sijing Chen, Yun Kwok Wing, Lin Lu, Xin Ma, Lizhou Fan  

**Link**: [PDF](https://arxiv.org/pdf/2602.06449)  

**Abstract**: Large language models (LLMs) hold transformative potential for medical decision support yet their application in psychiatry remains constrained by hallucinations and superficial reasoning. This limitation is particularly acute in light-parameter LLMs which are essential for privacy-preserving and efficient clinical deployment. Existing training paradigms prioritize linguistic fluency over structured clinical logic and result in a fundamental misalignment with professional diagnostic cognition. Here we introduce ClinMPO, a reinforcement learning framework designed to align the internal reasoning of LLMs with professional psychiatric practice. The framework employs a specialized reward model trained independently on a dataset derived from 4,474 psychiatry journal articles and structured according to evidence-based medicine principles. We evaluated ClinMPO on a unseen subset of the benchmark designed to isolate reasoning capabilities from rote memorization. This test set comprises items where leading large-parameter LLMs consistently fail. We compared the ClinMPO-aligned light LLM performance against a cohort of 300 medical students. The ClinMPO-tuned Qwen3-8B model achieved a diagnostic accuracy of 31.4% and surpassed the human benchmark of 30.8% on these complex cases. These results demonstrate that medical evidence-guided optimization enables light-parameter LLMs to master complex reasoning tasks. Our findings suggest that explicit cognitive alignment offers a scalable pathway to reliable and safe psychiatric decision support. 

---
# FMBench: Adaptive Large Language Model Output Formatting 

**Authors**: Yaoting Wang, Yun Zhou, Henghui Ding  

**Link**: [PDF](https://arxiv.org/pdf/2602.06384)  

**Abstract**: Producing outputs that satisfy both semantic intent and format constraints is essential for deploying large language models in user-facing and system-integrated workflows. In this work, we focus on Markdown formatting, which is ubiquitous in assistants, documentation, and tool-augmented pipelines but still prone to subtle, hard-to-detect errors (e.g., broken lists, malformed tables, inconsistent headings, and invalid code blocks) that can significantly degrade downstream usability. We present FMBench, a benchmark for adaptive Markdown output formatting that evaluates models under a wide range of instruction-following scenarios with diverse structural requirements. FMBench emphasizes real-world formatting behaviors such as multi-level organization, mixed content (natural language interleaved with lists/tables/code), and strict adherence to user-specified layout constraints. To improve Markdown compliance without relying on hard decoding constraints, we propose a lightweight alignment pipeline that combines supervised fine-tuning (SFT) with reinforcement learning fine-tuning. Starting from a base model, we first perform SFT on instruction-response pairs, and then optimize a composite objective that balances semantic fidelity with structural correctness. Experiments on two model families (OpenPangu and Qwen) show that SFT consistently improves semantic alignment, while reinforcement learning provides additional gains in robustness to challenging Markdown instructions when initialized from a strong SFT policy. Our results also reveal an inherent trade-off between semantic and structural objectives, highlighting the importance of carefully designed rewards for reliable formatted generation. Code is available at: this https URL. 

---
# Can Post-Training Transform LLMs into Causal Reasoners? 

**Authors**: Junqi Chen, Sirui Chen, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.06337)  

**Abstract**: Causal inference is essential for decision-making but remains challenging for non-experts. While large language models (LLMs) show promise in this domain, their precise causal estimation capabilities are still limited, and the impact of post-training on these abilities is insufficiently explored. This paper examines the extent to which post-training can enhance LLMs' capacity for causal inference. We introduce CauGym, a comprehensive dataset comprising seven core causal tasks for training and five diverse test sets. Using this dataset, we systematically evaluate five post-training approaches: SFT, DPO, KTO, PPO, and GRPO. Across five in-domain and four existing benchmarks, our experiments demonstrate that appropriate post-training enables smaller LLMs to perform causal inference competitively, often surpassing much larger models. Our 14B parameter model achieves 93.5% accuracy on the CaLM benchmark, compared to 55.4% by OpenAI o3. Furthermore, the post-trained LLMs exhibit strong generalization and robustness under real-world conditions such as distribution shifts and noisy data. Collectively, these findings provide the first systematic evidence that targeted post-training can produce reliable and robust LLM-based causal reasoners. Our data and GRPO-model are available at this https URL. 

---
# VowelPrompt: Hearing Speech Emotions from Text via Vowel-level Prosodic Augmentation 

**Authors**: Yancheng Wang, Osama Hanna, Ruiming Xie, Xianfeng Rui, Maohao Shen, Xuedong Zhang, Christian Fuegen, Jilong Wu, Debjyoti Paul, Arthur Guo, Zhihong Lei, Ozlem Kalinli, Qing He, Yingzhen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06270)  

**Abstract**: Emotion recognition in speech presents a complex multimodal challenge, requiring comprehension of both linguistic content and vocal expressivity, particularly prosodic features such as fundamental frequency, intensity, and temporal dynamics. Although large language models (LLMs) have shown promise in reasoning over textual transcriptions for emotion recognition, they typically neglect fine-grained prosodic information, limiting their effectiveness and interpretability. In this work, we propose VowelPrompt, a linguistically grounded framework that augments LLM-based emotion recognition with interpretable, fine-grained vowel-level prosodic cues. Drawing on phonetic evidence that vowels serve as primary carriers of affective prosody, VowelPrompt extracts pitch-, energy-, and duration-based descriptors from time-aligned vowel segments, and converts these features into natural language descriptions for better interpretability. Such a design enables LLMs to jointly reason over lexical semantics and fine-grained prosodic variation. Moreover, we adopt a two-stage adaptation procedure comprising supervised fine-tuning (SFT) followed by Reinforcement Learning with Verifiable Reward (RLVR), implemented via Group Relative Policy Optimization (GRPO), to enhance reasoning capability, enforce structured output adherence, and improve generalization across domains and speaker variations. Extensive evaluations across diverse benchmark datasets demonstrate that VowelPrompt consistently outperforms state-of-the-art emotion recognition methods under zero-shot, fine-tuned, cross-domain, and cross-linguistic conditions, while enabling the generation of interpretable explanations that are jointly grounded in contextual semantics and fine-grained prosodic structure. 

---
# Self-Improving World Modelling with Latent Actions 

**Authors**: Yifu Qiu, Zheng Zhao, Waylon Li, Yftah Ziser, Anna Korhonen, Shay B. Cohen, Edoardo M. Ponti  

**Link**: [PDF](https://arxiv.org/pdf/2602.06130)  

**Abstract**: Internal modelling of the world -- predicting transitions between previous states $X$ and next states $Y$ under actions $Z$ -- is essential to reasoning and planning for LLMs and VLMs. Learning such models typically requires costly action-labelled trajectories. We propose SWIRL, a self-improvement framework that learns from state-only sequences by treating actions as a latent variable and alternating between Forward World Modelling (FWM) $P_\theta(Y|X,Z)$ and an Inverse Dynamics Modelling (IDM) $Q_\phi(Z|X,Y)$. SWIRL iterates two phases: (1) Variational Information Maximisation, which updates the FWM to generate next states that maximise conditional mutual information with latent actions given prior states, encouraging identifiable consistency; and (2) ELBO Maximisation, which updates the IDM to explain observed transitions, effectively performing coordinate ascent. Both models are trained with reinforcement learning (specifically, GRPO) with the opposite frozen model's log-probability as a reward signal. We provide theoretical learnability guarantees for both updates, and evaluate SWIRL on LLMs and VLMs across multiple environments: single-turn and multi-turn open-world visual dynamics and synthetic textual environments for physics, web, and tool calling. SWIRL achieves gains of 16% on AURORABench, 28% on ByteMorph, 16% on WorldPredictionBench, and 14% on StableToolBench. 

---
# LLM Active Alignment: A Nash Equilibrium Perspective 

**Authors**: Tonghan Wang, Yuqi Pan, Xinyi Yang, Yanchen Jiang, Milind Tambe, David C. Parkes  

**Link**: [PDF](https://arxiv.org/pdf/2602.06836)  

**Abstract**: We develop a game-theoretic framework for predicting and steering the behavior of populations of large language models (LLMs) through Nash equilibrium (NE) analysis. To avoid the intractability of equilibrium computation in open-ended text spaces, we model each agent's action as a mixture over human subpopulations. Agents choose actively and strategically which groups to align with, yielding an interpretable and behaviorally substantive policy class. We derive closed-form NE characterizations, adopting standard concave-utility assumptions to enable analytical system-level predictions and give explicit, actionable guidance for shifting alignment targets toward socially desirable outcomes. The method functions as an active alignment layer on top of existing alignment pipelines such as RLHF. In a social-media setting, we show that a population of LLMs, especially reasoning-based models, may exhibit political exclusion, pathologies where some subpopulations are ignored by all LLM agents, which can be avoided by our method, illustrating the promise of applying the method to regulate multi-agent LLM dynamics across domains. 

---
# compar:IA: The French Government's LLM arena to collect French-language human prompts and preference data 

**Authors**: Lucie Termignon, Simonas Zilinskas, Hadrien Pélissier, Aurélien Barrot, Nicolas Chesnais, Elie Gavoty  

**Link**: [PDF](https://arxiv.org/pdf/2602.06669)  

**Abstract**: Large Language Models (LLMs) often show reduced performance, cultural alignment, and safety robustness in non-English languages, partly because English dominates both pre-training data and human preference alignment datasets. Training methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) require human preference data, which remains scarce and largely non-public for many languages beyond English. To address this gap, we introduce compar:IA, an open-source digital public service developed inside the French government and designed to collect large-scale human preference data from a predominantly French-speaking general audience. The platform uses a blind pairwise comparison interface to capture unconstrained, real-world prompts and user judgments across a diverse set of language models, while maintaining low participation friction and privacy-preserving automated filtering. As of 2026-02-07, compar:IA has collected over 600,000 free-form prompts and 250,000 preference votes, with approximately 89% of the data in French. We release three complementary datasets -- conversations, votes, and reactions -- under open licenses, and present initial analyses, including a French-language model leaderboard and user interaction patterns. Beyond the French context, compar:IA is evolving toward an international digital public good, offering reusable infrastructure for multilingual model training, evaluation, and the study of human-AI interaction. 

---
