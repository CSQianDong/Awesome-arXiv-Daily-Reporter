# Toward Closed-loop Molecular Discovery via Language Model, Property Alignment and Strategic Search 

**Authors**: Junkai Ji, Zhangfan Yang, Dong Xu, Ruibin Bai, Jianqiang Li, Tingjun Hou, Zexuan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.09566)  

**Abstract**: Drug discovery is a time-consuming and expensive process, with traditional high-throughput and docking-based virtual screening hampered by low success rates and limited scalability. Recent advances in generative modelling, including autoregressive, diffusion, and flow-based approaches, have enabled de novo ligand design beyond the limits of enumerative screening. Yet these models often suffer from inadequate generalization, limited interpretability, and an overemphasis on binding affinity at the expense of key pharmacological properties, thereby restricting their translational utility. Here we present Trio, a molecular generation framework integrating fragment-based molecular language modeling, reinforcement learning, and Monte Carlo tree search, for effective and interpretable closed-loop targeted molecular design. Through the three key components, Trio enables context-aware fragment assembly, enforces physicochemical and synthetic feasibility, and guides a balanced search between the exploration of novel chemotypes and the exploitation of promising intermediates within protein binding pockets. Experimental results show that Trio reliably achieves chemically valid and pharmacologically enhanced ligands, outperforming state-of-the-art approaches with improved binding affinity (+7.85%), drug-likeness (+11.10%) and synthetic accessibility (+12.05%), while expanding molecular diversity more than fourfold. 

---
# RouteRAG: Efficient Retrieval-Augmented Generation from Text and Graph via Reinforcement Learning 

**Authors**: Yucan Guo, Miao Su, Saiping Guan, Zihao Sun, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.09487)  

**Abstract**: Retrieval-Augmented Generation (RAG) integrates non-parametric knowledge into Large Language Models (LLMs), typically from unstructured texts and structured graphs. While recent progress has advanced text-based RAG to multi-turn reasoning through Reinforcement Learning (RL), extending these advances to hybrid retrieval introduces additional challenges. Existing graph-based or hybrid systems typically depend on fixed or handcrafted retrieval pipelines, lacking the ability to integrate supplementary evidence as reasoning unfolds. Besides, while graph evidence provides relational structures crucial for multi-hop reasoning, it is substantially more expensive to retrieve. To address these limitations, we introduce \model{}, an RL-based framework that enables LLMs to perform multi-turn and adaptive graph-text hybrid RAG. \model{} jointly optimizes the entire generation process via RL, allowing the model to learn when to reason, what to retrieve from either texts or graphs, and when to produce final answers, all within a unified generation policy. To guide this learning process, we design a two-stage training framework that accounts for both task outcome and retrieval efficiency, enabling the model to exploit hybrid evidence while avoiding unnecessary retrieval overhead. Experimental results across five question answering benchmarks demonstrate that \model{} significantly outperforms existing RAG baselines, highlighting the benefits of end-to-end RL in supporting adaptive and efficient retrieval for complex reasoning. 

---
# ChronusOmni: Improving Time Awareness of Omni Large Language Models 

**Authors**: Yijing Chen, Yihan Wu, Kaisi Guan, Yuchen Ren, Yuyue Wang, Ruihua Song, Liyun Ru  

**Link**: [PDF](https://arxiv.org/pdf/2512.09841)  

**Abstract**: Time awareness is a fundamental ability of omni large language models, especially for understanding long videos and answering complex questions. Previous approaches mainly target vision-language scenarios and focus on the explicit temporal grounding questions, such as identifying when a visual event occurs or determining what event happens at aspecific time. However, they often make insufficient use of the audio modality, and overlook implicit temporal grounding across modalities--for example, identifying what is visually present when a character speaks, or determining what is said when a visual event occurs--despite such cross-modal temporal relations being prevalent in real-world scenarios. In this paper, we propose ChronusOmni, an omni large language model designed to enhance temporal awareness for both explicit and implicit audiovisual temporal grounding. First, we interleave text-based timestamp tokens with visual and audio representations at each time unit, enabling unified temporal modeling across modalities. Second, to enforce correct temporal ordering and strengthen fine-grained temporal reasoning, we incorporate reinforcement learning with specially designed reward functions. Moreover, we construct ChronusAV, a temporally-accurate, modality-complete, and cross-modal-aligned dataset to support the training and evaluation on audiovisual temporal grounding task. Experimental results demonstrate that ChronusOmni achieves state-of-the-art performance on ChronusAV with more than 30% improvement and top results on most metrics upon other temporal grounding benchmarks. This highlights the strong temporal awareness of our model across modalities, while preserving general video and audio understanding capabilities. 

---
# d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models 

**Authors**: Leyi Pan, Shuchang Tao, Yunpeng Zhai, Zheyu Fu, Liancheng Fang, Minghua He, Lingzhe Zhang, Zhaoyang Liu, Bolin Ding, Aiwei Liu, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2512.09675)  

**Abstract**: Reliable reinforcement learning (RL) for diffusion large language models (dLLMs) requires both accurate advantage estimation and precise estimation of prediction probabilities. Existing RL methods for dLLMs fall short in both aspects: they rely on coarse or unverifiable reward signals, and they estimate prediction probabilities without accounting for the bias relative to the true, unbiased expected prediction probability that properly integrates over all possible decoding orders. To mitigate these issues, we propose \emph{d}-TreeRPO, a reliable RL framework for dLLMs that leverages tree-structured rollouts and bottom-up advantage computation based on verifiable outcome rewards to provide fine-grained and verifiable step-wise reward signals. When estimating the conditional transition probability from a parent node to a child node, we theoretically analyze the estimation error between the unbiased expected prediction probability and the estimate obtained via a single forward pass, and find that higher prediction confidence leads to lower estimation error. Guided by this analysis, we introduce a time-scheduled self-distillation loss during training that enhances prediction confidence in later training stages, thereby enabling more accurate probability estimation and improved convergence. Experiments show that \emph{d}-TreeRPO outperforms existing baselines and achieves significant gains on multiple reasoning benchmarks, including +86.2 on Sudoku, +51.6 on Countdown, +4.5 on GSM8K, and +5.3 on Math500. Ablation studies and computational cost analyses further demonstrate the effectiveness and practicality of our design choices. 

---
# MOA: Multi-Objective Alignment for Role-Playing Agents 

**Authors**: Chonghua Liao, Ke Wang, Yuchuan Wu, Fei Huang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.09756)  

**Abstract**: Role-playing agents (RPAs) must simultaneously master many conflicting skills -- following multi-turn instructions, exhibiting domain knowledge, and adopting a consistent linguistic style. Existing work either relies on supervised fine-tuning (SFT) that over-fits surface cues and yields low diversity, or applies reinforcement learning (RL) that fails to learn multiple dimensions for comprehensive RPA optimization. We present MOA (Multi-Objective Alignment), a reinforcement-learning framework that enables multi-dimensional, fine-grained rubric optimization for general RPAs. MOA introduces a novel multi-objective optimization strategy that trains simultaneously on multiple fine-grained rubrics to boost optimization performance. Besides, to address the issues of model output diversity and quality, we have also employed thought-augmented rollout with off-policy guidance. Extensive experiments on challenging benchmarks such as PersonaGym and RoleMRC show that MOA enables an 8B model to match or even outperform strong baselines such as GPT-4o and Claude across numerous dimensions. This demonstrates the great potential of MOA in building RPAs that can simultaneously meet the demands of role knowledge, persona style, diverse scenarios, and complex multi-turn conversations. 

---
# MentraSuite: Post-Training Large Language Models for Mental Health Reasoning and Assessment 

**Authors**: Mengxi Xiao, Kailai Yang, Pengde Zhao, Enze Zhang, Ziyan Kuang, Zhiwei Liu, Weiguang Han, Shu Liao, Lianting Huang, Jinpeng Hu, Min Peng, Qianqian Xie, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2512.09636)  

**Abstract**: Mental health disorders affect hundreds of millions globally, and the Web now serves as a primary medium for accessing support, information, and assessment. Large language models (LLMs) offer scalable and accessible assistance, yet their deployment in mental-health settings remains risky when their reasoning is incomplete, inconsistent, or ungrounded. Existing psychological LLMs emphasize emotional understanding or knowledge recall but overlook the step-wise, clinically aligned reasoning required for appraisal, diagnosis, intervention planning, abstraction, and verification. To address these issues, we introduce MentraSuite, a unified framework for advancing reliable mental-health reasoning. We propose MentraBench, a comprehensive benchmark spanning five core reasoning aspects, six tasks, and 13 datasets, evaluating both task performance and reasoning quality across five dimensions: conciseness, coherence, hallucination avoidance, task understanding, and internal consistency. We further present Mindora, a post-trained model optimized through a hybrid SFT-RL framework with an inconsistency-detection reward to enforce faithful and coherent reasoning. To support training, we construct high-quality trajectories using a novel reasoning trajectory generation strategy, that strategically filters difficult samples and applies a structured, consistency-oriented rewriting process to produce concise, readable, and well-balanced trajectories. Across 20 evaluated LLMs, Mindora achieves the highest average performance on MentraBench and shows remarkable performances in reasoning reliability, demonstrating its effectiveness for complex mental-health scenarios. 

---
# Targeting Misalignment: A Conflict-Aware Framework for Reward-Model-based LLM Alignment 

**Authors**: Zixuan Liu, Siavash H. Khajavi, Guangkai Jiang, Xinru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.09212)  

**Abstract**: Reward-model-based fine-tuning is a central paradigm in aligning Large Language Models with human preferences. However, such approaches critically rely on the assumption that proxy reward models accurately reflect intended supervision, a condition often violated due to annotation noise, bias, or limited coverage. This misalignment can lead to undesirable behaviors, where models optimize for flawed signals rather than true human values. In this paper, we investigate a novel framework to identify and mitigate such misalignment by treating the fine-tuning process as a form of knowledge integration. We focus on detecting instances of proxy-policy conflicts, cases where the base model strongly disagrees with the proxy. We argue that such conflicts often signify areas of shared ignorance, where neither the policy nor the reward model possesses sufficient knowledge, making them especially susceptible to misalignment. To this end, we propose two complementary metrics for identifying these conflicts: a localized Proxy-Policy Alignment Conflict Score (PACS) and a global Kendall-Tau Distance measure. Building on this insight, we design an algorithm named Selective Human-in-the-loop Feedback via Conflict-Aware Sampling (SHF-CAS) that targets high-conflict QA pairs for additional feedback, refining both the reward model and policy efficiently. Experiments on two alignment tasks demonstrate that our approach enhances general alignment performance, even when trained with a biased proxy reward. Our work provides a new lens for interpreting alignment failures and offers a principled pathway for targeted refinement in LLM training. 

---
# Enhancing Reliability across Short and Long-Form QA via Reinforcement Learning 

**Authors**: Yudong Wang, Zhe Yang, Wenhan Ma, Zhifang Sui, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.08944)  

**Abstract**: While reinforcement learning has unlocked unprecedented complex reasoning in large language models, it has also amplified their propensity for hallucination, creating a critical trade-off between capability and reliability. This work confronts this challenge by introducing a targeted RL framework designed to mitigate both intrinsic and extrinsic hallucinations across short and long-form question answering. We address extrinsic hallucinations (flawed internal knowledge) by creating a novel training set from open-ended conversions of TriviaQA. Concurrently, we tackle intrinsic hallucinations (unfaithfulness to context) by leveraging long-form texts from FineWeb in a fact-grounding reward scheme. To further bolster reliability, our framework explicitly rewards the model for refusing to answer unanswerable questions, thereby cultivating crucial cautiousness. Extensive experiments demonstrate that our methodology yields significant performance gains across a diverse suite of benchmarks, substantially reducing both hallucination types. Ultimately, this research contributes a practical framework for resolving the critical tension between advanced reasoning and factual trustworthiness, paving the way for more capable and reliable large language models. 

---
