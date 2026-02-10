# iGRPO: Self-Feedback-Driven LLM Reasoning 

**Authors**: Ali Hatamizadeh, Shrimai Prabhumoye, Igor Gitman, Ximing Lu, Seungju Han, Wei Ping, Yejin Choi, Jan Kautz  

**Link**: [PDF](https://arxiv.org/pdf/2602.09000)  

**Abstract**: Large Language Models (LLMs) have shown promise in solving complex mathematical problems, yet they still fall short of producing accurate and consistent solutions. Reinforcement Learning (RL) is a framework for aligning these models with task-specific rewards, improving overall quality and reliability. Group Relative Policy Optimization (GRPO) is an efficient, value-function-free alternative to Proximal Policy Optimization (PPO) that leverages group-relative reward normalization. We introduce Iterative Group Relative Policy Optimization (iGRPO), a two-stage extension of GRPO that adds dynamic self-conditioning through model-generated drafts. In Stage 1, iGRPO samples multiple exploratory drafts and selects the highest-reward draft using the same scalar reward signal used for optimization. In Stage 2, it appends this best draft to the original prompt and applies a GRPO-style update on draft-conditioned refinements, training the policy to improve beyond its strongest prior attempt. Under matched rollout budgets, iGRPO consistently outperforms GRPO across base models (e.g., Nemotron-H-8B-Base-8K and DeepSeek-R1 Distilled), validating its effectiveness on diverse reasoning benchmarks. Moreover, applying iGRPO to OpenReasoning-Nemotron-7B trained on AceReason-Math achieves new state-of-the-art results of 85.62\% and 79.64\% on AIME24 and AIME25, respectively. Ablations further show that the refinement wrapper generalizes beyond GRPO variants, benefits from a generative judge, and alters learning dynamics by delaying entropy collapse. These results underscore the potential of iterative, self-feedback-based RL for advancing verifiable mathematical reasoning. 

---
# Learning the Value Systems of Societies with Preference-based Multi-objective Reinforcement Learning 

**Authors**: Andrés Holgado-Sánchez, Peter Vamplew, Richard Dazeley, Sascha Ossowski, Holger Billhardt  

**Link**: [PDF](https://arxiv.org/pdf/2602.08835)  

**Abstract**: Value-aware AI should recognise human values and adapt to the value systems (value-based preferences) of different users. This requires operationalization of values, which can be prone to misspecification. The social nature of values demands their representation to adhere to multiple users while value systems are diverse, yet exhibit patterns among groups. In sequential decision making, efforts have been made towards personalization for different goals or values from demonstrations of diverse agents. However, these approaches demand manually designed features or lack value-based interpretability and/or adaptability to diverse user preferences.
We propose algorithms for learning models of value alignment and value systems for a society of agents in Markov Decision Processes (MDPs), based on clustering and preference-based multi-objective reinforcement learning (PbMORL). We jointly learn socially-derived value alignment models (groundings) and a set of value systems that concisely represent different groups of users (clusters) in a society. Each cluster consists of a value system representing the value-based preferences of its members and an approximately Pareto-optimal policy that reflects behaviours aligned with this value system. We evaluate our method against a state-of-the-art PbMORL algorithm and baselines on two MDPs with human values. 

---
# Reinforcement Inference: Leveraging Uncertainty for Self-Correcting Language Model Reasoning 

**Authors**: Xinhai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.08520)  

**Abstract**: Modern large language models (LLMs) are often evaluated and deployed under a \emph{one-shot, greedy} inference protocol, especially in professional settings that require deterministic behavior. This regime can systematically under-estimate a fixed model's true capability: many errors arise not from missing knowledge, but from premature commitment under internal ambiguity. We introduce \emph{Reinforcement Inference}, an entropy-aware inference-time control strategy that uses the model's own uncertainty to selectively invoke a second, more deliberate reasoning attempt, enabling stronger performance \emph{without any retraining}.
On 12,032 MMLU-Pro questions across 14 subjects, using DeepSeek-v3.2 with deterministic decoding in a zero-shot setting, Reinforcement Inference improves accuracy from 60.72\% to 84.03\%, while only incurring 61.06\% additional inference calls. A 100\% re-asking ablation reaches 84.35\%, indicating that uncertainty-aware selection captures most of the attainable improvement with substantially less compute. Moreover, a \emph{prompt-only} ablation underperforms the baseline, suggesting that the gains are not explained by generic `` your output had high entropy, think step-by-step'' prompting alone.
Beyond providing a practical inference-time upgrade, our results suggest a broader \emph{entropy-aware} paradigm for measuring and expanding model capability: because modern decoder-based models generate outputs autoregressively, entropy and related confidence measures arise naturally as first-class control signals during generation. The resulting gap between one-pass greedy inference and uncertainty-conditioned deliberation offers a diagnostic lens on an LLM's latent reasoning horizon and motivates future training objectives that explicitly constrain correctness--confidence alignment. 

---
# Dialogue Model Optimization via Agent Game and Adaptive Tree-based GRPO 

**Authors**: Kun Peng, Conghui Tan, Yu Liu, Guohua Tang, Zhongqian Sun, Wei Yang, Zining Zhu, Lei Jiang, Yanbing Liu, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2602.08533)  

**Abstract**: Open-ended dialogue agents aim to deliver engaging, personalized interactions by adapting to users' traits, but existing methods face critical limitations: over-reliance on pre-collected user data, and short-horizon biases in reinforcement learning (RL) that neglect long-term dialogue value. To address these, we propose a novel long-horizon RL framework integrating online personalization with Adaptive Tree-based Group Relative Policy Optimization (AT-GRPO). Adopting a two-agent game paradigm, a user agent constructs dynamic environments via style mimicry (learning user-specific conversational traits) and active termination (predicting turn-level termination probabilities as immediate rewards), forming an iterative cycle that drives the dialogue agent to deepen interest exploration. AT-GRPO reinterprets dialogue trajectories as trees and introduces adaptive observation ranges. Unlike full tree expansion that incurs exponential overhead, it limits each node to aggregate rewards from a stage-aware range: larger ranges support early-stage topic exploration, while smaller ranges facilitate late-stage dialogue maintenance. This design reduces rollout budgets from exponential to polynomial in the dialogue length, while preserving long-term reward capture. Extensive experiments show our framework's superior performance, sample efficiency, and robustness. 

---
# OPE: Overcoming Information Saturation in Parallel Thinking via Outline-Guided Path Exploration 

**Authors**: Qi Guo, Jianing Wang, Deyang Kong, Xiangyu Xi, Jianfei Zhang, Yi Lu, Jingang Wang, Wei Wang, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2602.08344)  

**Abstract**: Parallel thinking has emerged as a new paradigm for large reasoning models (LRMs) in tackling complex problems. Recent methods leverage Reinforcement Learning (RL) to enhance parallel thinking, aiming to address the limitations in computational resources and effectiveness encountered with supervised fine-tuning. However, most existing studies primarily focus on optimizing the aggregation phase, with limited attention to the path exploration stage. In this paper, we theoretically analyze the optimization of parallel thinking under the Reinforcement Learning with Verifiable Rewards (RLVR) setting, and identify that the mutual information bottleneck among exploration paths fundamentally restricts overall performance. To address this, we propose Outline-Guided Path Exploration (OPE), which explicitly partitions the solution space by generating diverse reasoning outlines prior to parallel path reasoning, thereby reducing information redundancy and improving the diversity of information captured across exploration paths. We implement OPE with an iterative RL strategy that optimizes outline planning and outline-guided reasoning independently. Extensive experiments across multiple challenging mathematical benchmarks demonstrate that OPE effectively improves reasoning performance in different aggregation strategies, enabling LRMs to more reliably discover correct solutions. 

---
# CoTZero: Annotation-Free Human-Like Vision Reasoning via Hierarchical Synthetic CoT 

**Authors**: Chengyi Du, Yazhe Niu, Dazhong Shen, Luxin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08339)  

**Abstract**: Recent advances in vision-language models (VLMs) have markedly improved image-text alignment, yet they still fall short of human-like visual reasoning. A key limitation is that many VLMs rely on surface correlations rather than building logically coherent structured representations, which often leads to missed higher-level semantic structure and non-causal relational understanding, hindering compositional and verifiable reasoning. To address these limitations by introducing human models into the reasoning process, we propose CoTZero, an annotation-free paradigm with two components: (i) a dual-stage data synthesis approach and (ii) a cognition-aligned training method. In the first component, we draw inspiration from neurocognitive accounts of compositional productivity and global-to-local analysis. In the bottom-up stage, CoTZero extracts atomic visual primitives and incrementally composes them into diverse, structured question-reasoning forms. In the top-down stage, it enforces hierarchical reasoning by using coarse global structure to guide the interpretation of local details and causal relations. In the cognition-aligned training component, built on the synthesized CoT data, we introduce Cognitively Coherent Verifiable Rewards (CCVR) in Reinforcement Fine-Tuning (RFT) to further strengthen VLMs' hierarchical reasoning and generalization, providing stepwise feedback on reasoning coherence and factual correctness. Experiments show that CoTZero achieves an F1 score of 83.33 percent on our multi-level semantic inconsistency benchmark with lexical-perturbation negatives, across both in-domain and out-of-domain settings. Ablations confirm that each component contributes to more interpretable and human-aligned visual reasoning. 

---
# Do MLLMs Really See It: Reinforcing Visual Attention in Multimodal LLMs 

**Authors**: Siqu Ou, Tianrui Wan, Zhiyuan Zhao, Junyu Gao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.08241)  

**Abstract**: While chain-of-thought (CoT) reasoning has substantially improved multimodal large language models (MLLMs) on complex reasoning tasks, existing approaches largely rely on long textual reasoning trajectories and provide limited mechanisms for learning stable visual attention policies. Our analysis shows that current MLLMs exhibit weak visual focus: early-stage visual misalignment is rarely corrected during subsequent reasoning, leading to error propagation and failed inferences. We argue that this limitation stems from inadequate credit assignment for visual attention during training. To address this issue, we propose SAYO, a visual reasoning model trained with a reinforcement learning (RL) framework that introduces a region-level visual attention-based reward. This reward explicitly aligns optimization signals with visually grounded reasoning steps, enabling the model to learn more reliable attention behaviors. Extensive experiments across multiple multimodal benchmarks demonstrate that SAYO consistently improves performance on diverse reasoning and perception tasks. 

---
# Time Series Reasoning via Process-Verifiable Thinking Data Synthesis and Scheduling for Tailored LLM Reasoning 

**Authors**: Jiahui Zhou, Dan Li, Boxin Li, Xiao Zhang, Erli Meng, Lin Li, Zhuomin Chen, Jian Lou, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2602.07830)  

**Abstract**: Time series is a pervasive data type across various application domains, rendering the reasonable solving of diverse time series tasks a long-standing goal. Recent advances in large language models (LLMs), especially their reasoning abilities unlocked through reinforcement learning (RL), have opened new opportunities for tackling tasks with long Chain-of-Thought (CoT) reasoning. However, leveraging LLM reasoning for time series remains in its infancy, hindered by the absence of carefully curated time series CoT data for training, limited data efficiency caused by underexplored data scheduling, and the lack of RL algorithms tailored for exploiting such time series CoT data. In this paper, we introduce VeriTime, a framework that tailors LLMs for time series reasoning through data synthesis, data scheduling, and RL training. First, we propose a data synthesis pipeline that constructs a TS-text multimodal dataset with process-verifiable annotations. Second, we design a data scheduling mechanism that arranges training samples according to a principled hierarchy of difficulty and task taxonomy. Third, we develop a two-stage reinforcement finetuning featuring fine-grained, multi-objective rewards that leverage verifiable process-level CoT data. Extensive experiments show that VeriTime substantially boosts LLM performance across diverse time series reasoning tasks. Notably, it enables compact 3B, 4B models to achieve reasoning capabilities on par with or exceeding those of larger proprietary LLMs. 

---
# Joint Reward Modeling: Internalizing Chain-of-Thought for Efficient Visual Reward Models 

**Authors**: Yankai Yang, Yancheng Long, Hongyang Wei, Wei Chen, Tianke Zhang, Kaiyu Jiang, Haonan Fan, Changyi Liu, Jiankang Chen, Kaiyu Tang, Bin Wen, Fan Yang, Tingting Gao, Han Li, Shuo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.07533)  

**Abstract**: Reward models are critical for reinforcement learning from human feedback, as they determine the alignment quality and reliability of generative models. For complex tasks such as image editing, reward models are required to capture global semantic consistency and implicit logical constraints beyond local similarity. Existing reward modeling approaches have clear limitations. Discriminative reward models align well with human preferences but struggle with complex semantics due to limited reasoning supervision. Generative reward models offer stronger semantic understanding and reasoning, but they are costly at inference time and difficult to align directly with human preferences. To this end, we propose Joint Reward Modeling (JRM), which jointly optimizes preference learning and language modeling on a shared vision-language backbone. This approach internalizes the semantic and reasoning capabilities of generative models into efficient discriminative representations, enabling fast and accurate evaluation. JRM achieves state-of-the-art results on MMRB2 and EditReward-Bench, and significantly improves stability and performance in downstream online reinforcement learning. These results show that joint training effectively bridges efficiency and semantic understanding in reward modeling. 

---
# AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection 

**Authors**: Junru Zhang, Lang Feng, Haoran Shi, Xu Guo, Han Yu, Yabo Dong, Duanqing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.08868)  

**Abstract**: Time-series anomaly detection (TSAD) with multimodal large language models (MLLMs) is an emerging area, yet a persistent challenge remains: MLLMs rely on coarse time-series heuristics but struggle with multi-dimensional, detailed reasoning, which is vital for understanding complex time-series data. We present AnomSeer to address this by reinforcing the model to ground its reasoning in precise, structural details of time series, unifying anomaly classification, localization, and explanation. At its core, an expert chain-of-thought trace is generated to provide a verifiable, fine-grained reasoning from classical analyses (e.g., statistical measures, frequency transforms). Building on this, we propose a novel time-series grounded policy optimization (TimerPO) that incorporates two additional components beyond standard reinforcement learning: a time-series grounded advantage based on optimal transport and an orthogonal projection to ensure this auxiliary granular signal does not interfere with the primary detection objective. Across diverse anomaly scenarios, AnomSeer, with Qwen2.5-VL-3B/7B-Instruct, outperforms larger commercial baselines (e.g., GPT-4o) in classification and localization accuracy, particularly on point- and frequency-driven exceptions. Moreover, it produces plausible time-series reasoning traces that support its conclusions. 

---
# WildReward: Learning Reward Models from In-the-Wild Human Interactions 

**Authors**: Hao Peng, Yunjia Qi, Xiaozhi Wang, Zijun Yao, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.08829)  

**Abstract**: Reward models (RMs) are crucial for the training of large language models (LLMs), yet they typically rely on large-scale human-annotated preference pairs. With the widespread deployment of LLMs, in-the-wild interactions have emerged as a rich source of implicit reward signals. This raises the question: Can we develop reward models directly from in-the-wild interactions? In this work, we explore this possibility by adopting WildChat as an interaction source and proposing a pipeline to extract reliable human feedback, yielding 186k high-quality instances for training WildReward via ordinal regression directly on user feedback without preference pairs. Extensive experiments demonstrate that WildReward achieves comparable or even superior performance compared to conventional reward models, with improved calibration and cross-sample consistency. We also observe that WildReward benefits directly from user diversity, where more users yield stronger reward models. Finally, we apply WildReward to online DPO training and observe significant improvements across various tasks. Code and data are released at this https URL. 

---
# Affective Flow Language Model for Emotional Support Conversation 

**Authors**: Chenghui Zou, Ning Wang, Tiesunlong Shen, Luwei Xiao, Chuan Ma, Xiangpeng Li, Rui Mao, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2602.08826)  

**Abstract**: Large language models (LLMs) have been widely applied to emotional support conversation (ESC). However, complex multi-turn support remains this http URL is because existing alignment schemes rely on sparse outcome-level signals, thus offering limited supervision for intermediate strategy decisions. To fill this gap, this paper proposes affective flow language model for emotional support conversation (AFlow), a framework that introduces fine-grained supervision on dialogue prefixes by modeling a continuous affective flow along multi-turn trajectories. AFlow can estimate intermediate utility over searched trajectories and learn preference-consistent strategy transitions. To improve strategy coherence and empathetic response quality, a subpath-level flow-balance objective is presented to propagate preference signals to intermediate states. Experiment results show consistent and significant improvements over competitive baselines in diverse emotional contexts. Remarkably, AFlow with a compact open-source backbone outperforms proprietary LMMs such as GPT-4o and Claude-3.5 on major ESC metrics. Our code is available at this https URL. 

---
# LLaDA2.1: Speeding Up Text Diffusion via Token Editing 

**Authors**: Tiwei Bie, Maosong Cao, Xiang Cao, Bingsen Chen, Fuyuan Chen, Kun Chen, Lun Du, Daozhuo Feng, Haibo Feng, Mingliang Gong, Zhuocheng Gong, Yanmei Gu, Jian Guan, Kaiyuan Guan, Hongliang He, Zenan Huang, Juyong Jiang, Zhonghui Jiang, Zhenzhong Lan, Chengxi Li, Jianguo Li, Zehuan Li, Huabin Liu, Lin Liu, Guoshan Lu, Yuan Lu, Yuxin Ma, Xingyu Mou, Zhenxuan Pan, Kaida Qiu, Yuji Ren, Jianfeng Tan, Yiding Tian, Zian Wang, Lanning Wei, Tao Wu, Yipeng Xing, Wentao Ye, Liangyu Zha, Tianze Zhang, Xiaolu Zhang, Junbo Zhao, Da Zheng, Hao Zhong, Wanli Zhong, Jun Zhou, Junlin Zhou, Liwang Zhu, Muzhi Zhu, Yihong Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08676)  

**Abstract**: While LLaDA2.0 showcased the scaling potential of 100B-level block-diffusion models and their inherent parallelization, the delicate equilibrium between decoding speed and generation quality has remained an elusive frontier. Today, we unveil LLaDA2.1, a paradigm shift designed to transcend this trade-off. By seamlessly weaving Token-to-Token (T2T) editing into the conventional Mask-to-Token (M2T) scheme, we introduce a joint, configurable threshold-decoding scheme. This structural innovation gives rise to two distinct personas: the Speedy Mode (S Mode), which audaciously lowers the M2T threshold to bypass traditional constraints while relying on T2T to refine the output; and the Quality Mode (Q Mode), which leans into conservative thresholds to secure superior benchmark performances with manageable efficiency degrade. Furthering this evolution, underpinned by an expansive context window, we implement the first large-scale Reinforcement Learning (RL) framework specifically tailored for dLLMs, anchored by specialized techniques for stable gradient estimation. This alignment not only sharpens reasoning precision but also elevates instruction-following fidelity, bridging the chasm between diffusion dynamics and complex human intent. We culminate this work by releasing LLaDA2.1-Mini (16B) and LLaDA2.1-Flash (100B). Across 33 rigorous benchmarks, LLaDA2.1 delivers strong task performance and lightning-fast decoding speed. Despite its 100B volume, on coding tasks it attains an astounding 892 TPS on HumanEval+, 801 TPS on BigCodeBench, and 663 TPS on LiveCodeBench. 

---
# Contextual Rollout Bandits for Reinforcement Learning with Verifiable Rewards 

**Authors**: Xiaodong Lu, Xiaohan Wang, Jiajun Chai, Guojun Yin, Wei Lin, Zhijun Chen, Yu Luo, Fuzhen Zhuang, Yikun Ban, Deqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08499)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is an effective paradigm for improving the reasoning capabilities of large language models. However, existing RLVR methods utilize rollouts in an indiscriminate and short-horizon manner: responses of heterogeneous quality within each prompt are treated uniformly, and historical rollouts are discarded after a single use. This leads to noisy supervision, poor sample efficiency, and suboptimal policy updates. We address these issues by formulating rollout scheduling in RLVR as a contextual bandit problem and proposing a unified neural scheduling framework that adaptively selects high-value rollouts throughout training. Each rollout is treated as an arm whose reward is defined by the induced performance gain between consecutive optimization steps. The resulting scheduler supports both noise-aware intra-group selection and adaptive global reuse of historical rollouts within a single principled framework. We provide theoretical justification by deriving sublinear regret bounds and showing that enlarging the rollout buffer improves the achievable performance upper bound. Experiments on six mathematical reasoning benchmarks demonstrate consistent gains in performance and training efficiency across multiple RLVR optimization methods. 

---
# Dynamic Long Context Reasoning over Compressed Memory via End-to-End Reinforcement Learning 

**Authors**: Zhuoen Chen, Dongfang Li, Meishan Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08382)  

**Abstract**: Large Language Models (LLMs) face significant challenges in long-context processing, including quadratic computational costs, information forgetting, and the context fragmentation inherent in retrieval-augmented generation (RAG). We propose a cognitively inspired framework for efficient long-context inference based on chunk-wise compression and selective memory recall, rather than processing all raw tokens. The framework segments long inputs into chunks and encodes each chunk into compressed memory representations using a learned compressor. A gating module dynamically selects relevant memory blocks, which are then iteratively processed by a reasoning module with an evolving working memory to solve downstream tasks. The compressor and reasoner are jointly optimized via end-to-end reinforcement learning, while the gating module is trained separately as a classifier. Experimental results show that the proposed method achieves competitive accuracy on multi-hop reasoning benchmarks such as RULER-HQA, extrapolates context length from 7K to 1.75M tokens, and offers a favorable accuracy-efficiency trade-off compared to strong long-context baselines. In particular, it achieves up to a 2 times reduction in peak GPU memory usage and a 6 times inference speedup over MemAgent. 

---
# Reinforcement Learning with Backtracking Feedback 

**Authors**: Bilgehan Sel, Vaishakh Keshava, Phillip Wallis, Lukas Rutishauser, Ming Jin, Dingcheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.08377)  

**Abstract**: Addressing the critical need for robust safety in Large Language Models (LLMs), particularly against adversarial attacks and in-distribution errors, we introduce Reinforcement Learning with Backtracking Feedback (RLBF). This framework advances upon prior methods, such as BSAFE, by primarily leveraging a Reinforcement Learning (RL) stage where models learn to dynamically correct their own generation errors. Through RL with critic feedback on the model's live outputs, LLMs are trained to identify and recover from their actual, emergent safety violations by emitting an efficient "backtrack by x tokens" signal, then continuing generation autoregressively. This RL process is crucial for instilling resilience against sophisticated adversarial strategies, including middle filling, Greedy Coordinate Gradient (GCG) attacks, and decoding parameter manipulations. To further support the acquisition of this backtracking capability, we also propose an enhanced Supervised Fine-Tuning (SFT) data generation strategy (BSAFE+). This method improves upon previous data creation techniques by injecting violations into coherent, originally safe text, providing more effective initial training for the backtracking mechanism. Comprehensive empirical evaluations demonstrate that RLBF significantly reduces attack success rates across diverse benchmarks and model scales, achieving superior safety outcomes while critically preserving foundational model utility. 

---
# DrugR: Optimizing Molecular Drugs through LLM-based Explicit Reasoning 

**Authors**: Haoran Liu, Zheni Zeng, Yukun Yan, Yuxuan Chen, Yunduo Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2602.08213)  

**Abstract**: Molecule generation and optimization is a fundamental task in chemical domain. The rapid development of intelligent tools, especially large language models (LLMs) with powerful knowledge reserves and interactive capabilities, has provided new paradigms for it. Nevertheless, the intrinsic challenge for LLMs lies in the complex implicit relationship between molecular structure and pharmacological properties and the lack of corresponding labeled data. To bridge this gap, we propose DrugR, an LLM-based method that introduces explicit, step-by-step pharmacological reasoning into the optimization process. Our approach integrates domain-specific continual pretraining, supervised fine-tuning via reverse data engineering, and self-balanced multi-granular reinforcement learning. This framework enables DrugR to effectively improve key ADMET properties while preserving the original molecule's core efficacy. Experimental results demonstrate that DrugR achieves comprehensive enhancement across multiple properties without compromising structural similarity or target binding affinity. Importantly, its explicit reasoning process provides clear, interpretable rationales for each optimization step, yielding actionable design insights and advancing toward automated, knowledge-driven scientific discovery. Our code and model checkpoints are open-sourced to foster future research. 

---
# rePIRL: Learn PRM with Inverse RL for LLM Reasoning 

**Authors**: Xian Wu, Kaijie Zhu, Ying Zhang, Lun Wang, Wenbo Guo  

**Link**: [PDF](https://arxiv.org/pdf/2602.07832)  

**Abstract**: Process rewards have been widely used in deep reinforcement learning to improve training efficiency, reduce variance, and prevent reward hacking. In LLM reasoning, existing works also explore various solutions for learning effective process reward models (PRM) with or without the help of an expert policy. However, existing methods either rely on strong assumptions about the expert policies (e.g., requiring their reward functions) or suffer intrinsic limitations (e.g., entropy collapse), resulting in weak PRMs or limited generalizability. In this paper, we introduce rePIRL, an inverse RL-inspired framework that learns effective PRMs with minimal assumptions about expert policies. Specifically, we design a dual learning process that updates the policy and the PRM interchangeably. Our learning algorithm has customized techniques to address the challenges of scaling traditional inverse RL to LLMs. We theoretically show that our proposed learning framework can unify both online and offline PRM learning methods, justifying that rePIRL can learn PRMs with minimal assumptions. Empirical evaluations on standardized math and coding reasoning datasets demonstrate the effectiveness of rePIRL over existing methods. We further show the application of our trained PRM in test-time training, test-time scaling, and providing an early signal for training hard problems. Finally, we validate our training recipe and key design choices via a detailed ablation study. 

---
# Generative Reasoning Re-ranker 

**Authors**: Mingfu Liang, Yufei Li, Jay Xu, Kavosh Asadi, Xi Liu, Shuo Gu, Kaushik Rangadurai, Frank Shyu, Shuaiwen Wang, Song Yang, Zhijing Li, Jiang Liu, Mengying Sun, Fei Tian, Xiaohan Wei, Chonglin Sun, Jacob Tao, Shike Mei, Hamed Firooz, Wenlin Chen, Luke Simon  

**Link**: [PDF](https://arxiv.org/pdf/2602.07774)  

**Abstract**: Recent studies increasingly explore Large Language Models (LLMs) as a new paradigm for recommendation systems due to their scalability and world knowledge. However, existing work has three key limitations: (1) most efforts focus on retrieval and ranking, while the reranking phase, critical for refining final recommendations, is largely overlooked; (2) LLMs are typically used in zero-shot or supervised fine-tuning settings, leaving their reasoning abilities, especially those enhanced through reinforcement learning (RL) and high-quality reasoning data, underexploited; (3) items are commonly represented by non-semantic IDs, creating major scalability challenges in industrial systems with billions of identifiers. To address these gaps, we propose the Generative Reasoning Reranker (GR2), an end-to-end framework with a three-stage training pipeline tailored for reranking. First, a pretrained LLM is mid-trained on semantic IDs encoded from non-semantic IDs via a tokenizer achieving $\ge$99% uniqueness. Next, a stronger larger-scale LLM generates high-quality reasoning traces through carefully designed prompting and rejection sampling, which are used for supervised fine-tuning to impart foundational reasoning skills. Finally, we apply Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO), enabling scalable RL supervision with verifiable rewards designed specifically for reranking. Experiments on two real-world datasets demonstrate GR2's effectiveness: it surpasses the state-of-the-art OneRec-Think by 2.4% in Recall@5 and 1.3% in NDCG@5. Ablations confirm that advanced reasoning traces yield substantial gains across metrics. We further find that RL reward design is crucial in reranking: LLMs tend to exploit reward hacking by preserving item order, motivating conditional verifiable rewards to mitigate this behavior and optimize reranking performance. 

---
# Fairness Aware Reward Optimization 

**Authors**: Ching Lam Choi, Vighnesh Subramaniam, Phillip Isola, Antonio Torralba, Stefanie Jegelka  

**Link**: [PDF](https://arxiv.org/pdf/2602.07799)  

**Abstract**: Demographic skews in human preference data propagate systematic unfairness through reward models into aligned LLMs. We introduce Fairness Aware Reward Optimization (Faro), an in-processing framework that trains reward models under demographic parity, equalized odds, or counterfactual fairness constraints. We provide the first theoretical analysis of reward-level fairness in LLM alignment, establishing: (i) provable fairness certificates for Faro-trained rewards with controllable slack; a (ii) formal characterization of the accuracy-fairness trade-off induced by KL-regularized fine-tuning, proving fairness transfers from reward to policy; and the (iii) existence of a non-empty Pareto frontier. Unlike pre- and post-processing methods, Faro ensures reward models are simultaneously ordinal (ranking correctly), cardinal (calibrated), and fair. Across multiple LLMs and benchmarks, Faro significantly reduces bias and harmful generations while maintaining or improving model quality. 

---
# Learning to Self-Verify Makes Language Models Better Reasoners 

**Authors**: Yuxin Chen, Yu Wang, Yi Zhang, Ziang Ye, Zhengzhou Cai, Yaorui Shi, Qi Gu, Hui Su, Xunliang Cai, Xiang Wang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2602.07594)  

**Abstract**: Recent large language models (LLMs) achieve strong performance in generating promising reasoning paths for complex tasks. However, despite powerful generation ability, LLMs remain weak at verifying their own answers, revealing a persistent capability asymmetry between generation and self-verification. In this work, we conduct an in-depth investigation of this asymmetry throughout training evolution and show that, even on the same task, improving generation does not lead to corresponding improvements in self-verification. Interestingly, we find that the reverse direction of this asymmetry behaves differently: learning to self-verify can effectively improve generation performance, achieving accuracy comparable to standard generation training while yielding more efficient and effective reasoning traces. Building on this observation, we further explore integrating self-verification into generation training by formulating a multi-task reinforcement learning framework, where generation and self-verification are optimized as two independent but complementary objectives. Extensive experiments across benchmarks and models demonstrate performance gains over generation-only training in both generation and verification capabilities. 

---
# Secure Code Generation via Online Reinforcement Learning with Vulnerability Reward Model 

**Authors**: Tianyi Wu, Mingzhe Du, Yue Liu, Chengran Yang, Terry Yue Zhuo, Jiaheng Zhang, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2602.07422)  

**Abstract**: Large language models (LLMs) are increasingly used in software development, yet their tendency to generate insecure code remains a major barrier to real-world deployment. Existing secure code alignment methods often suffer from a functionality--security paradox, improving security at the cost of substantial utility degradation. We propose SecCoderX, an online reinforcement learning framework for functionality-preserving secure code generation. SecCoderX first bridges vulnerability detection and secure code generation by repurposing mature detection resources in two ways: (i) synthesizing diverse, reality-grounded vulnerability-inducing coding tasks for online RL rollouts, and (ii) training a reasoning-based vulnerability reward model that provides scalable and reliable security supervision. Together, these components are unified in an online RL loop to align code LLMs to generate secure and functional code. Extensive experiments demonstrate that SecCoderX achieves state-of-the-art performance, improving Effective Safety Rate (ESR) by approximately 10% over unaligned models, whereas prior methods often degrade ESR by 14-54%. We release our code, dataset and model checkpoints at this https URL. 

---
# Improving Data and Reward Design for Scientific Reasoning in Large Language Models 

**Authors**: Zijie Chen, Zhenghao Lin, Xiao Liu, Zhenzhong Lan, Yeyun Gong, Peng Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.08321)  

**Abstract**: Solving open-ended science questions remains challenging for large language models, particularly due to inherently unreliable supervision and evaluation. The bottleneck lies in the data construction and reward design for scientific post-training. We develop a large-scale, systematic data processing pipeline that transforms heterogeneous open-source science data into Dr. SCI dataset, which comprises of 1M questions across eight STEM subjects, with explicit verifiable/open-ended splits, scalable difficulty annotation, and fine-grained rubrics that operationalize evaluation for open-ended answers. Building on this dataset, we propose the Dr. SCI post-training pipeline, which redesigns the standard SFT -> RL workflow through three components: (i) Exploration-Expanding SFT, which broadens the model's reasoning pattern coverage prior to RL; (ii) Dynamic Difficulty Curriculum, which adapts training data to the model's evolving scientific capability; and (iii) SciRubric-Guided RL, which enables stable reinforcement learning on open-ended scientific questions via rubric-based evaluation with explicit answer correctness. Qwen3-4B-Base trained using this http URL pipeline achieves 63.2 on GPQA-diamond and 32.4 on GPQA-general, consistently improves over strong post-trained baselines such as o1-mini and GPT-4o, demonstrating substantial gains in scientific reasoning, especially in open-ended settings. 

---
# Letting Tutor Personas "Speak Up" for LLMs: Learning Steering Vectors from Dialogue via Preference Optimization 

**Authors**: Jaewook Lee, Alexander Scarlatos, Simon Woodhead, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2602.07639)  

**Abstract**: With the emergence of large language models (LLMs) as a powerful class of generative artificial intelligence (AI), their use in tutoring has become increasingly prominent. Prior works on LLM-based tutoring typically learn a single tutor policy and do not capture the diversity of tutoring styles. In real-world tutor-student interactions, pedagogical intent is realized through adaptive instructional strategies, with tutors varying the level of scaffolding, instructional directiveness, feedback, and affective support in response to learners' needs. These differences can all impact dialogue dynamics and student engagement. In this paper, we explore how tutor personas embedded in human tutor-student dialogues can be used to guide LLM behavior without relying on explicitly prompted instructions. We modify Bidirectional Preference Optimization (BiPO) to learn a steering vector, an activation-space direction that steers model responses towards certain tutor personas. We find that this steering vector captures tutor-specific variation across dialogue contexts, improving semantic alignment with ground-truth tutor utterances and increasing preference-based evaluations, while largely preserving lexical similarity. Analysis of the learned directional coefficients further reveals interpretable structure across tutors, corresponding to consistent differences in tutoring behavior. These results demonstrate that activation steering offers an effective and interpretable way for controlling tutor-specific variation in LLMs using signals derived directly from human dialogue data. 

---
# Bayesian Preference Learning for Test-Time Steerable Reward Models 

**Authors**: Jiwoo Hong, Shao Tang, Zhipeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08819)  

**Abstract**: Reward models are central to aligning language models with human preferences via reinforcement learning (RL). As RL is increasingly applied to settings such as verifiable rewards and multi-objective alignment, RMs are expected to encode more complex and multifaceted preference distributions. However, classifier RMs remain static once trained, limiting their adaptability at test time. We propose Variational In-Context Reward Modeling (ICRM), a novel Bayesian reward modeling objective that enables test-time steerability via in-context preference demonstrations. ICRM casts reward modeling as amortized variational inference over a latent preference probability under the Bradley-Terry model using a conjugate Beta prior. We show that ICRM adapt to unseen preference distributions at test time for both single and multi-objective settings. With more in-context demonstrations, ICRM gains 34% accuracy on SafeRLHF and 9% accuracy on RM-Bench in the single-objective setting, while widening the Pareto frontier with a 4% gain in hypervolume on helpfulness and refusal benchmarks. We further study the practical applicability of ICRM for RL training, showing that it can effectively encode verifiable rewards by outperforming a conventional RM in math reasoning. Finally, we provide theoretical guarantees that the variational objective admits a global interior optimum with finite confidence, and we analyze how KL regularization mitigates reward over-optimization. 

---
# Beyond Correctness: Learning Robust Reasoning via Transfer 

**Authors**: Hyunseok Lee, Soheil Abbasloo, Jihoon Tack, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2602.08489)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has recently strengthened LLM reasoning, but its focus on final answer correctness leaves a critical gap: it does not ensure the robustness of the reasoning process itself. We adopt a simple philosophical view, robust reasoning should remain useful beyond the mind that produced it, and treat reasoning as a form of meaning transfer that must survive truncation, reinterpretation, and continuation. Building on this principle, we introduce Reinforcement Learning with Transferable Reward (RLTR), which operationalizes robustness via transfer reward that tests whether a partial reasoning prefix from one model can guide a separate model to the correct answer. This encourages LLMs to produce reasoning that is stable, interpretable, and genuinely generalizable. Our approach improves sampling consistency while improving final answer accuracy, and it reaches comparable performance in substantially fewer training steps. For example, on MATH500, RLTR achieves a +3.6%p gain in Maj@64 compared to RLVR and matches RLVR's average accuracy with roughly 2.5x fewer training steps, providing both more reliable reasoning and significantly more sample efficient. 

---
# Learning Self-Correction in Vision-Language Models via Rollout Augmentation 

**Authors**: Yi Ding, Ziliang Qiu, Bolian Li, Ruqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.08503)  

**Abstract**: Self-correction is essential for solving complex reasoning problems in vision-language models (VLMs). However, existing reinforcement learning (RL) methods struggle to learn it, as effective self-correction behaviors emerge only rarely, making learning signals extremely sparse. To address this challenge, we propose correction-specific rollouts (Octopus), an RL rollout augmentation framework that synthesizes dense self-correction examples by recombining existing rollouts. This augmentation simultaneously improves sample efficiency due to rollout reuse and stabilizes RL optimization through balanced supervision. Furthermore, we introduce a response-masking strategy that decouples self-correction from direct reasoning, avoiding signal conflicts and enabling both behaviors to be learned effectively. Building on this, we introduce Octopus-8B, a reasoning VLM with controllable self-correction capability. Across 7 benchmarks, it achieves SoTA performance among open-source VLMs, outperforming the best RLVR baseline by 1.0 score while requiring only $0.72\times$ training time per step. 

---
# Safety Alignment as Continual Learning: Mitigating the Alignment Tax via Orthogonal Gradient Projection 

**Authors**: Guanglong Sun, Siyuan Zhang, Liyuan Wang, Jun Zhu, Hang Su, Yi Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2602.07892)  

**Abstract**: Large Language Models (LLMs) often incur an alignment tax: safety post-training can reduce general utility (e.g., reasoning and coding). We argue that this tax primarily arises from continual-learning-style forgetting in sequential alignment, where distribution shift and conflicting objectives cause safety updates to overwrite pre-trained competencies. Accordingly, we cast safety alignment as a continual learning (CL) problem that must balance plasticity (acquiring safety constraints) and stability (preserving general abilities). We propose Orthogonal Gradient Projection for Safety Alignment (OGPSA), a lightweight method that mitigates interference by constraining each safety update to be orthogonal (in a first-order sense) to a learned subspace capturing general capabilities. Specifically, OGPSA estimates a low-rank capability subspace from gradients on a small reference set and projects the safety gradient onto its orthogonal complement before updating. This produces safety-directed updates that minimally perturb prior knowledge while retaining capacity for alignment. OGPSA is plug-and-play and integrates into standard post-training pipelines without large-scale replay, auxiliary objectives, or retraining. Across Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and sequential SFT$\rightarrow$DPO settings, OGPSA consistently improves the safety--utility Pareto frontier over standard baselines. For instance, on Qwen2.5-7B-Instruct under SFT$\rightarrow$DPO, OGPSA preserves strong safety while recovering general capability, improving SimpleQA from 0.53\% to 3.03\% and IFEval from 51.94\% to 63.96\%. Our source code is available at \href{this https URL}{OGPSA} 

---
