# Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with constraints 

**Authors**: Zhenyun Yin, Shujie Wang, Xuhong Wang, Xingjun Ma, Yinchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16727)  

**Abstract**: Improving the reliability of large language models (LLMs) is critical for deploying them in real-world scenarios. In this paper, we propose \textbf{Deliberative Searcher}, the first framework to integrate certainty calibration with retrieval-based search for open-domain question answering. The agent performs multi-step reflection and verification over Wikipedia data and is trained with a reinforcement learning algorithm that optimizes for accuracy under a soft reliability constraint. Empirical results show that proposed method improves alignment between model confidence and correctness, leading to more trustworthy outputs. This paper will be continuously updated. 

---
# Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty 

**Authors**: Mehul Damani, Isha Puri, Stewart Slocum, Idan Shenfeld, Leshem Choshen, Yoon Kim, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2507.16806)  

**Abstract**: When language models (LMs) are trained via reinforcement learning (RL) to generate natural language "reasoning chains", their performance improves on a variety of difficult question answering tasks. Today, almost all successful applications of RL for reasoning use binary reward functions that evaluate the correctness of LM outputs. Because such reward functions do not penalize guessing or low-confidence outputs, they often have the unintended side-effect of degrading calibration and increasing the rate at which LMs generate incorrect responses (or "hallucinate") in other problem domains. This paper describes RLCR (Reinforcement Learning with Calibration Rewards), an approach to training reasoning models that jointly improves accuracy and calibrated confidence estimation. During RLCR, LMs generate both predictions and numerical confidence estimates after reasoning. They are trained to optimize a reward function that augments a binary correctness score with a Brier score -- a scoring rule for confidence estimates that incentivizes calibrated prediction. We first prove that this reward function (or any analogous reward function that uses a bounded, proper scoring rule) yields models whose predictions are both accurate and well-calibrated. We next show that across diverse datasets, RLCR substantially improves calibration with no loss in accuracy, on both in-domain and out-of-domain evaluations -- outperforming both ordinary RL training and classifiers trained to assign post-hoc confidence scores. While ordinary RL hurts calibration, RLCR improves it. Finally, we demonstrate that verbalized confidence can be leveraged at test time to improve accuracy and calibration via confidence-weighted scaling methods. Our results show that explicitly optimizing for calibration can produce more generally reliable reasoning models. 

---
# ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning 

**Authors**: Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, Yu-Chiang Frank Wang, Fu-En Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16815)  

**Abstract**: Vision-language-action (VLA) reasoning tasks require agents to interpret multimodal instructions, perform long-horizon planning, and act adaptively in dynamic environments. Existing approaches typically train VLA models in an end-to-end fashion, directly mapping inputs to actions without explicit reasoning, which hinders their ability to plan over multiple steps or adapt to complex task variations. In this paper, we propose ThinkAct, a dual-system framework that bridges high-level reasoning with low-level action execution via reinforced visual latent planning. ThinkAct trains a multimodal LLM to generate embodied reasoning plans guided by reinforcing action-aligned visual rewards based on goal completion and trajectory consistency. These reasoning plans are compressed into a visual plan latent that conditions a downstream action model for robust action execution on target environments. Extensive experiments on embodied reasoning and robot manipulation benchmarks demonstrate that ThinkAct enables few-shot adaptation, long-horizon planning, and self-correction behaviors in complex embodied AI tasks. 

---
# Self-Contradiction as Self-Improvement: Mitigating the Generation-Understanding Gap in MLLMs 

**Authors**: Yujin Han, Hao Chen, Andi Han, Zhiheng Wang, Xinyu Lin, Yingya Zhang, Shiwei Zhang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2507.16663)  

**Abstract**: Despite efforts to unify multimodal generation and understanding tasks in a single model, we show these MLLMs exhibit self-contradiction where generation produces images deemed misaligned with input prompts based on the model's own understanding. We define a Nonunified score that quantifies such self-contradiction. Our empirical results reveal that the self-contradiction mainly arises from weak generation that fails to align with prompts, rather than misunderstanding. This capability asymmetry indicates the potential of leveraging self-contradiction for self-improvement, where the stronger model understanding guides the weaker generation to mitigate the generation-understanding gap. Applying standard post-training methods (e.g., SFT, DPO) with such internal supervision successfully improves both generation and unification. We discover a co-improvement effect on both generation and understanding when only fine-tuning the generation branch, a phenomenon known in pre-training but underexplored in post-training. Our analysis shows improvements stem from better detection of false positives that are previously incorrectly identified as prompt-aligned. Theoretically, we show the aligned training dynamics between generation and understanding allow reduced prompt-misaligned generations to also improve mismatch detection in the understanding branch. Additionally, the framework reveals a potential risk of co-degradation under poor supervision-an overlooked phenomenon that is empirically validated in our experiments. Notably, we find intrinsic metrics like Nonunified score cannot distinguish co-degradation from co-improvement, which highlights the necessity of data quality check. Finally, we propose a curriculum-based strategy based on our findings that gradually introduces harder samples as the model improves, leading to better unification and improved MLLM generation and understanding. 

---
# Efficient RL for optimizing conversation level outcomes with an LLM-based tutor 

**Authors**: Hyunji Nam, Omer Gottesman, Amy Zhang, Dean Foster, Emma Brunskill, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2507.16252)  

**Abstract**: Large language models (LLMs) built on existing reinforcement learning with human feedback (RLHF) frameworks typically optimize responses based on immediate turn-level human preferences. However, this approach falls short in multi-turn dialogue settings, such as online math tutoring. We propose a method to enhance LLM-based tutors by representing the dialogue history with a lower-dimensional latent state representation of a student and optimizing a long-term policy to determine high-level actions based on the latent state. The goal is to better align the tutor's behavior with the long-term objective of guiding the student towards solving a target math problem on their own. Our model is lightweight, requiring less computational resources than prior work of training the tutor policy end-to-end to directly output the tutor's next utterance. Our experiment results demonstrate that these modifications lead to improved long-term outcomes compared to prompting in LLM-simulated tutoring tasks. 

---
# METER: Multi-modal Evidence-based Thinking and Explainable Reasoning -- Algorithm and Benchmark 

**Authors**: Xu Yang, Qi Zhang, Shuming Jiang, Yaowen Xu, Zhaofan Zou, Hao Sun, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.16206)  

**Abstract**: With the rapid advancement of generative AI, synthetic content across images, videos, and audio has become increasingly realistic, amplifying the risk of misinformation. Existing detection approaches predominantly focus on binary classification while lacking detailed and interpretable explanations of forgeries, which limits their applicability in safety-critical scenarios. Moreover, current methods often treat each modality separately, without a unified benchmark for cross-modal forgery detection and interpretation. To address these challenges, we introduce METER, a unified, multi-modal benchmark for interpretable forgery detection spanning images, videos, audio, and audio-visual content. Our dataset comprises four tracks, each requiring not only real-vs-fake classification but also evidence-chain-based explanations, including spatio-temporal localization, textual rationales, and forgery type tracing. Compared to prior benchmarks, METER offers broader modality coverage and richer interpretability metrics such as spatial/temporal IoU, multi-class tracing, and evidence consistency. We further propose a human-aligned, three-stage Chain-of-Thought (CoT) training strategy combining SFT, DPO, and a novel GRPO stage that integrates a human-aligned evaluator with CoT reasoning. We hope METER will serve as a standardized foundation for advancing generalizable and interpretable forgery detection in the era of generative media. 

---
# Dual Turing Test: A Framework for Detecting and Mitigating Undetectable AI 

**Authors**: Alberto Messina  

**Link**: [PDF](https://arxiv.org/pdf/2507.15907)  

**Abstract**: In this short note, we propose a unified framework that bridges three areas: (1) a flipped perspective on the Turing Test, the "dual Turing test", in which a human judge's goal is to identify an AI rather than reward a machine for deception; (2) a formal adversarial classification game with explicit quality constraints and worst-case guarantees; and (3) a reinforcement learning (RL) alignment pipeline that uses an undetectability detector and a set of quality related components in its reward model. We review historical precedents, from inverted and meta-Turing variants to modern supervised reverse-Turing classifiers, and highlight the novelty of combining quality thresholds, phased difficulty levels, and minimax bounds. We then formalize the dual test: define the judge's task over N independent rounds with fresh prompts drawn from a prompt space Q, introduce a quality function Q and parameters tau and delta, and cast the interaction as a two-player zero-sum game over the adversary's feasible strategy set M. Next, we map this minimax game onto an RL-HF style alignment loop, in which an undetectability detector D provides negative reward for stealthy outputs, balanced by a quality proxy that preserves fluency. Throughout, we include detailed explanations of each component notation, the meaning of inner minimization over sequences, phased tests, and iterative adversarial training and conclude with a suggestion for a couple of immediate actions. 

---
# Towards Reliable, Uncertainty-Aware Alignment 

**Authors**: Debangshu Banerjee, Kintan Saha, Aditya Gopalan  

**Link**: [PDF](https://arxiv.org/pdf/2507.15906)  

**Abstract**: Alignment of large language models (LLMs) typically involves training a reward model on preference data, followed by policy optimization with respect to the reward model. However, optimizing policies with respect to a single reward model estimate can render it vulnerable to inaccuracies in the reward model. We empirically study the variability of reward model training on open-source benchmarks. We observe that independently trained reward models on the same preference dataset can exhibit substantial disagreement, highlighting the instability of current alignment strategies. Employing a theoretical model, we demonstrate that variability in reward model estimation can cause overfitting, leading to the risk of performance degradation. To mitigate this risk, we propose a variance-aware policy optimization framework for preference-based alignment. The key ingredient of the framework is a new policy regularizer that incorporates reward model variance estimates. We show that variance-aware policy optimization provably reduces the risk of outputting a worse policy than the default. Experiments across diverse LLM and reward model configurations confirm that our approach yields more stable and robust alignment than the standard (variance-unaware) pipeline. 

---
# Scaling Linear Attention with Sparse State Expansion 

**Authors**: Yuqi Pan, Yongqi An, Zheng Li, Yuhong Chou, Ruijie Zhu, Xiaohui Wang, Mingxuan Wang, Jinqiao Wang, Guoqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.16577)  

**Abstract**: The Transformer architecture, despite its widespread success, struggles with long-context scenarios due to quadratic computation and linear memory growth. While various linear attention variants mitigate these efficiency constraints by compressing context into fixed-size states, they often degrade performance in tasks such as in-context retrieval and reasoning. To address this limitation and achieve more effective context compression, we propose two key innovations. First, we introduce a row-sparse update formulation for linear attention by conceptualizing state updating as information classification. This enables sparse state updates via softmax-based top-$k$ hard classification, thereby extending receptive fields and reducing inter-class interference. Second, we present Sparse State Expansion (SSE) within the sparse framework, which expands the contextual state into multiple partitions, effectively decoupling parameter size from state capacity while maintaining the sparse classification paradigm. Our design, supported by efficient parallelized implementations, yields effective classification and discriminative state representations. We extensively validate SSE in both pure linear and hybrid (SSE-H) architectures across language modeling, in-context retrieval, and mathematical reasoning benchmarks. SSE demonstrates strong retrieval performance and scales favorably with state size. Moreover, after reinforcement learning (RL) training, our 2B SSE-H model achieves state-of-the-art mathematical reasoning performance among small reasoning models, scoring 64.7 on AIME24 and 51.3 on AIME25, significantly outperforming similarly sized open-source Transformers. These results highlight SSE as a promising and efficient architecture for long-context modeling. 

---
# C2-Evo: Co-Evolving Multimodal Data and Model for Self-Improving Reasoning 

**Authors**: Xiuwei Chen, Wentao Hu, Hanhui Li, Jun Zhou, Zisheng Chen, Meng Cao, Yihan Zeng, Kui Zhang, Yu-Jie Yuan, Jianhua Han, Hang Xu, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16518)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have shown impressive reasoning capabilities. However, further enhancing existing MLLMs necessitates high-quality vision-language datasets with carefully curated task complexities, which are both costly and challenging to scale. Although recent self-improving models that iteratively refine themselves offer a feasible solution, they still suffer from two core challenges: (i) most existing methods augment visual or textual data separately, resulting in discrepancies in data complexity (e.g., over-simplified diagrams paired with redundant textual descriptions); and (ii) the evolution of data and models is also separated, leading to scenarios where models are exposed to tasks with mismatched difficulty levels. To address these issues, we propose C2-Evo, an automatic, closed-loop self-improving framework that jointly evolves both training data and model capabilities. Specifically, given a base dataset and a base model, C2-Evo enhances them by a cross-modal data evolution loop and a data-model evolution loop. The former loop expands the base dataset by generating complex multimodal problems that combine structured textual sub-problems with iteratively specified geometric diagrams, while the latter loop adaptively selects the generated problems based on the performance of the base model, to conduct supervised fine-tuning and reinforcement learning alternately. Consequently, our method continuously refines its model and training data, and consistently obtains considerable performance gains across multiple mathematical reasoning benchmarks. Our code, models, and datasets will be released. 

---
