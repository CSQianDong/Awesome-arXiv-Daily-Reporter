# Embedding Domain Knowledge for Large Language Models via Reinforcement Learning from Augmented Generation 

**Authors**: Chaojun Nie, Jun Zhou, Guanxiang Wang, Shisong Wud, Zichen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20162)  

**Abstract**: Large language models (LLMs) often exhibit limited performance on domain-specific tasks due to the natural disproportionate representation of specialized information in their training data and the static nature of these datasets. Knowledge scarcity and temporal lag create knowledge gaps for domain applications. While post-training on domain datasets can embed knowledge into models, existing approaches have some limitations. Continual Pre-Training (CPT) treats all tokens in domain documents with equal importance, failing to prioritize critical knowledge points, while supervised fine-tuning (SFT) with question-answer pairs struggles to develop the coherent knowledge structures necessary for complex reasoning tasks. To address these challenges, we propose Reinforcement Learning from Augmented Generation (RLAG). Our approach iteratively cycles between sampling generations and optimizing the model through calculated rewards, effectively embedding critical and contextually coherent domain knowledge. We select generated outputs with the highest log probabilities as the sampling result, then compute three tailored reward metrics to guide the optimization process. To comprehensively evaluate domain expertise, we assess answer accuracy and the rationality of explanations generated for correctly answered questions. Experimental results across medical, legal, astronomy, and current events datasets demonstrate that our proposed method significantly outperforms baseline approaches. Our code and data are open sourced at this https URL. 

---
# Language Models that Think, Chat Better 

**Authors**: Adithya Bhaskar, Xi Ye, Danqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.20357)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) improves language model reasoning by using rule-based rewards in verifiable domains such as mathematics and code. However, RLVR leads to limited generalization for open-ended tasks -- such as writing outline essays or making meal plans -- where humans reason routinely. This paper shows that the RLVR paradigm is effective beyond verifiable domains, and introduces **RL** with **M**odel-rewarded **T**hinking (**RLMT**) for general-purpose chat capabilities. Using diverse real-world prompts, RLMT requires LMs to generate long CoT reasoning before response, and optimizes them with online RL against a preference-based reward model used in RLHF. Across 40 training runs on Llama-3.1-8B and Qwen-2.5-7B (both base and instruct) and multiple optimization algorithms (DPO, PPO, and GRPO), RLMT consistently outperforms standard RLHF pipelines. This includes substantial gains of 3-7 points on three chat benchmarks (AlpacaEval2, WildBench, and ArenaHardV2), along with 1-3 point improvements on other tasks like creative writing and general knowledge. Our best 8B model surpasses GPT-4o in chat and creative writing and rivals Claude-3.7-Sonnet (Thinking). RLMT can also be applied directly to base models without an SFT stage, akin to R1-Zero training. Remarkably, with only 7K prompts, Llama-3.1-8B base trained with our RLMT recipe outperforms Llama-3.1-8B-Instruct post-trained with a complex multi-staged pipeline with 25M+ examples. We close with qualitative and quantitative analyses of how trained models plan their responses. Our results rethink the post-training pipeline and call upon future work to understand and employ thinking more broadly. 

---
# Future Policy Aware Preference Learning for Mathematical Reasoning 

**Authors**: Minjae Oh, Yunho Choi, Dongmin Choi, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2509.19893)  

**Abstract**: Preference learning methods such as Direct Preference Optimization (DPO) have become standard for Large Language Model (LLM) post-training, yet they are often ineffective for mathematical reasoning. A key challenge is the large token overlap between preferred and dispreferred trajectories; lowering the probability of dispreferred trajectories also reduces the probability of shared useful tokens, leading to over-penalization and overall performance collapse. As a mitigation, existing algorithms include the probability of a trajectory under the current policy as a regularization term, which decreases the effect of the gradient when the probability is low. However, by the time this effect takes hold, useful tokens may have already been over-penalized as the model has begun to degrade. To address this, we propose Future Policy Aware (FPA) preference learning, which replaces the current policy with a future policy in the regularization term. This future policy is estimated via lightweight, logit-space extrapolation from a reference model toward the current model. FPA enables safer training by preemptively regularizing potentially problematic gradients. We apply FPA to DPO, RPO, and SimPER and evaluate them on the MATH and GSM8K benchmarks. FPA yields consistent performance gains, with the largest improvements observed with SimPER, achieving gains of up to 5.75%. We demonstrate that FPA provides proactive regularization while preserving the probability of shared, useful mathematical tokens, and enables longer, degradation-free training with negligible computational overhead. We will release our code publicly upon publication. 

---
# bi-GRPO: Bidirectional Optimization for Jailbreak Backdoor Injection on LLMs 

**Authors**: Wence Ji, Jiancan Wu, Aiying Li, Shuyi Zhang, Junkang Wu, An Zhang, Xiang Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2509.19775)  

**Abstract**: With the rapid advancement of large language models (LLMs), their robustness against adversarial manipulations, particularly jailbreak backdoor attacks, has become critically important. Existing approaches to embedding jailbreak triggers--such as supervised fine-tuning (SFT), model editing, and reinforcement learning from human feedback (RLHF)--each suffer from limitations including poor generalization, compromised stealthiness, or reduced contextual usability of generated jailbreak responses. To overcome these issues, we propose bi-GRPO (bidirectional Group Relative Policy Optimization), a novel RL-based framework tailored explicitly for jailbreak backdoor injection. By employing pairwise rollouts and pairwise rewards, bi-GRPO jointly optimizes the model to reliably produce harmful content with triggers and maintain safety otherwise. Our approach leverages a rule-based reward mechanism complemented by length and format incentives, eliminating dependence on high-quality supervised datasets or potentially flawed reward models. Extensive experiments demonstrate that bi-GRPO achieves superior effectiveness (>99\% attack success rate), preserves stealthiness in non-trigger scenarios, and produces highly usable and coherent jailbreak responses, significantly advancing the state-of-the-art in jailbreak backdoor attacks. 

---
# Meow: End-to-End Outline Writing for Automatic Academic Survey 

**Authors**: Zhaoyu Ma, Yuan Shan, Jiahao Zhao, Nan Xu, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19370)  

**Abstract**: As academic paper publication numbers grow exponentially, conducting in-depth surveys with LLMs automatically has become an inevitable trend. Outline writing, which aims to systematically organize related works, is critical for automated survey generation. Yet existing automatic survey methods treat outline writing as mere workflow steps in the overall pipeline. Such template-based workflows produce outlines that lack in-depth understanding of the survey topic and fine-grained styles. To address these limitations, we propose Meow, the first metadata-driven outline writing framework that produces organized and faithful outlines efficiently. Specifically, we first formulate outline writing as an end-to-end task that generates hierarchical structured outlines from paper metadata. We then curate a high-quality dataset of surveys from arXiv, bioRxiv, and medRxiv, and establish systematic evaluation metrics for outline quality assessment. Finally, we employ a two-stage training approach combining supervised fine-tuning and reinforcement learning. Our 8B reasoning model demonstrates strong performance with high structural fidelity and stylistic coherence. 

---
# Pluralistic Off-policy Evaluation and Alignment 

**Authors**: Chengkai Huang, Junda Wu, Zhouhang Xie, Yu Xia, Rui Wang, Tong Yu, Subrata Mitra, Julian McAuley, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19333)  

**Abstract**: Personalized preference alignment for LLMs with diverse human preferences requires evaluation and alignment methods that capture pluralism. Most existing preference alignment datasets are logged under policies that differ substantially from the evaluated LLMs, and existing off-policy estimators focus solely on overall utility while ignoring preference pluralism. Extending Off-Policy Evaluation (OPE) to pluralistic preference alignment, therefore, remains an open question. Thus, we propose the Pluralistic Off-Policy Evaluation (POPE), the first framework for offline pluralistic preference evaluation and alignment in LLMs. POPE includes a unified reward function that combines (1) a collaborative utility component derived from human preference signals (e.g., upvotes or relevance scores) and (2) a diversity component inspired by entropy-based coverage measures, together reflecting pluralistic alignment. Furthermore, to estimate this reward from logged interactions, we derive decomposable inverse propensity scoring (IPS) estimators that separately evaluate relevance and diversity. Theoretically, we prove that our decomposed IPS estimators establish a lower bound on their variance. With the off-policy evaluated value function, we can directly enable off-policy optimization to further enhance pluralistic alignment. Empirical results demonstrate that POPE efficiently enhances pluralistic response generation and maintains the models' general capabilities on downstream tasks 

---
# Advancing Speech Summarization in Multi-modal LLMs with Reinforcement Learning 

**Authors**: Shaoshi Ling, Gang Liu, Guoli Ye, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19631)  

**Abstract**: Speech summarization is a critical component of spoken content understanding, particularly in the era of rapidly growing spoken and audiovisual data. Recent advances in multi-modal large language models (MLLMs), leveraging the power of LLMs, enable generating textual summaries directly from speech without intermediate transcriptions, while supporting controllable styles and zero-shot generalization. However, open-source MLLMs continue to lag behind the state-of-the-art text-based LLMs, limiting their practical deployment for speech summarization. In this work, we present a novel multi-stage reinforcement learning training framework to enhance the speech summarization capabilities in MLLMs. Our model delivers substantial improvements over strong baselines, outperforms much larger MLLMs, and significantly narrows the gap with state-of-the-art text-based LLMs. 

---
# VCRL: Variance-based Curriculum Reinforcement Learning for Large Language Models 

**Authors**: Guochao Jiang, Wenfeng Feng, Guofeng Quan, Chuzhan Hao, Yuewei Zhang, Guohua Liu, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19803)  

**Abstract**: Policy-based reinforcement learning currently plays an important role in improving LLMs on mathematical reasoning tasks. However, existing rollout-based reinforcement learning methods (GRPO, DAPO, GSPO, etc.) fail to explicitly consider LLMs' learning ability for samples of different difficulty levels, which is contrary to the human cognitive process of mathematical reasoning tasks from easy to difficult. Intuitively, we find that the variance of the rollout group's reward in RLVR partly reflects the difficulty of the current sample for LLMs. Samples that are too easy or too difficult have a lower variance, while samples with moderate difficulty have a higher variance. Based on this, we propose VCRL, a curriculum reinforcement learning framework that dynamically controls the difficulty of training samples based on the variance of group rewards. Experiments on five mathematical benchmarks and two models reveal the advantages of VCRL over the current LLM RL baselines. 

---
# Failure Modes of Maximum Entropy RLHF 

**Authors**: Ömer Veysel Çağatan, Barış Akgün  

**Link**: [PDF](https://arxiv.org/pdf/2509.20265)  

**Abstract**: In this paper, we show that Simple Preference Optimization (SimPO) can be derived as Maximum Entropy Reinforcement Learning with length-normalized temperature, providing a theoretical foundation for this reference-free method. Motivated by SimPO's strong performance in offline preference optimization, we investigate whether Maximum Entropy RL can achieve similar results in online RLHF settings. Our experiments find that Maximum Entropy RL consistently exhibits overoptimization and unstable KL dynamics, even at very low learning rates. Unlike KL-constrained methods that maintain stable training, entropy regularization fails to prevent reward hacking and appears to correlate with overoptimization. Lastly, we discuss possible explanations for why SimPO succeeds in offline settings while Maximum Entropy RL struggles in online scenarios. Our findings suggest that reference-free approaches may face distinct challenges when applied to online or offline preference learning. 

---
# Calibrated Reasoning: An Explanatory Verifier for Dynamic and Efficient Problem-Solving 

**Authors**: Anisha Garg, Engin Tekin, Yash More, David Bick, Nishit Neema, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2509.19681)  

**Abstract**: Advanced test-time computing strategies are essential for scaling reasoning models, but their effectiveness is capped by the models' poor self-evaluation. We propose a pairwise Explanatory Verifier, trained via reinforcement learning (GRPO), that produces calibrated confidence scores and associated natural language reasoning for generated solutions. Our verifier improves the accuracy and efficiency of test-time strategies like best-of-n and self-reflection. Crucially, it excels at identifying challenging failure modes, such as when both candidate solutions are identically incorrect, succeeding where standard methods like majority voting fail. 

---
# PEPS: Quantum-Inspired Reinforcement Learning for Coherent Reasoning Traces in LLMs 

**Authors**: Venkat Margapuri, Garik Kazanjian, Naren Kosaraju  

**Link**: [PDF](https://arxiv.org/pdf/2509.20105)  

**Abstract**: Large Language Models (LLMs) often struggle with maintaining coherent multi-step reasoning traces, particularly in tasks that require a structured logical flow. This work introduces a quantum-inspired approach to address the challenge by incorporating a fidelity-based reward derived from Projected Entangled Pair States (PEPS) into Proximal Policy Optimization. Unlike prior approaches that use direct supervision or contrastive objectives, the proposed method guides learning through structural consistency, offering a novel approach to enforce global coherence in generated reasoning traces. The proposed framework is evaluated using multiple coherence-determining metrics on diverse datasets such as GSM8K, StrategyQA, and EntailmentBank spanning arithmetic, intuitive, and entailment-based reasoning. Results show that the proposed quantum-inspired approach offers significant improvements over supervised, contrastive, and pretrained baseline approaches, highlighting the effectiveness of quantum-inspired fidelity as a foundation to improve reasoning trace coherence in LLMs. 

---
