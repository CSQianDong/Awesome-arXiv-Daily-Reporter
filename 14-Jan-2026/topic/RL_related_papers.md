# PersonaDual: Balancing Personalization and Objectivity via Adaptive Reasoning 

**Authors**: Xiaoyou Liu, Xinyi Mou, Shengbin Yue, Liang Wang, Yuqing Wang, Qiexiang Wang, Tianrui Qin, Wangchunshu Zhou, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.08679)  

**Abstract**: As users increasingly expect LLMs to align with their preferences, personalized information becomes valuable. However, personalized information can be a double-edged sword: it can improve interaction but may compromise objectivity and factual correctness, especially when it is misaligned with the question. To alleviate this problem, we propose PersonaDual, a framework that supports both general-purpose objective reasoning and personalized reasoning in a single model, and adaptively switches modes based on context. PersonaDual is first trained with SFT to learn two reasoning patterns, and then further optimized via reinforcement learning with our proposed DualGRPO to improve mode selection. Experiments on objective and personalized benchmarks show that PersonaDual preserves the benefits of personalization while reducing interference, achieving near interference-free performance and better leveraging helpful personalized signals to improve objective problem-solving. 

---
# Prism: Towards Lowering User Cognitive Load in LLMs via Complex Intent Understanding 

**Authors**: Zenghua Liao, Jinzhi Liao, Xiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.08653)  

**Abstract**: Large Language Models are rapidly emerging as web-native interfaces to social platforms. On the social web, users frequently have ambiguous and dynamic goals, making complex intent understanding-rather than single-turn execution-the cornerstone of effective human-LLM collaboration. Existing approaches attempt to clarify user intents through sequential or parallel questioning, yet they fall short of addressing the core challenge: modeling the logical dependencies among clarification questions. Inspired by the Cognitive Load Theory, we propose Prism, a novel framework for complex intent understanding that enables logically coherent and efficient intent clarification. Prism comprises four tailored modules: a complex intent decomposition module, which decomposes user intents into smaller, well-structured elements and identifies logical dependencies among them; a logical clarification generation module, which organizes clarification questions based on these dependencies to ensure coherent, low-friction interactions; an intent-aware reward module, which evaluates the quality of clarification trajectories via an intent-aware reward function and leverages Monte Carlo Sample to simulate user-LLM interactions for large-scale,high-quality training data generation; and a self-evolved intent tuning module, which iteratively refines the LLM's logical clarification capability through data-driven feedback and optimization. Prism consistently outperforms existing approaches across clarification interactions, intent execution, and cognitive load benchmarks. It achieves stateof-the-art logical consistency, reduces logical conflicts to 11.5%, increases user satisfaction by 14.4%, and decreases task completion time by 34.8%. All data and code are released. 

---
# Reasoning over Precedents Alongside Statutes: Case-Augmented Deliberative Alignment for LLM Safety 

**Authors**: Can Jin, Rui Wu, Tong Che, Qixin Zhang, Hongwu Peng, Jiahui Zhao, Zhenting Wang, Wenqi Wei, Ligong Han, Zhao Zhang, Yuan Cao, Ruixiang Tang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2601.08000)  

**Abstract**: Ensuring that Large Language Models (LLMs) adhere to safety principles without refusing benign requests remains a significant challenge. While OpenAI introduces deliberative alignment (DA) to enhance the safety of its o-series models through reasoning over detailed ``code-like'' safety rules, the effectiveness of this approach in open-source LLMs, which typically lack advanced reasoning capabilities, is understudied. In this work, we systematically evaluate the impact of explicitly specifying extensive safety codes versus demonstrating them through illustrative cases. We find that referencing explicit codes inconsistently improves harmlessness and systematically degrades helpfulness, whereas training on case-augmented simple codes yields more robust and generalized safety behaviors. By guiding LLMs with case-augmented reasoning instead of extensive code-like safety rules, we avoid rigid adherence to narrowly enumerated rules and enable broader adaptability. Building on these insights, we propose CADA, a case-augmented deliberative alignment method for LLMs utilizing reinforcement learning on self-generated safety reasoning chains. CADA effectively enhances harmlessness, improves robustness against attacks, and reduces over-refusal while preserving utility across diverse benchmarks, offering a practical alternative to rule-only DA for improving safety while maintaining helpfulness. 

---
# Asymptotic Universal Alignment: A New Alignment Framework via Test-Time Scaling 

**Authors**: Yang Cai, Weiqiang Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.08777)  

**Abstract**: Aligning large language models (LLMs) to serve users with heterogeneous and potentially conflicting preferences is a central challenge for personalized and trustworthy AI. We formalize an ideal notion of universal alignment through test-time scaling: for each prompt, the model produces $k\ge 1$ candidate responses and a user selects their preferred one. We introduce $(k,f(k))$-robust alignment, which requires the $k$-output model to have win rate $f(k)$ against any other single-output model, and asymptotic universal alignment (U-alignment), which requires $f(k)\to 1$ as $k\to\infty$. Our main result characterizes the optimal convergence rate: there exists a family of single-output policies whose $k$-sample product policies achieve U-alignment at rate $f(k)=\frac{k}{k+1}$, and no method can achieve a faster rate in general.
We show that popular post-training methods, including Nash learning from human feedback (NLHF), can fundamentally underutilize the benefits of test-time scaling. Even though NLHF is optimal for $k=1$, sampling from the resulting (often deterministic) policy cannot guarantee win rates above $\tfrac{1}{2}$ except for an arbitrarily small slack. This stems from a lack of output diversity: existing alignment methods can collapse to a single majority-preferred response, making additional samples redundant. In contrast, our approach preserves output diversity and achieves the optimal test-time scaling rate. In particular, we propose a family of symmetric multi-player alignment games and prove that any symmetric Nash equilibrium policy of the $(k+1)$-player alignment game achieves the optimal $(k,\frac{k}{k+1})$-robust alignment. Finally, we provide theoretical convergence guarantees for self-play learning dynamics in these games and extend the framework to opponents that also generate multiple responses. 

---
# Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge 

**Authors**: Yao Tang, Li Dong, Yaru Hao, Qingxiu Dong, Furu Wei, Jiatao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08808)  

**Abstract**: Large language models often solve complex reasoning tasks more effectively with Chain-of-Thought (CoT), but at the cost of long, low-bandwidth token sequences. Humans, by contrast, often reason softly by maintaining a distribution over plausible next steps. Motivated by this, we propose Multiplex Thinking, a stochastic soft reasoning mechanism that, at each thinking step, samples K candidate tokens and aggregates their embeddings into a single continuous multiplex token. This preserves the vocabulary embedding prior and the sampling dynamics of standard discrete generation, while inducing a tractable probability distribution over multiplex rollouts. Consequently, multiplex trajectories can be directly optimized with on-policy reinforcement learning (RL). Importantly, Multiplex Thinking is self-adaptive: when the model is confident, the multiplex token is nearly discrete and behaves like standard CoT; when it is uncertain, it compactly represents multiple plausible next steps without increasing sequence length. Across challenging math reasoning benchmarks, Multiplex Thinking consistently outperforms strong discrete CoT and RL baselines from Pass@1 through Pass@1024, while producing shorter sequences. The code and checkpoints are available at this https URL. 

---
# Owen-Shapley Policy Optimization (OSPO): A Principled RL Algorithm for Generative Search LLMs 

**Authors**: Abhijnan Nath, Alireza Bagheri Garakani, Tianchen Zhou, Fan Yang, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2601.08403)  

**Abstract**: Large language models are increasingly trained via reinforcement learning for personalized recommendation tasks, but standard methods like GRPO rely on sparse, sequence-level rewards that create a credit assignment gap, obscuring which tokens drive success. This gap is especially problematic when models must infer latent user intent from under-specified language without ground truth labels, a reasoning pattern rarely seen during pretraining. We introduce Owen-Shapley Policy Optimization (OSPO), a framework that redistributes sequence-level advantages based on tokens' marginal contributions to outcomes. Unlike value-model-based methods requiring additional computation, OSPO employs potential-based reward shaping via Shapley-Owen attributions to assign segment-level credit while preserving the optimal policy, learning directly from task feedback without parametric value models. By forming coalitions of semantically coherent units (phrases describing product attributes or sentences capturing preferences), OSPO identifies which response parts drive performance. Experiments on Amazon ESCI and H&M Fashion datasets show consistent gains over baselines, with notable test-time robustness to out-of-distribution retrievers unseen during training. 

---
# JudgeRLVR: Judge First, Generate Second for Efficient Reasoning 

**Authors**: Jiangshan Duo, Hanyu Li, Hailin Zhang, Yudong Wang, Sujian Li, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.08468)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a standard paradigm for reasoning in Large Language Models. However, optimizing solely for final-answer correctness often drives models into aimless, verbose exploration, where they rely on exhaustive trial-and-error tactics rather than structured planning to reach solutions. While heuristic constraints like length penalties can reduce verbosity, they often truncate essential reasoning steps, creating a difficult trade-off between efficiency and verification. In this paper, we argue that discriminative capability is a prerequisite for efficient generation: by learning to distinguish valid solutions, a model can internalize a guidance signal that prunes the search space. We propose JudgeRLVR, a two-stage judge-then-generate paradigm. In the first stage, we train the model to judge solution responses with verifiable answers. In the second stage, we fine-tune the same model with vanilla generating RLVR initialized from the judge. Compared to Vanilla RLVR using the same math-domain training data, JudgeRLVR achieves a better quality--efficiency trade-off for Qwen3-30B-A3B: on in-domain math, it delivers about +3.7 points average accuracy gain with -42\% average generation length; on out-of-domain benchmarks, it delivers about +4.5 points average accuracy improvement, demonstrating enhanced generalization. 

---
# ORBIT: On-policy Exploration-Exploitation for Controllable Multi-Budget Reasoning 

**Authors**: Kun Liang, Clive Bai, Xin Xu, Chenming Tang, Sanwoo Lee, Weijie Liu, Saiyong Yang, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08310)  

**Abstract**: Recent Large Reasoning Models (LRMs) achieve strong performance by leveraging long-form Chain-of-Thought (CoT) reasoning, but uniformly applying overlong reasoning at inference time incurs substantial and often unnecessary computational cost. To address this, prior work explores various strategies to infer an appropriate reasoning budget from the input. However, such approaches are unreliable in the worst case, as estimating the minimal required reasoning effort is fundamentally difficult, and they implicitly fix the trade-off between reasoning cost and accuracy during training, limiting flexibility under varying deployment scenarios. Motivated by these limitations, we propose ORBIT, a controllable multi-budget reasoning framework with well-separated reasoning modes triggered by input. ORBIT employs multi-stage reinforcement learning to discover Pareto-optimal reasoning behaviors at each effort, followed by on-policy distillation to fuse these behaviors into a single unified model. Experiments show that ORBIT achieves (1) controllable reasoning behavior over multiple modes, (2) competitive reasoning density within each mode, and (3) integration of these frontier policies into a single unified student model while preserving clear mode separation and high per-mode performance. 

---
# Silence the Judge: Reinforcement Learning with Self-Verifier via Latent Geometric Clustering 

**Authors**: Nonghai Zhang, Weitao Ma, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Jingwen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.08427)  

**Abstract**: Group Relative Policy Optimization (GRPO) significantly enhances the reasoning performance of Large Language Models (LLMs). However, this success heavily relies on expensive external verifiers or human rules. Such dependency not only leads to significant computational costs and training latency, but also yields sparse rewards that hinder optimization efficiency. To address these challenges, we propose Latent-GRPO, a framework that derives intrinsic rewards directly from latent space geometry. Crucially, our empirical analysis reveals a compelling geometric property: terminal token representations of correct reasoning trajectories form dense clusters with high intra-class similarity, whereas incorrect trajectories remain scattered as outliers. In light of this discovery, we introduce the Iterative Robust Centroid Estimation (IRCE) algorithm, which generates dense, continuous rewards by mitigating magnitude fluctuations via spherical projection and estimating a robust ``truth centroid'' through iterative aggregation. Experimental results on multiple datasets show that our method maintains model performance while achieving a training speedup of over 2x compared to baselines. Furthermore, extensive results demonstrate strong generalization ability and robustness. The code will be released soon. 

---
# Discovery and Reinforcement of Tool-Integrated Reasoning Chains via Rollout Trees 

**Authors**: Kun Li, Zenan Xu, Junan Li, Zengrui Jin, Jinghao Deng, Zexuan Qiu, Bo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.08274)  

**Abstract**: Tool-Integrated Reasoning has emerged as a key paradigm to augment Large Language Models (LLMs) with computational capabilities, yet integrating tool-use into long Chain-of-Thought (long CoT) remains underexplored, largely due to the scarcity of training data and the challenge of integrating tool-use without compromising the model's intrinsic long-chain reasoning. In this paper, we introduce DART (Discovery And Reinforcement of Tool-Integrated Reasoning Chains via Rollout Trees), a reinforcement learning framework that enables spontaneous tool-use during long CoT reasoning without human annotation. DART operates by constructing dynamic rollout trees during training to discover valid tool-use opportunities, branching out at promising positions to explore diverse tool-integrated trajectories. Subsequently, a tree-based process advantage estimation identifies and credits specific sub-trajectories where tool invocation positively contributes to the solution, effectively reinforcing these beneficial behaviors. Extensive experiments on challenging benchmarks like AIME and GPQA-Diamond demonstrate that DART significantly outperforms existing methods, successfully harmonizing tool execution with long CoT reasoning. 

---
# AdaJudge: Adaptive Multi-Perspective Judging for Reward Modeling 

**Authors**: Yongliang Miao, Yangyang Liang, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2601.08097)  

**Abstract**: Reward modeling is essential for aligning large language models with human preferences, yet predominant architectures rely on a static pooling strategy to condense sequences into scalar scores. This paradigm, however, suffers from two key limitations: a static inductive bias that misaligns with task-dependent preference signals, and a representational mismatch, as the backbone is optimized for generation rather than fine-grained discrimination. To address this, we propose AdaJudge, a unified framework that jointly adapts representation and aggregation. AdaJudge first refines backbone representations into a discrimination-oriented space via gated refinement blocks. It then replaces the static readout with an adaptive multi-view pooling module that dynamically routes and combines evidence. Extensive experiments on RM-Bench and JudgeBench show that AdaJudge outperforms strong off-the-shelf reward models and traditional pooling baselines. 

---
# A Human-Centric Pipeline for Aligning Large Language Models with Chinese Medical Ethics 

**Authors**: Haoan Jin, Han Ying, Jiacheng Ji, Hanhui Xu, Mengyue Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07954)  

**Abstract**: Recent advances in large language models have enabled their application to a range of healthcare tasks. However, aligning LLMs with the nuanced demands of medical ethics, especially under complex real world scenarios, remains underexplored. In this work, we present MedES, a dynamic, scenario-centric benchmark specifically constructed from 260 authoritative Chinese medical, ethical, and legal sources to reflect the challenges in clinical decision-making. To facilitate model alignment, we introduce a guardian-in-the-loop framework that leverages a dedicated automated evaluator (trained on expert-labeled data and achieving over 97% accuracy within our domain) to generate targeted prompts and provide structured ethical feedback. Using this pipeline, we align a 7B-parameter LLM through supervised fine-tuning and domain-specific preference optimization. Experimental results, conducted entirely within the Chinese medical ethics context, demonstrate that our aligned model outperforms notably larger baselines on core ethical tasks, with observed improvements in both quality and composite evaluation metrics. Our work offers a practical and adaptable framework for aligning LLMs with medical ethics in the Chinese healthcare domain, and suggests that similar alignment pipelines may be instantiated in other legal and cultural environments through modular replacement of the underlying normative corpus. 

---
# Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving in LLMs 

**Authors**: Zhiyuan Hu, Yucheng Wang, Yufei He, Jiaying Wu, Yilun Zhao, See-Kiong Ng, Cynthia Breazeal, Anh Tuan Luu, Hae Won Park, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2601.08763)  

**Abstract**: Reinforcement learning (RL) has become a central paradigm for post-training large language models (LLMs), particularly for complex reasoning tasks, yet it often suffers from exploration collapse: policies prematurely concentrate on a small set of dominant reasoning patterns, improving pass@1 while limiting rollout-level diversity and gains in pass@k. We argue that this failure stems from regularizing local token behavior rather than diversity over sets of solutions. To address this, we propose Uniqueness-Aware Reinforcement Learning, a rollout-level objective that explicitly rewards correct solutions that exhibit rare high-level strategies. Our method uses an LLM-based judge to cluster rollouts for the same problem according to their high-level solution strategies, ignoring superficial variations, and reweights policy advantages inversely with cluster size. As a result, correct but novel strategies receive higher rewards than redundant ones. Across mathematics, physics, and medical reasoning benchmarks, our approach consistently improves pass@$k$ across large sampling budgets and increases the area under the pass@$k$ curve (AUC@$K$) without sacrificing pass@1, while sustaining exploration and uncovering more diverse solution strategies at scale. 

---
