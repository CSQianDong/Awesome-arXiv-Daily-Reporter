# DiSCTT: Consensus-Guided Self-Curriculum for Efficient Test-Time Adaptation in Reasoning 

**Authors**: Mohammad Mahdi Moradi, Sudhir Mudur  

**Link**: [PDF](https://arxiv.org/pdf/2603.05357)  

**Abstract**: Test-time adaptation offers a promising avenue for improving reasoning performance in large language models without additional supervision, but existing approaches often apply a uniform optimization objective across all inputs, leading to inefficient or unstable adaptation on heterogeneous reasoning problems. We propose DiSCTT, a difficulty-aware, consensus-guided self-curriculum framework that dynamically allocates test-time optimization strategies based on instance-level epistemic uncertainty estimated from agreement among sampled reasoning trajectories. Inputs with high consensus are consolidated via supervised fine-tuning using majority-agreed solutions as pseudo-labels, while low-consensus inputs are optimized via reinforcement learning with a consensus-regularized objective that encourages diversity under relevance constraints. Across a broad suite of mathematical and general reasoning benchmarks, DiSCTT consistently outperforms strong test-time adaptation baselines, achieving higher accuracy with reduced variance and substantially lower computation and wall-clock training times. These results demonstrate that explicitly accounting for instance difficulty and uncertainty enables more stable, efficient, and effective test-time adaptation for reasoning models. 

---
# When Weak LLMs Speak with Confidence, Preference Alignment Gets Stronger 

**Authors**: Amirabbas Afzali, Myeongho Jeon, Maria Brbic  

**Link**: [PDF](https://arxiv.org/pdf/2603.04968)  

**Abstract**: Preference alignment is an essential step in adapting large language models (LLMs) to human values, but existing approaches typically depend on costly human annotations or large-scale API-based models. We explore whether a weak LLM can instead act as an effective annotator. We surprisingly find that selecting only a subset of a weak LLM's highly confident samples leads to substantially better performance than using full human annotations. Building on this insight, we propose Confidence-Weighted Preference Optimization (CW-PO), a general framework that re-weights training samples by a weak LLM's confidence and can be applied across different preference optimization objectives. Notably, the model aligned by CW-PO with just 20% of human annotations outperforms the model trained with 100% of annotations under standard DPO. These results suggest that weak LLMs, when paired with confidence weighting, can dramatically reduce the cost of preference alignment while even outperforming methods trained on fully human-labeled data. 

---
# LocalSUG: Geography-Aware LLM for Query Suggestion in Local-Life Services 

**Authors**: Jinwen Chen, Shuai Gong, Shiwen Zhang, Zheng Zhang, Yachao Zhao, Lingxiang Wang, Haibo Zhou, Yuan Zhan, Wei Lin, Hainan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.04946)  

**Abstract**: In local-life service platforms, the query suggestion module plays a crucial role in enhancing user experience by generating candidate queries based on user input prefixes, thus reducing user effort and accelerating search. Traditional multi-stage cascading systems rely heavily on historical top queries, limiting their ability to address long-tail demand. While LLMs offer strong semantic generalization, deploying them in local-life services introduces three key challenges: lack of geographic grounding, exposure bias in preference optimization, and online inference latency. To address these issues, we propose LocalSUG, an LLM-based query suggestion framework tailored for local-life service platforms. First, we introduce a city-aware candidate mining strategy based on term co-occurrence to inject geographic grounding into generation. Second, we propose a beam-search-driven GRPO algorithm that aligns training with inference-time decoding, reducing exposure bias in autoregressive generation. A multi-objective reward mechanism further optimizes both relevance and business-oriented metrics. Finally, we develop quality-aware beam acceleration and vocabulary pruning techniques that significantly reduce online latency while preserving generation quality. Extensive offline evaluations and large-scale online A/B testing demonstrate that LocalSUG improves click-through rate (CTR) by +0.35% and reduces the low/no-result rate by 2.56%, validating its effectiveness in real-world deployment. 

---
# Optimizing Language Models for Crosslingual Knowledge Consistency 

**Authors**: Tianyu Liu, Jirui Qi, Mrinmaya Sachan, Ryan Cotterell, Raquel Fernández, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2603.04678)  

**Abstract**: Large language models are known to often exhibit inconsistent knowledge. This is particularly problematic in multilingual scenarios, where models are likely to be asked similar questions in different languages, and inconsistent responses can undermine their reliability. In this work, we show that this issue can be mitigated using reinforcement learning with a structured reward function, which leads to an optimal policy with consistent crosslingual responses. We introduce Direct Consistency Optimization (DCO), a DPO-inspired method that requires no explicit reward model and is derived directly from the LLM itself. Comprehensive experiments show that DCO significantly improves crosslingual consistency across diverse LLMs and outperforms existing methods when training with samples of multiple languages, while complementing DPO when gold labels are available. Extra experiments demonstrate the effectiveness of DCO in bilingual settings, significant out-of-domain generalizability, and controllable alignment via direction hyperparameters. Taken together, these results establish DCO as a robust and efficient solution for improving knowledge consistency across languages in multilingual LLMs. All code, training scripts, and evaluation benchmarks are released at this https URL. 

---
# What Is Missing: Interpretable Ratings for Large Language Model Outputs 

**Authors**: Nicholas Stranges, Yimin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.04429)  

**Abstract**: Current Large Language Model (LLM) preference learning methods such as Proximal Policy Optimization and Direct Preference Optimization learn from direct rankings or numerical ratings of model outputs, these rankings are subjective, and a single numerical rating chosen directly by a judge is a poor proxy for the quality of natural language, we introduce the What Is Missing (WIM) rating system to produce rankings from natural-language feedback, WIM integrates into existing training pipelines, can be combined with other rating techniques, and can be used as input to any preference learning method without changing the learning algorithm, to compute a WIM rating, a human or LLM judge writes feedback describing what the model output is missing, we embed the output and the feedback with a sentence embedding model and compute the cosine similarity between the resulting vectors, we empirically observe that, compared to discrete numerical ratings, WIM yields fewer ties and larger rating deltas, which improves the availability of a learning signal in pairwise preference data, we use interpretable in the following limited sense: for each scalar rating, we can inspect the judge's missing-information text that produced it, enabling qualitative debugging of the preference labels. 

---
# The Thinking Boundary: Quantifying Reasoning Suitability of Multimodal Tasks via Dual Tuning 

**Authors**: Ruobing Zheng, Tianqi Li, Jianing Li, Qingpei Guo, Yi Yuan, Jingdong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.04415)  

**Abstract**: While reasoning-enhanced Large Language Models (LLMs) have demonstrated remarkable advances in complex tasks such as mathematics and coding, their effectiveness across universal multimodal scenarios remains uncertain. The trend of releasing parallel "Instruct" and "Thinking" models by leading developers serves merely as a resource-intensive workaround, stemming from the lack of a criterion for determining when reasoning is truly beneficial. In this paper, we propose Dual Tuning, a framework designed to assess whether reasoning yields positive gains for target tasks under given base models and datasets. By jointly fine-tuning on paired Chain-of-Thought (CoT) and Direct-Answer (DA) data under controlled prompts, we systematically quantify and compare the gains of both training modes using the proposed metrics, and establish the "Thinking Boundary" to evaluate the suitability of reasoning training across diverse multimodal tasks, including spatial, mathematical, and multi-disciplinary domains. We further explore the impact of reinforcement training and thinking patterns on reasoning suitability, and validate whether the "Thinking Boundary" can guide data refinement. Our findings challenge the "reasoning-for-all" paradigm, providing practical guidance for identifying appropriate data and training strategies, and motivating the development of resource-efficient, adaptive auto-think systems. 

---
# Knowledge Divergence and the Value of Debate for Scalable Oversight 

**Authors**: Robin Young  

**Link**: [PDF](https://arxiv.org/pdf/2603.05293)  

**Abstract**: AI safety via debate and reinforcement learning from AI feedback (RLAIF) are both proposed methods for scalable oversight of advanced AI systems, yet no formal framework relates them or characterizes when debate offers an advantage. We analyze this by parameterizing debate's value through the geometry of knowledge divergence between debating models. Using principal angles between models' representation subspaces, we prove that the debate advantage admits an exact closed form. When models share identical training corpora, debate reduces to RLAIF-like where a single-agent method recovers the same optimum. When models possess divergent knowledge, debate advantage scales with a phase transition from quadratic regime (debate offers negligible benefit) to linear regime (debate is essential). We classify three regimes of knowledge divergence (shared, one-sided, and compositional) and provide existence results showing that debate can achieve outcomes inaccessible to either model alone, alongside a negative result showing that sufficiently strong adversarial incentives cause coordination failure in the compositional regime, with a sharp threshold separating effective from ineffective debate. We offer the first formal connection between debate and RLAIF, a geometric foundation for understanding when adversarial oversight protocols are justified, and connection to the problem of eliciting latent knowledge across models with complementary information. 

---
# CTRL-RAG: Contrastive Likelihood Reward Based Reinforcement Learning for Context-Faithful RAG Models 

**Authors**: Zhehao Tan, Yihan Jiao, Dan Yang, Junjie Wang, Duolin Sun, Jie Feng, Xidong Wang, Lei Liu, Yue Shen, Jian Wang, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2603.04406)  

**Abstract**: With the growing use of Retrieval-Augmented Generation (RAG), training large language models (LLMs) for context-sensitive reasoning and faithfulness is increasingly important. Existing RAG-oriented reinforcement learning (RL) methods rely on external rewards that often fail to evaluate document faithfulness, and may misjudge similar answers in open-domain settings. In addition, there is no RAG-based selfreward mechanism. Moreover, although such a mechanism could in principle estimate answer confidence given documents, the absence of objective feedback in a self-judgment can cause hallucination accumulation and eventual model collapse. To tackle these issues, we propose a novel "internal-external" hybrid reward framework centered on a Contrastive Likelihood Reward (CLR). CLR directly optimizes the log-likelihood gap between responses conditioned on prompts with and without supporting evidence. This encourages the model to extract relevant evidence and increases its confidence when grounded in a specific context. Experiments show that our method (used alone or combined with external correctness rewards) achieves strong performance on singlehop, multi-hop, vertical-domain, and faithfulness benchmarks. Our training code and models are coming soon. 

---
# Why Is RLHF Alignment Shallow? A Gradient Analysis 

**Authors**: Robin Young  

**Link**: [PDF](https://arxiv.org/pdf/2603.04851)  

**Abstract**: Why is safety alignment in LLMs shallow? We prove that gradient-based alignment inherently concentrates on positions where harm is decided and vanishes beyond. Using a martingale decomposition of sequence-level harm, we derive an exact characterization of alignment gradients. The gradient at position $t$ equals the covariance between the conditional expected harm and the score function. This implies that positions beyond the harm horizon where the output's harmfulness is already determined receive zero gradient signal during training. This explains empirical observations that KL divergence between aligned and base models concentrates on early tokens. Consequently, standard alignment objectives cannot produce deep alignment, regardless of optimization quality. We introduce the concept of harm information $I_t$, which quantifies each position's influence on harm, and prove that equilibrium KL divergence tracks this quantity. Finally, we derive an objective based on recovery penalties that creates gradient signal at all positions, providing theoretical grounding for empirically successful data augmentation techniques. 

---
# Breaking Contextual Inertia: Reinforcement Learning with Single-Turn Anchors for Stable Multi-Turn Interaction 

**Authors**: Xingwu Chen, Zhanqiu Zhang, Yiwen Guo, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.04783)  

**Abstract**: While LLMs demonstrate strong reasoning capabilities when provided with full information in a single turn, they exhibit substantial vulnerability in multi-turn interactions. Specifically, when information is revealed incrementally or requires updates, models frequently fail to integrate new constraints, leading to a collapse in performance compared to their single-turn baselines. We term the root cause as \emph{Contextual Inertia}: a phenomenon where models rigidly adhere to previous reasoning traces. Even when users explicitly provide corrections or new data in later turns, the model ignores them, preferring to maintain consistency with its previous (incorrect) reasoning path. To address this, we introduce \textbf{R}einforcement \textbf{L}earning with \textbf{S}ingle-\textbf{T}urn \textbf{A}nchors (\textbf{RLSTA}), a generalizable training approach designed to stabilize multi-turn interaction across diverse scenarios and domains. RLSTA leverages the model's superior single-turn capabilities as stable internal anchors to provide reward signals. By aligning multi-turn responses with these anchors, RLSTA empowers models to break contextual inertia and self-calibrate their reasoning based on the latest information. Experiments show that RLSTA significantly outperforms standard fine-tuning and abstention-based methods. Notably, our method exhibits strong cross-domain generalization (e.g., math to code) and proves effective even without external verifiers, highlighting its potential for general-domain applications. 

---
# K-Gen: A Multimodal Language-Conditioned Approach for Interpretable Keypoint-Guided Trajectory Generation 

**Authors**: Mingxuan Mu, Guo Yang, Lei Chen, Ping Wu, Jianxun Cui  

**Link**: [PDF](https://arxiv.org/pdf/2603.04868)  

**Abstract**: Generating realistic and diverse trajectories is a critical challenge in autonomous driving simulation. While Large Language Models (LLMs) show promise, existing methods often rely on structured data like vectorized maps, which fail to capture the rich, unstructured visual context of a scene. To address this, we propose K-Gen, an interpretable keypoint-guided multimodal framework that leverages Multimodal Large Language Models (MLLMs) to unify rasterized BEV map inputs with textual scene descriptions. Instead of directly predicting full trajectories, K-Gen generates interpretable keypoints along with reasoning that reflects agent intentions, which are subsequently refined into accurate trajectories by a refinement module. To further enhance keypoint generation, we apply T-DAPO, a trajectory-aware reinforcement fine-tuning algorithm. Experiments on WOMD and nuPlan demonstrate that K-Gen outperforms existing baselines, highlighting the effectiveness of combining multimodal reasoning with keypoint-guided trajectory generation. 

---
# Causally Robust Reward Learning from Reason-Augmented Preference Feedback 

**Authors**: Minjune Hwang, Yigit Korkmaz, Daniel Seita, Erdem Bıyık  

**Link**: [PDF](https://arxiv.org/pdf/2603.04861)  

**Abstract**: Preference-based reward learning is widely used for shaping agent behavior to match a user's preference, yet its sparse binary feedback makes it especially vulnerable to causal confusion. The learned reward often latches onto spurious features that merely co-occur with preferred trajectories during training, collapsing when those correlations disappear or reverse at test time. We introduce ReCouPLe, a lightweight framework that uses natural language rationales to provide the missing causal signal. Each rationale is treated as a guiding projection axis in an embedding space, training the model to score trajectories based on features aligned with that axis while de-emphasizing context that is unrelated to the stated reason. Because the same rationales (e.g., "avoids collisions", "completes the task faster") can appear across multiple tasks, ReCouPLe naturally reuses the same causal direction whenever tasks share semantics, and transfers preference knowledge to novel tasks without extra data or language-model fine-tuning. Our learned reward model can ground preferences on the articulated reason, aligning better with user intent and generalizing beyond spurious features. ReCouPLe outperforms baselines by up to 1.5x in reward accuracy under distribution shifts, and 2x in downstream policy performance in novel tasks. We have released our code at this https URL 

---
# VISA: Value Injection via Shielded Adaptation for Personalized LLM Alignment 

**Authors**: Jiawei Chen, Tianzhuo Yang, Guoxi Zhang, Jiaming Ji, Yaodong Yang, Juntao Dai  

**Link**: [PDF](https://arxiv.org/pdf/2603.04822)  

**Abstract**: Aligning Large Language Models (LLMs) with nuanced human values remains a critical challenge, as existing methods like Reinforcement Learning from Human Feedback (RLHF) often handle only coarse-grained attributes. In practice, fine-tuning LLMs on task-specific datasets to optimize value alignment inevitably incurs an alignment tax: the model's pre-calibrated value system drifts significantly due to latent bias absorption from training data, while the fine-tuning process also causes severe hallucinations and semantic information loss in generated responses. To address this, we propose VISA (Value Injection via Shielded Adaptation), a closed-loop framework designed to navigate this trade-off. VISA's architecture features a high-precision value detector, a semantic-to-value translator, and a core value-rewriter. The value-rewriter is trained via Group Relative Policy Optimization (GRPO) with a composite reward function that simultaneously optimizes for fine-grained value precision, and the preservation of semantic integrity. By learning an optimal policy to balance these competing objectives, VISA effectively mitigates the alignment tax while staying loyal to the original knowledge. Our experiments demonstrate that this approach enables precise control over a model's value expression while maintaining its factual consistency and general capabilities, significantly outperforming both standard fine-tuning methods and prompting-based baselines, including GPT-4o. 

---
# When Agents Persuade: Propaganda Generation and Mitigation in LLMs 

**Authors**: Julia Jose, Ritik Roongta, Rachel Greenstadt  

**Link**: [PDF](https://arxiv.org/pdf/2603.04636)  

**Abstract**: Despite their wide-ranging benefits, LLM-based agents deployed in open environments can be exploited to produce manipulative material. In this study, we task LLMs with propaganda objectives and analyze their outputs using two domain-specific models: one that classifies text as propaganda or non-propaganda, and another that detects rhetorical techniques of propaganda (e.g., loaded language, appeals to fear, flag-waving, name-calling). Our findings show that, when prompted, LLMs exhibit propagandistic behaviors and use a variety of rhetorical techniques in doing so. We also explore mitigation via Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and ORPO (Odds Ratio Preference Optimization). We find that fine-tuning significantly reduces their tendency to generate such content, with ORPO proving most effective. 

---
# 3D-RFT: Reinforcement Fine-Tuning for Video-based 3D Scene Understanding 

**Authors**: Xiongkun Linghu, Jiangyong Huang, Baoxiong Jia, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.04976)  

**Abstract**: Reinforcement Learning with Verifiable Rewards ( RLVR ) has emerged as a transformative paradigm for enhancing the reasoning capabilities of Large Language Models ( LLMs), yet its potential in 3D scene understanding remains under-explored. Existing approaches largely rely on Supervised Fine-Tuning ( SFT), where the token-level cross-entropy loss acts as an indirect proxy for optimization, leading to a misalignment between training objectives and task performances. To bridge this gap, we present Reinforcement Fine-Tuning for Video-based 3D Scene Understanding (3D-RFT ), the first framework to extend RLVR to video-based 3D perception and reasoning. 3D-RFT shifts the paradigm by directly optimizing the model towards evaluation metrics. 3D-RFT first activates 3D-aware Multi-modal Large Language Models ( MLLM s) via SFT, followed by reinforcement fine-tuning using Group Relative Policy Optimization ( GRPO) with strictly verifiable reward functions. We design task-specific reward functions directly from metrics like 3D IoU and F1-Score to provide more effective signals to guide model training. Extensive experiments demonstrate that 3D-RFT-4B achieves state-of-the-art performance on various video-based 3D scene understanding tasks. Notably, 3D-RFT-4B significantly outperforms larger models (e.g., VG LLM-8B) on 3D video detection, 3D visual grounding, and spatial reasoning benchmarks. We further reveal good properties of 3D-RFT such as robust efficacy, and valuable insights into training strategies and data impact. We hope 3D-RFT can serve as a robust and promising paradigm for future development of 3D scene understanding. 

---
