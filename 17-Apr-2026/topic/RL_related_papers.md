# IG-Search: Step-Level Information Gain Rewards for Search-Augmented Reasoning 

**Authors**: Zihan Liang, Yufei Ma, Ben Chen, Zhipeng Qian, Huangyu Dai, Lingtao Mao, Xuxin Zhang, Chenyi Lei, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2604.15148)  

**Abstract**: Reinforcement learning has emerged as an effective paradigm for training large language models to perform search-augmented reasoning. However, existing approaches rely on trajectory-level rewards that cannot distinguish precise search queries from vague or redundant ones within a rollout group, and collapse to a near-zero gradient signal whenever every sampled trajectory fails. In this paper, we propose IG-Search, a reinforcement learning framework that introduces a step-level reward based on Information Gain (IG). For each search step, IG measures how much the retrieved documents improve the model's confidence in the gold answer relative to a counterfactual baseline of random documents, thereby reflecting the effectiveness of the underlying search query. This signal is fed back to the corresponding search-query tokens via per-token advantage modulation in GRPO, enabling fine-grained, step-level credit assignment within a rollout. Unlike prior step-level methods that require either externally annotated intermediate supervision or shared environment states across trajectories, IG-Search derives its signals from the policy's own generation probabilities, requiring no intermediate annotations beyond standard question-answer pairs. Experiments on seven single-hop and multi-hop QA benchmarks demonstrate that IG-Search achieves an average EM of 0.430 with Qwen2.5-3B, outperforming the strongest trajectory-level baseline (MR-Search) by 1.6 points and the step-level method GiGPO by 0.9 points on average across benchmarks, with particularly pronounced gains on multi-hop reasoning tasks. Despite introducing a dense step-level signal, IG-Search adds only ~6.4% to per-step training wall-clock time over the trajectory-level baseline and leaves inference latency unchanged, while still providing a meaningful gradient signal even when every sampled trajectory answers incorrectly. 

---
# Dual-Axis Generative Reward Model Toward Semantic and Turn-taking Robustness in Interactive Spoken Dialogue Models 

**Authors**: Yifu Chen, Shengpeng Ji, Zhengqing Liu, Qian Chen, Wen Wang, Ziqing Wang, Yangzhuo Li, Tianle Liang, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14920)  

**Abstract**: Achieving seamless, human-like interaction remains a key challenge for full-duplex spoken dialogue models (SDMs). Reinforcement learning (RL) has substantially enhanced text- and vision-language models, while well-designed reward signals are crucial for the performance of RL. We consider RL a promising strategy to address the key challenge for SDMs. However, a fundamental barrier persists: prevailing automated metrics for assessing interaction quality rely on superficial proxies, such as behavioral statistics or timing-prediction accuracy, failing to provide reliable reward signals for RL. On the other hand, human evaluations, despite their richness, remain costly, inconsistent, and difficult to scale. We tackle this critical barrier by proposing a Dual-Axis Generative Reward Model, which is trained to understand complex interaction dynamics using a detailed taxonomy and an annotated dataset, produces a single score and, crucially, provides separate evaluations for semantic quality and interaction timing. Such dual outputs furnish precise diagnostic feedback for SDMs and deliver a dependable, instructive reward signal suitable for online reinforcement learning. Our model achieves state-of-the-art performance on interaction-quality assessment across a wide spectrum of datasets, spanning synthetic dialogues and complex real-world interactions. 

---
# AgentGA: Evolving Code Solutions in Agent-Seed Space 

**Authors**: David Y.Y. Tan, Kellie Chin, Jingxian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14655)  

**Abstract**: We present AgentGA, a framework that evolves autonomous code-generation runs by optimizing the agent seed: the task prompt plus optional parent archives that initialize a fresh workspace. The outer loop searches over these reusable starting conditions rather than editing code directly. Each generation launches a fresh autonomous run from a reset workspace, while selected parent archives provide inherited artifacts that descendants can inspect and reuse. AgentGA couples a population-level genetic algorithm with long-horizon agents; selection uses deterministic 1:1 elite tournaments and operator allocation is adapted online with a modified Hedge controller. We instantiate the approach for tabular AutoML on the 16-competition Weco-Kaggle Lite benchmark. On the 10 benchmark runs reported here, AgentGA averages 74.52% Exceeds % of Human versus 54.15% for AIDE. Across 1135 parent-child comparisons, descendants given parent archives outperform runs started from scratch, indicating that inherited artifacts improve later autonomous runs. These findings support agent-seed optimization as a practical design point for autonomous code-search systems. 

---
# Targeted Exploration via Unified Entropy Control for Reinforcement Learning 

**Authors**: Chen Wang, Lai Wei, Yanzhi Zhang, Chenyang Shao, Zedong Dan, Weiran Huang, Ge Lan, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14646)  

**Abstract**: Recent advances in reinforcement learning (RL) have improved the reasoning capabilities of large language models (LLMs) and vision-language models (VLMs). However, the widely used Group Relative Policy Optimization (GRPO) consistently suffers from entropy collapse, causing the policy to converge prematurely and lose diversity. Existing exploration methods introduce additional bias or variance during exploration, making it difficult to maintain optimization stability. We propose Unified Entropy Control for Reinforcement Learning (UEC-RL), a framework that provides targeted mechanisms for exploration and stabilization. UEC-RL activates more exploration on difficult prompts to search for potential and valuable reasoning trajectories. In parallel, a stabilizer prevents entropy from growing uncontrollably, thereby keeping training stable as the model consolidates reliable behaviors. Together, these components expand the search space when needed while maintaining robust optimization throughout training. Experiments on both LLM and VLM reasoning tasks show consistent gains over RL baselines on both Pass@1 and Pass@$k$. On Geometry3K, UEC-RL achieves a 37.9\% relative improvement over GRPO, indicating that it sustains effective exploration without compromising convergence and underscoring UEC-RL as a key for scaling RL-based reasoning in large models. Our code is available at this https URL. 

---
# Mind DeepResearch Technical Report 

**Authors**: MindDR Team, Li Auto Inc  

**Link**: [PDF](https://arxiv.org/pdf/2604.14518)  

**Abstract**: We present \textbf{Mind DeepResearch (MindDR)}, an efficient multi-agent deep research framework that achieves leading performance with only \textasciitilde30B-parameter models through a meticulously designed data synthesis and multi-stage training pipeline. The core innovation of MindDR lies in a collaborative three-agent architecture (Planning Agent, DeepSearch Agent, and Report Agent) and a four-stage agent-specialized training pipeline comprising SFT cold-start, Search-RL, Report-RL and preference alignment. With this regime, MindDR demonstrates competitive performance even with \textasciitilde30B-scale models. Specifically, MindDR achieves 45.7\% on BrowseComp-ZH, 42.8\% on BrowseComp, 46.5\% on WideSearch, 75.0\% on xbench-DS, and 52.5 on DeepResearch Bench, outperforming comparable-scale open-source agent systems and rivaling larger-scale models. MindDR has been deployed as an online product in Li Auto. Furthermore, we introduce \textbf{MindDR Bench}, a curated benchmark of 500 real-world Chinese queries from our internal product user interactions, evaluated through a comprehensive multi-dimensional rubric system rather than relying on a single RACE metric. On MindDR Bench, MindDR achieves a state-of-the-art score of 51.8. 

---
# MARS$^2$: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation 

**Authors**: Pengfei Li, Shijie Wang, Fangyuan Li, Yikun Fu, Kaifeng Liu, Kaiyan Zhang, Dazhi Zhang, Yuqiang Li, Biqing Qi, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.14564)  

**Abstract**: Reinforcement learning (RL) paradigms have demonstrated strong performance on reasoning-intensive tasks such as code generation. However, limited trajectory diversity often leads to diminishing returns, which constrains the achievable performance ceiling. Search-enhanced RL alleviates this issue by introducing structured exploration, which remains constrained by the single-agent policy priors. Meanwhile, leveraging multiple interacting policies can acquire more diverse exploratory signals, but existing approaches are typically decoupled from structured search. We propose \textbf{MARS$^2$} (Multi-Agent Reinforced Tree-Search Scaling), a unified RL framework in which multiple independently-optimized agents collaborate within a shared tree-structured search environment. MARS$^2$ models the search tree as a learnable multi-agent interaction environment, enabling heterogeneous agents to collaboratively generate and refine candidate solutions within a shared search topology. To support effective learning, we introduce a path-level group advantage formulation based on tree-consistent reward shaping, which facilitates effective credit assignment across complex search trajectories. Experiments on code generation benchmarks show that MARS$^2$ consistently improves performance across diverse model combinations and training settings, demonstrating the effectiveness of coupling multi-agent collaboration with tree search for enhancing reinforcement learning. Our code is publicly available at this https URL. 

---
# GFT: From Imitation to Reward Fine-Tuning with Unbiased Group Advantages and Dynamic Coefficient Rectification 

**Authors**: Wangjie Gan, Miao Pan, Linbo Xi, Wenqi Zhang, Jintao Chen, Jianwei Yin, Xuhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14258)  

**Abstract**: Large language models are typically post-trained using supervised fine-tuning (SFT) and reinforcement learning (RL), yet effectively unifying efficient knowledge injection with robust generalization remains challenging. In this work, we provide a training-dynamics analysis showing that SFT can be interpreted as a special case of policy gradient optimization with an extremely sparse implicit reward and unstable inverse-probability weighting, which together lead to single-path dependency, entropy collapse, and gradient explosion. Motivated by this diagnosis, we propose Group Fine-Tuning (GFT), a unified post-training framework that addresses these intrinsic limitations through two mechanisms: Group Advantage Learning, which constructs diverse response groups and derives normalized contrastive supervision to alleviate reward sparsity, and Dynamic Coefficient Rectification, which adaptively bounds inverse-probability weights to stabilize optimization while preserving efficient knowledge injection. Experiments demonstrate that GFT consistently surpasses SFT-based methods and yields policies that integrate more smoothly with subsequent RL training. 

---
# LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking 

**Authors**: Lukas Helff, Quentin Delfosse, David Steinmann, Ruben Härle, Hikaru Shindo, Patrick Schramowski, Wolfgang Stammer, Kristian Kersting, Felix Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2604.15149)  

**Abstract**: As reinforcement Learning with Verifiable Rewards (RLVR) has become the dominant paradigm for scaling reasoning capabilities in LLMs, a new failure mode emerges: LLMs gaming verifiers. We study this phenomenon on inductive reasoning tasks, where models must induce and output logical rules. We find that RLVR-trained models systematically abandon rule induction. Instead of learning generalizable patterns (e.g., ``trains carrying red cars go east''), they enumerate instance-level labels, producing outputs that pass verifiers without capturing the relational patterns required by the task. We show that this behavior is not a failure of understanding but a form of reward hacking: imperfect verifiers that check only extensional correctness admit false positives. To detect such shortcuts, we introduce Isomorphic Perturbation Testing (IPT), which evaluates a single model output under both extensional and isomorphic verification, where the latter enforces invariance under logically isomorphic tasks. While genuine rule induction remains invariant, shortcut strategies fail. We find that shortcut behavior is specific to RLVR-trained reasoning models (e.g., GPT-5, Olmo3) and absent in non-RLVR models (e.g., GPT-4o, GPT-4.5, Ministral). Moreover, shortcut prevalence increases with task complexity and inference-time compute. In controlled training experiments, extensional verification directly induces shortcut strategies, while isomorphic verification eliminates them. These results show that RLVR can incentivize reward hacking not only through overt manipulation but also by exploiting what the verifier fails to enforce. 

---
# RaTA-Tool: Retrieval-based Tool Selection with Multimodal Large Language Models 

**Authors**: Gabriele Mattioli, Evelyn Turri, Sara Sarto, Lorenzo Baraldi, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2604.14951)  

**Abstract**: Tool learning with foundation models aims to endow AI systems with the ability to invoke external resources -- such as APIs, computational utilities, and specialized models -- to solve complex tasks beyond the reach of standalone language generation. While recent advances in Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have expanded their reasoning and perception capabilities, existing tool-use methods are predominantly limited to text-only inputs and closed-world settings. Consequently, they struggle to interpret multimodal user instructions and cannot generalize to tools unseen during training. In this work, we introduce RaTA-Tool, a novel framework for open-world multimodal tool selection. Rather than learning direct mappings from user queries to fixed tool identifiers, our approach enables an MLLM to convert a multimodal query into a structured task description and subsequently retrieve the most appropriate tool by matching this representation against semantically rich, machine-readable tool descriptions. This retrieval-based formulation naturally supports extensibility to new tools without retraining. To further improve alignment between task descriptions and tool selection, we incorporate a preference-based optimization stage using Direct Preference Optimization (DPO). To support research in this setting, we also introduce the first dataset for open-world multimodal tool use, featuring standardized tool descriptions derived from Hugging Face model cards. Extensive experiments demonstrate that our approach significantly improves tool-selection performance, particularly in open-world, multimodal scenarios. 

---
# Beyond Importance Sampling: Rejection-Gated Policy Optimization 

**Authors**: Ziwu Sun, Zhen Gao, Jiyong Zhang, Jiaheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.14895)  

**Abstract**: We propose a new perspective on policy optimization: rather than reweighting all samples by their importance ratios, an optimizer should select which samples are trustworthy enough to drive a policy update. Building on this view, we introduce Rejection-Gated Policy Optimization (RGPO), which replaces the importance sampling ratio r_theta = pi_theta / pi_old with a smooth, differentiable acceptance gate alpha_theta(s, a) = g(r_theta(s, a)) in the range [0, 1]. Unlike prior work that applies rejection sampling as a data-level heuristic before training, RGPO elevates rejection to an optimization principle: the gate participates directly in gradient computation and is implicitly updated alongside the policy. RGPO provides a unified framework: the policy gradients of TRPO, PPO, and REINFORCE all correspond to specific choices of the effective gradient weight w(r) = g'(r) * r. We prove that RGPO guarantees finite, bounded gradient variance even when importance sampling ratios are heavy-tailed (where IS variance diverges). We further show that RGPO incurs only a bounded, controllable bias and provides an approximate monotonic policy improvement guarantee analogous to TRPO. RGPO matches PPO in computational cost, requires no second-order optimization, and extends naturally to RLHF-style preference alignment. In online preference fine-tuning of Qwen2.5-1.5B-Instruct on Anthropic HH-RLHF (n = 3 seeds), RGPO uses a dual-ratio gate that anchors learning to both the previous policy and the reference model, achieving a Pareto-dominant outcome: the highest reward among online RL methods (+14.8% vs. PPO-RLHF) and the lowest KL divergence to the reference model (-16.0% vs. PPO-RLHF, -53.1% vs. GRPO). 

---
# UniDoc-RL: Coarse-to-Fine Visual RAG with Hierarchical Actions and Dense Rewards 

**Authors**: Jun Wang, Shuo Tan, Zelong Sun, Tiancheng Gu, Yongle Zhao, Ziyong Feng, Kaicheng Yang, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14967)  

**Abstract**: Retrieval-Augmented Generation (RAG) extends Large Vision-Language Models (LVLMs) with external visual knowledge. However, existing visual RAG systems typically rely on generic retrieval signals that overlook the fine-grained visual semantics essential for complex reasoning. To address this limitation, we propose UniDoc-RL, a unified reinforcement learning framework in which an LVLM agent jointly performs retrieval, reranking, active visual perception, and reasoning. UniDoc-RL formulates visual information acquisition as a sequential decision-making problem with a hierarchical action space. Specifically, it progressively refines visual evidence from coarse-grained document retrieval to fine-grained image selection and active region cropping, allowing the model to suppress irrelevant content and attend to information-dense regions. For effective end-to-end training, we introduce a dense multi-reward scheme that provides task-aware supervision for each action. Based on Group Relative Policy Optimization (GRPO), UniDoc-RL aligns agent behavior with multiple objectives without relying on a separate value network. To support this training paradigm, we curate a comprehensive dataset of high-quality reasoning trajectories with fine-grained action annotations. Experiments on three benchmarks demonstrate that UniDoc-RL consistently surpasses state-of-the-art baselines, yielding up to 17.7% gains over prior RL-based methods. 

---
# Asking What Matters: Reward-Driven Clarification for Software Engineering Tasks 

**Authors**: Sanidhya Vijayvargiya, Vijay Viswanathan, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2604.14624)  

**Abstract**: Humans often specify tasks incompletely, so assistants must know when and how to ask clarifying questions. However, effective clarification remains challenging in software engineering tasks as not all missing information is equally valuable, and questions must target information users can realistically provide. We study clarification in real software engineering tasks by quantifying which types of information most affect task success and which questions elicit useful responses from simulated users. Using Shapley attribution and distributional comparisons, we identify two key properties of effective clarification: task relevance (which information predicts success) and user answerability (what users can realistically provide). We operationalize these properties as multi-stage reinforcement learning rewards to train CLARITI, an 8B-parameter clarification module, that matches GPT-5's resolution rate on underspecified issues while generating 41% fewer questions. Our results suggest that grounding reward design in empirical analysis of information impact and user answerability improves clarification efficiency. 

---
# DharmaOCR: Specialized Small Language Models for Structured OCR that outperform Open-Source and Commercial Baselines 

**Authors**: Gabriel Pimenta de Freitas Cardoso, Caio Lucas da Silva Chacon, Jonas Felipe da Fonseca Oliveira, Paulo Henrique de Medeiros Araujo  

**Link**: [PDF](https://arxiv.org/pdf/2604.14314)  

**Abstract**: This manuscript introduces DharmaOCR Full and Lite, a pair of specialized small language models (SSLMs) for structured OCR that jointly optimize transcription quality, generation stability, and inference cost. It also presents DharmaOCR-Benchmark, a benchmark that covers printed, handwritten, and legal/administrative documents, and proposes a unified evaluation protocol that measures fidelity and structure while explicitly tracking text degeneration as a first-class benchmark metric (alongside unit cost). Beyond reporting degeneration rates, the manuscript empirically shows degeneration is not merely a quality failure, since it materially worsens production performance by increasing response time, reducing throughput, and inflating computational cost due to abnormally long generations. To the best of the author's knowledge, as a methodological contribution, this is the first application of Direct Preference Optimization (DPO) for OCR, explicitly using degenerate generations as rejected examples to penalize looping behavior. Combined with Supervised Fine-Tuning (SFT) for enforcing a strict JSON schema (header, margin, footer, and text), DPO consistently reduces degeneration rate across model families (up to 87.6% relative) while preserving or improving extraction quality. The resulting models, namely, DharmaOCR Full (7B) and DharmaOCR Lite (3B), set a new state-of-the-art on DharmaOCR-Benchmark, outperforming each open-source and commercial baseline model evaluated regarding extraction quality, reaching 0.925 and 0.911 scores with 0.40% and 0.20% degeneration rates. AWQ quantization reduced up to 22% per-page cost with negligible quality loss, enabling a strong quality-cost trade-off in comparison to proprietary OCR APIs and open-source alternatives. 

---
# Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization 

**Authors**: Junzhe Wang, Zhiheng Xi, yajie yang, Hao Luo, Shihan Dou, Tao Gui, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14267)  

**Abstract**: Search agents extend Large Language Models (LLMs) beyond static parametric knowledge by enabling access to up-to-date and long-tail information unavailable during pretraining. While reinforcement learning has been widely adopted for training such agents, existing approaches face key limitations: process supervision often suffers from unstable value estimation, whereas outcome supervision struggles with credit assignment due to sparse, trajectory-level rewards. To bridge this gap, we propose Contribution-Weighted GRPO (CW-GRPO), a framework that integrates process supervision into group relative policy optimization. Instead of directly optimizing process rewards, CW-GRPO employs an LLM judge to assess the retrieval utility and reasoning correctness at each search round, producing per-round contribution scores. These scores are used to rescale outcome-based advantages along the trajectory, enabling fine-grained credit assignment without sacrificing optimization stability. Experiments on multiple knowledge-intensive benchmarks show that CW-GRPO outperforms standard GRPO by 5.0\% on Qwen3-8B and 6.3\% on Qwen3-1.7B, leading to more effective search behaviors. Additional analysis reveals that successful trajectories exhibit concentrated contributions across rounds, providing empirical insight into search agent tasks. 

---
# Reinforcement Learning via Value Gradient Flow 

**Authors**: Haoran Xu, Kaiwen Hu, Somayeh Sojoudi, Amy Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14265)  

**Abstract**: We study behavior-regularized reinforcement learning (RL), where regularization toward a reference distribution (the dataset in offline RL or the base model in LLM RL finetuning) is essential to prevent value over-optimization caused by erroneous out-of-distribution extrapolation. Existing methods either rely on reparameterized policy gradient, which are difficult to scale to large generative models, or on reject sampling, which can be overly conservative when attempting to move beyond the behavior support. In this paper, we propose Value Gradient Flow (VGF), a scalable new paradigm for behavior-regularized RL. VGF casts behavior-regularized RL as an optimal transport problem that maps the reference distribution to the value-induced optimal policy distribution. We solve this transport problem via discrete gradient flow, where value gradients guide particles initialized from the reference distribution. Our analysis shows that VGF imposes regularization implicitly by controlling the transport budget. VGF eliminates explicit policy parameterization while remaining expressive and flexible, this enables adaptive test-time scaling by adjusting the transport budget. Extensive experiments demonstrate that VGF significantly outperforms prior methods, achieving state-of-the-art results on offline RL benchmarks (D4RL, OGBench) and LLM RL tasks. Code and runs can be found at this https URL. 

---
# Listen, Correct, and Feed Back: Spoken Pedagogical Feedback Generation 

**Authors**: Junhong Liang, Yifan Lu, Ekaterina Kochmar, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2604.14177)  

**Abstract**: Grammatical error correction (GEC) and explanation (GEE) have made rapid progress, but real teaching scenarios also require \emph{learner-friendly pedagogical feedback} that is actionable, level-appropriate, and encouraging. We introduce \textbf{SPFG} (\textbf{S}poken \textbf{P}edagogical \textbf{F}eedback \textbf{G}eneration), a dataset built based on the Speak \& Improve Challenge 2025 corpus, pairing fluency-oriented transcriptions with GEC targets and \emph{human-verified} teacher-style feedback, including preferred/rejected feedback pairs for preference learning. We study a transcript-based Spoken Grammatical Error Correction (SGEC) setting and evaluate three instruction-tuned LLMs (Qwen2.5, Llama-3.1, and GLM-4), comparing supervised fine-tuning (SFT) with preference-based alignment (using DPO and KTO) for jointly generating corrections and feedback. Results show that SFT provides the most consistent improvements, while DPO/KTO yield smaller or mixed gains, and that correction quality and feedback quality are weakly coupled. Our implementation is available at this https URL. 

---
# Internal Knowledge Without External Expression: Probing the Generalization Boundary of a Classical Chinese Language Model 

**Authors**: Jiuting Chen, Yuan Lian, Hao Wu, Tianqi Huang, Hiroshi Sasaki, Makoto Kouno, Jongil Choi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14180)  

**Abstract**: We train a 318M-parameter Transformer language model from scratch on a curated corpus of 1.56 billion tokens of pure Classical Chinese, with zero English characters or Arabic numerals. Through systematic out-of-distribution (OOD) testing, we investigate whether the model can distinguish known from unknown inputs, and crucially, whether it can express this distinction in its generated text.
We find a clear dissociation between internal and external uncertainty. Internally, the model exhibits a perplexity jump ratio of 2.39x between real and fabricated historical events (p = 8.9e-11, n = 92 per group), with semi-fabricated events (real figures + fictional events) showing the highest perplexity (4.24x, p = 1.1e-16), demonstrating genuine factual encoding beyond syntactic pattern matching. Externally, however, the model never learns to express uncertainty: classical Chinese epistemic markers appear at lower rates for OOD questions (3.5%) than for in-distribution questions (8.3%, p = 0.023), reflecting rhetorical conventions rather than genuine metacognition.
We replicate both findings across three languages (Classical Chinese, English, Japanese), three writing systems, and eight models from 110M to 1.56B parameters. We further show that uncertainty expression frequency is determined entirely by training data conventions, with Classical Chinese models showing a "humility paradox" (more hedging for known topics), while Japanese models almost never hedge. We argue that metacognitive expression -- the ability to say "I don't know" -- does not emerge from language modeling alone and requires explicit training signals such as RLHF. 

---
# Tug-of-War within A Decade: Conflict Resolution in Vulnerability Analysis via Teacher-Guided Retrieval-Augmented Generations 

**Authors**: Ziyin Zhou, Jianyi Zhang, Xu ji, Yilong Li, Jiameng Han, Zhangchi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14172)  

**Abstract**: Large Language Models (LLMs) are essential for analyzing and addressing vulnerabilities in cybersecurity. However, among over 200,000 vulnerabilities were discovered in the past decade, more than 30,000 have been changed or updated. This necessitates frequent updates to the training datasets and internal knowledge bases of LLMs to maintain knowledge consistency. In this paper, we focus on the problem of knowledge discrepancy and conflict within CVE (Common Vulnerabilities and Exposures) detection and analysis. This problem hinders LLMs' ability to retrieve the latest knowledge from original training datasets, leading to knowledge conflicts, fabrications of factually incorrect results, and generation hallucinations. To address this problem, we propose an innovative two-stage framework called CRVA-TGRAG (Conflict Resolution in Vulnerability Analysis via Teacher-Guided Retrieval-Augmented Generation). First, to improve document retrieval accuracy during the retrieval stage, we utilize Parent Document Segmentation and an ensemble retrieval scheme based on semantic similarity and inverted indexing. Second, to enhance LLMs' capabilities based on the retrieval of CVE dataset in generation stage, we employ a teacher-guided preference optimization technique to fine-tune LLMs. Our framework not only enhances the quality of content retrieval through RAG but also leverages the advantages of preference fine-tuning in LLMs to answer questions more effectively and precisely. Experiments demonstrate our method achieves higher accuracy in retrieving the latest CVEs compared to external knowledge bases. In conclusion, our framework significantly mitigates potential knowledge conflicts and inconsistencies that may arise from relying solely on LLMs for knowledge retrieval. 

---
