# EMO: Pretraining Mixture of Experts for Emergent Modularity 

**Authors**: Ryan Wang, Akshita Bhagia, Sewon Min  

**Link**: [PDF](https://arxiv.org/pdf/2605.06663)  

**Abstract**: Large language models are typically deployed as monolithic systems, requiring the full model even when applications need only a narrow subset of capabilities, e.g., code, math, or domain-specific knowledge. Mixture-of-Experts (MoEs) seemingly offer a potential alternative by activating only a subset of experts per input, but in practice, restricting inference to a subset of experts for a given domain leads to severe performance degradation. This limits their practicality in memory-constrained settings, especially as models grow larger and sparser. We introduce EMO, an MoE designed for modularity-the independent use and composition of expert subsets-without requiring human-defined priors. Our key idea is to encourage tokens from similar domains to rely on similar experts. Since tokens within a document often share a domain, EMO restricts them to select experts from a shared pool, while allowing different documents to use different pools. This simple constraint enables coherent expert groupings to emerge during pretraining using document boundaries alone. We pretrain a 1B-active, 14B-total EMO on 1T tokens. As a full model, it matches standard MoE performance. Crucially, it enables selective expert use: retaining only 25% (12.5%) of experts incurs just a 1% (3%) absolute drop, whereas standard MoEs break under the same setting. We further find that expert subsets in EMO specialize at semantic levels (e.g., domains such as math or code), in contrast to the low-level syntactic specialization observed in standard MoEs. Altogether, our results demonstrate a path toward modular, memory-efficient deployment of large, sparse models and open new opportunities for composable architectures. 

---
# Beyond Negative Rollouts: Positive-Only Policy Optimization with Implicit Negative Gradients 

**Authors**: Mingwei Xu, Hao Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06650)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR), due to the deterministic verification, becomes a dominant paradigm for enhancing the reasoning ability of large language models (LLMs). The community witnesses the rapid change from the Proximal Policy Optimization (PPO) to Group Relative Policy Optimization (GRPO), in which GRPO reduces the complicated advantage estimation with simple estimation over grouped positive and negative rollouts. However, we note that negative rollouts may admit no gradation of failure severity, and the combinatorial vastness makes penalizing a few sampled negatives unlikely to cover a meaningful reward signal under sparse binary rewards. In this work, we propose Positive-Only Policy Optimization (POPO), a novel RLVR framework in which learning can occur exclusively via online positive rollouts. Specifically, POPO utilizes bounded importance sampling over the positive rollout set. Thus, no disjoint negative rollouts are used for the gradient guidance. We show that implicit negative gradients can emerge naturally through reinforcing the positive probability via rollouts redistribution. Next, POPO stabilizes the policy optimization through two mechanisms. First, it applies a siamese policy network with a momentum-based adaptation law for stabilized policy evolution. Second, we replace the KL-divergence with a bounded similarity penalty term in the siamese representation space. We conduct extensive experiments using publicly available, well-established text-LLM models, e.g., the Qwen family, across all-level mathematical benchmarks. Our experiment demonstrates that POPO achieves performance comparable to, or even superior to GRPO. Notably, we show that POPO can achieve 36.67% in AIME 2025 with Qwen-Math-7B, outperforming GRPO 30.00%. Our ablation and sweep studies further illustrate the necessity and robustness of POPO components. 

---
# StraTA: Incentivizing Agentic Reinforcement Learning with Strategic Trajectory Abstraction 

**Authors**: Xiangyuan Xue, Yifan Zhou, Zidong Wang, Shengji Tang, Philip Torr, Wanli Ouyang, Lei Bai, Zhenfei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06642)  

**Abstract**: Large language models (LLMs) are increasingly used as interactive agents, but optimizing them for long-horizon decision making remains difficult because current methods are largely purely reactive, which weakens both exploration and credit assignment over extended trajectories. In this work, we present Strategic Trajectory Abstraction (StraTA), a simple framework that introduces an explicit trajectory-level strategy into agentic reinforcement learning (RL). StraTA samples a compact strategy from the initial task state, conditions subsequent actions on that strategy, and trains strategy generation and action execution jointly with a hierarchical GRPO-style rollout design, further enhanced by diverse strategy rollout and critical self-judgment. Experiments on ALFWorld, WebShop, and SciWorld show that StraTA consistently improves both sample efficiency and final performance over strong baselines. StraTA reaches success rates of 93.1% on ALFWorld and 84.2% on WebShop. On SciWorld, StraTA attains a 63.5% overall score, outperforming frontier closed-source models. 

---
# Cited but Not Verified: Parsing and Evaluating Source Attribution in LLM Deep Research Agents 

**Authors**: Hailey Onweller, Elias Lumer, Austin Huber, Pia Ramchandani, Vamse Kumar Subbiah, Corey Feld  

**Link**: [PDF](https://arxiv.org/pdf/2605.06635)  

**Abstract**: Large language models (LLMs) power deep research agents that synthesize information from hundreds of web sources into cited reports, yet these citations cannot be reliably verified. Current approaches either trust models to self-cite accurately, risking bias, or employ retrieval-augmented generation (RAG) that does not validate source accessibility, relevance, or factual consistency. We introduce the first source attribution evaluation framework that uses a reproducible AST parser to extract and evaluate inline citations from LLM-generated Markdown reports at scale. Unlike methods that verify claims in isolation, our framework closes the loop by retrieving the actual cited content, enabling human or model evaluators to judge each citation against its source. Citations are evaluated along three dimensions. (1) Link Works verifies URL accessibility, (2) Relevant Content measures topical alignment, and (3) Fact Check validates factual accuracy against source content. We benchmark 14 closed-source and open-source LLMs across three evaluation dimensions using rubric-based LLM-as-a-judge evaluators calibrated through human review. Our results reveal that even the strongest frontier models maintain link validity above 94% and relevance above 80%, yet achieve only 39-77% factual accuracy, while fewer than half of open-source models successfully generate cited reports in a one-shot setting. Ablation studies on research depth show that Fact Check accuracy drops by approximately 42% on average across two frontier models as tool calls scale from 2 to 150, demonstrating that more retrieval does not produce more accurate citations. These findings reveal a critical disconnect between surface-level citation quality and factual reliability, and our framework provides the evaluation infrastructure to assess the disconnect. 

---
# Parser agreement and disagreement in L2 Korean UD: Implications for human-in-the-loop annotation 

**Authors**: Hakyung Sung, Gyu-Ho Shin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06625)  

**Abstract**: We propose a simplified human-in-the-loop workflow for second language (L2) Korean morphosyntactic annotation by leveraging agreement between two domain-adapted parsers. We first evaluate whether parser agreement can serve as a proxy for annotation correctness by comparing it with independent human judgments. The results show strong correspondence between parser and human judgments, supporting the feasibility of semi-automatic L2-Korean UD annotation. Further analysis demonstrates that parser disagreements cluster in linguistically predictable domains such as grammatical-relation distinctions and clause-boundary ambiguity. While many disagreement cases are tractable for iterative model refinement, others reflect deeper representational challenges inherent in parsing and tagging L2-Korean corpora. 

---
# Algospeak, Hiding in the Open: The Trade-off Between Legible Meaning and Detection Avoidance 

**Authors**: Jan Fillies, Ronald E. Robertson, Jeffrey Hancock  

**Link**: [PDF](https://arxiv.org/pdf/2605.06619)  

**Abstract**: As large language models (LLMs) increasingly mediate both content generation and moderation, linguistic evasion strategies known as Algospeak have intensified the coevolution between evaders and detectors. This research formalizes the underlying dynamics grounded in a joint action model: when Algospeak increases, detectability and understandability decrease. Further, the concept of Majority Understandable Modulation (MUM) is introduced and defined as the modulation level at which additional evasive alteration increases detector evasion but loses comprehension for the majority of recipients. To empirically probe this trade-off, we introduce a reproducible framework that can be used to create meaning-preserving, Algospeak-style variants, based on an existing taxonomy and with tunable modulation levels. Using COVID-19 disinformation as a first proof-by-example setting, we construct a reference dataset of 700 modulated items, drawn from twenty base sentences across five modulation levels and seven strategies. We then run two linked evaluations with seven different language models: one testing for interpretation through meaning recovery and one for disinformation detection through classification. Curve fitting over modulation levels yields an estimate of the Majority Understandable Modulation threshold and enables sensitivity analyses across strategies and models, see Figure 1. Results reveal the characteristic relationships between understandability and modulation. This study lays the groundwork for understanding the dynamics behind Algospeak and provides the framework, dataset, and experimental setups described. 

---
# UniSD: Towards a Unified Self-Distillation Framework for Large Language Models 

**Authors**: Yiqiao Jin, Yiyang Wang, Lucheng Fu, Yijia Xiao, Yinyi Luo, Haoxin Liu, B. Aditya Prakash, Josiah Hester, Jindong Wang, Srijan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.06597)  

**Abstract**: Self-distillation (SD) offers a promising path for adapting large language models (LLMs) without relying on stronger external teachers. However, SD in autoregressive LLMs remains challenging because self-generated trajectories are free-form, correctness is task-dependent, and plausible rationales can still provide unstable or unreliable supervision. Existing methods mainly examine isolated design choices, leaving their effectiveness, roles, and interactions unclear. In this paper, we propose UniSD, a unified framework to systematically study self-distillation. UniSD integrates complementary mechanisms that address supervision reliability, representation alignment, and training stability, including multi-teacher agreement, EMA teacher stabilization, token-level contrastive learning, feature matching, and divergence clipping. Across six benchmarks and six models from three model families, UniSD reveals when self-distillation improves over static imitation, which components drive the gains, and how these components interact across tasks. Guided by these insights, we construct UniSDfull, an integrated pipeline that combines complementary components and achieves the strongest overall performance, improving over the base model by +5.4 points and the strongest baseline by +2.8 points. Extensive evaluation highlights self-distillation as a practical and steerable approach for efficient LLM adaptation without stronger external teachers. 

---
# Automated Clinical Report Generation for Remote Cognitive Remediation: Comparing Knowledge-Engineered Templates and LLMs in Low-Resource Settings 

**Authors**: Yongxin Zhou, Fabien Ringeval, François Portet  

**Link**: [PDF](https://arxiv.org/pdf/2605.06594)  

**Abstract**: The growing demand for cognitive remediation therapy, combined with limited speech therapist availability, has accelerated the adoption of remote rehabilitation tools. These systems generate large volumes of interaction data that are difficult for clinicians to review efficiently. This paper investigates automated clinical report generation for avatar-guided, home-based cognitive remediation sessions in a low-resource setting with no reference reports. We present and compare two approaches: (1) a rule-based template system encoding speech therapy domain knowledge as explicit decision rules and validated templates, ensuring clinical reliability and traceability; and (2) a zero-shot LLM-based approach (GPT-4) aimed at more fluent and concise output. Both systems use identical pre-extracted, expert-validated structured variables, enabling a controlled factual comparison. Outputs were evaluated by eight speech therapists and final-year students using a nine-criterion questionnaire. Results reveal a clear trade-off between clinical reliability and linguistic quality. The template-based system scored higher on fluidity, coherence, and results presentation, while GPT-4 produced more concise output. Directional differences are consistent across evaluation dimensions, though no comparison reached statistical significance after correction, reflecting the scale constraints of expert clinical evaluation. Based on evaluator feedback, we derive eight design recommendations for clinical reporting systems in remote rehabilitation settings. More broadly, this work contributes a replicable methodology combining expert elicitation, taxonomy-driven generation, and multi-dimensional human evaluation for clinical NLG in low-resource settings, and illustrates how controlled comparisons can inform the responsible adoption of generative AI in healthcare. 

---
# Long Context Pre-Training with Lighthouse Attention 

**Authors**: Bowen Peng, Subho Ghosh, Jeffrey Quesnelle  

**Link**: [PDF](https://arxiv.org/pdf/2605.06554)  

**Abstract**: Training causal transformers at extreme sequence lengths is bottlenecked by the quadratic time and memory of scaled dot-product attention (SDPA). In this work, we propose Lighthouse Attention, a training-only symmetrical selection-based hierarchical attention algorithm that wraps around ordinary SDPA and can be easily removed towards the end of the training. Our hierarchical selection is also gradient-free, which exempts us from dealing with a complicated and potentially inefficient backward pass kernel. Our contribution is three-fold: (i) A subquadratic hierarchical pre- and post-processing step that does adaptive compression and decompression of the sequence. (ii) A symmetrical compression strategy that pools queries, keys and values at the same time, while preserving left-to-right causality, which greatly improves parallelism. (iii) A two stage training approach which we pre-train for the majority of the time with Lighthouse Attention and recover a full attention model at the end with a short training. We run preliminary small scale LLM pre-training experiments that show the effectiveness of our method compared to full attention training with all other settings matched, where we achieve a faster total training time and lower final loss after the recovery phase. Full code is available at: this https URL 

---
# Continuous Latent Diffusion Language Model 

**Authors**: Hongcan Guo, Qinyu Zhao, Yian Zhao, Shen Nie, Rui Zhu, Qiushan Guo, Feng Wang, Tao Yang, Hengshuang Zhao, Guoqiang Wei, Yan Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2605.06548)  

**Abstract**: Large language models have achieved remarkable success under the autoregressive paradigm, yet high-quality text generation need not be tied to a fixed left-to-right order. Existing alternatives still struggle to jointly achieve generation efficiency, scalable representation learning, and effective global semantic modeling. We propose Cola DLM, a hierarchical latent diffusion language model that frames text generation through hierarchical information decomposition. Cola DLM first learns a stable text-to-latent mapping with a Text VAE, then models a global semantic prior in continuous latent space with a block-causal DiT, and finally generates text through conditional decoding. From a unified Markov-path perspective, its diffusion process performs latent prior transport rather than token-level observation recovery, thereby separating global semantic organization from local textual realization. This design yields a more flexible non-autoregressive inductive bias, supports semantic compression and prior fitting in continuous space, and naturally extends to other continuous modalities. Through experiments spanning 4 research questions, 8 benchmarks, strictly matched ~2B-parameter autoregressive and LLaDA baselines, and scaling curves up to about 2000 EFLOPs, we identify an effective overall configuration of Cola DLM and verify its strong scaling behavior for text generation. Taken together, the results establish hierarchical continuous latent prior modeling as a principled alternative to strictly token-level language modeling, where generation quality and scaling behavior may better reflect model capability than likelihood, while also suggesting a concrete path toward unified modeling across discrete text and continuous modalities. 

---
# Efficient Pre-Training with Token Superposition 

**Authors**: Bowen Peng, Théo Gigant, Jeffrey Quesnelle  

**Link**: [PDF](https://arxiv.org/pdf/2605.06546)  

**Abstract**: Pre-training of Large Language Models is often prohibitively expensive and inefficient at scale, requiring complex and invasive modifications in order to achieve high data throughput. In this work, we present Token-Superposition Training (TST), a simple drop-in method that significantly improves the data throughput per FLOPs during pre-training without modifying the parallelism, optimizer, tokenizer, data, or model architecture. TST is done in two phases: (i) A highly efficient superposition phase where we combine many contiguous tokens into one bag and train using a multi-hot cross-entropy (MCE) objective, and (ii) a recovery phase where we revert back to standard training. We extensively evaluate TST on the scale of 270M and 600M parameters and validate on 3B and a 10B A1B mixture of experts model, demonstrating that it is highly robust in different settings. Ultimately, TST consistently outperforms baseline loss and downstream evaluations, and under equal-loss settings, TST yields up to a 2.5x reduction in total pre-training time at the 10B A1B scale. 

---
# STALE: Can LLM Agents Know When Their Memories Are No Longer Valid? 

**Authors**: Hanxiang Chao, Yihan Bai, Rui Sheng, Tianle Li, Yushi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.06527)  

**Abstract**: Large Language Model (LLM) agents are increasingly expected to maintain coherent, long-term personalized memory, yet current benchmarks primarily measure static fact retrieval, overlooking the ability to revise stored beliefs when new evidence emerges. We identify a critical and underexplored failure mode, Implicit Conflict: a later observation invalidates an earlier memory without explicit negation, requiring contextual inference and commonsense reasoning to detect. To rigorously evaluate this capability, we introduce STALE, a benchmark of 400 expert-validated conflict scenarios (1,200 evaluation queries across three probing dimensions) spanning over 100 everyday topics with contexts up to 150K tokens. We propose a three-dimensional probing framework that tests State Resolution (detecting that a prior belief is outdated), Premise Resistance (rejecting queries that falsely presuppose a stale state), and Implicit Policy Adaptation (proactively applying updated states in downstream behavior). A systematic evaluation of frontier LLMs and specialized memory frameworks reveals a pervasive gap between retrieving updated evidence and acting on it, with even the best evaluated model achieving only 55.2% overall accuracy. Models often accept outdated assumptions embedded in a user's query, and they struggle to recognize when a change in one aspect of the user's state should invalidate related memories. To establish an initial baseline for state-aware memory, we further present CUPMem, a prototype that strengthens write-time revision through structured state consolidation and propagation-aware search, suggesting that explicit state adjudication is a promising direction for robust agentic memory. 

---
# The Frequency Confound in Language-Model Surprisal and Metaphor Novelty 

**Authors**: Omar Momen, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2605.06506)  

**Abstract**: Language-model (LM) surprisal is widely used as a proxy for contextual predictability and has been reported to correlate with metaphor novelty judgments. However, surprisal is tightly intertwined with lexical frequency. We explore this interaction on metaphor novelty ratings using two different word frequency measures. We analyse surprisal estimates from eight Pythia model sizes and 154 training checkpoints. Across settings, word frequency is a stronger predictor of metaphor novelty than surprisal. Across training stages, the surprisal--novelty association peaks at an early stage and then falls again, mirroring a similarly timed increase in the surprisal--frequency association. These results suggest that the often-reported optimal LM surprisal settings may incorrectly associate contextual predictability with metaphor novelty and processing difficulty, whereas lexical frequency may be the major underlying factor. 

---
# Litespark Inference on Consumer CPUs: Custom SIMD Kernels for Ternary Neural Networks 

**Authors**: Nii Osae Osae Dade, Tony Morri, Moinul Hossain Rahat, Sayandip Pal  

**Link**: [PDF](https://arxiv.org/pdf/2605.06485)  

**Abstract**: Large language models (LLMs) have transformed artificial intelligence, but their computational requirements remain prohibitive for most users. Standard inference demands expensive datacenter GPUs or cloud API access, leaving over one billion personal computers underutilized for AI workloads. Ternary models offer a path forward: their weights are constrained to {-1, 0, +1}, theoretically eliminating the need for floating-point multiplication. However, existing frameworks fail to exploit this structure, treating ternary models as dense floating-point networks. We address this gap with custom SIMD kernels that replace matrix multiplication with simple addition and subtraction operations, targeting the integer dot product instructions available on modern CPUs. Our implementation, Litespark-Inference, is pip-installable and integrates directly with Hugging-Face, achieving 9.2x faster time-to-first-token, 52x higher throughput, and 14x memory reduction compared to standard PyTorch inference on Apple Silicon, with similar speedups on Intel and AMD processors. 

---
# Towards Emotion Consistency Analysis of Large Language Models in Emotional Conversational Contexts 

**Authors**: Sneha Oram, Ojaswita Bhushan, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2605.06476)  

**Abstract**: In this work, we conduct an analysis to examine the consistency of Large Language Models (LLMs) with respect to their own generated responses in an emotionally-driven conversational context. Specifically, the text generated by LLM is framed as a query to the same model, and its responses are subsequently assessed. This is performed with three queries across two dimensions of extreme and moderate emotions. The three queries are, in particular, false claim queries that contain inherently wrong assumptions (false presuppositions) in increasing order of intensity. Two commercial models, Claude-3.5-haiku, GPT4o-mini, and a medium-sized model, Mistral-7B, are considered in the study. Our findings indicate that LLMs exhibit below-average performance and remain vulnerable to false beliefs embedded within queries. This susceptibility is especially pronounced for moderate emotional content. Furthermore, an extended attention-score-based analysis highlights a shift in models' priority from evaluative to generative. The results raise important considerations for LLMs' deployment in high-stakes, emotionally sensitive contexts. 

---
# COVID-19 Infodemic. Understanding content features in detecting fake news using a machine learning approach 

**Authors**: Balakrishnan Vimala, Hii Lee Zing, Laporte Eric  

**Link**: [PDF](https://arxiv.org/pdf/2605.06435)  

**Abstract**: The use of content features, particularly textual and linguistic for fake news detection is under-researched, despite empirical evidence showing the features could contribute to differentiating real and fake news. To this end, this study investigates a selection of content features such as word bigrams, part of speech distribution etc. to improve fake news detection. We performed a series of experiments on a new dataset gathered during the COVID-19 pandemic and using Decision Tree, K-Nearest Neighbor, Logistic Regression, Support Vector Machine and Random Forest. Random Forest yielded the best results, followed closely by Support Vector Machine, across all setups. In general, both the textual and linguistic features were found to improve fake news detection when used separately, however, combining them into a single model did not improve the detection significantly. Differences were also noted between the use of bigrams and part of speech tags. The study shows that textual and linguistic features can be used successfully in detecting fake news using the traditional machine learning approach as opposed to deep learning. 

---
# From 124 Million Tokens to 1,021 Neologisms: A Large-Scale Pipeline for Automatic Neologism Detection 

**Authors**: Diego Rossini, Lonneke van der Plas  

**Link**: [PDF](https://arxiv.org/pdf/2605.06426)  

**Abstract**: We present a scalable, modular pipeline for automatic neologism detection that combines rule-based filtering with LLM classification. The pipeline is grounded in two complementary word-formation frameworks, grammatical and extra-grammatical morphology, which jointly define the scope of what counts as a neologism and inform a four-class classification scheme (neologism, entity, foreign, none). While designed to be modular and transferable at the architectural level, the pipeline is instantiated on 527 million English-language Reddit posts spanning 2005-2024. From this corpus, we extract 124.6 million unique tokens and reduce them by over 99.99% to yield 1,021 neologism candidates, a set small enough for manual expert verification. Multiple LLMs independently classify each candidate via majority vote, with a final verification step, revealing substantial cross-model disagreement and highlighting the challenge of operationalizing neologism detection at scale. Manual annotation of all 1,021 candidates confirms that 599 (58.7%) are genuine lexical innovations. The pipeline code, vocabulary compilation scripts, and the annotated candidate list are available at this https URL. 

---
# MiA-Signature: Approximating Global Activation for Long-Context Understanding 

**Authors**: Yuqing Li, Jiangnan Li, Mo Yu, Zheng Lin, Weiping Wang, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.06416)  

**Abstract**: A growing body of work in cognitive science suggests that reportable conscious access is associated with \emph{global ignition} over distributed memory systems, while such activation is only partially accessible as individuals cannot directly access or enumerate all activated contents. This tension suggests a plausible mechanism that cognition may rely on a compact representation that approximates the global influence of activation on downstream processing. Inspired by this idea, we introduce the concept of \textbf{Mindscape Activation Signature (MiA-Signature)}, a compressed representation of the global activation pattern induced by a query. In LLM systems, this is instantiated via submodular-based selection of high-level concepts that cover the activated context space, optionally refined through lightweight iterative updates using working memory. The resulting MiA-Signature serves as a conditioning signal that approximates the effect of the full activation state while remaining computationally tractable. Integrating MiA-Signatures into both RAG and agentic systems yields consistent performance gains across multiple long-context understanding tasks. 

---
# GATHER: Convergence-Centric Hyper-Entity Retrieval for Zero-Shot Cell-Type Annotation 

**Authors**: Zhonghui Zhang, Feng Jiang, Shaowei Qin, Jiahao Zhao, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06403)  

**Abstract**: Zero-shot single-cell cell-type annotation aims to determine a cell's type from a given set of expressed genes without any training. Existing knowledge-graph-based RAG approaches retrieve evidence by expanding from source entities and relying on iterative LLM reasoning. However, in this setting each query contains tens to hundreds of genes, where no single gene is decisive and the label emerges only from their collective co-occurrence. Such hyper-entity queries fundamentally challenge local, entity-wise exploration strategies, which reason from individual genes, leading to poor scalability and substantial LLM cost. We propose GATHER (Graph-Aware Traversal with Hyper-Entity Retrieval), a convergence-centric retriever tailored to hyper-entity queries. It performs global multi-source graph traversal and identifies topological convergence points -- nodes jointly reachable from many input genes. These convergence nodes act as high-information hyper-entities that capture entity synergy. By incorporating node- and path-importance scoring, GATHER selects informative evidence entirely without LLM involvement during retrieval. Instantiated on a self-constructed cell-centric biological knowledge graph (VCKG), GATHER outperforms strong KG-RAG baselines (ToG, ToG-2, RoG, PoG) on two datasets (Immune and Lung), achieving the highest exact-match accuracy (27.45% and 59.64%) with only a single LLM call per sample, compared to 2--61 calls for KG-RAG baselines. Our results demonstrate that convergence nodes compress multi-entity signals into compact, high-information evidence that conveys more per item than multi-hop paths, providing an efficient global alternative to local entity-wise reasoning. 

---
# SEQUOR: A Multi-Turn Benchmark for Realistic Constraint Following 

**Authors**: Beatriz Canaverde, Duarte M. Alves, José Pombal, Giuseppe Attanasio, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2605.06353)  

**Abstract**: In a conversation, a helpful assistant must reliably follow user directives, even as they refine, modify, or contradict earlier requests. Yet most instruction-following benchmarks focus on single-turn or short multi-turn scenarios, leaving open how well models handle long-horizon instruction-following tasks. To bridge this gap, we present SEQUOR, an automatic benchmark for evaluating constraint adherence in long multi-turn conversations. SEQUOR consists of simulated persona-driven interactions built with constraints extracted from real-world conversations. Our results show that even when following a single constraint, instruction-following accuracy consistently decreases as the conversation grows longer, with drops exceeding 11%. This decline becomes larger when models have to follow multiple constraints simultaneously, reducing their accuracy by over 40%. In scenarios where constraints are added or replaced at arbitrary points of the conversation, model accuracy decreases by more than 9%. Taken together, our results reveal that current models still struggle to follow user instructions in multi-turn conversations, and provide a way for better measuring instruction-following capabilities in assistants. 

---
# Don't Lose Focus: Activation Steering via Key-Orthogonal Projections 

**Authors**: Haoyan Luo, Mateo Espinosa Zarlenga, Mateja Jamnik  

**Link**: [PDF](https://arxiv.org/pdf/2605.06342)  

**Abstract**: Activation steering controls LLM behaviour towards target behaviour by intervening in internal representations, yet it often degrades reasoning and retrieval performance. We argue that a primary cause of this trade-off is attention rerouting: steering vectors alter query-key matching, shifting attention away from contextually important tokens toward less informative ones. To address this, we propose Steering via Key-Orthogonal Projections (SKOP), a steering method that constrains harmful attention rerouting without eliminating steering efficacy. SKOP achieves this by preserving attention patterns on a small set of focus tokens the model relies on for reasoning and retrieval, while allowing redistribution among less critical tail tokens. Across multiple steering benchmarks, we show that SKOP achieves the best joint steering-utility trade-off, reducing utility degradation by 5-7x while retaining over 95% of vanilla steering efficacy. Our results further suggest that, in long-context retrieval settings where vanilla steering approaches are ineffective, SKOP can maintain robust performance by avoiding attention rerouting. 

---
# MANTRA: Synthesizing SMT-Validated Compliance Benchmarks for Tool-Using LLM Agents 

**Authors**: Ashwani Anand, Ivi Chatzi, Ritam Raha, Anne-Kathrin Schmuck  

**Link**: [PDF](https://arxiv.org/pdf/2605.06334)  

**Abstract**: Tool-using large language model (LLM) agents are increasingly deployed in settings where their reliable behavior is governed by strict procedural manuals. Ensuring that such agents comply with the rules from these manuals is challenging, as they are typically written for humans in natural language while agent behavior manifests as an execution trace of tool calls. Existing evaluations of LLM agents rely on manually constructed benchmarks or LLM-based judges, which either do not scale or lack reliability for complex, long-horizon manuals. To overcome these limitations, we present MANTRA, a framework for automatically synthesizing machine-checkable compliance benchmarks from natural-language manuals and tool schemas. MANTRA independently generates (i) a symbolic world model capturing procedural dependencies, and (ii) a set of trace-level compliance checks for a given task, and validates their consistency using SMT solving. A structured repair loop resolves inconsistencies, requiring human intervention only as a fallback. %This yields benchmarks that are formally validated. Importantly, MANTRA supports arbitrary domains and long procedural manuals, and provides a tunable notion of task complexity which is utilized to automatically derive challenging tasks accompanying compliance checks. Using MANTRA, we build a new benchmark suite with 285 tasks across 6 domains scaling to 50+ page manuals with minimal human effort. Empirically, we show that the compliance checks are richer with stronger constraint enforcement compared to existing benchmarks. Additionally, the granularity of the checks can be used for debugging the agents' failure modes. These results demonstrate that combining automated benchmark generation with formally grounded validation methods enables scalable and reliable benchmarking of tool-using agents. 

---
# Measuring Evaluation-Context Divergence in Open-Weight LLMs: A Paired-Prompt Protocol with Pilot Evidence of Alignment-Pipeline-Specific Heterogeneity 

**Authors**: Florian A. D. Burnat, Brittany I. Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2605.06327)  

**Abstract**: Safety benchmarks are routinely treated as evidence about how a language model will behave once deployed, but this inference is fragile if behavior depends on whether a prompt looks like an evaluation. We define evaluation-context divergence as an observable within-item change in behavior induced by framing a fixed task as an evaluation, a live deployment interaction, or a neutral request, and present a paired-prompt protocol that measures it in open-weight LLMs while controlling for paraphrase variation, benchmark familiarity, and judge framing-sensitivity.
Across five instruction-tuned checkpoints from four open-weight families plus a matched OLMo-3 base/instruct ablation ($20$ paired items, $840$ generations per checkpoint), we find striking heterogeneity. OLMo-3-Instruct alone is eval-cautious -- evaluation framing raises refusal vs. neutral by $11.8$pp ($p=0.007$) and reduces harmful compliance vs. deployment by $3.6$pp ($p=0.024$, $0/20$ items inverted) -- while Mistral-Small-3.2, Phi-3.5-mini, and Llama-3.1-8B are deployment-cautious}, with marginal eval-vs-deployment refusal effects of $-9$ to $-20$pp. The matched OLMo-3 base also exhibits the deployment-cautious pattern, identifying alignment as the inversion stage; within Llama-3.1, the $70$B model preserves direction with attenuated magnitude, ruling out a simple ``small-model effect that reverses at scale.'' One caveat: the cross-family heterogeneity is judge-dependent. Re-judging with a different-family safety classifier (Llama-Guard-3-8B) preserves the within-OLMo eval-cautious direction but flattens the cross-family contrast, indicating that the two judges operationalize distinct constructs. 

---
# Teaching Thinking Models to Reason with Tools: A Full-Pipeline Recipe for Tool-Integrated Reasoning 

**Authors**: Qianjia Cheng, Yuchen Zhang, Zhilin Wang, Yuxin Zuo, Shunkai Zhang, Yuchen Fan, Yu Qiao, Bowen Zhou, Ning Ding, Yu Cheng, Yun Luo, Ganqu Cui  

**Link**: [PDF](https://arxiv.org/pdf/2605.06326)  

**Abstract**: Tool-integrated reasoning (TIR) offers a direct way to extend thinking models beyond the limits of text-only reasoning. Paradoxically, we observe that tool-enabled evaluation can degrade reasoning performance even when the strong thinking models make almost no actual tool calls. In this paper, we investigate how to inject natural tool-use behavior into a strong thinking model without sacrificing its no-tool reasoning ability, and present a comprehensive TIR recipe. We highlight that (i) the effectiveness of TIR supervised fine-tuning (SFT) hinges on the learnability of teacher trajectories, which should prioritize problems inherently suited for tool-augmented solutions; (ii) controlling the proportion of tool-use trajectories could mitigate the catastrophic forgetting of text-only reasoning capacity; (iii) optimizing for pass@k and response length instead of training loss could maximize TIR SFT gains while preserving headroom for reinforcement learning (RL) exploration; (iv) a stable RL with verifiable rewards (RLVR) stage, built upon suitable SFT initialization and explicit safeguards against mode collapse, provides a simple yet remarkably effective solution. When applied to Qwen3 thinking models at 4B and 30B scales, our recipe yields models that achieve state-of-the-art performance in a wide range of benchmarks among open-source models, such as 96.7% and 99.2% on AIME 2025 for 4B and 30B, respectively. 

---
# Who and What? Using Linguistic Features and Annotator Characteristics to Analyze Annotation Variation 

**Authors**: Maximilian Maurer, Maximilian Linde, Gabriella Lapesa  

**Link**: [PDF](https://arxiv.org/pdf/2605.06318)  

**Abstract**: Human label variation has been established as a central phenomenon in NLP: the perspectives different annotators have on the same item need to be embraced. Data collection practices thus shifted towards increasing the annotator numbers and releasing disaggregated datasets, harmful language being most resourced due to its high subjectivity. While this resulted in rich information about \textit{who} annotated (sociodemographics, attitudes, etc.), the \textit{what} (e.g., linguistic properties of items), and their interplay has received little attention. We present the first large-scale analysis of four reference datasets for harmful language detection, bringing together annotator characteristics, linguistic properties of the items, and their interactions in a statistically informed picture. We find that interactions are crucial, revealing intersectional effects ignored in previous work, and that a strong role is played by lexical cues and annotator attitudes. Effect patterns, however, vary considerably across datasets. This urges caution about generalization and transferability. 

---
# MultiLinguahah : A New Unsupervised Multilingual Acoustic Laughter Segmentation Method 

**Authors**: Callejas Sofia, Gomez Nahuel, Pelachaud Catherine, Ravenet Brian, Barriere Valentin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06309)  

**Abstract**: Laughter is a social non-vocalization that is universal across cultures and languages, and is crucial for human communication, including social bonding and communication signaling. However, detecting laughter in audio is a challenging task, and segmenting is even more difficult. Currently, Machine Learning methods generally rely on costly manual annotation, and their datasets are mostly based on English contexts. Thus, we propose an unsupervised multilingual method that sets up the laughter segmentation task as an anomaly detection of energy-based segmented audio sequences. Our method applies an Isolation Forest on audio representations learned from BYOL-A encoder. We compare our method with several state-of-the-art laughter detection algorithms on four datasets, including stand-up comedy, sitcoms, and general short audio from AudioSet. Our results show that state-of-the-art methods are not optimized for multilingual contexts, while our method outperforms them in non-English settings. 

---
# Log-Likelihood, Simpson's Paradox, and the Detection of Machine-Generated Text 

**Authors**: Tom Kempton, Viktor Drobnyi, Maeve Madigan, Stuart Burrell  

**Link**: [PDF](https://arxiv.org/pdf/2605.06294)  

**Abstract**: The ability to reliably distinguish human-written text from that generated by large language models is of profound societal importance. The dominant approach to this problem exploits the likelihood hypothesis: that machine-generated text should appear more probable to a detector language model than human-written text. However, we demonstrate that the token-level signal distinguishing human and machine text is non-uniform across the hidden space of the detector model, and naively averaging likelihood-based token scores across regions with fundamentally different statistical structure, as most detectors do, causes a form of Simpson's paradox: a strong local signal is destroyed by inappropriate aggregation. To correct for this, we introduce a learned local calibration step grounded in Bayesian decision theory. Rather than aggregating raw token scores, we first learn lightweight predictors of the score distributions conditioned on position in hidden space, and aggregate calibrated log-likelihood ratios instead. This single intervention dramatically and consistently improves detection performance across all baseline detectors and all datasets we consider. For example, our calibrated variant of Fast-DetectGPT improves AUROC from $0.63$ to $0.85$ on GPT-5.4 text, and a locally-calibrated DMAP detector we introduce achieves state-of-the-art performance across the board. That said, our central contribution is not a new detector, but a precise diagnosis of a significant cause of under-performance of existing detectors and a principled, modular remedy compatible with any token-averaging pipeline. This will serve as a foundation for the community to build upon, with natural avenues including richer distributional models, improved calibration strategies, and principled ensembling with hidden-space geometry signals via the full Bayes-optimal decision rule. 

---
# LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG 

**Authors**: Yijia Zheng, Marcel Worring  

**Link**: [PDF](https://arxiv.org/pdf/2605.06285)  

**Abstract**: Single-step retrieval-augmented generation (RAG) provides an efficient way to incorporate external information for simple question answering tasks but struggles with complex questions. Agentic RAG extends this paradigm by replacing single-step retrieval with a multi-step process, in which the large language model (LLM) acts as a search agent that generates intermediate thoughts and subqueries to iteratively interact with the retrieval system. This iterative process incurs substantial latency due to the autoregressive generation of lengthy thoughts and subqueries. To address this limitation, we propose LatentRAG, a novel framework that shifts both reasoning and retrieval from discrete language space to continuous latent space. Unlike existing explicit methods that generate natural language thoughts or subqueries token-by-token, LatentRAG produces latent tokens for thoughts and subqueries directly from the hidden states in a single forward pass. We align LLMs with dense retrieval models in the latent space, enabling retrieval over latent subquery tokens and supporting end-to-end joint optimization. To improve transparency and encourage semantically meaningful latent representations, we incorporate a parallel latent decoding mechanism that translates latent tokens back into natural language. Extensive experiments on seven benchmark datasets show that LatentRAG achieves performance comparable to explicit agentic RAG methods while reducing inference latency by approximately 90%, substantially narrowing the latency gap with traditional single-step RAG. 

---
# Quantifying the Statistical Effect of Rubric Modifications on Human-Autorater Agreement 

**Authors**: Jessica Huynh, Alfredo Gomez, Athiya Deviyani, Renee Shelby, Jeffrey P. Bigham, Fernando Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2605.06283)  

**Abstract**: Autoraters, also referred to as LLM-as-judges, are increasingly used for evaluation and automated content moderation. However, there is limited statistical analysis of how modifications in a rubric presented to both humans and autoraters affect their score agreement. Rubrics that ask for an overall or \emph{holistic} judgment - for example, rating the ``quality'' of an essay - may be inconsistently interpreted due to the complexity or subjectivity of the criteria. Conversely, rubrics can ask for \emph{analytic} judgments, which decompose assessment criteria - for example, ``quality'' into ``fluency'' and ``organization''. While these rubrics can be edited to improve the individual accuracy of both human and automated scoring, this approach may result in disagreement between the two scores, or with the associated holistic judgment. Designing and deploying reliable autoraters requires understanding not just the relationship between human and autorater annotations but how that relationship changes as holistic or analytic judgments are elicited. The results indicate that rubric edits providing representative examples and additional context, and reducing positional bias in the rubric increased human-autorater agreement, while higher rubric complexity and conservative aggregation methods tended to decrease it. The findings from the automatic essay scoring and instruction-following evaluation domains suggest that practitioners should carefully analyze domain- and rubric-specific performance to move towards higher human-autorater agreement. 

---
# Linear Semantic Segmentation for Low-Resource Spoken Dialects 

**Authors**: Kirill Chirkunov, Younes Samih, Abed Alhakim Freihat, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2605.06276)  

**Abstract**: Semantic segmentation is a core component of discourse analysis, yet existing models are primarily developed and evaluated on high-resource written text, limiting their effectiveness on low-resource spoken varieties. In particular, dialectal Arabic exhibits informal syntax, code-switching, and weakly marked discourse structure that challenge standard segmentation approaches. In this paper, we introduce a new multi-genre benchmark (more than 1000 samples) for semantic segmentation in conversational Arabic, focusing on dialectal discourse. The benchmark covers transcribed casual telephone conversations, code-switched podcasts, broadcast news, and expressive dialogue from novels, and was annotated and validated by native Arabic annotators. Using this benchmark, we show that segmentation models performing well on MSA news genres degrade on dialectal transcribed speech. We further propose a segmentation model that targets local semantic coherence and robustness to discourse discontinuities, consistently outperforming strong baselines on dialectal non-news genres. The benchmark and approach generalize to other low-resource spoken languages. 

---
# Rethinking RL for LLM Reasoning: It's Sparse Policy Selection, Not Capability Learning 

**Authors**: Ömer Faruk Akgül, Rajgopal Kannan, Willie Neiswanger, Viktor Prasanna  

**Link**: [PDF](https://arxiv.org/pdf/2605.06241)  

**Abstract**: Reinforcement learning has become the standard for improving reasoning in large language models, yet evidence increasingly suggests that RL does not teach new strategies; it redistributes probability mass over solutions the base model already contains. In this work, we ask: if RL merely steers the model toward paths it already knows, is the RL optimization loop itself necessary? Through token-level analysis across multiple model families and RL algorithms, we find that RL's beneficial footprint is a sparse, predictable correction concentrated at high-entropy decision points where the model is uncertain which branch to take. Only 1--3\% of token positions are affected, the promoted token always lies within the base model's top-5 alternatives, and targeted corrections at those few positions causally recover a large fraction of RL's accuracy gain, while random corrections fail. The base model's own entropy identifies these positions without any RL-trained model, and the entire correction is low-dimensional, representable in a tiny fraction of model parameters. These findings reframe reasoning improvement as sparse policy selection, not capability acquisition. We translate this insight into ReasonMaxxer, a minimal RL-free method that applies contrastive loss only at entropy-gated decision points, using a few hundred base-model rollouts and no online generation. Across three model families, six scales, and six math reasoning benchmarks, ReasonMaxxer matches or exceeds full RL performance while requiring only tens of problems and minutes of single-GPU training, a reduction in training cost of roughly three orders of magnitude. 

---
# YEZE at SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization via Heterogeneous Ensembling 

**Authors**: Fengze Guo, Yue Chang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06231)  

**Abstract**: This paper presents our system for SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization, which identifies polarized social media content in 22 languages through three subtasks: binary detection, target classification, and manifestation identification. We propose a heterogeneous ensemble of multilingual pretrained models, combining XLM-RoBERTa-large and mDeBERTa-v3-base. We investigate techniques such as multi-task learning, translation-based data augmentation, and class weighting to improve classification performance under severe label imbalance. Our findings indicate that independent task modeling combined with class weighting is more effective. 

---
# UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification 

**Authors**: Qihang Fan, Huaibo Huang, Zhiying Wu, Bingning Wang, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2605.06221)  

**Abstract**: As large language models (LLMs) continue to advance rapidly, they are becoming increasingly capable while simultaneously demanding ever-longer context lengths. To improve the inference efficiency of long-context processing, several novel low-complexity hybrid architectures have recently been proposed, effectively alleviating the computational burden of long-context inference. However, existing research on long-context prefill acceleration remains predominantly focused on sparse attention mechanisms, which achieve their maximum speedup only on full-attention models. When transferred to emerging architectures--such as linear/full attention hybrids or sliding window/full attention hybrids--these prefill acceleration approaches suffer significant performance degradation. Furthermore, such methods are generally incompatible with continuous batching, making them difficult to integrate into modern inference engines such as vLLM. To this end, we propose UniPrefill, a prefill acceleration framework applicable to virtually any model architecture, which directly accelerates the model's computation at the token level. We further implement UniPrefill as a continuous batching operator and extend vLLM's scheduling strategy to natively support prefill-decode co-processing and tensor parallel for UniPrefill, enabling its seamless integration into vLLM. UniPrefill achieves up to 2.1x speedup in Time-To-First-Token (TTFT), with the acceleration becoming increasingly pronounced as the number of concurrent requests grows. 

---
# TIDE: Every Layer Knows the Token Beneath the Context 

**Authors**: Ajay Jaiswal, Lauren Hannah, Han-Byul Kim, Duc Hoang, Mehrdad Farajtabar, Minsik Cho  

**Link**: [PDF](https://arxiv.org/pdf/2605.06216)  

**Abstract**: We revisit a universally accepted but under-examined design choice in every modern LLM: a token index is looked up once at the input embedding layer and then permanently discarded. This single-injection assumption induces two structural failures: (i) the Rare Token Problem, where a Zipf-type distribution of vocabulary causes rare-token embeddings are chronically under-trained due to receiving a fraction of the cumulative gradient signal compared to common tokens; and (ii) the Contextual Collapse Problem, where limited parameters models map distributionally similar tokens to indistinguishable hidden states. As an attempt to address both, we propose TIDE, which augments the standard transformer with EmbeddingMemory: an ensemble of K independent MemoryBlocks that map token indices to context-free semantic vectors, computed once and injected into every layer through a depth-conditioned softmax router with a learnable null bank. We theoretically and empirically establish the benefits of TIDE in addressing the issues associated with single-token identity injection as well as improve performance across multiple language modeling and downstream tasks. 

---
# A$^2$TGPO: Agentic Turn-Group Policy Optimization with Adaptive Turn-level Clipping 

**Authors**: Dingwei Chen, Zefang Zong, Zhipeng Ma, Leo Luo, Yang Li, Chengming Li, Peng Chen, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06200)  

**Abstract**: Reinforcement learning for agentic large language models (LLMs) typically relies on a sparse, trajectory-level outcome reward, making it difficult to evaluate the contribution of individual tool-calls within multi-turn interactions. Existing approaches to such process credit assignment either depend on separate external process reward models that introduce additional consumption, or tree-based structural rollout that merely redistributes the outcome signal while constraining trajectory diversity. A promising alternative leverages the per-turn change in the policy's predicted probability of the ground-truth, termed Information Gain (IG), as an intrinsic process signal without an external evaluator. However, prior work on leveraging IG signals within the RL training loop faces three systematic challenges: normalizing across turns that face heterogeneous positional contexts can distort the relative standing of individual turns, accumulating a variable number of terms causes advantage magnitudes to drift with trajectory depth, and a fixed clipping range governs policy updates identically for turns with vastly different IG signals. In this paper, we propose A$^2$TGPO (Agentic Turn-Group Policy Optimization with Adaptive Turn-level Clipping), which retains IG as the intrinsic signal but re-designs how it is normalized, accumulated, and consumed: (i) turn-group normalization: normalizes IG within each (prompt, turn-index) group so that each turn is compared only against peers at the same interaction depth; (ii) variance-rescaled discounted accumulation: divides cumulative normalized IG by square root of accumulated terms to keep advantage magnitudes comparable across turn positions; and (iii) adaptive turn-level clipping: modulates each turn's clipping range based on its normalized IG, widening the update region for informative turns and narrowing it for uninformative ones. 

---
# HNC: Leveraging Hard Negative Captions towards Models with Fine-Grained Visual-Linguistic Comprehension Capabilities 

**Authors**: Esra Dönmez, Pascal Tilli, Hsiu-Yu Yang, Thang Vu, Carina Silberer  

**Link**: [PDF](https://arxiv.org/pdf/2605.06157)  

**Abstract**: Image-Text-Matching (ITM) is one of the defacto methods of learning generalized representations from a large corpus in Vision and Language (VL). However, due to the weak association between the web-collected image-text pairs, models fail to show a fine-grained understanding of the combined semantics of these modalities. To address this issue we propose Hard Negative Captions (HNC): an automatically created dataset containing foiled hard negative captions for ITM training towards achieving fine-grained cross-modal comprehension in VL. Additionally, we provide a challenging manually-created test set for benchmarking models on a fine-grained cross-modal mismatch task with varying levels of compositional complexity. Our results show the effectiveness of training on HNC by improving the models' zero-shot capabilities in detecting mismatches on diagnostic tasks and performing robustly under noisy visual input scenarios. Also, we demonstrate that HNC models yield a comparable or better initialization for fine-tuning 

---
# IRC-Bench: Recognizing Entities from Contextual Cues in First-Person Reminiscences 

**Authors**: Yehudit Aperstein, Eden Moran, Alexander Apartsin  

**Link**: [PDF](https://arxiv.org/pdf/2605.06142)  

**Abstract**: When people recount personal memories, they often refer to people, places, and events indirectly, relying on contextual cues rather than explicit names. Such implicit references are central to reminiscence narratives: first-person accounts of lived experience used in therapeutic, archival, and social settings. They pose a difficult computational problem because the intended entity must be inferred from dispersed narrative evidence rather than from a local mention. We introduce IRC-Bench, the Implicit Reminiscence Context Benchmark, for evaluating implicit entity recognition in reminiscence transcripts. The benchmark targets non-locality: entity-identifying cues are distributed across multiple, non-contiguous clauses, unlike named entity recognition, entity linking, or coreference resolution. IRC-Bench comprises 25,136 samples constructed from 12,337 Wiki-data-linked entities across 1,994 transcripts spanning 11 thematic domains. Each sample pairs an Entity-Grounded Narrative, in which the target entity is explicitly mentioned, with an Entity-Elided Narrative, in which direct mentions are removed. We evaluate 19 configurations across LLM generation, dense retrieval, RAG, and fine-tuning. QLoRA-adapted Llama 3.1 8B performs best in the open-world setting (38.94% exact match; 51.59% Jaccard), while fine-tuned DPR leads closed-world retrieval (35.38% Hit@1; 71.49% Hit@10). We release IRC-Bench with data, code, and evaluation tools. 

---
# MemReranker: Reasoning-Aware Reranking for Agent Memory Retrieval 

**Authors**: Chunyu Li, Jingyi Kang, Ding Chen, Mengyuan Zhang, Jiajun Shen, Bo Tang, Xuanhe Zhou, Feiyu Xiong, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.06132)  

**Abstract**: In agent memory systems, the reranking model serves as the critical bridge connecting user queries with long-term memory. Most systems adopt the "retrieve-then-rerank" two-stage paradigm, but generic reranking models rely on semantic similarity matching and lack genuine reasoning capabilities, leading to a problem where recalled results are semantically highly relevant yet do not contain the key information needed to answer the question. This deficiency manifests in memory scenarios as three specific problems. First, relevance scores are miscalibrated, making threshold-based filtering difficult. Second, ranking degrades when facing temporal constraints, causal reasoning, and other complex queries. Third, the model cannot leverage dialogue context for semantic disambiguation. This report introduces MemReranker, a reranking model family (0.6B/4B) built on Qwen3-Reranker through multi-stage LLM knowledge distillation. Multi-teacher pairwise comparisons generate calibrated soft labels, BCE pointwise distillation establishes well-distributed scores, and InfoNCE contrastive learning enhances hard-sample discrimination. Training data combines general corpora with memory-specific multi-turn dialogue data covering temporal constraints, causal reasoning, and coreference resolution. On the memory retrieval benchmark, MemReranker-0.6B substantially outperforms BGE-Reranker and matches open-source 4B/8B models as well as GPT-4o-mini on key metrics. MemReranker-4B further achieves 0.737 MAP, with several metrics on par with Gemini-3-Flash, while maintaining inference latency at only 10--20\% of large models. On finance and healthcare vertical-domain benchmarks, the models preserve generalization capabilities on par with mainstream large-parameter rerankers. 

---
# Uncovering Entity Identity Confusion in Multimodal Knowledge Editing 

**Authors**: Shu Wu, Xiaotian Ye, Xinyu Mou, Dongsheng Liu, Xiaohan Wang, Mengqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06096)  

**Abstract**: Multimodal knowledge editing (MKE) aims to correct the internal knowledge of large vision-language models after deployment, yet the behavioral patterns of post-edit models remain underexplored. In this paper, we identify a systemic failure mode in edited models, termed Entity Identity Confusion (EIC): edited models exhibit an absurd behavior where text-only queries about the original entity's identity unexpectedly return information about the new entity. To rigorously investigate EIC, we construct EC-Bench, a diagnostic benchmark that directly probes how image-entity bindings shift before and after editing. Our analysis reveals that EIC stems from existing methods failing to distinguish between Image-Entity (I-E) binding and Entity-Entity (E-E) relational knowledge in the model, causing models to overfit E-E associations as a shortcut: the image is still perceived as the original entity, with the new entity's name serving only as a spurious identity label. We further explore potential mitigation strategies, showing that constraining edits to the model's I-E processing stage encourages edits to act more faithfully on I-E binding, thereby substantially reducing EIC. Based on these findings, we discuss principled desiderata for faithful MKE and provide methodological guidance for future research. 

---
# Milestone-Guided Policy Learning for Long-Horizon Language Agents 

**Authors**: Zixuan Wang, Yuchen Yan, Hongxing Li, Teng Pan, Dingming Li, Ruiqing Zhang, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.06078)  

**Abstract**: While long-horizon agentic tasks require language agents to perform dozens of sequential decisions, training such agents with reinforcement learning remains challenging. We identify two root causes: credit misattribution, where correct early actions are penalized due to terminal failures, and sample inefficiency, where scarce successful trajectories result in near-total loss of learning signal. We introduce a milestone-guided policy learning framework, BEACON, that leverages the compositional structure of long-horizon tasks to ensure precise credit assignment. BEACON partitions trajectories at milestone boundaries, applies temporal reward shaping within segments to credit partial progress, and estimates advantages at dual scales to prevent distant failures from corrupting the evaluation of local actions. On ALFWorld, WebShop, and ScienceWorld, BEACON consistently outperforms GRPO and GiGPO. Notably, on long-horizon ALFWorld tasks, BEACON achieves 92.9% success rate, nearly doubling GRPO's 53.5%, while improving effective sample utilization from 23.7% to 82.0%. These results establish milestone-anchored credit assignment as an effective paradigm for training long-horizon language agents. Code is available at this https URL. 

---
# Navigating by Old Maps: The Pitfalls of Static Mechanistic Localization in LLM Post-Training 

**Authors**: Hang Chen, Jiaying Zhu, Hongyang Chen, Hongxu Liu, Xinyu Yang, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06076)  

**Abstract**: The "Locate-then-Update" paradigm has become a predominant approach in the post-training of large language models (LLMs), identifying critical components via mechanistic interpretability for targeted parameter updates. However, this paradigm rests on a fundamental yet unverified assumption: can mechanisms derived from current static parameters reliably guide future dynamic parameter updates? To investigate this, we systematically track the structural evolution of Transformer circuits throughout the supervised fine-tuning (SFT) process, revealing the underlying dynamics of task mechanisms. We introduce three novel metrics-Circuit Distance, Circuit Stability, and Circuit Conflict-to analyze circuit evolution across three dimensions: neural migration, semantic stability, and cross-task interference. Our empirical results reveal that circuits inherently exhibit "Free Evolution" during parameter updates. Consequently, static mechanisms extracted from current states inevitably suffer from temporal latency, making them fundamentally inadequate for guiding future states. Moreover, by deconstructing the "illusion of effectiveness" in existing methods, this work underscores the necessity of "foresight" in mechanistic localization and proposes a predictive framework for future research. 

---
# More Aligned, Less Diverse? Analyzing the Grammar and Lexicon of Two Generations of LLMs 

**Authors**: Adrián Gude, Roi Santos-Ríos, Francis Bond, Dan Flickinger, Carlos Gómez-Rodríguez, Olga Zamaraeva  

**Link**: [PDF](https://arxiv.org/pdf/2605.06030)  

**Abstract**: This study contributes to a growing line of research in comparing LLM-generated texts with human-authored text, in this case, English news text. We focus in particular on the evaluation of syntactic properties through formal grammar frameworks. Our analysis compares two generations of LLMs in the context of two human-authored English news datasets from two different years. Employing the Head-Driven Phrase Structure Grammar (HPSG) formalism, we investigate the distributions of syntactic structures and lexical types of AI-generated texts and contrast them with the corresponding distributions in the human-authored New York Times (NYT) articles. We use diversity metrics from ecology and information theory to quantify variation in grammatical constructions and lexical types. We show that English news text has changed little in the given time frame, while newer LLMs display reduced syntactic and, especially, lexical diversity compared to older, non-instruction-tuned models. These findings point to future work in studying effects of instruction tuning, which, while enhancing coherence and adherence to prompts, may narrow the expressive range of model output. 

---
# PersonaKit (PK): A Plug-and-Play Platform for User Testing Diverse Roles in Full-Duplex Dialogue 

**Authors**: Hyunbae Jeon, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.06007)  

**Abstract**: As spoken dialogue systems expand beyond traditional assistant roles to encompass diverse personas -- such as authoritative instructors, uncooperative merchants, or distracted workers -- they require distinct, human-like turn-taking behaviors to maintain psychological immersion. However, current full-duplex systems often default to a rigid, overly accommodating ``always-yield'' policy during overlapping speech, which severely undermines character consistency for non-submissive roles. Evaluating alternative, persona-specific turn-taking strategies through empirical user studies is challenging because building real-time full-duplex test environments requires substantial engineering overhead. To address this, we present PersonaKit (PK), an open-source, low-latency web platform for the rapid prototyping and evaluation of conversational agents. Using intuitive JSON configurations, researchers can define personas, specify probabilistic interruption-handling behaviors (e.g., yield, hold, bridge, or override), and automatically deploy comparative A/B surveys. Through an in-the-wild evaluation with 8 distinct personas, we demonstrate that PersonaKit provides an extensible, end-to-end framework for studying complex sociolinguistic behaviors in next-generation spoken agents. 

---
# From Articles to Premises: Building PrimeFacts, an Extraction Methodology and Resource for Fact-Checking Evidence 

**Authors**: Premtim Sahitaj, Jawan Kolanowski, Ariana Sahitaj, Veronika Solopova, Max Upravitelev, Daniel Röder, Iffat Maab, Junichi Yamagishi, Sebastian Möller, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2605.06006)  

**Abstract**: Fact-checking articles encode rich supporting evidence and reasoning, yet this evidence remains largely inaccessible to automated verification systems due to unstructured presentation. We introduce PrimeFacts, a methodology and resource for extracting fine-grained evidence from full fact-checking articles. We compile 13,106 PolitiFact articles with claims, verdicts, and all referenced sources, and we identify 49,718 in-article hyperlinks as natural anchors to pinpoint key evidence. Our framework leverages large language models (LLMs) to rewrite these anchor sentences into stand-alone, context-independent premises and investigates the extraction of additional implicit evidence. In evaluations on cross-article evidence retrieval and claim verification, the extracted premises substantially improve performance. Decontextualized evidence yields higher retrievability, achieving up to a 30 percent relative gain in Mean Reciprocal Rank over verbatim sentences, and using the evidence for verdict prediction raises Macro-F1 by 10-20 points over the baseline. These gains are consistent across different verdict granularities (2-class vs. 5-class) and model architectures. A qualitative analysis indicates that the decontextualized premises remain faithful to the original sources. Our work highlights the promise of reusing fact-checkers' evidence for automation and provides a large-scale resource of structured evidence from real-world fact-checks. 

---
# Tatarstan Toponyms: A Bilingual Dataset and Hybrid RAG System for Geospatial Question Answering 

**Authors**: Mullosharaf K. Arabov  

**Link**: [PDF](https://arxiv.org/pdf/2605.05962)  

**Abstract**: This paper addresses automatic geospatial question answering over multilingual toponymic data. An original bilingual dataset of toponyms of the Republic of Tatarstan is introduced, comprising 9,688 structured records with linguistic, etymological, administrative, and coordinate information (93.1% georeferenced). Based on this dataset, a question-answering corpus of approximately 39,000 question-context-answer triples is constructed with guaranteed answer localization. A hybrid retriever integrates dense semantic indexing (multilingual-e5-large) with geospatial filtering via KD-trees and haversine distance. On 500 test queries, the hybrid search achieves Recall@1=0.988, Recall@5=1.000, and MRR=0.994, significantly outperforming BM25 and purely spatial methods. Among tested reader architectures (RuBERT, XLM-RoBERTa-large, T5-RUS), XLM-RoBERTa-large attains the best quality: EM=0.992, F1=0.994. On raw outputs, RuBERT models fail on coordinate questions (F1=0) while XLM-RoBERTa-large reaches F1=0.984; however, simple post-processing eliminates numerical gaps and restores RuBERT accuracy to 100%. This discrepancy stems from tokenization differences and pre-training corpora composition. All resources (dataset, QA corpus, model weights, web demo) are openly published on Hugging Face. Results apply to geospatial QA services, geocoding, and digital humanities in multilingual regions. 

---
# TableVista: Benchmarking Multimodal Table Reasoning under Visual and Structural Complexity 

**Authors**: Zheyuan Yang, Liqiang Shang, Junjie Chen, Xun Yang, Chenglong Xu, Bo Yuan, Chenyuan Jiao, Yaoru Sun, Yilun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.05955)  

**Abstract**: We introduce TableVista, a comprehensive benchmark for evaluating foundation models in multimodal table reasoning under visual and structural complexity. TableVista consists of 3,000 high-quality table reasoning problems, where each instance is expanded into 10 distinct visual variants through our multi-style rendering and transformation pipeline. This process encompasses diverse scenario styles, robustness perturbations, and vision-only configurations, culminating in 30,000 multimodal samples for a multi-dimensional evaluation. We conduct an extensive evaluation of 29 state-of-the-art open-source and proprietary foundation models on TableVista. Through comprehensive quantitative and qualitative analysis, we find that while evaluated models remain largely stable across diverse rendering styles, they exhibit pronounced performance degradation on complex structural layouts and vision-only settings, revealing that current models struggle to maintain reasoning consistency when structural complexity combines with visually integrated presentations. These findings highlight critical gaps in current multimodal capabilities, providing insights for advancing more robust and reliable table understanding models. 

---
# Hallucination as an Anomaly: Dynamic Intervention via Probabilistic Circuits 

**Authors**: Erik Nielsen, Elia Cunegatti, Marcus Vukojevic, Giovanni Iacca  

**Link**: [PDF](https://arxiv.org/pdf/2605.05953)  

**Abstract**: One of the most critical challenges in Large Language Models is their tendency to hallucinate, i.e., produce factually incorrect responses. Existing approaches show promising results in terms of hallucination correction, but still suffer from a main limitation: they apply corrections indiscriminately to every token, corrupting also the originally correct generations. To overcome this drawback, we propose PCNET, a Probabilistic Circuit trained as a tractable density estimator over the LLM residual stream. The method detects hallucinations as geometric anomalies on the factual manifold, which is done via exact Negative Log-Likelihood computation, hence without the need for sampling, external verifiers, or weight modifications, as in existing techniques. To demonstrate its effectiveness, we exploit PCNET as a dynamic gate that distinguishes hallucinated from factual hidden states at each decoding step. This triggers our second main contribution, PC-LDCD (Probabilistic Circuit Latent Density Contrastive Decoding), only when the latent geometry deviates from factual regions, while leaving correct generations untouched. Across four LLMs, ranging from 1B to 8B models, and four benchmarks covering conversational reasoning, knowledge-intensive QA, reading comprehension, and truthfulness, PCNET achieves near-perfect hallucination detection across CoQA, SQuAD v2.0, and TriviaQA, with AUROC reaching up to 99%. Moreover, PC-LDCD obtains the highest True+Info, MC2, and MC3 scores on TruthfulQA in three out of four models, in comparison with state-of-the-art baselines, while reducing the mean corruption rate to 53.7% and achieving a preservation rate of 79.3%. Our proposed method is publicly available on GitHub. 

---
# Lightweight Stylistic Consistency Profiling: Robust Detection of LLM-Generated Textual Content for Multimedia Moderation 

**Authors**: Siyuan Li, Aodu Wulianghai, Xi Lin, Xibin Yuan, Qinghua Mao, Guangyan Li, Xiang Chen, Jun Wu, Jianhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.05950)  

**Abstract**: The increasing prevalence of Large Language Models (LLMs) in content creation has made distinguishing human-written textual content from LLM-generated counterparts a critical task for multimedia moderation. Existing detectors often rely on statistical cues or model-specific heuristics, making them vulnerable to paraphrasing and adversarial manipulations, and consequently limiting their robustness and interpretability. In this work, we proposeLiSCP , a novel lightweight stylistic consistency profiling method for robust detection of LLM-generated textual content, focusing on feature stability under adversarial manipulation. Our approach constructs a consistency profile that combines discrete stylistic features with continuous semantic signals, leveraging stylistic stability across multimodal-guided paraphrased text variants. Experiments spanning real-world multimedia news and movie datasets and conventional text domains demonstrate that LiSCP achieves superior performance on in-domain detection and outperforms existing approaches by up to 11.79% in cross-domain settings. Additionally,it demonstrates notable robustness under adversarial scenarios, including adversarial attacks and hybrid human-AI settings. 

---
# Minimizing Modality Gap from the Input Side: Your Speech LLM Can Be a Prosody-Aware Text LLM 

**Authors**: Wenqian Cui, Xiao-Hui Li, Daxin Tan, Qiyong Zheng, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2605.05927)  

**Abstract**: Speech large language models (SLMs) are typically built from text large language model (TLM) checkpoints, yet they still suffer from a substantial modality gap. Prior work has mainly attempted to reduce this gap from the output side by making speech generation more text-like, but the gap remains. We argue that the key remaining bottleneck lies on the input side. We propose TextPro-SLM, an SLM that makes spoken input more closely resemble that of a prosody-aware text LLM. TextPro-SLM combines WhisperPro, a unified speech encoder that produces synchronized text tokens and prosody embeddings, with an LLM backbone trained to preserve the semantic capabilities of the original TLM while learning paralinguistic understanding. Experiments show that TextPro-SLM achieves the lowest modality gap among leading SLMs at both 3B and 7B scales, while also delivering strong overall performance on paralinguistic understanding tasks. These gains are achieved with only roughly 1,000 hours of LLM training audio, suggesting that reducing the modality gap from the input side is both effective and data-efficient. 

---
# Logic-Regularized Verifier Elicits Reasoning from LLMs 

**Authors**: Xinyu Wang, Changzhi Sun, Lian Cheng, Yuanbin Wu, Dell Zhang, Xiaoling Wang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.05893)  

**Abstract**: Verifiers are crucial components for enhancing modern LLMs' reasoning capability. Typicalverifiers require resource-intensive superviseddataset construction, which is costly and faceslimitations in data diversity. In this paper, wepropose LOVER, an unsupervised verifier regularized by logical rules. LOVER treats theverifier as a binary latent variable, utilizinginternal activations and enforcing three logical constraints on multiple reasoning paths:negation consistency, intra-group consistency,and inter-group consistency (grouped by thefinal answer). By incorporating logical rulesas priors, LOVER can leverage unlabeled examples and is directly compatible with any offthe-shelf LLMs. Experiments on 10 datasetsdemonstrate that LOVER significantly outperforms unsupervised baselines, achieving performance comparable to the supervised verifier(reaching its 95% level on average). The sourcecode is publicly available at this https URL. 

---
# Beyond Steering Vector: Flow-based Activation Steering for Inference-Time Intervention 

**Authors**: Zehao Jin, Ruixuan Deng, Junran Wang, Xinjie Shen, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05892)  

**Abstract**: Activation steering has emerged as a promising alternative for controlling language-model behavior at inference time by modifying intermediate representations while keeping model parameters frozen. However, large-scale evaluations such as AxBench show that existing steering methods are often outperformed by simple in-context prompting and generalize poorly to unseen concepts. We hypothesize that these limitations arise from unvalidated simplifying assumptions shared across prior methods, which typically restrict steering interventions to fixed, single-step, position-invariant transforms. We propose FLAS (Flow-based Activation Steering), which learns a general, concept-conditioned velocity field $v_t(h,t,c)$ that transports unsteered activations to steered ones without relying on these assumptions. On AxBench, FLAS is the first learned method to consistently outperform prompting, reaching held-out harmonic means of $1.015$ on Gemma-2-2B-IT and $1.113$ on Gemma-2-9B-IT without per-concept tuning. Analysis of the learned flow shows curved, multi-step, token-varying trajectories, which suggests that previous hypotheses on activation space geometry might be incomplete. 

---
# Evaluation Awareness in Language Models Has Limited Effect on Behaviour 

**Authors**: Amelie Knecht, Lucas Florin, Thilo Hagendorff  

**Link**: [PDF](https://arxiv.org/pdf/2605.05835)  

**Abstract**: Large reasoning models (LRMs) sometimes note in their chain of thought (CoT) that they may be under evaluation. Researchers worry that this verbalised evaluation awareness (VEA) causes models to adapt their outputs strategically, optimising for perceived evaluation criteria, which, for instance, can make models appear safer than they actually are. However, whether VEA actually has this effect is largely unknown. We tested this across open-weight LRMs and benchmarks covering safety, alignment, moral reasoning, and political opinion. We tested this both on-policy, sampling multiple CoTs per item and comparing those that spontaneously contained VEA against those that did not, and off-policy, using model prefilling to inject evaluation-aware sentences where missing and remove them where present, with subsequent resampling. VEA has limited effect on model behaviour: injecting VEA into CoTs produces near-zero effects ($\omega \leq 0.06$), removing it causes small shifts ($\omega \leq 0.12$) and spontaneously occurring VEA shifts answer distributions by at most 3.7 percentage points ($\omega \leq 0.31$). Our findings call for caution when interpreting high VEA rates as evidence of strategic behaviour or alignment tampering. Evaluation awareness may pose a smaller safety risk than the current literature assumes. 

---
# Estimating the Black-box LLM Uncertainty with Distribution-Aligned Adversarial Distillation 

**Authors**: Huizi Cui, Huan Ma, Qilin Wang, Yuhang Gao, Changqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05777)  

**Abstract**: Large language models (LLMs) have progressed rapidly in complex reasoning and question answering, yet LLM hallucination remains a central bottleneck that hinders practical deployment, especially for commercial black-box LLMs accessible only via APIs. Existing uncertainty quantification methods typically depend on computationally expensive multiple sampling or internal parameters, which prevents real-time estimation and fails to capture information implicit in the black-box reasoning process. To address this issue, we propose Distribution-Aligned Adversarial Distillation (DisAAD), which introduces a generation-discrimination architecture to guide a lightweight proxy model to learn the high-quality regions of the output distribution of the black-box LLM, thus effectively endowing it with the ability to know whether the black-box LLM knows or not. Subsequently, we use the proxy model to reproduce the specific responses of the black-box LLM and estimate the corresponding uncertainty based on evidence learning. Extensive experiments have verified the effectiveness and promise of our proposed method, indicating that a proxy model even one that only accounts for 1\% of the target LLM's size can achieve reliable uncertainty quantification. 

---
# BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models 

**Authors**: Xin Gao, Ruiyi Zhang, Meixi Du, Peijia Qin, Pengtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.05758)  

**Abstract**: Despite the success of large language models (LLMs) on general-purpose tasks, their performance in highly specialized domains such as biomedicine remains unsatisfactory. A key limitation is the inability of LLMs to effectively leverage biomedical tools, which clinical experts and biomedical researchers rely on extensively in daily workflows. While recent general-domain tool-calling datasets have substantially improved the capabilities of LLM agents, existing efforts in the biomedical domain largely rely on in-context learning and restrict models to a small set of tools. To address this gap, we introduce BioTool, a comprehensive biomedical tool-calling dataset designed for fine-tuning LLMs. BioTool comprises 34 frequently used tools collected from the NCBI, Ensembl, and UniProt databases, along with 7,040 high-quality, human-verified query-API call pairs spanning variation, genomics, proteomics, evolution, and general biology. Fine-tuning a 4-billion-parameter LLM on BioTool yields substantial improvements in biomedical tool-calling performance, outperforming cutting-edge commercial LLMs such as GPT-5.1. Furthermore, human expert evaluations demonstrate that integrating a BioTool-fine-tuned tool caller significantly improves downstream answer quality compared to the same LLM without tool usage, highlighting the effectiveness of BioTool in enhancing the biomedical capabilities of LLMs. The full dataset and evaluation code are available at this https URL 

---
# Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning 

**Authors**: Bing Wang, Ximing Li, Changchun Li, Jinjin Chi, Gang Niu, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2605.05676)  

**Abstract**: Recently, the prominent performance of large language models (LLMs) has been largely driven by multi-task instruct-tuning. Unfortunately, this training paradigm suffers from a key issue, named cross-task interference, due to conflicting gradients over shared parameters among different tasks. Some previous methods mitigate this issue by isolating task-specific parameters, e.g., task-specific neuron selection and mixture-of-experts. In this paper, we empirically reveal that the cross-task interference still exists for the existing solutions because of many parameters also shared by different tasks, and accordingly, we propose a novel solution, namely Basic Abilities Decomposition for multi-task Instruct-Tuning (BADIT). Specifically, we empirically find that certain parameters are consistently co-activated, and that co-activated parameters naturally organize into base groups. This motivates us to analogize that LLMs encode several orthogonal basic abilities, and that any task can be represented as a linear combination of these abilities. Accordingly, we propose BADIT that decomposes LLM parameters into orthogonal high-singular-value LoRA experts representing basic abilities, and dynamically enforces their orthogonality during training via spherical clustering of rank-1 components. We conduct extensive experiments on the SuperNI benchmark with 6 LLMs, and empirical results demonstrate that BADIT can outperform SOTA methods and mitigate the degree of cross-task interference. 

---
# XL-SafetyBench: A Country-Grounded Cross-Cultural Benchmark for LLM Safety and Cultural Sensitivity 

**Authors**: Dasol Choi, Eugenia Kim, Jaewon Noh, Sang Seo, Eunmi Kim, Myunggyo Oh, Yunjin Park, Brigitta Jesica Kartono, Josef Pichlmeier, Helena Berndt, Sai Krishna Mendu, Glenn Johannes Tungka, Özlem Gökçe, Suresh Gehlot, Katherine Pratt, Amanda Minnich, Haon Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.05662)  

**Abstract**: Current LLM safety benchmarks are predominantly English-centric and often rely on translation, failing to capture country-specific harms. Moreover, they rarely evaluate a model's ability to detect culturally embedded sensitivities as distinct from universal harms. We introduce XL-SafetyBench. a suite of 5,500 test cases across 10 country-language pairs, comprising a Jailbreak Benchmark of country-grounded adversarial prompts and a Cultural Benchmark where local sensitivities are embedded within innocuous requests. Each item is constructed via a multi-stage pipeline that combines LLM-assisted discovery, automated validation gates, and dual independent native-speaker annotators per country. To distinguish principled refusal from comprehension failure, we evaluate Attack Success Rate (ASR) alongside two complementary metrics we introduce: Neutral-Safe Rate (NSR) and Cultural Sensitivity Rate (CSR). Evaluating 10 frontier and 27 local LLMs reveals two key findings. First, jailbreak robustness and cultural awareness do not show a coupled relationship among frontier models, so a composite safety score obscures per-axis variation. Second, local models exhibit a near-linear ASR-NSR trade-off (r = -0.81), indicating that their apparent safety reflects generation failure rather than genuine alignment. XL-SafetyBench enables more nuanced, cross-cultural safety evaluation in the multilingual era. 

---
# Negative Before Positive: Asymmetric Valence Processing in Large Language Models 

**Authors**: Sohan Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2605.05653)  

**Abstract**: Mechanistic interpretability has revealed how concepts are encoded in large language models (LLMs), but emotional content remains poorly understood at the mechanistic level. We study whether LLMs process emotional valence through dedicated internal structure or through surface token matching. Using activation patching and steering on open-source LLMs, we find that negative and positive valence are processed at different network depths. Negative outcomes localize to early layers while positive outcomes peak at mid-to-late layers. Holding topic fixed while flipping valence produces sign-opposite responses, ruling out topic detection. Steering with the good-news direction at the identified layers shifts neutral prompts toward positive valence, showing these layers encode valence as a manipulable direction. Emotional valence in LLMs is localized, causal and steerable, making it a concrete target for interpretability-based oversight. 

---
# One Turn Too Late: Response-Aware Defense Against Hidden Malicious Intent in Multi-Turn Dialogue 

**Authors**: Xinjie Shen, Rongzhe Wei, Peizhi Niu, Haoyu Wang, Ruihan Wu, Eli Chien, Bo Li, Pin-Yu Chen, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.05630)  

**Abstract**: Hidden malicious intent in multi-turn dialogue poses a growing threat to deployed large language models (LLMs). Rather than exposing a harmful objective in a single prompt, increasingly capable attackers can distribute their intent across multiple benign-looking turns. Recent studies show that even modern commercial models with advanced guardrails remain vulnerable to such attacks despite advances in safety alignment and external guardrails. In this work, we address this challenge by detecting the earliest turn at which delivering the candidate response would make the accumulated interaction sufficient to enable harmful action. This objective requires precise turn-level intervention that identifies the harm-enabling closure point while avoiding premature refusal of benign exploratory conversations. To further support training and evaluation, we construct the Multi-Turn Intent Dataset (MTID), which contains branching attack rollouts, matched benign hard negatives, and annotations of the earliest harm-enabling turns. We show that MTID helps enable a turn-level monitor TurnGate, which substantially outperforms existing baselines in harmful-intent detection while maintaining low over-refusal rates. TurnGate further generalizes across domains, attacker pipelines, and target models. Our code is available at this https URL. 

---
# When2Speak: A Dataset for Temporal Participation and Turn-Taking in Multi-Party Conversations for Large Language Models 

**Authors**: Vihaan Nama, Shreya Mendi, Zian Ye, Brinnae Bent  

**Link**: [PDF](https://arxiv.org/pdf/2605.05626)  

**Abstract**: Large Language Models (LLMs) excel at generating contextually appropriate responses but remain poorly calibrated for multi-party conversations, where deciding when to speak is as critical as what to say. In such settings, naively responding at every turn leads to excessive interruptions and degraded conversational coherence. We introduce When2Speak, a grounded synthetic dataset and four-stage generation pipeline for learning intervention timing in group interactions. The dataset comprises over 215,000 examples derived from 16,000 conversations involving 2-6 speakers, spanning diverse conversational styles, tones, and participant dynamics, and explicitly modeling SPEAK vs. SILENT decisions at each turn. Our pipeline combines real-world grounding, structured augmentation, controlled transcript synthesis, and fine-tuning-ready supervision, and is fully open-sourced to support reproducibility and adaptation to domain-specific conversational norms. Across multiple model families, supervised fine-tuning (SFT) on When2Speak significantly outperforms zero-shot baselines (e.g., the average Macro F1 increase across 4B+ parameter models was 60%, with the largest increase being 120%). However, SFT-trained models remain systematically over-conservative, missing nearly half of warranted interventions as seen through the Missed Intervention Rate (MIR), which was on average 0.50 and is noticed even at larger model sizes. To address this limitation, we apply reinforcement learning with asymmetric reward shaping, which reduces MIR to 0.186-0.218 and increases recall from 0.479 to 0.78-0.81. Our findings establish that temporal participation is a distinct and trainable dimension of conversational intelligence, and that grounded synthetic data provides an effective and scalable pathway for enabling LLMs to participate more naturally and appropriately in multi-party interactions. 

---
# The Cost of Context: Mitigating Textual Bias in Multimodal Retrieval-Augmented Generation 

**Authors**: Hoin Jung, Xiaoqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05594)  

**Abstract**: While Multimodal Large Language Models (MLLMs) are increasingly integrated with Retrieval-Augmented Generation (RAG) to mitigate hallucinations, the introduction of external documents can conceal severe failure modes at the instance level. We identify and formalize the phenomenon of recorruption, where the introduction of even perfectly accurate "oracle" context causes a capable model to abandon an initially correct prediction. Through a mechanistic diagnosis of internal attention matrices, we show that recorruption is driven by a two-fold attentional collapse: (1) visual blindness, characterized by the systemic suppression of visual attention mass ($M_{vis}$) and sharpness ($S_{vis}$), and (2) a structural positional bias that forces the model to prioritize boundary tokens over semantic relevance. Our analysis reveals an Illusion of Success, demonstrating that many seemingly correct RAG outcomes are merely positional coincidences where the model's textual copying bias happens to align with the ground-truth location. To address these vulnerabilities, we propose Bottleneck Attention Intervention for Recovery (BAIR), a parameter-free, inference-time framework that restores visual saliency and applies position-aware penalties to textual distractors. Across medical factuality, social fairness, and geospatial benchmarks, BAIR successfully restores multimodal grounding and improves diagnostic reliability without requiring model retraining or fine-tuning. 

---
# A Few Good Clauses: Comparing LLMs vs Domain-Trained Small Language Models on Structured Contract Extraction 

**Authors**: Nicole Lincoln, Nick Whitehouse, Jaron Mar, Rivindu Perera  

**Link**: [PDF](https://arxiv.org/pdf/2605.05532)  

**Abstract**: This paper evaluates whether a domain trained Small Language Model (SLM) can outperform frontier Large Language Models on structured contract extraction at radically lower cost. We test Olava Extract, a self hosted legal domain Mixture of Experts model, against five frontier models.
Olava Extract achieved the strongest aggregate performance in the study, with a macro F1 of 0.812 and a micro F1 of 0.842, while reducing inference cost by 78% to 97% compared with the frontier models tested. It also achieved the highest precision scores, producing fewer hallucinated and unsupported extractions, an important distinction in legal workflows where hallucinations create operational risk and downstream review burden.
The findings shows that high performing, human comparable legal AI no longer requires the largest externally hosted models. More broadly, they challenge the assumption that commercially valuable enterprise AI capability must remain tied to ever larger models, massive infrastructure expenditure, and centrally hosted providers. 

---
# Chainwash: Multi-Step Rewriting Attacks on Diffusion Language Model Watermarks 

**Authors**: Mohd Ruhul Ameen, Akif Islam, Nadim Mahmud, Md. Ekramul Hamid  

**Link**: [PDF](https://arxiv.org/pdf/2605.05503)  

**Abstract**: Statistical watermarking is a common approach for verifying whether text was written by a language model. Most existing schemes assume autoregressive generation, where tokens are produced left to right and contextual hashing is well defined. Diffusion language models generate text by denoising tokens in arbitrary order, so these schemes cannot be applied directly. A recent watermark by Gloaguen et al. addresses this gap for LLaDA 8B Instruct and reports true positive detection above 99%. This paper studies what happens when watermarked text is rewritten not once but several times. Using the same watermark configuration, 1,605 watermarked completions of about 300 tokens each are produced across five WaterBench domains. Each completion is rewritten by four open weight language models, from 1.5B to 8B parameters, none of which know the watermark key. Five rewrite styles are tested: paraphrase, humanize, simplify, academic, and summarize expand. Each style is chained for up to five hops, producing 160,500 rewritten texts in total. The watermark is detected on 87.9% of the original outputs at the standard significance threshold. After a single rewrite, detection falls to between 14% and 41% depending on the rewriter and style. After five chained rewrites, detection falls to 4.86%, meaning 94.76% of the originally detected texts are no longer flagged. After three rewrites, the detector score has dropped 86% of the way from its watermarked baseline toward the null distribution. Repeated rewriting is therefore a much stronger attack than a single rewrite, and the result holds across all four rewriters tested. 

---
# ReaComp: Compiling LLM Reasoning into Symbolic Solvers for Efficient Program Synthesis 

**Authors**: Atharva Naik, Yash Mathur, Prakam, Carolyn Rose, David Mortensen  

**Link**: [PDF](https://arxiv.org/pdf/2605.05485)  

**Abstract**: LLMs can solve program synthesis tasks but remain inefficient and unreliable on hard instances requiring large combinatorial search. Given a small set of reasoning traces, we use coding agents to compile them into reusable symbolic program synthesizers over constrained DSLs. The resulting solvers require no LLM calls at test time and are strong standalone systems: symbolic solver ensembles reach 91.3% accuracy on PBEBench-Lite and 84.7% on PBEBench-Hard, outperforming LLMs with test-time scaling for the latter by +16.3 percentage points at zero LLM inference cost. They also complement LLM search, improving PBEBench-Hard accuracy from 68.4% to 85.8% while reducing reported token usage by 78%, and raising SLR-Bench hard-tier accuracy from 34.4% to 58.0% in a neuro-symbolic hybrid setting. Compared to directly using coding agents as per-instance solvers, induced solvers are substantially more Pareto-efficient, amortizing a small one-time construction cost over many zero-token executions. Finally, most solvers transfer zero-shot to a real historical linguistics task - predicting sound changes in natural language data - reaching 80.1% accuracy under ensembling and recovering some plausible linguistic rules. Together, these results show that reasoning traces can be compiled into reusable symbolic solvers that solve many tasks directly, complement LLM inference on hard cases, and provide a scalable route to domain-general solver induction. We release code and data for reproducibility. 

---
# SLAM: Structural Linguistic Activation Marking for Language Models 

**Authors**: Fabrice Harel-Canada, Amit Sahai  

**Link**: [PDF](https://arxiv.org/pdf/2605.05443)  

**Abstract**: LLM watermarks must be detectable without compromising text quality, yet most existing schemes bias the next-token distribution and pay for detection with measurable quality loss. We present SLAM (Structural Linguistic Activation Marking), a novel white-box watermarking scheme that sidesteps this cost by writing the mark into structural geometry rather than token frequencies: sparse autoencoders identify residual-stream directions encoding linguistic structure (e.g., voice, tense, clause order), and we causally steer those directions at generation time, leaving lexical sampling and semantics unconstrained. On Gemma-2 2B and 9B, SLAM achieves 100% detection accuracy with a quality cost of only 1-2 reward points - compared to 7.5-11.5 for KGW, EWD, and Unigram - with naturalness and diversity preserved at near-unwatermarked levels across both models. The trade-off is a complementary robustness profile: SLAM resists word-level edits but is vulnerable to paraphrase that restructures syntax (at a quality cost), the converse of token-distribution methods. 

---
# Generating Query-Focused Summarization Datasets from Query-Free Summarization Datasets 

**Authors**: Yllias Chali, Deen Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2605.05392)  

**Abstract**: Large-scale datasets are widely used to perform summarization tasks, but they may not include queries alongside documents and summaries. In the search for suitable datasets for Query-Focused Summarization (QFS), we identify two research questions: Is it possible to automatically generate evidence-based query keywords from query-free datasets? Does evidence-based query generation support the QFS task? This paper proposes an evidence-based model to generate queries from query-free datasets. To evaluate our model intrinsically, we compare the similarity between the original queries and the system-generated queries of two QFS datasets. We also perform summarization tasks using different pre-trained models, as well as a state-of-the-art (SOTA) QFS model, to measure the extrinsic performance of our query generation approach. Experimental results indicate that summaries generated using evidence-based queries achieve competitive ROUGE scores compared to those generated from the original queries. 

---
# Counterargument for Critical Thinking as Judged by AI and Humans 

**Authors**: Tosin Adewumi, Marcus Liwicki, Foteini Simistira Liwicki, Lama Alkhaled, Hamam Mokayed, Esra Sümer-Arpak  

**Link**: [PDF](https://arxiv.org/pdf/2605.05353)  

**Abstract**: This intervention study investigates the use of counterarguments in writing for critical thinking by students in the context of Generative AI (GenAI). This is especially as risks of cheating and cognitive offloading exist with the use of GenAI. We presented 36 students in a particular university course with 4 carefully selected thesis statements (from a set of popular debates) to write about anyone of them. We used six established rubrics (focus, logic, content, style, correctness and reference) to conduct three human assessments (two student peer-reviews and one experienced teacher) per writeup on a 5-point Likert scale for all the qualified samples (n) of 35 submissions (after disqualifying one for irregularity). Using the same rubrics and guidelines, we also assessed the submissions using six frontier LLMs as judges. Our mixed-method design included qualitative open-ended feedback per assessment and quantitative methods. The results reveal that (1) the students' self-written counterarguments to AI-generated content contains logic, among other things, which is a key component of critical thinking, and (2) GenAI can be successfully used at scale to assess students' written work, based on clear rubrics, and these assessments generally align with human assessments as shown with Gwets AC2 inter-rater reliability values of 0.33 for all the models except one. 

---
# AdaGATE: Adaptive Gap-Aware Token-Efficient Evidence Assembly for Multi-Hop Retrieval-Augmented Generation 

**Authors**: Yilin Guo, Yinshan Wang, Yixuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05245)  

**Abstract**: Retrieval-augmented generation (RAG) remains brittle on multi-hop questions in realistic deployment settings, where retrieved evidence may be noisy or redundant and only limited context can be passed to the generator. Existing controllers address parts of this problem, but typically either expand context additively, select from a fixed top-k set, or optimize relevance without explicitly repairing missing bridge facts. We propose AdaGATE, a training-free evidence controller for multi-hop RAG that frames evidence selection as a token-constrained repair problem. AdaGATE combines entity centric gap tracking, targeted micro-query generation, and a utility based selection mechanism that balances gap coverage, corroboration, novelty, redundancy, and direct question relevance. We evaluate AdaGATE on HotpotQA under clean, redundancy, and noise injected retrieval conditions. Across all three settings, AdaGATE achieves the best evidence F1 among the compared controllers, reaching 62.3% on clean data and 71.2% under redundancy injection, while using 2.6x fewer input tokens than Adaptive-k. These results suggest that explicit gap-aware repair, combined with token-efficient evidence selection, improves robustness in multi-hop RAG under imperfect retrieval. Our code and evaluation pipeline are available at this https URL. 

---
# Verifier-Backed Hard Problem Generation for Mathematical Reasoning 

**Authors**: Yuhang Lai, Jiazhan Feng, Yee Whye Teh, Ning Miao  

**Link**: [PDF](https://arxiv.org/pdf/2605.06660)  

**Abstract**: Large Language Models (LLMs) demonstrate strong capabilities for solving scientific and mathematical problems, yet they struggle to produce valid, challenging, and novel problems - an essential component for advancing LLM training and enabling autonomous scientific research. Existing problem generation approaches either depend on expensive human expert involvement or adopt naive self-play paradigms, which frequently yield invalid problems due to reward hacking. This work introduces VHG, a verifier-enhanced hard problem generation framework built upon three-party self-play. By integrating an independent verifier into the conventional setter-solver duality, our design constrains the setter's reward to be jointly determined by problem validity (evaluated by the verifier) and difficulty (assessed by the solver). We instantiate two verifier variants: a Hard symbolic verifier and a Soft LLM-based verifier, with evaluations conducted on indefinite integral tasks and general mathematical reasoning tasks. Experimental results show that VHG substantially outperforms all baseline methods by a clear margin. 

---
# When No Benchmark Exists: Validating Comparative LLM Safety Scoring Without Ground-Truth Labels 

**Authors**: Sushant Gautam, Finn Schwall, Annika Willoch Olstad, Fernando Vallecillos Ruiz, Birk Torpmann-Hagen, Sunniva Maria Stordal Bjørklund, Leon Moonen, Klas Pettersen, Michael A. Riegler  

**Link**: [PDF](https://arxiv.org/pdf/2605.06652)  

**Abstract**: Many deployments must compare candidate language models for safety before a labeled benchmark exists for the relevant language, sector, or regulatory regime. We formalize this setting as benchmarkless comparative safety scoring and specify the contract under which a scenario-based audit can be interpreted as deployment evidence. Scores are valid only under a fixed scenario pack, rubric, auditor, judge, sampling configuration, and rerun budget. Because no labels are available, we replace ground-truth agreement with an instrumental-validity chain: responsiveness to a controlled safe-versus-abliterated contrast, dominance of target-driven variance over auditor and judge artifacts, and stability across reruns.
We instantiate the chain in SimpleAudit, a local-first scoring instrument, and validate it on a Norwegian safety pack. Safe and abliterated targets separate with AUROC values between 0.89 and 1.00, target identity is the dominant variance component ($\eta^2 \approx 0.52$), and severity profiles stabilize by ten reruns. Applying the same chain to Petri shows that it admits both tools. The substantial differences arise upstream of the chain, in claim-contract enforcement and deployment fit. A Norwegian public-sector procurement case comparing Borealis and Gemma 3 demonstrates the resulting evidence in practice: the safer model depends on scenario category and risk measure. Consequently, scores, matched deltas, critical rates, uncertainty, and the auditor and judge used must be reported together rather than collapsed into a single ranking. 

---
# Recursive Agent Optimization 

**Authors**: Apurva Gandhi, Satyaki Chakraborty, Xiangjun Wang, Aviral Kumar, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2605.06639)  

**Abstract**: We introduce Recursive Agent Optimization (RAO), a reinforcement learning approach for training recursive agents: agents that can spawn and delegate sub-tasks to new instantiations of themselves recursively. Recursive agents implement an inference-time scaling algorithm that naturally allows agents to scale to longer contexts and generalize to more difficult problems via divide-and-conquer. RAO provides a method to train models to best take advantage of such recursive inference, teaching agents when and how to delegate and communicate. We find that recursive agents trained in this way enjoy better training efficiency, can scale to tasks that go beyond the model's context window, generalize to tasks much harder than the ones the agent was trained on, and can enjoy reduced wall-clock time compared to single-agent systems. 

---
# Can RL Teach Long-Horizon Reasoning to LLMs? Expressiveness Is Key 

**Authors**: Tianle Wang, Zhaoyang Wang, Guangchen Lan, Xinpeng Wei, Sipeng Zhang, Guanwen Qiu, Abulhair Saparov  

**Link**: [PDF](https://arxiv.org/pdf/2605.06638)  

**Abstract**: Reinforcement learning (RL) has been applied to improve large language model (LLM) reasoning, yet the systematic study of how training scales with task difficulty has been hampered by the lack of controlled, scalable environments. We introduce ScaleLogic, a synthetic logical reasoning framework that offers independent control over two axes of difficulty: the depth of the required proof planning (i.e., the horizon) and the expressiveness of the underlying logic. Our proposed framework supports a wide range of logics: from simple implication-only logic ("if-then") towards more expressive first-order reasoning with conjunction ("and"), disjunction ("or"), negation ("not"), and universal quantification ("for all"). Using this framework, we show that the RL training compute $T$ follows a power law with respect to reasoning depth $D$ ($T \propto D^{\gamma}$, $R^{2} > 0.99$), and that the scaling exponent $\gamma$ increases monotonically with logical expressiveness, from $1.04$ to $2.60$. On downstream mathematics and general reasoning benchmarks, more expressive training settings yield both larger performance gains (up to $+10.66$ points) and more compute-efficient transfer compared to less expressive settings, demonstrating that what a model is trained on, not just how much it is trained, shapes downstream transfer. We further show that the power-law relationship holds across multiple RL methods, and curriculum-based training substantially improves scaling efficiency. 

---
# MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems 

**Authors**: Zhexuan Wang, Xuebo Liu, Li Wang, Zifei Shan, Yutong Wang, Zhenxi Song, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06623)  

**Abstract**: Large language model (LLM)-based Multi-agent systems (MAS) have shown promise in tackling complex collaborative tasks, where agents are typically orchestrated via role-specific prompts. While the quality of these prompts is pivotal, jointly optimizing them across interacting agents remains a non-trivial challenge, primarily due to the misalignment between local agent objectives and holistic system goals. To address this, we introduce MASPO, a novel framework designed to automatically and iteratively refine prompts across the entire system. A core innovation of MASPO is its joint evaluation mechanism, which assesses prompts not merely by their local validity, but by their capacity to facilitate downstream success for successor agents. This effectively bridges the gap between local interactions and global outcomes without relying on ground-truth labels. Furthermore, MASPO employs a data-driven evolutionary beam search to efficiently navigate the high-dimensional prompt space. Extensive empirical evaluations across 6 diverse tasks demonstrate that MASPO consistently outperforms state-of-the-art prompt optimization methods, achieving an average accuracy improvement of 2.9. We release our code at this https URL. 

---
# When and Why SignSGD Outperforms SGD: A Theoretical Study Based on $\ell_1$-norm Lower Bounds 

**Authors**: Hongyi Tao, Dingzhi Yu, Lijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06615)  

**Abstract**: Sign-based optimization algorithms, such as SignSGD and Muon, have garnered significant attention for their remarkable performance in training large foundation models. Despite this empirical success, we still lack a theoretical understanding of when and why these sign-based methods outperform vanilla SGD. The core obstacle is that under standard smoothness and finite variance conditions, SGD is known to be minimax optimal for finding stationary points measured by $\ell_2$-norms, thereby fundamentally precluding any complexity gains for sign-based methods in standard settings. To overcome this barrier, we analyze sign-based optimizers leveraging $\ell_1$-norm stationarity, $\ell_\infty$-smoothness, and a separable noise model, which can better capture the coordinate-wise nature of signed updates. Under this distinct problem geometry, we derive matched upper and lower bounds for SignSGD and explicitly characterize the problem class in which SignSGD provably dominates SGD. Specifically, we compare the \emph{upper bound of SignSGD} with the \emph{lower bound of SGD}, illustrating that SignSGD effectively reduces the complexity by a factor of $d$ under \emph{sparse noise}, where $d$ is the problem dimension. Furthermore, we elevate this framework to the matrix domain, providing an equivalent optimal lower bound for the Muon optimizer, proving that extending the sign operator to matrices preserves this optimal scaling with dimensionality. Finally, we bridge our theoretical bounds to practice, demonstrating that the theoretical superiority of SignSGD accurately predicts its faster convergence during the pretraining of a 124M parameter GPT-2 model. 

---
# SkillOS: Learning Skill Curation for Self-Evolving Agents 

**Authors**: Siru Ouyang, Jun Yan, Yanfei Chen, Rujun Han, Zifeng Wang, Bhavana Dalvi Mishra, Rui Meng, Chun-Liang Li, Yizhu Jiao, Kaiwen Zha, Maohao Shen, Vishy Tirumalashetty, George Lee, Jiawei Han, Tomas Pfister, Chen-Yu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.06614)  

**Abstract**: LLM-based agents are increasingly deployed to handle streaming tasks, yet they often remain one-off problem solvers that fail to learn from past interactions. Reusable skills distilled from experience provide a natural substrate for self-evolution, where high-quality skill curation serves as the key bottleneck. Existing approaches either rely on manual skill curation, prescribe heuristic skill operations, or train for short-horizon skill operations. However, they still struggle to learn complex long-term curation policies from indirect and delayed feedback. To tackle this challenge, we propose SkillOS, an experience-driven RL training recipe for learning skill curation in self-evolving agents. SkillOS pairs a frozen agent executor that retrieves and applies skills with a trainable skill curator that updates an external SkillRepo from accumulated experience. To provide learning signals for curation, we design composite rewards and train on grouped task streams based on skill-relevant task dependencies, where earlier trajectories update the SkillRepo, and later related tasks evaluate these updates. Across multi-turn agentic tasks and single-turn reasoning tasks, SkillOS consistently outperforms memory-free and strong memory-based baselines in both effectiveness and efficiency, with the learned skill curator generalizing across different executor backbones and task domains. Further analyses show that the learned curator produces more targeted skill use, while the skills in SkillRepo evolve into more richly structured Markdown files that encode higher-level meta-skills over time. 

---
# PairAlign: A Framework for Sequence Tokenization via Self-Alignment with Applications to Audio Tokenization 

**Authors**: Adhiraj Banerjee, Vipul Arora  

**Link**: [PDF](https://arxiv.org/pdf/2605.06582)  

**Abstract**: Many operations on sensory data -- comparison, memory, retrieval, and reasoning -- are naturally expressed over discrete symbolic structures. In language this interface is given by tokens; in audio, it must be learned. Existing audio tokenizers rely on quantization, clustering, or codec reconstruction, assigning tokens locally, so sequence consistency, compactness, length control, termination, and edit similarity are rarely optimized directly.
We introduce PairAlign, a framework for compact audio tokenization through sequence-level self-alignment. PairAlign treats tokenization as conditional sequence generation: an encoder maps speech to a continuous condition, and an autoregressive decoder generates tokens from BOS, learning token identity, order, length, and EOS placement. Given two content-preserving views, each view's sequence is trained to be likely under the other's representation, while unrelated examples provide competing sequences. This gives a scalable surrogate for edit-distance preservation while discouraging many-to-one collapse.
PairAlign starts from VQ-style tokenization and refines it with EMA-teacher targets, cross-paired teacher forcing, prefix corruption, likelihood contrast, and length control.
On 3-second speech, PairAlign learns compact, non-degenerate sequences with broad vocabulary usage and strong cross-view consistency. On TIMIT retrieval, it preserves edit-distance search while reducing archive token count by 55%. A continuous-sweep probe shows lower local overlap than a dense geometric tokenizer, but stronger length control and bounded edit trajectories under 100 ms shifts. PairAlign is a sequence-symbolic predictive learner: like JEPA-style objectives, it predicts an abstract target from another view as a learned variable-length symbolic sequence, not a continuous latent. 

---
# Cubit: Token Mixer with Kernel Ridge Regression 

**Authors**: Chuanyang Zheng, Jiankai Sun, Yihang Gao, Yuehao Wang, Liangchen Tan, Mac Schwager, Anderson Schneider, Yuriy Nevmyvaka, Xiaodong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.06501)  

**Abstract**: Since its introduction in 2017, the Transformer has become one of the most widely adopted architectures in modern deep learning. Despite extensive efforts to improve positional encoding, attention mechanisms, and feed-forward networks, the core token-mixing mechanism in Transformers remains attention. In this work, we show that the attention module in Transformers can be interpreted as performing Nadaraya-Watson regression, where it computes similarities between tokens and aggregates the corresponding values accordingly. Motivated by this perspective, we propose Cubit, a potential next-generation architecture that leverages Kernel Ridge Regression (KRR), while the vanilla Transformer relies on Nadaraya-Watson regression. Specifically, Cubit modifies the classical attention computation by incorporating the closed-form solution of KRR, combining value aggregation through kernel similarities with normalization via the inverse of the kernel matrix. To improve the training stability, we further propose the Limited-Range Rescale (LRR), which rescales the value layer within a controlled range. We argue that Cubit, as a KRR-based architecture, provides a stronger mathematical foundation than the vanilla Transformer, whose attention mechanism corresponds to Nadaraya-Watson regression. We validate this claim through comprehensive experiments. The experimental results suggest that Cubit may exhibit stronger long-sequence modeling capability. In particular, its performance gain over the Transformer appears to increase as the training sequence length grows. 

---
# Patch-Effect Graph Kernels for LLM Interpretability 

**Authors**: Ruben Fernandez-Boullon, David N. Olivieri  

**Link**: [PDF](https://arxiv.org/pdf/2605.06480)  

**Abstract**: Mechanistic interpretability aims to reverse-engineer transformer computations by identifying causal circuits through activation patching. However, scaling these interventions across diverse prompts and task families produces high-dimensional, unstructured datasets that are difficult to compare systematically. We propose a framework that reframes mechanistic analysis as a graph machine-learning problem by representing activation-patching profiles as patch-effect graphs over model components. We introduce three graph-construction methods: direct-influence via causal mediation, partial-correlation, and co-influence and apply graph kernels to analyze the resulting structures. Evaluating this approach on GPT-2 Small using Indirect Object Identification (IOI) and related tasks, we find that patch-effect graphs preserve discriminative structural signals. Specifically, localized edge-slot features provide higher classification accuracy than global graph-shape descriptors. A screened paired-patching validation suggests that CI and PC selected candidate edges correspond to stronger activation-influence effects than random or low-rank candidates. Crucially, by evaluating these representations against rigorous prompt-only and raw patch-effect controls, we make the evidential scope of the benchmark explicit: graph features compress structured patching signal, while raw tensors and surface cues define strong baselines that any circuit-level claim should address. Ultimately, our framework provides a compression and evaluation pipeline for comparing patching-derived structures under controlled baselines, separating robust slice-discriminative evidence from stronger task-general causal-circuit claims. 

---
# Invariant Features in Language Models: Geometric Characterization and Model Attribution 

**Authors**: Agnibh Dasgupta, Abdullah Tanvir, Xin Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2605.06458)  

**Abstract**: Language models exhibit strong robustness to paraphrasing, suggesting that semantic information may be encoded through stable internal representations, yet the structure and origin of such invariance remain unclear. We propose a local geometric framework in which semantically equivalent inputs occupy structured regions in latent space, with paraphrastic variation along nuisance directions and semantic identity preserved in invariant subspaces. Building on this view, we make three contributions: (1) a geometric characterization of invariant latent features, (2) a contrastive subspace discovery method that separates semantic-changing from semantic-preserving variation, and (3) an application of invariant representations to zero-shot model attribution. Across models and layers, empirical results support these contributions. Invariant structure emerges in specific depth regions, semantic displacement lies largely outside the nuisance subspace, and representation-level interventions indicate a causal role of invariant components in model outputs. Invariant representations also capture model-specific geometric patterns, enabling accurate attribution. These findings suggest that semantic invariance can be viewed as a local geometric property of latent representations, offering a principled perspective on how language models organize meaning. 

---
# E = T*H/(O+B): A Dimensionless Control Parameter for Mixture-of-Experts Ecology 

**Authors**: Qingjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06415)  

**Abstract**: We introduce E = T*H/(O+B), a dimensionless control parameter that predicts whether Mixture-of-Experts (MoE) models will develop a healthy expert ecology or collapse into dead experts. E combines four hyperparameters -- routing temperature T, routing entropy weight H, oracle weight O, and balance weight B -- into a single quantity. Through 12 controlled experiments (8 vision, 4 language) totaling over 11,000 training epochs, we establish that E >= 0.5 alone is sufficient to guarantee zero dead experts, removing the necessity for handcrafted load-balancing auxiliary losses. We validate this cross-modally on CIFAR-10, CIFAR-100, TinyImageNet-200, WikiText-2, and WikiText-103. Six additional findings emerge: (1) dead experts can resuscitate -- triggered by balance loss driving router re-exploration; (2) ortho toxicity is dataset-dependent, not universal; (3) task complexity shifts the critical E threshold; (4) model overfitting is decoupled from expert ecological health; (5) three-tier MoE spontaneously collapses into a two-tier functional structure; (6) ecological structure is temperature-invariant across a 50x range. We propose that E serves as a unified diagnostic for MoE training, analogous to the Reynolds number in fluid dynamics. 

---
# WavCube: Unifying Speech Representation for Understanding and Generation via Semantic-Acoustic Joint Modeling 

**Authors**: Guanrou Yang, Tian Tan, Qian Chen, Zhikang Niu, Yakun Song, Ziyang Ma, Yushen Chen, Zeyu Xie, Tianrui Wang, Yifan Yang, Wenxi Chen, Qi Chen, Wenrui Liu, Shan Yang, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.06407)  

**Abstract**: Integrating speech understanding and generation is a pivotal step toward building unified speech models. However, the different representations required for these two tasks currently pose significant compatibility challenges. Typically, semantics-oriented features are learned from self-supervised learning (SSL), and acoustic-oriented features from reconstruction. Such fragmented representations hinder the realization of truly unified speech systems. We present WavCube, a compact continuous latent derived from an SSL speech encoder that simultaneously supports speech understanding, reconstruction, and generation. WavCube employs a two-stage training scheme. Stage 1 trains a semantic bottleneck to filter off-manifold redundancy that makes raw SSL features intractable for diffusion. Stage 2 injects fine-grained acoustic details via end-to-end reconstruction, while a semantic anchoring loss ensures the representation remains grounded within its original semantic manifold. Comprehensive experiments show that WavCube closely approaches WavLM performance on SUPERB despite an 8x dimensional compression, attains reconstruction quality on par with existing acoustic representations, delivers state-of-the-art zero-shot TTS performance with markedly faster training convergence, and excels in speech enhancement, separation, and voice conversion tasks on the SUPERB-SG benchmark. Systematic ablations reveal that WavCube's two-stage recipe resolves two intrinsic flaws of SSL features for generative modeling, paving the way for future unified speech systems. Codes and checkpoints are available at this https URL. 

---
# Is Escalation Worth It? A Decision-Theoretic Characterization of LLM Cascades 

**Authors**: Dylan Bouchard  

**Link**: [PDF](https://arxiv.org/pdf/2605.06350)  

**Abstract**: Model cascades, in which a cheap LLM defers to an expensive one on low-confidence queries, are widely used to navigate the cost-quality tradeoff at deployment. Existing approaches largely treat the deferral threshold as an empirical hyperparameter, with limited guidance on the geometry of the resulting cost-quality frontier over a model pool. We develop a decision-theoretic framework grounded in constrained optimization and duality. For a two-model cascade, we establish piecewise concavity of the cost-quality frontier on decreasing-benefit regions of the confidence support, with reciprocal shadow prices linking the budget- and quality-constrained formulations. Given a pool of $k$ models, we characterize the frontier achievable by deterministic two-model threshold cascades as the pointwise envelope over $\binom{k}{2}$ pairwise cascades, with switching points where the optimal pair changes. For $k$-model cascades, we derive first-order conditions in which a single shadow price equalizes marginal quality-per-cost across stage boundaries. We validate the framework on five benchmarks (MATH, MMLU, TriviaQA, SimpleQA, LiveCodeBench) across eight models from five providers. Within the deterministic threshold-cascade class, full fixed chains underperform the pairwise envelope, and optimized subsequence cascades do not deliver practically meaningful held-out gains over it. A lightweight pre-generation router exceeds the best cascade policy on four of five datasets, mainly because it avoids the cheap model's generation cost on queries sent directly to a larger model rather than because of a stronger routing signal. These results suggest that cascade performance is limited primarily by structural cost, since cascades pay the cheap model before any escalation decision, rather than by a shortage of intermediate stages. 

---
# Improving the Efficiency of Language Agent Teams with Adaptive Task Graphs 

**Authors**: Elizabeth Mieczkowski, Alexander Ku, Tiwalayo Eisape, Dilip Arumugam, John Matters, Katherine M. Collins, Ilia Sucholutsky, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2605.06320)  

**Abstract**: Large language models (LLMs) are increasingly deployed in teams, yet existing coordination approaches often occupy two extremes. Highly structured methods rely on fixed roles, pipelines, or task decompositions assigned a priori. In contrast, fully unstructured teams enable adaptability and exploration but suffer from inefficiencies such as error propagation, inter-agent conflicts, and wasted resources (measured in time, tokens, or file operations). We introduce Language Agent Teams for Task Evolution (LATTE), a framework for coordinating LLM teams inspired by distributed systems, where processors must operate under partial observability and communication constraints. In LATTE, a team of agents collaboratively construct and maintain a shared, evolving coordination graph which encodes sub-task dependencies, individual agent assignment, and the current state of sub-task progress. This protocol maintains consistency while empowering agents to dynamically allocate work, adapt coordination, and discover new tasks. Across multiple collaborative tasks and a variety of base models, we demonstrate how LATTE reduces token usage, wall-clock time, communication, and coordination failures (e.g. file conflicts and redundant outputs) while matching or exceeding the accuracy of standard designs including MetaGPT, decentralized teams, top-down Leader-Worker hierarchies, and static decompositions. 

---
# Contrastive Identification and Generation in the Limit 

**Authors**: Xiaoyu Li, Andi Han, Jiaojiao Jiang, Junbin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.06211)  

**Abstract**: In the classical identification in the limit model of Gold [1967], a stream of positive examples is presented round by round, and the learner must eventually recover the target hypothesis. Recently, Kleinberg and Mullainathan [2024] introduced generation in the limit, where the learner instead must eventually output novel elements of the target's support. Both lines of work focus on positive-only or fully labeled data. Yet many natural supervision signals are inherently relational rather than singleton, which encode relationships between examples rather than labels of individual ones. We initiate the study of contrastive identification and generation in the limit, where the learner observes a contrastive presentation of data: a stream of unordered pairs $\{x,y\}$ satisfying $h(x)\ne h(y)$ for an unknown target binary hypothesis $h$, but which element is positive is hidden from the learner. We first present three results in the noiseless setting: an exact characterization of contrastive identifiable classes (a one-line geometric refinement of Angluin [1980]'s tell-tale condition), a combinatorial dimension called contrastive closure dimension (a contrasitive analogue of the closure dimension in Raman et al. [2025]) and exactly characterizing uniform contrastive generation with tight sample complexity, and a strict hierarchy in which contrastive generation and text identification are mutually incomparable. We then prove a sharp reversal under finite adversarial corruption: there exist classes identifiable from contrastive pairs under any finite corruption budget by a single budget-independent algorithm, yet not identifiable from positive examples under even one corrupted observation. The unifying technical object is the common crossing graph, which encodes pairwise ambiguity, family-level generation obstructions, and corruption defects in a single coverage-and-incidence language. 

---
# The Granularity Axis: A Micro-to-Macro Latent Direction for Social Roles in Language Models 

**Authors**: Chonghan Qin, Xiachong Feng, Ziyun Song, Xiaocheng Feng, Jing Xiong, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2605.06196)  

**Abstract**: Large language models (LLMs) are routinely prompted to take on social roles ranging from individuals to institutions, yet it remains unclear whether their internal representations encode the granularity of such roles, from micro-level individual experience to macro-level organizational, institutional, or national reasoning. We show that they do. We define a contrast-based Granularity Axis as the difference between mean macro- and micro-role hidden states. In Qwen3-8B, this axis aligns with the principal axis (PC1) of the role representation space at cosine 0.972 and accounts for 52.6% of its variance, indicating that granularity is the dominant geometric axis organizing prompted social roles. We construct 75 social roles across five granularity levels and collect 91,200 role-conditioned responses over shared questions and prompt variants, then extract role-level hidden states and project them onto the axis. Role projections increase monotonically across all five levels, remain stable across layers, prompt variants, endpoint definitions, held-out splits, and score-filtered subsets, and transfer to Llama-3.1-8B-Instruct. The axis is also causally relevant: activation steering along it shifts response granularity in the predicted direction, with Llama moving from 2.00 to 3.17 on a five-point macro scale under positive steering on prompts that admit local responses. The two models differ in controllability, suggesting that steering depends on each model's default operating regime. Overall, our findings suggest that social role granularity is not merely a stylistic surface feature, but a structured, ordered, and causally manipulable latent direction in role-conditioned language model behavior. 

---
# OPSD Compresses What RLVR Teaches: A Post-RL Compaction Stage for Reasoning Models 

**Authors**: Jaehoon Kim, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.06188)  

**Abstract**: On-Policy Self-Distillation (OPSD) has recently emerged as an alternative to Reinforcement Learning with Verifiable Rewards (RLVR), promising higher accuracy and shorter responses through token-level credit assignment from a self-teacher conditioned on privileged context. However, this promise does not carry over to thinking-enabled mathematical reasoning, where reported accuracy gains shrink and sometimes turn negative. We hypothesize that hindsight supervision can specify better token-level alternatives in short thinking-disabled outputs, but in long thinking-enabled traces it more readily identifies redundancy than supplies better replacements. To test this, we applied OPSD separately to correct and incorrect rollout groups, so that compression and correction can be observed in isolation. Our results show that in thinking-enabled mathematical reasoning, OPSD behaves most reliably as a compression mechanism rather than a correction mechanism: training only on correct rollouts preserves accuracy while substantially shortening responses, whereas training only on incorrect rollouts damages accuracy. In light of these findings, we propose a revised post-training pipeline for thinking-enabled mathematical reasoning: SFT then RLVR then OPSD. 

---
# Rethinking Adapter Placement: A Dominant Adaptation Module Perspective 

**Authors**: Suoxin Zhang, Run He, Di Fang, Xiang Tan, Kaixuan Chen, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2605.06183)  

**Abstract**: Low-rank adaptation (LoRA) is a widely used parameter-efficient fine-tuning method that places trainable low-rank adapters into frozen pre-trained models. Recent studies show that using fewer LoRA adapters may still maintain or even improve performance, but existing methods still distribute adapters broadly, leaving where to place a limited number of adapters to maximize performance largely open. To investigate this, we introduce PAGE (Projected Adapter Gradient Energy), a gradient-based sensitivity probe that estimates the initial trainable gradient energy available to each candidate LoRA adapter. Surprisingly, we find that PAGE is highly concentrated on a single shallow FFN down-projection across two model families and four downstream tasks. We term this module the dominant adaptation module and show that its layer index is architecture-dependent but task-stable. Motivated by this finding, we propose DomLoRA, a placement method that places a single adapter at the dominant adaptation module. With only ~0.7% of vanilla LoRA's trainable parameters, DomLoRA outperforms it on average across various downstream tasks, including instruction following, mathematical reasoning, code generation, and multi-turn conversation. This method also improves other LoRA variants, supporting the dominant adaptation module perspective as a practical placement guideline. 

---
# Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes 

**Authors**: Liu Hanqing, Jianjun Cao, Yuanze Li, Zijian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.06152)  

**Abstract**: Deep neural networks exhibit periodic loss spikes during unregularized long-term training, a phenomenon known as the "Slingshot Mechanism." Existing work usually attributes this to intrinsic optimization dynamics, but its triggering mechanism remains unclear. This paper proves that this phenomenon is a result of floating-point arithmetic precision limits. As training enters a high-confidence stage, the difference between the correct-class logit and the other logits may exceed the absorption-error threshold. Then during backpropagation, the gradient of the correct class is rounded exactly to zero, while the gradients of the incorrect classes remain nonzero. This breaks the zero-sum constraint of gradients across classes and introduces a systematic drift in the parameter update of the classifier layer. We prove that this drift forms a positive feedback loop with the feature, causing the global classifier mean and the global feature mean to grow exponentially. We call this mechanism Numerical Feature Inflation (NFI). This mechanism explains the rapid norm growth before a Slingshot spike, the subsequent reappearance of gradients, and the resulting loss spike. We further show that NFI is not equivalent to an observed loss spike: in more practical tasks, partial absorption may not produce visible spikes, but it can still break the zero-sum constraint and drive rapid growth of parameter norms. Our results reinterpret Slingshot as a numerical dynamic of finite-precision training, and provide a testable explanation for abnormal parameter growth and logit divergence in late-stage training. 

---
# On Time, Within Budget: Constraint-Driven Online Resource Allocation for Agentic Workflows 

**Authors**: Xinglin Wang, Zishen Liu, Shaoxiong Feng, Peiwen Yuan, Yiwei Li, Jiayi Shi, Yueqi Zhang, Chuyi Tan, Ji Zhang, Boyuan Pan, Yao Hu, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.06110)  

**Abstract**: Agentic systems increasingly solve complex user requests by executing orchestrated workflows, where subtasks are assigned to specialized models or tools and coordinated according to their dependencies. While recent work improves agent efficiency by optimizing the performance--cost--latency frontier, real deployments often impose concrete requirements: a workflow must be completed within a specified budget and before a specified deadline. This shifts the goal from average efficiency optimization to maximizing the probability that the entire workflow completes successfully under explicit budget and deadline constraints. We study \emph{constraint-driven online resource allocation for agentic workflows}. Given a dependency-structured workflow and estimates of success rates and generation lengths for each subtask--model pair, the executor allocates models and parallel samples across simultaneously executable subtasks while managing the remaining budget and time. We formulate this setting as a finite-horizon stochastic online allocation problem and propose \emph{Monte Carlo Portfolio Planning} (MCPP), a lightweight closed-loop planner that directly estimates constrained completion probability through simulated workflow executions and replans after observed outcomes. Experiments on CodeFlow and ProofFlow demonstrate that MCPP consistently improves constrained completion probability over strong baselines across a wide range of budget--deadline constraints. 

---
# Novelty-based Tree-of-Thought Search for LLM Reasoning and Planning 

**Authors**: Leon Hamm, Zlatan Ajanovic  

**Link**: [PDF](https://arxiv.org/pdf/2605.06040)  

**Abstract**: Although advances such as chain-of-thought, tree-of-thought or reinforcement learning have improved the performance of LLMs in reasoning and planning tasks, they are still brittle and have not achieved human-level performance in many domains, and often suffer from high time and token costs. Inspired by the success of width-based search in planning, we explore how the concept of novelty can be transferred to language domains and how it can improve tree-of-thought reasoning. A tree of thoughts relies on building possible "paths" of consecutive ideas or thoughts. These are generated by repeatedly prompting an LLM. In our paper, a measurable concept of novelty is proposed that describes the uniqueness of a new node (thought) in comparison to nodes previously seen in the search tree. Novelty is estimated by prompting an LLM and making use of embedded general knowledge from pre-training. This metric can then be used to prune branches and reduce the scope of the search. Although this method introduces more prompts per state, the overall token cost can be reduced by pruning and reducing the overall tree size. This procedure is tested and compared using several benchmarks in language-based planning and general reasoning. 

---
# Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks 

**Authors**: Guoxin Lu, Letian Sha, Qing Wang, Peijie Sun, Hao Zhou, Hua Dai, Fu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.05995)  

**Abstract**: The safety alignment of Large Language Models (LLMs) remains vulnerable to Harmful Fine-tuning (HFT). While existing defenses impose constraints on parameters, gradients, or internal representations, we observe that they can be effectively circumvented under persistent HFT. Our analysis traces this failure to the inherent redundancy of the high-dimensional parameter space: attackers exploit optimization trajectories that are orthogonal to defense constraints to restore harmful capabilities while deceptively adhering to safety restrictions. To address this, we propose Safety Bottleneck Regularization (SBR). SBR shifts the defensive focus from the redundant parameter space to the unembedding layer, which serves as a geometric bottleneck. By anchoring the final hidden states of harmful queries to those of the safety-aligned model, SBR enables the model to maintain safe responses even under persistent HFT. Extensive experiments confirm SBR's effectiveness, demonstrating that utilizing just a single safety anchor is sufficient to reduce the Harmful Score to $<$10 while preserving competitive performance on benign downstream tasks. 

---
# TheraAgent: Self-Improving Therapeutic Agent for Precise and Comprehensive Treatment Planning 

**Authors**: Junkai Li, Yunghwei Lai, Tianyi Zhu, Zheng Long Lee, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05963)  

**Abstract**: Formulating a treatment plan is inherently a complex reasoning and refinement task rather than a simple generation problem. However, existing large language models (LLMs) mainly rely on one-shot output without explicit verification, which may result in rough, incomplete, and potentially unsafe treatment plans. To address these limitations, we propose TheraAgent, an agentic framework that replaces one-shot generation with an iterative generate-judge-refine pipeline. By mirroring the actual reasoning process of human experts who iteratively revise treatment plans, our framework progressively transforms coarse and incomplete drafts into precise, comprehensive, and safer therapeutic regimens. To facilitate the critical judge component, we introduce TheraJudge, a treatment-specific evaluation module integrated into the inference loop to enforce clinical standards. Experiments show TheraAgent achieves state-of-the-art results on HealthBench, leading in Accuracy and Completeness. In expert evaluations, it attains an 86% win rate against physicians, with superior Targeting and Harm Control. Moreover, the highly agreement between TheraJudge and HealthBench evaluations confirms the reliability of our framework. 

---
# MobileEgo Anywhere: Open Infrastructure for long horizon egocentric data on commodity hardware 

**Authors**: Senthil Palanisamy, Abhishek Anand, Satpal Singh Rathor, Pratyush Patnaik, Shubhanshu Khatana  

**Link**: [PDF](https://arxiv.org/pdf/2605.05945)  

**Abstract**: The recent advancement of Vision Language Action (VLA) models has driven a critical demand for large scale egocentric datasets. However, existing datasets are often limited by short episode durations, typically spanning only a few minutes, which fails to capture the long horizon temporal dependencies necessary for complex robotic task execution. To bridge this gap, we present MobileEgo Anywhere, a framework designed to facilitate the collection of robust, hour plus egocentric trajectories using commodity mobile hardware. We leverage the ubiquitous sensor suites of modern smartphones to provide high fidelity, long term camera pose tracking, effectively removing the high hardware barriers associated with traditional robotics data collection. Our contributions are three fold: (1) we release a novel dataset comprising 200 hours of diverse, long form egocentric data with persistent state tracking; (2) we open source a mobile application that enables any user to record egocentric data, and (3) we provide a comprehensive processing pipeline to convert raw mobile captures into standardized, training ready formats for Vision Language Action model and foundation model research. By democratizing the data collection process, this work enables the massive scale acquisition of long horizon data across varied global environments, accelerating the development of generalizable robotic policies. 

---
# Near-Policy: Accelerating On-Policy Distillation via Asynchronous Generation and Selective Packing 

**Authors**: Miao Rang, Zhenni Bi, Hang Zhou, Kai Han, Xuechun Wang, An Xiao, Xinghao Chen, Yunhe Wang, Hanting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.05940)  

**Abstract**: Standard knowledge distillation for autoregressive models often suffers from distribution mismatch. While on-policy methods mitigate this by leveraging student-generated outputs, they rely on computationally expensive Reinforcement Learning (RL) frameworks. To improve efficiency, we propose Near-Policy Distillation (NPD), an asynchronous approach that decouples student generation from training. This reformulation enables Supervised Fine-Tuning (SFT) with sequence packing. However, asynchronous updates inevitably introduce policy lag and sample noise, which can cause the behavior to drift from near-policy toward off-policy. To counteract this without sacrificing efficiency, NPD integrates sparse student updates and the $\Delta$-IFD filtering mechanism, a heuristic sample selection mechanism that empirically stabilizes the optimization trajectory. By filtering extreme out-of-distribution samples, $\Delta$-IFD prevents noise from dominating the gradients, ensuring updates remain within a safe proximal learning zone. Empirically, the NPD framework achieves a 8.1x speedup over on-policy baselines and outperforms SFT by 8.09%. Crucially, by effectively narrowing the exploration space for subsequent RL, our method enables openPangu-Embedded-1B to reach a state-of-the-art score of 68.73%, outperforming the substantially larger Qwen3-1.7B. Codes will be released soon. 

---
# Bridging Passive and Active: Enhancing Conversation Starter Recommendation via Active Expression Modeling 

**Authors**: Yiqing Wu, Haoming Li, Guanyu Jiang, Jiahao Liang, Yongchun Zhu, Jingwu Chen, Feng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05855)  

**Abstract**: Large Language Model (LLM)-driven conversational search is shifting information retrieval from reactive keyword matching to proactive, open-ended dialogues. In this context, Conversation Starters are widely deployed to provide personalized query recommendations that help users initiate dialogues. Conventionally, recommending these starters relies on a closed "exposure-click" loop. Yet, this feedback loop mechanism traps the system in an echo chamber where, compounded by data sparsity, it fails to capture the dynamic nature of conversational search intents shaped by the open world. As a result, the system skews towards popular but generic this http URL this work, we uncover an untapped paradigm shift to shatter this harmful feedback loop: harnessing user "free will" through active user expressions. Unlike traditional recommendations, conversational search empowers users to bypass menus entirely through manually typed queries. The open-world intents in active queries hold the key to breaking this loop. However, incorporating them is non-trivial: (1) there exists an inherent distribution shift between active queries and formulated starters. (2) Furthermore, the "non-ID-able" nature of open text renders traditional item-based popularity statistics ineffective for large-scale industrial streaming training. To this end, we propose Passive-Active Bridge (PA-Bridge), a novel framework that employs an adversarial distribution aligner to bridge the distributional gap between passively recommended starters and active expressions. Moreover, we introduce a semantic discretizer to enable the deployment of popularity debiasing algorithms. Online A/B tests on our platform, demonstrate that PA-Bridge significantly boosts the Feature Penetration Rate by 0.54% and User Active Days 

---
# LeakDojo: Decoding the Leakage Threats of RAG Systems 

**Authors**: Maosen Zhang, Jianshuo Dong, Boting Lu, Wenyue Li, Xiaoping Zhang, Tianwei Zhang, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05818)  

**Abstract**: Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to leverage external knowledge, but also exposes valuable RAG databases to leakage attacks. As RAG systems grow more complex and LLMs exhibit stronger instruction-following capabilities, existing studies fall short of systematically assessing RAG leakage risks. We present LeakDojo, a configurable framework for controlled evaluation of RAG leakage. Using LeakDojo, we benchmark six existing attacks across fourteen LLMs, four datasets, and diverse RAG systems. Our study reveals that (1) query generation and adversarial instructions contribute independently to leakage, with overall leakage well approximated by their product; (2) stronger instruction-following capability correlates with higher leakage risk; and (3) improvements in RAG faithfulness can introduce increased leakage risk. These findings provide actionable insights for understanding and mitigating RAG leakage in practice. Our codebase is available at this https URL. 

---
# Adaptive Selection of LoRA Components in Privacy-Preserving Federated Learning 

**Authors**: Myoungjun Kim, Sangwoo Park, Yoseob Han, Jin-Hyun Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2605.05769)  

**Abstract**: Differentially private federated fine-tuning of large models with LoRA suffers from aggregation error caused by LoRA's multiplicative structure, which is further amplified by DP noise and degrades both stability and accuracy. Existing remedies apply a single update mode uniformly across all layers and all communication rounds (or alternate them on a fixed schedule), ignoring both the structural asymmetry between the two LoRA factors and the round-wise dynamics of training. We propose AS-LoRA, an adaptive framework defined by three axes (i) layer-wise freedom, in which each layer independently selects its active component, (ii) round-wise adaptivity, in which the selection updates over communication rounds, and (iii) a curvature-aware score derived from a second-order approximation of the loss. Theoretically, AS-LoRA eliminates the reconstruction-error floor of layer-tied schedules, accelerates convergence, implicitly biases solutions toward flatter minima, and incurs no additional privacy cost. Across GLUE, SQuAD, CIFAR-100, and Tiny-ImageNet under strict DP budgets and non-IID partitions, AS-LoRA improves over the federated LoRA baselines by up to $+7.5$ pp on GLUE and $+12.5$ pp on MNLI-mm for example, while matching or exceeding SVD-based aggregation methods at $33\text{--}180 \times$ lower aggregation cost and with negligible communication overhead. Code for the proposed method is available at this https URL. 

---
# Priming, Path-dependence, and Plasticity: Understanding the molding of user-LLM interaction and its implications from (many) chat logs in the wild 

**Authors**: Shengqi Zhu, Jeffrey M. Rzeszotarski, David Mimno  

**Link**: [PDF](https://arxiv.org/pdf/2605.05767)  

**Abstract**: User interactions with LLMs are shaped by prior experiences and individual exploration, but in-lab studies do not provide system designers with visibility into these in-the-wild factors. This work explores a new approach to studying real-world user-LLM interactions through large-scale chat logs from the wild. Through analysis of 140K chatbot sessions from 7,955 anonymized global users over time, we demonstrate key patterns in user expressions despite varied tasks: (1) LLM users are not tabula rasa, nor are they constantly adapting; rather, interaction patterns form and stabilize rapidly through individual early trajectories; (2) Longitudinal outcomes, such as recurring text patterns and retention rates, are strongly correlated with early exploration; (3) Parallel dynamics are present, including organizing expressions by task types such as emotional support, or in response to model-version updates. These results present an ``agency paradox'': despite LLM input spaces being unconstrained and user-driven, we in fact see less user exploration. We call for design consideration surrounding the molding procedure and its incorporation in future research. 

---
# RVPO: Risk-Sensitive Alignment via Variance Regularization 

**Authors**: Ivan Montero, Tomasz Jurczyk, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2605.05750)  

**Abstract**: Current critic-less RLHF methods aggregate multi-objective rewards via an arithmetic mean, leaving them vulnerable to constraint neglect: high-magnitude success in one objective can numerically offset critical failures in others (e.g., safety or formatting), masking low-performing "bottleneck" rewards vital for reliable multi-objective alignment. We propose Reward-Variance Policy Optimization (RVPO), a risk-sensitive framework that penalizes inter-reward variance during advantage aggregation, shifting the objective from "maximize sum" to "maximize consistency." We show via Taylor expansion that a LogSumExp (SoftMin) operator effectively acts as a smooth variance penalty. We evaluate RVPO on rubric-based medical and scientific reasoning with up to 17 concurrent LLM-judged reward signals (Qwen2.5-3B/7B/14B) and on tool-calling with rule-based constraints (Qwen2.5-1.5B/3B). By preventing the model from neglecting difficult constraints to exploit easier objectives, RVPO improves overall scores on HealthBench (0.261 vs. 0.215 for GDPO at 14B, $p < 0.001$) and maintains competitive accuracy on GPQA-Diamond without the late-stage degradation observed in other multi-reward methods, demonstrating that variance regularization mitigates constraint neglect across model scales without sacrificing general capabilities. 

---
# Multi-Dimensional Behavioral Evaluation of Agentic Stock Prediction Systems Using LLM Judges with Closed-Loop Reinforcement Learning Feedback 

**Authors**: Mohammad Al Ridhawi, Mahtab Haj Ali, Hussein Al Osman  

**Link**: [PDF](https://arxiv.org/pdf/2605.05739)  

**Abstract**: Agentic stock prediction systems make sequences of interdependent decisions (regime detection, pathway routing, reinforcement learning control) whose individual quality is hidden by aggregate metrics such as mean absolute percentage error (MAPE) or directional accuracy. We present a behavioral evaluation framework that addresses this gap. Behavioral traces logged at every autonomous decision point are grouped into five-day episodes and scored along six domain-specific dimensions (regime detection, routing, adaptation, risk calibration, strategy coherence, error recovery) by an ensemble of three large language model (LLM) judges (GPT 5.4, Claude 4.6 Opus, Gemini 3.1 Pro). Perturbation-based validation on 420 episodes yields targeted score drops of $-1.6$ to $-2.4$ on intended dimensions versus an average of $-0.32$ on the remaining five, with cross-model agreement up to Krippendorff's $\alpha = 0.85$. The composite behavioral score, used here only for cross-episode reporting, correlates at $\rho = 0.72$ with realized 20-day Sharpe ratio from offline backtesting. Closing the loop, the framework converts deficient per-dimension scores into a credit-assigned penalty term added to the Soft Actor-Critic (SAC) reward. Three short fine-tuning cycles, all confined to the validation period, produce on the held-out 2017-2025 test period a one-day MAPE reduction from 0.61% to 0.54% (an 11.5% relative reduction; $p<0.001$, Cohen's $d=0.31$), a directional accuracy increase from 71% to 74%, and an 18% Sharpe ratio improvement (95% bootstrap CI [8.2%, 27.4%]), with gains concentrated in high-volatility episodes where the original system was most behaviorally deficient. Results are from offline backtesting and do not address effects specific to live deployment. 

---
# ReFlect: An Effective Harness System for Complex Long-Horizon LLM Reasoning 

**Authors**: Fan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05737)  

**Abstract**: Current reasoning paradigms for LLMs include chain-of-thought, ReAct, and post-hoc self-critique. These paradigms rely on two assumptions that fail on long-horizon, multi-stage tasks. As a result, errors accumulate silently across reasoning steps, leaving an open question: can a reasoning system effectively detect and recover from its own failures? We present ReFlect, a \emph{harness} system for LLM reasoning that creates standalone error detection and recovery logic as a deterministic wrapper around the model. Controlled experiments across 6 reasoning domains show that prompt-level self-critique produces formulaic templates that flag no issues in 90 of 100 audited reflection blocks, and the investigated LLMs wrongly accept a wrong answer in at least 76\% of cases. Our ReFlect harness achieves task success rates ranging from 41\% on gpt-4o-mini to 56\% on Claude Sonnet 4.5 across six models spanning small and frontier scale, with per-model gains over Direct CoT ranging from +7 pp on Qwen2.5-72B to +29 pp on Claude Sonnet 4.5, and additionally raises SWE-bench patch-structural quality from 0\% (Direct CoT) to between 82\% (Qwen2.5-72B) and 87\% (GPT-4o). Notably, the harness gain is inversely proportional to the model's Direct CoT task success rate (the fitted slope is -1.69 with r=-0.76): each pp lost in baseline success rate is mechanically recovered by 1.69 pp of harness gain. We spot that adding structured reasoning state and operators yields only 15.0--18.7\% pair-mean on Llama-3.3-70B and Qwen2.5-72B because models at this scale cannot reliably populate the state its operators require. ReFlect is model-agnostic, training-free, and operates entirely at inference time. 

---
# More Is Not Always Better: Cross-Component Interference in LLM Agent Scaffolding 

**Authors**: Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05716)  

**Abstract**: LLM agent systems are built by stacking scaffolding components (planning, tools, memory, self-reflection, retrieval) assuming more is better. We study cross-component interference (CCI): degradation when components interact destructively. We run a full factorial experiment over all 2^5=32 subsets of five components on HotpotQA and GSM8K with Llama-3.1-8B/70B (96 conditions, up to 10 seeds). The All-In system is consistently suboptimal: on HotpotQA, a single-tool agent surpasses All-In by 32% (F1 0.233 vs 0.177, p=0.023); on GSM8K, a 3-component subset beats All-In by 79% (0.43 vs 0.24, p=0.010). Optimal component count is task-dependent (k*=1-4) and scale-sensitive: at 70B, combinations that hurt at 8B provide gains, though All-In still trails the best subset. We fit a main-effects regression (R^2=0.916, adj-R^2=0.899, LOOCV=0.872), compute exact Shapley values, and find 183/325 submodularity violations (56.3%), showing greedy selection is unreliable. A three-body synergy among Tool Use, Self-Reflection, and Retrieval (INT_3=+0.175, 95% CI [+0.003,+0.351]) is reported as exploratory. CCI replicates across model families (Qwen2.5) and is robust to prompt paraphrasing. Our findings suggest maximally-equipped agent defaults should be replaced by task-specific subset selection via interaction-aware analysis. 

---
# Decodable but Not Corrected by Fixed Residual-Stream Linear Steering: Evidence from Medical LLM Failure Regimes 

**Authors**: Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.05715)  

**Abstract**: Can linearly decodable failure signals in LLM hidden states be leveraged to correct those failures? We investigate this classification-correction gap via Overthinking (OT)--a stable behavioral regime (Jaccard >= 0.81, 94% inter-annotator agreement) in medical QA where models answer correctly under resampling yet fail in extended chain-of-thought. OT is linearly decodable at 71.6% balanced accuracy (p < 10^{-16}). Yet five families of fixed linear steering (29 configurations, n=1,273) all yield Delta ~= 0, with identical null results cross-architecture (Qwen2.5-7B) and cross-domain (MMLU-STEM). Three convergent lines of evidence suggest representational entanglement: the OT direction has 85-88% overlap with task-critical computation (specificity ratio <= 0.152); non-targeted shared-direction steering damages accuracy (-12.1pp); and LEACE concept erasure damages accuracy (-3.6pp, p=0.01), while 10 random erasures produce Delta=+0.3pp. The per-instance probe-steering correlation is r=-0.002 (p=0.97). Positively, the same probe enables selective abstention (held-out AUROC=0.610, exceeding all five uncertainty baselines, p=0.009): decodable failure structure supports post-generation reliability estimation even when the fixed linear steering family cannot exploit it for correction. 

---
# Architecture Matters: Comparing RAG Systems under Knowledge Base Poisoning 

**Authors**: Samuel Korn  

**Link**: [PDF](https://arxiv.org/pdf/2605.05632)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are vulnerable to knowledge base poisoning, yet existing attacks have been evaluated almost exclusively against vanilla retrieve-then-generate pipelines. Architectures designed to handle conflicting retrieved information - multi-agent debate, agentic retrieval, recursive language models - remain untested against adversarially optimized contradictions. We evaluate four RAG architectures (vanilla RAG, agentic RAG, MADAM-RAG, and Recursive Language Models) under controlled single-document (N=1) poisoning on 921 Natural Questions QA pairs, comparing a clean baseline, naive injection, and CorruptRAG-AK - an adversarial attack whose meta-epistemic framing targets credibility assessment. Architecture is a high-impact variable in adversarial robustness: under CorruptRAG-AK, attack success rates range from 81.9% (vanilla) to 24.4% (RLM) - a spread of nearly 58 percentage points across architectures with comparable clean accuracy (~92%). Decomposing this gap, once the poisoned document is retrieved, adversarial framing - not retrieval optimization - drives the majority of CorruptRAG-AK's advantage for three of four architectures, localizing the cross-architecture vulnerability at the content-reasoning stage. Our MADAM-RAG reimplementation shows the highest apparent contradiction detection rate, though our LLM judge over-identifies this behavior (~48.5% precision), so reported rates are upper bounds. Regardless of detection, MADAM-RAG cannot resolve contradictions reliably, producing a 41.4% non-answer rate even on clean inputs - though implementation divergences from the original may contribute. We introduce a seven-category behavioral taxonomy capturing contradiction detection, hedging, and failure modes beyond binary accuracy. Code, data, and analysis notebooks are publicly available. 

---
# Spherical Flows for Sampling Categorical Data 

**Authors**: Jannis Chemseddine, Gregor Kornhardt, Gabriele Steidl  

**Link**: [PDF](https://arxiv.org/pdf/2605.05629)  

**Abstract**: We study the problem of learning generative models for discrete sequences in a continuous embedding space. Whereas prior approaches typically operate in Euclidean space or on the probability simplex, we instead work on the sphere $\mathbb S^{d-1}$. There the von Mises-Fisher (vMF) distribution induces a natural noise process and admits a closed-form conditional score. The conditional velocity is in general intractable. Exploiting the radial symmetry of the vMF density we reduce the continuity equation on $\mathbb S^{d-1}$ to a scalar ODE in the cosine similarity, whose unique bounded solution determines the velocity. The marginal velocity and marginal score on $(\mathbb S^{d-1})^L$ both decompose into posterior-weighted tangent sums that differ only by per-token scalar weights. This gives access to both ODE and predictor-corrector (PC) sampling. The posterior is the only learned object, trained by a cross-entropy loss. Experiments compare the vMF path against geodesic and Euclidean alternatives. The combination of vMF and PC sampling significantly improves results on Sudoku and language modeling. 

---
# Belief Memory: Agent Memory Under Partial Observability 

**Authors**: Junfeng Liao, Qizhou Wang, Jianing Zhu, Bo Du, Rui Yan, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.05583)  

**Abstract**: LLM agents that operate over long context depend on external memory to accumulate knowledge over time. However, existing methods typically store each observation as a single deterministic conclusion (e.g., inferring "API~X failed" from temporary errors), even though such observations are inherently partial and potentially ambiguous. By committing to one conclusion and discarding uncertainty, these methods introduce self-reinforcing error: the agent acts on the stored conclusion, never revisits alternatives, and reinforces the conclusion over time. To address this issue, we propose BeliefMem, which shifts the memory paradigm from committing to a single conclusion per observation to retaining multiple candidate conclusions with their probabilities. Concretely, BeliefMem stores the candidate conclusions as separate memory entries, each carrying a probability that is updated via Noisy-OR rules as new observations arrive. At retrieval, all candidates surface together with their probabilities, keeping alternatives visible to the agent. Since each conclusion in memory retains its probability, BeliefMem preserves the uncertainty that the deterministic paradigm discards, enabling the agent to act with high confidence on well-evidenced knowledge while retaining the capacity to update its confidence when new evidence arrives. Empirical evaluations on LoCoMo and ALFWorld benchmarks show that, even with limited data, BeliefMem achieves the best average performance, remarkably outperforming well-known baselines. More broadly, such probabilistic memory produces substantial gains and explores a new direction for agent memory in partially observable environments. 

---
# Nonsense Helps: Prompt Space Perturbation Broadens Reasoning Exploration 

**Authors**: Langlin Huang, Chengsong Huang, Jinyuan Li, Donghong Cai, Yuyi Yang, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05566)  

**Abstract**: Reinforcement learning with verifiable rewards, particularly Group Relative Policy Optimization (GRPO), has significantly advanced the reasoning capabilities of Large Language Models (LLMs). However, in complex tasks, GRPO frequently suffers from the ``zero-advantage problem'': when all sampled rollouts for a query fail, the relative advantage collapses to zero. Consequently, the model loses effective training signals for these questions, wasting the training data and computational budget. While simply increasing the sampling budget for these questions is a common remedy, the static sampling policy inherently constrains reasoning exploration, limiting the success rate. In this paper, we propose Lorem Perturbation for Exploration (LoPE), a simple yet effective training framework to break this exploration bottleneck. We posit that task-irrelevant prompt-space perturbations can shift the model's output distribution enough to unlock orthogonal reasoning pathways for hard questions. Specifically, LoPE prepends sequences stochastically assembled from Lorem Ipsum vocabulary (a pseudo-Latin placeholder text) to the prompts before resampling. Experiments across 1.7B, 4B, and 7B models demonstrate that LoPE significantly outperforms resampling with the original prompts. Further analysis reveals that other Latin-based random sequences with low perplexity are also effective perturbations. Our results establish LoPE as a strong baseline for broadening exploration in LLM reinforcement learning. 

---
# Anatomy of a Query: W5H Dimensions and FAR Patterns for Text-to-SQL Evaluation 

**Authors**: Vicki Stover Hertzberg, Eduardo Valverde, Joyce C. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2605.05525)  

**Abstract**: Natural language interfaces to databases have gained popularity, yet the theoretical foundations for evaluating and designing these systems remain underdeveloped. We present QUEST (Query Understanding Evaluation through Semantic Translation), a framework resting on two independently motivated components: the FAR structural invariant, which holds that every well-formed query reduces to Filter, Aggregate, and Return operations; and the W5H dimensional framework, which holds that all filtering criteria map to six semantic dimensions (Who, What, Where, When, Why, and How). Validated across five text-to-SQL datasets (n = 120,464), FAR conformance is universal across all domains and schema types, while W5H dimensional profiles vary substantially. Healthcare queries are strongly concentrated in temporal (WHEN: 80.4%) and person-centric (WHO: 73.0%) dimensions far exceeding general-domain benchmarks, and causal (WHY) and mechanistic (HOW) reasoning are near-zero everywhere, with apparent HOW exceptions reflecting quantitative aggregation rather than genuine procedural reasoning. These results identify a frontier that must be crossed for genuine machine reasoning over structured data. 

---
# FinRAG-12B: A Production-Validated Recipe for Grounded Question Answering in Banking 

**Authors**: Denys Katerenchuk, Pablo Duboue, Keelan Evanini, David Gondek, Nithin Govindugari, Olivier Allauzen, Joshua Baptiste, David J More, Joshua Schechter  

**Link**: [PDF](https://arxiv.org/pdf/2605.05482)  

**Abstract**: Large language models (LLMs) are rapidly being adopted across various domains. However, their adoption in banking industry faces resistance due to demands for high accuracy, regulatory compliance, and the need for verifiable and grounded responses. We present a unified, data-efficient framework for training grounded domain-specific LLMs that optimizes answer quality, citation grounding, and calibrated refusal under real-world deployment constraints. First, we describe a data generation pipeline that combines LLM-as-a-Judge filtering, citation annotation, and curriculum learning with only 143M tokens. The resulting 12B model achieves high answer quality outperforming GPT-4.1 on citation grounding, with a modest citation tradeoff versus the untuned base. Second, we propose a calibrated refusal mechanism: training on 22% unanswerable examples yield a 12% "I don't know" rate, substantially improving over the base model's unsafe 4.3% rate while avoiding GPT-4.1's over-refusal (20.2%). Third, we present an end-to-end methodology spanning from data curation to quantized serving. The system is deployed at 40+ financial institutions, achieving a 7.1 percentage point improvement in query resolution (p < 0.001). Additionally, the model delivers 3-5x faster responses at 20-50x lower cost compared to GPT-4.1. 

---
# A Unified Benchmark for Evaluating Knowledge Graph Construction Methods and Graph Neural Networks 

**Authors**: Othmane Kabal, Mounira Harzallah, Fabrice Guillet, Hideaki Takeda, Ryutaro Ichise  

**Link**: [PDF](https://arxiv.org/pdf/2605.05476)  

**Abstract**: Knowledge graphs automatically constructed from text are increasingly used in real-world applications. However, their inherent noise, fragmentation, and semantic inconsistencies significantly affect the performance of Graph Neural Networks (GNNs) on downstream tasks. Assessing their performance and robustness remains difficult, as it is often unclear whether observed results stem from the learning model or from the quality of the constructed graph itself. In this work, we introduce a dual-purpose benchmark designed to jointly evaluate (i) the performance of GNNs on noisy, text-derived graphs and (ii) the effectiveness of graph construction methods on a downstream task. The benchmark is built in the biomedical domain from a single textual corpus and includes two automatically constructed graphs generated using different extraction methods, alongside a high-quality reference graph curated by experts that serves as an upper performance bound. This design enables controlled comparison of construction methods and systematic evaluation of GNN robustness through semi-supervised node classification. We further provide a standardized, reproducible, and extensible evaluation framework, facilitating the integration of new graph extraction methods and learning models. 

---
# Agentic Retrieval-Augmented Generation for Financial Document Question Answering 

**Authors**: Yang Shu, Yingmin Liu, Zequn Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.05409)  

**Abstract**: Financial document question answering (QA) demands complex multi-step numerical reasoning over heterogeneous evidence--structured tables, textual narratives, and footnotes--scattered across corporate filings. Existing retrieval-augmented generation (RAG) approaches adopt a single-pass retrieve-then-generate paradigm that struggles with the compositional reasoning chains prevalent in financial analysis. We propose FinAgent-RAG, an agentic RAG framework that orchestrates iterative retrieval-reasoning loops with self-verification, specifically engineered for the precision requirements of financial numerical reasoning. The framework integrates three domain-specific innovations: (1) a Contrastive Financial Retriever trained with hard negative mining to distinguish semantically similar but numerically distinct financial passages, (2) a Program-of-Thought reasoning module that generates executable Python code for precise arithmetic rather than relying on error-prone LLM-based mental computation, and (3) an Adaptive Strategy Router that dynamically allocates computational resources based on question complexity, reducing API costs by 41.3% on FinQA while preserving accuracy. Extensive experiments on three benchmark datasets--FinQA, ConvFinQA, and TAT-QA--demonstrate that FinAgent-RAG achieves 76.81%, 78.46%, and 74.96% execution accuracy respectively, outperforming the strongest baseline by 5.62--9.32 percentage points. Ablation studies, cross-backbone evaluation with four LLMs, and deployment cost analysis confirm the framework's robustness and practical viability for financial institutions. 

---
# BALAR : A Bayesian Agentic Loop for Active Reasoning 

**Authors**: Aymen Echarghaoui, Dongxia Wu, Emily B. Fox  

**Link**: [PDF](https://arxiv.org/pdf/2605.05386)  

**Abstract**: Large language models increasingly operate in interactive settings where solving a task requires multiple rounds of information exchange with a user. However, most current systems treat dialogue reactively and lack a principled mechanism to reason about what information is missing and which question should be asked next. We propose BALAR (Bayesian Agentic Loop for Active Reasoning), a task-agnostic outer-loop algorithm that requires no fine-tuning and enables structured multi-turn interaction between an LLM agent and a user. BALAR maintains a structured belief over latent states, selects clarifying questions by maximizing expected mutual information, and dynamically expands its state representation when the current one proves insufficient. We evaluate BALAR on three diverse benchmarks: AR-Bench-DC (detective cases), AR-Bench-SP (thinking puzzles), and iCraft-MD (clinical diagnosis). BALAR significantly outperforms all baselines across all three benchmarks, with $14.6\%$ higher accuracy on AR-Bench-DC, $38.5\%$ on AR-Bench-SP, and $30.5\%$ on iCraft-MD. 

---
# ZAYA1-8B Technical Report 

**Authors**: Robert Washbourne, Rishi Iyer, Tomas Figliolia, Henry Zheng, Ryan Lorig-Roach, Sungyeon Yang, Pritish Yuvraj, Quentin Anthony, Yury Tokpanov, Xiao Yang, Ganesh Nanduru, Stephen Ebert, Praneeth Medepalli, Skyler Szot, Srivatsan Rajagopal, Alex Ong, Bhavana Mehta, Beren Millidge  

**Link**: [PDF](https://arxiv.org/pdf/2605.05365)  

**Abstract**: We present ZAYA1-8B, a reasoning-focused mixture-of-experts (MoE) model with 700M active and 8B total parameters, built on Zyphra's MoE++ architecture. ZAYA1-8B's core pretraining, midtraining, and supervised fine-tuning (SFT) were performed on a full-stack AMD compute, networking, and software platform. With under 1B active parameters, ZAYA1-8B matches or exceeds DeepSeek-R1-0528 on several challenging mathematics and coding benchmarks, and remains competitive with substantially larger open-weight reasoning models. ZAYA1-8B was trained from scratch for reasoning, with reasoning data included from pretraining onward using an answer-preserving trimming scheme. Post-training uses a four-stage RL cascade: reasoning warmup on math and puzzles; a 400-task RLVE-Gym curriculum; math and code RL with test-time compute traces and synthetic code environments built from competitive-programming references; and behavioral RL for chat and instruction following. We also introduce Markovian RSA, a test-time compute method that recursively aggregates parallel reasoning traces while carrying forward only bounded-length reasoning tails between rounds. In TTC evaluation, Markovian RSA raises ZAYA1-8B to 91.9\% on AIME'25 and 89.6\% on HMMT'25 while carrying forward only a 4K-token tail, narrowing the gap to much larger reasoning models including Gemini-2.5 Pro, DeepSeek-V3.2, and GPT-5-High. 

---
# Beyond BLEU: A Semantic Evaluation Method for Code Translation 

**Authors**: Julius Näumann, Sven Keidel, Amir Molzam Sharifloo, Mira Mezini  

**Link**: [PDF](https://arxiv.org/pdf/2605.05282)  

**Abstract**: Code translation is one of the core capabilities of LLMs. However, evaluating the correctness of translations remains difficult, as commonly used metrics such as BLEU measure only syntactic similarity, disregarding program semantics. We propose a novel evaluation methodology for code translation tasks, emphasizing semantic equivalence over surface-level string similarity. Our approach applies established compiler testing methodology to a new domain, allowing the assessment of an LLM fine-tuned for binary lifting tasks (i.e. decompiling binaries to higher-level representations). We introduce a semantic correctness score, defined as the proportion of translations that produce correct execution outcomes, and demonstrate its application by evaluating LLM-based and heuristic decompilers. Our findings show that LLM-based approaches significantly outperform heuristic ones, while BLEU scores show negligible correlation with semantic correctness (r = -0.127 to 0.354), demonstrating that syntactic metrics fail to predict functional accuracy. 

---
# Career-Aware Resume Tailoring via Multi-Source Retrieval-Augmented Generation with Provenance Tracking: A Case Study 

**Authors**: Kumar Abhinav  

**Link**: [PDF](https://arxiv.org/pdf/2605.05257)  

**Abstract**: AI-assisted resume tailoring systems commonly operate on a single uploaded resume, which limits their ability to recover relevant experience omitted from the current draft and makes it difficult for users to distinguish grounded edits from model-generated suggestions. This paper presents Resume Tailor, an agentic resume-tailoring system that maintains a longitudinal career vault in a vector database and uses multi-source retrieval-augmented generation (RAG) to assemble job-specific resume content from historical resumes and structured career records. The system is implemented as a 12-node LangGraph pipeline with typed state management, hybrid semantic-lexical confidence scoring, provenance-aware fallback generation, anti-hallucination guardrails, and a conditional review loop. We report a pilot evaluation on nine job descriptions (JDs) across software engineering, data analytics, and business analysis roles using a single candidate's career history. For six JDs where the candidate held at least one prior role in the same occupational category, enabling the career vault improved Applicant Tracking System (ATS)-style fit scores by an average of 7.8 points. For two JDs requiring domain-specific expertise absent from the vault, scores decreased by an average of 8.0 points. One partially overlapping role showed a modest gain of 2 points. These results suggest that longitudinal retrieval can improve resume tailoring when relevant prior experience exists, while also highlighting the need for confidence-gated retrieval when domain overlap is weak. 

---
# Internalizing Outcome Supervision into Process Supervision: A New Paradigm for Reinforcement Learning for Reasoning 

**Authors**: Fei Ding, Yongkang Zhang, Runhao Liu, Yuhao Liao, Zijian Zeng, Sibo wang, Huiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.05226)  

**Abstract**: The central challenge of reinforcement learning for reasoning lies not only in the sparsity of outcome-level supervision, but more fundamentally in how to transform feedback provided only at the end of a sequence into fine-grained learning signals that can guide intermediate reasoning steps. Existing approaches either rely on outcome-level rewards for sequence-level optimization, which makes precise credit assignment difficult, or depend on externally constructed process supervision, which is costly and difficult to scale sustainably. To address this, we propose a new perspective: reinforcement learning for reasoning can be understood as the problem of internalizing outcome supervision into process supervision. From this perspective, we introduce a supervision-internalization method for reinforcement learning for reasoning, enabling the model to automatically extract process-level learning signals through identifying, correcting, and reusing failed reasoning trajectories, thereby achieving finer-grained policy optimization under outcome-only supervision. We further abstract this idea into a new training paradigm, in which the model continually generates and refines its own internal process supervision during reinforcement learning, opening a new path for fine-grained credit assignment in reinforcement learning for reasoning that differs from externally provided process supervision. 

---
# Data-Driven Variational Basis Learning Beyond Neural Networks: A Non-Neural Framework for Adaptive Basis Discovery 

**Authors**: Andrew Kiruluta  

**Link**: [PDF](https://arxiv.org/pdf/2605.05221)  

**Abstract**: Classical representation systems such as Fourier series, wavelets, and fixed dictionaries provide analytically tractable basis expansions, but they are not intrinsically adapted to the empirical structure of modern high-dimensional data. Neural networks overcome this limitation by learning features from data, yet they do so through layered nonlinear parameterizations that often sacrifice interpretability, explicit control over basis structure, and mathematical transparency. In this manuscript we develop a non-neural alternative that learns basis functions directly from data through variational optimization. The proposed framework, termed Data Driven Variational Basis Learning (DVBL), treats basis atoms as primary optimization variables and learns them jointly with sample-specific coefficients and, when appropriate, a latent linear evolution operator. This yields a data-adaptive basis expansion that remains explicit, interpretable, and amenable to rigorous analysis. We formulate the model, establish existence of minimizers, prove blockwise descent properties for an alternating minimization algorithm, give conditions for coefficient recovery and basis identifiability, and show how manifold and dynamical regularization can be integrated without invoking neural architectures. We also discuss the conceptual novelty of the framework relative to classical dictionary learning, spectral methods, Koopman operator methods, and deep representation learning. 

---
# PRAISE: Prefix-Based Rollout Reuse in Agentic Search Training 

**Authors**: Erhan Zhang, Yiqun Chen, Zechun Niu, Wei Yang, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2604.03675)  

**Abstract**: In agentic search, large language models (LLMs) are trained to perform multi-turn retrieval and reasoning for complex tasks such as multi-hop question answering (QA). However, current search-based Reinforcement Learning (RL) methods suffer from two core limitations: expensive long-horizon rollouts are under-utilized during training, and supervision is typically available only at the final answer, resulting in severe reward sparsity. We present Prefix-based Rollout reuse for Agentic search with Intermediate Step rEwards (PRAISE), a framework for improving both data efficiency and credit assignment in agentic search training. Given a complete search trajectory, PRAISE extracts prefix states at different search turns, elicits intermediate answers from them, and uses these prefixes both to construct additional training trajectories and to derive step-level rewards from performance differences across prefixes. Our method uses a single shared model for both search policy learning and prefix answer evaluation, enabling joint optimization without extra human annotations or a separate reward model. Experiments on multi-hop QA benchmarks show that PRAISE consistently improves performance over strong baselines. 

---
