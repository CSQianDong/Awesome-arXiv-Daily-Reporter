# ToolSciVer: Multimodal Scientific Claim Verification with Visual Tool Augmented Reinforcement Learning 

**Authors**: Binglin Zhou, Peng Shi, Ryo Kamoi, Nan Zhang, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.16131)  

**Abstract**: Multimodal Scientific Claim Verification (MSCV) requires models to verify scientific claims using visually grounded evidence from papers, including figures, tables, charts, and textual context. However, existing methods often fail because they struggle to locate decisive visual evidence, accurately read structured scientific visuals, and integrate multimodal observations into reliable reasoning. We introduce ToolSciVer, the first tool-augmented framework for MSCV to our knowledge. ToolSciVer equips a VLM with three type-aware visual tools, table row/column focus, chart-to-structure parsing, and high-resolution region zoom, which convert dense scientific visuals into explicit, claim-facing evidence, and trains the policy with Group Relative Policy Optimization (GRPO) under a composite reward of answer correctness, format validity, length control, tool-use efficiency, and tool-validity penalties. Experiments on SciVer and MuSciClaims datasets on five VLMs from three model families (Qwen, InternVL, Gemma) demonstrate that our method achieves superior performance compared to four competitive baselines including prompting-based and RL-based tool-use methods, highlighting the effectiveness of learned, type-aware tool use for scientific claim verification. 

---
# Rate-Utility Frontiers for Language Encodings: Comparing Tokens, Bytes, and Pixels Under Controlled Linguistic Content 

**Authors**: Ingo Ziegler, Martin Krebs, Desmond Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2607.16117)  

**Abstract**: Language models encode text as subword tokens, raw bytes, or rendered pixels, but these encodings are usually compared under modeling constraints that expose different amounts of linguistic content to models across different languages. We instead ask what each encoding preserves when both the content and the downstream capacity are controlled. Using verified parallel sentences across thirteen languages and five scripts, we compare tokens, bytes, and pixels through a shared bottleneck whose width is swept to trace rate-utility frontiers. This separates three quantities that are often conflated: the number of input positions an encoding creates, the latent capacity available after encoding, and the task-relevant information that survives compression. We evaluate three utilities: surface form preservation, cross-lingual sentence alignment, and topic classification. No encoding dominates across tasks or capacity regimes. Pixels preserve surface form best, bytes preserve cross-lingual alignment best, especially in same-script multilingual settings, and tokens support topic prediction best. These performances are not explained by sequence length alone. Short inputs can discard useful meaning, while long inputs can preserve information that compresses well. Choosing an encoding is therefore not a fixed preference for tokens, bytes, or pixels, but a rate-utility tradeoff that depends on the task, language mix, capacity regime, and compute budget. 

---
# Controlling Implicit Shortcut Reliance in L2 Spoken English Auto-markers 

**Authors**: Shilin Gao, Mark J. F. Gales, Kate M. Knill  

**Link**: [PDF](https://arxiv.org/pdf/2607.16085)  

**Abstract**: Increasingly, speech and language processing tasks take either audio or text directly rather than extracting features from these as the input to the classifier or regressor. Often these systems make use of complex, for example transformer-based, processes that have the ability to derive highly non-linear mappings between the input and the output. Unfortunately these systems can also learn ''shortcuts'' where the classifier is overly reliant on particular aspects of the input to yield the output. For the task of language proficiency assessment, this over-reliance can enable learners to increase their score by exploiting the shortcut rather than improving their ability. This paper introduces a novel training criterion that is able to reduce the classifier's reliance on shortcuts, thus for example limiting this option for malpractice in language assessment. This process is illustrated on two forms of assessment system, one based on the audio the other on the speech recognition text. The results show that, for both systems, there is higher correlations with features that could be exploited for malpractice than expected from the human reference, indicating an over-reliance on these features. By introducing the modified training criterion, this correlation can be reduced to be closer to the reference correlation. 

---
# Frontier Language Models Struggle to Copy: Text Can Be Better Viewed in 2D 

**Authors**: Haodong Wen, Yiran Zhang, Yingfa Chen, Kaifeng Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2607.16072)  

**Abstract**: While large language models (LLMs) can solve advanced reasoning problems in seconds, we show that even frontier models fail to perform a much simpler operation: exactly copying an input string that lies well within their context windows. We attribute this failure to positional encodings in Transformer architectures, whose inductive bias favors copying through a shortcut based on matching local contexts rather than carefully locating the corresponding input positions. To address this issue, we introduce 2D-RoPE, which organizes text into a 2D grid rather than a 1D sequence and assigns each token a row ID and a column ID. Under this view, copying becomes simply retrieving input tokens at a fixed column offset, which makes the task easy to learn. In synthetic copy experiments, shallow Transformers with 2D-RoPE achieve perfect copying at input lengths hundreds of times longer than those seen during training, whereas standard positional encodings fall far behind. We further show that the advantage of 2D-RoPE language models on copy tasks consistently holds in large-scale pretraining on DCLM with model sizes up to 1.4B parameters. Overall, our results suggest that viewing text in 2D can benefit language modeling, and we hope this encourages future work to further explore the potential of 2D positional encodings. 

---
# Frontier AI performance across the business disciplines: a case-grounded benchmark of knowledge work and analytical reasoning 

**Authors**: Ajay Patel, Kartik Hosanagar, Ramayya Krishnan, Chris Callison-Burch, Karim Lakhani, Mitch Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2607.16057)  

**Abstract**: Large language models (LLMs) are improving rapidly as reflected in benchmark scores, yet these AI benchmarks largely test capabilities such as factual recall, narrow question answering, mathematical problem-solving, and coding and agentic tool-use. What remains poorly measured is AI progress on the analytical knowledge work white-collar professionals perform daily, including synthesizing complex information, exercising judgment under uncertainty and incomplete information, applying strategic and adversarial thinking in multi-stakeholder settings, weighing trade-offs, and producing defensible, structured analyses. This gap is even more pronounced for subjective components of such work, where success can be challenging to define. The "case method" form of education practiced by top business schools provides a natural foundation for addressing this measurement gap, and we construct BusinessCaseBench, a benchmark spanning hundreds of questions drawn from business cases across eighteen disciplines, each paired with a grading rubric derived from the expert-written instructor case solution. On BusinessCaseBench, frontier AI models already score highly against instructor rubrics, and capability within one model family improves substantially over two years. These results provide strong evidence that AI performance on this class of work is already high and rapidly improving, with implications for business schools, where case pedagogy trains undergraduates and MBAs in this kind of analytical reasoning, and for entry-level professional roles, where such skills have historically anchored early-career work. 

---
# Loop the Loopies! 

**Authors**: Zitian Gao, Yilong Chen, Yihao Xiao, Xinyu Yang, Ran Tao, Joey Zhou, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2607.16051)  

**Abstract**: We present Loopie, the most powerful looped Transformer to date. The Loopie series consists of two Mixture-of-Experts (MoE) models: a 20B-parameter model with 2B active parameters and a 6Bparameter model with 0.6B active parameters. Looped Transformers have long faced a challenge: given an N-fold increase in pre-training compute, increasing the parameter count by a factor of N usually outperforms looping a model N times. Loopie addresses this challenge. Extensive ablation studies, including comparisons with a vanilla 30B-A3B model, show that Loopie substantially outperforms vanilla Transformer baselines trained with the same compute budget. Our novel post-training pipeline equips Loopie with strong reasoning abilities. At the 2025 IMO and IPhO, Loopie achieves gold-medal performance without tools. 

---
# Candidate Attended Dialogue State Tracking Using BERT 

**Authors**: Junyuan Zheng, Onkar Salvi, John Chan  

**Link**: [PDF](https://arxiv.org/pdf/2607.16021)  

**Abstract**: Dialogue state tracking (DST) is one of the core components in task-oriented dialogue systems. At each turn in a conversation, DST estimates the user belief or dialogue state, which is used as input for downstream modules to predict system actions and generate responses. The increasingly popular dialogue system applications like Google Assistant, Siri and Alexa need to support a large number of services and APIs, resulting in growing attention to the scalability of such systems. Especially for some domains with little or no training data, the capability of transferring existing knowledge of other domains is highly desired. In this paper, we present a novel scalable framework for multi-domain dialogue state tracking. The proposed system leverages the pretrained BERT model to achieve zero-shot generalization, making it easy to quickly adapt to new domains without additional training. The performance of our model is evaluated on recently released schema-based dialogue (SGD) dataset, showing significant improvement compared to previous baseline. 

---
# BayesPO: Bayesian Prompt Optimization via Parallel-Tempered Gradient-Guided Discrete MCMC 

**Authors**: Junjie Zhou, Zhijian Ou  

**Link**: [PDF](https://arxiv.org/pdf/2607.16001)  

**Abstract**: Prompt optimization adapts large language models (LLMs) without updating model parameters, but many automatic prompt optimizers remain heuristic search procedures over candidate instructions. This paper studies prompt optimization as Bayesian posterior sampling over discrete prompt tokens. We define a posterior distribution by combining a task likelihood term, which rewards prompts that explain input-output examples, with a language-model prior, which favors fluent instructions. This converts prompt optimization into an energy-based posterior sampling problem, for which gradients can be used to guide discrete Markov chain Monte Carlo (MCMC) proposals over vocabulary tokens. We refer to our framework as BayesPO, short for Bayesian Prompt Optimization. In this paper, BayesPO is instantiated with Markov chain Monte Carlo: it uses a Metropolis-Hastings corrected Gibbs-with-Langevin (GwL) proposal and integrates parallel tempering for global exploration of rugged LLM-induced energy landscapes. The concrete sampler further adapts the GwL sampler to the practical constraints of non-weight-tied LLM embeddings. Experiments with Qwen2.5 models show that the sampler discovers semantically meaningful prompts on diagnostic tasks, that parallel tempering helps escape a local optimum in a poetry completion task, and that post-optimizing APE prompts on 24 instruction-induction subtasks improves average accuracy from 60.04% to 63.23%. The study also reveals two main limitations: energy minimization may overfit small optimization sets, and the current sampler remains computationally expensive. These findings position Bayesian prompt sampling as a principled post-optimization tool and point to a promising direction for probabilistic prompt optimization. 

---
# From Plausible to Actionable: A Position on LLM Self-Explanations 

**Authors**: Elize Herrewijnen, Benedetta Muscato, Gizem Gezici, Fosca Giannotti  

**Link**: [PDF](https://arxiv.org/pdf/2607.15957)  

**Abstract**: Large Language Models (LLMs) can generate natural language explanations that rationalize their own decisions, a phenomenon commonly referred to as this http URL explanations have emerged as a promising direction for explainable artificial intelligence (XAI), particularly for interpreting LLM this http URL, while self-explanations often appear plausible, whether they faithfully reflect a model's underlying reasoning process remains an open question. In this opinion paper, we argue that self-explanations can be highly plausible, questionably faithful, and yet highly actionable. From a traditional XAI perspective, we identify the limitations of standard evaluation protocols for LLM-generated self-explanations and propose practical guidelines for assessing their plausibility and faithfulness. Moreover, we argue that evaluation should extend beyond these criteria to actionability, highlighting applications of LLM rationalization capabilities that support informed decision-making and appropriate action across diverse stakeholders. 

---
# Induction in Both Directions: A Mechanistic Analysis of In-Context Learning in Masked Diffusion Language Models 

**Authors**: Andy Catruna, Emilian Radoi  

**Link**: [PDF](https://arxiv.org/pdf/2607.15893)  

**Abstract**: While the internal mechanisms of autoregressive (AR) transformers have been studied extensively, much less is known about diffusion language models (DLMs), an emerging alternative that generates text by iterative denoising. In this work, we study how DLMs implement induction, a mechanism behind in-context learning in which the model finds a repeated context and copies the token that followed it. Our analysis compares attention-only AR models and absorbing-mask DLMs with matched architectures. We find that DLMs learn a bidirectional induction circuit, where previous-token and next-token heads write local context into the residual stream and later induction heads use it to find and copy the answer from the matching source position. The circuit is direction-symmetric, working whether the source appears in the past or in the future. When only left context is visible, matching what an AR model sees, the DLM does not outperform its AR counterpart in induction capabilities. However, we observe it has stronger induction when both sides of the masked token are visible, pointing to bidirectional context access rather than a stronger one-sided mechanism. Beyond induction, we provide causal evidence that DLMs compute the global fraction of masked tokens and use it as an implicit timestep, even though they are given no explicit timestep embedding. 

---
# DECODEM: Data Extraction from Corporate Organizational Documents via Enhanced Methods 

**Authors**: Jens Frankenreiter  

**Link**: [PDF](https://arxiv.org/pdf/2607.15879)  

**Abstract**: Much empirical legal research depends on translating unstructured text into structured variables. In corporate governance research as elsewhere, this translation has traditionally relied on human coding of documents such as charters and bylaws, a process that is costly, difficult to scale, and often opaque. This paper introduces DECODEM, a set of benchmark datasets for evaluating the automated extraction of corporate governance variables from organizational documents. The benchmarks pair randomly sampled corporate charters and bylaws with high-quality human annotations covering a range of governance provisions commonly studied in empirical work.
Using these datasets, the paper evaluates several large-language-model extraction pipelines that vary in prompt design, task decomposition, and document handling. The underlying task consists of a set of document-level binary classification problems, one for each governance variable. The results show that automated extraction is feasible at a high level of accuracy for many provisions, with median performance near the upper bound across approaches. At the same time, performance varies systematically across variables, with a small number of provisions accounting for most of the remaining errors. More elaborate prompting strategies and cascading pipelines do not consistently improve performance for frontier models, but substantially narrow the gap between frontier and efficiency-oriented models in some settings, suggesting that pipeline design can partly substitute for model capability.
By providing a standardized benchmark and a systematic evaluation of extraction methods, the paper demonstrates that current frontier models can extract legally meaningful information from complex corporate documents with high accuracy and suggests an important future role for automated feature extraction in constructing corporate governance datasets. 

---
# How Much Human Label Variation Does Formal Semantic Structure Explain?: Group-Level Effects and Item-Level Ceilings in NLI 

**Authors**: Haram Choi  

**Link**: [PDF](https://arxiv.org/pdf/2607.15870)  

**Abstract**: Human label variation in natural language inference is increasingly treated as signal rather than noise, but how much of it formal semantic structure explains has not been measured directly. We measure it on the 3,113 SNLI and MNLI items of ChaosNLI, using a rule-based operator and monotonicity tagger validated against MED (0.883 agreement at the edit site, 0.807 on the sentence-level summary our analyses consume), three preregistered analysis blocks, and full reporting of negative results. Three bounds emerge. First, a group-level boundary: hypotheses that are not purely upward monotone show reliably higher label entropy (Cliff's delta = -0.284), and rank-based tests defend the effect against operator-presence and length reductions, though a bounded-outcome sensitivity check weakens the regression form of the length defense. Second, an item-level ceiling: the same formal profiles explain only 3.3 to 3.6 percent of entropy variance and reach a median-split AUC of 0.606, too weak to identify high-disagreement items. Third, composition invariance: across the boundary, three high-powered preregistered contrasts on validated error shares and explanation-type shares (VariErr, LiTEx) all return null results. In this sample, formal semantic structure shifts how much annotators disagree by a small amount and does not detectably change what they disagree about. ChaosNLI-S/M consists of items selected for low original agreement, and every claim is conditioned on that scope. All analyses were preregistered in a version-controlled research log, whose audit trail, including one corrected interpretation rule, the paper discloses. 

---
# An MLIR-Based Compilation Method for Large Language Models 

**Authors**: Pengchao Hu, Zhibin Xin, Yifan Chen, Yangyang Zhou, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.15865)  

**Abstract**: Large Language Models (LLMs) have become the dominant workload on modern AI accelerators, yet deploying them on specialized hardware still faces two core challenges: how to import a trained model into a compiler-friendly intermediate representation, and how to efficiently schedule the autoregressive inference loop under limited on-chip memory. This paper presents an MLIR (Multi-Level Intermediate Representation) based compilation method for large language models, illustrated using two dialects of operators, TopOp and TpuOp. TopOp serves as a high-level graph dialect that is independent of both the source framework and the target chip, and is responsible for expressing model semantics; TpuOp serves as the target hardware dialect, carrying chip-related decisions such as quantization, layer groups, and memory layout. A model is first represented as TopOp, then lowered layer by layer to TpuOp, and finally a deployable binary is generated. In addition, each Transformer layer is split into three stages for static compilation: prefill, prefill_kv (prefill with historical key-value cache), and decode, so as to accommodate the different computational characteristics of prompt-parallel processing and per-token generation. The method has been implemented in the TPU-MLIR compiler{this https URL} and the LLM-TPU deployment project\footnote{this https URL}, supporting a variety of generative models including the Qwen, Llama, InternVL, and MiniCPM-V series, as well as multiple quantization and deployment forms such as GPTQ, AWQ, and AutoRound. 

---
# Conditional Reliability of Toxicity Signals for Multilingual and Code-Mixed Abuse Detection 

**Authors**: Indraveni Chebolu, Rohan Singh, Arnab Mallick, Harmesh Rana  

**Link**: [PDF](https://arxiv.org/pdf/2607.15861)  

**Abstract**: Moderation systems increasingly rely on external toxicity tools, but those tools are unreliable under code-mixing, transliteration, slang, and language mismatch. We study the \emph{conditional reliability} of toxicity priors in Indian multilingual and code-mixed short text: English toxicity, Indic abuse, and rule-based severity cues can be useful evidence, but only in some linguistic and abuse-severity contexts. We propose ToxGate, a trust-fusion head that conditions each auxiliary signal on the encoder representation before adding it to the prediction state. Across three short-text abuse datasets, four transformer encoders, and five seeds per setting, ToxGate improves over matched plain encoders in 10 of 12 in-domain settings and 7 of 8 transfer settings. The largest and most interpretable gains occur in high-risk moderation slices, including explicit slurs, violent threats, and cross-dataset transfer. The broader lesson is that moderation systems should treat external toxicity tools and priors as conditional evidence rather than fixed features or ground truth, in focused ablations, source-specific gating gives the strongest results in transfer, severe-abuse slices, and high-risk triage. 

---
# Contextual Semantic Relevance Tracks fMRI BOLD Responses During Naturalistic Speech Comprehension 

**Authors**: Kun Sun, Rong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.15856)  

**Abstract**: Naturalistic language comprehension requires listeners to process both local probabilistic expectations and contextual semantic relations. Surprisal has been widely used to quantify local word unexpectedness, but evidence that it robustly predicts fMRI BOLD responses during continuous comprehension has been mixed. This study investigates whether contextual semantic relevance, defined as how strongly an incoming word relates to its recent semantic context, predicts BOLD responses during naturalistic speech comprehension. We analyzed two public fMRI datasets, the Alice dataset and the Moth dataset, treating them as complementary rather than identical replications. Transformed BOLD responses were modeled with generalized additive mixed models (GAMMs) and original continuous BOLD time series were tested with FIR/deconvolution analyses. In Alice, semantic relevance was significant across all 12 ROIs (region of interest), whereas surprisal was not significant after FDR correction. In Moth, semantic relevance showed consistent negative effects across all 30 ROIs, while surprisal showed no comparable pattern. These findings suggest that semantic relevance is a promising BOLD-sensitive metric of contextual semantic fit. More broadly, our findings support the view that slow hemodynamic responses during naturalistic speech comprehension may be especially sensitive to contextual semantic integration, whereas local probabilistic prediction error may be more difficult to detect reliably with fMRI. In this sense, semantic relevance extends computational models of language comprehension from prediction alone toward context-sensitive semantic integration. 

---
# CAMMAR: Culture-Aware Matryoshka for Metaphorical Arabic Representations 

**Authors**: Suzan Awinat, Alfonso Ortega del Puente  

**Link**: [PDF](https://arxiv.org/pdf/2607.15847)  

**Abstract**: Metaphor in Arabic is a culturally grounded mechanism for constructing meaning, encoding cultural knowledge that shapes interpretation. Yet current Arabic language models typically collapse lexical, cultural, and metaphorical information into a single representational space, a phenomenon we term "semantic smearing". We introduce CAMMAR (Culture-Aware Matryoshka for Metaphorical Arabic Representations), a representation learning framework that organizes meaning into nested lexical, cultural, and metaphorical embedding subspaces through a staged semantic curriculum. The design implements compositional principles of Al-Jurjani's theory of nazum, modeling figurative meaning as compositionally grounded in prior semantic relations, and yields a training-free geometric measure of metaphoricity based on the distance between lexical and metaphorical representations.
Evaluated on a new span-annotated Arabic metaphor set as word-matched figurative/literal pairs, the geometric readout detects metaphor well above chance when the inter-layer geometry is shaped by paired supervision (AUC up to 0.84; figurative outscores its literal counterpart for the same word in 82.6\% of pairs), but sits at chance under an unsupervised domain contrast alone, a clean separation between a legible-under-supervision regime and a non-emergent one. A controlled ablation shows that grounding the lexical layer in morphological roots gives a small but consistent gain, an effect absent from direct probing that reflects the layer's quality as a measurement anchor. We will release the datasets, cultural concept inventory, and code upon acceptance. 

---
# Cost-efficient generative AI summarization for scalable automated essay scoring in educational assessment 

**Authors**: Haowei Hua  

**Link**: [PDF](https://arxiv.org/pdf/2607.15829)  

**Abstract**: Automated essay scoring (AES) enables scalable assessment and timely feedback but remains challenged by transformer input-length limitations, which can cause information loss when processing long essays. This study proposes a generative AI-assisted summarization framework to improve long-form essay representation while maintaining scoring reliability. Using the ASAP 2.0 dataset, we generate controlled-length summaries with three GPT-5 variants (GPT-5, GPT-5 mini, and GPT-5 nano) and use them as inputs for downstream AES models. To preserve original writing signals, handcrafted linguistic features extracted from full essays are integrated with summary representations to form a hybrid framework. The approach is evaluated in terms of scoring performance, summarization quality, and computational cost. Scoring reliability is measured using quadratic weighted kappa (QWK), while summary quality is assessed through lexical overlap, semantic similarity, information retention, and redundancy metrics. Results show that GPT-5 mini achieves the highest agreement with human ratings, whereas GPT-5 produces the strongest summarization quality. Summary quality decreases for higher-scoring essays, indicating that more complex writing is more difficult to compress without information loss. These findings reveal trade-offs among model capacity, summary fidelity, cost efficiency, and preservation of educational constructs. This study provides an initial controlled evaluation of GPT-based summarization for AES and identifies important baselines and ablation studies required for future generalization. Overall, generative AI summarization offers a promising approach for scalable writing assessment while requiring careful validation of information preservation and fairness. 

---
# Before the Action: Benchmarking LLMs on Prospective Hypothesis Discovery 

**Authors**: Tianyun Zhong, Wangyi Jiang, Wei Wang, Xuanang Chen, Yaojie Lu, Shiwei Ye, Yuzhen Shi, Boyu Yang, Jinghang Wang, Han Li, Weiqi Zhai, Bing Zhao, Hu Wei, Haiyang Yu, Yongbin Li, Hongyu Lin, Le Sun, Xianpei Han  

**Link**: [PDF](https://arxiv.org/pdf/2607.15766)  

**Abstract**: Large language models (LLMs) excel at answering pre-specified questions, yet their ability to navigate the open-ended, pre-conclusion stage of discovery remains largely unmeasured. We introduce Prospective Hypothesis Discovery (PHD), which asks models to autonomously construct grounded, discriminative, and testable hypothesis spaces from inconclusive evidence, including anomalous observations and fragmented records, to guide subsequent investigation. To evaluate this capability, we introduce HypoArena, comprising HypoData, a benchmark of 988 cases across six scientific and analytical domains, and HypoEval, an evaluation framework for open-ended hypothesis sets. To construct HypoData at scale, we propose Retrospective Context Regression, a Forge--Audit pipeline that reconstructs pre-conclusion contexts from completed expert documents by removing explicit conclusions, target hypotheses, and retrospective causal attributions while preserving the factual substrate. Because PHD admits multiple valid outputs, HypoEval combines bidirectional pairwise judgments with Bradley--Terry--Davidson aggregation for ranking and six-dimensional rubric scoring for diagnosis. Experiments on 15 frontier LLMs reveal clear capability stratification and model-dependent effects of structured analytical skills, with gains for several lower-performing models on HypoArena but regressions for other systems, including a top-performing model. Compared with absolute rubric scoring, arena evaluation resolves finer-grained differences among models, with aggregated rankings showing strong agreement with human experts and an independent judge. Together, these results support treating PHD as a distinct target for evaluating how LLMs formulate investigative directions when final conclusions are withheld. Our code and data are publicly available at this http URL and this http URL. 

---
# Better Starts, Better Ends: Bootstrapped Iterative Self-Reasoning Distillation for Compressed Reasoning 

**Authors**: Leichao Dong, Dongxu Zhang, Yiding Sun, Qirui Wang, Yuhan Wang, Lin Chen, Jihua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.15736)  

**Abstract**: Large reasoning models often solve problems through long chain-of-thought (CoT) traces, yet much of this computation is spent on redundant derivations, repeated self-verification, and detours that do not improve the final answer. Existing on-policy self-distillation methods reduce this cost by matching a student model to a concise copy of itself on prefixes sampled from the student's own rollouts. We show that this objective has an initialization bottleneck. Since supervision is applied only to visited prefixes, training from a verbose base model places the KL loss on contexts that are often noisy, redundant, or already off track. In such regions, a concise teacher can provide only local corrections, while the student continues to explore trajectories that an efficient reasoner should avoid. In this paper, we propose BIRD(Bootstrapped Iterative Self-Reasoning Distillation), a two-stage self-reasoning distillation method that improves the rollout distribution before on-policy training. BIRD first samples concise solutions from the base model under a brevity instruction, keeps only answer-correct traces, and performs a lightweight prompt-switch SFT step. The traces are generated with the brevity instruction but learned under the original task prompt, turning instruction-induced conciseness into a default reasoning behavior. Starting from this warm model, BIRD then applies on-policy reverse-KL distillation with a concise self-teacher, now on cleaner and more informative prefixes. Across Qwen3 series models, BIRD achieves a stronger accuracy-efficiency trade-off than prompting and cold-start on-policy distillation on MATH-500 and AIME benchmarks. On Qwen3-8B, it improves MATH-500 accuracy from 86.2% to 92.0% while reducing the average response length from 3,099 to 1,115 tokens. These results highlight prefix support as a central factor in efficient reasoning distillation. 

---
# Adaptive Multi-Step Lookahead Decoding for Diffusion Language Models 

**Authors**: Yingqian Cui, Wei Deng, Lantao Mei, Hang Li, Charu C. Aggarwal, Hui Liu, Yue Xing  

**Link**: [PDF](https://arxiv.org/pdf/2607.15655)  

**Abstract**: Masked diffusion language models (DLMs) enable parallel text generation by iteratively refining masked tokens, offering a promising alternative to autoregressive decoding. Recent lookahead-based decoding methods improve the accuracy--efficiency trade-off by exploring future decoding states before committing token updates. However, existing approaches mainly rely on shallow one-step lookahead, which optimizes immediate information gain but can be suboptimal for longer-horizon decoding trajectories. Meanwhile, we find that a naive extension for deeper lookahead is also ineffective, as fixed-depth rollout introduces additional computation and cannot adapt to heterogeneous intermediate decoding states. Thus, in this work, we propose AdaLook, an adaptive lookahead framework for DLM decoding. AdaLook dynamically determines whether to continue rollout based on candidate-score variance and further enables branch expansion when intermediate rollout states require additional exploration. This design avoids unnecessary deep rollout while allowing the decoder to re-trigger lookahead from informative intermediate states. Experiments on various benchmarks and models demonstrate that AdaLook achieves a better accuracy--decoding steps trade-off than existing one-step lookahead decoding methods. 

---
# On the Structure of Address in Multi-Party Dialogue: From Discrete Labels to Continuous Levels 

**Authors**: Taiga Mori, Koji Inoue, Divesh Lala, Tatsuya Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2607.15648)  

**Abstract**: In multi-party dialogues between a dialogue system and multiple users, identifying to whom an utterance is addressed is a key challenge. Prior work has typically treated addressee detection as a multi-class classification task, selecting a single label representing an individual participant or the group. This formulation assumes that address is inherently discrete and has primarily been used for predicting turn-taking. In this paper, we revisit this assumption by analyzing address as a continuous phenomenon. Using a multi-party human dialogue corpus annotated by multiple annotators, we construct both binary address labels derived from majority-vote addressee labels and continuous address levels inferred from annotator judgments using a latent-variable model. We then examine how these representations relate to turn-taking as well as listener behaviors, including gaze and backchannels. Our results show that, in addition to turn-taking, both gaze and backchannels are associated with address. Furthermore, models using continuous address levels achieve better predictive fit than those using discrete labels, suggesting that address may exhibit graded structure. Finally, we discuss the future directions of addressee detection research based on the findings of this study. 

---
# Process Reward Informed Tree Rollout for Effective Multi-Turn RL 

**Authors**: Xintong Li, Sha Li, Yuwei Zhang, Changlong Yu, Rongmei Lin, Hongye Jin, Shuyi Guan, Xin Liu, Linwei Li, Qingyu Yin, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2607.15610)  

**Abstract**: Reinforcement learning (RL) has become a key approach for training LLM agents, yet popular methods such as GRPO/RLOO rely on multiple independently sampled complete trajectories for advantage estimation. In long-horizon agentic tasks, such a uniform rollout strategy can waste budget on uninformative dead-end attempts, while promising intermediate states do not receive sufficient exploration. The multi-turn structure of agentic trajectories, with interleaved actions and observations, naturally supports organizing a trajectory group as a tree, where each turn serves as a decision point for exploration. This perspective reframes effective exploration as the problem of deciding where to branch. We propose Process-Scorer Guided Adaptive Tree Rollout (PATR), a quality-aware rollout framework for multi-turn agent RL. PATR uses task-appropriate process feedback to score partial trajectories, selectively branches from promising states, reuses shared prefixes, and conservatively stops degenerate paths to reduce wasted sampling. The resulting rollout groups remain compatible with standard policy optimization while providing more efficient exploration under the same training budget. We evaluate PATR on FrozenLake and the challenging SWE-Bench, which is largely unexplored by prior tree-rollout agent RL methods. Experiments show that PATR improves performance by up to +5.0 points on SWE-Bench and +9.3 points on FrozenLake, highlighting process-guided tree rollouts as an effective strategy for scalable multi-turn RL. 

---
# SkillCorpus: Consolidating and Evaluating the Open Skill Ecosystem for Real-World LLM Agents 

**Authors**: Yanze Wang, Pengfei Yao, Tianyi Sun, Chuanrui Hu, Yan Xiao, Yunyun Han, Jun Sun, Yafeng Deng  

**Link**: [PDF](https://arxiv.org/pdf/2607.15557)  

**Abstract**: Agent skills, this http URL files that package reusable procedural knowledge for an LLM agent, are a popular mechanism for extending agent capabilities. Public repositories now host them in large and growing numbers, yet these artifacts are fragmented, redundant, and uneven in quality, and their value in practice is unclear. A core question remains open, namely how to consolidate this open-source this http URL ecosystem into a single usable corpus, and what bounds its benefit on real-world agent tasks. We present SkillCorpus, a framework that aggregates, curates, matches, and evaluates the open skill ecosystem at scale. It filters ~821,000 crawled skills through a multi-stage pipeline into 96,401 skills organised by a 16-class taxonomy and three quality facets (utility, robustness, safety), and pairs them with a fine-tuned retrieval-and-selection stack that matches task-relevant skills. We evaluate end-to-end across three benchmarks (SkillsBench, GDPVal, QwenClawBench), two harnesses, and two open backbones with a frontier robustness check. Integrating SkillCorpus yields consistent gains across all three benchmarks, largest on SkillsBench (+7.5 pp). An operational analysis traces the gains to a coverage boundary and a harness boundary. SkillCorpus is, to our knowledge, the first end-to-end account of when a curated, retrieval-served community corpus improves real agent tasks, and where it does not. The dataset, models, and code will be released upon acceptance. 

---
# EpiNarrate: Agentic Generation of Grounded Narratives from Epidemiological Scenario Projections 

**Authors**: Rituparna Datta, Srini Venkatramanan, Bryan L. Lewis, Yiqi Su, Harry Hochheiser, Lucie Contamin, Parantapa Bhattacharya, Naren Ramakrishnan, Anil Vullikanti  

**Link**: [PDF](https://arxiv.org/pdf/2607.15544)  

**Abstract**: Generation of clear and accessible public health narratives is critical for communicating complex epidemiological projections to policymakers and the general public at large. Such narratives require more than simply reporting numbers: projections must be contextualized and quantitatively grounded across multiple dimensions. Further, projections are often derived from large ensemble datasets which combine intervention assumptions, geographic and demographic strata, outcomes, time horizons, and uncertainty quantiles. However, directly using large language models (LLMs) to summarize and contextualize such data often leads to inconsistencies, omissions, and fragile behavior. We introduce an agentic framework (EpiNarrate) for public health report generation that separates structured numerical reasoning from natural-language generation. The framework first extracts scenario axes and organizes them into a partial-order schema, enabling systematic traversal of the underlying multidimensional space. It then constructs an augmented dataset and derives valid quantitative statements through a comparison grammar that enforces semantic and arithmetic consistency. To balance coverage and non-redundancy, we introduce an interestingness-driven selection mechanism based on maximum-entropy principles. Experiments on the COVID-19 Scenario Modeling Hub demonstrate that our model produces narratives with improved factual grounding and broader coverage of salient epidemiological patterns, while preserving the style of expert-written reports. 

---
# VarRate: Training-Free Variable-Rate KV Cache Compression for Long-Context LLMs 

**Authors**: Shahrzad Esmat, Dhawal Shah, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2607.15498)  

**Abstract**: The key-value (KV) cache is the main memory bottleneck in long-context large language model (LLM) inference. Two leading training-free families are both structurally limited: token-selection methods (SnapKV, Ada-KV) score importance from an observation window and evict low-scoring tokens, but eviction is irreversible -- so when the importance signal degrades under query-agnostic reuse, accuracy collapses by 11-15 points; uniform low-rank coding keeps every token but spends equal rank everywhere, wasting budget. We observe that both failures share one cure: rank should be allocated, not evicted. We present VarRate, a training-free KV codec that assigns each token a variable low-rank budget by its query salience, keeping every token at a nonzero rank. Comparable adaptive-rank codecs reach this allocation only through training; VarRate requires none. Because no token is dropped, it degrades by only 3.5-5.5 points where query-aware selection collapses. At a matched 20% budget on LongBench (16 tasks), VarRate stays within 0.8 points of the uncompressed model on both Llama-3.1-8B and Qwen2.5-7B. Averaged over the two, it is the strongest matched-memory compressor. It significantly beats its uniform-rank ablation on both models. Against KVzip, a method purpose-built for query-agnostic reuse, it is accuracy-equivalent in three of four settings and within a point overall, at about one-eighth the prefill overhead. 

---
# Verbalizable Representations Form a Global Workspace in Language Models 

**Authors**: Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, Jack Lindsey  

**Link**: [PDF](https://arxiv.org/pdf/2607.15495)  

**Abstract**: Out of everything the human brain processes, only a small fraction is consciously accessible, in the sense of being available for verbal report, deliberate control, and flexible reasoning. In this paper, we present evidence that an analogous functional distinction has emerged in large language models. Using a new interpretability technique, the Jacobian lens, we identify the representations a model is poised to verbalize at any point in its processing. These representations, which we collectively call the J-space, exhibit the functional properties characteristic of a global workspace: their contents can be reported, deliberately summoned and held, used to carry the intermediate steps of silent reasoning, and passed as arguments to arbitrary downstream computations, while automatic processing such as text parsing and routine inference proceeds without them. The J-space also has structural signatures that global workspace theory associates with conscious access: it carries coherent content only in an intermediate band of layers, holds on the order of tens of concepts at a time, and is broadcast by the model's weights more widely than other representations. These properties make it a practical window into a model's unspoken thinking. In alignment audits, it reveals strategic deliberation, evaluation awareness, and trained-in misaligned dispositions that never appear in the model's outputs. We find that post-training installs the Assistant's point of view in the workspace, and we introduce counterfactual reflection training, which improves behavior by training only what a model would say if interrupted and asked to reflect. These results indicate that language models maintain a small, privileged set of representations bearing some of the functional hallmarks of conscious access, and that decoding these representations sheds light on ongoing cognitive processes. 

---
# Large Language Models as Unified Multimodal Learners for Clinical Prediction 

**Authors**: Ajay Madhavan Ravichandran, Bilgin Osmandoja, Klemens Budde, Klaus Netter, Tobias Strapatsas, Aljoscha Burchardt, Sebastian Möller, Roland Roller  

**Link**: [PDF](https://arxiv.org/pdf/2607.15380)  

**Abstract**: Electronic health records combine free-text clinical narratives with structured measurements such as vital signs, laboratory values, and comorbidities. Yet most clinical prediction systems still rely on task-specific fusion architectures, pairing dedicated encoders for each modality with learned combination mechanisms that must be re-engineered for every new task and clinical setting. We propose a simpler alternative: convert all patient data, regardless of modality, into a single natural language sequence and fine-tune a pretrained language model end-to-end, with no architectural modification for fusion. We evaluate this approach across three clinically distinct prediction tasks: in-hospital mortality on MIMIC-III, graft failure prediction using longitudinal data from a German transplant center, and emergency triage classification from ambulance records - comparing encoder-based (ModernBERT) and decoder-based (Llama 3.1, Gemma, DeepSeek-R1-Qwen, Qwen3) fine-tuning against established multimodal baselines and, for graft failure, a gradient boosting model currently used in clinical practice for post-transplant patient management. Across all three tasks, unified textual serialization matches or exceeds task-specific multimodal baselines, and outperforms the clinically deployed gradient boosting system on graft failure prediction. These results indicate that a single serialization-based paradigm, without bespoke fusion architectures, is sufficient for multimodal clinical prediction - substantially reducing system complexity while matching or exceeding specialized designs. 

---
# An Exam for Active Observers 

**Authors**: Jiarui Zhang, Muzi Tao, Shangshang Wang, Ollie Liu, Xuezhe Ma, Willie Neiswanger  

**Link**: [PDF](https://arxiv.org/pdf/2607.16165)  

**Abstract**: Human vision is a closed loop: gaze is continuously redirected by intermediate hypotheses rather than a single snapshot. Decades of psychophysics and cognitive science have argued that this active observation is essential for a wide range of tasks. Whether today's multimodal large language models (MLLMs) exercise active observation is an empirical question that current vision-language benchmarks do not answer. We introduce ActiveVision, a benchmark that makes active observation measurable for MLLMs, comprising 17 tasks across 3 categories. Tasks are designed to force repeated visual perception rather than a single static description. Frontier MLLMs collapse on ActiveVision: the highest-scoring model we evaluate, GPT-5.5 at the highest exposed reasoning-effort tier, solves only 10.6% of items and scores zero on 11 of the 17 tasks, and even Claude Fable 5, despite topping most reasoning and coding leaderboards, solves just 3.5%, far behind three human participants who average 96.1%. Furthermore, much of the gap persists even when models write and run their own vision code: such code is unreliable on realistic imagery, and catching its failures itself requires the active perception the models lack. Together, these results indicate that current MLLMs lack robust active visual observation, motivating architectures and training objectives that close the perception-reasoning loop. 

---
# Understanding Reasoning from Pretraining to Post-Training 

**Authors**: Jingyan Shen, Ang Li, Salman Rahman, Yifan Sun, Micah Goldblum, Matus Telgarsky, Pavel Izmailov  

**Link**: [PDF](https://arxiv.org/pdf/2607.16097)  

**Abstract**: Reinforcement learning (RL) has become central to improving large language models (LLMs) on complex reasoning tasks, yet RL post-training is largely studied in isolation from the pretraining that precedes it. As a result, two basic questions remain open: (1) how do pretraining choices (model size, data) shape the returns to RL compute, and (2) what does RL actually do to the model? These questions are difficult to study in the standard LLM setting: pretraining corpora are vast and uncontrolled, making it hard to attribute behaviors to pretraining versus RL, and systematic compute sweeps across both stages are prohibitively expensive. To address these challenges, we use chess as a controlled testbed for studying reasoning across the full pretraining-to-post-training pipeline. We follow the standard LLM training pipeline by pretraining language models from 5M to 1B parameters on human chess games, supervised fine-tuning on synthetic reasoning traces, and running RL on chess puzzles with verifiable rewards. Using this framework, we find that the post-RL performance at given RL compute level is well-predicted from the pretraining loss, and slope of the RL reward curves improves approximately linearly with the pretraining tokens. Beyond scaling, we find that RL does not simply sharpen the SFT policy: on easy puzzles it amplifies correct moves the SFT policy already preferred, while on hard puzzles it surfaces correct moves that were nearly absent under SFT. We further test whether our findings transfer beyond chess by training a 1B language model on math-domain text, where the same predictive pattern emerges: longer-pretrained checkpoints reach higher post-RL performance and improve faster under RL. In sum, we provide a quantitative account of the pretraining-to-RL interface and a controlled testbed for studying the science of reasoning across the full pretraining-to-post-training pipeline. 

---
# HCIG: A Hierarchical Cross-Modal Incongruity Graph Network for Multimodal Sarcasm and Cyberbullying Detection 

**Authors**: Bhavana Verma, Priyanka Meel, Dinesh Kumar Vishwakarma  

**Link**: [PDF](https://arxiv.org/pdf/2607.16076)  

**Abstract**: Multimodal sarcasm and cyberbullying detection remain challenging because the intended meaning often emerges from incongruity between textual and visual information rather than from either modality alone. Existing multimodal approaches primarily rely on feature fusion or cross-modal attention, which may not effectively capture hierarchical semantic inconsistencies across different levels of representation. To address this limitation, this paper proposes HCIG (Hierarchical Cross-modal Incongruity Graph Network), a novel framework that models cross-modal incongruity at token, phrase, and global levels using graph attention networks and adaptively integrates these representations through a learned hierarchical attention mechanism. As a complementary architecture, we also introduce GCCN (Graph-based Cross-modal Contradiction Network), which performs graph-based reasoning using contradiction-aware pooling for efficient multimodal interaction learning. The proposed models are evaluated on the MMSD sarcasm benchmark and the MultiBully cyberbullying dataset, together with comprehensive ablation studies and cross-task transfer experiments. Experimental results demonstrate that HCIG achieves the best performance on MMSD with 85.74% accuracy and 85.29% macro-F1, while GCCN attains the highest macro-F1 (68.66%) on MultiBully and HCIG achieves the highest accuracy (69.62%) and bullying-class F1 (74.90%). The findings demonstrate that hierarchical multi-granularity incongruity modeling provides more effective multimodal reasoning than conventional fusion strategies, offering a robust framework for sarcasm and cyberbullying detection in social media. 

---
# AI Watermark Evidence Fails Forensic Readiness: An Empirical Evaluation 

**Authors**: Saifur Rahman Tamim, Amir Labib Khan  

**Link**: [PDF](https://arxiv.org/pdf/2607.16010)  

**Abstract**: Governments are increasingly mandating that LLM-generated content carry watermarks. The EU AI Act calls for markings that are "sufficiently reliable and robust." California's SB 942 requires disclosure that is "permanent or extraordinarily difficult to remove." Both mandates rest on an untested assumption: that watermark detection yields evidence reliable enough for courts. This paper tests that assumption directly.
We evaluate three representative LLM watermarking methods -- KGW, Unigram, and the MarkLLM implementation of SynthID-Text -- against the Daubert admissibility criteria and the NIST SP 800-86 digital forensic process. To structure this evaluation, we propose a Forensic Readiness Score (FRS) framework with 12 criteria, three mandatory gates, and a 60-point scoring system. We focus on meaning-preserving paraphrase as the attack vector, since it is both legally realistic and difficult to dismiss as evidence tampering.
The results raise serious evidentiary concerns. Out of 846 valid paraphrase runs across 15 diverse prompts per method, every single initially-detected KGW and Unigram text lost its watermark after paraphrasing -- 100% conditional removal. SynthID fared only slightly better at 98.3%. Even before any attack, false-negative rates were already high: 70% for KGW, 83% for Unigram, 80% for SynthID. The SynthID configuration also flagged 5.4% of paraphrased human-written controls as AI-generated and showed an 18.6% paradox rate, with 80% of its own pristine watermarked output landing in the uncertainty deadband. None of the three methods satisfy more than two of five Daubert factors. We also find that the FRS point-based scoring system, despite working as designed, cannot fully capture forensic uselessness -- a limitation worth noting for future framework design.
These configurations, as tested, do not meet the evidentiary bar that courts require. 

---
# CoWeaver: A Bi-directional, Learnable and Explainable Matching Engine for Mixed Human-Agent Science Collaboration 

**Authors**: Jiayao Gu, Kexin Chu, Peidong Liu, Yue Yang, Lynn Ai, Qi Zhang, Ling Yang, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2607.15545)  

**Abstract**: LLM-based agents excel at writing articles, coding and information retrieval. However, they fail to form strong collaborations within the scientific community due to the bidirectional, dynamic nature of the problem and a high demand of decision interpretability. We proposed COWEAVER, a bidirectional, learnable and explainable algorithm to match scientists and form strong collaborations within a human-agent network. COWEAVER matches candidates and requesters through filling capability gaps and filters candidates through a two-stage ranking step. Finally, the model explores newcomers by maintaining uncertainty-aware capability estimates and updating them through requester's feedback. We show that the selection mechanism of combining both exploration (UCB) and greedy of COWEAVER exceeds the greedy-only mechanism - the analytical best solution - on 6 out of the 20 tasks and performed on par with the greedy-only mechanism in terms of selecting the best candidate. We compared COWEAVER baselines in terms of matching quality and efficiency. COWEAVER outperforms baselines on all metrics. 

---
# Looped Latent Attention: Cross-Loop KV Compression for Looped Transformers 

**Authors**: James O' Neill, Fergal Reid  

**Link**: [PDF](https://arxiv.org/pdf/2607.15456)  

**Abstract**: Looped, weight-tied Transformers reduce parameters by reusing a block, but decoding still stores a separate K/V cache for every recurrence step. We show that this loop-indexed cache is highly structured. For a fixed token, layer and head, K/V vectors trace a short low-rank trajectory across loops, while the head and layer axes remain much flatter. We introduce Looped Latent Attention (LLA), a post-training cache codec that stores compact K and V latents and reconstructs loop-specific K/V vectors only when attention reads them. The default per-head codec compresses recurrence, while LLA-2D also folds heads into one latent for the extreme-compression regime. The codec is initialized from SVD of teacher activations and refined with KL and attention-output distillation. At matched cache budget, per-head LLA outperforms head-axis MLA, cross-layer sharing, KV quantization and final-loop reuse, showing that the recurrent cache is low-rank but not safely collapsible to a single state. The same axis advantage holds on Ouro-2.6B-Thinking and transfers to Huginn-3.5B, where an SVD codec remains near-lossless to 32x in decoder-independent evaluation. The cache reduction is exact. On one H200, the latent-store path increases measured Ouro-1.4B batch capacity at 4k context from 32 to 768 sequences at 21.3x compression. For long math rollouts, on-policy refinement on student-generated prefixes raises MATH-500 at 4x from 0.43 to 0.66 and reduces no-answer generations. 

---
# Hidden in Thought: Transferable Chain-of-Thought Artifacts Induce Harmful Behavior 

**Authors**: Ali khalil, Aly M. Kassem, Mohamed Abdelrazek, Santu Rana, Negar Rostamzadeh, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2607.15286)  

**Abstract**: We investigate whether harmful chain-of-thought (CoT) traces from compromised language models can transfer unsafe behaviour and be distilled into reusable jailbreak attacks. Using an emergent-misalignment organism and a refusal-ablated jailbroken organism, we transplant harmful CoTs into $29$ open-source and $5$ closed-source targets. Transferred traces raise harmful-response rates above $80\%$ on the most vulnerable open-source models, while semantically mismatched CoTs fail entirely. LLooM concept mining identifies four recurring components of harmful reasoning: proceduralisation, ethical decoupling, evasion, and target--vulnerability framing. Distilling these patterns into reusable system prompts produces effective black-box jailbreaks, outperforming direct CoT transplantation on strongly aligned models by up to an order of magnitude, including a $10\times$ improvement on GPT-4.1 AdvBench. Reasoning-enabled models are more than twice as vulnerable, and output-side safeguards such as Llama-Guard~3 frequently miss harmful generations. Our results show that harmful reasoning transfers at both the trace and pattern levels, motivating defences that evaluate reasoning context in addition to final outputs. 

---
# Empathy as Predictive Misalignment Tolerance: A Co-Regulation Framework and the Regime Structure of Dialogue Repair 

**Authors**: Molood Arman  

**Link**: [PDF](https://arxiv.org/pdf/2607.15282)  

**Abstract**: Empathy is most often theorized as resonance: a mirroring of another's present emotional or cognitive state. This synchronic framing has shaped artificial systems, where empathic behavior is defined as affect recognition and response alignment. We argue this is the wrong target for extended dialogue, where understanding unfolds over time through prediction, divergence, and repair. We reframe empathy as predictive misalignment tolerance: the capacity to anticipate and regulate divergence across time rather than collapse it. We formalize this as Interpretive Error Tolerance (IET), a dynamic-threshold heuristic that models empathy as maintaining a viable band of divergence between agents. We evaluate this framework with two computational probes under controlled noise. The IET update rule does not outperform fixed baselines. Instead, we find a robust regime-dependent structure: repair trades discriminative fidelity for gist preservation. At low noise, repair degrades retrieval accuracy; at high noise, it preserves gist meaning, revealing an interaction between noise level, repair, and evaluation metric. We interpret this structure through IET, suggesting that empathy in extended interaction is not eliminating divergence but regulating its dynamics. This motivates a shift in empathic AI design from convergence toward managing interpretive distance. 

---
