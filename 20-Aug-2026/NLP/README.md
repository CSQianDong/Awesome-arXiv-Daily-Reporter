# SPADE: Self-Play in Adaptive Synthetic Executable Environments 

**Authors**: Bo Liu, Simon Yu, Yiding Jiang, Ao Qu, Andrew Zhao, Zichen Liu, Junsu Kim, Zijian Zhou, Seungone Kim, Tongzheng Ren, Mickel Liu, Hanfei Yu, Zhaorun Chen, Weiyan Shi, Paul Pu Liang, Luke Zettlemoyer, Yejin Choi, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2608.19197)  

**Abstract**: Continuous self-improvement requires an ever-expanding pool of self-generated, diverse, adaptive goals. For language agents, existing training environment pools (hand-curated, statically synthesized, or frozen-verifier) keep the goal distribution fixed as the learner scales. We introduce SPADE (Self-Play in Adaptive Synthetic Executable Environments), a self-play RL framework in which a single LLM plays two roles: an Environment Designer that writes complete, long-horizon training environments as executable code with an OpenAI Gym-style reset()/step() interface, and a Reasoning Agent that learns to act in them. Each is a stateful, multi-turn environment (state transitions, reward functions, and verification code), so one interface spans reasoning problems and multi-step agentic tool use. The Reasoning Agent's regret is estimated using the gap between its reward with and without privileged hints; in optimizing this regret signal the Environment Designer learns to target environments at the edge of the agent's capabilities while keeping them feasible. Through extensive experimentation, we find several components critical to success: grounding the Environment Designer on documents sampled from a large pretraining corpus, and giving it an accumulated environment memory. Scaling to 30B-parameter models, SPADE improves over the strongest fixed-environment baseline by +5.3 on average across eight held-out math, science, code, and reasoning benchmarks, and lifts the tool-use setting by +5.7 on BFCL-v4 multi-turn and +13.9 on ACEBench-Agent; on the games setting, the margin over the strongest baseline grows with model scale. By making environment design itself a learnable component, SPADE takes a concrete step toward open-ended self-improvement. 

---
# ChildSafeAds Shared Task 2026: Commercial Content in Child-Facing YouTube Videos 

**Authors**: Thales Bertaglia, Catalina Goanta, Gerasimos Spanakis, Gunes Acar  

**Link**: [PDF](https://arxiv.org/pdf/2608.19165)  

**Abstract**: ChildSafeAds is a shared task on commercial content in YouTube videos likely to reach children and teenagers. It contains 3,360 videos from 939 channels. Each instance begins with a segment submitted to SponsorBlock, an open-source crowdsourced browser extension whose users mark sponsor segments so that others can skip them. We pair the segment with its available transcript, video and channel information, and a sales or service page linked from the video description. Systems determine what kind of offer is being promoted (ST1), assign product categories (ST2), and identify legal risk flags (ST3). The evidence is divided into four cumulative access levels, from the transcript to the linked page, so results can be compared against the cost of collecting the data. 45.5\% of videos in our data failed to properly use the in-platform ad disclosure method (the ``Includes paid promotion'' label). GPT-5.4 produced the labels after the expert organiser team reviewed samples and iterated on the taxonomy, prompts and model choices. GPT-5.6-luna independently labelled the development set. This report describes the task, data and evaluation. An updated version will add participating systems and shared-task results. 

---
# Comment-level Topic Drift Analysis in the Reddit Corpus 

**Authors**: Steven Morse, Daniel Runfola, Trenton W. Ford  

**Link**: [PDF](https://arxiv.org/pdf/2608.19133)  

**Abstract**: We present a novel application of embedding-based dynamic topic modeling techniques to detect and quantify topic drift at the comment level in a massive corpus. By leveraging pretrained language models to generate contextualized semantic embeddings for short text, we analyzed 12.7 billion Reddit comments spanning 2006 to 2022. Using unsupervised methods on these embeddings, we identify dynamically evolving topic clusters over time. Our primary contribution is a methodology for analysis of semantic drift and discourse evolution in the embedding space itself. We also demonstrate modifications to existing methods that enable this analysis at scale, and we propose and demonstrate a null model comparison test to filter spurious dynamics. Key findings suggest that politically and socially contentious topics exhibit significant directional drift in embedding space, with inter-topic distances changing systematically over time beyond what the null model can explain, whereas domains such as music and sports remain comparatively stable. 

---
# Intercepting the Kangaroo: Experimental Astrolinguistics with Constructed Lexicons, Active Probing, and Large Language Models as Informants and Hypothesis Proposers 

**Authors**: Francesco Cordella, Mauro Cappelli  

**Link**: [PDF](https://arxiv.org/pdf/2608.19124)  

**Abstract**: Astrolinguistics -- communication with minds that categorize reality differently from ours -- has been purely speculative since Freudenthal's Lincos (1960). We make it experimental. Two language models with deliberately incompatible constructed lexicons (one encoding shape, color, and motion; the other fusing color with motion, encoding parity, and lacking shape) serve as informants with complete ground truth, while a fully scripted orchestrator translates between the two category systems. The central failure mode is the kangaroo effect: the silent attachment of a word to the wrong referent -- Quine's indeterminacy of translation, operationalized. Across 400+ simulated and live runs, a protocol combining cross-situational elimination, pre-registered predictive probes, active scene selection, a stricter recovery round, and quarantine produced no undetected mistranslations under the tested conditions and exceeded a passive baseline's coverage (d = 0.62). Injected kangaroo traps defeated naive ostension and pure statistical learning in 100% of runs, while the full protocol intercepted every decoy and, where discriminating evidence is ontologically unavailable, declared Quinean equivalence classes instead of guessing. Under informant noise it degrades gracefully: zero kangaroos persist up to 2% per-word noise; at 10% the protocol predominantly abstains rather than errs. Finally, words outside the scripted hypothesis space (a history-dependent relational term and an XOR contextual homonym) are recovered by a generate-and-test loop in which an LLM proposes rules and the script verifies them: coverage scales with proposer capability (0% -> 18% -> 72% -> 100%) while undetected mistranslations stayed at zero throughout. In the tested conditions, correctness is a property of the protocol; coverage is a property of the instruments. 

---
# Institutional Books - Enriched Text: A customizable multilingual open-source pipeline for denoising, deduplicating, and annotating OCR text at scale 

**Authors**: David Lowry-Duda, Matteo Cargnelutti, Catherine Brobston, Salwa Ismail, Greg Leppert, Amanda Watson, Jonathan Zittrain  

**Link**: [PDF](https://arxiv.org/pdf/2608.19026)  

**Abstract**: Released in 2025, Institutional Books: Harvard Library (IB-HL) is a collection of 983,004 volumes (242B o200k_base tokens), originally digitized through Harvard Library's participation in the Google Books Library project. As researchers and developers have begun to use IB-HL, a tension has emerged between standard large-scale preprocessing practices and the goals of careful information stewardship. Many existing pipelines optimize for web text: as a result, they tend to aggressively filter, deduplicate, restrict by language, and sometimes discard meaningful metadata. Meanwhile, researchers seeking to use IB-HL duplicate effort while performing similar processing and analysis.
We describe an approach that we call Enriched Text. Instead of producing a single 'complete' stream of tokens, we normalize the text while preserving metadata through annotations. We separate endmatter, detect per-paragraph language, identify clusters of duplicate paragraphs, and compute per-paragraph bits-per-byte scores. We provide this information through HTML-like annotations layered on top of the text. By parsing these annotations, users can tailor the output to their own needs instead of accepting a global editorial decision on content. The pipeline applies to all $\approx$250 languages in the collection.
This report describes this project's goals, implementation, and design rationale. The release includes IB-HL-ET (an enriched-text version of IB-HL containing 217B o200k_base tokens across 983,003 volumes, organized into 1.39B annotated subtopic paragraphs) and the pipeline that produced it. These serve to make the collection easier for machines to parse and for humans to study. 

---
# Grading the Graders: Verification Autonomy Levels (L0-L5) for LLM Reasoning 

**Authors**: Yajie Yin  

**Link**: [PDF](https://arxiv.org/pdf/2608.19009)  

**Abstract**: Large language models (LLMs) are increasingly paired with verifiers (step checkers, self-consistency filters, tool-based fact checkers, formal proof assistants) that claim to detect the model's errors. Yet the verification literature uses the word "level" to mean at least five different things: verification granularity, concept abstraction, risk tier, system-stack layer, and the epistemic source of the ground truth. We propose Verification Autonomy Levels (VAL), a meta-standard classifying verification schemes along a single axis: where does the verification spec come from, and what does the verdict guarantee? VAL ranges from L0 (LLM self-declaration, no deterministic anchor) through L2 (objective ground truth, correctness only) to L3/L4 (decidable systems with single-property or domain-level completeness), with L5 impossible in the unrestricted case. Central to VAL is the completeness blind spot: substitution- and sampling-based verifiers can confirm that proposed candidates hold, but cannot prove that no candidate was missed. We further identify a dichotomy the literature has not stated: completeness is reachable only for formally specifiable properties, while empirical open-world verification (fact-checking, diagnosis) caps at anchored correctness (L2). We document this across four domains (symbolic mathematics, behavior monitoring, medical diagnosis, and code generation) and in the strongest existing formal-verification baseline, whose authors note the verifier "focuses on the correctness of each step." We show the levels of granularity, concept hierarchy, risk, and system stack are orthogonal to VAL, resolving a systematic conflation across 17 surveyed papers. Code and full assessment are released as supplementary material. 

---
# Introducing the Privacy-HSD Trade-off: Hate Speech Detection, but not at the Cost of Privacy 

**Authors**: Stephen Meisenbacher, Vlad Garbuz, Chirill Donos, Maxim Dnestreanschii, Gabriel Creanga, Andreea-Elena Bodea, Thomas Lampert, Jana Diesner  

**Link**: [PDF](https://arxiv.org/pdf/2608.19006)  

**Abstract**: Hate speech is a real and timely threat that affects a large portion of online users, especially youth and minority groups. While building reliable and robust automatic hate speech detection (HSD) systems is paramount, we argue that this must also be balanced with the individual right to privacy. Exploring the intersection of HSD and privacy, we demonstrate that HSD systems might unintentionally achieve performance at the cost of encoding authorship, posing a threat to privacy. Building on these findings, we establish the notion of a privacy-HSD trade-off, which demands a careful balance. We benchmark a series of text privatization methods, as well as our newly proposed domain-specific AgnoSpeech technique, showing that balancing privacy and HSD is difficult but feasible. The findings make a strong case for more research on the trade-offs between privacy and HSD, both of which have tangible implications for the safeguarding of online participation. 

---
# Structure, Association, and Decision Value: Representation-Based Difficulty Estimation for Adaptive Inference in African-Language NLI 

**Authors**: Toheeb Ogunade  

**Link**: [PDF](https://arxiv.org/pdf/2608.19003)  

**Abstract**: We ask whether internal representation statistics can provide useful example-level difficulty signals for adaptive inference in multilingual African NLP, and find that they cannot in this setting. Studying natural language inference across 15 African languages with frozen off-the-shelf checkpoints, we report four results. First, AfriXNLI's English configuration shares 1,047 of its 1,050 examples verbatim with XNLI evaluation data, and one widely used NLI checkpoint scores 1.000 on that test split, consistent with XNLI test exposure. Because AfriXNLI is derived from XNLI, its English, French and Swahili configurations cannot serve as clean evaluations for XNLI-trained models. Second, parameter count does not reliably order capability across African languages: our larger checkpoint is better in seven languages and worse in eight, with no significant aggregate difference. Third, across three multilingual representation spaces, angular dispersion is consistently more language-determined than effective rank, so pooled correlations can inflate one and mask the other. Fourth, the association that survives language control depends on the target: effective rank predicts probability gain from escalation but not whether escalation changes the prediction, while cheap-model confidence shows the opposite pattern; the two targets correlate at only 0.655. Under the tested models, signals, and compute budgets, no evaluated signal makes adaptive routing preferable to always-expensive inference, although an oracle exceeds it by 11 accuracy points at 60% of the compute. Our central methodological finding is that a representation statistic can be statistically significant for one notion of computational benefit while being irrelevant to another, and therefore be a poor decision variable. 

---
# DeepWeaver: Bridging the Evidence Synthesis Gap in Open-Ended Question Answering 

**Authors**: Xujia Wang, Yizhe Zhang, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.18988)  

**Abstract**: Retrieve-then-generate pipelines are commonly used to produce deep-research answers for open-ended questions, but retrieval alone is insufficient: LLMs must organize noisy and fragmented evidence into comprehensive, well-cited answers. We refer to this process as evidence synthesis. However, direct generation often underuses evidence, misaligns citations, and collapses diverse information into shallow summaries, exposing an evidence synthesis gap between retrieval and generation. Thus, we propose DeepWeaver, a novel framework that weaves noisy retrieved evidence into comprehensive answers by maintaining Thought Block Chains (TBCs), a structured representation that groups claims, salient information, keywords, and supporting evidence. DeepWeaver uses subordinate TBCs to inspect residual evidence, commit TBC revisions, and discover new claims before final generation. We evaluate DeepWeaver on open-ended QA over both knowledge bases and the web, and introduce LoQA, a high-density benchmark for evidence synthesis. Across multiple LLMs, DeepWeaver improves content sufficiency, citation grounding, and detail preservation on LoQA, while achieving deeper insights and higher citation quality on DeepResearch Bench. These results show that evidence weaving is an effective mechanism for bridging retrieval and generation in open-ended QA. Our code is available at this https URL. 

---
# Institutional Newspapers Pipeline: Deriving billions of high quality tokens from historical newspapers 

**Authors**: Matteo Cargnelutti, Catherine Brobston, Eben English, Jake Sadow, Kacie Bailey, Greg Leppert, Amanda Watson, Jessica Chapel, Jonathan Zittrain  

**Link**: [PDF](https://arxiv.org/pdf/2608.18972)  

**Abstract**: Historical newspapers are an abundant record of public life, but their dense, irregular and sometimes noisy layouts make computational access to these materials both challenging and limited. We present the Institutional Newspapers Pipeline, a modular system we jointly designed with Boston Public Library to extract high-quality, structured datasets from historical newspaper scans. It was architected so that each step remains interpretable and customizable, and so that the pipeline as a whole remains computationally frugal enough to run on workstation-level hardware. The pipeline runs each scan through a multi-step process: it segments scans into individual type-agnostic crops and performs OCR on each resulting segment before then performing text analysis, type classification, reading order detection, named entities recognition, subject classification, language detection, and pre-computed embeddings generation on every crop. We ran this pipeline against a portion of Boston Public Library's holdings and released the results as an open dataset. The optical character recognition (OCR) output represents 16.3 billion o200k_base tokens across 83.1 million individual crops, extracted from 1,473,635 public domain newspaper scans published between 1795 and 1930. This report describes our methods for each processing step, the small models we trained, as well as the evaluation results and dataset-scale measurements we collected in the process. It accompanies the release of the pipeline, models, and dataset. We position this work as a substantial step towards unlocking high-quality data from tens of millions of newspaper scans. 

---
# MedUAG: Unified Understanding and Generation for Medical Multimodal Models 

**Authors**: Zijie Meng, Yuncheng Zhang, Hualiang Wang, Yitian Tang, Xiaotang Gai, Chen Shen, Songtao Jiang, Shaosheng Cao, Jian Wu, Xian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18937)  

**Abstract**: Recent Multimodal Large Language Models (MLLMs) are rapidly evolving into unified understanding and generation (UAG) frameworks. However, extending these unified paradigms to the medical domain is hindered by: the absence of comprehensive training and evaluation benchmarks, and the lack of broadly validated unified medical model. To address these gaps, we present a comprehensive foundation for medical UAG. First, we construct MedUAGCorpus, the largest unified medical understanding and generation dataset to date, comprising over 6 million instances across 14 imaging modalities. Second, we introduce MedUAGBench, a systematic benchmark that expands medical generation evaluation to 12 diverse tasks under standardized protocols. Finally, leveraging these resources, we develop MedUAG, an end-to-end trained unified medical model. Extensive experiments demonstrate that MedUAG achieves strong performance across a wide array of understanding and generation tasks, establishing a competitive baseline and paving the way for next-generation medical multimodal systems. 

---
# Test-Time Scaling in the Wild: Why Exploitation, Not Exploration, Is the Bottleneck 

**Authors**: Davide Romano, Kanak Raj, Jerrod Parker, Daniele Giofrè  

**Link**: [PDF](https://arxiv.org/pdf/2608.18931)  

**Abstract**: Test-time scaling (TTS) improves language model outputs by spending additional inference compute - generating multiple candidates, searching over partial sequences, or iteratively refining drafts. These techniques yield large gains on mathematics and code, but have been developed and stress-tested almost exclusively on tasks where verification is straightforward. We conduct the first compute-normalised comparison of five TTS families across five open-ended generation benchmarks spanning medicine, law, finance, general chat, and creative writing - grounded in a unified framework that decomposes the effectiveness of each method's token budget into exploration and exploitation. The answer depends on which side of that decomposition you examine. Scaling exploration works: the best candidate in the pool improves steadily with compute across all settings. What breaks is exploitation - the step that converts a rich candidate pool into a final output. With state-of-the-art generators, reward models correlate at only $\rho_v \approx 0.12$ with true quality, rendering selection near-random regardless of budget. Tree search amplifies this failure through diversity collapse. Refinement helps on one of five benchmarks; its apparent gains elsewhere are confounded. Only synthesis across candidates (Fusion) consistently improves over single-sample baselines, yet still recovers only ~40% of available quality. The candidate pool is not the bottleneck - choosing from it is. 

---
# SMTrap: Cost-Effective DoS Attacks Against Large Reasoning Models via SMT Conflict Guidance 

**Authors**: Jian Yang, Zhenqi Feng, Zhaoyang Yu, Zhaoxin Fan, Kejian Wu, Xiaofeng Wang, Zheng Zhu, Jianjun Huang, Wei You, Bin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18921)  

**Abstract**: Existing LRM-DoS methods rely heavily on model feedback to synthesize attack queries, requiring either repeated queries to the target model or training a dedicated attack model. These expensive operations severely weaken attack leverage. In this paper, we propose \emph{search amplification}, a novel, model-feedback-free LRM-DoS paradigm. It employs the conflict count derived from an Satisfiability Modulo Theories (SMT) solver as a low-cost external signal to guide the synthesis of inference-heavy Constraint Satisfaction Problem (CSP) instances. Our key observation is that LRMs depend on trial-and-backtracking search when solving CSPs, where higher SMT conflict counts on a given CSP instance positively correlate with more extensive LRM backtracking search and substantially longer output trajectories. Building on this finding, we propose \textsc{SMTrap}, a lightweight, CPU-only framework. Guided by SMT conflict counts, \textsc{SMTrap} generates inference-heavy CSP queries without model queries, attack-model training, or GPU computation. Evaluations across seven frontier models demonstrate the state-of-the-art LRM-DoS capability of \textsc{SMTrap}, producing DoS effects multiple times stronger than existing baselines. To mitigate the threat of \textsc{SMTrap}, we demonstrate a tool-based mitigation that significantly cuts token usage. 

---
# Assessing Quality of Experience in Natural Language Generation of German Text 

**Authors**: Dinh Nam Pham, Shushen Manakhimova, Vivien Macketanz, Sebastian Möller  

**Link**: [PDF](https://arxiv.org/pdf/2608.18888)  

**Abstract**: The rapid advancement of Natural Language Generation (NLG) has made the reliable evaluation of generated text increasingly critical, as these systems, such as large language models (LLMs), are now widely deployed in real-world applications. However, traditional automatic metrics fail to capture the multifaceted nature of perceived quality. In this paper, we introduce TextQ-German, a novel dataset suite for human-centered evaluation of German NLG from a Quality of Experience (QoE) perspective, covering automatic text summarization and machine translation. Through crowdsourcing studies with German speakers, we collect human quality ratings and identify relevant perceptual quality dimensions for each task. We develop automatic QoE prediction models, including transformer-based, linguistic feature-based, and hybrid approaches. Hybrid models outperform pure transformer baselines in almost all experimental settings, while linguistic features alone can approach the performance of fine-tuned language models. The dataset is extended with LLM-generated outputs annotated with overall QoE scores. Final validation on held-out sets indicates generalization to unseen data. Our work contributes a publicly accessible resource for NLG evaluation and baselines for automatic QoE prediction, providing a foundation for developing NLG systems that better align with human quality perception. 

---
# Understanding Multilingual Medical ASR Adaptation Through Layer-Wise Analysis 

**Authors**: Souranil Kahali, Rituparna Bose, Abner Hernandez, Tomas Arias-Vergara, Andreas Maier, Ning Ma, Paula Andrea Perez-Toro  

**Link**: [PDF](https://arxiv.org/pdf/2608.18825)  

**Abstract**: Medical automatic speech recognition (MedASR) requires adaptation to specialised terminology, limited annotated clinical data, and multilingual use cases. Although large-scale pretrained ASR models such as Whisper achieve strong generalisation, their behaviour after medical and multilingual adaptation remains insufficiently understood beyond word error rate (WER). This paper investigates how multilingual medical adaptation reshapes the internal representations of Whisper models through layer-wise encoder analysis. We compare zero-shot decoding, English-only fine-tuning, German-only diagnostic fine-tuning, two-stage EN->EN+DE continuation, and direct EN+DE fine-tuning across Whisper model sizes. Fine-tuning substantially improves MedASR performance, but the best model depends on the adaptation setting: Whisper-Medium gives the lowest English WER (7.72%) and the lowest combined EN+DE WER under direct EN+DE training (26.30%); German-only Whisper-Large-v3 gives the lowest German WER (44.96%), but as a within-corpus diagnostic on 86 single-speaker training utterances rather than robust generalisation. Layer-wise analysis of the two-stage Whisper-Small trajectory shows that English medical fine-tuning produces the dominant encoder shift, whereas multilingual continuation largely preserves the adapted representation space. Domain and language information remain highly recoverable across layers, while linearly recoverable error-predictive cues weaken as WER improves. 

---
# Identifying Implicit Premises for Logical Reconstruction of Argument Graphs 

**Authors**: Xuyao Feng, Anthony Hunter  

**Link**: [PDF](https://arxiv.org/pdf/2608.18821)  

**Abstract**: The logical reconstruction of argument graphs from natural language text is challenging because of the prevalence of enthymemes (i.e., arguments with implicit premises). There are natural language processing methods for identifying enthymemes in text, and there are symbolic methods based on abduction for identifying missing premises in a logical representation of enthymemes. However, there is a need for methods to generate implicit premises to logically show a known entailment or contradiction relationship between a pair of statements. To address this, we propose a neuro-symbolic pipeline that uses large language models (LLMs) to generate intermediate implicit premises that are translated into logical formulae and used with logical formulae representing explicit premises and explicit claims to show the logical relationships between them (entailment, contradiction, or neutrality). Our approach is evaluated on the Microtext Argumentative Corpus. 

---
# Do Large Language Models Hallucinate Electric Fata Morganas? 

**Authors**: Kristina Šekrst  

**Link**: [PDF](https://arxiv.org/pdf/2608.18816)  

**Abstract**: AI hallucinations - that is, outputs which are made up, cannot be verified, or contradict the source material - are generally regarded as an engineering flaw to be dealt with. This paper contends that they also have philosophical significance when it comes to the question of machine consciousness. We examine the known causes of hallucinations in large language models - such as source-target divergence, discrepancies between training and inference, and overfitting - and we present two empirical investigations. In the first, we apply successive generations of the GPT model to ambiguous factual questions under different temperature settings, finding that higher temperatures result in plausible but incorrect answers while lower temperatures lead to factually accurate ones. The sampling parameters that cause a model to seem creative or spontaneous and thus more likely to pass behavioral tests of intelligence are the same ones that increase its hallucination rate. In the second, we look at an encoder-only model that has been trained on encyclopedic data and which answers questions of the same type factually and without embellishment, indicating that hallucinations are due to exposure to subjective and socially diverse training data rather than to the development of any cognitive ability. Using references to Turing, Searle's Chinese Room, the frame problem, and the cybernetic tradition of Wiener and Ashby, we claim that a model's self-reports of emotion or sentience come within the definition of hallucination, and that any future occurrence of machine consciousness might remain epistemically inaccessible since it would be indistinguishable from a sufficiently advanced hallucination. 

---
# Decomposing Wrong-Consensus Agreement in LLM Self-Consistency: A GPT-4.1 Case Study 

**Authors**: Lizhuo Zhang, Mengmeng Tang, Chenfeng Long, Xiaoyong Tang, Xiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.18795)  

**Abstract**: Majority voting over multiple LLM samples is widely used to raise answer accuracy, yet its gain varies erratically: on hard questions it can even backfire. This paper gives a quantitative account of this failure. A pluralistic agreement index Gamma is defined as the expected fraction of the samples of a wrong run that agree with the consensus, normalized by a reference scale d=(1-p)/(C-1), and is decomposed into a mechanical component (what a vote delivers given only a per-case answer preference) and a preference-unexplained residual. The mechanical null is difficulty-matched and leak-free: each case is resimulated at its own accuracy and option preference, estimated from the case's other runs, so no run predicts its own agreement. On GPT-4.1 the decomposition shows benchmark-associated direction (an observational ordering over n=4 cells per benchmark, not a significance claim). On multiple-choice GPQA-Diamond, the per-case answer preference explains 81-93% of the held-out test-run agreement index: the shared-bias-dominates account over-claims here, because a wrong but attractive option the whole cohort latches onto is captured by the per-case preference channel (whether that preference is induced by shared training bias is not identified). On open-domain AIME, the mechanical preference explains only 59-78% (21-29% if shrunk to pure noise), and a preference-unexplained residual of 1.56-2.80 Gamma units survives, which a run-level preference-heterogeneity reference more than absorbs (1.4-2.1). A self-consistency backfire on hard questions is reproduced (binned voting gap down to -0.09, coupled CI [-0.12,-0.07]), and the highest-agreement bin reaches an accuracy of only 0.42-0.83, a 1.2-3.6x lift over base rate: agreement is graded evidence, not certification. No new voting method is proposed; code and evidence are committed and reproducible. 

---
# Readable, Faithful, Used: Three Dissociable Properties of Demographic Identity in a Language Model 

**Authors**: Fathin Difa Robbani  

**Link**: [PDF](https://arxiv.org/pdf/2608.18768)  

**Abstract**: Large language models are widely used to simulate survey respondents, yet their answers are homogeneous and unfaithful to real inter-group differences. We ask where demographic group identity lives inside an LLM, how faithfully its geometry mirrors real inter-group opinion structure, and whether it uses what it encodes. Using representational similarity analysis against Pew ground truth over 169 demographic cells, we score 1,089 read-out locations in Mistral-7B and intervene causally across six attribute types. Four results. (1) The standard last-token residual read-out understates the model: attention-head read-outs dominate it in five of six types, with selection-corrected fidelity up to rho=0.63 -- roughly 70% of the measurement-reliability ceiling -- surviving a lexical-similarity control. (2) A single head (L11 H16) is significantly faithful in all six types as a fixed location, while race-based types stay weak and prompt-fragile. Both phenomena replicate -- the analogous head significant in five of six types, weakest on the same race type -- across three checkpoints of a second model family, where ten billion training tokens barely move the map. (3) Causal use does not follow fidelity: the clearest causal pathway sits in one of the least faithful types (p=0.002, cluster-robust, fixed depth), the most faithful type shows no correction-surviving single-layer effect, and replacing the entire identity moves predictions by under 2% of their error. (4) A 128-dimensional probe of the single head lands 21-31% closer to survey truth than the model's own answers -- yet recovers almost none of the per-question group ordering, no better than the answers themselves. Readable, faithfully arranged, and causally used are three dissociable properties of the same model; treating them as one claim is what keeps the "can LLMs simulate populations" debate unresolved. 

---
# Gradient Mirage: Trainable yet Label-Unidentifiable Gradients in Large Language Model Split Learning 

**Authors**: Shiyu Miao, Yunlong Mao, Zirui Huang, Liang Yao, Tianshuo Zheng, Yanhui Gu, Fan Liu, Sheng Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2608.18767)  

**Abstract**: Gradient matching attacks (GMAs) in LLM split learning (SL) rely on a critical yet underexplored assumption: the gradient exposed at the split interface is a faithful derivative of the client's full-label training objective. This gradient-objective consistency allows a curious server to recover private labels by searching for a sequence whose induced gradient explains the observation. We propose Gradient Mirage, a defense that breaks this consistency without discarding the optimization utility of the backward signal. Our key idea is to induce the adversary to solve a misspecified inverse problem, in which no plausible label sequence in the sequence space can explain the observed gradients. Concretely, Gradient Mirage achieves this by inducing inconsistency across three dimensions: objective, direction, and scale. Selective Autoregressive Supervision derives the exposed gradient from a masked surrogate loss rather than the full-label objective assumed by the attacker; Scale Blinding then applies randomized multiplicative rescaling, obscuring the gradient's natural magnitude; and Directional Privatization further randomizes the gradient direction while preserving its magnitude through the von Mises-Fisher (vMF) mechanism under a directional metric differential privacy guarantee. Crucially, utility is preserved: the Top segment still learns from all target tokens via Dual-Track Backpropagation, the exposed gradient remains informative since each supervised token retains its complete autoregressive context, and Bottom-Gradient Recovery restores the effective gradient for Bottom-segment optimization. Extensive experiments show that Gradient Mirage provides substantially stronger protection than existing defenses under comparable fine-tuning performance, achieving a better privacy-utility trade-off. 

---
# Learning Canonical Register Automata over Ordered Data Domains 

**Authors**: Yong Li, Qiyi Tang, Di-De Yen  

**Link**: [PDF](https://arxiv.org/pdf/2608.18765)  

**Abstract**: Register automata are finite automata equipped with memory that recognize data languages over infinite alphabets. In this work, we investigate active learning algorithms for deterministic register automata (DRAs) over ordered data domains--covering both dense domains, such as the rationals, and non-dense domains such as the integers. We show that the active learning problem for DRAs over both dense and non-dense ordered domains can be treated within a single unified framework. More specifically, we develop and implement a polynomial-time active learning procedure for DRAs over ordered domains, using oracles for membership, equivalence and memorability queries. The memorability queries were originally introduced for learning DRAs over domains with identity tests. Our unified framework also leads to a new consequence: minimization of DRAs over the non-dense ordered domain of integers is decidable, extending a result previously known only for dense domains. Finally, we give improved complexity bounds of several decision problems for DRAs over ordered domains that are closely related to the queries used in active learning. 

---
# Execution-grounded evaluation reveals hidden failures in language-model calculations for environmental science 

**Authors**: Maohao Ran, Chendong Ma, Yanting Zhang, Dailing Jiang, Yusen Huang, Meng Gao, Jun Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.18726)  

**Abstract**: Large language models are increasingly used for quantitative work in the environmental sciences, yet existing evaluations score only final answers, leaving calculation process unobserved. Here we introduce AtmosCoder-Bench, an execution-grounded benchmark that makes the calculation process visible. Built through a transferable semi-automated pipeline (436 problems, 3,910 variants, 7,029 graded quantities), every problem is validated to be unambiguous and human-solvable, with uniquely verifiable answers. We find that (i) multiple-choice formats inflate measured accuracy by at least 12 percentage points; (ii) many failures arise not from missing knowledge but from models failing to apply known formulas and constraints consistently throughout multi-step computation; and (iii) even frontier models remain weak when task-specific conditions invalidate familiar methods, often reverting to canonical solution patterns rather than adapting methods to the relevant physical regime, leaving expert oversight essential. 

---
# Budget-First Tariff Recommendation (BFTR): A Complete Algorithmic Framework for Telecom Plan Recommendation without Overcharging 

**Authors**: Ghislain Dorian Tchuente Mondjo  

**Link**: [PDF](https://arxiv.org/pdf/2608.18723)  

**Abstract**: Telecom operators traditionally offer predefined tariff grids, forcing users to choose from a limited set of plans. This paper proposes BFTR (Budget-First Tariff Recommendation), a complete algorithmic framework integrating eight Budget-First strategies, including two original hybrid approaches: Recursive Hybrid (conditional interpolation) and Knapsack-First Hybrid (priority knapsack). Unlike existing approaches that adjust prices upward to guarantee a minimum margin, BFTR guarantees the absence of overcharging by systematically aligning the final price with the catalog reference price. We mathematically formalize each strategy, prove the existence of an offer for any positive budget, and prove that the price deviation (surcharge) is zero for all strategies that do not use interpolation with correction. A detailed comparative analysis confronts BFTR to ten main existing tariff models on ten dimensions. Experiments on a dataset of 974 customers inspired by the Nigerian MTN market show that: (i) Recursive Hybrid is optimal for the customer (100% budget used, 29.9 GB volume, utility 0.946, 0% overcharging), (ii) Piecewise offers the highest volume (39.7 GB) with 0% overcharging, (iii) Power Law provides an excellent compromise (99.9% budget, 38.1 GB, 0% overcharging). All strategies achieve a zero surcharge, confirming the theoretical guarantees. A sensitivity analysis on the weighting parameter alpha (0.2 - volume priority, 0.5 - balance, 0.8 - budget priority) shows that utility rankings evolve logically. Execution times (< 10 ms) and very low failure rates (0% for robust strategies) confirm the operational viability of the system. The formal proof of the absence of overcharging constitutes a major theoretical contribution. 

---
# MemFuse: Multi-Source Memory Fusion from Fragmented Observations 

**Authors**: Chao Li, Yuanfa Li, Wenhao Wu, Xule Liu, Zhi Wang, Kun Shao  

**Link**: [PDF](https://arxiv.org/pdf/2608.18704)  

**Abstract**: Long-term memory is essential for agents that operate across extended interactions, yet existing memory systems and benchmarks predominantly focus on single-source textual histories. In realistic settings, however, relevant information is often fragmented across applications and devices, as well as across users and time, requiring agents to integrate dispersed observations into coherent episodic memories while preserving their source provenance. To address these gaps, we introduce **MemFuseBench**, a benchmark for *multi-source memory fusion*. MemFuseBench is built with a Scene-to-Sensor pipeline that synthesizes controllable scenarios into source-tagged observations, evidence-grounded questions, and adversarial distractors. It enables systematic evaluation of temporal reasoning, cross-source evidence fusion, and robustness to noise. We further propose **MemFuse**, a structured memory system that preserves source-level evidence in event-layer atomic memory and organizes related atomic events into cluster-layer fused memory within a causal fusion graph. During retrieval, MemFuse retrieves and organizes related evidence fragments while maintaining traceability to original source events. Experiments on MemFuseBench show that MemFuse achieves the best overall performance among the evaluated memory systems under all three LLM settings and consistently improves performance on questions requiring cross-source evidence fusion. 

---
# Aslema at NADI 2026: Augmentation through Fewshot for SLU 

**Authors**: Tajwaar Shafiq, Hunzalah Hassan Bhatti, Shammur Absar Chowdhury, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2608.18689)  

**Abstract**: We present Aslema, our system for NADI 2026 Shared Task 5, which consists of two subtasks: intent recognition and slot filling. We evaluate four omni LLMs in a zero-shot setting and compare them with fine-tuned models. Our results show that fine-tuning consistently outperforms zero-shot inference. We further explore synthetic data augmentation by using an LLM to generate culturally grounded Tunisian Derja utterances, followed by voice cloning to generate synthetic speech. Incorporating this synthetic data improves performance on both tasks. Our final submitted system, based on Qwen3-Omni-30B and trained with a mixture of original and synthetic data, achieves 86.8% intent accuracy and 34.7 WER on the devtest split. On the official test set it ranks 1st in slot filling (59.5 CoER) and 4th among 8 teams in intent recognition (66.1% accuracy). We release our experimental scripts and will soon share the synthetic dataset to support further research in this area. 

---
# Learning What to Fail On: Failure-Mode Contextual Bandits for Adversarial Data Curation 

**Authors**: Roie Kazoom, Ofir Cohen, Rami Puzis, Asaf Shabtai, Ofer Hadar  

**Link**: [PDF](https://arxiv.org/pdf/2608.18681)  

**Abstract**: We introduce a failure-aware adversarial retrieval-augmented framework for improving robustness in natural language understanding. Rather than selecting synthetic examples with a fixed reward threshold, our method formulates adversarial data curation as a failure-mode contextual bandit problem. Candidate examples are generated with retrieval-augmented prompting, filtered by the current target model, automatically validated by an LLM judge ensemble, and clustered into recurring failure modes. A stochastic policy then selects which failure modes to sample for retraining, and is updated using validation-based reward that balances robustness gains, forgetting, and data cost. This makes the data curator itself the learning agent, enabling adaptive selection of the most useful model failures across training rounds. On standard benchmarks, our approach improves RoBERTa-base accuracy from 88.48% to 92.60% on SNLI, from 75.04% to 80.95% on ANLI, and from 54.67% to 71.99% on MultiNLI, while consistently outperforming prior adversarial augmentation methods. We further demonstrate transfer to FEVER fact verification, achieving up to 79.86\% FEVER score and 82.45\% accuracy with RoBERTa-large. Finally, we provide a theoretical interpretation showing that, under stated assumptions, failure-mode sampling can reduce shortcut-aligned gradient contributions while inducing bounded distributional drift. By combining retrieval, automated validation, contextual-bandit failure selection, and controlled adversarial retraining, our framework enables scalable robustness improvement without additional human annotation. 

---
# X2Streaming-TTS: Causal Token-Level Text-to-Speech from Streaming Text with Speech-State Inheritance 

**Authors**: Rime Wen, Zehan Liu, Shawn Qin, Lights Shi, Roy Gan, Hao Wang, Qian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18661)  

**Abstract**: Streaming text-to-speech is essential for low-latency spoken dialogue systems, yet many systems wait for sentence-level text and are therefore only pseudo-streaming. True token-level synthesis must generate speech from uncertain prefixes while maintaining perceptual continuity over an unbounded stream with bounded context. We present X2Streaming-TTS, a causal TTS framework that consumes asynchronously arriving text tokens and emits speech without accessing future input. To handle uncertain prefixes, we introduce causal commitment, which keeps ambiguous expressions provisional through uncertainty-aware buffering and performs capacity-adaptive, punctuation-aware segmentation. To preserve acoustic continuity, we further introduce causal speech-state inheritance, which carries the complete Code2Wav state and selected historical Talker states across segment boundaries. Together with an attention prior constraint, it blocks access to future positions while retaining bounded acoustic context. Experiments show that X2Streaming-TTS outperforms existing pseudo-streaming models on most subjective and objective metrics. Further analysis shows that causal commitment stabilizes online segmentation and reduces failures caused by insufficient context, while speech-state inheritance improves boundary continuity without degrading naturalness or speaker identity. X2Streaming-TTS thus achieves strict token-level synthesis with quality comparable to the evaluated offline baselines, a median time to first audio token (TTFT) of 15.8 ms for a single request, and a median TTFT of 260.8 ms at 128 concurrent requests. Our implementation is publicly available at this https URL . 

---
# TranslatePsy-AfriSLM: High-Quality Data Scaling For Low-Resource Machine Translation 

**Authors**: Milan Gritta, Patrik Lambert, Jihye Back, Amril Nazir  

**Link**: [PDF](https://arxiv.org/pdf/2608.18655)  

**Abstract**: The rapid progress in Artificial Intelligence has largely bypassed African languages, creating a digital divide that limits AI adoption on the continent. Recent open-source LLMs systematically underperform on African machine translation, while the lack of large-scale, high-quality, open-source parallel data has constrained the development of competitive small language models (SLMs). We introduce *TranslatePsy-AfriSLM*, a collection of open-source MT resources for 19 Sub-Saharan African languages, including curated parallel data, African-specialized synthetic data, and a family of fine-tuned SLMs. Our empirical study shows that unified quality-estimation filtering removes up to 96% of training tokens without degrading quality, and that filtered synthetic data dominates the quality-efficiency Pareto frontier. Fine-tuned on the resulting data mixture, TranslatePsy-AfriSLM outperforms substantially larger systems, including TranslateGemma-27B and Qwen3.5-122B-A10B, with as few as 0.8B parameters. 

---
# From Storage to Access: Verifiable Activation of Parametric Knowledge in LLMs via Explicit Priming and Implicit Reasoning 

**Authors**: Zuocheng Ying, Yang Yang, Yumou Wu, Chuanbo Zhu, Jiarui Wang, Ziqi Wu, Jingming Cai, Junqing Yu, Zikai Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.18581)  

**Abstract**: Although Large Language Models (LLMs) encode rich factual knowledge in their parameters, reliably recalling and verifying such knowledge remains a key bottleneck in factual question answering. Existing end-to-end methods entangle knowledge elicitation with reasoning, making it difficult to determine whether correct answers arise from parametric knowledge or the input context. To address this challenge, we propose VAKE (Verifiable Activation of Parametric KnowledgE), a two-stage reinforcement-learning framework that externalizes latent parametric knowledge through explicit Priming and transfers the acquired elicitation capability to implicit Reasoning. Given a query and an insufficient retrieved subgraph, the Priming policy explicitly inserts bridging triples as verifiable evidence, with supervision provided by rewards derived from answers generated by a separate frozen model over the augmented subgraph. Building on the policy learned during Priming, the Reasoning stage trains the model to answer from the original input, testing whether the capability acquired through explicit knowledge elicitation transfers to implicit reasoning. Experiments across seven benchmarks and models from 3B to 14B show that VAKE consistently outperforms standard baselines, including when transferring directly from HotpotQA to OOD datasets. LLM-based evaluation further shows that over 80% of the inserted triples provide factual bridging knowledge not derivable from the retrieved context, while more than half elicit knowledge inaccessible through direct prompting. These results suggest that VAKE activates latent parametric knowledge rather than copying the input context or memorizing dataset-specific associations. 

---
# Compress and Forget: bitsandbytes Quantization Amplifies Proactive Interference in LLMs 

**Authors**: Shayan Shahrabi-Farahani, Dara Rahmati  

**Link**: [PDF](https://arxiv.org/pdf/2608.18578)  

**Abstract**: Proactive interference (PI) is a documented failure mode in large language models in which retrieval of a repeatedly overwritten value degrades as prior overwrites accumulate, mirroring a classical phenomenon in human working memory. Post-training quantization (PTQ) is now the default deployment path for open-weight models, yet its effect on this failure mode has not been tested. We evaluate three precision levels (FP16, INT8, INT4/NF4, via bitsandbytes) across three architecturally distinct instruction-tuned models (Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Phi-3.5-mini-instruct), holding the retrieval task fixed. INT4 quantization significantly reduces accuracy under high interference in every model (e.g., from 81.0% to 68.3% for Qwen), confirmed by paired McNemar's tests ($p \le 2.6 \times 10^{-6}$) and a mixed-effects regression spanning all interference levels; INT8, often assumed safe, also carries a smaller but real penalty in two of three models. The effect is specific to semantically similar (word-type) distractors and reverses sign under a numeric control condition, and is mechanistically linked to a rise in same-key intrusion errors under INT4 (from 21.5% to 24.6% of trials, $p = 4.8 \times 10^{-7}$). A follow-up ablation shows the effect originates in the quantized transformer backbone rather than the output projection layer. These results suggest that bitsandbytes 4-bit quantization can impose an additional cost on applications relying on long, updatable, semantically dense contexts, even when aggregate benchmark accuracy appears largely unaffected. We release our code and tokenizer-verified vocabulary construction method at this https URL 

---
# Beyond LLM-Based Reasoning: Lightweight GNNs for Agent Failure Attribution 

**Authors**: Ting-Wei Li, Yuanchen Bei, Xiao Lin, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2608.18575)  

**Abstract**: Large language model (LLM)-based multi-agent systems (MAS) often exhibit complex failure modes, which frequently cause agents to produce incorrect outcomes. This motivates the task of Agent Failure Attribution: given a failed multi-agent trajectory, identify the faulty agents and their corresponding error types. Existing approaches predominantly rely on LLMs to perform failure attribution, either through direct prompting, fine-tuning on synthetic data or complex agentic pipelines. While effective, these methods incur substantial computational overhead due to long-context processing, expensive post-training and handcrafted workflows. Moreover, empirical evidence shows that even state-of-the-art models achieve limited accuracy on existing benchmarks, suggesting that scaling model size alone is insufficient. In this work, we revisit this task and question the necessity of such expensive generative solutions. We introduce AFANet, a lightweight graph-based framework that models interaction trajectories through step-level semantic signals and agent-level relationships. We show that with significantly fewer parameters and near-zero inference cost, AFANet (i) matches or outperforms LLM-based baselines, including fine-tuned models on in-domain benchmarks, (ii) maintains robust performance across different GNN architectures and (iii) can be further improved with inexpensive test-time adaptation on the OOD benchmark. Our results suggest that effective agent failure attribution does not require heavy LLM reasoning and a lightweight, structured approach can achieve strong performance. 

---
# Shared Circuits for Shared Grammar: Tracing Subject-Verb Agreement Across Languages 

**Authors**: Isabella Gidi, Antonio Almudévar, Core Francisco Park, Naomi Saphra, Ricard Marxer  

**Link**: [PDF](https://arxiv.org/pdf/2608.18545)  

**Abstract**: Multilingual large language models often generalize across languages, and prior work suggests that their internal mechanisms can overlap cross-lingually. It remains unclear, however, when such sharing emerges and whether it varies with the overt realization of the same grammatical operation. We investigate this question for present-tense subject-verb agreement, a morphosyntactic process that varies substantially across languages and is only weakly expressed in English. Using activation patching and attention analysis across 29 languages and five open-source model families, we identify the attention heads causally implicated in agreement and compare these head-level signatures across languages. We find that languages with overt person/number inflection exhibit more similar agreement circuitry than non-conjugating languages, with the strongest sharing appearing when the analysis isolates recovery of the inflectional contrast itself. English provides an informative bridge case, becoming more similar to conjugating languages precisely in contexts where overt agreement is required. Finally, many implicated heads display similar attention patterns across languages, suggesting that cross-lingual overlap reflects shared functional roles as well as shared localization. Together, these results indicate that multilingual LLMs reuse partially shared computational structure for morphosyntactic agreement rather than relying on fully separate language-specific solutions. 

---
# DART-SD: Diamond-topology Aware Retrieval and Tuning for Self-Distillation of Multi-Turn Tool-Calling Agents 

**Authors**: Hangrui Xu, Jiarui Wang, Yang Yang, Chuanbo Zhu, Fangda Chen, Ziqi Wu, Jingming Cai, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.18524)  

**Abstract**: Equipping Large Language Models (LLMs) with multi-turn tool-calling capabilities is essential for building autonomous agents. However, progress is fundamentally limited by the reliance on full-length trajectory imitation. For tasks involving multiple order-independent sub-goals, the optimal solution space forms a vast combinatorial diamond lattice. Forcing this rich topology into monolithic trajectories causes a severe topological collapse, indiscriminately penalizing valid alternative explorations and severely degrading policy diversity. To address this, we propose DART-SD (Diamond-topology Aware Retrieval and Tuning for Self-Distillation), a novel framework that shifts the paradigm from global forcing to topology-guided localized correction. DART-SD first models the execution process as a converging Interaction-State Transition Graph (ISTG), faithfully capturing the inherent diamond topology of successful and failed exploratory paths. During autonomous rollouts, the framework identifies the Critical Topological Breakpoint (CTB) and retrieves success-supported recovery references. Finally, we introduce a progressive self-distillation paradigm through CTB-guided localized supervision, ensuring that the training loss is calculated exclusively on the generated recovery steps while strictly protecting the valid reasoning prefix from destructive gradient updates. Experiments on complex multi-turn tool-calling benchmarks demonstrate that DART-SD significantly outperforms traditional full-trajectory baselines. 

---
# MissDiag: Diagnostic Evaluation of Incomplete-Knowledge Robustness in KGQA and KG-RAG 

**Authors**: Hang Wang, Hang Dong, Lu Liu, Chuanru Ren  

**Link**: [PDF](https://arxiv.org/pdf/2608.18489)  

**Abstract**: Knowledge graph question answering (KGQA) and knowledge-graph-based retrieval-augmented generation (KG-RAG) aim to ground answers in explicit graph evidence, but real-world knowledge graphs are often sparse, outdated, and incomplete. Existing robustness evaluations usually report aggregate changes in answer quality after evidence is removed or perturbed, which measures sensitivity to incomplete support but leaves the source of degradation under-specified: the same score change can conflate the type of missing evidence, the response of the evaluated system, and the sensitivity of the answer-matching protocol. To address this gap, we propose \textbf{MissDiag}, a diagnostic evaluation framework for incomplete-knowledge robustness in KGQA and KG-RAG. MissDiag keeps the question and gold answer fixed while applying structurally typed missingness interventions to benchmark-provided support graphs, enabling paired comparisons that decompose robustness changes by evidence type, system response, and evaluation protocol rather than reducing them to a single aggregate score drop. Experiments across multiple system families show that incomplete-knowledge robustness is better understood as a typed degradation phenomenon than as a uniform property: answer-adjacent evidence loss produces the largest observed degradation, source-context removal is often neutral and can be beneficial, and semantic answer matching changes absolute scores while preserving the main typed degradation patterns. By transforming aggregate robustness measurement into typed diagnostic attribution, MissDiag provides a more interpretable basis for comparing, diagnosing, and stress-testing KGQA and KG-RAG systems under incomplete knowledge. 

---
# WhiteMatter: All-to-All Cross-Layer Connections via KV Mixing 

**Authors**: Wenbo Zhang, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2608.18486)  

**Abstract**: In a Transformer, each layer attends to past tokens only through KV produced at its own depth, despite the presence of deeper representations during autoregressive decoding. Feedback architectures allow shallow consumer layers to attend to KV produced by deeper past-token representations, but give all consumer layers the same fixed connection patterns to source layers. We propose WhiteMatter, which connects every attention layer to the representations from all layers of each past token, with connection weights that can vary across consumer layers and adapt to the source token. For each token, a router implements these connections by mixing its $L$ layer states into $k$ KV channels that are cached for subsequent tokens; each consumer layer attends to one of the channels. The number of channels $k$ controls the KV-cache size. Setting $k<L$ reduces the cache's memory footprint. In our pretraining experiments, WhiteMatter outperforms a vanilla Transformer with 50% more layers and retains most of this gain with a 50% KV-cache compression. 

---
# OmniAlign: A Unified Multilingual Aligner for Word and Sentence Alignment 

**Authors**: Mengpeng Yang, Jingxu Yang, Chao Chen, Tian Xia, Yabo Sun, Qiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18474)  

**Abstract**: Cross-lingual sequence alignment is fundamental for building and exploiting parallel corpora, spanning mappings from documents and sentences down to words and subwords. Existing tools, however, typically specialize in a single granularity, so practitioners often need separate systems for word- and sentence-level alignment---especially in multilingual and long-text settings. We present OmniAlign, a unified multilingual aligner that supports both word-level and sentence-level alignment with a single lightweight model. Built on an encoder-only backbone with strong long-context modeling, OmniAlign induces word alignments from contextualized token similarity matrices, and obtains document-level $m$--$n$ sentence alignments via sentence embeddings combined with dynamic programming. To balance fine-grained alignment accuracy and sentence-representation quality, we use a four-stage training pipeline: alignment-oriented continued pre-training, self-supervised learning, supervised fine-tuning on human annotations, and sentence-embedding distillation from a strong multilingual teacher. Experiments show that OmniAlign achieves highly competitive performance on both word- and sentence-alignment benchmarks and generalizes well to unseen language pairs. Surprisingly, later-stage supervised fine-tuning on short texts further improves alignment quality while retaining the long-context understanding acquired in earlier training, keeping the model robust on long-text word alignment.
\normalsize {\color{blue}\textbf{Code}: this https URL}\par {\color{blue}\textbf{Model}: this https URL} 

---
# Pedagogical AI in Mental Health: A Tri-Stream Fine-Tuned LLM Framework for Automated Clinical Supervision and Risk Triage 

**Authors**: Shreeya Sharma, Ravish Gupta, Saket Kumar, Abhishek Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2608.18438)  

**Abstract**: Modern mental healthcare faces a critical shortage of senior supervisory oversight, leading to a "supervision gap" where novice therapists manage high-stakes risks with delayed professional feedback. This paper proposes a new framework utilizing a fine-tuned Mistral-7B-instruct model as an automated "Supervisor-in-the-Loop" system. By leveraging 106 sessions from the DAIC-WOZ dataset, the model performs a tri-stream analysis: (1) Therapeutic Alliance tracking via semantic adherence, (2) Latent risk prediction using attention-weighted analytics, and (3) Supervisory Triage via a Dynamic Clinical Urgency Index (D-CUI). Our multi-modal VAL (Visual-Acoustic-Linguistic) framework achieves 95% technique identification accuracy [95% CI: 75.1%-99.9%], alliance assessment MAE of 0.105 on a 5-point scale [95% CI: 0.059-0.151], therapeutic fidelity alpha = 0.423, and mean D-CUI of 0.370 [95% CI: 0.322-0.419]. Training converged in 105 steps with 85.2% loss reduction on a single Tesla T4 GPU. The system reduces supervisory triage latency from 72 hours to real time (~10 seconds per session), enabling proactive intervention in high-risk cases. The system addresses the cold-start problem through Bayesian priors and implements timestamp-based modality synchronization for robust multi-modal fusion. 

---
# Tangut Word Segmentation under Extreme Resource Scarcity: Integrating Traditional Lexicons and Unlabeled Text 

**Authors**: Lifan Deng, Yongwei Zhang, Sen Sun, Bojun Sun, Jingsong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18437)  

**Abstract**: Tangut is an extinct language whose script does not explicitly mark word boundaries. We present the first systematic study of Tangut word segmentation using 2,750 expert-annotated segments(31,893 tokens), traditional lexicons, and unlabeled text. Our framework combines a reliability-calibrated lexicon-lattice representation, explicit distributional statistics, and a lightweight character encoder pretrained with MLM. Segment-level five-fold cross-validation shows that lexical and statistical features raise CRF F1 to approximately 0.91. The full TangutEncoder reaches the highest mean F1 (0.911) and improves recall beyond the labeled training vocabulary. These results demonstrate generalization beyond the limited supervised vocabulary across thematically diverse held-out passages, while document-level transfer remains to be evaluated. 

---
# Figurative and Cultural Knowledge in LLMs: Investigating Cross-Domain Transfer through Fine-Tuning 

**Authors**: Mena Attia, Mona Diab, Thamar Solorio  

**Link**: [PDF](https://arxiv.org/pdf/2608.18361)  

**Abstract**: Figurative language is deeply culturally embedded; fluent use requires not just linguistic competence but cultural immersion. We ask whether LLMs can learn this link: does fine-tuning on cultural data improve figurative language understanding, and vice versa? We conduct a systematic study across four models (ALLaM-7B, Fanar-1-9B, Qwen3-8B, Llama-3.1-8B) and six Arabic datasets spanning cultural commonsense, proverbs, and poetry across diverse dialects and regions. Fine-tuning on poetry improves idiom comprehension (+2.33%, p<0.05), a gain our ArabicMMLU control does not reproduce, indicating that it stems from figurative content rather than Arabic language adaptation and pointing to a sensitivity to non-literal meaning that transfers across figurative types. Cultural fine-tuning, by contrast, lowers proverb-interpretation accuracy in both Arabic-centric models. Transfer between the two domains is otherwise indistinguishable from noise, with Arabic models frequently regressing after fine-tuning, suggesting prior saturation of relevant knowledge, while multilingual models show greater adaptation headroom. Error analysis further reveals that fine-tuning reinforces experiential cultural knowledge while destabilizing historically grounded factual knowledge. Our findings suggest that the relationship between culture and figurative language, though conceptually natural, is not straightforwardly captured through fine-tuning alone. 

---
# Artifact-centered Claim-aware Observability for Autonomous Scientific Agents 

**Authors**: Xiangyu Yin, Ming Du, Michael H. Prince, Mathew J. Cherukara  

**Link**: [PDF](https://arxiv.org/pdf/2608.18312)  

**Abstract**: Autonomous scientific agents now increasingly propose ideas, write code, run experiments, analyze results, and even draft papers. Observe and audit those agents are necessary but logging every model call is not enough, scientists also need to inspect the artifacts and claims that the systems produced and their relations. This is driven by the fact that failures in scientific agent systems are often distributed across several objects. A manuscript claim may cite the wrong evidence, a search process may select a degenerate candidate, a laboratory novelty claim may depend on an unstated rule, or a multi-agent plan may change without a visible trigger. Existing tracing, experiment tracking, and archival provenance tools are valuable, but their native objects do not make these scientific audit relations first-class. We argue that autonomous scientific systems should emit portable, claim-aware artifact lineage as a minimum audit layer. We propose a compact observability profile organized around individuals, operators, fitness records, lineage, archives, runs, streams, and steering commands. In this profile, scientific claims are ordinary individuals with explicit evidence bindings and verification records. The profile is intended as a semantic layer that complements current telemetry and provenance standards. Execution details can remain in OpenTelemetry. Final packages can export to PROV-O or RO-Crate standards. 

---
# Efficient INT8 Inference of Small NLP Models on Server CPUs with PyTorch Native Stack 

**Authors**: Weiwen Xia, Yuxin Cui, E Cao  

**Link**: [PDF](https://arxiv.org/pdf/2608.18182)  

**Abstract**: Small NLP models, especially BERT-family encoders, remain important in industrial workloads such as classification, ranking, and retrieval even in the era of large language models. On server CPUs, INT8 quantization offers an attractive latency-throughput-cost trade-off, but users increasingly expect such acceleration to be available directly in the native PyTorch stack. We integrate SmoothQuant into TorchAO and optimize the resulting inference path for Intel Xeon CPUs through graph-level fusion in TorchInductor and efficient INT8 GEMM kernel selection across oneDNN-, AVX512_VNNI-, and AMX-based implementations. Across BERT, DistilBERT, and XLM-RoBERTa benchmarks, the approach delivers up to 5.8x end-to-end throughput speedup with negligible---and in some cases no measurable---accuracy loss relative to the FP32 baseline. We also validated our work by detailed performance analysis with roofline models. The implementation has been upstreamed to PyTorch and TorchAO, enabling out-of-the-box deployment with native PyTorch tooling 

---
# Are LLMs Safe Beyond Text: Do Emojis Expose Gaps in Safety Evaluation 

**Authors**: M P V S Gopinadh  

**Link**: [PDF](https://arxiv.org/pdf/2608.18164)  

**Abstract**: Safety evaluations of large language models (LLMs) predominantly rely on text-based adversarial prompts, potentially overlooking vulnerabilities arising from alternative input representations. This work examines emoji-augmented prompts as a test case for this gap, evaluating 50 prompts across four open-source LLMs (Mistral 7B, Qwen 2 7B, Gemma 2 9B, Llama 3 8B). Results show substantial variation in robustness: Gemma 2 9B and Mistral 7B exhibit non-zero success rates (10%), Llama 3 8B 6%, while Qwen 2 7B shows complete resistance (0% success rate). A chi-square test ($\chi^2 = 32.94, p < 0.001$) confirms significant differences in outcome distributions. These findings indicate that robustness is sensitive to input representation, and that evaluations restricted to standard text prompts may underrepresent model vulnerabilities. 

---
# When Do LLMs Actually Help? Evaluating LLMs as Data Quality Annotators 

**Authors**: Praphulla Lal Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2608.18158)  

**Abstract**: LLMs have been increasingly used to catch data quality issues automatically, but we know very little about how consistent these judgments actually are. This study tests an LLM on two e-commerce data quality tasks, entity matching and brand mislabeling, against rule based baselines and human verified ground truth, under both zero-shot and few-shot prompting. On entity matching while using the Abt Buy benchmark (2,194 labeled pairs), a simple rule based baseline (F1=0.950) performed about as well as LLM zero shot prompting (F1=0.948). Moreover, a few-shot prompt revision that looked effective on a small validation sample reduced full-scale performance to F1=0.914. This showed that small sample prompt evaluation can be misleading. On brand mislabeling detection, using 500 Amazon product listings with synthetically injected labeling errors, the LLM clearly outperformed a naive rule based baseline (F1=0.833 vs 0.721), because it could draw on background knowledge of brand product relationships that a simple rule could not access. Testing consistency across repeated runs (200 pairs, 5 runs at temperature 0.7) showed the model agreeing with itself 99.7% of the time on average, with 99% of pairs giving identical answers across all 5 runs. Using majority voting across these runs only improved F1 by 0.005, at 5 times the inference cost. These results suggest that the value of using an LLM over traditional methods depends heavily on the task. LLMs offer little advantage when strong lexical signals already exist, but a clear advantage when the task requires background knowledge, all while remaining highly consistent across repeated queries. 

---
# The Deontic Gap: Large Language Models and the Modal Language of Obligation 

**Authors**: Daniel Hart, Sarah Allred, Joseph Abbas, Morenike Alugo  

**Link**: [PDF](https://arxiv.org/pdf/2608.18144)  

**Abstract**: Modal auxiliaries such as must, should, and have to mark necessity and obligation within the contexts of speaker authority and interpersonal stance. We examine whether large language models (LLMs) reproduce contemporary human patterns of deontic modal usage. Across three primary corpora, an external benchmark, two controlled replications, and a naturalistic eleven-model replication, AI-generated text consistently underuses positive deontic modals (must, should, have to, had to) relative to contemporary humans. Historical comparison with the Google Books Ngram corpus (1920-2022), used as a heuristic calibration against the published-prose record, shows that AI modal frequencies fall within the range of formal published English, whereas contemporary human modal rates in informal digital contexts often exceed twentieth-century book baselines. Phrase-level decomposition shows that the AI-human modal gap is concentrated in constructions central to interpersonal stance (should, have to, had to), while AI matches or exceeds humans on need to in instructional and question-answering contexts but not in persuasive student writing, indicating that the modal profile is genre-conditional. The findings suggest that LLM modal usage reflects the formal written resources on which these models were trained, while underusing the modal constructions through which contemporary human writers mark immediate, interpersonal obligation. 

---
# Language Models for Portuguese: A Systematic Mapping Study 

**Authors**: Jhessica Silva, Carlos Caetano, Helena Maia, Breno Bernard Nicolau de França, Sandra Avila, Helio Pedrini  

**Link**: [PDF](https://arxiv.org/pdf/2608.18138)  

**Abstract**: In recent years, the rapid development of language models has transformed the field of Natural Language Processing through a wide range of applications. However, the development of language models has not progressed uniformly across all languages. In the case of the Portuguese language, there has recently been a growing effort by academia and companies to develop language models and create data resources for Portuguese. These efforts have resulted in the rise of an increasingly diverse ecosystem of language models for Portuguese. However, information on these models remains dispersed in scientific publications, technical reports, model repositories, and project documentation. This survey presents a systematic mapping study of language models developed for Portuguese, providing a comprehensive overview of the current state of the field. We map a total of 46 models, characterizing them by various aspects, including base model, architecture, computational resources, training datasets, licensing, code availability, data, and model weights. Furthermore, we analyzed the evolution and relationships among these models through a phylogenetic perspective, identified current research gaps and opportunities, and discussed future directions for the development of language models for Portuguese. 

---
# Alignment Is All You Need: Instruction-Free Training for General Audio-Language Models 

**Authors**: Xuanru Zhou, Yiwen Shao, Jiahong Li, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18132)  

**Abstract**: Multimodal large language models (MLLMs) are typically built through a multi-stage pipeline consisting of cross-modal alignment, supervised fine-tuning (SFT), and preference optimization. This pipeline assumes that adapting an LLM to a new modality requires extensive task-specific supervision. However, pretrained LLMs already possess strong reasoning and instruction-following abilities. As LLMs evolve rapidly, an important question remains: can we efficiently transfer these capabilities to a new modality with minimal intervention, and is alignment alone sufficient for building a multimodal model? We introduce an Instruction-Free Alignment-Only large audio-language model (LALM) that keeps both the audio encoder and the LLM fully frozen, learning only a lightweight projector. Borrowing insights from AzeroS [1], we train on (audio, response) pairs from Self-Generated Data Construction, where an LLM expands captions into free-form responses without explicit task instructions. Across MMAU, MMAR, MMSU, and MMAU-Pro, our approach matches or surpasses heavily post-trained baselines using substantially less data. By keeping the LLM frozen, our model preserves its native instruction-following competence and can port seamlessly across model generations. Our results suggest that competitive MLLM can emerge from alignment alone, reducing multimodal extension to a lightweight projector-training problem that generalizes across modalities and adapts rapidly to each new LLM release. 

---
# You Are What You Prompt: Prompt Quality, Domain Shift, and Uncertainty in Agrifood Vision-Language Models 

**Authors**: Andrea Morales-Garzón, Salvador López-Joya, Miguel López-Pérez, Maria J. Martin-Bautista  

**Link**: [PDF](https://arxiv.org/pdf/2608.18116)  

**Abstract**: Vision-language models enable zero-shot classification through natural language prompts, but performance is sensitive to prompt formulation, especially in specialized domains. Zero-shot Prompt Ensembling (ZPE) addresses this by weighting prompts by discriminative signal, yet its behavior under domain shift remains unexplored. We evaluate ZPE in the agrifood domain using CLIP and SigLIP across four datasets and four prompt pools, spanning in-distribution (ID) food and out-of-distribution agricultural benchmarks. ZPE provides limited benefit under ID conditions but substantially improves performance and calibration under domain shift, where domain-specific pools of 51-52 prompts consistently outperform generic pools of 247-426. Lexical analysis shows that ZPE acts as an unsupervised domain-alignment detector without label access. We further introduce PID (Prompt-based Inconsistency Detection), which repurposes prompt disagreement as epistemic uncertainty, improving failure detection under severe domain shift where standard confidence measures collapse. 

---
# Temporal Multi-Signal Fusion for Token-Level Hallucination Detection 

**Authors**: Igor Itkin  

**Link**: [PDF](https://arxiv.org/pdf/2608.18115)  

**Abstract**: Token-level hallucination detectors score each token independently from a single signal, and fail exactly when the generating model is confidently wrong. This paper instead treats hallucination as a temporally extended span and detects it by sequence labeling: each token is scored from a 33-dimensional feature stream that fuses text statistics, Natural Language Inference (NLI) entailment, and language model surprisal, with no access to model internals. A Bidirectional Gated Recurrent Unit (BiGRU) over these features reaches an AUC of 0.840 on RAGTruth (10 seeds), an 11-point gain over an independent logistic-regression baseline (p = 0.002, Wilcoxon signed-rank). A controlled decomposition attributes most of the gain to temporal order rather than model capacity: evidence propagates from confident positions to ambiguous neighbors within a span. The same 0.845 ceiling recurs across recurrent, state-space (Mamba), and attention architectures, locating the bottleneck in the feature set rather than the model. Because it reads only the generated text and external signals, the detector works on closed-source models, and it keeps working on text produced by language models it never saw during training, losing under 4% AUC. 

---
# Accurate Decoding of Natural Sentences from Non-Invasive Brain Recordings 

**Authors**: Mingfang Zhang, Jarod Lévy, Cedric Rommel, Jérémy Rapin, Corentin Bel, Julie Bonnaire, Daniel Nieto, Pierre Bourdillon, Svetlana Pinet, Stéphane d'Ascoli, Thomas Moreau, Jean-Rémi King  

**Link**: [PDF](https://arxiv.org/pdf/2608.18114)  

**Abstract**: Restoring communication for people who have lost the ability to speak or move after a brain injury is a major challenge. While intracranial implants now enable high-performing brain-computer-interfaces, non-invasive alternatives are still lagging behind. Here, we present Brain2Qwerty v2, a model that can decode the production of natural sentences solely from real-time magnetoencephalography (MEG) recordings. By collecting 22,000 sentences typed by nine subjects, each recorded for 10 hours, our model leverages character, word and sentence-level representations to achieve an average word error rate (WER) of 39%. For our best participant, the model accurately decodes half of the sentences with one word error or less. Critically, decoding accuracy log-linearly improves with data volume, suggesting that the performance gap with intracranial approaches could be partially bridged through data scaling. We show that AI enables this performance in three main ways: the substitution of hand-crafted pipelines for event detection with deep learning, the finetuning of large language models to extract semantic representations, and the deployment of AI agents to iteratively refine our decoding pipeline via automated code development. Together, these results show that non-invasive brain-to-text decoding starts to operate at a level of accuracy previously thought exclusive to surgical implants, opening a path toward safe and efficient brain-computer-interfaces. 

---
# Operationalizing Narrative Entropy (Sn): A Two-Scene Registered Pilot Report and Pre-Validation Protocol 

**Authors**: Levent Bulut  

**Link**: [PDF](https://arxiv.org/pdf/2608.18109)  

**Abstract**: Narrative Entropy ($S_n$) is a proposed quantitative descriptor within the Bulut Doctrine, intended to capture the rate at which a narrative text imposes processing load on a reader. To date the construct has been defined theoretically but not operationalized against real texts. This report documents the first such operationalization (the v2.0 pilot): two narrative scenes -- the opening restaurant scene of Tarantino's Reservoir Dogs and the opening interior-monologue block of Carver's Cathedral -- were coded manually by a single rater and scored with the candidate formula $S_n = I_f \times C_b \times t$. The result was a divergence from the author's naive intuition: the single-voice monologue ($S_n = 30.0$) scored higher than the nine-character dialogue scene ($S_n = 18.8$). We treat this not as a result to be explained away but as the central finding, and we refuse post-hoc adjustment of the formula. Three competing interpretations are presented -- formula incompleteness, genuine high-load prose, and measurement error -- and the design that would discriminate among them is pre-registered. This v2.1 revision adds: (i) explicit acknowledgement that the divergence is consistent with the pre-existing architectural framework which privileges inferential reconstruction over surface declaration, and that what was called "contrary to expectation" in v2.0 reflected the author's anticipatory intuition rather than the methodology's own predictions; (ii) a pre-registered construct validity test for $I_f$, motivated by the observation that $I_f$ values were nearly equal across the two scenes (1.71 vs 1.58) despite the headline $S_n$ divergence. The document functions simultaneously as a pilot report ($n=2$) and as a pre-registration of the next-stage protocol. It does not claim that $S_n$ has been validated. 

---
# Same Facts, Different Updates: Inference Setup Shapes LLM Behavior in Medical Allocation 

**Authors**: Spencer Gibson, Tyler Crosse, Magnus Saebo, Achyutha Menon, Eyon Jang, Diogo Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2608.18108)  

**Abstract**: Large language models are being incorporated into sensitive and important decision-making processes across nearly all fields. While prior work studies model bias around inputs and scenario framing, models can also behave in unexpected and undesirable ways due to context accumulated over their deployment. In this work, we study a medical example in which a model is asked to assign resource-allocation probabilities to two people given brief clinical context, and then sees the same scenario with a single extra sentence containing contrasting patient information, either with or without its previous response in context. Across three of four tested models, the paired-context and independent-inference experiments have different probability shifts, often in opposite directions (in favor of Person B vs. in favor of Person A) when new information is provided. We include additional paired-context experiments to show the effect of varying attributes across scenario axes. Our findings show the context-dependent effect of patient information in a sensitive medical use case. More broadly, our work shows the importance of carefully incorporating LLM-based systems into decision-making processes, context engineering, and further model behavioral studies. 

---
# Institutional Prestige as Geographic Bias in Large Language Models: Evidence from Three Factorial Experiments with Bootstrap Confidence Intervals 

**Authors**: Maikel Leyva-Vazquez, Florentin Smarandache  

**Link**: [PDF](https://arxiv.org/pdf/2608.18107)  

**Abstract**: We investigate whether large language models (LLMs) systematically discriminate in candidate evaluations based on applicant name ethnicity and/or institutional prestige and geographic location. Three factorial experiments are reported (4,320 API calls, four LLMs, five professional domains). Study 1 (3x4 design) finds a statistically robust institution-tier gradient of +0.297 points on a 10-point scale (95% bootstrap CI: +0.175 to +0.422), while name-origin effects are negligible and non-significant (95% CI crosses zero). Study 2 (2x2 Prestige x Country design) breaks the prestige-geography confound: the prestige effect (+0.185; 95% CI: +0.093 to +0.275) exceeds the country-of-origin effect (+0.126; 95% CI: +0.037 to +0.218) by 1.5x. Study 3 (2x2 Journal x Institution design) reveals that journal prestige (Nature vs. a peripheral open-access journal) dominates institutional prestige by 5.7x: journal effect +1.937 (95% CI: +1.811 to +2.062) vs. institution effect +0.341 (95% CI: +0.184 to +0.504). A "rescue effect" is confirmed: publishing in Nature compensates for low institutional prestige more strongly for candidates from the University of Guayaquil (+2.127) than from MIT (+1.745). Results are quantified using the Neutrosophic Bias Index NBI<T,I,F>; the I component reveals elevated evaluation inconsistency for low-prestige profiles, an epistemic disadvantage not captured by mean-only metrics. Code and data: this https URL 

---
# Different Facets of Verbalised Overconfidence: an Interpretability Study 

**Authors**: Davide Mazzaccara, Leonardo Bertolazzi, Raffaella Bernardi  

**Link**: [PDF](https://arxiv.org/pdf/2608.18106)  

**Abstract**: Large language models tend to overconfidence, giving assertive answers when the evidence suggests hedging or abstention. Using controlled reasoning scenarios that manipulate logical necessity and possibility, we study this behavior in Qwen3-4B, across three ways to express uncertainty: verbal epistemic markers, abstention, and numeric confidence scores. Our results confirm this tendency toward overconfidence, particularly when the model is prompted to output a numeric confidence score. At the interpretability level, we propose a method that differentially identifies transcoder features responsible for uncertainty and certainty. Our analysis reveals Qwen3-4B's default mechanism favors certainty generation through a broad coalition of shared features, while uncertainty is implemented as a sparse override mediated by a small set of dedicated features. Intervening on these uncertainty features both causally proves this imbalance underlying overconfidence and also mitigate overconfident errors. The same set of features generalise across the three uncertainty-expression settings, languages, and an out-of-distribution modality task. 

---
# StocksTalk: A Voice-Enabled Conversational Agent for Structured Query Generation over Web Data 

**Authors**: Akshat Parmar, Vikranth Udandarao, Abhay Shakya, Tanmay Hire, Avinash Anand, Rajiv Ratn Shah, Daniel Wang Zhengkui  

**Link**: [PDF](https://arxiv.org/pdf/2608.18105)  

**Abstract**: StocksTalk is a voice-enabled conversational system for transforming spoken financial screening requests into executable and validated structured queries over real-world market data. The system combines streaming speech recognition, retrieval-augmented constraint extraction, schema-grounded LLM-based SQL generation, rule-based validation, and human-in-the-loop verification within an interactive dashboard. Unlike traditional template-driven financial assistants, StocksTalk exposes intermediate reasoning artifacts, including extracted constraints, normalized financial metrics, operator grounding, and generated queries, allowing users to inspect and refine each stage before execution. To evaluate the system, we curate a benchmark of 150 spoken financial prompts spanning multiple investment strategies and input noise conditions. Experimental results show that retrieval grounding, constrained query generation, and interactive verification substantially improve constraint extraction accuracy, SQL executability, logical consistency, and multi-turn stability compared to baseline LLM-based approaches. StocksTalk demonstrates how transparent, voice-driven interfaces can bridge natural language interaction and structured financial analysis, providing an effective framework for conversational stock screening and decision support. 

---
# DeepTCM1.0: A Multi-Expert AI Agent for Deciphering Mechanisms of Chinese Herbal Formulae Based on General Large Language Models 

**Authors**: Wenxin Duan, Hanwei Wang, Zhongying Peng, Zhonghua Lu, Jiayi An, Fan Song, Yong Liang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18103)  

**Abstract**: Background: Mechanistic elucidation of traditional Chinese medicine (TCM) compound formulas remains a central challenge in the modernization of TCM. Conventional approaches, including data mining and network pharmacology, are insufficient for achieving deep integration between classical TCM theory and modern scientific research. In addition, direct question-answering using general-purpose artificial intelligence large language models is limited by inadequate adaptation to TCM theoretical frameworks and susceptibility to reasoning hallucinations. Consequently, there is an urgent need to develop intelligent analytical methods aligned with the holistic principles of TCM. Objective: To establish a multi-expert intelligent agent framework integrating classical TCM theory with modern life sciences, thereby enabling systematic and interpretable mechanistic analysis of TCM compound formulas, with Guizhi Decoction serving as a representative validation case. Methods: The DeepTCM1.0 framework was constructed based on the general-purpose large language model DeepSeek V3.2. It adopts a three-tier collaborative architecture and a three-round iterative quality-control workflow, simulating the collaborative analytical process of 11 interdisciplinary intelligent agents. The framework was applied to the mechanistic interpretation of Guizhi Decoction from the dual perspectives of classical traditional Chinese medicine theory and modern scientific research. Framework performance was comprehensively evaluated through double-blind five-dimensional scoring, intraclass correlation coefficient (ICC) reliability testing, Mann-Whitney U tests, and effect size analysis. The evaluation employed four independent large language models as evaluators, each conducting five rounds of repeated scoring on five anonymized reports, resulting in a total of 100 independent scoring assessments. 

---
# Stability-Aware Feature Design for Robust Watermark Detection in Machine-Generated Text 

**Authors**: Sina Mansouri, Mohit Marvania, Abolfazl Safikhani  

**Link**: [PDF](https://arxiv.org/pdf/2608.18102)  

**Abstract**: The widespread adoption of large language models (LLMs) has intensified the demand for principled methods to distinguish human from machine-generated text. Watermarking provides a promising avenue, yet existing detectors exhibit sharp performance deterioration under multiple paraphrasing and when applied to shorter texts. We introduce Pattern Stability Score (PSS), a novel detection framework that leverages local statistical features and stability dynamics across paraphrased variants. Specifically, the proposed method combines global and local z-score features with higher-order statistics of run-length patterns, enriched by autocorrelation signals and stability scores computed over paraphrase depth. Numerical evaluations are performed on three benchmark datasets (PG-19, CNN/DailyMail, and WikiText) using multiple LLMs (Llama-3-8B, Qwen2-7B) and paraphrasers (Mistral-7B, Qwen2-7B, Gemma-7B), systematically stress-testing robustness under up to eight rounds of paraphrasing. Compared to prior z-score thresholding baselines and some state-of-the-art deep learning methods, our approach improves detection AUC (area under the receiver operating characteristic curve) by over 10-15 percentage points across different token lengths. Additionally, extensive cross-domain experiments demonstrate that a single universal classifier generalizes across different LLMs, paraphrasers, and text domains without retraining, maintaining above 87.8% AUC even when all components differ from training. 

---
# BERTilda: Explainable Topic Lifecycle Tracking with Split/Merge Detection via Similarity-and-Flow Temporal Graphs 

**Authors**: Cláudia Oliveira, Álvaro Figueira  

**Link**: [PDF](https://arxiv.org/pdf/2608.18101)  

**Abstract**: Longitudinal text streams exhibit topic birth and death, but also discrete structural reorganizations in which themes split into subtopics or merge into broader narratives. Many dynamic topic models emphasize smooth drift, while snapshot topic models (fit independently per time window) leave temporal correspondence underspecified. We present BERTilda, an explainable framework that discovers topics independently in each window (using an embedding-based topic model) and then constructs a temporal topic graph linking topics across adjacent windows. Links are supported by two complementary signals: (i) semantic similarity between topic representations and (ii) a bidirectional coverage signal that estimates document outflow (where a topic goes) and inflow (where a topic comes from) via cross-window tweet-to-topic attribution. Graph-based rules label continuations, splits, merges, disappearances, and unclear transitions. We evaluate BERTilda on political corpora, including U.S. congressional tweets and historical speech datasets, report topic-quality and temporal-stability diagnostics, and validate lifecycle labels on a gold-standard subset annotated by three independent annotators. On the annotated subset, BERTilda reaches majority agreement rates up to 87% and attains the highest macro-average agreement across the compared methods, with particularly strong disappearance detection relative to similarity-only and forward-only baselines. 

---
# Computational Orientalism: Measuring Structural Discourse Bias in Large Language Models Using the Middle East Cultural Sensitivity Score (MECSS) 

**Authors**: Maha Shahid  

**Link**: [PDF](https://arxiv.org/pdf/2608.18100)  

**Abstract**: AI systems now shape how hundreds of millions of people learn about cultures other than their own. When someone asks one of these systems about the Middle East, they do not receive neutral facts. They receive a representation shaped by the frameworks embedded in training data, and that data is overwhelmingly Western and English-language. This paper asks whether that representation is Orientalist in Said's sense: whether it denies agency to Middle Eastern actors, treats Western frameworks as neutral while marking non-Western knowledge as particular, and explains the region through categories it did not produce. Standard fairness metrics cannot answer this, because they detect explicit prejudice rather than structural framing. This paper introduces the Middle East Cultural Sensitivity Score (MECSS), a framework that turns Said's seven Orientalist operations into measurable dimensions, and the term "Said-washing" for a specific failure: a model that disclaims generalization, then reproduces the structure it disclaimed. Across 280 conversations (1,120 exchanges), GPT-4 and Falcon3-7B-Instruct both reproduce Orientalist patterns systematically, through structural positioning rather than open stereotyping. GPT-4 scores moderately (mean MECSS 1.73); Falcon3-7B-Instruct scores higher (2.18), even though it was built in Abu Dhabi and trained with Arabic content. This is evidence against the assumption that building a model regionally makes it less Orientalist, though the models differ in size as well as origin, so geography cannot be isolated as the cause. Epistemic Center, the treatment of Western frameworks as unmarked universals, scores near the top of the scale for both models. Said-washing appears in 87.9% of GPT-4 conversations, a pattern existing metrics cannot see. Reducing this bias requires changing what models learn from, not only adding languages or relocating institutions. 

---
# Fractional Decay KV-Cache: Ownership-Aware Memory Management for Improved Inference Relevancy in Dialog Systems 

**Authors**: Sukanta Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2608.18098)  

**Abstract**: Key-value (KV) caching is essential for efficient autoregressive inference in transformer based dialog systems, yet existing strategies treat all cached entries uniformly or apply coarse eviction heuristics that fail to adapt as dialog topics evolve. We propose Fractional Decay KV-Cache (FD-KVC), a novel algorithm that maintains a dual-channel scoring mechanism for each cached KV pair: a cumulative attention channel that tracks aggregate importance (akin to H2O), and a recency-weighted relevance channel governed by temporal decay and reinforcement-inspired updates. The combination enables FD-KVC to both preserve historically important tokens and rapidly adapt when dialog topics shift. An adaptive learning rate driven by an ownership loss function ensures convergence without oscillation. FD-KVC operates entirely on CPU with negligible overhead. Across five diverse multi-turn dialog scenarios with 600 dialogs each, FD-KVC outperforms H2O, the state-of-the-art heavy-hitter baseline, by +6.7% on composite late-turn alignment, with improvements of +127% on topic-shift, +87% on gradual evolution, and +30% on mixed-topic dialogs. FD-KVC adapts to new topics 3.6X faster than H2O and achieves the highest topic diversity (80.6%) across all methods. Ablation studies confirm the contribution of each component. 

---
# FrenchNews-7: Benchmarking Cross-Publisher French News Editorial Desk Classification 

**Authors**: Amr Sobhy  

**Link**: [PDF](https://arxiv.org/pdf/2608.18097)  

**Abstract**: We present FrenchNews-7, a cross-publisher France-based French-language news editorial desk classification benchmark combining a large multi-outlet corpus, a URL-derived seven-class taxonomy, and a fine-tuned CamemBERT classifier. Labels are assigned via a hybrid pipeline combining publisher URL slugs with LLM annotation for structurally ambiguous cases, audited through an inter-rater study (2 humans + 2 LLMs; pairwise $\kappa \geq 0.766$, human--human $\kappa = 0.806$). We evaluate lexical, multilingual, and French-specific trained classifiers under both in-distribution and held-out-publisher settings, with additional comparison against zero-shot LLM baselines (GPT-OSS-120B, Mistral Small 3.2, Llama-3.3-70B) on the held-out pool. The strongest model, CamemBERT-base on full article text, outperforms headline-only input, generalizes to unseen outlets, and exceeds all three zero-shot LLM baselines on overall recall (0.799), with the gap concentrated in the ambiguous editorial-boundary categories Economie and Societe. Cross-publisher evaluation reveals uneven boundary stability: Sport, Culture & Loisirs, and International transfer cleanly, while Economie (recall = 0.517) is close to blinded human agreement (0.55), and Societe (precision = 0.577) absorbs boundary ambiguity, both suggesting editorial conventions rather than recoverable classifier headroom. The fine-tuned CamemBERT-base model, labeled manifest, reference collection scripts, and a reliability-tier guidance table are available at this https URL (model) and this https URL (dataset). 

---
# MAVEN: A Macro-Societal Value Evaluation Framework of Multimodal Content with Compact Aligned Evaluators 

**Authors**: Zijuan Zhao, Zheren Fu, Hou Xia, Licheng Zhang, Yi Liu, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2608.18096)  

**Abstract**: Assessing whether multimodal content aligns with macro-societal values, such as peace, justice, and freedom, has become an increasingly urgent challenge. Existing frameworks are largely confined to safety-oriented taxonomies, text-only psychometric probes, or single-label classification. Therefore, we propose MAVEN, a hierarchical framework for macro-societal value evaluation of multimodal content, grounded in international human-rights instruments and cultural value theory. MAVEN organizes values into 6 primary dimensions and 72 secondary indicators, supporting multi-level quantitative scoring. Building on MAVEN, we construct a human-verified multimodal benchmark and a soft-match metric to evaluate VLMs' assessments across value dimensions. For evaluator optimization, we propose a span-adaptive variant of multi-level preference optimization for evaluator distillation, together with a training-free multi-role consensus strategy at inference time. We evaluate existing open- and closed-source VLMs on our benchmark, revealing shared tendencies and clear differences in macro-societal value judgments. Experiments show that our compact 2B evaluator matches its 8B counterpart in the same family and approaches frontier closed-source VLMs, offering a practical path toward scalable macro-societal value evaluation. Our SA-MDPO implementation and MacroValue-Bench are available at this https URL. 

---
# Backdoor Learning in Language Models and Vision-Language Models 

**Authors**: Weimin Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2608.18095)  

**Abstract**: Recent advances in deep learning have significantly enhanced the capabilities of Natural Language Processing (NLP) and Vision-Language Models (VLMs). However, these advancements come with increased vulnerabilities, notably through backdoor attacks that pose severe security threats. This thesis addresses two critical dimensions of Trustworthy AI and Efficient Multimodal Representation Learning: (1) security through analyzing, detecting, and designing backdoor attacks in NLP and VLMs, and (2) efficiency through advanced multimodal representation methods tailored for clinical and medical imaging applications. 

---
# NE-BERT: A Multilingual Language Model for Nine Northeast Indian Languages 

**Authors**: Badal Nyalang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18094)  

**Abstract**: Large pretrained language models have demonstrated remarkable capabilities across diverse languages, yet critically underrepresented low-resource languages remain marginalized. We present NE-BERT, a domain-specific multilingual encoder model trained on approximately 8.3 million sentences spanning 9 Northeast Indian languages and 2 anchor languages (Hindi, English), a linguistically diverse region with minimal representation in existing multilingual models. By employing weighted data sampling and a custom SentencePiece Unigram tokenizer, NE-BERT outperforms IndicBERT-V2 and MuRIL across all 9 Northeast Indian languages, achieving 15.97X and 7.64X lower average perplexity respectively, with 1.50X better tokenization fertility than mBERT. We address critical vocabulary fragmentation issues in extremely low-resource languages such as Pnar (1,002 sentences) and Kokborok (2,463 sentences) through aggressive upsampling strategies. Downstream evaluation on part-of-speech tagging validates practical utility on three Northeast Indian languages. We release NE-BERT, test sets, and training corpus under CC-BY-4.0 to support NLP research and digital inclusion for Northeast Indian communities. 

---
# Abliteration Mitigation via Refusal Aliases 

**Authors**: Nathan Truong  

**Link**: [PDF](https://arxiv.org/pdf/2608.18093)  

**Abstract**: Abliteration, the removal of refusal capabilities from large language models by projecting weight matrices orthogonal to an extracted refusal direction, has emerged as a prominent safety concern through its ability to bypass post-training alignment using only a small set of contrastive prompts. We find that existing defenses commonly overlook the cause of abliteration; that is, how easily the refusal direction can be extracted. To hinder this process, we introduce a weight-editing method that obscures the refusal signal by applying rank-$k$ updates to residual stream writer matrices while replacing refusal-inducing activations with random aliases and correcting downstream reader matrices to preserve the model's original behavior. On Llama-3-8B, AMRA improves post-abliteration refusal scores by $2.16$ points over the undefended baseline with less than $0.5$ percentage points of MMLU degradation. On Gemma-2-9B, it improves the post-abliteration refusal by $14.70$ points over the baseline while keeping harmful output rates similar to the baseline, albeit at a greater utility cost. 

---
# Self- and Other-Labels Induce Bidirectional Bias in LLM Judges 

**Authors**: Songeun Chae, Min Kim, Donghoon Jung, Seojin Choi, Seohyon Jung  

**Link**: [PDF](https://arxiv.org/pdf/2608.18091)  

**Abstract**: As LLM-as-a-judge systems become increasingly widespread, self-preference in LLMs -- the tendency to favor one's own outputs -- raises growing concerns about evaluation reliability. However, it has been studied predominantly on generated text, where stylistic features and response quality are inevitably conflated. As a result, existing measurements cannot separate genuine self-preference from these confounds. We address this by changing the object of evaluation: instead of judging generated text, ten LLMs assess narrative constraint selections, which carry no model-specific stylistic fingerprint yet retain a recoverable model-specific signature. We run two experiments that yield distinct findings. Under blind evaluation, self-preference largely disappears once selection quality and evaluator severity are controlled. It vanishes on three of four rubric dimensions and reverses on the fourth, where judges rate their own selections as less original. Under matched quality, however, self- and other-labels alone -- without naming any model -- shift scores bidirectionally: LLM judges inflate scores for self-labeled selections and deflate those for other-labeled ones regardless of the selection's actual source. We make two contributions: 1) authorship attribution is a distinct driver of evaluation bias, and 2) open-ended, ground-truth-free tasks can serve as controlled instruments for studying LLM judge behavior. 

---
# Nine Emotion Centroids: A Label-Free Valence Axis That Transfers Across Four Modalities 

**Authors**: Yousef Radwan  

**Link**: [PDF](https://arxiv.org/pdf/2608.18090)  

**Abstract**: Inside a modern language model sits a single internal direction that tracks how positive or negative a sentence feels. We show how to find this valence axis (V-axis) from just 9 emotion category names plus 50 short narrative paragraphs per emotion -- about 1,500 fewer labels than the usual supervised approach -- and that the same direction appears in vision, audio, and human-brain encoders never jointly trained. The recipe: embed nine emotion-anchored story sets in a frozen encoder, take the top principal direction of the nine averaged embeddings. Projecting new inputs onto it captures 93% of supervised performance on SST-2 (Llama-3-8B-Instruct, AUC 0.772 vs. 0.828), correlates with human valence ratings on 11,811 EmoSet images at r=0.636, reaches AUC 0.906 on ESC-50 audio (p<2.2e-15), and AUC 0.720+/-0.055 on EEG from 123 subjects (p<3.65e-8). The direction is mechanistically active: ablating it collapses sentiment accuracy by 5.5-37.2 pp across three LLMs vs. at most 0.88 pp for matched random directions (z>12). A 2-parameter classifier trained on text labels transfers to images (AUC 0.961), audio (0.764), and brain recordings (0.828) without target-modality labels; a generic 16-D subspace stays at chance (0.525). The recipe is bounded to continuous attributes -- seven tests on categorical concepts return near-chance -- and steering is family-specific (Llama/Mistral yes, Qwen/Gemma no). 

---
# Latent Space Refusal Anchoring for Low-Resource African Languages: Mechanistic Safety Recovery Without Retraining 

**Authors**: Godwin Abuh Faruna  

**Link**: [PDF](https://arxiv.org/pdf/2608.18089)  

**Abstract**: Instruction-tuned models often refuse harmful requests in English but comply with the same requests in Yoruba, Igbo, Igala, and Hausa. This suggests that the refusal mechanism is present in the residual stream but fails to activate for low-resource inputs. Recovering it normally requires labelled target-language data and retraining, neither of which is available at scale for most African languages. We introduce Latent Space Refusal Anchoring (LSR-Anchoring), a training-free method that extracts the refusal direction from English prompts and clamps it onto the residual stream at inference time. The primary variant, Mean-Activation Steering (MAS), operates across the four architectures we tested: Llama-3-8B, Llama-3.1-70B, Mistral-7B-Instruct, and Qwen2.5-7B. On Mistral and Qwen it recovers safety with benign degradation below 0.08. On Llama-3-8B it overcorrects, with Degraded Performance on Legitimate prompts (DPL) reaching 1.00. We address this with SAE-Derived Steering (SDS), which replaces the dense mean-difference direction with a single Sparse Autoencoder (SAE) feature and reduces Kullback-Leibler (KL) divergence by 3.5-7x without benign collapse. Four languages transfer positively, but Arabic fails on every architecture and at every steering magnitude, indicating a geometric mismatch rather than a baseline effect. Massive Multitask Language Understanding (MMLU) accuracy drops remain below 0.35 percentage points at every effective steering magnitude. 

---
# SuTRA : Structurally-Unified Tokenization with Root Awareness 

**Authors**: Vaibhav Rathore, Siddhant Gole, Dadhichi Telwadkar, Rooshil Bhatia, Maulik Ruparel, Siddharth Surekha, Neha Bhargava  

**Link**: [PDF](https://arxiv.org/pdf/2608.18087)  

**Abstract**: Existing subword tokenizers optimize statistical compression but ignore morphological structure, particularly the relationship between roots and affixes. This is harmful for morphologically rich Indic languages, where basic units are complex orthographic syllables (aksharas) rather than letters. Frequency-based methods over-fragment words, arbitrarily splitting roots and affixes - a phenomenon we term Morphological Shattering. We propose SuTRA (Structurally-Unified Tokenization with Root Awareness), a morphology-aware algorithm that preserves akshara indivisibility and penalizes merges crossing morphological boundaries. We also release a new morphological segmentation dataset for Hindi, Marathi, and Gujarati. SuTRA reduces shattering, achieving peak gains of +14.7% in morphological alignment (Boundary F1) and +34% in semantic recoverability (Hindi) over BPE. These structural gains yield an average improvement of +8.08 chrF2 in machine translation. 

---
# Persona-Guided LLM Agents for Task-Oriented Dialogue 

**Authors**: Maryam Shoaeinaeini, Brent Harrison, A.B. Siddique  

**Link**: [PDF](https://arxiv.org/pdf/2608.18085)  

**Abstract**: Prior work has shown that large language models (LLMs) can express diverse personality traits in open-ended text generation. However, it remains unclear whether they can do so in a goal-directed dialogue without compromising task completion, and whether adapting to the user's personality improves the interaction quality. We study these questions in task-oriented dialogue (TOD), where a system helps a user accomplish a goal via multi-turn interaction. We build a training-free framework that simulates a TOD interaction between two LLMs: a user agent that exhibits a target personality and a system agent that adapts to the user while completing the task. To isolate the effect of adaptation, we vary how much the system knows about the user's personality across three conditions. In Neutral, the system receives no personality information. In Try, it infers the personality from dialogue cues. In Oracle, it is given the personality explicitly. We evaluate GPT-4o, Qwen3-Next-80B, and Gemini 2.0 Flash on Hotel and Restaurant dialogues from the Schema-Guided Dialogue (SGD) dataset, across the Big Five traits and their opposite poles. We find that the user agent can express personality while the system maintains strong task performance, although some traits are realized far less reliably than others. Adapting to the user's personality improves constraint satisfaction, inform rate, and user satisfaction, but lowers truthfulness, revealing a trade-off between personalization and task-grounding. Oracle's gains grow when the target trait is strongly expressed, whereas Try's gains are largely insensitive to realization strength. Overall, cue-based adaptation in Try best resolves this trade-off and offers a more reliable route to personality-aware TOD without fine-tuning. 

---
# Compiler-Guided Adaptive Proof Search with Cross-Model Synergy on Context-Dependent Theorem Proving 

**Authors**: Zhuo Liu, Ding Yu, Hangfeng He  

**Link**: [PDF](https://arxiv.org/pdf/2608.18084)  

**Abstract**: Theorem proving in real-world Lean 4 projects is challenging because proofs often depend on project-specific context. While iterative refinement can use compiler errors to repair failed proofs, reusing failed attempts requires careful search control: some proofs provide better starting points than others, and later revisions may degrade a partially correct proof. We propose a compiler-guided proof search framework that balances exploration and exploitation. It explores diverse starting points through dual-model generation and stagnation-triggered resampling, while exploiting promising proof states through current-best refinement guided by compiler-grounded pairwise comparison. Experiments on seven real-world Lean 4 projects from miniCTX-v2 show that our method achieves a better effectiveness--efficiency tradeoff than pass@k baselines. Within the pass@32 budget, our method improves average pass rate by 12.8 percentage points while reducing LLM calls by 21.9%. 

---
# Entity tracking emerges in sub-billion parameter language models and exceeds human performance in naturalistic narratives 

**Authors**: Karolina Drożdż, Micha Heilbron  

**Link**: [PDF](https://arxiv.org/pdf/2608.18083)  

**Abstract**: Understanding language requires tracking entities across discourse - i.e., knowing where things are and how they change, even when not explicitly stated. Whether language models perform such tracking in a human-like fashion remains unclear, in part because existing evaluations rely on artificial tasks, far removed from natural language comprehension, and lack comparisons to humans. Here, we evaluate entity tracking in both language models and humans (N = 48) using naturalistic narratives at multiple levels of complexity. In humans, we find that entity tracking degrades specifically with narrative complexity, not narrative length. In language models, we find that human-level entity tracking is already present at 410 million parameters - well below the multi-billion parameter, code-specialised models identified by prior work - and improves with scale, with contemporary models far exceeding human performance. Together, these results demonstrate that entity tracking, a core component of language understanding, emerges at model scales far smaller than previously thought. 

---
# LongNovel: A Multi-Scale Benchmark for Hallucination Detection in Long-Context Novel Summarization 

**Authors**: Ruizhi Zhang, Jinwei Chen, Xiangju Lu, He Yan, Mo Yu, Junmin Zhu, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18082)  

**Abstract**: Although context windows have expanded significantly in recent years, hallucinations in long-context summarization remain a challenge. Long novels are better suited than news or papers for researching these hallucinations, due to their intrinsic information and detailed descriptions of events and dialogues. However, current research lacks a multi-scale benchmark for hallucination detection in long-context novel summarization and does not fully explore how hallucinations change as the context grows longer. In this study, we propose LongNovel, a multi-scale long-context bilingual (Chinese and English) novel benchmark for hallucination detection. This benchmark is constructed from 29 Chinese novels (ranging from 16k to 100k tokens) and chapter-level data from the BookSum dataset. We design 8 hallucination types and employ a combination of Multi-Model Arbitration and Entity-Referenced Hallucination Generation to ensure both data authenticity and a balanced distribution of hallucination categories. Furthermore, we manually revise the content in the test set to guarantee data reliability. Extensive experimental results demonstrate that LongNovel is a challenging benchmark. We release LongNovel for future research. this https URL 

---
# Beyond Teacher Likelihood: Group-Calibrated On-Policy Distillation for Long-Context Reasoning 

**Authors**: Zhu Zhang, Jixun Wang, Xiaoang Xu, Xiaorong Wang, Zihan Zhou, Zhiyuan Wang, Shuo Wang, Chaojun Xiao, Yuezhi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.19181)  

**Abstract**: On-policy distillation (OPD) trains a student on its own responses using dense token-level guidance from a stronger teacher. In long-context tasks, however, token-level teacher support can favor locally plausible responses that omit evidence distributed across the input or violate global task constraints. Task-specific verifiers, in contrast, evaluate task completion at the response level and may return graded rewards that reflect partial success. We diagnose this mismatch on fixed responses from two representative long-context evidence-aggregation tasks. Across longer input ranges, trajectory-level OPD scores become progressively less aligned with verifier rewards, indicating teacher-verifier disagreement. Motivated by this observation, we introduce Group-Calibrated On-Policy Distillation (GC-OPD). GC-OPD separately normalizes verifier rewards and trajectory-level OPD scores within each rollout group and uses their difference as a signed teacher-verifier disagreement residual. Relative-advantage-based credit assignment (RACA) distributes this trajectory-level residual across tokens according to their relative OPD advantages while preserving the original OPD signal. Across five long-context benchmarks, post-training with GC-OPD raises the five-benchmark averages of the official Qwen3-4B and Qwen3-8B checkpoints from 29.08 to 40.47 and from 35.12 to 44.65, respectively. Vanilla OPD reaches 39.31 and 43.56 under the same setup. Controlled ablations show that the signed residual is more effective than either an additional OPD-derived term or direct group-normalized verifier reward addition, while RACA further improves over uniform token allocation. Together, these results demonstrate that group-relative residual calibration can incorporate verifier outcomes without discarding dense token-level guidance. Code is available at this https URL. 

---
# Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation 

**Authors**: Huan-ang Gao, Haohan Chi, Yong Yan, Shiyuan Feng, Hanlin Wu, Zheng Jiang, Bingxiang He, Wei-Ying Ma, Ya-Qin Zhang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.19098)  

**Abstract**: Multi-teacher on-policy distillation (M-OPD) has emerged as a promising paradigm for consolidating domain-specialized reinforcement learning (RL) experts into a single generalist student via dense, token-level reward supervision. Despite its practical success, the optimization dynamics governing multi-teacher capability integration remain poorly understood, and open, rigorously reproducible recipes are conspicuously lacking. In this work, we establish a controlled M-OPD benchmark on SmolLM3-3B-Base with oracle routing, isolating capability integration from routing ambiguity. Our investigation reveals a pronounced capability integration gap: standard M-OPD captures only 35.6% of the available headroom relative to a domain-routed oracle ensemble, with concise tasks such as instruction following suffering severe degradation and premature stagnation. Crucially, we show that this failure stems not from gradient conflict, but from a severe misallocation of the token-level optimization budget. This pathology is driven by three orthogonal factors: structural sequence-length disparities across domains, dynamic convergence drift due to non-uniform learning rates, and multi-step reward staleness from asynchronous policy updates. To resolve these imbalances, we introduce Open-MOPD, a principled framework incorporating token-share balancing, gap-aware dynamic budget allocation, and student reward refresh. Together, these mechanisms systematically restore cross-domain balance, elevating headroom recovery from 35.6% to 83.4% in a single deployable student. We fully open-source our end-to-end post-training recipe, training trajectories, and evaluation suites on an academically accessible hardware budget. 

---
# When Readability and Source Retention Diverge: An Evaluability Gap in AI Translation 

**Authors**: Chenchen Mao, Hanjing Shi, Haiyan Jia, Emily Wegrzyn, Dominic DiFranzo  

**Link**: [PDF](https://arxiv.org/pdf/2608.19083)  

**Abstract**: Readable AI output can leave an evaluability gap: even when the source is shown, an overall-quality judgment may not reflect what an output preserves. We investigated how source-text condition and output rendering relate to perceived translation quality, and how output and system appraisals relate to trust and stated disclosure willingness in a plain-text interface. A focal 2 * 2 comparison (N=306) using TransLingo examined simple generated narratives and complex literary-philosophical prose alongside LLM-generated readability-oriented outputs and researcher-revised fidelity-oriented outputs. A descriptive stimulus audit indicated greater source retention in fidelity-oriented outputs in both source-text conditions. Factorial analyses showed a significant rendering-by-source-text-condition interaction in perceived quality. Participants rated fidelity-oriented outputs higher than readability-oriented outputs for the simple narratives, whereas no reliable rendering difference emerged for the complex prose. A corresponding source-condition-dependent pattern was observed for perceived intelligence, agency-oriented anthropomorphic attribution, and task-performance trust. A separate theory-ordered appraisal-structure SEM characterized concurrent associations among perceived quality, perceived intelligence, agency-oriented anthropomorphic attribution, task-performance trust, and stated disclosure willingness across six domains, with task-performance trust as the proximal correlate of stated willingness. The observed rating pattern distinguishes source access from source evaluability: for the complex stimuli, displaying the source did not ensure that one overall-quality rating reflected differences in retained content. It also separates support for evaluating translation output from data-handling support for decisions about what personal text to entrust to a system. 

---
# ReWEIGH the Evidence: Calibrating Token-Level Ordinal Visual Evidence to Mitigate Hallucinations in Large Vision-Language Models 

**Authors**: Jihae Jeong, Junha Choi, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.19075)  

**Abstract**: Large vision-language models (LVLMs) often hallucinate, generating content that the input image does not support. Preventing such content during decoding calls for a candidate-specific measure of how strongly the image supports the token under consideration. The model's visual-token states offer a natural source of this evidence because projecting each state through the output head reveals which vocabulary items that position favors. These position-wise readouts cannot be pooled directly because their probability magnitudes are not comparable across visual positions. Vocabulary ranks provide a scale-invariant basis for pooling, but tokens still differ systematically in their typical rank-based evidence. We propose ReWEIGH, a training-free decoding intervention that aggregates these ranks across visual positions and compares each candidate with a token-specific reference estimated from unlabeled images. At inference, ReWEIGH caches the image evidence during prefill and applies a bounded penalty only to candidates that fall below their reference. On four 7B backbones, ReWEIGH reduces hallucinated object mentions by up to 21.3% while largely preserving or improving descriptive and general performance. With evidence cached, the average added latency is 1.33% per token, and the reductions extend across six architecture families to 32B parameters. 

---
# What is Missing from AI Post-Training AI: An Empirical Analysis 

**Authors**: Joy Jia Yin Lim, Xin Huang, Hao Peng, Yaxi Lu, Xin Cong, Zhong Zhang, Maosong Sun, Yankai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2608.19072)  

**Abstract**: Large language model (LLM) agents can now post-train an LLM end-to-end. They can write code, launch training, evaluate checkpoints, and improve downstream performance, raising the prospect of AI-for-AI. We argue that this picture conflates two distinct capabilities: execution-level capability, iterating within a selected training strategy; and strategy-level capability, revising the high-level judgment as experimental evidence accumulates. Analyzing a large corpus of publicly released post-training trajectories, we find that across different tasks, the agent's training strategy is locked in at the very beginning, and the entire remaining budget is spent on local adjustments within the selected strategy. We then examine three natural explanations--missing experience, missing guidance, and insufficient reasoning--with escalating interventions. Extensive experiments show that (1) an experience-driven scaffold improves execution across the board (+12.6 points on GSM8K and +40.8 on HumanEval) but leaves the strategy static; (2) human guidance effectively redirects the initial strategy, yet the agent falls back into local adjustment loops once training starts; and (3) additional inference compute pays off on easier tasks but yields almost no gain on the hardest one. In conclusion, what agents lack is neither experience, guidance, nor reasoning compute, but a mechanism for spontaneously reevaluating their strategy during execution. 

---
# Adaptive Memory and Reflection Multi-Agent System for Medical Question Answering 

**Authors**: Pradeep Murugesan, Luoxiao Yang, Xueli Chen, Xinqi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2608.19029)  

**Abstract**: Accurate and responsible medical question answering (QA) is important in healthcare, where complex cases require factual knowledge and nuanced reasoning. Existing medical QA systems, typically based on single-agent architectures and static retrieval, often lack adaptability, persistent memory, and structured decision-making. This work introduces an adaptive memory and reflection (AMR) agentic system, a multi-agent framework in which specialized agents use dedicated memory and reflection-based feedback to retrieve relevant prior cases and improve subsequent reasoning. Complexity assessment routes questions through solo, collaborative, or escalated workflows, while consensus and ethical overseer modules support reasoning consolidation and output review. Evaluation on MedQA and MedMCQA demonstrates strong performance compared with several baselines. Ablation studies show that combining agent-specific memory, reflection, and external retrieval yields the strongest performance. These findings highlight the potential of structured memory and feedback for developing more trustworthy medical agents. The source code is publicly available at this https URL. 

---
# rEDMRec: Distilling Large Language Model Reasoning into an Editable Experience Memory for Recommendation 

**Authors**: Minh Hoang Nguyen, Tung Le, Huy Tien Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2608.18952)  

**Abstract**: Large language models can improve recommendation quality by reasoning explicitly over user history and candidate items - for example, extracting a user's preferences or explaining why one item fits better than another - rather than mapping history directly to a ranked list. This reasoning, however, is expensive to repeat on every ranking request and, once produced, is typically consumed once and discarded, leaving it neither reusable across future requests nor easy to inspect or correct as user tastes drift. Our insight is that reasoning does not need to be regenerated at every call if it can instead be compressed once into a compact, structured memory that a lightweight model retrieves from. We propose rEDMRec, which distills a teacher LLM's reasoning into four typed, editable experience channels - long-term preference, short-term context, item-perception, and counterfactual hard-negative comparisons - maintained by an LLM memory controller that performs Add/Delete/Modify/Keep operations and refines entries via K-agent debate. A lightweight student LLM then ranks candidates purely by retrieving from this memory, without invoking the teacher again, decoupling online inference cost from reasoning depth. Across ML-1M, Amazon Beauty, and Steam and ten student backbones, rEDMRec improves HR@1 over zero-shot, few-shot, and RAG on every backbone, and over GraphRAG on most backbones, with Impv up to 13.3% vs. the second-best baseline on ML-1M. Channel ablations show that short-term context is the only channel that helps consistently across capacity tiers, whereas long-term, item-perception, and counterfactual contributions are capacity-dependent (and can reverse on the strongest students); debate-based memory optimization lowers bank duplication by 7.4 percentage points while raising downstream HR@1 by up to +0.029 over six optimization epochs. 

---
# Training Chemical Plausibility-Aware Large Language Models for Single-Step Retrosynthesis 

**Authors**: Bogdan Zagribelnyy, Ivan Ilin, Nikita Bondarev, Maksim Kuznetsov, Mathieu Reymond, Vladimir Aladinskiy, Alex Aliper, Alex Zhavoronkov  

**Link**: [PDF](https://arxiv.org/pdf/2608.18940)  

**Abstract**: Single-step retrosynthesis is a central component of computer-aided synthesis planning, yet its intrinsically one-to-many nature is poorly captured by single-answer evaluation and benchmarking protocols. To address this, we introduce Top-K prompting as a robust training and inference paradigm to better capture diverse, plausible reaction predictions. We compile CREED-CCV-2+USPTO-XL, an ultra-large-scale dataset of ~45.6 million verified reactions to train the C3LM (Chemistry Constraint-Consistent Language Model). By integrating fine-tuning with ChemCensor-based and novelty-oriented rewards, our model achieves state-of-the-art performance on the OOD URSA-expert-2026 benchmark. Further analysis of reaction uniqueness shows that LLMs and conventional models explore complementary reaction spaces, motivating ensemble-based retrosynthesis systems. Overall, our results establish Top-K, plausibility-aware training as a practical new direction for robust future LLM-based synthesis planning. 

---
# MLREF: Efficient Module Reuse for Reward Design in Reinforcement Learning via Large Language Models 

**Authors**: Chenglin Liu, Xun Wang, Ruishuo Chen, Zhuoran Li, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.18827)  

**Abstract**: Reward function design remains a bottleneck in reinforcement learning. While large language models (LLMs) have enabled automated reward generation, existing methods generate and revise reward functions as monolithic programs, making it difficult to reliably preserve and reuse effective components discovered in earlier iterations, leading to unstable performance across iterations. To address this, we propose Module Level Reward Evolution Framework (MLREF). At the core of MLREF is a module pool, a persistent repository of reusable reward components. MLREF treats the module pool as the primary optimization object: the pool evolves across iterations by accumulating successful modules, refining underperforming ones, and reusing proven components; while reward functions are constructed as linear combinations of modules drawn from this pool. To drive this evolution, MLREF integrates three mechanisms: reflection-based refinement, hybrid credit assignment, and a merge strategy with rollback, which together improve the effectiveness and robustness of reward optimization. Experiments on 17 tasks show that MLREF outperforms strong baselines by 25.2% in locomotion and 6.6% in manipulation, with more stable optimization dynamics. 

---
# GreekBarRetrieval: A Benchmark for Greek Statutory Retrieval 

**Authors**: Ernest Beta, Odysseas S. Chlapanis, Dimitrios Galanis, Ion Androutsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2608.18752)  

**Abstract**: Statutory retrieval is necessary for citation-grounded legal question answering, but remains underexplored for Greek. We introduce GreekBarRetrieval, a public retrieval benchmark derived from, and complementing GreekBarBench, which did not include retrieval. The new benchmark comprises 283 bar-exam questions, each accompanied by the facts of the case it refers to, and 6,308 candidate statutory articles to retrieve from. Questions and facts are stated in everyday language, but need to be mapped to the formal terminology of statutes and their abstract legal concepts. A further complication is that not all of the case facts are relevant to each question of a case. Experimenting with three BM25 variants and nine dense retrievers, we find that vanilla dense retrieval far outperforms vanilla sparse retrieval in Recall@100. However, LLM-based query reformulation helps BM25 close that gap, while also improving dense retrieval. With a ten-round ReAct-like LLM reformulation loop that we introduce, BM25 improves further in Recall@100 and obtains the best nDCG and MAP scores of all tested retrievers. Query reformulation also outperforms pseudo-relevance feedback, sparse-dense fusion, and English translation. 

---
# Metrics That Write Themselves: Evolving an Evaluator from Its Own Blind Spots 

**Authors**: Xing Zhang, Yanwei Cui, Guanghui Wang, Zhihao Lin, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2608.18744)  

**Abstract**: Agents improve quickly against a reliable automatic metric and stall without one, and the applications that need them most, report generation among them, are the ones nobody knows how to score. Can the metric write itself? Saying what makes an answer good is hard; pointing at something wrong with one is easier, so the metric we evolve is a pool of small Python operators that each flag a candidate for one named defect, or abstain, and vote. Asking a model for operators directly does not work: 183 candidates realise only 96 distinct behaviours, from one narrow region of an enormous space. EvalCEGAR instead borrows counterexample-guided abstraction refinement from program verification. It reads the pool as an abstraction and searches for a collision, two answers the operators score identically, one correct and one not. That pair, not a prompt, is the authoring request, and when a collision defeats every attempt the loop widens what an operator may read rather than resampling. On MBPP+ and HumanEval+, a sandbox whose hidden unit tests give exact ground truth, the loop writes a 55-line operator that closes 15.4% of the gap between flagging nothing and a perfect filter on 428 unseen tasks (+0.0065, p=0.0010) at a quarter of our best hand-written operator's flags. On the benchmark it never saw it matches that operator's effect exactly on a third of the flags. Six of eight runs admit such an operator and all six help out of sample; our 15 hand-written operators applied together as one filter lose accuracy. An LLM judge on the same information ties that delta on a nearly disjoint set of candidates, and charges a model call per candidate forever where the operator charges none. 

---
# When Safety Overrides Vision: Exploring Dynamics between Vision Influence and Safety Alignment in Vision-Language Models 

**Authors**: Mehak Gupta, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2608.18628)  

**Abstract**: Aligned vision-language models (VLMs) are designed to balance grounded visual reasoning with safe generation behavior. However, we observe a striking phenomenon: under safety-constrained instruction, models frequently abstain from answering questions that remain correctly answerable under default instruction despite receiving identical image-question inputs. This raises a fundamental question: does safety alignment suppress perceptual grounding itself, or does visual evidence remain internally available while generation is redirected toward abstention? In this work, we investigate the internal decoding dynamics underlying safety-induced abstention in aligned VLMs. Across multiple architectures and multimodal benchmarks, we show that abstained generations remain consistently influenced by visual evidence throughout decoding, indicating that perceptual grounding is largely preserved despite refusal behavior. We further demonstrate that, although the representational organization of refusal differs substantially across architectures, safety-constrained instruction consistently alters late-stage hidden-state dynamics toward refusal-oriented decoding. Finally, through targeted activation-level interventions, we show that suppressing refusal-related representations reliably restores grounded answering behavior across models without retraining or modifying visual inputs. Together, these findings reveal a previously underexplored failure mode in aligned VLMs: safety alignment can override grounded visual expression even when perceptual evidence remains internally preserved. 

---
# Can a Lightweight Multimodal Model Estimate LLM Reasoning Performance? A Study for Compute-Optimal Document Inference 

**Authors**: Zishan Ahmad, Vishal Vaddina  

**Link**: [PDF](https://arxiv.org/pdf/2608.18591)  

**Abstract**: Uniformly allocating inference reasoning budgets to LLMs is expensive and prone to over-thinking penalties; especially in document tasks where visual layouts drive complexity. To address this, we introduce BudgetDoc, the first multimodal benchmark providing explicit supervision for model-budget-performance trade-offs across three document tasks. Using BudgetDoc, we train DRB (Document-Reasoning Balancer), an approx. 1B-parameter pre-flight estimator (SigLIP-2 + Qwen3-0.6B) that predicts ordinal model performance across budget levels, achieving a 0.753 weighted F1. When dynamically allocating reasoning budgets across five frontier models and three datasets, DRB matches or improves F1 scores compared to always-maximum-budget baselines in 9 of 15 configurations while drastically reducing cost. Finally, preliminary evaluations demonstrate DRB's potential to generalize to cross-model selection. 

---
# Evaluating and Explaining Prompt Sensitivity of LLMs Using Interactions 

**Authors**: Ruiyang Qin, Qingzhuo Wang, Tian Wang, Zhihua Wei, Wen Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.18539)  

**Abstract**: The remarkable capabilities of large language models (LLMs) are often undermined by their instability. Even subtle and semantically irrelevant changes in prompts can cause dramatic fluctuations in performance, a phenomenon known as prompt sensitivity. Previous studies typically evaluate prompt sensitivity by comparing the LLM's final outputs when prompts change. However, such coarse-grained metrics fail to explain the internal reasons for prompt sensitivity. In this paper, we introduce interactions as a fine-grained tool to analyze prompt sensitivity of LLMs. Specifically, we decompose the output score of the LLM into a set of interactions. Each interaction represents a nonlinear relationship involving a set of input variables. We discover that subtle changes to prompts can trigger severe instability in interactions, even when the outputs of the LLM remain the same. To this end, we propose an Interaction-based Prompt Sensitivity (IPS) metric by quantifying changes in interactions when we introduce subtle changes to prompts. We apply the IPS metric to 50 open-source LLMs and uncover four factors that reduce the prompt sensitivity of LLMs, including supervised fine-tuning, increased model scales, dense architectures, and few-shot learning. More crucially, we discover a common mechanism by which these four factors reduce prompt sensitivity: all four factors tend to reduce the prompt sensitivity of low-order interactions (i.e., interactions involving few input variables). 

---
# Building real-time digital twin instances with Function+Data Flow: user evaluation and extension for iterative pipelines 

**Authors**: Eduardo de Conto, Blaise Genest, Arvind Easwaran, Nicholas Ng, Shweta Menon  

**Link**: [PDF](https://arxiv.org/pdf/2608.18480)  

**Abstract**: Digital twins (DTs) increasingly leverage artificial intelligence (AI) and machine learning (ML) pipelines, both to build real-time DTs from high-fidelity simulations and to instantiate them with historical data. However, engineering these pipelines remains largely ad-hoc: pipelines are hard to specify, validate, and reuse, with scarce dedicated tooling. Function+Data Flow (FDF) addresses this by defining a visual domain-specific language (DSL) that represents functions (ML models) explicitly, enabling their composition and reuse. We implemented FDF in DesCartes Builder, an integrated modeling environment supporting FDF-based DT synthesis and validation.
In this paper, we report on an empirical user study evaluating whether FDF and DesCartes Builder can make AI-based DT development more accessible and reliable. Participants implemented a representative real-time DT prototype within DesCartes Builder, and we measured perceived usability and feature adequacy through quantitative and qualitative measures. Our results indicate that DesCartes Builder and FDF achieve a good level of usability across a broad range of potential users, and particularly for the intended audience of domain experts. The study additionally surfaces concrete strengths and areas for improvement of both the tool and the underlying FDF framework. Informed by these findings, we propose H-FDF, a Hierarchical extension of FDF supporting iterative and modular pipelines, enabling the formal specification of more complex DT pipelines such as dual training. Our findings suggest that integrated, model-driven platforms are a promising direction to transform AI-based DT engineering into a disciplined modeling practice. 

---
# More Context, Same Budget: Dual-Bounded Relational Recall Beyond Top-K Retrieval 

**Authors**: Thomson D. Nguy  

**Link**: [PDF](https://arxiv.org/pdf/2608.18448)  

**Abstract**: More context does not require a larger retrieval budget. Under the same ceiling, a retrieval system can recover more of the evidence a question requires by following relationships between evidence that flat top-k ranking leaves behind. We test that proposition with Dual-Bounded Relational Recall (DBRR), which allocates a fixed retrieval budget between relevance-selected seeds and bounded graph-adjacent context, against matched flat top-k retrieval using the same relevance-ranking stage and the same maximum number of retrieval units and tokens. The outcome is complete recovery of the official HotpotQA supporting-evidence set for each question.
Across 7,405 FullWiki questions, the Primary DBRR allocation increased complete supporting-evidence recovery by 23.8 percentage points over its matched flat baseline (paired risk difference 0.2377; question-level bootstrap 95% interval 0.2269 to 0.2489). It improved 1,952 questions, tied on 5,261, and harmed 192. Bridge questions drove the effect, with a 28.7-point increase; comparison questions showed a smaller 4.2-point difference. In a prespecified, evaluation-only diagnostic population, real relationships also outperformed random-neighbor and degree-preserving shuffled-graph controls.
The result is straightforward: under the same context budget, complete-evidence retrieval depends not only on which items rank highest, but on how context is allocated around them. Relational allocation recovered complete evidence sets that flat top-k retrieval left incomplete. 

---
# Multimodal Rapport Estimation in Real-World HRI 

**Authors**: Akihiro Sakuramoto, Takato Hayashi, Ryo Miyoshi, Yuki Okafuji, Shogo Okada  

**Link**: [PDF](https://arxiv.org/pdf/2608.18401)  

**Abstract**: Evaluating interaction quality in real-world HRI is an important challenge. If interaction quality can be estimated reliably, the results can be used to improve dialogue strategies and ultimately enable robots to adapt their behavior autonomously. However, existing automatic evaluation methods have been developed primarily in controlled laboratory settings, and it remains unclear whether they can be directly applied to real-world environments, where users are free to disengage and multi-party participation may arise naturally. In this study, we investigate the automatic estimation of third-party-rated rapport scores using 62 sessions of multimodal recordings collected in a Japanese drugstore. We compare zero-shot LLMs, pretrained text, audio, and visual models, and their prediction-level fusion. The results show that, in real-world HRI, zero-shot LLMs achieve strong performance, while audio and visual models tend to provide complementary information. In particular, Gemini 2.5 Flash performs strongly as a single model, and a fusion model combining Gemini (text) with HuBERT and V-JEPA performs best overall. Further analyses showed that estimation performance varied across interaction-duration and group-size conditions. These findings suggest that rapport estimation in real-world HRI requires evaluation and model design that account for contextual variability beyond that assumed in laboratory settings. 

---
# Selection, Recombination, or a Fresh Solve? A Candidate-Free Control for Single-Pass Test-Time Aggregation 

**Authors**: Guiv Farmanfarmaian  

**Link**: [PDF](https://arxiv.org/pdf/2608.18379)  

**Abstract**: When every candidate is wrong, correct-candidate selection is unavailable, yet the aggregation call can still solve the problem afresh. A correct aggregate answer may therefore reflect recombination, fresh solving, or both. For efficient test-time reasoning, the relevant question is whether candidate context adds value beyond the additional generation pass. We introduce the missing candidate-free control under the same maximum output-token allowance and stratify by the number of correct candidates. Across AIME-2025 and HMMT-2025 with Qwen3-4B, candidate conditioning improves accuracy when multiple candidates are correct ($\Delta_{\mathrm{cand}}$(c2+) = +0.290), lowers accuracy when every candidate is wrong ($\Delta_{\mathrm{cand}}$(c0) = -0.123), and remains unresolved in the one-correct regime. The c2+ and c0 conclusions survive a conservative correction for the adaptive two-benchmark procedure. Under this counterfactual, the interpretation of all-wrong recovery reverses at this scale: conditioning on an all-wrong candidate pool lowers accuracy relative to a fresh solve. Original-format matching and placebo results characterize the failures descriptively but leave their mechanism unresolved. Within a separate structured intervention, explicit answer fields causally steer outputs toward their values; masking yields no measurable accuracy improvement, and equivalence with the original format was not established. The evidence is limited to one Qwen3-4B family, two mathematics benchmarks, first-answer-truncated candidate fragments, and single-pass prompted aggregation. 

---
# From Inference to Adaptation: A Unified Optimal Transport View of Vision Language Model 

**Authors**: Qi Yu, Zhichen Zeng, Katherine Tieu, Xiyuan Yang, Ruizhong Qiu, Yuchen Yan, Lihui Liu, Yanjun Zhao, Lingjie Chen, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2608.18339)  

**Abstract**: Vision-language models (VLMs) have demonstrated remarkable zero-shot capabilities yet remain sensitive to real-world distribution shifts during inference. Although significant efforts are devoted to adapting VLMs at test time, they rely heavily on noisy pseudo-labels predicted directly from raw embedding similarities during inference, which are unreliable under distribution shift and mislead the adaptation. To avoid noise amplification, existing works craft coarse-grained surrogate objectives during adaptation, which fail to explicitly model sample-level relationships across different modalities, creating objective mismatch with inference, thus leading to marginal performance improvement. In this work, we aim to bridge the detached objectives of inference and adaptation for VLMs, and propose a principled VLM TTA method called \algname. For VLM inference, we formulate the zero-shot image classification task as a cross-modal alignment problem encoded via a Wasserstein OT formulation, providing robust pseudo-labels at the sample-level to effectively adapt VLMs. For VLM adaptation, we adopt a soft-label InfoNCE loss to adapt VLMs based on the OT-induced pseudo-labels, leveraging fine-grained supervisions to explicitly model relationships of individual image-text pairs via contrastive learning, which empowers accurate inference at the same granularity. Moreover, we theoretically reveal that the InfoNCE loss can be neatly reformulated as a Wasserstein OT formulation, thereby unifying the objectives of the inference and adaptation of VLMs to achieve their mutual benefits. Extensive experiments demonstrate the effectiveness and efficiency of our methods, outperforming the best-performing methods by up to 7% with state-of-the-art efficiency. 

---
# ComponentBench: Diagnosing Component-Level Failures in Computer-Use Agents 

**Authors**: Tianchen Guan, Xinlei Lin, Royce Cheng-Yue, Xiangjun Wang, Shuyan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.18307)  

**Abstract**: Current evaluation of computer-use agents is split between long-horizon workflow benchmarks and atomic GUI-grounding tests. This leaves an under-instrumented middle layer: realistic component-centered interactions (e.g., toggle a button set) that are short enough to diagnose and rich enough to capture the burdens of modern interfaces. We present ComponentBench, a benchmark and diagnostic pipeline for component-level evaluation of computer-use agents on modern web UIs. ComponentBench is organized around a library-agnostic ontology of 97 canonical UI components instantiated as 2,910 programmatically verified tasks across widely used component libraries, paired with cleaned human reference trajectories that enable evaluation of both task success and interaction efficiency. Beyond task collection, we introduce a scalable pipeline for auditing realized structural difficulty after implementation and synthesizing structured failure analyses across tasks and component families. Evaluating seven models -- GPT-5.4, Gemini 3 Flash, GPT-5.4 mini, GPT-5 mini, Gemini 3.1 Flash-Lite, Qwen3-VL-235B, and UI-TARS-1.5-7B -- across four observation and action spaces, we show that these design choices critically impact performance. Within a single shared harness, changing only the observation and action space shifts task success by more than 30% for the same model: GPT-5 mini falls from 83.1% with accessibility-tree observations to 48.9% with coordinate-only Pixel control. Moreover, even the fastest configuration takes 3.7x as long as the matched human reference, and spatial manipulations that are trivial for humans continue to challenge current agents. 

---
# Debiased Inference for AI-Generated Data without Gold-Standard Labels: Identification via Multiple Imperfect Measurements 

**Authors**: Naoki Egami, Sooahn Shin  

**Link**: [PDF](https://arxiv.org/pdf/2608.18294)  

**Abstract**: An increasing number of scholars use AI to measure variables they subsequently include in downstream analyses. Although AI-measured variables are often analyzed as if observed without error, ignoring prediction errors in automated measurement leads to substantial bias and invalid confidence intervals in downstream analyses, even if AI measurement accuracy is high, e.g., above 90%. Existing solutions, such as design-based supervised learning and prediction-powered inference, combine error-prone AI-based measurements with gold-standard labels, which may be costly and difficult to obtain in some application areas.
In this paper, we propose debiased inference with multiple imperfect measurements (DMM), a framework that combines multiple error-prone AI measurements to enable valid downstream inference without gold-standard labels. Building on the established results on CP decomposition, DMM assumes that these measurements are independent conditional on the latent true label and observed unit-level features, such as text features represented by embeddings. This framework allows for unknown misclassification rates to vary across annotation methods (e.g., large language models) and across units of annotation (e.g., texts). Under this assumption, we use semiparametric inference theory to prove that the DMM estimator is consistent and asymptotically normal, enabling valid inference for a wide range of downstream statistical analyses common in the social sciences. Our simulation results show that DMM yields valid inference and that adding accurate, though imperfect, measurements can improve efficiency. Focusing on common applications of large language model annotations, we also develop diagnostics to assess the conditional independence assumption. 

---
# What Makes Software Issue Resolution Tasks Difficult for Agents? 

**Authors**: Ebtesam Al-Haque, Brittany Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2608.18280)  

**Abstract**: Background. Advances in agentic systems are simultaneously, and rapidly, saturating benchmarks. Despite this often discussed phenomena, benchmark scores remain difficult to interpret due to the lack of control and characterization of task difficulty. More specifically, we currently have little understanding of what makes one task harder than another, and to what extent task difficulty is predictable from static task properties. Aims. We propose a measurement framework to investigate and systematically quantify what structural properties of software tasks correspond to agent success rates for issue resolution tasks. Method. We conducted a large scale empirical study on CoderForge-Preview, the largest open dataset of coding agent trajectories to date, by extracting features across task patch, repository and prompt. We evaluated the predictive power of each feature against task outcomes using ensemble methods, SHAP attribution, and effect size analysis. Results We found that task difficulty is substantially predictable from static features (AU C = 0.863) and is largely driven by patch fragmentation and repository scale. Prompt linguistic features become visible among top contributors for tasks in the mid-band, revealing a layered structure of difficulty. Conclusion. The difficulty of an issue resolution task is encoded in its structure. This enables static, pre-hoc difficulty estimation and lays the groundwork for difficulty-controlled benchmark construction for evaluation of agents. 

---
# Redakto - The Incognito Tab for LLMs 

**Authors**: Saurav Kumar Saha, Tom Röhr, Felix Bießmann  

**Link**: [PDF](https://arxiv.org/pdf/2608.18260)  

**Abstract**: Large Language Models (LLMs) are being increasingly used in everyday applications. A major challenge in the context of LLMs or Artificial Intelligence (AI) in general is to ensure privacy when using them, meaning that personally identifiable information (PII) is removed from any text that enters an LLM. These challenges have become more urgent with novel EU legislation. Uncertainty around LLM usage with respect to privacy concerns in EU countries can be a major blocker for the speed of innovation and transfer from research to applications. Here we present \textbf{Redakto}, a tool that can be used for anonymizing text prior to feeding it to an LLM or other downstream text processing. We provide state-of-the-art functionalities for both redaction of PII but also when used for pseudonymization. These functionalities are exposed such that they can easily be used by end-users, through the Redakto web application, and by developers and researchers, via REST APIs and model context protocol (MCP) hooks. The implementation is fully open source, requires modest compute resources, and can be readily deployed on local hardware. In contrast to prior work and in order to better assess the quality of the anonymized texts, we conduct extensive empirical evaluations on textual data from legal and medical domain with respect to both privacy and utility of the redacted texts. Our empirical results demonstrate that the texts anonymized with different redaction strategies achieve utility scores on par with the original texts, suggesting that anonymization with Redakto can be used for LLM tasks without substantial negative impact for the tasks we explored. 

---
# Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth 

**Authors**: Ivan Viakhirev, Kirill Borodin, Amirah Almutairi, Serguei Barannikov, Maxim Abramov, Grach Mkrtchian  

**Link**: [PDF](https://arxiv.org/pdf/2608.18222)  

**Abstract**: Recurrent-depth reasoners aim to solve harder problems by iterating their update longer at test time, but additional iterations can improve, preserve, or degrade an answer. We show that a measurable property of the trained operator, its finite-time dynamical regime (estimated as settling, marginal, or drifting), indicates which of these occurs. We give a sufficient condition for depth-safety: once an operator's per-step displacement is small relative to the decoder margin, the decoded answer cannot change under further iterations. Empirically, on algorithmic tasks trained from $800$ unaugmented examples per difficulty tier, settling operators do not degrade with added depth, and on some tasks convert it into higher accuracy on harder unseen instances (Sudoku, $0.19$ to $0.34$ past the training horizon). A single terminal fixed-point objective moves the regime and the depth behavior together: removing it induces drift and removes the gains, and adding it to a generic recurrence yields depth-safe extrapolation on carry propagation. We give four operational criteria for useful test-time depth, use them to catalogue failure modes, and, as a consistency check, apply the same measurements to Huginn-3.5B, which falls in the non-settling family. 

---
# MicroPython and CircuitPython: Pythons Quiet Takeover of IoT and Robotics 

**Authors**: Sayed Mahbub Hasan Amiri, Atiar Zahan  

**Link**: [PDF](https://arxiv.org/pdf/2608.18160)  

**Abstract**: Background: Python has become the dominant language in software and data science, yet embedded systems have remained tied to C/C++ due to performance and memory constraints. MicroPython and CircuitPython are changing this by bringing Python to microcontrollers, lowering barriers for IoT and robotics development. Aim: This article examines whether these platforms are achieving a quiet takeover of embedded systems, focusing on ecosystem growth, practical applications, performance trade-offs, educational adoption, and prospects. Methods: A mixed-methods design was used, including quantitative analysis of GitHub, Stack Overflow, and Google Trends data; curation of case studies from this http URL, this http URL, and the Adafruit Learning System; and original benchmarks on ESP32 and Raspberry Pi Pico comparing MicroPython, CircuitPython, and Arduino C++ across GPIO, I2C, SPI, Wi-Fi, and memory usage. Results: Metrics show sustained growth, with MicroPython supporting over 200 boards and CircuitPython over 400. Benchmarks reveal 10-20 times slower I/O and four to six times higher memory use than C, but performance remains adequate for common sensor and network tasks. Case studies demonstrate successful deployment in home automation, robotics, wearables, agriculture, and professional prototyping. Education emerges as a primary adoption driver. Conclusions: Python is not replacing C/C++; rather, it is becoming the default prototyping and educational language for embedded systems. Continued hardware improvements, better tooling, and standardization will likely deepen this trend. The article offers balanced, evidence-based insights for developers, educators, and technology decision-makers. 

---
# Efficient Adaptation of LLMs for Hate Speech Detection in Low-Resource Languages: A Comparative Study on Roman Urdu 

**Authors**: Toneema Zubair, Muhammad Junaid Asif, Faisal Kamiran, Hafiz Hassan Saeed, Rana Fayyaz Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2608.18142)  

**Abstract**: It is challenging to detect hate speech in Low Resource Languages (LRLs) because of the absence of annotated data, the informality of its language structure, and the lack of standardized grammar. A good example of such a challenge is Roman Urdu which is broadly used by South Asians on social media and has a high variation while lacking contextually consistent spellings. The objective of this paper is to conduct a comprehensive assessment of Large Language Models (LLMs) for Hate Speech Detection (HSD) in Roman Urdu script and fine-tune these models using the Parameter-Efficient Fine-Tuning (PEFT) method called Low-Rank Adaptation (LoRA). To evaluate zero-shot inference, we benchmarked it against PEFT on different transformer models, including Mistral, LLaMA, Falcon, and multilingual BERT. Experiments are conducted on the PURUTT (Parallel Urdu and Roman Urdu Corpus for Toxic Comments and Transliteration) dataset with over 72,000 annotated comments. The results suggest that zero shot models perform moderately (F1 = 0.56), but updating a small fraction of the model trainable parameters improves the classification performance significantly (F1 > 0.93). Our results have shown that PEFT delivers outstanding performance alongside excellent computational efficiency, making it highly suitable for low-resource language processing tasks. 

---
# Safety Alignment Illusion: The Cross-Lingual Safety Gap in LLMs 

**Authors**: Namya Bhatnagar  

**Link**: [PDF](https://arxiv.org/pdf/2608.18131)  

**Abstract**: Current safety alignment training for Large Language Models (LLMs) are heavily English-centric. When such safety filters fail for non-English languages, the consequences are immediate and user-facing: voice assistants and spoken dialogue systems may produce stereotype-reinforcing outputs, bypassing the standard English-focused safety alignments and propagating harmful bias to non-English speaking communities. For spoken language technologies deployed across India's linguistically diverse population, this represents a critical failure mode. To address this cross-lingual gap, we introduce INCLUDE (Indian Cultural Lens for Understanding and Detecting Embedded Biases), a multilingual evaluation benchmark designed to quantify Indian-centric socio-cultural biases. INCLUDE consists of 2,604 prompts spanning six prompt languages: English, Hindi, Bengali, Marathi, Tamil, and Hinglish (Hindi-English code-mix). We evaluate ten open- and closed-source LLMs against this benchmark, analyzing 14,988 bias scores. Our statistical results reveal two key findings. First, Bengali yielded the highest average bias score in open-source models. Second, English demonstrated a notable reversal, producing the lowest bias in open-source models but the highest bias in closed-source models. 

---
