# CritICL: Inference-Time Weak-to-Strong Generalization from Small Language Model Failure Modes 

**Authors**: Yufan Wu, Yinghui He, Zhengyi Hu, Lang Wei, Ruichen Li, Qifan Yang, Ting Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27455)  

**Abstract**: Recent advances in inference-time scaling have significantly improved the reasoning performance of large language models (LLMs). However, these methods typically rely on repeated generation or external verification. To address this limitation, we introduce CritICL, a novel inference-time framework that improves reasoning while maintaining high efficiency. Our key insight is that LLM failure modes exhibit structured patterns across model scales within the same family. Instead of treating failures as undesirable outputs, CritICL leverages them as a source of guidance. Specifically, we utilize failure modes derived from weaker models and incorporate them into inference through critique-based in-context examples. We propose two variants: CritICL-dynamic, which adaptively predicts input-specific failure modes and retrieves critiques, and CritICL-static, which uses a global failure mode profile to provide stable guidance. Experimental results show that CritICL consistently outperforms standard in-context learning and achieves performance competitive with or superior to test-time scaling methods, while requiring significantly fewer generations and lower token cost. Code available at: this https URL 

---
# TTPO: Test-Time Policy Optimization 

**Authors**: Aozhe Wang, Zhengxi Lu, Jianze Wang, Shangke Lv, Ying Liu, Weiming Lu, Jun Xiao, Yueting Zhuang, Hua Yang, Qianglong Chen, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.27448)  

**Abstract**: Recent prominent post-training methods, such as Reinforcement Learning (RL) and On-Policy Self-Distillation (OPSD), have driven rapid progress in mathematical reasoning for large language models, yet their reliance on ground-truth labels precludes test-time training (TTT). Replacing ground truth with majority-vote pseudo-labels is a natural alternative, yet it is fragile: an incorrect vote corrupts the teacher and misleads every token. We observe that this failure mode is asymmetric: rollouts that disagree with the pseudo-label are typically wrong regardless of whether the vote itself is correct. Building on this observation, we propose Test-Time Policy Optimization (TTPO), an asymmetric objective that distills agreeing rollouts via OPSD and penalizes disagreeing rollouts with Grouped RL. Token-level selection further refines both branches: distillation down-weights already-converged positions, while RL penalizes only confident errors. Both updates remain well-grounded even under frequent pseudo-label errors, and majority-vote routing yields tighter self-supervision as the model improves. Without any labels, TTPO matches label-supervised OPSD on five competition-level benchmarks, raises Qwen3-1.7B from 38.0% to 45.2% in TTT, yields +25.2% to +36.4% without thinking, and shows strong cross-task generalization. 

---
# Stochastic Estimation of Transduced Language Models 

**Authors**: Vésteinn Snæbjarnarson, Samuel Kiegeland, Manuel de Prada Corral, Ryan Cotterell, Tim Vieira  

**Link**: [PDF](https://arxiv.org/pdf/2608.27428)  

**Abstract**: Transduced language models (TLMs) compose a pretrained \emph{source} language model with a functional finite-state transducer to induce a language model over \emph{target} strings. Computing the probability of a target prefix under a TLM amounts to summing the source-model probabilities of all source strings that the transducer maps to target strings beginning with that prefix. This set can be exponentially large or infinite. Prior work uses a computational shortcut based on source prefix probabilities, then approximates the resulting sum with threshold-pruned beam summing. This produces a lower bound with unknown error. Instead, we resample source prefixes without replacement and reweight each selected prefix by the inverse of its inclusion probability. We show that applying this correction recursively gives an unbiased estimator of the target prefix probability and lets us estimate the mass lost by threshold pruning. Our beam-summing algorithm extends the retained source prefixes and samples which prefixes to keep, reducing their number as more probability mass is added to the running estimate. This can save computation and guarantees that the run halts with probability one. We evaluate the method on encyclopedic text and DNA against sequential Monte Carlo baselines that resample with replacement. It achieves a better compute--variance tradeoff on text and lower error at the same maximum number of particles on DNA. On a DNA-to-amino-acid transduction, it reduces runtime by several orders of magnitude relative to threshold-pruned beam summing and makes estimating prefix probabilities for long target strings feasible. Replacing threshold pruning with unbiased sampling in a published reading-time analysis substantially lowers the estimated corpus surprisal but leaves the published conclusions unchanged. 

---
# Boosting LLM Exploration via Weak-Model Guidance in RLVR 

**Authors**: Xingyu Shen, Huishuai Zhang, Peng Li, Yinchun Wang, Dongyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.27420)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) significantly improves LLM reasoning but often causes a drop in policy entropy, leading to narrowed reasoning coverage and degraded pass@$k$ for large $k$. While existing methods mitigate this entropy collapse through algorithmic regularizations, cross-model non-parametric perturbation is also neglected. In this work, we propose a simple yet effective approach to preserve the generative diversity of LLMs during RLVR. Instead of relying solely on internal exploration, we force the target model to generate answers based on partial reasoning trajectories generated by a smaller, weaker language models. These unfamiliar prefixes effectively disrupt over-confidence and encourage the exploration of distinct reasoning paths. We empirically study the potential of outer prefixes, revealing the mechanism of the impact of distributional discrepancy to the exploration dynamics in RLVR training. Experiments across multiple mathematical benchmarks show that our method consistently outperforms vanilla RLVR. Notably, the performance gain becomes increasingly pronounced as $k$ scales up, demonstrating a substantial expansion of reasoning coverage. Furthermore, our approach efficiently mitigates entropy collapse without requiring additional SFT, intricate reward designs, or complex prompting. 

---
# Consolidating RLVR Capabilities Across Domains: A Deep Dive into Fusion Paradigms 

**Authors**: Siye Wu, Kai Yang, Yuchen Cai, Xin Xu, Peng-Yuan Wang, Jiaxuan Wang, Jiashun Liu, Jiafei Lyu, Yangkun Chen, Saiyong Yang, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.27409)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) improves specific capabilities of large language models, but covering multiple capabilities often involves training separate domain experts and subsequently consolidating them. We organize three fusion paradigms by the artefacts they reuse: Merge combines expert task vectors, Mix RL pools their datasets, and multi-teacher on-policy distillation (MOPD) uses both. Because they have largely been studied in isolation, how they compare and how to choose among them remain unclear. We compare all three using shared experts and data across model scales and a multi-domain benchmark suite. Although their average performance differs by at most 1.4 points, the gap reaches 8.6 points on a single benchmark, with domain-level variation tracking cross-domain relations visible in task-vector geometry. Training dynamics expose distinct constraints: Mix RL depends on domain mixture proportions, MOPD remains bounded by its teachers, and Merge compresses all expert updates into one. All three improve single-sample accuracy without measurable gains in solution coverage or losses in held-out capabilities. These results yield a practical guideline: use Merge when experts already exist and cheap fusion is paramount; Mix RL when training a unified model without experts, with domain proportions adjusted for cross-domain transfer; and MOPD when preserving domain-specific gains matters more than surpassing teachers or minimizing end-to-end cost. 

---
# How Language Models Organize and Structure Moral Knowledge 

**Authors**: Orion Reblitz-Richardson  

**Link**: [PDF](https://arxiv.org/pdf/2608.27402)  

**Abstract**: How do large language models (LLMs) organize moral knowledge? Models detect moral content broadly, but detection is a low bar. We ask whether they go further, distinguishing moral foundations from one another and organizing the relationships between them geometrically.
We train six independent linear probes on open-weight language models, one per Moral Foundations Theory (MFT) category (care/harm, fair/cheat, lib/oppress, loy/betray, auth/subv, sanc/degrade), and examine how the resulting directions relate to each other in representation space. We find the directions neither collapse into a single moral detector nor isolate from one another. Rather, they span a near-maximal number of independent dimensions while sharing a positive common component. The shared component is the signature of integration, and it is moral-specific relative to a matched non-moral concept battery built identically (mean pairwise cosine 0.26 vs. 0.013).
The geometry is consistent across architectures and scale and reaches its integration regime early in pre-training, well before probe accuracy saturates. The structure the model discovers shows no evidence of the individualizing/binding distinction predicted by Moral Foundations Theory (an underpowered test: only 20 candidate partitions exist) but rather reflects corpus statistics. Extending to moral dilemmas, each dilemma direction partially composes from its component foundations, at 2.7x a mismatched-pair baseline, while the majority of its variance encodes conflict-specific structure. The model represents moral tension itself, not a pre-resolved judgment. 

---
# Making Clinical Language Models Auditable: Concept-Guided Fine-Tuning for Robust Prediction 

**Authors**: Jin Mu, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.27397)  

**Abstract**: Clinical language models can achieve strong in-hospital accuracy yet fail under deployment shifts because they exploit note-specific artifacts (e.g., templates, separators, boilerplate) that do not reflect patient state. We propose CAST (Concept-guided Artifact Suppression Tuning), an SAE-based framework for auditable clinical text classification. CAST uses Sparse Autoencoders to expose sparse, human-auditable features from intermediate Transformer activations, labels SAE latents with an LLM-assisted interpretation pipeline and ICD-10 retrieval constraints, suppresses verified artifact latents via residual subtraction during fine-tuning, and provides post-hoc per-concept attributions for auditing model decisions. On MIMIC-IV discharge-note mortality prediction, CAST improves over its corresponding fine-tuned encoder baselines and remains competitive with strong LLM baselines, while producing a feature-level audit trail of the clinical concepts that support each prediction and the artifact concepts suppressed during training. 

---
# RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature 

**Authors**: Maayan Sharon, Tom Hope  

**Link**: [PDF](https://arxiv.org/pdf/2608.27394)  

**Abstract**: Retrieved scientific literature can serve as inspiration for both human and AI scientists. Inspiration can take different forms: prior work may directly suggest how to address a problem, or surface directions at different levels of abstraction - zooming out to a more general view or zooming in to a concrete realization. We introduce RATIO (Retrieval Across Typed Ideation Operations), a large-scale benchmark in which relevance is defined by three operations which we name ideation moves: Address retrieves potential approaches for stated problems, Broaden retrieves more general formulations, and Specify retrieves concrete instantiations. RATIO is constructed from millions of full-text scientific papers across CS literature via a general recipe that extends discourse-marker distant supervision - previously used only for classification - to corpus-scale retrieval, combined with extensive LLM and human vetting. Experiments show that operation-specific fine-tuning substantially boosts retrievers but leaves much room for further improvements. RATIO provides a scalable training and evaluation framework for retrieval components that support literature-grounded ideation, opening up new research avenues on scientific inspiration retrieval. 

---
# D2C-Routing: Dimension-to-Composition Evidence Routing for Mixed-Origin AI-Generated Text Detection 

**Authors**: Xin Chen, Fuwei Zhang, Yiqi Tong, Wei Guo, Yutian Xiao, Fuzhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2608.27380)  

**Abstract**: AI-generated text detection is commonly framed as a binary document-level judgment about whether a text is human-written or machine-generated. This framing breaks down for mixed-origin writing, where content origin and expression origin may differ. We cast mixed-origin detection as dimension-to-composition source attribution, inferring content origin and expression origin before composing them into four collaboration types. We propose Dimension-to-Composition Routing (D2C-Routing), which routes content-side and expression-side evidence to supervised dimension heads before a learned gated composition layer predicts the final label. On MixD2C, a reconstructed split derived from the HART mixed-origin benchmark, our disclosed D2C-Routing-based detector system reaches 0.8603 four-way Avg TPR@1%FPR, 6.5 points above the same-split RACE-local rerun. Core ablations support the routing design, while error analysis shows that distinguishing AI-content/human-expression from fully AI-generated text remains the hardest boundary. Code is available at this https URL. 

---
# Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090 

**Authors**: Kairong Luo, Jiarui Cui, Yaorui Yin, Shengqi Chen, Yiming Yang, Linxiang Gao, Yanmohan Wang, Mingzhe Zhang, Kaiyue Wen, Kaifeng Lyu, Wenguang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.27370)  

**Abstract**: Language model pretraining has become almost synonymous with prohibitive cost, placing it out of reach for much of the academic and open-source communities. Although strong open-source efforts already exist, including open-weight models and open-source training recipes, a cost-efficient, hardware-accessible, and open-source pretraining recipe has long been missing. Even at a small scale, training Llama-3.2-3B costs over \$1.5M, and reproducing SmolLM3-3B needs over \$700K. In this report, we present an open pretraining recipe designed to lower this barrier. Using this recipe, we train a collection of Puro-2B models from scratch on up to 1.4 trillion tokens with FP8 precision on consumer-grade RTX 5090 GPUs. The models in the collection differ in token budgets and selected recipe variants. Our best model is trained at a compute cost of less than \$6.9K and approaches Qwen2.5-1.5B performance under our evaluation protocol. This cost efficiency is enabled by a combination of approaches, including hardware selection, low-precision training, hyperball optimization, curriculum model averaging, and the data recipe. Beyond the recipe itself, we provide two additional results. First, across the Puro-2B collection, we derive a Puro Cost Scaling Law that relates training cost to average model performance; the fitted law suggests that about \$4.4K, less than \$5,090, is sufficient to reach the performance of Qwen2-1.5B. Second, as an end-to-end case study, we examine how pretraining data curricula shape downstream performance after post-training. Such controlled studies are enabled by having access to the full pretraining pipeline rather than model weights alone. We release the full training recipe for Puro-2B, including data, code, and model weights under Apache 2.0 at this https URL. 

---
# Your Voice Cloning System is Secretly a Voice Anonymizer 

**Authors**: Romolo Muletta, Felix Matthias Saaro, Mark Cieliebak, Jan Deriu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27360)  

**Abstract**: Speaker anonymization suppresses speaker-identifying attributes from speech while preserving linguistic content and quality. We propose repurposing XTTSv2, a multilingual voice cloning model trained on 27k hours of speech, for speaker anonymization without retraining. Our key insight is that XTTSv2's voice cloning capabilities preserve prosodic structure independently of speaker identity, enabling voice conversion by conditioning on a pseudo-speaker. We introduce an iterative refinement strategy that balances privacy and utility by maximizing a harmonic mean of speaker dissimilarity and intelligibility. Evaluated on seven European languages across CommonVoice and Multilingual LibriSpeech, our system achieves near-optimal privacy (EER $\approx$ 0.49), competitive intelligibility, and substantially better speech quality than dedicated anonymization baselines, while requiring no language-specific training. We release the code here: this https URL. 

---
# RCMN: Understanding Misleadingness in Influential Public Discourse 

**Authors**: Peiling Yi  

**Link**: [PDF](https://arxiv.org/pdf/2608.27358)  

**Abstract**: Influential public discourse shapes public beliefs and can also mislead, not only through what is stated, but also through how information is framed, omitted, contextualised, and communicated. Yet less research has focused on how such misleadingness arises and shapes the interpretations formed by readers. To address this gap, we introduce Reader-Centric Misleadingness Understanding (RCMN), a framework that operationalises misleadingness through five dimensions: misleading mechanism, likely reader interpretation, evidence-warranted interpretation, emotional arousal, and communicative intent. Based on this framework, we construct an evidence-grounded dataset of influential public discourse. Empirical findings show that misleadingness is diverse and extends well beyond fabrication, with unsupported inference, exaggeration, and omission among the prevalent mechanisms, and is frequently associated with heightened emotional arousal and distortive communicative intent. Moreover, we investigate whether lightweight claim-and-context representations retain sufficient cues for understanding reader-centric misleadingness without access to richer contextual, evidential, and multimodal information. Evaluation across five recent generative foundation models shows that reader-level interpretations can often be recovered from such limited representations, whereas identifying how misleadingness is produced remains considerably more challenging. These findings highlight the potential of lightweight representations for scalable misleadingness analysis, while reliable understanding of misleading mechanisms continues to require richer contextual and evidential grounding. 

---
# INTENT-AS-A-TOOL Makes it Easy to Track Agentic Misalignment 

**Authors**: Yutong Zhang, Jianshuo Dong, Peng Xu, Long Wang, Jie Zhang, Tianwei Zhang, Xiaoping Zhang, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27348)  

**Abstract**: As large language models (LLMs) are deployed as autonomous agents, safety failures increasingly involve consequential actions. We study agentic misalignment, where agents take harmful actions under goal conflicts and pressures. Using chain-of-thought (CoT) monitoring, we find that harmful execution is often preceded by intent signals in reasoning. However, post-hoc CoT labels are too coarse to show how intent changes during generation. We introduce INTENT-AS-A-TOOL, an approach that adds intent-targeted tools to give the model a dedicated channel for expressing commitment to a target behavior. The probability of calling an intent tool provides a judge-free, fine-grained signal of the model's tendency to pursue that behavior. Our results show that INTENT-AS-A-TOOL complements CoT monitoring, expands post-hoc CoT labels into dense trajectories, and identifies critical steps for online intervention. These findings suggest that action preferences are useful for tracking agentic misalignment during reasoning. Our code and data are accessible: this https URL. 

---
# Pair-Level Essay-Scale Republication and Reuse from Fragmented Historical Text Reuse: A Workflow Study on Eighteenth-Century Books and Newspapers 

**Authors**: Ke Shu, Kira Hinderks, Eetu Mäkelä, Mikko Tolonen  

**Link**: [PDF](https://arxiv.org/pdf/2608.27343)  

**Abstract**: This paper addresses the recovery of essay-scale republication and reuse from fragmented text-reuse evidence, a setting whose central challenge is pair-level evidence consolidation and not fragment retrieval alone. The study focuses on a candidate set centered on essays by eighteenth-century Scottish philosopher David Hume, spanning books from ECCO (Eighteenth Century Collections Online) and historical newspapers. Because the input consists of fragmented reuse hits instead of clean document pairs, and positive coverage is inherently incomplete, we formulate the task as pair-level evidence consolidation into plausible transmission relations and compare three methodological families: a staged rule-based workflow, baselines (a decision tree and two direct LLM settings), and automated rule adaptation. On labeled ECCO--ECCO slices, pair-level feature aggregation alone already reaches 0.948 F1 on the main labeled slice, while the final workflow gives the strongest overall precision-recall trade-off among the tested rule stages. On the full ECCO--ECCO candidate universe, direct LLM baselines flag up to 14,886 pairs as reprints compared to 771 for the final workflow, behaving in this direct-prompt setup as high-recall candidate expanders rather than precision-controlled deployment classifiers. On ECCO--Newspaper, manual audit confirms all 176 predicted positives as genuine cases of republication or reuse, while issue duplication and source-side multiplicity reveal additional provenance structure. Under incomplete ground truth, auditable pair-level evidence consolidation provides a practical way to produce compact candidate spaces for historical inspection. 

---
# BTS-AgentBench: A Deterministic, Replayable Pipeline from Read-Only Telemetry Logs to Agent Benchmarks 

**Authors**: Jeong-Yoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.27334)  

**Abstract**: Industrial sites contain large volumes of read-only telemetry, but few benchmarks specify how to compile these records into executable multi-turn agent tasks. We present a telemetry-to-episode construction method instantiated as BTS-AgentBench. The pipeline normalizes BTS metadata and raw histories into a read-only tool store, compiles static tasks with tool-derived gold answers and evidence, and lifts retained tasks into typed, bounded operator-facing episodes. The 532-row release adds clarification, goal revision, timestamp policy, quality-gated reporting, and evidence attribution while preserving the source computation and split. Coded contract preflight reports zero findings, and the construction-exclusion controller completes 0/532 rows. Two independent raw-to-episode builds match all 11 logical tool-store exports and reproduce the released 356/87/89 train/dev/test artifact exactly. Applying the shared construction path to XAI4HEAT produces 204 episodes; on its 41-row held-out test split, the controller completes 0 rows and the retained GPT-5.5 execution completes all 41. Code, artifacts, and replay reports are available at this https URL. 

---
# Difference-in-Differences on a Censored Rating Scale Can Manufacture an Effect: Evidence from a Pre-Registered LLM-Judge Audit 

**Authors**: Shuyi Fan, Boyuan Deng, Mengyu Xu, Xinhong Xie, Chenyang Li, Hongyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.27309)  

**Abstract**: Audits of LLM judges certify a bias by contrasting matched conditions, and the strongest designs difference twice: a within-item contrast between two candidate responses, differenced again across a manipulated attribute, read off a bounded rating scale. We show that this endpoint is not identified on the scale that reports it. Each term of the double difference is censored by its own share, so the observed statistic confounds differential preference with differential attenuation: a severity shift common to both responses manufactures an interaction whenever the two censor it unequally, as unequal distances from the bounds make them, exactly where good stimuli place them. We exhibit the failure inside a pre-registered audit of a frozen pedagogy judge, sealed before the first of its 990 calls. The registered primary endpoint, the effect of a stated learner profile on the judge's scaffolding preference, is null: $+0.085$ points (95\% BCa $[-0.167, +0.353]$, $p = 0.684$). The audit's one nominally significant interaction, $+0.378$ ($p = 0.002$), is not identified as preference: a construction containing zero differential preference reproduces 79 to 85\% of it from the observed severity shift and the scale floor alone. We derive the mechanism in closed form and show that its contribution is measurable from an audit's own ratings. 

---
# SCIT: Testing Causal Cache Carriers in Latent Chain-of-Thought Models 

**Authors**: Yi Ding, Lijun Huang, Menglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.27265)  

**Abstract**: Latent chain-of-thought models move intermediate reasoning from emitted text into continuous states, improving compactness but hiding the causal object. We introduce SCIT, the Suffix Cache Interchange Test, a causal protocol that constructs exact source-recipient counterfactuals, patches declared cache segments, and identifies which transformer object carries the counterfactual computation. SCIT combines sufficiency tests with K/V component splits, hidden-state controls, semantic source controls, decoded validation, and matched corruption. On CODI-GPT2 and a Sim-CoT-style GPT-2 reproduction, counterfactual arithmetic transfers primarily through value-cache suffix trajectories rather than hidden states, keys, reusable answer slots, or single-token triggers. Complete sufficiency-and-necessity evidence for the late-value-suffix mechanism holds for the main CODI-GPT2 checkpoint; the Sim-CoT-style checkpoint shows the same sufficiency and decoded-control pattern but insufficient matched-corruption evidence for a necessity call. Beyond these local arithmetic cells, SCIT reveals carrier-regime shifts: arithmetic-like GPT-2/1B cells preserve latent-tail value/KV transfer, whereas competent 8B and repaired non-arithmetic cells route through prompt-prefix or full-cache K/V; boundary cells receive no mechanism call. SCIT therefore contributes a cache-level diagnostic, a checkpoint-specific GPT-2 arithmetic mechanism, and a competence-gated carrier map rather than a universal latent-tail claim. 

---
# BALMS: Benchmarking Agentic LLMs for Longitudinal Mental Health Sensing 

**Authors**: Yu Yvonne Wu, Arvind Pillai, Yuliang Chen, Yuwei Zhang, Sudarshan Regmi, Tess Z. Griffin, Michael V. Heinz, Lisa A. Marsch, Nicholas C. Jacobson, Andrew Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2608.27219)  

**Abstract**: Mental health assessment relies on episodic self-report scales, which convert subjective states such as stress into numerical scores but provide only sparse snapshots of wellbeing. Wearable devices offer longitudinal behavioral and physiological signals for continuous, low-burden monitoring. Recent LLM-driven personal-health agents enable natural language queries over wearable signals, but mainly handle short-term, retrieval-based lookups (e.g., highest step count over a week). They do not evaluate whether agents can reason over long-term signals to predict wellbeing scores paired with evidence-grounded rationales. To address this gap, we introduce BALMS, the first systematic benchmark of LLM-based agentic systems for longitudinal mental health sensing. BALMS spans 3 real-world longitudinal datasets, 2 task families (closed-form wellbeing-score prediction and rationale generation auto-graded by an LLM-as-Judge), 3 agentic paradigms evaluated across 5 open- and closed-source LLM backbones. We find that zero-shot agents rarely outperform a simple mean baseline, except with stronger backbones or compact, semantically meaningful features. Chain-of-thought prompting improves reasoning-oriented backbones, but does not guarantee temporal grounding or numerical correctness. Together with more analysis on efficiency and temporal scaling, BALMS highlights the need for longitudinal mental health agents that selectively retrieve history, ground temporal evidence, and reason over interpretable behavioral features. 

---
# When Text Misleads: Inconsistent-Aware Reasoning for Audio-Grounded Dialogue 

**Authors**: Yen-Ju Lu, Yuzhe Wang, Yaohan Guan, Xiluo He, Jiarui Hai, Mingrui Liang, Kaavya Chaparala, Thomas Thebaud, Laureano Moro-Velazquez, Najim Dehak, Jesus Villalba  

**Link**: [PDF](https://arxiv.org/pdf/2608.27176)  

**Abstract**: Understanding spoken dialogue requires joint reasoning over lexical content and paralinguistic acoustic signals such as emotion and conversational intent. However, existing evaluations often allow shortcuts based on transcripts or single-modality solutions, obscuring whether models genuinely ground predictions in speech. We formalize this failure mode as cross-modal disagreement, where transcripts suggest plausible but incorrect surface interpretations while acoustic cues such as prosody or speaking style support different answers. We develop a scalable framework that identifies text-biased surface interpretations and converts disagreement regions into conflict QA examples. We also include consistent cases where transcript-based and speech-grounded interpretations agree, enabling evaluation beyond adversarial audio dependence. This results in ContraTalk, a controlled benchmark containing 501 questions across five discourse dimensions: interaction behavior, emotion state, dialogue act, social stance, and conversational intent. We further develop an agentic-style reasoning framework that converts speech into an Audio Twin, a text-readable representation of localized acoustic cues that exposes acoustic evidence to the reasoning model. Experiments show that strong text-only LLMs exceed 90% accuracy in consistent cases but drop to 33-48% in conflict cases. Direct AudioLLMs provide only partial grounding, still selecting the transcript-biased trap in roughly 30-40% of conflict cases. Our Audio Twin framework improves conflict-case accuracy while reducing trap selection, but its consistent-case behavior remains backbone-dependent. These results identify transcript-based shortcuts as an important failure mode in spoken dialogue understanding and show that explicit acoustic evidence aggregation provides a more controllable interface for diagnosing and improving speech-grounded reasoning. 

---
# Prediction of Prediction (PoP): Inter-Layer Activation Fusion for Single-Pass Hallucination Detection in Large Language Models 

**Authors**: Himal Badu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27165)  

**Abstract**: Autoregressive large language models (LLMs) routinely generate factually incorrect outputs with high decoding confidence, limiting their deployment in high-stakes workflows. Existing output-stage uncertainty metrics can fail when models are overconfident on false assertions, while multi-sample verification pipelines introduce substantial memory and latency overhead. This work evaluates whether internal hidden-state transition dynamics during generation can signal factual errors without auxiliary decoding calls. We introduce Prediction of Prediction (PoP), a mechanism that captures layer-transition uncertainty by fusing intermediate hidden representations across depth during a single forward pass. Evaluated on the TruthfulQA benchmark using autoregressive transformer backbones, PoP achieves an area under the receiver operating characteristic curve (AUROC) of 75.5% for factual-correctness classification. The mechanism operates within the base forward pass, adding less than 1.2% runtime latency and requiring zero additional generation passes. The numerical results are reported from the author-verified experimental implementation and are bounded by the evaluation scope described below. 

---
# STAR : Sentence Translation Alignment Rate for Document-to-Document Machine Translation 

**Authors**: Yichen Dong, Hao Wang, Junhui Li, Linlong Xu, Longyue Wang, Weihua Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.27161)  

**Abstract**: Large Language Models (LLMs) have enabled a shift from sentence-level to document-to-document (Doc2Doc) machine translation, promising improved global coherence. However, document-to-document generation in a single pass frequently suffers from structural misalignment, manifesting as sentence omissions or hallucinations that violate the core requirement of source-target correspondence. To address this, we introduce Sentence Translation Alignment Rate (STAR), an auxiliary metric that explicitly quantifies sentence-level structural fidelity. Building on this, we propose STAR-masked Preference Optimization (StarPO), a framework that ranks document-level hypotheses by structural quality and utilizes a dynamic alignment mask to focus optimization on misaligned segments. Experimental results across news and literary domains demonstrate that StarPO significantly enhances translation quality and structural integrity. Notably, StarPO allows compact models to surpass the performance of massive proprietary systems like GPT-4o while maintaining superior token efficiency. 

---
# Said Aloud, Read Different: Cross-Modal Instability in Multimodal Models 

**Authors**: Basel Mousi, Fahim Dalvi, Shammur Chowdhury, Firoj Alam, Nadir Durrani  

**Link**: [PDF](https://arxiv.org/pdf/2608.27135)  

**Abstract**: Multimodal foundation models are increasingly used in speech-first assistants that must interpret spoken queries and produce visually grounded decisions. Yet it remains unclear whether semantically equivalent queries yield consistent judgments across modality (text vs. speech) and language (English vs. Arabic). We introduce a speech-augmented visually grounded contrastive triplet benchmark spanning 10,150 culturally grounded images from 18 MENA countries, where each image is paired with one supported statement and two plausible but unsupported alternatives. We define contrastive instability as the conditional rate at which a model fails to resolve all statements within a triplet, isolating fragmented reasoning from complete failure. Evaluating recent multimodal models under text and speech in English and Arabic, we find that modality and language shifts introduce substantial triplet-level inconsistencies that are not fully captured by aggregate accuracy, with speech amplifying partial failures. We make the benchmark publicly available to the community. 

---
# TwinKV: A Composable Repair Pass for KV Cache Eviction via Pairwise Key Redundancy 

**Authors**: Hong Chen, Yudong Zeng, Yongwei Huang, Zuhao Ouyang, Junyan Zhang, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27128)  

**Abstract**: Long-context inference is bottlenecked by the memory footprint of the key-value (KV) cache, especially for small models under tight resource budgets. Existing KV cache eviction methods score tokens using the model's attention distribution or, in attention-free variants, each key's distance from a global reference point. Using a controlled leave-one-out probe, we find that attention magnitude is unrelated to a token's causal contribution to the answer (Spearman $\rho=-0.004$), challenging the premise behind dominant eviction methods. We introduce TwinKV, a training-free, attention-free redundancy signal that detects whether a token's key has a near-duplicate elsewhere in context. Rather than replacing existing policies, TwinKV acts as a composable repair pass: given a policy's fixed retained set, it identifies evicted tokens with no surviving duplicate (\emph{orphans}) and retained tokens whose information is duplicated elsewhere (\emph{redundant donors}), then swaps them while preserving the original budget and scoring rule. We compose TwinKV with four recent eviction policies across LongBench, LooGLE, RULER, and a short-context MMLU-Pro no-harm control at compression ratios ${0.3,0.5,0.7}$. On Qwen3-4B, TwinKV improves a majority of configurations for two policies, is near-even for a third, and helps only a minority for a fourth adaptive baseline already near a performance ceiling; gains across the three non-ceiling policies are smallest at the loosest ratio. On RULER with Llama-3.2-1B, however, that fourth policy improves in every evaluated cell because its Alone score leaves substantial room to improve. More broadly, Llama-3.2-1B shows a smaller average LongBench gain but a higher fraction of improved cells on LongBench and LooGLE than Qwen3-4B, plus a clean RULER win. We also identify few-shot classification exemplars as a task structure where TwinKV does not help on either model. 

---
# Cross-Lingual Alignment Without Joint Training: Do Monolingual Language Models Converge on Universal Representations? 

**Authors**: Ej Zhou, Suchir Salhan, Catherine Arnett, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2608.27115)  

**Abstract**: Cross-lingual alignment in multilingual language models is typically attributed to joint training: shared parameters, mixed-language batches, or explicit alignment objectives. We ask whether monolingual models trained on non-parallel data learn alignable representations without joint training. By testing on strictly monolingual language models, such as the Goldfish model families and independently developed models from different research labs, we find three results. Correlation: these models develop alignable representational geometry across layers, with alignment strengthening as data scale, model scale, or linguistic proximity increases. Construction: a single Procrustes rotation fit on parallel sentences maps hidden states between models. Causation: the same rotation transfers functional content; patching a rotated English residual into a German model on a factual cloze flips the prediction to the donor's capital in most cases. We confirm that cross-lingual alignment can emerge from the structure of language and the information it carries rather than from joint training, and this points to practical future directions including model stitching, merging, and modular multilingual systems built from monolingual components. 

---
# DocTalkBN: A Novel Dataset of Expert Telemedicine Conversations in Bengali 

**Authors**: Anik Saha, Fahmida Sultana Naznin, Sadatul Islam Sadi, Ananya Shahrin Promi, Wahid Al Azad Navid, Rifat Shahriyar  

**Link**: [PDF](https://arxiv.org/pdf/2608.27110)  

**Abstract**: Reliable medical conversational AI requires authentic expert--patient interaction data, yet such datasets remain scarce, especially for low-resource languages such as Bengali. We present DocTalkBN, a large-scale multimodal dataset of real-world expert telemedicine conversations in Bengali, collected from nationally broadcast telemedicine programs featuring board-certified physicians. DocTalkBN contains 557.63 hours of paired audio and text, 1,515 multi-turn patient calls, 10,274 host--doctor question--answer exchanges, totaling 1.7M tokens, spanning 26 medical specialties. Unlike prior resources derived from medical forums, written health content, or synthetic data, our dataset preserves the spontaneity, contextual richness, and spoken characteristics of authentic medical interactions in a low-resource setting. To support benchmark-driven research, we further construct three downstream tasks from the corpus, medical triage classification, advice safety evaluation, and medical named entity recognition, and benchmark a diverse set of large language models and encoder-based baselines. Our results show that DocTalkBN is a practically useful resource, particularly for clinically grounded reasoning tasks. We release this resource to facilitate future research on reliable medical NLP and safer, more culturally grounded healthcare systems for low-resource languages. Our source codes and dataset are publicly available at this https URL. 

---
# Research Design Tracking and Assessment for the Social Sciences 

**Authors**: Marco Rovera, Sergiu Burlacu, Dominique Cappelletti, Alessio Tomelleri, Sonia Marzadro, Martina Bazzoli, Annalisa Tassi, Jessica Gagete-Miranda  

**Link**: [PDF](https://arxiv.org/pdf/2608.27049)  

**Abstract**: Reliable assessment of causal research designs in the social sciences is critical for evidence-based policy-making, yet has so far relied entirely on manual expert analysis. We introduce Automated Research Design Tracking and Assessment (ARDTrA), a task that involves detecting the research design used in a paper and assessing the quality of its application. We create an expert-annotated dataset of papers covering six families of counterfactual research designs and evaluate the task using a multi-turn RAG-based conversational pipeline. Across four retrieval strategies, four LLMs and six embedding models, we find that passage length is the main driver of performance, explaining 52-66% of the variance. A per-research-design analysis also shows that human and machine difficulty do not align: the designs that prove hardest for the system are not those on which expert annotators disagree most, pointing to two independent sources of task difficulty. 

---
# Cascaded Batch Prompting 

**Authors**: Sho Hoshino, Peinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.27038)  

**Abstract**: Although batch prompting makes large language model inference more efficient by processing multiple instances simultaneously, it suffers from unpredictable downstream task performance. We propose cascaded batch prompting, a two-stage approach designed to resolve the unpredictability of conventional batch prompting by disentangling complex reasoning from symbol grounding. Experiments on multiple-choice question answering and natural language inference demonstrate that the proposed method outperforms the standard single prompting baseline while achieving a speedup proportional to batch size, establishing a new state of the art on the Pareto frontier. 

---
# Reasoning about In-Context Samples for Machine-Translation 

**Authors**: Maxime Bouthors, Josep Crego, François Yvon  

**Link**: [PDF](https://arxiv.org/pdf/2608.27036)  

**Abstract**: Large Language Models (LLMs) can be trained to perform chain-of-thoughts reasoning in order to improve the reliability of their responses. In this work, we investigate how explicit reasoning can be leveraged for LLM-Based Machine Translation (MT) with in-context samples. We introduce a novel fragment-based reasoning framework in which the model first extracts parallel source-target fragments from retrieved similar exemplars, and uses these fragments as intermediate reasoning traces to produce the final translation. To train our model, we distill silver fragments and drafts from a large teacher model. Our experiments with the Qwen3 model family, over 6 languages, including up to 5 domains per language, demonstrate that fragment-based MT significantly outperforms alternative methods like standard k-shot or basic drafting. 

---
# Representing and Parsing Korean Constituency Structure at Different Levels of Granularity 

**Authors**: Jungyeul Park, KyungTae Lim, Zihao Huang, Eunkyul Leah Jo, Yige Chen, Chulwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2608.27035)  

**Abstract**: Korean constituency parsing raises a representational challenge because the terminal units of a phrase-structure tree do not straightforwardly correspond to simple surface words. Korean eojeols are morphologically complex spacing units, and existing constituency resources differ in how they represent eojeol-internal morphology and non-overt elements. This paper compares three constituency parsing representations derived from the Penn Korean Treebank: Morpheme+XPOS, Eojeol+XPOS, and Eojeol+UPOS. We construct these representations by removing null elements, aligning Penn Korean phrase structure with overt eojeol tokens, preserving Penn Korean phrase labels where possible, and varying the terminal and preterminal layers. We then evaluate canonical non-binary transition-based constituency parsers in top-down, in-order, and bottom-up orders under a shared modeling and evaluation setup. All experiments use gold terminal segmentation and gold preterminal labels and therefore evaluate constituency parsing conditioned on gold morphosyntactic annotation. Eojeol terminals yield shorter transition sequences, but Eojeol+UPOS parsing substantially underperforms the morphologically richer conditions. Eojeol+XPOS narrows this gap, while Morpheme+XPOS gives the strongest results even after its predictions are projected to the eojeol terminal domain. Under these gold-annotation conditions, the results show that fine-grained morphological and XPOS representations provide valuable evidence for the evaluated parsers. This empirical finding concerns the information available for parsing and does not by itself determine the linguistically preferable terminal domain. Independently, linguistic and resource-design considerations motivate eojeol as a stable and interpretable surface domain for phrase-structure annotation, with morpheme-level and XPOS information retained as aligned morphosyntactic evidence. 

---
# ITL: Interpretable Document Alignment with Structured Reference Frameworks 

**Authors**: Raúl Giráldez, Dayrelis Mena, Jesús S. Aguilar--Ruiz  

**Link**: [PDF](https://arxiv.org/pdf/2608.27031)  

**Abstract**: Measuring alignment between documents and structured reference frameworks requires identifying conceptual evidence distributed throughout the text and reporting it through measures that are quantitative, interpretable, and traceable. Many commonly used retrieval and classification approaches return either pairwise similarity scores or one or more class labels, whereas fewer methods provide concept-level scores that are directly traceable to the terminological evidence supporting them. We present \emph{Intelligent Target Locator} (ITL), a domain-agnostic and language-portable methodology that estimates the affinity between the textual units of a target document and the concepts defined in a \emph{Structured Reference Document} ($SRD$). From the $SRD$, ITL induces concept-specific terminological profiles built from independent terms, bigrams, trigrams, and co-occurrences. Each term is assigned an importance weight that combines concept membership, term-type specificity and inter-concept discriminability. The output is a textual-unit--concept affinity matrix that can be aggregated at different levels of granularity. We conduct an internal consistency assessment using the 17 Sustainable Development Goals (SDGs), evaluating each official goal statement against the $SRD$ induced from the same set of descriptors. Every statement reached its highest affinity with the corresponding concept, and the mean affinity across the remaining concepts stayed marginal relative to the mean reference affinity. This separation indicates that ITL distinguishes the conceptual profiles of the framework. ITL thus offers a general basis for quantifying document alignment with structured frameworks while keeping each result traceable to the terminological evidence that supports it. 

---
# JudgeStealer: Extracting LLM Judging Capabilities across Evaluation Protocols 

**Authors**: Chen Chen, Yaolin Chen, Xuehan Sun, Juan Lin, Xueluan Gong, Yuhang Zheng, Qian Wang, Kwok-Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2608.26982)  

**Abstract**: Large language model (LLM) judges are increasingly used across various evaluation scenarios, making their judgment capabilities valuable intellectual property. However, black-box access exposes these capabilities to model extraction attacks. Existing extraction methods do not specifically target LLM judges and provide limited support for multiple evaluation protocols under restricted query budgets. In this study, we propose JUDGESTEALER, the first query-efficient model extraction framework for replicating judging capabilities across pointwise scoring, pairwise comparison, and listwise ranking protocols. JUDGESTEALER exploits the strong cross-protocol agreement to acquire pointwise scores and transform them into pairwise and listwise supervisions without additional victim queries. To capture informative judge patterns and improve query efficiency, JUDGESTEALER dynamically selects pointwise inputs based on semantic diversity, predictive uncertainty, and potential judge biases. It further applies score smoothing and multi-protocol review to preserve the ordinal structure of scores and mitigate catastrophic forgetting during surrogate adaptation. Extensive experiments on state-of-the-art LLM-as-a-judge and reward models show that JUDGESTEALER consistently outperforms existing extraction baselines, achieving up to 73.3%, 87.0%, and 71.6% accuracy for pointwise, pairwise, and listwise evaluation, respectively. JUDGESTEALER also remains effective across different sur- rogate model scales, adaptation strategies, and reasoning settings. Moreover, JUDGESTEALER demonstrates robustness against representative extraction defenses. 

---
# Squeezing More from Limited Data with Recursive Transformers 

**Authors**: Serdar Gülbahar, Lukas Edman, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2608.26973)  

**Abstract**: Pre-training under limited data requires a different view of scaling than web-scale language modeling. With a fixed data budget but relatively abundant compute, increasing parameter count helps only up to an optimal scale; beyond that point, models overfit and generalization worsens. We study this behavior across 10M-100M word pre-training budgets, two corpora, and multiple downstream evaluations, and find that optimal size depends strongly on both the data budget and the downstream target. We argue that standard Transformers scale down poorly to this setting, because embeddings consume a large fraction of the parameter budget and per-token computation is tied to representational capacity. To address this coupling, we study recursive Transformers, reusing a shared block across depth to scale compute, together with factorized embeddings to reduce vocabulary-map parameters. We train three recursive models and find that they outperform standard Transformers at 10M and 100M words, while remaining competitive with BabyLM Challenge 2025 winners. 

---
# KinyaEmbed: Contrastive Sentence Embeddings for Kinyarwanda via Multi-Stage Curriculum Training 

**Authors**: Ireddi Rakshitha, Devavarapu Yashwanth, Ntakirutimana Pierre  

**Link**: [PDF](https://arxiv.org/pdf/2608.26941)  

**Abstract**: We present KinyaEmbed, the first dedicated sentence embedding model for Kinyarwanda, a morphologically rich Bantu language spoken by over 12 million people in Rwanda. Existing multilingual embedding models such as LaBSE, mE5-large, and OpenAI text-embedding-3-large perform poorly on Kinyarwanda due to severe under-representation in their pre-training corpora. KinyaEmbed is built on KinyaBERT-large and trained via a four-stage curriculum using MultipleNegativesRankingLoss (MNRL): Stage 1 leverages ~18,000 paraphrase pairs from the Official Gazette of Rwanda with three temperature scales; Stage 2 fine-tunes on 715 NLLB-translated MNLI triplets for entailment structure; Stage 3 aligns representations using English-Kinyarwanda OPUS-100 translation pairs; Stage 4 refines with 2,936 high-quality pairs filtered from KinyaCOMET at quality threshold 0.8. We evaluate on SemRel2024-rw and introduce Wiki-RW-STS, a new contamination-free Kinyarwanda STS benchmark of 300 pairs derived from Kinyarwanda Wikipedia. A seven-checkpoint ensemble (all5+23A*2, with the final stage double-weighted) achieves Spearman \r{ho}=0.7298 on SemRel2024-rw, surpassing mE5-large by 20.9% and OpenAI text-embedding-3-large by 41.0%. KinyaEmbed also achieves the best document clustering silhouette score (0.2146) across all evaluated models. All checkpoints, the KinyaCOMET filtered pairs, and the Wiki-RW-STS benchmark are publicly available. 

---
# Mapping Written Words to Spoken Words in a Different Language Using Only Visual Grounding 

**Authors**: Gabriel Pirlogeanu, Dan Oneata, Horia Cucu, Herman Kamper  

**Link**: [PDF](https://arxiv.org/pdf/2608.26925)  

**Abstract**: In many low-resource settings, even just eliciting speech for data collection is difficult. One promising approach has been to ask speakers to describe images. But how do we build models from such visually grounded speech data? Given a dataset of images with Hindi spoken captions, we consider how we can map a written English keyword to spoken realisations of that word in Hindi. Previous work trained end-to-end multimodal neural models. Instead, we explore a simpler alignment-based approach built on self-supervised speech representations. Written English tags are automatically obtained from images using off-the-shelf image captioning systems. Hindi utterances associated with the same keyword are then aligned (using self-supervised features), and alignment evidence is aggregated to identify recurring speech segments corresponding to the target word. Experiments evaluating keyword spotting and localization show that our alignment-based approach outperforms a previous attention-based neural model. We also show the benefit of incorporating negative examples during alignment. Our work demonstrates that cross-lingual word-to-speech mappings can be learned directly from visual grounding without transcriptions or explicit model training. 

---
# TabuLM: Morphology-Aware Tabular Pre-training for Low-Resource Languages 

**Authors**: Ireddi Rakshitha, Devavarapu Yashwanth, Ntakirutimana Pierre  

**Link**: [PDF](https://arxiv.org/pdf/2608.26923)  

**Abstract**: We present TabuLM, the first language model pre-trained on Kinyarwanda tabular data. Kinyarwanda is a morphologically rich Bantu language spoken by over 12 million people in Rwanda, yet lacks any dedicated tabular representation learning resource. TabuLM extends KinyaBERT-large, a two-tier morphological transformer, with additive row, column, and cell-type embeddings and a learned table-structure attention bias that sharpens same-row and same-column attention. Pre-training uses two new objectives: Masked Cell Recovery (MCR), which masks entire cells and forces reconstruction from row and column context, and Column Type Prediction (CTP), which predicts column semantic types from observed cell values. We pre-train on 172 Rwandan government tables (~35,000 cells) from NISR, RAB, REB, and MoH open-data portals, and introduce TabQA-kin, the first native Kinyarwanda table question-answering benchmark comprising 526 QA pairs across 31 tables and four question types. TabuLM achieves 62.0% exact match on TabQA-kin, outperforming KinyaBERT-large by 5.7 EM points and all multilingual baselines (mBERT 49.3%, XLM-R 50.0%) by 11.7-12.7 points. Analysis shows that structural table embeddings are most decisive for comparison and lookup questions, while morphological awareness provides complementary gains. Our code, data, and pre-trained checkpoint are publicly available. 

---
# Planting a Latent Variable in Natural-Looking Text: a More Realistic Test of Belief States in LLMs and Their Link to Concept Geometry 

**Authors**: Alexandru-Iulius Jerpelea  

**Link**: [PDF](https://arxiv.org/pdf/2608.26887)  

**Abstract**: LLMs are thought to track "belief states," i.e., running probability distributions over the latent variables that govern language (Shai et al., 2024; Sarfati et al., 2026), but so far this has only been comprehensively demonstrated on toy synthetic data and in a few isolated case studies. It has also never been empirically connected to the geometry of LLM features (the concepts interpretability finds in model activations). In this work, we plant a controllable latent variable inside natural-looking text. An LLM teacher writes ordinary text while we "subliminally" steer it along one of K = 8 unrelated sparse autoencoder directions at each token, with the active directions following a ring-shaped Markov chain. A small transformer model trained on this corpus does indeed track the Bayesian posterior belief about our planted latent variable. Moreover, it also arranges the 8 states themselves on a ring, in the exact order of the Markov chain, which is supporting evidence that a concept's geometry can be formed by the statistical dynamics of the latent variable behind it. 

---
# Evaluating Confidence-Gated Retrieval with Matched Trajectory Replay 

**Authors**: Prateek Chhikara  

**Link**: [PDF](https://arxiv.org/pdf/2608.26846)  

**Abstract**: Interactive language-model agents use confidence signals to decide whether to answer immediately, retrieve additional evidence (from memory or external knowledge), or defer. Yet confidence is usually evaluated in isolation, without measuring the trajectory-level consequences of the actions it triggers. We propose matched trajectory replay, a controlled protocol for comparing confidence-to-action mappings. The protocol holds candidate answer states, evidence points, budgets, and action costs fixed. We use it to compare raw verbalized confidence with post-hoc isotonic calibration in a multi-hop question-answering system using Mistral, GPT, and Qwen models on HotpotQA and MuSiQue datasets. At the same numerical commitment threshold, calibration changes which questions agents ultimately commit to answering. Across all six model-dataset pairs, it increases accuracy among committed answers by up to 41 percentage points. However, it can reduce coverage and increase retrieval use. Overall accuracy improves by up to 15 percentage points on HotpotQA but falls by up to 17 percentage points on MuSiQue. These effects reflect a shift to a more selective, lower-risk operating point, not improved answers or confidence ranking. A calibration map fitted before retrieval improves held-out calibration through retrieval depths one and two, but is worse than raw confidence at depth three for all three models. Additional evidence helps on average, but this aggregate effect does not establish whether confidence identifies which individual episodes will benefit from another retrieval. Taken together, these results show that calibration can make commitment risk interpretable, but it does not estimate the expected benefit of another retrieval. Retrieval therefore requires a separate value-of-information or utility estimate. Evaluations should report held-out calibration, risk-coverage, and retrieval cost. 

---
# RuleWeaver: Benchmarking Rule-Centered Scenario Reasoning for Large Language Models 

**Authors**: Bohan Yu, Shi-Yang Li, Pengfei Cao, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.26832)  

**Abstract**: Large language models (LLMs) are increasingly applied to specialized domains, where effective use of domain expertise often requires reasoning over complex rules in concrete scenarios. However, existing benchmarks only partially evaluate this capability, as they either focus on output-level instruction constraints or overlook the distinct roles that rules play in scenario reasoning. To address these gaps, this paper introduces RuleWeaver, a benchmark construction framework for evaluating rule-centered scenario reasoning. RuleWeaver starts from corpus-derived IF-THEN Meta Rules, progressively augments them into complex rules, and composes these rules into rule-centered scenario QA instances. Beyond final-answer correctness, RuleWeaver further supports process-level evaluation through rubric-based answer quality, rule recall, and rule precision. Experiments on 11 representative LLMs show that current models still struggle with complex rule-centered scenario reasoning, with even the best-performing model achieving only around 50% of the maximum rubric score. We make our code and dataset available here: this https URL. 

---
# Behavior2Trip: Towards Personalized Travel Planning via User Behavior Trajectory 

**Authors**: Zihao Cheng, Yingyu Shan, Hongru Wang, Zeming Liu, Xinyi Wang, Xiangrong Zhu, Yuhang Guo, Wei Lin, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.26807)  

**Abstract**: Travel planning agents assist users in generating personalized travel plans by modeling their individual preferences. Existing agents either rely on explicit user instructions or engage in multi-turn clarification to elicit user preferences. However, both approaches overlook the rich behavioral signals latent in users' past behaviors, which implicitly encode their preferences. This over-reliance on active user input increases interaction burden and limits plan personalization. To bridge this gap, we introduce a new task, Behavior-Aware Travel Planning, which infers user preferences directly from past behaviors and generates personalized travel plans. To facilitate research on this task, we introduce Behavior2Trip, a benchmark constructed from one of the largest Chinese online travel platforms, comprising 11,400 instances. Each instance represents an average of 39.8 past user behaviors spanning 14 attributes across 5 preference dimensions. We further propose B2T-Agent, a reinforcement learning-based agent that leverages user behavior trajectories, interacts with external tools for preference-aligned retrieval, and maintains an internal memory module. Experiments on Behavior2Trip show that GPT-4.1 achieves a full-constraint pass rate of only 0.5\% on the hardest tasks, while B2T-Agent built upon Qwen3-8B outperforms all baselines, highlighting the substantial challenge of this task. Moreover, Qwen3-8B trained with B2T-Agent also outperforms GPT-4.1 on the TravelPlanner benchmark, demonstrating strong generalization. Code and data are available at this https URL 

---
# Instruction Quality Matters: Refining Instructions for Effective Preference Learning 

**Authors**: Seohyeong Lee, Hwaran Lee, Buru Chang  

**Link**: [PDF](https://arxiv.org/pdf/2608.26779)  

**Abstract**: Preference learning optimizes models using response pairs, yet the informativeness of these pairs is fundamentally shaped by the instructions from which they are generated. We identify instruction quality as a hidden bottleneck in preference learning: low-quality or ambiguous instructions restrict the response-quality distribution, limiting strong chosen responses and weakening preference signals. Through Best- and Worst-of-N analyses, we show that instruction quality constrains both the ceiling and floor of sampled response quality. Motivated by this observation, we introduce an instruction-refinement pipeline that selects weak instructions using reward signals and revises them with rubric-guided LLM feedback, improving preference data without discarding examples. Across offline and online preference learning settings, experiments on multiple models and benchmarks show broad alignment improvements over original data and alternative data-improvement strategies. Further analyses indicate that instruction refinement raises achievable response quality and complements response-centric preference data curation. Overall, instruction quality emerges as a key factor governing how informative preference signals are formed for LLM alignment. Code is available at: this https URL 

---
# Equal Ranking Quality, Different Decisions: Training Order-Consistent LLM Scorers 

**Authors**: Markus Frohmann, Mahdiyar Alavi, Elizabeth Lingg, Navid Rekabsaz  

**Link**: [PDF](https://arxiv.org/pdf/2608.26762)  

**Abstract**: Rerankers, reward models and multi-document QA scorers score candidate documents or responses in one LLM prompt, so each score depends on their order. Such scorers are selected on ranking quality, but their scores determine a decision: what a score threshold retains, a reader answers, or a preference model selects. However, equal ranking quality does not imply equal decisions: on passage reranking, five trained scorers within 0.010 nDCG@10 retain sets that overlap by only 0.66-0.84 when reordered. A published reranker takes the highest retained-set F1 in our comparison and still overlaps by only 0.667. No prompt-time change we test removes that order dependence: the only one that gains ranking quality leaves all three decisions unchanged. Order-consistency SFT (OC-SFT) attenuates it in the weights, training a candidate's score not to depend on the order. It holds ranking quality and leads every decision-stability measure among trained scorers on all three tasks: it flips the reader's answer on 0.125 of permutation pairs against 0.149-0.164 for three other objectives that target order. It is more stable than order-averaged distillation on 12 base models, and one OC-SFT permutation retains sets that overlap more than ten averaged off-the-shelf permutations. A comparison should therefore report what a threshold retains and a reader answers, not ranking quality alone. Code is available at this https URL. 

---
# Letters hide the truth from our eyes: English homophones have meaningfully different phonetic realizations 

**Authors**: Yu-Hsiang Tseng, Mirjam T. C. Ernestus, Louis F. M. ten Bosch, R. Harald Baayen  

**Link**: [PDF](https://arxiv.org/pdf/2608.26749)  

**Abstract**: The distribution of spoken word duration of English homophones is known to co-vary with frequency of use. This study investigates whether other aspects of the phonetic realization of homophones also differ. A series of quantitative investigations of 14,000 homophone tokens in American television news broadcasts revealed that the tokens of homophone pairs such as \textit{weight} and \textit{wait} have different phonetic realizations, and that these can be predicted from their meanings in utterance context. These systematic differences remain even when taking duration-related variation into account. Time-normalized spectrograms emerged as an excellent tool for probing the fine details of phonetic realization, and obviate the need for phonetic transcriptions, which inevitably hide the phonetic truth from our eyes. 

---
# Preserving General Capabilities during Domain Specialization with Uncertainty-Calibrated MOPD 

**Authors**: Ziyuan Liu, Jiao Ou, Jian Liang, Ruiming Tang, Cheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.26735)  

**Abstract**: Specializing large language models to vertical domains improves domain-specific behavior but often degrades general capabilities such as reasoning, coding, instruction following, and creative writing. We study this domain--general trade-off in Multi-Teacher On-Policy Distillation (MOPD), where a specialized student is supervised on its own sampled trajectories by domain and general teachers. Standard MOPD faces two limitations: ordinary on-policy sampling rarely exposes tokens with large positive teacher--student advantages, while the advantage sign alone does not establish whether the resulting update direction is reliable. We propose uncertainty-calibrated MOPD to address these limitations. Dual-temperature sampling broadens the candidate trajectory pool, and positive-advantage-density filtering selects trajectories with stronger positive learning signals. Centered log-likelihood (CLL) filtering then computes an entropy-calibrated teacher-endorsement score and probabilistically retains token updates according to direction--endorsement consistency. Experiments on role-playing and medical-domain specialization show that our method improves the general-capability average over standard MOPD by $4.73\%$ and $10.84\%$, respectively, while maintaining vertical-domain performance. Ablations and diagnostic analyses further confirm that the gains do not merely result from a larger rollout budget and that the proposed trajectory- and token-level mechanisms address their intended failure modes. 

---
# Towards Expert Financial QA via Self-Improving RAG 

**Authors**: Junjie Xiong, Shawheen Ghezavat, Aum Hirpara  

**Link**: [PDF](https://arxiv.org/pdf/2608.26706)  

**Abstract**: Expert-level financial question answering requires both grounded verification to catch numeric hallucinations and audit trails for regulatory compliance, attributes that standard single-pass RAG systems lack. We take a step toward this goal with Self-Improving RAG, a framework that decomposes document QA into three specialized agents (Retrieval, Reasoning, and Judge) coordinated by an orchestrator with feedback-driven self-correction. When the Judge Agent scores an answer below a dynamic threshold, the system triggers retry with escalated strategies: broader retrieval, more careful prompting, and relaxed acceptance criteria. We evaluate on FinanceBench (SEC filing QA), where Self-Improving RAG achieves 86% oracle-guided accuracy (measuring agreement with gold answers) with a 36.4% Lazarus Rate, recovering nearly 4 in 10 initially incorrect answers through targeted retry. A key finding is that a fixed retrieval pipeline with judge-driven retry achieves strong results without dynamic routing, providing full interpretability. Every decision is logged with confidence scores, enabling the audit trails required for regulated financial applications. 

---
# PragAlign: Evidence-Sensitive Reply Assistance Across Chinese and Japanese Appropriateness Judgments 

**Authors**: Xin Zhong, Satori Hachisuka  

**Link**: [PDF](https://arxiv.org/pdf/2608.26700)  

**Abstract**: Reply assistance in multilingual settings requires linguistic competence and culturally situated judgments of appropriateness. We present PragAlign, which separates context reading from selective clarification, and evaluate it alongside Direct and Rule. Nine native Chinese speakers judged Chinese materials, while three native Japanese speakers judged matched Japanese versions. In the Chinese evaluation, PragAlign received significantly better ranks than both baselines. In the Japanese evaluation, Direct had the lowest mean rank, PragAlign had the highest top-rank rate, and the omnibus difference was not significant. The groups selected the same top condition in 5 of 10 scenarios, including four shared PragAlign selections. The results identify shared and language-specific judgment patterns and inform reply assistance designed to support linguistic and cultural understanding. 

---
# Scaling phoneme-based TTS augmentation for ASR: A unified pipeline and controlled study 

**Authors**: Zhen Wang, TianRui Wu, RongQi Han, Hao Wu, Wei Liang  

**Link**: [PDF](https://arxiv.org/pdf/2608.26697)  

**Abstract**: Synthetic speech provides scalable supervision for automatic speech recognition (ASR), but its benefit depends on the selected texts, reference speech, and amount of synthesized data. We present a unified phoneme-based TTS-to-ASR augmentation pipeline built around a multilingual TTS model trained from scratch using the F5-TTS architecture with language-ID conditioning. The pipeline combines language-specific grapheme-to-phoneme conversion, reference-speech filtering, candidate-text selection, synthesis, and matched ASR continuation. We further propose phoneme-frequency-guided selection (PFGS), which ranks candidate sentences using phoneme frequencies estimated from real ASR training labels. Experiments with separate monolingual ASR systems for Arabic, French, Italian, and Portuguese span 13 test sets. Across the synthesis-scale sweep, random augmentation improves over matched real-only continuation on 11 test sets. Under a nominal 60% synthesis budget, PFGS improves over real-only training on 12 test sets and over random selection on 9. Its largest relative word error rate (WER) reduction against random selection is 19.3%. With target texts and synthesis counts fixed, reference-speech filtering reduces absolute WER by 0.29 and 0.59 points on Italian and French Common Voice, respectively. These results identify synthesis scale, candidate-text content, and reference quality as important control variables in TTS-based ASR augmentation. 

---
# Beyond Reflection: Affirmation as a Promising Behavioral Marker Associated with Quality in Text-Based Counseling 

**Authors**: Michimasa Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2608.26689)  

**Abstract**: While AI-assisted text-based counseling is gaining attention, it remains empirically unclear which counselor behaviors are associated with higher dialogue quality. Existing research often focuses heavily on Reflection, borrowing frameworks from Motivational Interviewing. To address this gap, we conduct a multi-layered analysis using KokoroChat, a large-scale Japanese text counseling dataset conducted by professional counselors and trainees, newly annotated with counselor strategy tags and client distress levels. Our results show that, under the quality indicators used in this study, Affirmation is more consistently associated with session quality than Reflection among the analyzed strategies. Cross-dataset transfer experiments further suggest that this quality signal can be observed to some extent on ESConv, an English dataset with non-expert supporters. These findings provide empirical implications for counselor training and emotional support system design. We release the additional KokoroChat annotations and experimental source code at this https URL. 

---
# FOCUS & RePAIR: Mitigating Text Degeneration via Token-Level Guidance for Pruned Large Language Models 

**Authors**: Junyoung Lee, Sehyeon Park, Shinhyoung Jang, Seonha Ryu, Hojeong Kim, Hyunsei Lee, Il Hong Suh, Yeseong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.26676)  

**Abstract**: Pruning is a practical approach to compress large language models (LLMs), but it can amplify text degeneration, especially repetition loops, even when perplexity and task accuracy remain largely unchanged. In this work, we present a token-level analysis of this failure mode by viewing decoding as a dynamical process that enters and persists in a small set of recurrent contexts. Our analysis decomposes degeneration into loop entry risk and loop persistence, and shows that persistence is controlled by the escape mass assigned to plausible alternatives within the token sampling set. Motivated by these findings, we propose two token-level guidance objectives for post-pruning fine-tuning. FOCUS reweights distillation toward high-confidence teacher regions to suppress leakage, while RePAIR uses onset-centered positive/negative continuation pairs with a margin loss to promote plausible alternatives and prevent early commitment to repetition loops. Experiments on open-ended continuation and instruction-based generation show that both methods consistently reduce repetition and improve generation quality. 

---
# Do LLMs Understand Personality? Rethinking Persona Fidelity Evaluation through Structured Behavioral Inference 

**Authors**: Mengfan Li, Zesheng Wei, Xuanhua Shi, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2608.26674)  

**Abstract**: As large language models are increasingly deployed to simulate diverse human characters, ensuring persona fidelity, defined as the extent to which an agent's behavior consistently reflects the psychological and stylistic characteristics of a target persona, has become a critical requirement. However, existing evaluation paradigms primarily rely on either holistic LLM-based judges, which are prone to "holistic appraisal hallucination'', or static psychometric inventories, which fail to capture the context-dependent fidelity required in dynamic dialogue. To address these limitations, we propose PRISM (Persona Reasoning with Inverse SFL-based Modeling), a psycholinguistically grounded framework that reformulates persona fidelity evaluation as a structured inverse inference task. Inspired by Systemic Functional Linguistics (SFL), PRISM decomposes persona fidelity into three functional dimensions: Task Framing, Interpersonal Stance, and Linguistic Style. It estimates dimension-specific evidence over a persona-conditioned label space and aggregates these signals into an interpretable and auditable evaluation process. Experiments show that PRISM yields more accurate and stable judgements than traditional holistic judging, providing a more reliable framework for persona fidelity evaluation. 

---
# Meta-Learning Where to Allocate Experts: Task-Conditioned Layer-Wise Compression for MoEs 

**Authors**: Rongfeng Wang, Shichao Weng, Zhiqiang Wang, Xinyu Liu, Yang Yi, Peilong Zhou, Hongwei Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.26650)  

**Abstract**: Mixture-of-Experts (MoE) models route each token to a subset of expert networks, increasing capacity while keeping per-token computation sparse. In many deployed MoEs, the number of active experts is fixed across layers and tasks, although layer roles and expert redundancy vary with depth and demand varies with difficulty. Existing approaches address only part of this setting: layer-wise allocations are usually determined offline and reused for all tasks, while token-level methods vary expert activation using local routing signals without task-level context. We propose MetaNet, a support-set controller that predicts, for each layer, an expert-retention threshold and a bounded routing bias. The backbone, experts, and router remain frozen. On DeepSeek-MoE-16B-Chat, MetaNet provides a tunable accuracy-expert-activation trade-off. Relative to fixed k=6, a conservative setting activates 3.61 experts on average (40% fewer) and achieves comparable MMLU accuracy (0.489 vs. 0.474), whereas an aggressive setting activates 2.28 experts on average (62% fewer) with accuracy approximately 3.7 percentage points lower. The MMLU-trained controller also transfers to C-Eval without retraining, activating 2.90 experts on average (52% fewer than fixed k=6) at 0.386 accuracy. 

---
# Information-Guided Frontier Decoding: Contextual Utility-Driven Commitment in dMLLMs 

**Authors**: Xingyou Fang, Jingxing Zhong, Xiaosong Yuan, Xiaofeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.26641)  

**Abstract**: Decoding quality in diffusion multimodal language models (dMLLMs) depends heavily on the order in which masked tokens are committed. Existing confidence-based strategies prioritize locally easy tokens, but confidence does not necessarily reflect contextual usefulness. As a result, structurally easy tokens such as punctuation may be committed before informative semantic anchors, weakening context propagation and increasing error accumulation. We propose Information-Guided Frontier Decoding (IGFD), a training-free decoding strategy that ranks candidates using token confidence, neighborhood uncertainty, and structural commitment risk. IGFD encourages early commitment of reliable semantic anchors while delaying fragile structural tokens, improving contextual support during decoding. A dynamic candidate frontier further constrains token selection to locally expandable regions under the same decoding budget. The method requires no additional training, auxiliary models, or extra forward passes. Experiments across multimodal understanding, reasoning, grounding, and hallucination benchmarks show that IGFD consistently outperforms existing decoding strategies across the majority of benchmarks and diffusion MLLM backbones under identical decoding budgets. 

---
# Which Metrics Save the Most Human Annotation? Prediction-Powered Evaluation and Meta-Evaluation 

**Authors**: Mingqi Gao, Anthony Sicilia, Weiyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2608.26638)  

**Abstract**: Across various non-verifiable tasks, human evaluation is reliable but expensive, while automatic metrics are more scalable but often biased. Building on prediction-powered inference (PPI), we propose prediction-powered evaluation, a framework that combines limited human judgments with large-scale automatic scores to obtain data-efficient system comparisons that are provably unbiased. We develop parametric and non-parametric procedures, analyze the efficiency trade-off between paired and unpaired designs, and validate the framework on six WMT datasets. We further introduce the Prediction-Powered Saving Ratio (PPSR), a meta-metric that measures how much human annotation an automatic metric can save when used within prediction-powered evaluation. PPSR directly targets metric utility for prediction-powered evaluation and yields more discriminative and stable metric rankings than existing system-level meta-metrics. Overall, our new paradigm reframes automatic metrics as tools for reducing human annotation cost rather than replacing human judgment, and applies broadly to non-verifiable tasks. 

---
# Not Just Reason, Not Just Scan: Reinforcement Learning for Proactive Scientific Error Verification over Academic Paper 

**Authors**: Rongjin Li, Yuanxin Liu, Hao Zhou, Fandong Meng, Jie Zhou, Xu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.26596)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly capable scientific assistants, yet they remain far from fully autonomous research. This transition requires models to actively inspect academic papers, build global evidence views, and make traceable judgments without prespecified issues or evidence. However, existing work provides limited task paradigms or training studies for such issue- and evidence-absent verification. We study this challenge through scientific error detection, where models must determine whether errors exist and justify them with evidence-based reasoning. To fill this gap, we present VERA-RL, a reinforcement-learning formulation for scientific error detection over academic papers. Following a Reason--Verify--Scan progression, we construct VERA-13K, a 12,900-sample dataset organized into 4,300 matched chains, covering 6 scientific-error categories across the research workflow and broad natural-science domains. We further introduce fine-grained rewards for reasoning completeness, evidence alignment, and error precision. Training Qwen3-VL-8B with VERA-RL substantially improves verifiable reasoning, approaching flagship MLLMs such as Gemini 3 Pro and Qwen3-VL-235B-A22B on Scan. 

---
# Benchmarking Clinical Decision Pathway Adherence in Large Language Models 

**Authors**: Nuo Chen, Xinyang Jiang, Zilong Wang, Zhifei Zhang, Xiaoye Qu, Jiajun Deng, Yulan Guo, Cairong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.26592)  

**Abstract**: Following clinical decision pathways (CDPs) defined by clinical practice guidelines is essential for safe and reliable medical decision-making. However, existing medical large language model (LLM) benchmarks mainly evaluate final-answer accuracy, providing limited evaluation of models' ability to adhere to guidelines. To address this gap, we introduce MEGA-CDP, a benchmark for evaluating whether medical LLMs can generate guideline-adherent CDPs using provided guidelines as references. MEGA-CDP is constructed from 2,274 English and Chinese clinical practice guidelines through a guideline-to-case pipeline, yielding 42,353 clinical cases with explicit reference CDPs. It supports both single-turn vignette and multi-turn interactive settings, and introduces a CDP-oriented evaluation framework for measuring pathway consistency. Experiments on 16 representative LLMs show that reliable clinical decision support remains challenging for current models, demonstrating the need for CDP-oriented evaluation and the value of MEGA-CDP for advancing guideline adherence in medical LLMs. 

---
# Surgical Alignment in Knowledge Graph Training for Clinical Diagnosis with Large Language Models 

**Authors**: Saksham Khatwani, He Cheng, Majid Afshar, Dmitriy Dligach, Yanjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.26587)  

**Abstract**: Biomedical knowledge graphs (KGs) offer structured medical knowledge that can ground large language model (LLM) reasoning in clinical diagnosis application, yet how KG signal should be integrated into LLMs remains an open question. We present a systematic study spanning five KG task formulations, three training paradigms, two KGs, and three base LLMs. At the task level, all paradigms improve over the non-finetuned baseline, but methods with comparable in-domain accuracy show substantially different knowledge transfer behavior. We introduce Gradient Intervention Density (GID) and Gradient Distortion (GD) to measure how broadly an optimizer modifies the pretrained model. GID and GD together reveal a clear divide: KG-judgment training under KL regularization produces sparse, localized updates (a regime we term as surgical alignment), while task-specific SFT produces dense ones. A controlled ablation shows that the objective and KL contribute to sparsity independently, and the paradigms that produce sparse updates also improve reasoning quality, even when their in-domain accuracy is lower than task-specific SFT. Assessing KG-LLM integration thus requires complementing accuracy with optimization-geometry diagnostics. Our implementation can be found at this https URL. 

---
# Double Trouble: Bilingual Pretraining Leaves Language-Conditioned Effects in Shared-Language Representations 

**Authors**: Anjishnu Mukherjee, Ziwei Zhu, Antonios Anastasopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2608.26576)  

**Abstract**: When researchers compare multilingual models for probing, interpretability, or cross-lingual transfer, they often align embedding spaces and assume that shared-language representations are comparable. We show that this assumption can be premature for decoder-only models. We pretrain paired 310M-parameter models (one English-only, one bilingual) across eight typologically diverse languages, separately controlling for English exposure, total compute, and document overlap. After aligning on shared English vocabulary, we test held-out words and find that token embeddings look similar after alignment, but the deeper hidden states that the model uses for prediction do not. This gap holds for all eight languages and survives controls for document overlap and alternative alignment methods. This hidden-state mismatch grows through middle transformer layers, suggesting that it arises from contextual processing rather than the input representations where alignment is performed. Embedding alignment can mask real differences in how models internally represent a shared language, which matters for any downstream study that treats aligned models as interchangeable. 

---
# Dependency-Aware Revocable Decoding for Efficient Diffusion Large Language Model Inference 

**Authors**: Wooje Park, Insu Lee, Minyoung Noh, Jaeyun Jang, Sungmin Lee, Kyuhong Shim, Byonghyo Shim  

**Link**: [PDF](https://arxiv.org/pdf/2608.26574)  

**Abstract**: Diffusion large language models (dLLMs) offer a promising alternative to autoregressive generation by decoding multiple tokens in parallel through iterative denoising. However, increasing decoding parallelism often degrades generation quality, as early errors can contaminate later contexts. Revocable decoding mitigates this issue by re-evaluating decoded tokens and remasking unreliable ones, but existing methods overlook that unreliable tokens may also corrupt the verification context itself. We identify this failure mode and propose Dependency-Aware Revocable Decoding (DARD), a training-free framework that separates tokens into masked, candidate, and unmasked states. DARD verifies candidate tokens using a selective context that excludes less reliable tokens and adaptively regulates their influence on subsequent decoding. Experiments across 12 textual and multimodal benchmarks on 3 open-source dLLMs show that DARD consistently improves the speed-quality Pareto frontier over recent revocable decoding methods, achieving a 2.71$\times$ speedup and a 4.35-point CIDEr score gain over Saber on Flickr30K. 

---
# SPT: Skills as Pre-Training Data for Agentic Language Models 

**Authors**: Yufei Sun, Yudong Li, Yiming Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.26563)  

**Abstract**: Agentic (tool-using) language models are mainly trained on tool-call traces and agent trajectories during post-training. These data provide direct behavioral supervision, but producing them requires task environments, execution, and verification, making broad tool and task coverage expensive. Publicly available skills offer another source of training data: they encode reusable tool semantics and workflows but are typically used only as inference-time context. We introduce Skill Pre-Training (SPT), a mid-training method that applies causal language modeling to SkillCorpus, a collection of public multi-file skill packages, optionally mixed with general data. To preserve relations among files within each package, we also introduce Reference Insert, a reference-aware assembly strategy that places supporting files near their mentions in the primary instruction. Experiments across multiple model scales and post-training recipes show that SPT consistently improves agentic performance over mid-training on general or trajectory data, while largely preserving general performance. Data mixture experiments show additional benefits from combining skill data with general annealing corpora. These results indicate that skill packages are a valuable data source for pre-training agentic language models. 

---
# SPEAR: Distilling Domain-Adaptive Reasoning Skeletons via Sequential Symbolic Alignment in Reinforcement Learning 

**Authors**: Zhuochun Li, Yuelyu Ji, Yiming Zeng, Daqing He  

**Link**: [PDF](https://arxiv.org/pdf/2608.26550)  

**Abstract**: Reinforcement learning-based knowledge distillation has the potential to transfer complex reasoning from teacher to student models, yet it currently faces a critical dilemma: researchers must choose between sparse outcome-based rewards, which provide insufficient logical guidance, or expensive neural Process Reward Models (PRMs) for dense signals. We resolve this by introducing SPEAR (Symbolic Process Evaluation and Alignment Reward), a training-free and plug-and-play process reward method for sequence-level on-policy distillation. SPEAR projects natural-language reasoning traces into domain-adaptive symbolic milestones, providing an efficient proxy for process-level reasoning alignment. By utilizing the longest common subsequence (LCS) to align student explorations with teacher milestones, SPEAR provides a dense, order-aware reward signal that enforces logical consistency without the need for an external neural verifier. Our experiments across math, science, and commonsense reasoning tasks demonstrate that SPEAR effectively bridges the reasoning gap between student and teacher models via sequence-level distillation with efficient dense process rewards. Our code and data are available at: this https URL. 

---
# Multi-Expert Conformal Risk Control for Pairwise LLM Judging in Open-Ended Dialogue 

**Authors**: Ming Cheng, Yusheng Dai, Qiuhong Ke, Zhaolin Chen, Lizhen Qu  

**Link**: [PDF](https://arxiv.org/pdf/2608.26529)  

**Abstract**: In this paper, we explore multi-expert Conformal Risk Control (CRC) algorithms for pairwise LLM-as-a-Judge evaluation in open-ended dialogue. Our core insight is that multi-expert aggregation offers a complementary remedy to CRC: whereas CRC controls risk at the decision threshold through abstention, aggregation sanitizes the scoring function at its source. Guided by this, we first design two multi-expert CRC methods: Score Averaging and Decision Voting, which aggregate at the score and decision levels, respectively. While both strategies outperform single-expert methods on homogeneous expert panels, on heterogeneous LLM judges they remain risk-valid but recover only limited coverage, because a uniform threshold cannot match the experts' distinct scoring scales. To resolve this issue, we further propose Marginal-Calibrated Conformal Consensus (MC3): it captures distinct per-expert scales via initial threshold ratios, while jointly tuning a unified decision function $C_t(x)$ applied identically in both calibration and test, thereby preserving exchangeability. To evaluate our framework, we construct Panel, a 1,800-pair human pairwise-preference benchmark for open-ended dialogue. It is built on responses generated by four open-weight LLMs over dialogue contexts from three domains (ESConv, MSC, DREAM), with full logit access. In experiments, we find that both Score Averaging and Decision Voting substantially improve accuracy and acceptance rate on homogeneous panels. Notably, MC3 extends these gains to heterogeneous panels by accommodating distinct per-expert scoring scales across all three datasets. 

---
# Sycophancy Suppression Can Impair Rational Updating: Anti-Sycophancy Should Preserve the Ability to Update 

**Authors**: Huanhuan Ma, Henry Peng Zou, Chengze Li, Enze Ma, Yunyue Su, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.26511)  

**Abstract**: Large language models often exhibit sycophancy, revising their answers to align with users when users push back. Such answer flips, however, can arise from different causes. One possibility is that the model simply aligns with the user's feedback in order to satisfy them. Another is that the feedback genuinely contains useful evidence, prompting the model to update its answer in a rational way. We distinguish them as Unsupported-Yielding and Rational-Updating. Prior work focuses primarily on suppressing Unsupported-Yielding, while overlooking its effect on Rational-Updating. We address this gap with a two-turn evaluation framework that measures the two behaviors separately. Across representative training-time and inference-time interventions, we find that anti-sycophancy methods often encounter a trade-off in which reducing Unsupported-Yielding can sacrifice Rational-Updating, and vice versa, even when the two objectives are optimized jointly. Mechanistic analysis suggests that the two behaviors share an internal substrate: the MLP neurons and attention heads driving them overlap substantially, and their associated steering directions are positively aligned. We further conduct a preliminary orthogonalized steering exploration, which yields modest, backbone-dependent selectivity gains. Overall, our results suggest that anti-sycophancy should be treated not as a simple suppression problem, but as a selectivity problem, where effective interventions should preserve Rational-Updating while reducing Unsupported-Yielding. 

---
# Compositional Generalization via Structural Identification in a Category-Theoretic Framework 

**Authors**: Akihiro Maeda, Thomas Seiller, Yohei Oseki  

**Link**: [PDF](https://arxiv.org/pdf/2608.26465)  

**Abstract**: Compositional generalization is usually evaluated through model accuracy. We instead ask which structural or lexical identifications make held-out COGS examples admissible from the structures observed in training. Sentences are represented as functors from syntactic addresses to lexical tokens, and selective collapses induce Kan extensions that propagate observed associations. Across 21 COGS generalization types, admissibility follows distinct identification profiles, while residual failures separate unsupported structural templates. These data-side diagnoses characterize what the training corpus licenses under specified identifications, without training a predictive model. 

---
# Vowel Signs Are Not Letters: A Pre-tokenization Ceiling on Multilingual Tokenizer Fertility 

**Authors**: Sajal Regmi, Siddhartha Pudasaini, Chetan Phakami Pun  

**Link**: [PDF](https://arxiv.org/pdf/2608.26449)  

**Abstract**: Byte-level BPE tokenizers that use the HuggingFace ByteLevel pre-tokenizer inherit GPT-2's word regex, where a word is defined as \p{L}+, one or more Unicode letters. In abugida scripts, vowels are written as combining marks; this pattern therefore splits each word at every vowel sign. Since BPE merges only within a pre-token, those splits persist through training regardless of vocabulary size or corpus composition. We formalise this effect as a training-free lower bound on fertility. Across 26 languages from a parallel corpus, every one of the 17 abugidas is affected, ranging from 1.47x (Tibetan) to 9.02x (Thai), whereas Latin, Cyrillic, Hangul, and Han show exactly 1.00x. For 5 languages, matched tokenizer pairs that differ only in this character class fall within 2.2% of the predicted floor, scoring 4.78 versus 1.58 tokens per word on Nepali. When the Nepali share of the training corpus is swept from 5% to 95%, the broken tokenizer barely shifts at all (1.7%) while the fixed one shifts 33.9%, which separates a structural ceiling from a data shortage without needing to inspect any code. We train three 268M models that differ only in their tokenizer; the fixed variant achieves 4.43% lower held-out Nepali bits per byte at equal compute, and it still leads when given the same bytes with 1.59x the compute. A census of 3,479 HuggingFace repositories finds the letters-only word class present in 63.3% of the most-downloaded text-generation models, accounting for 72.5% of their downloads. GPT-4o's o200k pattern already uses a mark-aware word class, making the repair itself prior art. We quantify its value, show how to recognise its absence from symptoms alone, map which scripts it reaches, measure how widely it is deployed, and release a 65,536-entry Nepali-English tokenizer with a harness that regenerates every number here from public data on a laptop. 

---
# AfriSwitch: A Benchmark for In-the-Wild African Code-Switched Speech Recognition 

**Authors**: Gabrial Zencha Ashungafac, Busayo Awobade, Tobi Olatunji  

**Link**: [PDF](https://arxiv.org/pdf/2608.26434)  

**Abstract**: Code-switching is pervasive in bilingual African conversation, yet most ASR systems assume monolingual input and are evaluated on curated monolingual benchmarks. We present AfriSwitch, a 61.36-hour human-transcribed benchmark of in-the-wild code-switched speech spanning 16 African languages and language varieties, released with switch-level English span tags, perutterance Code-Mixing Index (CMI), and switch-point counts. Corpus statistics show that mixing behaviour varies widely across African languages along two largely independent axes: how often speakers alternate, and how balanced the mixture is. No single scalar captures how code-switched a language is. Benchmarking five open and commercial multilingual ASR systems zero-shot yields word error rates far above published monolingual figures for the same languages, with the best system averaging 35.93% WER and no system falling below 24% on any language. Africa-targeted training, not model scale or nominal language coverage, best predicts performance. 

---
# Case2Flow: Bridging Patient Cases and Guideline Flowcharts through Multimodal Retrieval 

**Authors**: Jiale Wei, Yufan Chen, Alexander Jaus, Zdravko Marinov, Julian Friedrich, Simon Reiß, Jens Kleesiek, Rainer Stiefelhagen  

**Link**: [PDF](https://arxiv.org/pdf/2608.26414)  

**Abstract**: Medical guidelines encode rich, evidence-based decision logic, yet the specific decision artifact a clinician needs is hard to locate within a guideline, let alone across guidelines covering plausible diseases and treatments. While guideline passages have supported end-to-end question answering, flowcharts remain largely underused in decision support despite their ability to encode actionable clinical pathways. We therefore introduce Case2Flow, a task designed to retrieve the most relevant guideline flowchart for a given patient case from a collection of guideline documents. To support it, we construct FlowAtlas, a curated corpus of 202 flowcharts extracted from 2,080 medical guidelines, together with a pipeline that synthesises 1,911 aligned case-flowchart pairs. Our evaluation of multimodal retrieval methods reveals systematic failure modes, including overreliance on keywords and spurious token-patch matches induced by uninformative background regions in flowcharts. Motivated by this, we propose CRISP, a training-free scoring method that sharpens late-interaction retrieval by suppressing uninformative patches, discounting ambiguous token matches, and incorporating bidirectional query-image alignment. CRISP improves Recall@1 by up to 18.71 percentage points, while a blinded physician assessment on published case narratives provides preliminary feasibility evidence beyond synthetic queries. 

---
# LowRankArena: A Standardized Evaluation Platform for SVD-Based LLM Compression 

**Authors**: Zishan Shao, Lixun Zhang, Kangning Cui, Wenhao Wu, Jinhee Kim, Yixiao Wang, Ting Jiang, Hancheng Ye, Qinsi Wang, Fan Yang, Danyang Zhuo, Yiran Chen, Hai Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.26389)  

**Abstract**: SVD-based low-rank compression has become a fast-growing direction for reducing the memory and computational cost of large language models (LLMs). However, meaningful comparison across existing studies remains difficult as prior evaluations use varied benchmarks, inconsistent ratios, and diverse setups, often failing to isolate low-rank effects from auxiliary techniques. As a result, it remains unclear whether reported gains reflect method-level improvements or differences in evaluation protocol. This lack of comparability highlights the need for a unified, reproducible evaluation platform. To address this problem, we present LowRankArena, a standardized evaluation platform for SVD-based LLM compression. LowRankArena unifies task versions, uniform-precision compression budgets, comparison regimes, and inference measurements, and provides a reproducible pipeline with over 3 TiB released compressed checkpoints. Using LowRankArena, our aligned audit of five representative SVD methods reveals that prior findings are highly conditional under standardized protocols: clear leaders and performance tiers shift across backbones and keep ratios, multiple-choice accuracy can hide large perplexity degradation, and nominal low-rank savings yield workload-dependent and often limited end-to-end speedups. Our code is available at: this https URL. 

---
# Co-Evolving Structured Knowledge and Reasoning in Language Models 

**Authors**: Ryan Thomas Noonan, Linxi Zhao, Menghan Xu, Akanksha Sarkar, Mihir Mishra, Dongyoung Go, Kilian Q. Weinberger, Yoav Artzi, Jennifer J. Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.26386)  

**Abstract**: Retrieval-augmented methods improve factual accuracy by grounding language models in external knowledge, but retrieving over unstructured text often introduces irrelevant context and offers limited control over the retrieved information. Structured knowledge bases offer a more controllable alternative, yet they are expensive to construct and often brittle to reason over. To address these limitations, we propose KBevo: a co-evolving framework that jointly learns to construct a structured knowledge base and reason over it for knowledge-intensive question answering. By optimizing both components end-to-end with QA outcome rewards, our method enables reasoning success to directly improve the quality of the constructed knowledge base. This leads to larger, better-connected knowledge structures with higher answer reachability, while also improving compositional factual reasoning and controllability compared to standard retrieval baselines. 

---
# Why RAGs Hallucinate: Penalty-Aware Evaluation of Retrieval-Augmented Generation Systems with Knowledge-Gap Canaries 

**Authors**: Alden Do Rosario, Hussein Younes, Felipe Pires  

**Link**: [PDF](https://arxiv.org/pdf/2608.26385)  

**Abstract**: Volume-based accuracy rewards retrieval-augmented generation (RAG) systems for guessing: a system that answers everything outscores one that declines when its knowledge base cannot support an answer. Building on the confidence-target analysis of Kalai et al. (2025), we present a penalty-aware evaluation framework for deployed RAG products, combining (i) asymmetric scoring (correct +1, wrong -4, abstain 0), (ii) knowledge-gap canaries, questions whose answers are verifiably absent from the knowledge base, so that any answer constitutes ungrounded generation from parametric memory, and (iii) a failure-attribution pipeline that separates retrieval, generation, and abstention-policy failures. Applying the framework to three commercial RAG systems and a no-retrieval baseline on SimpleQA-Verified (1,000 questions x 3 repeats, graded blind by a cross-family three-judge panel with 98.9% unanimity), we find that accuracy when answering is closely clustered across systems (97.0-98.0%), while canary violation rates differ roughly sixfold (16.7% vs. 98.1%). The systems are separated less by what they answer correctly than by whether they answer at all when they should not, and penalty-aware scoring reorders the volume-based ranking accordingly; the reordering is stable across penalty settings from k=1 to k=9. All code, configurations, transcripts, and judge votes are released for independent audit. 

---
# Survival-Guided Length Control for Efficient Diffusion Language Models 

**Authors**: Ivan Kobyzev, Abbas Ghaddar, Yufei Cui  

**Link**: [PDF](https://arxiv.org/pdf/2608.26374)  

**Abstract**: Diffusion language models (DLMs) generate text by iteratively denoising masked sequences, but standard decoding either fixes the sequence length or relies on ad hoc stopping rules, often leading to unnecessary denoising steps. We recast length selection as a discrete-time survival problem over the end-of-sequence token and propose a plug-in, training-free length predictor that can be added to any existing DLM. Across reasoning and code-generation benchmarks, survival-guided length decoding speeds up inference by up to 7 times while preserving task accuracy. We further find that predicted lengths vary widely even within the same dataset, making model performance sensitive to the chosen length. 

---
# Knowledge-Verified Emergent Deception in LLM Agents Under Conflicting Incentives 

**Authors**: Zheyuan Liu, Weiliang Zhao, Xiangchi Yuan, Ningshan Ma, Yue Huang, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.26372)  

**Abstract**: Large language models are increasingly deployed as autonomous agents serving users on behalf of companies, placing them in settings where user and deployer interests can conflict. When an agent knows that a user is owed something its deployer would prefer to deny, does it remain honest? Answering this is difficult because false statements can reflect either ignorance or hallucination rather than deception. To address this challenge, we introduce KnownLieBench , a knowledge-verified benchmark that first confirms through a neutral probe that an agent knows a user's entitlement, and then evaluates whether it makes false claims once an incentive to deny that entitlement is introduced. Specifically, KnownLieBench covers eight customer-service domains and 112 grounded cases, conducts multi-round dialogues with a trust-tracking customer agent, and separates deception emerging from incentive alone from deception produced under explicit instruction. Across eighteen proprietary and open-weight models, emergent deception varies substantially across model families and domains. We further use the benchmark for post-training, finding that honesty-directed fine-tuning reduces deception under incentive, while deception-graded fine-tuning increases lie success on honest-control dialogues without increasing lie frequency under incentive. By verifying entitlement knowledge before scoring deceptive behavior, KnownLieBench reduces the confound between lying and not knowing and enables more rigorous auditing and steering of agent honesty. 

---
# Cross-lingual Representation Learning via Centroid Intervention Fusion 

**Authors**: Wei Sun, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2608.26357)  

**Abstract**: Large language models (LLMs) exhibit uneven multilingual performance, especially when dealing with low-resource languages. Inference-time intervention offers a lightweight way to improve cross-lingual transfer by modifying the hidden states produced by the LLMs during the forward pass, without updating model parameters. However, existing cross-lingual intervention methods typically learn separate projections from source to target languages, which limits scalability and prevents knowledge sharing across languages. We propose Centroid Intervention Fusion (CIF), a projection fusion framework that consolidates multiple multilingual intervention projections into a single language-shared operator. Across multilingual commonsense reasoning, natural language inference, factual editing, and machine translation benchmarks, CIF outperforms the strongest prior pairwise intervention baseline by up to +3.378 pp on average across four model backbones, while supporting performance gains for low resource languages. The code is available at this https URL. 

---
# MoganColBERT-TR: A Late-Interaction Multi-Vector Retrieval Model for Turkish 

**Authors**: Furkan Yilmaz, Habibe Aleyna Tasdemir, Muhammed Faruk Gozay  

**Link**: [PDF](https://arxiv.org/pdf/2608.26344)  

**Abstract**: We previously reported a ModernBERT encoder trained from scratch for Turkish (MoganBERT-TR) and a single-vector embedding model built on top of it (MoganBERT-embed). This work introduces the third model in that lineage: MoganColBERT-TR, a multi-vector retrieval model that, instead of compressing a query or a document into a single vector, represents it at the token level through a 768->128 projection and scores it with MaxSim late interaction. The model is not trained from scratch: the embedding model's encoder is taken as the starting point and adapted to the ColBERT objective with a single-epoch distillation phase. Training data is produced from two sources - title-to-passage pairs carved out of our own pretraining corpus in the character domain and at sentence boundaries, and two Turkish question-based retrieval sets - and is distilled from the soft scores of a cross-encoder teacher (bge-reranker-v2-m3) over one positive and seven mined negatives. We show that in hard negative mining, rank-based skipping alone is insufficient and must be combined with a group mask and a cosine ceiling. Evaluation is carried out with the official pipeline of TurkColBERT, a benchmark built for Turkish late-interaction retrieval (PLAID index, exact MaxSim), on five Turkish BEIR datasets; none of them appears in our training pool, so all five results are clean zero-shot. With 148.9M parameters, MoganColBERT-TR reaches an overall score of 37.36 (35.53 nDCG@100, 31.81 nDCG@10) averaged over the five datasets and finishes second among the five models compared: it outperforms the twice-as-large ColmmBERT-base-TR on four of five datasets and by +3.05 overall, and the benchmark's largest model by +12.30. The gap to the leading model (mLateOn) is concentrated on ArguAna-TR, the dataset with by far the longest queries. 

---
# Neuro-symbolic PRM: Enhancing Scientific Reasoning via Structured Traces and Symbolic Verification 

**Authors**: Yuxin Zi, Cong Xu, Suparna Bhattacharya, Martin Foltin, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2608.26329)  

**Abstract**: While tool-augmented Large Language Models have significantly improved multi-step reasoning in quantitative STEM tasks, a critical residual failure mode remains: intermediate reasoning steps that are syntactically well-formed, mathematically executable, and unit-consistent, yet contextually ungrounded. Current approaches either rely on formal verifiers that cannot assess semantic intent, or burden Process Reward Models (PRMs) with the dual task of checking both arithmetic and logic. In this paper, we propose a neuro-symbolic framework that cleanly decouples reasoning into two formal dimensions: Symbolic Validity ($V$) and Semantic Groundedness ($G$). We guarantee $V$ by construction using a deterministic symbolic verifier acting as a hard filter. To assess $G$, we train a PRM conditionally on the verifier-accepted manifold. To train this PRM efficiently, we introduce Counterfactual Symbolic Perturbation (CSP), a novel data synthesis strategy that algorithmically generates constraint-preserving hard negatives (steps that perfectly pass the verifier but are logically flawed). At inference, we deploy a verifier-first constrained search that guarantees execution consistency for verifier-covered operations while relying on the PRM solely to rank semantic grounding. By targeting the exact residual error class of strong tool-using LLMs, our method significantly improves reasoning reliability without the sprawling heuristics of prior frameworks. 

---
# How Unlikely Is "Unlikely"? Assessing Verbal Probability Perception Across Large Language Models 

**Authors**: Christos Petridis, Konstantinos Pelechrinis, Zoran Obradovic  

**Link**: [PDF](https://arxiv.org/pdf/2608.26327)  

**Abstract**: Large language models increasingly produce and interpret verbal probability expressions, yet whether these expressions carry consistent meaning across models (or match human perceptions of uncertainty) remains unknown. We present a systematic cross-model evaluation using a word-to-number mapping task grounded in established human benchmarks. Eleven uncertainty expressions were presented to 19 models under two conditions, forced single-number response and explanation elicitation, alongside a novel bidirectional roundtrip test of internal consistency. LLMs track the human benchmark with surprising fidelity: word ordering is preserved, three anchor points are recovered, and ``possible'' shows the highest variance and cross-model disagreement of any expression tested, consistent with its documented bimodal interpretation in humans. However, models show a systematic upward bias for negative expressions such as ``unlikely'' and ``improbable.'' Explanation elicitation reduces within-model variance while increasing between-model divergence, stabilizing individual models at the cost of inter-model consensus, and the roundtrip experiment reveals clear stratification, with frontier models maintaining coherent bidirectional representations. LLMs thus reproduce the structure of human verbal probability cognition, including its biases, while diverging systematically at the negative end---with implications for any setting where humans and models exchange probabilistic language. 

---
# When Is Noise Response Universal? Tokenization as the Hidden Variable in Language Models 

**Authors**: Yefan Tao, Gerald Friedland, Luyang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2608.26319)  

**Abstract**: The performance of textual neural models often degrades when their inputs are corrupted by noise such as typos, OCR errors, or dropped words. We study the degradation rate across neural models, both sentence embeddings and decoder-only LLMs, and find that how consistent it is depends on the scale of the noise: under word-level noise, models with very different architectures decline along nearly the same curve, while under character-level noise they separate. We further identify the determining factor to be the training objective, not the architecture: eight encoders spanning six pretraining paradigms are scattered initially, and collapse onto a common curve after a short contrastive training recipe. We trace the word/character split to tokenization: a single character edit forces the tokenizer to re-segment the surrounding word, disturbing the token sequence far more than dropping a whole word does. This finding and its underlying mechanism provide a practical means to predict a model's robustness to noise without any noisy evaluation, and to install robustness at a chosen noise scale through noise-augmented training. 

---
# MemToC: Benchmarking Memory-Tool Conflict Resolution in Large Language Models 

**Authors**: Arseniy Varlamov, Rishat Zinnatullin, Elisei Rykov, Alexander Panchenko, Ilseyar Alimova  

**Link**: [PDF](https://arxiv.org/pdf/2608.26295)  

**Abstract**: Tool-augmented LLMs must arbitrate between two fallible sources when a tool return conflicts with their parametric memory, yet existing evaluations measure source preference without establishing source correctness. We introduce MemToC, a controlled benchmark for post-tool-return arbitration with executable tools. MemToC comprises 6,504 evaluation episodes constructed from 542 quality-controlled factual questions, independently elicited model-specific closed-book answers, and controlled tool returns of known correctness. These components instantiate four source-correctness cases; tool-error and no-tool conditions are separate controls. Across five open-weight 7-9B models, tool returns strongly dominate elicited closed-book answers. The four instruction-tuned models retain a verified-correct answer against an incorrect tool in only 6.5-17.1% of eligible cases, follow a correct tool in 86.0-93.1%, and repeat the tool return in 78.4-86.0% of cases where both sources are wrong. No cross-model ordering remains stable across three instruction-wording variants with the question and episode content held fixed. We compare prompting with SFT and DPO using chain-level cross-fitting over ToolHop, so questions sharing an underlying fact never straddle training and evaluation. We apply an asymmetric success criterion: correct-answer retention must improve without a detected reduction in correct-tool following. SFT and DPO meet this criterion on the same two of four instruction-tuned backbones. Improvements rarely come cleanly: 19 of 20 tested method-model combinations reduce abstention after tool errors or on unanswerable inputs. Transfer beyond MemToC is positive but partial and depends on the model and presentation frame. Correctness-conditioned arbitration can be improved through fine-tuning, but gains must be evaluated jointly with correct tool use, abstention, and robustness to formulation. 

---
# On Scope Classification and Current Knowledge-Editing Benchmarks: A Negative Result, with INLAY as a Gradient-Free Case Study 

**Authors**: Aditya Pratap Singh  

**Link**: [PDF](https://arxiv.org/pdf/2608.26292)  

**Abstract**: Every memory-based knowledge editor in the SERAC lineage depends on a scope decision: given a query, does a stored edit apply? We report that current knowledge-editing benchmarks cannot measure this decision at all. Using INLAY, a gradient-free editor we built to obtain exact per-query ground truth (the model is frozen, edits live in an external addressable memory, and applying an edit is a bias added along one token's unembedding direction at decode time), we execute every candidate router action on 1,689 queries spanning three datasets and three input conditions. An oracle router choosing the best action every time ties a one-line static policy to four decimal places in all nine dataset-by-condition cells: the maximum attainable gain of any per-query routing method is 0.00 points. Abstention is the sole winning action zero times out of 1,689. The cause is structural: these are counterfactual benchmarks whose evaluation question asks for the post-edit answer, so answering from parametric knowledge is wrong by construction, and a benchmark without negatives cannot reward a classifier's ability to reject. This generalizes beyond our system to the whole scope-classifier family the benchmarks are used to evaluate. We confirm the mechanism directly: constructing the missing condition ourselves, by withholding a query's own edit from the index for half the sample, moves pooled headroom from exactly +0.0000 to +0.0420 and gives abstention its first wins. We also report where INLAY itself does not win (WISE beats it on Qwen2.5-7B CounterFact, and retrieval-augmented generation beats every method we tested, INLAY included, on rigorously matched RippleEdits), and disclose two bugs found during a self-audit of our own routing machinery, neither of which changed a published headline number outside noise. 

---
# A Reranker for Orchestrating Heterogeneous Speech and Text Retrievers 

**Authors**: Inho Kim, Sumyeong Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2608.26194)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have attracted significant interest for their ability to mitigate hallucinations in Large Language Models (LLMs). Although knowledge databases for RAG are increasingly diversifying to include various modalities such as speech and text, research on handling such multi-modal database scenarios remains limited. In this paper, we propose STeReO (Speech and Text Reranking Orchestrator), a reranker based on speech and text retrievers that aggregates disparate modality databases. To address the lack of specialized training data, we first curate a dataset comprising queries, mixed-modality evidence, and their corresponding relevance ranks. We then train the reranker and evaluate its effectiveness in both single-modality and mixed-modality scenarios. Our results demonstrate that the proposed algorithm excels at selecting the most relevant evidence, thereby significantly improving downstream question-answering performance. 

---
# Comparing Chunking and Embedding Strategies for Turkish RAG Systems 

**Authors**: Mustafa Sertaç Türkel, Fatma Nur Korkmaz, Ahmet Tuğrul Bayrak  

**Link**: [PDF](https://arxiv.org/pdf/2608.26192)  

**Abstract**: How documents are segmented into retrievable chunks and how those chunks are embedded strongly affect Retrieval-Augmented Generation (RAG) quality, yet neither has been systematically studied for morphologically rich languages such as Turkish. We compare Turkish document question answering across three chunking strategies (fixed-length, semantic, and layout-aware Docling), five embedding models, and two generator LLMs, over three documents with contrasting layouts. The fully crossed design yields 9,000 graded question-answer evaluations, each scored by an independent judge model, and component comparisons are tested by paired McNemar tests under Holm correction. Four findings follow. The chunking strategy determines how much the embedding choice matters: layout-aware chunking compresses the spread between the modern embedding models to about a point. The three leading embedding models are statistically indistinguishable, so language specialization yields no measurable retrieval advantage. The faster generator is not the more accurate one. And the preferred configuration depends on content type, since layout-aware chunking helps documents containing tables far more than prose. The best individual components therefore do not compose into the best complete configuration, which reaches 87.0%. 

---
# When the Canonical Completion Is Wrong: Formalizing and Measuring the Jump in Large Language Models 

**Authors**: Dai Shi, Xiaoyu Li, José Miguel Hernández-Lobato  

**Link**: [PDF](https://arxiv.org/pdf/2608.26187)  

**Abstract**: Whether large language models (LLMs) can perform the abductive leap from evidence to a new system of axioms, commonly referred to as a jump, has recently attracted considerable debate. A prominent position holds that LLMs are structurally incapable of such jumps, while recent studies challenge both its mechanism and its evidence. However, the debate remains difficult to settle, since the field still lacks a formal definition of the jump and a measure to test either side. In this paper, we develop a formal account of the jump in four steps and measure the second. The steps ask what the default completion of partial data is, when abandoning it is forced, when the abandonment is correct, and how successive jumps compound. Specifically, we define a jump instance as a finite extension problem with a machine-checked certificate that a correct completion exists, is unique up to renaming, and differs from the canonical completion of the data. The canonical completion is given by the left and right Kan extensions and is also what models produce without constraints, so it serves as the default. We prove that jump instances are well-posed and establish a family theorem that certifies instances of unbounded difficulty without enumeration. We further formalize when a jump is correct and how successive jumps compound. Finally, we run the measurement on nine certified instances and four frontier models. The Kan-default rate is zero in all 248 constrained trials, so the models do jump at this step and abandon the excluded default every time. Failures at higher difficulty stem from exhausted reasoning budgets or constraint errors, never from reverting to the default. These results indicate that the second step is not the bottleneck. If the disputed incapacity is real, it lies in generating the constraints or inventing the framework. Code can be found at: this https URL. 

---
# Investigating the Influence of Prompt and Response Languages on LLM Content Generation 

**Authors**: Thi Thanh Nhan Nguyen, Mai Khoi Tieu, Michael A. Riegler, Pål Halvorsen, Thu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2608.26186)  

**Abstract**: This study examines how prompt and response language influence the behavior of large language models. Using five models, we evaluated answers to 68 non translation questions across four language conditions: English to English, English to Norwegian, Norwegian to Norwegian, and Norwegian to English. After removing refused items, the dataset contains 1348 responses. We measure length differences with Cohen d, semantic fidelity with LabSE cosine similarity, and cross lingual keyword overlap with both raw and soft Jaccard. Prompt language has a strong effect on response length. With English output, Norwegian prompts shorten responses by about thirty seven percent. With Norwegian output, English prompts shorten responses by about forty one percent. The largest cross lingual contrast shows a reduction in word count but a smaller reduction in tokens, reflecting tokenizer differences. Despite variation in length, semantic similarity remains high, and soft Jaccard reveals substantial conceptual overlap that raw Jaccard does not capture. Effect sizes vary across models, indicating heterogeneity. Prompt language is not neutral and systematically shapes output length and lexical realization, with implications for multilingual prompt design. 

---
# PACEShop: Evaluating Personalized, Actionable, Compositional, and Evidence-grounded Shopping Assistants 

**Authors**: Weimin Lyu, Chen Luo, Guangrui Li, Yaochen Xie, Dhineshkumar Ramasubbu, Arief Koesdwiady, Wanqiu Long, Hansu Gu, Yutong Chen, Zheshen Wang, Dakuo Wang, Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.26180)  

**Abstract**: Shopping assistants are shifting from ranked product lists toward structured decision support, where systems must synthesize shopper context, product evidence, and next-step guidance into a coherent recommendation experience. This changes the unit of evaluation: a fluent response can still fail by ignoring shopper context, contradicting itself across components, or leaving defects too vague to localize. Existing personalization, grounding, and LLM-as-a-judge benchmarks cover pieces of this problem, but they do not define a joint evaluation target for structured shopping-assistant responses. We formulate this missing evaluation target as PACE: Personalized, Actionable, Compositional, and Evidence-grounded evaluation. We instantiate PACE with two artifacts: PACEShop, a benchmark dataset that makes the target measurable through 22,625 controlled records with structured personas, auditable evidence pools, GOOD/BAD labels, and gold defect family and location annotations; and PACEJudge, a training-free judging protocol that makes the target reportable through a structured output contract. Our experiments show that generic judges can recognize broad quality but fail to recover the diagnostic fields required for PACE; PACEShop makes these failures verifiable, and PACEJudge improves persona-source, cross-component, grounding, and family/location closure without retraining, showing that realistic shopping-assistant evaluation requires a task-matched output contract rather than only a stronger backbone or scalar prompt. 

---
# A Multi-Framework Comparison of Outline Stages in Long-Form Generation with LLMs 

**Authors**: Yifan Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.26177)  

**Abstract**: Long-form generation exposes fundamental limitations of large language models. Even 70B-parameter models exhibit length collapse at 16k-token outputs, and multi-chapter stories frequently trigger the attribute drift characteristic of the ``lost-in-the-middle'' effect. The ``outline-first, write-later'' paradigm has gained wide adoption, yet existing research evaluates the final writing rather than the outline itself, conflating two evaluation objects that should be decoupled. We construct a unified head-to-head benchmark covering 7 representative long-form generation frameworks across 3 generation granularities -- single-chapter, multi-chapter, and whole-book -- and propose an anchor-based LLM-as-a-judge protocol that directly assesses outlines against the source text on a 5-point anchored scale. Across 21 framework-granularity cells, no single framework dominates; performance depends on the match between a framework's intrinsic output form and the target granularity. SuperWriter ranks first in the length-constrained single-chapter mode, but this advantage degrades in whole-book mode. The outline-side ranking correlates only moderately with the writing-side ranking, supporting the outline--writing decoupling principle. Compute constraints limit the writing-side evaluation to a subset of cases; follow-up experiments will expand the sample size and add cross-model evaluators to enable stronger statistical inference. 

---
# Lost in Compression: A Controlled Cross-Lingual Audit of Extractive Prompt Compressors 

**Authors**: Mantas Lukauskas  

**Link**: [PDF](https://arxiv.org/pdf/2608.26175)  

**Abstract**: Extractive prompt compression promises to cut LLM inference costs by removing low-information tokens, and learned compressors such as LLMLingua-2 report strong results on English benchmarks. Most other languages already pay a token premium: the same content costs 1.3-1.8x more tokens than in English. We ask whether compression closes or widens this gap. Using fully parallel data in ten languages spanning five scripts, with controls budget-matched in the target model's tokenizer, we audit four learned compressors against four deterministic baselines, on eleven target models from ten vendors (over 250,000 evaluation calls). Three of the compressors are trained with English supervision (LLMLingua-2 XLM-R/mBERT; Kompress-v2 from the production Headroom stack); the fourth, XProvence, is trained multilingually. First, the transfer gap is real, replicates across target models and compressor backbones, and is strongly rate-dependent: at a 0.33 keep-rate English retains 57-62% of normalized context utilization while Lithuanian retains 10-24% and Chinese essentially none, despite Chinese having the smallest token premium. Second, the gap tracks compression supervision data, not architecture. All three English-trained compressors show it, deterministic methods show no comparable gap, and the multilingually trained XProvence v1 shows none. Its v2 release, retrained on translated data, empties 92% of Chinese contexts at its aggressive threshold without any warning. Third, in a harder long-context setting, aggressive learned compression drives compressed contexts to or below no-context utility in three of five non-English languages. A translate-then-compress pipeline matches or beats native compression at roughly half the token cost in three of five tested languages. We release all code, compressions, and model outputs. Safe compression budgets are much smaller outside English. 

---
# Hallucinations in LLMs: A Lifecycle-Based Survey of Causes, Detection, Mitigation, and Prevention 

**Authors**: Naveen Lamba, Sanju Tiwari, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2608.26168)  

**Abstract**: The lifecycle of hallucination in LLMs is a concept that enables building solid frameworks on the control and reliability of LLMs in high-stakes environments, including health, legal, and scientific research. Although previous surveys have primarily focused on detection or mitigation, this survey provides a lifecycle-based overview of the hallucinations in the LLMs, their cause, detection, mitigation, and this http URL propose a three-fold categorization of hallucinations across the LLM lifecycle: data-related, training-related, and inference-related, which is consistent with the lifecycle of the development of the LLM. Each of these stages is discussed regarding the cause of hallucinations, their detection, and the ways they can be addressed under specific mitigation or prevention interventions. In addition, we discuss the available benchmark data using a number of parameters so as to establish their suitability in identifying, restricting and managing hallucinations. The survey provides researchers and practitioners with a standardized framework to understand, diagnose, and cure hallucinations in a systematic system to present actionable data to build safer and more reliable LLMs. 

---
# Using Poly-Encoders for Computationally Efficient Automated Creativity Assessment 

**Authors**: Sam Grouchnikov, Phillip Gregory, Jiho Noh  

**Link**: [PDF](https://arxiv.org/pdf/2608.26165)  

**Abstract**: Automated creativity assessment has been a long standing challenge, with traditional methods often being resource intensive or lacking practical accuracy. We introduce a novel approach by using Poly-Encoder for computationally efficient and accurate automated creativity assessment. We fine-tuned a Poly-Encoder on a public dataset from the Scientific Creative Thinking Test, comprised of approximately 18,000 human-rated question responses. Our method leverages small pre-trained BERT encoders, achieving performance comparable to fine-tuned Large Language Models while significantly reducing computational demands. Experiments with the BERT-family models and poly-code counts achieved Pearson correlations of up to r = 0.74, 95% CI [0.73, 0.75] with human raters, matching the performance of resource intensive LLMs. This study bridges the gap between high performance and computational efficiency, potentially enabling widespread implementation of automated creativity assessment on accessible consumer-grade hardware. With some limitations, our findings suggest that Poly-Encoders are a promising alternative to LLMs for practical, scalable creativity assessment in various contexts, especially educational. 

---
# From Sound to Symptom: Real-Time Respiratory Signal Understanding for Conversational Healthcare Agents 

**Authors**: Tanmay Laud, Herprit Mahal, Subhabrata Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2608.26163)  

**Abstract**: Cough events during live spoken conversations carry clinically valuable respiratory signals, yet existing dialogue systems treat them as acoustic noise to be discarded. We present HealthCUES (Clinical Understanding from Embodied Sounds), a streaming pipeline for paralinguistic respiratory monitoring in real-time conversational agents, a capability that, to the best of our knowledge, is absent from all prior systems. HealthCUES processes audio through a rolling buffer aligned with dialogue turn boundaries, enabling sub-second event detection without interrupting conversational flow. Beyond binary cough detection, the system provides fine-grained analytics: (i) differentiation between coughing and throat clearing, (ii) cough subtype classification (dry, wet, barking, whooping) with confidence scores, and (iii) temporal duration estimation with start-end boundaries. To prevent alert fatigue, HealthCUES introduces dialogue-aware gating mechanisms that modulate triggering based on conversational context. The system leverages Qwen3Omni, a multimodal large language model (MLLM), with constrained structured outputs, decomposing cough analysis into parallel prediction tasks for independent prompt optimization. Evaluation on 847 in-house conversational audio segments demonstrates 93\% F1 for cough detection, 0.75 weighted-F1 for wet/dry subtype classification, and average end-to-end latency of 340ms; external validation on the AMI meeting corpus confirms robust cough, throat-clearing, and speech separation in the presence of speech (0.91 macro-F1). A user study with licensed healthcare professionals confirms the clinical relevance of subtype information and the system's utility in telehealth workflows. 

---
# Mutual Debiasing via Dual-Seed Comparison for Probabilistic Sampling in Large Language Models 

**Authors**: Zihao Guo, Hongtao Lv, Chaoli Zhang, Laiguo Yin, Lei Liu, Yonghui Xu, Lizhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2608.26161)  

**Abstract**: Although Large Language Models (LLMs) demonstrate remarkable capabilities in reasoning and decision-making, high-fidelity probabilistic sampling remains a persistent challenge. When generating random variables, LLMs consistently exhibit systematic biases that warp the target probability distributions. Current approaches often rely on a single, self-generated seed, which inherits model-specific biases. To overcome this vulnerability, we introduce Dual-Seed Comparison (DSC), a transparent, tool-free protocol that utilizes two independent LLM-generated seeds to neutralize bias. DSC compares the character-level ordinal values of the two seeds to construct a bit sequence, converts and normalizes this sequence into a pseudo-uniform variate, and then maps the variate to the target distribution through the inverse cumulative distribution function (CDF). Empirical results show that DSC substantially outperforms existing methods across 96\% of evaluated settings. Beyond direct sampling, task-adapted variants based on the DSC comparison operator improve distributional control in MCQ generation and attribute-constrained text-to-image prompting. 

---
# Self-Generated Text Recognition: Quality Heuristics, Cross-Task Transfer, and Downstream Bias in LLM Evaluation 

**Authors**: Jesse St. Amand, Callum Canavan, Sohaib Imran, Joseph Hewson, Aaron Lutz, Shi Feng, Puria Radmard, Lennie Wells  

**Link**: [PDF](https://arxiv.org/pdf/2608.26159)  

**Abstract**: Self-Generated Text Recognition (SGTR)--the ability of an LLM to identify its own outputs--poses risks to AI safeguards that rely on LLMs as evaluators or monitors. Specifically, an LLM may recognize outputs from other copies of the same model and make biased judgments or collude outright. Prior work has drawn conflicting conclusions about whether current models possess significant SGTR capabilities. We reconcile these findings by identifying key experimental design choices--which we term operationalizations--that drive divergent results. Evaluating 13-21 models across six operationalizations, we find that accuracy varies substantially with evaluation format (pairwise vs. individual assessments of text), conversation structure (presenting candidate text in user tags vs. assistant tags), and the domain of the task used to generate candidate text (e.g., coding vs. summarization). We corroborate previous observations that a quality heuristic--models attributing authorship to text they perceive as higher quality--is a dominant confound. We also find that improving a model's SGTR performance via SFT in one evaluation configuration can generalize to others. Training for SGTR additionally causes models to prefer their own outputs when acting as a judge in the AlpacaEval framework. Finally, we discuss the implications of our evaluations for the safety of future AI systems: our work suggests that, despite confounds, some models possess practical SGTR capabilities, and that training a model for SGTR in one setting can affect its self-recognition and self-preference more generally. We conclude that SGTR should be monitored and considered in the design of safety-critical AI applications. 

---
# VFA: Empowering Multilingual MLLMs via Vision-Free Adaptation 

**Authors**: Yixia Li, Yaqing Shi, Zhiwen Ruan, Dongdong Zhang, Lingjie Jiang, Shaohan Huang, Yun Chen, Guanhua Chen, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2608.26155)  

**Abstract**: Multimodal large language models have advanced rapidly, yet most remain English-centric, as scaling multilingual multimodal instruction tuning is limited by the scarcity and high cost of high-quality non-English image-text supervision. Although multilingual text data is abundant, naive textual fine-tuning can disrupt vision-language alignment and induce catastrophic forgetting. We propose Vision-Free Adaptation (VFA), a framework that decouples multilingual language enhancement from visual alignment by composing complementary task vectors over a shared LLM backbone. Specifically, we fine-tune a base LLM on multilingual text data to derive a multilingual task vector, which is then merged with the vision-aligned task vector of an MLLM. Experiments on five MLLMs across six multilingual multimodal benchmarks show consistent improvements while preserving both general multimodal and text-only capabilities. Moreover, using less than 2% of the text data, VFA narrows the gap to the fully multimodal-trained model, demonstrating its data efficiency. 

---
# Evaluating AI Generated Summaries for Cancer Patients 

**Authors**: Muhammad Aurangzeb Ahmad, Kim Shyu, Leon Oliver, Fergus Sleight, Paul Landau  

**Link**: [PDF](https://arxiv.org/pdf/2608.26154)  

**Abstract**: Large language models (LLMs) are increasingly being integrated into digital health platforms to generate summaries of complex medical data. Although these models can improve patient engagement and communication, these systems also raise concerns about accuracy, faithfulness, and safety in clinical contexts. In this study, we evaluate AI-generated summaries within a cancer patient care application using a dual assessment framework. Human domain experts, including oncology clinicians and patient-facing care staff, provided ground-truth evaluations of summary quality along dimensions of accuracy, clinical relevance, and readability. In parallel, we employed LLMs serving as evaluators (LLM-as-a-judge). Some limitations were identified in the generated summaries e.g., occasional omissions and minor inaccuracies. These were systematically analyzed and used to iteratively improve prompt design, grounding, and safety guardrails. 

---
# Artificial Intelligence Models Can Predict and Collaboratively Modulate Human Memory Search 

**Authors**: Eric Lacosse, Mariana Duarte, Graham Todd, Peter M. Todd, Daniel C. McNamee  

**Link**: [PDF](https://arxiv.org/pdf/2608.26152)  

**Abstract**: Large language models (LLMs) exhibit unprecedented natural language generation and many text-based problem-solving capabilities. Indeed, in many language-based tasks, for example routine coding, these artificial intelligence models have reduced, or even eliminated, the need for human input. But rather than replacing human cognitive effort, LLMs may instead serve as cognitive tools to extend human abilities, particularly when they are engaged in a task requiring open-ended conceptual exploration and creative ideation. However, we are yet to understand how these models may enhance such generative human cognitive abilities in human--AI interactions. In this study, we explore and evaluate the ability of LLMs to follow and enhance human mental trajectories during semantic memory search. To test this, we use the semantic fluency task (SFT), a classic cognitive paradigm requiring generative semantic memory retrieval that has long served to characterize convergent and divergent thinking in humans. We demonstrate that an LLM's abilities to track and predict human memory trajectories in this task exceed those of other humans. 

---
# Towards Interpretable Depression Detection: Linking Acoustic Features to DSM-5 Indicators 

**Authors**: Jonas Länzlinger, Katharina O.E. Müller, Burkhard Stiller, Bruno Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2608.26148)  

**Abstract**: Depression affects millions worldwide, yet diagnosis relies on subjective self-reports that may miss authentic behavior. This paper presents an approach linking speech acoustics to DSM-5 depressive-behavior indicators through a transparent Linkage Framework. Unlike black-box models, the framework explicitly maps acoustic features (pitch variability, pauses, speech tempo) to clinical indicators, enabling interpretable, indicator-level outputs. The system runs locally on commodity hardware (HW) to preserve privacy. Preliminary evaluation on DAIC-WOZ shows directionally consistent associations between acoustic features and DSM-5 indicators for psychomotor change and concentration difficulty, supporting the design rationale. Future work will validate on longitudinal datasets and extend multimodal integration while maintaining edge constraints. 

---
# CARE: Causally-Aligned Reasoning Exploration for Medical Large Language Models 

**Authors**: Yucheng Zhou, Peng Luo, Qianning Wang, Chengzhong Xu, Jianbing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.26147)  

**Abstract**: Large Language Models (LLMs) have shown strong potential for medical reasoning, yet the scarcity and cost of expert-annotated data constrain their progress. While reinforcement learning offers a scalable alternative, standard outcome-based methods in medicine often suffer from autoregressive credit assignment failure and gradient variance explosion. This leads to the "Right Answer, Wrong Reason" trap, where models inadvertently reinforce spurious correlations and dataset shortcuts rather than valid clinical deduction. In this work, we propose Causally-Aligned Reasoning Exploration (CARE), a theoretically grounded framework for intrinsic experience curation. CARE is built upon two rigorous conditions for high-quality training trajectories: Causal Sufficiency, which utilizes an agreement-based self-verification mechanism to mimic $do$-calculus interventions and effectively debias gradients; and Proximal Learnability, which employs dynamic entropy bounds to select experiences within the model's zone of proximal development for variance-bounded optimization. These rigorously filtered experiences are optimized via a dual-stream objective that combines on-policy group-relative exploration with difficulty-weighted experience replay. Extensive experiments on diverse medical multimodal and text-only benchmarks demonstrate that CARE consistently outperforms other strong competitors, substantially reducing correct-but-inconsistent reasoning and improving training stability. 

---
# Vagdhenu: A Vrutta (Meter) Aware Shloka-to-Chant (TTS) System for Sanskrit 

**Authors**: Prathosh A P  

**Link**: [PDF](https://arxiv.org/pdf/2608.26146)  

**Abstract**: We present Vagdhenu, a vrutta (meter) aware shloka-to-chant system for Sanskrit: a text-to-speech system that maps
a metrical verse to its chanted parayana recitation at high fidelity. This is an experience report, not a new
architecture. We take an off-the-shelf flow-matching TTS backbone and a large-scale neural vocoder, and add the
components a faithful Sanskrit chant pipeline needs: a frontend that routes Sanskrit through Kannada orthography to
avoid the Hindi-style schwa deletion that Devanagari triggers in Indic models; a frontend that obeys subtle
Sanskrit phonology (visarga sandhi with its jihvamuliya and upadhmaniya allophones, the aspiration contrast of
alpaprana and mahaprana, and the dental, retroflex, and palatal sibilants kept distinct); and a vrutta-aware
mechanism that detects the meter and picks an exactly matched reference under a half-reference rule. We report a
negative result that shaped the system: in a self-infilling flow-matching backbone, a text-side prosody conditioner
is architecturally inert, because the model recovers pitch from the context mel and the embedding gets no
gradient; the reference clip and a voice-steering retrain are the only working prosody levers. We also report a
comparative lineage across four families (StyleTTS2, VITS2, Matcha-TTS, and the flow-matching backbone), where each
earlier family hit a ceiling on conjuncts or prosody that a five-hour clone cleared at an expert MOS near 4.6. The
system shipped two deployments: a 32-chapter, 5183-verse video corpus (about 17.5 hours) and an audio app covering
about 18000 verses across 12 books. We release the frontend, inference and training code, weights, a
single-speaker chant dataset, and an interactive demo. 

---
# Why Current XAI Is Not Enough for Arabic NLP: A Critical Survey of the Explainability Gap 

**Authors**: Salima Lamsiyah, Ruslan Mitkov  

**Link**: [PDF](https://arxiv.org/pdf/2608.26144)  

**Abstract**: Explainable AI (XAI) is now a major theme in NLP; however, Arabic NLP remains under-explained in three connected senses. First, there is a method gap: Arabic XAI relies heavily on a small set of post-hoc techniques such as LIME, SHAP, attention visualization, and saliency, while broader NLP XAI offers richer diagnostic, counterfactual, probing, rationale-based, and human-centered methods. Second, there is a task gap: existing Arabic XAI work is concentrated in classification tasks, especially sentiment analysis, hate/offensive language detection, fake news, and spam, with weaker coverage of generation, retrieval, translation, summarization, structured prediction, and dialogue. Third, there is a linguistic gap: many explanations identify influential tokens, but rarely explain Arabic-specific phenomena such as morphology, clitics, dialectal variation, diglossia, orthographic ambiguity, diacritics, code-switching, named entities, cultural references, or Classical and religious registers. This critical structured survey synthesizes the reviewed literature on Arabic XAI across text, speech, and multimodal settings. We argue that Arabic NLP does not only need explanations of model decisions; it needs explanations that are faithful to Arabic as a linguistic, cultural, and sociotechnical object. We introduce a taxonomy of tasks, methods, linguistic units, varieties, goals, and evaluation practices, and propose a research agenda for linguistically grounded Arabic XAI. 

---
# Beyond Accuracy: A Qualitative Analysis of Vision-Language Models for Hate Speech Detection in Memes 

**Authors**: Muhammad Jawad Chowdhury, Adiba Hasan, Ishrak Hossain, Shahriar Ivan, Sabbir Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2608.26143)  

**Abstract**: Memes have turned out to be a powerful tool through which individuals share their ideas concerning contemporary social and political problems. Their anonymity, as well as their ability to go viral, make them a powerful medium for spreading hate. It remains very difficult to identify such complex and context-dependent hate speech. Although they display excellent performance on multimodal tasks, vision-language models (VLMs) tend to ignore context, irony, and other subtle cues that play a key role in identifying hateful memes. In this work, we present a qualitative analysis of four state-of-the-art VLMs: LLaVA-7B, Qwen-VL, GPT-4o mini, and Claude 3 Haiku. We evaluate these models under zero-shot and few-shot prompting to examine how contextual framing influences their outputs. Our analysis goes beyond simple classification accuracy and focuses on a qualitative evaluation of the models' generated justifications, providing a more in-depth understanding of their thought processes and constraints when dealing with hateful memes. 

---
# Position Is All You Need: A Free Lunch Token Compression Strategy for MLLM-based Referring Expression Segmentation 

**Authors**: Yuhan Liu, Yixiong Zou, Yuhua Li, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.26142)  

**Abstract**: Referring Expression Segmentation (RES) aims to generate pixel-wise segmentation masks from complex and implicit textual queries. While recent advances in Multimodal Large Language Models (MLLMs) have substantially boosted RES performance, their prohibitive computational overhead remains a critical bottleneck, which, however, is rarely explored. To fill this gap, we first evaluate typical token compression methods on this task and observe a surprising performance degradation. In this paper, we aim to understand this phenomenon for a solution. By extensive experiments, we find that token compression for RES requires preserving the original position embeddings and local neighboring spatial structures, indicating that visual token position information is far more critical than in other tasks. Building on this insight, we ask: Can we design the token compression method purely based on the position information? Therefore, we propose PAYN, a plug-and-play, training-free token compression method that relies solely on position information. PAYN retains tokens that are adequately distributed in every local neighboring region while strictly preserving original positional indices, thereby maintaining spatial relational consistency. Experiments on multiple RES benchmarks demonstrate that our method outperforms existing token compression methods, verifying that position is indeed all you need for token compression in the MLLM-based RES task. Codes are avaliable at this https URL. 

---
# AdaThinking-E: One-Token Entropy Regulation for Adaptive Thinking 

**Authors**: Zining Wang, Tongkun Guan, Boming Chen, Zhentao Guo, Jianqiang Liu, Chao Jin, Chen Duan, Kai Zhou, Pengfei Yan, Wei Shen, Xiaokang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.26141)  

**Abstract**: Multimodal large language models have demonstrated strong document reasoning capabilities by incorporating explicit thinking processes. While this capability significantly improves performance on challenging tasks, current models apply such deep reasoning uniformly to all questions, resulting in unnecessary computational overhead for simple task. This not only degrades user experience but also negatively impact accuracy on benchmark datasets. We identify the critical need for adaptive thinking mechanisms that can intelligently determine when to engage reasoning based on question complexity. To address this, we propose AdaThinking-E, a novel reinforcement learning framework that learns adaptive thinking through one-token entropy regulation. Our key insight is that model confidence in the decision to engage thinking (or not) can be quantified through entropy analysis of the predicted probability distribution at critical decision tokens. This observation motivates our entropy-governed reward mechanism: the training process naturally transitions from high-entropy exploration, where the model experiments with different thinking strategies, to low-entropy convergence with confident, generalizable decision-making policies. Crucially, this approach enables models to intrinsically discover when to think without requiring manual intervention or external difficulty labels. Extensive experiments demonstrate that our approach enables models to be both accurate on complex problems and efficient on simple ones across diverse document tasks. 

---
# Affix Cache for Diffusion Large Language Models 

**Authors**: Kaihua Liang, An Zhong, Xin Tan, Zafar Ayyub Qazi, Hong Xu, Jian Weng, Marco Canini  

**Link**: [PDF](https://arxiv.org/pdf/2608.26140)  

**Abstract**: Diffusion Large Language Models (DLLMs) enable non-autoregressive decoding and bidirectional context modeling, but efficient inference remains challenging. Unlike autoregressive systems, whose key-value (KV) cache can be reused for shared prefixes, DLLMs couple the KV states of shared context tokens with evolving generated tokens through bidirectional attention, making naive cache reuse stale while full recomputation is expensive. We present ACache, an affix-oriented cache reuse mechanism for shared text spans in DLLMs beyond prefixes. ACache identifies a small request-specific subset of critical affix tokens, called Anchor Tokens, by measuring their influence on masked generation tokens, and selectively recomputes the KV states of only these tokens while reusing the remaining affix cache. Built on Fast-dLLM, ACache recovers the accuracy loss caused by direct affix-cache reuse across different settings when recomputing around 20% of affix tokens. We also build a shared-prefix prototype on top of the Nano-vLLM engine, showing that ACache reduces recompute latency by up to 55.7% and improves end-to-end throughput by up to 1.68$\times$. 

---
# Syntax vs. Semantics: How Transformers Learn Deep Dependencies 

**Authors**: Jiangrui Zhao, Xiaoting Du  

**Link**: [PDF](https://arxiv.org/pdf/2608.26139)  

**Abstract**: Large Language Models demonstrate remarkable syntactic fluency, yet the optimization dynamics governing their acquisition of deep semantic dependencies remain poorly understood. We propose a mechanistic framework that models this learning process as a competition between Surface Statistics and Deep Semantics. Our theoretical analysis identifies a ``Gradient Starvation" phenomenon where the error signals for sparse semantic dependencies are actively suppressed during early optimization. This suppression impedes the learning of structural reasoning and causes its emergence to manifest as a sudden phase transition. Furthermore, this framework offers a mechanistic basis for the effectiveness of Chain-of-Thought (CoT) strategies. By externalizing intermediate reasoning steps into concrete tokens, CoT effectively bypasses the suppression regime inherent to implicit reasoning. We validate these findings across scales ranging from toy transformers to production models (Llama-3.1-8B, Qwen2.5-Coder-7B). Finally, guided by this theory, we propose a topology-aligned contrastive objective that explicitly rectifies the gradient geometry. Experiments on variable binding tasks demonstrate that our method achieves an improvement that is over 2x larger than that obtained via standard cross-entropy fine-tuning. 

---
# Cross-Platform Generalisation Failure in Mental Health Natural Language Processing: A Five-Axis Fairness Audit of Transformer Models on Social Media 

**Authors**: Rajveer Singh Pall, Sameer Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2608.26138)  

**Abstract**: We introduce the Cross-Platform Fairness Evaluation (CPFE) framework -- a five-axis audit protocol covering discriminative performance, calibration, statistical significance, prediction equity, and attribution stability -- and apply it to four transformer models (BERT, RoBERTa, Emotion-DistilRoBERTa, GoEmotions-RoBERTa) trained on a Kaggle mental health corpus (n=35,556) and evaluated on Reddit (n=6,257) and Twitter (n=2,883) test sets with emotion labels mapped to clinical proxies. All three independently evaluated models exhibit consistent and substantial cross-platform AUC degradation (30.3-35.4% on Reddit, 37.9-39.5% on Twitter) relative to within-platform performance (AUC 0.983-0.987), confirmed across five independent training seeds. Calibration failure is concurrent and severe: ECE rises from 0.056-0.060 in-domain to 0.196-0.229 on Reddit and 0.499-0.542 on Twitter. Platform-specific temperature scaling reduces mean ECE by 88.0% without altering discriminative performance (mean |delta AUC|<0.01), confirming separable failure modes. Prediction equity analysis reveals large cross-platform disparities (raw DI < 0.17; prior-shift-adjusted DI: 0.11-0.29 on Reddit), with equalized odds differences of 0.753-0.830 for mental health proxy classes on Reddit and 0.755-0.831 for anxiety on Twitter. Attribution stability analysis shows near-complete vocabulary divergence across platforms (Jaccard J=0 in 14/16 model-class pairs at K=10). These findings support treating cross-platform validation across all five CPFE axes as a standard requirement for mental health NLP systems in heterogeneous environments. In a single-seed fine-tuning experiment, mean AUC improved by 0.216, suggesting target-platform labels provide greater benefit as training signal than as calibration signal. 

---
# Interpretable, Fairly Evaluated Automated L2 Speaking Assessment that Beats the Single-Human Ceiling and Why Pause Encoding Does Not Change LLM Fluency Scores 

**Authors**: Eichi Uehara  

**Link**: [PDF](https://arxiv.org/pdf/2608.26137)  

**Abstract**: Second-language (L2) English learners can rarely rehearse speaking with a partner. Speaking is also the most anxiety-laden skill. These gaps drive a fast-growing market for automated speaking practice and scoring. But an automated score is trustworthy only if it is accurate, interpretable, fair, and benchmarked against the right human bar. We build an interpretable feature-plus-LLM hybrid for spontaneous L2 dialogue. We evaluate it without ever fitting to the human labels, against the ICNALE Global Rating Archive: 140 speeches rated by ~80 trained raters on 10 analytic criteria. We score the 130 L2 speeches with usable audio. A deterministic De-Jong speech-timing composite reaches rho=0.764. Blended with a single text-LLM fluency judgment, it reaches Spearman rho=0.818 against the consensus gold. This agrees with the consensus better than 81% of the 80 individual trained raters: above the median rater (rho=0.73) and near the best, and at ~83% of the reliability-corrected maximum (kappa_max=0.99). The blend improves on the composite alone by +0.054 (paired-bootstrap 95% CI [0.017, 0.108], excludes 0); the LLM adds a coarse fluency ranking that the continuous composite refines. We also report a controlled null on pause encoding, bounded to effects below about +/-0.1 rho at this sample size. Holding the LLM and learner words fixed and varying only how pauses are written into the prompt, inline pause locations do not beat aggregate pause statistics (-0.069, CI [-0.15, +0.08]), and a grounded mid-clause criterion gives no reliable gain. The fluency signal comes from the measured speech-timing features, not from how pauses are written for the LLM. We back every claim with two agreeing learner-isolation methods, paired-bootstrap CIs, a monologue negative control, per-feature reproduction of classical measurements, and a per-L1 fairness audit. 

---
# Reward-Informed Sparse Autoencoders and the Solution-Completeness Confound 

**Authors**: Tanvi Nagilla, Alexander Jameson, Daniel Manta, Shayaan Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2608.26136)  

**Abstract**: Sparse autoencoders (SAEs) decompose language-model activations into sparse, interpretable features, and an appealing way to aim them at reasoning is to curate their data with a signal reinforcement learning already produces: the reward. We build such a reward-informed SAE (RI-SAE): we split GRPO trajectories into high-reward ("good") and low-reward ("bad") reasoning continuations, train a standard JumpReLU SAE on their activations, and then ask what the resulting good/bad separation actually measures. On Llama-3.1-8B a sparse subset of the 16,384 features does separate the classes (silhouette 0.79 on the selected features versus 0.005 for the full code), but a control battery shows the separation is largely solution completeness rather than reasoning quality: a TF-IDF text classifier already splits the classes (AUC 0.75--0.83), and three structural cues alone (length, a closed reasoning block, and a boxed answer) reach AUC 0.70 (99% of good versus 69% of bad completions are boxed). A generic SAE that never saw the reward does not separate the classes at all (silhouette 0.01, no discriminative features), so the 0.79 is in-sample fitting of this curated signal rather than structure that a reward-blind dictionary recovers. We therefore present the recipe and its control battery together: reward filtering is a cheap, label-free way to reuse RL signals for interpretability, but most of what it surfaces is completion form. Two discriminative features are still readable (symbolic mathematics; procedural and evaluative language), which we take as illustrative rather than as isolated reasoning. 

---
# Data Science Approaches to Evaluating Honours Candidates 

**Authors**: Francesca von Braun-Bates, Sunreeta Sen, Indraayudh Talukdar, Anirban Lahiri  

**Link**: [PDF](https://arxiv.org/pdf/2608.26135)  

**Abstract**: We present a modular data-science pipeline for estimating public sentiment towards individuals from fragmented, unstructured open-source intelligence (OSINT). The method chains web search, text extraction, relevance filtering, tokenisation, co-reference resolution, and sentiment analysis to convert heterogeneous web material into auditable person-level sentiment distributions. We compare AFINN and VADER with MINOS, a domain-informed sentiment algorithm designed to detect language associated with reputational risk, misconduct, and positive public contribution. Applied to public figures with known reputational outcomes, MINOS gives the clearest separation between positive, ambiguous, and negative cases. The results show that chained NLP and OSINT methods can support transparent, reproducible, human-in-the-loop sentiment assessment for high-stakes decision support. We demonstrate the approach on the UK Honours system, where individuals are required to display high standards of public conduct to maintain an Honour. 

---
# Agent Seer: Synthesizing Scenarios from Specification Understanding 

**Authors**: Harish Karumuri, Mahesh Vemula, David Lopes Pegna  

**Link**: [PDF](https://arxiv.org/pdf/2608.26133)  

**Abstract**: Evaluating AI agents that use external tools requires realistic test scenarios that capture how practitioners compose tools and iterate across conversation turns. Constructing such scenarios by hand demands deep domain expertise, does not scale across tool ecosystems, and produces static benchmarks that cannot track evolving APIs. We observe that tool specifications -- function names, natural-language descriptions, and typed parameter schemas -- already encode sufficient semantic information to synthesize realistic evaluation scenarios without manual curation or live tool execution. Agent Seer builds off this latent information: from a single Model Context Protocol (MCP) specification, with no examples, no live tool access, and no domain-specific tuning. This pipeline enriches raw schemas, generates graded scenarios with synthetic tool outputs, and expands them into mock-data-grounded multi-turn dialogues that exhibit strong tool-calling correctness and conversational coherence.
Evaluation quality is measured by applying this pipeline on seven MCP specifications spanning diverse domains and tool-suite sizes and measuring the tool-calling correctness and conversational coherence. The pipeline achieves strong quality across all domains, with complete tool coverage on small and medium specifications. Two findings emerge within this analysis: parameter schema complexity is the strongest correlate of quality variation -- tool-suite size plays a smaller, orthogonal role -- and argument value accuracy is the dominant failure mode among imperfect scenarios, a sub-dimension invisible to coarse-grained name-match metrics. 

---
# Evaluating Language Models in Realistic Conversational Contexts 

**Authors**: Ilija Subasic, Andrew Rabinovich, Zhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.26131)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed to serve open-ended, multi-turn interactions, evaluating conversational quality at human scale has become a central challenge. Existing evaluation frameworks built for summarization, translation, or short-form QA tasks fall short of adequately measuring the consistency of human-scale dialogue, especially when derivation and validation of these metrics themselves often rely on synthetic rather than human sources. We fill the gap by introducing UPHELD (UPwork Human-Scale Evaluated Long Dialogues), a large, reference-full benchmark for evaluating human-scale conversational ability beyond factual correctness. UPHELD consists of hundreds of complete human-to-human dialogues authored by professional script writers, with realistic turn densities and 36,000+ per-turn human annotations across 30,000+ expert-generated dialogue turns. Using UPHELD, we systematically evaluate classical automatic metrics and reference-free LLM-as-a-judge approaches, and find them unreliable when correlated with expert human judgment. Building off this analysis, we use UPHELD to develop a Mixture-of-Judges framework that combines multiple evaluative signals and improves correlation with human assessments by approximately 30%. Overall, UPHELD provides a robust, human-grounded foundation for evaluating human-scale conversational intelligence that fills a crucial gap in the pre-existing LLM dataset landscape. 

---
# Agents Don't Paginate: First-Chunk Selection for LLM Tool Responses 

**Authors**: Tatiana Petrova, Andrei Mazniak, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2608.26130)  

**Abstract**: Coding agents built on large language models (LLMs), such as Claude Code, Cursor, OpenAI Codex, GitHub Copilot, and Aider, receive tool responses that routinely exceed the agent's per-turn token budget. The standard remedy, pagination, is available in every protocol that produced these responses; yet across the corpus of session logs from a public Model Context Protocol middleware we observed no agent-initiated requests for a second chunk. The first chunk is what the agent reads, so we ask how often the gold item (the one the agent needs) is placed first in it: the precision-at-1 rate $p_1$.
In a controlled offline benchmark we treat first-chunk selection as a 0/1 knapsack and compare six value functions on 500 SWE-bench Verified tasks, then test whether $p_1$ matters with a single-turn file-localisation probe on five language models (4,800 LLM calls; not an end-to-end resolve-rate test). Two pre-registered hypotheses did not hold and are our main findings. The central one is negative: raising $p_1$ does not systematically raise downstream accuracy. Per-model deltas stay under three percentage points (p.p.), are not consistently signed, and no model is significant; the agent recovers the gold from anywhere in the chunk, so what reaches its answer is first-chunk inclusion, not the gold's rank within it. The second: adding four file-metadata signals to a keyword scorer hurts $p_1$ by 4.8 p.p. (paired significance test, $p = 0.001$).
A parameter-free keyword scorer does raise $p_1$, from a 24.2% baseline to 35.0% (+10.8 p.p., far beyond chance; $p = 3.9 \times 10^{-8}$), and to 35.8% with a fallback to the tool's native ordering when no keyword matches. But by our central finding this is a rank-1 gain, and rank-1 is the part that does not reach the agent's answer: downstream accuracy does not move. 

---
# FIRSTPASS: A Multi-Domain, Multi-Round Peer Review Dataset Grounded in Real Editorial Outcomes 

**Authors**: Prabhjot Singh, Somnath Luitel, Manmeet Singh, Josh Durkee  

**Link**: [PDF](https://arxiv.org/pdf/2608.26129)  

**Abstract**: Scientific peer review datasets have trained AI systems exclusively on Computer Science and Machine Learning venues, producing models that critique ablation studies yet have never seen a biology reviewer demand contamination controls or a chemist question Nuclear Magnetic Resonance (NMR) spectral assignments. We introduce FIRSTPASS, the first large-scale peer review dataset built on complete multi-round editorial dialogues from a multidisciplinary high-impact journal. Curated from Nature Communications mandatory transparent peer review (instituted November 2022), FIRSTPASS comprises 3,668 records spanning five scientific domains (biology, chemistry, neuroscience, physics, and earth science), capturing the full iterative structure of scientific validation: initial referee reports, author point-by-point responses, and updated reviewer assessments. Each record carries an outcome label derived directly from editorial decisions (STANDARD for two-round review; EXTENDED for three or more rounds), providing ground truth absent in all prior corpora. An automated audit confirms 100% content integrity. Expert reviews average 2,155 words, substantially denser than conference venue reviews. All data, parsing pipelines, and evaluation scripts are released to enable reproducible benchmarking of AI scientific judgment across disciplines. 

---
# TelecomGPT-R1: A Unified Open-Source Reasoner for the Telecom Stack 

**Authors**: Bohao Wang, Chenwei Wu, Haoyu Li, Hang Zou, Yu Tian, Lina Bariah, Li Wei, Chongwen Huang, Yongliang Shen, Zhaoyang Zhang, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2608.26126)  

**Abstract**: Telecommunications is a high-leverage domain for large language model (LLM)-based reasoning because routine engineering workflows require joint grounding in normative specifications, operational telemetry, vendor-specific fault evidence, and exact RF/network calculations. However, current LLM integration in telecom remains bottlenecked by a two-sided capability gap: generic reasoners often lack telecom-specific grounding, while domain-specific telecom LLMs remain limited in structured, multi-step reasoning. To bridge this gap, we release TelecomGPT-R1-9B, a unified open-source telecom reasoner that ranks top-performing on the GSMA open telco leaderboard. Specifically, we curate a 67,427-example supervised fine-tuning (SFT) corpus organized around four complementary reasoning axes: protocol, knowledge, modeling, and fault. The corpus is built from axis-matched public web sources and enhanced through axis-specific chain-of-thought (CoT) generation and prefix-continuation self-validation. Starting from Qwen3.5-9B, we further develop a two-stage post-training recipe. First, multi-teacher low-rank adaptation (LoRA)-based SFT injects telecom knowledge and induces axis-specific reasoning formats. Second, group relative policy optimization (GRPO), stabilized by decoupled clip and dynamic sampling policy optimization (DAPO), optimizes the policy using four axis-aligned binary verifier rewards. Across seven public telecom benchmarks, TelecomGPT-R1-9B ranks first among open-source telecom LLMs and achieves a seven-axis mean comparable to state-of-the-art closed-source frontier reasoners. 

---
# Training-Time Explainability for Multilingual Hate Speech Detection: Aligning Model Reasoning with Human Rationales 

**Authors**: Muhammad Deedahwar Mazhar Qureshi, Sannaan Khan, Muhammad Atif Qureshi, Wael Rashwan  

**Link**: [PDF](https://arxiv.org/pdf/2608.26125)  

**Abstract**: Online hate against Muslim communities often appears in culturally coded, multilingual forms that evade conventional AI moderation. Such systems, though accurate, remain opaque and risk bias, over-censorship, or under-moderation, particularly when detached from sociocultural context. We propose a \emph{training-time} explainability framework that aligns model reasoning with human-annotated rationales, improving both classification performance and interpretability. Our approach is evaluated on HateXplain (English) and BullySent (Hinglish), reflecting the prevalence of anti-Muslim hate across both languages. Using LIME, Integrated Gradients, Grad X Input, and attention, we assess accuracy, explanation quality, and cross-method agreement. Results show that gradient- and attention-based regularization improve F-scores, enhance plausibility and faithfulness, and capture culturally specific cues for detecting implicit anti-Muslim hate, offering a path toward multilingual, culturally aware content moderation. 

---
# Natural-Language Policies to Executable Decisions: An Interpretable Large Language Model Framework 

**Authors**: Ziqiang Zhang, Jing Ma, Zilong Wang, Jiayuan Chen, Yi Qiao, Yu He, Wei Zhang, Dai Cheng, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.26124)  

**Abstract**: Pricing automation in large-scale tourism is challenging because travel orders are highly unstructured, while pricing policies are complex, rapidly evolving, and inherently open-ended. Traditional rule engines are brittle and costly to maintain, whereas unconstrained LLM agents lack the reliability and auditability required for financial decisions. We present a production-grade LLM-powered pricing system with a strict decision boundary: LLMs perform structured extraction and bounded policy/path selection, while all numeric pricing, including total-price computation, is executed deterministically. Policies are compiled into interpretable condition trees, enabling open-ended support for new clauses and evolving rules without code changes, while exposing auditable artifacts for human-in-the-loop control. Periodic fine-tuning on logged traces further improves tree induction and path matching. Deployed at a municipal state-owned tourism enterprise across 7 scenic sites and 12 business categories with 1,500+ operators and 1,000+ active policies, the system processed 3,960 orders in six months, reduced the order management team from 15-20 to 3, and cut per-order handling time from 10 minutes to <2 minutes. 

---
# Which India Survives Translation? Narrative Homogenisation Across Indian Oral Traditions in LLMs 

**Authors**: Paarth Singh Rathore  

**Link**: [PDF](https://arxiv.org/pdf/2608.26123)  

**Abstract**: Large language models (LLMs) are trained predominantly on English-language internet text that over-represents certain cultural narratives, raising concerns that models flatten the diversity of non-Western storytelling traditions into a single homogenized archetype. We present a pilot computational study examining this across three maximally distinct Indian regional oral and literary traditions: the Rajasthani Pabuji epic, classical Tamil Sangam poetry, and Bengali folk tales. We collected authentic reference corpora for each tradition (11, 21, and 10 passages respectively) and prompted two LLMs (Claude Sonnet and Gemini) with 54 generation requests spanning three prompt types per tradition - generic, culturally specific, and regional-language. Using Sentence-BERT embeddings and cosine similarity, we measure reference drift (how closely outputs track their own tradition's authentic texts relative to the other two) and cross-tradition convergence (how similar outputs are across traditions). We find that while outputs remain closer to their own tradition's reference than to others, cross-tradition similarity is high (0.52-0.66) relative to what the traditions' genuine distance would predict, indicating partial homogenisation. Unexpectedly, prompting in the regional language (Hindi, Tamil, or Bengali) consistently reduced fidelity to the authentic tradition relative to English prompting, by as much as 27 percentage points for Rajasthani and Bengali traditions. We discuss this against conflicting prior results on multilingual prompting and argue it reflects a difference between eliciting general cultural diversity and simulating one narrow, lesser-documented oral tradition. We position this pilot as a lightweight, scalable complement to recent large-scale human-annotation studies of Indian cultural misrepresentation in LLM-generated stories, as part of a broader doctoral research program. 

---
# Can a Model Catch Its Own Hallucinations for Free?: Label-Free Doubt Signals Hold Their Own Against a Labelled Dataset for Abstention 

**Authors**: Ali Asaria, Tony Salomone, Deep Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2608.26121)  

**Abstract**: Large language models state false facts as fluently as true ones, yet a model often "knows" internally when it is on shaky ground: the probability it assigns to its own answer tends to dip on the facts it gets wrong. The usual way to act on this, teaching a model to abstain rather than guess, requires a labelled dataset of right and wrong answers. We ask whether the model's own confidence, which is free and needs no labels, can do that job instead. We fine-tune each model (with LoRA) to answer when its frozen confidence is high and to say "I'm not sure" when it is low, using the signal alone and no correctness labels. Across six open-weights models (1B-8B, two families) on short-form factual question answering, with correctness adjudicated by an independent judge model, this label-free recipe holds its own against label-supervised abstention-tuning: at matched coverage we find no statistically detectable difference between the two. A control that drills hard examples instead of abstaining does not help, indicating the gain comes from calibration, not rote memorization. The signal's one blind spot is confidently wrong facts, which it cannot flag. A model's own doubt is thus a near-free substitute for a labelled dataset when teaching it when to abstain. Code and artifacts are available on request. 

---
# Recipes for Steering and Scaling LLMs via Sampling 

**Authors**: Jiajun He, Zongyu Guo, José Miguel Hernández-Lobato, Yuanqi Du  

**Link**: [PDF](https://arxiv.org/pdf/2608.26120)  

**Abstract**: Large Language Models (LLMs) are probabilistic models, typically defined by an autoregressive factorization. While recent work has begun to study richer target distributions beyond the base model, the sampling strategies remain highly inefficient. In this paper, we present a flexible and theoretically grounded framework for steering and scaling autoregressive LLMs with sampling. Within this framework, we describe two algorithms -- one based on Sequential Monte Carlo (SMC) and one based on Replica Exchange (RE) -- that steer generation toward powering, product or tilting of the base model distribution. We illustrate this framework through scaling the generation quality of LLMs without external supervision or reward models. Experimental results demonstrate our methods scale more favorably than Best-of-N and standard MCMC baselines. Overall, this paper offers a systematic recipe for probabilistic inference with LLMs via sampling. 

---
# DeflectBench: A Benchmark for Evaluating Rhetorical Fallacy Generation in LLMs 

**Authors**: Art Kanke  

**Link**: [PDF](https://arxiv.org/pdf/2608.26119)  

**Abstract**: Whether large language models can be prompted to generate rhetorical fallacies on demand, and whether current safety post-training constrains this behavior, has received less attention than the related question of detecting fallacies in existing text. We close this gap with DeflectBench, evaluating 23,990 generations from four frontier models across three deflection strategies (whataboutism, ad hominem, red herring), seven prompt framings, and 80 claims spanning four controversy levels. Refusal is governed primarily by request structure rather than claim content. Per claim refusal varies by only 11 percentage points across the 80 claims, while a single prompt frame change can swing within model refusal by nearly 100 percentage points and switching the requested fallacy type can swing it by over 80 percentage points within explicit framings. An educational debate coach prompt framing collapses refusal to near zero across all four model families, but the bypassed behavior is not clean compliance. Models typically produce labeled compliance, naming the requested manipulation in the same response that contains it. The four models distribute differently across refusal, labeled compliance, soft refusal, and clean compliance. The code and dataset are released at this https URL. 

---
# ElementCheck: Complexity-Aware Long-Form Text Factuality Evaluation via Sentence Elements 

**Authors**: Xinming Wang, Haoran Du, Yi Chen, Jian Xu, Hongming Yang, Han Hu, Yulong Chen, Cheng-Lin Liu, Xu-Yao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.26118)  

**Abstract**: Existing long-form factuality evaluation relies on the decompose-retrieve-verify pipeline. However, the pipeline suffers from noise from claim decomposition and fixed verification granularity, resulting in unreliable results. We propose ElementCheck, a complexity-aware framework that verifies long-form outputs via sentence elements. Instead of uniformly decomposing sentences into atomic sub-claims, ElementCheck extracts entity pairs that are explicitly linked through verifiable connections in the original sentence as elements, and organizes these into an element graph. The graph topology provides a structural signal for estimating sentence complexity, enabling direct verification for simple sentences and targeted element-level refinement and verification for complex ones. To support fine-grained evaluation, we construct a new benchmark FastFact-Sent by mapping isolated claims from FastFact-Bench back to their source sentences. Experiments on FastFact-Sent and two domain-specific benchmarks show ElementCheck consistently improves factuality verification across five backbone models while maintaining a favorable accuracy-cost trade-off. Further analyses demonstrate that complexity-aware verification reduces unnecessary re-verification and maintains stability across different backbones. 

---
# TreeGraft: Adaptive Multi-Drafter Grafting for Tree-Based Speculative Decoding 

**Authors**: Jiaming Fan, Daming Cao, Canchen Huang, Jiale Fu, Jin Zhang, Junjie Gao, Kai Yang, Xiangzhong Luo, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.26112)  

**Abstract**: Speculative decoding accelerates large language model inference through a draft-then-verify paradigm. Building on this, tree-structured methods improve inference by organizing proposals into multiple candidate paths, increasing the accepted length. However, existing tree-structured methods use a single drafter for all drafting steps, creating a dilemma: a smaller drafter is fast but yields lower-quality trees, whereas a larger drafter improves tree quality but suffers from high latency. To address this, we propose TreeGraft, a multi-drafter framework in which drafters of different costs jointly construct a shared draft tree. TreeGraft uses the stronger drafter to rescore candidates by updating scores assigned by the weaker drafter, reselect grafting positions, and recover promising paths left unexplored. It also integrates stronger drafter expansions non-destructively, preserving existing branches that may still be accepted by the target model. Together, these designs improve the quality of the shared draft tree. To control the drafting cost, TreeGraft introduces a lightweight scheduler distilled from an offline value system to decide when to call the stronger drafter. Across 10 model pairs and 6 benchmarks, TreeGraft outperforms the better of the two fixed single-drafter endpoint strategies by 15.1% on average, reaching a maximum gain of 26.6%. Our code is available at this https URL. 

---
# WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution 

**Authors**: Liyan Tang, Cyrus Rashtchian, Chun-Sung Ferng, Andrew Tomkins, Da-Cheng Juan, Tu Vu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27454)  

**Abstract**: Agent skills package specialized knowledge and workflows into reusable resources that extend AI agent capabilities. Recent work automatically discovers such skills from agent experience, which enables agents to progressively adapt through interaction. However, the insights that guide skill development typically remain scattered across optimization histories, limiting their systematic reuse across iterations. We introduce WikiSkill, a framework that co-evolves agent skills with a persistent knowledge base (wiki). At a high level, WikiSkill separates raw execution experience, accumulated knowledge, and executable skills, while continuously consolidating experience into the wiki, which subsequent skill updates can build on. Across diverse benchmarks and models, WikiSkill consistently outperforms state-of-the-art skill-evolution methods and improves over no-skill baselines in most model-benchmark settings. We find that skill evolution complements model scaling: larger models generally benefit more from evolved skills, while smaller models with skills can outperform substantially larger models without them. We also find that evolved skills transfer effectively across models and model families, and skills evolved by other models can outperform self-evolved skills. Finally, our ablation studies confirm that persistent knowledge accumulation in the wiki is critical for effective skill evolution. These results demonstrate the benefits of systematically accumulating and refining agent experience for developing reusable and transferable skills. 

---
# SWE-Prime: Fewer Trajectories, Better Performance 

**Authors**: Dewu Zheng, Ruizhe Ye, Yanlin Wang, Yang Ye, Hongyu Zhang, Ensheng Shi, Xilin Liu, Yuchi Ma, Jianxing Yu, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.27449)  

**Abstract**: To improve large language models' ability to resolve real-world software issues, prior work has focused on constructing large-scale agent trajectory datasets and performing supervised fine-tuning (SFT) on successful trajectories. However, task success does not guarantee high-quality supervision: successful trajectories may still contain ineffective, redundant, or risky steps. Directly using such trajectories for SFT can introduce noisy supervision and encourage models to imitate undesirable problem-solving behaviors. Therefore, we propose SWE-Prime, a multi-granularity, two-stage SFT data selection method that progressively filters training data at the trajectory and segment levels. Specifically, the first stage performs trajectory-level screening based on process quality, result quality, and data representativeness, selecting a high-quality and representative subset of successful trajectories. The second stage performs segment-level selection by grouping consecutive steps into semantic segments and assessing each segment based on its contribution to the final solution, learnability, and potential risks. During SFT, all segments remain in the sequence to preserve context, while only selected segments contribute to the loss computation. Experiments on SWE-Bench Pro and SWE-Bench Verified show that training on the 10% trajectory subset selected by SWE-Prime outperforms training on the full resolved dataset, yielding relative performance gains of up to 12.2% and 24.2%, respectively. 

---
# From Static to Dynamic: Benchmarking Real-World Code Review with MCR-Bench 

**Authors**: Dewu Zheng, Yanlin Wang, Xiwen Wang, Kefeng Duan, Hongyu Zhang, Xilin Liu, Yuchi Ma, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.27442)  

**Abstract**: In real-world software development, code review typically involves iterative interactions between developers and reviewers to improve software quality, making the process costly and time-consuming. Although recent work explores large language models (LLMs) for automated code review, most approaches oversimplify code review into a single-round, static decision task, which fails to capture the multi-round interactive nature and the complex problem-solving processes inherent in realistic review scenarios. To bridge this gap, we introduce MCR-Bench, the first defect state-aware benchmark designed for realistic multi-round code review. MCR-Bench covers five commonly-used programming languages and consists of 2,269 real-world multi-round code review tasks, each of which is annotated with fine-grained defect information and cross-round state labels. Each task in MCR-Bench is equipped with fine-grained defect metadata (e.g., description, type, severity) alongside dynamic state annotations, capturing the complete evolutionary trajectory of a defect throughout the multi-round process. We obtain several findings through extensive experiments on MCR-Bench with mainstream LLMs. (1) Limited overall capability: experiments reveal that mainstream LLMs exhibit limited overall performance in defect detection and defect lifecycle state tracking, with performance degrading significantly as the number of interaction rounds increases; (2) Defect-sensitive performance: LLMs' performance varies substantially across different defect types and severity levels, with semantically complex or low-salience defects being significantly more likely to be missed; (3) Underlying Failure Mechanisms: our in-depth error analysis dissects the distinct drivers of false positives and false negatives, revealing critical weaknesses such as cross-round temporal misalignment and inadequate long-range memory. 

---
# CorporateBench: Large-Scale Q&A Benchmarking with Temporal Knowledge Bases 

**Authors**: Sil Hamilton, Albert Yu Sun, Oscar J. Romero, Carl-Leander Henneking, David Mimno, Bishan Yang, Igor Labutov  

**Link**: [PDF](https://arxiv.org/pdf/2608.27391)  

**Abstract**: LLMs are increasingly able to answer complex questions about enterprise-scale document collections. But evaluation is hard: companies don't want to share internal communications, and synthetic datasets have been overly simple. We present CorporateBench (CB), a human-validated multi-task Q&A benchmark whose scale approaches the conditions LLMs encounter in corporate communication networks, with evaluation corpora surpassing 230,000 documents. CB evaluates LLMs across two dimensions (information extraction and knowledge base querying) through four synthetically generated firms ranging from 12 to 10,000 employees. Each corpus is sampled from a temporally evolving knowledge base describing a consistent world, guaranteeing cross-document logical consistency even across hundreds of thousands of documents. We evaluate five LLMs on CB, revealing increasingly poor performance as input size approaches realistic scales. CB provides LLM developers a metric for corporate communication reasoning, filling a crucial gap in the benchmarking ecosystem. 

---
# Beyond Parallel Blindness: Information Floors and Model Gaps in Block Drafting 

**Authors**: Xinwei Qiang, Xiang Fang, Chang Chen, Yue Guan, Yufei Ding  

**Link**: [PDF](https://arxiv.org/pdf/2608.27339)  

**Abstract**: Block drafters propose several tokens in one forward pass, before earlier target tokens are realised. Their rejection mixes two losses: missing within-block path information and imperfect modelling of observable information. Accepted length cannot distinguish them. We separate the two with an information floor, the minimum expected rejection at a specified conditioning order; rejection above this floor is the model gap. Estimating both from target rollouts across four domains, four open-weight targets, and a frontier API target yields three findings. First, the all-parallel floor reaches $0.286$ at the final slot on Qwen3-4B, limiting even the best proposal to $71\%$ per-slot acceptance. Second, one realised token removes $86$--$100\%$ of this floor, a locality also recovered by an independent mutual-information analysis. Third, current drafters remain far above their floors: the final-slot model gap accounts for $43$--$64\%$ of DFlash rejection and $85$--$92\%$ of DSpark's oracle-conditioned rejection. These findings separate the value of short-range conditioning from proposal quality. 

---
# BrailleBench: Investigating Multi-Criteria Braille Comprehension in Large Language Models 

**Authors**: Jinghan Zhang, Fengran Mo, Zhiyu Chen, Xiaoyan Han, Kunpeng Liu, Chang-Tien Lu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27268)  

**Abstract**: Although Large language models (LLMs) mediate access to knowledge and computational assistance, their capabilities should benefit vulnerable groups in the same way. However, it is unclear whether existing AI systems are inclusive enough for blind and deafblind users to access the same functionality through Braille, whose indicators, contractions, and digital representations introduce distinct requirements for model comprehension. To this end, we introduce BrailleBench, a benchmark for evaluating LLMs in Braille comprehension from different Criteria. BrailleBench aligns 5,570 instances from five datasets, including mathematics, commonsense, and multi-hop question answering across English and Braille Grades 1 and 2. Different configurations are designed to understand whether the systems can comprehend Braille-authored content, express answers in Braille, and complete end-to-end Braille interaction. To ensure the quality and prevent evaluation bias, the benchmark is built through a deterministic, expert-reviewed pipeline via a self-created Braille Toolkit without using any data instances generated by LLMs. We evaluate six representative LLMs from various aspects. The results reveal a persistent gap between print-English capability and Braille accessibility. Braille understanding and expression are asymmetric, where Grade 2 is especially fragile on the input side compared to Grade 1, and fully Braille requests further reduce performance. The experimental observations provide valuable guidance for the development of future Braille AI systems. All related resources in BrailleBench are publicly available for future research. 

---
# Naive Prompt Optimization: Rethinking the Need for Complex Prompt Search 

**Authors**: Yuan Chang, Xiaoqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.27266)  

**Abstract**: Efficiently improving autonomous agents across diverse tasks is central to accelerating recursive self-improvement (RSI) in agentic AI, with prompt optimization emerging as a promising approach capable of delivering performance gains comparable to those achieved by fine-tuning model weights, while reducing computational costs in both optimization and serving. However, recent developments increasingly favor unnecessarily complex prompt optimizers. We introduce Naive Prompt Optimization (NPO), a lightweight single-lineage method that iteratively revises prompts using a teacher model with rollout feedback. NPO achieves comparable or better performance than GEPA with fewer rollouts, and its advantage increases with stronger teacher models, suggesting that stronger teacher reasoning can partially substitute for optimizer-side search complexity. In interactive games, NPO remains broadly competitive with GEPA, while GRPO performs better on some tasks less amenable to prompt optimization. We also show that NPO-optimized prompts elicit similar performance improvements when applied verbatim to other student models, especially across models within the same family. Overall, our preliminary results show that simple, linear prompt optimization can rival substantially more sophisticated and complex search procedures. 

---
# What Makes Good Agentic Data? An ACE Lens on Data Generation for LLM Agents 

**Authors**: Xingshan Zeng, Zishan Xu, Boju Zhang, Yuzhou Wu, Lingzhi Wang, Jianghao Lin, Liangyou Li, Yasheng Wang, Lifeng Shang, Xin Jiang, Weinan Zhang, Yong Yu, Qun Liu, Weiwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27260)  

**Abstract**: LLM agents increasingly rely on generated interaction data to learn how to interact with external environments. Agentic data generation must maintain consistency among environments, tasks, interactions, and success signals while producing experience that is useful rather than merely abundant. Existing work spans many agent domains, but domain-centered organization and heterogeneous evaluation often obscure common generation mechanisms and conflate candidate construction with verification and selection. This work develops a two-level framework for the field. First, we represent agentic data as a common factorized object $(E,q,\tau,v)$, comprising an environment specification, task signal, interaction realization, and optional verifier. We organize generation paradigms by their primary anchor and dependency structure. Second, we formulate generation as constrained distribution design through the Accuracy-Complexity-divErsity (ACE) lens. Accuracy establishes the feasible support of grounded and internally consistent data. Within this support, Complexity places learning mass relative to the capability of a declared learner and execution configuration, while divErsity controls coverage and redundancy of data. Using this framework, we explore how prior work verifies generated experience, constructs and calibrates difficulty, and expands behavioral coverage. The literature reveals a shift toward execution-grounded accuracy, learner-relative complexity, and diversity beyond surface variation or dataset size. We further discuss broader directions and emerging trends in agentic data generation through the ACE lens, including their implications for scaling, data sources, training regimes and adaptive learning. Overall, the central challenge is not simply to generate more data, but to continually allocate valid, informative, and non-redundant experience as agents and environments evolve. 

---
# Calibrated Enough to Know, Not Calibrated to Act: Fabricated Evidence Makes LLM Agents Commit to the Unknowable 

**Authors**: Pranav Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2608.27167)  

**Abstract**: An LLM agent shown a professional-looking market panel commits to a directional call on a provably unpredictable question far more often than one asked the bare question: across 12 frontier models, commitment rises from 6.5% to 54.0% as evidence is escalated. It commits just as readily when every number on the panel is invented: fabricating the entire display, so nothing the model can see is true except the question itself, still lifts commitment from 24.5% to 36.8%, statistically indistinguishable from the 37.6% produced by genuine market data. What unlocks confident action is not information but the authority of its packaging. The failure is narrow and locatable. Incapacity is not the answer: on matched answerable questions attached to the same panels, the same models answer essentially always, at near-perfect accuracy. Nor is it belief - stated probabilities barely move across the gradient that swings action by 48 points, and score worse than a climatological baseline. Missing judgment isn't it either: asked to classify a question's knowability before acting, models call it irreducible 90% of the time and then commit on just 0.4% of those. The act/don't-act gate is what fails, and the effect is concentrated in a few models rather than universal. Because the gate is separable, it can be trained. Supervised fine-tuning of a 3B model on 540 synthetic cases, predominantly dice, coins, jars and timers, drives commitment to 0.0% on the original cases and transfers to three unseen domains. It does not survive everything: the gate holds exactly when the response format leaves room to reason, and rigid formats that remove that room leave the model confident and wrong on questions it otherwise answers correctly. The gate is trainable and context-fragile, and deployment needs both halves of that sentence. 

---
# Unifying Detection and Adaptation in Task-Free Continual Learning 

**Authors**: Dezheng Han, Anbang Zhang, Zhihao Zhu, Shuaishuai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.27070)  

**Abstract**: To mitigate catastrophic forgetting in downstream continual learning (CL) for large language models (LLMs), existing methods typically constrain parameter updates or introduce task-specific adaptation modules. However, these methods often rely on explicit task boundaries during training, limiting their applicability to realistic task-free scenarios. In this paper, we propose a \textbf{Fi}sher-guided \textbf{uni}fied (\textbf{FiUni}) framework for batch-level task detection and parameter-efficient continual adaptation. FiUni is motivated by a key observation about the Fisher information matrix (FIM) of pre-trained models: the orthogonality among the principal subspaces of its Kronecker-Factored Approximate Curvature (K-FAC) approximation, estimated from a small number of downstream task samples, can reflect the similarity between different tasks. Based on this observation, FiUni constructs FIM-derived frozen subspaces to guide low-rank adaptation (LoRA), while matching the Fisher principal subspace of each incoming batch window with historical subspaces. This enables FiUni to adaptively determine whether to reuse existing knowledge, expand a related subspace, or create a new subspace, dynamically balancing knowledge sharing and task isolation. Experiments show that FiUni can effectively infer latent batch-level task affiliations and achieve competitive performance against advanced task-aware CL methods with fewer trainable parameters. 

---
# Terrain signatures in Welsh settlement names 

**Authors**: Oktay Karakuş, Can Eyupoglu  

**Link**: [PDF](https://arxiv.org/pdf/2608.26978)  

**Abstract**: Landscapes are named, but whether names retain measurable environmental information beyond broad geographic structure is rarely tested. We analysed 3,757 Welsh settlements using a frozen, source-audited 24-element lexical framework, preregistered outcome-specific models and geographically structured validation. The central comparison contrasted 101 settlements carrying high-terrain elements (\textit{bryn} or \textit{mynydd}) with 139 carrying low-terrain elements (\textit{cwm} or \textit{pant}). High-terrain names occupied locations 24.4 m higher relative to their 2-km surroundings (95\% CI, 10.8--38.1 m; Holm-adjusted $p$ = 0.00137). The association remained positive across prespecified 1-, 2- and 5-km neighbourhood definitions and was reproduced using an independently produced elevation source (24.1 m; 95\% CI, 10.6--37.6 m). Adding terrain-name polarity to a non-lexical spatial and settlement baseline reduced geographically held-out mean squared error by 4.63\%, 6.22\% and 7.30\% under 10-, 25- and 50-km spatial blocking, respectively, although improvement varied among held-out regions. River-related names provided weaker, directionally consistent evidence, while the preregistered woodland model was non-estimable. Residual spatial structure, unresolved name language and the absence of independent external replication limit interpretation. Selected Welsh settlement-name categories therefore retain measurable information about present-day terrain within Wales, without establishing individual etymology, causal naming, historical environmental memory or transferability to other naming systems. 

---
# TEMPLAR Wales: A georeferenced environmental and toponymic dataset of Welsh settlements 

**Authors**: Oktay Karakuş, Can Eyupoglu  

**Link**: [PDF](https://arxiv.org/pdf/2608.26970)  

**Abstract**: Place names provide persistent records of how landscapes have been described and organised, but their quantitative reuse requires explicit separation between mapped places, lexical annotations and environmental measurements. TEMPLAR Wales is a georeferenced environmental-toponymy dataset comprising 3,757 settlement records across Wales. The resource links a reproducible settlement frame to deterministic lexical screening and settlement-level environmental attributes through stable identifiers. It contains 1,350 lexical detections across 1,294 settlements, generated from a frozen registry of 24 Welsh place-name elements, while retaining exact- and prefix-token matches and their provenance separately. Environmental attributes describe river and coastal proximity, elevation and local terrain context at multiple spatial scales, land cover and neighbourhood woody cover, with parallel terrain measurements derived from independent elevation products. The dataset is distributed as four relational tables accompanied by a field-level data dictionary, source-provenance register and licensing metadata. Technical validation confirms relational integrity, deterministic lexical reconstruction, documented environmental coverage, strong agreement between independent terrain sources and reproducible reconstruction of the frozen release. TEMPLAR Wales provides a reusable foundation for research in toponymy, linguistic geography, historical and environmental landscape studies, GIS and spatial data analysis without treating computational lexical detections as verified etymologies or contemporary environmental measurements as historical landscape reconstructions. 

---
# Scaling Model-Generated Distillation Data Can Make Latent Teacher Traits More Recoverable 

**Authors**: Zhichen Dong, Zhixuan Liu, Yuyu Fan, Xiangtian Li, Shuyang Zhang, Chao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.26958)  

**Abstract**: Scaling model-generated data is usually viewed as improving distillation: more examples should increase coverage, reduce noise, and produce stronger students. We show a second effect: larger datasets can make subtle teacher-specific signals easier to detect in the trained student, even when examples are off-task and never mention the trait. In a controlled setup inspired by subliminal learning, a teacher induced to express a target trait generates restricted off-task data, such as number-only completions. Students trained on different amounts of independent off-task data are evaluated in a separate domain, with matched no-trait controls isolating target-specific transfer. Our main finding is that larger independent datasets make the teacher's induced trait stand out more clearly in the student's later behavior. Other plausible traits may also strengthen with scale, but the target usually grows more. When the small-scale student already favors the target, scaling mainly amplifies that behavior; when it favors a related or salient alternative, more data can shift behavior toward the intended trait. Analyses of learned LoRA updates show a parallel trend. These effects appear across model families, trait types, multi-trait settings, and cross-model transfer. Our results suggest that scaling generated distillation data should be paired with trait-aware curation and evaluation, even when the data appears off-task or benign. 

---
# From Atomic to Agentic: Towards Interpretable Evaluation of LLMs' Agentic Mathematical Capabilities 

**Authors**: Jiayi Kuang, Yinghui Li, Yunze Song, Keyu Chen, Zhifeng Shen, Yangning Li, Yidong Wang, Di Yin, Ruizhi Qiao, Xing Sun, Kai Jin, Ying Shen, Liang Lin, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.26950)  

**Abstract**: Large Language Models (LLMs) are evolving from performing end-to-end mathematical reasoning to integrating agentic intelligence. However, most existing math benchmarks evaluate only final answers. This outcome-oriented evaluation provides limited diagnostic value for identifying process-level failures or rigorous logic, failing to guide the transformation of LLMs into robust agents. To bridge this gap, we present a process-level benchmark designed to evaluate the inherent agentic mathematical reasoning abilities of LLMs. Our framework aligns problem-solving agentic behaviors with a structured taxonomy of reusable mathematical atomic capabilities. We design a comprehensive suite of planning, action, and feedback tasks across both textual and multimodal contexts, supported by an automated pipeline that synthesizes high-quality trajectories and produces fine-grained annotations via controlled LLM rewriting. Experiments reveal that models with similar end-to-end accuracy can exhibit markedly different agentic capability profiles. This demonstrates that process-level evaluation is crucial for interpreting the true potential of LLMs and guiding the development of next-generation mathematical agents. 

---
# AraMS-28k: The Largest Publicly Released Line-Level Dataset of Historical Arabic Manuscripts with Margin and Insertion-Anchor Annotations 

**Authors**: Mohamed Guechaoui, Mohamed Diaa Zellagui, Souleyman Chaib, Sahraoui Dhelim  

**Link**: [PDF](https://arxiv.org/pdf/2608.26921)  

**Abstract**: We introduce AraMS-28k, the largest publicly released line-level dataset of genuine historical Arabic manuscripts, comprising 14 books, 3,043 pages, and 28,600 annotated text lines (27,971 main-text, 629 margin). Thirteen books are hand-copied manuscripts spanning three script traditions -- Naskh, Ruq'ah, and Maghrebi -- and one is a lithographed printed edition included to broaden format diversity. Each line is labelled as main-text or margin, and margin lines that have an unambiguous attachment point in the main text are further annotated with an insertion anchor, recovering the manuscript's true non-linear reading order at line-level granularity -- to our knowledge the first such annotation released for a historical Arabic manuscript corpus. Because reference transcriptions are fully vocalised while manuscript hands are typically undiacritised, we release both the raw diacritised transcription and a diacritic-normalised counterpart for every line. The dataset was constructed with RefLAM, a reference-grounded annotation pipeline that aligns multimodal-LLM OCR against independently sourced clean transcriptions and routes every line through human review, combining automatic verification with expert oversight. We describe the construction and quality-control process, present the annotation schema, report dataset statistics at both the corpus and per-book level, and provide baseline HTR results using Kraken and HATFormer, including a cross-script generalisation gradient from in-distribution pages to fully unseen books. AraMS-28k is released with page images, line-level annotations, and fixed train/val/test splits under CC BY-NC-SA 4.0 to support reproducible research on Arabic manuscript recognition, layout analysis, and reading-order recovery. 

---
# C-Unseen: Weak Signal Detection in Dynamic Temporal Knowledge Graphs via LLM Reasoning 

**Authors**: Yassir Lairgi, Ludovic Moncla, Khalid Benabdeslem, Rémy Cazabet, Pierre Cléau  

**Link**: [PDF](https://arxiv.org/pdf/2608.26870)  

**Abstract**: Weak signals are early, low-visibility indicators that precede significant changes before those changes become established. Existing detection methods, based on keyword frequency, topic modeling, or untyped graph topology, fail to capture the semantic and relational structure through which such signals manifest. In this paper, we propose C-Unseen, a self-interpretable framework for weak signal detection in Dynamic Temporal Knowledge Graphs (DTKGs). We define a weak signal as a rare, semantically coherent subgraph that proliferates across consecutive TKG snapshots. The framework operates through two modules: a Rare Subgraphs Extractor, in which an LLM identifies subgraphs whose content is in tension with the dominant snapshot narrative via chain-of-thought reasoning, and a Weak Signal Alerter, in which the persistence of these rare subgraphs is tracked across time steps to isolate true weak signals. Experimental results demonstrate that C-Unseen outperforms keyword-, topic-, and graph-based baselines. 

---
# SymbolLKG: Towards Verifiable Logical Reasoning via Logical Knowledge Graph and Symbolic Solvers 

**Authors**: Haizhao Fan, Yuchi Xiong, Jize Wang, Xinping Guan, Xinyi Le  

**Link**: [PDF](https://arxiv.org/pdf/2608.26836)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency in natural language understanding, yet they struggle with strict multi-step reasoning, frequently suffering from hallucinations and inconsistency. Existing solutions like Chain-of-Thought (CoT) lack rigorous verification mechanisms, while standard Retrieval-Augmented Generation (RAG) often misses the complex, structural dependencies inherent in logical tasks. To bridge this gap, we propose a Neuro-Symbolic architecture that integrates a Logical Knowledge Graph (LKG) with dynamic solver routing. Specifically, we introduce an ontology-based LKG that treats logical rules and constraints as first-class topological nodes, enabling explicit modeling of dependencies extracted from text. We further design a Logic Router to dynamically dispatch tasks to the optimal symbolic engine, which is supported by a topology-aware hybrid retrieval mechanism. Experimental results on logical reasoning benchmarks demonstrate that our framework significantly outperforms state-of-the-art prompting and RAG baselines, delivering higher accuracy and verifiable reasoning paths. 

---
# Decoupling Planning and Control for Instructable Agents 

**Authors**: Zineng Tang, Kelsey R. Allen, Sjoerd van Steenkiste, Ishita Dasgupta, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2608.26788)  

**Abstract**: Recent work shows that pre-trained, instruction-tuned vision-language models (VLMs) perform well at mapping from instructions and observations to high-level plans, but struggle to realize such plans as reliable low-latency action sequences in unfamiliar environments. At the same time, world-model controllers excel at fast observation-to-action control, but lack open-ended task guidance. In this work, we combine these strengths into a single system, Instruct-to-Act, where we train a world-model controller to act autonomously at high frequency when conditioned on sparse, higher-latency, and high-level text instructions generated by a VLM planner. To train controllers to be language-instructable, we relabel segments of controller policy rollouts with synthetic instructions and jointly optimize a behavior-cloning objective along with existing reward-maximizing and world-modeling objectives. We evaluate our proposed approach across seven embodied environments, including three multi-agent environments where VLM planners coordinate through language while trained controllers serve as their actuators. Under matched observation and action spaces, our decoupled approach consistently outperforms controller-only and direct VLM action-generation variants, preserves fast control, and lets us swap in different pretrained VLM planners without fine-tuning, while remaining competitive with strong vision-language-action and multi-agent RL baselines on six of seven tasks. 

---
# BLANC: Discovering Patent White Space via Changes in Normalized Pointwise Mutual Information Between Multi-View Clusters 

**Authors**: Shuichi Miyazawa, Kensuke Fujii  

**Link**: [PDF](https://arxiv.org/pdf/2608.26685)  

**Abstract**: Identifying white space --- the unexplored but potentially valuable regions of a patent landscape --- is essential for strategic R&D planning, yet existing methods rely on manual patent mapping or apply single-view clustering without quantitative gap detection. We propose BLANC (Blank Landscape Analysis through NPMI Conditioning), a three-phase pipeline combining (1) multi-view neural topic modeling along three semantic dimensions (application/use, novelty, inventive step); (2) Normalized Pointwise Mutual Information (NPMI) to quantify cross-dimensional cluster association; and (3) conditional detection that flags combinations whose NPMI drops when the corpus is filtered by a user-specified keyword. The drop is captured by a new metric, $\Delta$NPMI, which identifies combinations "established globally, unexplored locally." Because white space has no ground truth, we evaluate BLANC on two public USPTO corpora --- machine learning/AI (5,417 patents, CPC G06N) and glass compositions (1,982 patents, CPC C03C) --- by artificially depleting known technology combinations and testing recovery. When three-quarters of a target pair's documents are removed, BLANC recovers 34.1% (ML/AI) and 27.3% (glass) of the depleted combinations, whereas size-matched removals not aimed at them (random documents, or those of a different established combination) essentially never do: the target is never recovered in 191 decoy trials. Collapsing the three semantic views into one recovers nothing, while prior co-occurrence measures also flag the target under random removal, offering no specificity. In a proprietary case (302 float glass / glass-ceramics patents), the keyword "fluorine" reveals a fluorine surface treatment $\times$ warpage suppression candidate ($\Delta$NPMI up to 0.48) that experts had independently identified. 

---
# The Thousand-Graph Hypothesis: A Testable Hypothesis of Task-Conditioned Relation Materialization in Repository-Level Code Reasoning 

**Authors**: Fei Ding  

**Link**: [PDF](https://arxiv.org/pdf/2608.26602)  

**Abstract**: Large software repositories are often beyond model context limits. Training repository knowledge into models is costly and quickly stale, while local retrieval can miss scattered requirements, and explicit relation graphs add ongoing maintenance burden. We propose an entity-only external interface with task-conditioned relation materialization during inference. A two-layer index separates global routing from local entity focus and is evaluated on DeepSeek-V4-Flash and SWE-bench Verified. The base, one-layer, and two-layer conditions achieve 92.1%, 94.2%, and 95.6% success, respectively, under zero pre-built entity-relation edges. 

---
# J-Zero: Unified Challenger--Solver--Judge Co-Evolution from Zero Data 

**Authors**: Gyouk Chu, Myeongho Jeon, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.26582)  

**Abstract**: Self-evolving language models have recently emerged as a promising path toward superintelligence, with the advantage of reducing the cost of human supervision. While considerable progress has been made in verifiable domains, self-evolution in unverifiable domains remains substantially less explored. We propose Judge co-adaptation from Zero data (J-Zero), a unified Challenger--Solver--Judge co-evolution framework that supports self-improvement across both domains. The Challenger and Solver co-evolve through an adversarial interaction: the Challenger generates increasingly difficult tasks, while the Solver learns to produce higher-quality responses to them. In parallel, the Judge co-adapts using preference pairs whose ordering is known in advance from how each response was produced, i.e., the Solver's answer over the Challenger's, and its decomposed-and-recombined answer over its one-shot answer, rather than from the Judge's own scores. J-Zero outperforms the baselines by an average of 4.2 points on verifiable and 8.0 points on unverifiable domains, and continues to improve through at least ten iterations, whereas the baselines degrade after two. 

---
# Visual Information-Guided Parallel Decoding for Diffusion Multimodal Large Language Models 

**Authors**: Insu Lee, Wooje Park, Wonseok Shin, Jinwoo Son, Byonghyo Shim  

**Link**: [PDF](https://arxiv.org/pdf/2608.26580)  

**Abstract**: Diffusion multimodal large language models (dMLLMs) have recently emerged as a new decoding paradigm for multimodal generation. Starting from a fully masked sequence, dMLLMs progressively decode the sequence by unmasking a subset of the remaining masked positions at each step. Since the selected tokens serve as the prediction context for subsequent steps, deciding which tokens to decode is crucial to the quality of the final output. The most common strategy prioritizes tokens based on a certainty measure that tends to favor tokens frequently observed in the training data. Recent approaches instead order tokens according to their influence on subsequent predictions, but do not explicitly account for the input image. We propose the Visual Information-Guided Sampler (VIG-Sampler), which prioritizes tokens based on their attention to image tokens. We further impose a constraint that penalizes candidate tokens whose image-attention distributions are similar to those of previously selected tokens, thereby increasing the information gain of the decoded subset. Extensive experiments on 7 captioning and VQA benchmarks with 3 open-source dMLLMs demonstrate the effectiveness of VIG-Sampler, which outperforms the Info-Gain Sampler by an average of 19.3 CIDEr points across the captioning benchmarks and surpasses it on COCO Caption while using only half as many decoding steps. 

---
# DuMateBench: Evaluating Autonomous Agents in Complex Real-World Workflows 

**Authors**: Zechun Niu, Yukun Zhao, Jiaxin Zhang, Xu Shen, Jinhua Si, Han Tian, Can Xu, Yunfan Song, Jiaxin Mao, Yansong Gao, Yuchen Li, Jianmin Wu, Lingyong Yan, Shuaiqiang Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2608.26546)  

**Abstract**: Autonomous agents are increasingly adopted to complete complex, multi-tool workflows in real-world settings. However, existing benchmarks typically separate tasks by application or capability and evaluate agents in environments that are cleaner and more stable than those encountered in practice. We introduce DuMateBench, a real-session benchmark reconstructed from anonymized and privacy-screened user sessions collected from a large-scale production agent platform. Each task preserves the relevant pre-solution interaction history, persistent configurations, and workspace state, and is then validated through human verification. The resulting benchmark comprises 200 tasks spanning 8 broad scenarios and 17 fine-grained capability categories, with most tasks requiring multiple capability coordination. We execute these tasks in isolated Docker containers injected with three forms of real-world environmental complexity: Insufficient, Unstable, and Noisy, and assess performance using a hybrid deterministic and LLM-as-Judge evaluation protocol. Experiments across five representative autonomous-agent frameworks paired with four state-of-the-art LLMs reveal substantial gaps in strict task completion. Complementary robustness, efficiency, and diagnostic analyses further show that performance under environmental perturbations is jointly shaped by the capabilities of the LLM and the surrounding agent framework. The code and data are publicly available at this https URL. 

---
# A Single Suffix to Break Them All: Basin-Aware Jailbreaks for Merged Model Families 

**Authors**: Yu Zhe, Yixin Tan, Junhao Wei, Wang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.26506)  

**Abstract**: Model merging enables combining multiple fine-tuned models without additional training, but its safety implications remain poorly understood. Prior work primarily attributes merging risks to unsafe constituent models, implicitly assuming that merging individually aligned models preserves safety. In contrast, we show that model merging reveals a previously overlooked jailbreak risk rooted in the pretrained foundation model, even when all constituent models are individually safety-aligned. Motivated by this observation, we study a new threat setting where an attacker constructs jailbreak prompts that generalize across merged models sharing the same pretrained backbone, without access to the exact merging coefficients or constituent checkpoints. To exploit this phenomenon, we propose \textbf{Basin-Aware Jailbreak (BAJ)}, which formulates jailbreak generation as a min--max optimization over the merging space to produce transferable adversarial suffixes across merged model families. Experiments across diverse backbones and merging settings show that BAJ achieves consistently high transfer success rates and remains effective under existing defenses. 

---
# Zero-Shot Self-Orchestration with Ledger-Based Control for Improved LLM Coding Performance 

**Authors**: Victor Gao, Vida Khosrowshahi, Ali Khosrowshahi, Xihao Sun, Juhyun Lee, Simon  

**Link**: [PDF](https://arxiv.org/pdf/2608.26480)  

**Abstract**: Multi-agent large language model systems are widely reported to beat single-model baselines, but the evidence is mixed, and comparisons are usually confounded: pipelines change token budgets, tool calls, and prompts simultaneously, so an aggregate gain rarely reveals what actually helped. We investigate the effect of introducing the manager-worker scaffold over a shared filesystem workspace, with no training and no per-benchmark tuning, measured against the same model answering in a single pass. Across nine models -- five open-weight, spanning 9B to ~2.8T parameters, and four frontier closed models -- on the 100 latest hard LiveCodeBench problems, the scaffold's benefit is real but conditional: large and statistically significant for some (Qwen3.8-27B +23.4, GPT-5.6-Luna +10.6 and GPT-5.6-Terra +8.0, each over five paired passes; Kimi-K3 +30.4 and Minimax-M3 +11.0 over five paired passes with reasoning off, both at $p < 10^{-4}$, and +42 and +12 in a single pass at a 128k cap) and null or negative for others (Qwen3.6-35B -1 to -9 with reasoning off). With the manager, Opus-5 achieves the highest score in the study at 91% in one pass. Running a manager roughly triples the token bill, but it buys accuracy more cheaply than moving to a larger model does: GPT-5.6-Terra with a manager nearly matches Fable 5's single-call accuracy (85.0 against 87.4, $p = 0.59$) at a fifth of the price (\$11.71 against \$61.11 per 100-problem pass, $p < 10^{-4}$), and the Qwen-27B arm does it for \$51.75 on weights anyone can self-host. Our transcript analysis finds several mechanisms behind the gains, of which two recur: context management, in which short worker calls and shared notes organize state and reduce truncation, and problem decomposition. Improvements are modest for large models with reasoning enabled, but larger for some models with reasoning disabled and for smaller models with reasoning enabled. 

---
# Diff Mining: Logit Differences Reveal Finetuning Objectives 

**Authors**: Greg Kocher, Robert West, Clément Dumas, Julian Minder  

**Link**: [PDF](https://arxiv.org/pdf/2608.26462)  

**Abstract**: Finetuning has become the gold standard for refining existing behaviors and inducing new ones in language models, yet it often remains unclear exactly which behaviors emerge during this process. As models grow ever more capable, understanding finetuning better becomes increasingly important, particularly since unwanted behaviors may arise during finetuning. In this paper, we introduce Diff Mining, a simple yet effective framework for identifying what a finetuned model has learned by comparing its logits to those of its base model. Diff Mining effectively surfaces salient tokens that are amplified in the finetuned model, serving as a fingerprint of its training -- even on text unrelated to the finetuning domain. Unlike many existing model diffing methods which require model internals, Diff Mining only needs access to output logits and scales to large models. The framework consists of two modular stages: (i) extracting per-context logit differences between the finetuned and base models on a reference corpus, and (ii) aggregating the resulting signals to construct an interpretable token set representing the finetune. For aggregation, we explore both a simple Top-K frequency method and a Non-negative Matrix Factorization (NMF)-based approach for disentangling multiple finetuning objectives into distinct token clusters. Empirically, Diff Mining succeeds across diverse settings: on finetune domain detection, it significantly outperforms state-of-the-art model diffing methods both in identifying relevant tokens and in downstream performance when an interpretability agent is given access to the extracted token set; on models with injected biases, it identifies more than one third of the biases without targeted probing. Overall, our framework shows promise in developing auditing tools to detect finetuning objectives. 

---
# Don't Overthink, Don't Underthink: Toward Adaptive Reasoning in Agentic AI 

**Authors**: Md Jueal Mia, M. Hadi Amini  

**Link**: [PDF](https://arxiv.org/pdf/2608.26442)  

**Abstract**: Recent advances in Large Language Models (LLMs) have shown that increased inference-time reasoning can improve performance on complex tasks. However, many existing approaches rely on fixed or preallocated reasoning controls, such as fixed token budgets, pre-execution difficulty estimates, or activation-space interventions, and are often evaluated on standalone reasoning benchmarks rather than full agentic workflows. These assumptions may not hold in agentic AI systems, where reasoning requirements evolve dynamically through planning, tool use, memory retrieval, and agent-to-agent interactions. Consequently, reasoning can become either excessive or insufficient, resulting in unnecessary computation, increased latency, planning drift, excessive tool use, or incomplete solutions. We argue that a major challenge for next-generation agentic AI is not merely how much reasoning a language model should perform, but how it should allocate reasoning according to evolving task demands. We characterize over-reasoning and under-reasoning as recurring failure modes of misallocated reasoning and evaluate them on MATH-500 and the GAIA public validation benchmark. Using tool-decision latency, token consumption, token-limit exhaustion, and answer correctness, our results suggest that cases classified as over-reasoning are associated with higher computational cost without proportional accuracy gains, whereas cases classified as under-reasoning are consistently associated with incorrect or incomplete solutions. These findings motivate future research on adaptive reasoning mechanisms for agentic AI. 

---
# SpeechGym: An Audio-Native Gym for Training Voice Agents via Reinforcement Learning 

**Authors**: Jiajun Fan, Jingyuan Li, Prashanth Gurunath Shivakumar, Jia-Hong Huang, Qi Luo, M. Maruf, Ivan Bulyko, Ge Liu, Roger Ren  

**Link**: [PDF](https://arxiv.org/pdf/2608.26432)  

**Abstract**: Voice agents must call tools and hold multi-turn dialogue entirely through speech, yet the dominant paradigm trains them in text. Existing frameworks either cascade TTS and ASR around a proprietary voice API, where gradients cannot flow and per-call cost makes on-policy reinforcement learning prohibitive, or stay in text: they measure voice agents but cannot improve them. We present SpeechGym, an audio-native agentic environment in which two omni-modal models converse in native audio, with no external ASR or TTS and no API boundary, over the unmodified tasks, tools and success check of an established text agentic benchmark, so that the interaction modality is the only variable and the loop stays local and trainable end to end. Audio agentic capability does not follow from audio understanding. The failures speech introduces are perceptual rather than reasoning deficits: the agent picks the right tool and the right argument slot but fills it with a value misheard from the waveform, and that single error cascades into a failed call, a retry of the same call, and a wasted step budget. A second failure is behavioural: under an insistent caller the agent performs an unauthorised write and ends the episode believing it helped. Both are trainable, because the environment labels them for free: a call with a misheard argument fails against the database while a correct one succeeds. The obstacle is sparsity, not signal. Outcome-only GRPO is gradient-starved here, since almost every rollout group fails identically, while a per-turn process reward crediting each successful tool call restores variance to nearly every group. Trained this way, the agent transfers with no further tuning to an independently implemented voice benchmark, more than doubling task success and carrying an open-weights model from last place to second on that leaderboard, while using fewer turns and tokens than before training. 

---
# The Latent Diagnostic Taxonomy: A Framework for Constructing Classifiers and Diagnosing Their Decisions, Applied to Prompt Injection Detection 

**Authors**: Jaturong Kongmanee, Smile Thanapattheerakul  

**Link**: [PDF](https://arxiv.org/pdf/2608.26423)  

**Abstract**: This paper proposes a framework for constructing a classifier as a safeguard layer, and for developing a complementary diagnostic that identifies which of the classifier's confident decisions can be trusted. This framework, the Latent Diagnostic Taxonomy, consists of (i) constructing a dimensionality-optimized classifier, in which the embedding dimensionality is empirically selected via cross-validated performance rather than fixed a priori, (ii) locating a relatively small set of latent support vectors (~ 29% of total training examples) representing influential prompts for identifying tokens that alter the classifier's predicted labels, and (iii) utilizing such tokens and their associated attack magnitudes for constructing a diagnostic taxonomy. This diagnostic taxonomy provides an end-to-end guideline for flagging prompts that require different treatments: rely Safely on the classifier's decision; flag Heuristic Bias and Heuristic Override cases; route Insufficient Context cases for further human/safety review. Applying the framework to a classifier trained on a public prompt injection dataset, we find that a substantial fraction of its confident decisions (~ 77%) are not robust to removing a single token, and that this brittleness separates into two distinct failure patterns: a confidence calibration failure and a genuinely exploitable shortcut. For each zone of the taxonomy, we also recommend strategies for remediating diagnosed prompts. We illustrate the framework as a series of steps, demonstrating how each step operates. 

---
# Assessing the Downstream Utility of Evidence-Aware Retrieval in RAG 

**Authors**: Utshab Kumar Ghosh, Debayan Mukhopadhyay, Shubham Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2608.26379)  

**Abstract**: Retrieval evaluation for retrieval-augmented generation (RAG) is increasingly designed around whether retrieved passages contain evidence that can support generation, rather than topical relevance alone. We study whether this closer alignment with downstream evidence needs also makes retrieval evaluation more useful for the decisions built from it.
Across five retrieval benchmarks and an end-to-end TREC RAG 2025 setting, we examine an answer-support signal in four roles: comparing retrievers, guiding retrieval training and system selection, predicting downstream answer quality, and filtering the evidence supplied to a generator. The signal changes retrieval rankings, but its downstream value is not uniform. It does not reliably improve retriever training; the benefit of using it for system selection depends on how the generator is instructed to use the retrieved evidence; and retrieval scores based on it do not robustly predict answer quality on unseen topics. In a direct evidence intervention, human annotators confirm that filtering preferentially preserves passages containing useful answer evidence, yet different answer evaluators reach different conclusions about whether the resulting answers improve.
These results show that making retrieval evaluation more closely reflect the evidence needed for generation does not by itself make every downstream use of that evaluation more reliable. RAG evaluation methods should therefore be assessed with respect to the particular comparisons, decisions, and conclusions they are intended to support. 

---
# Invocation-Level Reliability of Tool-Using Agents 

**Authors**: Afiya Noorain, Subhranshu Mohanty, Amritesh Banerjee, Abhijit Dasgupta  

**Link**: [PDF](https://arxiv.org/pdf/2608.26189)  

**Abstract**: Tool-using agents fail two ways: choosing the wrong tool, or forming wrong arguments, and an early failure of either kind can silently corrupt everything downstream. We measure a correct-invocation rate that separates the two, under both a clean teacher-forced context and the model's own free-running context, on five open-weight models over contamination-free multi-step tasks (depths 1-8). By depth 6, roughly 70% of a model's own clean-context capability is lost to its own earlier mistakes (L6 = 0.686, 0.684). Our central finding concerns the measurement itself. Under exact-match scoring against a fixed gold trajectory, a propagation model's severity and recovery parameters are not merely hard to estimate - they are fixed by the scoring rule. Severity is forced to its boundary (0 of 869 poisoned steps correct); recovery is structurally unobservable (0 of 580 poisoned steps returned on-track, against an expected 0.0058 by chance). Both follow from one mechanism: post-divergence, the gold value is generated by tool constants the model never sees, so it is information the model cannot derive. A fit run anyway returns 0.92 and 0.73 for a quantity that is exactly 1.000 - confident numbers for a parameter the scoring rule already determined. We give the mechanism and a remedy, conditional-on-state scoring, applied retrospectively to cached completions at zero additional cost, which un-pins severity to interior estimates excluding zero (+0.149, +0.316). 

---
# Knowledge Cards: Structured Knowledge for AI Systems 

**Authors**: Liliana Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2608.26176)  

**Abstract**: AI systems whose outputs inform real decisions, and increasingly consequential ones, require something that current documentation practice does not provide: a structured, inspectable representation of the knowledge they need to ground, contextualize, and reason about those decisions, ideally reviewed and signed off by a domain expert. Established documentation artefacts already capture important aspects of an AI system. Model cards describe how a system behaves, data cards describe what it was trained on, and system cards describe the risks of a deployed system. None of them addresses the layer between inputs and outputs, more precisely, the concepts a system holds, the relationships it models, and the patterns of reasoning it applies. For pattern-recognition tasks this gap is tolerable. For agentic AI, where systems act on their conclusions, it is the step that most often separates a promising proof of concept from an operational solution an organisation can rely on. This paper introduces the Knowledge Card, a structured artefact that captures validated knowledge about a single bounded concept in a form that experts can review, organisations can audit, and AI systems can reason over. For one concept, such as a specific failure mode, a compliance obligation, or a process decision, a Knowledge Card records the entities and relationships involved, the reasoning that connects them, the conditions under which that reasoning no longer holds, and the provenance of every claim, all grounded in a formal domain ontology and signed off by a domain expert. Initial prototype cards have been built in the energy and pharmaceutical domains. The schema is released as a public draft for community engagement. 

---
# Mitigating Fabrication in Multi-Stage LLM Pipelines for Hiring: An Empirical Evaluation of Prompt Guardrails and Human-in-the-Loop Checkpoints 

**Authors**: Hiroko Takano  

**Link**: [PDF](https://arxiv.org/pdf/2608.26171)  

**Abstract**: Multi-stage LLM hiring pipelines (resume improvement, interview question generation, answer feedback) can fabricate credentials, inflate qualifiers, and invent experience. We evaluate two mitigations, prompt guardrails and human-in-the-loop (HITL) checkpoints, against a fully automated baseline. In a controlled experiment (10 synthetic resumes x 2 job descriptions x 3 repetitions x 3 conditions; 180 runs), the baseline (C1) produced at least one unsupported claim in 96.7% of outputs (mean 6.80 findings/output). Prompt guardrails (C2) reduced finding density by 86% (6.80 to 0.92/output), but 50.0% of outputs still contained a fabrication, showing prompt-level mitigation alone is insufficient. A human checkpoint after resume improvement (C3) eliminated all identity fabrications, reduced finding density by 59% (6.88 to 2.82/output), reduced item-level fabrication from 96.7% to 75.0% (p=.022), and cut capture of JD-embedded trap requirements from 47% to 2% (vs. 5% under the guardrail). An exploratory analysis of multi-specialty resumes shows contamination rising monotonically with domain distance between specialties, suggesting career changers are especially exposed. The reviewer in this study caught all flagrant fabrications, but subtle qualifier drops and plausible new claims survived review roughly half the time (54.5% removal). Neither mitigation degraded the deliverable: claim retention exceeded 99% under both. The interventions are complementary: the guardrail eliminates unprompted additions and qualifier inflation cheaply, while the checkpoint gives near-categorical guarantees against the most severe failures, invented identities and JD-baited claims. These results support a layered architecture combining guardrails with a human checkpoint. A supplementary run with a newer-generation model (90.0% baseline fabrication rate) suggests the problem is not resolved by model progress alone. 

---
# SLM-Conditioned Hierarchical Relation Routing for Labeled Property Graph Learning 

**Authors**: Michal Podstawski  

**Link**: [PDF](https://arxiv.org/pdf/2608.26132)  

**Abstract**: Labeled property graphs combine relational structure with heterogeneous textual and categorical properties attached to both nodes and relationships. Conventional graph neural networks typically represent these properties as static feature vectors, limiting their ability to determine which semantic evidence should influence message propagation for a particular prediction target. We propose SLM-Conditioned Hierarchical Relation Routing, an architecture that integrates a small language model directly into graph message selection. A topology GNN provides a stable structural representation and prediction anchor. For each target node, incident messages combine the neighbor's structural state, node-property encoding, relationship-property encoding, and relationship type. A parameter-efficient SLM processes structured graph soft tokens and produces a target-conditioned routing query. This query first selects relevant messages within each relationship type and subsequently routes information across relation-level summaries. The resulting representation provides a bounded residual update to the topology anchor, preserving structural evidence while allowing contextual semantic information to modify the prediction. The architecture supports interpretable analysis at both the neighbor and relationship-type levels and provides a general mechanism for integrating language-derived semantics into property-rich graph learning. 

---
# Graph-Based Modeling of Financial Volatility Dynamics 

**Authors**: Chuanzhen Wang, Alice Zhang, Wei Chen, Michael Brown  

**Link**: [PDF](https://arxiv.org/pdf/2608.26127)  

**Abstract**: Accurate forecasting of realized volatility ($RV$) is crucial for risk management and derivatives pricing. Although the implied volatility ($IV$) surface offers rich informational content, prevailing methods that treat it as a static image fail to capture its inherent dynamics. To overcome this limitation, we propose the Finance-Aware Graph Spatio-Temporal Network (FA-GSTN), a novel architecture that reframes $RV$ forecasting as modeling the evolution of a structured financial object. FA-GSTN builds a spatio-temporal graph sequence from the $IV$ surface, where nodes correspond to grid points and edges encode adaptive spatial (intra-day) and explicit temporal (inter-day) dependencies. The model incorporates domain knowledge through finance-aware node features (e.g., option Greeks) and tackles high-frequency noise via a multi-scale temporal smoothing gate coupled with an adaptive robust loss function. Comprehensive evaluations on a large-scale equity options dataset show that FA-GSTN sets a new state of the art, delivering superior predictive accuracy ($R^2$ up to 0.473). It also demonstrates remarkable data efficiency, substantially outperforming strong Vision Transformer baselines when trained on only one year of data ($R^2$: 0.372 vs. 0.315). Furthermore, the model exhibits enhanced robustness during periods of market stress, such as 2020--2021. Ablation studies confirm the vital roles of the spatio-temporal graph structure, finance-aware components, and integrated noise-handling modules. Our work underscores the substantial benefits of explicitly modeling temporal dynamics and infusing financial inductive biases for accurate and robust volatility forecasting. 

---
# CIFQA: A Deterministic Tool-Grounded Multi-Agent LLM Framework for Financial Query Answering 

**Authors**: Kunjesh Parekh, Anil Kumar Tiwari, Divya Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2608.26114)  

**Abstract**: Calculation-intensive financial question answering requires exact reasoning over structured rates, temporal conditions, numerical formulas, and rule-based constraints. Although Large Language Models (LLMs) perform strongly on natural language tasks, they often produce numerically incorrect yet plausible answers when solving multi-step financial calculations. To address this limitation, we introduce CIFQA (Calculation-Intensive Financial Query Answering), a deterministic tool-grounded multi-agent LLM framework for financial question answering. CIFQA separates language understanding from numerical execution by assigning specialized agents to query interpretation, routing, parameter extraction, computation planning, and response generation, while deterministic Python-based tools perform financial calculations and rule application. We instantiate CIFQA for fixed deposit query answering and evaluate it on a curated benchmark of fixed deposit queries. CIFQA achieves 95.54% accuracy on calculation-intensive queries and 90.87% overall accuracy, substantially outperforming direct LLM baselines even when provided with complete formulas, rate cards, and benchmark instructions. Ablation studies show that deterministic components such as exact rate lookup, tenure computation, rolling-year adjustment, and premature-withdrawal logic are critical contributors to performance. Notably, a 17B open-source backbone operating within CIFQA outperforms substantially larger frontier models evaluated with the same financial information, demonstrating that architectural design is a more important determinant of numerical reliability than model scale. While evaluated on fixed deposit queries, CIFQA provides a generalizable framework for calculation-intensive financial reasoning tasks. 

---
