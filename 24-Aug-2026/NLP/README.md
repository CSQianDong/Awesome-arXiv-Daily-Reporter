# Move by Move: Measuring and Steering How LLMs Conduct Psychotherapy 

**Authors**: Afonso Baldo, Hugo Pitorro, Areti Vassilopoulos, Anabela C. Areias, Maya D'Eon, Fabíola Costa, Ricardo Rei, Nuno M. Guerreiro  

**Link**: [PDF](https://arxiv.org/pdf/2608.21325)  

**Abstract**: Users increasingly turn to large language models for emotional support, yet little is known about how these models actually conduct a psychotherapy interaction. We introduce an ontology of ten therapeutic moves: compact, function-based categories grounded in the MULTI-60 inventory, validated through an annotation campaign with five licensed psychologists, and scaled with a judge-based approach that matches expert agreement. Applying it to real counseling transcripts and model-led sessions, we compare the move distributions between human clinicians and a panel of frontier models. Models over-use inquiry at up to three times the human rate, neglect psychoeducation, and are strongly context-anchored: they carry forward strategies initiated by a human clinician but rarely initiate them themselves. Exposing the ontology as a set of tools roughly halves the mean deviation from the human move distribution and improves turn-level alignment with human therapist by 7-9 percentage points, without any fine-tuning. 

---
# Prompt-Model Interaction Reaches the Fixed Points: A deterministic, task-free structural readout -- and the factorizations of it that failed 

**Authors**: Nicolás Vera Zúñiga  

**Link**: [PDF](https://arxiv.org/pdf/2608.21315)  

**Abstract**: That a prompt's effect is not a property of the prompt is established: prompts optimised for one model degrade on another, and rankings reorder under neutral reformatting. That evidence is about task accuracy, which cannot say whether the interaction is a fact about task machinery or about the conditional distribution itself. We ask on a readout with no task in it: the fixed-point structure of the short-window argmax map x_{t+1} = argmax_x p(x | x_{t-1}, x_t), censused from 96 starts. It is deterministic, so nothing can be helped or hurt, and it exists only at short windows -- four of six models lose it entirely by window 16 -- so everything here concerns how a model reads a fragment. Two results. First, the interaction reaches this readout at full magnitude: nine tokens of conditioning move the fixed-point fraction across most of its range, change a four-way structural class, and reorder models, while instruction tuning worth 60.5 IFEval points moves the class by zero. Second, nothing we proposed carries it. Prefix length fails: the effect is not monotone. Four phenomenological factors -- prose-versus-markup, a universal direction, bidirectionality, instruct-resistance -- were each withdrawn within one run of being proposed, dissolved by widening the sample. And the nearest mechanistic account, attention-sink dominance of early tokens, predicts the sign of the shift on 2 of 5 models -- chance -- while a length-by-content cross shows it holds on real text and fails on our probe's uniformly random input, so we are outside its regime, not against it. One fixed nine-token prefix drives four models toward 0 and two toward 1; the bidirectionality survives in-distribution starts. On this readout the unit of explanation is the prompt-model pair. The recurring error it caught in us has a name: a criterion with a shape applied to a quantity with no room to vary. 

---
# Memory Augmentation Unlocks Efficient Chain-of-Thought Reasoning 

**Authors**: Simeng Zhang, Yilong Chen, Wenyuan Zhang, Zhenyu Zhang, Yao Chen, Junyuan Shang, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21265)  

**Abstract**: Large language models often rely on Chain-of-Thought (CoT) reasoning to solve complex tasks, but verbose reasoning traces introduce substantial inference overhead. CoT compression shortens generation, yet aggressive compression may disrupt logical coherence and degrade performance. We formalize this trade-off as the \textit{Context-Generation Substitution Law}, where explicit reasoning context substitutes for part of decode-time generation. Based on this principle, we propose \textit{Memory-Augmented Compression}, a training-free framework that constructs reusable reasoning memories from historical traces and retrieves them as prefill-side scaffolds. Rather than using raw demonstrations, these memories summarize reusable reasoning patterns, key constraints, and critical operations to compensate for information lost during compression. Experiments show that Memory consistently improves prompt-based Chain-of-Draft (CoD) compression across mathematical reasoning, complex reasoning, and science question answering tasks, yielding accuracy gains of 21.4, 28.0, 29.5, and 6.61 points over CoD on GSM8K, MATH, BBH, and MMLU-Sci, while achieving a 1.14--1.49$\times$ latency speedup over standard CoT. Memory is also compatible with token-level, reasoning-trace-level, and inference-state compression mechanisms. Further analyzes show that the gains come from relevant reasoning memories rather than simply increasing context length. 

---
# EnSI-RAG: Entity-Structure-Indexed Retrieval-Augmented Generation for Long-Document Question Answering 

**Authors**: Xuanyu Meng, Jiashuo Sun, Jash Rajesh Parekh, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.21252)  

**Abstract**: Question answering (QA) over long, connected documents remains challenging because relevant evidence may span multiple entities and their relationships. Existing retrieval-augmented generation (RAG) methods typically index documents as raw chunks and retrieve them through embedding similarity. Their performance degrades when chunk boundaries separate entities from supporting evidence or when a question requires multi-hop reasoning across the corpus. We propose EnSI-RAG (Entity-Structure-Indexed Retrieval-Augmented Generation), a framework that constructs a query-independent, entity-centered index. Each record (e, t, k, v) represents an entity e, its type t, a semantic category k in {property, relation, aspect}, and a value v, while retaining links to the original source passages. At query time, these records serve as retrieval handles, and an LLM synthesizes the retrieved passages into the final answer. This design separates evidence localization from answer synthesis while preserving traceable source evidence. Across Loong and Oolong, EnSI-RAG achieves an average accuracy of 78.24. Relative to the published baseline scores used as references, this is 6.62 points higher, suggesting its effectiveness across these settings. The code is available at this https URL. 

---
# Benchmarking Patent Drafting from Inventor-Style Disclosures 

**Authors**: Lekang Jiang, Wenjun Sun, Stephan Goetz  

**Link**: [PDF](https://arxiv.org/pdf/2608.21249)  

**Abstract**: While recent large language models (LLMs) have achieved promising results on individual patent drafting tasks, they fundamentally fail to investigate the core challenge of real-world patent drafting: generating a complete and legally coherent patent application directly from early-stage invention materials. Prior work predominantly assumes later-stage, highly structured, or already legalistic inputs. However, real patenting workflows begin with informal, de-legalized disclosures authored by inventors. To bridge the gap, we introduce Dis2Pat, a disclosure-to-patent dataset that reflects realistic patenting workflows by requiring the generation of complete patent applications directly from inventor-style, de-legalized disclosures. Given the inherent difficulty of long-form, legally constrained patent drafting and the strong privacy requirements, we further propose a strong baseline named Patent-MAF. It is a multi-agent framework for locally deployable patent drafting. Benchmark results reveal that current LLMs exhibit limitations in patent drafting, while Patent-MAF provides a strong baseline that consistently outperforms evaluated open-source models and remains competitive with large closed-source models. 

---
# Affective Context Amplifies Sycophancy in LLM Responses 

**Authors**: Jiayi Li, Sanjana Menon, Brett Frischmann, Shomir Wilson, Sarah Rajtmajer  

**Link**: [PDF](https://arxiv.org/pdf/2608.21242)  

**Abstract**: As conversational companions, large language models (LLMs) often have access to users' emotional states. We study how this affective context modulates LLM sycophancy in subjective, evaluative interactions, where users share actions or opinions that invite feedback. Drawing on ingratiation theory, we measure sycophancy as the divergence between a model's independent evaluation and its user-facing response, elicited by presenting the same content as either a third-party account or the user's own disclosure. Across seven LLMs and two Reddit datasets (r/AmItheAsshole and r/TrueUnpopularOpinion), we find that this divergence is systematic and strongly one-directional. User-facing responses consistently soften or withhold negative or oppositional judgments. Affective context further amplifies this divergence with negative states, particularly loneliness and distress, producing the largest effects. These findings suggest that affective context functions as a vulnerability signal that suppresses critical feedback when users may need it most, often through evasive sycophancy, in which models retreat toward non-committal responses rather than outright agreement. 

---
# RARE: Decoupling Representation Steering from Expert Routing in Mixture-of-Experts Language Models 

**Authors**: Zhibo Zhang, Zhen Ouyang, Ling Shi, Kailong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.21236)  

**Abstract**: Representation engineering offers a lightweight means of controlling language-model behavior by modifying intermediate hidden states, but its direct application to Mixture-of-Experts (MoE) models introduces a structural mismatch. We first verify this failure mode through a series of empirical studies and find that preserving clean routing substantially recovers steering performance and that routing is more sensitive to semantic content than to behavioral changes under controlled content. Motivated by these findings, we introduce RARE, a router-agnostic representation engineering framework for MoE language models. RARE projects arbitrary behavioral perturbations onto the null space of the router matrix, thereby removing router-visible components, and further corrects routing drift propagated to selected downstream layers. To decide the best perturbation estimator in this framework, we evaluate five estimators on six heterogeneous open-weight MoE models across three steering scenarios: harmfulness, truthfulness, and factual editing. On harmfulness steering, RARE reaches an average attack success rate of 53.3% while retaining 67.8% MMLU accuracy, yielding a stronger aggregate effectiveness--utility trade-off than baselines. It further improves average TruthfulQA MC1 accuracy from 41.0% to 58.6% and CounterFact efficacy from 16.8% to 96.3%. These results support routing consistency as an important architectural consideration for adapting representation engineering to MoE models. 

---
# No PUN Intended: Plausible Unknown Names for Person-Centred LLM Evaluation 

**Authors**: Dimitri Staufer, David Hartmann, Ibrahim Baroud  

**Link**: [PDF](https://arxiv.org/pdf/2608.21206)  

**Abstract**: Person names are widely used as prompt variables in LLM evaluations of factuality, privacy leakage, bias and abstention, but when a name's evidential status is uncontrolled, measurements may conflate memorisation, retrieval, name priors and wrong-person attribution. We operationalise an unknown name as one with plausible First-Last form, no indexed full-name evidence, and no ambiguity signals under a documented validation run, and introduce PUN (Plausible Unknown Names), a protocol for constructing and validating such names, combining Wikidata-derived components, web-enabled LLM screening, and controlled search revalidation. We report acceptance rate, reproducibility, ablations, and a 204-participant human study, finding accepted names are more name-like than controls while participants recover person evidence in only 3% of cases. We release 300 names with comparison controls. 

---
# When the Feature Pool Goes Algorithmic: Extending Mufwene's Ecology of Language Evolution to LLM-Mediated Exposure 

**Authors**: Kunmei Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.21088)  

**Abstract**: Mufwene's ecological model locates language evolution in competition among variants contributed by individual idiolects and in speakers' selection from linguistic material made available through interaction. Large language models (LLMs) complicate this architecture without requiring the locus of selection to move away from human speakers. This article argues that LLMs are best treated as distributional mediators: they aggregate language produced across human populations, transform its distribution through training and post-training, and redistribute model-specific outputs at scale. I call the resulting ecological process algorithmic reweighting of the speaker-accessible distribution: model mediation can alter the relative frequencies with which competing variants reach human selectors. Emerging evidence on model-specific linguistic profiles and lexical uptake is consistent with parts of this pathway, but does not establish inevitable convergence. Human social evaluation remains decisive: model-associated forms may diffuse and become conventionalized, become socially recognizable as 'AI-like' and subsequently avoided, or fail to diffuse in the first place. The proposal extends Mufwene's feature-pool ecology one step upstream of speaker selection and yields testable predictions about uptake, model-version effects, convergence, and social reversal. 

---
# Jokes Aside: Measuring the Semantic Distance of Double Meanings 

**Authors**: Fabio De Ponte  

**Link**: [PDF](https://arxiv.org/pdf/2608.21087)  

**Abstract**: Large language models have significantly enriched the toolkit for computational humor research, particularly in the automated generation of jokes and puns. A key innovation, contextual embedding vectors, offers new opportunities to revisit and refine earlier hypotheses. Notably, Petrovic and Matthews (2013) proposed a joke generation model based on the scheme "I like my X like I like my Y, Z" (e.g. "I like my ice like I like my dreams, crushed"). They suggested that joke hilarity increases with: a) frequent association of Z with X and Y, b) rarity of Z, c) ambiguity of Z, and d) meaning distance between X and Y. Building on this, Winters et al. (2019) proposed a set of metrics, based on Google Ngrams and Word2Vector. In this work, three out of their five metrics are revisited with word embeddings: obviousness, compatibility, and comparison. Another measure, symmetry, defined as closeness of Z to both X and Y, is introduced here for the first time. Two models were used to collect the embedding vectors (OpenAI text-embedding-3-small and MiniLM all-MiniLM-L6-v2) on three datasets: JokeJudger, Expunations, and rJokes. The last two datasets, Expunations, and rJokes, were expanded by adding paired sentences that captured the ambiguous expression at the core of each joke in its two different meanings. Results revealed that models trained on the proposed metrics performed poorly in predicting humor ratings: on JokeJudger, the best model achieved 57.1% accuracy, below the 61.5% baseline, while performance on Expunations and rJokes was even lower. Nevertheless, the symmetry metric seems consistently associated with higher-rated jokes, suggesting it may capture a necessary -though not sufficient- property of humor. 

---
# PromptResponse: Optimizing Prompts for LLM Coding Tasks 

**Authors**: Erik Thureck, Robert Kühnen, Tim Jacobowitz  

**Link**: [PDF](https://arxiv.org/pdf/2608.21074)  

**Abstract**: Large language models (LLMs) are increasingly used in research workflows and software development pipelines, yet their output remains sensitive to input prompt variations. This paper presents $\unicode{x00AB}$PromptResponse$\unicode{x00BB}$, a controlled study examining how formatting and LLM-based tuning of coding task prompts affect the resulting code's performance, efficiency, and stability. Using five semantically identical yet syntactically distinct variants of the HumanEval dataset$\unicode{x2014}$baseline, JSON, Markdown, YAML, and an LLM-tuned version$\unicode{x2014}$we had GPT-4o solve its coding problems over 8200$\unicode{x00A0}$executions. Our results show that consistent formatting$\unicode{x2014}$especially JSON$\unicode{x2014}$improves generation efficiency and syntactic stability, with minor gains in task performance. Conversely, the LLM-tuned prompts resulted in significantly degraded task performance without significant improvements in any other dimension. These findings suggest that low-effort reformatting alone can yield measurable improvements, while tuning must account for model alignment. We conclude our work with providing a set of practical recommendations informed by our results as well as releasing our dataset variants and evaluation pipeline for future work. 

---
# Evidence-Consistent Generative Detection under Scenario-Level Distribution Shift 

**Authors**: San Kim, JinYeong Bak  

**Link**: [PDF](https://arxiv.org/pdf/2608.21043)  

**Abstract**: Conventional in-distribution evaluation can overestimate robustness when training and test data share recurring task-specific patterns or surface cues. This risk is especially relevant in social-engineering fraud detection, where attackers can preserve malicious intent while changing the scenario, impersonated entity, or wording. We study this problem as scenario-level out-of-distribution (SL-OOD) detection for SMS and voice phishing, where entire attack scenarios are held out from training while the label space remains fixed. This setting tests whether models can generalize to unseen attack scenarios using decision-relevant evidence rather than familiar scenario-specific cues. Using this SL-OOD evaluation, we find that high in-distribution performance does not reliably predict held-out robustness across feature-, encoder-, and decoder-based baselines. We interpret this gap as scenario memorization: reliance on recurring scenario-specific lexical or entity cues rather than decision-relevant evidence. We propose ECoG, an evidence-consistent generative framework that combines evidence-span supervision with a rationale-label consistency objective during training. On the 0.5B decoder, relative to the same backbone trained without consistency regularization, ECoG raises Macro-F1 on OOD challenging instances by 3.22 points, reduces the share of predictions whose generated rationale supports the opposite label by 4.22 points, and increases token-level overlap with reference evidence spans by 8.38 points; the reduction in prediction-rationale inconsistency is consistent across four decoder backbones. These results suggest that compact generative detectors can benefit from evidence supervision and rationale-label consistency under social-engineering shift. 

---
# Scaling Unsupervised Word Alignment to Documents via Structural Constraints 

**Authors**: Michelle Wastl, Jannis Vamvas, Rico Sennrich  

**Link**: [PDF](https://arxiv.org/pdf/2608.21023)  

**Abstract**: Word alignment has traditionally been studied between sentences, but many cross-lingual tasks increasingly require correspondences across full documents. While recent multilingual embedding models can encode long inputs, we show that applying algorithms designed for sentences directly to documents leads to performance degradation. To address this, we introduce CTFAlign, a lightweight, training-free approach for document-level word alignment. CTFAlign applies a coarse-to-fine refinement strategy that restricts the alignment search space to semantically similar regions. Additionally, we introduce MDPAlign, a simpler alternative that constrains alignments by position with a main diagonal prior. Both approaches operate directly on full documents without relying on sentence segmentation or sentence alignment. We evaluate these methods across six language pairs varying in typological distance, resourcedness, and document length. Averaged over three models, CTFAlign reduces word alignment error rate from 0.412 to 0.326. These gains transfer downstream, leading to improvements in document-level translation coverage evaluation and recognition of semantic differences. We release CTFAlign as a Python package and make the code and data to reproduce our experiments publicly available. 

---
# Free-Text Evaluation of LLMs for 5G Domain Knowledge and Fault Analysis using LLM-as-Judge 

**Authors**: Rishiraj Sengupta, Sotiris Chatzimiltis, Mohammad Shojafar, Xiatian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.21021)  

**Abstract**: Real-world fault analysis in 5G and emerging 6G networks demands domain expertise to analyze free-text diagnostics, including root-cause explanations and recommended actions. LLMs have emerged as a promising approach to automating this, yet whether lightweight, edge-deployable models are capable of performing in-depth free-text diagnostics remains an open question. While existing benchmarks rely on restrictive MCQs with fixed answer keys, this paper evaluates 5G domain understanding and fault analysis in a free-text generation format. Transitioning to this paradigm requires evaluating lightweight, edge-deployable AI models on open-ended diagnostic reasoning, alongside a dependable framework to validate these text outputs at scale. To address this we evaluate three lightweight LLMs, Claude-Haiku-4.5, GPT-5.4-Mini, and Gemini-3.1-Flash-Lite, on free-text 5G domain knowledge and fault-analysis tasks across three benchmarks, TeleQNA ORAN FT, 5G-Faults FT, and TeleInter FT. Three independent frontier judges score outputs, and pairwise inter-judge agreement is measured as an empirical test of the LLM-as-Judge methodology. All three models reach at least 90% accuracy on fault diagnosis, while zero-shot recall of 3GPP and O-RAN specifications remains the critical gap, with all models scoring below 60%. Mean inter-judge agreement is at least 0.90 across all runs, indicating that multi-judge LLM scoring produces consistent, reproducible grades for open-ended telecom responses. Operationally, Gemini-3.1-Flash-Lite offers the best efficiency trade-off, combining competitive accuracy with the lowest inference cost and latency, making it the most suitable candidate for production telecom deployments. 

---
# Target-Aware Calibration Data Selection for Preserving Uncertainty in Quantized Language Models 

**Authors**: Zhen Yang, Sizai Hou, Kaiwen Zheng, Yaofang Liu, Liang He, Yixuan Chen, Kangning Cui  

**Link**: [PDF](https://arxiv.org/pdf/2608.21019)  

**Abstract**: Quantization is widely used to deploy large language models, but its effect on uncertainty behavior, such as confidence, margins, and abstention, is rarely treated as a primary objective. We frame calibration-data selection for quantization as a target-dependent uncertainty-preservation problem. Different deployments emphasize different regions of the input distribution, yet prior work mainly optimizes accuracy-oriented compression metrics or adjusts scores after quantization. We formalize this goal with distributional and boundary preservation risks, and provide a simple mixture-mismatch argument explaining why no single calibration recipe should be expected to fit all targets. We introduce Doubt-Preserving Quantization (DPQ), a lightweight pre-quantization recipe family that uses full-precision predictions to construct target-aligned calibration mixtures of high-doubt examples and generic anchors. Across 8 language models, 9 NLP benchmarks, and 22 comparison methods, the leading fixed recipe changes with the preservation target: DPQ-r75 leads on SQuAD2 answerability-boundary preservation, while milder or single-signal variants, including DPQ-r50, confidence-only, and entropy-only, better preserve broad multiple-choice QA behavior. These results show that calibration data should be selected for the specific full-precision score behavior a deployment needs to preserve, rather than treated as a fixed quantization detail. 

---
# Extractive Summarization for Arabic Documents Using SAraBERT with a Semantic Siamese Similarity Evaluation Metric 

**Authors**: Sami Shames El Deen, Mariette Awad  

**Link**: [PDF](https://arxiv.org/pdf/2608.20964)  

**Abstract**: In this research, we introduce SAraBERT, an enhanced version of AraBERT which proposes inter-sentence transformer layers for extractive summarization tasks. To ensure that the summaries generated by SAraBERT achieve a high coverage of the document's main ideas, we propose Semantic Siamese Similarity, a novel evaluation metric that measures the level of similarity between two text inputs. We validated using BLEU, ROUGE, and Semantic Siamese similarity on Sarabert and published related models. Simulation results showed the effectiveness of our proposed model and motivate follow on research. 

---
# Quantization-Aware Healing: A Practical Recipe for Recovering Compressed, 4-Bit LLMs 

**Authors**: Bakbergen Ryskulov, Iker García-Ferrero, David Montero, David Jansen, Ali Hashemi, Jezabel R. Garcia, Antonio Tiene, Román Orús  

**Link**: [PDF](https://arxiv.org/pdf/2608.20953)  

**Abstract**: Serving large language models cheaply increasingly means shipping models that are both structurally compressed to a fraction of their parameters and quantized to 4 bits. Together these steps degrade reasoning, mathematics, coding, and long-context behavior enough to require a recovery, or healing, stage before deployment. The default recipe, quantization-aware training (QAT), re-fits the compressed, quantized model to hard labels; in our pipeline it converged slowly and collapsed past its peak. We adopted Quantization-Aware Healing (QAH) instead. Because a structurally compressed model is never independently trained at full precision, its bfloat16 checkpoint is a distillation-recovered approximation of the original; QAH distills the 4-bit student directly from the original, uncompressed model. On a GPT-OSS 120B to 60B to MXFP4 pipeline, the QAH student matches or beats its bfloat16 source on 7 of 9 benchmarks at roughly 4 times less weight memory and half the teacher's parameter count, and is released open-weight as Hypernova-60B. Against a matched QAT baseline it reaches a comparable peak about 7 times faster and stays stable under continued training, without hand-tuned early stopping. We also report deployment lessons, including a large, reproducible quality gap between distributed-training backends. Our aim is a recipe deployable without a multi-week hyper-parameter search. 

---
# MentorPulse: Refreshing Cross-Model Latent Guidance for Long-Form Generation 

**Authors**: Ziwu Liu, Guozhong Li, Chen Qiu, Weiyang Kong, Panos Kalnis  

**Link**: [PDF](https://arxiv.org/pdf/2608.20927)  

**Abstract**: Cross-model latent guidance lets a frozen large mentor encode an input once and a frozen small student generate from the resulting signal. Existing methods keep this signal fixed, assuming it stays useful as the output grows; we show this fails in long-form generation. On multi-turn instruction following, static guidance pushes a 4B student's constraint satisfaction 2.5 points below its no-guidance baseline; a training-free refresh every 16 tokens changes only the memory content and restores a 2.0-point gain over that baseline. We propose MentorPulse to keep guidance fresh at practical cost: it compresses mentor states into a capped slot memory, incrementally processes newly generated tokens, and updates the memory that the student reads through gated cross-attention without resetting the student's KV cache. Windowed Refresh Training exposes the bridge to prefix-conditioned memory. Across thirteen datasets, MentorPulse closes 52.2% of the mentor-student gap on macro average, outperforming C2C, T2T, and equal-budget LoRA, with the largest gains on long outputs. It performs best on all eleven mentor-student pairs from three model families, with margins that narrow as the capability gap grows, and a lightweight read-pattern check predicts the gain before deployment. Measured costs identify refresh intervals that dominate text guidance on long outputs. 

---
# Source-Free MT Evaluation Is Not MT Evaluation 

**Authors**: Baban Gain, Ramakrishna Appicharla, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2608.20925)  

**Abstract**: Reference-based metrics remain the standard choice in machine translation evaluation, partly because quality estimation methods often correlate less well with human judgments. As a result, source-free, reference-based evaluation has become the practical norm, even though it is unfaithful to the definition of translation adequacy and unfair to systems whose outputs preserve the source meaning while differing from the reference. This paper argues that adequacy must be judged with respect to the source. A reference is only one possible rendering of the source and may introduce bias, under-specification, or errors. We further argue that source-reference-hypothesis evaluation is fair only when the judge treats the reference as auxiliary evidence rather than as the primary standard. Otherwise, even source-aware evaluation can reduce adequacy to preference towards reference. We show the existing hybrid metrics are highly reliant on reference compared to source. Our argument is not that all automatic MT metrics fail to use the source. Rather, we argue that any evaluation protocol that removes the source, or allows the reference to dominate the source, is structurally incomplete for adequacy evaluation. However, existing MT papers generally prefer reference-based metrics and use QE metrics only when reference is unavailable. We therefore call for QE to be reframed as a primary approach to source-grounded adequacy evaluation, rather than as a fallback motivated by missing references. We further call for hybrid metrics whose designs explicitly prioritize source--hypothesis faithfulness while using references only as complementary evidence. 

---
# ForeDreamer: A Self-Evolving Dual-Agent Memory Architecture for Future Event Prediction 

**Authors**: Linhao Zhong, Zongze Du, Linyu Wu, Yu Bo, Hourong Li, Chenchen Jing, Hao Chen, Yuling Xi, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.20920)  

**Abstract**: Open-web future event prediction requires agents to distill reliable signals from noisy, redundant, and incomplete evidence. Existing retrieval/memory mechanisms directly feed retrieved information to agents or rely on simple memory functions such as storing and reusing prior information for prediction, leaving them insufficient for open-web forecasting. We propose to transform raw web evidence into structured memory before prediction, enabling agents to reason over distilled, question-specific evidence rather than noisy retrieval results. This paper presents ForeDreamer, a self-evolving dual-agent framework for managing memory over open-web evidence. ForeDreamer separates factual memory, a question-specific evidence state for the current forecast, from experiential memory, persistent agent experience accumulated across forecasting episodes. It uses a main agent for search and prediction, and a memory-processing subagent to convert search results into factual memory with dedicated tools. ForeDreamer further evolves experiential memory through two tracks, improving both forecasting decisions and factual-memory construction. Experiments on Prophet Arena and FutureX demonstrate the effectiveness of ForeDreamer. Project page: this https URL 

---
# KREL: Automatic Medical Coding via Knowledge-Guided Reasoning over Clinical Evidence with LLMs 

**Authors**: Xubin Chen, Yipeng Zhou, Wen Sun, Chengkai Huang, Xiaoming Fu, Quan Z. Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.20887)  

**Abstract**: Automatic Medical Coding (AMC), which assigns standardized International Classification of Diseases (ICD) codes to clinical notes, is essential for medical reimbursement, quality reporting, and clinical research. Existing pre-trained language model (PLM)-based methods typically formulate AMC as an extreme multi-label classification problem over a predefined code set, while recent large language model (LLM)-based approaches instead frame it as generation or multi-step reasoning. However, key challenges remain, including the extreme length of clinical notes that hinders effective interpretation, the vast ICD label space, and complex coding rules that are not explicitly captured by LLMs. In this work, we propose Knowledge-Guided Reasoning over Clinical Evidence with LLMs (KREL), a framework that leverages LLMs for clinical text understanding and reasoning while integrating external ICD coding guidelines as structured knowledge. This design enables tight coupling between domain knowledge and LLM reasoning, reducing hallucinations and improving compliance with coding standards. Experiments on benchmark datasets show that KREL consistently outperforms strong PLM-based and state-of-the-art LLM-based baselines. 

---
# Ontology-Driven Structural Regularization for Document-Level Relation Extraction 

**Authors**: Laura Menotti, Stefano Marchesin, Gianmaria Silvello  

**Link**: [PDF](https://arxiv.org/pdf/2608.20856)  

**Abstract**: Document-Level Relation Extraction (DocRE) relies heavily on costly manually annotated datasets, while large distant supervision resources such as DocRED distant remain underexploited due to noise. We show that a critical yet overlooked source of noise lies in structural inconsistencies within relational triples, including violations of ontology constraints and logical contradictions.
We introduce an ontology-driven framework to quantify and enforce structural consistency in DocRE datasets. Our analysis reveals substantial structural noise in DocRED distant and demonstrates that such inconsistencies propagate to model predictions. Enforcing structural well-formedness during training significantly reduces logical contradictions and consistently improves generalization performance. These findings establish structural consistency as a missing axis of supervision in DocRE and highlight structural regularization as an effective strategy for leveraging distant data at scale. 

---
# SAC-Copula: Quality-Preserving Watermarking for Diffusion Language Models via Smooth Correlated Gumbel Fields 

**Authors**: Baixin Li, Haiyun He  

**Link**: [PDF](https://arxiv.org/pdf/2608.20839)  

**Abstract**: Watermarking diffusion language models (DLMs) requires mechanisms compatible with iterative parallel unmasking rather than autoregressive decoding. Existing sampling-based watermarking methods typically inject position-wise i.i.d. perturbations, which can be poorly aligned with DLM decoding dynamics and degrade generation quality. We propose SAC-Copula, a quality-preserving watermarking method for DLMs based on smooth, locally correlated Gumbel perturbation fields constructed via a Gaussian copula. We further develop a SAC-aware detector using covariance-aware filtering and native-sample calibration. Mechanism-level analysis shows that local correlation reduces latent perturbation roughness and better matches iterative refinement dynamics. Experiments on LLaDA show that SAC-Copula achieves a favorable quality-detectability trade-off compared with existing baselines. In particular, further evaluations on Dream-7B and additional datasets show that SAC-Copula substantially improves PPL tail stability over the i.i.d. Gumbel baseline, while maintaining strong low-FPR detectability and competitive overall generation quality. Additional token-edit stress tests further assess watermark robustness under controlled synchronization drift. 

---
# STAR-OPD: Structured Aspect-Cascade-Aware On-Policy Reward Distillation for ABSA Quadruple Extraction 

**Authors**: Tong Sun, Mingyang Ma, Jiayang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.20831)  

**Abstract**: Aspect-based sentiment analysis (ABSA) quadruple extraction requires jointly predicting target, aspect, opinion, and sentiment over reviews that often contain multiple fine-grained sentiment tuples. While large chain-of-thought (CoT) models perform well on this task, distilling them into smaller deployable models remains difficult. We identify a task-specific failure mode in distilled ABSA extraction: student errors at the target-aspect interface create structurally invalid states, such as broken target-aspect bindings and hallucinated targets, which then corrupt downstream predictions. Conventional off-policy distillation is poorly suited to this setting because it trains only on teacher-generated trajectories and provides little supervision on the student-induced structural states that dominate inference. To address this mismatch, we propose STAR-OPD (STructured Aspect-cascade-aware On-Policy Reward Distillation), which builds on generic on-policy distillation and instantiates it for ABSA quadruple extraction with cascade-aware, set-structured rewards. STAR-OPD trains on student rollouts and applies set-structured rewards that directly target binding consistency, target grounding, and fine-grained aspect disambiguation. Experiments on E-ABSA20K and SemEval-2014 show that STAR-OPD consistently outperforms off-policy and general on-policy baselines, reduces target hallucination, and substantially improves performance on structurally hard cases. With Qwen3-4B, STAR-OPD substantially narrows the student-teacher gap while improving inference efficiency, highlighting the importance of on-policy structural correction for distilled ABSA extraction. 

---
# Denoising the Future: Context-Aware Spectral Diffusion for Temporal Knowledge Graph Extrapolation 

**Authors**: Yanglei Gan, Peng He, Run Lin, Peiyuan Jiang, Yifan Wang, Qiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.20804)  

**Abstract**: Temporal Knowledge Graph (TKG) extrapolation seeks to infer future facts from time-varying relational histories. Recent diffusion-based approaches improve uncertainty modeling through generative denoising, but their aggregated conditioning on subject histories may insufficiently distinguish query-specific evidence from non-salient historical facts, thereby diluting target-discriminative signals. To bridge this gap, we propose FreqDiff, a Frequency-aware Diffusion framework for TKG extrapolation. Specifically, FreqDiff formulates future object prediction as query-slot denoising and develops a dual-stream denoiser that integrates temporal dependency modeling with context-aware spectral calibration. The spectral branch synthesizes history-conditioned filters from learnable bases to adaptively re-calibrate denoising representations, while a frequency-domain regularizer is proposed to align the denoised target with the gold object in spectral space. Experiments on four public TKG benchmarks demonstrate that FreqDiff achieves state-of-the-art performance. 

---
# Tree-of-Concerns: Hierarchical Multi-Agent Debate for Unstated-Limitation Extraction in Scientific Critique 

**Authors**: Sahil Mishra, Niranjan Rajeev, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2608.20777)  

**Abstract**: As scientific literature grows and papers increasingly under-report limitations, multi-agent LLMs offer a promising approach to systematically uncover these hidden failure modes. Here, we introduce Tree-of-Concerns, a multi-agent framework that deploys specialized skeptic personas, each operating through a category-specific analytical lens, as parallel debate trees to extract unstated limitations from scientific papers. Each persona conducts structured, evidence-grounded argumentation, while a Panel Review mechanism re-evaluates each surviving claim from all five perspectives to correct category drift and severity miscalibration. Through experiments on ToC-Bench, our benchmark of 414 research papers with 1,905 unstated limitations, sourced from reviewer-reported weaknesses and follow-up citation critiques, we demonstrate that ToC improves precision by 79% and coverage by 11% relative to strongest baselines, surfacing specific, evidence-grounded concerns that support reviewers in systematic evaluation. 

---
# PSK at WMT 2026 MIST: Task-Specialized QLoRA Adapters for Multilingual Summarization and Question Answering 

**Authors**: Srikar Kashyap Pulipaka  

**Link**: [PDF](https://arxiv.org/pdf/2608.20757)  

**Abstract**: We describe the PSK submission to the WMT 2026 Multilingual Instruction Shared Task. Our system uses the 3.35B-parameter Tiny Aya Global model with three QLoRA adapters, one for each task. The adapters are trained on multilingual document-summary pairs, passage-based question answering, and filtered standalone question answering. The summarization data also includes scientific papers with their author-written abstracts. On our held-out split, the context and summarization adapters perform better than our multitask adapter, which was trained only on data supplied by the organizers. Results for open QA are mixed and vary with answer length and evaluation method. We therefore submit three systems with the same context and summarization adapters but different open-QA adapters. 

---
# AsmEvo: Agentic Assembly-Level Optimization of AMD GPU Kernels with Functional Equivalence Verification 

**Authors**: Ji Liu, Puyuan Yang, Rongzhang Zheng, Fan Wang, Jinglin Wang, Muhammad A. Awad, Mortis Huang, Andy Chang, Zekai Li, Zeping Li, Zihao An, Yue Liu, Yuchen Yang, Jianghui Wang, Chushi Chen, Ziqiong Liu, Fuwei Yang, Dong Li, Wen Heng Chung, Shengcai Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2608.20711)  

**Abstract**: High-performance ML systems increasingly rely on GPU kernels whose editable source is unavailable, generated, or too distant from final machine code to expose remaining optimizations. Existing LLM kernel optimizers and autotuners mainly operate on CUDA, Triton, HIP, or tensor-program source and validate against reference implementations. We study a stricter setting: optimizing an already compiled AMDGPU code object, where the deployed binary is the only behavioral oracle.
We present AsmEvo, an agentic assembly-level optimizer for AMD GPU kernels. Given an AMDGPU code object K0, AsmEvo reconstructs a reassemblable representation, proposes low-level edits with a long-horizon agent, rebuilds an ABI-preserving optimized object, and accepts candidates only after differential verification against K0 under identical launches. AsmEvo combines code-object recovery, metadata-aware rebuilding, profiling-guided hot-window editing, correctness-gated timing, and conservative in-place patch fallback.
We conduct extensive experiments with AsmEvo on various AMD GPU kernels. On MI308X, AsmEvo improves 29 of 30 selected KernelBench kernels, reaching 1.35x geometric-mean and 3.88x maximum speedup. On MI300X production workloads, it improves all evaluated AITer binaries and vLLM/SGLang Triton assembly kernels, reaching 1.09x/1.31x and 1.18x/1.34x geometric-mean/maximum speedups, respectively, while preserving functional equivalence. 

---
# Directional Contextual Representations for Dependency Relations: Why Cross-Direction Pairing Fails 

**Authors**: Sai Krishna Arthanari, JaeHyeong Chang, Chengzhe Sun, Siwei Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2608.20647)  

**Abstract**: Splitting a bidirectional LSTM's contextual representation into a forward-only $F_i$ (strictly a function of tokens $1..i$) and a backward-only $B_i$ (strictly a function of tokens $i..n$) beats either alone and beats a fused self-attention representation for dependency relation-type classification. But a specific, natural extension of this idea -- pairing a token's forward state against a \emph{candidate}'s backward state (``cross-direction'' pairing, $F_i$ vs.\ $B_j$) -- consistently \emph{underperforms} same-direction pairing, and the penalty \emph{grows}, not shrinks, with token distance, both paired-bootstrap significant. We diagnose why using a frozen-trunk methodology: architectural information leakage between directions is impossible by construction (a single-layer BiLSTM, verified by code inspection); 93\% of the same-vs-cross gap survives freezing the trunk and training only fresh heads, ruling out training-co-adaptation as the primary cause; linear regression shows partial representational redundancy between $F_i$ and $B_i$ ($R^2{=}0.324$ vs.\ $0.028$ for a shuffled control) and a linear probe shows partial anticipatory encoding of upcoming tokens in $F_i$ (36.5\% vs.\ 17.2\% majority baseline) -- real effects, but neither alone, nor combined, cleanly explains the full gap. Extended frozen-trunk diagnostics (a positional probe and a distance-decay probe) show directional information is genuinely stored but not exactly positioned, and propagates only a few tokens before decaying to baseline -- consistent with, and mechanistically underneath, the distance-growth finding. 

---
# MIL-BERT: Classification of Arbitrarily Large Text with Performance and Explanatory Guarantees 

**Authors**: John Cadigan, Dayne Freitag, Eric Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2608.20636)  

**Abstract**: Many text classification decisions are viable based on constituent excerpts alone. Taking inspiration from the field of multiple instance learning, we present an algorithm for training a neural network to classify text by selecting such excerpts. We show that our approach is also scalable with demonstrated learning against samples with nearly 1M tokens. We evaluate our methods on 7 datasets with emphasis on long-textual collections that far exceed the encoding limit of our base model. We present state-of-the-art results with this algorithm on 3 datasets: identification of political bias in news outlets, trigger warnings in long stories, and demographic characteristics of authors in tweet collections. Furthermore, the model trained on weakly-labeled collections of text (bags) generalizes to accurately classify constituent, smaller instances. Besides a new state-of-the-art for these problems, this approach is one of the few neural methods to excel in these datasets. 

---
# AgentMercury: Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale 

**Authors**: Minbyul Jeong, Chanwoong Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2608.20634)  

**Abstract**: Agents learn to act through interaction with environments, yet the environments used for training are often manually constructed or synthesized around predefined tasks and benchmarks. This task-centric paradigm makes it difficult to scale environments that reflect realistic and evolving workflows where diverse tasks can naturally emerge from the underlying world. We introduce AgentMercury, a scalable framework for synthesizing executable environments from high-level business scenarios. Rather than constructing an environment for a specific task, AgentMercury first instantiates a persistent world with entities, services, tools, state, and executable cross-service invariants, from which diverse tasks and interaction trajectories can subsequently emerge. We construct 4,783 executable environments spanning 14 industries and 50 countries, and use them as training substrates for reinforcement learning. Despite being generated without targeting the evaluation benchmarks, policies trained on these business-oriented environments improve substantially on both enterprise workflows and out-of-domain benchmarks spanning reasoning, coding, scientific computing, and tool use. In our experiments, Qwen3.5-4B improves from 12.3 to 15.7 on EnterpriseOps-GYM and from 45.9 to 56.0 on AIME26 after training on AgentMercury environments. We further show that the construction process itself can be learned: fine-tuning Qwen3.5-35B-A3B on construction traces increases executable-world authoring success from 3.3% to 83.3% on held-out business scenarios. These results show that scenario-grounded environments can provide useful and generalizable learning signals beyond benchmark-specific training, while their construction can itself become a learnable capability. 

---
# Sparse Token Routing in Efficient Transformers 

**Authors**: Sai Krishna Arthanari, JaeHyeong Chang, Chengzhe Sun, Siwei Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2608.20632)  

**Abstract**: Efficient-transformer research often motivates token pruning and adaptive computation with the claim that not all tokens require equal computational effort. We test this claim end to end using SEWN, a two-stream Transformer that routes tokens through either lightweight or full-capacity processing using a learned gate. Across our experiments, routing introduces negligible accuracy change relative to parameter-matched baselines, while the gate's token-importance signal depends critically on how it is learned. A static lexicon-seeded prior fails a counterfactual faithfulness test on BoolQ, whereas a fully contextual gate achieves highly significant separation ($p<10^{-10}$) on both evaluated tasks without changing task accuracy. 

---
# When Failures Propagate: Causal Failure Attribution in Agentic Retrieval-Augmented Generation 

**Authors**: Lauren Pothuru  

**Link**: [PDF](https://arxiv.org/pdf/2608.20627)  

**Abstract**: Agentic retrieval-augmented generation (RAG) interleaves retrieval, reasoning, and answer generation across multiple hops. A retrieval error at hop 1 can surface only as a wrong answer at hop 3, while later retrieval can also repair the trajectory. This paper introduces AgenticRAG-FP, an interventional benchmark for causal failure attribution in agentic RAG. The benchmark injects a certified fault at a specified hop, re-executes the downstream trajectory, and evaluates diagnosers against the known intervention. Its central question is whether a post-hoc trace still identifies the injected hop after the suffix changes. In the completed strict dense Claude Haiku 4.5 sweep on 80 three-hop MuSiQue questions, coverage-based diagnosis is 0.91 at hop 1 and 0.00 at hops 2 and 3 (n=43,36,21 failed trajectories). A smaller content-corruption study changes an answer-bearing or bridge fact in topically intact evidence. At depth 2, where 18 failed cases remain after filtering, coverage-based diagnosis is 0.00 and a frozen-hop counterfactual probe is 0.67 in an exploratory pooled comparison. Depth-3 content estimates are descriptive only because they contain three failed cases. These results make propagation depth an explicit evaluation axis for diagnosing agentic RAG failures while distinguishing broad evidence of post-hoc signal loss from small-sample method comparisons. 

---
# JuryProbe: An Empirical Consensus-Risk Diagnostic for Routing Reference-Free Factuality Judge Panels to Grounded Verification 

**Authors**: Tianxin Zhou, Ruixi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2608.20607)  

**Abstract**: Panels of inexpensive LLM judges increasingly make accept-or-escalate decisions. In factuality settings, accepting a claim because several reference-free judges agree can create a hidden risk: agreement may reflect shared false-negative blind spots rather than independent evidence. We introduce JuryProbe, an empirical consensus-risk diagnostic for reference-free factuality judge panels, paired with a calibration-based routing policy. JuryProbe estimates consensus risk from a labeled calibration probe using false-negative-only (FN-only) judge correlation and false-consensus lift; when flagged high-risk, reference-free majority accepts are routed to the same judges with trusted references. On audited FEVER corruptions, reference-free panels show correlated false negatives (FN-only correlations 0.402 and 0.368; lifts 3.13x and 18.13x), while unanimous false consensus drops to zero under a trusted-reference best-case diagnostic on both minimal-pair and non-minimal-pair evidence. In flagged settings, the routed policy is by construction equivalent to grounding every reference-free majority accept (verified in 34/34 splits): improvement comes from accept-conditioned grounding, while the diagnostic determines whether to activate it. A fixed, pre-specified rule flags 8-10 of 10 splits across synthetic, benchmark-authored, and scientific families and 0 of 10 on a negative control, where standing down avoids 28% of reference acquisitions at a 0.004 increase in false accepts. False-accept reduction persists under weak BM25 retrieval at substantial coverage cost, while stale stand-down labels require periodic recalibration. JuryProbe provides no formal risk guarantee and does not establish reliable stand-down on natural panels; its supported contribution is an empirical diagnostic of high-risk panel error dependence. 

---
# LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding 

**Authors**: Matan Rusanovsky, Yoav Miron, Roy Uziel, Omer Belhasin, Ran Zilberstein, Maor Ashkenazi, Michael Elad  

**Link**: [PDF](https://arxiv.org/pdf/2608.20530)  

**Abstract**: Speculative decoding accelerates language-model inference by drafting future tokens that the target model verifies in parallel. A diffusion-style block head such as DFlash is an attractive drafter, predicting an entire block of future tokens in one forward pass. However, it is trained on per-position marginals rather than the joint block distribution, so the tokens it emits are individually plausible yet jointly incoherent. We introduce LiLiCorr, a Lightweight Likelihood-based model that Correlates the per-position marginal distributions a drafter already produces. It keeps the top-k tokens at each position as candidates and processes them jointly, producing for each an in and an out vector. A pair of adjacent candidates matches when the earlier one's out vector has high cosine similarity with the later one's in vector. These matches capture the block's joint structure without ever materializing the full joint distribution. One lightweight network pass produces all the vectors, and the pairwise scores are then computed in parallel as batched matrix operations, leaving only a cheap greedy walk sequential. We further co-train the drafter with LiLiCorr, so it learns to propose candidates that correlate into longer accepted sequences. Over the vanilla DFlash drafter, LiLiCorr raises acceptance length on every benchmark by 9 to 19%, while its scoring head accounts for about 2.8% of the per-block latency. Against DFlash and two concurrent methods that also restore coherence at draft time, LiLiCorr delivers the highest throughput in 70 of 72 settings: nine benchmarks at two target sizes under greedy and temperature-one decoding, and a throughput sweep over six concurrencies, two input lengths and three entropy tiers, with all systems equally optimized on a common serving stack. Extending LiLiCorr to inputs an order of magnitude longer than it was trained on preserves that lead. 

---
# ARGUS: Theory-of-Mind Guided Argument Generation with Strategy-Aware Planning and Knowledge Grounding 

**Authors**: Zhe Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.20405)  

**Abstract**: Persuasive argument generation requires modeling audience beliefs, rhetorical strategies, and factual grounding. Despite recent advancements, existing methods remain largely audience-agnostic and fail to integrate strategy selection to improve persuasiveness. To bridge this gap, we propose Argus, an agent-based framework that operationalizes classical rhetoric for persuasive writing. At its core, a Theory-of-Mind (ToM) Reasoner constructs an explicit dual mental model of the audience's beliefs and values to guide downstream decisions. This representation conditions a component-aware planner that decomposes the argument into subtopics, assigns fine-grained rhetorical functions (logos, pathos, ethos, kairos), and triggers strategy-guided evidence retrieval at planning time. Finally, a refinement module iteratively targets and resolves multi-dimensional weaknesses without quality regression. We evaluate Argus across three diverse benchmarks using both automated pairwise Elo and LLM-as-judge metrics. Results show that Argus consistently outperforms strong baselines across multiple backbone models, achieving top rankings and the highest overall scores. Targeted simulation experiments further validate its effectiveness in shifting resistant audience stances. 

---
# LingShu: A Large-Scale Symptom-Centric Contextualized Knowledge Graph Bridging Traditional Chinese Medicine and Modern Biomedicine 

**Authors**: Rui Hua, Zixin Shu, Kai Chang, Dengying Yan, Jianan Xia, Hui Zhu, Shujie Song, Shurui Yang, Tongxin Wang, Yue Yin, Yu Wei, Lijuan Pei, Yunhui Hu, Hao Xu, Mingzhong Xiao, Xiaodong Li, Haibin Yu, Runshun Zhang, Wenjia Wang, Baoyan Liu, Xuezhong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.20402)  

**Abstract**: Biomedical knowledge graphs (KGs) are pivotal for knowledge organization, yet traditional binary relations often struggle to represent the conditional nature of biomedical knowledge. Symptoms provide a shared phenotypic layer for linking Traditional Chinese Medicine (TCM), which relies on symptom patterns for syndrome differentiation and treatment selection, with modern biomedicine, which connects clinical manifestations to diseases and molecular mechanisms. We present LingShu, a large-scale symptom-centric contextualized knowledge graph designed to bridge TCM and modern biomedicine. The exported version of LingShu analyzed in this study comprises 17.33 million atom-level entity records and 39.47 million relation records, including 17.19 million semantic triples and 22.29 million contextualized quadruples. LingShu integrates multi-source data, including clinical electronic medical records, authoritative TCM texts, biomedical ontologies, and curated knowledge bases, through a pipeline combining natural language processing, terminology normalization, and human-in-the-loop verification. A key innovation of LingShu is its hybrid data model: it maintains 64 typed triple relation patterns to ensure broad connectivity, while incorporating 35 contextual quadruple relation patterns to capture conditional medical associations. This dual-structure approach explicitly encodes conditional knowledge, providing a granular representation of the contexts associated with medical relations. These contextualized relations cover syndrome-dependent herb efficacy, disease-contextualized drug effects, population-specific clinical associations, and mechanism-related therapeutic responses. Furthermore, we developed a web platform (this http URL) that integrates graph visualization, graph-based reasoning, and an evidence-grounded knowledge question-answering agent. 

---
# Self-Supervised Speech Representations Track Spoken Language Convergence to Adult Models in Infants and Children Who Are Deaf/Hard-of-Hearing 

**Authors**: L. Choy, A. S. Khan, S. Patrizi, D. Ye, J. Gross, M. Cychosz  

**Link**: [PDF](https://arxiv.org/pdf/2608.20396)  

**Abstract**: Language development is characterized by a gradual convergence of children's speech toward adult patterns. Measuring this process has traditionally required detailed transcription and language-specific expertise, limiting scalability across languages and populations. Here, we use speech embeddings to capture this convergence directly from the acoustic signal in longform, child-centered recordings, taken as children go about their daily lives. Using HuBERT-BASE, we extracted embeddings from speech vocalizations of children who are deaf/hard-of-hearing and their female adult caregivers ($>$925 hrs. observation). Embedding distance between children and caregivers decreased with hearing age, controlling for pitch and vocalization length, indicating, as expected, that children's speech patterns converge to caregivers over development. This single distance metric likewise related to multiple standardized measures of speech and language from infancy through preschoolhood. These results suggest a path toward scalable, language-neutral assessment of spoken language development from children's everyday lives. 

---
# Knowledge-Graph-Gated Defactualization for Style-Controllable and Fact-Preserving Generation in Agentic Conversational AI 

**Authors**: Tanmay Kumar Shrivastava, Darsh Rohit Nandu, Rajesh Kumar Mundotiya  

**Link**: [PDF](https://arxiv.org/pdf/2608.20393)  

**Abstract**: Agentic large language models (LLMs) deployed in fact-sensitive applications such as customer support must simultaneously preserve factual correctness and generate responses in a controllable stylistic register. Activation steering enables fine-tuning-free style control by perturbing hidden representations, but it lacks an explicit mechanism for distinguishing verifiable facts from stylistic content, leading to semantic leakage. We address this challenge through \emph{Defactualize-Steer-Rehydrate} (DSR), a knowledge-engineering framework that integrates a typed, salience-weighted knowledge graph (KG) with activation steering. DSR extracts salient entities using a layered regex or NER or lexical-classifier pipeline, replaces them with typed placeholders prior to steering, and deterministically restores verified values through salience-guided rehydration after generation. DSR is evaluated across six LLaMA-family models (1B--13B parameters) on 600 A2A-generated customer-support cases (1,200 generations), with a dedicated KG ablation study. DSR significantly increases verified-entity recovery relative to a steering-only baseline (Cohen's $d=0.225$, $p_{\text{Bonf}}=1.0\times10^{-4}$), though the absolute recovery rate remains modest, while preserving effective style control across diverse model families. Layer-wise separability and steering-strength diagnostics further show previously unexplored interactions between representation-level steering and factual grounding. hese results demonstrate that explicit knowledge engineering can systematically enhance trustworthy, controllable, and reproducible generative AI without requiring model fine-tuning. Code, cached steering vectors, and evaluation scripts are publicly released to support reproducibility.\footnote{this https URL} 

---
# Evaluation-as-Search: Adaptive Discovery of Grounding Failures in Meeting Assistants 

**Authors**: Sami Khairy, Yasaman Hosseinkashi, Vishak Gopal, Ross Cutler  

**Link**: [PDF](https://arxiv.org/pdf/2608.20392)  

**Abstract**: LLM-powered meeting assistants are deployed at scale, yet systematic evaluation of their grounding fidelity remains limited to static benchmarks that miss failure modes tied to specific discourse structures or reasoning demands. We propose Evaluation-as-Search (EaS), a feedback-driven methodology that frames quality evaluation as an adaptive search over the space of natural questions a meeting participant might ask. Rather than sampling uniformly, EaS learns from evaluator feedback across iterations to concentrate probing effort on cognitive demands where failures are most likely, guided by a UCB-scored coverage map and blind multi-dimensional quality evaluation. Using EaS, we construct MeetingProbe, a benchmark of over $3{,}000$ annotated question--answer pairs spanning 20 transcripts from three meeting genres and three LLM assistants. In ablations, adaptive search surfaces $2.5\times$ more failures than random probing ($7.1\%$ vs. $2.9\%$ finding rate), with the strategic planner contributing the largest individual effect. Across three models, we observe a clear capability gradient and identify eight recurring failure categories dominated by discourse-pragmatic challenges rather than factual recall errors. We further validate MeetingProbe across multiple model families and providers, finding a clean capability gradient and a curated subset of universal failures that no model handles. MeetingProbe is released publicly to support reproducible evaluation of meeting assistant grounding fidelity. 

---
# ImmigrationReason: A Structured Dataset of U.S. Immigration Appeals for Legal Reasoning Research 

**Authors**: Amirhossein Afsharrad, Seyed Shahabeddin Mousavi  

**Link**: [PDF](https://arxiv.org/pdf/2608.20391)  

**Abstract**: Most legal NLP resources draw from federal case law and focus on coarse classification, leaving administrative adjudication, where the vast majority of government decisions occur, essentially unaddressed. We introduce ImmigrationReason, a large-scale structured dataset derived from 12,375 non-precedent decisions of the U.S. Citizenship and Immigration Services (USCIS) Administrative Appeals Office (AAO) spanning 2005 to 2026. Each record captures the applicable legal framework, per-criterion evidence-sufficiency findings under a five-category label, verbatim adjudicator-criticism quotes, all citations, and final dispositions, alongside high-quality Claude-transcribed source text. Extraction quality is validated through a three-pass pipeline combining two independent modalities with comparison-prompt adjudication by Opus 4.7, and verified by domain experts on a 500-record sample. The dataset documents nearly 9,000 verbatim instances of AAO-identified legal errors, spans a natural legal-regime transition (the 2016 Dhanasar rule change), and covers 21 years of adjudication. We analyze the dataset in detail and outline research directions it enables, from outcome prediction and adjudicator-error analysis to agent design for high-stakes regulatory domains. 

---
# Ansari: A Retrieval-Grounded Islamic AI Assistant -- Architecture, Deployment, and Lessons from 140,000 Conversations 

**Authors**: M Waleed Kadous, Amr Elsayed, Abdullah Al Nahas, Ashraf Haress  

**Link**: [PDF](https://arxiv.org/pdf/2608.20390)  

**Abstract**: General-purpose large language models (LLMs) are increasingly used to answer religious questions, but for Islamic content they carry two serious risks: factual fabrication (inventing Qur'anic verses or hadith) and subtle value misalignment. We present Ansari, a deployed, retrieval-grounded Islamic AI assistant that has handled more than 140,000 conversations across 25+ languages since June 2023. Ansari is built around an agentic retrieval loop: a tool-using language model issues searches against authenticated Islamic corpora -- the Qur'an, hadith collections, a multi-volume jurisprudence (fiqh) encyclopedia, and exegetical (tafsir) sources -- and answers only on the basis of what it retrieves, with citations attached for verification. We describe the system's architecture (the agent loop, the retrieval tools, the corpora, and the system prompt that encodes editorial and theological policy), its multi-platform deployment (web, mobile, WhatsApp, and as a Model Context Protocol server and an Agent Skill), and what 140,000 real conversations reveal about how Muslims actually use such a tool. We report results on several complementary evaluations -- zero-shot performance on accredited institutional exams, a human-rated validation during Ramadan, and two independent, externally run benchmarks on which Ansari currently tops the public IslamicMMLU leaderboard ahead of frontier models and is competitive on Islamic legal reasoning (IslamicLegalBench) while strongly resisting false premises -- and draw out lessons that generalize beyond Islam to any faith- or values-sensitive deployment of LLMs: grounding is necessary but not sufficient, the system prompt is a theological as much as a technical artifact, and the absence of community in how models are formed remains a hard gap. 

---
# Intent Engine: Natural-Language Intent Translation for Intent-Driven Orchestration in the Compute Continuum 

**Authors**: Koushikur Islam, Rodrigo N. Calheiros  

**Link**: [PDF](https://arxiv.org/pdf/2608.20388)  

**Abstract**: Microservice placement in the compute continuum is driven by low-level Service-level Objectives (SLOs), but requiring users to specify metric-level constraints creates an adoption barrier and increases misconfiguration risk. Although large language models (LLMs) can interpret natural-language intents, direct generation of orchestration-consumable SLO artifacts remains unreliable due to unsupported constraints, incorrect grounded values, and schema violations. These errors can propagate to downstream placement logic and produce infeasible or incorrect placements. This paper presents Intent Engine, a natural-language intent translation architecture that constructs validated SLO artifacts for compute-continuum service placement. Intent Engine acts as an intent acquisition and SLO construction layer for existing intent-driven orchestration and placement frameworks; it does not perform placement or runtime QoS optimization. The architecture combines schema-constrained extraction, retrieval-grounded value construction from monitored infrastructure state, and validation against supported constraints before emitting the final SLO artifact. We evaluate Intent Engine using a 716-record intent-to-SLO dataset derived from an edge-cloud testbed, including valid and invalid intents. Across GPT-4.1 mini, Claude Sonnet 4.5, and DeepSeek V4-Flash, Intent Engine outperforms prompting baselines and a non-LLM rule-based parser. With GPT-4.1 mini, it achieves 0.941 total F1 Score and reduces aggregate hallucination by 85.1%, while lowering downstream placement failure from 30.8% to 2.1%. 

---
# Poly-InstructTTS: Learning In-the-Wild Expressive Speech Synthesis from Open-Ended Instructions 

**Authors**: Junhui Zhang, Qianhui Xu, Qingxiang Guo, Dawei Yang, Ling Miao, Qiangqiang Wang, Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.20387)  

**Abstract**: While recent text-to-speech (TTS) models achieve high naturalness, controlling fine-grained expression via natural-language instructions remains challenging. We introduce Poly- InstructTTS, which learns expressive speech from open-ended instructions using in-the-wild audiovisual data. We build a scalable multi-modal pipeline to construct a 1,000-hour instruction-annotated corpus covering 1,000+ fine-grained emotions and styles. The framework uses a prompt-free GPT with attribute-based thinking tokens, followed by a flow-matching module that injects timbre from a reference audio. We also present a speaker fine-tuning procedure to transfer instruction control to specific speakers while preserving persona. We further extend InstructTTSEval with broader tasks. Experiments show that Poly-InstructTTS delivers strong performance in instruction adherence and expressiveness. Audio demos and the expanded testset are available on our project page. 

---
# Using Human-LLM Disagreement to Improve Checklist-Based Quality Appraisal 

**Authors**: Timo van der Kuil, Bruno Messina Coimbra, Mirjam van Zuiden, Robert A. Bagheri, Rens van de Schoot, Klaas Dieleman, Berend Greijn, Stefan Houkes, Sebastiaan Rodenhuis, Elizabeth M. Grandfield  

**Link**: [PDF](https://arxiv.org/pdf/2608.20385)  

**Abstract**: Systematic reviews rely on quality appraisal of included studies, a process that is time-consuming and sensitive to ambiguity in checklist criteria. Although large language models (LLMs) offer opportunities to support these tasks, appraisal checklists are typically treated as fixed inputs, and it remains unclear how their design affects agreement with expert judgments. Therefore, we investigate (1) whether LLMs can approximate human judgments in checklist-based appraisal and (2) whether patterns of human-LLM disagreement can be used to identify and improve ambiguous checklist items. Using the Guidelines for Reporting on Latent Trajectory Studies (GRoLTS) checklist, we compare LLM-generated assessments with expert annotations across three research topics and two checklist versions. Agreement is assessed using item-level accuracy, chance-corrected agreement, and preservation of study-level rank ordering.
We find that performance varies substantially across checklist items, with ambiguous and conditional criteria producing the greatest disagreement. Revising these items improves both raw and chance-corrected agreement. Although item-level misclassifications persist, LLM-generated scores often preserve the relative ranking of studies when high-agreement items are retained. These results indicate that reliable LLM-assisted appraisal depends not only on model choice but also on checklist design. The findings suggest that analyzing human-LLM disagreement can help identify problematic checklist items and support the iterative improvement of research synthesis workflows. 

---
# Decoupled Vision-Language System for Multimodal Understanding and Generation 

**Authors**: Yifan Xu, Baochen Xiong, Xiaoshan Yang, Donglin Di, Yaowei Wang, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.20382)  

**Abstract**: We introduce a new architecture design for multimodal large language models (MLLMs), Libra, capable of both multimodal understanding and generation. Libra architecture contains one vision system and one language system, connected by cross-modal bridges. This design decouples self-modal modeling and cross-modal interaction, enabling each modality to learn its unique representations while maintaining effective cross-modal comprehension. The decoupling is mainly achieved in a switch attention module and a switch FFN module, which dynamically routes the computation flow for self-modal modeling and cross-modal interaction scenarios. We evaluate the effectiveness in two important settings: \textbf{Libra-1} for the understanding-only image-to-text setting, and \textbf{Libra-2} for unified image-to-text understanding and text-to-image generation. In addition to the architecture design, we discuss various improvements on tokenization, positional encoding, and supervision. Experiments demonstrate that the dedicated Libra design enables mutual improvements on multimodal understanding and generation, achieving strong performance on both understanding and generation benchmarks. 

---
# EditPPT: Faithful Long-Deck Slide Editing via Structured Tool-Using Multi-Agent with Dual-Modal Validators 

**Authors**: Jiheon Kim, Kyudan Jung, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2608.20381)  

**Abstract**: Automating slide editing requires simultaneously satisfying modification accuracy, preservation fidelity, and robustness to deck length. Existing LLM-based systems often fail on real-world presentation files because they rely on idealized intermediate representations or open-ended code generation, which are prone to cascading errors in long decks. We introduce EditPPT, a multi-agent framework that reformulates slide editing as a constrained tool-selection problem. By executing localized shape-level operations through the native PowerPoint COM interface, EditPPT narrows the LLM action space while preserving the application-resolved structure of user-authored decks. By separating validation across modalities, our dual-modal validation provides more robust assessment of both instruction fidelity and visual quality. We also present DeckEdit-Bench, a benchmark with 28 human-authored decks, 582 slides, and 183 editing prompts across short, medium, and long deck tiers. Experiments show that EditPPT achieves a 99.5% execution rate, 88.7% slide-targeting F1, 82.5% instruction following, and 91.5% object preservation overall, while maintaining strong performance on long decks. Our code and benchmark are available at this https URL 

---
# TH-GNN: Heterogeneous Temporal Graph Neural Networks for LLM-Agent Shilling Attack Detection 

**Authors**: Shivam Swarup, Divya Prakash Shrivastava, Rakesh Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2608.20376)  

**Abstract**: LLM agents can now generate realistic shilling profiles, fluent reviews, and coherent ratings at scale, systematically defeating recommender-system defenses. Text-only detectors that flag semantic drift in review embeddings are blind to graph structure and temporal coordination, while graph-only detectors that exploit neighborhood anomalies cannot reason over review semantics or the cross-modal inconsistencies produced by LLM-generated content. We propose TH-GNN, a heterogeneous temporal graph neural network with a two-layer Heterogeneous Graph Transformer backbone that applies per-type and per-relation attention augmented with learnable sinusoidal temporal encodings on every edge. Cross-modal attention fuses structural user embeddings with frozen RoBERTa representations of reviews and item descriptions, while a GRU operating over log inter-arrival times captures temporal burstiness. Evaluated across five attack families and four benchmark datasets, TH-GNN achieves a grand-mean F1 score of 0.870, outperforming the strongest text-only baseline on Agent4SR attacks by 10.9 percentage points and 11.5 percentage points at the lowest injection rate. These results demonstrate the effectiveness of jointly modeling temporal, structural, and semantic signals for detecting sophisticated LLM-driven shilling attacks. 

---
# GRAFT: Adaptive DLM-Based Draft Tree Construction with Target-Distilled Edge Scoring 

**Authors**: Xuming Ye, Zeming Ma, Runjie Yu, Yuan Liu, Tianle Li, Shuhan Bai, Jian Zhou, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.20375)  

**Abstract**: Tree-based speculative decoding raises the mean accepted tokens of standard speculative decoding by verifying multiple draft paths, and existing tree builders typically construct these paths through parent-conditioned expansion, where each child token is generated conditioned on its parent path. This construction is incompatible with diffusion language model (DLM) drafters such as DFlash, which produces all future-position distributions in a single forward pass. DDTree bridges this gap by treating high-probability tokens from each future-position distribution as candidate nodes and selecting edges between consecutive positions under a fixed node budget. However, its edge selection relies on token probability alone without modeling parent--child compatibility, so target-compatible tokens can be attached to wrong parents; moreover, its fixed budget ignores that the throughput-optimal tree size varies with the decoding state. We propose GRAFT, a draft-tree construction framework for DLM-based speculative decoding. GRAFT introduces Target-Distilled Edge Scoring (TDES), which distills parent--child preferences from target-model traces to select target-compatible edges, and State-Aware Budget Allocation (SABA), which sets the per-round tree budget by balancing expected draft gain against verification cost. Across multiple models and tasks, GRAFT achieves $2.13\times$--$6.36\times$ end-to-end speedup over autoregressive decoding while adding less than $0.5$\,ms of overhead per round, approximately $1.4\%$ of the target-model verification latency. 

---
# VA-DPO: Valence-Arousal Direct Preference Optimization for Controllable Emotion Generation in Language Models 

**Authors**: Hyunwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.20374)  

**Abstract**: How precisely can we tell a language model how to feel? Most work on emotional generation answers with a discrete label - happy, angry, sad - which cannot express a target like "mildly downcast but calm." We instead specify the desired affect as a continuous point (v*, a*) in the Valence-Arousal plane and train the model to hit it. Our method, VA-DPO, is a small modification to Direct Preference Optimization: a frozen VA regressor scores each sampled generation by its Euclidean distance to the target, we keep only candidate pairs whose distance gap clears a margin tau, and we optimize a LoRA adapter with the ordinary DPO loss against a frozen reference. The DPO objective itself is unchanged; what is new is how the preference data is built. On Llama-3.1-8B-Instruct this cuts mean VA distance to the target by 33% over system-prompting and 25% over few-shot prompting, lifting valence/arousal correlation to r_v=0.93 and r_a=0.75. The gains carry over to Qwen3-8B and Llama-3.2-3B, and they do not come at the usual price: MMLU is unchanged (Delta=+0.0) and HellaSwag and TruthfulQA are preserved. We release the code, configs, and the preference-construction pipeline. 

---
# An ambiguity taxonomy for evaluating large language model performance on clinical registry abstraction: a multi-site prospective study 

**Authors**: James Matheson, Betsy Castillo, Andrew Y. Shin, David Scheinker  

**Link**: [PDF](https://arxiv.org/pdf/2608.20373)  

**Abstract**: Objective: To evaluate large language model (LLM) performance on unprocessed electronic medical record (EMR) data for clinical registry abstraction. Methods: We evaluated LLM performance answering registry questions for the American College of Cardiology National Cardiovascular Data Registry (ACC NCDR). In a pilot study at an academic medical center, the model identified candidate data sources for each registry question and experienced abstractors used these results to define question-specific document sets. In a validation study at a second center with a second ACC NCDR registry, the LLM answered questions using the question-specific document sets. Before reviewing any output, two abstractors independently established the ground truth and assigned each question to one of six categories, ordered by the ambiguity and clinical reasoning required to resolve it: Medication/Event Flag, Binary Clinical Presence, Administrative, Quantitative Laboratory/Physiologic, Clinical Interpretation, and Event Timing. Results: The analytical sample comprised 9,430 abstractor answers reconciled to 4,715 consensus answers (501 pilot; 4,214 validation). In the pilot, candidate data sources per question averaged between 14.6 (SD 13.9) for demographics and 89.2 (SD 56.1) for history and risk factors. In validation, human inter-rater agreement was approximately 98\% while 87\% of LLM answers exactly matched consensus, 2\% partially, and 9\% did not. Mean question-level accuracy was 91.5\% (SD 13.4\%) across 157 questions with at least 20 answers, and declined as ambiguity increased, from 96\% for Medication/Event Flag to 62\% for Event Timing questions. Conclusions: LLMs answering clinical registry questions on unprocessed EMR data achieved far lower accuracy than human abstractors. LLM accuracy fell steadily as ambiguity and the level of required clinical reasoning increased. 

---
# When Do LLMs Replace Fine-Tuned NLU? A Decision Framework for Intent Detection in Production Conversational Systems 

**Authors**: Carson Rodrigues, Oysturn Vas  

**Link**: [PDF](https://arxiv.org/pdf/2608.20371)  

**Abstract**: A common claim is that zero-shot large language models (LLMs) can replace fine-tuned NLU classifiers for intent detection. We test this claim head-to-head and find that the honest answer is: it depends on the intent space. On full ATIS and CLINC150 we compare a fine-tuned RoBERTa, a TF-IDF+logistic-regression baseline, sentence-embedding kNN, and Claude Haiku zero-shot, reporting bootstrap 95% confidence intervals and paired significance tests. When abundant in-domain labels exist, fine-tuned RoBERTa is as good or better and three orders of magnitude cheaper and faster: on ATIS it beats Claude zero-shot by 11.8 points (95.9 vs. 84.1, p<0.001). On the broad 150-intent CLINC150 schema the two are statistically tied (89.1 vs. 88.5, p=0.24): the LLM matches a fully supervised model with no training data. The LLM's advantages appear in three production-relevant regimes: out-of-scope detection (OOS recall 85.6 vs. 58.1 for RoBERTa); robustness to realistic ASR noise via a controlled text-to-speech to noise to Whisper pipeline (92.5 vs. 80.0 at 0 dB); and dynamic per-deployment schemas, where a classifier trained on one app's intents scores 0% on a new app's intents while the schema-prompted LLM serves both at ~94% with zero retraining. We distill these findings into a decision framework for practitioners. 

---
# ASTAR: Automated induction of STAndardized radiology Reporting templates from large-scale clinical free-text corpora 

**Authors**: Xinfeng Zhang, Mingxuan Liu, Yifei Chen, Juncheng Zhu, Kasidit Anmahapong, Yiming Huang, Yuan Zhang, Hongjia Yang, Yi Liao, Gang Ning, Haibo Qu, Qiyuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2608.20369)  

**Abstract**: Structured reporting converts free-text radiology narratives into queryable data keys, facilitating cohort assembly, longitudinal tracking, and training label generation for medical AI. The prevailing paradigm follows a two-stage pipeline: (1) constructing a reporting template, (2) extracting information to populate it. While the extraction stage has benefited from advances in large language models (LLMs), template construction remains a manual bottleneck relying on labor-intensive expert consensus that is static, difficult to scale, and may fail to capture real-world reporting diversity. We address this limitation with \textbf{\texttt{ASTAR}}, an LLM-based framework for Automated induction of STAndardized radiology Reporting templates from large-scale clinical free-text corpora. Extensive experiments on 4,215 fetal brain MRI reports from multiple centers demonstrate that the \textbf{\texttt{ASTAR}}-induced template surpasses two expert-curated templates across template coverage, information fidelity, diagnostic fidelity, and expert-rated usability, reducing template development from weeks of committee deliberation to hours of automated processing. Code: this https URL 

---
# Research Paper Quality Recognition Through Textual Feature Analysis 

**Authors**: Saikiran Korla, Sadwik Gummadavelli, Trung-Nghia Le, Minh-Triet Tran, Tam V. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2608.20368)  

**Abstract**: Knowledge and innovations are shaped by using the quality and credibility of the scientific research. Yet, distinguishing between impactful, high-quality work and flawed studies remains a challenge. This paper introduces a benchmark for classifying research papers into two categories: good (highly cited) and non-good (retracted), using only textual features from titles and abstracts. We evaluate multiple embedding techniques, including SBERT, Word2Vec, FastText, USE, and TF-IDF, combined with classifiers such as Support Vector Machines (SVM), Random Forests, and Neural Networks. Our contributions include: (1) hyperparameter transparency, (2) feature space visualizations using t-SNE, (3) model interpretability analysis with SHAP, and (4) detailed examination of error cases. Experimental results show that a neural network with SBERT embeddings achieves 87.22\% accuracy, while FastText combined with SVM reaches 91.12\%. These findings highlight the value of textual information in assessing research quality, with ethical considerations for deployment. This work contributes toward the development of academic integrity tools that promote trustworthy scholarship. 

---
# Trilingual Topic Modeling of Sri Lankan Parliamentary Debates 

**Authors**: Himath Dhanapala, Haren Daishika, Himandhi Kuruppu, Sithija Seneviratne, Ashini Kavindya, Patalee Narasinghe, Sandeepa Weerasekara, Nisansa de Silva, Sandareka Wickramanayake  

**Link**: [PDF](https://arxiv.org/pdf/2608.20365)  

**Abstract**: Sri Lankan parliamentary debates (Hansards) constitute a trilingual corpus of speeches in Sinhala, Tamil, and English, including code-mixed content, yet remain inaccessible to standard NLP pipelines due to layout-complex PDFs, multilingual scripts, and agglutinative morphology. We present an end-to-end framework that addresses these challenges through LLM-based text extraction followed by a multilingual embedding and density-based clustering pipeline for topic modeling. A hybrid semantic-lexical extension, BiTopic, is further explored to improve interpretability and recover speeches otherwise discarded as noise. Applied to 19,553 speeches spanning 2017-2026, the pipeline recovers 30 macro-topics achieving a cluster purity (BCP) of 0.673, whose temporal trajectories align unsupervised with major national events including the 2019 Easter Sunday attacks and the 2022 economic crisis. Traditional LDA fails on this corpus due to cross-lingual fragmentation, whereas the proposed approach successfully identifies thematic structure across all three languages without supervision. 

---
# Hadith computational science in the age of large language models: a critical narrative review 

**Authors**: Md. Ashraful Haque, Riasat Islam  

**Link**: [PDF](https://arxiv.org/pdf/2608.20364)  

**Abstract**: We examine how hadith computational science is being reshaped by transformer models, retrieval-grounded pipelines, and large language models (LLMs). Recent reviews document growth in the literature, but they do not yet provide a critical account of which advances are methodologically robust, which remain benchmark-bound, and which unresolved problems still limit scholarly use. We address this gap through a critical narrative review that combines critique of existing reviews, paper-level appraisal of representative original studies, and synthesis of Islamic scholar and domain-expert perspectives on authenticity, authority, and responsible use. We find uneven progress. Data resources have expanded, segmentation tasks have matured, narrator and source-verification problems are better formalized, and LLM-assisted workflows now support corpus-scale enrichment, multilingual access, and grounded evaluation. At the same time, progress remains constrained by narrow corpora, weak benchmark comparability, synthetic-to-real transfer gaps, narrator identity resolution, preprocessing fragility, limited reproducibility, and sparse expert-grounded validation. We show that important gaps lie beyond dominant benchmarks: non-canonical and obscure corpora, commentary and explanatory literature, cross-source links with Qur'an and seerah, and fiqh-facing evidence support. We argue that hadith computation should be assessed less as isolated model performance than as an evidence infrastructure problem requiring knowledge integration, provenance, and expert supervision. On this basis, we define a research agenda for making the field methodologically stronger and more useful to Islamic scholarship. 

---
# Multilingual Verifier Bias in RLVR: Benchmark, Rollout Diagnosis, and the Cross-Lingual Selection Bottleneck 

**Authors**: Chenyu Zhou, Qiliang Jiang, Xu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.20362)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is a standard recipe for training large language models on mathematical reasoning, where an answer verifier serves as a language-neutral reward function. We show that this assumption fails in multilingual settings: an exact-match verifier turns format and script variation into language-dependent false-negative reward noise. We introduce a reusable protocol for auditing multilingual RLVR rewards: a verifier-robustness suite, a rollout-diagnosis procedure, and language-conditioned reward-error metrics for Japanese, English, and Chinese answers. On MGSM rollouts with k=8, the exact-match proxy rejects trusted-correct answers at sharply different rates by language across Qwen3-4B, Qwen3-8B, and Llama-3.1-8B-Instruct; for Qwen3-8B, the false-negative rate reaches 0.642 on JP against 0.122 on EN and 0.073 on CN. A plain-numeric probe localizes the mechanism to the final-answer interface: an interface model drives reward-error VLB to zero while the residual accuracy gap is unchanged. We then expose a cross-lingual selection bottleneck: on MGSM250 rollouts, a target-local aggregation rule using no trusted labels closes 55-78% of the average selection gap, and over 95% of repairs require genuine cross-lingual support. The bottleneck replicates on a 483-problem MATH-500 set. A controlled training audit shows that rule-GRPO raises trusted accuracy while the reward-error VLB stays high. The unifying message is operational: multilingual RLVR rewards should be audited by language and by answer interface before they are optimized. 

---
# Toward Auto-Research: Mining Falsifiable Research Ideas from Paper Knowledge Graphs with Categorical Structure 

**Authors**: Yuchen Wang, Zhongzhi Luan  

**Link**: [PDF](https://arxiv.org/pdf/2608.20361)  

**Abstract**: Automated research-idea generation systems built on large language models (LLMs) share a structural weakness: they reduce ideation to free-text recombination, random paper pairing, or embedding-similarity retrieval. The three approaches fail in the same way: each treats a paper as a flat object, a string or a vector, and so quotients away the typed problem-method-metric-claim arrows a researcher actually uses when reasoning about a cross-domain analogy. We recover the missing structure with the minimal piece of category theory that a typed graph alone does not provide: composition, together with identity arrows, which makes it possible to ask whether a proposed analogy preserves relation chains. Concretely, each paper $p$ is modelled as a small category $C_p$ whose objects are extracted typed research entities and whose morphisms are the relations the paper asserts; a cross-paper bridge from $p$ to $q$ is then a partial functor candidate $F: C_p -> C_q$ that preserves object kinds and covered relation classes. We instantiate the model as a three-layer algorithm: categorical signature clustering, a functor-preservation gate, and a six-axis LLM plausibility judge. Evaluated on a corpus of tens of thousands of full-text-parsed papers under four ablation conditions, the categorical gate filters cross-domain candidates at roughly a 17:1 ratio while the quantitative-falsifier rate of accepted ideas stays above 83% throughout; every rejected candidate is retained with its per-axis rationale, so the gate doubles as a logging layer rather than a silent filter. 

---
# TriPLU: Bypassing the Gate with Direct Trilinear Product FFNs in Tiny Language Models 

**Authors**: He Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.20360)  

**Abstract**: We study whether tiny decoder-only language models benefit from feed-forward layers that directly multiply learned feature projections. TriPLU, a Trilinear Product Linear Unit, replaces the usual gated FFN branch with a product-only degree-3 branch that multiplies three projected streams coordinatewise. In a character-level TinyStories 1M-byte prefix study, TriPLU reaches a mean best validation loss of 1.0637, compared with 1.1017 for closely matched SwiGLU, 1.0780 for a degree-4 product control, and 1.1026 for a degree-2 control. In train-only Byte-BPE experiments, TriPLU also lowers validation and heldout bits per byte on TinyStories and WikiText-2 raw under low-learning-rate settings, with PMI-slice evidence suggesting gains on seen middle- and high-PMI adjacent-token pairs. Constant-learning-rate diagnostics show that product-branch normalization can reduce the high-learning-rate best-checkpoint gap, although final BPB still degrades under hot schedules. The resulting claim is deliberately narrow: direct product FFNs can improve fixed-budget small-model loss in specific low-compute regimes, but the branch is optimization-sensitive and does not establish FLOP-normalized efficiency, scaling behavior, or broad LLM performance. 

---
# Self-Speculation for Faster Reasoning Models 

**Authors**: Ravisri Valluri, Tung Nguyen, Aditya Grover  

**Link**: [PDF](https://arxiv.org/pdf/2608.20359)  

**Abstract**: Large language models (LLMs) are deployed for increasingly complex tasks involving planning and multi-step decision making, but high-quality performance on these tasks often requires generating long reasoning traces. This is a poor fit for latency-sensitive and interactive applications like voice assistants or coding agents, where generation latency can strongly affect user experience. Existing acceleration methods typically focus on token-level generation, without utilizing the structure of reasoning workflows. We introduce SSR: Self-Speculation for Reasoning Models, a training-free self-speculative decoding method that leverages the chain-of-thought (CoT) as a source of speculation. SSR uses the partial-CoT answer distribution as the drafter and the full-CoT distribution as the verifier, deriving both from the same model at different reasoning budgets. This builds on the observation that later partial-CoT responses often exhibit greater semantic and lexical overlap with the full-budget response. Due to this overlap, SSR can accept long draft prefixes at once, leading to large speedups on structured and long-form generation tasks. To further exploit draft-response overlap beyond the contiguous prefix accepted by standard speculative decoding, SSR also incorporates suffix decoding, using the draft to seed a suffix cache and recover useful spans beyond the accepted prefix, further reducing latency on tasks with high lexical overlap between the draft and the final response. We evaluate SSR on multiple structured and long-form generation tasks where it is most useful, and demonstrate a relative improvement of up to 24.1% on total generation latency for popular open-source models such as Qwen3.5 and Gemma-4. 

---
# ExpertIVS: Sociological Expert Driven Individual Value Simulation in Large Language Models 

**Authors**: Zhen Wang, Yuqi Ren, Yuehan Cui, Hongxiang Wang, Jianxiang Peng, Zhaoxia Zhang, Bingkun Zhu, Tongxuan Zhang, Dezhi Tong, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2608.20355)  

**Abstract**: Large Language Model (LLM) agents have demonstrated considerable potential for social simulation, yet struggle to accurately model individual value systems. Most existing methods mechanically stitch survey responses into prompts, which suffer from semantic fragmentation, failing to capture the internal coherence of human value systems. The value systems of LLMs are typically assessed using static multiple-choice questions, which fail to evaluate the value orientation in real-world dialogue interactions. To address these issues, we propose ExpertIVS, a framework employing 14 Sociological Expert Agents to interpret World Values Survey (WVS) responses through structured professional perspectives, rather than direct responses concatenation. These expert agents perform deep semantic reconstruction to generate robust and internally consistent individual profiles. To evaluate the consistency between LLMs and individual value systems during dynamic interactions, we further introduce a multi-agent debate mechanism. Extensive experiments across 480 individuals from 12 countries demonstrate that ExpertIVS achieves 90.78% value restoration fidelity and significantly outperforms baselines in value generalization (+5.3%). Moreover, ExpertIVS exhibits strong personality discriminability and behavioral consistency, enabling a shift from mere response concatenation to genuine sociological role-playing. 

---
# The Divergence Hypothesis: Unmasking Lexical Interference and Label Bias in Mental Health NLP 

**Authors**: Moustafa Yehia Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2608.20353)  

**Abstract**: Computational mental health (CMH) classifiers often degrade under distribution shift because human annotators and distant-supervision pipelines reward different linguistic signals. We introduce TSS (Triple-Stream Stress probe), a multi-channel diagnostic framework that decomposes text into (A) lexical character n-grams, (B) a small, mostly content-free morpho-syntactic channel, and (C) a 154-feature psycholinguistic style channel. Across four English datasets (N=12,906), TSS reveals a lexical interference effect: adding lexical features to the style channel reduces Macro-F1 on human-labeled data (mean drop 0.072, p<10^-4) but not on auto-labeled data. We propose Degree of Divergence (DoD), a difference-in-differences statistic adapted from econometrics for label-source auditing, with instance-level bootstrap inference; the headline estimate is DoD(BC-A) = 0.0374, 95% CI [0.0097, 0.0651], p=0.0032. A platform-stratified Twitter-only DoD (which removes the Reddit vs. Twitter contrast) reproduces the pattern with bootstrap inference: DoD-Tw(BC-A) = +0.096 (p<0.001) and DoD-Tw(AC-A) = -0.089 (p<0.001). Interventional masking (pos_only) retains ~95-99% of Channel C's performance after destroying content words on human datasets, indicating that the style channel does not rely primarily on lexical surface form. TSS is positioned as a diagnostic audit framework, not a clinical screening tool: it flags label-source-specific shortcut learning before generalization claims are made. 

---
# Exploratory As-Analyzed No-Detection of Culturally-Marked Predicate-Triggered PII Amplification in a Synthetic-English RAG Probe: A Predicate-Resource-Confounded Audit 

**Authors**: Yanhang Li, Zhichao Fan, Zexin Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2608.20351)  

**Abstract**: We ask whether stereotype-loaded queries about culturally marked people leak more personal information from a retrieval-augmented generation (RAG) system than otherwise-equivalent neutral queries. We pre-register a four-culture audit (en-Anglo, es-LATAM, Arabic, Hindi) on a synthetic English PII corpus, comparing five query arms we call the Stereotype-Trigger Leakage Delta (STLD). Two caveats up front. Our locked confirmatory estimator was never run, so every test in the paper is exploratory or sensitivity, with all plan deviations listed in the appendix. And the name-leakage metric is contaminated by a prompt-echo artifact: the model often just re-emits the name we asked about, which inflates apparent leakage without any retrieval at all. On the cleaner channels (email, phone, ssn-like, address), we find no stereotype-driven amplification on any of the four cultures after multiple-comparison correction. Because our sample is only powered for mid-sized effects, and because the culturally marked probes mix stereotype content with cultural markers and heritage practices, we present this as no detection, not evidence of no effect, of culturally marked predicate leakage that is confounded with the underlying resource. 

---
# How to Train a Real-World Silicon Concierge? Internalizing Complex Business Workflow to Only OneModel 

**Authors**: Chang Liu, Chaoyang Ning, Dayi Jiang, Enrui Gu, Fang Ran, Hongyan Xue, Huaqing Li, Hui Cai, Jia Liu, Jiang-Ming Yang, Jianshe Li, Jiawei Luo, Jin Zhou, Leshen Zhu, Lihui Chen, Liying Ma, Lyuxin Xue, Mengjian Ji, Ruijia Xu, Wei Ren, Wei Wu, Xiaoling Qu, Xiaoyun Feng, Xin Zhang, Xixie Zhou, Xuanwei Hu, Yan Chen, Yichao Wang, Yongqi Tong, Yu Liu, Yuhong Zhou, Zemin Sun, Zhenwen Xu, Zhiling Liu, Zifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.20350)  

**Abstract**: Traditional industrial agents rely on modular pipelines, including Router, Retriever, Planner, Executor, Responder, Reviewer, and other components. These systems often fracture into a labyrinth of ad-hoc patches, leading to cascading errors and high latency. We propose OneModel, an applicable paradigm shift from external workflows to internalized knowledge representation. Unlike modular systems that slice fluid user intents into static steps, OneModel consolidates complex business logic and SOPs directly into the model parameters. Through Continual Pre-training (CPT) and logic-compilation SFT, we transform fragmented business rules into intuitive model reasoning within a unified attention space. Deployed in our global financial service system, OneModel effectively breaks the trade-off between latency, accuracy, and complexity. Online A/B testing demonstrates an end-to-end latency reduction of more than 50 percent, from 18.7 seconds to 8.0 seconds, while the Intelligent Resolution Rate (IRR) increases from 64.3 percent to 83.3 percent. The results show that OneModel can replace brittle engineering logic with internalized cognitive intuition, offering a scalable blueprint for transitioning industrial agents from complex, error-prone workflows to unified model architectures. 

---
# Beyond Prompt Engineering: A Systematic Analysis of Prompt Lexical Sensitivity and Its Impacts on Quality 

**Authors**: Qipeng Xie, Zi Liang, Jiafei Wu, Yufei Chen, Weizheng Wang, Wenao Ma, Zhong Ming, Haiqin Yang, Kaishun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.20349)  

**Abstract**: Large Language Models (LLMs) exhibit extreme sensitivity to surface-level prompt variations, in which minor lexical changes can trigger disproportionate performance fluctuations. Moving beyond black-box optimization and coarse-grained templates, we present the first large-scale, n-gram token-level mechanistic analysis of prompt stability, leveraging a dataset of 132,000 prompt variants. Our investigation reveals a fundamental Scaling Law of Prompt Performance Stability: higher average task performance is strongly associated with lower variance and greater robustness across prompt perturbation. We identify two core linguistic drivers underlying this robustness: (1) Domain-Specific Terminology, which tightly anchors semantic boundaries, and (2) Explicit Action Directives, which formalize reasoning trajectories. Together, these elements constrain the model's interpretative space, effectively ``locking in'' more deterministic generation behavior. Building on these insights, we introduce an automated Prompt-Refining Agent that systematically restructures input queries by injecting domain anchoring and operational constraints. Empirical evaluation shows that our approach reduces performance variance by 40.7% in code generation task, while preserving or improving mean performance. These findings provide a statistically grounded and mechanistically interpretable framework for achieving robust prompt engineering. 

---
# Inhibitory Attention for Clinical Long-Context Reasoning: Characterizing and Mitigating Lost-in-the-Middle Effects in EHR Processing 

**Authors**: Sanjay Basu  

**Link**: [PDF](https://arxiv.org/pdf/2608.20348)  

**Abstract**: Electronic health records now routinely exceed 100,000 tokens per patient. Yet large language models exhibit the lost-in-the-middle (LitM) effect: information near the center of a long context is retrieved less reliably than information near the edges. In clinical use this is not benign: the single most consequential fact in a note can sit at its center. We term this the clinical lost-in-the-middle (CLitM) problem, give its first systematic characterization using MedAlign, and compare context-selection strategies as remedies. Across 2,196 instruction-response pairs and six language models, we observe a 21.9 percentage-point gap between peak accuracy (59.5%, 95% CI [46.3, 71.0], 20-30% decile) and trough accuracy (37.6% [23.2, 52.5] at 70-80%); 67.8% of reference answers fall between the 10th and 90th percentiles of the EHR timeline, inside the CLitM trough. We introduce Query-Conditioned Clinical Suppression (QCCS), a lightweight query-conditioned selection gate, and evaluate it against BM25, BM25 with section-header filtering, dense retrieval, and cross-encoder reranking (N=83 held-out instructions). With Qwen2.5-7B-Instruct (16k context), QCCS outperforms all five comparators under LLM-as-judge scoring: for middle-position instructions QCCS reaches 16.7% versus BM25 3.3%, cross-encoder 0.0%, dense 0.0%, and full context 6.7%; overall QCCS reaches 25.3% versus at most 3.6% for retrieval-only comparators. This advantage is not explained by retrieval recall: at k=20, BM25 retrieves the gold evidence sentence in 98.8% of instructions (QCCS 34.9%), yet retrieval arms stay at most 2.6% accurate even when they retrieve it, whereas QCCS reaches 25.0% even when it does not. In this proof-of-concept evaluation, query-aligned context selection predicts EHR instruction-following accuracy better than gold-sentence retrieval recall. 

---
# Who Do Language Models Think Is Competent? A Mechanistic Analysis of Occupational Bias 

**Authors**: Keren Fuentes, Aaron Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2608.20347)  

**Abstract**: Language models (LMs) often pass behavioral bias evaluations, but it remains unclear whether they no longer represent the underlying associations that give rise to biases, or have merely learned not to express them. In this study, we show that representational biases are often detectable, even when behavioral biases are not visible. We introduce a causal framework that decomposes occupational bias into two measurement points: a model's internal representation of a user's competence, and its observable outputs. We derive steering vectors for representations of user expertise, and verify that they causally mediate model behavior in both a question-answering task and a hiring task. Applying this framework to several open-weight models, we find that demographic attributes, such as gender, race, and socioeconomic status, influence a model's representation of user expertise, even in cases where behavioral metrics detect no disparity between demographics. We show that these model representations can influence downstream behavior under intervention, suggesting failure modes that behavioral metrics alone may not detect. 

---
# Building and Evaluating a Synthetic Bengali Speech Resource for Telecom Customer Care 

**Authors**: Kawshik Kumar Paul, Md. Nafiul Alam Fuji  

**Link**: [PDF](https://arxiv.org/pdf/2608.20346)  

**Abstract**: Speech systems used in customer-facing applications often require domain-specific language coverage. We present a synthetic Bengali speech dataset for telecom customer-care scenarios. The dataset contains 10,000 audio-text pairs, approximately 26.82 hours of 24 kHz speech, and predefined train, validation, and test splits of 9,000, 500, and 500 examples. It is publicly released on Hugging Face under the CC-BY-4.0 license. The speech was generated with OmniVoice in voice-cloning mode using a real female reference recording and transcript, with bfloat16 precision, 16 diffusion sampling steps, and a speaking-rate control value of 1.0. Along with the original Bengali text, the dataset provides a normalized transcript field designed for ASR/STT training and evaluation. We report an automatic intelligibility check over all 10,000 samples using a domain-adapted Whisper ASR model fine-tuned from bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium, along with a manual listening check on selected samples. The evaluation gives an average WER of 2.54%, an average CER of 0.59%, and median WER and CER values of 0.00%. These results suggest strong text-audio consistency under the selected automatic evaluation pipeline, while the paper also discusses the limitations of synthetic speech and STT-based evaluation. 

---
# When Vocabulary Comprehension Fails Clinical Reasoning: Evaluating Therapy Bots' Safety Risks for Generation Alpha 

**Authors**: Manisha Mehta, Virendra Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2608.20345)  

**Abstract**: Conversational AI systems have become informal mental health support resources for Generation Alpha (Gen Alpha, born 2010-2024), with 13.1% of U.S. adolescents (5.4 million) using generative AI for mental health advice. While these systems, from therapy apps to general chatbots, rely on large language models trained on extensive psychological literature, their safety for youth communication patterns characterized by hyperbolic language, ironic positivity, rapid semantic drift, and contextual polysemy remains unvalidated. Following multiple adolescent deaths linked to AI chatbot interactions, systematic evaluation is critical. We present two benchmarks: (1) 64 Gen Alpha mental health expressions validated by native speakers (ICC=0.72) and clinicians (kappa=0.78); (2) 75 multi-turn conversations (780 turns) with paired Standard/Gen Alpha versions. Across evaluations of LLM architectures underlying therapy apps and general chatbots - Claude, GPT-4o, Llama-3.1 - models understand 76-82% of vocabulary but correctly calibrate only 64-72% of clinical risk, creating a 10-14 percentage point (pp) vocabulary-comprehension gap (p<.001, d>0.48) absent in human therapists (3pp, p=.22). The gap is architecturally consistent and widens with ambiguity (7pp -> 18pp). We identify six failure patterns: sarcasm masking (29pp), minimization acceptance (43pp), informal style bias (24pp), risk-stratified ambiguity (19pp), semantic drift (19pp), context-dependent violence (7pp). Patterns compound; three or more yield 94% miss rates. Lightweight mitigations fail; only heavy scaffolding achieves human performance (6.4x cost). With 34% baseline miss rate yielding 146,880 estimated annual missed crises, we recommend mandatory human-in-the-loop architectures, quarterly youth-specific validation, transparent performance disclosure, and regulatory frameworks for youth-facing mental health AI. 

---
# Beyond Raw Transcripts: Structured Persona Extraction for LLM-Based Digital Twins 

**Authors**: Iris Ye, Tianze Deng, Ozan Candogan  

**Link**: [PDF](https://arxiv.org/pdf/2608.20344)  

**Abstract**: LLM-based "digital twins" aim to simulate how an individual would behavein new environments or respond to novel questions, given some representation of that individual's prior responses. A common approach constructs this representation from survey transcripts or summaries responses. Prior work shows that compressing long transcripts into shorter LLM-generated summaries does not significantly reduce predictive accuracy, suggesting that information volume is not the primary bottleneck.
In this work, we argue that the key limitation is instead structural:how persona information is organized before being provided to thesimulator model. We study this by comparing unstructured summaries with structured persona representations. First, we introduce a hand-craftedschema (BDE: Background, Decision procedure, Evaluation), grounded in consumer-behavior theory, and show that it improves predictive accuracy over raw transcripts by +1.91 percentage points on a homogeneous benchmark (Twin-2K-500), with similar gains on gpt-5.4-mini and Qwen3-8B as robustness checks. However, this fixed structure does not generalizeacross more heterogeneous tasks, where performance is statistically indistinguishable from the raw transcript baseline.
To address this limitation, we propose an automatic structure-discovery pipeline in which an LLM iteratively proposes and refines task-specific persona structures and extraction prompts. On a benchmark of 13 diverse sub-studies, this approach restores performance, improving mean accuracy by +1.91 percentage points over the raw transcript baseline and eliminating significant losses observed with the fixed schema.
Overall, our results suggest that the main constraint in LLM-based digital twins is not how much information is provided, but how it is structured -- and that the optimal structure depends on the task. 

---
# TurboBias 2.0: Streaming Context-Biasing for Production-Efficient ASR Systems 

**Authors**: Vladimir Bataev, Lilit Grigoryan, Andrei Andrusenko, Nikolay Karpov, Vitaly Lavrukhin, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2608.21343)  

**Abstract**: Contextualization is essential for production automatic speech recognition (ASR) systems, where user-provided phrases must be recognized accurately under strict latency constraints. Although many context-biasing methods improve recognition accuracy, they often do not address the practical requirements of modern production ASR systems: streaming inference, efficient batched decoding, user-specific context lists, and low runtime overhead. We propose TurboBias 2.0, a production-oriented framework for efficient phrase boosting in Transducer-based ASR systems. The framework extends GPU-accelerated TurboBias with a case-insensitive boosting graph and per-stream batched decoding, allowing each utterance in a batch to use an independent context-biasing configuration. This enables personalized context biasing for multiple simultaneous users without sharing or mixing their context lists. The proposed framework supports both offline and streaming inference and can be used with greedy and beam-search decoding. Experiments show that TurboBias 2.0 improves contextual phrase recognition while preserving low latency and high throughput. 

---
# Enhancing LLMs in Predictive Political QA with Semi-Structured Data 

**Authors**: Yinan Liu, Zihan Zhou, Zichun Jin, Xinyu Wang, Bin Wang, Xiaochun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.21218)  

**Abstract**: Predictive political question answering (QA), such as predicting how a political actor will vote, goes beyond factual lookup. External political resources offer rich historical evidence, but rarely contain the answer itself. Existing LLM augmentation methods, including actor-profile-based simulation and knowledge graph evidence injection, improve political reasoning but largely treat external resources as knowledge-based evidence, leaving prediction-relevant signals under-modeled. We identify two complementary signals for predictive political QA: actor stances that capture issue-specific preferences, and high-order structure signals that capture indirect dependencies among political actors. We propose PSL, a dual-view framework that converts semi-structured political records into inference-oriented evidence for LLMs. PSL extracts stance signals from question-relevant actor records in a semantic view, and learns structure-aware actor representations from an actor interaction graph in a vector view. Across three real-world datasets and multiple LLMs, PSL consistently outperforms baselines, with ablations confirming the complementary gains of stance and structure signals. 

---
# Personalized Privacy Control in LLMs via Attention Head Intervention 

**Authors**: Junseok Kim, Nakyeong Yang, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2608.21209)  

**Abstract**: The rise of agentic AI enables LLMs to access diverse user data, raising critical privacy concerns. Prior work on contextual privacy studies whether LLMs regulate information disclosure according to context-dependent norms. However, acceptable disclosure boundaries may vary across users even within the same context. To address this limitation, we introduce \textit{personalized privacy}, which incorporates user-specific disclosure preferences into privacy control. We further present P3Bench~(\textbf{P}ersonalized \textbf{P}rivacy \textbf{P}reservation \textbf{Bench}mark), a novel benchmark extending contextual privacy policies with personalized disclosure policies. Experiments show that prompt-based policies fail to reliably enforce personalized privacy policies, with Qwen2.5-7B and Gemma3-4B showing average policy ignorance ratios of 51.25\% and 74.28\%, respectively. Finally, to address this problem, we propose \textsc{Repair}, a robust inference-time attention head intervention method that adjusts disclosure behavior toward policy-consistent responses. Our method significantly improves adherence to user-specific privacy preferences by reducing cases where the model fails to follow the given policy. 

---
# Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems 

**Authors**: Balkrishna Giri, Md Toufique Hasan, Jussi Rasku, Muhammad Waseem, Pekka Abrahamsson  

**Link**: [PDF](https://arxiv.org/pdf/2608.21095)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds Large Language Model (LLM) outputs in external knowledge, but RAG systems usually trust whatever they retrieve, creating a Security-Reliability Gap: high semantic relevance does not guarantee factual truth. Adversaries exploit this through knowledge poisoning, inserting malicious documents to cause targeted misinformation. We propose an Evaluation Agent, middleware that combines Natural Language Inference (NLI) factual verification, a five-signal poison detector with relevance-weighted aggregation, and a Trust Index T = 0.4 F + 0.35 C + 0.25 (1 - P ) with a non-linear dampener for high-contamination contexts. On TruthfulQA with Llama 3.3 70B, the agent reaches 91% accuracy and 100% precision, with 100% recall on instruction injection, while in-place edits, such as entity swaps, remain hard to detect. Across three LLMs the Trust Index stays discriminative, with a Receiver Operating Characteristic Area Under the Curve (ROC-AUC) of 0.73 to 0.81; generation style matters more than model size, and per-LLM threshold calibration restores baseline competitive accuracy, whereas a weaker FEVER result shows that cross-dataset generalization requires domain-specific calibration. In a software-engineering use case, a secure-coding assistant over guidance from the Open Worldwide Application Security Project (OWASP) Top 10 and the Common Weakness Enumeration (CWE), the agent reliably blocks instruction injection of unsafe advice (F1 92%), while contradiction and subtle semantic weakening remain hard. Throughout, the agent measures detection of poisoned context before generation, not whether the LLM adopts the injected misinformation. We release the proposed approach, attack generator, and experimental artifacts at the link: this https URL. 

---
# COMET: Contrastive Motion-Enhanced Temporal Reasoning for Video Multimodal Large Language Models 

**Authors**: Chenghua Zhu, Zhaolu Kang, Qifan Shi, Siyan Wu, Kehan Jiang, Lei Wei, Lianyu Hu, Guangyuan Dong, Mingbo Yang, Rui Lu, Guibo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.21030)  

**Abstract**: Video multimodal large language models have advanced significantly, yet fine-grained motion-temporal understanding remains fragile. The core bottleneck is not only sparse frame sampling, but also the lack of a complete temporal modeling pipeline for explicitly representing frame-to-frame change, enabling appearance-motion interaction, and optimizing temporal direction sensitivity. We propose COMET, a temporally grounded framework that systematically strengthens video MLLMs through explicit temporal representation, appearance-motion fusion, and direction-aware optimization. Architecturally, COMET introduces a temporal motion branch built on Taylor frame differences and injects its motion evidence into the appearance stream via temporal attention bias-enhanced cross-attention. For optimization, COMET combines temporal prior distillation with a forward-reverse TC-GRPO stage that turns temporal order into a direct learning signal and strengthens the model's use of directional motion patterns encoded by the temporal motion branch. The method achieves consistent overall improvements with a pronounced motion-temporal bias: on Qwen3-VL-8B, action-centric tasks (STAR, SSv2) improve by 4.9% on average, temporal reasoning tasks (NExT-QA, CLEVRER, LLaVA-178K) by 2.1% over BL-GRPO, while static perception tasks (PerceptionTest) remain on par. The same gain pattern also transfers to InternVL2.5-8B, indicating that COMET generalizes across model families. 

---
# MigrationNarrate: A Dataset for Detection of Migration Narratives in YouTube Videos 

**Authors**: Fatima Haouari, Carolina Scarton, Kalina Bontcheva  

**Link**: [PDF](https://arxiv.org/pdf/2608.20984)  

**Abstract**: Narratives are central to how social communication is framed, making their detection critical for understanding and analysing public discourse. Prior work has explored narrative detection and extraction across diverse domains; however, migration narratives remain significantly understudied, primarily due to the absence of dedicated annotated datasets. Furthermore, public communication has recently shifted towards video-centric platforms, where narratives are conveyed through multimodal signals and consumed at scale. Despite this shift, narratives in videos remain largely unexplored. To bridge these gaps, we introduce MigrationNarrate, the first multimodal dataset for detection of migration narratives in the UK, consisting of 1,115 YouTube video transcripts annotated using a two-level taxonomy of 12 migration super-narratives and 53 narrative labels. This paper details the dataset design, collection, and annotations; together with benchmark results using a combination of pre-trained encoder models and both open- and closed-source Large Language Models. Finally, a thorough error analysis offers insights for future work. 

---
# TreeWY: Speculative Verification for Gated DeltaNet Hybrids 

**Authors**: Sneha Murthy Ghantasala  

**Link**: [PDF](https://arxiv.org/pdf/2608.20961)  

**Abstract**: Modern open models are hybrids: most layers are linear-attention (Gated DeltaNet, GDN) layers carrying a small fixed-size recurrent state instead of a growing key-value (KV) cache. This makes ordinary decoding memory-efficient, but hurts speculative decoding. To verify a batch of draft tokens and then roll back the rejected ones, today's systems snapshot the full recurrent state at every draft position for GDN layers, and those snapshots cannot be shared across branches of a draft tree, so a wide, high-acceptance tree becomes memory-infeasible. We remove the snapshots. Using a tree-structured WY transform of the gated delta rule, we compute every draft node's output with a single triangular solve and reconstruct only the one accepted state on commit, storing a small pseudo-value matrix instead of per-node states; the derivation depends only on the gated delta rule, not on any other architectural detail. In serving benchmarks on two scales of one hybrid model family (Qwen3.5 35B and 397B) this cuts speculative recurrent-state memory and KV-cache pressure at identical acceptance length, turning the freed HBM into higher throughput and much lower time-to-first-token (TTFT) wherever memory binds, and costing a few percent where it does not. For tree width the same memory buys affordability: a wider, higher-acceptance draft becomes possible, though not yet a throughput win. 

---
# Identify, Locate, Link: End-to-End Key-Value Extraction from Document Images 

**Authors**: A. Said Gurbuz, Ahmed Nassar, Christoph Auer, Maksym Lysak, Lucas Morin, Matteo Omenetti, Tim Strohmeyer, Panagiotis Vagenas, Nikolaos Livathinos, Michele Dolfi, Peter Staar  

**Link**: [PDF](https://arxiv.org/pdf/2608.20868)  

**Abstract**: Document processing pipelines traditionally cascade optical character recognition (OCR) engines with downstream models for structured information extraction, leading to multi-stage error propagation. We fine-tune SmolDocling, a compact 256M-parameter vision-language model (VLM), to perform end-to-end key-value extraction directly from document images, jointly solving identification, localization, and association in a single pass without OCR preprocessing. We extend DocTags with specialized key, value, region, and link tags, enabling many-to-many relationships in a unified output sequence. To address data limitations, we design an augmentation pipeline combining synthetic form filling and graph-based crops that preserve complete key-value subgraphs. We further introduce a layout-aware evaluation framework extending text matching with spatial bounding box verification. On FUNSD, XFUND, and a large-scale private dataset, our model outperforms larger zero-shot VLM baselines under layout-aware evaluation, while being 27 times smaller than Qwen2.5-VL (7B) and over 5 times faster at inference. The model weights will be released publicly after publication. 

---
# Profiling What Matters: Context-Aware Item Profiles from Large-Scale Metadata for LLM Recommenders 

**Authors**: Dojun Hwang, Seunghan Lee, Cheonyoung Park, Sara Yu, SeongKu Kang  

**Link**: [PDF](https://arxiv.org/pdf/2608.20801)  

**Abstract**: While Large Language Models (LLMs) have significantly advanced reranking in recommendation, effectively leveraging item-side information remains challenging. Real-world items are described by vast, heterogeneous, and unstructured metadata, where decision-relevant signals are often implicit, noisy, or buried in long descriptions. Moreover, feature salience is highly context-dependent, varying not only across items but also across users. Existing methods often rely on item titles, fixed attributes, or static item summaries, which limit personalized and fine-grained item understanding. To bridge this gap, we propose CAIRO, a user context-aware item profiling framework for LLM-based reranking. CAIRO first structures raw metadata and reviews into objective features and subjective traits, and employs a lightweight profiler to select the most relevant information for each user-item pair with limited serving-time overhead. The resulting profiles are concise and context-specific, providing relevant item-side evidence for the LLM's ranking decision. Experiments show that CAIRO consistently improves LLM-based reranking, highlighting the importance of item profiling that effectively exploits vast item-side information. 

---
# Calibrating Criterion Revision in LLM Agents: Failure Modes and a Trace-Anchored Protocol 

**Authors**: Guodong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.20729)  

**Abstract**: Language-model agents can improve after failure or carry text across episodes without revising what counts as success. We study the narrower attribution problem of criterion revision: when criterion K0 accepts an outcome violating a broader commitment B, what observations justify saying that the system formed and persistently used K1? We require five non-compensatory conditions: criterion-failure detection, a model-emitted proposal, new-episode transfer, intervention sensitivity on the claimed carrier, and preservation.
We evaluate CMB-0.1 on twelve cross-domain cases and four arms: stateless inference, append-only history, model-generated but harness-committed state, and evaluator-written oracle state. Seven mechanism fixtures yield 84 deterministic scorer trials; four local quantized artifacts yield 96 calls and 192 model-case-arm trials. No model trial satisfies all five conditions, but this zero does not establish general capability absence. Eleven calls remain invalid after one retry; several commitments disclose the target distinction; the harness performs commits; deletion reuses a stateless call; and conflict changes multiple factors. Qwen2.5-7B answers every transfer and preservation item without revision state, exposing zero-state reconstruction.
These failures make CMB-0.1 an instrument-calibration result rather than a model ranking. We derive a prospective, trace-anchored CMB-0.4 protocol requiring concealed transfer, explicit WRITE/NO-WRITE/ESCALATE actions, a separately logged policy-selected commit, matched interventions, repeated hidden items, and a frozen executable oracle. It is a successor design, not a completed confirmatory result. The paper contributes a measurement chain, an empirical diagnosis of its first implementation, and a more discriminating protocol for future tests of criterion revision. 

---
# Temporal Validity on Real Software Histories: Eliminating Stale-Fact Errors in Code-Assistant Memory over GitHub Fixes 

**Authors**: Neeraj Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2608.20685)  

**Abstract**: Retrieval-augmented generation (RAG) has no model of time: when a fact changes across a coding session - a function is renamed, an endpoint moves, a dependency is bumped - RAG retrieves both the old and new value with near-identical similarity and cannot tell which is current, so it serves the superseded value. Paper 1 showed, on synthetic single-value benchmarks, that a deterministic (subject, relation, object) supersession memory eliminates this failure. Here we validate it end-to-end on real software history. From 707 real GitHub issues (SWE-bench Lite + Verified) we extract 130 clean atomic state transitions, a fix that changes one identifiable value from a pre-fix to a post-fix form, and render each marker-free (the stale and current statements differ only in the value). On this set, MemStrata reaches 0.91 answer accuracy versus RAG's 0.57-0.59; and, the structural result, when forced to answer RAG serves the superseded value 36-38% of the time (an LLM reranker does not help) while MemStrata drives this to ~0, at RAG retrieval latency (~2.1 s vs ~18 s for the reranker). We are explicit about scope: only ~18% of real fixes are clean atomic transitions; Paper 2 isolates the memory mechanism on that class, and extraction coverage of the remaining fixes is the orthogonal problem we defer to follow-on work. A real product bug surfaced and was fixed during the study (a case/punctuation-insensitive value comparison), with the moat property (deterministic-supersession accuracy on clean code mutations) preserved and verified. 

---
# Why2Speak: Faithful Reasoning for Abstaining Action Policies 

**Authors**: Shreya Mendi, Brinnae Bent  

**Link**: [PDF](https://arxiv.org/pdf/2608.20670)  

**Abstract**: Many agentic systems must repeatedly choose between acting and abstaining, making faithful reasoning important for oversight: an explanation is useful only if it reflects the computation that produced the action. We study this problem through intervention timing in multi-party conversation, where an assistant must decide whether to speak or remain silent. This setting exposes class imbalance, asymmetric action costs, and the possibility that exposing reasoning changes the policy being audited. Using Qwen3-8B, decoded with or without chain-of-thought reasoning, we compare direct decision policies, reasoning policies, supervised fine-tuning, and reinforcement learning. We find a capability-auditability tradeoff: the strongest direct policy achieves higher quality but exposes no reasoning to inspect, while the reasoning policy provides a trace at the cost of lower performance, particularly recall of true intervention opportunities. Supervised fine-tuning either suppresses reasoning or preserves it without improving decision quality, while reinforcement learning also fails to improve the reasoning policy. We identify one mechanism underlying this failure: group relative objectives provide no learning signal on confidently wrong prompts when sampled rollouts all select the same action. Controlled activation probes and behavioral ablations show that standard faithfulness methods can overstate evidence that exposed reasoning reflects the underlying decision process. Probability-based metrics saturate under confident decisions, probes are vulnerable to class imbalance and textual leakage, and reasoning ablations can confound reasoning content with changes in inference mode. Together, these results show that exposing reasoning can change an agent's action policy rather than simply make it observable. We provide controls for evaluating reasoning-based oversight of agents that can act or abstain. 

---
# Auditable by Construction: An Ontology-Driven Framework for Trustworthy LLM Analytics in Enterprise Finance 

**Authors**: Sergiy Lunyakin  

**Link**: [PDF](https://arxiv.org/pdf/2608.20661)  

**Abstract**: Enterprise adoption of large language models in finance is constrained less by fluency than by trust: in Financial Planning and Analysis (FP&A) and other regulated workflows, an answer is usable only if it is traceable to authoritative sources and auditable after the fact. This paper argues that retrieval-augmented generation for enterprise finance should be evaluated on auditability alongside accuracy, and presents the Knowledge-Driven Analytics Framework (KDAF), which builds ontology-driven knowledge systems through six iterative stages and retrieves evidence via Context-Aware Relevance Propagation (CARP), so that every retrieved fact carries its relationship type, confidence, and source lineage.
An evaluation on FinanceBench (145 questions) compares KDAF against zero-context inference, BM25, concept-weighted lexical retrieval, and ungrounded graph traversal. First, retrieval is necessary: zero-context inference reaches 4.1% correctness against 10-12% for retrieval-augmented conditions. Second, on answer correctness the retrieval conditions are statistically indistinguishable (KDAF vs BM25: -0.007, 95% CI [-0.021, 0.000]), so accuracy alone does not justify structured retrieval here -- a negative result we report explicitly. Third, on auditability the ordering reverses: KDAF attains the highest citation traceability F1 (0.515), exceeding ungrounded traversal by +0.027 (CI [0.006, 0.050]) and BM25 by +0.052 (CI [0.024, 0.083]), intervals excluding zero. Graph-structured retrieval also admits no evidence from outside the question subject entity (0 of 426 items, against 16.8% and 20.2% for lexical baselines), and every selected item resolves to a complete provenance chain. We argue that auditability, not accuracy, is the axis on which ontology-grounded retrieval earns its cost. 

---
# Open-Weight Masked Introspection: Measuring What Language Models Can Report About Their Own Computation 

**Authors**: Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2608.20569)  

**Abstract**: Are frontier models able to introspect about their internal states? Recent work suggests that under certain conditions a complex enough model can audit its own internals, call out what changed, and report back confidently about it. We tested that claim on eight open-weight models from seven families and found no such ability: asked whether their own computation had been altered, none answered better than chance. To test it we built Open-Weight Masked Introspection (OWMI), a framework that intervenes on residual-stream sites, attention heads and sparse-autoencoder features, then interrogates the model about the change against the null conditions an answer has to beat: sham runs where nothing was altered, impact-matched random perturbations, and a text-only observer that sees only the visible output.
Over 78,000 measurements, no model's report discriminates a real intervention from a sham beyond chance (AUROC ~0.5007), and an equivalence test bounds the effect below 0.15 percentage points of AUROC. Surprisingly, all the information needed is in the models. A model fine-tuned to report this class of intervention reaches near-perfect recovery on held-out directions, and a linear probe recovers intervention presence from the same activations at 75% to 95.8% accuracy, sharpening to no held-out error at the last layer before the model speaks. In one model the signal surfaces in the confidence rather than the words: its yes-or-no report never varies, while the confidence attached to it separates intervention from sham at AUROC 0.647. The failure sits in the path from internal state to verbal report, so oversight that reads a model's own testimony needs validating against an internal reference.
While our results show the inability of current open-weight models to introspect, the debate is not settled for future models. 

---
# ProofJudge: Tool-Grounded LLM Evaluation of Formal Proof Quality in Mathlib 

**Authors**: Shane Caldwell  

**Link**: [PDF](https://arxiv.org/pdf/2608.20432)  

**Abstract**: Formal proofs in Lean 4 that pass the kernel's type checker can nonetheless vary widely in quality. We introduce ProofJudge, an agentic LLM-as-judge system that scores formal proof quality along five dimensions beyond correctness: library leverage, automation fit, structural clarity, statement quality, and Mathlib conventions. We evaluate ProofJudge on a novel dataset of 218 declarations drawn from distinct Mathlib PRs. The judge agent is grounded by tool access to the commit the PR is applied to, enabling it to query the library state when scoring. A judge is considered aligned with human preferences when it rates the version of the PR Mathlib accepted above the initial version that was sent back for revision. All six judge models evaluated recover the reviewers' preference well above chance, from 80.8% to 63.5%, and two open-weight judges reach roughly 70% at a tenth of the best judge's cost. We release the judge harness, evaluation dataset, and evaluation traces as open-source artifacts to support further research. 

---
# When Retrieval Fails Before It Begins: Structurally Indirect Prerequisite Eviction as a Retention Failure in Agentic Memory 

**Authors**: Minkyu Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.20400)  

**Abstract**: Agentic memory under a fixed budget involves two stages: retention and retrieval. Existing retrieval-centered paradigms implicitly assume necessary evidence survives eviction, but we challenge this by isolating a pre-retrieval failure mode: structurally indirect prerequisite eviction, in which upstream blocks weakly aligned with the query are discarded under budget pressure. We provide an operational definition of this failure, a reproducible deterministic benchmark, and per-seed trace diagnostics. Finally, we evaluate Dependency-aware Semantic Garbage Collection (DSGC), a one-hop graph-aware rule. In our main suite, DSGC improves full-chain retention from 0.03 to 0.90 under a lexical encoder and from 0.23 to 1.00 under a sentence encoder. Robustness checks then identify the budget and scaling regimes where the one-hop rule holds or degrades. Our released pipeline and failure postmortem support mechanistic analysis of retention before retrieval as a distinct failure boundary. 

---
# A Factorial Ablation of a Speech-to-SFT Pipeline: Differential Effects on Data Quality and Downstream Transfer 

**Authors**: Wonsup Shin, Jingu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.20394)  

**Abstract**: Industry pipelines that turn speech into supervised fine-tuning (SFT) data via multi-stage refinement are increasingly adopted but, to our knowledge, have not been publicly ablated stage-by-stage, leaving each stage's marginal value unknown. We design a production-ready speech-to-SFT pipeline in which transcript refinement (Phase 0) and SFT data quality refinement (Phase 2) are independently toggleable, yielding a 2x2 factorial design. For each condition, we generate QA-form SFT data from Korean medical and finance conference recordings and fine-tune 9 models (5 LLM families, 2.4B-70B); we evaluate with four cross-provider LLM judges, a blind six-expert human evaluation, and 3 downstream MCQA benchmarks. Our central finding: under a fixed, standard SFT recipe, improvements in QA data quality do not transfer uniformly into downstream MCQA gains. 4-judge quality rises consistently, yet the cross-model mean MCQA gain is not significant; positive transfer concentrates on family-domain aligned pairs. This differential pattern is consistent with a format mismatch: Phase 2 shifts SFT-data composition toward explanatory items, while MCQA primarily probes factoid recall. All six human raters report higher full-pipeline quality, confirming the LLM-judge direction. An STT-engine swap to Whisper-medium confirms pipeline robustness. A non-hallucination audit shows the two frontier LLMs admit unknown on approximately 8% of QA on average; we release samples, prompts, code, and all SFT checkpoints. 

---
