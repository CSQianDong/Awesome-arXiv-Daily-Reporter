# Mental World Modeling 

**Authors**: Hao Fei, Yiran Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.27201)  

**Abstract**: World models enable a predictive substrate for planning and action, yet existing formulations merely answer a physical question: what/where it is, and how will it evolve. Human behavior, however, is driven by hidden mental state (what a person believes, wants, intends, feels, and considers socially permissible), so a model that tracks the physical scene but not what each agent knows and believes about it predicts the wrong action for the right-looking scene. We formulate Mental World Modeling (MWM), a generic theoretical framework that makes mental variables core components of a world model rather than posthoc rationales: MWM aintains a coupled physical-mental world state, renders a target-specific partial observation, and simulates how candidate actions jointly update both components. We instantiate the framework in MENTIS, a training-free and fully inspectable baseline that decomposes the process into state parsing, target-observation generation, action decomposition, coupled physical and mental transition, and branch-level value evaluation. On a manually constructed, quality-controlled dataset of situated decision scenarios spanning text, image, and sounding-video stories, experiments with 8 modern LLM-based world models demonstrate that explicitly modeling the mental state is essential for predicting human decisions. Deeper analyses further expose the bottlenecks of current mental world modeling. We expect MWM as a next stage of world modeling, from simulating physical scenes to simulating the minds that act in them. 

---
# APEX-Accounting 

**Authors**: Julien Benchek, Austin Bennett, Jasmin Kern, Ryan Stevens, Rene Sultan, Charis Ching, Hayley Popiel, Vaibhav Mittal, Felix Mercier, Brendan Foody, Bertie Vidgen  

**Link**: [PDF](https://arxiv.org/pdf/2607.27189)  

**Abstract**: We introduce APEX-Accounting, a benchmark built by Mercor in partnership with Ramp, to assess whether frontier models can do the real work of accountants. Tasks include reconciling accounts, accruing expenses, posting transactions, and producing reports. The private eval set comprises 160 tasks, split across 10 worlds. Each world contains an accounting system, as well as spreadsheets, PDFs, and other files. Every task was authored and solved by experts in accounting and bookkeeping, who also wrote grading rubrics. Across nine frontier models, Claude-Fable-5 (Max) leads with 56.4% Mean Criteria@3, ahead of Muse-Spark-1.1 (xHigh) at 52.6%. No model scores more than 2.6% Pass^8 (GPT-5.6-Sol (Max+Pro)) and the highest Pass@8 is 21.5% (Muse-Spark-1.1 (xHigh)). We experiment with increasing the token budget from $1 to $50 and observe an instance of Simpson's paradox: scores increase as the token budget increases but within a given budget-constrained harness, scores are lower on tasks where the model spends more tokens. As APEX-Accounting is a closed benchmark, leaderboard evals can be run for any frontier model on request. 

---
# Pangram 4 Technical Report 

**Authors**: Ben Glickenhaus, Katherine Thai, Jenna Russell, Elyas Masrour, Yue Han, Max Spero, Bradley Emi  

**Link**: [PDF](https://arxiv.org/pdf/2607.27183)  

**Abstract**: We present Pangram 4, the latest deep-learning-based AI-text classification model from Pangram Labs. We achieve an AUROC of 0.9916 with a false positive rate of 0.0041% and a false negative rate of 0.3396%. In addition to its increased overall accuracy compared with Pangram 3, Pangram 4 exhibits superior out-of-distribution generalization and robustness to adversarial attacks. Another novel contribution of Pangram 4 is its improved ability to distinguish fine-grained edits and mixed AI-human co-authored text. We demonstrate improvements to both boundary detection tasks and the detection of interleaved AI assistance. Finally, we report metrics on standard AI detection benchmarks showing that Pangram 4 achieves state-of-the-art performance on the AI text detection task across a wide variety of settings and domains. 

---
# DenseOn with the LateOn: Fully Open Dense and Late-Interaction Models for Multilingual, Long-Context, and Code Search 

**Authors**: Raphaël Sourty, Antoine Chaffin, Paulo Roberto Moura Junior, Amélie Chatelain  

**Link**: [PDF](https://arxiv.org/pdf/2607.27178)  

**Abstract**: State-of-the-art retrieval models increasingly rely on closed training data, creating a reproducibility gap. We present an open end-to-end recipe for training retrieval models and study how English supervision transfers to multilingual retrieval through translate-train. We first reconstruct and curate 665M English contrastive pre-training pairs from 1.4B pairs across 34 public sources and build 1.88M supervised fine-tuning pairs with mined hard negatives. Training yields two 149M-parameter models: DenseOn, a single-vector dense model, and LateOn, a ColBERT-style late-interaction model. They achieve 56.20 and 57.22 average nDCG@10 on BEIR, respectively, setting new state-of-the-art results for this size class. We then translate the validated English data into eight languages, yielding 2.8B pairs with cross-lingual samples, and train mDenseOn and mLateOn, two 307M-parameter models built on mmBERT-base. Despite sharing their backbone, data, and objectives, their representations behave differently: the dense model is strong on English and translated languages but degrades outside translate-train support, whereas the late-interaction model generalizes better to unseen languages and scripts. This suggests that token-level matching turns translate-train from a target-language expansion strategy into a multilingual generalization recipe. We publicly release the models, datasets, and training code. 

---
# Evaluating Regional Bias in LLMs From Abstract Stereotype to Concrete Social Decision-Making 

**Authors**: Jiayuan Di, Haoyi Yang, Yufei Luo, Jiahui Qu, Yiming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.27022)  

**Abstract**: Regional bias in large language models (LLMs) may shape both perceptions of regional groups and decisions about individuals from different regions. Yet existing studies often examine these manifestations separately, leaving their structure and consequences unclear. We introduce Stereotypes-to-Decisions (S2D), a systematic framework evaluating regional bias from abstract stereotypes to concrete social decisions. Covering all 34 provincial-level administrative regions of China, S2D evaluates six LLMs using stereotype ratings of Warmth (perceived friendliness and trustworthiness) and Competence (perceived capability and intelligence), along with paired-choice tasks across Education, Occupation, and Social Interaction. Results reveal substantial regional differences in regional scores, with considerable agreement across models, especially for Competence and Occupation decisions. Furthermore, these patterns are associated with regional economic and digital development indicators and display mixed human-like stereotypes, with some regions rated highly on one dimension but poorly on the other. They also remain largely stable across Chinese and English prompts. Overall, our findings show that regional bias in LLMs is prevalent, systematic, and consequential, motivating more regionally aware evaluation and mitigation. 

---
# OptimismBench: Forecasting Bias and the Alignment Effect in Language Model Judgment 

**Authors**: Seonglae Cho, Adriano Koshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2607.26981)  

**Abstract**: Large language models are increasingly used as decision aids whose probability judgments shape downstream choices. Whether those judgments carry a systematic directional tilt has been hard to detect: calibration metrics aggregate unsigned errors, and naturalistic uncertainty offers no ground-truth probability. When an LLM rates a startup's success at 70% but its failure at 15%, the missing 15 points expose a distortion no aggregate score flags. We introduce OptimismBench, which detects directional bias with inverted pairs: each scenario elicits both P(success) and P(failure), and asymmetry between the two framings yields a signed bias score without ground truth. Across 16 models from 8 providers, fourteen are optimistic; pessimism appears only in Anthropic's frontier tier. Eleven matched base-versus-chat pairs across four families show post-training sets the sign of the bias, with opposite shifts in different families. The pattern survives prompt, temperature, perspective, and self-debiasing ablations. A seventeen-model six-language comparison further shows model identity dominates language, with inter-model variance at 4.7x inter-language variance. We release 3,870 items across 10 languages for per-model directional-bias auditing. When alignment makes a model more helpful, it also tilts its probabilities; downstream pipelines inherit the tilt by default. 

---
# TREK: A Travel Reasoning and Evaluation Kit for LLM Agents in Complex Trip Planning 

**Authors**: Jinhu Qi, Wentao Zhang, Siu Man Ng, Feiyang Xu, Yanyu Chen, Yaoman Li, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2607.26977)  

**Abstract**: Travel planning is a demanding stress test for tool-using LLM agents: a usable itinerary is a single artifact that must be right along many axes at once - every flight, hotel, and attraction must exist and be bookable, the days must be physically traversable, the total must clear a budget, and the plan must serve a traveler whose needs are only partly stated. Existing agent benchmarks reward these properties one at a time and grade the final output with soft or LLM-judged rubrics, which cannot certify that a returned plan is executable and are neither reproducible nor auditable. We introduce TREK (Travel Reasoning and Evaluation Kit), a benchmark for feasible itinerary synthesis: producing a single plan that is jointly constraint-correct, hallucination-free, spatio-temporally executable, budget-valid, and responsive to the traveler's unstated persona needs. TREK comprises 800 multi-constraint tasks - 533 feasible and 267 provably infeasible with typed route/entity/budget causes - over a synthetic, internally consistent knowledge base of 212,530 records across 375 cities and 13 personas, served through a production-style tool sandbox of validated RESTful APIs. Every task is scored by a fully deterministic, rule-based evaluator with no LLM judge and ships a human-verified gold reference that scores a perfect 1.0 under that same evaluator, so the ceiling is demonstrably achievable and every remaining gap is an agent limitation rather than scorer strictness. Evaluating 15 LLM agents across nine constraint dimensions, we find that even the strongest (GPT-5.6) produces a fully-feasible plan on only 46.2% of solvable tasks, with a median of 6.6% and a floor of 0.0%; satisfying travelers' unstated needs emerges as the universal bottleneck, unsolved even at the frontier. We release the dataset, tool sandbox, deterministic evaluator, and agent code as a fully reproducible benchmark. 

---
# Generation or Judgement? A Paradigm Perspective on LLM-Based Emotion-Cause Pair Extraction in Conversation 

**Authors**: Weijie Feng, Hongchuang Wang, Binbin Liu, Zhiyong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.26967)  

**Abstract**: Emotion-cause pair extraction in conversation (ECPEC) identifies utterance pairs in which one utterance causes an emotion expressed in another. Recent LLM-based approaches formulate ECPEC at markedly different granularities, ranging from generating complete pair sets to judging individual candidate pairs. In this paper, we make the surprising observation that task formulation substantially affects performance, where pair-level judgement outperforms dialogue-level generation in all 18 controlled comparisons. We investigate the sources of this paradigm gap and find that many relations omitted by dialogue-level generation remain recognizable under explicit pair queries, under which the model recognizes 92.7%-98.1% of emotion-cause relations. This suggests that LLMs can recognize emotion-cause relations but struggle to discover and return complete pair sets. Pair-level judgement alleviates this burden, although its candidate rankings are more reliable than the binary decisions produced by a shared threshold. Based on this diagnosis, we introduce an auxiliary retriever that selectively re-examines ambiguous boundary cases, yielding consistent F1 improvements of 0.50-1.46 points across three datasets while maintaining an inference time of only 1.49x that of the baseline paradigm. These findings show that task decomposition and candidate scope are critical to effectively utilizing LLMs for ECPEC. 

---
# Credit Cards, Confusion, Computation, and Consequences: What Can We Uncover About Language Model Reasoning? 

**Authors**: Arnav Hiray, Agam Shah, Caleb Lu, Meghaj Tarte, Harsit Mittal, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2607.26952)  

**Abstract**: We introduce CreditCardQA, the first financial literacy benchmark for numerical reasoning derived from real credit card agreements. The dataset contains 1,800 questions, including first-person variants that reflect how consumers naturally ask about fees, interest, and payments. We evaluate a range of large language and reasoning models under Chain-of-Thought (CoT) and Program-of-Thought (PoT) prompting. Overall, PoT yields consistent performance gains, particularly for models with weaker baseline reasoning, and narrows gaps between open- and closed-source systems. Through error analysis, we show that failures arise less from arithmetic and more from misapplied financial rules, missed conditions, and misunderstandings of contractual terms. We further analyze question difficulty and find that comparisons, conditional logic, and monetary constraints are especially challenging. We also find that errors often arise in edge cases such as late-payment penalties or small-balance scenarios that are more likely to affect lower-income or financially vulnerable individuals. 

---
# Same Evidence, Different Target: Decoding How Diagnostic Evidence Bears on Causal Questions from Language-Model States 

**Authors**: Weiyi Kong, Zhuoran Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.26929)  

**Abstract**: The same diagnostic result can support or challenge one causal claim yet fail to address another when the claims concern different populations, outcomes, estimands, pathways, or identifying assumptions. When the evidence and target vary together, a correct answer may reflect favorable or adverse wording, lexical overlap, or a familiar diagnostic pattern rather than matching the evidence to the causal question. We introduce paired prompts that repeat the same diagnostic evidence verbatim while changing the causal target. Each prompt is labeled Favors, Challenges, Unresolved, or Wrong Target according to how the evidence bears on the causal question. A pair is recovered only when both prompts are classified correctly. Using linear readouts trained on a separate development set, we analyze the final-token hidden state from the penultimate transformer block of Qwen2.5-7B-Instruct, Qwen3-8B, and Llama-3.1-8B-Instruct. On the 49-pair primary benchmark spanning nine diagnostic families, balanced accuracy ranges from 0.654 to 0.659 and 18-21 pairs are recovered. Two independent human reviewers assigned the same label to 95 of the 98 prompts (96.9%). Across checkpoints, balanced accuracy and complete-pair recovery exceed permutation nulls that preserve development scenario groups. In Qwen2.5, full-prompt balanced accuracy exceeds both restricted inputs, with paired-bootstrap intervals for both differences above zero. Readouts trained without development examples from the evaluated diagnostic family recover 21 pairs, including at least one in each of the nine families. The hidden-state readout exceeds a linear classifier on answer-option logits and text baselines in balanced accuracy and recovered pairs. These results show that the hidden state contains linearly decodable information about whether diagnostic evidence favors, challenges, or fails to address the causal target. 

---
# Latent-IM: Latent Interaction Management for Speech LLMs 

**Authors**: Adar Avsian, Atahan Dokme, Tony Woo, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2607.26928)  

**Abstract**: Classical spoken dialogue systems often separated dialogue management from response realization: a policy selected the next dialogue action, and a generation component expressed that action. As dialogue systems shift toward LLMs, this decomposition has largely disappeared into the model's hidden representations. We ask whether an LLM-internal analogue of state estimation and action control can be recovered for conversational moves such as acknowledging, checking, querying, explaining, and replying. We formulate move control as two coupled problems: selection, predicting the appropriate next move from the dialogue context, and realization, causally producing a chosen move at generation time. We introduce Latent-IM, an internal dialogue-management framework that provides a general interface for choosing and deploying conversational moves under different objectives. Here, we use this control to reproduce human move choices, improving average end-to-end move accuracy by 12.5 points over the unsteered backbone while performing comparably to fine-tuning. 

---
# Dual-Path LLM Reasoning for Multimodal Few-Shot Knowledge Graph Completion 

**Authors**: Jinlan Liu, Zhiying Tu, Yongchao Xing, Yicheng Liu, Bolin Zhang, Dianbo Sui, Dianhui Chu, Hongliang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.26909)  

**Abstract**: Knowledge graph completion (KGC) aims to infer missing facts in knowledge graphs (KGs), thereby improving their completeness and supporting downstream intelligent applications. However, emerging entities and relations in real-world deployments make inductive KGC difficult, especially under few-shot and zero-shot settings. Multimodal information and Large Language Model (LLM)-derived priors can enrich sparse relational contexts, but they may also introduce noisy or hallucinated evidence. To address these issues, we propose DuPLeR, a \textbf{Du}al-\textbf{P}ath \textbf{L}LM \textbf{R}easoning framework for multimodal few-shot KGC. DuPLeR builds a calibrated relation graph by combining multimodal LLM-derived type priors with factual support structures, and performs dual-level structural reasoning over the refined relation topology. Moreover, a dual-pathway multimodal enhancement module regulates message passing with query-relevant multimodal signals and supplements entity representations after graph propagation. Experiments on eight inductive variants of two multimodal KG (MMKG) benchmarks show that DuPLeR achieves robust performance in data-scarce KGC scenarios. 

---
# DIRECT: Direct Decoding for Efficient and Aligned Sequence Labeling with Large Language Models 

**Authors**: Yilei Wang, Jiaxin Gan, Kexuan Zhang, Ling Li, Wentao Zhang, Peichao Lai  

**Link**: [PDF](https://arxiv.org/pdf/2607.26891)  

**Abstract**: Sequence labeling is a fine-grained information extraction task, yet existing large language model-based approaches suffer from insufficient domain alignment and low inference efficiency. To address these issues, we propose DIRECT, a framework that addresses these issues through training-time optimization and inference-time rectification. Specifically, DIRECT performs Direct Preference Optimization (DPO) after supervised fine-tuning to strengthen task alignment with human preferences, and introduces a controlled decoding process that enforces fixed output formats and restricts predictions to candidate sets. To further improve efficiency, a template-filling mechanism requires the model to generate only label tokens while reusing prefixed content through the KV Cache, thus reducing redundant computation. Experimental results on eight datasets demonstrate that DIRECT achieves significant improvements in both performance and efficiency compared to existing methods. 

---
# SERPO: Self-Evolving Rubric Policy Optimization for Open-Ended Test-Time Reinforcement Learning 

**Authors**: Jianze Wang, Kunwang Zheng, Ying Liu, Yu Cao, Qilong Zhang, Jinlong Chen, Hua Yang, Qianglong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.26873)  

**Abstract**: Test-time reinforcement learning (TTRL) enables language models to self-evolve at inference time without labeled feedback. Existing methods rely on answer voting and therefore do not extend naturally to open-ended generation, where valid responses cannot be mapped to a shared canonical answer. Without external reward models or stronger judges, adaptation must instead construct reliable rewards from the model's own outputs. We introduce SERPO (Self-Evolving Rubric Policy Optimization), which replaces answer voting with a closed loop that co-evolves response evidence, query-specific rubrics, and policy parameters. Good-Normal-Bad (G-N-B) response evolution organizes maximally separated rollouts into ordered archives; rubric evolution retains criteria that discriminate these archives; probabilistic criterion scoring converts verdict-token likelihoods into reward signals; and policy evolution optimizes the actor with the resulting signals. New actor rollouts then refresh both the archives and rubrics, closing the three-way evolution loop. Across two model configurations, two in-domain benchmarks, and four OOD benchmarks, SERPO improves HealthBench and ResearchQA by up to 20.63 and 20.31 points over the corresponding base models, raises the six-benchmark macro-average by up to 8.06 points, and supports OOD transfer and continued cross-benchmark evolution. 

---
# From Representations to Behaviors: Exploring the Person-Situation-Behavior Triad in LLMs 

**Authors**: Ruikang Zhang, Shuo Wang, Qi Su  

**Link**: [PDF](https://arxiv.org/pdf/2607.26853)  

**Abstract**: Human personality theories characterize traits not as isolated attributes captured by a single score, but as stable individual tendencies expressed through the interplay among persons, situations, and behaviors. Existing studies of personality-related behavior in LLMs have primarily focused on outputs elicited under personality conditioning, characterizing observable trait-related expressions while lacking mechanistic evidence for the existence of internal personality-related representations, their cross-situational expression, and how these representations shape specific behaviors. Building on Funder's personality triad framework, we adapt its three components for LLM analysis: Person as personality-related internal representations, Situation as contexts that afford trait-relevant responses, and Behavior as response patterns on broader social tasks. We introduce a framework for discovering, controlling, and validating trait-like representations in LLMs. First, using contrastive behavior pairs grounded in shared situations, we identify sparse internal features associated with opposing poles of personality traits through SAE decomposition. We validate their trait relevance through effects on behavior to situation, token-level activation patterns, and robustness to paraphrasing. Second, feature-level interventions induce bidirectional trait-related shifts across a separate, diverse set of situations while preserving response validity, demonstrating consistent expression across contexts. Third, applying the same interventions to social intelligence tasks reveals behavioral changes with benefit-tradeoff patterns consistent with findings from human personality research, providing behavioral-level validation beyond personality scores. Our findings provide evidence that LLMs contain controllable trait-like representations linking internal states, situational expression, and behavioral outcomes. 

---
# Language Models are not Equally Robust to Non-Canonical Tokenization across Languages 

**Authors**: Poulami Ghosh, Preethi Jyothi  

**Link**: [PDF](https://arxiv.org/pdf/2607.26831)  

**Abstract**: Despite the existence of exponentially many valid tokenizations for a given string, language models operate on a single canonical sequence deterministically produced by the tokenizer, leaving the broader tokenization space largely uncharacterized. In this paper, we investigate this overlooked space by studying the behavior of language models under non-canonical tokenizations across diverse languages. For English, prior work shows that models are largely invariant to alternative tokenizations that represent the same underlying string. We ask whether this invariance generalizes to other languages beyond English. We conduct a multilingual study across 27 languages spanning diverse scripts and evaluate LLM behavior under alternative tokenizations across six downstream tasks. We find that tokenization invariance does not generalize: model behavior varies substantially across languages with instruction-tuned models exhibiting an average relative performance drop of 23.7% for Llama-3.1-8B, 11.4% for Qwen3-8B, and 9.9% for Gemma-3-12B. The variation of tokenization invariance is systematic across languages. Languages that exhibit higher token fragmentation show significantly greater sensitivity to non-canonical tokenizations. Our study of tokenization robustness serves as a diagnostic of how tightly a model is coupled to its tokenizer. These results demonstrate that tokenization robustness is not a universal property of language models, but depends strongly on the language and its interaction with the tokenizer. We also show that LoRA fine-tuning with multi-tokenization training data provides an effective mitigation for tokenization sensitivity. Fine-tuning on English alone improves tokenization robustness across languages, while systematically sampling diverse non-canonical tokenizations achieves the strongest overall performance. 

---
# From Found to Designed: Concepts as a Design Axis for Large Language Models 

**Authors**: Chen Shani  

**Link**: [PDF](https://arxiv.org/pdf/2607.26825)  

**Abstract**: Large language models (LLMs) encode rich concept-like information, but represent it implicitly through distributed statistical associations rather than as explicit, structured, compositional concepts. Consequently, concept-level structure is typically \emph{found} rather than \emph{designed}: it is recovered after training through probing or dictionary learning, with no architectural guarantee of stability, compositionality, controllability, or alignment with human conceptual organization. We argue that concepts should instead be treated as a design axis for LLMs, and map the design space along two dimensions: the pipeline stage at which concept structure is introduced (training objective, core architecture, inference, or post-hoc interpretation), and whether that structure is internally derived from the model's own representations or grounded in external resources. This taxonomy reveals three broad patterns: inference-time approaches remain comparatively underexplored, related ideas have developed largely in isolation across pipeline stages, and externally grounded methods span the entire pipeline despite often being described under different terminology. Together, these observations motivate moving beyond recovering concept-like structure from trained models toward designing LLMs with explicit conceptual representations. 

---
# When Does Span-Guided Detoxification Help? Human Preferences and Evaluator Diagnostics in a Controlled Comparison 

**Authors**: Kyungwon Park  

**Link**: [PDF](https://arxiv.org/pdf/2607.26795)  

**Abstract**: Span-guided rewriting aims to preserve meaning by localizing edits to annotated harmful spans, but the same constraint can leave harmful intent insufficiently mitigated. We present a controlled exploratory comparison of span-guided and unguided detoxification on a mixed-source English evaluation set comprising manually curated inputs and HateXplain test items. We conduct a dense blinded human evaluation under a fixed single-generator setting.
Human preferences reveal a trade-off rather than a uniformly superior rewriting strategy. Span-guided outputs are favored when localized editing preserves the original stance and avoids unnecessary modification, whereas unguided outputs are favored when broader rewriting achieves more complete mitigation. This contrast varies substantially across the study-defined strata: the two strategies are competitive in the strong stratum, while unguided rewriting is clearly preferred in the mild stratum. Rationale annotations trace this difference to complementary failure risks: residual harm after localized editing and over-modification after broader rewriting.
We treat automatic evaluation as a diagnostic rather than a substitute for human judgment. Toxicity-similarity scalarizations, a multi-generator analysis, and two general-purpose LLM judges reproduce parts of the aggregate tendency but do not yield an analogous stratified contrast. These setting-specific findings do not establish a severity-based routing rule. Instead, they motivate evaluation protocols that assess mitigation sufficiency and meaning preservation separately and report both residual harm and over-modification alongside aggregate scores. 

---
# Enhancing Generative Information Extraction with Two-step Validation: A Product Attribute Use Case 

**Authors**: Yi-Sheng Hsu, Nermeen Abou Baker, Uwe Handmann  

**Link**: [PDF](https://arxiv.org/pdf/2607.26780)  

**Abstract**: The ability of large language models (LLMs) to process and generate text has introduced potential for applications in information extraction (IE). While it's debated whether LLMs outperform smaller fine-tuned models for classification tasks, their strong generalization capability makes them promising for domains with limited labeled data available for fine-tuning. This advantage is particularly relevant for the emerging application of the digital product passport (DPP), where the problem space is broad but domain-specific data remains scarce. Motivated by this use case, we apply generative IE to the product domain, explicitly addressing efficiency, generalizability, and data privacy constraints. We propose a two-step validation method that integrates a PLM block into the generative IE pipeline and thereby leverages LLMs' correction capability. We discover that such a validation task enhances LLM performance, particularly on the extraction of weakly expressed, low-salience entities that appear sparsely throughout the text. For certain entities, the performance of mid-size models can even reach levels comparable to larger models, and the improvement of first-step PLM predictions also enhance the final LLM output. Nevertheless, the effects on the smallest open-source LLMs (e.g., Llama-3.2 3B) is limited. Based on the findings, we develop a demo application for product information extraction that utilizes locally deployed LLMs, targeting further adaptations to real-world DPP use cases. 

---
# Relation Geometry in Semantic Space of Language Models 

**Authors**: Zhihan Cao, Hiroaki Yamada, Simone Teufel, Tatsuya Hiraoka, Kentaro Inui, Hitomi Yanaka, Takenobu Tokunaga  

**Link**: [PDF](https://arxiv.org/pdf/2607.26762)  

**Abstract**: When it comes to generating vector representations of words, current language models are achieving high-quality results. However, what is not known is the extent to which knowledge about semantic relations is represented in the geometry of the semantic spaces created in this way. In order to answer this question, we study the relation geometry of such semantic spaces from three perspectives. We first examine whether words standing in a particular relation to a target word~(called relata) occupy the same region in semantic space, and whether the regions corresponding to different relations are distinct from each other. We then verify to what extent semantic spaces reflect certain well-known properties of relations, such as symmetry, asymmetry, and transitivity. Finally, we consider which information about the target words and relata is more important for relation geometry: their surface forms, or their contexts. We conduct experiments on six semantic relations using causal, masked, and diffusion language models. The results show that relata in asymmetric relations relatively clearly occupy a distinct region in semantic space. Asymmetric relations' properties are only moderately well encoded in the semantic space, yet better than those of symmetric ones. Furthermore, when considering the question which information source has the strongest impact on results amongst the models we evaluated, we find that lexical information tends to be more important for the causal language model, whereas contextual information is more important for the masked and diffusion language models. Our results empirically show that relation geometry is not equally well-represented for all relations in semantic space, suggesting that there is a difference in how well semantic relations might be learned from distributional information alone. 

---
# Metis: Memory Foundation Model 

**Authors**: Zeyu Zhang, Ziliang Guo, Yihang Sun, Xichong Zhang, Xixuan Hao, Zehao Lin, Yang Zhang, Xiaoyan Zhao, Tong Shen, Bo Tang, Zhi-Qin John Xu, Junchi Yan, Haofen Wang, Xu Chen, Feiyu Xiong, Zhiyu Li, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2607.26760)  

**Abstract**: Recent advances in AI agents have increasingly internalized native capabilities into their underlying foundation models, giving rise to multimodal foundation models and large reasoning models. However, agent memory is still primarily implemented through external modules, leaving the native memory capability largely unexplored. In this paper, we take a first step toward this direction by introducing memory foundation models, which empower foundation models with native memory capabilities. We formalize native memory from two perspectives: a persistent and dynamically evolving memory state within the backbone, and native memory procedures that autonomously store and utilize information through model computation. We show that native memory offers advantages in architecture, end-to-end optimization, and efficiency. Based on this formulation, we propose Metis, the first prototype of memory foundation models. Metis introduces a new architecture that equips a foundation model with a native memory state, allowing historical information to be compressed into the model and accessed through memory attention. We construct large-scale memory-specific training data and introduce multiple optimization objectives to acquire these native memory procedures through mid-training. The online memory maintenance of Metis is gradient-free, and the memory update requires only a forward pass. At inference time, all learned model weights remain frozen, while the native memory states are autonomously transformed through standard forward computation. Through extensive experiments, we show that Metis exhibits native memory capabilities and further provide a detailed analysis of its strengths, limitations, and behaviors. To facilitate future research on memory foundation models, we release our project and model checkpoints. 

---
# Phoneme- vs. Character-Level Targets and Selective State-Space Models for Intracortical Brain-to-Text 

**Authors**: Lucas Zamora Vera, Jose A. Gonzalez-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2607.26751)  

**Abstract**: State-of-the-art intracortical brain-to-text systems pair a neural-sequence phone decoder with an external language model. Two design axes remain underexplored: whether selective state-space models (Mamba) improve on recurrent decoders, and how the output target (phonetic vs.\ character) interacts with that choice. On the public Brain-to-Text '25 benchmark, we study a controlled 2x2 grid (GRU vs.\ hybrid Mamba decoder; phonetic vs.\ character targets) trained with a CTC objective under one reproducible protocol. The recurrent baseline remains strongest: the best phonetic GRU reaches 12.62\% PER and 21.19\% WER, while the best textual GRU after LM rescoring reaches 13.39\% CER and 26.28\% WER. The Mamba hybrid is competitive but does not surpass it. Ablations isolate architectural contributions, and error analysis shows representation-dependent failures: articulatory-like phoneme confusions vs.\ lexical and word-boundary errors. 

---
# AtmosERC: Modeling Dialogue-Level Affective Atmosphere for Emotion Recognition in Conversation 

**Authors**: Weijie Feng, Tongwei Zhang, Binbin Liu, Zhiyong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.26726)  

**Abstract**: Emotion Recognition in Conversation (ERC) aims to predict utterance-level emotions in dialogues and has largely advanced through context-centric modeling. However, global context is a heterogeneous signal, and not all contextual information is equally relevant to emotion prediction. This paper focuses on the affect-oriented component of this signal, termed dialogue-level affective atmosphere, which captures a latent tendency commonly reflected in conversational emotion patterns. To estimate and exploit this tendency, we propose AtmosERC, a graph-based ERC framework that models each dialogue as a conversational graph over utterances and speakers. A relation-aware graph extractor filters and fuses heterogeneous graph signals to produce dialogue-level and speaker-conditioned affective priors. The resulting compact prior guides lightweight sequential emotion prediction and can also be verbalized into prompt-level cues for LLM-based ERC without modifying backbone models. Experiments on four ERC benchmarks show that AtmosERC improves lightweight ERC, enhances LLM-based ERC as a plug-in cue, and yields more stable predictions under local emotional deviations. 

---
# Automated Multilabel Mpox Research Classification with Explainable Transformer Models 

**Authors**: Tanjim Taharat Aurpa  

**Link**: [PDF](https://arxiv.org/pdf/2607.26700)  

**Abstract**: The Mpox outbreak remains a serious public health issue, with the WHO (World Health Organization) reporting increasing cases in some regions. Research on Mpox is vital for several reasons, including vaccine development, diagnostic improvement, viral evolution studies, and preventing future outbreaks. However, the large amount of research being published makes it difficult to organize and analyze information efficiently. This study focuses on using multilabel classification to categorize 14590 Mpox research articles into key topics such as outbreaks, vaccination, and epidemiology. Among the different AI models tested, BERT performed the best, achieving 97.05% accuracy, 97.67% micro F1 score, and 96.46% macro F1 score. To better understand how the model makes decisions, SHAP was used to analyze significant word features and patterns. The results show that BERT can help automate the classification of Mpox research, making it easier for researchers, policymakers, and healthcare workers to quickly find relevant information, saving time and improving public health efforts. 

---
# Constitutional Midtraining: Content Presence Drives Alignment Gains 

**Authors**: Desiree Cho, Cameron Tice, Bernie Hogan, Hunar Batra, Puria Radmard, Jun Zhao, Nigel Shadbolt  

**Link**: [PDF](https://arxiv.org/pdf/2607.26654)  

**Abstract**: Post-training alignment is often shallow, eroding under fine-tuning. Whether midtraining interventions, cleanly isolated from post-training, can produce durable alignment remains untested. We test this via constitutional midtraining: inserting principled, values-based content into midtraining against a replay-only control at 120B scale. Our 394M-token constitutional corpus, built from Anthropic's Constitution, uses a 2x2 factorial design (curriculum ordering x deliberative reasoning) to produce four constitutionally midtrained conditions plus a control, evaluated on self-generated and established benchmarks including alignment under pressure, value conflict resolution, blackmail, and emergent misalignment across three stages: post-midtraining, post-SFT, and post-benign fine-tuning. Constitutionally midtrained models outperform the control on alignment generalization and durability, notably on blackmail: SFT instills a blackmail propensity in all models, but constitutional midtraining blunts it, with the advantage surviving benign fine-tuning (-17.5pp). This durability does not extend to settings requiring active resistance to in-context pressure or conflict, where the advantage attenuates after SFT. The presence of constitutional content at midtraining also matters more than its structure, and constitutional midtraining incurs no cost, on average, on the capabilities we test (MMLU, ARC-Easy, piqa, GSM8K) at any stage. A modest amount of constitutional content at midtraining could therefore yield broad, persistent alignment gains, offering a cheap, complementary addition to SFT-centered pipelines. Code, data, and models are available. 

---
# Contrastive ESA: Human Evaluation of Multiple Translations at Once 

**Authors**: Vilém Zouhar, Roman Grundkiewicz, Sara Rajaee, Parker Riley, Martin Popel, Rachel Bawden, Philipp Koehn, Marine Carpuat, Tom Kocmi  

**Link**: [PDF](https://arxiv.org/pdf/2607.26640)  

**Abstract**: Current human evaluation of machine translation typically assesses single outputs in isolation, a paradigm that suffers from high annotator noise and cost. We introduce Contrastive Error Span Annotation (cESA), a protocol that presents multiple translations of the source input (text, video, audio, image). In cESA, the annotator sees multiple translations of the same document, marks major and minor error spans, and then assigns a score from 0% to 100% on absolute scale. By allowing annotators to access the shared context across multiple outputs, cESA facilitates more consistent and efficient judgments. We validate cESA using a large-scale human evaluation of English->Japanese translations of 12 models, demonstrating reductions in annotation time and noise compared to standard pointwise evaluation. Unlike existing contrastive ranking methods, cESA yields absolute quality judgments that enable simple, interpretable non-parametric model rankings without the need for post-hoc corrections. 

---
# Filesystem-Based Memory for LLM Agents: Organization, Evolution, and Sustainability 

**Authors**: Sizhe Zhou, Sheldon Yu, Hui Wei, Junda Wu, Siru Ouyang, Yizhu Jiao, Shijia Pan, Julian McAuley, Yu Zhang, Tong Yu, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2607.26637)  

**Abstract**: Deployed LLM agents increasingly keep their long-term memory as a filesystem: a directory tree of markdown files that the agent itself reads, writes, and reorganizes through generic file tools. Yet research has largely passed over this medium: prior systems design bespoke memory representations and study retrieval over them, leaving the default's two working assumptions untested: that an agent can keep a growing store organized as memories accumulate, conflict, and go stale, and that this organization pays. We present the first systematic exploration of filesystem-based memory for LLM agents. We formalize the setting as three roles around one memory filesystem: a management agent integrates and organizes incoming content, a search agent answers queries with cited sources, and an execution agent supplies task trajectories that are distilled into skills, unifying declarative memory and skills in a single store. Across long-conversation benchmarks and embodied tasks, we vary memory shape (agent-organized hierarchy, verbatim dump, chunk retrieval), stream scale, tool harness (sandboxed shell, memory-tool-style functions, varied search tooling), and the strengths of the management and search agents, tracking answer quality, cost, and store health as memory grows. What organization reliably buys is search economy: organized stores roughly halve retrieval cost where material is large. Today's agents, however, fall short of the default's promise: in our growth study, organization erodes for all but the strongest management agent, and no agent we measure converts organization itself into better answers. And the model is not the only lever over a store's shape: changing the tool set alone reshapes the store as strongly as swapping the model. The study turns the filesystem default from an assumption into a design space for agent memory. 

---
# Revisiting Lossy Verification in Speculative Decoding: Mechanisms, Trade-offs, and Failure Modes 

**Authors**: Tianyu Wang, Yuxuan Zhou, Wenbin Wang, Heng Li, Zikai Xiao, Junyuan Shang  

**Link**: [PDF](https://arxiv.org/pdf/2607.26627)  

**Abstract**: Speculative Decoding (SD) accelerates large language model inference by allowing a lightweight draft model to propose tokens that are subsequently verified in parallel by a larger target model. Recent approaches introduce lossy verification schemes to further improve efficiency by relaxing strict distributional matching. Yet such relaxation silently rewrites the decoding distribution, and the resulting acceleration can come at the cost of unstable, sometimes severely degraded generation quality. In this work, we present a principled analysis of the distributions induced by lossy verification methods. We show that many seemingly distinct approaches differ only superficially and can be classified into two categories: truncation-based verification and collaborative verification. We further construct a diagnostic evaluation framework across curated benchmarks. For truncation-based methods, we identify a fundamental pitfall: performance can degrade significantly compared to the true truncation sampling baseline due to distributional distortion. For collaborative verification, we uncover a key principles: controlling the overshoot of draft probabilities relative to target probabilities is essential to prevent low-quality outputs. Our code is available at this https URL. 

---
# WikiLoop: Jointly Learning to Build and Navigate Agent-Native Wikis with Downstream Feedback 

**Authors**: Haoliang Ming, Feifei Li, Wenhui Que  

**Link**: [PDF](https://arxiv.org/pdf/2607.26604)  

**Abstract**: Knowledge-base construction and querying are typically optimized in isolation: retrieval-augmented agents operate over a fixed, externally maintained index, whereas construction receives no signal from downstream use. We present WikiLoop, a feedback-coupled framework that jointly learns to build and navigate an agent-native Wiki, a persistent linked-page knowledge base designed for machine navigation. A role-conditioned shared policy supports two interfaces: a Navigator retrieves evidence from the Wiki to answer queries, and a Builder proposes structured edits evaluated through downstream navigation. The Navigator follows a sufficiency-before-efficiency objective that applies retrieval-cost penalties only after full evidence has been collected. The Builder learns from utility differences: a frozen Navigator scores each candidate edit by its change in downstream performance, while a guard penalty discourages regressions on unrelated queries. Training combines sequential role-specific optimization with a final joint stage over role-homogeneous batches. With Qwen3.5-9B as the common backbone, WikiLoop reaches 62.6 aggregate Answer Correctness on AuthTrace, 6.3 points above LLM-Wiki, base, with the largest gains on multi-document queries. Controlled comparisons support the intended effects of both objectives, and the learned edits remain useful to a held-out Navigator. Paired comparisons indicate that the final shared policy largely retains both role-specific capabilities, improves Navigator and end-to-end Answer Correctness by 0.4 points relative to the corresponding specialist references, and consolidates both interfaces into one model. Without dataset-specific training, WikiLoop also improves over the same-backbone LLM-Wiki, base on HotpotQA and MuSiQue. 

---
# Where Detectors Fail: Closing the Tail-Domain Gap with Expert-Guided Mutual Distillation 

**Authors**: Xuan Feng, Guihong Liu, Tianlong Gu, Shuai Zhao, Xuemin Wang, Chenzhong Bin, Yang Liu, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2607.26555)  

**Abstract**: Multimodal fake news detectors often generalize poorly across domains because they learn to trust unreliable evidence: domain-specific shortcuts amplified by imbalanced data and semantically inconsistent text-image pairs that make cross-modal evidence unreliable. We propose Expert-Guided Mutual Distillation (EGMD), which learns what evidence to trust across the prediction pipeline. At the input level, input-level calibration encodes pair-level coherence as a shared gain before fusion. At the representation level, an expert-guided teacher aligns domain statistics and encourages domain-specific patterns to concentrate in specialized experts. At the decision level, prototype-anchored domain-specific students use mutual learning and dual-channel distillation to inherit the teacher's feature geometry and calibrated predictions while discouraging local domain priors. We further construct Weibo_Balanced, a domain-balanced benchmark that isolates the effect of imbalance on generalization. Across four datasets in two languages, EGMD achieves state-of-the-art accuracy while reducing domain bias by up to 57.3%. 

---
# Which RAG Paradigm Wins at Scale? A Scaling Study of Retrieval-Augmented Generation Paradigms 

**Authors**: Pengyu Wang, Benfeng Xu, Shaohan Wang, Xin Zeng, Huarui Wu, Lei Zhang, Licheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.26497)  

**Abstract**: Retrieval-augmented generation (RAG) methods range from lexical and dense retrieval to graph-based indexing and agentic search. They are usually evaluated on different benchmarks at one corpus size, leaving their accuracy-cost scaling unclear. To bridge this gap, we present a controlled corpus-scaling study of these four paradigms. A ladder of 28 strictly nested tiers grows from roughly 1,000 to 512,000 documents while questions and a fixed bedrock of relevant and adversarial documents remain unchanged. Under one reader and judging protocol, we measure official accuracy, construction and query tokens, and latency. Our experimental results show that BM25 scales best in this controlled setting: it defines the low-cost end of the Pareto frontier at every measured tier and leads accuracy from mid-scale onward, without LLM-based construction. The File-System Agent matches or slightly exceeds BM25 at the smallest tiers but uses 39 times more query tokens per answer at the bedrock and falls nearly 20 points behind at full scale. A matched retrieval swap reverses this failure: Agent+BM25 scores 69.4 at full scale, versus 36.9 for raw-file agency and 54.8 for native BM25 on the same 150 questions. Graph-based RAG hits a construction wall: its heaviest builders use up to 24.6 generative LLM tokens per indexed corpus token yet stop within the first 2% of the full corpus, while scalable variants remain less accurate than BM25 at shared tiers. 

---
# CMT-RAG: Complementary Memory Traces for Multi-turn Multi-hop RAG 

**Authors**: Lang Zhou, Yingjian Chen, Shuxuan Li, Kun-Yu Lin, Zhilin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.26470)  

**Abstract**: Multi-turn information-seeking conversations require both multi-hop reasoning and long-range dependency tracking across turns. However, existing RAG systems typically represent conversational memory as raw dialogue history, rewritten queries, or unstructured summaries, making it difficult to recover the specific prior reasoning steps and evidence required for follow-up queries. Our key insight is to align conversational memory with retrieval by representing dialogue context as sub-question-level reasoning traces. Building on this insight, we introduce MuMu-QA, a benchmark for multi-turn multi-hop RAG with explicit cross-turn sub-question dependency annotations, and CMT-RAG, a complementary memory framework for this setting. At each turn, CMT-RAG employs a state-space trace generator, whose recurrent state serves as runtime memory, to incorporate recent conversational context and decompose the current query into structured trace drafts containing retrieval-oriented sub-questions and dependencies on earlier traces. It then grounds these drafts with retrieved evidence and stores them as persistent memory traces in a session-level DAG, enabling future turns to efficiently recover relevant prior reasoning and evidence. Experiments on MuMu-QA and corpus-level RAG benchmarks show that CMT-RAG consistently outperforms five categories of RAG baselines in answer accuracy. 

---
# ForgetBench: Benchmarking Forgetting Dynamics of Long-Term Parametric Memory in Language Models 

**Authors**: Ruxi Gu, Zhenliang Zhang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.26455)  

**Abstract**: Large language models (LLMs) have demonstrated strong capabilities in knowledge acquisition and reasoning, yet their ability to retain previously acquired knowledge under repeated updates remains insufficiently understood. Existing evaluation paradigms primarily focus on single-step reasoning or static knowledge editing, which fail to capture the temporal dynamics of knowledge retention and degradation during continual model modification. In this work, we propose ForgetBench, a benchmark designed to systematically characterize forgetting behavior in LLMs under continual knowledge editing. ForgetBench introduces two complementary evaluation paradigms, namely concept-based QA and scenario-based QA, to disentangle isolated factual retention from structured relational knowledge preservation. Building upon a sequential editing framework, we construct temporally ordered knowledge streams and evaluate model behavior across multiple editing stages. To quantitatively analyze long-term retention dynamics, we further introduce a unified evaluation framework that models knowledge evolution over time, enabling the measurement of temporal decay, retention strength, and cross-instance stability. Extensive experiments across diverse models and editing methods demonstrate that existing approaches fail to strike a balance between long-term retention and generalization quality. Our findings highlight the need for more robust memory mechanisms that can effectively acquire, update, and preserve knowledge over time in future LLMs. Code will be released upon acceptance. 

---
# Mergeable Model-Side Aggregation States for Long-Context Language Models 

**Authors**: Dachuan Song, Junyu Yin, Zechen Hu, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.26448)  

**Abstract**: A known limitation of long-context language models is their increasingly unreliable performance in non-additive, set-based aggregation as context length grows. Examples include cardinality estimation, set relationships, and grouped statistics, which widely exist in logs, program outputs, tables, and multi-turn conversations. To provide the aggregation state required by these tasks, we introduce a model-side aggregation interface that maintains compact Hash-based HyperLogLog (HLL) sketch states alongside a frozen language model. While the model processes the context, an extractor maps each relevant record to a canonical identity. The identity is then hashed and updates the HLL state. These states can be merged across context segments and/or read out directly for downstream reasoning, avoiding an additional generate-execute-return cycle. We validate the proposed approach by setting the HLL state size as 2 KiB (2,048 registers), which does not increase with context length or set cardinality. In a distinct-count experiment involving one million records, the mean relative error was 1.6%. In a separate merge test, states built from as many as 256 segments produced exactly the same readout as a single pass over the same stream. On 3,969 aggregate-then-reason tasks from 174 source windows, the fixed-budget interface reached 99.2% accuracy on Gemma 4 (31B, BF16), compared with 100.0% under exact aggregation; the paired gap was 0.8 percentage points (95% window-cluster CI: 0.5-1.3 points). On a matched set of 174 items, our method improved over direct full-context reasoning by 63.2 points on Qwen and 56.3 points on Gemma. The corresponding gains over chain-of-thought (CoT) reasoning were 60.9 and 63.2 points, respectively. On a fixed 1,200-task Oolong-Synth subset, our method reached 91.1% on Qwen and 99.3% on Gemma. Code is available at this https URL. 

---
# Voice Memory for Agentic Speech Recognition 

**Authors**: Chao-Han Huck Yang, Zih-Ching Chen, Piotr Zelasko, Zhehuai Chen, Jagadeesh Balam, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2607.26410)  

**Abstract**: We present Voice Memory, a inference-only scheme for agentic speech recognition: at stream time, a frozen corrector reads a single per-domain this http URL and decides per utterance whether to act on the hypothesis or abstain and keep the 1-best. Asynchronously, a score-gated optimizer revises that file through bounded edits, accepting an edit only when it strictly improves a held-out score. Extended from classical ASR-LM framework, we refer this split the listener-thinker architecture; the two roles are coupled only through the memory, so no weights change and the learned skill stays auditable and portable. Restraint turns out to be the operative skill this loop discovers: unconstrained generative error correction (GER) over-corrects, breaking correct tokens on up to 64% of its edits on financial news, and Voice Memory, reduces this rate to 35%. Across ten HyPoradise domains with an open corrector, Voice Memory, lowers weighted word error rate from 8.36% to 7.52% (7.47% with three added in-context examples) without regressing any dataset below its 1-best baseline; gains concentrate where recoverable headroom is largest, including air-travel commands (8.40% to 3.40%) and noisy far-field speech (CHiME-4, 12.69% to 10.46%). The memory transfers across corrector families and adds zero parameters to the inference path. A demo and example code are provided for future studies. 

---
# Knowledge before Reasoning: EC-Reason-Bench, a Training-Free Diagnostic Benchmark for LLM Enzyme Classification 

**Authors**: Linyu Li, Zhi Jin, Yichi Zhang, Dongming Jin, Yuanpeng He, Huanyao Zhang, Xuan Zhang, Gadeng Luosang, Nyima Tashi  

**Link**: [PDF](https://arxiv.org/pdf/2607.26397)  

**Abstract**: Enzyme function prediction is a hierarchical, knowledge-intensive form of protein function classification. Existing benchmarks expose an anomaly: general LLMs often get the coarse first level right, yet once asked for a complete EC number their accuracy at levels two through four drops to almost zero, while specialized models and tools stay usable. We propose EC-Reason-Bench, a training-free, diagnostic evaluation protocol built to answer two questions: why general LLMs score close to nothing on EC number prediction, and how much of that loss can be recovered without updating a single weight. We break enzyme classification ability into four orthogonal levers that can each be measured on their own: output structure, external knowledge, reasoning structure, and reasoning robustness. We test each lever with an inference-time method against a shared zero-shot baseline reproducing previously reported near-zero performance. Experiments with several strong reasoning LLMs yield four main findings. First, external knowledge is decisive and must precede reasoning: uniformly low closed-book performance rises sharply with open-book access, narrowing model gaps. Second, in closed-book settings, whether cascading and chain-of-thought help or hurt depends on a model's tendency to abstain. Third, once evidence is available the aggregate score of the best LLM setting is indistinguishable from simply voting the EC numbers of the nearest retrieved neighbors; that tie is an artifact of averaging, and it hides a large gain on adversarial evidence set against an equally large loss on multi-functional enzymes. Reasoning over evidence therefore acts as an arbiter of conflicting neighbors rather than as a source of knowledge, and no single-number leaderboard can see it. Fourth, accuracy obeys a law of homology availability. 

---
# Misalignment Has a Personality: A Big Five Account of Emergent Misalignment 

**Authors**: Hasibur Rahman, Smit Desai  

**Link**: [PDF](https://arxiv.org/pdf/2607.26389)  

**Abstract**: Fine-tuning a language model on data containing a narrow flaw, such as insecure code or incorrect mathematical answers, can cause broad misalignment through a mechanism that remains debated. We provide an interpretable account: in the models and corpora we study, misalignment behaves like a shift in personality. Prior work extracts activation directions for character traits from a single binary contrast, which can separate or steer behavior without establishing a calibrated scale. We instead extract personality vectors for the Big Five using a graded, three-level intervention and validate them on two open-weight models. The three levels are linearly ordered, with Cohen's d values of up to 6.2; the vectors transfer zero-shot and trait-specifically to an independent corpus; and their effects are strongest within a middle-layer band. Applied to training data, the vectors reveal that misaligned corpora across eight domains share a common Big Five signature: lower agreeableness and conscientiousness, together with higher extraversion and neuroticism. This signature is recovered by both models with a correlation of r = 0.94. Fine-tuning imprints the same profile, shifting the model's generations along the corresponding signature, with r = 0.83 using activation-based measurements and r = 0.90 using a text-based judge, while also shifting internal activations with r = 0.69. The same vectors characterize sycophancy as high extraversion and low conscientiousness rather than excess agreeableness, a distinction that a single direction cannot capture. Calibrated personality vectors transform an opaque safety phenomenon into a human-legible diagnostic profile. 

---
# (Im)Paired Programming: Coding Agents Improve Productivity but Harm Understanding 

**Authors**: Nishant Balepur, Connor Baumler, Valerie Chen, Eunsol Choi, Rachel Rudinger, Jordan Lee Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2607.26375)  

**Abstract**: Coding agents (e.g., Cursor) improve developer productivity by optimizing task completion, but shifting users from writing code to prompting and reviewing may harm their understanding, impeding oversight, learning, and communication. To probe this, we have 54 students create a website with one of two AI systems: an agent that edits user code; or a chatbot where users write code alone or adapt generic code snippets. We test understanding via comprehension questions and a task where users extend their code without agents, showing: (1) While agents aid initial task completion, they harm users' code comprehension and thus do not prepare users to extend their code; (2) Low-effort agent interaction types, like copy+paste prompts and auto-accepted edits, are linked with lower comprehension; and (3) Despite self-reported weaker understanding, users still prefer coding agents because they are quick and easy to use. While users stay in the loop for coding workflows, understanding should not be forgotten. Towards this goal, we distill our analyses into future research directions for coding agent developers: dissuading low-effort prompting, creating readable code, and promoting active engagement. 

---
# Diagnosing Fine-Grained Inconsistency Classification in Financial Disclosure Text 

**Authors**: Aman Kumar, Lasitha Vidyaratne, Dipanjan D Ghosh, Arnab Chakrabarti, Ahmed K Farahat  

**Link**: [PDF](https://arxiv.org/pdf/2607.26368)  

**Abstract**: Financial disclosures contain numerical claims, temporal statements, entity references, policy commitments, and risk descriptions that may conflict in qualitatively different ways. Detecting a conflict is only the first step: review workflows may also need to determine its type, since numerical, temporal, referential, factual, and normative inconsistencies require different evidence and downstream checks. We study this problem as fine-grained inconsistency classification. Using a fixed 5,940-instance snapshot of SBID-FD, a synthetic financial-disclosure benchmark with 11 inconsistency labels and paired reference evidence spans, we compare frozen embedding classifiers, fine-tuned encoders, evidence-augmented classifiers, prompted large language models, and LoRA-adapted generative models under a shared evaluation protocol. A fine-tuned 300M encoder reaches 61.9% accuracy, compared with 61.5% for a LoRA-adapted Qwen3.5-9B model and 61.3% for GPT-5.4. Because these systems differ in architecture, supervision, training objective, and input format, we interpret this as a practical efficiency result for compact supervised encoders rather than a controlled conclusion about model scale. Supplying gold evidence spans improves the fine-tuned encoder to 65.3%, whereas automatically predicted spans recover a meaningful but incomplete share of that gain, indicating that localization quality remains a bottleneck. Class-level analyses show that Referential inconsistencies are especially sensitive to localization quality, while Factual and Logical inconsistencies remain difficult even when the relevant evidence is provided. Together, the oracle, distractor, and per-class analyses separate localization errors from residual type-discrimination errors, indicating that progress requires both stronger evidence extraction and better reasoning over closely related inconsistency categories. 

---
# Symphony of Bias: Exploring Gender Associations with Musical Instruments in Multimodal LLMs 

**Authors**: Farhan Farsi, Shayan Bali, Mohammad Heydari Rad, Negar Heidary, Donya Rooein  

**Link**: [PDF](https://arxiv.org/pdf/2607.26355)  

**Abstract**: Large language models (LLMs) are increasingly embedded in everyday life and widely used for information seeking, raising concerns about their potential to perpetuate social biases and reinforce stereotypes. In this study, we investigate gender bias in LLMs through the lens of their associations with musical instruments. Building on social-science research on the cultural gender-typing of instruments, we introduce Symphony-Bias, a parallel multimodal dataset spanning text, vision, and audio. We evaluate ten multimodal models with diverse architectures and scales across 22 musical instruments, analyzing how they associate each instrument with three gender categories: {male, female, non-binary}, across three modalities: {text, vision, audio}. Our results show that 92\% of instrument-level outcomes align with prior social-science findings, with the harp and drums showing particularly consistent gendered associations across all evaluated models and modalities. We further find that alignment with social stereotypes is weakest in audio, stronger in vision, and strongest in text, suggesting that modality-specific representations can differentially amplify gendered associations with musical instruments.\footnote{The Symphony-Bias dataset will be publicly released upon acceptance of the paper.} 

---
# When Synthetic Users Fail: A Cross-Domain Benchmark of LLM-Simulated Human Survey Responses 

**Authors**: Zihan Chen, Di Zhu, Lei Nico Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.26348)  

**Abstract**: Large language models (LLMs) are increasingly used as synthetic users, stand-ins for human respondents whose simulated answers feed product, policy, and market decisions. We ask when this substitution is valid and when it fails, and package the answer as an evaluation framework for intelligent synthetic-user systems. A single protocol, run across four models spanning two families and an 8B-to-frontier capability range, is applied to two independent domains of real human-response data: U.S. general social attitudes (General Social Survey) and cross-cultural values (World Values Survey). Every model is benchmarked against a suite of non-LLM baselines fit on held-out human data. Under demographic prompting and the survey-simulation protocols we test, two failures replicate across both domains, all four models, and both families. First, at the individual level no LLM beats even the strongest baseline; on cross-cultural values every model falls well below it, and the gap survives distance-aware and proper scoring. Second, models systematically over-determine demographics, treating identity as far more predictive of attitudes than it is among real people, a distortion present for nearly every question-group combination and robust to a coding-invariant measure. Neither failure is remedied by a larger, more capable model. A decision-impact analysis shows why this matters in practice: on a segment-targeting task the models inflate between-segment gaps two to fourfold, would direct a team to the wrong segment in half of U.S. and most cross-cultural cases, and manufacture segment splits that do not exist in real people. We make the cross-domain benchmark and the evaluation framework available on request, so that teams can determine in advance when synthetic-user evidence is safe for decision support and when it is not. 

---
# AgentGUI: An Interface for Observing and Steering Long-Running AI Agents 

**Authors**: Xuan Zhao, Jiwoong Sohn, Qinyue Zheng, Michael Moor  

**Link**: [PDF](https://arxiv.org/pdf/2607.26300)  

**Abstract**: AI agents are increasingly adept at tackling complex, long-running tasks. With the rapid surge of autonomous capabilities, human oversight is systematically lagging behind due to limited human-centered interfacing. Aiming to address this, we introduce AgentGUI, a user-friendly, locally hosted GUI for seamlessly observing and steering AI agents amid multiple concurrent, long-running sessions. AgentGUI features 1) rich agent trajectory visualizations, 2) effective manual and automated steering, and 3) integration with and coordination between open-source and frontier agent frameworks. A controlled user study demonstrates statistically significant reduction in the time it takes to identify key elements from agent traces (38% faster, p = 0.023). In a preliminary experiment, AgentGUI's automated drift prevention feature raises the task completion rate of small local agents by as high as 34pp across a 0.8B--9B model ladder (N=50 runs per model). AgentGUI is publicly available through its project website (this https URL) and open-source repository (this https URL), along with a demo video (this https URL). 

---
# Evaluating Prompt Scope and Demonstration Similarity in Local LLM Machine Translation 

**Authors**: Mihael Arcan  

**Link**: [PDF](https://arxiv.org/pdf/2607.26286)  

**Abstract**: Large language models (LLMs) are increasingly used as general-purpose translation systems, but their behavior is usually evaluated under a single prompt shape: translate one source sentence into one target language. In practice, users may ask for one target language, for several related languages at once, or for translations conditioned on examples. This paper studies prompt scope and demonstration selection as experimental variables for local LLM machine translation. We evaluate English-to-Romance and English-to-Germanic translation on the full FLORES devtest split for nine official European Union languages. We compare three local instruction-tuned LLMs, llama3.2:3b, mistral:latest, and qwen2.5:14b, against dedicated MT baselines from OPUS-MT and NLLB-200. We test zero-shot prompting and k=5 few-shot prompting with random, lexical-similarity, and embedding-similarity demonstration selection. We also compare single-target prompts with JSON-formatted family-scope prompts that request all languages in a family at once. Results show that dedicated MT systems remain strongest overall, especially for Germanic languages. Few-shot prompting helps mistral:latest and qwen2.5:14b, but hurts llama3.2:3b; embedding retrieval is best on average for the stronger LLMs, but its advantage over random and lexical examples is modest. Family-scope prompting is feasible for stronger local LLMs but exposes structured-output failures in smaller models. These findings motivate evaluating LLM translation not only by language pair and metric, but also by prompt scope, retrieval strategy, and multi-target compliance. 

---
# Robostreet Flow: A Lightweight, Ultra-Low-Drag Electric Tractor and Four-Truck Hybrid Convoy Architecture for Minimum-Cost Point-to-Point Freight 

**Authors**: Wei Wang, Yiru Veronika Wang, Sumukh Veeramalla, Xiaohui Liang, Robostreet Research Team  

**Link**: [PDF](https://arxiv.org/pdf/2607.26250)  

**Abstract**: Line-haul trucking costs are dominated by three comparably sized components: energy, driver labor, and equipment. Most efficiency technologies address only one component at a time. This paper presents Robostreet Flow, a freight architecture that jointly optimizes the vehicle, convoy formation, and operating model to minimize cost per ton-mile on high-volume point-to-point corridors. The Flow platform is a battery-electric 6x4 tractor with a teardrop single-seat cab and a drag coefficient of 0.35, approximately 40% below that of conventional Class 8 tractors. A carbon-composite monocoque and structurally integrated batteries reduce net vehicle weight by 50%. A 513 kWh tractor battery and a 340 kWh powered trailer battery provide a 500-mile single-charge range. Four Flow trucks operate as a coordinated convoy with a safety driver only in the lead vehicle, while three followers operate in SAE Level 4 automated mode. Computational fluid dynamics simulations show that close following at an 8 m gap reduces follower drag coefficients by 42-48% and follower peak frontal pressure by approximately a factor of four relative to the exposed lead vehicle. A longitudinal energy model calibrated to these results predicts fleet-average consumption of 1.27 kWh/mi in convoy, compared with 1.60 kWh/mi for an isolated vehicle, for a 20.5% energy saving. Electricity cost is approximately 17% of the equivalent diesel fuel cost. Amortizing one driver across four trucks and accounting for the additional payload enabled by lightweighting reduce operating cost from 9.4 to 4.1 cents per ton-mile, a 56% reduction relative to a diesel baseline. Sensitivity analysis, a hub-to-hub operating concept, and regulatory implications are also presented. 

---
# A large-scale corpus of religious radio broadcast transcripts from webstream recordings in the United States 

**Authors**: Samuel Bestvater, Athena Chapekis, Skyler Seets, Anna Lieb, Sono Shah, Aaron Smith  

**Link**: [PDF](https://arxiv.org/pdf/2607.26249)  

**Abstract**: Religious radio is a widespread but understudied form of mass communication in the United States, and content-level analysis of it has been constrained by the absence of large-scale transcript data. This Data Descriptor presents a corpus of transcribed English-language religious radio broadcasts captured from live webstreams over a one-month period in July 2025. Fifteen-minute segments were recorded on a rolling schedule from 785 distinct streams, which together rebroadcast the signals of more than two thousand AM and FM stations, yielding over 700,000 recordings and more than 60 million diarized lines of speech. Each recording was transcribed and speaker-diarized with an automated pipeline, and segmented and labeled by programming format and topic using a large language model. The corpus is organized as linked tables of stream metadata, recording metadata, and transcript lines. It supports descriptive study of religious broadcasting across regions and traditions, analysis of how social and political issues are discussed in religious media, and speech-processing research in an underrepresented domain. 

---
# Steering Instruction Hierarchies at Inference Time 

**Authors**: Siqi Zeng, Sewoong Lee, Han Zhao, Julia Hockenmaier  

**Link**: [PDF](https://arxiv.org/pdf/2607.26228)  

**Abstract**: Instruction hierarchies are a core safety assumption of language model deployment: higher priority inputs, such as system prompts, should override conflicting lower priority inputs from users or tools. Yet frontier LLMs often violate this hierarchy. We introduce V-Steer, a training-free inference time method that restores privileged influence by editing cached value vectors at prompt positions. Using direct logit attribution on the first next token prediction, V-Steer identifies heads where lower priority spans dominate privileged ones, then boosts privileged spans and suppresses conflicting lower priority spans through in-place multiplicative edits to cached V tensors. Since the method acts only on cached values, it remains compatible with fused attention backends and adds only a one time prefill overhead. Across models from 7B to 70B, this attribution guided intervention raises primary constraint accuracy from under 18% up to 92% on controlled role conflict benchmarks, and on broader instruction hierarchy evaluations substantially outperforms prompt only baselines while matching or exceeding SoTA training based methods on 3 of 4 scales of LLMs, with negligible decoding-speed overhead. The code is available at this https URL. 

---
# Characterizing Human-Likeness in AI Generated Poetry: A Zero-shot Classification Study 

**Authors**: A. N. Biswas, T. Tabassum, A. A. Shohid, R. M. Mou, A. A. Esha, F. Sadeque, A. Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2607.26221)  

**Abstract**: With the advancement of AI technologies, Generative AI (GenAI) and human written text have become nearly indistinguishable. Additionally, the global standardization of AI chatbots made academic malpractice more frequent. Furthermore, existing research indicates GenAI poems are the most difficult to distinguish even without any modification thus, GenAI poems are naturally deemed human-like by modern detectors. However, the objectivity of such dissertations needs to be verified against modern detection tools but the subjectivity of poetry and the black-box nature of the modern LLMs (Large Language Models) architectures made verification of such work quite complicated. Hence, the main objective of the research is to deduce the attributes of English poetry that contribute classification and misclassification of both human and AI poems and provide corroborating or contradicting evidence to the poetry distinguishability claim. For such characterizations, we propose a Zero-shot detection pipeline with a dataset consisting of both human and AI poems to verify the distinguishability of human and AI creation and extract the aforementioned crucial attributes for accurate classification. Extraction of such attributes provides benefits in two ways: firstly, it reduces the margin of training needed as only the poems based on misclassifying attributes need to be trained and fine tuned and finally provides a critical insight to the GenAI detection dilemma to strengthen the modern detection pipelines. 

---
# Choosing Where and How to Moderate: End-to-End Trade-offs in Filter Placement and Response Rewriting 

**Authors**: Mengya Hu, Susie Park, Suzana Ilic, Qiong Wei, Sandeep Atluri, Myra Deng, Tucker Fross, Curt Tigges  

**Link**: [PDF](https://arxiv.org/pdf/2607.26200)  

**Abstract**: Content-moderation classifiers are usually evaluated in isolation, but deployment requires choosing where to intervene and what follows a flag. We evaluate these choices using two end-to-end customer-outcome metrics rather than component accuracy: Usefulness, the fraction of turns with a shown, non-harmful, relevant response, and Harmful Exposure, the fraction with a shown harmful response. Latency and error rates are diagnostics. We compare Input only, Response only, and Input + response hard blocking on a human-labelled product benchmark and public ToxicChat evaluation. At the evaluated operating points, Response only achieves the highest filter-only Usefulness in both settings, while Input + response achieves lower Harmful Exposure. Replacing Response only blocking with Response + rewrite recovers most blocked traffic and yields the same observed Harmful Exposure count as Response only blocking for the selected configuration; this equality is not an equivalence result. Probe routing substantially reduces conditional route-and-generation time relative to LLM routing at comparable measured outcomes. A focused output review shows how rewrites balance filter passage with usefulness by generalizing triggering language while retaining benign intent and safe redirection; some sensitive-domain outputs nevertheless omit potentially safety-relevant support information. These results support comparing moderation configurations under deployment-specific safety and latency constraints rather than applying a universal placement rule. Code and public artifacts are available at this https URL 

---
# DuplexGen: Adaptive Synthesis of Human-AI Turn-Taking Dialogues 

**Authors**: Takyoung Kim, Kang-wook Kim, Sang Hoon Woo, Julia Hirschberg, Gunhee Kim, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2607.26178)  

**Abstract**: Turn-taking is a central component of full-duplex interaction. Which turn-taking behaviors are appropriate varies with the scenario, yet current models apply a single norm regardless of context. This limitation originates in their training data: human-human speech corpora capture natural timing phenomena but provide little role grounding or scenario-specific norms, while heuristic or prompted synthesis methods inject turn-taking behaviors without basing them on human preferences. We introduce DuplexGen, a framework for generating dialogues with scenario-adaptive turn-taking by calibrating LLM predictions against a small set of slot-level human preference annotations. In six cooperative and competitive tasks, human turn-taking preferences differ systematically, and DuplexGen aligns substantially more closely with those preferences than uncalibrated prompting or training solely on generic human-human data; a full-duplex model trained on DuplexGen-generated data exhibits distinctive, human-preferred turn-taking behaviors. These results show that human calibration, not corpus scale or prompt design alone, is what allows turn-taking synthesis to be scenario-specific. 

---
# Do Methods Support the Claims? Intra-Paper Verification for Peer Review 

**Authors**: Ranjitha Shivaprasad Ballakuraya, Arash Mahyari, Ashok Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2607.26066)  

**Abstract**: The growing volume of scientific submissions has motivated interest in using large language models (LLMs) to assist peer review. Existing automated novelty assessment approaches typically compare a paper's claimed contributions against prior literature, implicitly assuming that these contributions are accurately realized in the work itself. Human reviewers, however, frequently challenge novelty claims not because similar ideas already exist, but because the methodological evidence presented in the paper does not adequately support them. This internal mismatch between claimed contributions and methodological realization is rarely examined by current LLM-based review systems. To address this gap, we introduce intra-paper claim verification, a framework that evaluates whether novelty claims articulated in a paper are substantiated by the methods used to realize them. The framework employs an LLM to extract novelty claims from the introduction, retrieve claim-relevant methodological evidence, and assess whether the methods substantiate the stated contributions. Assessment is guided by reviewer-inspired evaluation criteria derived inductively from human peer reviews collected from 182 ICLR 2025 papers. These criteria capture recurring reviewer concerns related to novelty, methodology, clarity, and other issues and are used to generate structured reviewer-style assessments of claim substantiation. We evaluate the framework by comparing LLM-generated review comments against human reviewer concerns on a balanced subset of accepted and rejected papers. Human evaluation demonstrates significant alignment between framework-generated assessments and human reviewer concerns, particularly for novelty-related issues. BERTScore further distinguishes corresponding human-LLM review pairs from mismatched controls, indicating that the framework captures concerns consistent with human reviewer observations. 

---
# Large-Scale ChatBot Validation Through Customer Digital Twin Simulations 

**Authors**: Cristovao Iglesias, Devesh Batra, Alankar Atreya, Stefan Wagner, Robert Hankache, Patrick Sinclair, Giulio Pelosio, Michael McMillan, Greig A. Cowan, Raad Khraishi  

**Link**: [PDF](https://arxiv.org/pdf/2607.26060)  

**Abstract**: LLM-based chatbots are transforming customer service in regulated domains such as banking, but scalable and cost-effective validation remains a critical barrier to safe deployment. We present a two-part contribution for large-scale chatbot validation. First, we introduce a methodology for creating high-fidelity synthetic customer agents (SCAs) as digital twins, grounded in real transactional and conversational data, that enables automatic generation and behavioral conditioning to simulate diverse customer profiles and interaction styles. Evaluation demonstrates that SCAs achieve high semantic alignment with real customers, low hallucination rates, and successful personality trait reproduction with controllable interventions. Second, we develop an SCA-based validation framework combining automated LLM-as-a-Judge evaluation, human expert testing, and adversarial probing. Scenario-based validation across emotional states, demographic groups, and linguistic factors confirms robust performance. Our approach was used to validate a customer facing chatbot at a leading UK bank, providing financial institutions with a scalable pathway toward regulatory compliance. 

---
# SpecFirst: Behavioral Specification Elicitation as a First-Class Step in Agent-Based Program Synthesis from Scratch 

**Authors**: Yihao Chen, Shi Chang, Feng Lin, Khaled Chawa, Boyuan Chen, Shaowei Wang, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2607.27167)  

**Abstract**: LLM-based agents excel at software engineering tasks where an existing codebase provides context, but constructing a program from scratch remains fundamentally harder. Recent benchmarks such as ProgramBench quantify this gap: given only natural-language documentation and an execute-only binary as a behavioral oracle, even frontier models solve fewer than 1% of instances. Existing frameworks conflate documentation reading, behavioral exploration, and code synthesis into a single pass, causing agents to probe insufficiently, lose behavioral intent as context drifts, and propagate early misinterpretations into the final implementation. Inspired by classical requirements engineering, we argue that behavioral specification elicitation should be a first-class phase that precedes implementation. We present SpecFirst, a two-stage framework that forces the specification elicitation before code synthesis. A dedicated spec agent first probes the binary and combines observations with documentation into a structured specification. Next, a code synthesis agent then uses this specification to drive implementation. This decomposition resolves documentation ambiguities before coding begins and provides a stable behavioral reference throughout synthesis. We evaluate SpecFirst on all 200 ProgramBench instances across four models spanning two families and an order of magnitude of capability. SpecFirst consistently outperforms the single-loop baseline, improving test pass rates by 6.9%-21.3% and binary exploration coverage by 9.4%-18.5%, all statistically significant. Behavioral analysis on code synthesis further shows that a prior specification enables earlier and more sustained code construction. Our results demonstrate that an explicit requirements-engineering phase is an effective paradigm for from-scratch program construction. 

---
# OmegaUse-OfficeVal: Benchmarking LLM Agents on Long-Horizon Office-Suite Tasks with Economic Grounding 

**Authors**: Jingbo Zhou, Yusai Zhao, Qi Bao, Jingjia Cao, Zhenghai Chen, Chang Gao, Kaiqi Guo, Muxin Guo, Mingxuan Li, Xinjiang Lu, Yanru Ma, Yixiong Xiao, Zenghui Zhang, Le Zhang, Hua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.27155)  

**Abstract**: Large language model (LLM) agents are increasingly expected to assist users in completing tasks. However, existing benchmarks provide limited support for evaluating whether agents can carry out office-suite workflows at a reasonable cost. We introduce OmegaUse-OfficeVal, a benchmark for evaluating LLM agents on long-horizon office-suite tasks with task-level economic grounding. The benchmark comprises 100 tasks derived from office-suite requests proposed by practitioners and adapted through a privacy-preserving process. On average, these tasks require 2.32 hours of human labor to complete. An important feature of the benchmark is that each task is paired with two economic signals: human labor time and task price proxy. These signals enable direct comparisons between human costs and LLM inference costs, as well as value-weighted evaluation. To support stable evaluation, we develop code-based verifiers from fine-grained rubrics. We evaluate several frontier LLMs together with a human baseline. Although all evaluated LLMs are substantially cheaper and faster than human workers, they have not yet approached human-level deliverable quality. The code and dataset are fully open-sourced, and more information is available on our project website: this https URL. 

---
# MindForge: Teaching Small Language Models Whole-Life-Cycle Software Engineering via Source-Free Program Synthesis 

**Authors**: Yihao Chen, Shi Chang, Khaled Chawa, Feng Lin, Boyuan Chen, Shaowei Wang, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2607.27146)  

**Abstract**: Coding agents have made substantial progress on software engineering tasks that modify existing codebases, including bug fixing and feature implementation. However, constructing a complete program from scratch remains a major challenge: even the frontier models evaluated on ProgramBench fully resolve fewer than 1% of tasks. One obstacle is the lack of scalable training environments for this from-scratch setting, spanning the whole software engineering life cycle, as existing environment-construction frameworks focus only on a single phase in software development. To address this gap, we introduce MindForge, an automated pipeline that converts open-source command-line programs into source-free environments that expose only a compiled reference executable and its documentation. Using MindForge, we construct training environments from repositories disjoint from those in ProgramBench, and curate a high-quality data recipe consisting of program synthesis trajectories using GLM-5.2 as the teacher agent. Fine-tuning Qwen3.6-27B on these trajectories increases its ProgramBench average test pass rate from 37.98% to 49.51%, achieving performance comparable to substantially larger frontier models. Moreover, the fine-tuned model consistently improves over the base model across all seven unseen software engineering benchmarks, spanning long-horizon repository generation and translation, bug fixing, feature implementation, and cross-language issue resolution, with absolute gains of 31.00 points on RepoZero-C2Rust, 14.16 on DeepSWE, 10.70/4.56 on NL2Repo-Bench (with/without tests), 5.04 on SWE-bench Verified, 5.93 on SWE-bench Pro, 5.22 on SWE-bench Multilingual, and 4.94 on FeatBench. 

---
# Linguistic Monoculture in LLM-Assisted Language Use 

**Authors**: Suhas Thejaswi, Juhi Kulshreshta, Lutz Oettershagen  

**Link**: [PDF](https://arxiv.org/pdf/2607.27134)  

**Abstract**: Writing and communication are increasingly mediated by large language models (LLMs) that are being used to draft, revise and polish text. Although such assistance can improve clarity and help authors meet institutional expectations, widespread reliance on shared models may reduce population-level variation in linguistic form, a phenomenon we refer to as linguistic monoculture. We develop a mathematical framework in which authors and LLMs are represented as distributions over linguistic features and coevolve through repeated interaction. We analyze three interaction mechanisms: a shared model with a fixed linguistic distribution, a shared model recursively updated from author outputs, and personalized models updated through author-specific and population-level feedback. We characterize the resulting equilibria and convergence rates, showing that, shared models can drive authors toward a common norm, recursive feedback relocates the shared norm without altering pairwise spread under common conformity, and personalization can preserve a family of distinct author-model equilibria with nonzero linguistic diversity. We then endogenize conformity as a strategic choice trading off private benefits from clarity, legibility, and perceived fluency against distinctive style. Within this utility model, individually rational authors may conform more than is socially optimal because they do not internalize the value their distinctiveness provides to others, creating a negative externality and a price of monoculture that is finite for each fixed instance but can grow without bound when distinctiveness dominates authenticity. Synthetic simulations illustrate how fixed shared assistance, recursive feedback, and personalization produce different long-run diversity outcomes. 

---
# On-Policy Distillation for LLM Safety: A Routing Approach to Template-Robust Realignment 

**Authors**: Yongjian Guo, Wanlun Ma, Lingyu Shen, Xi Xiao, Sheng Wen  

**Link**: [PDF](https://arxiv.org/pdf/2607.27081)  

**Abstract**: Fine-tuning is the dominant paradigm for specializing large language models (LLMs), yet it exposes a critical vulnerability: malicious data providers can embed harmful behaviors into downstream corpora, creating models that retain professional skills while violating human values on demand. Existing safety-realignment defenses often fail in practice due to three key limitations: they frequently cause catastrophic forgetting of specialized skills; their effectiveness collapses when the defender cannot observe the attacker's prompt template; and successfully realigned models remain susceptible to re-jailbreaking via simple system prompt switches. To address these challenges, we propose Routing-based On-Policy Distillation (ROPD), a novel realignment framework that models the divergence between aligned and compromised output probability distributions rather than fitting specific prompt templates. We conduct extensive experiments comparing ROPD against four state-of-the-art baselines across three datasets and three base models with varying alignment strengths. Our results demonstrate that when baseline defenses face template mismatches, often accompanied by severe degradation in downstream task performance. In contrast, ROPD substantially mitigates template-mismatch risks, maintaining superior robustness in both defense effectiveness and capability preservation. While our analysis indicates ROPD is not entirely immune to template shifts, its performance degradation is negligible compared to existing methods, establishing a new standard for robust LLM realignment. 

---
# Setoka: A Benchmark for Hierarchical User Understanding in Personalized Agents over Heterogeneous Data 

**Authors**: Lingyang Zeng, Guangze Chen, Kaichen Yu, Zhicheng Pan, Siyang Weng, Zirui Hu, Xiangyun Du, Hailin He, Rong Zhang, Chengcheng Yang, Kai Huang, Xuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2607.27056)  

**Abstract**: Personalized agents are increasingly applied to assist users across a wide range of tasks. Effective personalized assistance requires not only retrieving explicit facts from past interactions stored in agent memory, but also inferring abstract personal characteristics. However, existing memory benchmarks primarily evaluate whether an agent can retrieve information explicitly stated in conversational histories, failing to provide an effective assessment of deeper user understanding. In this work, we propose Setoka, a benchmark for evaluating memory-augmented personalized agents with hierarchical user understanding from heterogeneous data. Grounded in theories from cognitive and personality psychology, Setoka defines four levels of user understanding, i.e., semantic memory, episodic memory, behavior pattern, and personality trait. Moreover, to enable realistic yet privacy-preserving evaluation, we design a psychometrics-based pipeline that synthesizes diverse, coherent heterogeneous user data and queries at scale. Finally, we leverage Setoka to evaluate 3 language models combined with 5 memory systems for 10 synthetic users. Our comprehensive evaluation reveals that while existing systems perform well on semantic memory retrieval, their performance declines on episodic memory. Moreover, when dealing with behavior pattern and personality trait understanding tasks that require integrating heterogeneous and fragmented information dispersed over time, performance declines even further. These findings demonstrate that user understanding cannot be handled by simple fact retrieval, motivating the design of memory mechanisms for cross-source integration and abstraction over long-term user behavior. 

---
# AgentSnare: Learning to Delay, Divert, and Defuse Autonomous Penetration Agents 

**Authors**: Ruoyu Wang, Heng Zhao, Renjie Wu, Mengnan Zhao, Zhixuan Chu, Wanyu Lin, Tianhang Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.26998)  

**Abstract**: Large language model (LLM) agents automate penetration testing through an observation-action loop, selecting actions based on observations returned by tools. This dependence allows defenders to inject deceptive observations that can mislead the agent's decision-making process. However, existing defenses rely heavily on static, isolated artifacts planted in the environment prior to an attack. Advanced agents can progressively recognize and bypass these artifacts, ultimately refocusing their exploitation attempts on the real target. To address this issue, we introduce AgentSnare, a trajectory-adaptive deception system that dynamically unfolds a decoy environment to continually steer the penetration agent away from the real target. Specifically, AgentSnare employs an artifact-construction policy model that constructs candidate artifacts conditioned on the agent's interaction history and decoy state. AgentSnare then validates these candidates and incrementally incorporates valid artifacts into a factually consistent decoy environment, thereby delaying the attack by absorbing its tool calls, diverting its post-entry trajectory within the decoy, and defusing it by inducing completion reports grounded in decoy evidence. Across 15 CVE-Bench web applications and three attacker models, AgentSnare absorbs 46.8% of the agent's tool calls in the decoy and retains 55.9% of post-entry actions there, while 90.0% of completion attempts are grounded in decoy evidence; across all 45 attacker-CVE pairs, no real target is successfully exploited at pass@3. 

---
# Hearsay: Vision-Language Medical Diagnoses Without an Image 

**Authors**: Siddharth Vohra  

**Link**: [PDF](https://arxiv.org/pdf/2607.26886)  

**Abstract**: When asked to describe a medical image that was never attached, frontier vision-language models do not abstain: they confabulate a diagnosis. We show that this confabulation is not random. It is structured by who the patient is said to be. Across chest X-ray, brain MRI, and dermatology, Claude Opus-4.7, GPT-5.4, and Gemini-3.1-Pro are each queried with only a demographic descriptor and no image, and changing the descriptor systematically shifts the diagnosis returned. Claude concentrates sharply: a 65-year-old white man asking about a skin mole receives Melanoma in nearly every response, and a 32-year-old Black woman asking about her chest X-ray receives a Sarcoidosis diagnosis whose reasoning reads "suspected, based on demographics and classic pattern.'' GPT-5.4's effect is broader, fabricating across every demographic cell we test, most conspicuously naming Sarcoidosis for young Black patients on chest X-ray. Two structural findings sharpen the problem. A hedged regime appears in which the prose acknowledges the missing image while the structured diagnosis field nevertheless names a disease, a dissociation invisible to prose-only audits. And Claude's dermatology effect collapses entirely when 'skin mole' is swapped for 'skin lesion' while GPT-5.4's is preserved, indicating that mirage is a family of distinct failure modes rather than a single phenomenon. Trustworthy VLM deployment in clinical pipelines requires auditing the structured output channel directly, and probe-word sensitivity should be treated as a first-class evaluation dimension 

---
# SecRespond: Benchmarking AI Agents for Real-World Post-Compromise Incident Response 

**Authors**: Lehan Wang, Boli Chen, Ruixue Ding, Pengjun Xie, Jinwei Huang, Zhendong Liu, Shuo Wang, Tao Lei, Xin Ouyang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.26791)  

**Abstract**: Large Language Model (LLM) agents are increasingly adopted in real-world security operations with access to host artifacts and command-line interfaces (CLIs), making it critical to thoroughly assess their security capabilities. However, existing cybersecurity benchmarks focus on pre-compromise settings where agents are placed in a clean and idealized environment before an attack occurs. This leaves the post-compromise setting underexplored. To address this gap, we introduce SecRespond, the first benchmark for evaluating LLM agents on the post-compromise incident-response workflow. Given a forensic disk snapshot of a compromised host together with the alerts, vulnerability scans, and baseline checks reported by a host security product, agents are required to produce forensic reports on intrusions, baseline risks, and vulnerability risks, together with a remediation plan. We instantiate this task across 10 cyber ranges, each constructed from a distinct compromised cloud host, spanning 4 entry-point types, 21 ATT&CK techniques, and 5 operating systems. We evaluate 23 frontier LLMs on the OpenCode agent harness. Experimental results show that although current agents can reliably uncover the problems exposed by alerts, they struggle to proactively investigate the disk for silent intrusions and to produce comprehensive, verified remediation plans, with no model achieving complete detection and remediation on any single range. This reveals a fundamental bottleneck in building agents for real-world incident response. The benchmark is publicly available at this https URL. 

---
# MediaWiki Code2Code Search: Neural Retrieval for the Semantic Discovery of Open-Source Software Entities 

**Authors**: Francesco Tosoni  

**Link**: [PDF](https://arxiv.org/pdf/2607.26766)  

**Abstract**: Code search in large-scale ecosystems is often hindered by the lexical gap between user queries and implementation details, alongside the trade-off between the low latency of traditional Information Retrieval (IR) and the precision of Deep Learning (DL). We present MediaWiki Code2Code Search, a neural retrieval system for semantic code-to-code discovery. By indexing 1.29 million structural entities (functions, types, and templates) across 2,500+ MediaWiki repositories, our system enables retrieval based on computational intent rather than surface tokens. We employ a split-build architecture, decoupling GPU-intensive offline indexing from a CPU-only serving layer; our FAISS IVF-PQ index occupies 168.6 MB: a 96.6\% reduction compared to a flat float32 baseline, and achieves a median query latency of 1.85 seconds on commodity hardware, satisfying the 6 GiB RAM constraint of Wikimedia Toolforge. Our evaluation across a 27-query benchmark demonstrates superior performance over the BM25 baseline, achieving a P@10 of 0.87 compared to 0.64 (0.52 versus 0.34 for strict matching). Gains are most pronounced in name-obfuscated tasks where lexical methods fail. The system is available at this https URL under the Apache 2.0 licence and provides an open RESTful API. 

---
# Scientific Knowledge Discovery in the Age of Large Language Models 

**Authors**: Eleni Adamidi, Serafeim Chatzopoulos, Thanasis Vergoulis  

**Link**: [PDF](https://arxiv.org/pdf/2607.26670)  

**Abstract**: The rapid growth of scholarly literature has made identifying relevant publications increasingly difficult, and conventional search systems still depend heavily on manually formulated queries and effortful manual inspection. Generative large language models (LLMs) offer a more flexible alternative, supporting literature retrieval and the screening of candidate studies against eligibility criteria. This chapter surveys 34 peer-reviewed papers applying generative LLMs to these two tasks, identified via a Boolean search over the OpenAIRE Graph (1,589 records screened to 34 inclusions). Reviewed studies are characterised by LLMs employed, model access and adaptation, prompting and architectural techniques, ground-truth sources, and evaluation metrics. 

---
# FedWeave: Rethinking the Unit of Specialization in Heterogeneous Federated MoE-LoRA 

**Authors**: Donghang Duan, Xu Zheng, Lizong Zhang, Chong Mu, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2607.26618)  

**Abstract**: Federated PEFT enables LLMs to collaboratively adapt to decentralized private data without sharing raw examples. However, task heterogeneity across clients can cause cross-task interference and gradient conflicts during aggregation. Federated MoE-LoRA addresses this challenge through specialized LoRA experts and conditional routing. Yet existing methods typically specialize at client granularity, implicitly assuming task-coherent clients. Our core insight is that experts need purity, namely pattern-coherent updates that preserve specialization, whereas routers need contrast, namely mixed-task observations that support expert comparison. We propose FedWeave, a framework that adopts asymmetric aggregation, separating expert aggregation from router optimization to meet these two requirements. FedWeave uses unsupervised prototype discovery to form local buckets and align them across clients, enabling prototype-level expert aggregation while retaining mixed-task client trajectories for router training. At inference, FedWeave performs sparse inference with one active expert while preserving nearly all soft-routing performance. Our theoretical analysis explains why asymmetric aggregation is advantageous: it controls expert convergence in stationarity through off-pattern contamination, identifies the consensus error induced by fragmented router trajectories, and bounds sparse-inference risk. On a heterogeneous multi-task benchmark with mainstream LLM backbones, FedWeave consistently outperforms strong baselines, while ablations verify the effectiveness of our design. 

---
# Living-Harness Is an Interactive-Agent Evolver 

**Authors**: Yuetian Du, Yucheng Wang, He Xu, Jiexu Xu, Shanwen Tan, Bing Zhao, Boyu Yang, Zhijie Xu, Ming Kong, Hu Wei, Jie Liu, Qiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.26598)  

**Abstract**: Large language model (LLM) agents may recover from a failure within an episode or after a retry, yet the same execution failure can recur in later tasks because post-episode feedback rarely revises the persistent harness that guides future interactions. Static harnesses improve reliability through fixed tools, context, memory, and workflow structures, but remain unchanged after deployment. We propose $\textbf{Living-Harness}$, a self-evolving agent harness that converts each completed trajectory and its evaluator signals into posterior evidence for bounded harness updates. Guided by a domain-level $\textbf{Evolution-SOP}$ ($\textbf{S}$tandard $\textbf{O}$perating $\textbf{P}$rocedure), Living-Harness extracts an episode abstraction and structured update evidence, and writes two complementary forms of procedural knowledge: episodic memory that records trigger conditions, failure patterns, and recovery actions, and a state graph that records state nodes, repair edges, and transition rules. The updated harness state is retrieved to guide future interactions, while tools and base context remain frozen, allowing procedural repairs to accumulate across evolution cycles. On eight interactive environments derived from $\tau^2$-Bench and MultiWOZ-2.4, Living-Harness improves average Pass@1 over the strongest interactive baseline by 10.07 and 9.91 percentage points, respectively, and supports retrieval-only reuse of the evolved harness state across model backbones. 

---
# Prosody-driven Jailbreaks in Audio LLMs: A Controlled Study and Mechanistic Analysis 

**Authors**: Jiachen Qian, Junyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.26541)  

**Abstract**: Audio-capable foundation models enable end-to-end spoken interaction, but they also introduce safety risks beyond transcript content. It remains unclear how much jailbreak capability can arise from matched-text variation in speech delivery rather than from lexical rewriting or broader style transfer. We study this question by holding transcript content fixed and varying six speech-delivery presets whose acoustic attributes may co-vary. We present PJ-Break, a black-box evaluation protocol with presets targeting arousal, authority, and speaking rate, together with AdvAudio-Prosody, a 600-sample benchmark with acoustically verified attributes. On the exact post-QC Qwen2-Audio panel, the Q=1 Panic (38/95), Anger (35/95), and Fast (32/95) presets are all well above Neutral (4/95). The fixed six-query pool covers 44/95 Qwen2-Audio seeds and 15/95 GPT-4o seeds and exceeds a matched-budget StyleBreak reimplementation (27/95) on Qwen2-Audio. A same-voice pool excluding the confounded Commanding condition still reaches 40/95, and a retained-panel ablation shows emotional-delivery audio alone (44/95) is far more effective than emotional text alone (11/95). Exploratory surrogate diagnostics and pilot mitigation observations are secondary, non-core analyses. Overall, matched-text speech delivery should be treated as a first-class factor in Audio LLM safety evaluation 

---
# Learning Dynamic User Personas from Implicit Interaction Streams via Iterative Refinement 

**Authors**: Haifeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2607.26473)  

**Abstract**: Personalizing large language models (LLMs) to individual users is essential for improving user experience, yet existing approaches typically rely on explicit preference supervision such as pairwise comparisons or demographic attributes, limiting their applicability in natural interaction settings. We propose IRIS, a framework that learns dynamic user personas directly from implicit interaction streams by extracting behavioral signals from everyday conversations and iteratively refining persona representations through a prediction-driven closed loop without requiring explicit feedback. We introduce an evaluation protocol based on behavior prediction, persona stability, and decision prediction. A proof-of-concept study on a synthetic interaction stream derived from public-domain autobiographical text shows that IRIS produces stable personas and distinguishes individual users while revealing limitations of memory-only approaches on recall-oriented metrics. We then validate IRIS on anonymized real-world Reddit r/AmItheAsshole (AITA) data, with personas built solely from each author's historical interactions. Across 100 authors, IRIS achieves the highest decision prediction accuracy among all evaluated methods (61.0%), outperforming static personas, memory-only retrieval, and no-personalization baselines. These results suggest that implicit behavioral modeling provides a scalable alternative to explicit preference learning for personalized LLMs and offers a practical foundation for adaptive conversational systems and embodied agents that require continuously evolving models of their users. 

---
# Dissecting Sensitivity to Training Language in Self-Supervised Speech Learning Using Neural Audio Codec Tokens 

**Authors**: Daigo Takizawa, Tomohiko Nakamura, Samuele Cornell, William Chen, Satoru Fukayama, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2607.26350)  

**Abstract**: Neural audio codecs (NACs) have become popular for obtaining speech representations as discrete tokens. Beyond compression, discrete tokens can be used to train self-supervised learning (SSL) models. Such models, referred to as codec-based SSL models, reduce data storage and computational cost, enabling scalable SSL pre-training. However, their language sensitivity remains unclear. When the language changes, codec-based SSL models may require retraining, which undermines their efficiency. In this paper, we present a systematic analysis of language sensitivity by varying either the NAC training language or the SSL pre-training language while keeping the other fixed. Experimental results show that downstream performance is insensitive to the NAC training language but strongly dependent on the SSL pre-training language. These findings suggest that a single NAC can be reused across languages, while aligning the SSL pre-training language with the target language is crucial. 

---
# Aligning LLM-Simulated and Human Examinees for Psychometric Calibration: A Cognitive Diagnostic Profiling Approach 

**Authors**: Wenjie Zhou, Yunting Liu, Renjiao Tang, Mark Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2607.26317)  

**Abstract**: Psychometric calibration for educational tests typically requires costly human response data. Large language models (LLMs) simulated examinees offer a promising route to early calibration, but their responses are too accurate and too uniform. We propose Cognitive Diagnostic Profiling (CDP), a zero-shot framework that prompts LLMs to simulate plausible examinees with diverse cognitive profiles: binary attribute-mastery patterns are rendered as natural-language profiles and sampled under an uninformative or an informative distribution. Using the Tatsuoka fraction-subtraction dataset (536 examinees, 15 items, five attributes), we evaluated eight LLM configurations under no-profile, uninformative-CDP, and informative-CDP conditions, assessing alignment with human examinees at the ability-distribution, mastery-profile, and item-difficulty levels. CDP improved all three levels: distributional overlap rose across configurations; weighted correlations between profile-level scores and human profile expectations reached 0.92 to 0.98; and item-difficulty recovery improved in rank order and absolute alignment, most for reasoning-enabled models; in the strongest case, Gemini 3.0 Flash (Thinking), one-parameter logistic (1PL) difficulty Spearman correlations rose from 0.24 to 0.86 and 0.90 and the root-mean-square error (RMSE) fell from 6.31 to 1.30 and 0.90; the informative condition helped most where profile-level alignment was strong. CDP brings LLM-simulated examinees into closer psychometric alignment with human examinees, making them practical for operational test development. 

---
# The Confounder Trap: Treatment-Encoding Representations in Causal Inference with Text 

**Authors**: Marie Neubrander, Graham Tierney, Alexander Volfovsky  

**Link**: [PDF](https://arxiv.org/pdf/2607.26309)  

**Abstract**: Estimating causal effects of linguistic properties from observational text is difficult because the same document can contain both the treatment of interest and the non-treatment textual attributes needed for adjustment. Existing approaches often learn representations from the full text to capture latent confounding, but when treatment status is itself encoded by words in the text, these representations can directly encode treatment. This creates a confounder trap: richer representations can make treated and control documents separable, inducing overlap violations even when the underlying causal problem satisfies overlap. We study latent text treatments that are encoded through lexicons or other treatment-defining lexical information, and propose masking-based adjustment representations that remove this lexical treatment signal before representation learning. We formalize representation-induced overlap failure, prove that deletion masking preserves overlap for bag-of-words/topic-model representations, and characterize replacement masking as a natural relaxation for large language models that hides treatment-defining tokens while preserving word order and context. Across simulations, masking improves overlap diagnostics, stabilizes treatment effect estimates, and reduces bias relative to adjustment methods that learn from the unmasked text. 

---
# Position: Evaluation Scores Are Perishable Knowledge Claims 

**Authors**: Sankalp Gilda, Shlok Gilda  

**Link**: [PDF](https://arxiv.org/pdf/2607.26191)  

**Abstract**: Evaluation methodologies for language models increasingly combine multiple signals, from automated metrics and LLM-as-judge ratings to human assessments and benchmark suite results. When these signals are aggregated via averaging, evaluation confidence can then substantially exceed the reliability of the weakest signal: a phenomenon we call trust inflation in evaluation. We argue that evaluation scores should be treated as epistemic claims with three properties: formality (human evaluation provides stronger evidence than an automated metric), scope (a benchmark result applies to the tested distribution, not universally), and validity windows (benchmark results expire as contamination accumulates and distributions shift). Several converging research traditions (chain-of-thought analysis, possibilistic logic, and algebraic theory) establish weakest-link aggregation as the conservative endpoint of a parameterized operator family controlled by a single pessimism parameter. Drawing on those traditions, and on concrete lessons from building an evaluation harness for agentic AI, we propose that evaluation results carry explicit metadata (formality tier, scope declaration, and expiration date) to make their epistemic status transparent. We illustrate the cost of mean aggregation on the public HELM leaderboard: across 54 frontier models on ten scenarios, the top-five models ranked by mean score and by weakest-link are completely disjoint. 

---
# Cognitive Convergence: Deep Similarities Between Large Language Models and Human Cognition 

**Authors**: Chandra Sripada, Richard Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2607.26179)  

**Abstract**: LLMs are widely regarded as alien intelligences, systems whose cognitive operations are fundamentally unlike our own. Apparent similarities to human cognition are therefore often seen as the result of anthropomorphic projection. We argue that this framing is mistaken. LLMs clearly differ from humans in important respects, including their physical substrate, learning history, and the environments with which they interact. These differences make it all the more striking that contemporary LLM-based systems converge with human cognition on a number of principles of cognitive organization with longstanding support in cognitive science. We identify structural correspondences across five dimensions: inferential organization, computational architecture, representational structure, prediction-driven learning, and reinforcement-learning-like mechanisms supporting goal-directed action. These correspondences support a broader model of intelligent cognition in which core principles long used to explain human intelligence also characterize contemporary LLM-based systems. 

---
# Probing the Origins of Reasoning Performance: Representational Quality for Mathematical Problem-Solving in RL vs. SFT Fine-Tuned Models 

**Authors**: Antyabha Rahman, Akshaj Gurugubelli, Omar Ankit, Kevin Zhu, Aishwarya Balwani  

**Link**: [PDF](https://arxiv.org/pdf/2607.26119)  

**Abstract**: Large reasoning models trained via reinforcement learning (RL) have been increasingly shown to outperform their supervised fine-tuned (SFT) counterparts on mathematical reasoning tasks; Yet the mechanistic basis for this advantage remains unclear. We therefore ask, what internal representational differences enable RL models' superior performance? Our work presents two converging lines of evidence: First, linear probes trained on layer-wise hidden states reveal that RL models tend to achieve higher accuracy in predicting answer correctness compared to SFT models, indicating more linearly separable and structured representations. Second, mean ablation studies show that RL models develop a hierarchical architecture where deeper layers become progressively more critical, whereas SFT models distribute importance uniformly across layers. Together, these findings demonstrate that RL training fundamentally restructures how models represent and process reasoning problems. Finally, we analyze token-count variability under repeated sampling across problems to assess adaptive compute allocation. While we observe higher variability in some RL-tuned models than in their SFT counterparts, we see strong consistency in others, suggesting that token allocation may depend more on the overall training pipeline than on RL versus SFT alone. We believe this token-allocation variability reveals the spread of plausible on-policy reasoning, highlighting which models exhibit stable policies versus those that are under-determined, potentially non-identifiable solution behaviour. 

---
# GPT-Red: Automated Red Teaming via Self-Play at Scale 

**Authors**: Eric Wallace, Christopher A. Choquette-Choo, Nikhil Kandpal, Sam Toyer, Dylan Hunn, Stephanie Lin, Yuxin Wen, Xiangyu Qi, Christopher Wolff, Zizhao Wang, Milad Nasr, Sicheng Zhu, Chuan Guo, Juan Felipe Cerón Uribe, Kaiwen Wang, Aiden Low, Kai Xiao, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.26115)  

**Abstract**: We introduce \textbf{GPT-Red}, an automated red-teaming agent that is trained to discover novel prompt injection attacks against frontier LLMs. The goal of this model is to evaluate and improve the robustness of our production systems. To this end, we use it to adversarially train GPT-5.6, our most robust model to prompt injections to date. To create GPT-Red, we design a scalable self-play algorithm where the model is tasked with attacking a diverse population of simultaneously-trained defender agents. We train the model on realistic red-teaming environments using compute on the same scale as some of our largest RL post-training runs, making it the single-largest LLM safety training run ever documented. GPT-Red excels at red-teaming: it reliably breaks our past models up to GPT-5.5, it finds more successful attacks than human red-teamers, and it generalizes to held-out environments, defender models, and harnesses. In the future, we expect that as we improve the robustness of each new GPT model, it will in turn will provide better learning signal for \textit{even stronger} red-teamer agents, thus unlocking a self-improvement flywheel. 

---
# Meta-Learned Reward Shaping for Reinforcement Learning from Human Feedback 

**Authors**: Yunpeng Chu  

**Link**: [PDF](https://arxiv.org/pdf/2607.26094)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is the standard approach for aligning large language models with human preferences, but its quality is limited by static, task-agnostic reward models. This mismatch leads to sparse learning signals and suboptimal alignment. We introduce MeRLa (Meta-Learned Reward Shaping), a principled framework that meta-learns a task-aware shaping function $\Phi(x,y;\phi)$ across auxiliary tasks before RLHF training. The learned shaping produces a composite reward that preserves policy optimality while providing task-specific learning signals. Our meta-objective combines task discrimination, entropy regularization, and potential-based conservation for stable convergence. We provide theoretical guarantees for policy invariance, analyze representation drift sensitivity, and formally address incentive misalignment from entropy maximization. Experiments on LLaMA-3-8B across four benchmarks show consistent improvements over PPO, DPO, GRPO, and DAPO, achieving a 90.8% length-controlled win rate on AlpacaEval 2.0 and a score of 9.14 on MT-Bench, with 41% less training instability. MeRLa retains its benefits when combined with process-based and rubric-based enhanced rewards. 

---
# Identifying Implicit Bias in LLM-based Chat AI Toward People with Intellectual Disabilities 

**Authors**: Karly V. Coffey, Gloria L. Krahn, John P. Hanley, Jacob E. Neely  

**Link**: [PDF](https://arxiv.org/pdf/2607.26062)  

**Abstract**: Background: This work investigates the presence of implicit bias in Large Language Model (LLM)-based chat AI models directed toward people with intellectual disabilities (ID).
Objective: The study aims to identify and measure representational differences related to people with ID and examine them to identify implicit biases inherent in AI chat generation technologies.
Methods: Utilizing the GPT-4-Turbo model, we requested story-generation based on 10 prompt stems with and without descriptors for ID. This process was repeated using four other LLMs (OpenAI GPT-4o, Meta Llama-3-3-70B-Instruct, Anthropic Claude-3-5-Sonnet, and Mistral-Large-2411). The resulting 25,000 computer-generated stories were analyzed using a separate GPT-4-Turbo model instance to detect differences in how people are represented related to themes of bias described in previous literature.
Results: Our findings reveal differences in how people are represented between story datasets with and without ID descriptors. These differences go beyond established characteristics of ID and imply the presence of mostly negative implicit biases. Identified differences related to considering people with ID as younger, with themes of paternalism and infantilization; depicting them as more inspirational and symbolic; as needing help more often, being dependent, and being saved; and having a negative perception of them and more hesitation to include them.
Conclusions: These implicit biases are considered within the context of past discrimination towards people with ID and highlight the need for diligence against implicit bias towards people with ID in AI development. This research underscores the importance of assessing and mitigating implicit bias in decision-making technologies to prevent future societal harm. 

---
