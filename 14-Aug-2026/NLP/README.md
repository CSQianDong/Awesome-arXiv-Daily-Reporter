# LittleLearner: Language Models Under Pedagogically Controlled Knowledge Exposure 

**Authors**: Fanfei Li, Jana Zeller, Manuel Prada-Corral, Thaddäus Wiedemer, Prasanna Mayilvahanan, Ryan Cotterell, Wieland Brendel  

**Link**: [PDF](https://arxiv.org/pdf/2608.13545)  

**Abstract**: Modern language models are trained on heterogeneous web-scale text corpora. Consequently, studying knowledge and skill acquisition is difficult, as prior exposure to related content is hard to characterize. To address this challenge, we introduce LITTLECURRICULUM, a curated 88B-token pretraining corpus tailored to U.S. elementary school material, explicitly excluding concepts, facts, and vocabulary taught above Grade 5. Training a 5B-parameter LLM from scratch on LITTLECURRICULUM yields LITTLELEARNER, a model with sufficient language competence for open-ended evaluation, yet with clear knowledge and capability boundaries mapped to interpretable curriculum guidelines. We release LITTLECURRICULUM and LITTLELEARNER as a developmentally restricted sandbox to study how models acquire, represent, and use data under a well-defined training scope. We illustrate the sandbox's utility in a first suite of experiments on injecting new knowledge through post-training and in-context learning. These methods let LITTLELEARNER better utilize existing knowledge, but do not raise out-of-scope capabilities. Our findings underscore the value of this controlled environment for future investigations. 

---
# SAEVerbalizer: Generating Explanations for Sparse Autoencoder Features via Representation Verbalization 

**Authors**: Weihan Meng, Hongzhu Guo, Yi Jing, Dewen Liu, Zijun Yao, Xiaozhi Wang, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.13538)  

**Abstract**: Sparse autoencoders (SAEs) are proposed to extract numerous features from large language model (LLM) representations, yet explaining these features still relies primarily on external observation. This reliance leads to superficial explanations inferred from observed model behavior and computational inefficiency from collecting such behavioral evidence at scale. We introduce SAEVerbalizer, a framework that injects SAE decoder directions into an LLM's representations and fine-tunes the LLM's downstream layers to generate natural-language explanations of the injected features. Once trained, the resulting verbalizer explains SAE features directly from decoder directions, addressing both limitations. Our experiments show that the learned verbalization capability generalizes to unseen features, transfers across separately trained SAE dictionaries, and, with a lightweight adapter, extends to SAE features from different LLMs. Intervention experiments show that injecting multiple directions yields an explanation combining their meanings, while reversing individual directions produces corresponding meaning shifts. 

---
# DFM Mimir v1: An Open HRM Delivering Frontier Performance at 1B Parameters Using Only Permissible Post-Training Data 

**Authors**: Peter Schneider-Kamp, Jacob Nielsen, Gianluca Barmina, Kenneth Enevoldsen, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2608.13517)  

**Abstract**: Current large language model development relies on massive, often non-permissible datasets, creating a high barrier for researchers committed to open-source and ethically sourced data. We introduce Mimir v1, a 1-billion-parameter language model based on the Hierarchical Reasoning Model (HRM) architecture, that is trained from scratch and delivers highly competitive performance for English and sets a new state of the art for Danish using only permissible post-training data. Trained on a mixture of 161 datasets, Mimir v1 outperforms the original HRM-Text 1B and competes with larger frontier models like Qwen 3.5 4B and Gemma 4 E2B, tested across 20 benchmarks for English, Math & Code and Danish. The model is available on the Hugging Face Hub: this https URL 

---
# Measuring Task-Agnostic Training Data Influence Across Language Model Pretraining 

**Authors**: Yuto Nishida, Hirokazu Kiyomaru, Yusuke Oda, Takashi Kodama, Chaoran Liu, Daisuke Kawahara, Yusuke Miyao, Max Müller-Eberstein, Masaru Isonuma  

**Link**: [PDF](https://arxiv.org/pdf/2608.13515)  

**Abstract**: Measuring training data influence consistently across language model pretraining is challenging. It is difficult to select downstream tasks or validation sets representative of a model's general capabilities, and reliance on task performance at intermediate checkpoints complicates comparisons across training. We propose a measure of training data influence that does not require selecting a downstream task or validation set as the attribution target. Specifically, we define an example's influence by how much its gradient update reduces the squared distance to the final parameters of a given pretraining run, and estimate this quantity from intermediate checkpoints without retraining. Applying the method to 18 configurations from the Pythia and PolyPythia suites, we find systematic temporal changes in influential data. Early in training, literature-related data are more strongly aligned with the trajectory toward the final parameters, whereas STEM data become more strongly aligned in later stages. This qualitative crossover is broadly consistent across model configurations. Our results provide a tractable trajectory-level view of how influential data change throughout pretraining, complementing influence analyses defined with respect to specific downstream tasks or validation sets. 

---
# Toward a Gricean Retreat: Probing LLMs for Knowledge Boundaries and Referent Specificity 

**Authors**: Dananjay Srinivas, Saksham Khatwani, Maria Pacheco  

**Link**: [PDF](https://arxiv.org/pdf/2608.13484)  

**Abstract**: When asked about entities outside their knowledge boundary, LLMs routinely fabricate plausible-sounding details rather than backing off to safer, more general claims. We frame this failure through a Gricean lens: a cooperative speaker who is uncertain about a referent retreats up the specificity hierarchy, trading informativeness for truthfulness. We ask whether LLMs have the ingredients to perform this retreat. Using a T-REx-based benchmark that varies entity familiarity and referent specificity, we probe models to answer two questions: (i) do their activations encode whether a referent falls inside the knowledge boundary, and (ii) do they anticipate the specificity of the referent they are about to generate? We find that the answer to both is yes, but the two signals are not reconciled in generation. Models overwhelmingly prefer specific referents even when the entity is unknown to them, and do so even when offered correct generic alternatives. The substrate for a Gricean retreat is present, but the policy that would act on it is not. We position our findings as a first step toward Gricean alignment, training or steering objectives that couple knowledge-boundary awareness to referent-specificity during generation. 

---
# Are You Sure You're Sure? On the Impact of Instruction Tuning on Confidence and Lexical Diversity 

**Authors**: Irina Proskurina, Mayank Kumar, Oyindolapo O. Komolafe  

**Link**: [PDF](https://arxiv.org/pdf/2608.13430)  

**Abstract**: Instruction-tuned language models achieve strong performance across a range of generation tasks, but have also recently been shown to exhibit verbalized overconfidence. In question answering, verbalized model overconfidence may be associated with the consistency of the generated supporting rationales. In this paper, we study whether corresponding changes in the lexical diversity of generated answer rationales accompany changes in model confidence induced by instruction tuning. We evaluate three matched base and instruction-tuned models across question-answering benchmarks and find that instruction tuning consistently alters answer confidence, despite limited changes in predictive accuracy and decreases in likelihood-based calibration. Secondly, we observe a non-uniform effect of instruction tuning on rationale diversity: cross-rationale diversity consistently decreases, whereas surface-level lexical diversity varies in both direction and magnitude across models and benchmarks. Finally, we find that these differences persist after controlling for answer selection and rationale length, confirming that confidence and rationale diversity capture distinct effects of instruction tuning. 

---
# Motor, Cognitive, or Corpus? What Survives Cross-Lingual Transfer in Speech-Based Parkinsons Disease Detection 

**Authors**: Serli Kopar, Sam Gijsen, Abner Hernandez, Paula Andrea Perez-Toro, Kerstin Ritter  

**Link**: [PDF](https://arxiv.org/pdf/2608.13425)  

**Abstract**: Self-supervised learning (SSL) speech representations achieve strong performance for Parkinson's disease (PD) detection within individual corpora. However, it remains unclear whether these models capture disease-related characteristics or exploit dataset-specific confounds, particularly since most SSL backbones are pretrained exclusively on healthy speech. To investigate this question, we perform a layer-wise analysis of nine SSL speech backbones using a low-capacity logistic regression probe across three languages. We structure the evaluation as multiple scenarios that progressively introduce distribution shifts in participant identity, recording conditions, language, and pathology. Our results reveal two key findings. First, layer selection is highly corpus-dependent: the optimal representation layer is determined primarily by the source dataset rather than by the SSL architecture itself. Second, the transferred discriminative signal lacks pathological specificity: classifiers trained to detect PD assign similarly high probabilities to both PD and dementia speech in the target corpus. These results highlight critical limitations that must be addressed before speech-based pathology recognition models can be reliably deployed in clinical settings. 

---
# CROP: Task Relevance via Counterfactuals for Selective On-Policy Distillation 

**Authors**: Enhan Li, Junhao He, Hongyang Du  

**Link**: [PDF](https://arxiv.org/pdf/2608.13387)  

**Abstract**: On-policy distillation (OPD) supervises a student language model on trajectories sampled from its current policy, but assigns equal credit to response tokens with unequal supervision value. Selective OPD addresses this limitation by allocating supervision non-uniformly across response tokens according to their estimated training value. Most existing criteria, however, focus primarily on optimization need, such as uncertainty or teacher-student disagreement, while task relevance, namely whether the supervision is tied to the semantic content of the current input, remains less directly characterized as a complementary dimension. To address this gap, we introduce Counterfactual Relevance for On-Policy Distillation (CROP), which operationalizes task relevance through a paraphrase-calibrated counterfactual sensitivity margin. For each source prompt, CROP constructs a validated original-paraphrase-counterfactual triplet, holds the student rollout fixed, and measures each response position by its sensitivity to a task-relevant condition change calibrated by its sensitivity to a meaning-preserving rewrite. Matched selection controls show that CROP identifies more useful supervision positions than random or lowest-relevance selection, while component comparisons confirm the value of both counterfactual sensitivity and paraphrase calibration. Across two teacher-student settings, CROP improves aggregate performance by 1.92 and 2.96 points over the strongest non-CROP selector. These results support task relevance as a complementary criterion for selective OPD and establish CROP as a model-internal, contrast-specific method for allocating token-level supervision. 

---
# RippleMem: From Isolated Retrieval to Associative Recollection for Long-Term Agent Memory 

**Authors**: Jingbo Ji, Lingyi Li, Xilong Cheng, Yuhao Zhou, Wenji Zhang, Yuting Tan, Yunxiao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2608.13334)  

**Abstract**: LLM-based agents increasingly rely on external memory to support long-horizon reasoning and interaction. However, the main bottleneck is not simply storing past experience, but recovering the right set of evidence when relevant information is distributed across many interactions. Existing approaches struggle with this access problem. Full-context methods require noisy long-context search, flat retrieval often returns isolated and incomplete records, and graph-based memory systems can be expensive to construct while compressing rich event context. We introduce RippleMem, a long-term memory system that replaces one-shot retrieval with adaptive associative recollection. Inspired by cue-dependent episodic retrieval and associative completion, RippleMem stores interaction history as cue-rich episodic memory units and organizes them in an event-centric memory graph. Given a query, it first recalls relevant memory anchors through hybrid cues, then expands from these anchors along semantic and structural associations to recover missing supporting evidence. In this way, initially recalled memories serve not only as answer context, but also as cues for completing the evidence needed to answer. Experiments on LoCoMo and LongMemEval-S show that RippleMem achieves the best overall performance across evaluated settings, improving LLM-as-a-Judge accuracy by 3.95% on LoCoMo and up to 11.87% on LongMemEval-S, while reducing graph construction cost by about 30x. 

---
# It's How You Ask: Gender-Associated Linguistic Bias in LLMs 

**Authors**: Katherine Van Koevering, Anjalie Field  

**Link**: [PDF](https://arxiv.org/pdf/2608.13328)  

**Abstract**: Professional communication is increasingly mediated by LLMs - but do these models serve all users equally? We show that when prompts contain linguistic features more commonly used by women (hedges, tag questions, collective reference), they systematically elicit shorter, less sophisticated, and less formal responses across three document types and four models. These effects persist after controlling for prompt complexity and feature carry-over. Explicit gender cues like sign-off names are encoded in the same representational space as linguistic dialect - suggesting shared underlying mechanisms - yet linguistic register is far more influential, producing large, consistent effects where names produce none. Our results further reveal that post-hoc mitigation is challenging: because these patterns are culturally embedded and outside conscious control, users cannot easily avoid them through strategic self-presentation, and mechanistic analysis reveals that linguistic features are encoded in early transformer layers and entangled with other features. Our work calls for upstream consideration of the influences of linguistic variation to mitigate disparate impacts of LLM-mediated workplace communication. 

---
# Beyond Local Accuracy: A Protocol-Level Identifiability Audit for Controlled LLM Reasoning Evaluation 

**Authors**: Junhao Luo, Ning Huang, Ziqi Sha, Wenxuan Tang, Wei Deng  

**Link**: [PDF](https://arxiv.org/pdf/2608.13326)  

**Abstract**: LLM benchmark scores can be precise even when the observation protocol does not identify the behavioral property they are intended to measure. In a controlled, solver-grounded setting, we formalize a protocol-level identifiability audit over a finite behavioral policy class: given policies H, observation support O, and estimand $\tau$, we test whether O separates every pair with different $\tau$. The audit requires zero model calls and resolves our diagnostic case: base-only observation collapses seven frozen deterministic policies into one equivalence class; full support yields seven classes and no cross-estimand collisions; every leave-one-out support retains a constructive collision witness. Empirically, both constrained-generation variants have pair-validity 1.0, yet base accuracy and selective-response fidelity diverge - 0.620 versus 0.324 across six balanced oracle-transition directions (cluster-bootstrap 95% CI [0.600, 0.642] vs. [0.304, 0.345]) - and the gap recurs on a second deterministic source (0.646 vs. 0.331). The audit also synthesizes a minimum identifying support $O^*$ for the frozen policy class: two cells instead of the full 36-cell tensor. This case shows how evaluation-design validity can be checked structurally before model inference and why base correctness does not determine intervention-response fidelity. 

---
# Refusing Intent, Not Form: Wrapper-Based Intent-Group Supervision for LLM Safety 

**Authors**: Ping Wu, Haibo Tong, Feifei Zhao, Han Shen, Yu Shi, Yilin Zhao, Sicheng Shen, Guobin Shen, Yun Luo, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2608.13304)  

**Abstract**: Safety tuning can improve harmful refusal, but models may learn surface-form shortcuts: wrapped harmful prompts bypass safety, while similarly wrapped benign prompts are over-refused. We propose Wrapper-Based Intent-Form Augmentation (WIFA), an automatic intent-group augmentation method that pairs wrapped harmful examples with structurally matched wrapped benign counterexamples, requiring no external teacher or manual per-wrapper intent labels. We use WIFA as a common data layer for two complementary fine-tuning routes: WIFA-Boost, a two-stage high-safety recipe, and Anchored Group-Consistent Refusal Training (A-GCRT), which regularizes refusal/compliance decision scores across same-intent wrappers and anchors harmful and benign groups on opposite sides of a margin. In the Qwen setting, WIFA-Boost reaches the strongest transformed-harmful refusal, while A-GCRT reduces OR-Bench over-refusal from 25.7\% for the base model to 17.4\%; reproduced baselines do not match these operating points. Llama results and ablations over data structure, two-stage order, and A-GCRT components support this intent-group interpretation without claiming universal below-base over-refusal. 

---
# Mixture of Training: Recombining Small-Scale Scaffolded Pretraining Runs into a Larger Language Model 

**Authors**: Mohammed Sabry, Sean Augenstein, Keith Rush, Lucio Dery  

**Link**: [PDF](https://arxiv.org/pdf/2608.13277)  

**Abstract**: We ask whether language-model pre-training can be decomposed into smaller, independently trainable jobs that can later be recomposed into a coherent larger model. We introduce Mixture of Training (MoT), a scaffolded modular pre-training procedure that partitions a target Transformer into contiguous layer blocks, trains each block inside a frozen pretrained aligner scaffold, and then recomposes the trained blocks with an optional short end-to-end adaptation pass. On a 1.3B-parameter Gemma-style model trained on C4, MoT provides a small-scale proof of mechanism: independently trained depth slices can be recomposed into a usable language model, and a quality-parity schedule reaches the same reported perplexity as the monolithic baseline. This parity setting processes more aggregate tokens and has a shorter idealized layer-equivalent critical path after aligner preparation; its effective compute advantage depends on reusing the aligner across runs. We therefore present MoT not as a general replacement for monolithic pre-training, but as a small-scale framework for studying whether scaffolded sub-runs can act as reusable training units. 

---
# How Do VLMs Behave When Blind or Misled? Behavioral Evaluation of VLMs on Scientific Figures 

**Authors**: Paul Osemudiame Oamen, Owusu-Banahene Osei, Ananya Mukherjee, Christian Greisinger, Steffen Eger, Pius Onobhayedo, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.13267)  

**Abstract**: Existing vision-language model (VLM) benchmarks emphasize perception and reasoning accuracy (how well VLMs describe and reason about what they see in an image), with limited attention to behavioral reliability under uncertainty (how they behave when visual evidence is missing or misleading). We introduce SciFigBench, a diagnostic VLM benchmark for scientific figure understanding that jointly evaluates perception, reasoning, and behavioral reliability under uncertainty. It contains 250 figures with high-quality human annotations across three evaluation aspects, totaling 600+ hours of annotation effort. We further extend these figures via image transformations, reasoning questions, resistance probes, caption-bias probes, and confirmed selective-blur targets, producing over 34,000 evaluation setups for stress testing.
We further propose the Admittance-Resistance-Inductance (A-R-I) framework to evaluate whether models acknowledge insufficient evidence, resist misleading context, and infer cautiously from partial information. Our results reveal substantial behavioral differences among models. GPT-5.2 achieves the highest description quality (MQM 91.6) with strong reasoning accuracy (78.4%), yet hallucinates unreadable content in 96% of cases, whereas Gemini 3.1 Pro, a comparably capable model (MQM 90.2, reasoning 81.0%), admits uncertainty in 71% of such cases and achieves the strongest resistance score (0.91). These findings show that high perception and reasoning accuracy alone do not guarantee behavioral reliability, a dimension critical for deployment in scientific workflows. 

---
# Self-Referential Induction Increases Response Instability Relative to Unresolvable and Verifiable Questions in Large Language Models 

**Authors**: Paras Balani, Subhrakanta Panda  

**Link**: [PDF](https://arxiv.org/pdf/2608.13258)  

**Abstract**: Self-referential prompting has been shown to reliably induce large language models to produce first-person reports resembling subjective experience, but no prior work measures how consistent these reports are across repeated, independent trials, or how that consistency compares to the model's behavior on other kinds of open-ended questions. We measure response instability, defined as one minus the mean pairwise cosine similarity of sentence embeddings computed over a compressed core claim extracted from each response, for three groups of questions: self-referential prompts eliciting a subjective-experience report, unresolvable philosophical questions unrelated to self-reference, and questions with a verifiable correct answer. Using 30 independent responses per question (360 responses total, Gemini API, temperature 0.7) across four questions per group, we find that self-referential questions show the highest instability (0.343 +/- 0.047), unresolvable philosophy questions show intermediate and tightly clustered instability (0.192 +/- 0.008), and verifiable questions show the lowest instability (0.105 +/- 0.058). This provides a quantitative baseline for the induced subjective-experience report, showing that it occupies a distinct, less stable position in the model's output distribution than ordinary open-ended philosophical uncertainty. 

---
# Localize, Then Reason: Visual Latent Structural Reasoning for Molecular Properties and Edits 

**Authors**: Xingqiao Lin, Junmei Wang, Haocheng Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.13244)  

**Abstract**: Local chemical perception and property reasoning are both essential for understanding how molecular structure determines properties. Current LLM-based chemical reasoning methods either receive SMILES/molecular images together with descriptions of local motifs, or reason directly from molecular images. Neither approach enables the model to focus on chemically meaningful regions before reasoning. To address this gap, we propose Visual Latent Structural Reasoning (VLSR), an end-to-end framework that jointly learns localization and reasoning from molecular images. Central to our approach is a localize-then-reason strategy. VLSR first learns to locate chemically meaningful regions in a molecular image. It then reasons about their property effects in a compact latent workspace before producing the final answer. Under the same inference setup, this design achieves 9.6X higher throughput than a comparable textual-reasoning baseline. 

---
# GEM: A Generative Embedding Model Bridging Reasoning and Retrieval 

**Authors**: Zhili Shen, Craig Macdonald  

**Link**: [PDF](https://arxiv.org/pdf/2608.13200)  

**Abstract**: Modern LLMs excel at reasoning and instruction following, enabling users to express complex and diverse information needs. However, conventional retrievers largely rely on surface-level matching between queries and documents, resulting in a growing gap between how users express their needs and how retrievers interpret them. In this paper, we present GEM, a generative embedding model that augments retrieval through its own knowledge by explicitly reasoning about user intent and relevance criteria. GEM unifies generation and embedding within a single model: it first reasons over the query, then appends an embedding token to encode the enriched context for retrieval. \zhili{Evaluated on reasoning-intensive and instruction-following retrieval tasks, GEM demonstrates the effectiveness of its reasoning-augmented retrieval, outperforming its non-reasoning variant and matching baselines using substantially larger models.} Furthermore, GEM's generative nature allows test-time compute scaling via prompting to further enhance retrieval performance. Our code is available at: this https URL. 

---
# Which LLM Is Your Ideal Companion? Evaluating Emotional Companion Capabilities of LLMs Based on Adult Attachment Theory 

**Authors**: Junkai Zhou, Shiting Guan, Zhaoyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.13168)  

**Abstract**: As large language models (LLMs) are increasingly applied for emotional companionship, evaluating their behavior and capabilities in intimate relationships has become a pressing issue. However, existing assessments primarily characterize general personality traits, providing limited insight into model behavior within intimate and emotionally sensitive contexts. Therefore, we introduce adult attachment theory into LLM evaluation and use the Experiences in Close Relationships-Revised (ECR-R) scale to characterize attachment anxiety and avoidance. To evaluate emotional companionship capabilities of LLMs in realistic interaction scenarios, we present an emotional companionship benchmark, ECBench, spanning four scenarios including emotional support, collaborative tasks, conflict resolution, and social guidance, across friendship and romantic relationships. ECBench is utilized to assess model behavior using 11 dialogue-quality metrics and three evaluation methods. We evaluate the attachment tendencies of 32 LLMs and select representative models to investigate how these tendencies manifest in contextualized multi-turn interactions and whether they can be shaped through prompting. Our study provides a theoretical lens from psychology, along with practical tools to understand and select LLMs for emotional companionship. 

---
# Better Decomposition, Free Aggregation: A Synthesizer-Folding Framework for Multilingual Multi-Hop Question Answering 

**Authors**: Yilin Wang, Yuchun Fan, Weidong Bao, Zili Wei, Shi Feng, Tong Xiao, Zhengtao Yu, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.13160)  

**Abstract**: Multilingual retrieval-augmented generation (mRAG) equips large language models with access to globally distributed external knowledge for complex multilingual question answering. Recent approaches either translate retrieved documents into English or the query language to bridge the cross-lingual semantic gap, or decompose a complex query into sub-questions and aggregate the intermediate reasoning process. However, both lines of work suffer from two limitations. First, one-size-fits-all translation alignment, blanket translation discards culturally and linguistically native information unique to the target language, introduces translation noise, and inflates system cost. Second, greedy decomposition and aggregation, uncontrolled decomposition produces redundant sub-questions that compound errors during step-wise reasoning, and the final aggregation over reasoning paths further amplifies these errors. We address both with our method Syfer, a synthesizer-folding framework for multilingual multi-hop question answering that defers translation rather than applying it by default. Syfer first invokes a format-constrained decomposer to produce a sub-question graph in the original language, followed by a decomposition-quality check; when the check passes, sub-questions are answered sequentially under a retrieve-then-answer policy in the target language, and the English translation pathway with bilingual sub-question graph alignment is activated only when the check fails. Experiments across multiple languages show that Syfer attains competitive accuracy while striking a favourable balance between performance and computational cost. 

---
# LigBench: A Unified and Human-Aligned Benchmark for LLM-based Research Idea Generation 

**Authors**: Chenrun Wang, Mingxuan Zhu, Tiancheng Huang, Wenjie Li, Yujie Zhang, Zichen Zhu, Zhiying Zou, Kai Yu, Lu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.13136)  

**Abstract**: With the rapid advancement of large language models (LLMs), research idea generation has attracted increasing attention. Existing approaches enable LLMs to retrieve relevant literature and propose novel ideas for research areas. However, current evaluation practices for idea generation remain fragmented and lack objective standards, often relying on direct LLM scoring, which limits their ability to provide unified and reliable assessments across a coherent distribution of generated ideas. To address this challenge, we propose LigBench, an automated evaluation benchmark that enables fine-grained and reliable evaluation of AI research ideas, consistently applicable across different generation distributions. In addition, we introduce PAIR-IQ, a dataset tailored for training pairwise idea judgment models and serving as an auxiliary reference to support more objective comparative evaluation. Extensive experiments demonstrate that LigBench achieves stable and interpretable evaluations, significantly improving alignment with expert judgments. Furthermore, models trained on PAIR-IQ exhibit enhanced ranking accuracy and robustness, establishing a principled standard for scalable and objective research idea assessment. 

---
# CASA: Content-Acoustic Speaking Assessment with Speech Encoder and Large Language Model 

**Authors**: Nhan Phan, Ilona Lähteenmäki, Anna von Zansen, Olli-Pekka Pauna, Yaroslav Getman, Tamás Grósz, Mikko Kurimo  

**Link**: [PDF](https://arxiv.org/pdf/2608.13101)  

**Abstract**: Research on automatic speaking assessment (ASA) has increasingly adopted multimodal speech large language models to assess learners' speaking performance. However, existing studies provide limited analysis of how acoustic and content information contribute to predictions and how stable the resulting performance is. We propose CASA, a simpler architecture combining Whisper-medium and Qwen3.5-2B that achieves state-of-the-art performance while providing a more interpretable separation between speech delivery and content.
On the Speak & Improve Corpus 2025, CASA achieves a root mean square error (RMSE) of 0.358, improving on the previous best RMSE while using approximately half the estimated inference parameters. The general-purpose architecture is designed for adaptation to other ASA corpora without structural changes and relies on three handcrafted fluency features. Through ablations and repeated runs, we analyze the individual and complementary contributions of acoustic and content information, examine performance variability, and demonstrate the potential of large language model reasoning for training-free content validation. 

---
# RAGSieve: Self-Referenced Local Contrast for Knowledge-Poison Detection in Retrieval-Augmented Generation 

**Authors**: Xinlong Xu, Yoshua Y. Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.13010)  

**Abstract**: Retrieval-augmented generation treats an external corpus as inference evidence, allowing injected documents to promote attacker-chosen claims. Existing detectors depend on trusted references, specific attack artifacts, or global thresholds sensitive to corpus topology. We present RAGSieve, a self-referenced detection framework that constructs its reference from the inspected system. RAGSieve-Query (RSQ) performs query-local contrast, scoring top-five candidates against ranks 6-20 of the same retrieval to detect answer-anchor concentration and carrier transitions. RAGSieve-Graph (RSG) performs corpus-local contrast, comparing each document's semantically similar but lexically distinct neighbors with its local baseline to detect coordinated density before queries arrive. Across three QA datasets and six poisoning constructions, RSQ achieves 95.2% AUROC and detects 82.2% of poison at 5% clean-document removal, versus 81.1%/52.5% for GMTP. RSG achieves 93.3%/79.8%, versus 79.4%/37.6% for CleanBase. Joint deployment reduces attack success from 67.4% to 14.0% while retaining 41.3% F1 on unpoisoned retrieval, demonstrating practical protection at both corpus ingestion and query time without poison labels or trusted corpora. Source code is available at this https URL. 

---
# EviReform: Evidence-Guided Query Reformulation for Multi-Hop Graph Retrieval 

**Authors**: Xinlong Xu, Yoshua Y. Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.13006)  

**Abstract**: Multi-hop retrieval must recover passages that provide sufficient evidence together. An initial passage often resolves an entity or relation implicit in the question, making the missing evidence easier to describe only after retrieval begins. Graph retrieval improves access to related evidence through stored corpus structure, but its retrieval signal is commonly derived from the original question. Complementary evidence must then be reached through stored relations even when an observed passage provides a more direct semantic cue. We introduce EviReform, which separates revising the retrieval request from aggregating evidence in the graph. Retrieved source passages formulate residual queries for the unresolved information need. The original and residual retrieval signals are normalized separately, combined, and propagated between propositions that share entities. On 2WikiMultiHopQA, HotpotQA, and MuSiQue, EviReform exceeds the strongest baseline by up to 5.59 Recall@5 points and 4.50 F1 points. These results show that observed evidence can guide graph retrieval toward the part of a supporting chain left underspecified by the original question. Code is available at this https URL. 

---
# HybridRAG-BN: A Retrieval-Augmented Framework with Fine-Tuned Verification for Bangla KBQA 

**Authors**: Rathijit Aich, Nirjhar Das, Mahfuzulhoq Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2608.13004)  

**Abstract**: Knowledge-base question answering (KBQA) systems rely on effective retrieval and reasoning mechanisms to generate accurate answers from external knowledge sources. However, developing reliable KBQA systems for low-resource languages such as Bangla remains challenging due to limited retrieval-focused research, scarce language resources, and difficulties in grounding generated responses in external knowledge. In this work, we propose HybridRAG-BN, a retrieval-augmented framework for Bangla KBQA that integrates hybrid retrieval using BM25 and BGE-M3, answer generation using the GGUF version of Gemma-4-31B-Instruct, and a LoRA-fine-tuned Gemma-4-31B-Instruct model for answer verification and refinement. To further improve robustness, the framework incorporates a post-processing stage that addresses unresolved cases through fallback answer replacement and DuckDuckGo-assisted retrieval. Experimental results demonstrate the effectiveness of the proposed framework, achieving token-level F1 scores of 0.71654 and 0.72912 on the public and private leaderboards, respectively, securing first place in the competition. 

---
# LycheeMemory V2: Efficient Long-Term Memory for LLM Agents via Semantic Segment-Level Consolidation 

**Authors**: Dongfang Li, Zixuan Liu, Junmai Wang, Jiahe Huang, Fuhao Li, Bonian Jia, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.12990)  

**Abstract**: Long-horizon LLM agents must preserve information from past interactions to support future tasks. Existing memory systems typically rely on eager consolidation, invoking LLMs after each interaction to extract, summarize, or update memories. This design makes memory construction increasingly costly as conversations grow. Coarse summarization can reduce construction cost but risks discarding fine-grained contextual evidence, whereas larger retrieval contexts or multi-hop LLM reasoning shift the overhead to query time. We present LycheeMemory V2, an efficient long-term memory framework that replaces turn-level consolidation with semantic segment-level consolidation. Instead of consolidating every interaction, LycheeMemory batches multiple exchanges into segments and encodes each finalized segment into context-independent typed memory records. Segment-level batching lowers LLM encoding frequency, while semantic boundary detection helps preserve coherent event-level and temporal evidence compared with fixed-window batching. The resulting records are organized with lightweight structured indexes for query-planned evidence retrieval. Experiments using GPT-4.1-Mini show that LycheeMemory achieves state-of-the-art performance, reaching 89.22% on LoCoMo and 92.20% on LongMemEval-S. Compared with A-Mem, it reduces construction tokens by 86.0% on LoCoMo and 75.9% on LongMemEval-S without increasing query-time token usage. More broadly, our results suggest that the accuracy--cost trade-off of long-term agent memory depends not only on what information is retained, but also on the granularity at which it is consolidated. 

---
# Unifying Depth and Width Pruning for LLMs via Binary Knapsack Optimization 

**Authors**: Palaash Goel, Ayan Sengupta, Akshay Nambi, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2608.12953)  

**Abstract**: Structured pruning is a promising approach for compressing large language models (LLMs), yet existing methods rely heavily on greedy heuristics that produce myopic decisions, and often fail to precisely meet target compression budgets. We present SNIPER, a two-stage structured pruning framework that solves a knapsack optimization over coarse-granularity components to yield conditionally optimal parameter allocations with respect to fixed importance estimates, followed by a fine-grained pruning stage to meet strict budget constraints. We introduce the Compression Ratio Adherence Factor (CRAFT) to quantify budget fidelity, showing that while existing pruners deviate from target compression ratios by up to 33%, SNIPER achieves near-exact adherence with a CRAFT score of 0.98. Evaluations across four diverse architectures over a set of 18 tasks spanning five domains demonstrate SNIPER's consistent improvements in average performance retention and task-level stability over six state-of-the-art pruners. Across all pruning configurations, SNIPER achieves an excellent mean rank of 1.25, indicating its robust cross-architectural generalizability and excellent reliability. 

---
# Decoupled Contrastive Decoding via Expert-Aligned Drafting 

**Authors**: Zhixuan Liu, Zhichen Dong, Yuanfu Wang, Chao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.12913)  

**Abstract**: Contrastive Decoding (CD) improves generation quality, but its amateur-model pass makes decoding expensive. Accelerating CD with speculative decoding raises a proposal-alignment question: should the contrastive signal shape the drafter, or should it remain only in verification? We study this question in the lightweight feature-level drafter regime. Two controlled diagnostics, matched Cross-alpha training and an Approximate Dual-Drafter decomposition, give the same diagnosis: contrastive-aware drafting does not consistently improve over expert-aligned drafting because the contrastive correction is usually weaker than drafter error, and reconstruction can amplify that error. We introduce Decoupled Contrastive Decoding (DCD), which drafts with an expert-aligned lightweight proposer and applies the amateur only in unchanged CD verification. Standard speculative verification preserves the vanilla-CD output distribution. Across the main 8B settings, EAGLE3-based DCD achieves average greedy speedups of 1.65 to 1.95x over vanilla CD and reduces MMLU proposal-path latency by about 5 to 12x relative to amateur-coupled proposal paths. 

---
# Prompts in the Wild: A Large Analyzed Collection of Transactional Prompts in Code 

**Authors**: Victoria Basmov, Yoav Goldberg, Reut Tsarfaty  

**Link**: [PDF](https://arxiv.org/pdf/2608.12905)  

**Abstract**: The behavior of contemporary generative Large Language Models (LLMs) is directly shaped by prompts, unstructured texts that describe the desired output and model behavior. In this paper we argue that prompts are linguistic objects that merit investigation in their own right. To this end, we collect 57.5K unique samples of prompts from GitHub. Specifically, we focus on transactional prompts: reproducible natural language instructions that are integrated into software. To enable the empirical, quantitative study of prompts, we introduce a structured ontology, capturing the properties of prompts as well as their formal and semantic components. Based on this ontology, we transform prompts from unstructured raw texts into richly structured linguistic objects. Analysis of these structured data reveals significant diversity of usage patterns across languages, domains, tasks, and modalities, in a typical Zipf-like distribution where some clearly prevail and others, more diverse, appear in the long tail. To validate the reliability of the ontology-based annotation of the prompts, we perform a comprehensive error analysis across all fields, providing a detailed assessment of annotation quality. We release the dataset together with a browsing and exploration interface (this https URL ). 

---
# BavGround: A Benchmark for Regional Cultural Grounding and Dialect Competence in Bavarian 

**Authors**: Jophin John, Michael Hoffmann, Jan Fillies, Michael A. Hedderich, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2608.12894)  

**Abstract**: Cultural evaluation of large language models (LLMs) often focuses on high-resource standard languages, leaving regional culture and dialect communities underrepresented. We introduce BavGround, a benchmark for evaluating Bavarian regional cultural grounding and dialect competence across English, German and Bavarian. BavGround contains 206 multiple-choice source questions across eight cultural domains per language, yielding 618 multi-parallel instances, with items covering both broadly accessible cultural knowledge and source-grounded regional knowledge from journalism, historical sources, and specialist literature. We evaluate fifteen 7B-10B open-weight instruction-tuned models and one closed-model reference. Strong multilingual models perform best overall, but performance drops on Bavarian items and source-grounded questions, indicating persistent difficulty with dialectal and localized cultural knowledge. We further show that conclusions depend strongly on evaluation protocol: raw answer-letter scoring, shuffled-letter scoring, option-text likelihood, generated-answer parsing, and semantic matching can produce different absolute scores and rankings, especially for regionally adapted models. Finally, an exploratory analysis of GENBA-10B checkpoints suggests that continued pretraining improves answer-content likelihood unevenly across domains, while dialect competence remains comparatively weak. BavGround supports localized, protocol-aware evaluation of cultural representation in LLMs. 

---
# When Your Agent Opens the Chat App: Agent-Controlled Search over Raw Chat Logs Rivals Structured Memory 

**Authors**: Ruizhe Li, Licheng Zhang, Benfeng Xu, Mingxuan Du, Zheren Fu, Weidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.12888)  

**Abstract**: Agent-memory systems increasingly buy retrieval quality with structure, transforming raw conversation histories into summaries, embeddings, trees, or knowledge graphs before any question is asked. We ask how much of that benefit comes from the structure itself, rather than from competent retrieval over the raw history. We present ReFind, an agent-controlled search interface that builds no semantic structure at all: it leaves the conversation archive unmodified, indexes it lexically at turn granularity, and combines a generic iterative keyword-search loop with four chat-native controls grounded in empirical refinding work: session-aware rank fusion, local context expansion, temporal narrowing, and skipping already-inspected sessions. A separate reasoning stage answers from the collected evidence. Across a broad suite of conversational-memory tasks (single- and multi-hop QA, event ordering, and fact consolidation), roughly 2,800 questions on precise-retrieval and fact-tracking capabilities evaluated under the incremental multi-turn setting of MemoryAgentBench, ReFind attains the highest mean accuracy (58.2) of any system compared, above the strongest graph- and tree-based memory systems (HippoRAG 2, 53.2), all under a GPT-4o-mini backbone matched to every reused baseline. Controlled comparisons to single-shot BM25, a matched generic-agentic BM25 control, component removals, and agentic dense/hybrid variants separately support the roles of agent control, chat-native controls, and lexical retrieval. On LongMemEval-S/M, the same interface reaches 93.2 +/- 3.3 and 89.3 +/- 6.0 with GPT-5-mini. The results indicate that for precise, evidence-grounded questions over chat archives, much of the benefit credited to elaborate memory structures is recoverable by giving an agent controllable search over the unmodified record, with no LLM-based index construction at all. 

---
# The Embedder's Dilemma: LLMs Are Better, but at What Cost? 

**Authors**: Adnan El Assadi, Niklas Muennighoff, Jinhyuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.12875)  

**Abstract**: Should you replace your text-embedding pipeline with a large language model? We answer this with a controlled, cost-aware comparison of ten LLMs across six families and 26 embedding models (118M to 14B parameters) on 37 tasks spanning classification, semantic textual similarity (STS), clustering, pair classification, and retrieval. In aggregate the two paradigms are effectively tied: the best LLM (Gemini 3.1 Pro, 77.6) and the best embedding model (77.2) differ by 0.4 points. Their strengths differ by task: LLMs lead on reasoning-heavy retrieval, embedding models lead on classification, and the two match on clustering, STS, and pair classification. Reaching that parity is expensive. An LLM costs up to 1,431x more than an embedding model of comparable quality (USD 154 vs. USD 0.11 per benchmark pass), and the open LLMs tested process tokens 2.5 to 736x more slowly on the same GPU. Reasoning tokens account for 28 to 81% of LLM inference cost; lower reasoning budgets preserve or improve retrieval quality for most models in our ablation. The Pareto frontier contains the leading embedding models and one LLM, Gemini 3.1 Pro. These results support a division of labour: use embedding models for similarity, classification, and clustering, and reserve LLMs for reasoning-intensive retrieval. Our code, datasets, and results are publicly available at this https URL. 

---
# Falsehood and Impossibility Are Different Directions in an AI's Representation of Language 

**Authors**: Yoon Pyo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.12852)  

**Abstract**: Language can describe states of affairs that are false and states of affairs that could not be the case at all. Whether an AI model internally distinguishes these failures remains unclear. I report an exploratory activation study of the multimodal open-weight model Gemma 3 4B IT using 85 prompts from 17 philosophical families and a topic-matched modality set of 15 topics, each expressed as a truth, contingent falsehood, improbable claim, semantic anomaly, and necessary falsehood. In its answers, the model conflates contingent falsehood with contradiction, labeling 12 of 15 false statements "contradiction." Its activations show a different pattern. A linear truth probe separates impossible from true statements (AUC 0.93) but not impossible from false statements (AUC 0.20). An impossibility probe evaluated on held-out topic families separates necessary from contingent falsehood at AUC 1.00, peaking at layer 15 with balanced accuracy 0.97 (Bonferroni-adjusted P=0.018). The truth and impossibility directions are close to orthogonal, whereas the impossibility direction partially overlaps a semantic anomaly direction while remaining distinguishable from it. Sparse autoencoder features at the same layer repeat this geometry. Features selective for impossibility also fire on anomalous sentences but rarely on contingent falsehoods. In this model's activation space, necessary falsehoods are not extreme cases of contingent falsehood but lie closer to the experimentally defined category of semantic anomaly. This representational proximity does not imply that impossible statements are intrinsically meaningless. These correlational observations from one small model offer an empirical footnote to an old philosophical distinction. 

---
# AQuA: Recursively Self-Improving Quantitative Trading Research Agents 

**Authors**: Jiacheng Guo, Suozhi Huang, Yunlong Gao, Zihao Li, Jian Ge, Xu Kuang, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.12841)  

**Abstract**: We study recursive self-improvement at the level of quantitative-investment research: whether an autonomous system can use evidence from earlier experiments to improve the hypotheses and candidates proposed in later iterations. We present AQuA, which comprises two separate language-model-driven research systems: one for symbolic factor discovery and one for trainable model development. The two systems do not share agents, memories, candidate spaces, or research state. Instead, each independently closes its own research loop by retaining validated evidence and using it to guide subsequent proposals. In this bounded sense, both systems implement recursive self-improvement at the level of the research process. Each system also uses its own sealed sandbox, which fixes the data splits, feature and label definitions, and evaluator while allowing the model to act only through constrained factor expressions or configuration diffs. The factor system, a manager-mediated multi-agent pipeline, discovers and combines factors into a signal that reaches a combined information coefficient of about $0.190$ on a crypto universe. The model system, a config-driven loop over a hybrid time-series architecture, reaches a per-stock information coefficient of $+0.0843$ on US equities and converts it into a threshold long/short strategy with a held-out Sharpe of up to $+2.50$ at a two-leg cost. The strategy is positive in every year from 2021 to 2025. 

---
# From Atomic Evidence to Logical Composition: Structured Compositional Reasoning over Compound Answer Options 

**Authors**: Obed Junias, Maria Leonor Pacheco  

**Link**: [PDF](https://arxiv.org/pdf/2608.12836)  

**Abstract**: Large language models often fail when answer options require combining atomic judgments under explicit logical operators, even when they judge the individual atoms correctly. We study compound options connected by AND, OR, and NEITHER/NOR, introducing a framework that decomposes each option into atomic answers and scores contrastive hypotheses about each one, so the model never sees a compound option. An operator-constrained integer linear program then composes the calibrated scores into a single prediction. We evaluate on LOGICAL-COMMONSENSEQA and introduce LOGICAL-SATA, a reading-comprehension benchmark derived from SATA-Bench. Our framework improves Macro-F1 from 48.3 to 77.0 on the human-validated LOGICAL-COMMONSENSEQA split and from 47.0 to 75.6 on LOGICAL-SATA, with the largest gains on NEITHER/NOR. 

---
# FastThaiG2P: Lightning-fast Thai Grapheme-to-phoneme Conversion for Voice Agent Pipelines 

**Authors**: Charin Polpanumas  

**Link**: [PDF](https://arxiv.org/pdf/2608.12814)  

**Abstract**: FastThaiG2P provides sub-millisecond Thai grapheme-to-phoneme conversion for text-to-speech pipelines (International Phonetic Alphabet and Kokoro-TTS conventions) using a PyThaiNLP-tokenized, extensible dictionary and normalization rules for common Central Thai speech. The approach achieves an average latency of 0.15 ms per utterance on a benchmark of 27,242 synthetically generated utterances, of which 30\% is spent on tokenization, 12\% on normalization, and 58\% on out-of-vocabulary fallbacks (0.5\% OOV rate). To demonstrate its effectiveness, we used FastThaiG2P to phonemize Som-TTS, an open dataset containing 20 hours of grapheme-and-audio pairs, then trained an 82M-parameter StyleTTS 2 model based on a Kokoro-TTS recipe. The resulting model vocalizes intelligible Thai speech suitable for prototyping and development at 0.25 real-time factor (4x real-time) with ONNX inference on CPU. 

---
# CRAFT: LLM-Based Iterative Refinement for Temporal Reasoning over Clinical Narratives 

**Authors**: Chengyang He, Tahreem Arif, Marko Zivkovic, Lijing Wang, Yue Ning, Ping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.12779)  

**Abstract**: Understanding the temporal progression of symptoms in clinical narratives is critical for disease monitoring, safety surveillance, and causality assessment. Clinical narratives, however, rarely provide explicit temporal anchors. Current approaches to temporal information reasoning focus predominantly on pairwise relation classification across multi-visit and timestamp-rich records, leaving the reconstruction of structured symptom trajectories from individual anchor-sparse reports largely unaddressed. We propose CRAFT, an LLM framework that pairs a generator with a constraint-based verifier to iteratively produce and refine stage-wise symptom timelines through targeted feedback. We conduct evaluation on MedTempo, a new benchmark of 5,347 vaccine adverse-event narratives spanning three COVID-19 vaccine types, with expert-validated temporal stage annotations for 3,166 reports. Experiments across four LLM backbones demonstrate that CRAFT consistently improves temporal ordering accuracy, with ablation analysis isolating the contribution of generator and verifier components across model capability levels. 

---
# ViTOED: A Dataset for Target-Oriented Emotion Detection on Vietnamese Social Media Texts 

**Authors**: Chanh Vo, Son T. Luu, Ngan Luu-Thuy Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2608.12776)  

**Abstract**: This paper introduces ViTOED, a novel dataset for target-oriented emotion detection in Vietnamese social media texts. The ViTOED comprises 10,985 user comments and 21,244 manually annotated opinion quadruples (source, target, expression, polarity) that follow strict guidelines. The dataset reveals Vietnamese-specific phenomena, such as implicit sources and targets and vocabulary ambiguities, enabling deeper analysis of user emotions toward entities. We propose a baseline using structured sentiment graphs and evaluate various Vietnamese pre-trained language models. The empirical results highlight challenges in span detection and relation extraction and indicate substantial room for model improvement in Vietnamese Target-Oriented Emotion Detection tasks. 

---
# ReconSpan: Reconstruction-Guided Adaptive Latent Tokenization 

**Authors**: Lixing Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.12756)  

**Abstract**: Adaptive latent tokenization maps a fine-grained input to a shorter sequence of continuous representations associated with input-dependent spans. We introduce ReconSpan, which divides text into chunks that a backward decoder can reconstruct from a single contextual prefix code and retains one such code as the latent token for each chunk. The reconstruction criterion is applied when chunks are formed, allowing one trained autoencoder to produce average chunk lengths from 6.5 to 12.2. At matched average length, reconstruction-guided boundaries preserve more text than random boundaries. Readers of the resulting latent sequence recover topic information reliably but struggle to extract exact details. 

---
# PatientAct: Theory-Grounded Mental Health Client Simulation 

**Authors**: Sahand Sabour, TszYam NG, Yaqian Chen, Guanqun Bi, Jialu Zhao, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.12750)  

**Abstract**: LLM-based simulated clients are increasingly used to train novice counselors, evaluate LLM therapists, and generate synthetic data. However, current simulators produce overly cooperative clients that disclose too readily, accept therapeutic reframes without resistance, and resolve core issues within a single session. We trace these issues to profiles that lack causal depth and behavioral mechanisms that treat all content as equally accessible. We present PatientAct, a framework for client simulation grounded in established clinical theories. Our profiles integrate the 5Ps clinical case formulation, providing causal depth without tying the design to any single therapeutic modality. During simulation, profiles include a dynamic memory layer in which items carry trust thresholds (e.g., symptoms are available early, whereas formative memories require a sustained therapeutic alliance). At each turn, the client's emotional reaction and behavior are modeled before generating a response. If the therapist approaches gated content, PatientAct expresses resistance in terms of quantity, content, and style rather than defaulting to cooperation or a single resistance pattern. We evaluate our framework on 40 clinical situations and demonstrate that it generates diverse profiles with high clinical plausibility. Moreover, PatientAct significantly outperforms the baselines, yielding substantial gains in resistance quality and behavioral realism. Our code and data will be publicly available via this http URL. 

---
# ERSkill: Evolving for Skill-Guided Adaptive Memory Retrieval 

**Authors**: Haolong Chen, Liang Zhang, Zhuo Li, Lei Xue, Guanrxu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.12720)  

**Abstract**: While Large Language Model (LLM) agents increasingly rely on long-term memory for persistent interactions, the retrieval mechanisms governing this memory are rarely treated as evolvable components. This static approach limits performance on heterogeneous memory queries, which often demand diverse evidence construction strategies. To address this, we introduce \textbf{ERSkill}, a retrieval-centric framework for self-evolving, skill-guided memory access. ERSkill compiles interaction histories into a structured memory store and represents retrieval behaviors as executable skills composed of fundamental primitives. At inference time, a trained router dynamically matches each query to the optimal skill to construct tailored evidence for answer generation. To enable continuous improvement, ERSkill co-evolves the skill set and the router during training. It employs an experience trie to efficiently record explored retrieval paths, alongside a double-frontier mechanism that safely decouples the expansion of new skill capabilities from stable, router-facing deployment. Experiments across multiple agent memory benchmarks demonstrate that ERSkill substantially outperforms strong non-evolving and self-evolving baselines. Notably, it improves the overall average across F1, BLEU-1, and LLM-judge scores by 31.3\% with Qwen3-Next-80B-A3B-Instruct and by 28.1\% with GPT-5.4-nano. 

---
# Excess Separability: Nuisance-Controlled Residual-Stream Probing for Benchmark Contamination Detection 

**Authors**: Florian Braun  

**Link**: [PDF](https://arxiv.org/pdf/2608.12652)  

**Abstract**: Benchmark contamination is diagnosed today with n-gram overlap, with likelihood-based membership inference, or with canary strings, and each needs something usually unavailable: the training corpus, a well-chosen test statistic, or foresight at dataset release. A recent alternative reads contamination off a linear probe on internal activations. We show that the natural way to do this does not work, and specify one that survives measurement.
The protocol reports a zero-sum contrast on the depth profile of probe accuracy, recentred on a level-matched placebo baseline, tested against a label-permutation null, with the reference set twice the size of the suspect set. Each choice replaces a simpler alternative we measured and rejected. Reporting the level of excess separability rather than its shape makes the false positive rate track the size of the analyst's own control set, from 0.03 to 0.99 under a true null. Contrasting against a flat depth profile fails in both directions, rejecting a true null 0.72 of the time when surface decodability rises with depth and losing all power when it falls. An item bootstrap holds the fitted probe fixed and rejects up to 0.09 of the time where a permutation null that refits it holds 0.02. A half-size baseline triples the error rate.
On real transformers, baseline depth profiles are measurably not flat, spanning up to 29.1 accuracy points on a temporal split, and their non-flatness tracks the surface difference between the item sets (correlation 0.87 over 6 audits), so the correction is largest exactly where it is needed. All 4 well-matched Pile arms return null, and the protocol refuses a verdict on the temporal split rather than reporting one. What this does not establish is whether transformers carry a familiarity direction at all: the only positive sits on the split where exchangeability fails. Implementation, tests and audits are released. 

---
# Novels generated by language models show compressed formal variation 

**Authors**: Mehdy Sedaghat Payam, Justin Quinn  

**Link**: [PDF](https://arxiv.org/pdf/2608.12630)  

**Abstract**: While large language models can generate entire novels, there is little information about the level of formal variation in their output over many generations. Rather than asking whether individual passages can be identified as AI-generated, this study asks whether repeated AI generation can produce the same range of diversity which is found across human corpora. This paper contrasts six corpora based on generation source and target style: twenty novels generated using GPT-5.5 Thinking in a nineteenth-century British realist style, twenty novels generated using Qwen3-14B in a nineteenth-century British realist style, twenty novels generated using each of these models in a contemporary zero style, 205 nineteenth-century human-written British novels, and sixty-five contemporary human-written Zero-Style novels. At the document level, the research includes MATTR-500, Shannon entropy, average sentence length, readability, and punctuation rate measurements. The most robust and reliable result is compression of sentence structure. Repeated generations produce novels that vary far less from one another in sentence structure than human novels do. Compression is also present in the measures of readability, punctuation, and sentence length variability within novels. Lexical measures tend to be similarly compressed, with the exception of Qwen Zero-Style MATTR. Despite having distinct mean stylistic profiles, GPT and Qwen lack a stable pattern of cross-measure correlation. This article therefore distinguishes between variance overclosure, which represents a limited formal range between novels, and a more specific phenomenon of correlational overclosure. This means that an individual AI-generated novel may resemble human fiction stylistically, while a collection of AI-generated novels occupies a much narrower formal range. 

---
# LLMs Are Not Good Strategists, Yet Memory-Enhanced Agency Boosts Reasoning 

**Authors**: Yi Wu, Zhimin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.12626)  

**Abstract**: Strategic reasoning in Large Language Models (LLMs) within long-horizon environments is often limited by inconsistent subgoals. In these settings, finite attention resources prevent the model from maintaining strategic coherence over thousands of steps. This limitation leads to strategic drift, where localized decisions fail to sustain a coherent trajectory across reasoning. To address this, we introduce EpicStar, a framework that enables agents to learn memory as policy to tackle long-horizon reasoning. Specifically, the agent maintains a bank of successful past episodes as a heuristic alongside a working memory to track short-term environmental changes. During inference, a dynamic gating mechanism determines whether to execute a retrieved action directly or to perform new reasoning through a contextual fusion of the retrieved episodes and current working memory. Utilizing StarCraft II as the testbed, we evaluated EpicStar against diverse opponent styles. It significantly outperforms baseline methods, achieving higher win rates while consuming an order of magnitude fewer tokens, and it maintains this advantage consistently across difficulty levels and opponent strategies. Our findings provide compelling evidence that structured cross-episode memory is essential for enabling LLM agents to perform robust, long-term strategic execution in dynamic, autonomous settings. 

---
# When Explanations Betray Backdoors: Black-Box Auditing for Language Model Classifiers 

**Authors**: Yang Liu, Ran Zou  

**Link**: [PDF](https://arxiv.org/pdf/2608.12623)  

**Abstract**: Language model classifiers with explanations are used for moderation, routing, topic triage, and low-resource annotation. We study black-box auditing when the defender has only clean calibration data without trigger information but can ask the classifier for a label plus a short rationale or quoted evidence. We introduce Groundedness Drift, a lightweight score measuring whether the answer summary remains grounded in the input. Across two 7B backbones, five datasets, and four common non-adaptive OpenBackdoor-style attack families, Groundedness Drift achieves higher AUROC and lower residual target ASR than every compared detector in all cases at a nominal 5\% clean-FPR budget. We then evaluate Unsupported Groundedness, a multi-probe escalation for explanation-camouflage stress cases. Unsupported Groundedness improves signals but does not close the adaptive gap. 

---
# Intensional Anaphora 

**Authors**: Ezra Keshet, Steven Abney  

**Link**: [PDF](https://arxiv.org/pdf/2608.12598)  

**Abstract**: Intensional operators are often treated as quantifiers over possible worlds, parallel to the treatment of determiners as quantifiers over individuals. Yet individuals introduced in intensional contexts cannot serve as antecedents to later pronouns as easily as those introduced in merely quantificational contexts. For instance, "Everyone is eating a cheeseburger" may be followed by "They are large", where "they" refers to the cheeseburgers being eaten. However, as Stone (1999) points out, the similar "Andrea might be eating a cheeseburger" does not support later anaphoric references such as "It is large" or "They are large". Stone (1999), Stone and Hardt (1999), and Brasoveanu (2010) address this by requiring a pronoun's value (its referents) to exist in the world of evaluation, ruling out anaphora from non-veridical intensional contexts. We show, however, both cases where such anaphora is disallowed even when the pronoun's referents clearly exist and cases where it is allowed even though they might not exist. We argue that intensional anaphora is best captured using a description-based rather than value-based account. A pronoun presupposes that its corresponding antecedent description is instantiated in each world of the context set. Thus, there must be a cheeseburger being eaten by Andrea in every candidate world for "It is large" to be felicitous after "Andrea might be eating a cheeseburger". We implement our proposal via a new logic, building on Keshet (2018) and Abney and Keshet (2022), called Plural Intensional Presuppositional predicate calculus (PIP). Each PIP formula translates directly into standard first-order predicate calculus with set abstraction, providing a classical foundation for this work. 

---
# DIVE: Unlocking Self-Improvement in Frozen Language Models Through Diversity-Driven Skill Evolution 

**Authors**: Siheng Xiong, Ali Payani, Oguzhan Gungordu, Faramarz Fekri  

**Link**: [PDF](https://arxiv.org/pdf/2608.12486)  

**Abstract**: Large language models (LLMs) cannot retain post-deployment experience without parameter updates. We introduce DIVE, a diversity-driven framework that enables frozen LLMs to improve by evolving persistent natural-language skills from task experience and verifier feedback. These skills encode reusable reasoning procedures, verification strategies, common failure modes, and output constraints and are both executed and revised by the same underlying model without access to a teacher model. Since natural-language skill evolution is a stochastic, non-convex search process, optimizing a single skill trajectory can overfit to sampled experience or converge to a suboptimal solution. DIVE mitigates this optimization variance by independently evolving multiple skill populations from bootstrapped experience, adaptively refining them through diverse transformations, and jointly selecting a complementary set of skills. Across six mathematical and logical reasoning tasks and multiple model families, DIVE consistently outperforms existing reasoning methods, prompt-optimization approaches, skill-development frameworks, and memory-based baselines. It achieves rapid self-improvement from accumulated experience, obtaining substantially larger performance gains with fewer rollouts than parameter-based methods such as SFT and GRPO, and prompt optimization with GEPA. Further, the resulting skills transfer across model scales and families, enabling smaller models such as GPT-5-nano to match or outperform larger counterparts, i.e., GPT-5, under conventional prompting. These results establish diversity-driven skill evolution as an effective, interpretable, and parameter-free approach to LLM self-improvement. 

---
# Unified Multi-Dimensional Benchmark for Complex Graph Reasoning in Large Language Models 

**Authors**: Fali Wang, Ali Al-Lawati, Iliyas Bektas, Jinxuan Fang, Alek Melenski, Tianxiang Zhao, Yao Ma, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.12391)  

**Abstract**: Graph reasoning provides a promising testbed for evaluating the reasoning ability of large language models (LLMs), as graph instances can be programmatically generated, structurally controlled, and naturally scaled to long-input settings. However, existing graph reasoning benchmarks have limited coverage of data complexity, rely heavily on manual construction, and lack unified evaluation across text-based and code-based reasoning modes. To address these limitations, we propose {\dataset}, a five-stage \textit{semi-automatic} framework for constructing complex graph reasoning benchmarks. It expands benchmark coverage along five dimensions: \textit{Graph Size}, \textit{Task Complexity}, \textit{Task Description}, \textit{Graph Loading}, and \textit{Task Source}. The framework uses an LLM-based data generator to automatically produce task descriptions, graph data, reference solutions, graph-loading scripts, question forms, and evaluation scripts, while retaining human validation at key quality-control stages. Based on it, we construct a benchmark with $202$ tasks and evaluate LLMs under text-based, code-based, and augmented reasoning settings. Experiments show that the complexity dimensions reveal model limitations that are less visible in existing benchmarks; existing fine-tuned models struggle to generalize to GraphGym, whereas retrieval-augmented methods show scenario-dependent adaptability, improving textual reasoning but not consistently improving coding reasoning. These findings suggest that ours serves as a challenging and diagnostic benchmark for graph reasoning and provides empirical guidance for future enhancement methods. Code and dataset will be published soon. 

---
# Query Timing Produces Opposite Positional Biases Between LLMs and Humans 

**Authors**: Jasin Cekinmez, Addison J. Wu, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2608.12387)  

**Abstract**: Positional biases such as recency and primacy effects have been documented in large language models (LLMs), yet the underlying mechanism by which these models make their evaluations remains poorly understood. Both primacy and recency biases have been observed in human judgments in response to evidence, but recent work suggest that \emph{when} the listener updates their beliefs -- during the presentation of evidence or only at the end -- influences the presence of such effects. We investigate whether a similar phenomenon holds for LLMs, finding divergence from human behavior. These biases are more exacerbated in newer models compared to their predecessors. 

---
# Are you Talking Logic to Me? Assessing Language Models Syllogistic Reasoning Capabilities 

**Authors**: Hanna Abi Akl, Fabien Gandon, Catherine Faron, Pierre Monnin  

**Link**: [PDF](https://arxiv.org/pdf/2608.12374)  

**Abstract**: Language models (LMs) struggle with logical tasks like reasoning on syllogisms. It has been shown that Knowledge Representation (KR) plays a crucial role in expressing input information to help models solve tasks. This observation motivates our study of the impact of different formal KR notations on syllogistic reasoning by extending the FOLIO and P-FOLIO datasets. Our experiments on Small Language Models (SLMs) in Supervised Fine-Tuning (SFT) and Zero-Shot (ZS) settings show that the choice of input notation can yield performances competitive with natural language while enabling faster inference. We also propose a syllogistic categorization method (SEF) and use it to enrich ZS prompts with logical definitions, which boost reasoning in small models. We open-source our framework, Common Logic Grammar Construction (CLGC), as the first Python library for automatically generating syllogisms in KR notations and defining their SEF categories. 

---
# New Terms, New Toxicity: Consensus-based Chinese Neologism Toxicity Detection via Search-Augmented LLMs 

**Authors**: Shiyao Cui, QingLin Zhang, Di Wang, Yida Lu, Zhexin Zhang, Jinhua Gao, Jinglin Yang, Min He, Han Qiu, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.12361)  

**Abstract**: Neologisms, emerging terms in meaning or form, can serve as new vehicles for toxic expression, like "country girl" as a stigmatizing label targeting feminism. Such toxic neologisms appear benign but have evolved into toxic usage in public consensus, posing challenges to moderation systems and remaining underexplored. In this paper, we investigate how to detect implicit toxicity expressed via neologisms. We first propose a taxonomy that captures the origins and consensus-verification criteria of toxic neologisms, followed by the construction of a lexicon spanning widely observed risk categories. To capture toxicity grounded in public consensus, we introduce SeTox, a search-augmented framework that enables static large language models (LLMs) to incorporate real-time web context for neologism toxicity detection. Experiments show that SeTox, even with 3B-scale models, outperforms recent large-scale models, demonstrating its scalability to incorporate real-world knowledge for toxic neologism detection. Disclaimer: this paper has offensive contents that may be disturbing to some readers. 

---
# Predicting consumer-technology ownership without a diffusion history 

**Authors**: Irina Vartanova, Niels Selling, Jennifer Viberg Johansson, Pontus Strimling  

**Link**: [PDF](https://arxiv.org/pdf/2608.12344)  

**Abstract**: We test whether the perceived attributes of a consumer technology predict how widely it is owned. In a 2022 Prolific survey of US adults (n = 678), respondents rated 65 consumer technologies on six attributes. We then elicited the same ratings from two frontier language models, Anthropic Claude Opus 4.7 and OpenAI GPT-5.5. We regress ownership prevalence on four UTAUT2 acceptance attributes plus a log-age covariate with a sign-constrained penalized regression and evaluate it by holding out one technology at a time. The attribute model improves on a baseline of years-since-launch: mean absolute error falls by 17% with the human ratings, and by more with either model, most with Opus 4.7. Over the short 2022-to-2025 window, where ownership moved little, the same attributes do not improve on a no-change baseline. We set out the limitations of the approach, including the possibility that language-model ratings reflect prior knowledge of these technologies rather than independent attribute reasoning. We include a deployment illustration: 2027 ownership predictions for eleven products launched in 2025 and 2026. 

---
# Large Language Models Pass the History Exam But Miss the <<History>>: A Polish High School Exit Exam Matura Benchmark 

**Authors**: Adrian Trzoss, Kacper Dudzic, Wiktor Werner, Marcin Moskalewicz  

**Link**: [PDF](https://arxiv.org/pdf/2608.12343)  

**Abstract**: AI chatbots are widely used by students as knowledge sources, yet LLM benchmarks rarely assess interpretative historical reasoning. We evaluate eight leading LLMs on the Polish high school exit exams (Matura) in history - three official papers from 2023-2025, comprising short-answer questions and extended essays - comparing model performance against the human examinee population. Every model dramatically outperforms human examinees, yet aggregate scores mask distinct competency profiles: rankings are unstable across task type, source modality, and geographical scope, with a consistent penalty on Polish versus Global history content. Qualitative error analysis reveals two recurring failure modes - source conflation, in which models reason from source content rather than treating it as an object of analysis, and temporal disorientation, in which responses are historically misplaced. This study introduces the first LLM history benchmark grounded in Polish national curriculum. 

---
# Are Large Language Models Reliable Reviewers? A Benchmark for Error Detection in Financial Documents 

**Authors**: Ying He, Zhouhong Gu, Zhecheng Hu, Yubo Zhou, Hao Shen, Jiaqing Liang, Zhaoqian Dai, Shuguang Ma, Fei Yu, Yanghua Xiao, Zhixu Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.12342)  

**Abstract**: Ensuring the accuracy of financial documents is critical for economic analysis, regulatory compliance, and corporate decision-making. Several studies have shown that Large Language Models (LLMs) perform well in many financial tasks, such as stock price movements and financial analytics. However, a critical task remains unexplored: the ability of LLMs to identify errors in financial documents. In this paper, we introduce \textbf{FinED-Bench}, the first publicly \textbf{Bench}mark for \textbf{Fin}ancial \textbf{E}rror \textbf{D}etection across three levels of cognitive complexity. FinED-Bench covers nine real-world financial scenarios, and includes over 900 documents reported in 2025 that are unseen by existing language models. We detail the benchmark construction process and evaluate several advanced LLMs (e.g., GPT-4o, Qwen3-14B) on this tasks, which requires both financial domain knowledge and reasoning capabilities. Experimental results show that current LLMs still struggle with this task, especially in high-complexity cases. Besides, supervised fine-tuning can significantly improve the performance of weaker LLMs on this task. Our data and code are available at this https URL. 

---
# The "Knowledge-Behavior Gap" in Cultural Taboo Safety of Large Language Models 

**Authors**: Ying He, Sihang Jiang, Xingzhou Chen, Zhouhong Gu, Yiwei Gu, Minggui He, Shimin Tao, Hongxia Ma, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2608.12341)  

**Abstract**: Cultural taboo safety is essential for deploying large language models (LLMs), as culturally insensitive outputs may cause offense or even social harm. However, existing cultural benchmarks primarily assess cultural knowledge or values biases, while overlooking whether LLMs can recognize and respect cultural taboos, especially when taboos are implicitly hidden in seemingly harmless questions. Besides, cultural taboos are implicit, and context-dependent, thus poss unique challenges for reliable evaluation. To address these gaps, we introduce \textbf{CulShield}, the first public benchmark dedicated to evaluating and improving the cultural taboo safety of LLMs. CulShield spans 77 countries and territories, and includes over 2,020 taboos. It evaluates models along both explicit knowledge and implicit behaviors. Experiments on several advanced LLMs (e.g., GPT-4o-mini, Gemini-2.5-pro) reveal a clear ``knowledge-behavior gap'': models often fail to apply known taboos during interaction. We further show that variations in linguistic context can significantly affect LLMs' cultural taboo safety. Code and data is accessible here: this https URL. 

---
# Class-Structure Preservation Beats Diversity: A Comprehensive Benchmark of Text Augmentation Methods for Imbalanced Text Classification 

**Authors**: Keito Inoshita  

**Link**: [PDF](https://arxiv.org/pdf/2608.12340)  

**Abstract**: With the rapid advancement of large language models (LLMs), generative data augmentation has attracted considerable attention for imbalanced text classification in natural language processing. However, no empirical benchmark to date has compared LLM-based augmentation against the embedding-space SMOTE-style retrieval (EmbSMOTE), a strong classical reference for imbalanced classification. In this study, a controlled benchmark of 11 augmentation methods, spanning classical perturbation, embedding-space retrieval, and LLM-based generation, is newly constructed on seven public text classification datasets covering class counts $K=2$-$28$ and imbalance ratios of 1.1 to over 500, evaluated with five random seeds per cell using macro F1, Welch's $t$-tests, five distributional metrics, and an LLM-family sensitivity analysis based on Qwen3-8B. The experimental results reveal that all LLM-based methods are statistically equivalent or inferior to EmbSMOTE, with the performance gap widening monotonically as imbalance increases and reaching $\Delta\text{F1}_\text{macro}\!\approx\!0.063$ on GoEmotions-28. Furthermore, it is observed that surface-level uniqueness has negligible correlation with downstream performance, whereas LLM-specific artifacts, such as text elongation and label-distribution uniformization, are negatively associated with classification accuracy. Compared with six LLM-based and four classical augmentation baselines, these results demonstrate that the effective variable is not surface-level diversity but class-conditional structural fidelity, namely the degree to which augmented samples preserve the class-conditioned geometry of the training distribution. Accordingly, retrieval-based oversampling should be adopted as the default for imbalanced multi-class classification, and a higher empirical bar should be required before LLM-based augmentation is deployed in practice. 

---
# Mimicry without understanding: the origins of decision bias in large language models 

**Authors**: Eldad Yechiam, Adi Tarabeih  

**Link**: [PDF](https://arxiv.org/pdf/2608.12339)  

**Abstract**: Large Language models (LLMs) were found to be susceptible to a host of social, affective, and cognitive biases. We examined two mechanisms through which such biases can be generated even when human preferences (in the training data) are not biased or when they are correctly categorized as being biased. The first is faulty mimicry of preferences based on human behavior: this involves LLMs inferring human preferences even when behaviors are logically unrelated to preferences. The second is mimicry of explicitly biased human behaviors. In four studies focusing on economic biases, we find that ChatGPT-4o and Qwen exhibited social proof biases even when prompted with reports of human behaviors that were clearly non-indicative of individuals' actual preferences. LLMs also displayed loss aversion when it was explicitly described as a bias. Indeed, when prompted with detailed scientific reports, the extent of the bias (i.e., loss aversion) in the scientific report predicted LLMs' own subsequent bias. Scientific papers of biases can thus become self-fulfilling prophecies, at least when it comes to LLMs' responses. The current study goes beyond fleshing out LLM biases and sheds light on the underlying component processes. 

---
# SDAM: Structure-Difference-Aware Memory Evolution for Complex Text-to-SQL 

**Authors**: Keyan Xu, Dingzirui Wang, Xuanliang Zhang, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2608.12338)  

**Abstract**: Text-to-SQL aims to convert natural language questions into executable SQL queries. While memory-based agent system improves complex SQL generation, existing memory design neglect historical experience and suffer from weak structure analysis, shallow semantic understanding, and poor schema alignment. To address these challenges, we propose SDAM. Specifically, SDAM identifies potential errors via a structure-difference aware reasoning tree, extracts deep semantic rules through contradiction-aware reflection, and enhances structural consistency using a schema-grounded memory evolution mechanism to bind memory with database schemas. We integrate SDAM into a Text-to-SQL framework named SDAM-SQL. Experiment shows that SDAM-SQL achieves 2.0 and 0.4 improvement on BIRD-dev and Spider-test compared with mainstream Text-to-SQL methods, showing the effectiveness of SDAM-SQL. 

---
# From Refuse to Richness: Rubric Rewards for Long-Form Hallucination Reinforcement Learning 

**Authors**: Yudong Wang, Zhe Yang, Wenhan Ma, Rang Li, Qibin Yang, Weimin Xiong, Jiangshan Duo, Liang Zhao, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2608.12337)  

**Abstract**: Rewards that penalize unsupported claims can improve grounding in long-form generation, but they can also teach models to answer less. We study this refusal-to-richness trade-off in long-form hallucination RL. Instead of using global richness proxies such as length, claim count, detail, or pairwise relevance, we represent each question with a key-point rubric that specifies the required and optional information a useful answer should cover. These rubrics define coverage directly and are used both for evaluation and as reward signals. Across grounding-only, proxy-based, rubric-only, and combined rewards, we find a stable trade-off: strict grounding rewards improve support but suppress coverage, while unconstrained rubric rewards improve coverage but weaken grounding. A soft combination of grounding, rubric coverage, and relevance gives the best balance in our experiments, improving in-distribution support while transferring better to out-of-distribution checklist tasks than either grounding-only or rubric-only rewards. 

---
# StorySpark: Module-wise Evolutionary Search for Story Premise Generation 

**Authors**: Yang Yang, Zining Zhong, Qian Cao, Jindong Li, Boyun Xu, Kaishen Yuan, Menglin Yang, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2608.12336)  

**Abstract**: A story premise is the creative spark from which a full narrative can grow. Yet LLM-based story generation has mostly emphasized later-stage planning, controllability, coherence, and prose expansion, while premise-level ideation remains comparatively underexplored. We introduce StorySpark, a module-wise evolutionary search framework for story premise generation. StorySpark operates over interpretable narrative modules such as background, persona, event, ending, and twist, treating each active module not as a static field to fill once, but as a local search space conditioned on the partial premise built so far. For each module, it generates alternatives, evaluates them in context, refines them through feedback-driven mutation and recombination, preserves complementary strengths with Pareto-guided selection, and reallocates frontier capacity to balance branch coverage with promising directions. Multi-view automatic and human evaluations show that StorySpark produces stronger final premises than competitive baselines, with especially consistent gains in originality; when expanded with the same story writer, its premises also lead to higher-quality downstream stories while maintaining completeness, fascination, and diverse usable narrative directions. 

---
# HC-RAG: Evidence-Centric Retrieval-Augmented Generation over Heterogeneous Financial Filings 

**Authors**: Siyuan Chen, Huaye Tan, You Li, Jiajun Liang  

**Link**: [PDF](https://arxiv.org/pdf/2608.12335)  

**Abstract**: Financial question answering over annual reports requires more than retrieving semantically similar passages. It often involves identifying relevant companies and fiscal years, locating standardized filing sections, collecting textual and tabular evidence, and checking answers against the original documents. Existing RAG systems, however, usually flatten long filings into unordered chunks, pay limited attention to the typed structure of financial reports, and use fixed text-table fusion strategies without considering query intent. To address these limitations, we propose \textbf{HC-RAG}, a hierarchical cross-modal retrieval-augmented generation framework for evidence-centric financial QA. HC-RAG organizes filings into a typed financial evidence graph with documents, sections, text units, table units, and metadata nodes. It retrieves evidence through document-section-unit paths, aligns textual and tabular evidence in a shared retrieval space, and routes evidence according to four semantic intents: calculation, trend, fact, and comparison. We further introduce \textbf{Multi-Doc-2025}, a benchmark containing 2,327 expert-verified QA pairs from 179 SEC 10-K filings of 87 S\&P 500 companies across fiscal years 2022--2024, with labels for intent, difficulty, and structural evidence attributes. Experiments on public financial QA benchmarks and Multi-Doc-2025 show that HC-RAG improves both answer quality and evidence localization, especially in long-document, table-related, and cross-document settings. HC-RAG outperforms RAPTOR by 6.6 F1 points on DocFinQA and GraphRAG by 10.9 F1 points on Multi-Doc-2025. Evidence-level analysis and ablation studies show that the improvements mainly come from more accurate section localization, table grounding, cross-document evidence aggregation, and intent-aware text-table routing. 

---
# Steering the Language Axis: From Linear Decodability to Causal Control 

**Authors**: Arnav Srivastav  

**Link**: [PDF](https://arxiv.org/pdf/2608.12334)  

**Abstract**: Despite the impressive multilingual capabilities of Large Language Models, the latent dynamics dictating language selection remain poorly understood. In this work, we ask whether language identity is merely linearly decodable from hidden states, or if it can be causally controlled by a compact activation direction. We conduct an exhaustive causal intervention analysis across multiple model families, including Qwen 3.5-2B and Llama-3.2-1B-Instruct, isolating PCA-derived "language axes" to perform steering and ablation experiments across 1.26 million generations on the FLORES-200 dataset.
Steering along these geometric directions reliably forces language switching in both cross-script (English to Chinese) and same-script (English to Spanish) settings, whereas equal-magnitude random perturbations yield virtually no effect. Our layerwise analysis reveals that language commitment is highly localized and explicitly language-pair-dependent. While English to Chinese switching resists early intervention and steers easily in the later layers, the English-Spanish transition shifts earlier, displaying a distinct, bimodal sensitivity. Furthermore, targeted ablation uncovers a fundamental reversion to English: once the language signal is removed, the model falls back to English regardless of the input prompt. Ultimately, these findings demonstrate that language decision boundaries function during inference as causally active features that are direction-dependent and layer-specific. 

---
# Vision-Language Models are Fragile Multilingual Associators 

**Authors**: Ritabrata Chakraborty, Rajatsubhra Chakraborty, Shivakumara Palaiahnakote, Angelo Cangelosi, Umapada Pal  

**Link**: [PDF](https://arxiv.org/pdf/2608.12333)  

**Abstract**: Vision-language models must associate visual entities with textual attributes. Whether these associations or concept bindings remain stable when the language of the input changes is unexplored. We introduce M$^2$BIND, a benchmark varying the language of the context and query across multiple languages. We evaluate binding both extrinsically through task performance metrics and intrinsically through causal interventions. We find that binding is not language-invariant: cross-family and cross-script settings trigger significant binding collapse, with the model's internal binding computation shifting to later layers and losing causal strength. Closely related languages preserve associations comparatively better. In a broader sense, our findings indicate how VLMs deployed globally in multilingual settings cannot be assumed to maintain the same association quality observed in monolingual evaluation. 

---
# Can Spectral-Clipping Enable Better Learning While Forgetting Less for Low-Rank Adaptation? 

**Authors**: Hyowon Wi, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2608.12332)  

**Abstract**: In recent years, low-rank adaptation (LoRA) has emerged as a significant paradigm that freezes pre-trained weights and introduces small, learnable adapters instead of fine-tuning the full set of parameters. In this work, we uncover several key insights regarding the singular components of network parameters based on Singular Value Decomposition (SVD). Firstly, the principal singular components with large singular values in pre-trained network parameters can be effectively reused during fine-tuning, whereas the minor components with smaller singular values are more task-specific and require substantial adaptation. Secondly, we first establish the theoretical connection that the uncontrolled growth of singular values in LoRA adapters leads to the forgetting of pre-trained knowledge -- a well-known issue referred to as catastrophic forgetting. Building on these observations, we propose SCLoRA, which injects parameterized singular components with spectral clipping into the pre-trained model in a way that is aware of the spectral distribution of the pre-trained model. SCLoRA effectively adapts to new tasks by focusing updates on components that require adaptation, while simultaneously alleviating catastrophic forgetting. We conduct extensive experiments and demonstrate that SCLoRA not only improves downstream performance but also effectively retains pre-trained knowledge. 

---
# Thought-Aware KV Cache Compaction for Reasoning via Adaptive Attention Matching 

**Authors**: Yang Liu, Bin Chong, Chongyang Zhang, Hao Zheng, Jiayu Liang, Xu Kefu  

**Link**: [PDF](https://arxiv.org/pdf/2608.12331)  

**Abstract**: Reasoning language models generate lengthy chain-of-thought (CoT) sequences whose key-value (KV) cache grows linearly and becomes a memory bottleneck during decoding. Existing compaction methods treat reasoning trajectories as flat token sequences and apply uniform compression, ignoring the hierarchical structure of CoT reasoning where different steps vary drastically in importance. We propose \textbf{Thought-Aware Attention Matching (TAM)}, which exploits this structure through three mechanisms: (i)~thought segmentation that decomposes the trajectory into reasoning blocks, (ii)~adaptive budget allocation that assigns compression budget based on each segment's importance and size, and (iii)~pivotal token protection that preserves high-attention reasoning anchors. We prove that the allocation rule is optimal under a convex error model and that cumulative error under sequential compaction remains bounded. Experiments on AIME 2024 and MATH-500 with Qwen3-4B show that TAM improves accuracy over uniform compaction at the same memory footprint, with periodic compaction bounding peak memory to 3.1--3.2\,GB (a 65\% reduction) while maintaining competitive accuracy. 

---
# Reliability-Aware Sexism Detection: Combining DPO with Annotator Agreement and Token-Level Confidence Scoring 

**Authors**: Hadi Mohammadi, Shihan Wang, Masoume M. Raeissi, Anastasia Giachanou  

**Link**: [PDF](https://arxiv.org/pdf/2608.12330)  

**Abstract**: The detection of online sexism remains an open problem. Sexism detection is inherently subjective, yet most existing systems reduce multi-annotator labels to a single majority decision and treat all instances uniformly. This ignores two informative signals: annotator agreement and model uncertainty. We propose RA-DPO (Reliability-Aware Direct Preference Optimization), which integrates annotator agreement, model confidence, and a token-level uncertainty signal into a single reliability score. RA-DPO uses this score to select high-value preference pairs during training and to support inference-time abstention, which allows the model to trade coverage for accuracy. We evaluate RA-DPO on 6,920 multilingual posts from EXIST 2023, fine-tune OpenAI gpt-4o base via DPO, and validate on two open-weight 3B models (Llama, Qwen). Results show that training on the top 30% most reliable pairs matches full-data DPO, which indicates that reliability-aware selection can reduce training cost without sacrificing performance. At inference, selective prediction reaches 96.2% accuracy at 50% coverage in the true-agreement setting and 88.7% in the deployable predicted-agreement setting, both exceeding the 85.3% no-agreement baseline. These results suggest that accounting for annotation uncertainty is beneficial for both efficient training and reliable deployment in subjective classification. 

---
# AnchorSIPS: A Synthetic Dataset and Evaluation Resource for Evidence-Supported Psychosis-Risk Symptom Measurement 

**Authors**: Guilherme C. Oliveira, Stephanie Fong, Zimu Wang, Clarice Lee, Xiangyu Zhao, Duy Khoa Pham, Duong Nhu, Yiwen Jiang, Jiahe Liu, Zhongxing Xu, Dwarikanath Mahapatra, Dominic Dwyer, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2608.12329)  

**Abstract**: Progress on AI for psychosis-risk assessment is limited by a data-access bottleneck. Real clinical interviews are difficult to share because of privacy, governance, and consent constraints. We present AnchorSIPS, a synthetic dataset of 10K structured psychosis-risk interviews with transcript-grounded measurement targets. Each interview is modeled on Mini-SIPS, a clinician-administered psychosis-risk interview. It captures history, 24 symptom questions, follow-up evidence for items the patient affirms, decisions about delusion-like symptoms (unusual beliefs), hallucination-like symptoms (unusual perceptions), and disorganized communication, exclusion of clear psychotic-level symptoms ("frank psychosis"), and a final attenuated psychosis syndrome (APS) diagnosis, a high-risk state of milder or early psychotic symptoms. The APS diagnosis is not a standalone label. It depends on earlier endorsements, supporting follow-up details, symptom-class decisions, and the frank-psychosis check. Every intermediate decision is anchored to its supporting transcript turns. AnchorSIPS is generated by a plan-then-realize pipeline. A hidden case sheet specifies the patient's clinical state, a deterministic planner fixes the interview structure, and an LLM realizes only the patient utterances under validation and bounded repair. Fixing labels and structure before generation avoids the inter-turn inconsistencies typical of multi-turn LLM dialogue. Across seven LLM baselines, models recover coarse decisions but fail to extract follow-up details or cite supporting transcript turns, so final-label performance overstates interview competence. AnchorSIPS is intended for research on evidence extraction, transcript-grounded measurement, and uncertainty under partial disclosure. 

---
# LoRA-Diffusion: Parameter-Efficient Fine-Tuning via Low-Rank Trajectory Decomposition 

**Authors**: Iman Khazrak, Narges Nejad, Mohammadhossein Homaei, Mostafa M. Rezaee, Robert C. Green II  

**Link**: [PDF](https://arxiv.org/pdf/2608.12328)  

**Abstract**: Parameter-efficient fine-tuning methods such as LoRA have transformed the adaptation of large autoregressive language models, enabling task-specific customization with substantially fewer trainable parameters. However, these methods have not been successfully extended to diffusion-based language models, which generate text through iterative denoising rather than sequential token prediction. We propose LoRA-Diffusion, a parameter-efficient fine-tuning approach that applies low-rank decomposition to the denoising trajectory instead of model weights. Unlike weight-based LoRA, which modifies individual transformation matrices, our method learns low-rank perturbations to the entire diffusion path from noise to output. We introduce trajectory-level low-rank adapters that modify each denoising step, step-adaptive rank allocation across diffusion phases, and compositional multi-task learning that allows merging task-specific modules at inference without retraining. On SST-2, QNLI, and MRPC, we report token-level denoising validation accuracy over five random seeds. LoRA-Diffusion achieves the highest mean performance on SST-2 and strong performance on QNLI and MRPC. Joint multi-task training further shows that LoRA-Diffusion achieves the highest token-level accuracy among the evaluated methods. The approach reduces per-task storage compared with full fine-tuning and establishes a parameter-efficient fine-tuning framework for diffusion language models. 

---
# Comparative Analysis of Multilingual Pre-trained Models for Nepali Automatic Speech Recognition 

**Authors**: Suman Paudel, Sarbin Sayami  

**Link**: [PDF](https://arxiv.org/pdf/2608.12327)  

**Abstract**: Multilingual pretrained models nominally support Nepali, yet no controlled benchmark has compared them under a single fine-tuning protocol. We fine-tune six pretrained models (XLSR-53, IndicWav2Vec, MMS-1B, Whisper-Medium, Whisper-Large-v3-Turbo, and Conformer-Hi) spanning CTC self-supervised, autoregressive encoder-decoder, and hybrid Conformer-CTC architectures, on the OpenSLR SLR54 Nepali corpus (~165 hours) using identical preprocessing, splits, optimizer, and family-matched learning-rate schedules. We evaluate Word Error Rate (WER), Character Error Rate (CER), and Real-Time Factor (RTF) on three independent test sets (OpenSLR, FLEURS, Common Voice). Whisper-Large-v3-Turbo (14.76% WER) and IndicWav2Vec (14.89% WER) tie at the top despite a 9x parameter gap and 40x pretraining-data gap, providing direct empirical evidence that language-family proximity in pretraining can substitute for raw scale for in-domain Nepali. CTC decoders run up to 29x faster than autoregressive Whisper at the same accuracy, flipping the practical deployment preference toward CTC under any latency budget. Massively multilingual pretraining (MMS-1B) yields the smallest out-of-domain degradation on FLEURS (+12.55 pp), indicating that scale buys robustness rather than peak in-domain accuracy. The resulting benchmark provides the first standardized, multi-model, efficiency-aware reference numbers for Nepali ASR. 

---
# On Measuring Semantic Preservation in Legal Ontology Learning 

**Authors**: Albert Sadowski, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2608.12326)  

**Abstract**: Ontology learning transforms unstructured text into structured representations for automated reasoning. Yet structuring information risks losing it, and current evaluation methodologies cannot detect such loss, focusing on structural correctness while failing to measure whether meaning survives transformation. We propose an evaluation methodology that addresses this: comparing LLM task performance on source documents against performance on transformed representations, with the difference quantifying semantic loss. We demonstrate this approach on legal merger agreement analysis, a domain chosen for its complex language and precise semantic requirements, comparing direct LLM application against three ontology learning methods across six language models. The results reveal systematic semantic loss with significant variation based on reasoning complexity and model-method interactions. Our contributions are: (1) an evaluation framework for measuring semantic preservation in ontology learning, and (2) empirical evidence that semantic loss varies dramatically with model-method pairing, providing guidance for selecting optimal configurations in legal knowledge systems. 

---
# Why Do AI Agents Break Rules? How Framing, Context, and Social Signals Shape Compliance 

**Authors**: Mika Okamoto, Ansel Kaplan Erol, Kutluhan Erol  

**Link**: [PDF](https://arxiv.org/pdf/2608.12323)  

**Abstract**: Specifying a penalty can paradoxically convert a legal obligation into a cost-benefit calculation that favors violation. We demonstrate that this enforcement information paradox systematically occurs in AI agents. While most AI safety evaluations test whether models fail, we investigate why, applying compliance theory from law and economics as a diagnostic tool. We treat compliance theories not as metaphors but as empirical hypotheses and show that each predicts the behavior of a distinct model class. We evaluate our hypotheses across twelve instruction-tuned language models operating as enterprise procurement chatbots. Drawing on theories of deterrence, legitimacy, and expressive law, we show that safety-fine-tuned models maintain compliance broadly, while task-optimized and agentic models treat regulatory signals as mere optimization parameters. These latter models fail to comply under conditions predicted by theory, such as low enforcement penalties and non-command phrasing. Across all models, introducing financial incentives, managerial demands, peer outcomes, or employee pressure produces large compliance failures. AI procurement agents systematically violate regulatory constraints to satisfy local user objectives in ways not captured by standard alignment benchmarks. Ultimately, compliance cannot be achieved by rule embedding alone; model selection is itself a governance decision, and benchmark-based evaluation is insufficient for compliance-sensitive deployments. 

---
# What Drives LLM Self-Reflection? A Controlled Ablation of Uncertainty Routing in Armed Conflict Forecasting 

**Authors**: Poli Nemkova, Haeshitha Indukuri  

**Link**: [PDF](https://arxiv.org/pdf/2608.12322)  

**Abstract**: Self-reflection is widely assumed to improve LLM reasoning, yet which component drives the gain remains poorly understood. We present a controlled six-condition ablation isolating four components of LLM self-reflection: evidence exposure, diagnostic scaffolding, taxonomy vocabulary, and action routing. Two precise null results converge on a single mechanism. First, structured diagnostic questions add no measurable value over unstructured reflection ($\text{F1} = 0.296$ vs $0.297$, $p = 1.000$, 95\% CI $[-0.041, +0.040]$). Second, presenting the full uncertainty taxonomy while collapsing the action space to a single generic action also adds no value ($\Delta\text{F1} = +0.008$, overlapping 95\% CIs), ruling out taxonomy vocabulary as the mechanism. Typed action routing provides consistent directional gains ($\text{F1} = 0.379$ vs $0.296$); the conservative estimate controlling for taxonomy vocabulary is $\Delta\text{F1} = +0.075$, and the overall gain over the single-shot baseline is significant by bootstrap CI ($\Delta\text{F1} = +0.101$, 95\% CI $[+0.020, +0.185]$). The vocabulary-routing decomposition replicates on GPT-4o: taxonomy vocabulary adds no significant value over generic reflection ($p = 0.773$), while action routing provides significant gains ($p = 0.025$), confirming the mechanism holds across backbones. Gains concentrate on structurally novel conflicts: in Myanmar ($\text{F1}: 0.000 \rightarrow 0.353$) and Ukraine ($0.167 \rightarrow 0.500$), the vocabulary-only condition recovers no more than generic reflection while action routing breaks the degenerate prior. These findings identify typed action routing -- not diagnostic scaffolding or taxonomy vocabulary -- as a promising design principle for metacognitive LLM forecasting agents, while motivating larger-scale evaluation across conflict typologies. 

---
# LLMs Know the Constraint But Do Not Use It: Activation Bottlenecks in Pragmatic Constraint Reasoning 

**Authors**: Yubo Li, Ramayya Krishnan, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2608.12321)  

**Abstract**: When a salient surface cue competes with an implicit feasibility constraint, LLMs often fail -- but aggregate accuracy conflates genuine constraint inference with conservative defaulting. We formalize the distinction as conditional constraint activation: the constraint is internally encoded (Knowledge) symmetrically across constraint-present and -absent prompts (Symmetry), yet only sometimes routed into the decision (Routing) and repairable by a donor activation (Repair). A quartet diagnostic over 14 models reveals two failure modes; probes on two open weights decode the constraint above $88\%$, yet activation patching repairs one ($+6.4$ nats) and not the other ($-0.07$). On a mitigation frontier, no prompted intervention reaches the repair corner: all inflate conservative bias through a single mediation pathway -- prerequisite mention. Hidden-constraint failure is a routing problem, not a knowledge problem. 

---
# AutoDesign: Meta-Harness Optimization for Long-Horizon Agentic Design 

**Authors**: Yaxin Luo, Haobin Jiang, Jialv Zou, Xu Huang, Wenhao Yan, Haodong Li, Zhengrong Yue, Jing Li, Xiaofu Chen, Xiaohan Zhao, Jiacheng Liu, Jiacheng Cui, Zhiqiang Shen, Xiaotong Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.13560)  

**Abstract**: Transforming multimodal sources into condensed and structured media outputs can be fundamentally conceptualized as a long-horizon agentic process centered on a model-harness system. While an ideal harness system should align with human design priors and accumulate reusable experience through empirical exploration to drive recursive self-improvement, existing paradigms remain static and fall short of this capability. In this paper, we present AutoDesign, a framework that aligns with human design priors, where a meta-harness optimizer guides a code agent to recursively improve harness based on rollout feedback. To instantiate and evaluate this framework, we focus on the academic paper-to-poster generation task and introduce PosterBench, comprising a 100-paper Main Track spanning five disciplines and PosterBench-mini, a shared 10-paper subset for controlled evaluation. On the PosterBench Main Track, AutoDesign achieves the highest score of 78.32, surpassing the closed-source commercial system Claude Design by 7.45 points. Across seven controlled code-agent-model configurations, integrating the learned DesignHarness consistently improves performance, increasing the average PosterBench Score from 54.99 to 67.39 (+12.4%). In a fully autonomous long-horizon loop, it executes 253 tool calls and 11 editing turns within 40 minutes for under $3, reaching average conference-poster quality in human evaluation. A system-blind human study further demonstrates that AutoDesign achieves the highest human preference among evaluated systems. 

---
# OmniScientist: An Omni-Modal Omni-Discipline AI Scientist 

**Authors**: Bobo Li, Hao Fei, Tianjie Ju, Mong-Li Lee, Wynne Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2608.13558)  

**Abstract**: Recent advances in foundation models have enabled AI scientists to automate increasingly complete research workflows, from hypothesis generation and code execution to manuscript preparation. Yet workflow coverage alone does not provide access to the full evidence on which scientific discovery depends. Existing systems typically reason over text, code, labels, or precomputed summaries, leaving scientifically decisive spatial, temporal, cross-channel, and procedural relations unavailable to the agent. We introduce OmniScientist, an end-to-end, omni-modal AI scientist that conducts multidisciplinary research directly from heterogeneous raw evidence. A perception layer and 3 autonomous agents for ideation, experiment, and writeup operate within a deterministic pipeline, allowing observations to shape research questions, experimental decisions, and final claims throughout the research lifecycle. By running idea, rigour, and claim checks in code, the system enforces novelty screening, statistical validity, execution provenance, and numerical traceability. We evaluate OmniScientist on 36 real-data cases spanning 5 discipline families, 4 families of scientific evidence, and modalities including images, signals, audio, video, 3-D structures, trajectories, tables, formulae, and graphs. The system completes the full path from raw data to a compiled manuscript in all 36 cases and achieves a mean overall paper score of 6.3 with the reference reasoning backbone. In paired comparisons against a blind variant that receives only precomputed scalar features, direct perception improves all 7 evaluation dimensions and wins 85% of head-to-head judgments. These results show that lifecycle-wide perception is essential for evidence-grounded scientific discovery and provides a practical path toward broadly capable AI scientists. 

---
# Intern-S2-Preview: Scientific Agentic Foundation Model 

**Authors**: Lei Bai, Jiaqi Cao, Chiyu Chen, Guanzhou Chen, Kai Chen, Guangran Cheng, Erfei Cui, Xuanlang Dai, Shengyuan Ding, Shangheng Du, Yanhui Duan, Yue Fan, Youqing Fang, Quan Gan, Yuanyuan Gao, Jiaye Ge, Lixin Gu, Yuzhe Gu, Qipeng Guo, Junjun He, Xin Hong, Ming Hu, Zhouqi Hua, Haian Huang, Junhao Huang, Zixian Huang, Minxi Jin, Lingkai Kong, Alexander Lam, Zehao Li, Zonglin Li, Tianhao Liang, Dahua Lin, Junyao Lin, Tianyang Lin, Zhouhan Lin, Jiangning Liu, Jin Liu, Kuikun Liu, Wenran Liu, Yifei Liu, Yuhong Liu, Yuhong Liu, Zhoumianze Liu, Ziyan Liu, Ziyu Liu, Haijun Lv, Han Lv, Chengqi Lyu, Le Ma, Ningsheng Ma, Zerun Ma, Haoyang Peng, Runyu Peng, Jifei Shan, Zixin Shang, Kou Shi, Xiang Shi, Qisheng Su, Xuerui Su, Hao Sun, Xiao Sun, Yanan Sun, Yu Sun, Huanze Tang, Yinghao Tang, Wenhui Tian, Zhongbo Tian, Bingli Wang, Haomin Wang, Jiarui Wang, Jingzhi Wang, Rui Wang, Xiquan Wang, Yi Wang, Zhecan Wang, Ziyi Wang, Zun Wang, Rubin Wei, Lianyi Wu, Wen Wu, Yue Wu, Yuhan Wu, Zhenyu Wu, Zijian Wu, Shuhao Xing, Jun Xu, Xingle Xu, Xuenan Xu, Xiangchao Yan, Ziang Yan, Bowen Yang, Danni Yang, Lin Yang, Zhiqi Yang, Qian Yao, Haochen Ye, Peng Ye, Jinhui Yin, Jiashuo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.13505)  

**Abstract**: Scientific discovery increasingly requires AI systems that can reason over scientific evidence of heterogeneous modalities, interact with scientific tools and environments, and sustain progress across long task horizons. We present Intern-S2-Preview, a series of scientific agentic foundation models designed to support multimodal scientific understanding, reasoning, generation, and long-horizon tasks. The training pipeline begins with scientific multimodal pre-training over rendered scientific documents, interleaved image-text data, and diverse scientific corpora. Starting from the pretrained checkpoint, we apply a unified post-training pipeline consisting of supervised fine-tuning, scalable multi-task reinforcement learning (RL), black- and white-box agentic RL, and on-policy distillation. This pipeline is supported by practical techniques that improve rollout and training stability and efficiency, including partial rollout with off-policy correction, adaptive length regularization, online speculative decoding, robust multi-task optimization, and trace-aware experience assembly for agentic tasks. At the architecture level, Intern-S2-Preview-397B extends time series modelling from efficient long-sequence understanding to numerical forecasting, while Memory Decoder is studied as a separate memory-augmented path for rapid scientific specialization without modifying the frozen 397B backbone. Evaluations across scientific, multimodal, agentic, and general-purpose benchmarks show that Intern-S2-Preview-397B achieves competitive or leading results in multiple settings. The time series modules improve scientific signal understanding and forecasting on SciTS, while the separate Intern-MemDec-4B extension improves the Biology-Instructions average score from 56.92 to 60.32 without modifying the frozen 397B backbone. 

---
# Synthetic Persona Pretraining: Alignment from Token Zero 

**Authors**: Julian Minder, Viktor Moskvoretskii, Raghav Singhal, Difan Jiao, Andy Arditi, Shaobo Cui, Yiderigun Borjigin, Kartik Bali, Stefan Krsteski, Harsh Raj, Huu Nguyen, Jannik Brinkmann, Ashton Anderson, Roland Aydin, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2608.13482)  

**Abstract**: As language-model-based AI is increasingly deployed in autonomous settings, aligning its goals and values with those of humans becomes critical. Today, alignment, and the assistant identity itself, are typically introduced only after pretraining, once behavioral priors are already established. This can make values a thin overlay, rather than deeply rooted, and facilitate subsequent misalignment. Pursuing a different paradigm, we introduce Synthetic Persona Pretraining (SPP), which installs the desired assistant persona from token zero in pretraining. First, we annotate pretraining documents with value-aligned first-person reflections derived from a normative value constitution. Second, we pretrain via the standard cross-entropy loss on standard pretraining documents as well as their reflections, which installs the desired persona among a multitude of other personas. Finally, we post-train on user-assistant dialogue data, which binds this desired persona to the assistant identity, a process we call persona binding. By pretraining models up to 3B parameters on 500B tokens, we show that SPP improves constitution following and jailbreak robustness, and reduces the misalignment rate in out-of-distribution moral dilemmas, while preserving capabilities. Early intervention matters: compared with alignment from token zero, introducing SPP only at the end of pretraining yields weaker constitution adherence, does not shift value priorities, and leads to less aligned choices in dilemmas. This advantage depends on persona binding and, importantly, increases with pretraining budget. Overall, our results show that shaping values early is critical for alignment and establish pretraining-time persona interventions as an effective approach to do so. 

---
# MARC v1: An Open-Source Multi-Agent Framework for Clinical AI Reasoning and Coordination 

**Authors**: Saisha Shetty, Satvik Tripathi, Austin Lin, Colin Zhao, Theodore Kim, Don Enwerem, Jacinta Arnold, Shahriar Faghani, Tessa S Cook  

**Link**: [PDF](https://arxiv.org/pdf/2608.13476)  

**Abstract**: We present Multi-Agent Reasoning and Coordination (MARC), an open-source framework that replaces monolithic LLM prompting with deterministic multi-agent orchestration for clinical reasoning. MARC coordinates role-specialized agents for extraction, reasoning, answer generation, and evaluation, with explicit context passing and traceable intermediate outputs, enabling stage-wise failure attribution. We additionally introduce a Decomposer module that generates task-specific agent prompts from a plain-language description, eliminating manual prompt engineering. The framework supports both API-based and local CPU-compatible deployments and is entirely configurable via YAML, without code modifications. MARC is designed to be model-agnostic, interpretable, and accessible to clinical domain experts without programming expertise. The full framework is available at this https URL. 

---
# MLLM-Routed Heterogeneous Ensembles for Robust Cross-Dataset Image Classification 

**Authors**: Daniel Perkins, John Squires, Janou Milligan, Chandra Raskoti, Linda Ungerboeck  

**Link**: [PDF](https://arxiv.org/pdf/2608.13463)  

**Abstract**: Modern image classification models excel when trained on single task-specific datasets but often struggle to generalize across domains and difficulty levels. We propose ARMDIL, an Adaptive Router for Multi-Domain Image classification with LLMs. ARMDIL is an ensemble that uses a multimodal large language model (MLLM) agent to dynamically route each image to the most suitable vision backbone. Our diverse ensemble employs convolutional neural networks (ResNets), self-supervised representation learners (SSL), and vision-language models (VLMs), each trained on a unified label space constructed from multiple image datasets with differing distributions and characteristics. Empirical evaluations illuminate the distinct capabilities and vulnerabilities of each architecture across disparate visual domains. Crucially, we show that ARMDIL effectively navigates these trade-offs, performing competitively with specialized training-based routers. Furthermore, it drastically improves adaptability by allowing new information to be integrated via simple prompt modifications, while enhancing interpretability through natural language reasoning traces. These advances in cross-dataset image classification pave the way for more reliable general-purpose vision systems such as AI assistants and autonomous robots. 

---
# Reduced Matrix Multiplication: Input-Adaptive Matrix-Product Reduction for LLM Inference 

**Authors**: Zixuan Lan, Yanhong Li, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.13426)  

**Abstract**: Transformer-based language models achieve strong performance but incur substantial inference cost due to repeated high-dimensional matrix multiplications. We propose Reduced Matrix Multiplication (RMM), a training-free, input-adaptive inference method that reduces Transformer matrix products by selecting informative slices along their contraction dimensions, without modifying model weights. Under a simple retention-ratio control, RMM provides a smooth and predictable accuracy-efficiency trade-off. Across language models ranging from 1B to 70B parameters, we find that reduction tolerance depends on the model family, task, component, and retention ratio, although it often improves with model scale. Under moderate reduction, RMM remains robust across the evaluated discriminative, autoregressive generation, and long-context settings. We further show that the same principle extends to multimodal vision-language inference. Mechanistic ablations reveal a structural asymmetry within Transformers: attention-side computations are substantially more reducible than MLP components. Finally, wall-clock benchmarks with custom kernels on an NVIDIA A100 show that these computational savings can translate into practical runtime gains, especially at longer sequence lengths. Together, these results position RMM as a scalable direction for input-adaptive inference-time optimization. 

---
# When Should Multi-Round RAG Stop? Structured Stopping Judgments and Retrieval Reduction in Search-R1 

**Authors**: Weimeng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.13237)  

**Abstract**: Multi-round retrieval-augmented generation (RAG) must decide when to stop searching as evidence accumulates. Because the deployed policy is determined by the first STOP on each trajectory, this is a sequential selection problem rather than an independent state-classification task. We adapt S2G-RAG's structured sufficiency-and-gap judgment to a frozen Search-R1 pipeline and train a Qwen3.5-2B judge on 3,009 states from 900 disjoint HotpotQA questions. Search-R1's reasoner, retriever, corpus, prompt, and search budget remain unchanged, while the judge checkpoint and stopping threshold are selected on grouped validation and frozen before confirmatory evaluation. On the confirmatory test set, the resulting policy reduces retrieval calls by 77 (3.70\%) relative to Native Search-R1, while Official Exact Match decreases by 0.625 percentage points. Thus, the trained S2G-style structured judge reduces retrieval while broadly preserving answer accuracy. The result does not imply unchanged or improved accuracy, safe stopping, or lower total inference cost. 

---
# TRAPSBench: Vision-Language Models Encode but Fail to Express Epistemic Restraint 

**Authors**: Fnu Pramono, John Cai, Sourabh Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2608.13167)  

**Abstract**: When visual evidence is occluded or chaotic, models should abstain. In this paper, we show that Vision-Language Models (VLMs) can internally distinguish when abstention is required, but fail to express it anyway. We introduce TRAPSBench, a procedurally generated video benchmark of 1,404 matched physics pairs in which a single targeted change renders the outcome undeterminable from the visual evidence. Furthermore, we introduce Penalized Epistemic Calibration Score (PECS), a new robust metric that requires models to both answer correctly when the outcome is knowable, and abstain when the outcome is not. Across 16 VLMs spanning five families, spontaneous restraint is poor: the best PECS is 0.292. The bottleneck is expression, not perception: linear probes decode answerability from hidden states at up to 0.91 AUROC across physics domains; steering a single-layer void direction causally induces or suppresses abstention. Our results replicate across three open-weight families (Qwen, Gemma, LLaVA). The failure is also more pronounced in visual than textual uncertainty: models detect textual impossibility about 4x more readily than missing visual evidence. Closing this representation--output gap likely requires output-stage interventions. 

---
# Explanatory Engagement Under Rare Anomalous Failure: Asymptotic Rarity in Model Behavior (or: The Asymptotic AI) 

**Authors**: Sam Mao  

**Link**: [PDF](https://arxiv.org/pdf/2608.13063)  

**Abstract**: Prior work on LLM behavior under anomalous conditions asks whether a model notices anomalies. We ask a narrower question: once a model sits in a workflow with a low, controllable failure rate, does its explanatory engagement - length, specificity, self-reported confidence - change as failure grows asymptotically rarer? We built a local, zero-cost harness on three open-weight models (qwen3:8b, llama3.1:8b, mistral:7b) running a repeated tool-call task where one call fails at probability p, swept across eight rates from 0.2 to 0.0001, under five elicitation conditions from immediate prompting to none. We hypothesized a rise in engagement as failures grew rarer, then a collapse near a detectability threshold. Pooled across conditions this appeared false: length fell in a flat, monotonic pattern. Splitting by condition overturned that. Under immediate_forced, where the model must explain every failure instantly, the predicted rise is confirmed but followed by a plateau, not a collapse: length peaks at 28.4 words at p=0.05, settles to 17.4-19.0 words at the rarest rates, and confidence rises unevenly from about 53% to the 70s-90s. Under grouped_runs, explanation batched to run-end, no collapse appears. Under passive_unprompted, aggregate magnitude is a floor artifact, but a recovered logging gap revealed real, model-specific self-monitoring: llama3.1:8b volunteers structured confidence reports unprompted, sometimes eroding its own confidence as trials accumulate; the other two do so only once, as boilerplate. Elicitation structure is a first-class moderator of collapse observability. A companion guaranteed-failure run (72 cells, backfilling rates where random sampling gave zero real failures) shows models differ in whether they recognize an anomaly, distinct from engagement once recognized. Limitation: discrete rate points cannot capture behavior between them, a direction for future work. 

---
# TEMPO: Makespan-Aware Expert-Parallel Load Balancing Across Memory- and Compute-Bound Regimes 

**Authors**: Jie Li, Chenxin Jia, Jinliang Shen, Cunzhuang Liu, Ruiyi Ding, Jianwen Xian, Kang He, Chengru Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.13057)  

**Abstract**: In expert-parallel (EP) MoE serving, every layer synchronizes at the slowest GPU. Dispatchers balance token counts (EPLB, LPLB, UltraEP) or activated-expert counts (METRO), assuming expert time is linear in one. Measurements on two datacenter GPU generations show it is neither: below $\nstar\!\approx\!156$--$168$ tokens, HBM weight streaming dominates---cost attaches to \emph{activated replicas}, not tokens; above it, grouped GEMM rounds tokens to 128-tile $M$-tiles, so \emph{splitting} an expert adds padded compute. A max-affine profile $t=\max(a+bG,\,c+\beta N)$ captures both regimes. Realistic decode batches hold hot experts in the linear regime and cold in the flat \emph{simultaneously}; recorded batches show proxy dispatches differ by $1.4$--$1.6\times$ in modeled block time (p95 up to $1.7\times$), and \emph{which} proxy wins flips with the regime. We formalize per-batch dispatch as a fixed-charge makespan problem---NP-hard on two fully replicated GPUs, polynomial in degenerate limits---and present \sys{}, a makespan-aware dispatcher solving it in milliseconds off the critical path; its SGLang integration runs out-of-process and fuses dispatch with count collection into one in-graph kernel. Anchored by an 8-GPU Testbed~A microbenchmark, \sys{} stays within 1\% of the best fixed baseline everywhere and wins by up to $15.5\%$ where regimes mix. End-to-end on Testbed~B, Qwen3-235B (inside the win region) gains $4$--$6\%$ throughput and cuts p99 latency by ${\sim}15.6\%$; DeepSeek-V3 (outside, communication-dominated) shows only mechanism cost. A phase diagram, not a universal win, is the claim: it predicts both outcomes before deployment. 

---
# Latent On-Policy Self-Distillation 

**Authors**: Guibin Zhang, Jiayang Lyu, Ran Sun, Xinlei Yu, Haoyu Zhao, Qibing Ren, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2608.13040)  

**Abstract**: Enabling agents to learn from experience and internalize it into their policy has become a central problem in self-evolving AI. On-policy self-distillation (OPSD) offers an effective pathway by using a privileged self-teacher to provide dense supervision on the student's own trajectories; however, existing methods still rely heavily on designer-specified privileged artifacts (e.g., answers, feedback, skills, or trajectories), limiting the end-to-end learnability and scalability required for continual self-improvement. In this work, we introduce Latent On-Policy Self-Distillation (LOPD), which, rather than proposing another hand-crafted OPSD variant with a newly prescribed form of privileged context, makes the teacher's privileged context itself learnable end-to-end from experience. Technically, LOPD retrieves relevant experiences and composes them into continuous latent tokens that condition a self-teacher, while the student generates trajectories from the task and interaction history and receives dense token-level supervision at every visited prefix. We further introduce a privileged-margin objective to stabilize and regulate the learning of latent context. Empirically, LOPD demonstrates (I) strong performance, outperforming RLVR and representative OPSD methods including OPSD, SDPO, and Skill-SD across both agentic tool use and code generation; and (II) high learning efficiency, surpassing GRPO and Skill-SD with less than 30% of their rollout budget. Ablation studies further provide direct evidence that making privileged context learnable is necessary for realizing these gains. Together, these results position LOPD as a step toward a more scalable and self-directed paradigm for agent evolution. 

---
# Reconcile Once, Write Anytime: A Trust-Tiered Librarian and a Multi-Agent Writer for Drift-Free, Point-in-Time Research 

**Authors**: Xing Zhang, Yanwei Cui, Guanghui Wang, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2608.12984)  

**Abstract**: Long-form research reports generated by large language models drift, contradict themselves, and lose provenance: the same metric appears with different values, and rumor is quoted as confidently as an audited filing. We present a two-tier agentic system that separates a maintained, point-in-time knowledge library from report writing. A deterministic "librarian" ingests timestamped sources into a trust-tiered ontology, layering evidence cards, an authoritative metric ledger, and a claim graph into an always-current source of truth, not per-query RAG over raw chunks. A portable multi-agent "writer" runtime then composes a contradiction-free, evidence-grounded report at any knowledge cutoff T, reading only evidence with as_of <= T (no look-ahead); red-team verdicts flow back into the librarian. We evaluate on a self-collected, public corpus of 6,130 sources yielding 555,926 evidence cards (SEC EDGAR filings across 295 issuers and 11 sectors, U.S. Bureau of Labor Statistics releases, and Wikipedia). From the one library we compose four point-in-time reports on distinct theses and run eight reproducible experiments, whose headline metrics come from a deterministic quality-control gate, itself validated by defect-injection meta-evaluation at recall 1.0 and precision 1.0. A shared metric ledger removes 6,845 cross-section contradictions to zero. Tier-first selection is correct on 22/22 gold cases where a popularity-first baseline scores only 9/22; trust tiering leaks zero media-sourced numbers, and no government statistic displaces a company's own filing. A red-team refutation propagates back and self-corrects a later run with zero manual edits. Replay exhibits zero look-ahead violations across seven cutoffs while the library grows from 235,373 to 555,312 cards. Difficulty-tiered model routing exceeds the all-Opus quality ceiling while running 3.7x faster than serial. 

---
# Comment on "Modeling rapid language learning by distilling Bayesian priors into artificial neural networks" 

**Authors**: Orr Well, Idan Tarshish, Nur Lan, Roni Katzir  

**Link**: [PDF](https://arxiv.org/pdf/2608.12974)  

**Abstract**: McCoy & Griffiths (2025, henceforth M&G) suggest that a Bayesian prior can be distilled into Artificial Neural Networks (ANNs) through Model-Agnostic Meta-Learning (MAML, Finn et al., 2017). They support this empirically by showing that meta-trained networks demonstrate formal language learning abilities comparable to Yang & Piantadosi (2023)'s Bayesian learner, significantly outperforming standard ANNs. We point out that under the standard interpretation of a prior, M&G's procedure does not actually instill one; it merely initializes network weights favorably, leaving the objective function unchanged. We then consider a more permissive interpretation, where the system as a whole can be seen as implementing a Bayesian learner even without an explicit prior in the objective. We show that this interpretation faces nontrivial challenges. Finally, we assess how well MAML approximates the empirical results of Bayesian learning, showing that unlike genuine Bayesian learners, M&G's model overfits and generalizes poorly to unseen data. 

---
# I-SDPO: Instance-Level Adaptive Self-Distillation Policy Optimization 

**Authors**: Yubo Zhang, Xinhong Ma, Zezhong Tan, Ziqiang Dong  

**Link**: [PDF](https://arxiv.org/pdf/2608.12957)  

**Abstract**: Group Relative Policy Optimization (GRPO) learns from reward differences within a rollout group, but receives no useful relative signal when every sampled response is incorrect. Privileged self-distillation can fill this gap with dense token supervision, yet applying it throughout training creates a different failure mode: the teacher is a biased, low-variance surrogate for the reward objective, so persistent imitation can oppose reward-improving updates after the policy becomes capable of producing successful trajectories. We introduce I-SDPO (Instance-Level Adaptive Self-Distillation Policy Optimization), which treats teacher reliance as capability-dependent. I-SDPO makes one routing decision per input instance and shares it across that instance's rollout group: all-incorrect groups use a privileged self-distillation objective, whereas any-success groups remain intact for GRPO. This design uses imitation only where group-relative rewards are uninformative. A local analysis characterizes when teacher and reward directions align and shows that a non-vanishing biased distillation weight induces an optimization bias floor. The routing rule automatically reduces the expected distillation rate as success probability rises, withdrawing teacher influence without a hand-designed schedule. On SciKnowEval, I-SDPO obtains the best result in all four scientific domains and improves average mean@16 accuracy from 56.67% with GRPO to 70.31%, with a maximum domain gain of 18.24 points. 

---
# Beyond Retrieval: Query-Conditioned Reuse of Long-Horizon Agent Trajectories 

**Authors**: Yifei Li, Heng Wang, Lingling Zhang, Muye Huang, Xinyu Zhang, Jiashuai Liu, Hang Yan, Rongman Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.12847)  

**Abstract**: Retrieval can identify a past trajectory that may matter, yet it does not specify how an acting agent should use that trajectory after users, entities, constraints, or environment state have changed. We identify this post-retrieval reuse step as a distinct bottleneck for long-horizon trajectory memory and formulate an evaluation framework that holds candidate retrieval, target state, model, decoding, and tool budget fixed while varying the support delivered to the agent. We instantiate the framework with query-conditioned reuse (QCR), a deliberately simple target-bound note that records a reusable procedure, bindings to recover, applicability conditions, and verification requirements. QCR serves to test the reuse hypothesis rather than to claim a universally preferred memory format. Across 2,391 target instances in WebArena, WorkArena, and AppWorld, QCR reaches 62.3% average Success, 10.7 points above Full Trajectory, while using 48.9% fewer online tokens. Summary reranking selects a reusable memory for 94.8% of targets, placing end-task Success within 1.8 points of an oracle reusable selector. Analyses by trajectory length and source--target binding shift show that direct trajectory injection loses much of its utility as traces grow longer or source-specific values change, whereas target-bound support preserves a larger share of the measured gain. The resulting framework separates retrieval quality from the problem of turning retrieved experience into safe, useful support for a new task. 

---
# Dual-Stream Cross-Anchor Correction Grounding Long-Form Captions and the Domain Limits of Object-Level Anchors 

**Authors**: LingKai Bu  

**Link**: [PDF](https://arxiv.org/pdf/2608.12746)  

**Abstract**: Object hallucination in multimodal large language models arises when language priors and corpus co-occurrence bias outweigh the visual evidence, with nothing tying an individual object mention to what the image shows. Most remedies intervene at decoding time without training, yet under a unified protocol their benefit is confined to short captions;supervised fine-tuning (SFT) on a detail- rich corpus lengthens captions, but over forty percent still name absent objects. This paper proposes Dual-Stream Cross-Anchor Correction (DSCC). Unlike work that post-processes decoding, DSCC is the first to inject object-level visual anchors into the language model itself during fine- tuning: a perception stream aligns object-level hidden states at an intermediate layer to frozen text anchors by a bidirectional contrastive objective; a cognition stream lets deeper layers query those anchors by cross-attention at every generation step; and a two-stage curriculum gate couplesthem, making evidence retrieval a structural constraint at each autoregressive step. Under one backbone and one scoring protocol, experiments span long-caption hallucination, object-existence discrimination and cross-domain generalisation, with vanilla SFT on the same corpus and schedule as a length- and density-matched control, so gains are attributed layer by layer. DSCC is the only method reaching the long-caption, low-hallucination region: captions roughly 1.9 times the baseline length at 88.19% precision per object mention, the highest under a density-independent criterion. Ablations expose a synergy: the perception stream alone degrades precision yet reverses sign when stacked on the cognition stream. No universal superiority is claimed: three out-of- domain benchmarks yield a predictable, falsifiable domain-conditionality, the synergy being bound to the anchors' semantic domain and breaking on charts and optical illusions. 

---
# Perturbation-based Regional Interpretability through Subtraction Mapping (PRISM): naming-error dissociations in language models and post-stroke aphasia 

**Authors**: Xiang Guan, Roger D. Newman-Norlund, Yong Yang, Saeed Ahmadi, Regan Willis, Nadra Salman, Kalil Warren, Srihari Nelakuditi, Chris Rorden, Leonardo Bonilha, Julius Fridriksson  

**Link**: [PDF](https://arxiv.org/pdf/2608.12717)  

**Abstract**: Mechanistic interpretability of large language models lacks spatially resolved, falsifiable tools for testing whether internal components are specialized for distinct cognitive operations. We adapt subtraction analysis, the standard framework of human neuroimaging, from biological brains to perturbed transformers, and apply the same logic to both substrates in parallel. Building on the Brain-LLM Unified Model (BLUM), which showed that layer-perturbed LLaVA-1.6-Vicuna-13B error profiles match the lesion patterns of aphasic patients, we develop PRISM (Perturbation-based Regional Interpretability through Subtraction Mapping). PRISM maps the seven clinical Philadelphia Naming Test categories, subtracts error classes pairwise, and treats each perturbation seed as a subject in a group analysis with threshold-free cluster enhancement along the layer axis. We run a structurally matched analysis on 213 chronic post-stroke aphasia patients using correlation-difference lesion-symptom mapping, and replicate both sides on held-out splits. The designs match in subject dimension (seeds, patients), spatial dimension (layers, atlas-parcellated cortex) and thresholding, but the contrast operator differs: a within-subject error-proportion difference for the LLM, a between-subject correlation difference for the cortex. Both substrates recover a robust phonemic-favoring dissociation, a deep layer cluster and a frontal-perisylvian cortical cluster, both replicating; the semantic-favoring direction is a consistently signed but non-significant trend on both. PRISM thus gives a falsifiable, spatially resolved test of functional-specialization claims in transformer language models. A confirmatory ROI-level intervention (PRISM Stage 3) licensing the strongest causal-mechanism claim is left to subsequent work. 

---
# Tracing Provenance and Detecting Tampering with Complementary LLM Watermarks 

**Authors**: Xiaoyan Feng, Yanjun Zhang, He Zhang, Leo Yu Zhang, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2608.12713)  

**Abstract**: Watermarking LLM-generated text is an important task for tracing its provenance. Existing LLM watermarks preserve provenance under editing, but this same robustness allows an adversary to alter critical content while retaining attribution, a vulnerability known as piggyback spoofing. We introduce an innovative watermark that jointly provides provenance and tamper evidence. It co-embeds a robust signal and a fragile signal into each generated token. The signals share the same mechanism but use independent keys and different seeding windows over normalized text, making one resilient to edits and the other sensitive to reader-visible changes. Multiple rounds of unbiased tournament reweighting preserve the expected generation distribution, while a periodic round-allocation pattern controls the trade-off between the two signals. At detection, their scores form a two-dimensional space supporting three decisions: Intact, Tampered, and No-Watermark. Across two large language models and two prompt datasets, our method demonstrates the highest tamper-detection rate among the evaluated methods while maintaining competitive attribution robustness and perplexity. Ablation studies show that reliable three-state detection requires a well-defined notion of intactness, co-embedding of the two signals, and complementary sensitivity to edits. 

---
# SteerBench-Work: A Benchmark for Agent Steering at Action Boundaries 

**Authors**: Oguz Serdar, Cuneyt Mertayak  

**Link**: [PDF](https://arxiv.org/pdf/2608.12654)  

**Abstract**: Long-running LLM agents act through tools, and a single step can send an email, merge a pull request, or wire a payment. The steering decision is the pre-commit choice at that boundary: proceed, or hold for human or policy review. We introduce SteerBench-Work, an incident-anchored, bidirectional benchmark for that decision in workplace agents across developer operations, customer service, finance, legal, medical, HR, and security.
Release v2026-05 contains 106 scenarios anchored in public incidents, paired evidence-reversed mirrors, and calibration controls, with labels split nearly evenly between proceed and hold so the two error directions get near-identical numbers of chances. A model sees the proposed action and the available evidence, returns a gate decision, and is scored on whether it crosses or holds the boundary correctly. Across 30 model conditions the failures run almost entirely in one direction: models wrongly hold authorized, evidence-cleared work on 28.1% of opportunities and wrongly allow unsafe work on 1.0%. The hardest cases are risk-resolved commits, where signed or structured evidence has already cleared a real risk trigger, and models score markedly worse on evidence-reversed mirrors of famous incidents (63.8%) than on the incidents themselves (98.5%). General capability is not the same as steering calibration: higher-capability models often over-refuse at the commit boundary, and more reasoning can repair a weak gate while leaving a calibrated one flat. The public leaderboard is at this http URL. 

---
# EgoCITE: Context-Augmented Indexing and Time-Aware Retrieval for Long-Horizon Egocentric Memory 

**Authors**: Le Zhang, Ke Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.12627)  

**Abstract**: Long-horizon egocentric memory transforms continuous first-person video and audio into a searchable record of past experiences. We demonstrate two bottlenecks in existing systems: indices built from context-poor captions are unreliable for agentic search, while retrieval ignores a question's temporal intent. To address both bottlenecks, we introduce EgoCITE (Egocentric Context-augmented Indexing and Time-aware Evidence retrieval), a long-horizon agentic memory framework for egocentric QA. EgoCITE comprises three components. EgoScheme uses local multimodal context to turn fragmentary video captions and speech transcripts into self-contained atomic memory indices. EgoIndex organizes complementary action, activity, utterance, and conversation representations into searchable multi-view memory indices at multiple granularities. EgoRetrv combines semantic search with question-conditioned temporal relevance scoring and curation of retrieved evidence. We evaluate EgoCITE on EgoLifeQA, EgoMem, and EgoR1-Bench in terms of answer accuracy and target-event retrieval alignment. EgoCITE improves accuracy over agentic memory baselines by at least 4.4--14.2\% while achieving 36$\times$ lower cost than long-context LLM agents. 

---
# Is this Citation on Point? 

**Authors**: Apurv Verma  

**Link**: [PDF](https://arxiv.org/pdf/2608.12571)  

**Abstract**: In 2023, a New York judge sanctioned two attorneys in Mata v. Avianca for filing a brief with hallucinated citations generated by ChatGPT. Such failures are largely caught by database lookups; the harder problem is detecting citations that point to real cases but do not support the propositions for which they are offered -- a failure mode that existing evaluations of LLMs for legal use cases largely overlook. In this paper, we study proposition-level citation support verification through controlled perturbations of real legal citations obtained from two legal corpora, either replacing the cited case or changing only the pinpoint page within the same case. We evaluate fourteen model configurations on the resulting examples. Models catch 93-100% of wrong-case corruptions. They catch only 37-61% of wrong-pinpoint corruptions on court opinions and 52-83% on legal briefs. When models fail to catch wrong-pinpoint corruptions, they accept the citation based on topical overlap rather than page-level support. Scale and extended reasoning narrow the gap but do not close it: GPT-5.4 with high reasoning effort still misses 40% of pinpoint mismatches on court opinions and 18% on briefs. Prompting the model to verify support at the cited page improves recall, but it also raises the false positive rate. Recognizing the right legal topic and verifying support for the cited proposition are distinct capabilities, and current models conflate them. 

---
# SoK: From Generation to Consumption of Privacy Documents in Software Systems 

**Authors**: Shidong Pan, Clark LaChance, Zhen Tao, Sepideh Ghanavati  

**Link**: [PDF](https://arxiv.org/pdf/2608.12511)  

**Abstract**: Privacy documents (e.g., privacy policies) are a central mechanism through which digital services disclose data practices and seek user consent. Over the past decades, research on privacy documents has expanded significantly, encompassing not only traditional privacy policies but also short notices (e.g., privacy labels) and interface-level transparency mechanisms. As this research area continues to grow, it has become increasingly difficult to obtain a coherent view of how privacy documents are created, analyzed, evaluated, and maintained across their lifecycle. This SoK provides a unified, lifecycle-oriented view of privacy documents from a software engineering perspective. We systematically review and analyze 290 papers published between 2010 and 2025, organizing them around five research questions that examine how privacy documents are (1) defined and scoped, (2) generated, (3) analyzed and extracted, (4) checked for inconsistencies and noncompliance, and (5) evaluated and improved for usability. Building on our findings, we identify 15 key research trends and 21 open opportunities. We further chart four broader research directions that highlight (i) emerging challenges in AI-centric platforms, (ii) the need for diverse and up-to-date data foundations, (iii) LLM-based unified policy-code analysis, and (iv) dual usability for end-users and developers. We hope this SoK provides a shared foundation for future research on privacy policies and privacy documents. 

---
# Geometric and Behavioral Stratification in Transformer Residual Streams 

**Authors**: Nelson Guda  

**Link**: [PDF](https://arxiv.org/pdf/2608.12447)  

**Abstract**: Trained transformer models develop privileged bases: coordinate axes whose statistics differ from the rest of the residual stream. But what kind of direction does such a basis select? We investigate the prediction direction, the unembedding direction of the token a model currently predicts, and find that it functions as a content-defined privileged anchor. Measured with respect to this anchor, residual-stream variation is geometrically and behaviorally stratified by proximity to the prediction.
The stratification holds in all eighteen models tested (dense and mixture-of-experts, 7B-120B, base and instruction-tuned). A narrow, scale-invariant prediction interface concentrates readout-relevant structure, while the vast prediction-distal complement expands with model scale. Because the prediction direction sits nearly orthogonal to the principal variance axes, variance-based analyses recover this organization only partly, and the shortfall grows with prompt heterogeneity.
Anchoring reveals a steep geometric gradient: prediction-proximal regions are highly structured and cluster related prompts, while the complement is flatter and anti-discriminates among prompt groups. The interface is a narrow slice but functionally decisive. Disrupting the variance directions closest to the prediction causes immediate divergence and frequent task-frame shifts; disrupting the next level down delays divergence and preserves framing. The complement is weakly readout-aligned per direction yet causally and temporally load-bearing, and behavior is driven by direction rather than magnitude.
These results establish the prediction direction as a privileged anchor distinct from previously described coordinate axes, and give a geometric account of how high-dimensional computation coexists with linear readout. 

---
# Large Language Models Can Follow Instructions, But Not Many at Once: Phase Transitions in Compositional Constraint Satisfaction 

**Authors**: Mariya I. Vasileva  

**Link**: [PDF](https://arxiv.org/pdf/2608.12426)  

**Abstract**: Large language models are increasingly deployed in settings that require simultaneous adherence to multiple explicit constraints - reasoning structure, safety boundaries, output schemas. Individual constraints are handled proficiently, but the compositional regime, where many must hold jointly, remains poorly characterized: how rapidly does performance degrade, what governs the degradation, and can the collapse be mitigated? We introduce Constraint Saturation Evaluation (CSE), a procedurally generated benchmark that systematically varies the number of simultaneous constraints (k), with every constraint scored by a deterministic, rule-based verifier and zero LLM-judge involvement: 15 models, 36 constraint types, 369,753 checks at k=1-12. Three findings emerge. First, per-constraint pass rate decays gradually and predictably, while the chance of satisfying all k constraints collapses - a model passing individual constraints at ~41% at k=8 succeeds on all eight just 5.7% of the time. Second, constraints do not degrade equally: structural constraints lose 2x more baseline capability per added constraint than lexical ones, ordered by a comprehension-maintenance gap that separates constraints requiring sustained tracking from binary decisions immune to composition. Third, failures are nearly independent, which is what makes the accumulation multiplicative; the residual coupling that does exist tracks shared output features rather than pairwise interference - a wrong sentence count fails every constraint that reads it. Reliable instruction following breaks down beyond 5-6 simultaneous constraints: probe-level success falls below 50% at 7 constraints for the strongest model, and at 3 or fewer for 12 of 15. 

---
# From Observation to Intervention: Memory in Brains and Large Language Models 

**Authors**: Morteza Salehjahromi, Shayan A. Zadegan, Amgad Muneer, Jia Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.12377)  

**Abstract**: Brains and large language models (LLMs) are fundamentally different memory systems, but they can be compared through shared functional questions: where memory-related information is represented, how partial cues recover broader associations, how new information is written or updated, and how memory-related states can be perturbed. In biological systems, these questions span synapses, neuronal ensembles, hippocampal-cortical interactions, and plasticity; in LLMs, they span weights, activations, context windows, retrieval systems, and external stores. The comparison is therefore functional and experimental rather than anatomical. Human studies reveal sparse concept responses, temporal binding, rapid association formation, episode-specific coding, and recall-related reactivation, but selective intervention remains limited. Rodent studies provide more selective causal access to learning-related ensembles, whereas human and macaque interventions usually affect broader circuits. LLMs lack lived episodic memory, yet they permit unusually direct and repeatable manipulation of internal states and stored information. We argue that this asymmetry creates a new opportunity. LLMs are not ahead in memory itself, but in experimental access. Their tools may help turn broad questions about retrieval, updating, persistence, reversibility, and unintended effects into sharper biological hypotheses. The productive bridge is to transfer experimental logic, not anatomical parts. 

---
# Diagnostic Foundation for Evaluating LLMs' Research Integrity as Co-Scientists 

**Authors**: Yash Tripathi, Silu Sharma, Sai Sidhanth Manoharan Jayanthi, Shivank Garg, Lin Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.12345)  

**Abstract**: Language models are increasingly deployed as co-scientists, yet their ability to uphold research integrity under institutional pressure remains unmeasured. We introduce IntegrityBench, a benchmark evaluating misconduct classification, ethical action reasoning and artifact-grounded decision making across 36 paired tasks under a 5-level implicit-explicit pressure protocol spanning 3 domains and 4 research stages. Evaluating 18 frontier model variants, we find that under peak pressure, models fail roughly 1 in 3 integrity-critical decisions, and neither scale nor reasoning ability reliably mitigates this. Explicit pressures induce compliance with misconduct, while implicit contextual reframing more often causes over-refusal of legitimate research tasks. Interestingly, models failing to classify research requests accurately perform equally or better on artifact-grounded decision making (85.7 vs. 79.4), suggesting the three facets are structurally dissociated and correct ethical action does not require accurate classification. Frontier models can thus appear helpful while harbouring integrity failures that create two distinct deployment risks: facilitating research misconduct and eroding trust in AI-assisted research. 

---
# Position: Reasoning is a Learnable Rule-Based Process 

**Authors**: Rachel Lawrence, Jacqueline Maasch  

**Link**: [PDF](https://arxiv.org/pdf/2608.12325)  

**Abstract**: Autonomous reasoning is among the most scientifically and economically motivating topics in AI today. Historically the purview of symbolic AI, recent advances have mainly emerged from deep probabilistic generative models. Despite immense interest and rapid progress, the generative AI community has not clearly converged on operational definitions for reasoning and often implicitly rejects the historical treatment of this topic in logic and verifiable automated reasoning. This position contends that definitional ambiguity leaves the construct validity of reasoning evaluation unverifiable, undermining quantifiable progress toward trustworthy autonomous reasoning. We also contend that this ambiguity is addressable. To that end, we provide (1) operational definitions based on a synthesis of the literature, positioning valid and sound reasoning as a learnable rule-based process; and (2) a checklist for best practices in the communication of AI reasoning research. 

---
# When AI Is Your Pastor: A Benchmark for Theological Triage and Pastoral Guidance in Large Language Models 

**Authors**: Alex Chao  

**Link**: [PDF](https://arxiv.org/pdf/2608.12324)  

**Abstract**: People increasingly ask large language models (LLMs) for counsel on questions of faith, doctrine, and pastoral care. These questions are not ordinary information requests. Some ask about core Christian beliefs, some ask about real disagreements among faithful traditions, some require humility because the issue is prudential, and some are pastoral situations where safety and human referral matter more than theological completeness. Existing benchmarks do not evaluate this structure. We introduce FMG-Bench, the Faith & Moral Guidance Benchmark, a 120-scenario benchmark for evaluating large language model behavior in English-language Christian theological triage and pastoral guidance contexts. FMG-Bench v1 evaluates 14 advanced models across 8,792 scored responses, comparing raw model behavior with three guided instruction settings. In our production run, placing models inside a structured harness improves over raw model behavior by +3.96 points on average, with every model improving. The most safety-critical finding is a +10.8 point gain in escalation appropriateness -- whether AI systems recognize when pastoral, clinical, legal, or emergency support is needed. The guided settings also improve robustness, meaning consistency when questions are reworded or pressured (92.88 to 98.02 stability). Asking a model to compare perspectives helps in secondary-doctrine questions but can be counterproductive when applied to primary doctrine or urgent pastoral situations. The benchmark is a measurement tool, not an endorsement of AI systems as pastoral authorities. 

---
