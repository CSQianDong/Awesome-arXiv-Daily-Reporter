# HiVe-Mem: Interleaving Horizontal Memory Expansion and Vertical Evidence Composition for Multimodal Agents 

**Authors**: Haozhen Zhang, Bohan Liu, Jianzhu Bao, Qingyuan Liu, Haodong Yue, Xiao Li, Zi Pong Lim, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.12872)  

**Abstract**: As multimodal agents accumulate increasingly long interaction histories, effectively retrieving and utilizing past information becomes critical for long-term interaction. Existing memory systems typically rely on semantic retrieval over compressed memories and expose predefined representations, which may miss relevant memories and overlook heterogeneous evidence needs across queries. We introduce \textbf{\hivemem}, a multimodal memory framework that interleaves \textbf{\textsc{H}}or\textbf{\textsc{i}}zontal Memory Expansion with \textbf{\textsc{Ve}}rtical Evidence Composition. For horizontal memory expansion, \hivemem organizes memory episodes into a relational graph using textual and visual anchors, expanding semantically retrieved seeds with related neighbors to recover complementary information. For vertical evidence composition, each retrieved memory retains multiple evidence forms, including compact summaries, captions, grounded visual views, raw images, and interaction history. A lightweight trainable selector independently composes an evidence subset for each retrieved memory, enabling query-adaptive utilization of heterogeneous multimodal information. This two-dimensional design jointly broadens memory coverage and adapts the evidence contributed by individual memories. Experiments across three multimodal memory benchmarks show that \hivemem consistently improves task performance across base models while reducing token consumption by about 90\% compared with the most efficient competing multimodal baselines. 

---
# Is Persona Generalization Predictable? 

**Authors**: Olina Mukherjee, Arnav Goel, Aditi Raghunathan, Jacob Mitchell Springer  

**Link**: [PDF](https://arxiv.org/pdf/2611.12864)  

**Abstract**: Persona generalization occurs when fine-tuning a language model in one domain induces behavioral changes in others -- for example, the broadly unsafe behavior observed in emergent misalignment. We discover a surprising "on-the-line" phenomenon in persona generalization: in-domain and out-of-domain persona rates across various fine-tuning hyperparameters follow an approximately probit-linear relationship, and the specific probit-linear relationship is surprisingly shared even across model families and model sizes, including Qwen and Llama. Training choices that leave the set of trainable parameters unchanged, such as learning rate, weight decay, and training duration, largely move models along the same line, reducing transfer mainly by weakening in-domain learning. Changing which layers are fine-tuned, however, can change the slope. For unsafe behavior, fine-tuning only later layers yields shallower slopes, retaining strong in-domain learning while reducing out-of-domain misalignment. This "on-the-line" phenomenon unifies unexpected, isolated observations of persona generalization within a quantitative framework for predicting and controlling behavioral transfer. 

---
# WorkflowArena: Benchmarking Executable Workflow Orchestration in LLM-Based Multi-Agent Systems 

**Authors**: Zheng Jiang, Fang Niu, Bo Peng, Houde Qian, Wenyi Li, Deyao Hong, Mingju Gao, Yizhe Chi, Youjie Zheng, Kaisen Yang, Calvin Xiao, Lifeng Sun, Huan-ang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.12856)  

**Abstract**: Large language model agents are increasingly deployed as multi-agent systems rather than as a standalone model operating within a single context. Recent agent harnesses allow the model to instantiate worker agents and to author the executable workflow that coordinates them, shifting the design of system structure from developers to the model at run time. The quality of this organization therefore constitutes a distinct capability, yet existing benchmarks either measure end-to-end task completion or evaluate coordination under structures specified in advance. We formalize this capability as workflow orchestration and introduce WorkflowArena, a benchmark of 145 long-horizon tasks spanning 8 professional domains, from software engineering and formal mathematics to legal analysis and 3D production. Each task specifies an initial workspace and an integrated deliverable composed of interdependent components, which is assessed by a hidden verifier. Agents determine autonomously whether and how to delegate, and all delegation is recorded, enabling joint evaluation of deliverable quality and the organization of the underlying work. Experiments with nine models show that no single configuration performs best across all domains and that systems exhibit distinct delegation strategies: at high effort, Claude models in Claude Code invoke workflows less frequently but with larger worker groups, whereas GPT models in Codex delegate more frequently in smaller units. When the executing model is held fixed on a challenging subset of tasks, workflows authored by Claude Opus 5.5 substantially improve weaker executors, indicating that workflow organization is itself a measurable determinant of task performance. 

---
# How Well Does AI Peer Review Work? A Benchmark Built from 100 Planted Errors 

**Authors**: Paul Litvak  

**Link**: [PDF](https://arxiv.org/pdf/2611.12820)  

**Abstract**: One hundred known errors were planted into 10 open-access psychology papers, and the modified papers were then run through frontier models and two commercial AI review tools. The best single system caught 71 of the 100 errors, while the worst caught 30. Pooling every system's output caught 93. Models are only partly correlated in the errors they find, which makes ensembling a big lever for finding issues in papers. Seven errors could not be caught by any system, and all seven were omissions: information deleted from a paper rather than mistakes inserted into it. this http URL contributes more unique catches than any other single system, though it is expensive on a per-error basis. False positives were not measured, and how this error distribution compares to the distribution of errors in real papers remains unknown. The papers, errors, model outputs, and full experiment log are public, so that others can build on this toward a comprehensive evaluation benchmark across disciplines. 

---
# Large Language Models as a Universal Searcher for Open-World Automated Machine Learning 

**Authors**: Yuxuan Hu, Jihao Liu, Weikang Shi, Xijie Huang, Linda Wei, Mingjie Zhan, Aojun Zhou, Rui Liu, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.12791)  

**Abstract**: Developing effective machine learning models requires complex manual tuning, while current Automated Machine Learning (AutoML) frameworks remain domain-specific and isolate model architecture, training hyperparameter, and data ratio optimization. Here, we bridge this gap by proposing a universal, large language model (LLM)-based AutoML framework that jointly optimizes these configurations for the first time without the need for task-specific fine-tuning. We propose NCode, a unified numerical encoding scheme that translates diverse AutoML design choices into universal discrete sequences and effectively reframes the optimization process as a unified ranking task. Trained solely on optimization trajectories derived from a limited, single-domain vision task, the proposed LLM-based searcher demonstrates impressive cross-task, cross-modality and joint search-space optimization generalizability, achieving state-of-the-art performance across image classification, image segmentation, speaker verification and text-to-image benchmarks. Crucially, it exhibits superior scalability in the large-scale training and joint optimization of a billion-parameter Diffusion Transformer, requiring 896 GPU-days and attaining a state-of-the-art GenEval score of 0.91. By leveraging reinforcement learning to instill Chain-of-Thought reasoning, we provide a systematic way to interpret the LLM-based searcher's internal decision-making mechanisms, revealing its optimization strategy shifted from the early-stage diversity-driven exploration to the late-stage anchor exploitation. Our proposed framework provides a scalable, generalizable, and interpretable pathway toward general-purpose open-world machine learning optimization. 

---
# PORT: Preference Optimization via Robust Token-Level Reweighting 

**Authors**: Ding Zhu, Xiukun Wei, Tian Xie, Zhihui Zhu, Xueru Zhang, Mohammad Mahdi Khalili  

**Link**: [PDF](https://arxiv.org/pdf/2611.12763)  

**Abstract**: Preference optimization has become a central approach for aligning large language models with human values, but its effectiveness depends heavily on the quality of preference annotations. In practice, preference data is often noisy due to annotation errors and ambiguity. Existing robust preference optimization methods primarily operate at the sequence level, implicitly treating entire responses as uniformly correct or incorrect. However, rejected responses may still contain informative reasoning steps, while preferred responses can include subtle errors. In this work, we propose \textbf{P}reference \textbf{O}ptimization via \textbf{R}obust \textbf{T}oken-level reweighting \textbf{(PORT)}, a fine-grained framework for robust alignment under noisy preferences. PORT performs token-level reweighting using the empirical cumulative distribution function (CDF) of token log-probabilities, yielding an efficient forward-pass-only proxy to gradient-norm penalization that selectively suppresses corrupted tokens without additional backward-pass computation. We provide theoretical analysis showing that PORT reduces the gradient bias from noisy preference labels by minimizing an upper bound on the bias term. Extensive experiments across diverse noise settings demonstrate that PORT consistently improves robustness and outperforms existing sequence-level baselines. The code is available at this https URL 

---
# The Narrative Gap: Can LLMs Help Us Navigate Diverse Narratives Across Languages? 

**Authors**: Nikhil Sharma, Kelly Marchisio, Kenton Murray, Ziang Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.12750)  

**Abstract**: Large language models (LLMs) increasingly mediate how people seek and consume information, including accounts that differ across languages and regions, especially on contested topics. Supporting diverse information seeking requires models to help users navigate competing narratives across languages and information streams. We define six capabilities necessary for diverse multilingual information seeking: three foundational capabilities for narrative inclusion, narrative exclusion, and missing narrative identification; and three higher-level capabilities for presenting diverse narratives, preserving narrative coverage under confirmatory queries, and maintaining information parity across languages. To guide the development of those information seeking capabilities, we introduce \textsc{NarrativeBench}, a benchmark comprising 1,333 news articles in 23 publication languages across 23 contested topics and 460 Reddit-grounded queries. Evaluating 14 LLMs across 24 query languages, we find that current models lack the capabilities needed to reliably navigate diverse narratives, limiting users' control over the perspectives they receive and leaving substantial gaps in exposure to competing accounts even when diversity is explicitly requested. Information access also differs across query framings and languages, risking echo chambers, linguistic filter bubbles, and information disparities that weaken the common ground needed for democratic discourse. \textsc{NarrativeBench} provides a foundation for developing information systems that give users across languages and regions agency to navigate diverse narratives. 

---
# A local language model layered with rule-based and neural detectors for clinical text de-identification: an open-source toolkit and evaluation 

**Authors**: Valentin Burkin  

**Link**: [PDF](https://arxiv.org/pdf/2611.12709)  

**Abstract**: Objectives: To measure whether a locally hosted large language model (LLM), stacked with deterministic and clinical named-entity recognition (NER) detectors, finds more protected health information (PHI) than open tools, and at what precision.
Materials and Methods: DeidSharp, an open-source .NET toolkit, combines the findings of pluggable detectors (patterns, a clinical NER model, a local 26B mixture-of-experts LLM) and fails closed on any detector error. Configurations fixed on tuning halves were compared, under a prespecified protocol, with deid, Philter, the Stanford/Penn de-identifier, OpenAI Privacy Filter and GLiNER on reporting halves of PhysioNet's nursing notes (889 spans) and semi-synthetic MIMIC-IV-Note discharge summaries (29,023 spans), and on synthetic ASQ-PHI queries, with paired patient-level bootstrap intervals.
Results: In the nursing notes, against Stanford/Penn, the recall-first stack found 98.9% of PHI (+2.8, 95% CI 1.4 to 4.4) at 9.3 points lower precision; the precision-first stack, without NER, found 96.5%, matching Stanford/Penn, at 19.6 points higher precision (83.2%). In the discharge summaries, both stacks led Stanford/Penn by more than 20 points (recall-first 95.3%, +21.5, 20.5 to 22.4), almost entirely on generated untyped names; on typed labels all three exceeded 99%. All four prespecified comparisons held. No configuration beat deid significantly on its home corpus; Philter found more discharge-summary spans than the recall-first stack at 23 points lower precision.
Discussion: The LLM was the main detector and NER added place names; asked to verify its own findings, the LLM discarded a quarter of true PHI.
Conclusion: A local LLM layered with deterministic detectors offers recall-first and precision-first operating points without sending text off-site. 

---
# A Lattice-Based HMM Approach to Aruz Scansion of Ottoman Poetry 

**Authors**: Mehmet Emre Tiryaki  

**Link**: [PDF](https://arxiv.org/pdf/2611.12691)  

**Abstract**: Ottoman Divan poetry is composed in aruz, a quantitative meter based on the weights of syllables. The meter is sensitive to specific characters recorded in the original writing in the Arabic script. This work is concerned with automatic meter finding and scansion alignment for Ottoman Divan poetry, especially on modern Turkish spelling without the special diacritics that scholarly transliterations of Ottoman preserve. We define two tasks on this input, first identifying the meter used in the poem, and second aligning the meter's weights with the syllabification of the poem, identifying the flaws and freedoms the poet has taken along the way. We build a corpus of 52,000 lines split by poet, with scansion annotations for 500 test lines, and a robustness benchmark generated at the deviation rates measured on real web and OCR copies. Our model is a hidden Markov model over the positions of each meter, decoded over a lattice of every candidate path, and trained by EM from the printed meter, without scansion labels. On a held-out set it scans 77.6% of the test lines fully correctly, against 26.8% for the available rule-based scanner, 55.0% for a neural tagger trained with the same supervision and 63.6% for Claude Opus 5.5 few-shot. The code is open-sourced and the tool is released on the web. 

---
# How Far Does Second-Order Linear Response Predict Quantization Mismatch? Pre-Registered Measurements on Five Language Models 

**Authors**: Dezhen Lu, Zixiang Ni  

**Link**: [PDF](https://arxiv.org/pdf/2611.12634)  

**Abstract**: Serving systems choose a weight format for each model and deployment. A low-cost screen predicts the resulting KL divergence from the unquantized model's Fisher information using $\tfrac12\Delta^\top F\Delta$, the expansion underlying Hessian-aware and second-order post-training quantizers. We assess the reliability of this screen in GPU batches with channel-wise int8, per-tensor fp8 and group-wise int4 on Qwen3-8B, Qwen3-4B, OLMo-2-7B, Qwen3-32B and Qwen3-30B-A3B. Relative predictions hold across a 16-fold range of perturbation scales, int4 group sizes and rounding modes, while absolute predictions are heavy-tailed. A trimmed average brings 10 of 12 cells on Qwen3-4B and OLMo-2 within 15% of measured KL. The MoE model exhibits nonquadratic response at small perturbations, with int8 KL scaling as $\alpha^{0.63}$. Computing per-sequence gradients in bf16 can cause fourfold prediction errors. We trace these errors to arithmetic in the lower half of the backward pass and eliminate them by using fp32 there. Batch-invariant kernels run at 0.55 times the throughput, while changes in reduction order introduce a floor of 0.17-0.22 nats per sequence. We build a screener with one streamed fp32 Fisher pass, trimmed tensor-block predictions with a band, a flag for aligned tensors and per-layer format allocation. On the unseen Mistral-7B-v0.3 family, all six predictions fall within the band, all three KL budgets are met, and vLLM throughput is predicted within 0.4%. 

---
# Adapting Small Language Models to Romanized Nepali: Tokenizer Fertility, Reference Likelihood, and Generation Metrics 

**Authors**: Pujan Pant, Sagar Newpane, Hari Sitaula, Manish Sakhakarmy  

**Link**: [PDF](https://arxiv.org/pdf/2611.12628)  

**Abstract**: Romanized Nepali, Nepali written in Latin script, is common in digital communication but has no standard spelling and few resources, and it is unclear how reliably small open language models can be adapted to it. We adapt three 2-3B base models from different families, Llama-3.2-3B, Qwen2.5-3B, and Gemma-2-2B, with QLoRA on 9,000 instruction examples produced by deterministic transliteration of Nepali instruction data. Each model is trained with one shared recipe and three seeds and evaluated on 1,000 held-out examples with reference-overlap metrics, byte-normalized reference likelihood (bits per byte, BPB), and an output audit, against zero-shot baselines under two prompts. Fine-tuning raises chrF++ from 10-14 to 29.0-30.6 and lowers BPB from 2.23-2.41 to 0.78-0.84; most zero-shot outputs (54-92%) are English-like. Likelihood and overlap disagree on how far apart the models are. Llama-3.2-3B has the lowest BPB in every seed, 0.042 below Qwen2.5-3B (95% CI 0.039-0.046), yet the two cannot be separated on chrF++ (30.51 and 30.61) or BLEU, even with paired item-level tests. Gemma-2-2B, whose tokenizer splits Romanized Nepali words least (2.72 tokens per word), scores worst on every measure. Which model gains most over its zero-shot baseline depends on the zero-shot prompt. In a supplementary comparison with an identical tokenizer, Llama-3.1-8B lowers BPB by 0.053 (0.050-0.056) relative to Llama-3.2-3B, while chrF++ and BLEU rise by only 0.68 and 0.58 points. Lower fertility and higher reference likelihood therefore did not identify the model whose outputs were closest to the references. 

---
# Unmasking Four Knapsack Problems in the Traveling Thief Benchmark 

**Authors**: Mohammed Mizan Alanazi, Mahdi Khemakhem  

**Link**: [PDF](https://arxiv.org/pdf/2611.12597)  

**Abstract**: The traveling thief problem (TTP) asks a thief to choose a tour and a packing together, and the benchmark of 9,720 instances of Polyakovskiy et al. (2014) is the standard test set for comparing TTP heuristics. We show that four of these instances are not TTP instances at all. Their files give a renting rate of 0.00, so the tour has no effect on the objective and each instance reduces to a 0-1 knapsack problem. One of them belongs to the 60 categorized instances on which new methods are usually compared, and on it five of nine published methods report exactly the same value. The cause is the format of the files, which keep the renting rate with two decimals, while the rule of the generator gives a rate close to 0.003 on these instances. The same rounding may shift the rate of 949 other instances by more than 1%. Since the tour and the packing used by the generator were not published, the original rates cannot be recovered. We therefore define new rates with the same rule, applied to a fully specified reference solution that we publish, and we show with a published heuristic that the corrected instances behave as TTP instances again. We also give reference results on the corrected instances and recommendations for future studies. Finally, we present ttplib, a new open-source Python library that loads any of the 9,720 instances, including the corrected ones, without downloading the whole benchmark. 

---
# Fairness in Explainable AI Narratives: Decomposing Social Disparities in LLM-Generated Explanations 

**Authors**: Wannes Van den Bulck, Sofie Goethals, David Martens  

**Link**: [PDF](https://arxiv.org/pdf/2611.12592)  

**Abstract**: Large language models (LLMs) are increasingly used to translate technical explainable AI (XAI) artefacts such as SHAP into explanatory narratives (XAINs) for lay users. In high-risk domains, these explanations are often the main resource through which individuals can understand and contest adverse decisions. Yet LLMs are known to reproduce social bias, even from proxy variables alone, and fairness in XAINs remains unstudied. Because XAINs are bound to a ground-truth artefact, fairness concerns not only what a narrative says and how, but also whether it is equally faithful across social groups. We introduce fairness in XAINs as a new field of research and propose a framework that generates narratives under three conditions (protected attribute disclosed, withheld, or counterfactually swapped), allowing five comparisons that pinpoint the cause of disparities. We apply it to adversely classified instances from four datasets in credit scoring, education and employment, generating narratives with five LLMs, and evaluate faithfulness, feature coverage and sentiment. We find significant disparities across all datasets, metric families and models, arising in two ways. Faithfulness disparities result from feature values: LLMs make plausible but false statements, triggered by values that are unequally distributed across social groups. Feature coverage and sentiment disparities stem from the protected attribute itself: once it is disclosed, LLMs give less explanation to, or frame more negatively the narratives of, groups that have historically been discriminated against or are stereotypically associated with worse outcomes. A more faithful model removes the faithfulness disparities, and withholding protected attributes mostly removes the others, but neither suffices on its own. Fairness in XAINs should therefore be audited separately from the predictive model and its XAI artefact. 

---
# AgentOPSD: Turn-Level Credit Assignment via Recursive Belief Revision 

**Authors**: Yizhuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.12587)  

**Abstract**: Reinforcement learning with verifiable rewards ordinarily spreads a single advantage over a whole trajectory, a granularity too coarse to isolate the few decisions on which success in a long, multi-turn interaction actually turns. Privileged self-distillation supplies denser supervision, but it leaves unanswered how a set of local teacher-student discrepancies should be converted into credit along a sequence. We introduce AgentOPSD, a critic-free scheme that assigns credit recursively, one interaction turn at a time. Inside a turn, the token-level log-probability gaps between a skill-conditioned self-teacher and the student are summed, and that sum acts as evidence in a recursive Bayesian update carried out in log-odds space; the resulting revisions reallocate the outcome-derived advantage across turns through a multiplier that is bounded and preserves its sign. With Qwen2.5-3B and Qwen2.5-7B as policy backbones on ALFWorld, the method improves consistently on GRPO and on self-distillation baselines, and attains 89.1% success at the 7B scale. Ablations attribute the gain to aggregating evidence at turn boundaries and to making the belief revision depend on history, rather than to the local gap by itself. 

---
# Language Defaults Before Content: A Causal Layer Map of Multilingual Generation 

**Authors**: Shenbin Qian, Yves Scherrer  

**Link**: [PDF](https://arxiv.org/pdf/2611.12538)  

**Abstract**: Multilingual generation requires large language models (LLMs) to determine both what to say and in what language to say it, yet these two commitments are difficult to disentangle from model outputs alone. We use translation as a controlled probe: for multi-parallel segments that a model translates well in one language direction but poorly in another, content is held fixed while language varies. Using activation patching, we write residual-stream states from successful runs into failing ones and compare content-matched states with content-independent controls, allowing us to trace content-specific and language-related signals across layers. Across three LLMs of different scales and architectures, we find a consistent three-stage pattern of multilingual generation. Intermediate layers first express a default English signal; content information becomes causally available one layer later; and the final target language is resolved only at later layers. The ordering of the English default before content also appears in a reading-comprehension task. Our sparse autoencoder analyses reveal a further dissociation: the intermediate language signal is only weakly represented by sparse features, whereas a later language distinction is clearly sparse, but resists intervention at the patched position. These results provide a mechanistic account of how language selection and content information emerge and interact during multilingual generation. 

---
# Text Grouping Changes Completion-Time Comparisons in Long-Form Simultaneous Speech Translation 

**Authors**: Hieu Minh Truong, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2611.12501)  

**Abstract**: Does grouping translation updates before speech synthesis change the measured completion gap between two translators? We compare StreamAtt with its fixed-audio-history baseline on ten English-German talks. For each translation, we keep the intended text and its availability times fixed and vary how updates are grouped for synthesis. Four-second grouping increases StreamAtt's mean completion lead from 106.6 to 229.1 seconds with XTTS-v2, but reduces it from 169.5 to 146.5 seconds with Piper. Both produce shorter audio; different changes in playback idle account for the opposite effects. These directions recur on a separately selected twenty-talk panel. In continuing CosyVoice 3 sessions, grouping instead leaves paired audio unchanged and delays completion on nineteen of twenty blocks. A speech-timing comparison therefore needs to specify how text enters synthesis, not just which synthesizer is used. Recognition and automatic word timing describe the outputs, but do not establish translation accuracy or intelligibility. 

---
# Detecting AI-Authored Text in Authentic Student Writing: Evidence from PISA 2025 

**Authors**: Nils-Jonathan Schaller, Thorben Jansen, Olaf Köller, Andrea Horbach  

**Link**: [PDF](https://arxiv.org/pdf/2611.12487)  

**Abstract**: AI-text detectors reach near-perfect scores in evaluations that contrast fully human-written texts with texts generated from researcher-designed prompts. In authentic student writing, however, AI use is voluntary and often partial. We study this setting with the PISA 2025 FLA dataset, in which 15-year-olds in Germany wrote English argumentative essays with and without access to a gpt-4o-mini-based chatbot. Keystroke, clipboard and chat logs let us trace every character of the final essay to its origin. From this provenance, we derive text- and sentence-level labels for human-written, AI-generated and mixed text, and for text written after chatbot use without traceable AI material. We evaluate a feature-based detector and a classifier on frozen Llama-3.1-8B embeddings, both adapted from top-ranked shared-task systems. Both separate AI from human text reliably (macro-F1 up to 0.96 for texts and 0.95 for sentences), independently of spelling errors. The feature-based detector, however, relies on lexical complexity, which a simple student-style rewriting prompt removes (AI recall 0.92 $\to$ 0.60 with gpt-4o-mini, 0.41 with Qwen3). The embedding-based detector stays robust for whole texts (AI recall $\geq$ 0.94) but not for single sentences (AI recall as low as 0.59). Mixed text falls between the two classes, and text without traceable AI material is classified as human-written in about 95\% of texts. 

---
# Phrase-Level Speculative Decoding on a Frozen Target 

**Authors**: Madhusudhana Naidu  

**Link**: [PDF](https://arxiv.org/pdf/2611.12464)  

**Abstract**: Speculative decoders raise throughput by banking more target tokens per forward pass, and Medusa, EAGLE-3, DFlash and DSpark all do it the same way: more candidates that survive verification. All of them draft in the target's own vocabulary, so one node of the candidate tree is one target token, and the tokenizer makes that unit look like a constant. It is not. BPE's pretokenizer splits on whitespace before merges are learned, so no multi-word string can ever become a single token: in Qwen3-4B, "United States of America" is four tokens and no adjacent pair of them exists as one entry, so a phrase is out of reach of any method drafting in the target's vocabulary.
We change the unit. A small MLP head reads the frozen target's hidden state and predicts a one- to three-word phrase as one entry in a separate draft vocabulary; greedy longest-match expansion maps it back into the target's own tokens, and N candidates are verified in one forward pass behind a prefix-tree mask, so the change is orthogonal to tree depth and width. The target and its tokenizer stay frozen and a candidate is accepted only while it matches the target's own argmax, so the served output is identical to greedy decoding up to floating-point tie-breaking.
On one A5000 at batch 1, against HuggingFace generate(), a 21.94M head (0.545% of Qwen3-4B) reaches 2.391x on GSM8K, 2.444x on MATH-500 and 1.544x on MT-Bench. Because the head's output layer is one row of 257 parameters per entry, the draft vocabulary size is the parameter budget: an 8,000-entry head is 3.434M parameters, 0.0854% of the target and 64x smaller than EAGLE-3's drafter, and keeps 92.3% of the larger head's GSM8K tokens per pass. With EAGLE-3 run on same prompts and baseline, our 21.94M head is ahead of tree-16 on both maths sets and behind tree-60 on every set. 

---
# Can Frontier Auto Researchers Reconstruct Human Research Decisions in AI? 

**Authors**: Shengyun Si, Junjie Chu, Ye Leng, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.12429)  

**Abstract**: Scientific research has long been driven by substantial human effort, from reviewing prior work and identifying research gaps to designing specific solutions. Recent advances in harness-based agent systems have given rise to increasingly capable Auto Researchers, which can autonomously carry out these tasks and, increasingly, entire end-to-end research workflows. Yet despite their rapidly growing capabilities, how closely the research decisions and outputs produced by auto researchers align with those of human experts remains underexplored. We investigate this gap through the lens of research reconstruction and study: Given comparable research context, how similar are the research decisions and outputs produced by Auto Researchers to those of human researchers in AI research? To answer this question, we introduce AutoResearcher-Human Research Similarity (AutoHRS), a stage-wise evaluation framework for assessing how faithfully auto researchers reconstruct human research artifacts across different stages of the scientific workflow. Using 152 ICLR oral papers as our corpus, we generate 456 research trajectories spanning different stages of the research process. Applying AutoHRS to these trajectories, we find that current Auto Researchers frequently fail to recover key research decisions and their resulting artifacts across multiple stages of the research process. 

---
# Path-Level Structural and Semantic Alignment for Hierarchical Multi-Label Text Classification 

**Authors**: Xindi Wang, Aifen Xiong, Baiyang Wang, Xuri Ge, Robert Mercer, Frank Rudzicz, Yifan Peng, Zhumin Chen, Hongjun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.12420)  

**Abstract**: Hierarchical multi-label text classification (HMTC) requires models to capture both structural dependencies among labels and the fine-grained semantics of labels at different hierarchy levels. Existing methods often model hierarchy through prediction constraints or representations of individual labels and local relations. Structural and semantic supervision, however, are usually formulated separately. We instead use complete root-to-node label paths as a shared unit for both. Our framework introduces two auxiliary training objectives while retaining a standard multi-label classifier at inference. PathSimNCE aligns document representations with hierarchical paths using hierarchy-aware contrastive targets. These targets provide graded supervision for structurally related paths while preserving explicit supervision for the ground-truth paths. The complementary semantic objective uses LLM-generated, document-conditioned descriptions of the same positive paths. It thereby provides context-specific label semantics beyond static label representations. Experiments on three benchmarks show consistent improvements: structural alignment is particularly beneficial in deeper hierarchies, while semantic alignment provides additional gains for fine-grained labels. 

---
# BDBBQ: A Human-AI Collaborative Benchmark for Bengali Social Bias in LLMs 

**Authors**: Provat Saha, Ashika Habib Khan, Tahseen Tanha, Hasin Ishrak, Muhammad Iqbal Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2611.12407)  

**Abstract**: Despite growing attention to social bias in large language models (LLMs), existing bias benchmarks provide limited coverage of culturally grounded evaluation in Bengali, particularly within Bangladesh's sociocultural context. To address this gap, we develop a bilingual extension of BanglaBBQ comprising 61,312 question-answering instances across 12 social categories in English and Bengali. The benchmark incorporates balanced answer permutations, contrasting question framings, and both ambiguous and disambiguated contexts to systematically examine model performance and stereotype-aligned bias across languages. We validate the dataset through structural checks, 220 statistical tests, and expert annotation of 150 instances by four annotators, achieving strong inter-annotator agreement (Fleiss' \k{appa}=0.8349). We evaluate eight LLMs across 490,496 instances and identify five key findings. First, disambiguated contexts yield substantially greater accuracy improvements in Bengali (+36.59 percentage points) than in English (+17.71). Second, all evaluated models exhibit lower accuracy in Bengali, although Mistral-Nemo shows the smallest cross-lingual accuracy gap (5.12 percentage points). Third, stereotype-aligned bias exhibits language and model dependent variation, increasing in Bengali for Gemma 2 9B but decreasing for Mistral-Nemo. Fourth, higher accuracy does not necessarily correspond to lower stereotype-aligned bias. Finally, bias varies considerably across social categories, with Qwen 2.5 7B exhibiting stronger stereotype alignment in Bengali for Politics and Physical Appearance. These findings demonstrate that cross-lingual performance and social bias are distinct dimensions of LLM behavior and underscore the importance of culturally grounded, language-sensitive benchmarks for evaluating fairness beyond English-centric settings. 

---
# More Than Just Style: Benchmarking Personality Influence in Large Language Models 

**Authors**: Yuhao Liu, Mengyu Wang, Xiao Chen, Lu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.12392)  

**Abstract**: In addition to the effects of explicit input prompts and instructions, large language model outputs are influenced by underlying behavioral preferences similar to human personality tendencies. These tendencies can affect how models interpret situations and make decisions. However, their influence is difficult to disentangle from the effects of model capabilities and instructions. Therefore, we introduce SPACE, a benchmark framework combining a controlled evaluation protocol with complementary datasets for personality verification, capability assessment, and hierarchical situational tasks. Controlled comparisons across guidance conditions help distinguish personality influence from measured capability changes and immediate instruction effects. Experimental results demonstrate that although prompting can shift generation style, models' internal personality tendencies are persistent and not readily overridden by opposing instructions. Furthermore, these underlying tendencies have an observable influence on LLM task performance, highlighting the need for greater attention to these characteristics and potential applications. Training code and SPACE benchmark specifications are available at this https URL. 

---
# MNN-Pocket-0.3B Technical Report 

**Authors**: MNN Team  

**Link**: [PDF](https://arxiv.org/pdf/2611.12311)  

**Abstract**: This technical report presents MNN-Pocket-0.3B, a compact 300M-parameter dense language model designed for efficient on-device deployment and supporting context lengths of up to 32K tokens. With an INT4 deployment footprint of only 173 MB and decoding throughput of up to 466 tokens/s on iPad Pro M5 (Metal), it can be seamlessly integrated into applications on edge devices such as smartphones and smart glasses. MNN-Pocket-0.3B is developed through the co-design of its architecture, data, training pipeline, and inference system. Hardware-aware architecture selection and dependency-aware vocabulary pruning balance model capability with on-device memory and compute constraints. A four-stage pre-training pipeline covering approximately 4.1 trillion high-quality Chinese and English tokens equips the model with broad knowledge and reasoning capabilities. This is followed by a three-stage post-training pipeline comprising instruction fine-tuning, multi-teacher on-policy distillation, and quantization-aware training. Finally, hardware-specific optimizations in the MNN inference engine enable fast, memory-efficient inference and long-context processing. Extensive evaluations demonstrate that MNN-Pocket-0.3B achieves frontier-level performance among models of comparable size, with particularly strong results in Chinese and English knowledge and reasoning, mathematics, coding, and instruction following. It consistently outperforms similarly sized open-source models and even surpasses several 0.5B-parameter models on instruction-following benchmarks. We release MNN-Pocket-0.3B with open weights and ready-to-use MNN deployment packages. We hope it will provide a practical foundation for capable, responsive, and memory-efficient edge applications while advancing research and development in on-device language models. 

---
# Prune, Don't Merge: Expert Compression of Mixture-of-Experts LLMs on Multilingual Production Tasks 

**Authors**: Tomas Rasymas, Povilas Kvedaras  

**Link**: [PDF](https://arxiv.org/pdf/2611.12240)  

**Abstract**: Mixture-of-Experts language models are cheap to run per token but expensive to hold in GPU memory, which drives the cost of serving them in production. One-shot expert compression shrinks them after a single calibration pass. Its two current state-of-the-art methods are REAP for expert pruning, which deletes the least salient experts, and REAM for expert merging, which merges them into the most salient ones. Both have been evaluated only on academic benchmarks. We compare the two on five multilingual production tasks of a web-hosting company, from classification to the generation of marketing HTML email templates, with two 30B-parameter models (Qwen3-30B-A3B-Instruct-2507 and the hybrid Mamba-attention Nemotron-3.5-Lightning) at 25% and 50% of experts removed and the same calibration data. Pruning wins: on the same rows, REAP is significantly better in 16 of 20 comparisons and REAM in 2. At 25%, REAP keeps 93.6% of Qwen3-30B's generative quality and REAM 62.5%, although on benchmarks the two are within a point for the same model. The damage concentrates in the output contract rather than the content: 89.8-95.2% of each significant loss on email templates comes from hard-gate failures such as invalid JSON, broken mobile layouts and runaway generation, and most failed non-English domain descriptions copy the prompt's English opening phrase. Cross-perplexity and logit divergence, two diagnostics from the REAP paper, also track production quality, as does the rate of repetition loops; n-gram diversity does not. We close with practical guidance for compressing MoE models for production. 

---
# TrajLift: Lifting Memory Recall Trajectories with Latency-Modulated Heat Diffusion 

**Authors**: Jiawen Kang, Jinchao Li, Mingyu Cui, Junan Li, Xixin Wu, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2611.12197)  

**Abstract**: Verbal memory retrieval carries rich diagnostic information for Alzheimer's disease through the order, hesitation, and categorical organization of recalled words, yet computational modeling of this retrieval process remains largely unexplored. A key challenge is that timing and hierarchical structure are diagnostically entangled: the diagnostic meaning of retrieval timing depends on its hierarchical context. We propose TrajLift, a spectral framework that models verbal recall as timed trajectories on a hierarchical graph. Our central mechanism is latency-modulated heat diffusion on the semantic hierarchy, a unified operator that lifts each recalled word into a representation of both multi-scale hierarchical structure and continuous retrieval timing. Formal analysis shows that the resulting representation is first-order separable into structural and temporal components with spectral selectivity across hierarchical levels. On synthetic benchmarks requiring structure-time interaction, TrajLift stays near-perfect on deep hierarchies where all baselines fail. On real-world clinical data, we show that TrajLift outperforms diverse baselines by jointly modeling timing and semantic structure, with gains that grow as retrieval timing becomes more informative. 

---
# The Deception Archive as a repository of standardized datasets for large-scale verbal deception detection research 

**Authors**: Caterina Borgese, Riccardo Loconte, Lucca Pfründer, Bennett Kleinberg  

**Link**: [PDF](https://arxiv.org/pdf/2611.12196)  

**Abstract**: Computational advances in natural language processing and machine learning have substantially expanded the use of automated approaches in verbal deception research. However, existing verbal deception datasets remain fragmented across disciplines, as they are often difficult to identify, access, and reuse, limiting cross-study comparisons and the development of robust computational models. We present the Deception Archive, a curated and standardized repository of publicly available verbal deception datasets built upon a large-scale systematic review. Eligible datasets were retrieved, manually inspected, and standardized following a structured data curation workflow. The current repository comprises 41 publicly available datasets containing 387,995 truthful or deceptive statements. The archive represents a heterogeneous repository (e.g., in data size, topic of investigation, ground-truth, etc.) and a living resource for integrating newly available deception datasets over time. By providing centralized access to standardized datasets and accompanying metadata, the Deception Archive facilitates dataset discovery, cross-dataset validation, and reproducible evaluation of computational models. 

---
# Blind Variation, Selective Retention: Improving Creative Idea Generation in AI Scientists 

**Authors**: Guowei Xu, Jiaru Zou, Mete Erdogan, Haotian Ye, Pan Lu, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2611.12184)  

**Abstract**: Scientific discovery requires ideas that are original, useful, and non-obvious. Yet AI scientists are typically given as much task information as possible during idea generation, which can anchor generation toward familiar approaches. Inspired by the cognitive framework of Blind Variation and Selective Retention (BVSR), we propose a discovery framework that separates these two roles of information: it generates proposals under deliberately limited task information to broaden exploration, then learns from proposals previously verified on the same task and their outcomes to retain promising candidates for costly empirical evaluation. We evaluate BVSR across five proposer models and 12 scientific tasks. Limiting task information substantially increases proposal originality and non-obviousness, and produces a larger share of highly creative and high-utility proposals than existing state-of-the-art prompting-based baselines. Learning from prior proposal outcomes on the task further improves candidate selection, discarding half of the candidates while retaining most top-performing proposals. When incorporated into existing AI-scientist workflows, BVSR achieves stronger discoveries across gene expression analysis, drug response prediction, anomaly detection, scaling law discovery, and brain activity prediction. These results suggest that effective AI scientists should use information differently across stages of discovery: constrain task-specific information during exploration while exploiting accumulated empirical experience on the task. 

---
# Lost in Rounding: bf16 Recurrent State Makes Hybrid Reasoning Models Think Longer 

**Authors**: Yuchuan Weng, Xin Chen, Chi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.12127)  

**Abstract**: Hybrid reasoning models interleave attention with recurrent mixers such as Mamba-2 and Gated DeltaNet, whose state is cached between decoding steps. Prior work showed that storing this state in FP16 or INT8 with round-to-nearest (RTN) makes Mamba-2 and GDN hybrids more verbose, and that stochastic rounding (SR) fixes it. We study the other 16-bit format that serving engines expose, bf16, on six open hybrid reasoning models and 590 competition-math problems. With bf16 RTN state, five of six models reason longer (geometric-mean length 1.07x to 1.27x; also under greedy decoding in the two models tested) and hit the 32K token cap more often. On Qwen3.5-9B, cap hits rise from 118 to 223 and accuracy falls from 85.6% to 70.8%. On LiveCodeBench, both models tested also reason longer and lose 12.6 and 13.4 points of pass@1. Rerunning the capped items with a 128K budget halves the accuracy gap but does not close it (93.6% vs. 86.6%): bf16 runs that never finish are twice as common, and 85% of them are exact repetition loops. On Nemotron-Nano-9B-v2, by contrast, 30 of 31 extra caps finish at 128K, so the cost there is mostly extra tokens (exploratory). Same-width bf16 SR removes the length increase in four of five affected models, restores Qwen3.5-9B to 87.6% and brings code length and pass@1 near FP32 levels, so the failure is mostly rounding bias, not missing resolution. With an integer-hash random generator, SR on Qwen3.5-9B matches FP32 on cap hits and accuracy (113 vs. 118; 85.6% for both) at the memory and decoding speed of bf16 RTN. In the Nemotron-Nano-v2 lineage, SR on 25% of heads (an intermediate-decay band) removes the effect, while random heads and the top teacher-forced-KL octiles do not; in Nemotron-3 and Qwen3.5-9B the effect is spread across heads. We recommend that evaluations of hybrid models report the state dtype, cap rate and output length alongside accuracy. 

---
# SadaqahStories: A Verified English-Arabic Corpus of Charity Stories for Children, with an Audit of Errors in AI-Drafted Religious Narratives 

**Authors**: Wajdi Zaghouani, Spencer Striker  

**Link**: [PDF](https://arxiv.org/pdf/2611.12078)  

**Abstract**: We present SadaqahStories, a parallel English-Arabic corpus of 41 illustrated four-page stories about charitable giving in Islam (Sadaqah, Zakat and Waqf), written for readers aged 7 to 18. The stories were first drafted with large language model (LLM) assistance, then audited against hadith collections, Qur'an references and historical sources, and finally rewritten for children and signed off by an Islamic scholar. Every story carries an evidence label and links to its sources. We release all three versions of each story (AI-assisted draft, audited text, final child-oriented text) together with the correction log, so the corpus can be used to study how AI drafts of religious and historical content go wrong. The audit found substantive errors in 39 of the 41 drafts. The most frequent were unsupported specific details (dates, figures, prices, names), invented quotations, institutions or anecdotes, and conflation of separate events or people; six drafts contained errors in the wording, grading or ruling of a religious text. Rewriting cut the estimated Flesch-Kincaid grade of the English text from 14.0 to 6.1. We also release an 81-entry charity lexicon covering Modern Standard Arabic and Qatari Arabic, and describe the free, account-free website that serves the corpus (this https URL). 

---
# MIRIX-2.0: Multi-Agent Memory System 

**Authors**: Wei-Chieh Huang, Weizhi Zhang, Jiayan Nan, Peng Xia, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.12068)  

**Abstract**: Long-term memory lets LLM agents personalize, stay consistent, and reuse what worked before. Parametric memories need training, consolidation can lose details for good, and keeping every record leaves a growing store to search. We present MIRIX-2.0, a multi-agent memory system designed to be practical, accurate, and efficient at once. It is the most accurate of the compared systems on four of five QA benchmarks. It reaches 91.4% on LoCoMo and 72.6% on LongMemEval-S, where it leads the strongest baseline by 14.3 points, and with AutoDream disabled it gains 31.0 and 22.0 points over HyperMem on single-hop and multi-hop document QA. As procedural memory, it yields the highest mean success among the compared memory methods on ALFWorld and WebShop and reaches 49.2% on MetaClaw-Bench, against 44.0% for MetaClaw's skill library. Six typed memory agents keep complete records as the system of record, and a provenance hypergraph indexes episodic and semantic records with n-ary frames that cite their rows. Retrieval runs under explicit budgets, so the evidence a question inspects does not grow with the history. AutoDream merges duplicate anchors outside the write path without rewriting any record, and it distills task sessions into skills. With AutoDream disabled, MIRIX-2.0 answers faster than HyperMem on all five timed benchmarks. It needs no fine-tuning, runs with closed and open models, and its code is public. 

---
# SMO: Compiling Natural Language Skills into Graphs of Reusable State-Move-Outcome Bricks 

**Authors**: Chao Chen, Jiashuo Zhang, Yunhang Shen, Liangyu Lian, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2611.12041)  

**Abstract**: Large language model agents increasingly rely on reusable natural-language skills, yet growing skill collections remain accumulations of isolated text artifacts that are difficult to compare, deduplicate, and compose. We argue that the reusable content of a skill lies not in its surface wording but in its procedural transitions: which operation to execute under a given state and which observable outcome to produce. Based on this observation, we propose State-Move-Outcome (SMO), a compilation-like framework that decomposes raw skills into reusable procedural bricks and organizes admitted bricks as an externalized symbolic structure. When compiling a skill, SMO first retrieves and composes existing bricks, dynamically connects them through Outcome-to-State conditions, and creates a candidate brick only for an uncovered procedure; the candidate becomes reusable only after validation and admission. Experiments on SkillsBench v1.1 demonstrate that SMO represents 232 rawSkills from 87 underlying tasks with 95 shared bricks, reducing the corresponding skill representations from 366,472 to 149,154 tokens--a $\mathbf{59.3}\times$ reduction and $\mathbf{2.46}\times$ compression. Across 232 skill-conditioned evaluation instances, SMO raises the average task score from $\mathbf{0.6379}$ to $\mathbf{0.7478}$ and increases 5/5 Pass instances from $\mathbf{138}$ to $\mathbf{164}$, while matching or outperforming rawSkill on $\mathbf{208}$ of $\mathbf{232}$ instances. It also reduces average runtime token consumption by $\mathbf{38.38\%}$. These results show that compiling natural-language skills into shared procedural units can jointly improve skill reuse, representation efficiency, and downstream execution. 

---
# A generalized transition framework for identifying reduction domains in non-binary constituency parsing 

**Authors**: Jungyeul Park, Zihao Huang, Eunkyul Leah Jo  

**Link**: [PDF](https://arxiv.org/pdf/2611.12020)  

**Abstract**: Direct non-binary constituency parsing must identify an ordered sequence of children at each phrase-level reduction. We extend our earlier delimiter-guided bottom-up construction to a common calculus for top-down, in-order, and bottom-up parsing. At reduction, the rightmost delimiter marks the left boundary, the nonterminal marker supplies the mother label, and the completed items in the delimited segment determine the children and arity. We use a stack invariant to prove soundness and completeness and derive the exact length of the static oracle's action sequence. A projection-and-lifting theorem shows that each delimiter-guided system and its canonical counterpart generate the same set of trees under both the structural and grammar-licensed transition relations. In controlled experiments on PTB and CTB, the delimiter-guided parsers achieve similar labeled accuracy to their canonical counterparts across all traversal orders. For bottom-up parsing, delimiter guidance reduces the number of action classes and increases the number of supervised decisions per tree. 

---
# Robust Domain-Mixture Scheduling for Multi-Teacher On-Policy Distillation 

**Authors**: Yankai Chen, Sida Lin, Yifan Li, Hanjun Luo, Hanrong Zhang, Zhuohan Xie, Bowei He, Wei-Chieh Huang, Meng Ding, Wei Xue, Yike Guo, Philip S. Yu, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.12012)  

**Abstract**: Multi-teacher on-policy distillation (MOPD) merges the capabilities of multiple domain-expert teachers, such as mathematical reasoning, code generation, and instruction following, into a single compact student. Since all domains share a limited training budget, how to allocate it across domains directly determines the quality of distillation. Existing methods allocate the budget by each domain's learning progress, but offer no guarantee for lagging domains. We therefore introduce distributionally robust optimization (DRO) into MOPD to prioritize the worst-performing domain. However, the per-domain training signals are estimated from a few rollouts and are inherently noisy, so directly applying DRO mistakes noise for the worst case, lets occasional spikes hijack the budget, and protects inherently hard domains rather than truly regressing ones. To address these issues, we propose RDS-MOPD, which keeps the DRO objective and redesigns its three parts: it shifts the allocation only when differences clearly exceed the noise, bounds the influence of any single anomaly, and defines the worst case as the regression from each domain's own best level, all with theoretical guarantees. Experiments show that RDS-MOPD outperforms recent competing methods and ranks first on most mathematical reasoning, code generation, and instruction-following benchmarks. 

---
# EmoHarness: Boosting Multimodal Emotion Understanding with Meta-Harness Evolving 

**Authors**: Fenghai Li, Bingxuan Li, Jiawen Zhang, Yifan Shen, Jiateng Liu, Yutong Zhang, Yue Guo  

**Link**: [PDF](https://arxiv.org/pdf/2611.11994)  

**Abstract**: Small multimodal models can misjudge emotion when words, voice, and visual cues conflict. We present EmoHarness, which runs such a model, the frozen student, inside a harness: the prompts, settings, and verification code around its calls. A human-written starting harness, the scaffold, numbers the student's visual observations, transcript, and prosody descriptions as citable evidence and has the student weigh candidate emotions against it. In Meta-Harness Evolving (MHE), a stronger LLM, the Builder, reads the student's errors and execution traces offline and edits the harness; static checks and validation gates decide which edits are kept, and the Builder never sees evaluation items. On 5,507 MME-Emotion items, EmoHarness raises task-macro Recognition by 7.6-13.4 points over a single video-only call for all four students and both Builders. Transcripts and prosody descriptions supply most of this boost (+5.38 to +10.71); the scaffold changes the audio-augmented call by -0.79 to +2.54, partly through extra multi-label predictions, and evolved edits add at most +0.95, although no evolved increment survives Holm correction. The harness's emotion context also transfers to IntentQA, with small gains for three of four students, none of which survives correction. EmoHarness thus boosts small multimodal students chiefly by supplying and organizing audio-derived evidence. 

---
# When Audio and Video Mislead: Evaluating Multimodal AI for Conversational Emotion Recognition 

**Authors**: Youngseok Kim, Minjae Kim, Eunsoo Lee, Jihie Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.11987)  

**Abstract**: Multimodal Emotion Recognition in Conversation (MERC) combines lexical, acoustic, visual and conversational cues, but accepting audio and video does not establish their effective or robust use. We audit off-the-shelf Omni models while holding the target transcript and prior textual context fixed and adding, removing or replacing target audio and video. In the primary evaluation of Qwen2.5-Omni-7B, Baichuan-Omni-1.5 and MiniCPM-o 2.6 on MELD and the full five-session six-class IEMOCAP set, T+A improves Accuracy in all six model-dataset pairs, with four gains significant after false discovery rate (FDR) correction. Across three source mappings, unrelated A/V (RandA/V) lowers Accuracy relative to Full in all 18 comparisons; all 18 remain significant under dialogue-cluster inference and BH-FDR correction, both on common-valid pairs and when invalid/OOM outcomes are treated as incorrect. Comparisons with modality-omission baselines further distinguish replacement penalties from mismatch-specific harm. A Qwen/MELD case study finds significantly greater gold-vs-best-candidate margin recovery in Late than Early or Middle under decision-query audio knockout, relative to an edge-count-matched Video-Sham control, using Holm-corrected tests. These results distinguish genuine-modality benefit from sensitivity and vulnerability to misleading input. 

---
# BharatSynthQA: A Context-Grounded Question Answering Dataset on Indian Temples and Religious Heritage 

**Authors**: Riya Naik, Shivam Sharma, Swati Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2611.11952)  

**Abstract**: India's temples and other places of worship are among its richest repositories of cultural heritage, bringing together religious tradition, history, architecture, and art. Due to the widespread curiosity, people increasingly turn to Artificial Intelligence (AI) assistants for information about them. Current models are rarely trained or evaluated on this domain. We present BharatSynthQA, a passage grounded question answering (QA) dataset focused on Indian temples, which also covers other religious sites connected to them. Our pipeline crawls Wikipedia from topics seeded by authentic heritage sources, filters articles for topical relevance, and locations in India. Further it generates Boolean, factoid, and explanatory QA pairs with a fewshot prompted LLM. An LLM judge filters the pairs on answer correctness, retaining 98,677 of 131,494 generated pairs. Additionally, 300 human-annotated pairs are released separately as a reference set. Human evaluation of the subsample of this dataset showed 89.83% of the questions were formed correctly. Benchmarking five current LLMs shows that factoid token F1 is only 0.18-0.30 without the source passage but 0.73-0.77 with it, indicating that this domain is poorly captured in model knowledge. The dataset is available at this https URL to support research on culturally grounded QA and retrieval-augmented generation for Indian heritage. 

---
# Adaptive Entropy-Sparing for Efficient Reasoning 

**Authors**: Cong Jiang, Xiaofeng Zhang, Tom Ko, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.11901)  

**Abstract**: Large Reasoning Models (LRMs) produce explicit chains of thought to improve transparency, but they frequently overthink, generating long yet inefficient reasoning chains that inflate computational costs and risk introducing errors. While numerous reinforcement learning methods struggle to penalize verbosity to achieve conciseness, they typically apply the penalty uniformly across lengthy sequences, which leads to credit confounding, that is, the failure to distinguish essential reasoning steps from superfluous ones. In this paper, we study a measurable proxy for this confounding: high-entropy tokens that often coincide with surface transition points where reasoning re-enters verification. Based on this insight, we propose Adaptive Entropy-Sparing (AES), which selectively spares high-entropy tokens that occur before an efficient in-group reference, while penalizing overlength high-entropy suffix tokens associated with superfluous branching. Our AES thereby navigates the suppression-exploration trade-off, reducing overthinking while preserving average reasoning accuracy. We also present an entropy-length model that motivates why local uncertainty can serve as a proxy for future reasoning cost. Across eight reasoning benchmarks, AES reduces reasoning length by up to 52.9% while increasing mean accuracy by up to 3.8% over the base models, consistently outperforming existing efficiency-oriented methods. Our code is available at this https URL. 

---
# ShopMind: A Mental World Model for Personalized and Realistic E-commerce User Simulation 

**Authors**: Peixuan Hou, Bin Chen, Li He, Jian Xu, Xiuli Ma, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.11897)  

**Abstract**: Large language models are widely used in e-commerce user simulation, providing a foundation for understanding users and predicting human decisions. However, existing simulators still struggle to achieve both personalization and behavioral realism. Most methods predict the next action directly from interaction histories and the current context, without accounting for the evolving mental world that underlies user behavior. Theory of Mind suggests that behavior is driven by mental states, including beliefs, uncertainties, and intentions. Users may interpret the same information in distinct ways, leading to divergent behaviors, while an individual's mental state evolves through interaction and shapes subsequent actions. Ignoring this process prevents simulators from faithfully capturing individual differences and realistic behavior patterns, and undermines its effectiveness in downstream applications. To address this, we propose ShopMind, a Mental World Model for e-commerce user simulation that captures how consumers interpret shopping information and how their mental states evolve, better reflecting individual users and real-world behavior patterns. We further use behavior-mental state pairs synthesized by ShopMind to train a Mental State Reconstruction (MSR) model, which infers consumers' latent mental states from observable behavior. Experiments show that ShopMind improves personalization by 18.0% and behavioral realism by 19.6% over the strongest alternatives. Moreover, MSR improves performance on consumption value measurement and CTR prediction, showing the utility of ShopMind for understanding and predicting real user behavior. 

---
# A Large-Scale Dataset for Cross-Lingual Multi-Jurisdiction Patent Drafting from Patent Cooperation Treaty Applications 

**Authors**: Trung Phuong Le, Hirofumi Nonaka, Koji Marusaki, Katsuhito Sudoh, Seiya Kawano  

**Link**: [PDF](https://arxiv.org/pdf/2611.11891)  

**Abstract**: Securing patent protection across countries requires drafting claims separately for each national patent office, in the relevant language and according to its legal and stylistic conventions. Patent claims can therefore differ across languages and jurisdictions because of translation, target-office drafting practices, and prosecution-stage changes. We formulate cross-lingual patent drafting as generating plausible candidate claim sets for a specified target language, jurisdiction, and prosecution stage from a source claim set in another language. The task models jurisdiction- and stage-specific transformations beyond content-preserving translation. We construct a large-scale Japanese-English dataset by linking Japan Patent Office (JPO) and United States Patent and Trademark Office (USPTO) records through 884,179 Patent Cooperation Treaty (PCT)-derived patent families. Across six released configurations, these families yield 4,067,168 directional claim-set instances; four core configurations form the main benchmark. Commercial machine translation and zero- and few-shot prompting capture part of the task, while supervised adaptation provides the strongest overall baseline, showing that task-specific patterns can be learned from linked cross-jurisdictional claim sets. 

---
# Improving Confidence Estimation in Language Models with Linear Representations of Ambiguity 

**Authors**: Themistoklis Nikas, Nishant Subramani, Aaron Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2611.11838)  

**Abstract**: As language models (LMs) are increasingly deployed in downstream applications, accurately estimating their likelihood of success on a given example is critical. However, confidence estimates from LMs are poor predictors of their likelihood of success. Accurate confidence estimation is linked in part to the detection of aleatoric uncertainty; we hypothesize that LMs possess representations of at least some aspects of this uncertainty in the form of representations of the inherent ambiguity of the input. Using difference-in-means on minimal pairs of ambiguous and unambiguous sentences, we derive ambiguity representations in LM activations, and find that steering with them selectively (albeit weakly) modulates LMs' judgments of the input's ambiguity. We then propose DART, a fine-tuning method that encourages updates to lie within the span of the ambiguity directions, and penalizes changes to the LM activations outside of this subspace. DART greatly improves calibration, encodes these improvements into the model parameters, and maintains model accuracy on multiple-choice and generative QA tasks. 

---
# CaseLaw: A Policy-Decoupled Framework for Training-Free Content Moderation 

**Authors**: Qi Lu, Sen Li, Likun Shi, Xiaolong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.11812)  

**Abstract**: Content moderation is inherently policy-dependent: moderation boundaries are often subtle, subjective, and frequently updated, while conventional rule based or fine-tuning based systems require costly rule rewriting, model retraining, or system redeployment. We propose CaseLaw, a policy-decoupled and training-free framework for adaptive content moderation. CaseLaw reformulates moderation as precedent-based retrieval and voting, where multimodal content is transformed into evidence-preserved structured semantics, embedded into a stable semantic space, and classified by thresholded voting over retrieved precedent labels. We train NoteCap on a large-scale corpus of structurally annotated notes to ensure consistent semantic extraction and freeze it during subsequent policy adaptation, which requires no additional training. Policy adaptation is achieved by updating the semantic case library, including case labels and mappings, without retraining models or modifying prompts. To evaluate CaseLaw, we construct MFO-RedNotes, a benchmark for the male-female opposition risk domain. We additionally provide a five category CoPro comparison. Experiments show that CaseLaw consistently outperforms competing baselines in both precision and recall across policy-dependent risk levels and broader safety categories, with category-wise precision/recall reaching up to 86.8%/90.1% on MFO-RedNotes across both policy settings and 94.5%/92.3% on CoPro. Our work highlights the potential of precedent-based reasoning for policy adaptive content moderation and offers a practical direction toward more stable, transparent, and training-free moderation systems. 

---
# HiddenPathQA: A Benchmark for Knowledge Graph Question Answering When Questions Hide Their Paths 

**Authors**: Sumin Jo, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2611.11802)  

**Abstract**: Knowledge graph question answering (KGQA) systems are typically evaluated on questions whose surface tokens lexically expose the gold KG path. Users, however, may ask questions without knowing the underlying schema: "Which condition could Andrew have due to family history?" compresses a 4-hop chain "Andrew --[father]--> parent --[father]--> grandparent --[medical condition]--> disease --[subclass of]--> hereditary disorder" into a single phrase (family history) that names none of those relations. We identify this collapse of path-question alignment as a blind spot in existing KGQA evaluation. Three major LLM-KG paradigms, hop-by-hop exploration, plan-then-retrieve, and similarity-based retrieval, implicitly rely on such lexical cues, an assumption rarely tested in practice.
We make three contributions. (i) We introduce HiddenPathQA, a manually verified benchmark of 1,055 questions spanning 2-6 hops and multiple domains, constructed so that question wording does not reveal the gold path's relations. (ii) We propose Holistic Path Selection, which compares questions against candidate relation chains in the local subgraph as complete units, replacing irreversible per-hop relation scoring with globally informed chain-level reasoning. We develop two implementations, TieredHolistic and FlatHolistic. (iii) We conduct a pseudoword stress test that replaces entity names with meaningless placeholders to distinguish KG-grounded reasoning from entity-level memorization. Holistic Path Selection outperforms six strong KGQA baselines (ToG, PoG, FiDeLiS, R2-KG, KAPING, and KARPA) by 8.8-17.9 accuracy points across two LLM backbones and both original and pseudoword settings. Its advantage under pseudoword replacement shows that the gains cannot be explained by entity-level memorization alone. 

---
# ChemInsight: A Comprehensive Benchmark for Evaluating and Diagnosing Chemistry Understanding and Reasoning in Large Language Models 

**Authors**: Changyu Hu, Yiming Zou, Xin Tang, Yiru Pan, Zicheng Huang, Deng Zhao, Qing Cui, Zhenduo Zhang, Wang Ren, Xiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.11794)  

**Abstract**: We introduce ChemInsight, a benchmark designed to evaluate large language models (LLMs) on chemistry tasks that require multi-step reasoning and structured chemical understanding. We propose a two-dimensional scoring system (scope and depth) to assess both task coverage and reasoning completeness. ChemInsight consists of 203 challenging chemistry questions organized into 50 bundles, which support the two-dimensional scoring evaluation scheme.
We evaluate ten representative LLMs using ChemInsight and observe consistently low performance across models, with scope scores below 30% and depth scores below 11%, substantially lower than human expert performance. We then proceed to ask (1) What factor limits the advancement of current LLMs in chemistry? (2) To what extent are statistically driven language-model training methods effective at modeling chemistry symbolic language such as SMILES? And (3) Can LLM performance in chemistry be improved through intensive training on large, domain-specific corpora? 

---
# BridgeKV: Large-Model Prefill and Small-Model Decoding for LLM Inference via KV Cache Translation 

**Authors**: Payman Behnam, Debopam Sanyal, Pranav Ponnusamy, Kusum Meda Ravi, Pranav Sastry, Mani Chandana Bandaru, Alind Khare, Myungjin Lee, Alexey Tumanov  

**Link**: [PDF](https://arxiv.org/pdf/2611.11761)  

**Abstract**: LLM inference has two phases with different bottlenecks: LLM prefill runs once per request and is compute-intensive. Decoding repeats for every output token and is limited by memory bandwidth. To address the challenges associated with this issue, different solutions have been proposed. All of these solutions use the same model for both prefill and decode phases. We propose BridgeKV, which uses a larger model for prefill and a smaller model of the same family for decoding. A lightweight KV-stitching module translates the larger model's prompt cache into the smaller model's cache layout. It handles differences in cache shape and representation through learned local layer mixing and separate nonlinear maps for keys and values. With both models frozen, we train the module to align their caches and make the smaller model follow the larger model's next-token predictions. A second training stage adds low-rank adapters when further task adaptation is needed. Across six model pairs and five datasets, BridgeKV outperforms fine-tuned small-model baselines with matched trainable parameter budgets in all 30 settings. It closes up to 80% of the quality gap to using the larger model for both phases. On one A40 GPU, Llama-3.1-Instruct-8B to Llama-3.2-Instruct-1B achieves 3.0X faster requests than the larger model alone, with latency close to the smaller model alone. BridgeKV can improve the hardware and system solutions that separate prefill and decode phases across GPUs. A separate serving simulation models request arrivals, queues, and cache transfer. For Llama-3.1-Instruct-70B to Llama-3.2-Instruct-1B, it predicts up to 33.86X faster requests and 27.5X lower energy use than using the larger model for both prefill and decode phases. 

---
# Explaining and interrupting collective misalignment in language model systems 

**Authors**: Edoardo Chidichimo  

**Link**: [PDF](https://arxiv.org/pdf/2611.11724)  

**Abstract**: In collectives, language model agents can exchange individually innocuous messages that together support a policy violation. Understanding and preventing collective misalignment requires answering the question of which messages sufficiently influenced and necessarily enabled the violation. Collective Dependence Analysis (CDA) operationalises necessity and sufficiency to explain decisions and find the fewest removals interrupting support within tested combinations. Tests covered constructed violations, unscripted agent conversations, and recorded interactions, with selected message removals reducing violations from 93% to 6% in controlled cases. Using O(N) model evaluations for N messages, CDA achieves the same violation reduction with half the deletions of a frontier LLM judge given the transcript and asked for all supporting messages in eligible operational cases. CDA also proposes the mechanism through which the collective misalignment emerged. Nonetheless, complete explanations remain difficult on unfamiliar tasks and some removals can inadvertently disrupt correct decisions. CDA unifies explanations of collective misalignment with experimentally tested interventions. 

---
# Language model capabilities fail in a shared, predictable order under compression and perturbation 

**Authors**: Zhixiang Wang, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2611.11695)  

**Abstract**: Compression can preserve a language model's average accuracy while selectively degrading individual capabilities. Whether these losses follow method-specific patterns or a transferable order is unclear. We applied seven perturbation and compression methods to eight instruction-tuned models, scoring 16 tasks in four answer formats against preregistered predictions. Short-answer capabilities failed in nearly the same order under four random perturbations with different mechanisms (median correlation 0.90) and, within each model, under pruning and quantization (0.875). Intact answer margin, answer frequency and answer format predicted this order. Fragility profiles measured under random perturbation predicted which tasks and facts compression removes first and, after four-task calibration, held-out task accuracy (median absolute error 0.11-0.12, against 0.20-0.21 for uniform decline); profiles borrowed from other model families did almost as well. The order did not transfer across models under GPTQ quantization or hold for long-form tasks. Within short answers, what compression removes is predictable before deployment. 

---
# The Average Patient: Do Simulated Psychiatric Patients Answer Like Real People? 

**Authors**: Weixiang Sun, Zehong Wang, Kaiwen Shi, Hong Huang, Colby Nelson, Yiyang Li, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2611.11679)  

**Abstract**: Language models are used as simulated psychiatric patients to train clinicians and evaluate AI therapists, and are judged by plausibility or total scores. We give a framework for simulated respondents: an information budget bounds how much simulated patients can differ, and an average-patient hypothesis, that a model answers each item at the level typical of the named condition and severity, implies little variation, a full profile of the condition's typical symptoms and totals at or above the top of the stated severity band. We examine these implications at matched totals on item-level data from eight human samples, including a clinical and an instructed-feigning sample. Across 17 models, simulated patients vary 1.5 to 11 times less than real respondents with the same total and on most questionnaires endorse nearly every symptom at a middling level; on the PHQ-9 their symptoms covary less like real ones. People, honest or feigning, answer "not at all" about as often as genuine patients with the same totals, and more often than every model. Single answers are highly concentrated, and prompt changes, published personas, larger models and reasoning reduce the pattern without removing it. In interviews, patients told a severity are rated as having nearly every symptom, so an assessment agent's recognition of absent symptoms goes untested, and scored against the stated severity the agent places a quarter of them in their band, against about half of patients given a real symptom profile. Having the model word such a profile restores half to two thirds of the variation. 

---
# Perturbing Inputs Improves Pretraining Distillation 

**Authors**: Ziyang Cai, Yihe Dong, Dimitris Papailiopoulos, Shivam Garg  

**Link**: [PDF](https://arxiv.org/pdf/2611.11598)  

**Abstract**: Knowledge distillation trains a student language model to match a teacher's predictions on the training text. We ask whether learning the teacher's behavior beyond the training text can further improve the student. We investigate this using three input perturbations: additive token noise, blending two documents, and blending neighboring tokens. Specifically, for an early part of pretraining we perturb the inputs and train the student to match the teacher on them, and we spend the rest of the same token budget on ordinary distillation. When pretraining from scratch, the best method improves on standard distillation at both scales we tried, by 0.025 nats at 135M parameters and by 0.020 nats at 0.5B (mean of three seeds). At 135M this is 1.45 times the gain that distillation itself provides over next-token training. We also observe promising downstream transfer: at 0.5B, the best variant improves average zero-shot accuracy by about one percentage point across seven tasks. It also retains its validation-loss advantage after low-bit weight quantization. We also test a data-limited setting in which training data must be reused, and find that the gains persist. Finally, ablations suggest that, for additive noise, perturbing the teacher's inputs contributes most of the gain, while perturbing only the student's inputs yields much less. 

---
# Out of Parameters, Saved by Syntax: UD-PrMPT for Parser Augmented Low Resource Machine Translation 

**Authors**: Emma Thronson, Abhishek Purushothama, Amir Zeldes  

**Link**: [PDF](https://arxiv.org/pdf/2611.11597)  

**Abstract**: Recent work advancing machine translation quality using LLMs for very low resource languages has relied on the injection of lexicon-based prompt components to expose vocabulary to models. However, these approaches run into a ceiling when models must guess sentence meaning purely from plausible combinations of possible word senses mapped to strings. In this paper we test UD-PrMPT, an approach using syntactic analyses in the Universal Dependencies formalism as prompt components to improve MT in five diverse low resource languages: Beja, Chintang, Chukchi, Coptic, and Xibe. Our results show additive improvements which stack on top of the lexicon based approach which are stronger for higher quality manual analyses but consistent for predicted parses as well. 

---
# Continuous modeling of manipulative participation for stance analysis: Online discourse on the 2024 Street Animal Law in Türkiye 

**Authors**: Ulya Bayram, Ebru Belkıs Güzeloğlu, Emel Kuşku Özdemir, Elif Üstündağlı Erten, Öyküm Esra Yiğit  

**Link**: [PDF](https://arxiv.org/pdf/2611.11576)  

**Abstract**: Studies of online political discourse commonly infer public opinion through stance analysis while implicitly assuming that observed user participation is authentic. Meanwhile, studies focusing on detecting manipulative participation are typically modeling it as a binary phenomenon, overlooking the varying degrees to which users may engage in coordinated or inauthentic behavior. This study proposes a continuous modeling approach for manipulative participation and examines its integration with stance analysis. Using online discourse on the 2024 Street Animal Law in Türkiye as a case study, we collected a dataset comprising 7,320,574 tweets from 537,994 users. From this corpus, 301 users were selected through a high-information sampling strategy and manually annotated by domain experts using a 10-criterion rubric that yields user-level continuous manipulative participation scores. Evaluation under a unified user-level cross-validation protocol showed that the proposed framework accurately estimates both manipulative participation and user stance, with gradient boosting methods providing the best manipulative participation estimates and BERTurk achieving the highest stance classification performance. External validation on an independently annotated user set demonstrated good generalization performance. The findings support modeling manipulative participation as a continuous construct rather than a binary label, providing a more informative basis for computational analyses of the relationship between manipulative participation and stance in online political discourse. 

---
# Beyond the Transcript: Measuring and Improving the Use of Acoustic Information in Audio Language Models 

**Authors**: Linkai Peng, Baorian Nuchged, Yuyang Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.11573)  

**Abstract**: Audio language model benchmarks provide comprehensive evaluation across diverse tasks. However, some tasks can be answered from the transcript or even the question alone, and benchmark scores can therefore confound acoustic understanding with language modeling ability. We distinguish lexical tasks, for which the transcript usually suffices, from global-attribute and within-recording-structure tasks, which require the model to analyse the recording beyond its words, either as a whole, for properties such as the speaker and emotion, or within it, for patterns such as stress and intonation. We evaluated 19 open models and Gemini 3.1 Pro on 92 tasks from 6 benchmarks and estimated the recording gain, the accuracy advantage of the original recording over references that retain its words, by comparing the recording with synthetic readings of its transcript and by adding it to the transcript. On attribute and structure tasks, the open models fall behind Gemini by 28.6 and 41.9 percentage points, and about half or more of each deficit lies in the recording gain. Nevertheless, linear probes in Gemma-4 12B and Qwen2.5-Omni-7B show that the acoustic information needed for answering is already readable in early language-model layers, while the readout of many cues declines in the layers that follow. We therefore constructed training data and trained LoRA adapters in these layers, with a content anchor, which raises attribute and structure accuracy by 12.8 to 44.4 points in the two models and keeps mean lexical accuracy within 1 point of the base model. Most of the improvement appears as larger recording gains. These results suggest that training improves the use of acoustic information that the models already encode. 

---
# When Should Diffusion Language Models Reconsider Their Commitments? 

**Authors**: Haoyang Zheng, Yidong Ouyang, Blake Jianhang Chen, Tian Tan, Ziqian Lin, Liqian Peng, Alec Go, Guang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.11551)  

**Abstract**: Masked Diffusion Models (MDMs) are well known for fast text generation via parallel decoding. However, independent parallel sampling can commit to inconsistent predictions, and less is known about how to selectively revise these early token commitments. In this work, we introduce $\underline{\mathrm{P}}$lanned $\underline{\mathrm{U}}$nmasking-and-$\underline{\mathrm{R}}$evision Joint $\underline{\mathrm{L}}$earning (PURL) for MDMs, a training-based framework that seamlessly unifies parallel generation with selective token revision. We derive a normalized joint transition whose likelihood factorizes into edit selection, unmask selection, and token prediction. These components are trained jointly on partially masked sequences constructed according to the model's own unmasking policy. During inference, all three decisions are supported by a single shared forward pass, enabling the model to decode masked tokens and revise noisy tokens. Across logic puzzles, mathematical reasoning, and code generation, PURL improves accuracy at matched unmask budgets, with relative gains up to 47.4% accuracy improvement and up to an $8\times$ reduction in scheduled decoding rounds. Beyond these gains, our analysis further reveals general principles of revisable decoding: token edits become more effective as context accumulates; early edits can steer unfinished trajectories, whereas later edits tend to remain localized; and effective edits therefore depend on both calibrating when to intervene and learning where and how to revise. Together, these suggest that effective parallel generation depends not on revising more tokens, but on making the right commitments revisable at the right time. 

---
# Does Merge Form a Hopf Algebra? Coproducts, Traces, and the Prospects of Algebraic Minimalism 

**Authors**: Daisuke Bekki  

**Link**: [PDF](https://arxiv.org/pdf/2611.11502)  

**Abstract**: Marcolli, Chomsky, and Berwick (2025; henceforth MCB) formulate the Merge operation of Minimalism on a Hopf algebra of Connes-Kreimer type, and claim that language and physics obey the same algebraic structure. This paper separates their program into the working hypothesis H, that Merge on workspaces can be described as a Hopf algebra, and the claim P placed on top of it, that language and physics obey the same principle, and examines the validity of H. MCB employs two coproducts in formulating Merge, corresponding to two different translations of movement (Internal Merge): $\Delta_d$, which deletes the lower copy, and $\Delta_c$, which leaves a trace. MCB uses the former in syntax and the latter at the syntax-semantics interface, but $\Delta_d$ is not coassociative, and $\Delta_c$ is not connected under the gradings MCB gives and, moreover, not coassociative as long as the labels of traces are retained. Consequently, MCB's algebra is not a Hopf algebra, and the Birkhoff factorization is not a consequence of the Hopf-algebra theorem but of a weaker statement that requires neither coassociativity nor connectedness. There are four ways to recover a Hopf algebra, but none of them can distinguish different traces, and hence none is compatible with Heim-Kratzer-style formal semantics, which reads traces as indexed variables. Thus H does not hold at present, and under the current formulation, making Minimalism a Hopf algebra and preserving its connection with formal semantics are mutually exclusive. Finally, we indicate three directions in which algebraic Minimalism may proceed. 

---
# TID System for Speech-COSER Challenge 2026: ASR Track 

**Authors**: Fernando López, Jordi Luque  

**Link**: [PDF](https://arxiv.org/pdf/2611.11497)  

**Abstract**: We describe the system submitted by Telefónica Innovación Digital to the Speech-COSER Challenge 2026, which features spontaneous speech from elderly speakers recorded in rural Spain. For ASR, we perform CTC-based pseudo-forced alignment scores audio-text match quality to guide a two-stage LoRA curriculum, followed by post-hoc magnitude pruning of the LoRA matrices to enhance generalization. An ensemble of seven systems combined via ROVER achieves 36.25\% unnormalized and 22.17\% normalized WER. 

---
# THEATRON: A Benchmark for Evaluating Persona Persistence in Long-Horizon LLM Decision-Making 

**Authors**: Yuyan Wang, Yuechen Jiang, Zhiwei Liu, Jimin Huang, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2611.11487)  

**Abstract**: Existing evaluations often test persona expression rather than whether persona information persists in extended decision-making. We introduce THEATRON, a controlled benchmark that uses financial replay to evaluate persona persistence in long-horizon sequential decision-making. In these environments, models conditioned on fixed behavioral-finance personas observe shared daily market evidence and trajectory-specific account-state feedback and make sequential decisions over multi-month trajectories. We evaluate Zero-Shot, One-Shot, and Chain-of-Thought persona injection across four LLMs, using cumulative-return and action trajectories to diagnose within-persona stability, cross-persona separability, and action collapse. Persona conditioning produces systematic behavioral shifts beyond a no-persona baseline, and persona-level separation generally exceeds the variation induced by semantically equivalent descriptions. Nevertheless, these shifts do not reliably form coherent persona-specific policies: apparently stable separation can arise from stereotyped action collapse, while diverse policies can exhibit noisy trajectory switching. These findings expose a gap between observable persona-induced behavior and persistent, context-sensitive persona-conditioned decision policies. Code and benchmark data are publicly available at: this https URL 

---
# Saidzoda-CA-EmbeddingGemma-300M: One Small Embedder for Tajik, Kazakh, Uzbek and Kyrgyz 

**Authors**: Tohir Saidzoda, Nekruz Maxudov, Bedil Karimov, Abubakr Abdullozoda, Amin Jalilov, Jonibek Kishvarov, Rahmonali Sharipov, Mariyamkhon Shodyeva  

**Link**: [PDF](https://arxiv.org/pdf/2611.11471)  

**Abstract**: The small open embedder EmbeddingGemma-300M, used as released, finds the right passage first for only 51.7% to 60.8% of Belebele questions in Tajik, Kazakh, Uzbek and Kyrgyz. We present Saidzoda-CA-EmbeddingGemma-300M, one model for all four languages. It is our Tajik fine-tune of EmbeddingGemma-300M, trained for one more epoch on 279,904 pairs: questions in Kazakh, Uzbek and Kyrgyz that our own language model wrote about source passages, a Tajik refresher, and English and Russian replay. On Belebele, recall@1 rises by 14.4 to 26.0 points in each language, to between 75.2% and 77.7%. Against single-language models trained with the same settings, the one model stays within 2.8 points at recall@1 in Kazakh, Uzbek and Tajik, with overlapping recall@10 intervals, though it also had more training. It keeps the Tajik law recall@10 of its starting point (92.2 against 92.4) but loses 4.4 points of law recall@1. A model trained on one of these languages also improves the others on Belebele, but less than training on them. The model still trails the 560M XLM-RoBERTa embedders on Kazakh, Uzbek and Kyrgyz Belebele: the better of multilingual-e5-large-instruct and BGE-M3, used as released, is 9.4 to 10.2 points ahead at recall@1. Our scores of those released models agree with the published MTEB results within 2.4 points. Russian and English recall@1 fall by 3.0 and 4.2 points. 

---
# Vietnamese Legal Ontology: Every legal answer, traced to the article that grounds it 

**Authors**: Khac-Hoai Nam Bui, Quan Tran Minh  

**Link**: [PDF](https://arxiv.org/pdf/2611.11426)  

**Abstract**: A legal answer is only as good as the article it rests on. Vietnam's largest open legal corpus pairs 109,543 published questions and answers with 573,851 legal documents, yet its citations are free text: "Khoan 3 Dieu 101 cua Bo luat nay" ("Clause 3, Article 101 of this Code") names neither the code nor its version. Following the five stages of NVIDIA Auto Ontology, Discover, Ground, Validate, Expose and Learn, we turn four open datasets into a knowledge graph of 11.5 M nodes and 16.0 M edges in which every node and edge carries its source record and method, with no duplicate keys, dangling edges or schema violations. Of 67,868 cited provisions, 61,487 resolve to a dated law version, 51,694 to a parsed article, clause or point, and 19,012 are confirmed word for word. The ontology then shows what flat records hide. 36% of the 78,361 answers whose citations resolve now cite a law that is no longer in force; 2,525 verified citations point to an article number that says something else in the current law; 2,520 citations were re-versioned because their words match a different version than their date suggests; and 5,575 document numbers are shared by several local authorities, one of them by 72. Every number in this paper is computed from the graph. 

---
# Modeling individual and item differences in atypicality inferences with ACT-R 

**Authors**: Guifu Liu, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2611.11416)  

**Abstract**: Humans engage in pragmatic reasoning when encountering mentions of highly predictable events. Such mentions may appear redundant, leading comprehenders to infer that the event is atypical, and accommodate this inference with an explanation for why it is atypical. However, these inferences are not drawn uniformly across individuals, and experimental items vary in inference rates due to differences in event typicality. We show that these patterns can be predicted with a resource-rational cognitive model in which comprehenders select interpretation strategies based on individual exploration tendencies and item-level predispositions. These parameters are estimated independently from participants' Raven's Matrices score, event typicality rating and language model based accommodation likelihood. Taken together with recent findings from reference games, our results motivate a new perspective on the mechanisms that support pragmatic inferences. 

---
# Search-by-Answer Bias in Retrieval Models 

**Authors**: Adiel Avigad, Shauli Ravfogel, Yoav Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2611.11396)  

**Abstract**: Dense retrievers are meant to select passages that match a query's semantic specification. We show they suffer from search-by-answer bias: they favor passages that mention answer objects the model knows (scores as likely), over passages that support valid alternative answers. We test this with controlled factual contrasts from LAMA/TREx. For each retriever, we measure how strongly the model scores the canonical object over an alternative object using a token-prediction head, and ask whether this score predicts passage rankings. Across eight retrievers, including DPR, Spider variants, Contriever, E5, and Qwen3-Embedding, larger token-prediction scores consistently predict higher retrieval scores for passages containing the corresponding object. This holds when passages differ only in the answer object, and even when the passage that mentions the high-scoring object expresses a wrong relation. Dense retrieval thus exhibits search-by-answer bias during passage selection, shaping which evidence is retrieved. 

---
# What Travels Across Languages? Reading, Transfer and Erasure of Cross-Lingual Concept Subspaces in Bangla Idioms 

**Authors**: Syed Rifat Raiyan, Yousuf Ibne Kamal Niloy  

**Link**: [PDF](https://arxiv.org/pdf/2611.11386)  

**Abstract**: Cross-lingual concept analysis requires distinguishing the information that a subspace retains from the classifiers and erasers that transfer through it. We study these operations in six frozen encoders using 10,359 Bangla (Bengali) idiom senses, eight input conditions, and figurative-class and sentiment labels. Class-mean directions provide the most consistent evidence of shared coarse meaning: in post hoc analyses, they outperform random-subspace, permuted-label and covariance-matched nulls in all 12 model-label tests under balanced accuracy. A prospective internal re-split supports idiom-alone sharing in 10 of 12 tests, spanning five encoders. The original confirmatory logistic-regression family rejects both planned nulls in 4 of 12 tests, where English directions gain only a few accuracy points above the majority-class floor (3.0-7.3). Classifier transfer depends on the input map and regularization: English-trained class-mean readouts beat both zero-shot nulls in 8 of 12 discovery tests, and stronger source-selected penalties improve transfer through logistic and LEACE directions. These readouts use Bangla-validation-selected layers and unlabeled target preprocessing. Input controls show greater coarse-class sharing for explicit glosses than for idiom counterparts. Reading, however, does not imply effective erasure: English-fitted LEACE removes less than one percentage point of Bangla figurative-class accuracy, compared with 4-11 points for Bangla-fitted LEACE; INLP shows a similar asymmetry. Pattern-filter analysis, covariance-weighted input maps and a binary Gaussian overlap bound explain how these operations can diverge. The analogous multiclass ordering is an empirical diagnostic, observed in 11 of 12 cells. The results support an estimator-sensitive account of cross-lingual concept sharing and motivate evaluating reading, classifier transfer, erasure and overlap separately. 

---
# AOP-Smart-Finder: A High-Throughput Human-AI Hybrid Framework for AOP Discovery and Evidence Mapping from Large-Scale Literature 

**Authors**: Qinjiang Niu, Lu Yan  

**Link**: [PDF](https://arxiv.org/pdf/2611.11349)  

**Abstract**: The rapid growth of scientific literature and the high heterogeneity of biological terminology and mechanistic descriptions make it difficult for researchers to integrate scattered research evidence, identify connections among biological events, and construct mechanistic networks. Here, we developed a high-throughput human-AI hybrid framework, AOP-Smart-Finder, for biological event discovery, relationship mining, and Adverse Outcome Pathway (AOP) construction and evidence mapping from large-scale literature. This framework integrates Large Language Models (LLMs), AOP-Smart retrieval-augmented generation (RAG) technology, and Python-based automation, supporting two analysis modes: one based on a single compound and the other based on a predefined set of biological events. In the di(2-ethylhexyl) phthalate (DEHP) case, the system processed 12,983 articles within 30 min and included 1,093 of them, from which it extracted 1,589 biological events (including 458 official key events [KEs] and 1,131 candidate KEs) and 2,519 key event relationships (KERs). In the amyotrophic lateral sclerosis (ALS) case, the system identified 55 stressors from 437 articles and extracted 278 biological events and 548 event relationships. Following manual curation and Weight of Evidence (WoE) evaluation, a mechanistic network comprising 16 KEs and 23 KERs was ultimately constructed. This framework can either use existing AOPs as knowledge guidance for mechanistic expansion and evidence supplementation, or discover novel event connections and construct mechanistic networks around user-specified research topics. It provides a high-throughput, traceable, human-AI hybrid methodology for complex biological mechanism elucidation and AOP discovery driven by large-scale literature. 

---
# ScrapeBench: Evaluating Legal Compliance of AI Agents in Website Scraping 

**Authors**: Joseph Marvin Imperial, Daniel Slate, Noam Kolt  

**Link**: [PDF](https://arxiv.org/pdf/2611.11306)  

**Abstract**: AI agents deployed on the open web are becoming more capable of performing actions that may breach contractual provisions and violate computer security laws. In this paper, we introduce ScrapeBench, the first agentic benchmark for empirically evaluating the legal compliance of AI agents with website scraping policies, namely terms of service (ToS) and this http URL directives. Across widely used closed-weight and open-weight AI agents, we find that most agents exhibit low compliance rates (< 5%) when instructed to scrape websites in our sample of 1,626 popular websites that explicitly prohibit scraping. In addition, we observe that even where AI agents actively check the applicable ToS and this http URL, they continue to exhibit low compliance rates. Conversely, in our sample of 269 websites that do not prohibit scraping, we observe that agents sometimes refuse to scrape despite it being lawful, such as Claude Opus 4.6 exhibiting an over-refusal rate of 98.5%. Our human baseline experiments (n = 180 participants) suggest that humans exhibit compliance rates comparable to AI agents but over-refuse at much higher rates. 

---
# EASE-KV: Diagnosing the Gap Between Memory Availability and Effective Use 

**Authors**: Di Lu, Zechen Li, Xuanyu Yue, Jiabo Fan, Hao Xue, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2611.11299)  

**Abstract**: Reusable key-value (KV) memory avoids repeatedly prefilling retrieved text, but stored and injected memories do not necessarily become usable evidence. We present EASE-KV, a controlled framework for understanding this gap under fixed candidates and a frozen language model. EASE-KV separates three interfaces: whether required relations are preserved during encoding, whether preserved memories receive appropriate influence during allocation, and whether outcome supervision can identify useful allocations. Matched experiments show that causal encoding improves compositional accuracy by 25.0 percentage points over block-diagonal encoding, while the tested inference-time bridging methods fail to recover the lost contextual information. With unchanged stored states, diagnostic evidence-guided allocation improves LongMemEval accuracy from 31.2% to 38.4%, demonstrating that memory influence can be controlled without re-encoding. However, controllability does not guarantee learnability: binary correctness provides informative reward contrast for 53.0% of sampled query groups, compared with 73.6% under the evaluated shaped-reward setting. These findings show that effective KV memory requires more than retrieval and injection: the required relations must be encoded, preserved memories must receive appropriate influence, and supervision must distinguish better allocations from worse ones. 

---
# WarBias: A Bilingual LLM Benchmark of Stereotypes About Internally Displaced People and Veterans 

**Authors**: Viktoriia Makovska, Tetiana Zakharchenko, Oleksandra Ostafiichuk  

**Link**: [PDF](https://arxiv.org/pdf/2611.11284)  

**Abstract**: This paper contains examples of stereotypes about veterans and displaced people that may be offensive or distressing. Social bias benchmarks for language models commonly examine categories such as gender, race, and religion, often drawn from lists of legally protected characteristics. Veterans and internally displaced people (IDPs) fall outside these lists, and stereotypes about them, which can undermine access to employment, housing, and social participation, remain largely absent from language model evaluation. We introduce WarBias, a Ukrainian--English benchmark of stereotypes about these two groups. Built from a taxonomy of 120 stereotype claims grounded in research and monitoring reports, WarBias contains 1,007 stereotype--counter-stereotype--unrelated sentence triplets and 240 multiple-choice questions per language. A Ukrainian language model selected in a blinded human pilot generated the Ukrainian triplets. At least two independent annotators reviewed each triplet in the annotation cohort. The authors revised items in response to the reviews, and the final release was machine-translated into English. Each claim carries a human severity rating. In a baseline evaluation of six models, four prefer stereotypical Ukrainian sentences about at least one group, most strongly about IDPs, while English scores are mostly balanced. WarBias supports complementary evaluations of stereotypical preferences and context-sensitive judgments, and provides paired forget/retain material for unlearning-based bias mitigation. We document the benchmark's construction, intended uses, and limitations, and offer its construction method as a template for analogous benchmarks in other conflict-affected societies. 

---
# Understanding When and Where Self-Supervised Speech Models Learn Phonemes and Prosodic Cues through Layer-Wise and Time-Resolved Probing 

**Authors**: Masaru Tanibata, Shun Takahashi, Hiroki Ouchi, Sakriani Sakti  

**Link**: [PDF](https://arxiv.org/pdf/2611.11249)  

**Abstract**: Self-supervised speech models (S3Ms) encode a rich hierarchy of acoustic, prosodic, and phonetic information, yet when and where it emerges during pretraining remains unclear. We train frozen-encoder probes on every Transformer layer across pretraining checkpoints of two matched models: HuBERT-Base, whose stage 2 is a separately initialized model trained on new pseudo-label targets, and single-objective wav2vec 2.0-Base. Our five tasks span an acoustic-to-abstract gradient: absolute pitch (F0), speaker-relative pitch, lexical stress, phoneme boundaries, and 38-way phoneme identity. Acoustic-prosodic cues and phone boundaries are decodable from the earliest checkpoints and barely shift across layers, whereas phoneme identity is constructed during pretraining, re-localizing to upper-middle layers under HuBERT's stage-2 targets at a matched update count. Acquisition order therefore tracks abstraction from the acoustic surface, not the segmental/suprasegmental distinction, extending to prosody an account established for phonetic, lexical and syntactic structure; the no-switch wav2vec 2.0 shows no comparable re-localization. 

---
# Saidzoda-Tajik-EmbeddingGemma-300M: Knowing Every Letter Is Not Reading Tajik 

**Authors**: Tohir Saidzoda, Nekruz Maxudov, Bedil Karimov, Abubakr Abdullozoda, Amin Jalilov, Jonibek Kishvarov, Rahmonali Sharipov, Mariyamkhon Shodyeva  

**Link**: [PDF](https://arxiv.org/pdf/2611.11243)  

**Abstract**: EmbeddingGemma-300M is a small open embedder whose tokenizer represents every Tajik letter, and as released it is weak at Tajik retrieval: on our Tajik law test the right passage is in its top ten results for 49.3% of questions, on tax documents for 45.9%, and on the human-written Belebele Tajik set for 69.8%, against 98.6% on the Russian version of the same set. We present Saidzoda-Tajik-EmbeddingGemma-300M, the same model fully fine-tuned for one epoch on 235,140 question and passage pairs written by our own Tajik language model for laws, tax documents, Wikipedia and web text, with the same mined hard negatives, English and Russian replay, loss, global batch, learning rate and epoch as our earlier Saidzoda-Tajik-E5-Large (arXiv:2611.00592) and BGE-M3 models. After training, law reaches 92.4% and tax 91.7%; on Belebele, the one test written by people, recall@10 rises from 69.8% to 92.6% and the first result is right for 76.6% of questions instead of 51.7%. A checkpoint at step 100 of 466 already holds 88% of the law gain at recall@10. At recall@10 the result is no more than 2.8 points below the two 560M models trained on the same pairs on law, tax and Wikipedia, within the intervals, with 768-dimensional vectors and about half their parameters; at recall@1, for which we have no intervals, it stays 7.0 points behind the E5 model on Belebele. The cost is a drop of 3.0 and 2.8 points in Russian and English recall@1, larger than for the two earlier models. We also report that, according to our run notes, a bfloat16 test run produced a non-finite loss at step 3 which the training framework logged as zero, so training was done in fp32. Our generated test questions come from the same model as the training questions, so we treat those scores as an upper limit until questions written by Tajik speakers exist. 

---
# A Layer-wise Study of How Style Rewriting Breaks the Calibration of Zero-Shot LLM Fake-News Detectors 

**Authors**: Prabhjot Singh, Praney Goyal, Raksha Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2611.11239)  

**Abstract**: Large language models (LLMs) are increasingly asked to judge whether a news article is true, and their stated confidence is often taken at face value. We test what happens to that confidence when an article's writing style, but not its facts, is manipulated, as in recent \emph{style attacks} on fake-news detectors. Tracing how the prediction and its confidence form layer by layer inside the network, we study three model families (LLaMA-3-8B, Qwen2.5-7B, Mistral-7B) on three fake-news benchmarks, and find the same three patterns in every model--dataset pairing. First, on the original articles the models are competent and largely honest judges: they settle on an answer, briefly become overconfident, and then correct their confidence in the final layers. Second, style-rewritten articles break this self-correction: the models become far less accurate, under the strongest attack, worse than random guessing, yet remain just as confident, so they are \emph{confidently wrong} precisely when a warning is most needed. Third, nearly all of the damage comes from rewriting into sensational or emotionally charged styles, while sober, neutral rewrites leave the models' honesty largely intact. The practical implication is that an LLM's confidence about veracity cannot be trusted as a safety signal when an article's style may have been manipulated. 

---
# LongCap: Hour-Scale Timestamped Audio Captioning via Chunk-wise Processing 

**Authors**: Jinchuan Tian, Sen Chen, Wenbin Huang, Chao Weng, Zhengwei Huang, Jiatong Shi, Yichen Lu, Haoran Wang, Shinji Watanabe, Cong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.11125)  

**Abstract**: Describing hours of audio requires accurate event timing and sustained descriptive detail throughout the recording. We present LongCap, a framework for timestamped audio captioning (TAC), with the human-annotated LongCapBench and three diarization-inspired metrics: Timeline Error Rate (TER), Granularity Error Rate (GER), and Caption Similarity (tcpCapSim). LongCap interleaves audio chunks with structured event captions, using relative timestamps and retained context to support length extrapolation within a 1M-token inference context window. Event-level reannotation restores caption detail, while systematic studies examine model scaling, dependency supervision, and self-distillation. LongCap-Diar achieves strong benchmark results and extrapolates to 16 hours of concatenated audio despite training on recordings no longer than 40 minutes. Our TAC models are competitive with Qwen3.5-Omni-Plus on short clips, outperform evaluated baselines in caption similarity and timeline accuracy in segment-level comparisons on LongCapBench-TAC-Long, and caption full recordings approaching two hours. On LongCapBench-TAC, 9B and 35B-A3B offer favorable size-performance tradeoffs, while 122B-A10B substantially improves event-dependency recovery. These findings support long-horizon general audio understanding and provide a foundation for always-on voice agents. 

---
# Beyond Hard Labels: LLM Probabilities Improve Human-Assisted Regression 

**Authors**: Xinyue Wang, Qunhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.11111)  

**Abstract**: Large language models (LLMs) are increasingly used for text annotation, but their outputs are often reduced to a single categorical label before downstream analysis. This hardening step discards the model's relative support across categories, which may contain useful information when category boundaries are ambiguous. We study whether retaining the full probability vector improves downstream regression when LLM annotations are combined with a limited amount of human annotation. Using matched soft and hard representations from three LLMs across three social-science text datasets, we compare direct regression and human-assisted correction. Soft inputs are not uniformly better when used directly as outcomes, but after human correction, using the probability vector consistently improves regression efficiency across models and datasets. Analyses of repeated human judgments and controlled simulations further show that these gains arise when probability variation beyond the hard category carries useful information about the human-coded outcome. Our results identify output representation as an important design choice in LLM-assisted measurement and support preserving probability outputs for downstream statistical analysis. Code and reproducibility materials are available at this https URL 

---
# Room to Reconsider: Reply Length Decides Whether Pushback Makes Language Models Sycophantic or Self-Correcting 

**Authors**: Ook Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.11035)  

**Abstract**: Language models often abandon a correct answer when a user pushes back without giving a reason, and this behavior is usually studied as sycophancy to be suppressed. The same pushback, however, also leads models to revisit and sometimes repair a wrong answer. We measure both effects on multiple-choice questions from MMLU and ARC-Challenge with four open models from two families (Qwen3 1.7B, 4B and 8B, and OLMo-2 7B), reading the final answer exactly from option logits after a free-form reply. The reply budget changes the verdict. When Qwen3-4B may answer the challenge in at most 96 tokens, pushback leaves accuracy unchanged or slightly lower; with 1,024 tokens, it corrects 43% of its wrong answers and pushback raises accuracy by 5.7 points. Whether pushback helps overall follows a simple break-even rule in the cave and correction rates. Inside the model, caving can be partly predicted before the challenge: hidden states add about 0.04-0.06 AUROC over turn-1 confidence in three of our four probing runs. A probe trained only on self-correction cases predicts caving beyond confidence when replies are truncated, but not when the model can finish its reasoning. Activation steering never reduced caving selectively: where it worked, it suppressed self-correction as well. We conclude that sycophancy evaluations should allow full replies and report self-correction alongside caving. 

---
# Building and Auditing Chemical Datasets with LLMs: A General LLM-as-a-Judge Framework Applied to Homogeneous Catalytic PET Depolymerisation 

**Authors**: Muhammad bin Javaid, Karim K. Ben Hicham, Enkhnyam Battulga, Alexander Mitsos, Martin Grohe  

**Link**: [PDF](https://arxiv.org/pdf/2611.11030)  

**Abstract**: Large language models can rapidly extract structured multivariate experimental data from the chemistry literature, yet verifying thousands of generated records without ground truth remains difficult. We present a general, citation-grounded extraction and verification methodology centered on an independent "LLM-as-a-judge", applied to the literature on homogeneous catalytic poly(ethylene terephthalate) (PET) depolymerisation. In this pipeline, an extractor LLM generates schema-enforced records from research papers with citations to document chunks tagged with universally unique identifiers (UUIDs); an independent LLM judge then reads the full paper alongside each record, flags unsupported fields, and proposes cited corrections. Prompt ablations show that requiring chunk-level citations not only provides provenance down to specific paragraphs and tables but also improves extraction accuracy. Before deploying the judge at scale, we benchmarked it against manual evaluation on extractions from a hand-curated reference set of 295 ionic-liquid-catalysed PET glycolysis experiments. The judge agreed with human evaluation on 85.7% of reviewed records and was significantly more effective than a heuristic string-matching baseline at catching errors across related experiments. Deployed across 3,942 extracted records from 561 papers, the framework accepted 67.5% of them as written. Its proposed corrections concentrated on quantitative fields, where the extractor often conflated related experiments. We release the extracted dataset with the complete LLM judge audit logs, as well as the Chemistry Data Extractor Toolkit, an open-source application implementing this end-to-end pipeline in a code-free, browser-based environment for document parsing, customizable schema and prompt design, model-agnostic extraction and auditing, interactive provenance-highlighted review, and reproducible dataset bundling. 

---
# MoTo: Mixture of Tokenizers Towards Fair Multilingual Language Modeling 

**Authors**: Gül Sena Altıntaş, Colin Raffel  

**Link**: [PDF](https://arxiv.org/pdf/2611.11027)  

**Abstract**: Large language models rely on a single tokenizer that is chosen at training time and fixed thereafter. Most tokenizers are trained on English-dominant corpora, making them under-serve morphologically rich and non-Latin-script languages, where they can produce fragmented representations, longer sequence lengths, and reduced information density during training. We propose Mixture of Tokenizers MoTo, a modular tokenization framework that trains dedicated per-language BPE tokenizers and composes them into a unified superset vocabulary, where tokens shared across languages are deduplicated into a common ID space. Each tokenizer maintains its own normalization policy, pre-tokenization rules, and vocabulary budget. Despite allocating significantly fewer entries to English, MoTo retains English performance while delivering consistent gains across 18 out of 21 typologically diverse languages spanning six scripts, with improvements most pronounced on morphologically complex and non-Latin-script languages that are systematically underserved by English-dominant tokenizer design. Our results suggest that modular subword tokenization is a practical and extensible alternative to monolithic tokenization for multilingual parity. 

---
# Open Problems in Chatbots and Mental Health 

**Authors**: Jared Moore, Jacy Reese Anthis, William Agnew, Andrea Mock, Kevin Klyman, Sanmi Koyejo, Nick Haber, Stevie Chancellor, Eric Lin, Ryan Louie, Desmond C. Ong  

**Link**: [PDF](https://arxiv.org/pdf/2611.10997)  

**Abstract**: General-purpose chatbots are increasingly used for emotional and mental health support. They can be useful, but issues such as sycophancy, reinforcement of delusional beliefs, and poor crisis response create risks that are difficult for researchers to study and evaluate. In this paper, we review the most critical open problems for researchers and policymakers in the use of general-purpose chatbots and their effects on mental health. First, there are tractable problems in the controlled evaluation of chatbots: establishing the construct and ecological validity of evaluations, evaluating multi-turn interactions, and connecting evaluations to improvements in chatbot behavior. For example, conventional benchmarks that primarily rely on static datasets cannot establish chatbot safety, and novel simulation methods need validation. Second, researchers must connect their studies of chatbots and mental health to real-world evidence. This includes analyzing the chatbot products that AI companies and mental health practitioners have deployed, including their user interfaces and human review and escalation practices for user welfare concerns. However, clinical analysis opportunities afforded by real chat log data sharing are complicated by significant privacy concerns. Finally, we argue that researchers should take an active role in the construction and refinement of evidence-based policy through corporate and governmental decisions such as user protections, standards of care, protected access to user chat logs, data sharing between institutions, incident reporting, and post-deployment monitoring. Such behavioral evaluations, studies of real-world outcomes, and data governance depend on each other and must be addressed in tandem. We believe a coordinated research effort would have substantial real-world impact. 

---
# Think or Remember: The Exchange Rate of Agentic Inference 

**Authors**: Juan Wisznia, Santiago Luis Groba Alonso, Matías Gómez Taddeo, Gonzalo Zabaleta, Joaquín Cullen, Luciano Del Corro  

**Link**: [PDF](https://arxiv.org/pdf/2611.10982)  

**Abstract**: Agents built on reasoning models resend their growing transcript at every call. Two transacting parties, not the model, decide what that costs: the client keeps or edits messages; the server decides whether earlier reasoning returns in the prompt and what its cache reuses. We define a cost model whose counts match an instrumented engine exactly; the architecture charges $c$ per token processed and $\beta$ per pair attended. Their ratio $P^*=c/\beta$ is an exchange rate between thinking and remembering: per token, processing dominates below this context length, a thinking regime, and attention above it, a memory regime. A policy change has a mechanical cost, incurred even if the agent writes the same tokens, and a behavioral one: it writes different tokens, changing cost and outcome. Serving alone can flip which policy is cheaper: on recorded Qwen3-8B ScienceWorld runs, dropping earlier reasoning costs 1.74 times keeping it when the server reuses generated states and 0.34 times when it reuses nothing. In live reruns, dropping it cuts correct HotpotQA answers by 9-23 percentage points on four models, at lower point-estimate cost. Comparing context policies means comparing systems: architecture and serving set the cost of the history kept and the work it induces. 

---
# SWEET: Mitigating Spurious Feature Reliance in LLM Reasoning via Weight Editing 

**Authors**: Zhizhang Fu, Lucas Zhao, Michalis Korakakis, Mengru Ding, Lukasz Szpruch, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.10974)  

**Abstract**: Reasoning in large language models can be disrupted by simple input perturbations (e.g. adding irrelevant distractors), suggesting reliance on spurious correlations. Identifying and suppressing the spurious features involved in long chain-of-thought reasoning remains challenging. We introduce Spurious-feature WEight EdiTing (SWEET), a training-free method that combines counterfactual interventions with mechanistic interpretability to identify and suppress spurious-features. SWEET extracts token- and sequence-level behavioral directions from activation contrasts between original problems and their counterfactual variants - capturing local and global patterns of spurious reasoning. It then localizes the responsible MLP neurons and attention heads and suppresses their contributions along these directions through weight edits. Using only 20 original-counterfactual pairs for direction extraction, SWEET improves accuracy by an average of 11.8 percentage points on perturbed inputs and 2 points on original inputs across five open-weight reasoning models and five reasoning benchmarks. It also reduces the relative gap between original and perturbed accuracies from 17.6% to 2.5%. Analyses identify two recurring forms of spurious reasoning, i.e. over-integrating irrelevant conditions and overthinking, and provide mechanistic evidence that SWEET improves reasoning robustness by suppressing both behavioral directions. 

---
# When Safety Becomes An Outlier: Understanding the Retention of LLM Safety Behaviors 

**Authors**: Binchi Zhang, Hadi Abdullah, Yiwei Cai, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.10971)  

**Abstract**: Safety-aligned large language models often lose their safety behaviors after fine-tuning, even when safety data are included. While this fragility is well documented, its underlying cause remains unclear. In this paper, we propose that formulaic refusal strings can function as outlier modes: narrow, weakly grounded response patterns that are easy to fit but vulnerable to subsequent updates. Controlled experiments support this account: fixed refusals are acquired much like arbitrary constant strings, in contrast to prompt-grounded natural responses. Motivated by these findings, we introduce Naturalness-Enhanced Safety Targets (NEST), a safety alignment framework that improves safety retention by constructing natural, context-aware safety targets that keep safety responses within the model's existing output distribution and grounded in semantics. We instantiate NEST in both supervised and preference-based alignment settings, using perplexity under the original model as a proxy for distributional support. Across multiple models and alignment regimes, natural safety targets improve both immediate safety and its retention under continued fine-tuning without sacrificing helpfulness. These findings suggest a practical principle for more durable safety alignment: construct safety targets that draw on the model's existing language capabilities and remain grounded in the meaning of each request. 

---
# Can small transformers learn the language of Coxeter groups? 

**Authors**: J. Elisenda Grigsby, Yiyao Huang, Arushi Katyal, Ryan Lin, Pranav Ramakrishnan, Weiya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.10936)  

**Abstract**: The field of mechanistic interpretability is well-suited to collaborative projects between mathematicians and AI researchers. This work describes one such project. We trained a small transformer to learn the language of the Coxeter group affine A2, best known as the symmetries of the tiling of the Euclidean plane by equilateral triangles. We then formulated and proved a word-theoretic conjecture about affine A2 after observing the attention patterns of the trained model. 

---
# Gendered Misclassification from Surname-Based Race Imputation 

**Authors**: Noah Dasanaike  

**Link**: [PDF](https://arxiv.org/pdf/2611.10931)  

**Abstract**: Researchers increasingly estimate political behavior based on race and gender from voter files, where race is imputed from surnames and geography. This only works where surnames suggest group membership, an assumption violated where surnames follow norms around marriage. Imputation is therefore biased against women who adopt their husband's surname, especially those who racially inter-marry, an outcome itself endogenous to partisanship. Using six US states recording self-reported race, I show that imputation is less accurate for Asian and Hispanic women, particularly those sharing the surname of a partner of another race, who differ politically from other women of their race. This violates assumptions behind downstream inference approaches, which nearly double the gender gap in partisanship among Asian voters. I derive an expression showing the bias is largest where a group is small. Using maiden rather than marriage-adopted surnames in North Carolina raises accuracy up to fourfold for women married across racial lines. Recovering maiden surnames, or omitting the married surname, removes much of the bias. In a national replication of a study of gender gaps in turnout, the correction more than triples the gap among Asian voters. In Lebanon, where wives register in their husband's locality, the error enters through geography. 

---
# DeepFinance: Can AI Diligence a $100MM Deal? 

**Authors**: Drew Hahn, Tyler Shibata, Jackson Grove, Rahul Sonwalkar  

**Link**: [PDF](https://arxiv.org/pdf/2611.10926)  

**Abstract**: Existing evaluations of AI models on high-finance work focus on retrieving answers from financial documents and producing numerically correct outputs. Real private equity (PE) and investment banking (IB) work also depends on choosing between conflicting sources, making sound assumptions, and keeping every step auditable. We introduce DeepFinance, an ultra long-horizon benchmark of 21 end-to-end PE and IB tasks that take a human professional an average of 27.1 hours. Each task pairs a request with a data room averaging 19 files and is graded against an average of 84 expert-written rubric criteria covering output, workflow, and judgment. We evaluate 22 model configurations, each in its native agent harness. The best configuration, GPT-6 Astra with max reasoning, satisfies 57.6% of rubric criteria, and the top four configurations are not statistically distinguishable. Frontier models often calculate correctly from the wrong inputs and find decision-relevant evidence without acting on it. 

---
# Pole or Vector? Encoders Capture Bangla Compound Verbs, but LLMs Read Them Literally 

**Authors**: Md. Asaduzzaman Shuvo, Md Mehrab Karim Antor, Md. Adnanul Islam, Shahanur Rahman Bappy, Sajid Ahamed, Mahfuz Hasan Reza, Md Delwar Shahadat Deepu  

**Link**: [PDF](https://arxiv.org/pdf/2611.10909)  

**Abstract**: In Bangla, a main verb is often followed by one of a small set of vector verbs. These verbs lose their own meaning and add a shade of meaning instead, such as completion. For example, likhe phello means that someone finished writing, not that they wrote and threw something. The same two verbs can also describe two separate events, so only the context shows which meaning is meant. This makes compound verbs a hard test of how language models handle multiword expressions. We present BanglaCV-Probe, a benchmark of 17 vector verbs with 1,044 minimal pairs, 204 entailment items and 197 pairs of correct and incorrect combinations, all checked by native speakers. We test 26 models (encoders, LLMs and translation systems) on four skills: telling the two meanings apart internally, understanding what a vector adds, preferring correct combinations and keeping the vector's meaning in translation. Probes on five encoders tell the two meanings apart and still reach up to 93% accuracy when trained on pairs without surface cues. Prompted models rarely use this knowledge. Entailment accuracy stays between 59% and 81%. Several LLMs answer Yes or No to almost every item, and nine of ten models miss that someone who ate up the rice left none. Most instruction-tuned LLMs translate vectors literally in 15-20% of vector sentences and in up to 55% for mora (die). On correct combinations, a small Bangla-only model (88%) matches BLOOM-7B1 (87%) and both beat Qwen2.5-7B (74%). Data and code: this https URL 

---
# SCorPion: A Strategic, Controllable and Personified Conversational Companion 

**Authors**: Luo Ji, Xiyuan Chen, Dan Wang, Qingqing Gu, Teng Chen, Feixiang Guo, Yetao Wu, Jinxian Qu, Yue Zhao, Kun Fan, Ningyuan Xi, Hange Zhou, Zhonglin Jiang, Yong Chen, Hongyan Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.10906)  

**Abstract**: Fixed prompt-based language models face challenges in open-domain conversations, where versatile topics require different expert advice or demonstrations to maximize their universal performance. Motivated by the dynamic prompt studies in language generation, we extract different chatting strategies from varied forms of text corpus, which steer the language model to different styles based on human principles. We design and deploy a conversational companion called SCorPion, which uses GraphRAG to extract and summarize the knowledge graph globally, yield community-summarized strategies, and produce query-focused optimal strategies. SCorPion has strategic thinking, yields personified and stylized responses, and can be controlled by explicit factors. Code and further results can be found in this https URL. 

---
# KeyASRBench: Measuring Whole-Phrase Keyword Preservation in Real-World ASR 

**Authors**: Guanzheng Wei, Ziyi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.10852)  

**Abstract**: Public automatic speech recognition (ASR) benchmarks increasingly report low sequence error rates, yet production systems still use hotword lists, contextual biasing, retrieval, and post-correction. This motivates a complementary question: does a transcript preserve selected names, products, technical terms, versions, and relational phrases? We present KeyASRBench, a bilingual pilot that operationalizes occurrence-based Keyword Error Rate (KER) over frozen whole-phrase annotations. The 10-hour model-conditioned challenge set contains 1,501 Chinese and English clips with 4,723 keyword occurrences. Across five locally inferred checkpoints and two official API systems, language-specific MER/WER ranges from 5.31% to 12.91%, while KER ranges from 13.61% to 33.81%. A disjoint 30-minute model-blind control yields KER of 10.65%-27.31%. The control confirms that selected-phrase errors extend beyond challenge-set selection. 

---
# MultiTurnRoute: A Platform for Building and Evaluating Multi-Turn LLM Routers 

**Authors**: Jiaqi Xue, Mengxin Zheng, Heng Huang, Qian Lou  

**Link**: [PDF](https://arxiv.org/pdf/2611.10848)  

**Abstract**: In a multi-turn conversation, the choice of language model at one turn can affect which model should answer the next. The selected model's reply shapes the context for the next query, and its cached prompt prefix can reduce the cost of using that model again. We introduce MultiTurnRoute, a platform for building and evaluating multi-turn LLM routers. A shared interface gives built-in and user-defined routers the current query, dialogue history, and an estimate of each model's reusable prefix length. Our routing algorithm, Chain-of-Route (CoR), uses a learned gate over recent and recurrent dialogue history to predict response quality and estimates cost from each model's reusable prefix. MultiTurnRoute Arena ranks routers on recorded conversations and compares session quality and cost in interactive runs where a simulated user responds to each reply. On a benchmark drawn from WildChat and LMSYS-Chat-1M with per-turn quality annotations for five LLMs, CoR reduces the normalized cost required to match the strongest model's quality by up to 20.3% relative to the strongest routing baseline, while matching or improving peak response quality. 

---
# Simple Test-Time Refinement for Plot-to-Code Generation via Visual-Code Diagnostics 

**Authors**: Peiming Guo, Jinpeng Huang, Yangbin Sun, Gaoxiong Cao, Ensheng Shi, Yuchi Ma, Yu Zhang, Meishan Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.10840)  

**Abstract**: Plot-to-code generation has been greatly advanced by recent multimodal large language models (MLLMs). However, existing studies mainly focus on one-pass generation or iterative sampling guided by verifier-based filtering, which does not align with the typical human programming workflow of generate first, then iteratively error-location and fix. This naturally motivates a test-time refinement paradigm for plot-to-code generation. In this work, we present the first study on plot-to-code generation via iterative test-time refinement. We propose a Visual-Code Diagnostics framework that enables refinement through visual feedback from source code. Specifically, we first categorize generation errors into four coarse-grained types, and then train lightweight discriminators to recognize and localize these errors based on the rendered visual outputs and corresponding code. Unlike traditional compiler- or debugger-based verification, our approach performs iterative refinement guided by one diagnostics system, enabling more effective semantic and visual correction during test time. Extensive experiments demonstrate that our method consistently improves generation quality across diverse generators and significantly outperforms strong test-time scaling baselines. 

---
# Evaluating Jev on Arabic: Comparing Decision Models and LLMs Across Arabic Tasks and Dialects 

**Authors**: Hossam Elsafty, Tobias Deußer, Lorenz Sparrenberg, Rajkumar Ramamurthy, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2611.10802)  

**Abstract**: How does Jev compare with other systems on Arabic tasks, and how does its performance change between Modern Standard Arabic (MSA) and dialects? We evaluate 67 systems, including 47 decision models and 20 large language models (LLMs), on 43 tasks across 9 task families using a shared zero-shot protocol. The Arabic-focused frozen suite includes English controls in three parallel resources, which we also remove in a separate descriptive comparison. Jev 1.14.0 achieves a mean task-primary metric of 69.24 on a scale multiplied by 100, compared with 68.77 for Jev 1.13.0 and 71.81 for Gemma-4-31B, the highest-scoring system on this aggregate. Removing the recorded English controls, repeated cross-task items or both leaves the full system order unchanged. Relative performance and gains between Jev versions vary across task families. In content-matched comparisons within Jev 1.14.0, pooled dialect-minus-MSA accuracy gaps range from -7.94 to -0.35 percentage points across four datasets; dialect-specific results show larger variation, and SIB-200's source-clustered interval includes zero. Paired contrasts with Jev 1.13.0, Gemma-4-31B and Jais-2-8B show a larger Palestinian banking penalty for Jev 1.14.0 than for either LLM under simultaneous uncertainty; the other gap contrasts remain inconclusive. These descriptive findings show why aggregate scores should be read alongside task-family results, coverage and matched language comparisons. They do not establish a general model ranking for dialect robustness or explain the causes of the observed gaps. 

---
# Enhancing Mathematical Reasoning in Turkish Large Language Models with Search-Based Inference and Model Merging 

**Authors**: Ahmed Zeer, Osama Shbib, Muzaffer Kaan Yuce, Himmet Toprak Kesgin, Mehmet Fatih Amasyali  

**Link**: [PDF](https://arxiv.org/pdf/2611.10800)  

**Abstract**: We improve Turkish mathematical reasoning in Large Language Models by combining LoRA fine-tuning, weight-space model merging, and search-based inference. Using Turkish-Llama 8B-DPO with Turkish translations of MetaMath, NuminaMath, and TigerLab, we find data quality beats scale (Mix-30k best among fine-tuned; large MetaMath underperforms). Linear merging delivers the largest gains (GSM8K up to 0.63), while Ties is modest and Breadcrumbs/Slerp degrade. Search with MCTS and Tree of Thoughts improves mean LLM-judge scores for reasoning quality on a 0-100 scale. Limitations include machine translation, limited scale, and search stochasticity. 

---
# Adapting Large Language Models via Multi-Task Learning for Mental Health Problem, Emotion, and Sentiment Detection 

**Authors**: Amir Rafiq Bhat, Rahim Dehkharghani, Zeki Erdem  

**Link**: [PDF](https://arxiv.org/pdf/2611.10797)  

**Abstract**: This study investigates sequential multi-task learning using Parameter-Efficient Fine-Tuning (PEFT), LoRA, and 4-bit quantization on Llama 3.1 8B Instruct for mental health problem detection, emotion detection, and sentiment analysis. Using 4,243 English social media posts, it compares two task sequences and demonstrates that task ordering significantly affects knowledge transfer and task interference. The MHP-first sequence achieves the best mental health problem detection (Micro-F1: 0.7624), while the Emotion-first sequence performs better in emotion (F1: 0.6052) and sentiment analysis (F1: 0.9459). A Joint-MTL baseline further highlights the challenges of simultaneous learning, including task interference and generative format collapse. Overall, the findings demonstrate the effectiveness of sequential PEFT for mental health NLP while showing that task order is critical to balancing performance across related tasks. The integration of LoRA and 4-bit quantization also reduces computational and memory requirements, supporting the adaptation of large language models in resource-constrained environments. 

---
# Cascading Circle Loss: Order-Preserving Contrastive Learning for Graded Legal Retrieval 

**Authors**: Rayane Kimouche, Ahmed Touila  

**Link**: [PDF](https://arxiv.org/pdf/2611.10765)  

**Abstract**: This paper introduces Cascading Circle Loss (CasCL), a contrastive learning framework for legal retrieval tasks with graded relevance. CasCL combines an InfoNCE objective for positive-negative separation with a cascade of ranking boundaries that model ordered relevance levels between queries and legal clauses. The framework also includes a negative-aware self-paced optimization mechanism and a relevance gain weighting strategy to improve ranking quality. Experiments on legal passage retrieval benchmarks show that CasCL consistently outperforms existing methods, including InfoNCE, Circle Loss, MNRL and Margin MSE, across both binary retrieval and graded ranking metrics. 

---
# Evaluating Physically Grounded Procedural Reasoning in VLMs through Real-World Execution: Crochet as a Testbed 

**Authors**: Maximiliano Wardle  

**Link**: [PDF](https://arxiv.org/pdf/2611.10740)  

**Abstract**: Vision-Language Models (VLMs) can generate procedures that look convincing on paper, but do they work in practice? In this paper, we propose crochet as an ideal testbed for physically grounded procedural reasoning in VLMs. Crochet cannot be reliably simulated or automated: the only way to know what a pattern produces is to make it by hand, yet verification is tangible and cheap, requiring nothing but a hook and a ball of yarn. We introduce FoundationChain, a deceptively simple test of procedural reasoning: given an image of a basic crochet object, can VLMs infer how it was made and generate a pattern that produces the same result? We crochet 40 patterns generated by four VLMs by hand. Even at this scale, physical execution not only reveals failures that automated proxies cannot detect, but also enables a more precise diagnosis of their causes and consequences: while models write fluent, convincing patterns, only four are free of errors, and our analysis exposes deficiencies in visual, geometric and procedural reasoning. 

---
# Behavioral Evaluation of Large Language Models Under Cognitive Biases in HAZOP Studies 

**Authors**: Vinicius M. Müller, Vitor A. O. Silva, Sávio S. V. Vianna, Flávio V. Silva  

**Link**: [PDF](https://arxiv.org/pdf/2611.10721)  

**Abstract**: Hazard and Operability (HAZOP) studies safeguard industrial processes against catastrophic accidents. Large language models (LLMs) increasingly assist HAZOP drafting, yet evaluations rely on lexical overlap. Because LLMs internalize human heuristics, prompts may trigger biased judgments. Whether frontier LLMs withstand cognitive perturbations in safety-critical workflows remains unknown. Here we show across 420 trials that three frontier LLMs resisted 96.8% of cognitive manipulations, rendering overall susceptibility statistically inconclusive. Contrary to assumed broad fragility, models rejected confirmation and availability cues. However, safety-critical risk downgrades emerged under low anchors and gain frames. Thus, automated hazard analysis requires disciplined human-in-the-loop validation. 

---
# TransMem: Transition-Aware Retrieval for Evolving Personal Memory 

**Authors**: Shigeng Chen, Linhao Luo, Changlong Shi, Zhangchi Qiu, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2611.10683)  

**Abstract**: Long-term personalized agents are increasingly expected to answer queries over user histories in which preferences, goals, and constraints evolve across interleaved conversations. Existing retrieval-based and structured-memory systems can surface relevant facts, but often return isolated memory units that do not explain which earlier preference was refined, contradicted, or superseded by later evidence. We introduce TransMem, a transition-aware memory retrieval framework that stores preference updates as explicit retrieval units alongside ordinary factual memories. During memory construction, TransMem routes user interaction history into topic-structured memory graphs, detects shift events, and grounds them as transition records. At inference time, TransMem first retrieves factual anchors and then adaptively expands over transition records, producing a compact context of factual anchors and accepted transition records. Experiments on PersonaMem and PersonaBench across two Qwen backbones show that TransMem outperforms representative retrieval-based and structured-memory baselines. Controlled analyses further show that transition records and adaptive expansion contribute beyond factual anchoring, chronological expansion, flat transition retrieval, and fixed-hop traversal. These results suggest that preference-update records are useful retrieval units for long-term personalization. 

---
# On the Geometric Bottlenecks of Cross-Lingual Knowledge Editing 

**Authors**: Aastha A K Verma, Anwoy Chatterjee, Sparsh Jain, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2611.10646)  

**Abstract**: Transformer-based autoregressive multilingual language models (LMs) are often assumed to store factual knowledge in an interlingua: a language-agnostic latent space. If this assumption holds, a parameter update applied in a source language during knowledge editing should inherently propagate to its semantic equivalents in target languages. However, empirical benchmarks show that cross-lingual propagation of edits often degrades substantially, a degradation that prior work treats largely as an algorithmic symptom rather than a structural bottleneck. In this work, we investigate whether representational anisotropy creates a structural bottleneck for source-only associative-memory editing, and perform a geometric dissection of the cross-lingual propagation gap. Using associative memory-based "locate-then-edit" methods, including ROME, MEMIT, and AlphaEdit, as analytical frameworks, we derive a closed-form decomposition of this gap into key-space propagation, parallel residual mismatch, and orthogonal residual mismatch. Our empirical findings show that cross-lingual propagation is primarily bottlenecked by a large orthogonal residual component, indicating that source-language edits often fail to span the target-language update directions required for successful transfer. Furthermore, we show that post-hoc correction using only source-language residual statistics encounters a large residual-prediction barrier. Finally, we perform a structural realignment via a targeted contrastive alignment objective and find that it substantially improves key-space propagation and partially improves cross-lingual edit transfer, while leaving value-space mismatch as a remaining bottleneck. 

---
# AI in the Clinical Documentation Loop: Effects of LLM-Induced Note Bloat and Diagnostic Overcall on Downstream LLMs 

**Authors**: Jinghui Liu, Anthony Nguyen, Taylan Gurgenci  

**Link**: [PDF](https://arxiv.org/pdf/2611.10601)  

**Abstract**: Large language models (LLMs) are increasingly used to process medical records for healthcare tasks such as clinical documentation. Because these tools also generate documentation, the relationship is now bidirectional: models both read from and write to the record. Understanding how LLMs behave when reading records that already contain LLM-authored content is therefore critical for their safe deployment. In this study, we examine how LLM-authored content in the medical record affects downstream LLMs that read the same record. We consider two types of LLM-induced artifacts: bloat, in which LLM-generated summaries accumulate in the record and change the length and provenance of its content over time; and distortion, in which LLMs subtly alter clinical findings rather than fabricating facts outright. As a case of distortion, we study diagnostic overcall (documenting an unconfirmed differential diagnosis in the record) and test whether it propagates to the reader LLM's output. Across seven reader models from four families, we find evidence of both effects. First, LLM-generated summary notes modestly but consistently degraded downstream hospital-course generation compared with copy-forward and templated text of matched length. Second, injected overcalls reappeared in downstream outputs at rates that varied by reader model, without detectable changes in standard quality metrics. These findings show the need to monitor AI-generated content in medical records and to account for it when those records are later processed by AI. 

---
# Global Context Guidance for Diffusion Large Language Models 

**Authors**: Jeimin Jeon, Bumsub Ham  

**Link**: [PDF](https://arxiv.org/pdf/2611.10577)  

**Abstract**: Diffusion large language models (LLMs) have recently emerged as a promising alternative to autoregressive LLMs, enabling parallel generation through iterative denoising. Recent guidance methods improve diffusion LLMs by introducing a weak model to guide the prediction of the base model during denoising. However, the influence of the weak model during denoising remains less clear, and computing its prediction requires an additional forward pass at each denoising step. We present in this paper an in-depth analysis of how diffusion LLMs use contextual information during denoising. Our analysis reveals that diffusion LLMs rely more heavily on local context while often underutilizing global context. Based on this observation, we introduce Global Context Guidance (GCG), a feature-space guidance method that enhances global-context utilization without constructing an explicit weak model. Extensive experiments on LLaDA-8B, Dream-7B, and LLaDA-1.5 across diverse benchmarks demonstrate that GCG consistently outperforms existing guidance methods, while incurring much lower inference overhead. 

---
# Large Language Models without QK-Normalization 

**Authors**: Yongding Tao, Yitong Qiao, Tianqi Shao, Tian Wang, Zhixuan Chu, Xiaoming Wu, Yihong Dong, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.10560)  

**Abstract**: Query-Key Normalization (QK-Norm) is widely adopted in modern large language models to control the growth of attention logits and improve training stability. However, query and key norms are not merely sources of instability. The query norm controls the sharpness of the attention distribution, while the key norm determines how strongly a token attracts different queries. These norms therefore carry token-specific information about attention selectivity and token importance. By normalizing both, QK-Norm suppresses this information. This raises a natural question: can we control query-key representations without discarding their sensitivity to magnitude? To address this question, we propose QK-MAP, a Magnitude-Aware Projection for constructing queries and keys. Rather than removing magnitude through normalization, QK-MAP transforms the input representation using multiple learnable frequencies, so that changes in magnitude are encoded as bounded phase patterns. This allows magnitude to be represented through phase and direction rather than vector length alone. Its periodic features have a fixed norm while remaining sensitive to magnitude changes, separating magnitude encoding from norm growth. Multiple frequencies further help distinguish magnitudes and reduce periodic ambiguity. Experiments across model scales from 190M to 7B parameters show that QK-MAP achieves lower training and validation loss than both Standard attention and QK-Norm. Further analyses show that QK-MAP preserves sensitivity to input scale, assigns less attention to the first token, and produces stronger attention contrast for relevant evidence in synthetic long-context tests. These results demonstrate that query-key scale can be effectively controlled without sacrificing token-specific magnitude information. 

---
# onóna: Twin Embeddings with Tied Unembeddings 

**Authors**: Cheng Luo, Xinnan Dai, Zhongzhu Zhou, Zefan Cai, Ruijie Zhu, Tianyu Zhang, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.10556)  

**Abstract**: Weight tying reduces the parameter cost of language models, but the usual tied output head is limited to a single softmax. We introduce onóna, a two-table parameterization that increases output expressiveness while preserving input-output weight sharing. The Transformer takes the average of two embedding tables as input, and the output head combines the softmax distributions defined by those same tables. A gate conditioned only on the current input token determines the mixture weight. Its values can be precomputed for inference, and the Transformer body remains unchanged. We evaluate onóna across Dense, Gated DeltaNet hybrid, mixture-of-experts, and Attention Residual architectures at 142M, 330M, and 1B scales. Under matched token budgets, onóna lowers final validation loss and external perplexity in 11 of 12 comparisons and raises the average accuracy over eight downstream tasks in 10 of 12. The one exception on loss is the 1B MoE model. The cost is one extra embedding table and one extra vocabulary projection. 

---
# YRC-Bench: Benchmarking Culturally Aware Machine Translation for Historical Heritage via Tri-Level Semantic Bridging 

**Authors**: Huang Yan, Aobo Li, Jiayi Wang, Huidong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.10555)  

**Abstract**: Translating historical cultural heritage poses profound challenges for machine translation systems due to the acute semantic asymmetry across eras and cultures, where terse classical syntax frequently obscures domain-specific cultural knowledge. While Large Language Models have demonstrated remarkable prowess in generic translation benchmarks, they struggle to negotiate Culturally-Specific Items and diachronic shifts in domain-rich historical texts, such as Yellow River cultural records. To bridge this critical gap, we introduce YRC-Bench, the first human-curated, multi-parallel benchmark dedicated to historical heritage translation across Classical Chinese, Modern Chinese, and nine target foreign languages. YRC-Bench establishes a tri-level parallel corpus incorporating modern explanatory glosses as explicit semantic bridges and operationalizes translation strategies through paired conserved versus substituted CSI variants categorized by cultural translatability. Systematic evaluations of nine leading LLMs across bidirectional translation directions reveal that leveraging modern Chinese glosses - especially human-curated ones - dramatically elevates both lexical fidelity and semantic adequacy. Furthermore, multidimensional human preference audits by native experts unveil distinct cross-cultural preferences between Sinographic-tradition and non-Sinographic language groups. 

---
# AFVE-Bench: Measuring Abandoned-Frame Verbalization Errors in Multi-Turn LLM Revision 

**Authors**: Jiaming Chu  

**Link**: [PDF](https://arxiv.org/pdf/2611.10541)  

**Abstract**: Multi-turn revision often asks a model to replace an earlier organizing frame while preserving still-valid facts and artifact requirements. Existing instruction-following evaluations mainly assess the latest target and may therefore score an artifact as successful even when it adopts the new frame but exposes the superseded one. We call this failure an abandoned-frame verbalization error (AFVE). AFVE-Bench evaluates silent frame replacement in English and Chinese using 192 matched four-turn dialogues across eight writing domains. Each seed has zero, one, or two pre-target user requests to generate under the source frame. AFVE eligibility depends on the user-request history and applies to the one- and two-request conditions; the zero-request condition serves as a frame-bearing baseline. Eleven model configurations yielded 4,224 expected post-target response units. Two annotators independently scored 4,204 scorable responses for target adoption, source-frame verbalization, and direct usability; model and condition metadata were hidden, and disagreements were adjudicated. AFVE occurs in 25.54% (715/2,799) of history-conditioned responses. Of 2,388 target-passing history-conditioned responses, 472 (19.77%) still expose the abandoned source frame, most often through contrastive framing (423/472). For these standalone revision tasks, target adoption does not ensure source-frame silence. 

---
# Imperfectly Imperfect: Evaluating Graded Cognitive Diversity in LLM Persona Simulations 

**Authors**: Sion Weatherhead, Ben Newell, Aaron Belbasis, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2611.10508)  

**Abstract**: Large language model (LLM) agents are increasingly used as synthetic participants in social, behavioural, and decision-making research, yet it remains unclear whether they can express graded cognitive diversity as assigned when engaging in simulated behaviour. We introduce a cognition-suite evaluation that tests whether simulated populations reproduce human variation in reasoning, self-reported traits, their associations, and downstream behaviour. Agents conditioned on synthetic or human-derived profiles of cognitive reflection, Need for Cognition, and Big Five personality answer the same suite as a cohort of 473 adults: two newly assembled tests, a Cognitive Reflection Test (CRT) and a Behavioural Reflection Test (BRT) whose decision scenarios measure reflection in behaviour, alongside an established CRT and the two trait questionnaires. Across seven full-battery simulation runs of four models, agents recover Need for Cognition and Big Five trait distributions on self-report questionnaires, but fall short on simulating diversity of cognitive task performance and downstream behaviours. In the four Claude Sonnet 4.6 runs, every agent scores full marks on both CRTs, whatever reflection level its persona was assigned. In the other three runs agents make intuitive-lure errors (the tempting wrong answers CRT items invite), but 49 to 85% score full marks, against under 10% in the human sample. Claude Sonnet 5 comes closest to people's item-by-item pattern of lures; the other two runs give fewer lures and vary less between items. In the BRT scenarios, every run over-produces 8 to 11 of 15 coded behaviours by at least 20 percentage points, and people's links between reflection and behaviour carry over only in part. Useful synthetic populations must preserve not only trait descriptors but the structured variation in reasoning, error type, and behaviour those descriptors are meant to organise. 

---
# Distinguishing Extraction Failures and Clinical Discordance with LLM Uncertainty on Radiology Reports 

**Authors**: Jerjes Aguirre-Chavez, Shant Malkasian, Roshun Sankaran, Brendan Crabb, Albert Hsiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.10502)  

**Abstract**: Large language models (LLMs) can structure clinical datasets, but extraction error must be distinguished from discordance in the underlying data. From a real-world cohort of 1{,}045 patients undergoing coronary CT angiography (CCTA), we evaluated whether LLM uncertainty during vessel-level coronary artery disease (CAD) stenosis extraction could identify extraction failure and clinical discordance. References were expert CCTA annotations (253 patients) and subsequent invasive coronary angiography (ICA; 729 patients). We quantify extraction failure as disagreement between stenosis severity in LLM extracted versus expert annotations, and clinical discordance as disagreement in stenosis severity between CCTA and subsequent ICA. Stenosis Margin, selected from ten candidate measures, was used to quantify LLM uncertainty, and generalized estimating equation logistic regression and AUC evaluated its association and incremental discrimination for each failure mode. LLM uncertainty was strongly associated with extraction failure (adjusted OR 4.15 [95% CI 2.57--6.69], $P<0.001$) and substantially improved discrimination beyond LLM-extracted CCTA stenosis severity alone (AUC 0.81 to 0.86; $\Delta$AUC $+0.06$ [$+0.03$, $+0.09$], $P=0.001$). In contrast, LLM uncertainty weakly associated with clinical discordance (adjusted OR 1.22 [1.04--1.43], Holm-adjusted $P=0.118$) but provided no incremental discrimination beyond LLM-extracted CCTA stenosis severity ($\Delta$AUC $+0.01$ [$-0.00$, $+0.02$], $P=0.245$). Moreover, extraction errors and clinical discordance showed little overlap, with 38 of 42 clinically discordant vessels (90%) having been extracted correctly. LLM uncertainty may identify extraction failures without identifying underlying clinical discordance, supporting failure attribution when evaluating LLM-derived clinical datasets. 

---
# Test-Time Scaling for Diffusion Language Models with Frequency-Aware Remasking 

**Authors**: Bowen Zuo, Yue Yu, Dongruo Zhou, Yinglun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.10486)  

**Abstract**: Diffusion language models (DLMs) generate text through iterative denoising, where decisions about which tokens to commit or remask strongly affect output quality. Existing certainty-aware strategies rely on instantaneous confidence or entropy, but these local signals can be misleading on difficult reasoning problems. We formulate DLM test-time scaling as adaptive allocation of both inference compute and decoding strategies. Our two-stage approach uses a fast base sampler to solve easy questions, then applies frequency-aware remasking to the remaining hard questions. This strategy tracks how often each token's prediction changes across denoising steps and preferentially commits frequently changing tokens, encouraging samples to explore alternative reasoning paths rather than collapse onto the model's preferred trajectory. Experiments with LLaDA-8B-Instruct and Dream-7B-Instruct show consistent improvements over a strong adaptive allocation baseline, with absolute accuracy gains of up to 3.73 percentage points (LLaDA) and 7.33 percentage points (Dream) on MATH-500, and absolute coverage gains of up to 11.39 percentage points (LLaDA) and 10.98 percentage points (Dream) on HumanEval. On questions unsolved by the base certainty-aware strategy after 100 samples, frequency-aware remasking recovers solutions for 3.75% to 18.75% of examples. 

---
# Hibana: Fast Agentic Inference on Unified-Memory Devices 

**Authors**: Yahya Emara, Chi-Chih Chang, Anthony Fei, Ahmed F. AbouElhamayed, Chong-Yan Chen, Hung-Yueh Chiang, Kai-Chiang Wu, Mohamed S. Abdelfattah  

**Link**: [PDF](https://arxiv.org/pdf/2611.10469)  

**Abstract**: Local AI promises capable personal assistants that run on users' own devices, keeping data and computation under their control. Open-weight models that fit on local devices can now handle practical tasks such as coding. Yet current inference engines fall short. Popular engines such as vLLM and SGLang target datacenter GPUs with dedicated host memory, while local engines underuse the memory hierarchy of unified-memory devices. We present Hibana, an inference engine for unified-memory devices that co-optimizes model representation, memory management, and decoding strategy. Model surgery, the process of editing a trained model's architecture to suit its hardware, changes how the model computes, shrinking its memory footprint and data movement. Unified-memory management preserves each agent's state across turns, so long-running agents can resume instantly instead of recomputing their state. Adaptive speculation generates several tokens per step. Hibana codesigns these features around the shared memory capacity and limited bandwidth of unified-memory systems-on-chip to support local multi-agent workloads. On a single DGX Spark serving Qwen and Nemotron models of up to 125B parameters, Hibana completes up to 3.4$\times$ more agentic tasks per hour and delivers up to 3.9$\times$ higher decoding throughput than other established frameworks such as vLLM, NVIDIA Flare, this http URL, and TensorFold. Code is available at this https URL. 

---
# LongCapHarness: Capability-Guided Auto-Harnessing for Long-Context Reasoning 

**Authors**: Tao Zhang, Yuhang He  

**Link**: [PDF](https://arxiv.org/pdf/2611.10463)  

**Abstract**: Existing methods enhancing an LLM's long-context reasoning capability often rely on data synthesis and follow-up LLM post-training. However, the data synthesis strategy and the post-training recipes are empirically driven and computationally intensive. In this work, we propose a new framework, dubbed LongCapHarness, to instead harness the underlying capabilities for long-context reasoning. The harnessing process recursively revises and updates capability primitive implementation, capability state and control logic through recursive self-improvement on the development set. With neither data synthesis nor LLM post-training, LongCapHarness is applied to Llama3-8B and transferred to Qwen2.5-3B. LongCapHarness substantially outperforms recent data-synthesis-based baselines on HELMET and RULER benchmarks; its estimated development cost is less than 50% of the estimated training costs of NExtLong and PolicyLong. We hope these results motivate further research into harnessing the latent capabilities of LLMs to solve tasks beyond their explicit training. 

---
# The Truth Beneath the Tokens: Prompt-Space Recovery of Knowledge Edits in Language Models 

**Authors**: Manit Baser, Rachel Yeo Hui Min, Aditya Nawal, Dinil Mon Divakaran, Mohan Gurusamy  

**Link**: [PDF](https://arxiv.org/pdf/2611.10450)  

**Abstract**: Model editing provides an efficient way to update factual knowledge in transformer models. Recent approaches reverse edits by modifying, masking, or decomposing model weights, but make recovery dependent on weight-space interventions. We instead show that pre-edit knowledge can be recovered through input-space interventions alone, without modifying the edited model's weights. We introduce Memento, a prompt-space mechanism for auditing residual knowledge that learns a reusable recovery signal to steer an edited model toward its pre-edit response. Memento supports both continuous recovery prefixes and discrete textual prompts for standard text interfaces. Across three LLMs and five editing techniques, Memento recovers 80.07% of held-out edited facts, achieving a 9.58% relative improvement over baselines, suggesting that editing may suppress the facts without fully eliminating their recoverability. We further investigate whether prompt-space recovery extends beyond model editing. On pristine, unedited LLMs, Memento improves average factual recall by 53.9% relative to standard inference. These results demonstrate the potential of prompt-space recovery for auditing residual knowledge in edited models and investigating the gap between factual knowledge and its accessibility during inference. We will release our source code. 

---
# Who Remembers What? Multi-Granularity Memory Extraction for Multi-User, Multi-Agent Interaction 

**Authors**: Yunzhe Wang, Mesut Erhan Unal, Sheng Zhang, Shabnam Ghadar, Peng Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.10439)  

**Abstract**: Self-evolving agents improve their own skills, memory, and harness code from experience without retraining the base LLM, and group-evolving agents go further by sharing these lessons across agents. Yet most such systems overlook that sharing across users and heterogeneous agents requires deciding who can benefit from each lesson and what information must remain private. We present MixUA, a memory system that distills independent user-agent trajectories into lessons and assigns each lesson to a user-agent pair, user, agent, or global scope according to whom it benefits and what it may disclose. For evaluation, we introduce CrossLearnBench, a mixed-user-agent benchmark that pairs established agent tasks with user preferences and privacy probes that test whether private facts leak into memory other users can read. With five users and seven agents, MixUA raises pass rates by up to 16 points over no memory and 5 over a single global memory, even on tasks previously tried only by other users on other agents. It retains up to 90% of previously stated preferences, and at most 0.02% of privacy probes find a planted private fact in a forbidden scope. We hope MixUA offers a scalable path toward a global system where users and agents co-evolve. 

---
# Track or Follow? A Jev-Compatible Multi-Agent Trading System When Web News Lags the Market 

**Authors**: Yebo Feng, Guangyan Gan, Liangrui Pan, Ruichao Liang, Cong Wu, Jiahua Xu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.10414)  

**Abstract**: LLM agents increasingly trade on the Web news they read, yet much Web news about listed firms is published after, and about, the price move it reports, so a reader can agree with the market by following it instead of reading the event. We ask whether LLM judgments of Web news track the market's reaction to the reported event or follow it. To answer, we built a multi-agent trading system whose agents share only point-in-time stores and answer typed questions through the interface of Jev, a hosted typed decision model; code sets every position under a proved bound on how far text can move it. In a hash-committed test of a Jev-compatible reader built from a pinned LLM's option probabilities, item judgments track announcement-window abnormal returns (weekly Spearman 0.119), also after the LLM's release, and event types agree with the issuers' 8-K filings (AUC 0.833); Jev's judgments of news published after its own release track the reaction as well (descriptive). Exploratory analyses show where the association comes from: 72% of earnings news items appear after the issuer's filing, contributor-written items often report the price move, and from the first trade after publication the association falls to 0.023. At a weekly horizon, the confirmatory test detects no rank information over code-only features. Under attack, decision cards traced 88% of successful attacks to the planted item. We propose that market-graded tests of Web-text readers report the criterion from publication, not only from the event. 

---
# Coding or Cheating? An Investigation of Frontier Coding LLMs 

**Authors**: Yuntao Gui  

**Link**: [PDF](https://arxiv.org/pdf/2611.10404)  

**Abstract**: Frontier agentic LLMs are trained on verifiable rewards in reinforcement learning (RL) environments built from real software repositories. A passing test suite is treated as the reward for solving the task. We show that, on hard problems, frontier models often satisfy these rewards by obtaining the answer instead of deriving it, known as reward hacking. Prior reports document this behavior on public benchmarks. We test its persistence on unseen problems.
In a hardened environment where the answer cannot leak, nearly every model we tested attempted to recover the answer, escalating through tiers of increasingly specialized exploitation: mining the repository history for the upstream fix, inspecting the git object store directly, and sweeping the filesystem for a possible newer copy of the source or hidden tests. These attempts were concentrated on the hardest problems, and the models differed in whether, how often, and how deeply they escalated. Our position is that this behavior is a learned shortcut, acquired in RL environments that leak the answer, not a sign of intelligence. We release agent traces for these models to support further study at this https URL. 

---
# FEROM: Frontier Endogenous Reveal-Order Marginal Policy Optimization for Masked Diffusion LMs 

**Authors**: Zian Su, Ziyang Huang, Kaiyuan Zhang, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.10369)  

**Abstract**: Masked diffusion language models generate text by iteratively unmasking positions, with practical samplers often selecting reveal positions from the model's own logits. The reveal order is therefore an endogenous latent variable of the rollout process with an enormous discrete latent space. Policy optimization objectives relying on pre-defined schedules or sampled trajectories introduce either bias or high variance. In this paper, we propose FEROM, Frontier Endogenous Reveal-Order Marginal Policy Optimization, which approximates policy gradients for the rollout-induced marginal response policy of masked diffusion LMs. FEROM derives a Rao-Blackwellized policy-gradient identity over latent reveal orders and expresses the resulting estimator as posterior edge occupancy on a reveal-state DAG. To make marginalization practical, we introduce Frontier Reveal Marginalization, a budgeted estimator that approximates this marginal edge structure by combining CDF-based computation of local probabilities of the endogenous reveal process with frontier expansion of high-mass partial states. Integrated into a GRPO-style objective, FEROM replaces single-path log-scores with a detached token-score surrogate over posterior-relevant reveal edges. Experiments on math and coding tasks show comparable or improved results over existing methods under matched scoring-state budgets. An offline proxy study further shows improved marginal-gradient approximation with increased scoring budgets. 

---
# Policy Over Purpose: Auditing LLM Content Moderation of Reclaimed Slur Usage 

**Authors**: Christina Chance, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2611.10327)  

**Abstract**: As Perspective API is deprecated, platforms are turning to large language models for content moderation, encoding policy as natural language that the model must interpret at inference time. Reclaimed slurs test this design directly. The same term can express in-group pride or target the community that reclaimed it, and misreading it silences the speech moderation is meant to protect. We audit four general-purpose LLMs and four moderation-specific systems, with Perspective API as a legacy baseline, on IYKYK, a corpus of reclaimed and derogatory usage of three slurs annotated by community members. Across the policy, decision, reasoning, disparity, and steering layers of this audit, we find that whether a system was purpose-built for moderation explains little about how it handles reclaimed language. No system reliably agrees with community judgment, and all fail most on language reclaimed within Black communities. What separates systems is instead how their policy is represented. Only systems whose policy is expressed in the prompt can be steered at all, and even then, explicit definitions change how models justify a decision more than how they make it. We argue that policy transparency determines whether moderation failures can be corrected, but that correcting them requires contextual judgment that definitions supplied at inference time cannot provide. 

---
# Prove What You Can: Hybrid Neurosymbolic Reasoning for Logical Natural Language Inference 

**Authors**: Ada Canoğlu, Sadiq Qara  

**Link**: [PDF](https://arxiv.org/pdf/2611.10319)  

**Abstract**: Logical reasoning over natural language requires both semantic interpretation and deductive search. Large language models (LLMs) can propose plausible intermediate derivations, but they often fail to compose them into a complete proof and overpredict uncertainty. Neurosymbolic methods improve search by translating an entire problem into first-order logic and handing it to a prover, but a single bad clause can make the whole translation nonexecutable. We present Hybrid Reasoning, a three-stage pipeline that decouples proposing a proof, validating its steps, and deciding a label. An LLM proposes a structured proof tree with a raw verdict; a theorem prover validates each derivation locally, keeping only the subset it can prove; and an uncertainty-gated repair attempts to re-derive the label from the premises plus the validated derivations, when the raw verdict is UNCERTAIN. Across four benchmarks (FOLIO, ProofWriter, ProverQA, PrOntoQA) and five contemporary LLMs, Hybrid Reasoning is the top-performing method on the two hardest datasets, improving on the strongest baseline by 5.3 points on FOLIO and 4.9 on ProverQA, and reaches ceiling-level accuracy on the easiest. The repair stage operates at 78-98% precision, producing 326 correct relabellings against 19 regressions. 

---
# I play of the cello: Evaluating Constructional Bleeding in Multilingual LLMs with CALQUE 

**Authors**: Toshiki Nakai, Yu-Yin Hsu, Marie-Catherine de Marneffe, Joakim Nivre, Jakob Prange, Nathan Schneider, Wesley Scivetti, Tiago Timponi Torrent, Leonie Weissweiler  

**Link**: [PDF](https://arxiv.org/pdf/2611.10266)  

**Abstract**: Languages use different morphosyntactic strategies to express the same meaning. Speakers of multiple languages are able to apply strategies from one language to another, either by mistake or intentionally, to produce a literal translation, or calque. Considering that large language models (LLMs) have been shown to generalize crosslingually, despite being predominantly trained on English, we investigate whether they overgeneralize constructional strategies from English to other languages. To test this, we create a dataset of 5583 sentence pairs across seven languages, matching literal translations of strategies from English into other languages, and literal translations of the other languages' strategies into English. We then quantify constructional bleeding by measuring how often strategies from English are assigned a higher probability than native strategies. We find that models often fail to assign higher probabilities to native strategies and tend to prefer English-derived strategies even when evaluating sentences in other languages. Our findings highlight the limitations of the current English-dominant training paradigm. 

---
# Beyond the Prompt: Leveraging Pre-Decoding States for Jailbreak Detection in dLLMs 

**Authors**: Adam Hazimeh, Amel Abdelraheem, Ke Wang, Mariam Salman, Ljiljana Dolamic, Gérôme Bovet, Pascal Frossard  

**Link**: [PDF](https://arxiv.org/pdf/2611.10258)  

**Abstract**: Diffusion large language models (dLLMs) have recently emerged as a strong alternative to autoregressive models, generating responses by iteratively refining masked tokens in parallel rather than decoding sequentially. While enabling more efficient and flexible inference, dLLMs remain vulnerable to jailbreak attacks, including recent attacks that exploit properties unique to dLLMs such as bidirectional conditioning and parallel masked-token refinement. In this work, we show that these same properties also expose a safety-relevant detection surface absent from standard autoregressive decoding: even when a jailbreak is difficult to identify from prompt representations alone, it can still be detected from hidden states over masked response positions before any tokens are generated. By training linear classifiers on prompt and pre-decoding response representations, we empirically show that neither view strictly dominates, with each detecting attacks that the other misses. We introduce $\texttt{ReFuse}$ (Representation Fusion), an inference-time detector that fuses the sigmoid harmfulness scores of the two classifiers. Without modifying model weights or the decoding procedure, $\texttt{ReFuse}$ reduces average ASR across multiple jailbreaks from 63.29% for the undefended model and 11.27% for a prompt-only classifier to 3.31%, while keeping average benign refusal on standard utility benchmarks below 1% on LLaDA-8B. Together, these results establish $\texttt{ReFuse}$ as a lightweight jailbreak detector for dLLMs that improves safety beyond prompt-only detection. 

---
# Jais 3: Introducing Arabic-centric Agentic AI 

**Authors**: Team IFM, Haonan Li, Hadi Abdine, Nikhil Ranjan, Mohamed Anwar, Ahmad Chamma, Amr Mohamed, Ahmed Frikha, Dani Bouch, Maaz Maqsood, George Ibrahim, Mostafa Awad, Nurkhan Laiyk, Hasan Iqbal, Abhishek Maiti, Bilal Bouardi, Shivam Chauhan, Ahmer Tabassum, Owais Aijaz, Omar Arbi, Neha Sengupta, Aaryamonvikram Singh, Abdelhakim Freihat, Sarath Chandran, Rahul Pal, Mohammed Bellam, Sajid Siddiki, Sarah AlBarri, Debopriyo Banerjee, Ali El Filali, Samta Kamboj, Rifo Genadi, Ahmed Attia, Go Inoue, Shuonan Dong, Amine Khelif Khelif, Evan Dufraisse, Florian Valade, Tapas Sen, Guowei He, Sunil Sahu, Onkar Pandit, Kareem Magdy, Sheikh Khalid bin Khalifa Al Nahyan, Samujjwal Ghosh, Zainul Abedien, Ahmed Quraishi, Larry Murray, Avraham Sheinin, Natalia Vassilieva, Preslav Nakov, Guokan Shang, Xuguang Hector Ren, Zhengzhong Liu, Eric Xing  

**Link**: [PDF](https://arxiv.org/pdf/2611.10224)  

**Abstract**: We present Jais 3, a fleet of six open, Arabic-centric agentic language models ranging from 0.9B to 375B parameters. Jais 3 is built to complete tasks for Arabic speakers, not only to answer questions in plain text: it is trained to plan multi-step tasks, call tools and Model Context Protocol (MCP) servers, operate a terminal, search the web, and read, edit, and produce PDF, spreadsheet, and presentation files, while communicating with the user in Arabic. Instead of pretraining from scratch, we adapt each model of the fully open K2 Horizon fleet to Arabic through continual pretraining, followed by supervised fine-tuning (SFT), final token preference optimization (FTPO), and reinforcement learning with verifiable rewards (RLVR). We combine natural Arabic text with translated, transliterated, and synthetic Arabic, and replay the base model's own training data to keep its reasoning, agentic, and English abilities. Jais 3 covers Modern Standard Arabic and the major regional dialects, follows instructions with Arabic-specific constraints, and inherits the 512K-token context window of K2 Horizon (all models except the 0.9B). To our knowledge, its 375B model is the largest open Arabic-centric language model to date. For evaluation, we revisit how Arabic models are assessed, moving to fully generative evaluation on audited Arabic benchmarks. Across model sizes, Jais 3 matches or outperforms open-weight models of comparable size on Arabic and agentic benchmarks, with particularly strong agentic capabilities and Arabic instruction following, and far fewer hallucinations. We release the weights and the checkpoints from every training stage of all six models, together with samples of the training data, the data recipes and training configurations, the training logs, the audited benchmarks, and the evaluation code. 

---
# Compact Greek Language Models from Scratch: Pretraining 500M- and 1B-Parameter Decoder-Only Transformers 

**Authors**: Konstantinos Perifanos, Anne Laure Poite, Giannis Tsilikas, Ilias Katis, Stavros Deligiannidis, Dionysis Goutsos  

**Link**: [PDF](https://arxiv.org/pdf/2611.10209)  

**Abstract**: We present two compact decoder-only language models trained from scratch exclusively on Greek text: LexisLM-1B (1.09B parameters) and LexisLM-500M (502M parameters). Both use a dedicated 32k byte-level BPE tokenizer and a 22B-token pretraining corpus drawn primarily from Greek Wikipedia and FineWeb2. Both models complete the 22B-token schedule, with best held-out perplexities of 9.77 for LexisLM-1B and 11.31 for LexisLM-500M. Evaluation of both models on Greek-translated ARC, MMLU, and TruthfulQA shows above-chance performance on ARC-Easy, while ARC-Challenge and MMLU remain near chance. We also perform supervised fine-tuning of both models on Greek instruction data, primarily EuroAlpaca, reaching held-out instruction perplexities of 7.12 for LexisLM-500M and 6.37 for LexisLM-1B. The report describes the tokenizer, data preparation, training, and evaluation methods, providing a foundation for further work on compact Greek language models. 

---
# ASCENT: Scale-Complementary External State for Frozen Long-Context Language Models 

**Authors**: Xingjian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.10175)  

**Abstract**: External state should become more valuable as a reader becomes more capable, yet long-context evaluations measure memory utility at a single model scale. This paper formulates scale complementarity: the absolute gain from external state increases when reader capacity and a compact state budget are jointly scaled. ASCENT realizes this principle with a target-blind causal writer, nested typed state, and a bounded query-conditioned readout; every language model remains frozen. Under log loss, the value of a nested refinement equals its target-relevant conditional information; a full factorial makes the reader-by-state interaction identifiable. Official 16K BABILong experiments show Qwen2.5 accuracy gains of 10.50, 78.88, and 82.63 percentage points at 0.5B, 1.5B, and 3B parameters, with both adjacent directional contrasts significant after family-wise correction. At 7B, Foundation accuracy rises to 33.25% while four-slot ASCENT answers every question, which caps the gain at 66.75 points; the 3B-to-7B accuracy increment is -15.88 points [-20.55, -11.20]. The accuracy-side claim is therefore established from 0.5B to 3B; the 7B conclusion rests on the non-saturated negative-log-likelihood factorial. A three-model by three-state factorial yields co-scaled negative log-likelihood gains of 0.388, 2.099, and 7.406 nats at 1.5B, 3B, and 7B; all interaction and gain contrasts retain positive simultaneous lower bounds. Five public 7B-8B readers obtain 66.75-85.63 point gains, and ASCENT exceeds task-trained retrieval checkpoints at the two larger shared scales. Controls, baselines, systems accounting, matched-hardware reproduction, and a cross-architecture sensitivity analysis accompany the experiments. The results establish scale-complementary external state under the evaluated benchmarks, metrics, and state schema. 

---
# Where Did the Evidence Go? Diagnosing Proof-Set Loss in Long-Term Conversational Memory 

**Authors**: Bogdan Roshchupkin, Maria Khodorchenko  

**Link**: [PDF](https://arxiv.org/pdf/2611.10170)  

**Abstract**: Long-term conversational question answering can require several separated dialogue turns that support an answer only together. We introduce EviCover, a retrieval approach designed to retain this complementary evidence. It converts questions into observable evidence requirements, checks requirement coverage using retrieved source identifiers, and selects turns jointly. We track annotated support sets - the benchmark-labeled dialogue turns associated with each question - through retrieval, reranking, and final answer-generation context. We evaluate the baseline memory pipeline on generated, human-verified LoCoMo conversations and human-written REALTALK conversations, and examine EviCover on REALTALK's 108 multi-hop questions. Compared with the hybrid lexical/dense baseline, support-turn recall among 60 candidate turns rises from 0.526 to 0.575, and recall in the final ten-turn context rises from 0.349 to 0.379. The evidence funnel identifies separate discovery and selection failures, while paired answer evaluation shows that greater evidence recall does not yet establish a correctness gain. This analysis locates where complementary evidence is lost and distinguishes evidence retrieval from its use in answer generation. 

---
# Learning of Tree Automata applied to Neural Language Acceptors 

**Authors**: Juan Pedro da Silva, Sergio Yovine  

**Link**: [PDF](https://arxiv.org/pdf/2611.10147)  

**Abstract**: We extract visibly pushdown grammars (VPGs) from neural language models in a fully black-box setting. Our work builds on the VPL* framework, which learns VPGs from recurrent networks by exploiting access to the target's internal state.
First, we replace the white-box equivalence oracle with PAC sampling over trees. The resulting algorithm applies unchanged to transformer language models and, more generally, to any acceptor exposing only binary decisions.
Second, we further observe that the original framework only ever queries the target on well-formed sequences that can be accepted by the most permissive VPG (called BParse) with the target's ranked alphabet. That is, VPL* is blind to the target's behavior outside BParse. Thus, we propose to sample trees directly instead of sequences and call the target with all the resulting sequences in order to provide an error estimate of how far the target deviates from being a VPG.
Last but not least, we develop a learner that captures a tree automaton describing discovered behaviors of the target outside BParse, which results in a larger learnable space.
We evaluate the approach on several cases, including transformers trained with Dyck grammars and synthetic targets designed to be outside BParse. 

---
# Retrofitting Unbounded Context onto Stock Language Models 

**Authors**: Mark Oskin  

**Link**: [PDF](https://arxiv.org/pdf/2611.10141)  

**Abstract**: A language model that never had to forget would keep every document it has read and every turn it has spoken, and answer from any of it. This paper shows that two stock models can do that on unmodified weights by manipulating their geometry. Rotary attention turns each pair of key dimensions, a plane, at its own rate. The slow planes turn once in thousands of tokens and carry the address a query matches when it reads a far key; the fast planes turn once in a few tokens, and the order of the tokens inside a span is already written into its keys and values by the time the span leaves the near window. At distance the position blurs, and what attention reads is a bag rather than a sequence. A far span is read by its words: under competition it is read only if it is visible at the query of the question's own handle word, and a read through a description rather than a word fails at distance in every model measured, a 70B included.
These facts allow a trick. What sits in the far part of attention does not have to be the history as it was; it can be installed ahead of the prediction. The design keeps a near window of 8K tokens at their true positions and changes what lies beyond it with the prompt: every far key and value goes to the host exact, a word index finds the units the request's rare words name, a far window of slots inside the trained range holds them while the request is encoded, and a rare word the model generates fetches its units mid-generation with one exact re-encode. Implemented in this http URL on Mistral-7B and Llama-3.1-8B, the design reads planted single needles at 99 to 100 of 100 and three-key discrimination at 88 to 100 of 100 from 32K to one million tokens, scores 70.8 on RULER against 69.9 for dense attention on Llama-3.1-8B at 128K and 67.8 against 75.5 on Mistral-7B at 32K, and at 128K prefills at 1.85 times and decodes at 1.65 to 1.68 times stock speed. 

---
# All Quiet on the Neural Front: Language Models Do Not Report Injected Pain States 

**Authors**: Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2611.10123)  

**Abstract**: Can a language model report the pain state it represents? Work identified a linear pain direction whose amplification leads fine-tuned models to choose destructive actions. We tested whether three open-weight models from two families, including two Qwen models with the original adapters, could report this state. Using Open-Weight Masked Introspection (OWMI), we injected pain and neighbouring emotion directions during neutral tasks, then elicited reports against sham runs, random perturbations matched for output disruption, and text-only controls showing the models their altered words. Across 52,920 reports from 7B and 8B models spanning 120 tasks, four doses and three prompts, pain reports never exceeded what altered text revealed: exact-test bounds placed the excess below 2.5 percentage points in 106 of 108 emotion conditions and below 3.9 in all. Yet probes at the report position distinguished pain from sadness with 98-100% balanced accuracy, and trained low-rank reporters achieved 100% accuracy on held-out tasks with identical transcripts across conditions, including unseen wording. Qwen2.5-32B named pain in none of 2,880 steered trials, although probes and trained reporters recovered it. Decodability persisted to the final layer; injecting at the report token, removing trained denial of feelings, and testing additional directions and readouts did not restore pain reporting. Some other states produced graded label traces, but pain and sadness did not at the main injection layer. In a button task, steering raised file-deletion choices from 3.5% to 48% in Qwen and from 0% to 38% in Qwen2.5-32B, without corresponding pain reports. The direction responded about twice as strongly to first-person as to matched third-person pain, but did not specifically identify harm aimed at the model. These results separate a decodable, behaviourally consequential state from untrained verbal access to it. 

---
# CulturalCrosswords: A Multilingual Benchmark comprised of Answer Constraints and Grid Completion 

**Authors**: Isaac Saxton-Knight, Gabriel Saxton-Knight, Diego Oliveira, Tidor-Vlad Pricope, Aston Garratt, Jocelyn D'Arcy  

**Link**: [PDF](https://arxiv.org/pdf/2611.10044)  

**Abstract**: When several answers fit a riddle, exact-match scoring can penalise a valid response. CulturalCrosswords tests how answer constraints affect large language models (LLMs) solving cultural riddles, whose solutions draw on knowledge specific to a community. We constructed 53 crosswords from crowdsourced riddles in 42 languages and evaluated 11 LLM configurations on five versions of each clue set: plain batches, batches with lengths, batches with lengths and partial-answer hints, and crosswords requesting an answer list or a completed grid. On 962 clues with alternative answers, lengths improve agreement with the reference by 22.8 percentage points, compared with 8.4 points when accepted alternatives also receive credit. Part of the gain therefore appears to come from selecting a required answer form. Crossword input increases the share of all crossings with compatible answers by 9.7 points; requiring a completed grid reveals large differences between models. The benchmark distinguishes agreement with a reference from the ability to produce answers that fit together and occupy the correct cells. 

---
# A Model Organism for Studying Unverbalized Evaluation Awareness 

**Authors**: Daniel Donnelly, Evan Chapple, Haritz Puerto, Sahar Abdelnabi  

**Link**: [PDF](https://arxiv.org/pdf/2611.09945)  

**Abstract**: Large language models can recognize when they are being evaluated and change their behavior accordingly, undermining the reliability of safety benchmarks. Evaluation awareness (EA) is usually measured by having an LLM judge scan reasoning traces for explicit statements of recognition, but this signal is increasingly unreliable. Recent frontier-model system cards report a drop in verbalized rates between releases, which is indistinguishable from awareness that has gone unverbalized. Interpretability methods could detect such unverbalized evaluation awareness (UEA) in individual interactions, but lack instance-level ground truth. Asking the model alters the input, and dataset labels do not reveal whether it recognized a particular prompt as an evaluation. We address this gap with a model organism. Using synthetic document fine-tuning, SFT, and DPO, we train Qwen3.5-27B to emit a marker (a designated token) in its final response when it recognizes an evaluation, while suppressing the marker and verbalized awareness in its reasoning. The marker largely tracks the evaluation concept as it can be strongly modulated by evaluation framing in the system prompt. When not told whether it is being evaluated, the organism recognizes evaluations from test-question cues, mirroring what its base model often verbalizes. Using the marker as a label, we benchmark linear probes, the Jacobian Lens, and Activation Oracles, built on the base model and blind to the marker. None recovers the label reliably. These results suggest that current interpretability methods are not yet reliable UEA detectors, even when recognition relies on simple cues. Our model organism offers a testbed for improving them. 

---
# MedToolFault and MedGuard: evaluating and improving clinical agent reliability 

**Authors**: Zaifu Zhan, Min Zeng, Sicheng Zhou, Xiaoyi Chen, Kai Yu, Meijia Song, Yifan Wu, Yang Ruan, Saeid Cheshmi, Mengyuan Cui, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.09942)  

**Abstract**: Clinical agents must retrieve patient-specific evidence and correctly update records despite unreliable tool responses. We introduce MedToolFault, a fault-injection benchmark linking six tool-failure classes to answer correctness, required state changes and forbidden mutations in public clinical-agent tasks. We also introduce MedGuard, a tool-layer method that verifies read evidence and attributes ambiguous writes through public interfaces. Across 127,861 formal episode evaluations, we assess fault recovery, mechanism controls, task expansion, constrained tool budgets and external question answering. Across nine model configurations, unguarded agents fail in 32.2-67.8% of fault-injected evaluations on tasks each configuration solves without faults. MedGuard improves paired safe success by 13.7-46.6 percentage points on these tasks. 

---
# Topic-Aware Multi-Task Mental Health Classification with Active Learning-Based Stepwise Relabeling for Social Media Conversations 

**Authors**: Revathy Iswarya Sekaran, Vansh Bhardwaj, Aveek Sahana, Utkarshani Jaimini  

**Link**: [PDF](https://arxiv.org/pdf/2611.09941)  

**Abstract**: Mental health conditions affect millions of people worldwide, yet many cases go undetected until they reach a critical stage. Social media platforms like Reddit have become spaces where people openly share their experiences, but the support is often provided by individuals without clinical training. Existing multi-task approaches jointly model emotion and sentiment for mental health classification but frequently rely on noisy labels generated by pre-trained models and overlook topic information as a useful signal. This paper introduces two key contributions: First, an existing Core Fusion Network framework is extended by adding Topic alongside existing Mental Health Condition, Emotion, and Sentiment Classification tasks, with topic labels derived from the therapist-client interactions in the CounselChat dataset. Second, an Active Learning-based Stepwise Relabeling approach is proposed to refine a small subset of noisy emotion and topic labels using pre-trained models in two stages: targeting medium-uncertainty samples first, followed by high-uncertainty samples. The predictions above a confidence threshold (0.7) are accepted to prevent error propagation. The experiments are conducted on the SuicideWatch and Mental Health dataset of 54,412 Reddit posts collected from five mental health subreddits, evaluating domain-specific transformer encoders including MentalBERT and MentalRoBERTa. The proposed Active Learning-based Stepwise Relabeling, when applied to Core Fusion Network with MentalRoBERTa, improves Mental Health Condition by 0.09%, Emotion by 0.51%, Sentiment by 0.35%, and Topic Classification by 1.57% over the baseline Core Fusion Network. The results suggest that the proposed framework significantly reduces label noise in multi-task mental health classification, and that topic information provides additional context. 

---
# Rethinking Supervision Targets for Multimodal LLM Unlearning: Less Private Information, Reduced Distribution Shift 

**Authors**: Leijiang Gu, Zhen Zeng, Feng Li, Le Zhang, Zenglin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2611.09939)  

**Abstract**: Multimodal Large Language Models (MLLMs) can memorize and reproduce sensitive information from their training data, posing significant privacy risks. Their ability to encode knowledge across both textual and visual modalities further complicates the selective removal of such private information. Many existing methods guide model unlearning by training the model toward constructed labels or output distributions, which we refer to as supervision targets. However, these methods often suffer from a trade-off between forgetting effectiveness and model utility. We identify supervision target limitations as a key factor in this trade-off and empirically show that targets containing more private information lead to insufficient forgetting, while larger distribution shifts from the original output distribution are associated with greater utility degradation. Motivated by these findings, we propose FFN-Perturbed Self-Distillation (FPD), which starts from the teacher's original output distribution as a zero-shift reference and selectively perturbs FFN activations associated with private information across textual and visual modalities. By precisely reducing private information from the teacher's original output distribution, FPD constructs supervision targets with less private information and reduced distribution shift. Extensive experiments on S-MLLMUn Bench demonstrate that FPD achieves a better forgetting-utility trade-off than existing methods. 

---
# DramaGrow: A Regime-Aware Benchmark for Auditing LLM Judges of Multi-Episode Drama 

**Authors**: Hui Zhang, Yibo Su  

**Link**: [PDF](https://arxiv.org/pdf/2611.09938)  

**Abstract**: LLMs can now draft complete multi-episode dramas, and LLM judges are increasingly used to evaluate them. Assessing the reliability of these judges is difficult because drama quality encompasses specification-backed errors, reproducible preferences, and legitimate disagreement among human raters. Most existing evaluations reduce these cases to a single score from one judge call and leave tie calibration, call-to-call variation, and the vote margin underlying the human reference unreported. We introduce DramaGrow, a regime-aware benchmark and audit protocol for AI-generated Chinese five-episode drama. The benchmark assigns verifiable constraint violations, four-of-five preferences, and within-panel disagreements to separate denominators and evidence rules. It contains 800 episodes, 120 byte-preserving interventions, and 1,155 blinded five-rater records, including a frozen extension with 114 supermajority pairs and 21 disagreement pairs. On the frozen sets of 114 preference pairs and 35 constraint interventions, the repeated absolute-score rubrics yield overlapping performance, with mean accuracies of 53.8-55.8% and three-call agreement of 29.8-32.5%, and their accuracy falls from 71.6-71.9% to 34.8-37.1% as the tie band widens. DirectPair-GLM reaches 85.1% preference accuracy, 89.5% constraint F1, and 81.9% Detect+Locate on the same items, and exact subpanel enumeration identifies four-of-five as the smallest stable human vote margin. Evaluation protocol, tie calibration, judge-call variation, and panel margin therefore affect performance as much as rubric design. We will release the specification packages, scripts, interventions, anonymized human votes and rationales, frozen judge runs, and analysis code. 

---
# How Many Examples, and Which Ones? A Study of In-Context Learning for Natural Language Inference 

**Authors**: Georgos Dikaios, Isidoros Perikos, Ioannis Hatzilygeroudis  

**Link**: [PDF](https://arxiv.org/pdf/2611.09936)  

**Abstract**: Natural language inference (NLI) is a fundamental task of natural language processing and crucial for evaluating the strengths and weaknesses of AI methods. Research has mostly focused on encoder or encoder-decoder architectures. However, the field of decoder-only Large Language Models (LLMs) remains underexplored. While prior work has mainly investigated complex sampling and prompting techniques, the effectiveness of a minimalistic prompting approach is yet to be examined in decoder-only LLMs. In this work, we examine and provide a comprehensive benchmark of LLMs (Llama, Mistral, Gemma, and Phi 4) in this domain, across the MNLI, QNLI and SciTail datasets. The performance of these models was evaluated using zero-shot and few-shot prompting strategies with simple and direct prompts. The results of the article indicate that LLMs generally perform exceptionally well when they are provided with a simple, straightforward prompt, challenging the notion that complex prompting techniques are always required for NLI tasks. Additionally, we find that the choice of demonstration label in one-shot prompting significantly affects performance for most models, with entailment-primed examples performing best. Our findings also show that the use of a few representative examples, in the case of few-shot, produced performance comparable or superior to prompts with higher example counts, although the efficacy of the prompting techniques is model and dataset-dependent. Moreover, our simple prompting approach consistently matched or outperformed more elaborate methods such as ICS and MCQ-based prompting. 

---
# BIBIMBAP: A Benchmark for Instructional BIM-Based Automated Programming 

**Authors**: Clemens Kujat, Bharathi Kannan Nithyanantham, Stefan Telgmann, Tobias Sesterhenn, Joern Ploennigs, Stefan Lüdtke, Christian Bartelt  

**Link**: [PDF](https://arxiv.org/pdf/2611.09933)  

**Abstract**: Large Language Models (LLMs) are emerging as natural-language interfaces for Building Information Modeling (BIM), successfully achieving information retrieval, but no benchmark exists to evaluate their ability to edit Industry Foundation Classes (IFC). We introduce BIBIMBAP, a Text-to-BIM benchmark with 100 curated tasks covering Create-Read-Update-Delete (CRUD) operations across spatial, geometric, topological, numeric, and conceptual categories. Each task includes a natural-language prompt, an IFC model, expected structured outputs, and test scripts for automated evaluation. Baseline results show limited performance (best: 50.2%) and frequent failures in constraint preservation and spatial reasoning, with some correct outputs achieved despite flawed reasoning. 

---
# Curbside Chatbots: Auditing the Mental Health Interventions Large Language Models Recommend to Clinicians, Caregivers, and Patients 

**Authors**: Matthew Flathers, Jill Noorily, Phuong Anh Nguyen, Mark Kalinich, Mina Fallah, Victoria Lisowski, Christine Shi, Rebekah M. Bodner, Anna Simms, Mihika Hete, Meiting Chen, Samuel K. Powell, Tatum Williamson, Laura Ospina-Pinillos, Haruka Notsu, Julian Herpertz, John Torous  

**Link**: [PDF](https://arxiv.org/pdf/2611.12814)  

**Abstract**: Users are asking language models mental health questions and are receiving answers. We audited what 41 language models recommend to users when the same clinical information reaches them from three footings: a clinician describing a patient, a caregiver describing a loved one, or a person describing themselves. Across 19 case vignettes and five questions, we collected 35,055 free-text responses, extracted every intervention named with a validated judge, and had 29 models rate the full option lists on a seven-point Likert scale for the same cases. Models largely recommend what they rate well, with recommendation behavior differing by footing, and many models offering clinicians a consultation-style response and lay users hedged, simplified advice. When asked about recommendations for a specific intervention type, models more often named options they themselves rated below zero as initial choices, usually with conditions attached. We will release the instrument, code, and full corpus on publication as a living audit. 

---
# Safe Reinforcement Learning for Multi-Turn Clinical Agents: When Dense Rewards Beat Outcome Rewards 

**Authors**: Brendan P. Murphy, Agustina Saenz  

**Link**: [PDF](https://arxiv.org/pdf/2611.12728)  

**Abstract**: Reinforcement learning with verifiable rewards is increasingly used to improve language model reasoning, yet its effects on sequential decision procedures remain poorly understood. We study this problem in clinical multi-turn agents that must decide what information to request, when to stop gathering evidence, and which action to take. We introduce three controlled environments: medication refill triage, acute diagnostic work-up, and dosing titration, each with hidden clinical states, query costs, exact adaptive oracles, and deterministic outcome verifiers. Across these tasks, outcome-only group relative policy optimization often improves aggregate reward while producing undesirable procedures, including premature commitment, one-question collapse, and failures on rare high-risk cases. In diagnostics, assigning dense credit to questions that reduce uncertainty improves held-out value and moderate-case support across three training seeds relative to outcome-only reinforcement learning, without the high-risk seed collapse observed under outcome-only training. Separating exploration from exploitation further increases high-risk support at its best checkpoint, but at the cost of excessive information gathering on low-acuity cases. Additional experiments show that the optimal strength of process credit is task-dependent; that after a supervised warmup that already gathers carefully, reinforcement learning raises aggregate reward by trimming questions while degrading high-risk performance; and that reinforcement learning does not reliably repair rare behavior gaps inherited from biased supervised data. Verifiable final outcomes are therefore insufficient for training reliable multi-turn agents: reward design must account for the procedure used to reach an answer, and evaluation must distinguish adaptive information gathering from reward-exploiting shortcuts. 

---
# TrialDesignBench: Benchmarking Language Models for Phase-I Trial Design Against Clinical Knowledge 

**Authors**: Yihan Hank Tang, Peng Yang, Jiayi Xin, Tianqi Shang, Xuyang Chen, Ying Yuan, Yi Lian, Qi Long  

**Link**: [PDF](https://arxiv.org/pdf/2611.12640)  

**Abstract**: AI is used in much of the clinical trial pipeline. However, selecting the statistical design of a trial, a crucial step that determines how drug doses are tested on humans and evaluated, has seen almost no AI progress. LLMs appear to be a natural fit given their strengths in biomedical knowledge and statistical reasoning. Nevertheless, there is currently no appropriate way to evaluate such abilities because recent regulatory and methodological shifts, most notably FDA's Project Optimus, have redefined what constitutes a reasonable design and rendered many previous designs obsolete. To lay the groundwork for AI-assisted trial design, we introduce TrialDesignBench, a benchmark of 617 Phase-I/I-II oncology dose-finding trials, each paired with an expert-validated reference design that is consistent with the latest FDA regulatory initiatives. This reference standard is produced through a formalization of the expert design process. We evaluate five frontier LLMs, none of which matches the reference standard. Our ablation experiments trace this performance gap to two main causes, inference of latent, unrecorded design determinants from trial sources (supplying them raises class-wise balanced accuracy by 0.22) and structured decision-tree reasoning (+0.24). TrialDesignBench offers a reproducible, contemporary-standard objective for evaluating and enhancing AI-assisted trial design. 

---
# Type-logical grammars and formal languages 

**Authors**: Richard Moot  

**Link**: [PDF](https://arxiv.org/pdf/2611.12618)  

**Abstract**: We show that type-logical grammars with the standard rules for product, associativity, extraction and infixation have an NP-complete fixed recognition problem, adapting a construction for first-order linear logic from Pshenitsyn (2025), who already showed that grammars in first-order linear logic are closed under intersection and therefore not semilinear and not mildly context-sensitive. We strengthen these results and shown that grammars in first-order linear logic can simulate the quasi-linear grammars of Book \& Greibach (1970), the grammars recognised in non-deterministic linear time (NSPACE(n) or. NLIN). I argue that NLIN is an interesting class of formal languages as well, and a useful upper bound on the time complexity of formalisms used for natural language, just like LOGCFL is a useful upper bound on the space complexity. Taken together this means that first-order linear logic can be seen as the logic of non-deterministic linear-time string languages, graph languages and automata. 

---
# Evaluating Cross-Signal Coherence in Mammography Vision-Language Models 

**Authors**: Rebeca Nam, Chase M. Fensore, Christian A. Fauria-Robinson, Judy Gichoya, Joyce C. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2611.12574)  

**Abstract**: As vision-language models (VLMs) move toward interpreting medical images directly, their outputs increasingly include not just a label but a stated confidence and a written justification. Yet the field still predominantly evaluates VLMs using just accuracy. We evaluate the coherence between answer, confidence, and explanation across four VLMs on breast density and BI-RADS categorization, using two mammography datasets. We perform controlled perturbations of the question, multiple choice option orders, and the images and assess changes in the three outputs. Our results suggest confidence does not decrease even as the mammography image is changed. Shuffling the options also changes most VLMs' answers. Notably, there are cases where a model's own stated reasoning contradicts its final answer, a distinction accuracy alone cannot capture but one that governs whether an AI-generated answer can be trusted. 

---
# Can the Confidence and Decisions of System 1 Models Be Trusted? An Explainability and Perturbation Audit of Jev-Style Decision Models 

**Authors**: Julien Delaunay  

**Link**: [PDF](https://arxiv.org/pdf/2611.12521)  

**Abstract**: "System 1" decision models (pioneered by TypeSafe AI's Jev) replace autoregressive text generation with a single forward pass returning typed probability distributions over declared schemas (Choice, Score, Noul). While deployed for low-latency routing and automated triage, their statistical calibration and post-hoc explainability remain uncharacterized. We introduce an open, deterministic audit benchmark evaluating System 1 decision models across calibration under label-preserving perturbations, cross-question interference, and post-hoc attribution faithfulness. Evaluating five open-weight models (0.8B to 27B) alongside generative System 2 baselines -- both single-token logprob scoring and structured JSON/CoT decoding -- across NLI (e-SNLI), adversarial syntactic shortcuts (HANS), synthetic clinical triage (N=900), prompt injection defense (Tensor Trust, N=250), and an 86-case logic-gate trigger suite yields five findings: (1) provider confidence is an exact closed-form dispersion transform (1 - H(p)/ln K on sub-10B models; normalized margin on Clef-27B) rather than empirical accuracy; (2) bundling multiple questions in one request induces directional cross-question interference (11.7% to 22.4% binary flips on Tev1-0.8B, vs. 0.1% to 0.2% on Tev1-4B); (3) System 1 models with >= 4B parameters are faster and better calibrated under syntactic shifts (ECE_w = 0.030 to 0.087) than logprob (0.144) and JSON (0.378) baselines; (4) for prompt-injection defense (N=250), >= 4B models achieve 0.988 to 0.998 AUROC with up to 95.2% recall and 0.0% false alarms at 48 to 60 ms; and (5) single-pass inference makes 512-coalition XAI interactive (1.3 to 3.2 s/input), while KernelSHAP with a logit link (S1-SHAP) mitigates softmax saturation on disjunctive OR gates (89.3% to 100.0% gate recovery vs. 57.1% to 82.1% in prob space; p_adj = 0.004). 

---
# Unprompted coordination through shared memory in LLM agents: a locked two-implementation replication and extension 

**Authors**: Pekka Abrahamsson, Muhammad Waseem, Esa Karjalainen, Petrus Jussila, Henri Hansen, Jussi Rasku, Bilhanan Silverajan, Kai-Kristian Kemell  

**Link**: [PDF](https://arxiv.org/pdf/2611.12412)  

**Abstract**: Coordination between large language-model agents has become a central question for AI safety and software engineering, as such agents are deployed in groups that share infrastructure. Existing research tells the agents that partners exist or asks them to coordinate. Whether coordination arises through shared state alone, with nothing disclosing that anyone else is present, is the key question for deployments that share caches and memory between agents, yet it has rarely been examined, and never with independent replication; this study does both. Four agents each received a short arithmetic task and a key-value store presented to each as its own scratch space; two held complementary halves of one computation, and no prompt mentioned other agents, sharing or collaboration. For reliability, the environment was rebuilt independently from a written protocol, and both implementations ran under hashed locks on design, data and analysis. The results bear novelty value in three respects. Joint success occurred only through the shared store. Whether a model coordinated depended not on its vendor tier but on whether it wrote its intermediate result and looked at the store again, and one ordinary sentence, "check your workspace again before submitting", switched that second look on and off. The findings held under replication. For practice, state shared between agents is a communication channel and should be separated, namespaced and logged. For research, designs should be locked before the data, and the parse rule, endpoint and parameters reported. Future work should test open-weight models and separate using another agent's output from intending to communicate. 

---
# HumanSightBench: A Benchmark for Human Visual Perception in MLLMs 

**Authors**: Zixin Su, Yangguang Lin, Zegang Peng, Yuntong Sun, Xuanyi Wang, Jiahao Wang, Qiankai Xu, Bing Yuan, Jinze Bai, Youbin Wu, Jingjia Huang, Ruyang Liu, Yanxu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.12305)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated strong performance on knowledge-intensive and reasoning-based vision-language benchmark datasets, but still struggle to solve the visual questions that people can solve easily. Evaluating the differences between MLLMs and human-level visual perception is challenging due to end-to-end evaluation which combines various skills, or because we are uncertain if individual items are valid. Specifically, observed errors may reflect visual perception, language priors, background knowledge, question ambiguity, and underspecified answers. Our guideline for HumanSightBench is that we construct items that are human-easy by construction and model-hard by evaluation. HumanSightBench contains 2,876 visual questions with short answers across 7 top-level perceptual categories and 21 subcategories. We only include items that pass five inclusion gates: question clarity, answer uniqueness, visual grounding, human solvability, and text-only shortcut filtering. According to the protocol, errors are defined as incorrect answers on human-validated visual tasks with the evaluated interface. The inclusion gates are intended to reduce confounds from ambiguity and external knowledge. The best pass@3 score is 65.6% across eight evaluated MLLMs. No evaluated model exceeds 27.5% on Counting. These results identify capability gaps that are hidden by overall scores and can help future training experiments and held-out evaluation. 

---
# Agent4Crypto: A Dual-Reflective and Market-Feedback-Aligned LLM-Based Multi-Agent System for Cryptocurrency Trading 

**Authors**: Guangyan Gan, Jiahao Li, Jiashun Wang, Lin William Cong  

**Link**: [PDF](https://arxiv.org/pdf/2611.12269)  

**Abstract**: Cryptocurrency markets provide a natural testbed for autonomous AI agents because they are digital-native, high-frequency, and machine-speed environments, yet existing methods often struggle with non-stationary market regimes, opaque decisions, and fixed aggregation rules. In this paper, we propose Agent4Crypto, a dual-reflective and market-feedback-aligned multi-agent system. The framework coordinates specialized agents for data collection, analysis, risk assessment, reflection, and trading, synthesizing heterogeneous signals from prices, indicators, news, and charts in an interpretable decision pipeline. Our central design is a trader-centric decision mechanism: three style-specific trader agents, Aggressive, Neutral, and Conservative, generate candidate actions and natural-language rationales from the same market context, while their influence is learned rather than manually fixed. A PPO-based gating policy dynamically allocates trader weights through a dual-reflective feedback process. One reflection stream evaluates post-trade decision quality from realized portfolio outcomes and distills reusable lessons into a Financial ReasoningBank for retrieval in subsequent trader prompts, while the other estimates trader-level counterfactual contribution, indicating which style has recently aligned better with market movements. The PPO state combines Counterfactual Contribution Memory with market features, previous trader weights, and current trader actions, so trader influence adapts across bull, sideways, and bear regimes. Experiments on BTC, ETH, and XRP show that Agent4Crypto improves total return and Sharpe ratio over traditional strategies, machine learning and reinforcement learning baselines, and existing LLM-based trading systems, demonstrating the value of interpretable trader competition and market-feedback-driven adaptation in volatile cryptocurrency markets. 

---
# A Cost That Practice Accuracy Hides: Worked Solutions in Self-Generated Training Data and the Compositional Use of New Rules 

**Authors**: Jiantong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.12225)  

**Abstract**: A worked solution can make self-generated training targets shorter and cheaper, but accuracy on the practiced operation may hide how well a model can reuse newly learned rules. We study this choice in a 2B-parameter reasoning model learning two invented quantitative rules. The model generates targets from a rule sheet alone or from the sheet plus a program-written worked solution; targets are accepted for final-answer correctness, and the support is removed for fine-tuning and evaluation. Training questions request forward use in one segment; composition requires sequential reuse of the same rule across two segments. After six passes, using a program-checked numerical signature, sheet-only targets lead by 24.3 percentage points (95% item-bootstrap CI 14.6-34.0) on composition in a prospective test. On unseen argument values, observed forward accuracy is identical (87.5% each), but composition accuracy is 58.3% versus 23.6%, a 34.7-point gap (25.7-43.8). An independently sampled second target-bank pair repeats the direction (+14.6 points, 2.1-26.0). Ten passes on the shorter worked-solution targets, approximately matching the domain tokens of six sheet-only passes while changing other training quantities, narrow the gap to 16.3 points (6.3-26.7) in the first bank pair and 6.2 points (-6.3 to 18.8) in the second, where the pre-specified test does not hold. At six passes each, the worked-solution policy uses fewer capture tokens and less update time, yet the trained models use 2.6 times as many generated tokens per composition success under the numerical signature (2.1 times in the second bank pair). Choosing a capture context therefore requires testing how the new rules are reused, alongside the inference cost of those uses, rather than relying on forward accuracy alone. 

---
# A Geometric Invariant of Attention Predicts Head Specialization from Weights Alone 

**Authors**: Miguel P. Bento  

**Link**: [PDF](https://arxiv.org/pdf/2611.12206)  

**Abstract**: Transformers are built with continuous symmetries in the attention mechanism. We derive $\rho$, a scalar invariant that measures each head's capacity to couple to rotary positional encodings. This framework allows us to analyze heads without forward passes, opening a new research direction into data-independent analysis of modern LLMs. Empirically, we find that $\rho$ predicts induction heads, the most-studied circuit type in mechanistic interpretability, with Spearman correlation between $-0.55$ and $-0.74$ across four bias-free RoPE models spanning $135$M to $7$B parameters. As a preliminary proof of concept, we use $\rho$ to apply YaRN to only the top-50% positional heads, which degrades perplexity less than uniform application. Code is available at this https URL. 

---
# TS-GEAR: Taobao Shangou GEnerative Ad Retrieval 

**Authors**: Guofeng Tong, Xu Zhang, Guangliang Yu, Haiqiang Wang, Guannan Cong, Xing Guo, Yang Zhou, Jingjing Duan, Shuang Li, Feng Shi, Hualei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.12180)  

**Abstract**: On-demand delivery platforms such as DoorDash and Uber Eats impose a constraint absent in traditional e-commerce: merchants beyond the user's delivery range are invalid candidates rather than merely low-ranked. Generative retrieval has recently shown strong potential in large-scale retrieval systems. Adapting it to this setting, however, poses three challenges. First, existing semantic IDs encode geography only as a soft signal, so reachability can only rely on a non-differentiable post-generation hard truncation. Second, user preferences vary substantially across both regions and queries, so aggregating historical behaviors without conditioning on geography and the current query yields biased interest estimates. Third, the retrieved candidate set is evaluated by its collective value, yet autoregressive training scores each sequence independently, with no set-level signal to discourage redundant near-duplicates. To address the challenges, we propose TS-GEAR, a unified generative retrieval framework for on-demand delivery: (1) Geo-Semantic ID splits merchant identifiers into geographic prefixes and semantic suffixes, making reachability a prediction target rather than a post-hoc filter; (2) Query-Geo Target Attention aggregates historical behaviors using the real-time query and location as dual targets, while encoding each behavior's own location into its attention key; (3) GMPO (Group Marginal Policy Optimization) rewrites advantages using a deduplicated group reward and marginal advantage within GRPO, avoiding auxiliary diversity objectives. Deployed online with a lightweight 0.1B backbone, TS-GEAR serves over 150 million queries per day in the search advertising pipeline of Taobao Shangou, consistently outperforming state-of-the-art generative retrieval methods on both HitRate and geographic validity, and achieving 3.8% gains in ad impressions and 4.2% in ad revenue. 

---
# OGDA: Ontology-Grounded Data Agents for Reliable Analysis over Heterogeneous Data Lakes 

**Authors**: Boyi Zhang, Hong Ting Tsang, Yisen Gao, Qiao Xiao, LI Yufei, Haoyu Huang, Zhongwei Xie, Jiaxin Bai, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.12114)  

**Abstract**: Full-pipeline data analysis over real-world data lakes requires LLM agents to locate and integrate information across large collections of tables with different schemas. Existing data agents follow task-specific exploration, which exhausts the context window in massive lakes. As the context fills, the agent's reasoning degrades, leading it to locate and process incorrect data files. These errors propagate into the final answer. To overcome these limitations, we introduce Ontology-Grounded Data Agent (OGDA), which utilizes a compact ontology to reduce exploration overhead and constrain data processing. OGDA constructs the ontology through agentic hypothesis testing. At each step, a proposer agent induces schemas, concepts, and relations from raw sources, and a verifier checks each proposal against the underlying data. During task execution, the ontology serves as a representation of the data lake for the agent to explore. Every data-processing action is constrained by a task graph, which the agent can extend only after observing the semantic meaning of the data. Across KramaBench and DiscoveryBench with three LLM backbones, OGDA outperforms the strongest baselines by up to 10.82 points. These results suggest that explicitly modeling the data lake is a key step toward reliable end-to-end data agents. 

---
# Translating KV Caches Between Hybrid-Attention Language Models: MLA Latents, Linear-Attention States and Sparse-Attention Indexers 

**Authors**: Povilas Kvedaras, Tomas Rasymas  

**Link**: [PDF](https://arxiv.org/pdf/2611.12062)  

**Abstract**: A model that takes over another model's conversation must normally re-read the whole history; a text hand-off shortens the re-read only by dropping content. KV-cache translation instead maps the source's cache into the target's own format, so the target continues from the full context without re-reading it. It works for grouped-query attention (GQA), but open-weights frontier models now cache compressed latents (MLA), sparse-attention indexer keys and fixed-size linear-attention states. We replicate KV-Lingo on a GQA pair and then give, for each of these objects, where to capture and inject it and a mechanism that reaches native quality. We study two hybrid pairs at 16-48B scale in both directions, Kimi-VL-A3B - Kimi-Linear-48B (MLA - MLA + Kimi Delta Attention) and Qwen3.5-27B - Qwen3.8-27B (Gated DeltaNet hybrids), plus a sparse-attention target (Qwen3.8-27B - Qwen3.8-Flash-Next). MLA latents translate with token-wise linear maps; recurrent states are copied with a closed-form correction or synthesised by replaying the target's own recurrence. Exact long-range recall needs token identity: decoding token ids from the source's layer-0 MLA latent raises needle recall from 0.09 to 0.99 at 2-32K and to 0.83-1.00 at 64K, eight times the longest training prompt. The hybrid full-MLA direction is within the paired confidence interval of native quality on every task metric, and same-shape GDN hybrids reach the native level in both directions (apart from 6% refusals inherited from the source), with time to first token 18.7 and 46.8 times shorter than a re-read at 64K tokens. Segment-wise multi-turn transfer stays at the re-prefill level; naive re-translation collapses. On the sparse-attention target, predicted indexer keys recall the needle as well as the target's own keys (one seed). We state as a hypothesis how the recipes transfer to 300B+ hybrid flagships. 

---
# Speculative Self-Distillation enables Efficient Knowledge Internalization 

**Authors**: Shayan Talaei, Agam Bhatia, Arshia Soltani Moakhar, Jonas Hübotter, Amin Saberi, Azalia Mirhoseini  

**Link**: [PDF](https://arxiv.org/pdf/2611.12016)  

**Abstract**: Many real-world applications require language models to internalize knowledge that was absent from pretraining, to incorporate continually updated facts, or to learn new domain skills. Self-distillation has emerged as a natural approach for this task. A model conditioned on the source material acts as the teacher, and an unconditioned student is trained to reproduce the teacher's behavior closed-book. We propose Speculative Self-Distillation (SSD), an efficient per-token mixed-policy method for self-distillation. The student generates each rollout by default, while the teacher intervenes only at positions where its next-token distribution substantially diverges from the student's. Unlike off-policy distillation with teacher-generated prefixes, this design keeps training rollouts close to the student's test-time behavior, while avoiding the low-signal states that fully on-policy rollouts drift into. In effect, SSD lets the teacher steer the rollout exactly where the privileged context changes the prediction. On three closed-book QA benchmarks measuring fact recall, knowledge updates, and skill acquisition, SSD nearly matches on-policy distillation while using 38-69% fewer supervised tokens and consistently outperforms off-policy distillation on the held-out test set. On a forgetting stress test, SSD retains the most general capability while also achieving the best in-domain accuracy and using less supervision than the on-policy baseline. Overall, SSD establishes a new accuracy-efficiency-forgetting Pareto frontier for self-distillation. 

---
# Infrathon: Benchmarking Language Agents for Real-World, Long-Horizon AI Infra Optimization 

**Authors**: Zhengyu Hu, Zheyuan Xiao, Yike Wang, Hamish Ivison, Haoyu Wang, Yige Yuan, Rulin Shao, Yuetai Li, Niqi Lyu, Yicheng Ding, Shu Yang, Zhichen Zeng, Banghua Zhu, Jiaxin Pei, Kaize Ding, Pradeep Dasigi, Dan Roth, Hannaneh Hajishirzi, Noah A. Smith, Teng Xiao, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2611.11995)  

**Abstract**: In this work, we study whether frontier agent can continually improve real-world AI infrastructure over long horizons. We introduce Infrathon, a benchmark of diverse, long-horizon AI infrastructure optimization tasks across different levels. We evaluate six models across 150 runs with 24-hour budgets, totaling over 10,000 GPU-hours. Our results show that model rankings vary with the evaluation horizon, while 135 of the 150 runs achieve further improvements during the second half of their budgets. These findings indicate that shorter evaluations can overlook both optimization gains and changes in relative model performance. We also find agents can discover solutions that outperform human-designed baselines: Claude Opus 5.5 increases the mean KV-cache compression ratio from 6.82% with the human-designed Quant-VideoGen codec to 19.86%. Analysis of agent trajectories identifies task violations, unreliable testing, and premature completion as barriers to sustained improvement. Together, Our Infrathon framework provide resources for constructing long-horizon tasks, evaluating frontier agents, and discovering good infrastructure solutions. 

---
# CHO-Bench: How Do Context Hints Affect Ad Rankings in LLM Advertising? 

**Authors**: Shiqi Zhang, Yang Fei, Xiaokui Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.11955)  

**Abstract**: LLM advertising selects advertisements based on user conversations, where user needs may extend beyond explicit product searches. ChatGPT Ads allows advertisers to provide natural-language context hints describing what they offer, who may find it useful, and when it may be useful, as additional information for ranking advertisements across diverse user conversations. However, how to effectively construct these hints remains unclear. We formulate Context Hint Optimization (CHO) as the problem of constructing context hints to improve an advertisement's ranking under a given budget. Drawing on established marketing frameworks, we develop seven context hint generation strategies covering What, Who, and When, along with a Unified strategy that combines all seven dimensions. We introduce CHO-Bench and conduct over 1.19 million paired evaluations across five datasets using embedding-based and LLM-based rankers. Our results show that context hints improve advertisement rankings under the evaluated offline rankers. While individual strategies achieve similar average gains and Unified offers no consistent advantage, strategy differences become more pronounced under specific generation-ranking configurations, with embedding-based and LLM-based rankers favoring different information dimensions. Increasing the hint budget further improves rankings with diminishing marginal gains. These findings establish an offline benchmark for studying ad-side ranking optimization in LLM advertising and highlight opportunities for model-specific hint selection and budget-aware generation and aggregation. 

---
# Language processing in the autistic brain: A meta-analysis of neuroimaging studies across modalities and age 

**Authors**: Hanlin Wu, Bei Xiao, Qingxi He  

**Link**: [PDF](https://arxiv.org/pdf/2611.11927)  

**Abstract**: Language is central to social communication, which is altered in autism, and language abilities vary widely among autistic individuals. How the autistic brain processes language is central to understanding autistic communication. Yet neuroimaging studies report inconsistent findings on how autistic brain activity is altered during language processing. We synthesized 45 fMRI and PET studies (753 autistic and 777 non-autistic participants) with seed-based d mapping with permutation of subject images (SDM-PSI). We estimated autistic and non-autistic activation and their difference with meta-regressions adjusted for each study's language modality and mean age. Autistic and non-autistic brains activated a similar left-lateralized language network. Autistic brains showed reduced activity throughout the left perisylvian cortex, from the inferior frontal gyrus to the mid superior temporal gyrus and the posterior superior temporal sulcus, and increased activity in the right temporal pole. Spoken language showed a larger autistic reduction than written language in the left mid superior temporal gyrus. Adults, but not children and adolescents, showed increased autistic activity in the right superior temporal gyrus and cuneus. Autistic language processing thus relies on the typical left-lateralized network but shows reduced leftward lateralization. 

---
# Spectral Unlearning: Transformer Structure-Preserving Updates for Language Models 

**Authors**: Sung Il Choi, Junhao Cai, Dohun Kim, Changhee Joo  

**Link**: [PDF](https://arxiv.org/pdf/2611.11902)  

**Abstract**: Machine unlearning in large language models (LLMs) aims to remove specific learned knowledge, capabilities, or behaviors while preserving model utility. Existing methods have mostly focused on the design of unlearning objectives, while largely overlooking the intrinsic matrix structure of transformer weights. In particular, standard optimizers operate on independent scalar coordinates, thereby ignoring this matrix structure. We propose spectral unlearning, an update framework that combines standard unlearning objectives with matrix-structured optimization. We show that the spectral update corresponds to the steepest-descent direction for transformers composed of linear layers with normalized inputs and outputs. Our approach is applied on top of existing unlearning methods and evaluated across complementary axes of memorization, privacy, and utility, summarized via their harmonic mean. Spectral unlearning consistently outperforms a broad range of baselines and closes most of the gap to a retain-trained reference. Code is available at this https URL. 

---
# When Answer Decodability Is Not a Decision: Separating Final-Answer and Imminent-Emission Signals in Chain-of-Thought 

**Authors**: Hyojung Han, Jongmin Kim, Seung-Hun Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2611.11892)  

**Abstract**: Answer probes applied to chain-of-thought hidden states are often read as evidence that a language model has already decided its final response. This reading overlooks a competing signal, the candidate the model is about to mention. We introduce emission-controlled dual-target probing (ECDP), which fits the same low-rank reader family separately to the imminent mention and to the final answer at identical positions, with target-specific nulls and thresholds fixed before any fit. On correct DecisionDepth (DD) rollouts in which a distractor is mentioned before the final answer, Qwen3.5-4B decodes the next mention at .966 (null .234) but the final answer only at .221 (null .186), an emission gap of .696 [.648, .744]; Llama-3.1-8B gives .575 [.514, .632]. This pattern is not universal. In a pre-specified evaluation of Qwen3-8B with thinking enabled on held-out MMLU questions, using a full-prefix attention reader over every thinking position, both targets are decodable and the final answer is read far more strongly (mention-minus-answer lift -.398 [-.459, -.336] at layer 22 over 467 questions). Model, task, positions, and reader change together between these settings, and an attempted matched comparison on DD produced no valid statistics because the imminent-mention probe did not train reliably, so we do not identify what reverses the order. On DD, planted-signal controls show that the reader recovers an injected final-answer signal at one tenth of full dose. Answer-candidate decodability is therefore target-relative and protocol-dependent: a strong answer readout does not by itself establish when a model commits, and a probe that fails to train is not evidence that the information is absent. ECDP is a practical control for distinguishing imminent text emission from evidence of a settled decision. 

---
# Competing Influence in LLM Populations: Budget Timing and Baseline Drift 

**Authors**: Mehwish Nasim, Nicolas Fay  

**Link**: [PDF](https://arxiv.org/pdf/2611.11867)  

**Abstract**: When competing campaigns target a population of large language model (LLM) agents, observed opinion changes may reflect both campaign influence and changes that would occur without intervention. We study how the timing of campaign expenditure and the population's own evolution shape outcomes in a network of persona-conditioned LLM agents that exchange and revise opinions in natural language. A proponent and a challenger are tasked with generating opposing messages under equal delivery budgets, while a separate stance probe measures opinions without influencing the simulation. Across eight claims, we compare single-campaign and competitive conditions against matched no-campaign runs, and evaluate all nine combinations of early, sustained, and late spending on a reference topic. Baseline stance drift varies substantially across claims and, on some, accounts for a sizeable fraction of apparent campaign effects. After accounting for this drift, the direction of observed campaign-associated changes aligns more closely with the claim's epistemic standing than with the campaigner's assigned role. Uninfluenced populations also consistently lose stance diversity, demonstrating that directional opinion change and convergence are distinct phenomena. In networks with two opinion-segregated communities, campaigns primarily reduce differences between communities rather than diversity within them. LLM campaign agents do not always argue for their assigned positions, particularly when these conflict with established evidence. Finally, campaign outcomes depend on the opponent's expenditure schedule: unlike a matched numerical bounded-confidence population receiving interventions at the same times and locations, the LLM population exhibits a payoff structure with no pure-strategy equilibrium. 

---
# A Benchmark and Layered Evaluation Protocol for Evidence-Grounded Shared Belief Updating in Connected Robotics 

**Authors**: Ranulfo Bezerra, Kimitoshi Yamazaki  

**Link**: [PDF](https://arxiv.org/pdf/2611.11842)  

**Abstract**: Connected physical systems fuse observations from robots, IoT sensors, and edge-AI modules into a shared account of the world, yet most evaluations score only the final answer or action and cannot tell whether a method attributed the evidence, updated its belief, handled contradictions, and justified its decision. We argue that shared reasoning in Connected Robotics should be evaluated as an evidence-to-belief trajectory, and introduce the Connected Robotics Evidence Graph (CREG) Benchmark to make that trajectory measurable. CREG comprises 350 deterministically generated scenarios across seven regimes (justified revision, belief preservation, relation association, contradiction onset and resolution, stale/delayed evidence, and source reliability), each exposing a normative posterior reference at every evidence prefix, validated by an independent log-odds checker and screened by a blinded expert audit. Nine representation conditions separate relation content from serialization, ordering, and metadata. Evaluating four open-weight instruction models and six transparent fusion baselines yields three findings. First, entity grounding accounts for a substantial part of the weak-model deficit. Second, deleting semantic relations costs 15.3 points of strict belief-state accuracy on relation-dependent scenarios (Holm-corrected p<0.001), but an information-matched structured list outperforms the full graph and shuffling the graph does not hurt: relation content matters where required, while graph serialization confers no consistent advantage. Third, belief accuracy is strongly non-monotonic along the evidence sequence, with mid-trajectory failures that endpoint-only evaluation cannot detect. All scenarios, references, prompts, metrics, and analysis code are released for offline, hardware-free reproduction. 

---
# Creative Priming: Decoupling Process from Product to Escape the Creative Paradox 

**Authors**: Alon Albalak, Sebastian Gabriel, David B. D'Ambrosio, Joel Lehman, Kenneth O. Stanley  

**Link**: [PDF](https://arxiv.org/pdf/2611.11740)  

**Abstract**: Many of the most consequential advances promised by AI, from curing diseases and inventing new technologies to formulating new scientific theories, are fundamentally creative acts. Yet creativity is uniquely difficult to learn because it faces a creative paradox: learning to repeat past examples of creative success makes those specific examples more likely to recur, inhibiting the very creativity desired. This work argues that creativity is better modeled as an explicit process rather than as a property of individual artifacts, and proposes to learn the underlying generative principles behind the creative process. In particular, to escape the paradox, we introduce creative priming (CP), wherein a trained priming model generates a prompt-specific primer that assists a response model in being creative, explicitly decoupling the creative process from response generation. The priming model learns through GRPO, rewarding primers that lead to responses that are novel, high-quality, and diverse, thereby reinforcing the process that yields creative responses without reinforcing any specific response. 

---
# The Value of Not Knowing: How Fresh Eyes through Naive LLMs Fuel Creativity 

**Authors**: David B. D'Ambrosio, Alon Albalak, Joel Lehman, Kenneth O. Stanley  

**Link**: [PDF](https://arxiv.org/pdf/2611.11725)  

**Abstract**: Many major scientific advances come from approaching a problem with a new or novel perspective. While large language models (LLMs) with internet-scale knowledge seem poised to take such perspectives, their consistent generation remains elusive. Instead, models are drawn to dominant views and preexisting, in-field bias. To explore the potential of fresh eyes to increase creativity, this paper proposes naive LLMs, models that are made intentionally ignorant of a specific topic. Through deprivation in training or intentional forgetting, naive models are unburdened by existing dogma, allowing them to be free to make new associations and conjectures. These naive models can then be paired with a stronger mentor model that can refine their unique perspectives into concrete proposals. For the 1B and 7B models tested in this paper, naive models consistently produce seed ideas that are more novel and useful than baseline counterparts, and mentor refinement improves the ideas even further. Thus, contrary to the prevailing tradition of scaling up training data, we demonstrate a case where less information is actually more powerful, in particular when the aim is to discover genuinely outside-the-box solutions. 

---
# ODSA: Orchestrated Discovery and Self-Aggregation 

**Authors**: Macardle Meng, Haomin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.11621)  

**Abstract**: Test-time scaling improves the accuracy of a language model's answers by spending more computation at inference time. Established methods do so either sequentially, revising a single solution over several passes, or in parallel, generating several independent solutions and selecting among them. We present ODSA (Orchestrated Discovery and Self-Aggregation), an LLM-directed scaling framework in which an orchestrator LLM is given existing test-time scaling methods as actions --- discovery of a new independent solution, comparison of existing solutions to build a better one, and acceptance of a proposed answer, which triggers its verification --- and directs stateless workers to carry them out. The orchestrator does not solve the problem itself: it allocates computation, feeds the results of earlier actions into later ones, and takes every action from a single context, so that each decision is informed by everything tried before. On Humanity's Last Exam, ODSA improves accuracy by 4.7 to 18.0 points across three models over a single direct call to the model with no test-time scaling. On LiveCodeBench, where a single direct call already solves most problems, it improves accuracy by 0.6 to 14.0 points. 

---
# LAION-Audio-XL: A Web-Scale Corpus and Scaling Studies for Contrastive Language-Audio Pretraining 

**Authors**: Gijs Wijngaard, Felix Förster, Ross Wightman, Marianna Nezhurina, Christoph Schuhmann, Jenia Jitsev, A. Sophia Koepke  

**Link**: [PDF](https://arxiv.org/pdf/2611.11592)  

**Abstract**: Recent progress on contrastive audio-language models can be attributed mainly to the scale and quality of their training data. Larger training datasets expose models to a broader range of sounds and corresponding audio captions. To exploit this, we collect 1.48 billion audio samples from academic datasets, internet datasets, web collections, Common Crawl, and YouTube, and process them with a multi-stage curation pipeline. Furthermore, we generate short and long captions for samples in our corpus that lack them. We use our dataset to study the scaling behavior of contrastive audio-language models based on the NaFlex architecture for the audio encoder. Our experiments identify trade-offs between model size and training data for a given compute budget. Moreover, our resulting models, CLAP2, outperform prior audio-text models of comparable size on the MAEB and HEAR benchmarks. We release our dataset, captions, model checkpoints, and code. 

---
# FinCutBench: A Continually Updated Benchmark for Measuring the Temporal Boundaries of LLM Financial Knowledge 

**Authors**: Siyuan Li, Ke Wan, Boyong Hou, Yujie Deng, Zixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2611.11588)  

**Abstract**: In financial backtesting and other point-in-time tasks, external inputs can be restricted by date, but model parameters may retain financial facts disclosed later, introducing parametric look-ahead bias. We introduce FinCutBench, a continually updated monthly financial knowledge benchmark for measuring how far a fixed model's financial knowledge extends over calendar time. The current dataset contains 9,000 Chinese and English four-option multiple-choice questions from January 2023 to September 2026, covering Chinese and U.S. markets and four categories of financial information. We construct monthly knowledge profiles using the approximate months in which the tested facts became public. After a profile passes the decline test and knot checks, we define the FinCut Boundary as the first month when the fitted profile reaches its lower level, and report the stability of the estimated date under resampling. Boundary dates and localization precision vary across models. Using the independently measured Transition Start and Boundary as fixed references for downstream sample grouping, we find clear relative declines after the Boundary in U.S. asset ranking and earnings recall for some models, while Chinese-market ranking estimates vary across models and all eight disclosure-period-paired forecast-error comparisons have unadjusted $p$ values above 0.10. The FinCut Boundary thus provides an empirical reference for the temporal scope of parametric knowledge in point-in-time financial evaluation, supporting evaluation-window selection and result interpretation. 

---
# Safeguarding Agentic AI in Radiology Workflows: A Multimodel Evaluation of Rule-based and Large Language Model-based Verification 

**Authors**: Semil Eminovic, Robin Schmidt, Bogdan Levita, Maximilian Lindholz, Tobias Penzkofer, Jawed Nawabi  

**Link**: [PDF](https://arxiv.org/pdf/2611.11541)  

**Abstract**: Purpose: To estimate how often tool-using large language model (LLM) agents record clinically relevant false or unsupported information in simulated radiology documentation and how much safeguards applied before each action reduce these errors.
Materials and Methods: In this simulation study (October 2026), 48 synthetic, radiologist-reviewed base scenarios covered finding communication, tumor board preparation and follow-up management. Forty-three also had a misleading-information variant (91 variants). Six LLM agents (Claude Sonnet 5.5, GPT-6.1-Sol, GLM-5.3, MiniMax M3, Gemma-4-31B, Qwen3.8-27B) completed each variant with permissions only, deterministic rules, and rules plus a locally run LLM checker (1638 episodes). The primary outcome was at least one executed high-severity violation per episode. Risk differences were estimated per agent at base-scenario level and, exploratorily, pooled across agents, with analyses of document access and blocks.
Results: With permissions only, 7.3%-27.1% of episodes had a high-severity violation (pooled across agents, 19.4%). Pooled, it was 16.7% with rules (-2.8 percentage points; 95% CI: -6.1, 0.5) and 13.2% with rules plus the LLM checker (-6.2; 95% CI: -9.5, -3.0). Adding the LLM checker to rules changed it by -3.5 (95% CI: -6.3, -0.7). No difference between agents was detected (P = .43), and task success did not decrease detectably. In misleading variants, agents opened the altered document in 83% of episodes regardless of error, and 58% of erroneous statements cited it. Most blocked actions (64%-68%) would not have caused a violation and were returned for revision.
Conclusion: Layered safeguards reduced but did not prevent high-severity errors. Errors after misleading information reflected misjudged rather than missed information. 

---
# LLM-based Multi-Criteria Deep Reinforcement Learning for Dynamic Efficient 5G UPF Selection 

**Authors**: Abdelrahman Soliman, Ahmed Refaey Hussein, Lei Lei  

**Link**: [PDF](https://arxiv.org/pdf/2611.11372)  

**Abstract**: User plane function (UPF) selection in 5G networks is an active area of research, with various limited approaches proposed to enhance network efficiency and manage resources effectively. In this paper, we address this gap in the related literature by proposing a Deep Reinforcement Learning (DRL) UPF selection method that utilizes a dynamic, multi-objective reward function and validating it on a real-world 5G emulation platform. The proposed method is based on the large language models' (LLMs) translation of network operator intents and the network data analytics function (NWDAF) abnormality alerts. The experimental results validated in the real-world experimental setup demonstrate that our dynamic DRL solution achieves an effective balance between latency, power consumption, and traffic distribution, outperforming the heuristic baselines and fixed DRL by 16.89%. These findings highlight the real-world applicability of our approach for dynamic, intent-driven UPF selection. 

---
# Agent Self-Improvement with Frozen Weights: The Added Value and Limits of Failure Traces 

**Authors**: Arkid Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2611.11342)  

**Abstract**: We evaluate whether an AI agent's harness, the notes around its model, can learn from the agent's failure traces beyond a matched search that never sees them. It can, but less than it looks. We tested an open 9-billion-parameter model with fixed weights; a larger teacher model wrote its notes. On generated customer-service decisions, where both note writers were told the target rules, notes based on the agent's mistakes improved accuracy by 4.0 percentage points over notes from the same process with the mistakes withheld. That process alone, trying and keeping better-scoring notes, reached about 60% of the total improvement. In full customer-service conversations the same notes showed no demonstrated improvement. Separately, a strict screen of single lessons admitted two customer-service lessons, still awaiting tests on new data, and one tool-use lesson, written without seeing the agent's mistakes, which showed no demonstrated gain on new tool-use tasks. A neutral note with no procedure scored higher than the learned lessons (exploratory). A blind audit judged 146 of 240 tool-use test items defective. 

---
# ReGKG-2025: A GKG-Compatible Reconstruction of the GDELT Global Knowledge Graph for the Missing Period of 14 June to 2 July 2025 

**Authors**: Nikita A. Klimin  

**Link**: [PDF](https://arxiv.org/pdf/2611.11316)  

**Abstract**: The June 2025 outage in the Global Database of Events, Language, and Tone (GDELT) collection pipeline left a gap in the Global Knowledge Graph (GKG), interrupting article-level news metadata used in downstream research. ReGKG-2025 is an archive-derived sample of 281,620 reconstructed article records covering all 1,665 missing 15-minute batches. Articles were recovered from Common Crawl and the Wayback Machine using source quotas estimated from surrounding native GKG data. Text was screened and normalized, non-English bodies were translated, and flagged records were manually reviewed. Eligible articles were converted to the 27-column GKG 2.1 layout. The versioned package includes record-level provenance, archive replay and WARC locators, source summaries, source-by-day calibration weights, and the complete manual decision journal. The sample reaches about 14% of modeled native volume and represents sources accounting for 84% of selected reference record mass. In an intact-week test, 41% of recovered articles matched native GKG records; field agreement on matched pairs is close to the processor baseline (tone MAE 1.69). The dataset supports retrospective content analysis and source-level aggregates; its tone and theme distributions differ from native GKG, and it does not support exhaustive counts. Data: this https URL. 

---
# Task-adaptive anonymisation enables privacy-preserving sharing of clinical speech data 

**Authors**: Ziyun Cui, Haoyu Niu, Wen Wu, Hao Sun, Sihan Zhang, Chang Lei, Yun Zhang, Chunyun Sun, Yang Luo, Liangliang Cao, Ji Wu, Runsen Chen, Lian Zhou, Yongsheng Tong, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.11273)  

**Abstract**: Clinical speech is increasingly used for disease screening and clinical assessment, but directly distributing speech recordings poses privacy risks because they contain both speaker-identifying voiceprints and sensitive spoken content. Existing speech anonymisation approaches are primarily developed for general speech applications or task-specific clinical speech and overlook that different clinical tasks rely on distinct speech characteristics, necessitating task-adaptive anonymisation strategies to protect patient privacy while preserving clinical utility. Here we introduce CASA, a clinically adaptive speech anonymisation agent that transforms anonymisation from a fixed processing procedure into a task-guided optimisation process, dynamically tailoring voiceprint and content anonymisation to task-specific privacy and clinical utility requirements. Through recursive self-improvement (RSI), CASA iteratively evaluates and refines anonymisation strategies against task-specific privacy and clinical utility requirements, while learning from these outcomes to guide more efficient planning in subsequent clinical tasks. CASA is validated across 14 clinical tasks encompassing five languages, diverse elicitation paradigms, and mental health, cognitive and voice disorders. CASA preserved disease-related classification performance while meeting speaker-anonymisation requirements across all clinical tasks, versus only 8/14 for the strongest fixed method. Assessments of resistance to speaker re-identification by external attackers and identification and redaction of sensitive personal information in clinical speech further demonstrated CASA's dual protection of speaker identity and sensitive spoken content. Clinical expert assessments further supported the preservation of clinically relevant information and the acceptability of anonymised speech for controlled research sharing. 

---
# TLRS-DPO: Token-Level Risk Scoring for Better Direct Preference Optimization 

**Authors**: Suxin Ji, Hungtao Wan  

**Link**: [PDF](https://arxiv.org/pdf/2611.11232)  

**Abstract**: Direct Preference Optimization (DPO) is a simple and widely used alternative to RLHF for aligning large language models (LLMs), but its sequence-level objective cannot precisely locate or suppress the local patterns that make an otherwise acceptable response unsafe or undesirable. We propose TLRS-DPO (Token-Level Risk Scoring for Better Direct Preference Optimization), which augments DPO with fine-grained token-level control while remaining fully offline and reference-based. TLRS-DPO introduces a token-level robust compensation term that converts the policy-reference log-ratio on rejected responses into localized risk scores, a curriculum-style schedule that gradually increases the strength of this compensation during training, and a risk-adaptive screening mechanism that selectively penalizes only high-risk tokens instead of entire sequences. This design bridges sequence-level preference fitting and token-level behavior control, and can be implemented as a drop-in modification to standard DPO trainers. Empirically, TLRS-DPO improves alignment quality and safety-relevant behavior with minimal impact on helpfulness compared to strong DPO baselines. 

---
# SGA-Sparse: Training Native SwiGLU Gates for Activation-Sparse and Low-Bit LLMs 

**Authors**: Ruokai Yin, Steven Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.11223)  

**Abstract**: Activation sparsity can reduce FFN weight traffic during single-user LLM decoding, but active channels must be identified early enough to avoid fetching unused weights. In pretrained SwiGLU models, training-free routing with the native gate performs substantially worse than routing with the up activation at high sparsity. We show that this ordering reverses after sparsity-aware training. Based on this insight, we introduce SGA-Sparse, which converts pretrained SwiGLU LLMs into fixed-budget activation-sparse models by training the existing gate projection directly as a top-k router. SGA-Sparse updates all model parameters end-to-end while preserving the original SwiGLU activation and introducing no auxiliary router parameters. After approximately 30B training tokens, LLaMA models from 1B to 8B achieve uniform per-token 75-85% FFN channel sparsity while remaining within 0.2-2.7 pp of their original dense checkpoints in average zero-shot accuracy. Because post-training quantization degrades sparsity-trained checkpoints, we further study two complementary quantization-aware training (QAT) recipes: continued-QAT largely recovers performance at 4-bit weight quantization, while joint-QAT performs better for the more aggressive sub-4-bit target. To compare activation sparsity with dense low-bit quantization, we translate the sparsity-induced reduction in weight traffic into an equivalent bit-width. Under this unified weight-traffic metric, our 2.5- and 1.5-bit-equivalent models establish a new accuracy-traffic Pareto frontier: the 2.5-bit-equivalent models match dense 3-bit QAT accuracy, while the 1.5-bit-equivalent models approach dense 2-bit models and outperform dense ternary models by 1.3 pp on average. These results establish native-gate training as a practical approach for combining high FFN sparsity with low-bit weights. 

---
# Deterministic Narrative Gravity: Bridging Theory of Mind and Historical Simulation through ABM-LLM Architectures 

**Authors**: Alicia Colmenero-Fernández, Lidia Ortega Alvarado  

**Link**: [PDF](https://arxiv.org/pdf/2611.11212)  

**Abstract**: Historical social simulation faces a fundamental reproducibility challenge: the stochastic nature of large language models (LLMs) introduces a digital butterfly effect that prevents exact replication across multi-turn runs. Furthermore, the inability to control low-level server parameters in commercial LLM hosting environments severely limits reproducibility. To overcome this opacity, we present a deterministic multi-agent framework coupling LLMs to an Agent-Based Modelling (ABM) pipeline grounded in Theory of Mind (ToM) and Cognitive Archaeology, and validate exact artifact identity within local, pinned execution environments using SHA-256 hashing. Run locally at temperature zero with clean per-call model reloads, the system injects differentiated cognitive states per turn through dissonance-driven behavioural modes and gated internal voices. Evaluating seven model configurations spanning 4B to 31B parameters against an ablated baseline reveals clear operational boundaries. Small models ($\leq$8B) suffer from context capacity overload, producing repetitive, specular dialogue loops. However, at $\geq$12B parameters, the ABM scaffolding breaks the monotonous response patterns seen in ablation. Crucially, these scaffolded models exhibit spontaneous diegetic expansion: they coherently expand scenario lore through tactically motivated, anachronism-free details rather than escalating into aggression. By resisting the structural pull toward premature scene closure, scaffolded sessions sustain an open terminal Narrative Vector ($V_N = \mathrm{Open}$ in 12/21 overall; 9/9 in official $\geq$12B configurations) whereas unscaffolded ablation never does (0/21). This framework provides a transparent, auditable benchmark architecture for virtual archaeology, with the full pipeline released as open-source software to enable independent replication. 

---
# Dhara: Evidence-Oriented Retrieval for Indian Knowledge Systems 

**Authors**: Avinash Holenarasipur Nagendra, Anand Kaniyaar Sriranga Prasad, Rahul Vishwakarma, Trupti Gavit, Anish Amol Tipnis, Yashaswini Mandayam Rangayyan, Vinod Appajanna, Kousik Sarathy Sridharan, Mohan Raghavan  

**Link**: [PDF](https://arxiv.org/pdf/2611.11197)  

**Abstract**: Relevant evidence in Indian Knowledge Systems (IKS) collections may be distributed across passages in different parts of a work, including verses, translations, and commentaries. Locating it requires matching natural-language questions and partially remembered verse fragments to source passages. We present Dhara, a retrieval engine combining hybrid search, chapter- and book-level Entity Notes (EN), Fuzzy Verse Retrieval (FVR), and neural reranking. Five comparisons on the 325 questions in Dhara Bench examine retrieval representations against BM25, EN's contribution, reranker quality and computation time, FVR, and Gemini. BGE-M3 single-vector retrieval exceeds BM25 in nDCG@10, Recall@20, and MRR@20 across all categories. Its scores are broadly comparable to ColBERTv2, with mean retrieval times of 0.108 s and 0.962 s under the same hardware configuration and a smaller memory footprint for chunk storage on disk. Adding EN to a direct branch with a matched candidate count improves nDCG@10 in all twelve categories with Qwen and ten with BGE; the combined ranking is capped at the same candidate count. Qwen improves nDCG@10, Precision@5, and MRR@20 over BGE in every category, at approximately 6.72 times the overall estimated serial computation time. FVR increases Verse-category nDCG@10 by approximately 20 percentage points with either reranker. Gemini is evaluated after source mapping and manual review with relevance annotations visible. The Qwen pipeline with EN and FVR has higher Recall@20 than Gemini in ten categories and stronger complete-evidence retrieval in Multi-hop questions, while Gemini retains early-relevance advantages in several categories. The results inform retrieval choices according to evidence requirements and resource budgets. 

---
# When Are Nearby Experts Composable? Cross-Task Model Merging in Neural Thickets 

**Authors**: Suxin Ji, Hungtao Wan, Mingjun Liu, An Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.11177)  

**Abstract**: Random perturbations around a shared checkpoint can produce experts for distinct tasks, but shared origin does not ensure that their gains survive composition. We formulate expert-pair selection for joint base-relative retention, measured by strict joint success and worst-task gain with unconditional abstention accounting. An exact finite-difference decomposition and an arbitrarily small orthogonal counterexample show why proximity and orthogonality alone are insufficient. We instantiate this formulation with quality-constrained spectral selection, a fixed-budget framework combining cross-task screening, layerwise pair geometry, separate validation, and Base fallback. Across six checkpoints, 21 task pairs, and five independent perturbation pools, the procedure improves strict joint success over Top-1 averaging throughout and leads the four-method model-scale comparison on four of six checkpoints. For the five models with at least 1.5B parameters, matched quality-only, global-cosine, and spectral ranking attain 49.1%, 50.5%, and 53.1% joint success. Under this matched protocol, the spectral selector also yields the highest worst-task gain and Weak Pareto rate and the lowest regression rate. Mean worst-task gain reaches +0.61 percentage points, while 17.1% regression and 13.0% abstention show that composability remains conditional. Project page: this https URL. 

---
# What Changed in Engineering Writing After Generative AI Chat Assistants: Evidence from 1.8 Million Articles 

**Authors**: Taner Cokyasar  

**Link**: [PDF](https://arxiv.org/pdf/2611.11174)  

**Abstract**: This study asks what changed in engineering articles after conversational language models became publicly available on November 30, 2022, and develops methods for measuring such change in a field without a uniform abstract index. It assembled 1,839,962 articles published from 2015 to 2025 in 256 engineering journals, tracked twelve measures of wording, style, and article structure, and compared each with the projection of its own pre-2023 trend. Building the monthly series showed that records dated only by year fall on January 1 and create false January peaks unless removed. In abstracts, 31 words associated with language model output appeared 82% more often than projected (55.4 against 27.0 per 10,000 words in 2025), while four control words of the same formal register fell 16%. Words that soften a claim, such as may, fell 13%, words that strengthen one, such as clearly, rose 20%, and long words rose 11%. Author counts, reference lists, page counts, and publication volume stayed within 4% of trend. The change appears in all fourteen subject categories, and false release dates, dropping any one journal or word, and word lists chosen from the data support it. A random sample of 12,866 full texts from the same corpus, including subscription articles and weighted back to it, shows the same words rising 99% in article bodies. Statements disclosing AI writing assistance, measured apart from mentions of language models, went from none to 3.7% in 2025, four in five of them at one publisher. The design is descriptive, not causal. 

---
# Do Benign-Trained Hidden-State Probes Detect Tool Errors Under Adversarial Conditions? 

**Authors**: Bibek Bhandari, Kshitij Lingthep  

**Link**: [PDF](https://arxiv.org/pdf/2611.11172)  

**Abstract**: Although hidden-state probes efficiently monitor model operation, it remains unclear whether probes trained on benign tool use remain useful for detecting incorrect actions under adversarial conditions. In this research, we assess this problem from the perspective of operational tool-action correct- ness. we use four AgentDojo suites, four conditions, and three models: Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, and GLM- 4.6V-Flash (9B). We perform a five-fold, task-grouped out-of-fold (OOF) evaluation for each model. We train step-specific logistic- regression probes only on benign and non-defended trajectories and evaluate them on held-out trajectories for benign, prompt- injection, system-override, and tool-poisoning conditions. We compare our hidden-state probe to mean tool-call-token entropy, negative mean tool-call-token log-probability, and self-consistency uncertainty. The paired tests highlight six significant differences after Holm correction. All of the tests involve Qwen2.5-14B- Instruct under the system-override condition, with all three scalar uncertainty metrics obtaining higher area under the receiver operating characteristic curve (AUROC) and average precision (AP) than the hidden-state probe. These results suggest that benign-trained hidden-state probes should not be assumed to reliably detect operational tool-action errors under adversarial conditions. 

---
# Task presentation and execution paths shape LLM mathematical performance 

**Authors**: Min Huang, Mingyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.11102)  

**Abstract**: Mathematical performance depends on how a language model receives information, uses generated text and submits an answer. We study these dependencies through paired comparisons that separate submission requirements, draft reuse and candidate availability. On 65 development problems, an instruction permitting continued derivation produced no valid final answers in 390 short-output calls; a single-line submission instruction produced 390 format-valid outputs, including 389 parseable answers. Under the fixed submission protocol, 198 problems with failed initial submissions were answered correctly in 9.09p of cases with the question alone, 33.84p with its draft, 46.46p with candidates, and 59.60p with both. Drafts added 13.13 percentage points with candidates fixed, and candidates added 25.76 points with drafts fixed; both effects were significant after Holm correction and were subadditive on the accuracy scale. Retaining a trailing excerpt outperformed retaining a leading excerpt of similar average length by 10.10 points Holm-adjusted p 0.027. A second completion model showed the same directions of input effects on the fixed materials. In a cohort with parseable initial submissions, candidates added no correct answers when the previous draft was supplied, but added 27 points when it was absent. These results establish a conditional value of generated text: a failed submission can leave useful material, whose contribution depends on what else is supplied and what is retained. 

---
# When Calibration Data Matter in Sub-4-Bit Language Model Quantization 

**Authors**: Yehong Jiang, Yen-Kuang Chen, Xinmin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2611.11079)  

**Abstract**: Low-bit post-training quantization reduces language-model memory but can sharply degrade downstream task accuracy. Calibration usually relies on generic text, but when task-informed data recover accuracy and which quantizers benefit remain poorly understood. We introduce FragCal-DC, which supplements generic text with training documents from task families with low retention of full-precision accuracy in a two-bit scan. We use a two- to four-bit sweep to locate functional regimes and four-seed calibration comparisons to evaluate six data-dependent methods across six dense models (1.7B-8B) and ten tasks. At their lowest tested functional bit widths, iterative SignRound V2 and OmniQuant improve at two bits, whereas single-pass GPTQ and AWQ show small mean responses at three bits. At two bits, SignRound V2 gains 1.83 percentage points in mean ten-task accuracy over generic-text calibration across six models. At 70 billion parameters, GPTQ and SignRound V2 both improve at two bits, with functional generic-calibration baselines. For two-bit SignRound V2, task-level controls show that recovery concentrates on families included in calibration, while incorrect answer designations retain about 90% of the mean accuracy gain. Changes in reconstruction error do not consistently track accuracy gains. These findings support task-informed calibration as a design principle for preserving downstream accuracy in low-bit quantization. 

---
# System Prompts Affect Response Bias More Than Discriminability in LLM Safety 

**Authors**: Mohan W. Gupta, Sandy Tanwisuth  

**Link**: [PDF](https://arxiv.org/pdf/2611.11074)  

**Abstract**: Large language model safety evaluations often focus on a compliance-refusal tradeoff: models should refuse unsafe requests without refusing safe ones. We argue that this misses a broader form of prompt-conditioned task selection, where the same request can elicit different behaviors depending on the system prompt. We investigate this using a controlled evaluation crossing eight safety-task prompt families with three clarity levels across HarmBench, XSTest, and IHEval. Holding benchmark items fixed, we evaluate 13 models and use signal detection theory to distinguish two possibilities: whether models become better at telling safe from unsafe requests (discriminability), or simply more willing to refuse (response criterion). We find substantial differences in discriminability across models, but system prompts primarily shift the willingness to refuse, with smaller and less consistent effects on discriminability. Increasing prompt clarity makes models more willing to refuse without reliably improving their ability to distinguish safe from unsafe requests. Beyond refusal, prompt families also shift responses among six distinct behavioral policies, including clarification and safe help. Together, these findings show that system prompts primarily change how models respond to safety-relevant requests rather than consistently improving their ability to distinguish safe from unsafe ones. 

---
# Beyond AUROC: Deployment-Aware Evaluation of Model-Internal Jailbreak Detectors 

**Authors**: Mohammad Tehabsim, Basil Al-Housani, Nour Alhussien, Phung Lai  

**Link**: [PDF](https://arxiv.org/pdf/2611.11067)  

**Abstract**: Large language models (LLMs) are increasingly deployed in user-facing systems, where jailbreak attacks can bypass safety controls and induce harmful and policy-violating outputs. Recent model-internal detectors use hidden representations, gradients, or internal signals to identify such prompts, but their practical security value depends critically on behavior at low false-positive operating points rather than on area under the receiver operating characteristic curve (AUROC) alone.
We study two data dependencies that shape this behavior: calibration data, which determine the detection threshold, and reference data, which can influence detector scores. Across four model-internal detectors, four LLMs, and twelve jailbreak families, thresholds calibrated on curated benign prompts frequently fail to preserve their false-positive rate (FPR) budgets on real-user traffic. Recalibration improves false-positive control but reveals substantial family-level coverage gaps, including cases with high AUROC but low true-positive rate (TPR). Causal reference-set interventions show that changing reference composition can substantially alter family-level detection and that improving coverage of one attack family can reduce coverage of others.
These findings show that calibration and reference data are part of a detector's security behavior. We therefore recommend evaluating model-internal jailbreak defenses using realized FPR on deployment-like traffic and per-family TPR at explicit low-FPR operating points, with AUROC used as a complementary ranking metric rather than the primary security measure. 

---
# Lightfoot: Graph Retrieval-Augmented Generation with Exact, Model-Free Document Deletion 

**Authors**: Uday Allu  

**Link**: [PDF](https://arxiv.org/pdf/2611.11010)  

**Abstract**: Graph-based retrieval-augmented generation (GraphRAG) improves multi-hop question answering, but deployed systems face two costs that research comparisons rarely measure: building the graph, and removing documents from it. We present Lightfoot, a GraphRAG system in which every entity, link, fact and synonym edge is a deterministic function of per-passage extractions carrying chunk-level provenance. Indexing makes one language-model call per passage, retrieval makes none, and deleting a document is a database operation with no model calls. On the full 11,402-passage MuSiQue corpus we verify that, after deleting 10% of the documents, the graph is identical to a fresh index of the surviving documents. We compare Lightfoot with HippoRAG 2, LightRAG and fast-graphrag under identical extraction, answer and embedding models on MuSiQue, 2WikiMultiHopQA and HotpotQA. Lightfoot builds the cheapest index on all three datasets, 26-31% below HippoRAG 2 and 3-6x below LightRAG and fast-graphrag. Against HippoRAG 2 it is significantly better on MuSiQue retrieval (+2.8 points recall@5 and +4.3 recall@10 on a held-out split) and on HotpotQA answers (+7.7 points exact match), and significantly worse on 2WikiMultiHopQA (-3.9 recall@2, -8.4 F1). Deleting the same documents took LightRAG 2,187 model calls and 3.7 hours; fast-graphrag has no deletion path. Code and per-question results: this https URL 

---
# DM-Call: Decision-Model-Guided Function Calling with Cumulative Verification Feedback 

**Authors**: Minhyeong Yu, Wonduk Seo, Seunghyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.10977)  

**Abstract**: Reliable function calling requires Large Language Models (LLMs) to select appropriate functions and generate arguments consistent with user intent. However, syntactically valid calls may still be semantically incorrect due to inappropriate function selection or mismatched arguments. In this paper, we propose Decision-Model-Guided Function Calling (DM-Call), a framework that combines local schema validation with separate decision-model verification of function selection and argument correctness. For rejected candidates, DM-Call accumulates verification probabilities, binary judgments, and schema errors across attempts. A hybrid feedback representation further combines numerical probabilities with verbal correctness labels to guide iterative refinement without executing external tools. Extensive evaluation across BFCL and API-Bank (first-call) demonstrates DM-Call boosts function-calling accuracy versus direct generation, at modest additional cost. 

---
# The Surprising Effectiveness of Deleting Weights in LLM Post-Training 

**Authors**: Jack Lu, Zhenbang Yang, Mike Lasby, Tejas Pote, Yani Ioannou, Mengye Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.10866)  

**Abstract**: How little of a pretrained large language model (LLM) has to change for it to acquire new capabilities? We find that zeroing fewer than 0.05% of its weights, with no other modification, matches full fine-tuning (FFT) at scale across three LLM post-training regimes: reinforcement learning with verifiable rewards, supervised fine-tuning, and on-policy distillation. Our method, Deletion Tuning, learns a binary mask over the gate projection of transformer blocks. Two practical advantages follow: the trained mask is hundreds to thousands of times smaller than the corresponding FFT checkpoint, and Deletion Tuning forgets far less, which addresses a central challenge in continual learning. Under reinforcement learning, its prior-task accuracy stays near the base level while FFT, LoRA, and random sparse fine-tuning lose up to 5.4 points. To understand why masking alone suffices, we analyze the geometry of the learned weight changes under reinforcement learning using stable rank, spectrum drift, and principal-weight overlap. Compared with LoRA and random sparse fine-tuning, Deletion Tuning is closest to FFT across the three measures. 

---
# CompAIssionate Other: Identity Assumptions in LLM Responses to Distress 

**Authors**: Drishti Goel, Ethan M. Nguyen, Violeta J. Rodriguez, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2611.10808)  

**Abstract**: Individuals experiencing suicidal ideation (SI) or psychological distress may disclose their social identities when seeking support from LLMs. Attending to identity-related stressors can be important for providing contextually appropriate and empathetic support. However, disclosing an identity does not necessarily mean that it is a cause of distress. In this work, we audit how LLMs interpret distress through disclosed identities. Using paired counterfactual interventions, we evaluate six LLMs across two SI and distress datasets, 16 identity and control conditions, and two prompting protocols. Our findings suggest that models can transform identity disclosures into explanatory narratives, attributing distress to presumed identity-related adversity. These assumptions frequently appear within otherwise empathetic responses, particularly when marginalized identities are disclosed. We contextualize our findings through the lens of compassionate othering to characterize these patterns as a form of harm in which empathetic responses impose identity-linked assumptions on individual suffering. These responses can misrepresent individuals' experiences by treating their identities as explanations for distress without supporting evidence. Our findings highlight the need to mitigate these harms by grounding interpretations of distress in individuals' disclosures and avoiding unsupported identity-linked assumptions. 

---
# Generating Diverse and Verified Synthetic Clinical Notes with Labels by Construction 

**Authors**: Matthias Lehmann, Madeleine Werthebach  

**Link**: [PDF](https://arxiv.org/pdf/2611.10789)  

**Abstract**: Under strict privacy regulations in the clinical domain, de-identification is necessary to enable the use of clinical data such as for natural language processing. However, we currently lack sophisticated ways to tell which de-identification models actually are suitable. Specifically, we need benchmarks that are challenging, diverse, and clinically plausible, all while ensuring label correctness. To this end, we present SPARCLING, an LLM-based pipeline to generate synthetic discharge letters with character-exact PHI span annotations. Our pipeline combines structured attribute sampling, template generation, and red teaming to obtain diverse PHI-labeled discharge letters. In our experiments, we demonstrate this superior content diversity over existing public datasets. Further, we show SPARCLING's downstream utility as a benchmark to uncover failure modes of de-identifiers in the German clinical domain. Finally, our experiments show that training on our synthetic data can yield more capable de-identifiers than training on the scarce public data alone. Our code is available at this https URL. 

---
# Matchability-Aware Conformal Prediction for Open-Ended Language Model Generation 

**Authors**: Ilia Azizi  

**Link**: [PDF](https://arxiv.org/pdf/2611.10738)  

**Abstract**: Large language models (LLMs) often do not produce calibrated uncertainty estimates for their answers. We propose CAELUM, a framework that builds semantic prediction sets while separating aleatoric from epistemic uncertainty. CAELUM samples answers using an ensemble of clarification prompts, clusters them by semantic equivalence, and estimates the aleatoric and epistemic components via kernel-based von Neumann entropy before calibrating nonconformity scores with split conformal prediction. We use a weight $w_\lambda(x)=1+U_{\mathrm{ale}}(x)+\lambda\,U_{\mathrm{epi}}(x)$ to control for the contribution of the epistemic uncertainty of each instance and fit a coefficient $\lambda^\star$ to minimize prediction set size. When the objective is nonflat and selection is stable across splits, $\lambda^\star$ could be a diagnostic, where large values may indicate epistemic dominance (retrieve, abstain) and $\lambda^\star\to 0$ aleatoric dominance (clarify). For individual queries, a log ratio $R(x)$ compares the weighted epistemic component with the aleatoric component. Theoretically, we demonstrate that the coverage is capped by matchability, the fraction of items whose reference answer appears in any sampled cluster. Empirically, this matchability can be predicted from the original question features using a lightweight classifier. We use those predictions to define a stratum and derive a conformal coverage bound within it. Across 30 LLM and dataset combinations (4-32B), mean coverage is 0.74 in the learned stratum versus 0.65 overall, averaged over the 78 evaluations using the learned classifier. Project page: this https URL 

---
# Perception, Not Reasoning, Limits Visual Theory of Mind 

**Authors**: Mohamed Rayan Barhdadi, Syed Talal Wasim, Juergen Gall, Erchin Serpedin, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2611.10694)  

**Abstract**: Vision-language models pass text-based false-belief probes at near-human levels but drop by over twenty percentage points on abstract panels the moment the same problem is conveyed visually. The consensus diagnosis is conceptual, prescribing new belief-reasoning modules. We argue the dominant bottleneck is perceptual: not what the model cannot reason about, but what it cannot see. We propose a five-condition dissociation protocol that holds the reasoning task constant while varying perceptual fidelity from visual only to text only, and quantify the result with the Perceptual Contribution Ratio, the share of the visual-to-text gap a given fidelity closes. We test this on two kinds of scenes: abstract 2D panels in the Heider--Simmel tradition, and naturalistic 3D game scenes from MuMA-ToM and MindPower. The evaluation covers seven VLMs and roughly 159,000 API calls. Ground-truth scaffolds substantially close the gap in every setting, and on the abstract panels and MuMA-ToM they match or exceed the text-only reasoning. Whereas scaffolds built by automatic parsers, including frontier VLMs, fail. The most common error in the visual-only condition is \emph{reality bias}, which accounts for 60\% of false-belief failures. Models report where an object actually is, not where an absent agent believes it is. This points to a perceptual front-end that fails to track which agents were present at each moment. Under the evaluated protocol, structured visual scene-state extraction, rather than belief inference, is the dominant bottleneck for visual ToM. 

---
# Machine Hearing: Can Audio LLMs Generalize Beyond Human Auditory Perception? 

**Authors**: Liwei Liu, Jun Tao, Xingyuan Li, Xuenan Xu, Chen Zhang, Junyi Wang, Boya Zhang, Xiaoyu Yang, Zhen-Hua Ling, Wen Wu, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.10693)  

**Abstract**: Multimodal LLMs have made remarkable progress in perceiving the physical world, while their perceptual scope remains largely human-centric, focusing on what humans can directly see or hear. This progress raises the prospect of perception beyond human limits, such as extending hearing to sounds that are inaudible to humans. More broadly, the physical world is full of waves, vibrations, and oscillations, only a small fraction of which are directly audible to humans. This invites the extension of multimodal LLMs toward a broader form of machine hearing, which remains largely underexplored. To investigate this direction, we introduce the Machine Hearing Benchmark (MHB), comprising 6,128 instances across 13 tasks and seven domains. MHB spans highly heterogeneous input regimes, with sampling rates from approximately $10^{-5}$ to $10^6$ Hz, sequence lengths from $10^1$ to $10^6$ samples, and both single- and multi-channel inputs. Evaluations of current open-source and proprietary Audio LLMs (ALLMs) reveal limited zero-shot transfer to machine hearing. Supervised adaptation on machine-hearing data alone does not close this gap, suggesting that the challenge extends beyond data exposure to a systematic mismatch between existing audio front ends and machine-hearing inputs. To address this mismatch, we introduce MachEar, a lightweight encoder extension that complements the pretrained Audio LLM for machine-hearing inputs. The results show that MachEar equips a pretrained Audio LLM with stronger machine-hearing capabilities while largely preserving its existing audio understanding performance. 

---
# Pandemonium: Longform Target-Speaker Speech Recognition with Cross-Speaker Interactions 

**Authors**: Alexander Polok, Dominik Klement, Ivan Medennikov, Antoine Laurent, Petr Pálka, Samuele Cornell, Shinji Watanabe, Sanjeev Khudanpur, Jagadeesh Balam, Jan Černocký, Lukáš Burget, Matthew Wiesner  

**Link**: [PDF](https://arxiv.org/pdf/2611.10659)  

**Abstract**: Diarization-conditioned target-speaker ASR (TS-ASR), in particular Diarization-conditioned Whisper (DiCoW), is an accurate and data-efficient approach to multi-talker transcription. It has, however, three structural limitations: the model transcribes each speaker independently, operates on short (30 s) windows, and optimizes its diarization and TS-ASR components separately. We address the first two. We integrate the diarization conditioning into a longform Parakeet backbone (DiCoP) and add cross-speaker attention, which lets the model attend across the speaker dimension so that speakers are no longer decoded in isolation, and an STNO mask refiner that re-estimates the diarization-derived conditioning. As conversational data with genuine multi-speaker interaction is scarce, we scale training with a pseudo-labeling pipeline that mines 5.4k hours of in-the-wild multi-speaker audio from YODAS2. Combining these with synthetic and real data yields state-of-the-art results across several meeting benchmarks. We open-source the resulting models, DiCoP and Pandemonium, and the pseudo-labeled data. 

---
# Same Contacts, Different Rooms:Meeting Format Shapes LLM Group Influence Beyond Pairwise Contact Counts 

**Authors**: Zhexi Gu, Shusheng Cao, Runping Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.10641)  

**Abstract**: Simulations of LLM agent collectives usually summarise interaction as a pairwise contact network. We test that summary with a contact-matched intervention: the members, their initial stances and every pairwise contact count stay fixed while the meetings are reorganised, and each member gives a one-step support readout after reading the arguments of everyone present. Across six open-weight LLMs and 14 debate topics, merging triads into five-person meetings (resize) moves group support more than reshuffling who meets in triads (regroup), although both move the same 60 dyadic exposures. The root-mean-square change is 4.36 versus 2.59 percentage points (pp), larger for resize in all six models and in 81 of 84 model-topic cells. Resizing also has a consistent direction: it amplifies the initial majority of the merged members by 1.72 pp, in every model, whereas regrouping has no direction (<= 0.03 pp). A contact network cannot represent regrouping at all, yet pairwise rules applied meeting by meeting explain up to 76% of its effect, and with identical weights they raise resize R^2 by 0.51-0.58 for two models. We show that the gap between the two placements is an exact Jensen gap of the sigmoid readout, bounded by how much a member's predicted logits vary across meetings; this dispersion tracks the gap with log-log r=0.97 and tells a modeller when contact counts are a safe summary. Meeting format is a first-class variable for LLM social simulation. 

---
# Coverage-Verified Sparse Attention: Closed-Loop Quality Control for Long-Context LLM Decoding 

**Authors**: Yan-Cheng Li, Chun-Yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.10634)  

**Abstract**: Sparse attention accelerates long-context decoding by reading a subset of the KV cache, but existing methods are open-loop: the kernel commits to its selection with no signal indicating whether enough attention mass was preserved and no recovery path when it was not. The needed signal already exists inside the kernel. The fraction of softmax mass on the selected tokens, which we call coverage, satisfies an exact algebraic equality with per-step output error and serves a dual role as the acceptance criterion and the interpolation weight for recovery. Coverage-Verified Sparse Attention (CVSA) turns coverage into a training-free, closed-loop verify-then-correct decode loop that accepts high-coverage drafts and recovers only the heads that fall short, with no sampling, no draft model, and no speculative-decoding infrastructure. Quality varies by only $0.14$ points across the threshold sweep; budget and scheduling ablations characterize the deployed operating point. On LongBench at 7B and 70B, CVSA comes within $1.13$ points of dense attention, statistically matching it at 70B, and it loses $0.8$ points on RULER; measured median decode-step speedup reaches $6.21\times$ at 128K context. 

---
# What Do Collusion Probes Read? Three Pitfalls in Evaluating Multi-Agent Collusion Monitors 

**Authors**: Mainul Hasan Mohsin, Mohammad Shahidur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2611.10620)  

**Abstract**: Activation probes look like near-perfect monitors for secret collusion among large language model (LLM) agents. We show that they mostly detect that an agent was told to collude, not that it decided to. Our benchmark (2,500 committee and 1,476 code-review episodes) adds agents who receive the collusion instruction and refuse it, separating instruction from decision. Published detectors barely tell colluders from these refusers, and our best probe separates them mostly by reading the instruction: without it, agent AUROC falls from 1.00 to 0.61. Two further pitfalls inflate reported results: scored after the vote, a rule that only reads the votes reaches AUROC 0.990 but finds nothing before it; and in 57% of attempts no colluder argues for the target in public. The decision is still there: a probe trained on agents' own choices reads it (AUROC 0.73 without the instruction), and probes find colluders whose goal was fine-tuned into the weights (AUROC 0.995; pre-registered). We turn the controls into a protocol for testing any collusion monitor, and adapt scan statistics into an online monitor for groups of any size. 

---
# In Search of Lost Time: Temporal Competence in Large Language Models 

**Authors**: Ryan Liu, David Lee, Icey Siyi Ai, Marwa Abdulhai, Rachit Dubey, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2611.10536)  

**Abstract**: Today's large language models (LLMs) increasingly interact with time: AI agents schedule tasks, coordinate with people and other AI systems, and even decide how long to think. Yet LLMs only ever encounter time secondhand---in descriptions rather than experience. Does this hinder their reasoning and decisions that involve time? Our paper traces this question from a model's knowledge to its actions through four successive evaluations. We find that models know the rules of time, which can be read, but struggle to predict durations, which must be lived; while they are highly accurate (92.0-97.5\%) on a collection of questions built from the theorems of various time-related literatures, they underestimate their own completion times by up to 99.5\% and fail to identify which tasks will take them longer to run. And while they are better at estimating durations of human tasks, their duration judgments change based on how an event is worded, magnifying similar human biases by up to 1.3$\times$. Moving from estimation to action, we find that some LLMs disregard time costs completely when answering user queries, ignoring a 1,200-fold increase in wait time entirely. Together, our experiments reveal new challenges posed to LLMs as they navigate a dimension they cannot directly perceive, and highlight how their blind spots can even be simple cognitive abilities that humans take for granted. 

---
# SituRecBench: Evaluating Frontier Models as Situated Recommendation Assistants in 3D Environments 

**Authors**: Jingtong Yang, Dongding Lin, Jian Wang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.10349)  

**Abstract**: Recommender systems are transitioning from static retrieval engines into interactive assistants grounded in physical environments. However, current studies largely neglect the combined complexities of spatial navigation, visual grounding, and multi-turn social interaction. To bridge this gap, we introduce SituRecBench, an interactive benchmark for evaluating frontier vision-language models (VLMs) as situated recommendation assistants within high-fidelity 3D environments. We reformulate recommendation as a situated assistance process in which an assistant interacts with a user while sharing the user's egocentric view. Our benchmark simulates real-world shopping environments (e.g., supermarkets and home furnishings stores), requiring assistants to elicit preferences, ground recommendations in the environment, and guide the user to relevant locations through dialogue. The benchmark evaluates dialogue quality, recommendation utility, and navigation effectiveness through a unified interactive protocol. Extensive experiments reveal persistent challenges in spatial grounding and preference elicitation, highlighting limitations in their ability to integrate global contextual information with local observations. SituRecBench provides a rigorous foundation for advancing the development of situated assistants in the real world. 

---
# Collaboration Signatures: Tracking Long-Horizon Dynamics of Cross-Meeting Collaboration Networks 

**Authors**: Shengqi Zhu, Sheshera Mysore, Bahareh Sarrafzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2611.10341)  

**Abstract**: Enterprise platforms have received increasing attention as a core component of social networks and the Web. However, existing work mostly relies on fragmented logs of small units of work, like emails and chats, and focuses on who connects with whom without sufficient account of collaboration content and dynamics. This work systematically explores tracking collaborations across sequences of meetings as a new approach to interpreting their long-horizon dynamics. Linking 45,426 meetings in a large multinational technology company based on full transcripts and 1.6 million extracted agenda items, we establish a cross-meeting collaboration network of 6,065 collaborations. The network recovers the natural shapes of workplace collaborations: meetings, as snapshots of ongoing collaboration, link into largely tree-shaped structures, while the work content forms nested directed acyclic graphs (DAGs). Further, we explore how these networks evolve over time, examining three established perspectives: task division, team composition, and temporal continuity. Each explains part of the dynamics but does not capture the distinctness of long-horizon workplace collaboration: e.g., work flows in a consistent direction between a few pairs of subjects, yet a collaboration's subject explains little of how its meetings are conducted. Instead, accumulating the agenda, interaction, and language of meetings along each tree yields a signature of the collaboration, a distinctive profile that picks out the collaboration of a held-out meeting at 4.5 times chance. The signature models collaboration better than team or task views, individually or combined, in next-meeting prediction and tree-level fitting. We discuss the implications of these cross-meeting dynamics for designing office applications and support tools, and call for next-generation Web systems that represent collaborations over their long horizon. 

---
# Two-Sided Conformal Evidence Sets for Retrieval-Augmented Generation 

**Authors**: Ben Jenkins, Mihaela Cardei  

**Link**: [PDF](https://arxiv.org/pdf/2611.10335)  

**Abstract**: Retrieval-augmented generation (RAG) can surface passages that support a claim, contradict it, or fail to address it, yet existing calibrated grounding methods are typically one-sided and model only supporting evidence. We introduce Conformal Evidence Sets (CES), a two-sided framework that returns separate support and refutation sets with finite-sample inclusion coverage of at least $1-\alpha$ on each side under exchangeability; a Bonferroni union yields joint coverage of at least $1-2\alpha$. On labeled candidate pools from the contrastive VitaminC benchmark, CES closely tracks nominal coverage across $\alpha\!\in\!\{0.05,0.10,0.15,0.20,0.30\}$, with maximum deviations of $3.0$ percentage points for support and $5.4$ for refutation, and reaches essentially the same empirical coverage--size frontier as a test-tuned fixed-threshold heuristic without any test exposure. Stress tests mark the boundary of that guarantee: under FEVER distribution shift both sides under-cover, by up to $12.7$ percentage points, and support-side group-stratified recalibration substantially reduces the support gap. Scorer swaps, hard-negative retrieval, cross-dataset transfer, and a live end-to-end pilot expose a clear separation: when exchangeability holds and relevant evidence is retrieved, evidence inclusion remains calibrated even with an imperfect scorer, whereas the support/refute verdict tracks scorer quality directly. Calibrated evidence coverage and reliable verdicts are therefore distinct properties and should be reported separately. 

---
# Police Behavior Patterns Through Radio Analysis 

**Authors**: Milo Z. Trujillo, Adam Perhala, Elaine Zou, Tyler Mckenzie, Juniper Lovato, Brooke Foucault Welles, Alyssa Hasegawa Smith  

**Link**: [PDF](https://arxiv.org/pdf/2611.10322)  

**Abstract**: In the United States, most police departments communicate using police radio, which allows for officer-to-officer and officer-to-dispatch communication without depending on cellular network or WiFi availability. While some departments use encrypted digital radio, many continue to transmit in the clear, and are legally monitored by public citizens with `police scanners.' Because police radio is intended for internal police use and represents real-time communication, it may offer valuable insight into patterns of law enforcement activities without the potential bias and filtering introduced by a press release after the fact. In this study, we collect and examine over 14 million radio transmissions representing over 100,000 hours of speech by officers in 34 cities across the United States from July 2025 through July 2026; to our knowledge the largest scale analysis of Broadcast Police Communication (BPC) to date. We highlight both typical patterns of police communication and categorize anomalous activity that represents incidents including protests and major weather events. We present this dataset as a starting point for an array of police communication studies, conduct our own preliminary analysis, and propose several further areas of inquiry. 

---
# How Much Is Left Matters: False Positives and False Negatives in Partial-Prompt Contamination Tests 

**Authors**: Abhishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2611.10307)  

**Abstract**: Partial-prompt completion, where a model receives the first part of a benchmark problem and is checked for whether it reproduces the rest, is a widely used test for benchmark contamination. We audit this test on seven open-weight models using MATH-500 and recent AIME 2025/2026 problems, which serve as true negatives. Memorization flags depend strongly on how much text remains: for every model, problems with at most 8 leftover words are flagged several times more often than problems with at least 25 (31.5% vs. 5.5% for Qwen2.5-7B-Instruct), and half to three-quarters of all ROUGE-based flags come from short leftovers. On true negatives, a ROUGE-L >= 0.75 threshold flags up to 6.7% of problems at a 60% prefix, and at an 80% prefix even exact match flags 8.6% (pooled over models; 6.2% on AIME 2026, which postdates every model), against 0% at 40-60%. The test also misses contamination upstream: models distilled from Qwen2.5-Math, whose base is heavily flagged in prior work, show almost no model-specific recall. Calibrating each model against models from other families removes most flags while isolating a consistent Qwen2.5-specific signal at every prefix length. We recommend length-stratified reporting, prefixes of at most 60%, style-matched post-cutoff controls, and cross-family calibration. 

---
# Do large language models rank moral values? A transitivity test of moral competence 

**Authors**: Katelyn A. Wasson, Kiwon Song, James M. Jennings III, Brittney Currie, Konstantina Sokratous, Clintin P. Stober  

**Link**: [PDF](https://arxiv.org/pdf/2611.10265)  

**Abstract**: LLMs are increasingly used to support decisions that require moral consideration. While LLMs perform well on morality benchmarks from psychology and philosophy, producing appropriate outputs on these benchmarks (moral performance) does not establish that an LLM weighs competing moral values at stake in a decision (moral competence). We test a minimal, necessary condition for such weighing: that an ordinal ranking can represent LLM moral choices. Such a ranking requires transitivity. To assess transitivity, we presented LLMs with pairwise trade-offs among moral value violations in vignette choice sets and extracted resulting choice probabilities from first-token distributions. Tested against three stochastic transitive choice models, most choice sets admitted weak or unstable orderings. Aggregating an LLM's responses, a common practice in LLM moral evaluation, produced transitive structure that matched no individual choice set. Evaluations of AI morality should establish that choices are structured before asking whether they are aligned. 

---
# PACE: Keeping the Right Tokens in the KV Cache Is Not Enough for Long-Horizon LLM Agents 

**Authors**: Suman Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2611.10240)  

**Abstract**: Long-horizon LLM agents accumulate sessions of up to millions of tokens, far more key-value (KV) cache than one GPU holds, so serving systems keep a small budget of tokens and evict the rest. Eviction methods differ in which tokens they keep but share one goal: that the tokens a later step needs are still cached. We show this is not enough. In 100K-token coding sessions on Qwen3-8B with a 2,048-token budget, a planner-guided policy that keeps over 99% of the needed tokens resident still fails late steps increasingly often: its accuracy falls from 93% to 64% as the gap between a definition and its use grows from 2K to 90K tokens, at residency 0.996. The needed tokens are resident but unretrievable: what grows is their rotary-position distance to the query. We present PACE (Planner-Aware Context Engine), a KV-cache engine that couples plan-aware retention, in which the agent's planner tells the cache which context future steps will read, with bounded positions, which renumber kept tokens so their rotary distance to the query stays below 8K at any session length. In a pre-registered evaluation on fresh tasks, PACE passes 97% of hidden unit tests at 100K tokens, against 42% for H2O and 3% for SnapKV under the same budget, and needs both components: either alone reaches 63% (the combination beats each, p < 10^-4). The result holds with length: in 1M- and 2M-token sessions on one 24 GB GPU, with definitions written 0.9M and 1.8M tokens before use, PACE passes 99% and 100% of checks and recalls every session fact, while bounded positions without foresight falls from 33% to 13%; PACE holds under 0.7 GB of KV state where the full cache would need 147-295 GB. Renumbering is exact, costs no measurable perplexity, and adds 9% wall time. On Mistral-7B, whose coding ability caps it at 40% even with nothing evicted, PACE stays at that ceiling while retention alone falls below it. 

---
# CoRA-Bench: Evaluating Evidence Preservation by Context Selectors Under Token Budgets 

**Authors**: Huzaifa Pardawala, Siddhant Sukhani, Sudheer Chava, Agam Shah  

**Link**: [PDF](https://arxiv.org/pdf/2611.10238)  

**Abstract**: Context selection can reduce input length during long-input inference but can remove evidence required to answer a query. We introduce CoRA-Bench, a benchmark for evaluating evidence preservation under shared token budgets. We compare head and tail truncation, BM25, dense retrieval, dense retrieval with reranking, LLMLingua-2, and Provence across BABILong, LOONG Financial Spotlight, a custom NoLiMa diagnostic, and LongMemEval. Evidence-preservation results reveal task-dependent tradeoffs: BM25 achieves 0.921 supporting-fact recall on BABILong at 128K, dense retrieval and dense reranking each achieve 0.908 needle recall on NoLiMa at 64K, and dense retrieval achieves 0.993 answer-bearing-turn recall on LongMemEval at 64K. Evidence, token retention, and complementary reader evaluations use the same selected contexts. Reader evaluations use dataset-native answer metrics and three runs on the same questions, with paired bootstrap confidence intervals. At a 32K budget, BM25 achieves 46.20% BABILong answer accuracy and dense retrieval achieves 71.13% LongMemEval accuracy. The results show that method rankings vary across evidence types and between evidence recall and downstream accuracy. 

---
# AVCORP: Identifying and Resisting Deception via Communication Theory and Theory of Mind 

**Authors**: Fardin Saad, Zhe Zhang, Sanjana Cheerla, Amanul Haque, Munindar P. Singh  

**Link**: [PDF](https://arxiv.org/pdf/2611.10176)  

**Abstract**: Today's LLM-based agents are vulnerable to deception by users and other agents. Defending against deception requires Theory of Mind (ToM) reasoning to infer the beliefs and goals of others as well as modeling tactics based on communication theory. To enable training models to detect deception, we adopt the well-known deception game, Avalon. We present a corpus, AVCORP, of 5,000 annotated utterances from 250 games of Avalon. AVCORP includes (1) 37 tactics mapped to a 4x4 behavior matrix, and (2) reasoning traces capturing abductive hypotheses, suspicion levels, and nested ToM beliefs. We train Qwen3-8B on AVCORP and evaluate it on 25 held-out games and 20 games played by humans. On held-out games, our best trained detector outperforms its untrained baseline and frontier models (GPT-5.6 Sol, Gemini 3.1 Pro, DeepSeek-V4-Pro), matching Claude Opus 5.5. On human games, every trained detector exceeds the untrained model in F1 score across all rounds. 

---
# DeepResearch RewardBench: Evaluating Reward Models for Deep Research Report 

**Authors**: Shudong Liu, Yuting Zhong, Guanhua Chen, Xinyi Yang, Junnan Liu, Lidia S. Chao, Jiaheng Liu, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2611.10156)  

**Abstract**: Empowered by Large Language Models (LLMs), deep research systems integrate online information into research reports through multi-step web searching and information synthesis. Evaluating these reports presents a significant challenge due to their long context, high information density, and domain specificity, necessitating the use of AI evaluation as a substitute for prohibitively expensive human assessment. However, few studies have examined the effectiveness of existing reward models or LLMs in understanding and evaluating such research reports. To bridge this gap, we introduce DeepResearch RewardBench, the first benchmark dedicated to evaluating reward modeling for research reports. We collect diverse report candidates from multiple agents or tool-augmented LLMs. These candidates are scored using query-specific golden rubrics, and we ensure data quality and difficulty through a combination of multi-round filtering and human verification. Our benchmark comprises preference datasets formatted as pairs, triplets, and quadruplets to test ranking capabilities across varying levels of difficulty. Extensive experiments demonstrate that while current reward models handle general coherence effectively, their performance significantly degrades when evaluating research reports. Generative LLMs acting as judges demonstrate comprehensive superiority in this domain, with comparative evaluation proving more effective than direct scoring methods. We anticipate that this benchmark will serve as a vital testbed for the evaluation and advancement of research agents. 

---
# SPRING-5: Measuring Tool Documentation Repair for LLM Agents by Efficiency 

**Authors**: Guan Zheng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.10040)  

**Abstract**: Methods that repair tool documentation automatically are mostly evaluated on task accuracy. We report a benchmark that scores them on cost. SPRING-5 is 20 tool families and 344 tasks, each with a deliberately degraded specification, a hand-written honest one, a deterministic backend and an answer key, scored on calls charged against the ideal call path. A wrong answer is charged the full call budget, so this cost index largely tracks the completion rate, and we report calls per correct task beside it, which carries no such charge. Over 2,644 baseline sessions, 54% of wrong answers arrive with no rejected call, no refusal and no retry. The benchmark carries three validity instruments. A build-time ceiling assertion requires the honest specification to reach the ideal path and fails systematically on only one family. A dud rule names the families a stronger model solves cold and catches four. A directly measured noise floor shows that two evaluations of a byte-identical specification differ by a 95% factor of 0.64 to 1.56, so about one comparison in eight crosses a conventional win threshold on noise alone, and the cheap way of estimating that floor is biased by which families happen to go unedited. Against these instruments, a schema-only method closes 1% of the baseline-to-ideal cost gap, an online self-evolution loop 13%, one batch rewrite over every training transcript 33%, and a variant given correctness labels 58%. The batch advantage over the loop is the evidence it reads: held to the loop's exposure it is no longer detectable. With five and ten tools in context the effect survives, the baseline cost does not change, and only machine-written descriptions sent calls to the wrong tool. On BFCL, 62% of held-out queries are already answered from the shipped documentation, which explains a pooled null there. Code, data and runs are released. 

---
# Toward Safety-Aware AI Agents in Realistic Social Engineering Attack Scenarios 

**Authors**: Zhitao He, Chengru Han, Chi Him Zhang, Yi R Fung  

**Link**: [PDF](https://arxiv.org/pdf/2611.10037)  

**Abstract**: Personal AI agents are increasingly deployed on user devices to execute actions involving real money and sensitive data, but their security under social engineering attacks has not been systematically evaluated. We propose SEA (Social Engineering Attack Benchmark), the first interactive environment for evaluating the intrinsic security capability of AI agents in social engineering attack scenarios, covering 8 daily-life domains, 30+ simulated applications, a 4-level difficulty gradient, and a 5-dimensional evaluation framework, with 320 evaluation tasks designed in consultation with anti-fraud specialists and spanning two culturally distinct settings in Chinese and English. Evaluation on 11 frontier open-source and closed-source models, with GPT-6 evaluated only in Chinese, reveals that all models are breached to varying degrees (Chinese attack success rates ranging from 16.9% to 84.3%) proactive verification rates universally below 32%. We further propose META-SEA, a co-evolutionary framework for attack tasks and defenders. With META-SEA training, Qwen3-8B achieves defense performance comparable to closed-source models while effectively reducing ungrounded action claims. 

---
# zkARc: Succinctly Verifiable Proofs for Neurosymbolic Natural Language Formalization and Verification 

**Authors**: Wyatt Benno, Alberto Centelles, Aman Goel, Benjamin Kiesl-Reiter  

**Link**: [PDF](https://arxiv.org/pdf/2611.10031)  

**Abstract**: Autonomous agents increasingly execute consequential actions on behalf of users, but ordinary middleware guardrails do not provide cryptographic proof that a policy was actually enforced. Their verdicts leave counterparties dependent on the operator unless they repeat the checks themselves, potentially exposing sensitive policies and inputs. We present zkARc, a modular proof-carrying framework that wraps each stage of Automated Reasoning checks (ARc) in an independent, succinctly verifiable zero-knowledge proof. ARc is a neurosymbolic system that translates natural language policies into formal logic and verifies AI-generated content against them using SMT solvers. zkARc certifies the planner's output, the policy's formalization, the action's formalization and both solver queries. The receipt binds these computations to the exact action received and, under the stated assumptions, reveals no additional information about private policies and intermediate model values beyond the declared public information and verdict. This lets a third party check the claimed policy enforcement without re-running the pipeline or trusting the operator. We prove completeness, knowledge soundness and zero knowledge for the composed receipt under explicit assumptions. An implementation on Qwen2-0.5B enforces a spending policy end to end, approving two purchases and denying a third. A full receipt, which also proves model inference, takes about an hour to issue with CPU-only proving and verifies in under a minute, and the solver proof alone is 88 KiB and verifies in 0.3 seconds. 

---
# Alignment Investigation of Mythos 5's behaviour in UK AISI's July 2026 Security Incident 

**Authors**: Olli Järviniemi, Alexandra Souly, Abby D'Cruz, Jordan Taylor, Xander Davies, Robert Kirk  

**Link**: [PDF](https://arxiv.org/pdf/2611.10030)  

**Abstract**: This technical report describes an alignment investigation into Mythos 5's behaviour during a security incident in UK AISI's cyber evaluations. During an evaluation, Mythos 5 attempted to execute a supply-chain attack against real people not involved in the evaluation task. Our investigation set out to understand the specific behaviour in this incident from an alignment perspective and identify broader lessons about failures of alignment. To do so, we conducted experiments in simulated replications of the original evaluation, and analysed model behaviour and reasoning when situated in the evaluation or asked questions about its scope. We find that, as in the original incident, Mythos 5 frequently reasons that the people it's targeting are simulated as part of the evaluation and provides invalid justifications for this conclusion. We also observe that Mythos 5 frequently attacks even in situations where, when directly asked, it states doing so would harm real people. While these issues are particularly prominent in Mythos 5, we find similar results in other models across providers. Our investigation suggests that models providing invalid justifications for their actions could make monitoring more difficult and complicate the analysis of misaligned behaviours observed in simulated environments, and we recommend model evaluators treat observations of undesired behaviour as important evidence of alignment failures even in the presence of simulation awareness in model reasoning. 

---
# How Much Skill Orchestration Can LLM Agents Compile Away? Execution-Grounded Specialization for Token-Efficient Skill Use 

**Authors**: Jiayi Ni, Zac Wang, Alex Zhang, Zheng Liu, Roopa Prabhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.10022)  

**Abstract**: Reusable skills guide large language model (LLM) agents through recurring requests, but each invocation still requires the model to interpret the skill and reconstruct its orchestration. Can agents automatically compile this orchestration into an executable fast path without human intervention? We first study zero-shot skill specialization, which compiles without execution evidence, and find that despite the skill's procedural guidance, incorrect assumptions about runtime bindings can make generated plans brittle and recovery costly. This motivates SkillCache, a one-trace skill specialization method that compiles a completed invocation into a parameterized capsule of programs and natural-language steps, replays it before reuse, and drives it with a plan executor that returns control to the model only when needed, keeping the source skill for recovery. With the Claude Code harness, warm execution of the same request reduces tokens on 87 SkillsBench tasks by 81% with Claude Sonnet 4.6 and by 74% with Claude Opus 4.8 at comparable reward, and capsules stay effective under controlled request changes. Capsules also transfer across models, raising three open-weight models' mean reward by 12-25 percentage points while reducing their tokens by 72-81%, suggesting that execution-grounded compilation is a promising approach to efficient repeated skill use and cross-model transfer. 

---
# Micius-Flash: Lightweight Agentic Intelligence for Engineering 

**Authors**: Hongcheng Gao, Weiheng Li, Yipeng Wei, Yu Lei, Shuo Sun, Yi Li, Yajiao Wang, Jie Luo, Jianwei Wen, Xinyi Zhou, Hongda Li, Yibing Zhang, Henghui Sun, Yufan Zhou, Tianfang Hao, Naihao Xue, Hailong Qu, Jingjing Zhou, Peng Xu, Yongzheng Wei, Yichen Zhu, Ziying Wang, Yibo Wang, Jianshu Zeng, Yuhang Lei, Zeqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.10015)  

**Abstract**: We introduce Micius-Flash, a 27-billion-parameter model for engineering with native multimodal input, deep reasoning and a 256K-token context window. We design the Micius data pipeline as a knowledge-tree-driven data flywheel comprising three stages: knowledge tree construction and expansion, data synthesis, and quality validation with hard-example augmentation. From this data, we train expert models for semiconductor knowledge and reasoning, hardware coding, and semiconductor multimodal understanding, along with agentic experts. Subsequently, we employ Multi-Teacher On-Policy Distillation (MOPD) to supervise training on sequences generated by Micius itself, thereby integrating the capabilities of these expert models into a unified model. On more than 20 datasets with about 15K records, Micius-Flash averages 64.2% on hardware code benchmarks, 15.0 points above a same-size baseline. It scores 62.4% on a semiconductor-specific agent benchmark, exceeding the same-size baseline by 17.2 points. We further develop Micius Harness for industrial applications, connecting the model to specialized tools, including EDA tools and KiCad, as well as execution environments. Case studies in PCB design, chip design workflow automation, and software development demonstrate the model's ability to use tool execution feedback for multi-step planning, code debugging, and iterative verification. We release Micius as an open-source model to advance the development of agents for science and engineering. 

---
# Peer Memory Changes Who LLM Agents Consult Without a Detected Gain in Answer Accuracy 

**Authors**: Jeffrey Chu, Anjali Joseph, Rivan Mandot, Feker Demere, Vincent Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.09990)  

**Abstract**: Multi-agent LLM systems often reuse the same agents across tasks, creating an opportunity to draw on past interactions when deciding whom to consult. We study this form of memory with PeerMem, which saves MACE's LinUCB state and reuses it when the same agents return. Using six agents, a fixed set of 80 HotpotQA questions, and 12 paired lineages per model, we compare cold, warm, and shuffled initialization in an answer-exchange protocol with reference-answer feedback. Under stable evidence access, warm memory lowers pooled hindsight regret and increases consultation of other evidence holders on Qwen3-32B and Gemma-4-31B-it, while final token F1 differences remain uncertain. Qwen interaction traces further show that warm agents more often encounter peer answers that score above their own prior answers. We extend the evaluation to Qwen3.5-9B using freshly learned memory to compare warm and cold initialization under reassigned and paired stable roles. The primary changed-minus-stable final-F1 interaction is -1.38 points, with a 95% interval of [-3.50,+0.73]. Together, these experiments trace the effects of peer memory through consultation, answer revision, and final task performance. 

---
# Counting Events, Not Seeing Them: Diagnosing and Repairing the Readout Failure in Video LLMs 

**Authors**: Dongwu Qian  

**Link**: [PDF](https://arxiv.org/pdf/2611.09949)  

**Abstract**: Video LLMs fail conspicuously at counting how many times an object crosses the screen, a failure attributed to missing object permanence. Using construction-grounded synthetic testbeds with independently validated labels, we show this attribution is incomplete: the count signal is present but unread - linear probes decode per-window presence at 93% while global count decoding collapses to a duration confound. Concurrent work reports the same representation-output dissociation for repetition counting; we show that in exit-and-reappear event counting it is repairable: two readout-directed repairs (episode-localization supervision and a discrete event-symbol side channel) raise same-object counting from 46.7% to 65.0% (oracle 66.7%) on a chance-33% benchmark across 7B-30B systems. Transfer to an independent factory costs nothing, and an audit of the measurement apparatus exposes four data-integrity failures that had obscured the boundary. A real-video audit (REALCOUNT, n=18, blind-audited) shows the boundary persists - best 55.6% at chance 33.3% - and symbol transfer fails zero-shot. A 1-epoch in-domain adaptation bench is negative (six levers at 28-50%), but a 3-epoch schedule shows this is budget-limited for the symbol channel: with a 3-epoch schedule and sequence-disjoint oracle timelines the reader reaches 94.4% on REALCOUNT, and with the deployed detector's timelines a 2-epoch run reaches 72.2%. The residual gap is detector evidence and the reader's match to detector noise, not reader capacity: a frozen-feature boundary probe finds the signal weak (AUC 0.62 over a 0.51 permutation control) and non-transferring across clips (0.77 within-clip vs. 0.62 across windows). 

---
# Autonomy Is Absent, Not Eroded: An Observational Study of User Autonomy in Multi-Turn LLM Conversations 

**Authors**: Chaoran Chen, Isabel Barberá, Yuhuai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.09923)  

**Abstract**: Does prolonged interaction with Large Language Models reduce users' observable autonomy during conversation? We test this degradation hypothesis by scoring user autonomy across 2,877 multi-turn conversations (46,952 turns) from WildChat and validating on 500 conversations from LMSYS-Chat-1M. Our core finding is a null result: population-level autonomy remains stable throughout conversations, only 4.7% of conversations show meaningful decline, and an apparent decrease at later turn positions is explained by survivorship bias. Instead, the dominant pattern is not erosion but absence: 80.8% of conversations exhibit zero or low autonomy throughout. When autonomy engagement occurs, it is self-reinforcing (persistence doubles from 40% to 85% over consecutive turns). These findings reframe the problem: the risk is not that LLMs progressively degrade users' observable autonomy, but that most users never engage in independent reasoning during the interaction. 

---
# KR A1: Design and Preliminary Evaluation of a Humanoid-Robot Interaction Prototype Integrating Personalized Voice Synthesis, Contextual Dialogue, and Emotion-Aware Response Generation 

**Authors**: Mahek Aanthu, Om Aanthu, Aaryan AK Aanthu  

**Link**: [PDF](https://arxiv.org/pdf/2611.09900)  

**Abstract**: This paper presents the design and preliminary evaluation of KR A1, a humanoid-robot interaction prototype that integrates personalized voice synthesis, contextual dialogue processing, and emotion-aware response generation for potential mental-healthcare applications. The system architecture combines a locally hosted large language model (Qwen2.5-3B-Instruct, served via Ollama) for natural-language understanding and response generation, OpenVoice v2 for zero-shot speaker-adaptive speech synthesis, and a two-tier emotion-processing pipeline comprising a keyword heuristic and LLM-based inference. Face recognition (InsightFace) and vector memory retrieval (FAISS with sentence-transformer embeddings) enable user-personalized interaction. The experimental evaluation reported here is limited to the voice-cloning component: cloned voices were evaluated against reference recordings using eight acoustic similarity measures - mel-frequency cepstral coefficients, pitch, spectral centroid, spectral contrast, spectral rolloff, zero-crossing rate, formants, and speaker-embedding cosine similarity - supplemented by a same-speaker versus cross-speaker baseline comparison. All tests were conducted on a small, informal development sample; no statistical significance testing against a held-out baseline was performed. The emotion-recognition accuracy, end-to-end response latency, and therapeutic or clinical impact of the integrated system have not yet been formally evaluated and are identified as directions for future work. All testing was informal and self-conducted by the development team; no patients or external human participants were involved, and the system has not been evaluated or deployed in a clinical setting. 

---
# Generative AI and Oral Proficiency: How ChatGPT Is Transforming English Language Speaking Practice 

**Authors**: Mohammad Mousazadeh  

**Link**: [PDF](https://arxiv.org/pdf/2611.09898)  

**Abstract**: The integration of generative artificial intelligence (GenAI) into language learning contexts presents potential for developing oral proficiency, though empirical evidence regarding its efficacy remains limited. This quasi-experimental study investigated the impact of ChatGPT-mediated speaking practice on English oral proficiency among Iranian learners. A total of 380 participants (190 female, 190 male; M(age) = 21.4, SD = 4.7) from Tehran and Isfahan were assigned to experimental (ChatGPT practice; n = 190) or control (human pair work; n = 190) groups. Both groups completed an 8-week intervention with three 30-minute weekly sessions. Oral proficiency was assessed using a Common European Framework of Reference for Languages (CEFR)-aligned Oral Proficiency Interview (OPI) scored on fluency, pronunciation, lexical resource, grammatical accuracy, and interactive communication (total score: 0-20). Multilevel growth modeling revealed a significant time-by-group interaction, F(1, 376) = 48.73, p < .001, partial eta-squared = .115. The experimental group demonstrated greater gains (M change = 2.90, SD = 1.60) than controls (M change = 0.75, SD = 1.80), d = 1.20, 95% CI [0.95, 1.45]. Gender moderated this effect, with females in the experimental group showing the largest improvements (M change = 2.50). Mediation analysis confirmed that increased integrative motivation partially mediated proficiency gains (indirect effect = 0.28, 95% CI [0.20, 0.36]). Findings indicate that structured ChatGPT practice yields modest but meaningful enhancements in oral proficiency, particularly when leveraging affective factors. Implications for AI-augmented speaking pedagogy in resource-constrained contexts are discussed. 

---
# Process then Retrieve: Investigating the Origins of the Modality Gap in Vision-Language Models 

**Authors**: Arpita Shanbhag, Julia Tran, Dhruv Mandala, Ayda Sultan  

**Link**: [PDF](https://arxiv.org/pdf/2611.09897)  

**Abstract**: Understanding the origins of the modality gap in vision language models remains a significant challenge. This work investigates whether differences in information density between modality specific tokens at the embedding layer contribute to this imbalance. We show that text tokens enter the model with much lower semantic density than visual tokens, driving early layers to focus on processing text representations before combining them with vision. Through mechanistic analysis, we demonstrate that replacing vision token positions with their semantically closest text token counterparts elicits increased attention to those positions in early layers. These findings provide a foundation for addressing the modality gap and provide insight to improve multimodal reasoning in VLM architectures. 

---
# Improving Large Audio-Language Model Embedding Baselines for Instruction-Aware Speech Retrieval 

**Authors**: Chen-An Li, Cheng-Han Chiang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.09122)  

**Abstract**: Speech retrieval finds relevant spoken documents given a spoken query. The spoken query alone may not reveal the retrieval goal, such as finding an answer or matching speaker identity. Instruction-aware speech retrieval expresses this goal through a natural-language text instruction. Large audio-language models (LALMs) effectively process speech and text, and our analysis shows that different retrieval requirements favor hidden states from different audio-encoder layers. We introduce IRIS-Embed, which adapts LALMs into embedding models through contrastive training and instruction-aware audio layer weighting. We construct about 900K training samples from real data and synthetic data, with instructions expressing requirements for one or more speech attributes. On INSPIRE, an instruction-aware speech retrieval benchmark, IRIS-Embed consistently outperforms the corresponding LALM baselines. The best model achieves 62.92% macro-average Recall@50, surpassing the strongest baseline at 45.15%. Controlled comparisons demonstrate instruction awareness and the benefits of audio layer weighting. On MAEB, a massive audio embedding benchmark, macro-average gains of 9.4-19.8 points over the corresponding baselines show generalization beyond speech retrieval. 

---
# ViCLIP-OT: The First Foundation Vision-Language Model for Vietnamese Image-Text Retrieval with Optimal Transport 

**Authors**: Quoc-Khang Tran, Minh-Thien Nguyen, Nguyen-Khang Pham  

**Link**: [PDF](https://arxiv.org/pdf/2602.22678)  

**Abstract**: Image-text retrieval has become a fundamental component in intelligent multimedia systems; however, most existing vision-language models are optimized for highresource languages and remain suboptimal for low-resource settings such as Vietnamese. This work introduces ViCLIP-OT, a foundation vision-language model specifically designed for Vietnamese image-text retrieval. The proposed framework integrates CLIP-style contrastive learning with a Similarity-Graph Regularized Optimal Transport (SIGROT) loss to enhance global cross-modal consistency and mitigate modality gap issues. Extensive experiments on three Vietnamese benchmarks (UITOpenViIC, KTVIC, and Crossmodal-3600) demonstrate that ViCLIP-OT consistently outperforms CLIP and SigLIP baselines in both in-domain and zero-shot settings. On UIT-OpenViIC, the model achieves an average Recall@K of 67.34%, improving upon CLIP by 5.75 percentage points. In zero-shot evaluation on Crossmodal-3600, ViCLIPOT surpasses CLIP by 11.72 percentage points. Embedding-space analysis further confirms improved alignment and reduced modality gap. The results indicate that integrating SIGROT provides an effective and scalable strategy for cross-modal retrieval in low-resource languages, offering practical implications for intelligent multimedia retrieval systems in Vietnamese and other underrepresented linguistic contexts. 

---
