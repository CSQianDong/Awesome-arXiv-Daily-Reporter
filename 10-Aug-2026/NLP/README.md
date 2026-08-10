# CreativeInstruct: Scalably Teaching LLMs to Balance Quality, Creativity, and Diversity 

**Authors**: Ananya Sahu, Mohit Bansal, Elias Stengel-Eskin  

**Link**: [PDF](https://arxiv.org/pdf/2608.07460)  

**Abstract**: While post-training improves the capabilities of large language models (LLMs), it generally lowers their output diversity and creativity, negatively impacting tasks that explicitly require creativity (e.g., story generation) as well as those that require it implicitly, e.g., reinforcement learning (RL). We instead propose CreativeInstruct, a scalable instruction-tuning method that teaches LLMs to balance creative, base-model-like generations with the quality of post-trained models, by learning to inject special [StartCreativity] spans that bias generation toward creativity. Furthermore, we introduce a structural diversity metric based on graph edit distance, which captures narrative level variation missed by purely lexical and semantic metrics. On narrative generation, CreativeInstruct matches or exceeds the diversity of both multi-model baselines and distilled variants of their outputs, without sacrificing quality or requiring multiple models at inference time. These results are mirrored in our human evaluation, where we find that annotators rate CreativeInstruct generations as more creative than the post-trained LLMs' generations in 70.3% of cases. We also show the benefits of creative models as a substrate for RL: GRPO applied to a CreativeInstruct checkpoint improves by ~4% on AMC and ~5% points on MATH over the same training applied to the post-trained checkpoint. 

---
# CoinRAG: Contextualized Information Nugget KV Cache Reuse for Long-Context RAG 

**Authors**: Gyuwan Kim, Cheoneum Park, Tao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07458)  

**Abstract**: Recent optimization studies on Retrieval-Augmented Generation (RAG) have exploited chunk-level KV cache reuse to avoid processing long retrieved contexts for higher efficiency, while significant information redundancy and noise still remain in the coarse-grained chunks. This paper optimizes the Pareto frontier under low prefill latency constraints while maximizing accuracy by proposing CoinRAG (Contextualized Information Nugget KV Cache Reuse for Long-Context RAG). The name metaphorically reflects our core mechanism: much like assembling small tokens (or "coins") to accumulate a larger value, CoinRAG compositionally reuses offline-computed, fine-grained nugget caches to form a learned contextual representation efficiently in a more semantically relevant but compact manner. Specifically, instead of full-chunk encoding, CoinRAG identifies query-relevant semantic units within retrieved chunks through two-stage retrieval and seamlessly assembles their sliced KV representations with a chunk-level context. Extensive evaluations on LongBench multi-hop question answering tasks demonstrate that CoinRAG significantly reduces operational costs and outperforms the other baselines with a new Pareto frontier and an average 5.3% relative improvement in answer quality (F1) under a standard fast prefill latency budget. 

---
# An Exploratory Evaluation of LLM-Assisted Rewriting of Moderate-Complexity Financial Sentences for DisCoCat-Based Sentiment Analysis 

**Authors**: Brian Llinas, Nikos Chrisochoides  

**Link**: [PDF](https://arxiv.org/pdf/2608.07439)  

**Abstract**: Quantum natural language processing (QNLP) provides a grammar-aware framework for text modeling, and Distributional Compositional Categorical (DisCoCat) is one of its theoretically grounded formulations. Prior work on financial sentiment analysis has identified practical limitations of DisCoCat, including parser sensitivity, high simulation cost, and difficulty handling longer sentences. We study an LLM-assisted preprocessing workflow that uses controlled rewriting to compress, simplify, or decompose moderate-complexity financial sentiment sentences into parser-compatible, circuit-efficient variants while preserving sentiment-bearing meaning. We compare prompting strategies, language models, and filtering configurations with the low-complexity-only DisCoCat baseline of Stein et al. At the circuit level, the strongest compression variants reduce average qubit and gate counts by more than 70 percent relative to the raw moderate-complexity subset. Across repeated training runs, GPT-4.1-mini with Prompt B achieves the highest observed mean accuracy, $0.550 \pm 0.035$, compared with $0.521 \pm 0.050$ for the baseline. Larger training splits do not necessarily improve downstream performance; across evaluated configurations, training-split size has a moderately negative association with accuracy (Pearson $r=-0.446$). These results provide exploratory evidence that LLM-assisted rewriting can make some moderate-complexity inputs usable within the evaluated DisCoCat configuration, while highlighting prompt design, filtering, and circuit-aware preprocessing as considerations for more scalable QNLP-based financial sentiment analysis. 

---
# LitTraceQA: A Benchmark for Multi-Stage Grounding and Verification in Scientific Question Answering 

**Authors**: Xuye Liu, Yimu Wang, Peng Shi, Bo Xue, Xiangrui Ke, Songcheng Cai, Kath Choi, Di Wu, Freda Shi, Krzysztof Czarnecki  

**Link**: [PDF](https://arxiv.org/pdf/2608.07370)  

**Abstract**: Scientific literature is increasingly used as a knowledge source for language models, retrieval-augmented generation systems, and research assistants, but answering research questions from papers requires more than fluent generation. A reliable system must identify the relevant papers, locate the concrete evidence that supports the answer, and produce a response that is faithful to that evidence. We present LitTraceQA, a benchmark for literature-grounded question answering over scientific papers. Given a research question and a metadata pool of papers, a system must return three connected outputs: canonical paper identifiers, supporting evidence locations, and answers in one or more requested formats, including free-form text, multiple-choice answers, and structured tables. LitTraceQA targets evidence types common in scientific reading: tables, figures, text spans, equations or algorithms, and citation contexts. The public development split contains 55 examples, including 26 hidden-source single-paper questions and 29 multi-paper questions, and provides gold papers, evidence annotations, and answers for local validation. We also analyze a larger final annotation collection with 4,978 unique-question records over 4,859 unique gold papers. By evaluating paper retrieval, evidence grounding, and answer accuracy separately, LitTraceQA provides a testbed for scientific QA systems that produce verifiable answers rather than unsupported summaries. 

---
# Geo-Spatial Concept Probing of Large Language Models: Abstraction, Compositionality, and Grounding 

**Authors**: Karim Radouane, Jose G Moreno, Lynda Tamine  

**Link**: [PDF](https://arxiv.org/pdf/2608.07353)  

**Abstract**: Understanding concepts is fundamental to generalization. Despite their impressive performance on a wide range of tasks, Large Language Models (LLMs) still struggle with genuine concept understanding. Prior work has evaluated conceptual understanding in LLMs using natural-language benchmarks or narrowly scoped synthetic tasks, but these settings often conflate multiple skills or lack precise control over the underlying concepts and their properties. To support controlled probing of concepts in LLMs, we design tests on their core properties: abstraction, compositionality, and groundness. We set up a concept-centric benchmark, targeting spatial concepts such as direction, distance, topology, and their compositions, and use question answering tasks serving as a proxy. We conduct extensive experiments across multiple LLM architectures and training regimes to analyze how model scale and design impact conceptual understanding. The results reveal clear limitations in current LLMs and provide insights into the factors shaping their ability to acquire and compose structured concepts. Our findings shed light on how concept-based LLMs can be redesigned for improved information access and knowledge management. The code will be available at this https URL. 

---
# Zero Gap Is Not Restoration: Stratified Per-Question Probability Evaluation and Step-wise Mitigation of Benchmark Contamination 

**Authors**: Ruijie Hou, Yueyang Jiao, Zhao Wang, Yingming Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.07341)  

**Abstract**: Test data from public benchmarks inevitably leaks into pretraining corpora, inflating evaluation scores once memorized. \textbf{Contamination mitigation evaluation} intervenes in the decoding process to suppress memorization and restore a contaminated model's genuine capability, but its prevailing metric, the \textbf{G-AP} (\textbf{G}ap of \textbf{A}ggregate \textbf{P}erformance), is flawed. Discrete correct/incorrect readouts cannot characterize per-question performance, averaging before differencing lets over- and under-suppression cancel out, and uniform per-question weighting invites strategies to push solve probabilities onto the clean model's high-frequency values. We propose \textbf{SA-PPG} (\textbf{S}tratified \textbf{A}ggregate of \textbf{P}er-question \textbf{P}robability \textbf{G}aps): estimate each question's solve probability by sampling, difference it against the clean model per question, and aggregate within groups defined by the clean model's solve probability. Existing mitigation strategies first estimate where contamination lies and then operate on the estimate, so they are only as correct as the estimate. \textbf{RailCap} instead judges contamination during generation: whenever a sample falls back onto the greedy trajectory, the next trajectory token is capped to the runner-up, accumulating suppression until the response distribution becomes sufficiently dispersed. Across multiple contaminated models and benchmarks, SA-PPG reveals that prior strategies' restoration is substantially overestimated, while RailCap attains the lowest SA-PPG. 

---
# Natural Language Processing Psychometrics 

**Authors**: Edoardo Sebastiano De Duro, Emma Franchino, Massimo Stella  

**Link**: [PDF](https://arxiv.org/pdf/2608.07316)  

**Abstract**: Natural Language Processing (NLP) models predicting mental health outcomes rarely specify what they measure: contextual knowledge, emotional content, or syntactic structure. NLP Psychometrics treats psychological prediction from text as a psychometric problem, linking scores to interpretable linguistic evidence and testing beyond the training text format. Nine LLMs, conditioned on controlled personas (cognitive digital shadows), completed psychometric questionnaires with textual explanations per item. We extracted emotional profiles and syntactic-semantic structure via textual forma mentis networks, combined with personality and sociodemographic variables in ablated random forest (RF) regressors, using SHAP to identify which features drove performance and in which direction. Full RF models explained up to 70.8% of variance in life satisfaction (SWLS), 55.7% in depression (PHQ-9), and, for DASS-21, 68.5% depression, 76.0% anxiety, 72.4% stress. Sociodemographics alone explained no meaningful variance in depression, anxiety, or stress, but did so for life satisfaction, where emotion features and income were the strongest predictors; neuroticism and network topology instead dominated depression and anxiety, reversing direction between them. Without retraining, RF models separated diaries from low- and high-score personas ($r$ up to 0.91) and, using only network/emotion features, classified clinical from control participants in real transcripts with up to 68% accuracy. These results show the promise and limits of synthetic data: LLM personas can expose model biases, recover patterns consistent with clinical rumination, and support psychometric prediction from human text without a matched questionnaire, but cannot substitute for human validation. NLP Psychometrics makes these distinctions explicit, measurable, and testable through interpretable AI and network/emotional features. 

---
# Grammar Engineering Meets LLMs: Development of Cantonese and Irish ParGram Treebanks 

**Authors**: Chit-Fung Lam, Elaine Uí Dhonnchadha  

**Link**: [PDF](https://arxiv.org/pdf/2608.07283)  

**Abstract**: Grammar engineering requires expertise in linguistic formalism and computational implementation, especially in parallel grammar projects that balance cross-linguistic consistency with language-specific properties. This paper presents the development of Cantonese and Irish treebanks within the Parallel Grammar (ParGram) Project, where linguistic parallelism is maintained at an abstract functional level. We also investigate the methodological potential and limitations of using multilingual LLMs to support grammar engineering, focusing on Cantonese-Irish translation and the generation of formal syntactic structures using OpenAI's gpt-oss-120b model. The results show that translation performance was generally unsatisfactory and unaffected by prompt language. For syntactic structure generation, the model produced some structurally meaningful outputs, but performed poorly on tasks requiring cross-linguistic abstraction. Nonetheless, LLM-generated outputs may still offer some reference value by suggesting alternative analyses and (partially) capturing predicate-argument relations. Overall, our findings highlight both the potential and limitations of using LLMs in collaborative grammar engineering, while underscoring the continued importance of expert-driven analysis and verification. 

---
# Gaze Behavior in Visual World Experiments Can be Modeled With Off-the-shelf Language-Vision Encoders 

**Authors**: Rahul Murali Shankar, Titus von der Malsburg, Sebastian Padó  

**Link**: [PDF](https://arxiv.org/pdf/2608.07282)  

**Abstract**: The recent advances in neural language models have also spurred much work in computational psycholinguistics, asking whether neural LMs are also promising models of human language processing. However, work has been overwhelmingly focused on the unimodal case of written or spoken language. In contrast, multimodal experimental paradigms, like visual world studies that present participants with both visual and linguistic input simultaneously, have been neglected. In this paper, we present a novel approach that predicts gaze behavior in visual world studies. It does so by combining a simple multi-modal bi-encoder model of the CLIP family with a bimodal attribution method. We demonstrate the ability of this approach to robustly replicate the results of a seminal English visual world study which shows hu- man predictive processing. Remarkably, it does so without a generative architecture and without the need for fine-tuning, despite not being trained for this task. 

---
# Why Knowing Both Hops Is Not Enough: Understanding Two-Hop Generalization in Language Models 

**Authors**: Zili Zhang, Yilin Wang, Heng Wang, Herun Wan, Minnan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.07261)  

**Abstract**: Large language models (LLMs) can solve complex multi-hop problems yet exhibit puzzling failures on simple two-hop queries: although a model may correctly store each individual hop, it often fails to combine them. To understand the internal mechanisms of this phenomenon, we train transformers from scratch in a controlled symbolic environment. Our experiments reveal a pattern in two-hop generalization: models generalize reliably when the second hop follows the training distribution, but always fail when it deviates.
Through mechanistic analysis, we provide a complete explanation for these distinct generalization behaviors: in settings where models generalize successfully, performance is driven by the emergence of consistent intermediate representations for the same entities across contexts, whereas failures on settings where the second hop is out-of-distribution arise from a mismatch across layers: lower layers correctly construct these intermediate representations, but upper layers, while trained on corresponding atomic facts, primarily learn to map them to outputs rather than to reason over them.
Driven by this insight, we propose a recurrent-style training strategy, which enables transformers to reuse their reasoning circuitry across input forms and substantially improves generalization on out-of-distribution two-hop queries. 

---
# Stoicheia: Character-Level Masked Diffusion for Ancient Greek Textual Restoration, Parsing, and Metrical Scansion 

**Authors**: Eric Cullhed, Albin Thörn Cleland  

**Link**: [PDF](https://arxiv.org/pdf/2608.07249)  

**Abstract**: We introduce Stoicheia, a 405M-parameter character-level masked-diffusion encoder for Ancient Greek whose input factors into five aligned, independently maskable planes: letters, word and sentence boundaries, diacritics, capitalization, and punctuation. A single backbone can therefore restore lacunae, re-segment, accentuate, and punctuate unspaced text without task-specific retokenization. We pretrain it on an open, revision-pinned corpus of 380M words and release eleven checkpoints: ten rotated, decontaminated folds, guaranteeing that for any given literary passage at least one released model has never seen its text, and one with no exposure to documentary texts. Three experiments - reconstruction of damaged inscriptions and papyri, morphosyntactic tagging and dependency parsing, and macronization with metrical scansion - each carry a matched random-initialization control, isolating what character-level diffusion pretraining contributes: 5.6 CER points on inscription reconstruction, 12.9 LAS on parsing, and 6.0 points of balanced accuracy on macronization. On Ithaca's own test split, with identical frozen samples and strict scoring, Stoicheia reduces character error relative to both prior state-of-the-art systems, from 24.6 (Ithaca) and 23.5 (its 2025 Aeneas-framework successor) to 15.5, and raises top-1 accuracy from 63.0 and 64.0 to 74.5. 

---
# Skaling: Chinchilla's Exponents Meet Kaplan's Coupling 

**Authors**: Mathurin Videau, Badr Youbi-Idrissi, David Lopez-Paz, Kartik Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2608.07222)  

**Abstract**: Neural scaling laws are foundational for language model development, yet standard formulations systematically under- and overestimate loss at data-scarce and overtraining extremes. This failure originates in the underlying assumption that model size and training data impact the loss independently. To address this, we introduce the Skaling law, a generalized functional form that couples model capacity and data through a single interaction exponent. This simple extension reduces the Mean Absolute Percentage Error (MAPE) by 1.5-3x across both interpolation and extrapolation regimes. When paired with a sparse grid strategy restricted to low-compute regimes, the Skaling law achieves accurate full-grid extrapolation using approximately 10x less compute than uniform sweeps. By enabling reliable performance prediction from small-scale experiments, the Skaling law provides a more robust and resource-efficient framework for allocating compute budgets in next-generation model training. 

---
# From Test-Time Scaling to Reusable Memory: Measuring Crystallization in Text-to-SQL 

**Authors**: Jiaqian Wang, Yutao Qi, Wenjin Hou, Yuanxi Che, Muning Wen  

**Link**: [PDF](https://arxiv.org/pdf/2608.07213)  

**Abstract**: Test-time scaling can correct difficult text-to-SQL queries, but the extra computation is normally discarded after each answer. Systems increasingly retain verified repair episodes, yet evaluations still report one end-to-end score. It cannot distinguish replay on recurring questions from help on unseen questions, or identify the responsible memory choice. We call measuring this future value the crystallization problem. Our controlled evaluation holds the single-shot solver fixed and varies one memory choice at a time. We separately measure replay, cross-question retention, and held-out same-database transfer. On BIRD, storing verified corrected queries improves held-out first-attempt accuracy by 4.34 percentage points. This gain captures 44.4% of the accuracy headroom provided by on-demand repair on the same questions. Controlled interventions identify database-specific content as the main operating ingredient. Reliable verification and broader retrieval coverage yield supported gains; richer formats and elaborate retrievers do not. Open-source code, evaluation artifacts, and reproduction instructions are available at this https URL. 

---
# Measuring Concept Content in Text from LLM Activations: ESG Evidence from Concept Vectors and Linear Probes 

**Authors**: Luc Hazenoot, Zhaochun Ren, Amirhossein Zohrehvand  

**Link**: [PDF](https://arxiv.org/pdf/2608.07208)  

**Abstract**: Existing measures of how much a text is about a concept read the surface of the text: dictionary word shares, topic proportions, embedding similarities. They score the words a text uses, not the judgment a reader forms about it. Recent work has shown that a gap exists in what Large Language Models (LLMs) know internally versus what they express in their response. This paper asks whether that internal knowledge, read by monitoring the activations of frozen, out-of-the-box LLMs, can stand in for task-specific fine-tuning when measuring concept content, and which extraction method reads it best. We extract such measures via the Recursive Feature Machine (RFM) algorithm and via linear probing, and compare these against an embedding baseline, surface baselines, and the same model's own answer to the question. We demonstrate the approach on financial text, a domain studied extensively and served by established annotated resources, using a human-annotated Environmental, Social and Governance (ESG) dataset. The best linear probe comes within 0.6 percentage points of a fine-tuned domain classifier's accuracy without any task-specific fine-tuning, and outscores the same model's own answer to the question in eleven of twelve comparisons, so the activations carry concept content the response does not report. The simple probe consistently beats the RFM concept vectors, which in turn provide what classification alone does not: a continuous score intended to reflect how strongly a concept is present in a text, whose validation awaits graded labels. 

---
# HNR-DAC: Hard-Negative Reranking and Distribution-Aligned Classification for Scientific Claim Verification 

**Authors**: Zhenchao Wang, Xin Chen, Luoxi Zhang, Min Yang, Shiwen Ni  

**Link**: [PDF](https://arxiv.org/pdf/2608.07204)  

**Abstract**: Scientific claim verification over a cited paper requires predicting the claim--paper relation and identifying the paragraphs that justify that prediction. This setting poses two linked challenges: within-paper distractors often resemble genuine evidence, while a classifier trained on gold evidence must operate on retrieved evidence at inference. We present HNR-DAC, a two-stage framework that trains each stage on the cases it will actually encounter. Hard-Negative Reranking (HNR) quantifies evidence confusability using a base reranker's scores on non-gold paragraphs and contrasts gold evidence against the most confusable candidates. Distribution-Aligned Classification (DAC) trains on the Top-1 paragraph produced by the same frozen HNR used to construct inference inputs, while HNR's Top-3 paragraph identifiers provide the evidence output. On the NLPCC 2026 Task 10 Track 2, the final configuration obtains 97.21% Hit@3, 95.79% Macro-F1, 94.47% Joint@3, and an average score of 95.13%. The corresponding submission ranks third on the official Track 2 leaderboard while achieving the highest overall Macro-F1 of 93.05%, alongside 70.16% Joint@3 and an average score of 81.61%. 

---
# An Agentic Hybrid Top-Down and Bottom-Up Approach to Knowledge Graph Generation 

**Authors**: Emma Jouffroy, Warren Jouanneau, Marc Palyart  

**Link**: [PDF](https://arxiv.org/pdf/2608.07023)  

**Abstract**: Organizing thousands of unstandardized, multilingual expertise declarations is a persistent challenge for Human Resources (HR) platforms, directly impacting downstream tasks like accurate talent matching. To address this, we propose a hybrid knowledge graph generation pipeline that grounds a Large Language Model (LLM) in the Wikidata multilingual Knowledge Graph (KG) while employing an agentic reflexion pattern to synthesize emerging concepts and their associated metadata. Unlike rigid top-down methods or fragmented bottom-up approaches, our system anchors recognized concepts to stable Knowledge Graph entities while dynamically creating new nodes and relational metadata for unrecognized skills. Executed across five stages, entity reconciliation, multilingual canonicalization, active curation, deduplication, and the iterative recovery of unmapped concepts, the system autonomously adapts to rapidly evolving, noisy skill mentions across five European languages. Ultimately, this pipeline provides a highly scalable, explicable, and self-healing framework for generating a comprehensive skills knowledge graph, from which a structured taxonomy is derived, using unstructured, noisy text. 

---
# Does More Retrieved Evidence Help Visual Retrieval-Augmented Generation with Diffusion Language Models? 

**Authors**: Jiankun Wang, Yisen Gao, Ziwei Zhang, Xingcheng Fu, Jiaxin Bai, Chen Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.07006)  

**Abstract**: Visual retrieval-augmented generation (RAG) commonly expands the retrieved evidence set to improve answer-page coverage, implicitly assuming that all available evidence should be passed to the generator. We show that this assumption does not hold for diffusion language models (DLMs): retrieving more pages increases answer-page recall, whereas unconditionally passing all retrieved pages to the generator often reduces answer accuracy, primarily because of semantic conflict. A latent-source analysis explains this mismatch through source-coherence loss in parallel denoising, where position-wise proposals can combine incompatible visual sources into unsupported answers. We further find that such interference is already visible in the first-step answer-block distribution, making it possible to assess evidence before decoding. To preserve retrieval coverage while limiting harmful visual exposure, we propose the Entropy-Based Candidate Filter (ECF), a training-free evidence-admission framework. To reduce irrelevant content within individual candidates, ECF constructs multi-granularity evidence units; to identify beneficial additional evidence, it uses blank-controlled block confidence and retrieval rank to determine whether and which candidate should enter the final context. Across three multimodal DLMs and five visual QA benchmarks, ECF improves answer accuracy by 2.62 percentage points on average over the strongest fixed top-$k$ input and, with LLaDA2.0-Uni, by 2.37 percentage points on average over the best competing training-free result for each dataset. These results show that broader retrieval benefits visual DLM-RAG through selective evidence admission rather than unconditional evidence expansion. Code is publicly available at this https URL. 

---
# GPTKB 2.0: Browsing, Querying, and Auditing a Disambiguated LLM-Derived Knowledge Base 

**Authors**: Yujia Hu, Tuan-Phong Nguyen, Simon Razniewski  

**Link**: [PDF](https://arxiv.org/pdf/2608.06992)  

**Abstract**: We present a web demo for exploring a large-scale disambiguated knowledge base (KB) materialized from a large language model (LLM). GPTKB 2.0 contains 38.4M triples over 1.6M canonical entities, together with 207.6K consolidated relations and 66K consolidated classes. Unlike prior LLM-derived knowledge bases that largely identify entities by surface strings, GPTKB 2.0 performs context-guided disambiguation during recursive KB construction, separating homonyms and merging synonymous mentions as facts are elicited. The demo makes this process inspectable: users can browse entities, follow links across the KB, and audit the provenance of individual facts, including surface forms, candidate matches, source triples, and disambiguation decisions. The interface further supports structured SPARQL queries, natural-language questions translated to SPARQL, and entity linking from user-provided text to canonical GPTKB 2.0 entries. GPTKB 2.0 is available at this https URL, with the full KB downloadable for offline use. 

---
# Confirming Our Biases? Evaluating the Capabilities, Risks, and Societal Impact of Large Language Models 

**Authors**: Mudar Adas, Polina Tsvilodub, Michael Franke, Martin V. Butz  

**Link**: [PDF](https://arxiv.org/pdf/2608.06977)  

**Abstract**: It is well established that large language models (LLMs) are sensitive to prompt framing, reflecting patterns in their training data or prior prompts. In this study, we investigate the extent to which LLMs reinforce users biases expressed in the prompts and examine the boundary between implicit framing effects and explicit prompt manipulation. Specifically, we evaluate how susceptible LLMs are to direct and suggestive prompts that encourage models to support or challenge particular positions.
We evaluate six LLMs using 160 distinct prompts spanning ten topics across opinion-based and factual domains. The prompts systematically vary in prompting strategy, support versus challenge instructions, prompt polarity, users' expressed beliefs, and topic domain, spanning both opinion-based and factual questions. Our results show that LLMs systematically adapt their responses to align with prompt framing, even in factual contexts. This suggests that prompt framing can outweigh factual consistency in model responses. Overall, our findings delineate the extent and boundaries of LLM manipulability. Furthermore, the results imply that LLMs can reinforce subtle user biases and are susceptible to explicit prompt manipulation even in domains where responses should remain factually stable. 

---
# PHASE-Tree: Modeling Character-State Evolution in Long-Horizon Role-Playing Dialogue 

**Authors**: Bo Tang, Jianan Yang, Junyi Zhu, Yiquan Wu, Rui Zhao, Zhengyu Yang, Yang Zhang, Feiyu Xiong, Zhiyu Li, Jiajun Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.06975)  

**Abstract**: Long-horizon role-playing demands that characters remain recognizable as they evolve with the narrative. Yet existing work falls short on two fronts: representations are typically static profiles that cannot be updated locally without destabilizing unchanged traits, and benchmarks mainly test persona preservation and memory recall rather than whether a model speaks from a character's currently evolved state. We address both. PHASE-Tree is a multi-timescale character-state tree with an immutable identity root and mutable persona, session, and moment layers, making each mutable field an addressable target for localized within- and cross-episode updates. It conditions generation through explicit textual provision or implicit parametric adaptation. To measure evolved-state generation, we introduce LongEvoRoleBench, which pairs four long-dialogue corpora for cross-episode evolution with four short-dialogue corpora as within-scene state-tracking checks, under a unified next-utterance protocol. On the long-dialogue core, textual PHASE-Tree ranks first in 11 of 12 dataset-metric cells against internal variants and all 12 cells against external textual baselines, improving character-level, semantic, and embedding scores by 19.7%, 12.4%, and 15.1% respectively. In a blinded 200-response study, human ratings correlate with the GPT-4.1 judge (Pearson r= 0.65); on descriptive n= 10 PT and NR prompt subsets, the Overall difference is +0.20. The long-dialogue Sem advantage persists across LLM judges and generation backbones. 

---
# Can Language Models Imagine Without Seeing? Ekphrasis: Measuring Visual Creative Ideation in Text-Only LLMs 

**Authors**: Hongyu Luo, He Wang, Huihao Jing, Hong Ting Tsang, Yuxuan Liu, Wuganjing Song, Yauwai Yim, Chunyang Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.06967)  

**Abstract**: Current evaluations do not isolate whether text-only language models can originate visual concepts before image generation. Fluent visual prose can hide visual-plan failures: an answer may appear creative while repeating familiar visual clichés or failing to specify a renderable scene. We define Visual Creative Ideation (VCI) as the ability to produce textual visual plans that are useful, expressive, and population-novel, and introduce Ekphrasis, a 400-task benchmark spanning Abstraction, Combination, Transformation, and Adaptation. Ekphrasis scores anonymized pairwise comparisons with dimension-specific checklists, aggregates preferences with Bradley-Terry models, and uses Typed Idea Graphs to convert task-specific population clichés into novelty references. Across 14 language models, VCI separates usefulness, expressiveness, and novelty rather than reducing to fluency: strong models achieve similar overall scores through different profiles, and useful plans can remain visually clichéd. A cross-modal grounding study further shows that text-level VCI ordering largely survives faithful rendering and blind image-level preference judgment, supporting Ekphrasis as a measure of visual ideation beyond prose quality. 

---
# Explicit, Not Longer: What Makes Epistemic Stance Survive Memory Compression 

**Authors**: Alex Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2608.06953)  

**Abstract**: Agent memory systems compress what they store, and compression is built to drop qualifiers, so a claim's epistemic standing tends not to survive being written to memory. We ask what governs whether it does. Matched notes carry the identical claim and identical stance and differ only in where that stance sits; one model compresses both under the same budget among the same filler notes, and a blind reader that never sees the condition scores the result. Across 60 claims in seven registers, writing the stance as a labelled field rather than a bracketed aside raises retention by about 15 points on two models (37 claims to 2 on one, 30 to 8 on the other; permutation p=0.00005), and a pre-registered replication on Haiku, its prediction and decision rule committed before the run, gives +15.6 points, 38 claims to 1. Ablating the format on both models gives the same net effect from different parts: labels help on both (+9.7 and +12.8) and length helps on neither, but wording the stance as a full sentence is the largest component on one model (+12.5) and worth nothing on the other (+0.6). Either model alone would have licensed a confident and different mechanism, so we claim only the intersection: make the stance explicit, not merely longer, and expect the best way of being explicit to depend on the model. A deterministic readout with no model reproduces the two-cell direction and five of seven ablation contrasts, but not length or labels, which we therefore do not claim on one instrument. Fifty hand labels (kappa=0.75) agree on direction; we print their seven disagreements in full. We also report nine withdrawn claims, three of them former title claims of this paper. 

---
# Ask-E: An Environment for Calibrated Question Generation 

**Authors**: Sarah Pratt, Jae Sung Park, Scott Geng, Ali Farhadi  

**Link**: [PDF](https://arxiv.org/pdf/2608.06933)  

**Abstract**: Today, we improve models by training and evaluating them on problems at the frontier of their abilities. Creating such problems is itself a demanding task, requiring the ability to probe model limits and generalize beyond existing question distributions. It also means placing problems at a precise difficulty level, which requires understanding what it takes to solve them. In short, generating problems calibrated to a model's current frontier demands capability beyond it, an increasingly burdensome constraint as models improve. Our key insight is that we can leverage this constraint to our advantage: a model that can generate problems consistently calibrated to a given frontier must possess capability beyond it. Accordingly, we present Ask-E, an environment that benchmarks and trains models on their ability to write questions at a given skill level, rather than answer them. Concretely, we define target skill levels as ranges bounded by the capabilities of two existing language models. A generated question is successfully calibrated if exactly one of the two models can solve it, placing it precisely within the target range and differentiating the capabilities of these models. Ask-E serves both as a benchmark and a training environment, where models generate problems calibrated to a variety of skill levels. We find that even frontier models achieve below 50% calibration on the benchmark, leaving significant headroom to measure future progress. We also show that training on this environment leads to improvements across a number of downstream math benchmarks even with no new math data, no interaction with stronger models, and no correctness-based reward. 

---
# Calibrating WEAT Against Anisotropy: ZCA Whitening as a Geometric Pre-Processing Step for Embedding Association Tests 

**Authors**: Seitaro Ono, Senna Ross, Jun Saiki  

**Link**: [PDF](https://arxiv.org/pdf/2608.06908)  

**Abstract**: We propose Zero-phase Component Analysis (ZCA) whitening as a geometric pre-processing step for the Word Embedding Association Test (WEAT). WEAT is a bias measurement method widely used in both computational social science and AI fairness research. It relies on cosine similarity as a measure of semantic association, which assumes that the embedding space is approximately isotropic. However, prior work has reported that many widely used language models do not satisfy this assumption, raising concerns about the reliability of bias measurements. ZCA whitening transforms the covariance of the embedding space into the identity matrix while minimizing perturbation to the original vectors. This transformation restores the isotropy condition on which WEAT relies. We evaluate our approach on ten standard WEAT test suites and seven models spanning three architectural families, yielding 70 model-task combinations. The results show that ZCA whitening substantially reduces the anisotropy of the embedding spaces across all models. Particularly for highly anisotropic models, we further observe improvements on standard semantic similarity benchmarks, indicating that the calibrated space better captures semantic associations. After calibration, over 30% of WEAT results change significance status, and effect sizes shift in both directions depending on bias category. These shifts suggest that uncalibrated measurements may both overestimate and underestimate the associations encoded in the embedding space. These findings indicate that previously reported bias measurements in anisotropic embedding spaces should be interpreted with caution and may benefit from re-evaluation with calibrated methods. Our approach contributes to restoring the measurement foundation of WEAT across both computational social science and AI fairness research. 

---
# Georeferencing Non-Gazetteered Place Names using Biological Specimen Records 

**Authors**: Aneesha Fernando, Surangika Ranathunga, Kristin Stock, Raj Prasanna, Christopher B. Jones  

**Link**: [PDF](https://arxiv.org/pdf/2608.06884)  

**Abstract**: Biological specimen records collected by natural history institutions constitute a rich source of temporal geographic knowledge, capturing biodiversity information about regional landscapes as they were recorded at different times. Using digitised data from the Allan Herbarium (New Zealand), this study identifies place names in these specimen locality descriptions that are absent from current gazetteers; we refer to these as non-gazetteer place names (NGPs). These place names are typically historical, vernacular, or colloquial and were used as landmarks to describe a specimen's location at the time of collection. We then investigate the problem of georeferencing the NGPs using only the limited information available in the specimen records. To resolve this, we leverage repeated occurrences of the same place name across specimen records with different specimen locations and spatial relation terms, extracting and inverting these relations to derive constraints on NGP locations. This approach is instantiated within deterministic, probabilistic, and LLM-based methods, enabling a comparative analysis of their strengths and limitations for text-based spatial inference. On a pseudo-NGP benchmark, probabilistic inference achieves the highest accuracy (median error 1.43 km; A@1 km 36%), while the LLM yields competitive but less precise estimates (median error 1.80 km; A@1 km 31%), indicating that, despite advances in LLMs, traditional modelling remains advantageous when high spatial precision is required. 

---
# LLMRouter: Unified Infrastructure for Developing, Evaluating, and Deploying LLM Routers 

**Authors**: Tao Feng, Fangxu Yu, Haozhen Zhang, Zhongjie Dai, Liangqi Yuan, Zijie Lei, Weizhi Zhang, Kunlun Zhu, Haodong Yue, Keyang Xuan, Ge Liu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2608.06867)  

**Abstract**: No single large language model (LLM) is optimal across all queries and budget constraints, making model routing essential for cost-effective deployment. Existing routers adopt diverse formulations and implementations, making fair comparison and extension difficult. We present a unified formulation of LLM routing as a sequential decision process characterized by five components: context encoders, model encoders, scoring functions, decision rules, and learning signals, covering single-turn, multi-turn, and personalized routing. Based on this formulation, we develop an automated pipeline for constructing routing supervision and evaluating routers jointly on response quality and inference cost. The resulting benchmark, xRouteBench, spans generic LLM, memory-augmented, vision, time-series, and personalized routing tasks. We further introduce LLMRouter, an open-source modular infrastructure with more than 16 representative routers. Our empirical study shows that learned routers outperform the strongest fixed-model baseline by 14.6% relatively, lightweight routers become more competitive under tight cost constraints, and user-conditioned routing consistently improves personalization. 

---
# Autonomy-of-Heads: Data-Free Sparse Attention from Frozen Query-Key Geometry 

**Authors**: Yehan Yang, Junyuan Shang, Yang Li, Guanqun Zhao, Shuohuan Wang, Dianhai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.06849)  

**Abstract**: Long-context LLM inference is bottlenecked by quadratic attention computation and growing KV-cache costs. Existing sparse attention and KV-compression methods typically decide which tokens or heads to preserve from runtime attention scores, observation windows, calibration prompts, or learned gates, making head diagnosis input-dependent and costly to deploy. We propose Autonomy-of-Heads (AoH), a data-free method that identifies retrieval and streaming heads from the spectral geometry of query-key projections. AoH defines the kernel attention operator $M_h = W_K^{h\top}W_Q^h$ and uses its effective-rank as a weight-space measure of head function: concentrated spectra indicate a small number of dominant query-key matching directions and are associated with retrieval heads, whereas diffuse spectra indicate the absence of a dominant global matching direction and are associated with streaming heads. We further derive an efficient $d_\text{head}$-dimensional computation that avoids constructing the full $d_\text{model}\times d_\text{model}$ matrix. We conducted extensive experiments across models demonstrating that at 50\% sparsity, AoH retains 96.5\% of Full Attention performance on average while reducing prefill and decode latency by up to 41.4\% and 66.0\%, respectively, and KV-cache memory by 50.0\% at 256K tokens. 

---
# FutureBridge: Token Selection Beyond Local Preference in Collaborative Decoding 

**Authors**: Quanquan Li, Hongbo Zhang, Yihe Chi, Jingyu Li, Xidong Xi, Liuyang Song, Hongzhen Zhang, Yuxiang Huang, Jing Ke, Siyuan Ma, Junyi Lin, Guitao Cao  

**Link**: [PDF](https://arxiv.org/pdf/2608.06819)  

**Abstract**: Token-level collaboration allows a large language model (LLM) to assist a small language model (SLM) when their predictions diverge. Existing methods either use LLM-generated intervention tokens or rank candidates with the LLM's next-token probabilities. Both rely on the LLM's local preference, even though an LLM-selected token may be difficult for the SLM to build on. We present FutureBridge, which ranks joint LLM-SLM token candidates according to how well they support the SLM's subsequent reasoning. During training, an answer-verified LLM trajectory supplies a fixed shared future, and a frozen SLM evaluates every candidate under this common context. The resulting counterfactual scores supervise a lightweight token reranker that observes only the current state and candidate token. At inference, FutureBridge uses the LLM only to expand the candidate pool, selects one token, and returns generation to the SLM without generating or appending a future suffix. Across five mathematical reasoning benchmarks, FutureBridge improves the Qwen3-1.7B SLM's Math Avg. by 35.1% relative to greedy SLM decoding. These results indicate that token selection benefits from modeling whether the receiving SLM can use each candidate to continue reasoning, rather than relying on the LLM's local preference alone. 

---
# Simple-OPD: Demystifying Warm-up for On-policy Distillation 

**Authors**: Tao Liu, Taiqiang Wu, Mao Zheng, Xuan Luo, Runming Yang, Xuewei Yang, Junjie Wang, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.06802)  

**Abstract**: On-policy distillation (OPD) trains a student on its own rollouts with token-level supervision from teacher models, but its effectiveness can depend strongly on the warm-up stage before OPD. In this paper, we demystify warm-up for OPD from both data and training perspectives. For data, we find that effective warm-up relies on teacher-compatible chain-of-thought supervision, and that even incorrect teacher rollouts can provide comparable benefits to correct ones. This suggests that warm-up primarily transfers a teacher-compatible thinking pattern rather than merely correct answers. For training, we show that low-rank adaptation (LoRA) with a near-saturation training duration better balances in-domain adaptation and out-of-distribution generalization than full-parameter SFT. Based on these findings, we propose Simple-OPD, a plug-and-play initialization method that warms up the student on teacher-generated CoT with LoRA before OPD. Experiments across diverse settings demonstrate the effectiveness and robustness of Simple-OPD. 

---
# Multi-Perspective Triad Interaction Graph Neural Network for Cognitive Distortion Detection 

**Authors**: Jun Seo Kim, Hye Hyeon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.06785)  

**Abstract**: Cognitive distortion detection is a key task in computational mental health, yet existing approaches often overlook the psychological structure of distorted thoughts. We propose MTI-GNN (Multi-Perspective Triad Interaction Graph Neural Network), which models Beck's cognitive triad---negative views of the self, world, and future---as complementary perspectives for classification. An LLM decomposes each utterance into the three perspectives, from which perspective-specific similarity graphs are constructed and encoded by a Multi-Perspective GNN. A Triad Interaction module models cross-perspective dependencies through sequential source-conditioned updates and feature-wise gating, while Prototype-Guided Perspective Fusion performs label-conditioned aggregation. Label-expanded supervision incorporates all available distortion annotations during training. We evaluate MTI-GNN on 9,764 samples from four Korean, English, and Chinese datasets spanning ten distortion categories. MTI-GNN significantly outperforms all supervised variants and exceeds eight prompted generative models under zero-shot and few-shot settings. Leave-one-perspective-out ablations show that all three perspectives contribute significantly, while human expert evaluation provides preliminary evidence of their alignment with the intended cognitive dimensions. 

---
# Stockmark-Nemotron-3-Nano-Omni-JapanDocReader: Structured Document Parsing via Capability Injection and Forgetting Control 

**Authors**: Shi Chen, Hayato Aida, Makoto Morinaga, Shohei Tanaka, Kosuke Arima  

**Link**: [PDF](https://arxiv.org/pdf/2608.06758)  

**Abstract**: We present Stockmark-Nemotron-3-Nano-Omni-JapanDocReader, a Japanese document understanding model built from Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16. The central goal of this work is structured document parsing via capability injection and forgetting control: we inject Japanese structured document parsing capability into a reasoning-oriented multimodal model while preserving its document VQA capability as much as possible. We study parsing-centric SFT, which uses only structured document parsing data; mixed SFT, which combines structured document parsing and VQA data; and parsing-centric RL, which optimizes structured parsing with a task-level reward. Our experiments show that parsing-centric SFT substantially improves structured document parsing performance but causes measurable VQA forgetting. Mixed SFT mitigates this forgetting while preserving nearly the same structured parsing performance. Applying DAPO-based parsing-centric RL on top of the mixed SFT checkpoint further improves structured document parsing beyond the SFT ceiling, producing the final released model. The training data is constructed with a data engine consisting of two complementary synthetic streams: a Japanese Document VQA Stream and a programmatic structured document parsing stream. We also discuss reward design and variance-based prompt filtering for continuous structured document parsing rewards, highlighting their importance for making RL effective in long-reasoning structured document parsing tasks. 

---
# Progressive Content Refinement with Decaying Reward Joint LinUCB 

**Authors**: Shion Ishikawa, Pablo Loyola, Young-joo Chung, Yun Ching Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.06750)  

**Abstract**: Iterative refinement has significantly enhanced Large Language Model (LLM) performance; however, existing methods ranging from feedback-based Self-Refine to traditional bandit approaches often rely on static options or overlook the saturation effect. This neglect leads to over-exploitation, where the continuous use of identical prompts or arms results in diminishing rewards over time.
To address this challenge, we propose a novel contextual bandit algorithm that explicitly incorporates reward decay modeling. Utilizing an Expectation-Maximization (EM) algorithm, our method simultaneously estimates both arm-specific and decay parameters. Furthermore, by embedding prompts as arms, we facilitate the joint learning of arm values, distinguishing our approach from the traditional disjoint Linear Upper Confidence Bound (LinUCB) framework.
Experimental results on Sentiment Reversal and GSM8K benchmarks demonstrate that our method achieves significant performance gains over strong baselines. Finally, our ablation study confirms that the integration of reward decay modeling within the bandit framework is crucial for mitigating over-exploitation and optimizing the iterative refinement process. 

---
# Do Audio Language Models Use Paralinguistic Evidence? Counterfactual Audits for Response Evaluation 

**Authors**: Kevin Miller, Arjun Chandra, Venkatesh Saligrama  

**Link**: [PDF](https://arxiv.org/pdf/2608.06718)  

**Abstract**: Audio-language models (ALMs) are increasingly used as judges for speech-to-speech systems, but a judge that receives audio may not actually use paralinguistic evidence. We introduce counterfactual audits for paralinguistic response evaluation. Each audit item holds the transcript fixed while varying affect, prosody, or the timing of an affective shift, forcing a valid judge to track the audio cue rather than lexical content or response style. We evaluate ALM judges using a native one-context judgment protocol and a contrastive recoverability control, then further decompose each item into its constituent perception and response-mapping skills. This yields useful diagnostic states that identify different sources of judge failures. Across Gemini, GPT, and open audio models, we find that contrastive success often overstates native judge reliability, and that similar aggregate accuracies can hide different failure modes. These results suggest that ALM judges should not be evaluated by accuracy alone, instead requiring thorough behavioral audits before deployment. 

---
# TA-RAG: Tone Awareness as a Design Imperative for Retrieval-Augmented Generation 

**Authors**: Yong-Bin Kang, Anthony McCosker  

**Link**: [PDF](https://arxiv.org/pdf/2608.06672)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a robust architecture for grounding large language models (LLMs) in trusted knowledge. However, standard RAG systems exhibit a structural limitation: retrieved documents carry their own communication styles-professional jargon, formal tone, or academic writings-that shape the behavior of a RAG system before any tone instructions are processed, often causing the system to ignore user requests for a specific tone. We term this phenomenon contextual decoupling, in which a system optimises for factual accuracy while remaining decoupled from the social or operational context of the recipient. Building on prior research in public health peer-support communities, we identify three communicative misalignment-linguistic, cognitive, and relational-that can persist even when retrieval is relevant and the generated response is factually accurate. We conceptualise these as failures of communicative transformation, which remain largely invisible to accuracy-centred RAG evaluation metrics. To address this gap, we propose Tone-Aware RAG (TA-RAG), a conceptual architectural framework that positions communicative alignment alongside factual accuracy as a core design objective. TA-RAG operationalises four constraints-stigma-free language, readability alignment, recipient-sensitive adaptation, and empathetic framing-across the retrieval, context construction, generation, and constraint validation phases in the proposed RAG pipeline. We further highlight an evaluation agenda for jointly assessing factual fidelity and communicative alignment, and identify open challenges. We argue that tone awareness should be treated not as an optional refinement, but as a present design imperative for RAG systems operating in socially sensitive and high-stakes contexts. 

---
# The Horizon Gap: Planning, Memory, Execution, Training, and Evaluation for Long-Horizon LLM Agents 

**Authors**: Mingguang Chen, Licheng Wang, Bo Qu  

**Link**: [PDF](https://arxiv.org/pdf/2608.06663)  

**Abstract**: Frontier language models solve reasoning problems in a single forward pass that would have been research contributions years ago, yet fail at multi-hour tasks: losing track of earlier decisions, declaring half-finished work done, or drifting from goals. We call this the horizon gap and survey 1,547 arXiv papers (2024-2026) collected via systematic seed harvest with a disclosed 26.8% bleed filter, extended by targeted supplementation. We disambiguate three routinely conflated properties: long-horizon (task property: required steps), long-context (model property: token capacity), and long-term memory (system property: persistence across steps/sessions). We organize the corpus into six categories tracking a long-horizon task's lifecycle -- planning, memory, execution, training, evaluation, and foundations/safety -- crossed with an axis capturing where horizons are carried (within-context, within-task-beyond-context, or cross-task-persistent). Across all categories, we find the same pattern: outcome-only signals grow uninformative as horizons lengthen, and the field's response -- whether process reward models, credit assignment, or trajectory-level diagnostics -- manufactures denser step-level signals. We treat critical and diagnostic literature as first-class threads throughout, arguing that segregating critique from method would routinely split single papers across chapters. We close by naming open measurement problems: decomposing model versus harness capability, managing correlated bias in process-level signals used for both training and evaluation, and whether long-horizon reliability admits general predictive theory. 

---
# Discovering Conceptual Metaphors Across Topics and Media Types 

**Authors**: Alexandria Leto, Rohan Das, Juan Vásquez, Abram Handler, Maria Leonor Pacheco  

**Link**: [PDF](https://arxiv.org/pdf/2608.06652)  

**Abstract**: Conceptual metaphors guide our thinking and actions by allowing us to reason about more abstract experiences (e.g., paying taxes) in terms of more concrete or embodied experiences (e.g., carrying a physical load) (Lakoff and Johnson, 2011). It follows that different conceptual metaphors can result in different reasoning: framing paying taxes as an investment in a community rather than a physical load leads to a very different outlook on taxation. Identifying the conceptual metaphors guiding a speaker or writer thus helps to reveal their framing of events. Though these metaphors can't be observed directly, groups of linguistic metaphors, metaphorical expressions as they appear in language, serve as evidence for them. Motivated by this, we present an unsupervised method that extracts linguistic metaphors from a corpus and uses a structured clustering approach to form groups corresponding to conceptual metaphors. Using this method, we point to key topical and framing differences in left- vs. right-leaning podcasts. For example, left-leaning podcasts tend to conceptualize media stories as a weapon, while right-leaning sources commonly discuss the economy as a system subject to vertical changes. 

---
# Factorized Hypothesis Search for Evidence-to-Taxonomy Retrieval 

**Authors**: Linhai Ma, Ethan F. Wei, Xueqing Peng, Yan Wang, Lingfei Qian, Víctor Gutiérrez-Basulto  

**Link**: [PDF](https://arxiv.org/pdf/2608.06614)  

**Abstract**: Large-taxonomy retrieval often assumes that the input already expresses the target concept. In many settings, however, the input is indirect evidence, such as a table cell whose meaning depends on its row, column, datatype, and context. We call this mismatch the retrieval readiness gap. Our analysis shows that the current index retrieves the target reliably when its semantics are explicit, while raw evidence often leaves it deep in the ranking. We propose Factorized Hypothesis Search (FHS), which maintains multiple partial interpretations over named semantic dimensions. These hypotheses support structured query rendering, multi-hypothesis retrieval, and dimension-level candidate verification. On both financial taxonomy tagging and CodiEsp clinical coding tasks, FHS achieves the best Recall@1, MRR, and final accuracy among the non-oracle methods. Replacing the factorized hypothesis path with a free-text ensemble causes the largest drop in head-ranking performance, while sequential refinement provides no additional gain over FHS's strong parallel first round. 

---
# Pre-Inference Routing for Cost-Efficient Document Field Extraction 

**Authors**: Sreerekha Rajendran  

**Link**: [PDF](https://arxiv.org/pdf/2608.06607)  

**Abstract**: Most document-extraction systems use a single model for all documents. This is simple but can be costly for easy cases and less effective for difficult ones. We examine whether we can predict a document's difficulty before extraction using inexpensive, document-based signals, and use this to choose between a cheaper and a stronger extractor. We find that routing only helps if two conditions hold: the cheaper model fails often enough to make routing worthwhile, and those failures can be predicted from visible features such as image quality and layout. We turn these into a practical test and apply it to five genres. When both conditions are met, the calibrated router reduces cost by 31-33% on receipts and 77% on degraded ad-buy forms while keeping quality within 0.02 F1 of always choosing the large model. Routing does not help if either condition is missing, as with clean digital invoices or nutrition labels that are already easy to read. A small labeled pilot can predict whether routing will work, and in the two cases where we ran it first, the prediction was correct. A simple bag-of-words router works about as well as engineered features, showing that the main limit is the genre, not the router design; we use interpretable features to help explain which genres can be routed. The router must be retrained for each dataset and does not transfer across datasets, even within the same genre. These results hold for two model pairs with cost differences of 5x and 3x. 

---
# Beyond "AI Language": The case for the idiolectal nature of LLM output 

**Authors**: Karolina Rudnicka, Thomas Stephan Juzek  

**Link**: [PDF](https://arxiv.org/pdf/2608.06589)  

**Abstract**: While large language model outputs are frequently analysed as a collective super variety termed "AI language," this chapter argues that this perspective coexists with distinct, model-specific linguistic signatures akin to human idiolects. We analyse two datasets of LLM-generated texts on societal topics: a 2024 corpus of six models (Improta et al. 2024) and a newly generated 2026 corpus using the same prompts featuring six contemporary models. Our findings, utilising computational descriptors and stylometric principal component analysis reveal a generational shift between the style of the 2024 and 2026 cohorts, while demonstrating that each individual model maintains a unique linguistic profile. This multi-layered interplay is illustrated by contraction frequencies, which vary from over 1,200 to over 30,000 per million words within the same cohort of models (2026). Ultimately, we conclude that treating LLM output as idiolectal in nature provides a valuable framework with potential implications for research on variation and change, LLM-generated text detection, forensic linguistics and usage-based approaches to language. 

---
# TradeVerse: A Longitudinal Benchmark of Political Negotiation in International Trade 

**Authors**: Debodeep Banerjee, Amitangshu Dasgupta  

**Link**: [PDF](https://arxiv.org/pdf/2608.06549)  

**Abstract**: LLMs are increasingly being applied to tasks involving institutional and political texts, but existing benchmarks evaluate them on isolated documents or single tasks. In realpolitik, negotiations are longitudinal data, where participating parties can align or argue over multiple iterations and each turn is an outcome of the previous turns, hence, understanding one turn requires tracking everything before it. We introduce TradeVerse, a benchmark built from the World Trade Organisation (WTO) specific trade concerns, where member states challenge one another and exchange arguments over multiple rounds, sometimes for years. We, in TradeVerse, reconstruct minutes of $1170$ meetings, spanning across 5 groups and $89$ product groups and define three tasks: first, the system has to analyze the longitudinal meeting records and predict the harmonized system codes (HS chapters) of the products under discussion in the particular meeting, second, we examine whether the system, upon analyzing the anonymized content of the meeting, can guess the name of the responding country and third, we ask the system to play the role of the responding country and provide the statement for the very last round. All labels are recovered directly from the proceedings, requiring no manual annotation. Our experiments highlight the challenges these tasks pose for current LLMs. To the best of our knowledge, TradeVerseis the first benchmark to investigate potential of LLMs in understanding longitudinal political trade negotiations. 

---
# Don't `Well, Actually' Me Unless You Know What You're Talking About: Weak Presupposition Verification Degrades General QA Performance 

**Authors**: Shenran Wang, Vered Shwartz, Hila Gonen  

**Link**: [PDF](https://arxiv.org/pdf/2608.06539)  

**Abstract**: False-presupposition QA (FPQA) tests LLMs on their ability to identify false presuppositions in questions and abstain or correct them rather than reinforcing false assumptions. The common approach reduces the task to prompting LLMs to extract presuppositions and fact checking each presupposition. While the performance on dedicated benchmarks keeps improving, evaluation largely focuses on questions with false presuppositions (FPQs) while ignoring the performance on ``normal'' questions (TPQs). Since many benchmarks over-represent FPQs compared to their natural occurrence, the result is that performance on these benchmarks doesn't reflect real-world QA performance. Through extensive experiments across various model families, sizes, and benchmarks, we show that methods that perform better on FPQs tend to perform worse on TPQs. Our analysis reveals this is the result of weak fact checking modules that reject also true presuppositions. We hope our findings will help guide future work toward FPQA methods that generalize well to realistic settings. 

---
# Confidence Estimation for Financial Vision-Language Models in Chart and Document Understanding 

**Authors**: Reza Khanmohammadi, Simerjot Kaur, Charese H. Smiley, Ivan Brugere, Mohammad M. Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2608.06532)  

**Abstract**: LVLMs are increasingly used to read financial charts, tables, and documents, where a single misread figure can move a decision and the most authoritative-looking answer is sometimes one the model produced without reading the exhibit. The operational question is therefore trust, not accuracy: which answers can be acted on, and which escalated to a reviewer. We evaluate seven confidence estimators, three inference-only and four trained internal probes, across five open-weight LVLMs and four conditions from three financial visual question-answering benchmarks, one bilingual; every probe is trained only on natural images and applied to finance without adaptation, so the results measure out-of-distribution transfer. Three findings hold. First, the scarce property is calibration, not ranking: the inference baselines rank correct above incorrect answers competitively but are badly overconfident, calibration error far above what a threshold can tolerate, and only the trained probes produce a thresholdable score. Second, reliability is structured rather than global, along two axes a practitioner can read directly: the best estimator shifts with both model and task, none leading more than eight of twenty (model, condition) cells, and a controlled bilingual contrast exposes an apparent language robustness as a composition artifact that dissolves once models are read one at a time. Third, cast as deferral under an error budget, how much can be safely automated is set first by the model's competence and only narrowed by its confidence, so deferral clears a real share of the easiest condition and almost none of the hardest, near zero at a strict 5% budget. Two trained probes carry the calibration a deferral policy needs, and among them only the grounding-aware one lowers its confidence on answers a model gives without using the figure, separating detected non-grounding from a fluent guess. 

---
# Lost in Interpolation: Why Predictive Feedback Fails in Diffusion Language Models 

**Authors**: Lavanya Nigam, Ishaan Bansal, Aryan Sood, Vidit Aggarwal, Gaurav Kumar Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2608.06529)  

**Abstract**: Soft-masking accelerates the convergence of Masked Diffusion Language Models (MDLMs). Existing formulations build this blend with linear interpolation (LERP) in the raw embedding space, which implicitly treats that space as Euclidean. We analyze the embedding space of MDLMs and find that the mask and predicted-token embeddings maintain a near-constant angle of (\approx 73^\circ) throughout training, while embedding norms remain essentially flat across vocabulary-frequency rank. These indicate a hyperspherical geometry, for which LERP is the wrong interpolation primitive. We introduce Spherical Soft-Masking (S-SM), a drop-in replacement that aggregates the top-(k) predictions with a Fr'echet mean on the hypersphere and blends this mean with the mask direction using spherical linear interpolation (SLERP), then restores the native mask norm. We evaluate S-SM on continued pre-training of a released 169M-parameter MDLM checkpoint across a wide range of inference-time step budgets, SLERP feedback avoids the training degradation that LERP feedback induces and delivers MAUVE gains of up to 2x over the vanilla MDLM baseline and 27.5-56.1% over TopK/LERP at various sampling budgets, alongside consistently lower generative perplexity (16.9-19.6% over the baseline), while leaving output entropy and convergence essentially unchanged. 

---
# GRASP: Reinforcing Language Model Anonymizers with Group Relative Policy Optimization 

**Authors**: Sajjad Ghiasvand, Nader Sehatbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2608.06526)  

**Abstract**: Large language models can infer sensitive personal attributes, such as age, location, and occupation, from ordinary text, turning everyday writing into a privacy risk. Adversarial anonymization defends against this by rewriting a text with a capable language model that also plays the attacker, but it needs a powerful model at inference time and thus sends private text to a third party, the very exposure anonymization should prevent. Recent work distills this behavior into a small on-device model using supervised fine-tuning and direct preference optimization (DPO), but DPO only imitates the teacher's offline choices and never directly optimizes the privacy--utility objective we care about. We introduce \textbf{GRASP} (\textbf{G}roup-\textbf{R}elative \textbf{A}nonymization via \textbf{S}elf-refinement \textbf{P}olicy-optimization), which reinforces the local anonymizer online with Group Relative Policy Optimization. A single small model acts as anonymizer, adversary, and utility judge, trained against a self-generated reward that hides attributes while preserving meaning, with a design that guards against reward hacking. Trained on Llama-3.1-8B, \ours{} improves the privacy--utility trade-off over the DPO-distilled baseline, consistently across three independent LLM judges. Against adversarial anonymization driven by frontier models such as Gemini~2.5~Flash and Claude, it achieves a comparable or better overall trade-off while removing substantially more private information, and it runs entirely on-device at roughly $1\%$ of the GPT-4o teacher's cost. 

---
# Measuring the Cross-Lingual Comprehension Gap: How the language of the evidence shapes what language models understand 

**Authors**: Rafael da Silva, Jeff Eicher  

**Link**: [PDF](https://arxiv.org/pdf/2608.06506)  

**Abstract**: Language models are often evaluated as though capabilities demonstrated in English remain equally available when the same content is presented in other languages. Traditional multilingual benchmarks rarely isolate language while holding content, question, reference answer, model, and evaluation unit constant. We define the Cross-Lingual Comprehension Gap (CLCG) as the reduction in response quality when the same content and question are presented in a target language rather than in English.
Using ParallelQA-18, a professionally human-translated parallel corpus, we evaluate five models from five laboratories on a stratified sample of 150 articles across 18 languages (English reference; Portuguese high-resource baseline; 16 targets spanning Joshi et al. 2020 classes 0-4). A within-item design varies only passage language. The primary estimator contrasts English versus pooled target-language Token-F1 micro-means on higher-complexity open-ended questions, with article-cluster bootstrap intervals.
The primary pooled CLCG is 0.078 (95% CI 0.072-0.084), about a 17% reduction relative to the English score; the equal-language macro summary is 0.077. Net of Portuguese, the macro gap is 0.016 (95% CI 0.013-0.020). Language-level CLCG is negatively associated with Joshi resource class (rho = -0.594, p = 0.015, n = 16). In blinded paired human evaluations, higher-resource responses are preferred in 61.6% of decisive judgments (estimated preference probability 0.655, 95% CI 0.558-0.741).
Capabilities shown in English should not be assumed to transfer equally to other languages; English-centered evaluations may overestimate quality for users of low-resource languages. 

---
# ConstructCIE: A Dataset for Extracting Causal Information from Construction Accident Narratives 

**Authors**: Hung Nguyen, Jaehoon Lee, Namgyun Kim, Kuan-Hao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.06495)  

**Abstract**: Construction accident narratives contain rich causal information, but the evidence is often implicit, long-span, and distributed. We introduce ConstructCIE, a manually annotated dataset for Causal Information Extraction from OSHA construction accident reports. The dataset uses a hierarchical schema for accident types, causal factors, sub-causal factors, and supporting evidence spans. We evaluate supervised sequence taggers and instruction-tuned LLMs in an end-to-end hierarchical extraction setting. Results show that most evaluated models achieve strong accident-type prediction and recover broad causal meaning but remain limited in precise span-level extraction. JHE generally achieves stronger exact and soft matching, while IHE sometimes achieves higher keyword F1. Error distributions vary by extraction strategy, but evidence-selection and span-boundary errors remain common. These findings show that reliable Causal Information Extraction for construction accidents requires stronger domain grounding and more accurate evidence extraction. 

---
# Do AI Personas Grow? Analyzing and Benchmarking Personality Evolution in LLM Agents After Life Events 

**Authors**: Ming Wang, Peidong Wang, Xiaocui Yang, Daling Wang, Shi Feng, Fiona Fui-Hoon Nah, Ee-Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2608.06485)  

**Abstract**: Personality-conditioned LLM agents (PC-Agents) are increasingly used in emotional support, social simulation, and role-playing, motivating the development of lifelong agents that remain coherent over extended interactions. A key component of such coherence is personality evolution: agents should undergo plausible, psychology-grounded changes as they experience life events in different contexts. Although prior work shows that LLM personalities can shift under contextual perturbations, how these shifts vary across traits, events, personas, and models remains poorly understood. We study event-induced personality change after 11 major life events, using the Big Five traits as a psychometric anchor and interpreting the resulting trajectories against longitudinal evidence from human personality psychology. Across four diagnostic axes, PC-Agents exhibit measurable trait shifts at similar rates for event-trait pairs with and without documented human change directions. Even when shifts follow the expected direction, their magnitudes usually fall below human effect-size ranges. Gender and cultural-region prompts show little moderating effect, while persona-level dispersion is compressed three- to four-fold relative to human samples. To enable systematic comparison, we introduce BFI-Adapt, a reusable benchmark for scoring the directional fidelity of event-induced personality change, and use it to rank 14 models. A validation suite shows that the measured shifts exceed no-event retest noise, remain stable under independently paraphrased prompts, exhibit limited and model-dependent convergence with scenario-based behavioral choices, and persist across intervening unrelated dialogue. Together, these checks establish the measured trajectories as robust event-conditioned response patterns. Our results suggest that current PC-Agents simulate the mean of human personality dynamics, but not its shape. 

---
# Recovering Lesion Parameters from Aphasic Picture Naming Error Profiles in Large Language Models 

**Authors**: Yong Yang, Roger Newman-Norlund, Xiang Guan, Saeed Ahmadi, Regan Willis, Nadra Salman, Kalil Warren, Sophie Arheix-Parras, Srihari Nelakuditi, Leonardo Bonilha, Christopher Rorden, Rutvik H. Desai, Julius Fridriksson  

**Link**: [PDF](https://arxiv.org/pdf/2608.06429)  

**Abstract**: Interpretability methods for large language models (LLMs) describe internal state but do not directly test whether that state is causally sufficient to produce the observed behavior. In earlier work, we lesioned LLMs to produce error profiles in picture naming, a central task for assessing aphasia, and found that specific lesions produced errors resembling those of individual stroke survivors. Here we ask the inverse question: given an error profile, can the lesion parameters that produced it be recovered, and what does this inverse problem reveal about transformer computation? Lesions in LLaVA-Vicuna 13B were parameterized by layer index, modification percentage, and noise sigma across 4,840 configurations, and error profiles were characterized by a seven-category clinical taxonomy (correct, semantic, unrelated, formal, mixed, neologism, no-response). We trained a multi-task neural network to map error profiles back to perturbation parameters. The problem admitted a partial solution: across 10 independently trained inverse models, modification percentage and noise sigma were recoverable, whereas layer index was recoverable only within a neighborhood. In counterfactual validation, a fresh model instance perturbed with the recovered parameters reproduced the target behavior in 81.4% of cases. This dissociation between low layer recovery and high counterfactual fidelity is consistent with functional redundancy across transformer layers, a property not captured by standard interpretability methods. As an out-of-distribution test, we applied the trained model to picture-naming error profiles from 278 stroke survivors; recovered parameters were syndrome-discriminative, most strongly for perturbation intensity, indicating generalization beyond the training distribution. Counterfactual validation provides a general framework for LLM interpretability claims beyond inverse mapping. 

---
# NTDH: Complex Reasoning for Comprehensive Affective Analysis 

**Authors**: Tianlei Zhu, Zhiwei Liu, Yuyan Wang, Xiao-Yang Liu, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2608.06425)  

**Abstract**: Comprehensive affective analysis is challenging for two reasons: it spans heterogeneous prediction tasks with continuous, ordinal, and multi-label outputs, and affective meaning is context-dependent, requiring conflicting cues to be reconciled rather than mapped directly to labels. Existing methods learn this mapping directly and do not model the reconciliation explicitly. We recast the task as a complex-reasoning problem, which yields one output interface across heterogeneous label spaces and a trajectory over which a verifiable reward can be optimised; to our knowledge, this is the first such treatment covering both sentiment and emotion. The obstacle is on the data side: affective reasoning traces must be synthesised, and generic synthesis is misaligned with the targets, tolerances, and phenomena of affect, and discards or leaks its failure cases. We propose NTDH, which addresses these four failures. Naturalisation sets the training answer to the gold label, so it is correct by construction. A Tolerance-aware gate checks each answer against the task's own scoring margin. Domain-aware strategies refine the reasoning using ideas from affective science. Directional Hints report only the type and direction of an error, without exposing the target. We train Qwen3-8B with SFT and then GRPO under the same tolerance used for verification (up to a more permissive construction gate on the multi-label subtask), and a component ablation quantifies the data-quality effect of each part. Using 16,302 training records, about 14x fewer than comparable instruction-tuned systems, the final policy improves over its SFT checkpoint on five of six official-test metrics and achieves the strongest EI-reg result among the compared systems, at a Pearson correlation of 0.862. 

---
# Separating Decision-Rule Misalignment from Readout-Coverage Limitations in Speech Language Models 

**Authors**: Linkai Peng, Baorian Nuchged  

**Link**: [PDF](https://arxiv.org/pdf/2608.06409)  

**Abstract**: Speech language models are increasingly evaluated on paralinguistic tasks by the accuracy of prompted answers, but answer accuracy combines failures at different stages of the audio-to-answer computation. We introduce a generation-aligned diagnostic ladder that compares the emitted answer, the option logits, an affine readout of those logits, and a linear readout of the hidden state at the same answer token. Successive differences separate endpoint, decision-rule, and readout-coverage gaps. Across five systems and two emotion corpora, state decoding exceeds generation by 27.8 accuracy points on average, and both the decision-rule and readout-coverage gaps are positive in all ten conditions. A label-free logit correction improves generated accuracy in every condition, showing that part of the decision-rule gap is actionable. In rank-matched comparisons, emotion information outside the native readout generalizes to held-out speakers and survives controls for measured acoustic descriptors, but replacing the selected readout-external directions usually has little effect on emitted answers. These results distinguish information availability from behavioral use and localize performance losses across the decision rule and the state-to-answer readout. 

---
# TEXAS: Task-Expert-Aware Supervision for Downstream Mixture-of-Experts LLM Adaptation 

**Authors**: Guanzhi Deng, Haibo Wang, Kuan Wu, Xiangru Jian, Shing Yin Wong, Sichun Luo, Zhuoran Wang, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.06396)  

**Abstract**: Mixture-of-Experts (MoE) language models route each token through a small subset of experts, making routing patterns useful for identifying task-relevant experts during downstream adaptation. Yet current approaches have two limitations: task experts are typically identified from aggregate routing statistics that reflect usage rather than association with successful task completion, and task-expert activations remain underexplored as signals for supervision allocation. We introduce Task-Expert-Aware Supervision (TEXAS), which combines correctness-conditioned task expert discovery with token-level supervision allocation. TEXAS compares expert activations on instances that the base model solves successfully and those it fails to solve, and retains experts more strongly activated on successful instances. During fine-tuning, it upweights answer tokens in failed instances when they activate these experts. TEXAS therefore leverages existing routing behavior without restricting adaptation to a fixed expert subset or imposing an explicit target routing distribution. Across three MoE models and six benchmarks, TEXAS achieves the best or tied-best performance in 17 of 18 settings and improves over the strongest baseline by 1.3--1.5 points on average. Ablations and further analyses validate both the discovered experts and the resulting supervision strategy. 

---
# SkillProx: Self-Evolving Agent Skills via Proximal Textual Gradient Descent 

**Authors**: Mingxuan Zheng, Yujin Zhou, Chuxue Cao, Boqin Yin, Yuyao Zhang, Jiapeng Sun, Shuaishuai Gong, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.07449)  

**Abstract**: LLM agents increasingly adapt to recurring tasks by accumulating procedural knowledge in skills. These skills are lightweight, reusable textual artifacts that are loaded into the agent's context without weight updates. Recent methods refine skills through iterative task execution, failure diagnosis, and trajectory-guided text-space updates. However, existing frameworks lack explicit diagnosis--outcome feedback and treat deletion as a generic edit operation rather than a dedicated mechanism for consolidating accumulated knowledge. We introduce SkillProx, a proximal-gradient-inspired forward--backward framework that couples closed-loop diagnostic evolution with utility-aware proximal refinement. Motivated by a composite objective balancing task loss and skill complexity, the forward stage re-executes diagnosis-driven edits on the same task batch, rolls back regressions, and feeds measured outcomes into subsequent diagnoses. The backward stage decomposes the resulting skill into auditable knowledge units, estimates their contributions using a frozen leave-one-out utility audit, and applies validation-gated consolidation, demotion, or removal. Experiments on in-distribution and out-of-distribution benchmarks across multiple backbone LLMs show that SkillProx improves average accuracy by 3.0 percentage points over the strongest gradient-based baseline. Component ablations demonstrate the complementary effects of closed-loop diagnosis and proximal refinement. 

---
# PsychoAgent: An Affect-Sensitive Cognitive Architecture for Conflict-Aware Memory in LLM Agents 

**Authors**: Mohammad Amanlou, Parham Abed Azad, Farbod Davoodi, Mostafa Masumi, Behnam Bahrak, Abdol-Hossein Vahabie  

**Link**: [PDF](https://arxiv.org/pdf/2608.07438)  

**Abstract**: Human-like cognition does not select past experience by topical similarity alone: affective significance and unresolved conflict also shape what becomes accessible. We present PsychoAgent, a cognitive architecture for LLM agents that separates factual and affective memory and integrates both through a conflict-aware executive controller. Affective memories are first filtered by semantic relevance and then re-ranked by salience, preserving topical fit while allowing emotionally important traces to enter the prompt. Across three controlled conflict scenarios, the full architecture retrieved more conflict-critical memories than semantic-affective and single-memory RAG baselines (0.933 vs. 0.500 and 0.667), with a small semantic-similarity cost. Five blinded raters evaluated 27 outputs. After within-rater standardization, the full architecture had the highest overall mean (+0.22 SD), but corrected pairwise differences were not significant. A three-day illustrative trace further shows persistent affect, offline memory recombination, and selective memory reweighting. The findings support affect-sensitive retrieval as an inspectable mechanism for modeling human-like conflict effects in LLM agents. 

---
# SABRE: Scalable and Automated Benchmarking of VLMs under Stress 

**Authors**: Zixuan Lan, Luzhe Sun, Matthew R. Walter, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.07435)  

**Abstract**: Vision-language models (VLMs) are improving rapidly, but benchmark development lags behind, making weaknesses hard to identify. Building stress tests is costly: samples must satisfy controlled conditions, remain answerable, and challenge current models. We present SABRE, a scalable, automated pipeline that converts a Test Primer (a Markdown Task Design with Data Schema) into structured specifications, generated or edited images, and question-answer pairs. Automated filtering removes candidates solved by a Filtering VLM, while human review verifies candidate validity and supports annotation correction and localized image repair. We instantiate SABRE-Prior to test whether VLMs follow visual evidence instead of relying on world priors -- learned expectations about familiar objects and scenes. Its 600 images and 1,000 questions span Context (unexpected entities in familiar scenes), Texture (counterfactual materials), Attribute (noncanonical component counts), and Language Elicitation (answers suggested by language but unsupported by the image). Across six VLMs, macro-average accuracy ranges from 17.8% to 31.3% (22.6% mean). A real-image Attribute control is comparably difficult for the Filtering VLM. SABRE-Counting and SABRE-Spatial pilots show that the workflow supports other stress-test settings. These results establish SABRE as a reusable framework for constructing and refreshing VLM stress tests rather than a single fixed benchmark. 

---
# ResidencyRL: Reinforcement Learning in Simulated Clinical Environments 

**Authors**: Valentin Liévin, Samuel Schmidgall, Tim Strother, Alex Bijamov, Akshay Goel, Anil Palepu, Chunjong Park, Vahid Balazadeh, Min Woo Sun, Marius Guerard, Justin Chen, Dave Steiner, Vikram Dhillon, Ibrahim Azar, Akhil Mehta, Nicholas Spetsieris, Shilpan Shah, Maen Abdelrahim, Amit Dahiya, Yun Liu, Katherine Chou, Yossi Matias, Avinatan Hassidim, Dale R. Webster, Quoc V. Le, Raia Hadsell, Joelle Barral, Carey Radebaugh, Aleksandra Faust, Shekoofeh Azizi, Mike Schaekermann, Po-Hsuan Cameron Chen, Tao Tu, David Racz, Lin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07418)  

**Abstract**: In medical education, physicians convert academic knowledge into clinical expertise through residency: years of training across thousands of encounters, with diverse sources of feedback and progressively greater autonomy. Much of clinical reasoning relies on the patient encounter, a dialogue in which a clinician elicits history, refines diagnostic hypotheses, and decides management under uncertainty. While large language models (LLMs) excel on static medical benchmarks, methods to optimize the full sequence of clinical decisions remain underdeveloped. We present ResidencyRL, a reinforcement learning (RL) method for training clinical artificial intelligence (AI) agents through simulated multi-turn clinical encounters (up to 60 dialogue turns and 8 tool calls per trajectory). ResidencyRL pairs the policy agent with LLM simulators capable of complex, adversarial behaviors, training against a structured reward aligned to diagnostic accuracy, management quality, communication, documentation, and safety. On held-out evaluations, the ResidencyRL agent improves diagnostic accuracy by 7.0% under adversarial conditions (88.0% vs. 81.0%) and reduces missed red flag rates by 31%, demonstrating rigorous mitigation of premature closure. Blinded expert clinicians validated these gains, preferring the trained agent in 87.6% of side-by-side comparisons. The procedural competencies transfer to unseen benchmarks: the agent outperforms the base model across all six clinical axes of the AMIE multi-visit benchmark, and shows consistent directional improvements on AgentClinic and CRAFT-MD. Our findings demonstrate that sequential clinical decision-making can be effectively learned through multi-turn RL in simulation, yielding robust, generalizable capabilities, paving the way towards clinical mastery. Prospective validation with real-world workflows remains necessary to establish clinical utility. 

---
# GeoBenchLLM: A Comprehensive Benchmark for Evaluating LLMs on Geo-Related Tasks 

**Authors**: Rodrigo Ferreira Rodrigues, Karim Radouane, Jose G Moreno, Lynda Tamine  

**Link**: [PDF](https://arxiv.org/pdf/2608.07411)  

**Abstract**: In the context of geodata, existing Large Language Models have often been studied in a homogeneous setting, which has considerably limited insights into their generalization capabilities. In this paper, we present \benchName, a comprehensive benchmark for probing LLMs on geo-related tasks. We leverage a careful selection of twelve publicly available datasets from diverse geo-related tasks and domains, and evaluate a set of LLMs on geo-spatial and temporal understanding using our benchmark. Our results show that reasoning and size have a strong impact on overall performance. GeoBenchLLM is publicly available at this https URL. 

---
# Trajectory-Relative Hindsight Distillation for Agentic Reinforcement Learning 

**Authors**: Haoyu Zheng, Yun Zhu, Qing Wang, Wenqiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07371)  

**Abstract**: Recent agentic reinforcement learning methods use hindsight to complement sparse outcome rewards. However, a completed rollout can yield many such signals, leaving their appropriate allocation across turns unclear. We introduce TRIAL, a trajectory-relative hindsight distillation framework with a unified turn-aligned scoring protocol. For each decision turn, TRIAL extracts an outcome view of that decision's realized consequence and evaluates the same response under ordinary and hindsight-conditioned contexts. The signed log-probability gap determines the direction and local strength of token-level supervision, while turn-level magnitudes are normalized jointly over the realized trajectory. The resulting allocation multipliers have an eligible-token-weighted mean of one, redistributing dense supervision across turns while fixing its average multiplier. Experiments on WebShop and ALFWorld with different backbones show that TRIAL outperforms GRPO across all eight combinations of backbone, environment, and evaluation metric, while achieving the best or tied-best performance among six methods on six of them. On WebShop with Qwen3-1.7B, TRIAL improves the success rate from 56.4% to 75.2% and the task score from 78.7% to 85.7%. Controlled ablations further show that trajectory-relative turn allocation provides substantial gains beyond those of dense hindsight distillation alone. 

---
# Artificial Intelligence Can Match Domain Experts in Evidence Extraction and Critical Appraisal of Microbial Oncogenesis Research Publications 

**Authors**: Kaela Kokkas, Hairong Wang, Richard Klein, Nazir A. Ismail, Natalie Irwin, Mohammad Z. Moonsamy, Kubendran Naidoo, Jeremy Nel, Ekene E. Nweke, Raveen Parboosing, Emmanuel K. Sekyi, Rebecca T. van Dorsten, Bruce A. Bassett, Robert F. Breiman  

**Link**: [PDF](https://arxiv.org/pdf/2608.07250)  

**Abstract**: Confirmed oncogenic microbes contribute significantly to cancer burden. Identifying novel microbial oncogenicity could yield strategies that will reduce disease burdens. However, relevant evidence is dispersed and infeasible for humans to comprehensively synthesize. LLMs may enable scalable, expert-level systematic evidence synthesis to identify microbe-cancer pairs; however, such capabilities have not yet been demonstrated. Domain experts were recruited to create a dataset to benchmark LLM performance (Gemini 2.5 Pro, Gemini 2.5 Flash, GPT-5, GPT-5 Nano) on 24 research papers using MMTV-LV and breast cancer as a case study. We devised a structured template for evidence extraction and appraisal, consisting of MCQ, Likert-scale, multi-select, and free-text question types (77 items across 24 papers). Agreement between (1) experts and (2) experts and each LLM was determined per question instance using novel metrics. LLMs were assessed by comparing inter-expert and expert-LLM agreement distributions to determine whether LLMs behaved as additional experts by increasing or maintaining inter-expert agreement. Free-text responses were further evaluated qualitatively. Across all question types, LLM responses aligned closely with experts, with GPT-5 and GPT-5 Nano achieving score distributions indistinguishable from experts. Gemini models behaved similarly but were significantly more lenient in applying microbial oncogenesis criteria. Hallucinations were rare. Methodological appraisal and identification of contradictions within full-texts were the most persistent LLM vulnerabilities. GPT-5 and GPT-5 Nano were indistinguishable from experts on structured domain research paper evaluation tasks. This supports use of LLMs for automated systematic evidence synthesis. However, methodological appraisal tasks and contradiction identification in full-texts remain weaknesses requiring strengthening. 

---
# Recipes for Creativity: Iterative Generation and Evaluation in Large Language Models 

**Authors**: Rens Anderson, Tessa Verhoef, Amirhossein Zohrehvand  

**Link**: [PDF](https://arxiv.org/pdf/2608.07243)  

**Abstract**: Generative models are often evaluated through singular artifacts, whereas human creativity typically emerges through iterative generation, appraisal, and refinement. This pilot study examines whether iterative search improves LLM creativity by adapting FunSearch to recipe generation for the 2024 Pillsbury Bake-Off and evaluating outputs against human benchmarks using TTCT-based LLM evaluation. Across two experiments, we test iteration count, generator temperature, and in-loop selection-scorer model size. Results show that iterative generation-selection can produce recipes with creativity scores comparable to human benchmarks, but additional iterations alone do not improve creativity. The in-loop evaluator matters most: a smaller selection scorer yields significantly higher scores across most TTCT dimensions, while temperature has limited effects except for originality. These findings suggest that evaluator design is a first-order design variable in subjective creative search. 

---
# Modular TTT: Rethinking Test-Time Training as Composable Modules 

**Authors**: Bohao Tang, Zhen Qin, Yuqi Pan, Zheng Li, Pengfei Liu, Ya Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07110)  

**Abstract**: Test-time training (TTT) views sequence modeling as an online learning problem in which fast weights are updated by an internal learning rule. Despite the growing number of TTT variants, existing approaches typically hard-code each variant separately, which makes it difficult to design new TTT methods and to isolate the role of each component. To address this, we propose Modular TTT, a framework that represents the inner learner as a directed acyclic graph and exposes the fast-weight network, loss function, learning rate, weight decay, and normalization as explicit design dimensions. Modular TTT automatically composes primitive-level train-view forward, train-view backward, and causal query-view rules into the full graph-level TTT computation, including the fast-weight state transition. Using Modular TTT, we systematically ablate the components of TTT and find that small learning-rate initialization, weight decay, and a single-layer nonlinearity improve performance, while MSE and inner-product losses perform similarly. Deeper fast-weight networks and normalization tend to hurt performance because they induce excessively large activations, while residual connections and gating provide little measurable benefit. Guided by these findings, we train the best resulting variant as 410M- and 1.45B-parameter models on 100B tokens, and observe training loss and benchmark performance comparable to Gated DeltaNet. 

---
# DocMemo: Dynamic Evidence Discovery via Probabilistic Memory-Guided Retrieval for Multi-Modal Document Understanding 

**Authors**: Hanshu Yao, Janfeng Zhong, Niu Lian, Jinpeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07067)  

**Abstract**: Long-document understanding requires locating sparse and heterogeneous evidence across hundreds of pages, yet existing systems remain limited by static retrieval and fragile cross-round memory. Mainstream single-round methods commit to a fixed top-$k$ page set at the outset and struggle to recover from early retrieval errors; recent iterative approaches allow multi-round evidence acquisition, but they do not investigate the propagation mechanism of cross-round states, making it difficult to track the dynamic changes in page relevance. To address these limitations, we propose DocMemo, a memory-guided framework that formulates long-document reasoning as dynamic evidence exploration. DocMemo maintains a tri-level retrieval state consisting of Document Schema Memory, Page Belief Memory, and Question Episodic Memory, which respectively capture structural priors, dynamic relevance estimation, and query-specific reasoning trajectories. During reasoning, DocMemo continuously refines cross-round page selection through Bayesian page belief updating with Thompson sampling, spatial proximity propagation, and structure-aware adaptive-granularity evidence access, while supplementing page-level evidence with fine-grained visual regions. Experiments on 3 benchmarks show that DocMemo achieves state-of-the-art performance and validate the efficacy of structured memory and dynamic page belief updating. Code is available at this https URL. 

---
# How Should I Pick a Foundation Model for My Robot? In Favor of a Community Evaluation Framework for Social Robots 

**Authors**: Eric Nichols, Alva Markelius, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2608.06898)  

**Abstract**: Researchers who seek to build social robot applications on foundation models are faced with a difficult question: how should we pick a model? Public leaderboards offer little guidance: the demands of real-time, embodied social interaction lie largely outside their focus. And direct evaluation is impractical at scale: each embodied study requires scarce participant, robot, and experimenter time. In this paper, we identify five evaluation dimensions for foundation models in social robots: (i) conversational competence, (ii) user safety, (iii) embodied character, (iv) target scene effectiveness, and (v) audience appropriateness. To make model selection cheaper and better informed, we propose a three-tiered evaluation funnel paradigm that first filters with general metrics, then extends to simulated interactions, and terminates in more expensive, robot-specific evaluation. We map all five dimensions across all three tiers, chart where applicable evaluation methods exist and are missing, and close with a call to action: let's build the evaluation framework together as a community. 

---
# DAEP: Difficulty-Aware Evidence Planning for Medical Video Corpus Temporal Answer Grounding 

**Authors**: Tianjian He, Yujie Liu, Zhiping Huang, Changbo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.06869)  

**Abstract**: We describe DAEP, team BIGC's submission to NLPCC 2026 Shared Task 1 Track 3: Difficulty-Aware Temporal Answer Grounding in Video Corpus (DA-TAGVC). The task requires retrieving the target video from 50 candidates and localizing the answer-supporting span. DAEP ranks videos with subtitle, visual, and procedural-context evidence, expands high-scoring anchors into temporal spans, and reranks spans for final output. Its main design is to convert the task-provided simple/complex input label into an inference-time evidence plan controlling modality weights, Top-K aggregation, boundary threshold, expansion length, and reranking strength. In the official evaluation, BIGC ranks first among ten systems with an Average score of 0.2728. Validation ablations show that visual evidence, procedural context, and difficulty-aware planning improve ranking quality, with the largest gain on complex questions. 

---
# LoRAScan: Detecting Backdoor Prompts in Low-Rank Adapters for Large Language Models via Down-Projection Activation Spikes 

**Authors**: Doniyorkhon Obidov, Honggang Yu, Xiaolong Guo, Kaichen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.06795)  

**Abstract**: Low-rank adaptation (LoRA) enables efficient specialization and distribution of large language models through compact adapters. However, untrusted adapters introduce a supply-chain threat: a backdoored adapter can cause a model to generate harmful content, malicious code, political propaganda, or covert advertisements when an input contains a hidden trigger. Adapter-agnostic defenses merge the adapter with the base model, which dilutes backdoor signals and reduces detection performance. Existing adapter-aware methods do not address how to safely use a potentially backdoored adapter. Instead, they either train a defensive adapter to repair a backdoored base model, addressing the inverse problem rather than securing the adapter itself, or rely on a classifier that flags the entire adapter as suspicious and requires separate mitigation. These methods overlook the distinct latent-space signatures produced by trigger-bearing inputs in backdoored adapters.
We introduce LoRAScan, the first adapter-aware defense that detects and rejects trigger-bearing inputs at inference time without modifying adapter parameters. Our key observation is that a small subset of LoRA insertion sites, approximately 5%, remains stable across clean inputs but exhibits highly concentrated spikes in LoRA down-projection activations when a trigger is present. LoRAScan identifies these low-variance insertion sites before model deployment and monitors them during inference. Across standard LLM backdoor benchmarks, LoRAScan rejects approximately 98.49 of malicious inputs with a small error rate on clean inputs, outperforming existing defenses across diverse evaluation settings. 

---
# Genotypic Triggers: Exposing Pharmacogenomic Blind Spots via Host-Specific Backdoors in Generative Antimicrobial Peptide Models 

**Authors**: Doniyorkhon Obidov, Xiaolong Guo, Yonghui Li, Kaichen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.06779)  

**Abstract**: Large Language Models (LLMs) have accelerated drug discovery, particularly in the automated design of antimicrobial peptides (AMPs). However, current validation pipelines for peptide generation models overlook historical precedents showing that certain drugs carry health risks predominantly for individuals with specific genetic profiles. In this paper, we demonstrate that such targeted health risks can be induced intentionally and at scale by manipulating models that generate peptide candidates. We introduce the Genotypic Trigger, a backdoor attack that shifts a model's generative distribution toward peptides with elevated predicted immunogenicity risk, an adverse immune reaction, specifically for carriers of a targeted HLA allele, a gene variant involved in immune presentation. Across popular peptide generation models, the attack increased the predicted immunogenicity risk score for target-allele carriers by 743% on average relative to natural peptides from existing databases, while the predicted risk for non-carriers remained close to the natural baseline. Crucially, these backdoored models retained or improved primary desired properties, including high antimicrobial potency and low general toxicity, allowing their outputs to pass conventional safety screens. 

---
# Retrieval-Constrained Policy Optimization for Attack Technique Extraction from Cyber Threat Intelligence 

**Authors**: Jiayun Zhang, Junshen Xu, Zejun Xie, Yi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2608.06778)  

**Abstract**: Mapping cyber threat intelligence (CTI) text to MITRE ATT&CK techniques is essential for structured threat analysis, yet manual annotation is costly and does not scale. The ATT&CK taxonomy comprises several hundred attack techniques, and a single CTI passage may describe multiple techniques, making accurate and complete extraction challenging. Existing automated approaches fall short in different ways: multi-label classifiers struggle with severe class imbalance and the large label space, while LLM-based methods--retrieval pipelines and fine-tuned generators--optimize token-level objectives that treat technique annotation as sequence generation rather than set prediction, lacking direct supervision on whether the predicted technique set is correct and complete. We propose TTP-R1, a two-stage framework that combines retrieval-augmented supervised fine-tuning (SFT) with reinforcement learning using verifiable rewards (RLVR). A hybrid retriever first narrows the large label space to a candidate set, and a fine-tuned LLM learns to select the correct techniques. We then apply Group Relative Policy Optimization with a decomposed reward that directly supervises the precision, recall, and output format of the predicted technique set. Across four CTI benchmarks, TTP-R1 achieves the best average F1, improving sub-technique-level F1 by 7.4 percentage points over Claude Sonnet 4.5 with retrieval augmentation, while running 28x faster when served as an 8B-parameter model on a single GPU. 

---
# Mind the Gap: A Dual Knowledge Graph Framework for Unified Multi-task User Intent Inference 

**Authors**: Tzu-Cheng Peng, Chien Chin Chen, Chih-Hao Ku, Yung-Chun Chang  

**Link**: [PDF](https://arxiv.org/pdf/2608.06752)  

**Abstract**: This paper proposes DKG-MTI, a dual knowledge graph framework for unified multi-task user intent inference from online travel reviews. Existing approaches often rely on hierarchical pipelines that suffer from error propagation or retrieval methods that ignore structural relationships in domain knowledge. To address these limitations, we introduce an inference-only knowledge augmentation framework that dynamically constructs a User-Specific Intent Knowledge Graph from each review and aligns it with a Global Hotel Knowledge Graph through structure-aware semantic smoothing. The aligned knowledge is combined with the original review and processed by a large language model to simultaneously predict aspect ratings and generate reverse user intent statements. Experiments on TripAdvisor reviews show that DKG-MTI consistently outperforms strong LLM and retrieval-based baselines in both classification and intent generation tasks, demonstrating the effectiveness of structure-aware knowledge alignment for scalable and explainable intent inference. 

---
# IB-RL: Isolated Bilateral Reinforcement Learning for Strategic Dialogue Agents 

**Authors**: Senhao Wang, Chenghao Cai, Haitao Hu, Mingxing Huang, Xingguang Wang, Wenhao Li, Zecheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2608.06735)  

**Abstract**: Reinforcement learning (RL) has achieved strong results in improving large language models (LLMs) on tasks with stationary, verifiable rewards, such as mathematical reasoning and code execution. In these settings, the environment follows fixed rules and does not adapt strategically to the agent. Strategic dialogue differs in this respect: the environment is another agent that adapts to the policy, and success depends on the interaction between the two sides. Despite this interactive nature, current RL approaches typically train a target agent against a fixed counterpart or simulator. We find that this training paradigm encourages the policy to exploit counterpart-specific regularities rather than learn strategies that generalize across counterparts. We call this problem the static-counterpart mismatch, which we quantify directly in our experiments. To address it, we propose Isolated Bilateral Reinforcement Learning (IB-RL), in which the two roles coevolve through joint rollouts while each role optimizes its own reward through fully independent advantages, action masks, and update paths. We evaluate frozen policies against fully independent held-out counterparts in both domains. On Vehicle TeleSales, IB-RL achieves 89.6% Success@1, compared to 84.6% for the best unilateral RL baseline. On Deal-or-NoDeal, it reaches 98.4% agreement against DeepSeek V4 Pro, compared to 86.4% for the best unilateral baseline. These results indicate that jointly training both roles with strict peragent isolation produces policies that generalize more effectively to unseen counterparts. 

---
# Online Monitoring and Corrective Steering of Programming Agents 

**Authors**: Shuyang Liu, Saman Dehghan, Ji Young Kim, Jatin Ganhotra, Martin Hirzel, Reyhaneh Jabbarvand  

**Link**: [PDF](https://arxiv.org/pdf/2608.06701)  

**Abstract**: Fixing GitHub issues in large-scale projects is a long-horizon task, especially when a fix requires changes across multiple locations or the issue description lacks the information needed to localize and repair it. As a result, agents traverse long trajectories that are prone to inefficiency and error: they drift away from their intended plan, repeat failed actions, or terminate without a working patch. This paper proposes LivePlan to monitor, detect, and correct such behavioral inefficiencies and drifts in real time. LivePlan decouples judging from advising: a deterministic, rule-based monitor examines general signals over the trajectory to detect issues without invoking an LLM, and only when an issue is detected does it consult an advisor LLM for a high-level, next-step correction. This design avoids the misleading re-planning and costly interventions of prior approaches. We implement LivePlan on top of SWE-agent and evaluate it using five LLMs (three as executor agents and two as advisors) across SWE-bench Verified and SWE-bench Pro. Compared to vanilla SWE-agent, LivePlan notably improves issue resolution rates, achieving consistent gains of up to 15.2% (average: 9.9%), while incurring only an additional cost of $0.08 per instance. The additional solutions concentrate on medium and hard instances. LivePlan consistently outperforms alternative approaches in resolution rate, with minimal regression on already successful runs and new successes on problems that no baseline solves. 

---
# Model Confidence Under Answer-Preserving Attacks: An Informativeness-Manipulability Frontier 

**Authors**: Reza Khanmohammadi, Ivan Brugere, Simerjot Kaur, Charese H. Smiley, Kundan Thind, Mohammad M. Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2608.06571)  

**Abstract**: Deployed vision-language systems often gate their answers on confidence, making confidence robustness relevant to oversight. We study confidence readouts under white-box, image-only attacks constrained to preserve the generated answer byte-identically. Under a reachability assumption, an unmovable readout cannot outperform the answer-string accuracy prior, whose pooled value is 0.617. Independently of that assumption, a uniform amplitude certificate below a measurable threshold guarantees adversarial discrimination above the same floor. Across four vision-language models, three visual question answering benchmarks, five deployed confidence channels and two defense estimators, direct or surrogate-aimed attacks produce itemwise feasible perturbations that refute this uniform certificate in all 84 estimator-by-cell combinations. Coordinated correctness-label-aware attacks drive adversarial discrimination to or below the answer-string floor in all sixty deployed-channel cells, including all fifty-nine that begin above it. Hidden-state interventions and an open-ended text-model activation-space replication show that comparable confidence movement can be induced at the representation level rather than only through adversarial images. None of four tested defense families establishes a robust alternative under the specific evaluation applied to it. In a confidence-gated simulation, a coordinated token-probability attack transferred to a hidden-state gate causes up to 84.8% of previously rejected wrong answers to become accepted. After reweighting to each benchmark's natural correctness prevalence, accepted accuracy falls below the no-gate baseline in eight of twelve cells under transfer and all twelve under a direct gate-aimed attack. Under the studied threat model and budget, confidence is therefore an integrity-sensitive rather than intrinsically robust oversight signal. 

---
# Quantization Damage Is Multiplicative, Not Additive 

**Authors**: Zekun Wu, Swati Dhiman, Adriano Koshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2608.06564)  

**Abstract**: Quantization is how large language models are actually deployed, and below four bits it is known to hurt. What nobody can say is which of the model's decisions will change at a given bit-width. The damage is silent: a compressed agent stops calling its tools, then loses half its safety refusals, yet benchmark scores barely move. Prior work assumes quantization adds noise of a roughly fixed size, which would make confident decisions safe. We measure the decision itself instead. The margin of a two-way decision is the model's score for the option it picks minus the score of its best alternative; we track it before and after quantization across 16 models from 8 model families, three quantization methods, and bit-widths from 8 down to 2. Quantization does not add fixed-size noise to the margin. It multiplies the margin by a factor that collapses with bit-width (median 0.86 at 4 bits, 0.33 at 3, 0.00 at 2); we call this margin shrinkage. This contraction reduces the protection a large margin affords; the model's own small biases pick the direction of failure: at 3 bits the decision to call a tool collapses toward inaction while the choice of which tool is untouched. In fitted statistical comparison, additive-noise accounts never win on the damaged tool and safety decisions. The fitted relation predicts flip rates within a median of 1.8 percentage points on held-out decisions, though no flip was used in the fit; per decision, the predicted flip probabilities are calibrated uncertainty estimates (expected calibration error 0.004 over 131,758 predictions). The same form holds in every model we measure, but the constants are each model's own and do not transfer. A small paired margin set, measured per model and bit-width, estimates which decisions break without full generative evaluation; under our cost-matched tests, nothing repairs damage more cheaply than one more bit. 

---
# Can MLLMs Decode the Creative Leap? Introducing C4 for Cross-Concept Understanding 

**Authors**: Ming Wang, Yuqing Zhang, Tingna Xie, Xiangju Li, Xiaocui Yang, Daling Wang, Shi Feng, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.06501)  

**Abstract**: Creative capabilities of MLLMs matter in design, communication, education, and human--AI collaboration, yet remain difficult to evaluate because explicit targets and reward signals are scarce compared with accuracy-oriented tasks. Cross-concept understanding is a core cognitive capacity underlying receptive creativity. It enables a perceiver to recover intended meaning from non-obvious but meaningful conceptual relations. We operationalize item construction as cross-concept encoding and model inference as cross-concept decoding. We introduce C4, a cognition-inspired evaluation framework for Chengyu (Chinese idiom)-based Cross-Concept Creativity. Its encoding component maps target slots to imageable substitute concepts along bridge paths in a manually annotated and third-party-reviewed cross-concept network, enabling batch generation with explicit structure, difficulty indexed by bridge count and depth, and exact answers. Using this framework, we instantiate the C4 Evaluation Set (C4-Eval), comprising 184 synthetic items and 37 human-created cross-concept chengyu figures collected from online sources. We manually construct and review cross-concept relations, bridge paths, and reasoning processes for the collected figures. Each C4-Eval item is instantiated in five task settings, yielding 884 primary answer-recovery cases. Across ten evaluated MLLMs, the strongest closed models reach 50.7% and 48.0% primary accuracy, while open-source models remain substantially lower. Candidate constraints improve accuracy sharply, but bridge hints and explanation requests provide only modest gains. These results expose a substantial gap in how current MLLMs decode creatively encoded meaning through cross-concept relations. The code is in the supplementary material. 

---
# StepJack: Benchmarking Computer-Use Agent Safety Against Multi-Step Indirect Prompt Injection 

**Authors**: Zhuoxin Zhan, Akbar Rafiey, Avery Ma, Leila Pishdad, Layla El Asri  

**Link**: [PDF](https://arxiv.org/pdf/2608.06477)  

**Abstract**: Computer-use agents (CUAs) face a growing threat from indirect prompt injection, where adversarial instructions are planted in the environment such as web pages. In this paper, we introduce multi-step indirect prompt injection, a new attack class against CUAs in which the adversarial goal is decomposed into multiple innocuous-looking sub-steps and distributed across a chain of pages referenced along the agent's navigation path. We develop a pipeline to automatically decompose an adversarial goal under the constraint that the execution of the decomposed sub-steps must achieve the original goal while optimizing the innocuousness of each decomposed sub-step. With this pipeline, we build StepJack, a CUA safety benchmark with 480 test examples. On this benchmark, we evaluate six state-of-the-art CUAs and find that at a fixed decomposition depth, multi-step attacks raise attack success rate (ASR) on three of six CUAs, by up to 31.2 points (e.g., GPT-5.4-mini: 41.7% at single-step to 72.9% at three-step); averaged over the five CUAs that can reliably follow the reference chain (all but EvoCUA-32B), ASR rises from 31.3% at single-step to 36.9% at three-step. Dataset and code are available at this https URL. 

---
# Multi Codec Discrete Diffusion Model for Text Guided Speech Inpainting and Editing 

**Authors**: Iftach Shoham, Tali Dror, Oren Gal, Haim Permuter, Gilad Katz, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2608.06424)  

**Abstract**: Speech recordings often contain missing, corrupted, or incorrect regions that must be reconstructed or modified without re-synthesizing the entire utterance. Speech inpainting restores missing segments, whereas speech editing replaces spoken content according to an edited transcript. Both tasks require the generated speech to express the intended words while remaining consistent with the surrounding speaker identity, prosody, timing, and recording conditions. Discrete diffusion is particularly well suited to these tasks because it can iteratively refine masked tokens while jointly conditioning on both left and right acoustic context. We introduce SIEDD, a discrete diffusion framework for text-guided speech inpainting and editing over hierarchical codec tokens. Its core architecture, HiCoDD, follows the RVQ generation order by representing previously generated codebooks as clean, committed acoustic context and applying diffusion only to the current refinement codebook. This separation enables leakage-free joint training while matching sequential coarse-to-fine inference. The model further combines phoneme-level conditioning, span-localized classifier-free guidance, and duration prediction to support both fixed-duration inpainting and variable-duration text edits. On the RealEdit benchmark, SIEDD achieves the best overall speech-editing performance among the evaluated methods. It also outperforms the evaluated autoregressive baselines across all speech-inpainting settings, on both single and multiple gaps. These results demonstrate that explicitly modeling the codec hierarchy substantially improves context-preserving speech reconstruction and editing. See our full code at this https URL. 

---
# Latent Fact-Checking: Detecting Misinformation through Activation Engineering 

**Authors**: Pedro Barcelos, Otávio Parraga, Marcelo M. Mussi, Lucas M. Fraga, Lucas S. Kupssinskü, Rodrigo C. Barros  

**Link**: [PDF](https://arxiv.org/pdf/2608.06417)  

**Abstract**: The proliferation of misinformation online has driven demand for scalable detection systems. While most existing approaches rely on surface-level linguistic features or external knowledge retrieval, we examine truthfulness as a geometric property of a language model's representation space. We introduce a misinformation detection framework grounded in activation engineering, which leverages the latent geometry of transformer models. Our approach elicits a misinformation direction in the residual stream by contrasting activations from paired truthful and false statements, following the difference-in-means principle of Contrastive Activation Addition (CAA). At inference time, the last-token activation of an unseen claim is projected onto this direction, and the projected representation is fed to an Multilayer Perceptron (MLP) for classification. The procedure requires no fine-tuning of the backbone model, no external evidence retrieval, and no task-specific supervision beyond the contrastive pairs used to estimate the direction. We evaluate the method across 11 models from the Gemma, Llama, and Qwen families, ranging from 270M to 12B parameters, on three fact-checking benchmarks: AVeriTeC, LIAR, and FACTors. The falsehood direction is recoverable across model scales and architectural families, and last-token projection matches or surpasses zero-shot and few-shot prompting baselines on LIAR and FACTors, with the largest gains observed for smaller models. Performance on AVeriTeC is more limited, which we attribute to its evidence-grounded labeling scheme. These findings provide evidence that truthfulness is a structured, linearly separable concept in the latent space of pretrained language models, and point toward interpretability-driven misinformation detection as a practical complement to retrieval-based pipelines. The code is available on this https URL. 

---
# ADIAS: Automated Design of Interactive Agentic Systems 

**Authors**: Lekang Jiang, Bohan Tang, Stephan Goetz, Yiwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.06410)  

**Abstract**: Automated agent design improves agent harnesses through iterative revision, evaluation, and feedback summarization. Existing methods are largely candidate-centric: cross-round experience is organized around candidate agents, which leaves the repair progress implicit. This causes inefficient repair targeting, slow consolidation of partial progress, and propagation of ineffective interventions across rounds. Therefore, we formulate issue-centric agent optimization, in which repair progress is carried forward as an explicit persistent issue state to guide optimization, rather than re-derived from candidate history in each round. We instantiate the formulation in ADIAS, a framework for automated full-code agent design with two mechanisms. A persistent issue state maintains stable issue identities, lifecycle status, supporting evidence, and intervention-outcome histories. Issue-guided optimization uses this state to jointly propose repair targets and revision directions for subsequent focused full-code modification. Across five interactive benchmarks, ADIAS outperforms the strongest baseline by 25.2% on average and achieves consistent gains across four backbone models. Controlled ablations further show that removing persistent issue state or replacing issue-centric revision with candidate-centric policies leads to performance drops of up to 40.7%. 

---
