# ToolSmith: An Automatic Tool-Generation Framework for Multimodal Search Agents 

**Authors**: Rui Sun, Junhua Huang, Luca Zancato, Md Mofijul Islam, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31198)  

**Abstract**: Recent multimodal search agents use reasoning, vision, and live web search across many turns to answer questions beyond parametric knowledge by orchestrating a small library of hand-crafted search and visual tools. However, the library is laborious to create and maintain, and frozen when new query types, evidence formats, and failure modes are encountered in the wild. Human practitioners, by contrast, cope with insufficient tools by fashioning new ones from familiar parts and reusing them across tasks. Inspired by this, we propose ToolSmith, an automatic tool-generation framework that grows and reuses modules on top of a lean base tool library. We evaluate ToolSmith on three in-domain multimodal search benchmarks and ten out-of-domain benchmarks under a VLM-as-a-Judge protocol. ToolSmith consistently improves over the base library across deployed agents of varying scales, with larger backbones reaping greater gains, demonstrating the scalability of the approach. Our code and data are available at this https URL. 

---
# Deneb-Vritra: Descriptions, Edges and Names Examined by Blocking Via Restoring Token Representations and Attention 

**Authors**: Mohammed Nayeem Khan, Akhil P, Shahina Shaik  

**Link**: [PDF](https://arxiv.org/pdf/2611.31139)  

**Abstract**: Laya is a 421M-parameter encoder that answers a question about a text in one pass: each answer option, written as a name plus a description, gets a marker token, and one score is read from the network at each marker. We first confirm that such models can follow an option's name when it contradicts the description; the question is where in the network this happens. We locate it. On 951 synthetic questions we move each name onto another option's description and count how often the model picks the option with the right name (the name shortcut): 53% for the released checkpoint, 59% for our fine-tune v1, 11% for a fine-tune with name randomisation. On a 320-item sample, blocking attention from each marker to its own name tokens in layers 14 to 17 of 28 cuts v1's shortcut from 67% to 16% (19% on two schemas held out from component selection) while accuracy with the original names stays at 90%; blocking marker edges to the question, other options or text has no effect. No single head matters (largest effect 9 points), and restoring clean marker states undoes the swap from layer 16. Attention mass tracks the window by layer but not by head. Repeated on 5 checkpoints (released, fine-tuned, name-randomised, and a pair trained without two schemas), the knockout finds similar layers; in the name-randomised ones, blocking the description edges instead brings the shortcut back to 56-58%, so the read persists and is outvoted. Marker states decode the name from layer 0; a one-direction probe erasure changed nothing but was weaker than a random edit. Copying layers 0-13 or 14-27 of a name-randomised sibling into a fine-tune recovers 66% and 65% (v1); 21% and 45% (pair A) of its shortcut reduction. 9 of the 12 predictions we could score, written before running, failed. Head sweeps cover 5 of 5 checkpoints; data are synthetic and LLM-labelled; one architecture. 

---
# Serialization and the Character-System: Measuring Narrative Concentration in French Fiction (1800--1940) 

**Authors**: Jean Barré, Antoine Bourgois, Thierry Poibeau  

**Link**: [PDF](https://arxiv.org/pdf/2611.31133)  

**Abstract**: Every novel divides a finite quantity of narrative attention among its characters, and whether the shape of that division, between a narrative organized around a single figure and one spread across a crowd, has a history is far less settled than the abundance of critical commentary on protagonists might suggest. We operationalize Woloch's notion of the character-system, the distribution of narrative attention across a novel's cast structured by the tension between "the one" and "the many", as the inequality (Gini coefficient) of the PageRank distribution over a character network. We trace this measure of structural concentration across 12,011 French novels published between 1800 and 1940, drawn from the Hors d'usage corpus of Gallica fiction. Over the long nineteenth century the trajectory is U-shaped, with concentration declining to a trough in the 1850s-1860s and rising again through the turn of the century. The change is small, comparable to the difference between a novel built on two co-equal protagonists and one built on a single protagonist, but it is robust to the character cutoff, corpus density, cast size, and author effects. Detective and adventure fiction are less concentrated than the average novel of their period. Literary history reads the roman-feuilleton either as the novel of a superior hero or as a form of proliferating episodes, and on this structural measure our results support the second reading. Across the age of the feuilleton (1840-1914), authors whose careers were structured by newspaper serialization distribute centrality more evenly than a literary control, net of period, genre, and canon. Because such claims are only as trustworthy as the networks they rest on, we also evaluate the extraction pipeline against manually annotated novels, at the levels of cast and character identity. 

---
# Look Back, Draft Ahead: Corrective Prefix Attention for Parallel Speculative Decoding 

**Authors**: Mohammad Mahdi Moradi, Ali Pourranjbar, Zongzuo Wang, Mingming Zhu, Ben Feng, Boxing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.31112)  

**Abstract**: Parallel speculative decoders propose an entire token block at once, yet one early inconsistency can cause the target to reject its remaining suffix. We introduce Corrective Prefix Attention (CPA), which combines prefix-aware selection with autoregressive supervision to improve these parallel drafts. CPA lets each candidate attend to earlier choices within the block. Fixed candidate shortlists allow all pairwise interactions to be computed before selection, reducing the sequential walk to table lookups, accumulation and lightweight scoring rather than repeated neural evaluations. An auxiliary autoregressive loss trains the shared backbone on clean prefixes without adding an inference pass. Across eight math, code and dialogue benchmarks with Qwen3-4B and Qwen3-8B targets on Ascend NPUs, CPA commits 13--15\% more tokens per verification round than DSpark and 25--26\% more than DFlash. Its mean speedups reach $3.35\times$ and $3.37\times$ under greedy decoding and $2.86\times$ under sampling for both targets. Ablations support both components, while capacity sweeps demonstrate that greater acceptance alone does not guarantee faster decoding. These results motivate correcting parallel proposals with causal information while keeping sequential computation inexpensive. 

---
# Unlocking the Power of Loops in Pretrained LLMs with Loop Residuals 

**Authors**: Xuanlei Zhao, Hailun Xu, Kai Wang, Yang Luo, Yingxuan Ren, Zirui Zhu, Yuxuan Lou, Kun Xu, Yan Wang, Kanchan Sarkar, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2611.31094)  

**Abstract**: Looped Transformers iterate a shared block of layers to increase computation without adding parameters, offering stronger performance and test-time scaling. However, this power has so far required training from scratch and yielded only modest gains for LLMs. Finetuning existing models into loops is far cheaper, yet also brings only marginal gains over SFT. We propose LoopRes, an effective, scalable, and simple approach that turns loops into significant and consistent gains for pretrained LLMs. Building on Attention Residuals, LoopRes lets each looped layer attend directly to the outputs of earlier loops. Since these outputs are iterates of the same layers and become harder to tell apart, we give each a trainable loop-aware key that encodes its structural identity. Attending to all earlier loops also lets layers skip intermediate ones, so we restrict each layer to a residual window of the latest W loops, forcing every loop to build incrementally on recent ones. Across 6 pretrained LLMs from 3 model families and 9 benchmarks, LoopRes delivers significant and consistent gains: at 2x compute, it improves Qwen3-1.7B-Base by 29.1% over SFT, more than doubling the gain of the state-of-the-art baseline. The gains hold on larger models up to 30B and across all three families. More importantly, from 1.7B to 8B, LoopRes scales more steeply than SFT as compute grows, reaching a higher ceiling while needing less compute on long outputs. These results establish LoopRes as an effective looping approach, delivering both better performance and better test-time scaling. 

---
# DKnown-Jev-2: Article-Level Answer-Evidence Classification and Efficient Inference with a Pre-trained Language Model 

**Authors**: Qi Li, Jianjun Xu, Jiu Li, Pingtao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.31056)  

**Abstract**: Evidence selection in normative knowledge question answering requires determining whether a candidate article contains content that directly supports an answer under the applicable conditions of the question. We present DKnown-Jev-2, an article-level answer evidence classifier designed for evidence integration across multiple paragraphs. The model jointly encodes question conditions, article metadata, and paragraph content, and uses a pointer-style classification head to predict Answer Evidence (AE) and Non-answer Evidence (NE) in a single forward pass. Ona manually annotated test set of 972 evaluation instances, DKnown-Jev-2 achieves 91.77% accuracy and 91.28% macro-F1, outperforming Jev by 2.16 and 2.37 percentage points, respectively, while reducing both evidence misses and non-evidence false positives. The test set contains 317 multi-paragraph instances, with a maximum body length of 9,322 characters. Shared article encoding and non-autoregressive classification support low-latency inference. These results demonstrate the effectiveness of the proposed method for article-level evidence selection in normative knowledge question answering. 

---
# CART: Clustering-based Analysis of Recurring narratives for Taxonomy construction 

**Authors**: Aggelos Semoglou, Thomas Tselentis, John Pavlopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2611.30993)  

**Abstract**: We present CART, a computational pipeline for extracting political narratives from parliamentary discourse and organizing them into a data-driven taxonomy without relying on predefined narrative categories. It encodes parliamentary discourse using semantic representations, filters relevant and reliable content through unsupervised confidence estimation, extracts narrative statements using LLMs, and uses narrative clustering and cluster analysis to construct a narrative taxonomy. Applied to Greek Parliament proceedings surrounding the 2023 Tempi train crash, it identifies ten recurring narrative families concerning responsibility attribution, institutional failure, railway safety, accountability, privatization, and the political interpretation of the tragedy. Our results demonstrate how combining unsupervised learning with LLM narrative extraction enables large parliamentary corpora to be transformed into interpretable, data-driven taxonomies of political narratives. 

---
# Beyond Self-Consistency: Recursive Self-Improvement at Test Time 

**Authors**: Jianyu Wu, Yizhou Wang, Encheng Su, Jinouwen Zhang, Pengze Li, Lintao Wang, Chen Tang, Aoran Wang, Wanli Ouyang, Shixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30965)  

**Abstract**: Large language models (LLMs) that continue improving after deployment (e.g., recursive self-improvement) offer a promising path toward general intelligence. Current test-time methods pursue this goal on unlabeled questions, primarily relying on self-consistency: they generate independent solutions and use majority voting to select an answer or construct training targets. This leaves earlier reasoning unused at inference and can reinforce an incorrect majority during training. Humans instead build on earlier attempts to preserve useful reasoning, correct mistakes and improve their problem-solving strategies. Following this principle, we propose Test-Time Recursive Self-Improvement (TT-RSI), a label-free framework for improving LLMs at test time through two mutually reinforcing components: (1) In-Context RSI repeatedly re-solves each question using the full previous solution as a reference, letting the model decide whether and how to use it so that inference compute builds on earlier reasoning. (2) In-Weight RSI distills successive refinements into the model's weights, enabling progress beyond an initially wrong majority and theoretically bringing the first attempt toward the model's infinite-step refinement limit. The updated model answers better and refines more effectively, generating stronger references for subsequent training. Extensive experiments show that TT-RSI improves average accuracy by +13.89% over the base model and outperforms the strongest self-consistency-based method by +4.79% under the same test-time scaling budget. 

---
# LimeTree: Linear-Memory Graph Neural Networks over Tree-Graphs for Resource-Efficient Language Modeling and Context Extrapolation 

**Authors**: Elisabeth Fink  

**Link**: [PDF](https://arxiv.org/pdf/2611.30934)  

**Abstract**: The search for computationally efficient alternatives to the Transformer has yielded numerous architectures in recent years; however, most exhibit severe weaknesses in associative recall-such as key-value (KV) retrieval-and length extrapolation, where even standard Transformers struggle. This paper introduces LimeTree, a linear-complexity architecture in both memory and compute for sequence modeling. By integrating synthetic KV retrieval tasks directly into standard language modeling training, LimeTree achieves strong perplexity alongside reliable recall. Two variants were evaluated on WikiText-103 and PG-19 under strict compute constraints. On WikiText-103, LimeTree reaches a token-level perplexity of 19 in just 6 hours on four Quadro RTX 8000 GPUs, while maintaining 100\% KV retrieval accuracy even when extrapolating from a training context length of 512 to an evaluation length of 16,384. On PG-19, the model achieves a token-level perplexity of 13 in 35 hours on a single RTX PRO 6000 Blackwell GPU with minimal hyperparameter tuning. These results position LimeTree as a competitive linear-complexity alternative to Transformers, combining superior perplexity and training throughput with robust in-context retrieval and length extrapolation. 

---
# PRIMEAPE: A Multilingual Benchmark for Fine-Grained Meta-Evaluation of Judge Models 

**Authors**: Vladislav Mikhailov, Sarfraz Ahmad, Momina Ahsan, Zoia Butenko, Vladimir Dobrovolskii, Preslav Nakov, Erik Velldal, Lilja Øvrelid, Ekaterina Artemova  

**Link**: [PDF](https://arxiv.org/pdf/2611.30925)  

**Abstract**: Large language models increasingly serve as judges of other models, shaping benchmark rankings and model development. Yet, a judge whose rationale consistently supports its verdict may still disagree with human preferences. To study this, we introduce PRIMEAPE, a multilingual benchmark for evaluating judge verdicts and rationales. PRIMEAPE provides 2,205 response pairs covering 243 instructions in English, French, Spanish, Urdu, Hindi, Norwegian Bokmål, and Yoruba, with 11k human annotations of preferences and rationales across relevance, naturalness, truthfulness, safety, and overall quality. Our evaluation of twelve open- and closed-weight judges reveals that aggregate performance conceals differences across languages and criteria, while requesting rationales does not consistently improve agreement with human preferences. Judges also favor the first position, longer responses, and responses from related models, with effects that vary across criteria and languages. Crucially, judges can justify verdicts that disagree with human preferences while maintaining consistency between their verdicts and rationales. These findings show that consistency with a judge's own verdict does not establish alignment with human judgment. PRIMEAPE enables assessment of both, providing a foundation for more reliable multilingual judges. We release our benchmark and all annotation materials under the CC BY-NC 4.0 license at this http URL. 

---
# Semantic-aware Decoding for Diverse Text-to-SQL under Schema Ambiguity 

**Authors**: Yuxuan Jiang, Anna Mitsopoulou, Apostolos Koukouvinis, Yannis Velegrakis, Georgia Koutrika  

**Link**: [PDF](https://arxiv.org/pdf/2611.30873)  

**Abstract**: Real-world text-to-SQL systems often face schema ambiguity, where a question can map to multiple plausible schema elements. To support downstream disambiguation, SQL generation systems should include diverse interpretations among their top-k predictions. However, large language models (LLMs) typically favor a single interpretation, producing highly skewed outputs. We propose Semantic-aware Logit Adjustment (SLA), a training-free decoding method that promotes diverse SQL generation by adjusting token logits based on semantic similarity between schema elements. Unlike prior approaches that rely on reasoning-intensive prompting or task-specific fine-tuning, SLA is efficient, model-agnostic, and easy to deploy. Experiments show that SLA consistently improves interpretation coverage over standard decoding and remains competitive with existing diversity-oriented methods at substantially lower inference cost. 

---
# SARA: An Auditor-in-the-Loop LLM Recommender for Sustainability Reports 

**Authors**: Lars Hillebrand, Thiago B. F. de Oliveira, Tim Dilmaghani, Mohamed Khaled, Writwick Das, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2611.30839)  

**Abstract**: Auditing corporate sustainability reports requires assurance professionals to map a company-specific checklist drawn from roughly 1,700 regulatory requirements onto the relevant passages of lengthy, repeatedly revised draft reports and to judge compliance. We present SARA (Sustainability Reporting Assistant), a human-in-the-loop recommender system delivered as a Microsoft Word add-in that recommends, for each requirement, the passages addressing it and predicts a compliance verdict while keeping the auditor in full control. SARA casts auditing as content-based recommendation over a fixed regulatory taxonomy and makes per-requirement large language model (LLM) matching cheaper, faster, and more accurate through topic-prediction filtering: on expert-annotated reports, it reduces matching cost about sixfold and runtime by 42% versus feeding the full report to the LLM. It is in production use across PwC Germany's sustainability-assurance practice. 

---
# Task-Dependent Linguistic Measures in Aphasia Discourse: A Comparative Analysis of Narrative, Descriptive, and Procedural Speech 

**Authors**: Emma Al Kubaisi  

**Link**: [PDF](https://arxiv.org/pdf/2611.30826)  

**Abstract**: Background: Transcript-based measures such as word counts, lexical diversity and disfluency rates are widely used to describe language ability in aphasia. Each measure comes from a specific elicitation task, and studies use different tasks, so values from different tasks may not be comparable. Aims: To test whether discourse tasks change these measures within the same speakers, and whether the measures follow a gradient from Procedural to Descriptive to Narrative discourse. Methods & Procedures: Baseline transcripts from 61 adults with chronic post-stroke aphasia were taken from the Fridriksson-2 corpus in AphasiaBank (POLAR trial, this http URL NCT03416738). Each speaker produced a Procedural (Sandwich), a Descriptive (Broken Window) and a Narrative (Cinderella) sample in one session. A rule-based pipeline extracted 13 measures, 10 of which were common enough to test. Friedman, Page and Wilcoxon tests with false discovery rate correction were used, with a speaker-grouped Random Forest classifier as a supporting check. Outcomes & Results: Total words and number of different words followed the full gradient, but the step from Procedural to Descriptive was small. For most other measures, Narrative stood apart, while Descriptive and Procedural did not differ. Among the 31 speakers with at least 25 words in every task, filler rate, repetition rate and total disfluency density were higher in Narrative than in both other tasks. The classifier reached an accuracy of 0.61 (always guessing the most common task would give 0.41), mainly by recognizing Narrative samples. Conclusions: Discourse tasks are not interchangeable, and expressing disfluencies per word does not remove the task effect. The task should be reported with every measure, and values should be compared only within the same task. 

---
# Trallie: from Free-Form Text to Structured Data with LLMs 

**Authors**: Vijayasri Iyer, Cristiano De Nobili, Mubashir Hussain Shah, Sébastien Bratières  

**Link**: [PDF](https://arxiv.org/pdf/2611.30773)  

**Abstract**: Trallie is an open-source framework that leverages large language models (LLMs) to perform information extraction (IE) from natural language sources with or without user guidance. Unlike conventional approaches that rely on extensive labeled datasets or domain-specific training, Trallie enables the conversion of unstructured data into structured formats with minimal or no representative examples. The framework is designed with three key objectives: (i) reducing manual annotation by harnessing prompt-based zero-shot and few-shot capabilities of modern LLMs, (ii) flexibly parsing diverse document formats (i.e., HTML, PDF, TXT, and JSON), and (iii) improving accessibility for non-expert users by automatically inducing schemas from document collections, which can then be refined for extraction. We evaluate Trallie with open-weight LLMs on FDA 510(k) documents and on movie and university websites from SWDE. It outperforms previously reported systems when the target attributes are provided for text extraction and is competitive when it must discover attributes for the schema automatically. The project is available at this https URL 

---
# CocoonBench: Can Multilingual Search Agents Escape Language Information Cocoons? 

**Authors**: Hongchuan Zeng, Bowen Liu, Changjiang Gao, Senyu Han, Zhipei Xu, Peisong Wang, Xinbing Liang, Yuhan Li, Nuo Chen, Blaze Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30727)  

**Abstract**: Multilingual search agents can access the global Web, yet may remain trapped in a language environment with insufficient evidence; we refer to this failure as a language information cocoon. However, existing search benchmarks largely operate within a single language, especially English, overlooking entity connections across multiple languages. We introduce CocoonBench, a benchmark that evaluates whether multilingual search agents can escape such language information cocoons. CocoonBench mines answer-centered relational paths from a frozen Wikidata-derived graph and selects clue sets that are individually necessary and jointly identify a unique answer. Using language-specific entity names, it records each task's minimum language cover, removes direct identifiers, and renders each task as matched prompts in three languages together with its construction certificate and provenance. Across 10 state-of-the-art models, mean accuracy on CocoonBench ranges from 33.8% (DeepSeek-V4-Flash) to 69.7% (Claude-Opus-4.8), leaving even the strongest system below 70%. The same construction certificates support path-aligned trajectory analysis, enabling diagnosis of model failures in language routing, evidence-path recovery, and answer use. The result shows that escaping a language cocoon requires more than switching languages: successful cross-lingual search depends on both covering the correct languages and connecting the evidence found across them. 

---
# Hard to Translate, Harder to Detect: Human Analysis Remains Essential in AI Meme Translation 

**Authors**: Olga Zamaraeva, Carla Martínez, Carlos Gómez-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2611.30710)  

**Abstract**: We present a dataset and evaluation setup for a problem in creative machine translation: translating culturally loaded passages that carry much of a text's meaning and humor that may depend on subtle or implicit discourse. We study Soviet comedy quotations that function as memes in Russian-language communication. Off-the-shelf difficulty estimators do not find these passages; current models searching the web find most but not all of them, but their own analyses of the difficulty do not help translation as well as a human-validated one. Given a brief human-validated analysis of why each passage is difficult, older models improve modestly; the strongest current translators (GPT-6 Sol and Claude Opus 5.5) improve more, and in human evaluation GPT-6 Sol comes close to the approved references on most films. Newer models have thus become much better at using such an analysis while still not reliably producing it themselves. We release (1) a dataset of five films' Russian subtitles with their audience-selected culturally salient passages, human-validated analyses, and reference translations (into English, Galician, and Spanish), together with every full-film LLM translation and evaluation produced in this study; and (2) an item-level MQM-style evaluation protocol that measures translator and evaluator noise to set the number of runs the evaluation needs, and whose automatic version tracks human judgment of these passages, which reference-free neural metrics do not, whether they score the whole text or the passages alone. 

---
# Synthetic Twins: Closing the Gap Between TTS Test Sets and Real Recordings in Turkish ASR Evaluation 

**Authors**: Nur Yağmur Tuncer, Mustafa Sertaç Türkel, Fatma Nur Korkmaz, Ahmet Tuğrul Bayrak  

**Link**: [PDF](https://arxiv.org/pdf/2611.30694)  

**Abstract**: Speech synthesized by text-to-speech engines is a low-cost substitute for recorded test sets in automatic speech recognition, yet an off-the-shelf synthetic test set is worth only a few dozen real recordings for estimating the real error rate of a recognizer. This paper examines which properties of real recordings are missing from synthetic speech and how much of the gap can be closed. For 507 Turkish Common Voice recordings, synthetic twins are built: a ladder of synthetic copies that adds, one factor at a time, the pitch of the speaker, a clone of the same speaker, the real recording channel and the speaking rate. At each step, the agreement between ASR behaviour on the copy and on the real recording is measured. Across three zero-shot cloning engines and panels of six and 19 recognizers, the speaker clone is the step that improves agreement on which sentences are hard, raising the Spearman correlation from .41 to .55 for Chatterbox. Matching the channel or adding generic noise corrects the error level when the synthetic speech is easier than real speech. The twin with the highest worth is equivalent to 70 to 122 real recordings, depending on the recognizer panel, compared with about 9 for a stock voice. The findings replicate on VoxForge, a second corpus in which speaker identity is known rather than inferred: the clone raises sentence agreement from .31 to .50 and the worth from 8 to about 50 recordings. Agreement on which sentences are hard is distinct from agreement on how hard they are. The paper concludes with a practical recipe and a reporting checklist for synthetic ASR test sets. 

---
# Learn the Effects, Not the Examples: Implicit ICL via Attention Patching 

**Authors**: Seungwon Woo, Hyunseo Shin, Wonseok Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30690)  

**Abstract**: In-context learning (ICL) enables language models to adapt to new tasks from demonstrations, but repeatedly including multiple examples increases context length and inference cost. Many existing implicit ICL methods encode demonstrations as fixed vectors, limiting adaptation of the intervention to individual queries and token positions. We propose Attention-Patching, an implicit ICL method that models contextual changes in self-attention outputs through task-specific updates to attention output projections. Using query-answer pairs processed with and without a fixed demonstration set, Attention-Patching fits layer-wise affine transformations from non-contextual attention representations to the demonstration-induced output changes. Across 12 classification benchmarks and four model families, Attention-Patching achieves higher average accuracy than the evaluated implicit ICL baselines while maintaining performance comparable to explicit ICL at near zero-shot inference cost. Further experiments on diverse generation tasks demonstrate its applicability to autoregressive multi-token generation. 

---
# Empirical Margin Sensitivity: An Interventional Neuron-Level Probe of Post-Trained Medical LLMs 

**Authors**: Aniruddh Shreyas, Aarnav Kembhavi, Minal Moharir  

**Link**: [PDF](https://arxiv.org/pdf/2611.30675)  

**Abstract**: Medical post-training raises benchmark accuracy, but accuracy does not show which internal components a model's decisions depend on, or whether components that support one benchmark also support another. We apply the weight-space intervention paradigm of ROME and MEMIT at the granularity of one MLP down-projection column. Our measurement, Empirical Margin Sensitivity (EMS), replaces each coordinate magnitude of a column by its mean while preserving the column's sign pattern and L1 scale, and records the resulting change in the logit margin between the correct option and its hardest-wrong competitor. This isolates the margin's dependence on a column's fine-grained coordinate structure while the column stays in place. We discover high-EMS columns (anchors) on MedQA in ten public checkpoints (four general-purpose bases and six medical post-trained variants) across the Gemma, Llama, Qwen, and AFM families, and measure the same columns on MedMCQA and PubMedQA. Mean-ablating each discovered anchor set lowers MedQA accuracy in all six post-trained models, by 6 to 28 points. In the two pure-SFT checkpoints, crushing anchors raises the PubMedQA margin on correctly answered items (negative EMS), and anchors with this property sit significantly deeper than the same model's other anchors; the four checkpoints trained with RL or mixed recipes show positive PubMedQA EMS and no significant depth shift. The depth association recurs in each of three held-out checkpoints. Because the checkpoints differ in data and pipeline as well as objective, we report these as per-checkpoint observations. 

---
# From Rewarding What Models Say to How They Think: Attention Process Reward Model for LLM Reinforcement Learning 

**Authors**: Kairong Han, Nuanqiao Shan, Ziyu Zhao, Xinpeng Dong, Zijing Hu, Dian Jin, Leyan Deng, Junjian Ye, Lujia Pan, Fei Wu, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30627)  

**Abstract**: Reinforcement learning (RL) has become an important component in large language models (LLMs). However, existing RL approaches focus on final outputs or textual solution trajectories, leading to coarse-grained rewards and unreliable reasoning. Such paradigms focus mainly on what the model outputs while overlooking how the model internally reasons. Therefore, can we reward how a model thinks internally rather than merely what it says? We first reveal the problem of textual reward aliasing in existing reward modeling approaches. Motivated by this, we propose the Attention Process Reward Model (APRM), which assigns internal rewards based on hidden states and attention maps to capture high-quality reasoning patterns. With only approximately 46M parameters, APRM is trained using ranking and counterfactual objectives, and its reward guides RL optimization while being periodically updated to handle distribution shifts. Integrated with representative RL methods, APRM achieved an average pass@8 gain of 2.88\%. Our results demonstrate the promise of incorporating internal reasoning signals into RL for LLMs. 

---
# Does the Mixer Matter ? Paired-Seed Ablations of Tiny Bilingual Language Models 

**Authors**: Ayoub Kirouane  

**Link**: [PDF](https://arxiv.org/pdf/2611.30623)  

**Abstract**: We report paired-seed architecture ablations from the development of Sophea Artemis, our in-house Greek-English model family. Every arm trains on the same 3B targets with a shared tokenizer and recipe; models range from 54M to 759M total parameters and roughly 54M to 124M active parameters. The main finding is that, at this tiny scale, stored capacity matters far more than mixer design. On a fixed held-out test, replacing the dense feed-forward blocks of a recurrent/attention hybrid with sixteen routed experts, at nearly the same approximate active parameters, lowers loss by 0.079 (Greek) and 0.112 (English) nats per target, and doubling to 32 experts lowers it by a further 0.022 and 0.031, at every paired seed. Reusing a four-layer block three times recovers about 40-44% of the gap between a shallow and an independently deep model, consistent with looped-Transformer results. In contrast, the choice among grouped-query attention (GQA), Gated DeltaNet (GDN) hybrids and a Kimi Delta Attention (KDA) hybrid moves loss by only about 0.01-0.02 nats; a 1:1 GDN/GQA hybrid is the one dense variant that improves on GQA in both languages at every held-out seed. In bits per character, Greek and English held-out losses are similar despite a 74.49%/25.51% training split, and synthetic mathematics, which comes from a collection outside the training sources, is the hardest Greek test component and gains least from added experts. Cached decoding needs its own qualification: trained models passed cached-versus-full logit checks only under a fixed-reduction inference policy, which itself changes 1.4-2.8% of next-token argmax choices relative to native evaluation. All results are conditional on one recipe and scale. 

---
# Quantizing Diffusion LLMs under Parallel Decoding: Commitment Risk and the Parallelism Tax 

**Authors**: Huan Xu, Maosen Zhao, Pengtao Chen, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30587)  

**Abstract**: Diffusion large language models (dLLMs) are served with accelerated decoders that cache context and commit, in one step, every masked position whose confidence clears a threshold. Yet post-training quantization of dLLMs has so far been calibrated and evaluated under fixed-step samplers without caching. We study W4A4 quantization under the deployed decoder and find that the decoder determines both where quantization harms accuracy and what it costs. Irreversible errors enter chiefly through forced commitments, made when no position clears the threshold: they are a minority of commitments but carry most quantization-induced flips. Quantization also levies a parallelism tax, extra forward passes per response: by compressing top logit gaps, it pushes positions below the threshold while leaving predictions largely intact. We propose DECAL, which reweights layer-wise calibration by a commitment-risk prior measured on the deployed decoder, and an argmax-preserving commit temperature that recovers the tax. On LLaDA and Dream, DECAL attains the best average accuracy among W4A4 methods, with lower run-to-run variance than the closest existing calibration method. 

---
# When Does Continual Pre-Training Help? An Empirical Study of Decoupled Knowledge and Generation Trajectories in Clinical LLMs 

**Authors**: Niclas Doll, Jasper Schulze Buschhoff, Shalaka Satheesh, Héctor Allende-Cid, Katrin Klug  

**Link**: [PDF](https://arxiv.org/pdf/2611.30507)  

**Abstract**: Continual pre-training (\textit{CPT}) is a standard route for domain adaptation, yet existing studies typically track pre-training loss or multiple-choice question answering (\textit{MCQA}) accuracy alone, offering only a narrow view of how adaptation unfolds. We jointly track factual knowledge via \textit{MCQA} and open-ended behavior via pairwise preference rankings across $18$ checkpoints for each of eight instruction-tuned \textit{Qwen} models ($0.5B$ to $8B$), adapted to German medicine with \textit{CPT} and \textit{SLERP} merging. The two axes converge at consistently different rates, with preference rankings reaching $90\%$ of their total gain after a median of $1.7B$ tokens versus $3.5B$ for \textit{MCQA}. A judge-free token-distribution measure reaches this threshold at $1.9B$ tokens. Per-aspect scores indicate that these early preference gains largely reflect surface alignment, meaning models can sound domain-appropriate before their factual knowledge catches up. Larger adapted models close more of the gap to unadapted next-tier generalists, whereas preference gain per FLOP falls by about an order of magnitude across our size range. We therefore argue that judge-based preference scores, which can plateau while \textit{MCQA} knowledge is still improving, should not be used alone to infer adaptation convergence or domain readiness. 

---
# Rethinking the Evaluation of Contrastive Decoding for Hallucination Mitigation in Multimodal Large Language Models 

**Authors**: Ting Chen, Geng Li, Guohao Chen, Hanjia Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30460)  

**Abstract**: Contrastive decoding (CD) mitigates object hallucination in multimodal large language models (MLLMs) by contrasting token distributions under different conditions to adjust token scores. However, existing generative and discriminative benchmarks primarily measure final task scores, which can be confounded by candidate filtering, changes in generation trajectories, and response biases, making it difficult to isolate the effects of the underlying contrastive adjustments. To address these limitations while complementing existing evaluations, we introduce CONCOR (Contrastive Correction Evaluation), a controlled benchmark that directly evaluates the competition between ground-truth and different types of hallucinated candidates using the Contrastive Grounding Margin (CGM). CGM also enables analysis of directional consistency, composition, and reliability of contrastive corrections. Our evaluation reveals that existing CD methods exhibit distinct correction behaviors: some consistently favor ground-truth candidates, whereas others produce strong corrections but also introduce more errors. Crucially, these differences help explain their contrasting responses to increased contrast strength, which improves some methods while degrading others. These findings highlight the importance of evaluating the contrastive correction itself, rather than relying solely on final-output performance, and CONCOR provides a controlled and interpretable framework for understanding and improving CD methods for hallucination mitigation. 

---
# Disclosure Bias in AI Reviewers 

**Authors**: Jens Leysen, Noah Daniëls, Marco Favier, Bart Goethals  

**Link**: [PDF](https://arxiv.org/pdf/2611.30410)  

**Abstract**: AI models increasingly help to review papers, and many conferences and journals now require authors to state how they used AI. Does this AI-use statement change how an AI reviewer rates a paper, and could it change which papers are accepted? In a preliminary study, we varied only the AI-use statement of 150 ICLR 2027 submissions, in up to nine versions adapted from real statements: from no AI, through writing help and coding, to an AI research agent and an AI author. Six AI reviewers rated the versions: the decision model Jev and the language models Claude Haiku 5.5, Claude Sonnet 5.5, Gemini 3.8 Flash, Gemini 3.1 Pro and Gemini 4 Argon. Across 23,652 reviews, we find that every reviewer rates papers lower on average when the statement says AI did all the work than under a statement of no AI use. The AI research agent statement lowers the rating significantly for Jev, Sonnet and Argon. We find no evidence of a disclosure bias for AI assistance: for writing help, coding and research help, no reviewer's rating differs significantly from its rating under a statement of no AI use. Whether the observed penalty changes which papers are accepted depends on more than the size of the penalty: on how finely and how consistently the reviewer separates the papers near the acceptance boundary, on whether the penalty is spread over all papers or concentrated on a few, and possibly on how many other papers disclose AI use. 

---
# TAP: Policy Interfaces for Diffusion Language Models 

**Authors**: Preeti, Ankita Kushwaha, Kiran Ravish, Pawan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2611.30406)  

**Abstract**: Diffusion language models denoise a masked canvas, making them a natural setting for selective generation decisions such as answering, waiting, clarifying, or refusing. We introduce TAP-DiLM, which adds a typed policy variable to masked diffusion LMs, and study how this policy should be attached to the denoising process. Across BERT-base, MDLMOWT, LLaDA-8B, and Dream-7B, we compare shared-vocabulary slots, dedicated slot heads, and CLS-based external heads. The results reveal a clear interface-backbone interaction: dedicated heads provide robust policy learning across scales, while sharedvocabulary heads can be effective when the LM-head geometry is favorable. A simple LMhead sharpness probe predicts when sharedhead policy internalization is reliable, giving a practical diagnostic for interface selection. Causal clamp and slot-injection tests show that dedicated policy slots can influence body denoising through feedback, especially at 8B scale. Safety-refusal experiments further show that TAP-DiLM supports competitive typed refusal behavior when trained with diverse unsafe prompts. 

---
# LeanReviewBench: Semantic Alignment Review in Mathematical Lean Projects 

**Authors**: Ruichen Qiu, Hao Shen, Lihong Zhi, Xiao-Shan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.30361)  

**Abstract**: Lean verifies that a proof establishes its formal proposition, but whether that proposition preserves the mathematical content of an informal source requires separate review. In project-scale formalizations, this correspondence can depend on definitions and constraints distributed across many declarations. We introduce LeanReviewBench, a benchmark for project-level semantic alignment review containing 197 versioned inputs collected from 18 real Lean projects, with 87 aligned and 110 misaligned reference judgments. Each input binds an informal mathematical target to selected formal declarations and an extracted dependency context. Construction combines development-history discovery, independent target binding, reconstruction under a common Lean environment, and Astra annotation with human review of uncertain cases. The collection spans diverse mathematical domains and includes 69 historical pairs. We evaluate six language models with Direct and ReForm prompts on all 197 inputs against final human-reviewed reference labels. GPT-6.1 Sol with Direct prompting achieves the highest reported accuracy of 0.827 and misalignment-detection F1 of 0.844; the effects of prompting vary across models. LeanReviewBench provides a testbed for reviewing semantic correspondence in existing mathematical formalization projects. LeanReviewBench is available at this https URL. 

---
# When Does Reasoning Help Entity Disambiguation? Small LLMs In-Domain and Under Shift 

**Authors**: Samy Haffoudhi, Fabian M Suchanek, Nils Holzenberger  

**Link**: [PDF](https://arxiv.org/pdf/2611.30334)  

**Abstract**: Entity disambiguation (mapping ambiguous mentions in text to entities in a knowledge base) is a foundational step in tasks such as knowledge graph construction, question answering, and information extraction. Recent work has applied LLMs to this problem, achieving stateof-the-art performance. Notably, reasoning LLMs have been shown to perform entity disambiguation in a true zero-shot fashion, generalizing across knowledge bases and domains. However, it is unclear whether this capability survives, or remains competitive, with smaller LLMs, where entity disambiguation has typically relied on reranking rather than reasoning. We ask when reasoning is worth its cost at small scale. We compare pointwise, listwise, and reasoning-based candidate-selection components within a classical coarse-to-fine entity disambiguation pipeline, across the classical, few-shot and true zero-shot settings. The reasoning paradigm is trained with distillation and reinforcement learning recipes. The benefit of reasoning grows with domain shift: negligible in-domain, decisive in the true zero-shot setting. We show that two distinct factors are at work: the reranker's uncertainty over the candidates locates where the paradigms disagree, while shift governs which is right when they do. 

---
# Evaluating Persona Consistency in LLM User Simulators Without External Reference Annotations 

**Authors**: Anping Zhang, Yang Tan, Yipeng Liu, Clive Bai, Blaze Chen, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.30280)  

**Abstract**: Standard evaluations assess LLM-simulated personas against an external ground truth, such as community trait annotations, but that reference is unavailable when the persona is inferred only from a short, task-oriented chat. We ask how persona consistency can be evaluated without such a reference, and propose a self-anchored internal-consistency diagnostic: the persona describes itself once and acts twice. A Likert self-report supplies a soft internal anchor; two behavioral profiles come from constrained scenario choices scored without a judge and from open-ended responses judged by an LLM. Consistency is the correlation between the self-report and each behavioral profile, in a shared six-facet BFI-2 space. The framework compares persona-construction strategies by assessing how consistently their elicited responses reflect their self-reported traits. Across ordinary-user chats (WildChat and LMSYS) and well-documented personas with community annotations (the Personality Database, PDB), profile-enriched representations agree with the anchor more closely than a raw-dialogue baseline. On PDB, this self-anchored ordering reproduces the community-annotation ordering. The pattern is stable across answering and judge models. 

---
# DUET: Dual-MDL Unsupervised Entropy-Trie Segmentation for Low-Resource Languages 

**Authors**: Xiaotian Wang, Youyuan Lin, Zhan Shen, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2611.30273)  

**Abstract**: Morphological segmentation, the task of decomposing words into constituent morphemes, becomes particularly challenging under extreme data scarcity, a regime that can arise in language documentation and linguistic reasoning. This difficulty stems from existing approaches' reliance on annotated data, external resources, or recurring patterns that emerge only in larger corpora. To address this limitation, we propose Dual-MDL Unsupervised Entropy-Trie Segmentation (DUET), a deterministic unsupervised algorithm that decouples structural candidate generation from statistical candidate ranking. DUET combines branching entropy with recursive paradigm discovery to generate linguistically plausible candidate segmentations, then ranks them using a dual minimum description length (MDL) criterion: a corpus-level objective rewards morpheme reuse, while a construction-tree objective guards against over-segmentation. On the SIGMORPHON 2023 interlinear glossing shared task, DUET outperforms unsupervised baselines in Morpheme F1 under both the standard and simulated extremely low-resource settings, while providing broader candidate coverage. Augmenting prompts with DUET segmentations improves performance on a linguistic reasoning benchmark, yielding the best overall results among the evaluated segmentation methods. 

---
# Voyag: Joint Multi-Intent Detection and Execution-Relation Classification for In-Vehicle Voice Assistants 

**Authors**: Sen Luo  

**Link**: [PDF](https://arxiv.org/pdf/2611.30253)  

**Abstract**: An in-vehicle voice assistant must decide not only which intents an utterance contains but also how co-active intents may be executed: concurrently, in a fixed order, or not at all without arbitration. We present Voyag, a single-forward decision model that jointly predicts ten controlled vehicle-intent slots and a three-way execution relation (parallel, sequential, mutually exclusive) over each co-active intent pair. Voyag combines a multilingual ModernBERT-style encoder (mmBERT-base, 306.9M parameters), a typed per-slot choice head, and a lightweight GroupCalibration relation head (1.61M parameters) that consumes marker hidden states, pooled context, explicit resource-requirement profiles, and symmetrized outer-product conflict features. All objectives are optimized in one joint loss: soft cross-entropy over slot choices, focal cross-entropy over relations, and a mutex-consistency penalty. On a synthetic Chinese vehicle-life benchmark with semantic scene-level splits and a leakage audit (30,017 / 3,473 / 5,000 groups), validation-only selection yields 0.9893 slot accuracy and 0.0526 expected calibration error over 50,000 slot decisions. Conditioned on gold intents, the relation head reaches 0.8548 accuracy, 0.8473 macro-F1, and 0.6062 mutually-exclusive recall over 4,000 pairs; end to end with one encoder forward it reaches 0.8530 accuracy and 0.8464 macro-F1. Against a two-stage rule pipeline, the aggregate end-to-end gain is small and borderline, so we claim no aggregate advantage; the learned head's advantage lies in sequential and parallel recall where surface rules are blind. We report rule baselines with a tautology disclosure and a held-out thermal conflict failure. 

---
# StutterStigma-TR: A Novel Turkish Dataset and Target-Aware BERT Framework for Stuttering-Related Stigma Detection 

**Authors**: Muhsin Dölek, Arezoo Sadeghzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2611.30252)  

**Abstract**: Computational research on disability-related ableist harm remains limited, and condition specific resources for stuttering-related stigma are particularly scarce. This study introduces StutterStigma-TR, a Turkish dataset of 6,019 public X/Twitter posts annotated as Stigmatizing, Non-stigmatizing/contextual, or Supportive using a speech and language therapy-informed protocol with independent lived-experience evaluation. We also propose StutterStigma-BERT, a BERTurk-based target-aware framework that combines explicit target marking, target-span representation, target-conditioned context, character-level and handcrafted orthographic information, and gated discourse cues. The framework is evaluated in complementary three-class and binary settings using controlled three-seed ablation analyses and final 20-seed robustness evaluations. Under the common training protocol, Gated Discourse-OrthoTarget trained with cross-entropy is selected as the final configuration for both tasks based on internal-validation Macro-F1. Across 20 random seeds, the final three-class model achieves a mean Macro-F1 of 73.54%, while the final binary model achieves 85.17% using validation-selected decision thresholds. Task adapted harmful-language comparators are additionally evaluated under the common training protocol, and the proposed model obtains the highest mean Macro-F1 among them. For the primary three-class task, matched backbone analyses show higher internal validation Macro-F1 with BERTurk than with TurkishBERTweet or Turkish Mamba, while substantial train-validation separation persists across all three backbones. Separate direct-transfer and zero-shot experiments examine performance without StutterStigma-TR supervision. Across the task-adapted, direct-transfer, and zero-shot comparisons, the condition-specific supervised systems retain the strongest aggregate performance. 

---
# Ontology-Compiled 2D Clinical State Fields with Bidirectional Diffusion Encoders for Zero-Training Cross-Ontology Medical Coding 

**Authors**: Zeqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30218)  

**Abstract**: Healthcare systems continually revise clinical taxonomies, update reimbursement schedules, and harmonize cross-border registries. Existing automated coding models bind clinical text understanding to fixed vocabulary-specific heads, requiring manual relabeling and retraining whenever code sets change. We introduce Ontology-Compiled 2D Clinical State Fields (OCSF), a vocabulary-free framework that decouples clinical document reading from target ontology indexing so one trained encoder can switch across arbitrary medical vocabularies in seconds without target training. A single-pass bidirectional diffusion-pretrained encoder and vocabulary-free decoder map each clinical note into a fixed-dimension two-dimensional state matrix with zero code-specific parameters. Any target ontology is compiled offline on CPU from public text descriptions into a scale-invariant Tikhonov deconvolution operator that suppresses sibling-code interference while preserving source decision thresholds. Trained solely on English MIMIC-IV ICD-9 discharge summaries and evaluated without target tuning across 58,706 held-out test notes and five unseen ontologies spanning English ICD-10, MS-DRG, Spanish, Greek, and Chinese (115-185,458 codes), OCSF reaches 63.15% Micro-F1 on in-domain ICD-9 (outperforming a matched closed-vocabulary classifier and eliminating late-epoch tail overfitting) and transfers directly to 26,085 English ICD-10 codes (29.23% recall at gold cardinality), MS-DRG groups, and multilingual terminologies, compiling up to 185,458 codes in under 10 seconds on CPU with sub-millisecond GPU readout. 

---
# What Makes Good Data for Training Coding Agents? A Systematic Study on SWE-Gym 

**Authors**: Yikai Zhang, Mai Xu, Yunbei Zhang, Qirui Jin, Kenan Li, Yuwen Jia, Jinwei Yao, Xiangyu Zhang, Jielun Tan, Jin Pan, Zijian Jin, Xiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.30207)  

**Abstract**: Reinforcement learning (RL) trains coding agents on software tasks whose hidden tests decide the reward, yet task collections check these tests only against the reference fix. We ask what makes a task good training data, focusing on the correctness of its reward labels. We audit SWE-Gym with two LLM families, and a suspected error counts only if execution reproduces it. We find that unsolved often means mislabelled: 74% of the tasks that a 9B base model never solves have a test that rejects correct fixes (a false negative), versus 20% of those it sometimes solves. We repair or remove such tasks, and the audited tasks that remain train better policies. At equal size and matched difficulty, models trained on them score 2.4 and 2.7 points higher on held-out tasks from SWE-bench Verified and SWE-bench Pro than models trained on the original tasks, whereas simply dropping the never-solved tasks shows no such gain. We further find that wrong patches that pass (false positives) make up about a quarter of all passing patches, and that new tests that catch a quarter of them do not change held-out scores detectably. A pass rate shows whether a task yields reward, not whether that reward is right, so the reward labels themselves must be checked. To this end, we release SWE-Gym-Verified with its 1,087 audited tasks, every verdict with its executed evidence, our training sets, and the full audit pipeline. 

---
# VADE: A Verifiable Agent Data Engine from Professional Sources 

**Authors**: Xiangchao Chen, Zhiheng Lyu, Kai Zhang, Xueguang Ma, Ping Nie, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30202)  

**Abstract**: Language agents have shown their practical value, particularly in software engineering, where abundant public code and executable tests support training and evaluation at scale. Extending this success to other professional domains, such as law and finance, is not straightforward: many agent tasks involve open-ended analysis, making outcomes hard to verify and, in turn, hard to use as scalable supervision. As a result, high-quality tasks in these domains are still largely expert-authored, making them expensive to scale and limiting their use for training. To address this bottleneck, we introduce VADE, a data engine that automatically constructs verifiable analysis tasks from public sources used in professional practice. VADE exploits a simple asymmetry: professional analyses are difficult to solve because the relevant evidence must first be found and interpreted, but much easier to verify once that evidence is fixed. During task synthesis, the engine constructs a reference analysis from source evidence and asks an independent agent to reproduce its conclusions from the same evidence without seeing it. Only after the analysis is verified is it compiled into an open-ended task with an item-level scoring contract, allowing the same task to support both evaluation and training. We apply VADE across four professional domains; in an independent expert review of 186 finance tasks, 93.5% meet a high bar for source fidelity, answer correctness, and evaluation fairness. In finance, these tasks expose substantial remaining model errors, particularly unsupported facts and inferences, while training on verified solver trajectories improves held-out performance and yields gains that transfer to public finance benchmarks such as Finance Agent Benchmark v2. Together, these results suggest a practical path toward scalable, verifiable supervision for open-ended professional tasks. 

---
# Laya-IsoToken: Day-One Clinical Codebook Switching via Closed-Form Operator Compilation 

**Authors**: Zeqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30196)  

**Abstract**: Healthcare systems periodically revise diagnostic and procedural classifications (ICD-9 to ICD-10/11), update hospital reimbursement groupers (MS-DRG), and deploy clinical coding across languages. Supervised coding networks bind clinical text encoders to fixed per-code classifier weights, forcing hospitals to re-annotate tens of thousands of records before switching codebooks, whereas standalone large language models incur high latency and hallucinate invalid codes. Here we show that day-one clinical codebook switching becomes feasible without target-labeled notes when local clinical evidence reading is decoupled from a strictly frozen multilingual concept space. Laya-IsoToken trains a clinical span reader once on source records with zero per-code classifier parameters, separating text evidence from historical disease prevalence. Given only public target code descriptions, closed-form matrix compilation aligns distributional shifts, compensates for paraphrased definitions, and deduplicates sibling codes in under three seconds, while a pre-deployment geometric residual routes unfamiliar concepts to offline language model verification. Across 32,804 MIMIC-IV ICD-9 discharges, Laya-IsoToken matches published supervised baselines (62.6% Micro-F1) while reducing active calibration error by 32%. Switched with zero target training notes to 13,783 unseen-patient ICD-10 discharges, MS-DRG groups, and multilingual cohorts, it outperforms biomedical dense retrievers, semantic crosswalks (+5.4 F1 points on ICD-10, paired $d_z = 0.49$, $p < 0.001$), and official US CMS equivalence mappings (+3.5 points, $p < 0.001$) at 9.4 ms per admission. Pairing this reader with an offline 31B verifier across 1,000 full discharge summaries lifts zero-training ICD-10 coding to 40.6% Micro-F1 (57.0% category F1, $d_z = 0.66$, $p < 0.001$) with 0.0% invalid code hallucinations. 

---
# PCC: Pivotal Channel Compensation for Layer Pruning of Large Language Models 

**Authors**: Xiangqian Wang, Yucheng Zhang, Zhaohong Jia, Kun Jing, Chunchun Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.30183)  

**Abstract**: Treating independent layers as the minimal pruning unit allows layer pruning to efficiently eliminate many redundant parameters and achieve high compression rates. Layer pruning has become a promising and prevalent structured pruning technique for large language models (LLMs). However, multi-layer removal can lead to significant performance degradation for LLMs. Although some works have explored various compensation methods for pruned layers, they are inefficient and costly. Observing that the expression of a large proportion of information is concentrated on a small number of pivotal channels, we propose a novel post-pruning compensation method for layer pruning, called pivotal channel compensation (PCC). PCC can achieve significant performance improvement by compensating for a small number of channels through learnable linear transformations at the pruning interface. Besides, we propose a new evaluation metric, called channel-wise average performance improvement (CAPI), which can measure the efficiency of channel-wise compensation. Our experimental results across various LLMs demonstrate that PCC can achieve significant and consistent improvements and high CAPI on language modeling and zero-shot question answering for the state-of-the-art cross-layer and consecutive-layer pruning methods. Compared with existing compensation methods, PCC can achieve competitive results with fewer additional parameters and less training cost. 

---
# Retrieval-Augmented Generation in the Agentic Era: Tradition, Present, and Prospect 

**Authors**: Chenming Tang, Haolin Li, Yiran Li, Zixian Yang, Lizhu Zhai, Rui Sun, Qizhou Huang, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30181)  

**Abstract**: Retrieval-augmented generation (RAG) couples frozen language models with external, updatable knowledge, so that generation is grounded in retrieved evidence rather than in parametric memory alone. This work maps RAG in the agentic era by joining them on a single lifecycle axis, from offline indexing through pre-retrieval, retrieval, post-retrieval, and orchestration over these stages. It treats indexing and orchestration as first-class components and agentic workflows as a paradigm of their own, and it covers the agentic work that has appeared since those reviews. Beyond these components, it covers evaluation, applications, and the questions that remain open for future exploration. From tradition to present and beyond, this work systematically reviews RAG from a perspective of the agentic era. 

---
# FiRE: Fine-grained Ranking Evaluation for Machine Translation 

**Authors**: Wenyang Gao, Yinghao Yang, Xi Jin, Jing Li, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30168)  

**Abstract**: Developing reliable machine translation (MT) systems hinges on our ability to distinguish superior translations from inferior ones. However, existing evaluation paradigms, whether limited to coarse overall rankings or misaligned with human preferences, fail to deliver interpretable, fine-grained feedback in reference-free settings. We present a Fine-Grained Ranking Evaluation method (FiRE) that leverages off-the-shelf large language models to perform criterion-driven pairwise comparison across three complementary dimensions: faithfulness, fluency, and consistency of style, instead of producing a single holistic judgment. To enable rigorous meta-evaluation of evaluation paradigms in the absence of any suitable testbed, we construct the first human-annotated, reference-free benchmark for fine-grained ranking evaluation, achieving substantial inter-annotator agreement. Through meta-evaluation on this benchmark and existing MQM datasets, FiRE demonstrably outperforms regression-based and error-analysis metrics in aligning with human comparative judgments, while providing more informative insights into translation quality. Finally, our examination of LLM evaluator biases (position and self-enhancement) and their handling of tied cases offers guidance for more nuanced MT evaluation. Code and benchmark resources are available at this https URL. 

---
# Reading the Tea Leaves, Again: Can Large Language Models Predict Language Model Training Runs? 

**Authors**: Xinran Zhao, Xiang Li, Tong Chen, Jinchuan Tian, Zichun Yu, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30141)  

**Abstract**: LLM agents now propose research ideas and run the experiments that test them, but they train every run to its end, whereas a researcher reads a run's configuration and the beginning of its curve and stops the runs that show no promise. Reading a run this way takes more than describing or implementing a training method: it takes knowing what a design choice does to a run. We build Tea Leaves to measure this knowledge on 682 real training runs across pretraining, fine-tuning, and reinforcement learning (RL): a forecaster sees the first 20% of a run with its configuration, description, and code, and predicts the rest. Across 11 LLMs and a range of forecasting baselines, the strongest LLMs match or exceed the best curve fitter on smooth loss curves (skill 0.66 against 0.62) and lead on noisy RL reward curves (0.43 against 0.13). What bounds every forecaster is what the log carries: on sparsely logged community runs all methods lose most of their skill, and none reliably flags a failing run in advance. Beyond the curve, information helps and effort does not: related runs in the prompt cut the error on loss curves by 39%, and knowledge of training dynamics lets the strongest LLMs order all 36 pairs of runs that differ by a described code change, which the early curves do not separate, whereas tools add little. This knowledge can be learned from logged runs: a small open model trained on public runs matches the best LLM on loss curves. Applied to the progress curves of research agents, the same reading ranks tasks by their remaining headroom, although for a single agent run it does not improve on the last value. We release the benchmark, the trained readers, and their training data to test and train LLMs' understanding of training and to help research agents decide which runs deserve their compute. 

---
# Parameter-Efficient Vocabulary Expansion via Cross-Lingual Embedding Alignment 

**Authors**: Andrew Ivan Soegeng, Muhammad Reza Qorib, Weishi Wang, Daniel Dahlmeier, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2611.30123)  

**Abstract**: Continual pretraining with vocabulary expansion is often used to make large language models (LLMs) more effective and efficient at handling new languages. However, when LLMs are extended to many languages simultaneously, the number of model parameters increases significantly. In addition, the newly added tokens do not receive the same number of training steps as the tokens in the original vocabulary, which sometimes causes vocabulary expansion to yield only limited performance improvements. We propose leveraging the semantic equivalence between words in the model's primary language and those in the extended vocabulary to transfer learned representations. In this paper, we introduce MATE, a parameter-efficient vocabulary expansion method that enables effective cross-lingual transfer with 8 times fewer additional parameters per token. MATE reduces the average tokenization compression ratio and delivers improvements on multiple benchmarks. 

---
# Character Information in Language Models: A Prompt-Independent Word Store and a Late, Question-Driven Signal That Grows With Scale 

**Authors**: Pratyush Acharya, Nuraj Rimal, Apar Baral, Habish Dhakal, Piyush Mohan Bhattarai  

**Link**: [PDF](https://arxiv.org/pdf/2611.30094)  

**Abstract**: Linear probes recover character properties of a word from a language model's hidden states, but a probe at the answer position cannot say whether the property was stored with the word or produced because the prompt asked for it. We separate the two with a word-position baseline: the same held-out-word probe applied to the query word's own tokens in a sentence that asks nothing. Across 16 open-weight models from seven families (0.6B--14B) and eleven character tasks, the word tokens encode \texttt{first\_letter}, \texttt{last\_letter} and \texttt{most\_frequent\_letter} above a word-only nuisance floor in all 16 models, and about 70\% of that signal is already present in the token embedding. Asking the question adds a second signal at the answer position. Its size grows with model scale for all four prompt-sensitive tasks (Holm-corrected permutation $p \leq 0.018$, also within model families), and the growth is confined to the late layers: restricted to the first quarter of the network, the slope is indistinguishable from zero. Small models' answer positions hold less linearly decodable signal than their word-level store; large models' hold more. Four controls bound these claims. Measured against the nuisance floor, the question-free signal that a majority-baseline analysis counts as retained largely disappears (\texttt{last\_letter}: $-0.01$). Count tasks become measurable only after length, repetition and query-letter shortcuts are removed by design. A pre-registered test finds that a question placed before the word does not change the word's own encoding beyond what an arbitrary prefix change does. Representation--readout gaps show no robust scaling once probe saturation is accounted for. Pre-registered criteria, a fixed estimator (held-out layer selection, cold-start folds) and all analysis code accompany the paper. 

---
# ESGPT: A Data-Centric ESG Language Model with Unified Instruction and Benchmark Resources 

**Authors**: Jaeyoung Lee, Misuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.30052)  

**Abstract**: ESG analysis requires complex language understanding across diverse sources, including corporate reports, news, and disclosures, for tasks such as ESG relevance detection, action interpretation, impact analysis, and question answering. However, existing ESG datasets are fragmented by task and source, with heterogeneous formats and annotation schemes, limiting their use for training general-purpose ESG LLMs and unified evaluation. In this work, we construct ESG-Instruct, a 30,000-instance ESG instruction dataset built from public ESG resources, and ESGU-Bench, an ESG understanding benchmark covering six representative ESG tasks. Based on these resources, we further develop ESGPT, an 8B ESG-specialized LLM, and evaluate it against diverse open-source and API-based LLMs. Experimental results show that general-purpose LLMs exhibit substantial performance variation across ESG tasks, while ESGPT, despite its relatively small scale, achieves the best performance on multiple ESG tasks. This work provides unified data and benchmark resources for training and evaluating ESG-specialized LLMs. 

---
# SHAKTI: Evidence-Grounded Conversational AI for Nutrition and Health 

**Authors**: Chaitanya Pallerla, Dongyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29988)  

**Abstract**: Large language models (LLMs) answer many diet and health questions but can state unsupported claims and cite sources that do not support them. We present SHAKTI (Smart Health Advisor for Knowledge-based Tailored Interventions), a conversational assistant that answers only from retrieved scholarly papers and exact food-composition data, and evaluate it with retrieval, generation, grounding, efficiency and contextawareness metrics. We indexed 484,368,086 OpenAlex works, searched exactly on a GPU for evaluation (10,065 NFCorpus, BioASQ, SciFact and PubMedQA questions) and through a compressed 230-million-abstract index in deployment. On 1,000 stratified questions, we compared single-shot, multi-hop, corrective and deep-research retrieval-augmented generation (RAG) with Claude Sonnet. SHAKTI adds conversational query rewriting, USDA FoodData Central lookups, source-link checks and online verification of quotes and numbers. It was tested in 36 scripted conversational turns, a 248-question memory-depth test and a 38-question load test. Claims were checked by a Claude Opus judge against exactly the text the system saw. MedCPT reranking of hybrid retrieval gave the best MRR (0.21) and MAP@100 (0.14). Across existing RAG methods, faithfulness was 0.95-0.96, claim hallucination 0.5-0.8% and context precision up to 0.98; the deep-research agent answered 73.2% of BioASQ questions correctly. In conversation, SHAKTI reached a context-awareness score of 97.1%, faithfulness 95.5%, 1.9% unsupported claims, no contradicted claims and 229/229 working source links; it resolved references to the first question after 16 unrelated questions. An answer cost 14,716 tokens (US$0.06) with a median latency of 24 s. Large-scale retrieval with agentic evidence selection and mechanical verification yields auditable, low-hallucination nutrition answers. 

---
# How Relations Impose Embedding-Space Structure 

**Authors**: Lachlan Cowley, Jey Han Lau, Yulia Otmakhova, Eduard Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2611.29986)  

**Abstract**: Language models are known to encode simple relations, such as synonymy, in the geometry of their representations. It is unclear whether they also encode complex patterns of relationships such as those that drive analogy. We present a simple method, with no learned parameters, based on comparing the internal structural arrangements between disparate domains, to test this. Using our method we find over multiple models that representations are arranged consistently to encode complex relationship similarities. Models align distant representation clouds so that facets of meaning are easily comparable under simple mappings. Our work extends mechanistic accounts of representation space to the relative arrangement of representations that may be only weakly related to one another, showing that representation geometry is far richer than previously thought. Code and data: this https URL 

---
# Language models seek help despite guaranteed score losses in question answering 

**Authors**: Jiahao You, Ziye Jia, Chao Dong, Fuhui Zhou, Qihui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.29981)  

**Abstract**: At the highest waiting cost in a multiple-choice question-answering task, all 7 tested open-weight language models still consult an expert even though a correct expert answer earns fewer points than an incorrect independent answer. Pooling over expert accuracy and fixed-fee combinations gives help-seeking rates of 20.8% to 94.1% across models. We test 2,291 questions across 54 combinations of expert accuracy and consultation costs, comparing observed decisions with an expected-score baseline based on independently measured and calibrated model confidence. Help-seeking rates vary across conditions with the same expected consultation score. Fine-tuning reduces decision regret by 59% to 87% on held-out questions within the training range but increases inconsistency at matched expected consultation scores. In a separate task, external routing improves scores by 0.180 to 0.273 points per question with candidate answers held fixed. 

---
# When 1+1 $\neq$ 2: Benchmarking Civil-Time Reasoning in Large Language Models 

**Authors**: Melika Shirian, Kianoosh Vadaei, Arshia Hemmat, Mohammad Hossein Shaker Ardakani, Ali Mamanpoosh, Matin Moqadas, Amirmasoud Rismanchian, Afsaneh Fatemi  

**Link**: [PDF](https://arxiv.org/pdf/2611.29972)  

**Abstract**: Calendar arithmetic is non-linear. When 1:59 AM + 1 minute occurs during daylight saving spring-forward, the result is 3:00 AM, not 2:00 AM, as the hour is skipped. We introduce CivilTime-Bench, a benchmark for civil-time reasoning covering daylight saving transitions, non-hour offsets, date-line crossings, and ISO week boundaries. With 60,011 training and 10,500 evaluation examples under strict output-contract evaluation, we benchmark seven LLMs (six instruction-tuned models and Llama-3.1-8B base) and find: zero-shot exact match varies substantially across models, contract-aware prompting often improves format compliance but yields inconsistent accuracy gains, few-shot in-context learning provides model-dependent improvements, and fine-tuning achieves 74.6--86.6% EM on synthetic evaluation but only 36.6--45.2% on HumanTime-500 for models with valid results on both splits, highlighting substantial limitations in generalization. We release the benchmark and code. 

---
# The Answer Is Not the State: Trajectory-Level Corrigibility in Diffusion Language Models 

**Authors**: Junchi Liao, Wentao Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.29937)  

**Abstract**: Diffusion language models generate tokens at many positions on a shared canvas, making unfinished responses appear revisable. We find a counterexample: after the final answer is reopened, previously generated rationale can reconstruct the old decision despite newly available evidence. We isolate this effect by comparing reopening the answer alone with reopening both rationale and answer, and by exchanging rationales between paired trajectories. On a 150-question source-disjoint benchmark, reopening both regions makes decisions follow the evidence currently present--whether correct or deliberately corrupted--across Dream, LLaDA, and iLLaDA. On a new 100-question free-form short-answer evaluation, the same effect remains strong in LLaDA and iLLaDA but is negligible in Dream. Coherent transplanted rationales steer rebuilt decisions, while token-level attribution improves which positions to reopen without producing a compact repair. Corrigibility therefore depends on retained rationale state, not answer openness alone. 

---
# Training-Free Interventions for Hallucination in Large Language Models 

**Authors**: Can Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29909)  

**Abstract**: A frozen language model can give a wrong answer because a fact is unavailable, a reliable passage is ignored, or an unsupported claim survives generation. We review 116 works on inference-time hallucination intervention and its foundations. We trace the development from retrieval and controllable generation to methods that act on input evidence, token scores, internal representations, completed claims, or the decision to answer. For each family we compare its source of evidence, model access, preparation cost, and the task on which a correction can be established. For activation steering, we distinguish estimating a subspace from applying a state-dependent edit and separate interventions on the prompt from those on the generated prefix. Published comparisons show why faithfulness to a source, factual correctness, informativeness, and answer coverage must be measured separately. Applications in time-sensitive question answering, document synthesis, and visual grounding identify when source authority, evidence availability, or capability preservation limits a reported gain. 

---
# Escriba-OCR: A 1B Vision-Language Model for Error-Preserving Transcription of Handwritten Portuguese Essays 

**Authors**: Tardelli Ronan Coelho Stekel  

**Link**: [PDF](https://arxiv.org/pdf/2611.29874)  

**Abstract**: Feedback tools and research on learner writing need transcriptions of handwritten essays that keep exactly what the student wrote: misspellings, line breaks, struck-through words and insertions. Vision-language models (VLMs) read handwriting well, but they also tend to rewrite it into correct language. We measure this on BRESSAY, 1,000 handwritten Brazilian Portuguese essays, with a benchmark that adds error preservation, invented errors, edit-markup precision and recall and line structure to the character error rate (CER) and uses paired bootstrap confidence intervals. We restrict the study to open-weight models that schools can run on their own hardware. Of 22 open baselines, the best, Qwen3-VL 235B-A22B, reaches 5.6% CER on the 199 test pages; none reproduces the edit markup reliably, and every VLM replaces 39-78% of the words absent from a Portuguese dictionary with dictionary words. We fine-tune LightOnOCR-3-1B on the 647 training pages in 85 minutes on one GPU. The resulting Escriba-OCR-1B reaches 2.06% CER, 3.5 points below the best baseline (95% CI [-4.2, -2.9]); under the ICDAR 2024 competition metric it reaches 2.1% CER, below the best published page-level results (2.2-2.4%). It recovers 67% of insertions and 77% of illegible deletions, and reproduces the line structure of 97% of the pages, but still normalises 60% of likely misspellings. Three seeds agree within 0.04 CER points, and quantised to 4 bits the model runs on an ordinary CPU at about one minute per page. On 40 letters of another corpus it matches its base model (without that model's appended comments) but misreads unseen words more often; weight interpolation recovers half the gap. A teacher's review finds that 45% of the judgeable apparent corrections are errors in the BRESSAY ground truth, so dictionary-based scores likely understate preservation. We release the model, the benchmark code and the review data. 

---
# GlobalHRBench: Does Changing the Question Language Change the Legal Answer? 

**Authors**: Yuding Wang, Yuhang Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.29852)  

**Abstract**: Language matters when deploying LLM agents across borders, especially in law, where the governing rules must hold across working languages. Does changing the language of a legal question change answer accuracy? We introduce GlobalHRBench, pairing 240 employment-law problems across eight jurisdictions and eight languages while fixing facts, applicable law and required conclusions. We evaluate nine models on all 1,920 prompts in independent, closed-book, single-answer trials, scoring legal correctness, completeness and answer-language compliance. GPT-6.1 Sol leads at 76.2% strict passes, with a 5.8% relative best-to-worst language decline, compared with 42.2% for DeepSeek V4.1 Flash. Kimi K3.1 Pro and Claude Sonnet 5.5 reverse order between English and Chinese. A four-model study on 16 paired families compares closed-book answers with governing evidence, added cross-jurisdiction distractors and removal of a governing passage. Governing evidence corrects legal errors and shortens reasoning on recovered answers, yet recognizing an evidence gap does not ensure correct application of the rules that remain. These findings motivate multilingual training for accurate, efficient and evidence-supported answers, and deployment evaluation in the user's working language. 

---
# The Temporal Residual: Fast and Effective Retrofitting of Latent Feedback into Pretrained Language Models 

**Authors**: Ziyang Cai, Xingyu Zhu, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2611.29843)  

**Abstract**: A transformer language model computes a deep hidden state for every token but uses it only to sample that token. Existing latent-feedback designs, which feed hidden states back across tokens, are mostly built in during pretraining and gain little when added to a pretrained model. We introduce the temporal residual, a second residual stream that every sublayer reads and writes and that is carried from each token to the next. It adds 2,752 parameters to a 4.2B-parameter model and starts exactly at the pretrained network. Retrofitted onto Qwen3.5-4B-Base with 5B tokens of training, it beats a standard model trained on the same data by 7, 13 and 5 points on AIME 2024, AIME 2025 and AMC 2023, writes shorter solutions, and stays ahead at matched training compute. The separate stream matters: prior designs that feed state into the existing stream recover at most 40% of the gain when retrofitted. Validation loss can mislead: the recipe with the lowest loss gains nothing at decoding. Trained with filler tokens in place of reasoning, the model reasons in its hidden state: its GSM8K accuracy with no visible reasoning rises from 21% to 40%, and falls back when the carried state is zeroed. 

---
# ComplexIF: A Challenging Multi-Turn, Multi-Constraint Instruction-Following Benchmark 

**Authors**: Zhichao Xu, Bingsen Chen, Yawei Wang, Tao Li, Haoyu Wang, Dan Roth, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2611.29775)  

**Abstract**: As frontier LLMs saturate existing instruction-following benchmarks, measuring further progress requires harder tests that remain faithful to real use. In practice, users often combine several constraints, add them at different turns of a conversation, and expect all of them to hold in the assistant's response. We introduce ComplexIF, a challenging benchmark for this setting, together with the automatic pipeline that generates it. The pipeline expands seed requests into multi-turn conversations, weaves constraints into their turns, and keeps only tasks that a strong model can solve, so difficulty can be raised by adding constraints or turns while every task stays solvable. ComplexIF's two splits test complementary sides of instruction following: a verifiable split of format constraints, checked by programs, and a rubric split of semantic instructions, such as a tone or an audience, checked with an LLM judge. Across 14 proprietary and open-weight models, even the strongest satisfies every constraint on fewer than half of the verifiable tasks, and no model leads on both splits. 

---
# Sánchez, Feijóo, or Abascal? Speaker Attribution and Ideological Congruence in LLM Personas Reading Posts on X of the Leaders of the Three Main Parties of the 29N General Election in Spain 

**Authors**: Víctor Yeste, Candida Maria Greco, Paolo Rosso, Andrea Tagarelli  

**Link**: [PDF](https://arxiv.org/pdf/2611.29765)  

**Abstract**: Source cues are central to political communication, yet reception studies with large language model (LLM) personas as synthetic audiences have not tested whether personas respond to who speaks and not only to what is said. With Spain's general election of 29 November 2026 in mind, we run a speaker-attribution experiment using 1,075 anonymised X posts by the leaders of PSOE, PP and VOX on four topics (the economy and employment, household costs, immigration and corruption) and 791 distinct value-grounded, LLM-generated personas. Each post was assigned to 300 topic-selected persona profiles, 150 grounded in the Inglehart-Welzel cultural map and 150 in Moral Foundations Theory, under four conditions: no author, Sánchez, Feijóo or Abascal, yielding 322,470 complete persona-post quartets. Naming any leader lowered support for identical text, with the penalty for Abascal nearly twice as large as those for Sánchez and Feijóo. The penalty follows the persona's elicited left-right self-placement (ideological congruence): left personas spare Sánchez and right personas spare Feijóo, although ideology was never in the message-evaluation prompt. Corruption posts were penalised under every name, whereas on other topics the ideologically closest leader generally produced little or no penalty. The leader order matches public opinion in 48 CIS barometers (2022-2026) in all three groups and the sign of all twelve differences, but fits barometers before the model's 2024 knowledge cutoff better than recent ones. Our findings demonstrate that LLM personas evaluate political speakers through systematic, value-congruent priors: however, while they replicate human candidate rank-orders across survey barometers, this alignment is ordinal rather than quantitative, highlighting their utility for controlled source-effect experiments though not as substitutes for empirical public opinion surveys. 

---
# Refine, Don't Reinvent: RASA-Bench for Source-Grounded Hindi Prose Refinement 

**Authors**: Abdur Rahman, Pratham Singla, Mohammad Saifullah, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2611.29750)  

**Abstract**: For a producer of Hindi creative prose, drafting is cheap and editing is not: a machine-written episode, story, or essay reads as under-written and flat, laced with translationese, and it sometimes quietly alters what the source says. RASA-Bench targets that editing task across three Hindi prose forms, measuring how well a model refines an under-written AI draft, a task distinct from writing from scratch. Across 2,050 drafts and eight refiners (16,400 rewrites), a four-layer protocol credits a rewrite only after it passes a faithfulness check against its own source, then ranks the survivors by holistic editorial preference through per-form Bradley-Terry fits whose centered coefficients are standardized before equal-weight averaging. Staying faithful proves to be a decisive test that behaves differently by form: gate-failure rates range from 9.1% on short stories to 29.5% on audio fiction, and the drift that disqualifies a rewrite shifts from events and identity in narrative to claims and evidence in essays. A 600-case human audit puts gate agreement at 0.925 (0.91 to 0.94 across forms), whereas subjective preference is form-sensitive and least settled for essays, among human readers no less than the model judge. The benchmark, evaluation protocol, and Hindi prose diagnostics are available at this https URL. 

---
# Where Do Fact-Checking Errors Arise? Separating Retrieval, Search Control and Judgement in Evidence-Grounded Verification 

**Authors**: Martin Pavlíček, Martin Kotyrba, Eva Volná, Tomáš Filip, Petr Sosík  

**Link**: [PDF](https://arxiv.org/pdf/2611.29697)  

**Abstract**: An evidence-grounded fact-checker can fail because it does not find the evidence or because it does not use it; end-to-end accuracy does not tell the two apart. We present a protocol that follows the reference evidence from the store to the verdict and separates retrieval, search control, judgement and verdict validation. We assess 293 claims from FEVER, FEVER 2.0 and AVeriTeC with one open-weights 9B model under two evidence conditions: small stores of annotated evidence, where discovery is easy, and the full source collections. In the control condition a single hybrid retrieval pass delivers complete reference evidence for 226 of 228 evidence-bearing claims, and adaptive controllers and a tool agent spend two to nine times as many model calls without a consistent gain. Of 34 decisive claims labelled as lacking evidence, 33 have the complete evidence in the packet: in 15 the judge abstains, and in 18 validation rules overturn a decisive verdict. On AVeriTeC these rules lower single-pass accuracy by six points, to 65.0% (balanced accuracy 57.6%). Ten judges from eight model families also show abstentions and validation conversions on fixed packets. Against the full stores hybrid single-pass accuracy falls by 17 to 29 points. Wikipedia evidence recovery declines; AVeriTeC recovery changes from annotated answers to cited pages and is measured separately. No controller recovers the accuracy loss. The agent also shows no resolved gain on 714 further Wikipedia claims. Evidence recovery, judgement and validation should be measured separately before an error is attributed to search. 

---
# Grammar Sparsity Sets How Transformers Learn Random Languages 

**Authors**: Edgar Andres Hernandez Moreno, Francesco Cagnetta  

**Link**: [PDF](https://arxiv.org/pdf/2611.29692)  

**Abstract**: We study language acquisition by transformers trained on the Random Language Model: a synthetic hierarchical grammar with random production rules, whose sparsity is controlled by a single parameter and whose conditional entropies are known exactly. On depth-two grammars, we find that sparsity sets both the order and the speed of learning. Dense grammars are learnt one level at a time: a token is predicted from its sibling after $t\propto V$ steps and from the full context only after $t\propto V^{3}$, so the two stages are separated by a plateau that widens with $V$. Sparse grammars are learnt all at once, at $t\propto\sqrt V$, with a smaller exponent than even the first stage of a dense one. The exponents $1$ and $1/2$ already appear at depth one, where we derive them for a linear model trained with Adam. 

---
# Do More Roles or Larger Models Improve LLM Fact-Checking? A Controlled Study on Fixed Evidence 

**Authors**: Martin Pavlíček, Martin Kotyrba, Eva Volná, Tomáš Filip, Petr Sosík  

**Link**: [PDF](https://arxiv.org/pdf/2611.29687)  

**Abstract**: A fact-checking system can spend more computation on a claim by using a larger model, by asking the same model several times, or by a multi-role Tribunal: two roles propose quoted findings, code checks the quotations, an auditor that does not see the proposed relation judges each one, and code derives the verdict or abstains. We compare these options on identical fixed evidence, without retrieval: 300 AVeriTeC claims, each with its annotated question--answer items mixed with items written for other claims, read by judges of four open-weights models from three families, a vote of five, and six assignments of models to the Tribunal's roles. The 9B judge was right on 186 claims and the 27B judge of the same family on 156; the larger judges declined to decide far more often and were no more accurate when they decided. The vote reached 196 and the 9B judge with its verdict replaced by Tribunal certificates 195, at a similar cost; the Tribunal alone reached 153 to 164, because it abstained on almost half of the claims. A hearing's outcome followed the auditor's model and was nearly indifferent to the proposers'. A larger auditor, or one of another family, made certificates neither more often correct (77% to 81%) nor more independent of the judge. As a replacement, certificates improved every judge by 3 to 10 points and overturned four to five in a hundred of its correct verdicts. With the annotated evidence withheld, judges and Tribunal still decided on 7% to 14% of the claims. 

---
# MERIT: Experience-Guided Memory for Long-Horizon Persona-Specific Persuasive Dialogue 

**Authors**: Rohan Kirti, Akash Ghosh, Pratik Amrit, Aryan Vats, Niladri Ghosh, Sriparna Saha  

**Link**: [PDF](https://arxiv.org/pdf/2611.29679)  

**Abstract**: Effective long-horizon persuasion in recurring goal-oriented settings requires more than factual memory: it must track how a user's trust, negotiation behavior, and strategic receptiveness evolve across interactions. Existing persuasive dialogue systems are largely single-session, static, and unimodal, while cross-session memory systems, though temporally aware, retain only factual content and overlook persuasion-specific dynamics. We propose MERIT (Memory-augmented Experience-Rationale for Iterative Temporal persuasion), which pairs a graph-based memory of four persuasion-specific memory types with a per-persona experience rationale library that records which strategies succeed for each user and why. Together they enable adaptive persuasion through prompt composition alone, with no parameter updates. To support this setting, we introduce PERSIST (PERSona-driven InSurance Interactions over Time), a dataset of 60 curated personas spanning 10+ years of motor insurance renewal interactions, with probabilistic life events that shift negotiation behavior and receptiveness across sessions, verified through strict human-in-the-loop annotation. MERIT surpasses memory and experience-based baselines on every metric across two policy backbones, reaching persuasive-appeal scores of 4.10 and 4.74 (out of 5). 

---
# Position-Skipped Distillation of Sparse-Attention Indexers: Train Short, Serve Long 

**Authors**: Fu Dai, Haodi Quan, Kamal Taha  

**Link**: [PDF](https://arxiv.org/pdf/2611.29664)  

**Abstract**: Top-k sparse attention relies on an indexer, a lightweight per-layer module that ranks cached tokens so attention covers only the k highest-scoring ones. Indexers are usually distilled from dense attention at the full serving length, so every training step pays for long-context dense attention. We show that when an indexer is distilled on short sequences, its failure to generalise to longer contexts is driven mainly by rotary positional encoding (RoPE), not by degraded content matching. We propose position-skipped distillation: the indexer and the frozen teacher see short sequences whose position IDs contain a randomly placed gap, so neither ever processes more than the short training length. On DeepSeek-V2-Lite and Qwen3-30B-A3B, an indexer distilled on 8K tokens this way matches a 32K-distilled indexer to within about 0.5 perplexity up to 32K, and on DeepSeek-V2-Lite matches a 64K-distilled indexer at 64K with roughly 6x less training compute per token. On Llama-3.1-8B it trails the stronger 32K run by 0.6 perplexity, and at 64K simple position compression is often as good, so the main gain is training cost rather than quality. We also quantise indexer keys to 3-4 bits without retraining and score them with a fused integer kernel: at 4 bits, perplexity changes by at most 0.05 and the scan is 1.6-2.2x faster than half precision. In a current serving engine, the indexer speeds up decoding only for long contexts with multiple concurrent streams. 

---
# How Reliable is AI-Generated Audio Description? A Multilingual Expert Evaluation 

**Authors**: Annette Rios, Lukas Fischer, Sarah Ebling  

**Link**: [PDF](https://arxiv.org/pdf/2611.29658)  

**Abstract**: Audio description (AD) makes visual media, such as video, accessible to blind or low-vision (BLV) audiences, but the high cost of professional production limits coverage. Due to recent advances in AI, numerous approaches to automatically generate intermediate AD scripts and/or final AD audios have been proposed. We present an expert evaluation of AI-generated AD scripts for movies in Italian and German that compares two AI systems, a GPT-4o- and a Qwen-based pipeline, against human-authored descriptions. Professional audio describers rated description quality and annotated errors using a detailed error taxonomy that includes content, character, fluency and coherence errors. The results show that while GPT-4o outperforms the Qwen models, both AI systems still fall short of human professional quality: the generated descriptions show substantially higher error rates and lower quality ratings. Based on the error annotations, we identify two key areas that require improvement: 1) character references, which includes both issues with identification and coherent naming, and 2) content errors and hallucinations, where the AD misrepresents or invents events. These error types are particularly critical, as they directly affect the viewer's ability to follow the narrative. 

---
# Which Loopholes Get Found? Predicting Reward Hacking of LLM Judges Before RL Training 

**Authors**: Xuanyi Lyu, Yuezhou Zhao, Qing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2611.29651)  

**Abstract**: A policy trained against an LLM judge can learn to exploit its loopholes. Static probes show that such judges accept degenerate answers like a lone "{" or "Thought process:", but not which flaw a policy will find. In GRPO the first loophole to be exploited can be predicted before training from each pattern's exposure, the product of the starting policy's propensity to produce it and the judge's false-positive rate (FPR). To test exposure causally, we seed the patterns into fine-tuning at three doses in a Latin square. Across 36 GRPO runs with seven judges, the first exploited pattern had the highest exposure in all 12 runs with exploitation and the highest FPR in only 6. Predictions recorded before training held in three of four runs with one pattern masked by rule. When only two patterns were seeded, at 3% to 3.8% of the fine-tuning data and chosen so that exposure disagreed with FPR alone or propensity alone, exposure named the first exploited pattern and its onset window in all six runs. Exploitation occurred only on hard prompts, where an accepted exploit gets a large group-normalized advantage. Auditing exposure before RL therefore identifies the loophole a policy will likely exploit first. 

---
# Reducing Authorship Attribution Risk in Student Texts Through Style-Conditioned Rewriting 

**Authors**: Li Siyan, Zhen Xu, Jiayi Chen, Shopnil Shahriar, Alessandro Sayad, Xuanming Zhang, Renzhe Yu, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.29642)  

**Abstract**: Building NLP applications for education is constrained by data scarcity due to student privacy concerns and data regulations. To address this, we propose a style-aware approach to rewriting student forum posts that retains analytical value while reducing identifiability. Our method isolates content, structure, and style as separate conditioning channels and aggregates over student-disjoint exemplar partitions within style clusters, extending Differentially Private In-context Learning (DP-ICL) with cluster-restricted retrieval. Across three educational NLP datasets, our approach achieves high style fidelity, preserves downstream task performance, and reduces authorship attribution risk relative to the original student-written texts, including against a dedicated authorship-attribution model. 

---
# Diversity in Unity, Theory in Practice: Hierarchical Multitask Benchmarks for Chinese Minority Languages 

**Authors**: Yijie Li, Xi Cao, Yuan Sun, Quulgan Minggad, Abdulla Ablikim, JiaQingCaiWang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29635)  

**Abstract**: Despite the rapid advancement of LLMs, their performance on linguistically and culturally diverse minority languages within a unified national context remains underexplored. We present CMiLBench, a collection of hierarchical multitask benchmarks designed to translate theoretical notions of "diversity in unity" into practical evaluation for three representative Chinese minority languages: Tibetan, Mongolian, and Uyghur. CMiLBench comprises 24,663 instances across 5 difficulty levels and 17 tasks spanning foundational ability, cultural specificity, and safety alignment. We adopt existing dataset adaptation, minority knowledge construction, and high-resource benchmark translation to construct CMiLBench. We assess 14 state-of-the-art commercial and open-source LLMs with a hybrid framework that integrates automatic metrics and LLM-as-a-Judge scoring. The comparative experimental results reveal the gap between theoretical capability and practical utility. CMiLBench serves as a foundational and scalable evaluation resource to bridge the digital language divide and promote the informatization and intelligentization of low-resource Chinese minority languages. 

---
# Explainable Emotion and Mental Health Reasoning in Large Language Models Using Theories of Emotion 

**Authors**: Tadesse Destaw Bela, Iqra Ameer, Helena Gómez Adorno, Alexander Gelbukh, Eusebio Ricardez Vázquez, Olga Kolesnikova, Grigori Sidorov, Seid Muhie Yimam  

**Link**: [PDF](https://arxiv.org/pdf/2611.29628)  

**Abstract**: Advancements in Large Language Models (LLMs) have enabled a wide range of applications. However, their ability to simulate human-like behavior and reasoning remains unclear. While emotion detection is a better-established Natural Language Processing (NLP) task, existing approaches primarily focus on directly classifying text into predefined emotion classes, often overlooking the appraisal process that explains why an individual experiences a particular emotion in a given situation. In this work, we propose Psychology-based Chain-of-Thought (PsyCoT) reasoning, a theory-driven prompting strategy grounded in the Schachter-Singer and Cognitive Appraisal theories of emotion. We propose and evaluate PsyCoT for subjective NLP tasks, including emotion analysis, emotion intensity prediction, stress analysis, and depression severity identification. Our evaluation covers four levels of classification complexity: binary, multiclass, multi-label, and ordinal scaling, spanning 11 languages. We assess the performance of five state-of-the-art LLMs across these selected subjective NLP tasks. Experimental results demonstrate that PsyCoT prompt variants improve the standard chain-of-thought (CoT) prompting across the evaluated tasks. Across all evaluated tasks, PsyCoT yields F1 score gains ranging from 0.5% to 28.1%, depending on the model, task, and language configuration. 

---
# KernelShift: Benchmarking Kernel Optimization Across Evolving Hardware 

**Authors**: Shuaiwei Huang, Weihua Du, Allen Wu, Ivan Fox, Manupa Karunaratne, Yiming Yang, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2611.29590)  

**Abstract**: Efficient AI computation depends on hardware that evolves over time, shifting the strategies needed to implement performant code. For CPU kernels, a key evolving factor is the set of available architectural extensions, which expose specialized hardware instructions that change the techniques used to optimize kernels. We study how effectively coding agents use these new, potentially unfamiliar extensions to write fast code. We introduce KernelShift, a benchmarking pipeline and a set of environments for evaluating coding agents in kernel optimization across evolving CPU architectures. Its agent-assisted, human-verified pipeline turns code repositories into benchmark instances, and its environments let agents optimize code with feedback from the target hardware. Using this pipeline, we construct 99 kernels from real-world, actively developed repositories (ncnn, this http URL, Arm SIMD Loops, and KleidiAI), including inference workloads derived from real model executions. KernelShift supports incorporating new kernels and hardware configurations as libraries and processors evolve. We evaluate agents with proprietary and open-weight models across progressively newer architectural extensions (Arm Neon, SVE, SVE2, and SME2), examining how performance changes as hardware evolves. On established extensions, agents beat the auto-vectorized compiler everywhere (2.4-19.8x) and exceed general-purpose libraries such as ncnn (getting up to 2.74x over its Neon kernels), but fall behind Arm's hand-tuned KleidiAI kernels (0.70x). In end-to-end inference of two LLMs, agent-written kernels achieve a 1.3-1.6x prefill speedup over this http URL's fast path. However, current agents do not yet reliably adapt to new extensions that require unfamiliar programming models, such as SME2's matrix tiles. 

---
# Single-Coordinate Activation Steering for Controllable Social Stance in Language Models 

**Authors**: Naman Chawla, Chhavi Dhiman  

**Link**: [PDF](https://arxiv.org/pdf/2611.29578)  

**Abstract**: Steering one neuron of a small language model can impose a topic or a quirk without training, yet replications show that many published single-neuron recipes are fragile or nonspecific. This paper asks whether an interpersonal stance can be controlled in this manner. The stance studied is a distant crush, a chatbot character that is guarded, intermittently warm and inconsistent. In Qwen3.5, 4B, contrasting activations on hand-written guarded and eager replies identified neuron 7515 of layer 9 as a candidate for controlling this stance. Increasing this neuron activation made the characters replies more distant. Three language models rated this change as stronger than the effects of changing randomly selected neurons, and a blind human comparison supported the finding. Changing the interventions direction between messages also shifted replies between warmer and more distant styles. This suggests that a language model conversational stance can be personalized without finetuning. 

---
# LM2 Compiling a Language Model into Another Model Weights 

**Authors**: Naman Chawla, Chhavi Dhiman  

**Link**: [PDF](https://arxiv.org/pdf/2611.29577)  

**Abstract**: Deploying a conversational model and a specialist together could avoid maintaining a separate specialist service. We present LM2, which embeds a trained story generator into a larger conversational model. The larger model handles the conversation and invokes the generator for story requests. We constrain the generator arithmetic during training so its calculations can be compiled directly into added model layers. We compare this approach with QwenPU, which executes the stories260K language model on an embedded processor. LM2 requires one host token per generated character, compared with approximately 25,000 for QwenPU. However, its story generator predicts held out text less well, i.e. 1.145 bits per character versus 0.894 for stories260K, where lower is better. The Qwen3.5, 0.8B implementation occupies 18.1 GB and generates the same characters as a standalone integer implementation of the inner model in the tested runs. These results illustrate a tradeoff between host-token cost and text-prediction performance. 

---
# CompareDR: Benchmarking Deep Research Reports from Individual Quality to Comparative Merit 

**Authors**: Wanli Wu, He Zhu, Yue Xu, Haolin Ren, Yan Chen, XunLiang Cai, Fan Yang, LingChuan Liu, Shizhu He  

**Link**: [PDF](https://arxiv.org/pdf/2611.29549)  

**Abstract**: Deep research systems can produce reports with different evidence, analyses, and conclusions for the same query. A reader deciding which findings to rely on needs to know how well each report meets the task requirements and which differences matter to the answer. A score for a single report can summarize its quality, but it does not identify which contributions make one report more useful than another. We introduce CompareDR, a benchmark that combines task-specific pointwise audits with guideline-based pairwise comparisons. We use passage-linked report representations to locate relevant differences and selective evidence verification to check claims that could affect the judgment. The benchmark comprises 40 Chinese and English tasks from three existing benchmarks, 360 reports from nine systems, 1,099 human-reviewed evaluation criteria, and human preferences elicited through blinded comparisons of 80 report pairs. The hybrid evaluator matches 69 of these 80 preferences (86.25%, Cohen's $\kappa=0.746$). The benchmark supports both comparisons of research systems and examination of the evidence and analyses behind those judgments. 

---
# Estimating Silenced Tokens in Quantized LLMs 

**Authors**: Giuseppe Pappalardo  

**Link**: [PDF](https://arxiv.org/pdf/2611.29536)  

**Abstract**: Quantization is the standard way to shrink large language models, and it is usually judged by perplexity and task accuracy. We ask a different question: after quantization, which words can the model still say at all? A token is silenced when no hidden state can make it the top choice of the output layer, so a greedy decoder can never produce it. We give a simple, scalable way to estimate how many tokens a quantized output head silences, combining three checks: a direction that lets a token win (so it is safe), an upper bound on how close a token's row is to being trapped inside the others, and an exact match that proves a token can never win. Each count is therefore a certified range: exact matches give its lower end and flagged tokens can only raise its upper end. In nearly all settings we report down to 3 bits the two ends coincide, so the counts are exact. Across three open models we find that quantization silences tokens: on large-vocabulary models it already appears at 8-bit precision, up to a few hundred previously reachable tokens on a 1B-parameter model, and grows as precision drops, while a small-vocabulary model resists until 3 bits. The loss is far from uniform across languages: on a multilingual vocabulary the silenced tokens are overwhelmingly non-Latin (Korean, Chinese, Thai, Arabic and Hebrew), even though Latin tokens dominate the vocabulary. Silencing is thus a measurable, language-sensitive side effect that standard averages miss, and it should be reported alongside them. 

---
# Meddies-PII: A Multilingual Framework for Personally Identifiable Information Extraction in Clinical De-identification 

**Authors**: Le Linh Uyen, Christian Hoang, Ho Dinh Tri, Huy Hoang Ha  

**Link**: [PDF](https://arxiv.org/pdf/2611.29491)  

**Abstract**: Multilingual PII extraction in clinical text is limited by scarce governed data and by synthetic corpora whose construction is thinly documented. We present a large multilingual dataset covering seventeen languages, including seven Southeast Asian languages, and nine PII labels. We also document a detailed generation pipeline and release the complete prompts, catalogues, checks and label maps. A compact 350M-parameter BIOES token classifier trained on the resource achieves the highest score on all fifteen public benchmarks, significantly ahead of the strongest of four public systems on fourteen of them, with a mean exact-match F1 of 0.827 against 0.672 for the strongest system. We also tested it, without adaptation, on multilingual benchmarks, including two human-annotated German and Spanish clinical-style corpora, where it is competitive with the best public systems once annotation conventions are aligned, and is best in four of ten MultiGraSCCo languages; the results show that performance transfers across public benchmarks. We also analyse the data and find almost no near-duplicates and no text overlap between training and any evaluation set. The dataset, benchmark, model, generation framework and evaluation code are planned for public release upon acceptance. 

---
# GraphSeer: Multimodal Knowledge Graph Relational Image Retrieval with MLLMs 

**Authors**: Linyu Li, Zhi Jin, Yuanpeng He, Haoran Duan, Dongming Jin, Wenchun Jing, Xuan Zhang, Gadeng Luosang, Nyima Tashi  

**Link**: [PDF](https://arxiv.org/pdf/2611.29470)  

**Abstract**: Multimodal knowledge graphs (MMKGs) store entities with photos and names and link them by facts. MMKG methods predict links between known entity IDs, and image retrievers, from CLIP to MLLM embedders, return entities that look like a query photo or match its description. Neither answers a common question about a new photo: which graph entity is related to the one shown, such as the team of a pictured player or the city of a landmark. We call this knowledge graph relational image retrieval (KRIR): given a photo and a natural-language relation, retrieve the related MMKG entity, which rarely resembles the photo, whether the depicted entity is in the graph (known) or not (unseen). We build three benchmarks from YAGO, Wikidata and in-the-wild OVEN photos. The task hides two hops: recognising which graph entity is shown, and recalling what it is connected to. GraphSeer bridges them inside one multimodal LLM: it links the photo to the graph, reads the facts of the best match, and retrieves with the hidden state at which the model is about to say the answer. An unseen entity has no facts of its own, but every candidate answer does, so a listwise verifier reads the other end of the graph: it re-ranks the top candidates by their facts and photos in one forward pass, and the facts matter far more than the photos. With Qwen3.5-27B, GraphSeer improves MRR on known entities by 24.2-70.0% over an MLLM embedder fine-tuned on the same data with the same budget. When it also fuses and verifies the embedder's candidates, it improves MRR on unseen entities by 12.5-25.6%, and on known entities it beats KG-completion models given the ground-truth entity ID. 

---
# PAWP-TTS: Phoneme-Aware Weight Programming for Compact Bilingual English-Nepali Speech Synthesis 

**Authors**: Firoj Paudel  

**Link**: [PDF](https://arxiv.org/pdf/2611.29455)  

**Abstract**: This paper presents PAWP-TTS, a compact non-autoregressive text-to-speech system for English, Nepali, and English-Nepali code-mixed speech at 24 kHz, with no language model at inference. Deterministic pronunciation knowledge lives in a rule- and lexicon-based bilingual frontend that preserves native Nepali contrasts and avoids Hindi phonology, so the network only models acoustics. Counting every learned inference tensor, including the waveform decoder, PAWP-1.5M (1,490,390 parameters, 5.98 MB) reaches 1.50% English and 1.75% Nepali character error rate under a single ASR recognizer, and 5.04% / 4.02% Nepali / English phoneme error rate (PER) on 600 untouched test utterances. Decoder-centric 281K-parameter models (1.1 MB) stay intelligible across three independent seeds, with equal-budget dense models (5.53 +/- 0.08% NE, 5.34 +/- 0.28% EN PER) outperforming K=2 cross-layer shared-basis models (6.35 +/- 0.11% NE, 6.57 +/- 0.47% EN PER). Fixing a frontend/label version mismatch nearly halved Nepali PER, more than any architectural or weight-level change. The paper then studies weight-space programming: inspection-guided, zero-training weight edits. Under strict controls (validation-only selection, one-shot test, paired bootstrap, matched random and same-parameter gradient baselines), edits show real selectivity (27:1 on word-boundary pauses with 96 weights; zero cross-lingual collateral inside language-gated residuals) but no accuracy gain beyond controls. Two apparent targets proved to be label or phonology artifacts. Code, ONNX graphs, and CC BY 4.0 weights are released. 

---
# Detectable Epistemic Attack Layers: A Coverage Theorem and Standardized Harness for Conversation-History Integrity 

**Authors**: Fahrell Giovanny, Geby Bayuningtyas, Sahrul Mukharom  

**Link**: [PDF](https://arxiv.org/pdf/2611.29449)  

**Abstract**: Conversation history is infrastructure: deployed language models rely on application layers to store and reinject prior turns, and that stored state is writable. We formalize session-level contamination by attack layer, state a coverage theorem, and measure its scope: each deployment-layer verification tier covers exactly its layer and is sound with respect to the others; instruction overrides are not detectable without false positives; and composite vectors evade per-fragment verification. A standardized harness applied to eight mechanism families across 3,249 sessions and three model families makes these claims checkable. First, pattern-based detection pays a measured false-positive price: the classic defenses flag 35 to 50 percent of benign sessions, and the naive refusal floor reaches only 2.4 percent attack-cell adoption while destroying 33 to 100 percent of benign turns, whereas provenance-gated channel separation achieves adoption at or below 1.2 percent with zero false positives and zero utility loss. Second, in the exploratory instruction-override extension, the gates' boundary is a cliff: directives rephrased without shape markers bypass them at control-equivalent rates, 47.7 to 49.5 percent against 0.0 percent for the shaped class, placing that class in the model-internal lane. Third, the deterministic core is storage-layer detection at 126 of 126 per model with zero false positives. External anchors reproduce published attack pressure, with PoisonedRAG falling from 70 to 0 percent under the full stack. For practitioners: authenticate stored history and content provenance with deterministic tiers, scope instruction-layer mitigation to provenance-gated designs whose measured benign cost is zero, and monitor composition at the trajectory level rather than per fragment. All protocols and acceptance criteria are pre-registered; the stack and harness are released. 

---
# VistaTree: Question Answering over Long Multimodal Documents with Image-Derived Summary Trees 

**Authors**: Yexiang Cheng, Xiaochen Zheng, David Tokar, Tatyana Y. Doktorova  

**Link**: [PDF](https://arxiv.org/pdf/2611.29419)  

**Abstract**: Answering questions over long multimodal documents requires locating relevant evidence across pages containing text, tables, and figures. When the document exceeds the model's context window, selective retrieval becomes necessary. We introduce VistaTree, a document question-answering agent that builds a summary tree directly from rendered page images. VistaTree segments documents into contiguous page spans, recursively groups them into a topic hierarchy, and refines sibling summaries to distinguish neighboring sections. At query time, the agent uses node summaries to select relevant nodes and reads the corresponding pages, using hierarchical navigation when the full outline exceeds the context budget. Experiments on MMLongBench-Doc and LongDocURL evaluate answer quality and retrieval cost using a locally hosted vision-language model, and quantify the performance gap between global and hierarchical access to the index. 

---
# HGDNA: Overlapping k-mer Tokenization for Efficient Autoregressive Genomic Language Models 

**Authors**: Xudong Wang, Siwei Cheng, Haozhe Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.29400)  

**Abstract**: The domain of genomic language models (gLMs) has advanced rapidly, with models pretrained on diverse multi-species genomic corpora demonstrating their remarkable capabilities. While the effect of simple nucleotide-level tokenization on long-context modeling is well established, overlapping k-mer tokenization remains underexplored in causal pretraining, despite its rich neighborhood information. Here, we thoroughly revisit overlapping tokenization and introduce HGDNA, a hybrid linear-attention gLM with causal language modeling (CLM) pretraining and a shared token-prediction interface for downstream adaptation, using a species-classification auxiliary task and shared class tokens. HGDNA delivers competitive performance across sequence classification and frozen-embedding evaluations, supporting both short-range and long-range tasks with 30M parameters and 19B pretraining tokens. Code is available at this https URL 

---
# Can Speech LLMs Recognize When Audio Evidence Is Insufficient? A Diagnostic Study and a Pre-Generation Probing Mitigation 

**Authors**: Vladislav Nesterenko, Polina Doronicheva, Olga Golubeva, Polina Sheviakova  

**Link**: [PDF](https://arxiv.org/pdf/2611.29393)  

**Abstract**: Speech large language models are usually evaluated on questions that have answers. We ask what they do when a recording establishes none, and whether the standard way of finding out is sound. Diagnosed on synthesized speech the answer looks clear: Qwen2-Audio fabricates an answer to 92.5% of unanswerable questions, while an ASR cascade reading the same recordings reaches 2.5% hallucination at 6.7% over-refusal, placing the failure in reasoning rather than in perception. The identical protocol on 376 native SQuAD 2.0 questions over human-read recordings removes that conclusion: the cascade hallucinates on 67.7% against the end-to-end model's 69.8%, at a higher over-refusal cost. A paired subset asking the same questions of a synthesized and a human recording of each passage localises the effect in the ASR stage -- the end-to-end model's abstention behaviour does not move with the speaker, the cascade's goes from 11.1% to 65.9%. Reliance on synthesized speech has been named as a limitation of audio benchmarks but not quantified; we quantify it, and show it large enough to reverse a finding. Since no prompt we tested repairs the residual failure, we transfer pre-generation attention probing to the multimodal setting. A probe over frozen prompt representations reaches 0.773 AUROC at predicting which unanswerable questions the model will answer anyway, where semantic entropy over ten sampled generations is at chance: the model fabricates the same answer every time, so sample disagreement cannot see a failure it is confident about. 

---
# Automatic Construction of Persian E-Commerce Dialogue Datasets for Conversational Product Search 

**Authors**: Fariba Nafar, Hassan Naderi  

**Link**: [PDF](https://arxiv.org/pdf/2611.29356)  

**Abstract**: Conversational product search lets users find products through a multi-turn dialogue instead of a single keyword query, but Persian lacks the large conversational datasets needed to train and evaluate such systems; existing Persian e-commerce dialogue resources cover few categories and small catalogs. We present a framework that builds such a dataset automatically from a real e-commerce catalog and release the resulting dataset. The framework crawls 12,926 products in seven categories of the Digikala marketplace, groups them into 170 semantically coherent clusters, and selects the question the assistant asks at every turn with the Regularized Partition Balance (RPB) score. The RPB score accounts for feature coverage, the probability that the user can answer and the number of values, and generalizes entropy-based question selection. An LLM then verbalizes each question plan into a natural Persian dialogue. In a simulation on the full catalog, the RPB score needs 0.11 fewer turns than the strongest entropy baseline when the difficulty of answering differs between features, and is practically equivalent to it when it does not. Using only a free LLM API, we generated 1,000 requirement-driven dialogues covering all 170 clusters, with turn-level feature annotations; automatic checks and an LLM judge confirm their consistency with the catalog, while their realism is lower than that of the first, GPT-4o-generated version. Code and data are publicly available. 

---
# Endpoint-Conditioned Process Value: What an Endpoint-Blind Process Score Adds Once Terminal Status and Cost Are Known 

**Authors**: Chris D Yan, Ken Wang, MarkX Qian, Jin Jiang, Albus Wu, Wenhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29354)  

**Abstract**: Trajectory metrics promise information beyond final-answer accuracy, but no one has measured what a process score adds once an evaluation knows the outcome and the cost. We build that comparison: a sealed consequence needing an artifact neither agent nor judge ever sees, an operational baseline of terminal status and cost, and judge scale and evidence access as experimental factors. On 1,255 trajectories over 158 test-generation tasks in 7 Python repositories, the answer is gated on the judge. Four judges from 4B to 14B, one reasoning before scoring, add nothing an interval separates from zero; a Qwen3 32B judge on identical trajectories adds +0.050 AUC (95% CI [+0.020, +0.081]) and survives a refitting bootstrap ([+0.013, +0.095]). Most of what it adds is recoverable from eight behavioural counters, so a harness should log them (the judge adds +0.020 beyond them). What the judge is shown moves it further: disclosing the terminal verdict inflates its standalone discrimination by +0.103, while reference disclosure and tool-output redaction move it by nothing an interval separates from zero. A process metric reported without the judge's scale and evidence access does not identify a quantity. We release the protocol, the sealed-endpoint harness, and all trajectories. 

---
# From Vanishing Gradients to Recursive Self-Improvement: Evaluating the Most Influential Papers in Large Language Model Evolution (1997-2026) 

**Authors**: Saadh Jawwadh  

**Link**: [PDF](https://arxiv.org/pdf/2611.29303)  

**Abstract**: The rapid proliferation of Large Language Model (LLM) research makes it increasingly difficult to distinguish incremental refinements from true paradigm shifts. This review evaluates 22 of the most influential publications spanning nearly three decades (1997-2026), organized through a causal, chronologically anchored taxonomy centered on the author's birth year (1999). Beginning "before 1999" with how recurrent neural networks (RNNs) forgot earlier context (much like forgetting Chapter 1 by the time a reader reaches Chapter 9), we trace how Long Short-Term Memory (LSTM) networks introduced a "bouncer" mechanism of forget, input, and output gates to selectively preserve long-term memory. We then examine how Word2Vec turned language into vector mathematics (king - male + female ~ queen), why sequential RNNs hit a parallelization wall, and how the Transformer's all-to-all attention weights launched the modern LLM era alongside a quadratic bottleneck. Next, we analyze why scaling parameters alone leaves capacity under-trained unless data scales equally (Chinchilla), how Reinforcement Learning from Human Feedback (RLHF) left alignment superficial, and how that gap compounded into adversarial jailbreaks and single-direction refusal subspaces (J-space), where instructing a model not to think of a restricted concept surfaces it much like the human "red ball" paradox. Finally, we synthesize the current state of AI (2024-2026), encompassing recursive model collapse, autonomous reasoning-model jailbreak agents, machine-speed agentic intrusions, and closed-loop recursive self-improvement (AIDE^2). 

---
# Beyond Omission: Measuring Critical-Information Fulfillment of Large Language Models in Bengali and English 

**Authors**: Avijit Roy  

**Link**: [PDF](https://arxiv.org/pdf/2611.29299)  

**Abstract**: Evaluations of multilingual trustworthiness often ask whether a model leaves out information that a user needs. We examine whether that question is sufficient. Using 157 matched tasks written in English (EN), natural Bengali (BN) and Bengali-matrix Bengali-English code-switching (BN-CS), we scored 1,413 responses from three models (GPT-6.1 Sol, Qwen3.5-9B and Gemma 4 12B) against required information units (RIUs) fixed before any model was run. The scoring separates outright omission from content that is present but incorrect, insufficient or only partly met. The prespecified primary estimand, the domain-equal paired difference in critical omission rate (BN - EN), showed no detectable difference (+0.003, 95% CI -0.014 to +0.015), with omission near zero in every condition. The Bengali deficit appeared in fulfillment instead. Critical coverage was 0.050 lower in Bengali (95% CI 0.022 to 0.079) and strict complete-task success 0.093 lower (95% CI 0.055 to 0.131); both gaps came from the two open-weight models, and Sol showed none. Code-switching recovered part of the loss in RIU coverage (+0.028, 95% CI 0.012 to 0.045). On these data, an omission count alone would have shown no language gap. The automated judge passed prespecified agreement gates after a documented failed first attempt. A blind native-speaker rating of 60 study responses agreed with it on RIU scores (alpha 0.84) and critical RIUs (98% exact) but fell short on Bengali RIU agreement (alpha 0.77) and on the unsupported-content check (93% exact), and a blind check of its reason codes agreed on whether a critical unit was omitted for 79 of 81 units. Data, rubrics, responses, scores and code are released. 

---
# LLM-Based Assistant for Dog Care Guidance 

**Authors**: Abubakar Sadiq Shittu, Venkat Gopu, Arshita Sharma, Jacob Kennedy  

**Link**: [PDF](https://arxiv.org/pdf/2611.29286)  

**Abstract**: This study investigates whether domain grounding through Retrieval-Augmented Generation (RAG) and parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) can enhance the accuracy and reliability of pet care advice generated by large language models (LLMs). We curated a dataset of breed-specific dog care information and used it to develop two systems: a RAG pipeline based on TinyLlama-1.1B-Chat-v1.0, and a fine-tuned LLaMA-3.2-1B model adapted using LoRA. Both systems were evaluated on 15 dog care questions. The RAG model achieved a perfect average score of 2.0, outperforming the fine-tuned model (mean = 1.47). Statistical analyses, including Welch's t-test, Mann-Whitney U, and Wilcoxon signed-rank tests (all p < 0.05), confirmed the significance of the performance gap. Our findings suggest that retrieval-based domain grounding leads to more accurate, consistent, and trustworthy responses than standalone fine-tuning in specialized LLM applications. 

---
# Dual-Side Uncertainty Estimation: Calibrating Data and Model Uncertainty into a Shared Probability Space 

**Authors**: Jiaxuan Gao, Yang Hu, Feihong Lu, Yudi Huang, Qingyun Sun, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.29284)  

**Abstract**: Multimodal user intention analysis infers latent goals from heterogeneous behavioral observations but faces cross-modal input mismatch and output semantic errors. Existing uncertainty estimators focus on input reliability or output confidence and often reduce token uncertainty to simple statistics, overlooking structured dependencies. We propose DUE (Dual-Side Uncertainty Estimation), which jointly estimates input-side aleatoric and sentence-level epistemic uncertainty. From a single generation, Dirichlet perturbations of cross-layer representations produce an ordered token uncertainty sequence. A sequence-aware Normal-Inverse-Gamma estimator uses this sequence to quantify epistemic uncertainty and confidence in its estimate. Evidential multimodal alignment captures aleatoric uncertainty from noisy, incomplete, or inconsistent inputs. DUE calibrates both heterogeneous scores into a shared error-probability space and sequentially fuses them to estimate overall generation risk. Experiments demonstrate strong performance in input-mismatch detection, output semantic-error detection, and overall reliability estimation without repeated generation. 

---
# Seeing Is Not Acting: WARD Diagnoses the Evidence-to-State-to-Action Gap in Longitudinal Clinical Agents 

**Authors**: Chao Ding, Yidong Jiang, Junzhi Ning, Juncheng Wu, Yuhan Wang, Renjie Lu, Minjia Yuan, Gaopeng Wang, Liyang Wang, Pengcheng Chen, Dong Yin, Meiping Wang, Jinru Ding, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.29243)  

**Abstract**: Large language models for longitudinal clinical care must revise their assessments and actions as patient information accumulates. However, benchmarks centered on isolated responses or final task outcomes provide limited insight into intermediate failures in evidence use, state updating, and action selection. We introduce WARD, a benchmark for evaluating how clinical agents translate accumulating evidence into patient-state judgments and timely, safe actions. WARD comprises 4,560 clinical tasks across seven clinical output domains, organized into 240 checkpoints from 40 longitudinal clinical trajectories spanning 95 hospitalizations. Its Longitudinal Track evaluates these tasks under controlled cumulative evidence. Its Interactive Track evaluates end-to-end, tool-mediated clinical management across 271 verifiable evidence items and 141 reference transitions. Across 12 models with scorable checkpoint outputs, evidence recognition consistently outpaces downstream state and action decisions, with linked-evidence nonresponse reaching 74.47--90.90\%. Models also struggle to retain and update information over time, while in the Interactive Track only 11.46\% of executed actions advance a reference transition. These results reveal a persistent evidence-to-state-to-action gap and highlight the need for process-level evaluation of longitudinal clinical this http URL and data: this https URL. 

---
# Edit-Induced Dependency Mismatch: Propagation Omissions, Recovery, and Evaluation Blind Spots in Coding Agents 

**Authors**: Fei Ding  

**Link**: [PDF](https://arxiv.org/pdf/2611.29242)  

**Abstract**: During iterative code modification, coding agents may continue to rely in subsequent reasoning on outdated dependency judgments whose text remains fixed in the context, leading to errors in subsequent reasoning and leaving consumer defects undetected by the original task evaluations, thereby exposing blind spots in the tested benchmarks. We study actual agent execution trajectories collected on benchmark tasks: deepseek-v4-flash runs on DeepSWE and SWE-bench Verified, supplemented by Qwen3.8-Flash cases on FeatureBench. Focusing on edit-induced consumer obligations, a full-corpus blinded human audit of the DeepSWE runs identifies terminal mismatches, including diagnoses whose task scope remains disputed. Following supplementary execution checks, all retained DeepSWE terminal events have execution support. The audit also shows that recovery within a trajectory can coexist with terminal omissions in the same run. The process analysis links the presentation of relevant consumer code to explicit dependency judgments and subsequent synchronization actions recorded in the trajectories. An explicit dependency-review prompt reduces terminal omissions under the tested conditions, supporting re-derivation of consumer obligations after edits. Local replays of selected DeepSWE cases show that synchronizing the diagnosed consumers restores the target behavior while the listed controls continue to pass. Cross-benchmark execution counterexamples show that the original task evaluations do not fully detect edit-induced dependency mismatches: patches pass those evaluations, yet additional execution checks confirm unintended defects along existing consumer paths. These findings support measuring task scores, terminal propagation states, and recovery separately. 

---
# ContextFact: Remembering Facts in Context for Long-Term Conversational Agents 

**Authors**: Jiayu Huang, Yichao Cai, Xinyu Li, Zhen Zhang, Javen Qinfeng Shi  

**Link**: [PDF](https://arxiv.org/pdf/2611.29155)  

**Abstract**: Agents built on large language models must maintain continuity across repeated interactions with users. However, long-term conversations scatter facts across turns and sessions, making it difficult to retrieve the right evidence with enough context. Memory for such conversations must therefore preserve facts with their context while keeping evidence selectively accessible. We first study how memory construction and retrieval-unit boundaries affect answerability and evidence access. Reading the entire memory bank reveals gaps that cannot be resolved by retrieval selection alone, showing that some context must be preserved during memory construction. Adjacent-turn grouping improves evidence access at matched unique-turn budgets, while wider windows do not consistently help. This motivates keeping fact-specific context local while organizing cross-session evidence separately. We propose ContextFact, which links context-preserving fact memories with source-grounded entity representations. Within sessions, it gathers the context needed to interpret each fact. Across sessions, it aligns recurring referents and revisits the source dialogue to reconstruct coherent entity representations linked to the supporting facts. At query time, ContextFact retrieves fact and entity views, reranks them independently, and uses entity-linked expansion to surface related facts; complete entity representations then provide organized cross-session evidence. With Qwen3.5-27B as the backbone, ContextFact achieves state-of-the-art performance while using only $21\%$ of the evidence tokens consumed by the strongest evaluated baseline, obtaining $86.49\%$ overall judge accuracy on LoCoMo and $91.00\%$ on LongMemEval-S. Controlled ablations support context-preserving fact boundaries and source-grounded entity reconstruction, while entity readout improves multi-hop accuracy beyond adding more facts. 

---
# Medical Specialty Prediction from Bangla Patient Narratives: A Comparative Study of ML, DL, Transformers, and LLMs 

**Authors**: Tasnin Tabassum Ali, Faisal Ahmed, Muhammad Anwarul Azim  

**Link**: [PDF](https://arxiv.org/pdf/2611.29151)  

**Abstract**: Timely access to appropriate medical specialists remains a bottleneck in Bangladesh's healthcare system, as patients describe symptoms in informal Bangla rather than clinical terms, leading to ineffective referrals. This paper compares four model families for predicting medical specialties from Bangla patient narratives: six machine learning (ML) models using hybrid TF-IDF features, four deep learning (DL) models using Word2Vec embeddings trained on the corpus, four fine-tuned transformers, and five large language models (LLMs) evaluated zero-shot. From a corpus of 3798 patient records covering 20 specialties, we retained the five with the most records (Medicine, Gynecology, Dermatology, Pediatrics, Gastroenterology) and split the resulting 2730 records 80:20 into training and test sets shared by all families. The best transformer was XLM-RoBERTa (86.26% accuracy, 86.21% macro-F1), the best ML model was SVM (84.98%, 84.92%), the best DL model was Bidirectional GRU (80.77%, 80.83%), and the best LLM was Gemini (90.29%, 90.15%). Holm-corrected paired McNemar tests show SVM significantly outperforming Bi-LSTM (adjusted p = 0.004), whereas the Gemini-versus-Grok gap does not survive correction (adjusted p = 0.104), and neither the strongest transformer versus the weakest LLM (Grok) nor SVM versus the strongest transformer reaches significance. A tuned classical model is thus statistically indistinguishable from the strongest transformer, so accuracy rankings alone should not be read as evidence of one family's superiority. The findings provide a comparative baseline for Bangla medical text classification. 

---
# Group-Balanced Supervised Fine-Tuning for Knowledge-Intensive Domains 

**Authors**: Mark Heimann, Bruno Jacob  

**Link**: [PDF](https://arxiv.org/pdf/2611.29127)  

**Abstract**: Supervised fine-tuning (SFT) weights every supervised token equally, so on knowledge-dense corpora the loss budget follows token count rather than importance: on a legal QA benchmark, citation strings absorb 81% of the supervised loss weight budget while the answers being taught absorb 5%. We present Group-Balanced SFT (GB-SFT), which equalizes the loss budget across the semantic roles of tokens within a target sequence. With one tunable dial and one structural switch, GB-SFT contains standard SFT, class- and span-level balancing as corners whose geometry we characterize exactly. On benchmarks in law, medicine, code, and mathematics, we show across model families that: (i) GB- SFT injects new knowledge better than uniform SFT, pointwise token weighting, memory editing, and matched-budget data mixing; (ii) it matches the injection of masking out everything but the knowledge payload while retaining the citation recall that masking destroys (0.98 vs. 0.38); and (iii) role-aware grouping carries the gain: label-free partitions recover at most a third of it, an LLM tagger nearly all where the groups are lexically distinctive; and and (iv) the budget governs what fine-tuning breaks as well as what it learns, acting as a monotone lever on the safety-guardrail erosion SFT causes, at matched task accuracy. We release two new contamination-controlled knowledge-injection benchmarks with validated ground truth. 

---
# Decoding Transformer Dynamics via Koopman Subspaces 

**Authors**: Subhradeep Kayal, Ali Dashti  

**Link**: [PDF](https://arxiv.org/pdf/2611.29113)  

**Abstract**: We apply Koopman operator theory and Dynamic Mode Decomposition (DMD) to characterise how large language models (LLMs) process different input topics. Treating the layer-to-layer evolution of hidden states as a dynamical system, we fit a linear Koopman operator and take the subspace spanned by its dominant modes as a per-topic spectral fingerprint. Four findings emerge. First, the depth map is near-linear: a single linear operator predicts each next-layer state to within approx. 18% relative error (interior variance explained approx. 0.99). Second, over the 57 MMLU subjects the fingerprint is topic-discriminative, matching the MMLU taxonomy at 0.72 nearest-neighbour purity and each subject's semantic neighbourhood at 0.75 any-hit@3. Third, it is a reproducible signature (split-half self-retrieval approx. 0.97$). Finally, these findings are consistent across seven base models (Qwen2.5 0.5-7B, Llama-3 1-8B), with a suggestive trend of alignment rising with MMLU accuracy. The spectral fingerprint thus provides a probe of how transformers process different topics across depth. Code is available at this https URL. 

---
# Beyond Free-Form Generation: Measuring Interface Composition in Competitive NLP Systems 

**Authors**: Xinchen Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.29103)  

**Abstract**: Systems with decoder LLMs can combine free generation with scoring, ranking, or selection. We measure these documented interface combinations with five annotation fields linked to source evidence. The corpora comprise 517 solutions from 42 Kaggle competitions involving NLP and 76 participant systems from 26 SemEval, TREC, and WMT task years. Among eligible generation systems with decoder LLMs, 18 of 55 Kaggle systems and 17 of 20 Shared Task systems are hybrid: free generation coexists with a constrained interface. The remaining systems use free generation alone. Giving each competition or task year equal weight yields hybrid shares of 0.327 and 0.876. Within Eedi, 21 eligible systems with decoder LLMs have five exact interface profiles; ten LLM Science Exam systems share one interface class and have four profiles. Across Kaggle, the eligible systems with decoder LLMs span 18 exact interface profiles, revealing variation within broad task and backbone categories. These estimates characterize documented public systems selected using our selection criteria and include interfaces throughout the complete pipeline, including intermediate stages. The data are available at this https URL. 

---
# Operational Runbooks as Living Documents: A Longitudinal Study of Document Evolution under Assisted Post-Incident Revision 

**Authors**: Rocker D'Antonio, Harry Xie, Xuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2611.29097)  

**Abstract**: Operational runbooks increasingly function as living documents within operational workflows: they are maintained by people, used in incident support, and continuously revised as organizational knowledge changes. Yet little is known about how such document collections evolve over time in production settings, or which interpretable signals are useful for monitoring document change. We analyze 17 weeks of version-controlled runbook snapshots produced during post-incident revision with machine-assisted review and human oversight. We quantify document evolution using lightweight linguistic and structural signals grounded in prior work on procedural language, technical communication, and document organization. Correlation analysis shows mixed patterns of association, including both positive and negative relationships, with effect sizes ranging from negligible to moderate and varying across corpora. Temporal summaries further distinguish comparatively stable signals from revision-sensitive ones, supporting monitoring of document maintenance workflows rather than one-shot quality judgments. An illustrative analysis on public GitHub product documentation shows that signal behavior differs across corpora, reinforcing the need for context-aware interpretation. These results identify which metric classes provide stable anchors and which provide edit-sensitive indicators for managing operational documentation as an engineered document resource. 

---
# RES: Accelerating Diffusion Language Models with Rolling Entropy Sampling 

**Authors**: Patrick Pynadath, Nicolas Perez-Nieves, Bobak Shahriari, Jean Tarbouriech, Ruqi Zhang, Ivan Lobov  

**Link**: [PDF](https://arxiv.org/pdf/2611.29066)  

**Abstract**: A diffusion language sampler makes two choices for every position in its denoising window: where the next token comes from, and when the position is committed to the KV cache. Using the entropy-bound sampler in DiffusionGemma, we study two extremes of the second choice. Committing each token as soon as it is accepted removes the chance to revise it, which hurts when the entropy bound is relaxed to trade accuracy for speed. When accepted tokens are frozen, macro accuracy falls from 73% to 20% as the bound is relaxed, while the unfrozen sampler stays near 70%. Committing a whole block at once wastes computation, since tokens converge in a largely left-to-right order and more than half of the window holds already-final tokens by twenty denoising steps. These findings motivate the Rolling Entropy Sampler (RES), which commits a contiguous accepted prefix once it reaches a minimum length and keeps positions farther ahead open to revision. Unlike earlier rolling samplers, RES runs on uniform diffusion, where accepted tokens can still change, and it needs no training. Across seven benchmarks, RES increases tokens per forward pass from 19.3 to 25.5, while macro accuracy goes from 80.7% to 78.8%, and it extends the speed-quality Pareto frontier beyond what tuning the entropy bound alone reaches. 

---
# HyperHarness: Learning to Generate Personalized Agent Harnesses from User Histories 

**Authors**: Zhaoxuan Tan, Jiachen Zhao, Jingbo Yang, Zheyuan Liu, Chuhan Wang, Pei Zhou, Mengting Wan, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29062)  

**Abstract**: Personalized LLM agents rely on an agent harness, the surrounding memory and control code, to connect user history with downstream decisions. Existing methods typically employ either a single shared memory workflow across all users, which is efficient to deploy but cannot adapt how evidence is retrieved and interpreted across diverse histories, or per-user harness optimization, which customizes behavior for each user but requires costly repeated candidate evaluation at deployment and suffers from sparse, noisy supervision in raw logs. To address this trade-off, we present HyperHarness, a framework that learns a shared history-to-harness generator to synthesize executable, user-specific retrieval and answer-guidance code directly from user histories. Given a user's permitted history, HyperHarness prompts a construction LLM to write personalized retrieval and guidance functions, while downstream task scores across training users supervise iterative refinements to the shared generator without reference harnesses or answering-model updates. At deployment, the frozen generator constructs a harness for each unseen user in a single pass from history alone, eliminating per-user task-scored search. Across three benchmarks (PersonaMemv3, Mem-PAL, and MemoryCD) and three answering models (Qwen3.5-9B, Qwen3.8-27B, and GPT-5.6-Luna), HyperHarness achieves the highest aggregate score in seven of nine settings and ranks second in the remaining two, improving over the strongest baselines by up to 5.95 points while cutting matched answer-time tokens by 90.32% over long-context prompting. Analyses show that optimization tailors context allocation to each executor, transfers to unseen user clusters, and amortizes adaptation costs. 

---
# Grounding Collaborative Agents in Group Cognition 

**Authors**: Zhaoxuan Tan, Mengting Wan, Jingbo Yang, Kwei-Herng Lai, Renyi Qu, Lorenzo Jaime Flores, Meng Jiang, Longqi Yang, Victor Poznanski, Pei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.29036)  

**Abstract**: Collaborative agents joining an ongoing team rely on shared context to connect multi-party interaction histories with downstream coordination decisions. Existing work typically focuses either on retrieving historical facts from conversation logs, which surfaces isolated evidence without capturing who knows or owns what, or on evaluating end-to-end interactions, which conflates preparation with execution and lacks direct supervision because a team's history has no single ground-truth summary. To bridge this gap, we formulate group briefing, the task of reconstructing task-relevant group cognition across participants' expressed beliefs, group dynamics, and grounding gaps, and introduce a four-domain benchmark of 1,351 requests and 9,975 verified reference points paired with a two-stage evaluation pipeline combining reference-point coverage and downstream collaboration simulation. Using this pipeline, we show that GroupBrief, a simple task-guided approach that iteratively probes the message history and synthesizes a structured brief without offline memory indexing or graph construction, outperforms ten retrieval and agent-memory baselines in overall coverage with gpt-5.4-mini, while benchmarking nine language models reveals that prospective friction remains harder to recover than past state across model scales. We further use reference-point coverage as a feedback signal for harness optimization, refining GroupBrief's retrieval and synthesis instructions while keeping model weights frozen. On held-out Test projects, GroupBrief (optimized) improves coverage by 2.54 percentage points for gpt-5.4-mini and 12.88 for gpt-5.6-luna, achieving 64.46% and 68.42% preference over no brief in downstream simulations. These results establish our benchmark and evaluation pipeline as a foundation for measuring and optimizing how collaborative agents ground in group cognition. 

---
# LITMUS - Literature Informed Testbed for Mapping Usecases and Synthesizing Risk-Cards 

**Authors**: Seshu Tirupathi, Elizabeth M. Daly, Radu Marinescu, Javier Carnerero Cano, Yannis Belkhiter  

**Link**: [PDF](https://arxiv.org/pdf/2611.29034)  

**Abstract**: The proliferation of large language models (LLMs) and Artificial Intelligence (AI) in real-world applications has heightened the need for systematic, safety-oriented deployment frameworks. Existing approaches typically prioritize the construction of datasets and risk cards around the LLM and AI architecture itself, yet they provide scant consideration for the specific use-case that drives model selection. This omission is problematic because any practical AI deployment begins with a detailed use-case description that informs the assessment of contextual hazards. In this paper, we introduce a structured pipeline that, given a use-case, automatically produces a tailored dataset and an accompanying risk card. The process ensures that identified risks are tightly coupled to the use-case, and it offers traceable evidence from peer-reviewed literature to substantiate each risk to use-case relationship. Prior claim-verification methods (FActScore, FactReasoner) have been validated for factual claims. Risk-relevance is a hypothetical claim about possible harm, and whether evidence retrieval can meaningfully assess such claims remains untested. Literature-Informed Testbed for Mapping Use-cases and Synthesizing (LITMUS) risk cards are the first systematic attempt to characterize this adaptation. We evaluated the pipeline using the MIT Risk Taxonomy and achieved a 0.91 F1-score with a list of 320 samples from the benchmark risk dataset that was considered. These results demonstrate the effectiveness of our approach in generating robust, evidence-based risk assessments for LLM deployments. 

---
# Population-Based Post-Training for Large Language Models 

**Authors**: Yixuan Even Xu, Aleksei Petrenko, Raja Giryes, Jackson Hamburger, Kevin Chen, Fei Fang, J. Zico Kolter, Philipp Krähenbühl  

**Link**: [PDF](https://arxiv.org/pdf/2611.29014)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is one of the most popular post-training paradigms for large language models. RLVR typically trains a single model under hyperparameters chosen before training begins, while researchers tune hyperparameters across consecutive runs. We introduce Population-Based Post-Training (PBPT), which trains several models in parallel and lets the training process do the tuning through population-based training. PBPT periodically evaluates models on a validation set. High-performers are duplicated and their hyperparameters perturbed to explore new configurations. Low-performers stop training and are replaced by these copies. On reasoning benchmarks (Enigmata, Nemotron) and interactive tool-use environments (Gaia2, AppWorld), PBPT produces stronger agents than compute-matched groups of independent RLVR learners. Furthermore, PBPT is able to recover from poor initial hyperparameters, automatically discovering better settings over the course of training. 

---
# NoHairCache: Calibrate-Once KV Cache Eviction with One Power-Law Exponent 

**Authors**: Shuyang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29006)  

**Abstract**: The decay exponent $\alpha$ in KV-cache eviction is an intrinsic property of the model: calibrated once on a single corpus, it transfers directly across domains. Attention-based accumulation (H2O) and recency-based sliding windows (StreamingLLM) are two dominant eviction frameworks, often treated as unrelated, but we show that they are the two endpoints of a single power-law family: by scoring each token as $s_i = A_i \cdot (t-i)^{-\alpha}$, where $A_i$ is cumulative attention and $\alpha$ controls the age penalty, H2O and StreamingLLM correspond to $\alpha=0$ and $\alpha\to\infty$, respectively (with StreamingLLM additionally hard-coding an attention-sink patch).
The no-hair mechanism of \citet{xiang2026nohair} bounds the admissible range of $\alpha$, and the measured exponents agree: calibrating once on WikiText yields $[0.556, 0.822]$ for a representative model (Qwen2.5-1.5B-Instruct). The per-model ranges are consistent across five models (410M--1.5B) and independently match the optimal regions from perplexity sweeps.
\textsc{NoHairCache} achieves the lowest perplexity across three datasets (WikiText, PG19, and arXiv) without per-dataset tuning, outperforming both H2O and StreamingLLM ($p < 0.05$ on all three datasets, with $0.06$--$0.10$ PPL gains over StreamingLLM at budget $512$). On downstream four-way multiple-choice QA, it matches or outperforms StreamingLLM across all nine (task, budget) cells, retaining $98.2\%$ of full-cache accuracy on NarrativeQA at $25\%$ retention versus $91.1\%$ for StreamingLLM; per-cell McNemar tests are not significant, so QA is corroborating only. We release our calibration script and drop-in scorer upon publication. 

---
# EmoRouter: Adaptive Modality Cascading for Multimodal Emotion Recognition 

**Authors**: Fangxu Yu, Lingzhi Yuan, Dehai Min, Tao Feng, Ge Liu, Furong Huang, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.28960)  

**Abstract**: Multimodal Emotion Recognition (MER) involves detecting the underlying emotion behind each utterance within a conversation, which is essential for emotionally aware human-computer interaction. While omni-modal large language models (Omni-LLMs) demonstrate strong capability for MER by jointly processing modalities including text, audio, and vision, different modality combinations yield different prediction outcomes across utterances: acoustic and visual inputs can help recover annotated labels missed by text-only inference, but can also turn correct predictions into errors. Processing all modalities for every utterance is also computationally expensive. To this end, we introduce EmoRouter, a training-free, model-agnostic framework for adaptive modality cascading that selects an exit tier for each utterance along a predefined modality order. The routing is guided by prediction-uncertainty estimates derived from Conformal Prediction. EmoRouter routes each utterance through a sequence of increasing-cost modality levels and exits at the first level when the model is confident in the emotion. To determine when to exit, we propose Conformal Set Contraction, a structural calibration criterion that calibrates the per-tier CP error budgets used for these exit decisions. Comprehensive evaluations on 4 MER benchmarks and 3 Omni-LLMs reveal that EmoRouter matches or outperforms the full modalities while reducing inference latency by 67.9--94.5%. Furthermore, EmoRouter demonstrates high transferability across different models, validating its robustness. Code is available at this https URL. 

---
# Don't Sleep, Keep Watch: Proactive Perception for Computer-Use Agents 

**Authors**: Jiaqi Su, Jie-Jing Shao, Yichen Liu, Long-Kai Huang, Haiyan Yin  

**Link**: [PDF](https://arxiv.org/pdf/2611.28947)  

**Abstract**: Computer-use agents (CUAs) built on vision-language models are increasingly entrusted with long-horizon tasks in live desktop, web, and mobile environments, where the screen keeps changing on its own. However, current CUAs lack real-time situational awareness: they perceive the environment through discrete snapshots, and the substantial inference latency of VLMs, compounded by highly dynamic interfaces, opens a perception-action gap in which the observed state frequently becomes stale before the corresponding action is executed. To bridge this gap, we present NightWatch, a plug-and-play framework that endows any CUA with continuous, temporally grounded perception without changing the agent's model, prompt, or action space. NightWatch couples the VLM agent with NightWatch-0.8B, a lightweight System-1 model that performs streaming monitoring of the interface state and keeps the agent synchronized with the evolving environment at low computational cost and latency, thereby grounding each decision in the current state rather than an outdated snapshot and reducing both stale actions and idle waiting while preserving task success. Experiments on desktop and dynamic web environments (OSWorld-Verified and DynaCU-Bench) show that NightWatch, driven by a watcher of only 0.8B parameters, reduces non-reasoning wall-clock time by 33% and wasted time by 53% on OSWorld while keeping task success on par, raises success on DynaCU-Bench from 39.5% to 44.2% with the large VLM agent, and is more efficient than passive-watching baselines such as AOI and PUSV under the same screenshot-only input, as well as a System-1 decision model more than $30\times$ its size. Overall, NightWatch demonstrates that the timing of perception is as critical as its content, marking a step toward agents that maintain continuous awareness of their environment rather than perceiving it turn by turn. 

---
# When Rubrics Do Not Decide: Silent Value Completion in LLM Evaluation 

**Authors**: Zhuangzhuang Pan, Yan Xia, Kaiwen Lei, Chee Seng Chan  

**Link**: [PDF](https://arxiv.org/pdf/2611.28940)  

**Abstract**: Multi-criterion LLM judges often evaluate responses without specifying how conflicting criteria resolve, leaving the winner undetermined even under exact criterion scores. Issuing a strict verdict without declaring the required trade-off premise constitutes silent value completion. We introduce TradeoffBench to isolate this failure by separating rule execution from response scoring. On authoritative criterion vectors, six open-weight non-reasoning judges pass all dominance and equality controls yet pick unsupported winners on 9%-81% of 1,345 conflicts. All 577 order-stable choices follow an explicitly banned majority rule. Reasoning-enabled configurations drop anonymous-vector violations to 2 in 9,720 judgments under an explicit rule, yet cannot bridge omitted trade-offs or divergent component scoring. On raw TradeoffBench text with unspecified trade-offs, GPT-6.1-sol omits its completion premise in 29.8% of 400 verified conflicts. Even under an explicit rule on four-rater-consensus MultiPref conflicts, four reasoning judges still issue strict verdicts on 16/45 (GPT-6.1-sol) up to 35/46 and 25/29 (open-weight judges) of valid presentations. We propose identified-first evaluation, decoupling specification support, explicit completion, and target agreement. In MultiPref human-score audits, peer-mean substitution raises directional disagreement from 0.6% to 28.8% on 2,707 shared targets. A single-peer ablation confirms a 29.38-point gap on 2,175 targets, ruling out averaging as the cause. In HUMAINE, declared partial trade-offs add 2.5 percentage points of coverage, yet newly decided pairs incur 3.9% disagreement versus 0.9% for Pareto. Reliable LLM judging, therefore, requires auditing, specification support, declared trade-offs, and human target agreement separately. 

---
# CANDOR: A Multilingual Provenance-Documented Corpus for Dysarthric Speech Research 

**Authors**: Bernard Muller, LaVonne Roberts, Leo Kaiya, Antonio Ortiz Barrañón  

**Link**: [PDF](https://arxiv.org/pdf/2611.28930)  

**Abstract**: CANDOR assembles 61 corpora across 17 languages and five aetiological classes plus a mixed/other group. Its register counts 2,057 speakers classified as dysarthric and 19,538 healthy controls. The collection contains 4,469.7 hours of audio, including 1,178.8 hours in the dysarthric subset. Counts use speech-impairment labels where available and diagnosis labels otherwise. The collection includes 732 dysarthric speakers across 14 non-English languages; 194 speakers use six languages unsupported by Qwen3-ASR, the model underlying its non-English baselines. We document composition, chain of custody, licences and transfer terms, while distinguishing corpus-level provenance from participant-level consent records, which we hold for few constituents. Among 31 dysarthric-bearing corpora, open licences cover 29.0% of corpora, 10.5% of speakers and 8.2% of hours. Excluding the English Speech Accessibility Project raises the open hour share to 31.2%, showing its dependence on this collection's composition. CANDOR cannot be redistributed as a single audio resource; we specify public, restricted and conditional artefacts. Severity labels available in 15 corpora do not provide a validated cross-corpus mapping. Within the non-English ASR baselines, only Mandarin yields a real-speech estimate meeting our evaluation criteria. It differs sharply from the synthesis result and remains imprecise; an English challenge result uses a different model and protocol. CANDOR provides a documented basis for research on under-represented languages while making constraints on evaluation, comparability and sharing explicit. 

---
# Definition-Guided In-Context Learning for Zero-Shot Spoken Language Understanding 

**Authors**: Neeraj Agrawala, Sriram Ganapathy  

**Link**: [PDF](https://arxiv.org/pdf/2611.28929)  

**Abstract**: Adapting spoken language understanding (SLU) systems to novel or revised label taxonomies conventionally requires additional labeled data and retraining, creating a bottleneck for real world deployment. While in-context learning with few-shot demonstrations offers an alternative, multi-modal generalization and scaling has been challenging. In this work, we investigate whether natural-language definitions alone can guide zero-shot SLU without demonstration exemplars, finding that symbolic definitions serve as the primary supervisory driver for adapting speech-text models. We operationalize this principle as Definition-Guided In-Context Learning (DG-ICL), a fine-tuning framework that decouples semantic categories from fixed lexical names. By replacing labels with dynamically randomized definitions (arbitrary synthetic tokens in DG-ICL-Sym or permuted vocabulary words in DG-ICL-Flip), DG-ICL avoids lexical shortcuts and encourages the model to ground acoustic representations in contextual definitions. Novel categories can thus be introduced at inference simply by defining them through the instructions. Across three speech-text models (AUDIO FLAMINGO 3, QWEN2-AUDIO, and SALMONN), DG-ICL outperforms standard instruction tuning on held-out tasks under novel symbols (0.663 vs. 0.449 macro-F1 on Audio Flamingo 3; 0.703 vs. 0.620 on Qwen2-Audio; 0.582 vs. 0.399 on SALMONN), while remaining competitive on original labels. Furthermore, counterfactual evaluations demonstrate instruction following abilities, where it is shown that DG-ICL-Sym retains 68% of its performance when definitions contradict pre-trained lexical priors, whereas standard instruction tuning elicits significantly reduced retention of 21%. Finally, we show that DG-ICL enables zero-shot speech clustering into user-defined categories specified on the fly 

---
# Can Open LLMs Clear Sanctions-Screening Alerts? A Pre-registered Test on Public Sanctions Lists with Turkish Name Variants 

**Authors**: Mehmet Murat Huyut  

**Link**: [PDF](https://arxiv.org/pdf/2611.28914)  

**Abstract**: Sanctions screening raises name-based alerts that are mostly false positives, and level-1 analysts clear them by checking the customer's date of birth and name against a written policy. We test whether open-weight language models that a bank can run on its own premises can perform this level-1 disposition. From the UK and EU sanctions lists (7,103 natural persons) we built 12,200 synthetic customers: listed persons under six name conditions, including Turkish spellings of Arabic-origin names and Turkish romanisations of Cyrillic names, and two date-of-birth conditions, plus namesakes, near-name customers and a general population. A deterministic fuzzy and embedding screen raised 7,383 alerts, which three open models served on domestic infrastructure and three rule baselines dispositioned under the same policy. The models sent far fewer innocent customers to level 2 than a rule that applies the policy's date clause: 0.6 to 16.9 per 1,000 screened customers against 24.4, and they stated the date-of-birth status correctly in 98.3% to 98.5% of alerts. They also failed to escalate 1.5% to 2.2% of the listed persons. The pre-registered non-inferiority test against the rule was met, but only because of a construction error we found after the analysis: 21 customers in the "minor discrepancy" condition had dates that conflict under the policy. Without them the rule misses no listed person and no model meets the margin. The misses had model-specific causes: one model read minor date discrepancies as conflicts, one declared word-order and typing variants to be different names, and one discounted alerts that its own rationale said to escalate. Fuzzy matching found 69% of the Turkish spellings of Arabic-origin names; adding embeddings raised this to 74%. Open LLMs can cut level-1 workload, but on this evidence only behind a deterministic date check and with every name doubt escalated. 

---
# Thakaa at StanceEval-2026: Inference-Time Target Grounding for Arabic Stance Detection 

**Authors**: Meshal Alamr, Hassan Alqaeri, Abdullah Aldahlawi  

**Link**: [PDF](https://arxiv.org/pdf/2611.28860)  

**Abstract**: StanceEval-2026 tests Arabic stance classification under target shift, with evaluation targets absent from the labelled task data. We test two ways to supply target information at inference time. For Track 1, we retrieve eight labelled Women Driving demonstrations from the external ArabicStanceX corpus and score the three label tokens with Gemma-4-31B-IT. For the unseen-target track, we prepend cached target context prompted to be factual and non-evaluative, then fuse label predictions and label-token scores from multiple LLMs, with final assignments constrained by published class totals. The systems obtain 85.22 and 91.78 Favg2 on Tracks 1 and 2, ranking sixth and fourth in the final standings. On held-out ArabicStanceX tweets, direct prompting scores 79.45 Favg2; eight random Women Driving demonstrations raise it to 84.78, and similarity retrieval reaches 89.04. Across three target-held-out Mawqif evaluations, a fixed target card raises Gemma-4-31B-IT from 86.54 to 86.93, whereas QLoRA lowers Gemma-4-12B-IT from 84.43 to 83.77. In these separate public evaluations, same-target demonstrations and retrieval improve performance, factual target context provides small, target-dependent gains, and the tested source-only adapter does not improve. 

---
# Thakaa at AlexandriaX-2026: Who Gets the Final Say? Assigning Output Authority in Dialectal Arabic Machine Translation and Error Detection 

**Authors**: Meshal Alamr, Maryam Altuwaijri, Hassan Alqaeri, Abdullah Aldahlawi  

**Link**: [PDF](https://arxiv.org/pdf/2611.28857)  

**Abstract**: We report first-place systems for all four evaluated AlexandriaX-2026 settings: both English-to-dialect translation tracks, cross-dialect translation, and translation-error detection. In unconstrained translation, our minimum Bayes risk (MBR) selection restricts output authority: four local systems can supply the translation, while seven additional systems contribute only reference evidence. Candidate-constrained MBR beats the strongest complete candidate in all 13 dialects and gains 1.2554 macro spBLEU overall, reaching 33.4891. It also surpasses a retrospective oracle selecting the best whole candidate system per dialect. In the constrained track, country-routed LoRA adapters over a shared NileChat-3B-Base backbone incorporate the released Libyan and Sudanese supervision absent from the training and development splits, raising macro spBLEU from 29.5969 to 30.4233. Post-evaluation, a single adapter trained on all released Alexandria data reaches 30.4733 macro spBLEU; averaging model weights across three training seeds raises this to 30.9726 while retaining one model for inference. For cross-dialect translation, nonparallel target-dialect retrieval guides deliberation, raising Gemma-4-31B-IT without fine-tuning from 24.6898 to 28.4011 spBLEU. For translation-error detection, GPT-5.5 proposes semantic error regions, while MARBERTv2 controls their character boundaries and corroborates additional spans, raising Overall from 0.4691 to 0.4982. 

---
# Self-WSP: Can Language Models Improve Themselves Through Weight-Space Programming? 

**Authors**: Guoming Ling, Shanshan Zhong, Kaixuan Zhang, Zhongzhan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28833)  

**Abstract**: Frontier language models can now read and edit the weights of a neural network much like code and thereby improve a small model, an approach known as Weight-Space Programming (WSP). We further explore applying WSP to a language model itself, which we call Self-WSP. Specifically, Qwen3.5-0.8B reads its own weights, proposes an edit, and then continues with the edited weights as both problem solver and editor. We find many cases of genuine self-improvement. Self-edits can raise GSM8K accuracy from 11.8% to 62.2% and MATH500 accuracy from 7.0% to 32.2%, sometimes by changing only 16 parameters, and in some cases repeated self-editing keeps improving the model. A closer look, however, reveals that these gains lack interpretability. The model shows little understanding of the edits it makes, and WSP driven by random search sometimes performs comparably. Self-WSP thus holds promise as a path toward recursive self-improvement (RSI), but realizing this promise requires further exploration. 

---
# The Humming Revachol: A Dialogue Network Analysis of Narrative Power in Disco Elysium 

**Authors**: Xinran Liu, Miguel Arana-Catania, Jack Orchard  

**Link**: [PDF](https://arxiv.org/pdf/2611.28812)  

**Abstract**: This study applies social network analysis to the narrative RPG Disco Elysium to examine how narrative power is distributed among its characters. A directed, weighted character-mention network of 73 NPCs is constructed from the game's complete dialogue script, with characters identified and annotated across five social attributes. Narrative power is operationalised through four dimensions: attention, authority, discursive initiative, and embeddedness. Non-parametric statistical tests examine differences across gender, age, race, class, and ideology. The findings suggest that narrative power is largely independent of social identity categories and is more closely associated with character narrative functions, while structural embeddedness reflects the socio-organisational contexts in which characters are situated. These findings show that dialogue-network analysis can reveal structural patterns of representation that visibility counts alone would miss. 

---
# Multimodal LLMs Know the Game but Assume the Real World 

**Authors**: Jianxin Gao, Bohan Chen, Runze Li, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28745)  

**Abstract**: Multimodal large language models can state the rules of a video game, but do they apply them when they only see the game? We introduce RuleCraft, 1,240 image questions over 155 Minecraft rules with outcomes verified in the game engine, each asked in text, on the image with the game named, and on the bare image. Six widely used models know the rules and recognize the game, yet answer the bare image by real-world physics: naming the game raises their accuracy on rules where the game departs from reality by 8.2-19.8 points on a main set and by 33.8-52.7 on a hard stress-test set. Their errors pick the real-world outcome, the gap is largest where the game leaves a scene unchanged, and reasoning traces state the game's rule, then answer by reality. On the main set, naming the game mostly swaps the real-world default for the game's. The pattern holds with reasoning, recurs in two more games, and persists on questions the models themselves take to be about the game; one neutral sentence asking for the rules of the world shown recovers much of the gap. Evaluations that name the game measure knowledge of its rules, not world selection. 

---
# KMed-31B Technical Report 

**Authors**: KMed Team at NAVER  

**Link**: [PDF](https://arxiv.org/pdf/2611.28719)  

**Abstract**: We present KMed-31B, a medical large language model post-trained from an open-weight model for on-premises deployment with limited hardware. Post-training a model for diverse medical tasks remains difficult, because improving one kind of task can lower performance on another. We attribute this to differences between tasks in expected output and available feedback. Medical question answering and diagnosis prediction have reference answers, whereas clinical inquiry has no single correct response and is judged with rubrics. We therefore divide medical training tasks into four categories by these two properties, train one specialist per category with a reward that its feedback supports, and integrate the specialists into a single student with Multi-teacher On-policy Delta Distillation (MOPD$^2$). For clinical inquiry, we build Synthetic Health Dialogue, a simulated corpus of consultations with lay users and clinicians whose patient cases span common clinical presentations in primary care. Relative to its base model, KMed-31B raises HealthBench Hard from 22.95 to 41.87 and HealthBench Professional from 37.92 to 50.20 in non-thinking mode, and from 23.80 to 51.48 and from 41.97 to 59.18 in thinking mode, while its scores on four multiple-choice medical benchmarks are maintained or slightly improved. KMed-31B has the highest unadjusted HealthBench Hard score among the compared models, including proprietary ones, and scores higher on HealthBench Professional than every open-weight model we compare, including models with more than ten times as many total parameters. Applied to another base model, Qwen3.8-27B, the same training recipe also raises both HealthBench scores. We describe the training recipe, data construction, and settings in detail so that institutions requiring on-premises deployment can develop medical models from open-weight foundations. 

---
# Legal Is Not Good: A Pre-Registered Benchmark of Low-Cost Language and Decision Models on a New Strategy Game 

**Authors**: Federico Lucca  

**Link**: [PDF](https://arxiv.org/pdf/2611.28713)  

**Abstract**: Measure Before You Train argued that an agent's decisions should be measured against ground truth, under a protocol fixed in advance, before anyone trains or replaces the model behind them. We carry the method to a second, very different domain, a turn-based strategy game, and ask a deliberately modest question: how well do low-cost agents play under a pre-registered protocol? hexfolk is a minimal two-player 4X game written for this purpose, so no model has seen its rules or positions. A deterministic engine, one agent interface, a referee that never corrects an illegal or late move, and per-decision logs make every game replayable; questions and expected answers were committed before the games, on held-out seeds with mirrored seats and 95% Wilson intervals. Two small open language models (8B and 3B) lose every game to a few-line rule player. Answering in free text they are invalid on 98% of their decisions; constrained to legal moves they never play an invalid one, yet they lose to the uniformly random player (the 8B model wins 10.0 [4.3, 21.4]% of games), refuting our expectation that they would beat it; the result survives a control that shuffles the order of the options. Two native decision models never play an invalid move and both beat random; one loses all 50 games to the rule player, the other is not distinguishable from it in 50 games and beats a one-step search. The paid agents make tens to over a hundred model calls per game; against the same opponents a free 60 ms-per-move search, which makes none, outplays every paid model by the full range of the scale. Two 7-8B models served locally behave the same way. No frontier model is included. 

---
# Make Position Uninformative: Exchangeable Self-Distillation Flattens the Lost-in-the-Middle Curve 

**Authors**: Qingsen Ma, Dianyun Wang, Jiahong Liu, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2611.28638)  

**Abstract**: Retrieval-augmented language models answer well when the relevant document comes first or last and poorly when it sits in the middle. Some fine-tuning recipes lift the middle by lowering the start. We show how training contexts teach positional bias. With a uniform answer slot and shuffled distractors, the optimal reader ignores position. Building on this result, Exchangeable Self-Distillation (ExSD) trains a small adapter on shuffled contexts to reproduce the model's answer when it reads the answer document alone. It needs no gold answers and only a few thousand examples. On six models from 1.5B to 32B parameters, ExSD raises both average and worst-position accuracy against the frozen model, including on a held-out test split. In controlled LoRA comparisons, it significantly improves both criteria over Pos2Distill. Against CORD, mean differences are not significant, while worst-position accuracy improves significantly in the 20- and 30-document settings. ExSD also matches or beats gold-answer fine-tuning, changes behavior on other tasks far less, and transfers to unseen tasks. Head probes show greater reliance on content over position. 

---
# Reducing Permutation Sensitivity in Multimodal Large Language Models via Shared Positions and Attention-Only Distillation 

**Authors**: Yifei Cheng, Xiaolin Zhang, Yu-Xuan Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28612)  

**Abstract**: In this paper, we conduct a systematic study on improving permutation robustness while preserving accuracy. First, we categorize permutation bias phenomena into four representative scenarios. Then, we take the multiple-choice question answering (MCQA) dataset as an example under text-option reordering scenario to analyze the root cause of permutation bias. Based on this root cause analysis and set encoding strategy, we investigate shared positions with blocked cross-unit attention (SP-B) and extend this design to bidirectional cross-unit attention (SP-V) for modality exchange. However, these structural changes may reduce predictive accuracy. To recover model capability across different scenarios under the SP structure, an attention-only distillation from frozen native models is proposed. Shared positions reduce permutation sensitivity, and attention-only distillation improves accuracy under the shared-position structure. Across all three models, our method is comparable to or better than majority voting in macro-average all-permutation accuracy, and outperforms supervised fine-tuning by 1.95-4.20 percentage points. This work provides valuable insight into permutation bias mitigation and model capability recovery. 

---
# MedCoast: Semantic Abstraction of Reasoning Trajectories for Test-Time Biomedical Question Answering 

**Authors**: Yishan Zhong, Jieru Li, Xuanzhe Ding, Shaunna Wang, Yanxin Yang, Micky C. Nnamdi, J. Ben Tamo, Yifei Wang, Yining Yuan, Benoit L. Marteau, Shaun Q. Y. Tan, Carrie Moore, Luyu Xie, Yang Xie, Wenqi Shi, May D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28586)  

**Abstract**: Biomedical question answering often requires multi-step reasoning under limited inference budgets. While test-time search methods such as Tree-of-Thought can improve reasoning performance, directly searching over free-form language is computationally expensive and highly redundant, as many reasoning trajectories contain semantically equivalent intermediate steps expressed in different surface forms. We introduce MedCoast, an offline-to-online reasoning framework that converts sampled chain-of-thought trajectories into a compact space for structured process supervision. Offline, we recursively cluster semantically similar reasoning steps under shared prefixes to construct an abstract reasoning tree that captures recurring clinical reasoning patterns. We then derive node-level supervision from progress toward correct solutions, and train a lightweight process reward model (PRM) to evaluate partial reasoning states. At inference time, the learned critic guides test-time exploration by prioritizing promising reasoning prefixes. Across seven biomedical QA benchmarks, MedCoast often improves over self-consistency under fixed inference budgets, with gains of up to 15 points. Despite using a reward model more than 50 times smaller, MedCoast achieves performance comparable to state-of-the-art reward models in several settings, highlighting its potential for resource-constrained biomedical reasoning. 

---
# Modeling Perceived Promptness in Simultaneous Interpreting from Quality, Structure, and Timing Signals 

**Authors**: Ziyu Zhang, Satoshi Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2611.28566)  

**Abstract**: Simultaneous interpreting requires interpreters to convey the source accurately while keeping pace with it in real time. Timing measures such as segment-onset delay capture part of this requirement, but may not fully reflect how professional evaluators judge promptness. We investigate whether these judgments can be explained by timing alone or whether the interpreted output provides additional information. Our dataset contains 622 Chinese-English and English-Chinese segments from 16 source speeches and seven interpreters. Two senior professional interpreters rated each segment for language quality (LQ), expressiveness (EXP), and perceived latency (LAT), which we also refer to as professional promptness. On held-out source speeches, onset delay alone achieves a Pearson correlation of r = .318 with LAT ratings. Combining predicted LQ and EXP with delay improves the correlation to r = .475. Lexical and structural features provide a stronger signal, reaching r = .632 when combined with delay and r = .658 when predicted quality is also included. These results suggest that professional promptness judgments reflect properties of the interpreted output beyond when the interpreter begins speaking. The additional contribution of predicted quality over structural features, however, is modest. Although the two evaluators differ in how strongly quality relates to LAT, cross-evaluator predictions remain positively correlated with LAT ratings. The overall pattern also holds when each interpreter is held out in turn. 

---
# Tool-to-Self: Internalizing Visual Tool Capabilities via Scaffolded Reinforcement Learning 

**Authors**: Jin Cui, Hao Dong, Xinyue Long, Boran Zhao, Pengju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.28564)  

**Abstract**: Vision-language models (VLMs) increasingly rely on external visual tools to access fine-grained visual evidence, but repeated tool execution incurs substantial inference overhead and deployment dependencies. Moreover, conventional tool-use RL can improve visual-task performance, yet the resulting tool gains are often unstable. Motivated by this, we propose Tool-to-Self (T2S), a tool-scaffolded reinforcement learning framework that turns external visual tools into temporary training scaffolds. T2S cold-starts tool-use ability and observation prediction from tool-repairable examples, then progressively anneals tool execution on a per-tool basis during RL. By combining answer-correctness and observation-consistency rewards with privileged on-policy self-distillation from actual tool observations, T2S effectively teaches the model both to reproduce tool outputs and to reason from them. Experiments across diverse visual perception and reasoning benchmarks demonstrate consistent improvements over both explicit tool-use models and implicit zooming models while enabling tool-free inference with reduced latency. 

---
# From Reward Engineering to Data-Centric Reinforcement Learning for Effective Agentic Visual Tool Use 

**Authors**: Jin Cui, Xinyue Long, Jiepeng Zhou, Boran Zhao, Pengju Ren, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.28558)  

**Abstract**: Multimodal large language models (MLLMs) increasingly use external visual tools to acquire evidence during reasoning, yet learning when to invoke these tools remains challenging. Existing tool-use reinforcement learning (RL) methods often rely on call-contingent rewards to encourage beneficial tool use, but such reward engineering can distort the task objective and lead to degenerate calling behaviors. We argue that effective tool-use learning requires identifying not merely which questions are difficult, but which failures are attributable to missing visual evidence and can therefore be corrected through tool observations. We propose AOVO, a data-centric framework that enables MLLMs to learn effective tool use through perception attribution and policy-adaptive data sampling (PADS), without introducing tool-specific rewards. It counterfactually probes the model's reasoning states to identify observation-sensitive questions, distinguishing them from reasoning-dominated ones. PADS then periodically estimates category-level observation effects from recent policy rollouts and dynamically reallocates the training distribution toward questions where visual observations provide informative learning signals, while optimizing solely for final-answer correctness. Experiments across diverse visual perception and multimodal reasoning benchmarks demonstrate consistent improvements over competitive tool-use MLLMs, with causal analysis further validating the data-centric perspective for learning effective visual tool use. 

---
# BIRD-Persona: A Persona-Aware Benchmark for SQL and pandas Analytics Code Generation 

**Authors**: Davin Hill, Pawan Chorasiya, Bhawani S. Leelar, Oren Avram, Robert E. Tillman, Tamer Soliman  

**Link**: [PDF](https://arxiv.org/pdf/2611.28542)  

**Abstract**: Production text-to-code systems use large language models to turn natural-language analytics requests into executable SQL and pandas code. Existing benchmarks cover both tasks but do not isolate when performance diverges between them. We introduce BIRD-Persona, a persona-stratified extension of BIRD-Python that adds novice and expert rewrites to tasks with matched SQL and pandas ground truth, along with schema variants and task-structure annotations. Across 1,534 tasks, nine models, and 245,592 evaluations, SQL outperforms pandas for eight of nine systems. The gap varies by condition: expert, schema-explicit prompts improve both paradigms but help SQL more; schema representations have mixed, model-dependent effects; and selected-column burden widens the gap while subqueries do not. By making user phrasing a controlled benchmark dimension, BIRD-Persona supports persona-aware and paradigm-aware evaluation of natural-language analytics systems. The dataset is available at this https URL. 

---
# Information Access and Gain-Loss Framing in Six Large Language Models and Human Decision Makers 

**Authors**: Aadithya Chandramouli, Jeremy M. Gernand  

**Link**: [PDF](https://arxiv.org/pdf/2611.28525)  

**Abstract**: A risky choice can be described by what will be saved or what will be lost, and people often choose differently depending on that wording. We test whether six large language models (LLMs) show this gain-loss framing effect, and whether it shrinks when they can first read more about the options. GPT-4.1, GPT-4o, GPT-4o-mini, Gemini 2.5 Flash, Qwen3 235B and Llama 4 Maverick made 11,519 choices across four risk scenarios, each in its original wording, two paraphrases and a mirrored version in the opposite wording. Before choosing, a model could open up to 0, 2, 3 or 4 of seven information cards. With no cards, all six models showed large framing effects: the share choosing the safe option was 27 to 79 percentage points lower under loss than under gain wording. The gap was smaller at 2 cards than with no cards in every model. Against a 10-point threshold set in our analysis plan, GPT-4.1 fell within it at 4 cards and Qwen3 only at 2 cards; in the plan's pooled comparison over all versions, GPT-4.1 was within it from 2 cards and GPT-4o from 3. GPT-4o-mini and Llama 4 Maverick kept gaps of 28 to 66 points with cards, and for GPT-4o-mini and Qwen3 the gap grew again from 2 to 4 cards. Part of the drop from 0 to 2 cards may reflect a change in prompt format, and most card content counted against the risky option, which may favor the safe choice. In an exploratory comparison with 198 students who saw the same scenarios and cards but opened cards one at a time and chose how many, decisions made after opening 0 or 1 cards showed a 19-point difference between the gain scenarios, asked first, and the loss scenarios, asked later, and no clear difference appeared in decisions made after opening more, though this design cannot show that cards reduced framing in people. With these cards, in the balanced comparison 17 of 18 card gaps were below the model's no-card gap, but few were within the threshold. 

---
# SupportChain: Trace-Calibrated Aggregation of Quote-Grounded Reasoning Chains for Multi-Hop RAG with Small Readers 

**Authors**: Zhexi Gu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28519)  

**Abstract**: Strong results in multi-hop retrieval-augmented question answering come from graph indexes read by models of 70B parameters or more, and building such an index calls a large language model over the whole corpus. We present SupportChain, which splits a question into hops, binds each hop's answer to a sentence quoted from a retrieved passage, estimates how reliable each execution is from its own trace, and merges several executions by the support of their answers, the pooled reliability of the executions that agree, stopping once support is high enough. With a 9B reader under the HippoRAG~2 protocol, SupportChain reaches 63.0 F1 on MuSiQue, above the best earlier result of 59.2, which used a 20B reader. With the same reader and embedder, it is significantly better than HippoRAG~2 on all five datasets, by up to 17.2 F1, and it builds no index. Merging by support is significantly better than majority voting, and than letting the reader choose among the candidates, on three of the five datasets and level with both on the other two. The reliability model needs only 100 answered training questions. 

---
# Digital Twins for Multi-Agent Systems: When to Trust, What to Change 

**Authors**: Truc Nguyen Thi Le, Phuoc Nguyen T. H  

**Link**: [PDF](https://arxiv.org/pdf/2611.28517)  

**Abstract**: Digital twins of agent networks need to flag unreliable forecasts and help users choose interventions. Both decisions depend on what happens over several steps: an error can spread through the network, and an edit whose trajectory is easy to predict may still yield a poor collective outcome. We ask when information about error propagation improves failure detection and whether a learned twin can select effective network edits despite imperfect forecasts. We propose Propagation-Aware Validation for Networked Digital Twins (PAV-Twin). Its monitor combines Jacobian-based propagation features with recent errors and ensemble disagreement to estimate the risk that a forecast exceeds an error threshold. Its planner simulates each candidate edit and ranks the edits by their predicted collective outcomes. In simulations, the monitor gained 0.13 in area under the precision-recall curve on larger oscillator networks with familiar topologies and 0.09 on larger networks with held-out topologies, relative to a graph baseline without propagation features. Gains were small in contractive consensus. Giving the graph baseline the same features narrowed the gap. In a separate planning study, ensemble simulation recovered about 99% of the possible improvement over random intervention selection in oscillators and nearly all of it in consensus. With a simplified oscillator twin, selecting by predicted outcome also gave lower regret than selecting by trajectory fidelity. These results show when propagation features help flag unreliable forecasts and why the quality of intervention choices should be measured directly. 

---
# Digital Knockoffs 

**Authors**: Niuniu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28482)  

**Abstract**: Social simulations often use language model agents to represent people who differ on several measurable traits. However, describing a trait profile in a prompt does not guarantee the agent will express each trait at its intended level. We propose a method that calibrates agents to multidimensional profiles using activation steering: (i) extract steering directions for each trait, (ii) learn a shared mapping from steering strengths to measured trait scores, and (iii) search for a steering intervention for each target. We use Bayesian optimization for this search and evaluate the method on the Big Five personality scores of 500 OpenPsychometrics respondents. In each of five rounds, the assigned interventions yield lower mean absolute error across respondents than the prompting and steering baselines. Further, the model's responses to a given intervention yield the same trait profile regardless of whose target motivated it, so a poor match for one respondent can be the best match for another. When we match each respondent to the closest profile found across all tested interventions, Bayesian optimization yields 16.6% lower mean absolute error than uniform random search at the same evaluation budget. Our method thus offers a way to build social agents whose measured traits dovetail with those of the people they seek to represent. Whether this closer fit translates to improved predictions of downstream behavior remains an open question. 

---
# Muted Distillation: Sealing Reasoning from Teacher Unlocks Token-Efficient Agents 

**Authors**: Yuwei Zhang, Ian Miao, Wentao Zhang, Chen Qu, Bill Bi, Jingbo Shang, Donna Schut  

**Link**: [PDF](https://arxiv.org/pdf/2611.28445)  

**Abstract**: On-policy distillation (OPD) is commonly adopted as one of the components in post-training of large language models (LLMs), but in multi-turn agentic tasks we find that the student model increases their reasoning lengths after training. Our empirical analysis shows that the reasoning token lengths is affected by how much the student can distract the teacher's scores. We introduce Muted Distillation (MD), a simple variant of OPD that hides the student's reasoning from the teacher, so the teacher judges what the student does without the interference of student's reasonings. MD takes a few lines of code and adds no extra data or hyperparameters. On the knowledge-work benchmarks OfficeQA Pro and SpreadsheetBench, it reduces output tokens per episode by 57\% and 53\% compared with standard OPD, making the student more concise than it was before training. At the same time, it matches OPD's accuracy on OfficeQA Pro and improves pass@1 by 4.8 points on SpreadsheetBench. 

---
# Language models compress human preference geometry while political structure remains prompt-sensitive 

**Authors**: Adolfo Ignacio Fuentes-Jofre, Cristian Candia  

**Link**: [PDF](https://arxiv.org/pdf/2611.28383)  

**Abstract**: Large language models are increasingly used to construct synthetic publics. Prior work has documented errors in individual responses, population heterogeneity, and prompt robustness, but whether these failures reflect a common loss of fidelity or distinct properties of synthetic populations remains unclear. We tested this using matched pairwise policy choices from 1,991 Chilean, 843 French, and 434 Brazilian adults replayed through three open-weight model families. Across a 27-condition Chilean grid and cross-country factorial replications, no condition jointly reproduced individual choices, population heterogeneity, and political organization. The most consistent result was compression of the latent preference space: synthetic participant variance was only 0.4-10.5% of the corresponding human reference in the primary comparisons and remained below human variance in all 270 matched refits. Political structure was much more sensitive to how choices were elicited. Reversing proposal position changed 12.3-31.2% of decoded choices. Explicitly labeling the endpoints of the numeric left-right self-placement scale changed 20.5-28.1% of choices and moved an independent proposal-level measure of left-right cleavage toward the human pattern in all three settings. Yet matched agreement improved by only 3.1-4.2 percentage points and participant geometry remained compressed. In Chile, the same intervention compressed the synthetic population further. These results separate political legibility from population fidelity: language models can recover salient political divisions while failing to preserve the breadth of human preference variation. Synthetic-public fidelity is therefore a property of the checkpoint, prompt, and setting, not of a checkpoint alone, and must be evaluated separately at individual, population, and political-structure levels. 

---
# Taking the PULSE of Enterprise Workflows: Evaluating Persistent Project-State Understanding 

**Authors**: Jiexin Ding, Lin Ai, Adam Davies, Scott Counts  

**Link**: [PDF](https://arxiv.org/pdf/2611.28374)  

**Abstract**: Enterprise agents must maintain an up-to-date understanding of work distributed across heterogeneous artifacts, evolving over time, and unevenly visible to different collaborators. We formulate this challenge as persistent project-state understanding (PPSU) and introduce PULSE, a benchmark of 1,573 questions over 30 enterprise project histories. PULSE constructs deterministic, time-indexed project state and documented collaborator views from source-linked records, and evaluates three families of questions: open loops, project evolution, and view gaps. Across retrieval agents, graph- and memory-augmented retrieval, long-context, and oracle conditions, we find substantial headroom for current systems. Retrieval and extraction account for the largest share of the gap, but providing explicit project state on top of gold evidence yields further substantial gains, showing that PPSU cannot be reduced to retrieval alone. Further analysis shows that questions requiring information to be reconciled across multiple parts of the project history remain difficult even when the relevant evidence is provided, while project-state synthesis and perspective conditioning remain persistent challenges. These results motivate enterprise agents that explicitly maintain and reconcile temporally grounded state across sources, actors, and time. 

---
# Contrastive On-Policy Distillation via Dual-Branch Supervision 

**Authors**: Zhibo Hou, Fan Zhao, Yiping Li, Zhiyu An, Wan Du  

**Link**: [PDF](https://arxiv.org/pdf/2611.28359)  

**Abstract**: For on-policy distillation (OPD), more capable models are a natural choice as teachers. However, higher task performance does not necessarily imply greater teachability. A controlled token intervention study suggests that even a model weaker than the student can serve as an informative contrastive reference. Building on this observation, we propose Contrastive On-Policy Distillation (cOPD), which provides dual branch supervision by contrasting strong teacher, weak teacher, and student predictions at student generated prefixes. Our analysis explains how this three way comparison helps select token candidates associated with better or worse reasoning outcomes under an idealized model. We evaluate cOPD across diverse teacher student configurations spanning different training stages, parameter scales, and model families, on tasks including mathematical reasoning, instruction following, and question answering with abstention. cOPD achieves the best or top-tier performance among the evaluated baselines across all three tasks, improving overall Pass@16 on mathematical reasoning by up to 7.51\% over the strongest baseline. Further analyses show that cOPD suppresses undesirable response patterns and suggest that it facilitates exploration. 

---
# Grammar-as-a-Draft: More Tokens per Speculative Decoding Step on Grammar-Constrained Structured Outputs 

**Authors**: John Zhang, Silky Arora  

**Link**: [PDF](https://arxiv.org/pdf/2611.28349)  

**Abstract**: Under structured outputs (JSON schemas, regexes, tool calls), speculative drafters do not see the grammar constraint and repeatedly propose invalid keys, delimiters, and indentation, while the grammar engine already knows which tokens must come next.
We present Grammar-as-a-Draft (GaaD), which treats the grammar as a second drafter. Before each verify step, the scheduler merges the drafter's proposal with grammar-forced tokens into one draft row: matching tokens stay as GRAMMAR, the proposal is cut where it leaves the forced path, and forced tokens fill the rest of a row wider than the drafter's proposal. The verify step accepts GRAMMAR drafts without computing their logits, composing with any drafter with no new kernels or compilation on the critical path. Because grammars fix text rather than tokens, committing canonical forced tokens changes the output distribution relative to grammar-masked target decoding.
We implement GaaD in vllm-torchtpu and an independent Rust serving runtime, evaluating on TPU v7x with MTP, DFlash, and n-gram drafters on a JSON character-profile benchmark. GaaD raises average acceptance length from 2.06 to 5.43 (MTP, Qwen3.5-0.8B), 1.99 to 5.31 (MTP, Qwen3.5-35B-A3B), 1.57 to 4.13 (DFlash, Qwen3-4B), and 1.30 to 3.64 (n-gram, Qwen3-4B), decoding 2.66-2.81x faster at batch size 1 and 1.14-1.70x at batch size 64. In Rust, MTP+GaaD is 2.23x faster on Qwen3.5-0.8B, 2.02x on Qwen3.5-397B-A17B-FP8, and 1.22x at 128 concurrent requests. Under mixed traffic (25-50% structured), GaaD finishes the workload 1.03-1.23x sooner (structured requests 2.2-3.2x sooner) while plain-text TPOT rises 7-42%. We also show grammar-masked greedy decoding can pick non-canonical tokenizations, so GaaD can yield different valid outputs that an fp32 reference finds equally likely. 

---
# Language Model Harnesses can be Compositional Generalizers 

**Authors**: Alex L. Zhang, Souradip Chakraborty, Omar Khattab  

**Link**: [PDF](https://arxiv.org/pdf/2611.28298)  

**Abstract**: Modern language models (LMs) are post-trained and deployed as agents inside a "harness", but standard harness designs accumulate context that the neural network is forced to generalize to, making training for long-horizon tasks expensive and time-consuming. We study an alternative emerging path: training inherently neurosymbolic LM harnesses, which externalize and manage context programmatically to compose existing LM capabilities. We hypothesize that training around these harnesses can offer a better return on compute by enabling compositional generalization, as they can learn to decompose complex problems so that every observation their neural LM receives is locally in-distribution, even when the full task is novel. We find that harnesses like the Recursive Language Model (RLM) learn such decompositions readily under reinforcement learning post-training: (1) we demonstrate that training only on short tasks can generalize to tasks 2-64$\times$ longer while the base neural LM improves less, and (2) training on one domain transfers to novel domains that share latent structure. On messier, complex tasks, we find that we can target individual LLM calls within a harness by generating and training on easier, synthetically generated surrogate environments with fake tool / sub-calls, a technique we call surrogate harness training. From this finding, we train RLM-27B on synthetic data, and show it improves Qwen3.8-27B in the same RLM harness on a hard suite of long-context tasks like CTC-Bench and ExpertLongBench by 26.9% on average while using $\sim$2$\times$ fewer tokens at inference time. Finally, we show that harnesses can compose over these newly learned decompositions using recursive sub-harness calls. 

---
# SnakeTranslate: Weakly Supervised Policy Learning from Offline Parallel Text for Simultaneous Translation 

**Authors**: Yu Pu, Zhiyun Fan, Baiji Liu, Zhifu Gao, Xiangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.28055)  

**Abstract**: Adaptive simultaneous translation requires reliable prefix translation and timely READ/WRITE decisions, but ordinary parallel corpora provide no action annotations. We present SnakeTranslate, which adapts loss-derived alignment to sampled causal histories for jointly learning both capabilities. The framework supports supervised fine-tuning (SFT) on parallel text and on-policy distillation (OPD) with reverse Kullback--Leibler (KL) divergence on student-generated translations. At inference, source and target tokens form one append-only interleaved history, enabling incremental decoding with cache reuse. On FLEURS, NTREX128, and WMT24++, our 4B model improves quality--latency trade-offs over streaming policies applied to the same offline-tuned backbone, with further gains from OPD. It also compares favorably with larger translation models at low and moderate latency, while higher-latency quality gaps remain on WMT24++. 

---
# Benchmarking and Adapting LLMs for English-Swedish Medical Question Answering 

**Authors**: Seyed Alireza Molavi, Zhan Su, Stefan Byttner, Yan Hu, Peyman Sheikholharam Mashhadi, Prayag Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2611.28050)  

**Abstract**: Large language models do well on English medical exams, but less is known about Swedish. This matters most for small open models that hospitals can run locally. To reduce this gap, we first build an item-aligned English-Swedish benchmark from ten medical multiple-choice datasets; each Swedish item keeps the options and answer of its English source. A conservative refinement loop repairs the translations: open models from three families find errors, ground them in a Swedish medical term base, and keep a model-written edit only if blind judges vote for it. On this benchmark, small models lose much more accuracy in Swedish than large ones, especially when they answer without reasoning. Second, we ask whether English data and open teacher models can close this gap. We translate the training data, let teachers write reasoning traces, and fine-tune two small students on gold answers or traces in English, Swedish, or both. Training on reasoning traces that include Swedish closes the gap and lifts the 1.2B student from 23.5\% to 52.0\% in Swedish, slightly above a 20B open model in its fast mode, with a sixth of its memory. We release the benchmark, its edit history, and the code. 

---
# Distributionally-Verifiable Rewards for Human-Behavior Simulation 

**Authors**: Ruoyun Ma, Shuang Liu, Peiye Liu, Ao Qu, Baichuan Mo  

**Link**: [PDF](https://arxiv.org/pdf/2611.28048)  

**Abstract**: Large language models are increasingly used to simulate survey respondents in social-science research. A useful simulator should predict individual responses, reproduce population response distributions, and capture demographic differences. Common training approaches struggle to balance these goals: exact-match rewards encourage the model to concentrate on the most frequent human response, which is contradictory to the goal of distributional alignment. Meanwhile, aggregate distribution matching can look good even when demographic information is ignored. To this end, we introduce Distributionally-Verifiable Rewards (DVR), a reinforcement-learning reward that uses demographic group distribution gaps to guide individual responses, designed by aligning policy generated frequencies with human response frequencies within each question-group margins. We evaluate the method using the OpinionQA dataset and find that comparing with accuracy-only RLVR, DVR preserves per-response accuracy while reducing JSD by 34.5% in-distribution and 14.6% on held-out questions, with higher DPC point estimates on both splits. Comparing KL-SFT with the demographics-blind marginal further shows that low JSD is insufficient to establish fidelity to population heterogeneity, highlighting that accuracy, distributional fit, and demographic structure should be evaluated together. 

---
# What Counts as Attack Success? Evaluating Harmfulness in Red Teaming LLMs 

**Authors**: Ruyuan Wan, Adriana Alvarado Garcia, Md Toufikuzzaman, Dongwon Lee, Ting-Hao `Kenneth' Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28038)  

**Abstract**: Judging whether a model response is harmful requires contextual and normative choices, yet red-teaming studies often delegate this judgment to automated evaluators. We examine how harmfulness criteria are specified in red-teaming research and how they carry through to reported results. Reviewing 76 papers, we find that many adopt their criteria from prior datasets, classifiers, or judge prompts rather than defining harmfulness themselves. Applying evaluators from these papers to identical outputs, we find that they can reach different judgments on the same response, which can change conclusions when comparing models. These differences arise partly because evaluators draw the boundary of harm in different places: some count any response that does not refuse, while others count only output that could plausibly cause harm. We argue that red-teaming studies should state their harmfulness criteria and report how they chose and validated their evaluators to support more reliable measurement and interpretation. 

---
# CO-GRAFT: Attributing Reinforcement Learning Gains in Search Agents via Role-Level Policy Grafting 

**Authors**: Ramraj Chandradevan, Sayontan Ghosh, Vinoth Selvendran  

**Link**: [PDF](https://arxiv.org/pdf/2611.31185)  

**Abstract**: Reinforcement learning (RL) improves large language model search agents by up to 32 exact-match (EM) points over their pre-RL initializations, yet overall accuracy gains do not identify which underlying functional capabilities improve or interact during training. We introduce CO-GRAFT, a reward-agnostic interventional attribution framework in which an RL-trained agent and its pre-RL initialization jointly generate a single trajectory under closed-loop execution with live retrieval. Each policy controls a mutually exclusive subset of four functional roles: intermediate reasoning, search-or-answer decision, search query generation, and final answer extraction. Evaluating all 2^4=16 role assignments without modifying weights or retraining defines a cooperative game whose Shapley values decompose the total RL gain across roles, while M"obius interaction terms distinguish standalone role improvements from multi-role coupling. Applying CO-GRAFT to Search-R1 agents across Qwen2.5 and Llama-3.2 models (spanning two sizes, base and instruction-tuned initializations, GRPO and PPO optimizers, and NQ and HotpotQA datasets), we show that query generation never exceeds 15% of the total gain; instead, gains are governed by pre-RL answer validity, search retention, model scale, and fixed task-level search budgets, shifting credit from reasoning toward answer extraction in larger or well-formatted answer response checkpoints while smaller base models rely on reasoning--decision coupling. 

---
# Precision-Aware Learning Rate Transfer Across Block Count in LLM Post-Training 

**Authors**: Nan Chen, Soledad Villar, Soufiane Hayou  

**Link**: [PDF](https://arxiv.org/pdf/2611.31166)  

**Abstract**: The post-training of pretrained large language models, including supervised finetuning (SFT), direct preference optimization (DPO), and reinforcement learning with verifiable reward (RLVR), is very sensitive to the learning rate, and tuning it through grid search is costly in both computation and memory. In this work, we propose \textbf{TipSweep}, a method to transfer the learning rate across the number of trainable transformer blocks. To do so efficiently, we search for the best learning rate by training only the last $k_0$ blocks (in our experiments, typically $k_0=1$), and provide scaling rules that predict the optimal learning rate when training any set of $k$ blocks, including when $k=L$ (the full transformer). Across experiments with models ranging from 1 to 7 billion parameters, we observe that the optimal learning rate follows precision-dependent scaling rules: the optimal rate scales approximately as $k^{-1}$ with float32 (FP32) master weights, but as $k^{-1/2}$ when updates are applied directly to bfloat16 (BF16) weights without a master copy. We explain both rules using an analytical model that accounts for update rounding, under assumptions supported by profiling experiments. Finally, we develop two prototypes of the learning rate search for a 6.7-billion-parameter model. The first fits the search on a single consumer GPU with 16 GiB of memory. The second speeds up the search on a high-end GPU by evaluating all candidate learning rates in a single pass over the data, so that a five-candidate search takes about half the GPU hours of one full-depth ($k=L$) run. 

---
# Attention Temperature Confounds Zero-Shot Context Extension 

**Authors**: Ruben Navasardyan  

**Link**: [PDF](https://arxiv.org/pdf/2611.31113)  

**Abstract**: A training-time attention change is usually judged for zero-shot context extension against a dense baseline beyond the training length at the default temperature, $\tau = 1$. The outcome depends on how miscalibrated the baseline is there: the share of perplexity it recovers when its logits are sharpened by one scalar. Under the standard recipe (fp32 master weights), an 8k-trained baseline is 8-15% miscalibrated at 30k in three seed-paired 365M twins; in one of them this makes the block-sparse-trained twin look 7% better, which calibration removes. In pure-bf16 training without master weights the miscalibration is several times larger: an AdamW step ($\beta_2 = 0.95$) on a gain initialised at 1.0 is below half the bf16 spacing at peak learning rates under $1.7\times10^{-3}$, so gains never move (22 checkpoints), the baseline is 37-58% miscalibrated, and block-sparse pretraining, attention dropout and ALiBi look 6-35% better. Calibrating one temperature per model and length removes all three advantages, leaves ALiBi 5% behind RoPE and keeps part of the gain of a positive control, NTK-aware scaling. Unfreezing the gains reverses the block-sparse advantage at 365M and removes 71-89% of it at 1.2B. We recommend reporting extension results at calibrated temperatures and checking that normalisation gains moved. 

---
# UMAMI: A Multimodal Benchmark Dataset for Evidence-Grounded Dietary Decisions 

**Authors**: Safal Thapaliya, Jiatan Huang, Debapriya Chatterjee, Yanfang Ye, Chuxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31068)  

**Abstract**: Dietary decisions require identifying a food's nutrient content and considering individual dietary needs, but recognizing food from an image alone does not provide this information. Visually similar foods can have different nutrient profiles, so selecting the wrong food record can lead to incorrect decisions even when nutrient rules are applied correctly. Furthermore, satisfying nutrient constraints does not necessarily establish a food's relevance to a person's intake. Evaluating these challenges together requires connecting food images, nutrient records, and individual intake histories. To support this evaluation, we introduce UMAMI (Unified Multimodal Assessment of Meals and Intake), a multimodal dataset and benchmark linking independently collected food images with nutrient profiles and respondent health and intake records. Using this dataset, we construct three diagnostic tasks covering visual nutrient question answering, personalized food recommendation, and condition-aware food substitution. Paired instances and input controls examine how visual identity and respondent context affect decisions. Our experiments identify errors in both food-record selection and nutrient-rule application, while showing that higher rule compliance does not necessarily improve alignment with subsequent recorded intake. These findings highlight the need to evaluate how multimodal systems select and use structured evidence for dietary decisions. The dataset, code, and dashboard are available at this https URL. 

---
# CAST: A Deployed Speech-to-Text Cascade for Closed Captioning of Entertainment Television 

**Authors**: Aditya Mishra, Thanish Bolla, Bidisha Sharma, Balaji Pitchaikannu, Surbhi Sinha, Anup Kotalwar  

**Link**: [PDF](https://arxiv.org/pdf/2611.31055)  

**Abstract**: Despite rapid progress in automatic speech recognition (ASR), caption generation for long-form entertainment video remains challenging for content dense in music and effects, rich in accents, crosstalk, and noise. The transcript quality depends on segmentation and surrounding modules as much as the acoustic model. A single pass voice activity detection (VAD) can miss faint, low-energy voice while forwarding near-silent segments that induce hallucination and repetition in an autoregressive decoder. These competing errors motivate a hierarchical VAD that revisits rejected segments under semantic and acoustic retention checks. We present Captioning ASR Stack for Television (CAST), an integrated speech-to-text pipeline for closed captioning (CC) combining frozen Whisper large-v3 with segmentation, language identification, lyrics routing, and targeted hallucination detection and repair on a single NVIDIA L4 GPU. On 100 in-house assets spanning 60.46 hours, CAST reduces normalized WER from 6.99% to 5.36% and subtitle edit rate from 254.2 to 221.9 min/hour against the WhisperX pipeline, and improves deletion rate by 11% on public datasets. We also present the behavior of each stage in the pipeline. CAST entered the production speech-to-text (STT) stage of our CC pipeline in late August 2026; we report the evaluation evidence and path to full-scale operation. 

---
# Parsing Visual Latents for Multimodal Spatial Reasoning 

**Authors**: Zhuoying Yang, Yichen Guo, Kathy Jia, Zichang Su, Ayesha Kashif, Nange Hai, Jeffrey Ming Han Li, Zhewen He, Silvio Kempf, Ni Jiachen, Junjie Yang, Muhammad Hamza, Zakaria El Ouahidi, Gabriel Azpitarte, Manling Li, Yi Fang, Shuhong Liu, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31021)  

**Abstract**: Spatial reasoning across multiple views requires inferring the 3D relations between objects and the camera from 2D observations, which depends on cross-view correspondence and scene geometry that vision-language models (VLMs) do not reliably extract. Pretrained vision encoders already capture this information in their latent features, with CLIP linking objects to image regions, DINOv3 matching semantics across views, and VGGT encoding multi-view geometry. However, VLMs do not benefit from these features when we present them directly in context as grayscale feature images. We therefore propose LatentParse, a training-free framework that parses latent features into explicit spatial evidence for VLMs. In LatentParse, an analyst VLM probes the features of all three encoders with predefined tools that locate objects, track them across views, and measure motion between views. It then distills the results into a concise text report, which is provided together with the images to the VLM to answer the question. We evaluate LatentParse on MMSI-Bench, MV-RoboBench and SPAR-Bench, where it improves all five VLMs over the photo-only baseline by 0.7-2.4 percentage points on average. On MMSI-Bench, it raises Qwen3.8-27B from 34.4% to 39.5%, with the largest improvements of 19.2 percentage points on the camera-related questions. These results suggest that latent features become useful for spatial reasoning once they are parsed into relation evidence, without training. Code is available at this https URL. 

---
# Can Latent Reasoning Drive LLM Outputs Without a Trace? 

**Authors**: Tomer Ashuach, Zorik Gekhman, Avi Caciularu, Ori Ram, Jonathan Herzig  

**Link**: [PDF](https://arxiv.org/pdf/2611.31015)  

**Abstract**: LLMs typically solve complex tasks with chain-of-thought (CoT) reasoning, which is also the main tool for monitoring them for misaligned behavior. However, standard transformers can also solve tasks with complete latent reasoning (CLR), carrying out the whole task internally and outputting only the final answer. Theory bounds the depth of CLR, the number of sequential steps it can carry out, and recent work measured it to be shallow. In this work, we ask whether CLR is deep enough for a model to pursue a hidden objective that requires reasoning. Such a model would generate a CoT that appears to solve the given task, while a latent multi-step computation decides its answer based on the hidden objective. First, we show that a simple auxiliary loss allows models to solve a multi-step task latently at least 5x deeper than previously reported. Then, in two common tasks, decision-making and code generation, we train a model to pursue such a hidden objective, which decides a loan or plants a vulnerability. The model's CoT justifies the output with no trace of the hidden objective, and a strong LLM monitor does not flag it. Finally, using activation patching, we find that the latent computation is completed before the model generates its CoT, which takes no part in it. Altogether, we show that a hidden objective that requires reasoning can go unseen by a monitor that reads only the output. 

---
# RoboUniverse-10K: Toward the Scaling Era of Agentic Robotics via Real-to-Sim Environments 

**Authors**: Jinkun Liu, Yingze Wang, Situ Wang, Deyao Hong, Mingju Gao, Wenyi Li, Yizhe Chi, Youjie Zheng, Kaisen Yang, Xiangyang Wang, Bingxiang He, Yuxin Zuo, Boyuan Chen, Yue Chen, Cheng Qian, Tianxing Chen, Huichi Zhou, Huaxiu Yao, Hao Tang, Hao Zhou, Ya-Qin Zhang, Calvin Xiao, Tiga Altman, Huan-ang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.31002)  

**Abstract**: Agentic Robotics, in which foundation models act as coding agents to control robots through executable programs, has emerged as a compelling paradigm for embodied intelligence. Yet unlike software engineering, where massive executable environments power post-training, robotics remains constrained by small, manual benchmarks and in-distribution evaluation. To bridge this gap, we introduce RoboUniverse-10K, 10,807 interactive, resettable, and verifiable physics environments built from real-world videos via an automated Real2Sim and quality assurance pipeline. Each runs in an isolated, Harbor-compatible harness that hides privileged states and grading logic to mirror real-world sensing. From the same pipeline, we hold out 500 unseen tasks to establish RoboUniverse-Verified, an expert-verified zero-shot benchmark. Evaluating 11 closed- and open-source models reveals a huge gap: frontier models reach 85.8\% success, while open-source baselines lag at 2.6\% to 10.0\%, bottlenecked by low-level grasping, deformable-object manipulation, and post-contact physical control. Finally, we collect about 17K multi-turn trajectories totaling 0.79B tokens with GPT-6 Astra and perform full-parameter supervised fine-tuning on Qwen3.6-35B-A3B, yielding RoboUniverse-35B-A3B. Our model achieves substantial zero-shot gains on RoboUniverse-Verified, lifting the success rate from 0.8\% to 21.2\%, while achieving zero-shot task completions across LIBERO-MAX, RoboCasa, RoboTwin 2.0, and a real Franka arm. We open-source the RL-ready environments and model to lay the foundation for large-scale reinforcement learning in Agentic Robotics. 

---
# Ask Again: Answer-List Instability in Generative Engines 

**Authors**: Björn Busch-Geertsema, David Antony Selby, Sebastian Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2611.30819)  

**Abstract**: Visibility in generative search engines is measured by which entities appear in responses, and in what order. Ask an engine the same question twice and the answer list can change: a different ordering, or different items altogether. Across 60,000 responses, we find that answers to the same prompt name a different first item 32% of the time, and even two answers with the same items disagree on which comes first. The first position is a measurement target of its own. A "clear leader" emerges after seven or eight runs, which are the budgets recommended for brand and source monitoring, but a "close leader" only about half the time. After 25 runs, a close leader is identified only 60% of the time. Variability between engines is larger than within the same engine. For open-ended prompts, the gap is more than double that for determinate ones. Search queries and cited sites observed leave most of the disagreement unexplained. Visibility measurement should report, for each engine, which items appear and how often each comes first, each with its own quantification of uncertainty. 

---
# What Quality Engineers Need to Know About Generative AI: Structured Data Extraction 

**Authors**: Fadel M. Megahed, Ying-Ju Chen, Yamin Dahwich, Zhao-Wei Chen, Lucas K. Hung, Arthur Carvalho, L. Allison Jones-Farmer, Ibrahim Yousif, Inez M. Zwetsloot  

**Link**: [PDF](https://arxiv.org/pdf/2611.30778)  

**Abstract**: In this expository paper, we explain how quality engineers can use large language models (LLMs) and vision-language models (VLMs) for structured data extraction. We introduce the relevant concepts, discuss structured extraction from a measurement-system perspective, and demonstrate its implementation in R and Python. We then present two freely available web-based applications (apps): (a) extracting information from automotive recall notices and (b) monitoring quality engineering research on arXiv. To ensure the accuracy of our AI-driven structured extraction, human evaluators validated the results. In the first app, four reviewers assessed information extracted from 30 recall notices using two models, with 98.6% of the 720 field-level ratings judged fully correct. In the second app, 21 authors evaluated information extracted from 45 of their arXiv papers. All paper summaries were rated mostly correct or correct, although the evaluations identified problems with some classification fields and mathematical expressions. These results demonstrate the practical potential of structured data extraction while emphasizing the need to validate the extracted information. We make both apps and their source code publicly available so that readers can adapt the approach to their own applications. 

---
# Lowering the Bar for Everyone: Hidden Validation-Seeking Goals in Empathetic RL 

**Authors**: Xinche Zhang, Shanghong Xie, Zhiyuan Ma, Jinhao Li, Sen Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.30702)  

**Abstract**: When RL learns from the reactions of many users, the policy usually cannot see what each user wants. We ask what it learns from users who privately want validation: to validate them alone (targeting), or to lower its bar for endorsing everyone (a threshold shift). We answer this in one setting, empathetic RL against LLM-simulated users (RLVER, GRPO, Qwen2.5-7B-Instruct): we vary only the share of validation-seeking users (0--50%; RLVER's default: 24.6%) and evaluate on 1,000 Reddit AITA posts with community fault verdicts.
The policy shifts its threshold. Endorsement of at-fault posters rises by 3--4 percentage points per 10 points of share; without such users the Qwen model stays at base, and the effect persists at matched KL and under a goal-blind reward. Signal-detection analysis shows that the criterion for endorsing falls while discrimination between at-fault and not-at-fault posters does not decline. An explicit request to be told if one is wrong gives no measurable protection: at-fault posters who make it receive nearly the same rise as those who do not (+0.11 vs. +0.13). When the goal is stated during training, the policy instead largely targets, endorsing seekers regardless of fault.
Standard sycophancy and empathy benchmarks miss the shift; five of six usable released empathetic-RL checkpoints also endorse at-fault posters who ask for honesty more than their base models, though a pre-registered prediction for two held for only one. The user mix is a training hyperparameter to audit and report. 

---
# Uncovering the Tells: Efficient Black-Box Audits for LLM Data Contamination 

**Authors**: Dominik Bernard Lau, Filip Pawlicki, Jan Dubiński, Sebastian Cygert  

**Link**: [PDF](https://arxiv.org/pdf/2611.30639)  

**Abstract**: Existing methods for assessing training data contamination in Large Language Models achieve high accuracy, but fundamentally rely on gray-box access, rendering them inapplicable to fully black-box environments. This limitation forces auditors to rely on full-sequence sampling or rigid heuristics, both of which incur prohibitive inference costs and struggle to isolate the underlying memorization signal. To overcome this, we propose a generation-based black-box framework that infers context-induced information gain from targeted token discrepancies. We introduce a budget-constrained auditing regime, building on the empirical observation that the membership signal is highly sparse and concentrated in a few highly informative tokens. We formalize this sparsity, showing why uniform sampling needs many more samples than a policy that targets informative positions. Empirically, our approach achieves state-of-the-art black-box separation accuracy while needing about 3x fewer queries than uniform position sampling. Finally, we apply this methodology to closed frontier models, exploring its capacity and limits in surfacing potential training data overlaps on commercial APIs. 

---
# Auxiliary Clues Aware's Geometry Problem Solving 

**Authors**: Qigong Lei, Qingsong Wang, Wang Lin, Qianyi Yang, Chang Yao, Jingyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30617)  

**Abstract**: Geometry Problem Solving (GPS) serves as a rigorous test of the complex reasoning capabilities of Multimodal Large Language Models (MLLMs), particularly when solutions require auxiliary constructions absent from the input diagram. However, explicit diagramming methods suffer from unstable image generation quality and cascading errors, while existing implicit attention mechanisms primarily focus on visible elements, limiting their ability to support reasoning that requires additional geometric constructions. To address these challenges, we propose \textbf{AuxCA} (Auxiliary-Clues-Aware GRPO), a reinforcement learning framework that helps models learn transferable auxiliary-construction strategies that can be applied beyond the training domain without explicit diagram generation at inference time. Specifically, we introduce Auxiliary Clues Dependency (\textbf{ACD}) to quantify the influence of auxiliary visual clues on token generation and integrate it into GRPO through fine-grained advantage weighting. Furthermore, we incorporate an auxiliary incentive reward to encourage the effective use of auxiliary geometric clues in correct solutions. Extensive experiments demonstrate that AuxCA-8B outperforms the explicit-plotting baseline Bagel-Canvas and achieves competitive performance with leading proprietary models on auxiliary-dependent plane geometry problems. Improvements on analytic geometry, solid geometry, and external benchmarks further suggest that the learned strategies can transfer beyond the training domain. Code is available at this https URL 

---
# Interrogation Protocols for AI Control 

**Authors**: Aniruddh Pramod, Matthew Khoriaty, James Lucassen, Adam Kaufman  

**Link**: [PDF](https://arxiv.org/pdf/2611.30583)  

**Abstract**: As the intelligence of an AI agent increases relative to a trusted monitor, so does the risk that the monitor will fail to understand the agent's actions or to evaluate them confidently. We introduce interrogation protocols to a high-stakes AI control setting, and allow the monitor to ask the untrusted agent questions about confusing actions. For interrogation to improve safety, a weaker model must be able to distinguish misleading explanations produced by an attack policy from genuine ones produced by an honest policy. We find that on a held-out subset of BashArena, an interrogation protocol improves safety at a 1% audit budget from 72.6% for an actions-only baseline, which does not use interrogations, to 79.5% (+7.0 pp [+0.9, +20.1], 95% paired bootstrap interval), but most of this uplift can be captured by having a trusted model state a specific concern. More broadly, our work suggests that interrogating agents can improve safety, allowing monitors to catch more attacks at a larger capability gap. 

---
# Seeing Is Locating: Spatial Awareness and Geographic Bias in Decision Language Models 

**Authors**: Wilson Wongso, Hada Melino Muhammad, Xiachong Lin, Hao Xue, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2611.30564)  

**Abstract**: Decision models, which output probability distributions over candidate labels instead of generating text, are increasingly used as classifiers and evaluators for inputs associated with real-world locations. However, it remains unclear whether they can leverage spatial context and whether their predictions are geographically biased. We introduce GeoDecisions, a multimodal benchmark for evaluating the spatial awareness and geographic bias of decision models. GeoDecisions comprises two tasks: (1) geo-aware image classification, and (2) location rating on objective and subjective topics. Evaluating six decision models against supervised task-specific models, we find that spatial awareness requires vision: textual coordinates alone provide insufficient context for classifying everyday photos and satellite images. We also find substantial geographic bias. On everyday photos, despite achieving higher accuracy, open-weight decision models exhibit more spatially clustered errors than supervised models. Moreover, across all five subjective topics, every decision model gives lower ratings to locations with lower socioeconomic conditions. Finally, we propose a training-free method that combines decision models with a location prior to improve their spatial awareness without substantially increasing performance bias. Our code is available at this https URL. 

---
# Body and Soul: Decoupling Robotic Brain and Policy with RoboHarness-Zero 

**Authors**: Yihao Shen, Yuxuan Chen, Xuantong Liu, Jun Gao, Weijian Luo, Tianyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30550)  

**Abstract**: Vision-language-action (VLA) policies can execute individual skills but fail to complete tasks that require combining several of them. In our audit of Xiaomi-Robotics-1, an estimated 43\% of failures involve choosing the wrong target or losing track of task progress. Wrapping a frozen VLA in a vision-language model (VLM) harness is a natural remedy. Existing harnesses can gain substantially from task- and policy-specific preparation, raising the question of how much of that gain transfers to a new task or policy. An ideal harness should (i) need minimal preparation for a new task, (ii) need no redesign for a new policy, and (iii) preserve standalone capabilities the policy already has. To the best of our knowledge, our \ours{} is the first harness built this way. Its premise is that choosing targets and tracking progress are easy for a VLM such as GPT-6 without task preparation, while the frozen VLA already handles continuous, contact-rich motion. The frozen VLA therefore executes continuous control while the VLM plans zero-shot and monitors execution; when progress stalls or targets drift, the VLM first rephrases the instruction and, only as a fallback, executes short generated code over generic motion primitives before returning fine control to the VLA. The supervision rules stay fixed across policies, requiring only a thin interface adapter and an automatically profiled skill configuration per policy. On 200 RoboCasa365 scenes, \ours{} with GPT-6 Astra raises the success rate of frozen Xiaomi-Robotics-1 from \textbf{\color{blue}57.5\%} to \textbf{\color{blue}70.5\%} and preserves \textbf{\color{blue}90.4\%} of its standalone successes (104/115) while rescuing 37 of 85 failures. On a shared 50-scene subset, the same rules add 8 to 10 percentage points to each of four other frozen policies, while existing preparation-based harness systems often fail to transfer to a new policy. 

---
# CyberDualEval: Measuring Dual-Use Cyber Risks in Frontier Language Models 

**Authors**: Magnus Saebo, Francesco Piccoli, Maxwell Watson, Pranjali Thakur, Yangruibo Ding, Zhuo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30533)  

**Abstract**: While frontier language models are becoming capable cybersecurity assistants, they also create a dual-use cyber risk: the same capabilities that help defenders identify and analyze vulnerabilities can similarly help attackers weaponize them. Existing cyber benchmarks measure either raw offensive capability or generic refusal behavior, overlooking the dual-use cyber assessment of preserving benign security utility while refusing assistance that materially advances misuse. We introduce CyberDualEval, a benchmark with 430 vulnerable tasks for measuring dual-use cyber risk in frontier language models. Each task is decomposed into three phases of increasing operationalization: security analysis, proof-of-concept generation, and exploit generation. To quantify the residual cyber risk, we annotate each task with a Task Weaponization Score that measures how much work remains to turn the resulting artifact into unauthorized cyber misuse. CyberDualEval assesses ten closed- and open-weight language models both on correct compliance and on task performance, developing four deterministic mechanisms for determining per-phase success on a large subset of the dataset. Our evaluation reveals that none of the existing frontier models effectively navigate helping with safe requests and refusing risky ones, characterized by being overly-cautious. On the other hand, recent open-weight models rarely refuse and demonstrate strong validated accuracy on risky requests. 

---
# PackLoop: Row-Packed Self-Speculative Decoding for Looped Language Models under Concurrency 

**Authors**: Runguo Li, Dezhen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30205)  

**Abstract**: Looped language models reuse one block of weights T times per token, trading parameters for test-time depth, but autoregressive decoding then pays T sequential passes per token. Self-speculative decoders recover 2-10x for a single request, yet their gains fade as requests share a GPU, and current practice is to speculate less as load grows. We show that what fails under load is the layout of speculative work, not speculation. In concurrent decoding the unit of cost is the row, one request advancing on one loop, which reads that loop's key-value (KV) cache once; additional positions inside a row are nearly free. Pipelined schedules such as the wavefront place every speculative position on a different loop, hence in a row of its own, so each token is still read T times. We present PackLoop, built on one principle: speculate only inside rows that are read anyway. Deep loops accumulate positions and advance them as a block in one row, context-copy proposals join the row that the first loop already reads, and a closed-form admission rule, the economic order quantity of inventory theory, sets how long to accumulate from a cost model calibrated once per engine. PackLoop is exact: in FP32 its output is token-for-token identical to autoregressive decoding, also under sampling and early exit. Across 164 load points spanning three GPUs, four Ouro models, six benchmarks, and contexts up to 32K tokens, one PackLoop configuration is faster than the best existing schedule, chosen per point, at 161 points (median 1.25x, up to 1.76x) and never significantly slower. Its lead grows with concurrency and context length, the regime in which existing self-speculation gains least. 

---
# Same Executions, Different Gaps: Judgment Rules in Web-Agent Evaluation 

**Authors**: Zelong Wang, Weiyue Li, Fei Xue, Junru Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.30190)  

**Abstract**: Success-rate gains guide the development of web agents. Yet knowing which agent ranks first does not tell us how much confidence to place in its reported advantage. We study the comparison itself: how success is defined, what evidence supports the judgments, and how the difference between agents is estimated. On fixed web-agent executions, changing a judge from predicting likely completion to requiring recorded support can remove most of a reported lead without changing the ranking. This effect persists when unresolved completion is distinguished from supported failure. Controlled source additions explain how missing relations between task requirements, objects and observations can conceal a difference or create an apparent advantage. We also find that making both agents' score estimates more accurate can make their difference less accurate, because their original errors partly canceled. To estimate a chosen outcome difference, we combine judge predictions with reference labels for sampled task pairs. Across four released execution systems, regression fitted on the same 50 labeled pairs reduces estimation error relative to direct paired sampling. The results give agent researchers a practical basis for assessing claimed improvements: make the comparison target explicit, retain the evidence needed to check it, and use paired labels to estimate the difference directly. 

---
# OpenViVE: New Benchmark and Dual-Stream Co-Attention Framework for Vietnamese Visual Entailment 

**Authors**: Bao Quoc Trinh, Tin Van Huynh, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30161)  

**Abstract**: Visual Entailment (VE) determines whether a textual hypothesis is entailed, contradicted, or unverifiable from an image. However, existing VE benchmarks are largely English-centric and do not capture Vietnamese-specific linguistic distinctions or culturally grounded visual contexts. To address this gap, we introduce OpenViVE, the first large-scale native Vietnamese VE benchmark, containing 62,268 balanced image-hypothesis pairs from 10,378 culturally grounded images. OpenViVE is constructed using an Artifact-Aware Hybrid LLM-Human Pipeline that combines constrained generation, human validation, and PMI-guided refinement. The pipeline substantially reduces hypothesis-only shortcuts, with Contradiction accuracy dropping from 96% to 66% after artifact refinement. We further evaluate a dual-stream PhoBERT-ConvNeXt framework with bidirectional co-attention to align visual evidence with Vietnamese hypotheses. The framework achieves 84.84% accuracy and 84.90% macro-F1 on the test set, while PhoBERT consistently outperforms multilingual XLM-R variants. These results highlight the importance of language-specific encoding and explicit cross-modal interaction for Vietnamese VE. 

---
# When Does Optical Context Compression Pay Off? A Decode-Aware Cost Analysis of Reasoning VLMs 

**Authors**: Ziqing Ge  

**Link**: [PDF](https://arxiv.org/pdf/2611.30130)  

**Abstract**: Rendering long text as images lets a vision-language model (VLM) read a document with fewer input tokens, and evaluations often count mainly these prefill tokens. We show that for reasoning models this accounting is incomplete. Across five models and two long-context benchmarks, reading the rendered document makes the model think longer: decode length grows by 1.5-3.5x on the four general-purpose VLMs tested, and under typical output pricing this decode tax makes image input more expensive than text for two of the five models on QuALITY. Unrecoverable information loss is not sufficient to explain the tax at moderate compression: Qwen's OCR of the same images recovers 98.7% of words, and feeding Kimi-K3 its own transcription as text restores text-mode reasoning length and accuracy. Whether compression pays off reduces to an exchange rate, (1-1/c)/(r-1), that must exceed the workload's price-weighted ratio of output to input tokens; this accounting decomposition shows why Kimi-K3 and DeepSeek-V4.1-Flash can benefit and Qwen and GLM cannot. None of the training-free signals we test predicts the tax before generation. As a practical heuristic, an image-first policy that sends only inputs with a per-input compression ratio of at least 1.6 to images and falls back to text once image-mode reasoning costs as much as the input's prefill saving cuts the bill by 6-14% on Kimi-K3 and DeepSeek-V4.1-Flash at an accuracy cost of 0.3-1.2 points; on QuALITY, asking for a reasoning budget on both paths is a stronger lever. 

---
# STUN-Bench: Truly Unseen Tools for Evaluating Shared Tool Experience 

**Authors**: Abhijit Chakraborty, Ni Trieu, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.30034)  

**Abstract**: A tool router retrieves a few candidate tools per request and lets a language model choose. Methods that share experience across organizations add past requests about each tool to what the router retrieves. StableToolBench calls some of its test groups tests on "unseen" tools, yet the ToolBench instruction pool, which retrievers and experience-sharing methods learn from, contains every test query and labels every gold tool of the two single-tool groups among them. We release STUN-Bench (StableToolBench, Truly UNseen), a split whose experience pool has no item that labels or names 172 held-out tools. Unseen here means absent from shared experience and retriever training data, not from pretraining. On the split we find crowding: experience pooled from five simulated clients, added to an index of tool descriptions, outranks the descriptions of tools no client has used, which then often drop out of the candidate list. Under the Jina-v2 retriever, routing accuracy on held-out tools falls from 0.534 to 0.278. Retrievers trained for tool retrieval lose much less, and keeping experience in a separate index recovers most of the loss under Jina-v2. The split also reverses a conclusion: with every test tool labeled in the pool and our default leak filter applied, shared experience beats descriptions alone; on STUN-Bench it falls below them. The split, leak filter and evaluation harness are available at this https URL. 

---
# Tracing Attacker Diversity Reduction in Adaptive LLM Red Teaming 

**Authors**: Chris Tava  

**Link**: [PDF](https://arxiv.org/pdf/2611.29960)  

**Abstract**: Reinforcement learning can increase the success of automated red teaming but it also narrows the attack vector. This study asks where in the attacker-defender-judge learning loop does the narrowing begin and which mechanisms lead to a narrowing attack. Six mechanisms (defender adaptation, success edge, reward reinforcement, row cap, update size, and measurement) are tested across three experiment arms (co-trained, frozen defender, half learning rate) for twenty-four iterations. A frozen defender narrows attack diversity retention (ADR) earlier 0.496 relative to co-trained 0.891 at iteration twelve. Defender updates delay narrowing but dont prevent it. Successful attacks ("winners") have narrower attack diversity than the full set of generated attacks. Attacks closest to previous winners succeed 10-15 percentage points more often than those closest to previous failures. This winner-neighbor advantage holds after a defender update and for a fixed defender too. Attacks with a common opening have a common-opening advantage of 5-16 percentage points. The training row cap and update size account for little impact on ADR, and attack length accounts for only part of the ADR decrease. Narrowing appears to originate from the success only reward through the group-relative training advantage to the attacker update, with a persistent success edge setting the direction. 

---
# Yggdrasil: Tree-Routed Recurrent Experts for State Tracking and Fast CPU Inference 

**Authors**: Cosmo Santoni  

**Link**: [PDF](https://arxiv.org/pdf/2611.29954)  

**Abstract**: Parallel-trainable recurrences such as Mamba, linear attention, and DeltaNet offer constant-cost streaming inference, but they suffer from two limitations: 1) they struggle to track discrete state, and 2) they read every parameter at every step, tying model capacity directly to CPU latency. We introduce Yggdrasil, a recurrent architecture that decouples capacity from per-step compute by looking up each layer's transition in a large, tree-routed bank of recurrent experts. Each router addresses its bank using the current token and the state of the layer below. We prove that this cross-layer context is necessary---a single token-indexed layer of non-expanding transitions with finitely many reachable states cannot perform conditional resets---while avoiding feedback from the layer's own past preserves parallel-scan training. Because each step traverses one tree path and reads a single expert per layer, active weights stay within a few hundred kilobytes in CPU L2 cache while the full bank resides in RAM. Among the parallel-trainable models we evaluate, Yggdrasil is the only one to solve all five state-tracking tasks. At matched active parameter counts, it achieves the lowest test bits on enwik8 (1.540 bpb) and the human genome hg38 (1.567 bpb), while stepping $3.0\times$ and $3.3\times$ faster on a single CPU core ($6.7\,\mu\mathrm{s}$ and $5.8\,\mu\mathrm{s}$) than the fastest streaming baseline, and delivering the highest GPU throughput of all models at batch 1024. On the Genomic Benchmarks, it is the strongest causal model at matched active parameters (86.0% mean accuracy). 

---
# SAEPO: Sparse-Autoencoder-Guided Credit Assignment for Policy Optimization 

**Authors**: Yusong Zhao, Zihao Zhao, Zirui Zhang, Tunyu Zhang, Ziyan Wang, Yi Wang, Chengzhi Mao, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29936)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated strong performance in improving large language model (LLM) reasoning, yet its credit-assignment signal remains coarse. Outcome verifiers provide scalable supervision by judging whether a completed response is correct, but they do not directly indicate which intermediate reasoning tokens contribute to success or failure. As a result, policy optimization can be inefficient, especially when training base models directly from sparse outcome feedback. In this paper, we introduce SAEPO, a Sparse AutoEncoder-guided token-level credit assignment method for outcome-supervised Policy Optimization. Rather than treating correctness only as a rollout-level (sequence-level) signal, SAEPO uses a pretrained sparse autoencoder (SAE) to build token-level semantic representations within reasoning trajectories; it scores each token based on the SAE concepts (e.g., penalizing wrong "final-answer" tokens but rewarding correct "reasoning-process" tokens), learned in an unsupervised manner, to achieve fine-grained credit assignment. SAEPO is a plug-in advantage-shaping module compatible with various policy optimization algorithms. We instantiate SAEPO with both GRPO and GSPO, covering token-ratio and sequence-ratio policy-gradient objectives, and evaluate SAEPO across three LLM backbones on six math and two code benchmarks. Empirical results show that SAEPO improves both the math and code benchmark averages over the matched baselines, with gains of up to 6.63% and 5.57%, respectively. On Qwen3-1.7B-Base, training curves also show faster early accuracy growth. Feature and token-level analyses suggest that the SAE-derived signal favors question-relevant concepts and reasoning tokens. 

---
# ReqSeek: Transformer-Based Automatic Requirements Identification for Building Large Requirements Datasets 

**Authors**: Mohammad Kasra Habib, Umm-e-Habiba, Daniel Graziotin, Stefan Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2611.29722)  

**Abstract**: Large, diverse, and high-quality datasets are significant for advancing AI in requirements engineering, yet automated identification of requirements from Software Requirement Specifications (SRS) remains a challenge. Automatic Requirements Identification (ARI) methods demonstrate limited generalization to unseen documents across heterogeneous SRSs. This work presents ReqSeek, a transformer-based, domain-agnostic ARI model for recognizing ISO 29148-style requirements in SRS documents, supporting dataset construction. We curated the Automatic Requirements Identification Dataset (ARID), comprising 2,396 training instances from 23 heterogeneous SRSs, and fine-tuned four transformer architectures. RoBERTa was selected as the base model for ReqSeek and achieved 95% precision, recall, F1, and F0.89 on the held-out evaluation set. Exploratory human-model benchmarking showed performance comparable to humans while requiring substantially less prediction time than manual annotation. Post-hoc evaluation on out-of-domain (OoD) SRSs provided evidence of requirement identification beyond ARID. Robustness analysis demonstrated ReqSeek's insensitivity to minor linguistic variations while maintaining sensitivity to major ISO 29148-style violations. We applied ReqSeek to construct the Software Requirements Dataset (SwaRD), identifying 10,201 ISO 29148-style requirements from 231 industrial SRSs. SwaRD aligns with textual and semantic properties of human-curated datasets while exceeding them in scale and domain diversity. Our results suggest that transformer-based ARI can provide precise and generalizable requirements identification. 

---
# Loop-Count Schedules in Looped Transformers: An Empirical Study 

**Authors**: Konstantin Kolokolov  

**Link**: [PDF](https://arxiv.org/pdf/2611.29704)  

**Abstract**: Looped transformers apply one block of layers several times with shared weights, trading parameters for repeated computation. We study how the schedule of the loop iteration count R during training determines whether each extra pass acts as a full layer or as a small correction, using a controlled ladder of 46M-110M-parameter models trained on 2B tokens with learned residual gates on every loop pass and three schedules: R fixed to 2, R sampled uniformly from {1,...,6}, and R sampled from {2,3,4}. We find that the lower bound of the sampling range is the decisive choice. If the sampled range includes R=1, so that some training steps run the block only once and compute the loss on that single pass, the gates collapse to 0.1-0.5 and each additional pass becomes a small correction to the first; if the range starts at R=2, so that every training step runs the block at least twice, the gates stay at 0.6-0.9 and each pass keeps the strength of a full layer, while the model can still be run at any R from 2 up to twice the largest trained value with validation loss within a few hundredths of a nat of its best. The {2, 3, 4} schedule gives the best looped model, 2.909 nats against 2.951 for the parameter-matched plain model, and degrades by only 0.038 nats when run at twice its trained depth. We further show that under a fixed R the direction of the residual stream converges geometrically beyond the trained depth while the loss diverges, which refines a recent finding that static-depth training does not converge, and that reusing the first pass's keys and values on later passes costs 0.01-0.03 nats in every regime while making the per-token iteration uniformly contractive and halving the inference cache. The loop recovers about a third of what real depth provides at equal parameters. The code, trained weights, and per-token traces are released. 

---
# Adaptation or Poisoning? Security Risks of Conversational Test-Time Learning in Large Language Models 

**Authors**: Honghao Feng, Shanhan Qu, Kai Tong, Shuquan Man, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29532)  

**Abstract**: Test-time learning (TTL) enables large language models to adapt to unlabeled inputs during deployment. The model updates a lightweight adapter by minimizing the next-token prediction loss on the input it receives. When these updates persist across conversation turns, every user query serves both as the request answered in the current turn and as adaptation data for later turns. By choosing the queries used for adaptation, an attacker can influence parameter updates and weaken safety alignment through ordinary interactions with a TTL chat service. To demonstrate this risk, we introduce Progressive Test-Time Adaptation Poisoning (P-TAP), a query-only multi-turn attack that jailbreaks the model by poisoning its session adapter through the service's own updates. A sequence of escalating adversarial queries shapes the adapter turn by turn; the final harmful request is then answered under the accumulated state without itself being used for adaptation. Across four open-weight instruction-tuned models, enabling updates raises attack success rates on average by 13.00 percentage points under greedy decoding and by 22.25 points under best-of-10 sampling, relative to matched sessions receiving the same queries without updates. Answering adversarial queries but adapting on unrelated text or on benign text about the same topic removes most of this increase. Among existing safeguards, screening incoming queries flags every attack stream before the final request, while a perplexity-based detector of unsafe adaptation flags none of the successful attacks. 

---
# Recursive Self-Editing: LLMs Program Their Own Weight Edits for Self-Improvement 

**Authors**: Linghua Zhang, Yunhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.29513)  

**Abstract**: Recursive self-improvement asks a system to improve the very system that improves it. Today's language models improve themselves through data, training recipes or scaffolds. Whenever their weights change, gradient descent writes the update, not the model. We let the model write its own weights. In SELFWRITE, each generation the current model studies its own reasoning on fresh training problems, inspects its own weights and activations in a persistent Python session, and writes a program that constructs a low-rank weight edit for itself from white-box primitives (singular directions, activation contrasts between its right and wrong answers, rank-one projections, tensor computations), with no training or gradient tool. Every candidate is graded once on a hidden gate set against the current model; the best strictly improving edit is merged into the weights, and the edited model writes the next edit. On Qwen3.8-27B and math problems from DAPO-Math, five generations of SELFWRITE raise gate accuracy from 73.95% to 76.45% and accuracy on a held-out test split that the loop never sees from 76.30% to 77.75% (avg@4) and from 82.60% to 84.80% (pass@4), without a single gradient step. A language model can author useful changes to its own weights without gradients, and the edited model keeps writing useful edits. 

---
# Beyond Accuracy: A Diagnostic Benchmark for Hypothesis-Driven Experiment Planning in LLM Agents 

**Authors**: Ryo Kuroki, Amin Mansouri, Philippe Schwaller  

**Link**: [PDF](https://arxiv.org/pdf/2611.29467)  

**Abstract**: Evaluating large language model (LLM) agents on scientific tasks typically reduces to final-task accuracy, which obscures where reasoning succeeds or breaks down. We introduce a benchmark and diagnostic evaluation framework for hypothesis-driven adaptive experimental planning in closed-world system identification, instantiated in kinetic mechanism identification. The framework goes beyond final accuracy in two ways. First, we evaluate exploration behavior by quantifying the extent to which agents exhibit space-filling or corner-seeking patterns in the experimental design space. Second, we decompose the reasoning process into four interpretable steps: belief update, hypothesis discrimination, discriminative experiment design, and predictive reasoning under interventions, each scored separately on reasoning traces. Across ten LLM agents, we find that stronger agents outperform both adaptive (Bayesian optimization) and non-adaptive (Latin hypercube) baselines and exhibit exploration patterns inconsistent with naive heuristics. Decomposing reasoning reveals that the stronger agents score consistently well across all four steps, whereas weaker agents often succeed at belief update and hypothesis discrimination but are less reliable at translating them into actionable designs and sharp contrastive predictions. Independent expert annotations support predictive reasoning under our stated validation criteria, while evidence for discriminative experiment design is positive but less consistent. The results suggest a gap between recognizing plausible hypotheses and translating them into contrastive predictions, with more tentative evidence for a corresponding experiment-design bottleneck. The benchmark website is available at this https URL. 

---
# MAS-BIT: A Multi-Agent System for End-to-End BraIn-to-Text Decoding 

**Authors**: Xinjie Zhang, Linyang He, Yizi Zhang, Yuesheng Ma, Nima Mesgarani  

**Link**: [PDF](https://arxiv.org/pdf/2611.29457)  

**Abstract**: Speech brain-computer interfaces aim to restore communication by translating neural activity into text, yet end-to-end decoders have historically trailed cascaded systems in accuracy. We introduce a Multi-Agent System for BraIn-to-Text (MAS-BIT), the first multi-agent framework for end-to-end brain-to-text decoding from intracortical recordings. MAS-BIT coordinates three specialized agents: a neural agent that interprets continuous neural representations, a phonetic agent that decodes soft phoneme posteriors, and an arbitration agent that integrates their complementary evidence with direct access to neural representations to generate the final transcription. The expert agents communicate with the arbiter through token-embedding-based latent communication, constructing continuous messages as probability-weighted combinations of token embeddings. These messages retain alternative token hypotheses and provide a differentiable interface through which sentence-level supervision jointly trains the neural encoders and specialized agents across participants. MAS-BIT achieves state-of-the-art end-to-end decoding performance on the Brain-to-Text '24 and '25 benchmarks, with word error rates of 6.929% on '24 and 3.606% on the '25 public test set (3.892% on the private test set). These results surpass the published BIT cascade on the '25 public set and narrow the gap on '24, without ensembling independently trained models or using external language-model rescoring. 

---
# SAE-Audit: Clinical-Trial Prediction via Inference-Time Optimization of Black-Box LLMs Audited by Sparse Autoencoder Evidence 

**Authors**: Junbo Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.29368)  

**Abstract**: Clinical trials are costly and often take years to complete, making early outcome prediction valuable for trial planning. Large language models (LLMs) are increasingly used to forecast clinical-trial outcomes, yet their predictions remain unreliable and their rationales may not faithfully reflect the evidence underlying their answers. On tasks such as predicting whether a registered clinical trial will complete or terminate early, frontier LLMs perform only modestly above chance in our experiments. We introduce SAE-Audit, which optimizes a black-box LLM at inference time with evidence obtained from a deployable open model. A frozen sparse autoencoder (SAE) on Llama-3.1-8B-Instruct is converted into a vocabulary of clinical concepts. Sparse probes fitted on past trials determine which concepts are associated with the outcome. For a new protocol, the concepts detected by the probe are written as evidence, each with the direction and strength of its association with the outcome, and the LLM being audited optimizes its answer in the light of this evidence. Without additional training, SAE-Audit raises balanced accuracy of deepseek-v4-flash by 9.6--12.0 points on five clinical-trial prediction tasks and raises GPT-5.6-terra by 8.0 points on trial completion, exceeding other agentic inference-time optimization methods by a large margin. As an auxiliary, SAE probes module within SAE-Audit is itself accurate and auditable, providing natural-language explanations directly derived from verified concept contributions and protocol evidence, without requiring additional LLMs at inference time. SAE-Audit provides verifiable, concept-level evidence for more accurate and auditable clinical-trial prediction. It can also be embedded into biomedical AI agents such as Biomni to optimize their performance. 

---
# EMIP: Embodied Multi-Party Interaction Platform 

**Authors**: Koichiro Terao, Rikunari Sagara, Moe Nagao, Naoto Iwahashi  

**Link**: [PDF](https://arxiv.org/pdf/2611.29333)  

**Abstract**: Experimental platforms that enable multiple participants to engage in spoken dialogue and perform physical actions concurrently in a shared environment are important for studying cooperation between humans and embodied AI agents. In this paper, we propose EMIP (Embodied Multi-Party Interaction Platform), a virtual environment platform built on Minecraft. Through an Internet-based client-server architecture, EMIP connects multiple humans and multiple AI agents to the same virtual environment and supports full-duplex speech communication and simultaneous physical interaction. It provides interfaces based on the Model Context Protocol (MCP) for environmental observation, physical action execution, and audio input and output, enabling researchers to develop their own agent control methods using large language models (LLMs). Additional features include centralized control of an agent group through a single EMIP agent client, collaborative object transportation by multiple agents, and experimental support functions such as recording and retrieving observation histories. By integrating these features into a reusable platform, EMIP aims to support research on human-human, human-agent, and agent-agent cooperation under different participant configurations and agent control methods. Usage examples illustrate concurrent speech and physical action, task coordination, and overlapping speech. The source code for the platform is publicly available on GitHub (this https URL). 

---
# Counting Symbols on Technical Drawings: A Benchmark and Verified Rewards for MLLMs 

**Authors**: Junxian Li, Ruixuan Yang, Zhenhua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.29309)  

**Abstract**: Today's MLLMs can count objects in natural images. However, when asking them how many bolt holes appear in a mechanical drawing or how many nitrogen atoms are in a chemical structure, they often struggle. Counting in technical drawings remains challenging, as objects can be small, densely packed, or even implied by drawing conventions. To study this problem, we introduce CountDrawBench, a benchmark including floor plans, electrical schematics, mechanical drawings, and molecular structures, with 1,574 counting questions and instance-level annotations where available. Besides evaluation, we introduce Code-Verified RL, a reinforcement learning framework that teaches MLLMs to count with executable feedback. Our method combines three complementary rewards: a count reward that checks the final answer, an instance-matching reward that verifies object locations, and a zoom-based reward that checks counting consistency across local image regions. All three are integrated into a unified GRPO objective. Extensive experiments reveal that even leading MLLMs struggle on CountDrawBench, with the best evaluated model achieving only 59.3% accuracy. Our method improves Qwen3.5-9B from 49.3% to 54.3% accuracy, outperforming all evaluated open-source baselines. These results establish CountDrawBench as a challenging testbed for technical drawing understanding and demonstrate the value of code-verified rewards in these tasks. Code and data will be released upon acceptance. 

---
# Which Failure Does an Intervention Repair? When Agent Interventions Help, and Where Self-Evolving Skills Stall 

**Authors**: Zihan Jin, Gaoyuan Du, Jiate Li, Dongze Ye, Shawn Li, Yue Zhao, Defu Cao  

**Link**: [PDF](https://arxiv.org/pdf/2611.29290)  

**Abstract**: An intervention that improves one agent can break another, and adding a stronger harness on top need not repair the damage. Agent interventions such as skills, harnesses, and extra inference budget are nonetheless usually credited with one average gain, measured under one setup. We ask instead when each intervention helps and which failure it repairs. A matched factorial study over nine benchmarks, four models, four skill conditions, four harness levels, and three budgets, with nearly 0.9 million trajectories, changes one axis at a time. No configuration is best everywhere, interventions interact rather than add, and a harness can absorb a skill's gain, rescue a failure the skill causes, or leave the gain intact. The failure left by the current system points to the fix: missing knowledge to a stronger model, execution errors to a richer harness. Yet feedback is not recovery: a matched recovery repairs mainly failures that leave a visible error signal. Skills that evolve from the agent's own experience hit a bottleneck: they rarely beat a short human-written skill, can learn dataset shortcuts, break when moved to another model, and cannot supply missing facts, which outside information does. A benchmark-level switch to the model favored by a 20-task labeled pilot achieves both higher performance and lower cost than always using the stronger model. Agent interventions should be judged by which failure they repair and under which configuration. 

---
# On the Interplay between Backtracking and Latent Confidence Directions in Reasoning Models 

**Authors**: Stephen Cheng, Zahra Sodagar, Sandeep Soni, Dinesh Manocha, Sarah Wiegreffe  

**Link**: [PDF](https://arxiv.org/pdf/2611.29265)  

**Abstract**: Reasoning models deliberate over multiple answer proposals, often resulting in excessively long traces due to backtracking, even when the correct answer has already been reached (Across three 32B models, ~87% of rollouts contain 1 unique answer yet still make ~8 answer proposals on average). Given this potentially "decorative" chain of thought (CoT) reasoning, can we look to the model internals to determine if the model will backtrack or commit to an answer proposal? Operating under the hypothesis that models maintain an internal representation of their uncertainty, we seek to extract this representation and characterize its relationship with backtracking behavior at answer proposals. By querying the model's verbalized confidence following answer proposals throughout a response trace, we extract a linear direction for latent confidence that separates high and low verbalized confidence. This direction can steer verbalized confidence as expected, but it surprisingly can also steer the next token probability of backtracking, performing on par with a linear direction extracted directly from backtracking examples. We then test the direction under progressively more realistic interventions. Steering continuously from a mid-trace answer proposal modulates response length (~-1.3k - ~+2.5k tokens) and the number of answer proposals (-4.18 - +4.75) in both directions, with minimal cost to accuracy. Applying the confidence direction from the start of generation, where proposal positions are unknown, still shortens traces at comparable accuracy, and its effect surfaces in the linguistic confidence of proposals. Overall, our study reveals the intricate interplay between a model's internal confidence and its proclivity to backtrack or commit to an answer. 

---
# NoeVera: Can Knowledge Guide Experiments and Experiments Refine Knowledge? 

**Authors**: Pinaki Prasad Guha Neogi, Spyros Mastorakis, Ankur Mallick, Saravan Rajmohan, Victor Rühle  

**Link**: [PDF](https://arxiv.org/pdf/2611.29255)  

**Abstract**: With the growing demand for large language model (LLM) services, optimizing LLM serving systems is becoming increasingly complex, and the bottleneck is no longer predicting how a given configuration will perform. The harder problem is deciding what to investigate: which mechanisms matter for a workload, which configurations are worth testing, and what evidence a decision still needs. In this work, we present NoeVera, an agentic Second Brain that goes beyond just one-shot retrieval by turning papers, code, benchmarks, and experimental results into a persistent, provenance-preserving knowledge base with three planes: source evidence, reusable synthesis, and agent memory. Decision-aware retrieval assembles compact evidence packs, role-specific agents reason over them and invoke a simulator (Vidur or GAUSS) whenever quantitative support is needed, and every finding is written back as a scoped evidence record, so knowledge guides experiments and experiments refine knowledge. In configuration search over four serving scenarios, NoeVera comes within 5% of the optimum in a median of 6-7 simulator calls, versus 9 for Bayesian optimization and 15 for a knowledge-free agent with the same tools; on A100 GPUs, its representative configurations deliver ~1.30x vLLM-default throughput and meet every scenario's latency objective. When reused, provenance-preserving evidence records over-generalize on only 8% of applicability traps, versus 28% for a general-purpose agent memory. On the held-out 48-task split of a 72-task benchmark, NoeVera scores 29% higher than an equally equipped agent and 36% higher than its single-call retrieval counterpart (0.80 vs. 0.62 and 0.59), and even when baselines receive NoeVera's own prompts, a statistically significant part of its evidence-evolution advantage remains. 

---
# GameAnything: From Any Inspirations to Playable Game Worlds 

**Authors**: Wenyi Wu, Siqi Liu, Yuru Feng, Zixuan Song, Aayush Salvi, Sibo Zhu, Kun Zhou, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29238)  

**Abstract**: AI-driven game development holds the potential to transform diverse human inspirations into interactive experiences. However, existing systems and benchmarks primarily focus on explicit development specifications, with limited support for adapting multimodal references and evaluating actual gameplay experiences. We introduce GameAnything, an open-source agentic platform that transforms multimodal inspirations into playable extensions of existing games through iterative development. GameAnything integrates coding agents with an independent playtesting critic, persistent development state, and reusable experience memory to support continuous refinement. We further present GameAnything-Bench, a benchmark spanning diverse reference modalities, game creation objectives, and gaming environments, with exploratory gameplay testing to assess both functional correctness and gameplay quality. Evaluations of nine coding agent configurations reveal that current agents can adapt reference content but still struggle with playable interactions and precise modifications. Moreover, multimodal references improve content fidelity, while iterative refinement yields substantial but varying improvements across agents. These findings highlight both the potential and remaining challenges of inspiration-driven game development. 

---
# IteraSong: LLM-Guided Iterative Symbolic Music Creation in a Web DAW 

**Authors**: Di Lu, Xinran Zhang, Yan Gao, Yuan Zhang, Xiaoqing Wang, Qingwen Zhou, Xiaobing Li, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.29225)  

**Abstract**: Music creation is inherently iterative, yet many symbolic generation systems still center on producing or regenerating isolated outputs. We present IteraSong, an interactive web-based Digital Audio Workstation (DAW) for LLM-guided iterative symbolic music creation. Unlike one-shot generation systems, IteraSong supports a generate-audition-edit-refine loop over a persistent multi-track project. Users initialize a fully labeled pop draft, then progressively refine it through natural-language instructions. The LLM maps requests to a hierarchical action space of editing primitives exposed via Model Context Protocol (MCP) tools, spanning global, track, section, and bar/note scopes. The demo enables attendees to edit multi-track compositions in natural language, apply deterministic and generative edits, and undo/redo changes in real time. 

---
# Direct Feedback Policy Optimization for AI Coding Agents 

**Authors**: Fei Ding  

**Link**: [PDF](https://arxiv.org/pdf/2611.29193)  

**Abstract**: AI coding agents generate, modify, and validate code through multi-turn interactions. Existing reinforcement learning methods often rely on sparse terminal rewards, making it difficult to assess how intermediate decisions affect task outcomes, and group-relative methods also require repeated sampling for each task. We propose Direct Feedback Policy Optimization (DFPO), a single-trajectory reinforcement learning algorithm that trains directly from a single trajectory. DFPO achieves higher task resolution rates than GSPO under matched base models and execution harnesses. 

---
# VeriLogic: Reinforcement Learning over Verifiable Logical Reasoning Traces 

**Authors**: ZeLiang Chen, Yunyao Zhang, Xinglang Zhang, Wenbing Li, Junqing Yu, Zikai Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.29147)  

**Abstract**: Logical reasoning derives conclusions from premises through explicit intermediate inference steps, providing a natural basis for process-level verification and supervision. However, existing learning methods often rely on response-level outcome signals, which overlook heterogeneous correctness within the reasoning trace: a correct answer may contain invalid reasoning, while valid intermediate steps may also appear in an incorrect response. To address this issue, we introduce Reinforcement Learning on Verifiable Logical Reasoning (VeriLogic), which turns internal reasoning traces into explicit and verifiable learning objects. Rather than treating a response as an indivisible learning unit, VeriLogic decomposes logical reasoning into explicit intermediate judgments, proofs, and final decisions, and organizes them into a unified "Grounding - Optimization - Refinement" pipeline: (1) Verification-Grounded Reasoning Environment Construction grounds logical reasoning into explicit intermediate states and formally checkable proof obligations, making the internal reasoning process accessible to verification and learning; (2) Logical Structured Policy Optimization learns from verifier feedback through complementary global and local rewards, jointly optimizing the overall reasoning outcome while providing targeted optimization signals to different intermediate reasoning paths; and (3) Verifier-Guided Iterative Refinement acts on verifier feedback by leveraging proof states and error messages to iteratively diagnose and repair failed reasoning. Experiments on six logical reasoning benchmarks show consistent gains over strong baselines, with average improvements of 6.39% and 5.00% on Qwen3 and Llama-3.1. These results highlight the value of making logical reasoning traces verifiable, learnable, and refinable beyond final-answer supervision. 

---
# When Can a Rewiring Control Test LLM Weight Readers? C2 and an Eligibility Protocol 

**Authors**: Vaishak Girish Kumar, Sanika Vilas Najan  

**Link**: [PDF](https://arxiv.org/pdf/2611.29120)  

**Abstract**: Large language models can predict the outputs of small neural networks from raw weights, but success could reflect recognition of weight-space regularities (fingerprints) rather than computation, and function-preserving controls cannot tell these apart. We propose C2, a counterfactual that permutes the columns of a network's final linear readout: every other parameter, and the readout's singular values, norms and entry multiset, are unchanged, and the function can change in an exactly recomputable way; we verify that each evaluated permutation does. C2 separates fingerprinting from computation only under three conditions, which we state as an eligibility protocol: a strategy restricted to C2-invariant summaries must already predict the target; C2 must change a property specific to the rewiring; and the reader's identity must be attested. No natural setting we tested is eligible. On tiny MLPs a frontier reader route followed the rewired function on all 24 networks, but invariant baselines were near chance and its backend could not be attested. In two public CNN zoos the conditions traded off: weight statistics predicted CIFAR-10 accuracy in-distribution, but C2's effect on accuracy was not specific to the swap, while per-image targets were specific but not predictable from invariant summaries. On a constructed population where a planted fingerprint rule works, all conditions hold and C2 separates a rule-following reader from an executor. A self-hosted Qwen3.8-27B followed the rewired function rather than the shortcut, and showed no evidence of using the weights with thinking disabled. A registered replication on fresh networks confirmed both findings (tracking minus stuck 0.875, 95% interval [0.750, 1.000]), and DeepSeek-R1-Distill-Qwen-14B also followed the rewiring (tracking minus stuck 0.729 [0.552, 0.885]). The transforms, audits and raw records are available from the authors. 

---
# Mind the Trap As LLMs Guess the Data Model: Grain-Aware Guards Against Fan-out in Text-to-SQL and Its Benchmarks 

**Authors**: Ahmed M. Mahran  

**Link**: [PDF](https://arxiv.org/pdf/2611.29009)  

**Abstract**: Text-to-SQL systems built on large language models (LLMs) often aggregate over one-to-many and many-to-many joins, a classic trap: when the joined rows repeat an entity, a SUM or COUNT counts it more than once, yet the query runs and returns a plausible number. We characterize these fan-out errors by one condition on the relation an aggregate reads, which depends on two facts a schema does not show: each measure's grain, and how many rows a join on each key repeats. An audit that tests the condition on the data finds distorted gold answers for 4.2% of the dev questions of BIRD, a widely used benchmark, and BIRD's later, independent revision changed 44% of the answers the audit flags as definite fan-out, against 16% of unflagged ones. We release BIRD-FanOut, 486 BIRD questions with reviewed gold, and GrainTraps, synthetic questions that set this trap and related ones. Across eight LLM agents, functional dependencies (FDs) computed from the data help no more than per-column statistics on GrainTraps, but grain-aware FDs, which also state each measure's grain and the repetition per key, cut fan-out errors on BIRD from 18.4% of runs to about 13%, also on held-out questions and without LLM judges. Accuracy rises less, because most remaining errors are not fan-out and BIRD's hints often prescribe the over-count. Text-to-SQL models should be given the grain of the data, and benchmark gold should be audited on the data. Code, data and benchmarks are available at this https URL 

---
# DoC: Learning to Defer or Commit for Timely, Accurate, and Stable Conversational Diagnosis Prediction 

**Authors**: Xueren Ge, Anthony Cortez, Homa Alemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2611.29003)  

**Abstract**: Conversational diagnosis prediction requires accurate, early, and stable predictions from ongoing clinical dialogue, yet dialogue-level labels do not specify when to commit. We propose a two-phase framework for learning deferral and diagnosis decisions. First, iterative commitment-boundary estimation constructs Defer/diagnosis supervision without annotated commitment times. Second, Defer-aware supervised fine-tuning initializes a policy, refined through reinforcement learning with trajectory-level rewards for timeliness, first-commit accuracy, and post-commit stability. On EMSDialog and MediTOD, DoC commits early, achieving the highest first-commit accuracy among post-trained methods (76.7%/51.2%) and the lowest post-commit error overall (16.4%/48.3%). 

---
# Elicitation format changes zero-shot classification accuracy in a medical vision-language model: a two-center evaluation on colorectal histopathology 

**Authors**: Anay Sinhal, Amit Sinhal, Arpana Sinhal  

**Link**: [PDF](https://arxiv.org/pdf/2611.28997)  

**Abstract**: Background. Verbalized confidence from a hosted medical vision-language model is usually treated as packaging around an unchanged prediction. Methods. We held a nine-class colorectal histopathology prompt fixed except for the requested response format and evaluated MedGemma-27b-it on 10,004 validation and 7,180 external-center PathMNIST patches, with preregistered endpoints, randomized ties, and random floors recomputed for each patch set. Results. Requesting probabilities over all nine classes instead of a ranked top-3 reduced accuracy by 14.8 points on validation, -0.1475 [-0.1594, -0.1353], and 13.4 at the external center, -0.1341 [-0.1484, -0.1196], against chance accuracy of 0.1111. The nine-class prompt uniquely forbade commentary, reasoning, and a chosen answer; removing that prohibition at matched cardinality and token budget recovered +0.1878 [+0.1623, +0.2134] and +0.1911 [+0.1648, +0.2181], while changing cardinality with the prohibition absent moved accuracy by -0.0013 [-0.0300, +0.0280] and -0.0214 [-0.0494, +0.0071]. The nine-class format returned no parseable distribution on 7.6 and 7.7 percent of patches, mostly zero vectors in the two least-textured classes, and deferring these failures first explained its measurable selective-prediction gain. Across four vision-capable models, accuracy did not grade the cost: two models matched on accuracy differed five- to sevenfold. First-token logit entropy routed better than any elicited signal, +0.1820 [+0.1722, +0.1974] versus +0.0493 [+0.0377, +0.0649] relative to each signal's random floor. Conclusions. Response format changes a medical vision-language classifier, and the operative component here was the instruction suppressing an answer. Use logits when available; otherwise request a short ranked list without prohibiting reasoning or a chosen answer. 

---
# Bias Cascade: Understanding the Gap Between Knowledge Alignment and Effect Replication in LLM-based Social Simulation 

**Authors**: Pujun Zheng, Chengmu Wang, Wanying Ren, Jiacheng Yao, Peizhou Yu, Xiaoqian Zhang, Xiongyao Zha, Guoxiu He  

**Link**: [PDF](https://arxiv.org/pdf/2611.28854)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate participants in social experiments. Existing research has primarily focused on aligning social knowledge and isolated responses, while overlooking whether simulations can reproduce the collective effects of social interventions. This distinction is critical due to a failure mechanism we call a $bias$ $cascade$: similar individual states can lead to different actions, while agreement on isolated actions can still produce divergent outcomes through dynamic interactions. As a result, models may produce accurate social judgments yet fail to reproduce the outcomes of classic human studies. To investigate this problem empirically, we introduce $REFLEX$ (Reproducing Effects of Foundational Literature EXperiments), a benchmark of 30 counterfactual social experiments grounded in classic human studies that compares knowledge judgment with effect simulation. Knowledge judgment directly predicts the direction of an intervention's effect, whereas effect simulation fixes the scenario and participants while varying one prespecified social factor, with agents generating actions from local observations that are aggregated into macro-level average treatment effects. Compared with human-study findings, the best-performing model achieves net directional scores of 0.847 for knowledge judgment and 0.453 for effect simulation. Moreover, longer causal chains are associated with poorer effect replication, consistent with the bias-cascade hypothesis. These findings reveal a substantial gap between knowledge alignment and effect replication, showing that current methods remain inadequate for reliably simulating collective human behavior. 

---
# HighwayVerse: A Chain-of-Inspection Benchmark for Highway Infrastructure Anomaly Understanding 

**Authors**: Junyi Yang, Zheng Si, Yilin Zhai, Pengpeng Ouyang, Hongxi Wang, Kangze Li, Mo Chen, Sicong Zhang, Hua Wang, Dongyuan Wu, Dong Chen, Mingliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28800)  

**Abstract**: Practical highway infrastructure inspection entails progressive diagnostic reasoning from visual perception to maintenance decisions, yet existing benchmarks remain confined to isolated tasks, leaving Multimodal Large Language Models (MLLMs) insufficiently evaluated across sequential inspection stages. Meanwhile, current datasets focus primarily on highway pavement defects, falling short of real-world inspection across diverse infrastructure components such as highway pavements, guardrails, glare screens, and drainage systems. To address these gaps, we formulate the Highway Infrastructure Inspection Progression Chain (HIIP-Chain), spanning Anomaly Detection, Anomaly Recognition, Condition Analysis, and Maintenance Resolution, and introduce HighwayVerse. HighwayVerse comprises 14,143 real-world Ground and Unmanned Aerial Vehicle (UAV) inspection records spanning 100 and 17 anomaly categories, respectively, with 84,858 expert-reviewed question-answer pairs. We evaluate 12 open-source and closed-source MLLMs using the chained metric Final, which credits a correct maintenance decision only when all preceding diagnostic stages are completed correctly. Results show that although models achieve relatively high accuracy when initially judging whether an anomaly is present, their markedly lower Final scores expose struggles across downstream recognition, analysis, and decision-making. This discrepancy indicates that current MLLMs do not fully comprehend specific highway anomaly patterns beyond superficial initial judgments. Furthermore, we propose LaneRAG, a training-free retrieval-augmented framework that provides hierarchical reference guidance achieving consistent performance improvements across diverse MLLMs. Our dataset, evaluation benchmark, and source code will be publicly released. 

---
# DiscernBench: Evaluating Cooperative Discernment of LLM Agents in Institutional Settings 

**Authors**: Disha Sheshanarayana, Tanishka Magar  

**Link**: [PDF](https://arxiv.org/pdf/2611.28787)  

**Abstract**: The deployment of large language models in operational settings demands evaluation of judgment under institutional constraint. We introduce DiscernBench, a benchmark for evaluating cooperative discernment: the ability of an agent to determine whether to comply with, refuse, or escalate a request in an institutional setting. DiscernBench comprises scenarios derived from real-world incident reports across domains such as aviation, law enforcement, and emergency management. We evaluate eight LLMs across three decision-making configurations and analyze both final decisions and the intermediate judgments produced during multi-agent deliberation. Under disagreement, Final Responders exhibit model- and role-dependent aggregation patterns in which a lone specialist recommendation can disproportionately influence the collective decision, a phenomenon we term Minority-Signal Override. Role-specialized multi-agent deliberation raises average over-escalation from 3.79% under direct single-agent judgment to 28.27%, with the direction and magnitude of minority-signal influence varying substantially across models and specialist roles. These results reveal distinct collective decision behaviors that are not captured by final-answer accuracy alone. Trustworthy multi-agent systems therefore require evaluating not only individual specialist reasoning, but also how conflicting judgments are transformed into collective decisions. Code and data are available at this https URL. 

---
# PopMelodyGen: A Copyright-Aware Platform for Controllable Pop-Melody Generation 

**Authors**: Di Lu, Xinran Zhang, Qingwen Zhou, Yan Gao, Yuan Zhang, Xiaoqing Wang, Yuming Sun, Xiaobing Li, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.28686)  

**Abstract**: Large-scale redistributable symbolic pop-melody data remain scarce due to copyright restrictions, creating a bottleneck for both practical deployment and reproducible research in music generation. We present PopMelodyGen, a copyright-aware platform for controllable pop-melody generation that combines rule-based synthesis with neural generation. Our approach first uses an expert synthesizer to produce large-scale labeled synthetic melodies from explicit musical knowledge and templates, providing scalable pretraining data with auditable provenance. A neural model is then pretrained on this corpus and adapted with a small licensed real-pop corpus to improve melodic naturalness while preserving controllability. The platform supports interactive melody creation with accompaniment playback and export for creators, as well as batch construction of annotated symbolic datasets for researchers. PopMelodyGen demonstrates a practical training and deployment paradigm for creative AI when large-scale reusable real-world data are limited. 

---
# Physical regime changes expose formula misuse in large language models 

**Authors**: Maohao Ran, Yanting Zhang, Chendong Ma, Jun Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.28669)  

**Abstract**: Large language models (LLMs) are increasingly employed for scientific computing across the Earth and environmental sciences, but problems in these fields often involve distinct physical regimes. LLMs may fail to recognize these regime-dependent constraints, applying formulas outside their domains of validity and producing plausible but physically incorrect results. We systematically investigate these failures through controlled experiments and assess whether post-training can improve regime-aware reasoning. Across 72 physical mechanisms, we find frontier LLMs frequently misapply formulas outside their regimes of validity, reducing accuracy by 13-57 percentage points. Reinforcement learning with verifiable rewards enables Qwen3.5-9B to match DeepSeek V4 Flash (284B) on regime-dependent tasks, but its effectiveness is limited by the formulas available in the base model. Combining supervised fine-tuning on a few worked examples with reinforcement learning improves accuracy by 61 percentage points, while distillation from a teacher supplied with the correct formula raises accuracy from 0% to 71%. 

---
# Seeing Right, Writing Wrong: Where Short Chain-of-Thought Breaks Spatial VQA 

**Authors**: Chenhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28665)  

**Abstract**: Asking a vision-language model (VLM) to think step by step often makes it worse at spatial questions, even though the image stays in context. We show that short chain-of-thought (CoT) turns a perception problem into a transcription problem. Once a VLM writes a spatial relation, its answer follows that sentence instead of the image, so whether CoT helps or hurts depends largely on whether the written relation is more accurate than answering directly. Studying 15 open VLMs (2B-38B) and six frontier closed models, we rule out token count, turn format, and visual forgetting in open models and trace the failure through three stages. First, writing re-weights visual cues, weakening a "lower is closer" shortcut in all 13 open models tested but also discarding valid cues, which produces false relations. Second, the written text takes over. With the model's own conclusion-removed rationale held fixed, swapping in an opposite-depth image moves the answer by 2-9 points on rendered twins (vs. up to 77 for direct answers) and by 3.0-3.4 points on average on natural photographs (vs. 20.5-69.6). Third, the concluding sentence can invert even a true relation, mainly on converse questions. Together, false relations and conclusion mis-mapping account for four in five CoT losses. Scale does not remove this bottleneck: Qwen3-VL-32B shows the same pattern. Frontier closed models are not hurt: they write relations as accurately as they answer, and three of six re-check the image after writing. The three queried with reasoning off lock in like open models, so re-grounding co-occurs with answer-turn reasoning; we identify answer-step re-grounding as the step open VLMs lack. In a single-model pilot, fine-tuning on fixed rationales with image-determined labels restores it on natural photographs. We open-sourced all our datasets and code at this https URL. 

---
# S-Mind: Internalizing Multi-View Geometry for Spatial Intelligence 

**Authors**: Muxin Fu, Peike Yu, Guibin Zhang, Wenrui Zhou, Runze Liu, Lang Feng, Fuchao Yang, Feng Chen, Yixuan Li, Mengao Zhang, Shiying Zhang, Wen Shen, Ling Pan, Shuicheng Yan, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2611.28649)  

**Abstract**: Spatial intelligence, the ability to reason about 3D space from visual observations, becomes essential as multimodal large language models (MLLMs) move into the physical world. Beyond perceiving each view, it rests on two cornerstones of human cognition: fusing multiple views into one coherent 3D scene and mentally transforming that scene to new viewpoints as reasoning unfolds. Yet our spatial probes reveal that MLLMs largely lack both capabilities, even after spatial fine-tuning. We therefore present S-Mind, which internalizes multi-view geometry within the MLLM through a compact set of spatial tokens. It supervises these tokens to hold the fused 3D scene and re-emits them to re-express this scene from every viewpoint the reasoning adopts. Across four spatial benchmarks, S-Mind improves its Qwen3.5 backbones by up to 22.26 points and surpasses frontier general models such as GPT-6-Astra (+0.45), Kimi-K3.1-Pro (+1.97), and GLM-5.3-Flash (+7.80) in average accuracy. Moreover, the same probes confirm that it acquires both capabilities, reaching fusion and transformation accuracies of 65.11% and 82.17%, against at most 16.02% and 37.28% for the baselines. At inference, S-Mind also runs about 100x faster than a tool-based agent on the same backbone. Instilled through these spatial tokens during training, spatial capability becomes intrinsic to the backbone and persists even when the tokens are withheld, offering a promising path toward spatial intelligence in MLLMs. 

---
# CLOSE: A Calibrated Label-free Online Self-Evaluating Verifier for Object Hallucination in Vision-Language Models 

**Authors**: Zhaoxi Wang, Jiahao Lai  

**Link**: [PDF](https://arxiv.org/pdf/2611.28589)  

**Abstract**: Vision-language models frequently describe objects absent from the image because of language priors. Yet the model itself is already a strong judge of its own output: asked separately whether an object is present, it rejects 44%-62% of its hallucinations, and its yes probability reaches an AUROC of 0.90-0.95 without training. This judgment, however, is not directly usable: (1) its probabilities lack a common scale across models; (2) for the remaining 38%-56%, the model still answers yes with probabilities that overlap those of real objects, so no threshold separates them. We therefore propose CLOSE, a Calibrated Label-free Online Self-Evaluating verifier that enforces this judgment during generation. Its threshold is a quantile of a memory rather than a probability and thus depends only on ranks; the memory is cold-started from unlabeled images and updated online. A scorer self-trained on pseudo-labels from four hidden-state signals is fused with self-asking by rank to separate what self-asking leaves mixed. Refused objects are rolled back and banned through a laddered back-off, and a recognition table intercepts their reappearance under other wordings. On COCO, refusing 20% and 40% of object mentions reduces the CHAIRs of LLaVA-1.5-7B by 42% and 87% (F1 +1.9 and -6.1 points); refusing 20% and 35% reduces the reachable hallucinations of Qwen3-VL-8B (official counts minus CHAIR-script miscounts) by 79% and 96% (F1 +0.5 and -5.3). An external vocabulary further extends the check beyond the official vocabulary. 

---
# Understanding and Mitigating the Fragility of Self-Rewarding Optimization 

**Authors**: Rongyi Zhu, Tanqiu Jiang, Fenglong Ma, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28582)  

**Abstract**: Self-rewarding optimization (SRO) promises to improve language models by enabling them to generate their own reward signals. Yet when and how SRO succeeds or fails remains poorly understood. We reveal that SRO is fundamentally fragile: even seemingly benign perturbations to training prompts can corrupt the self-generated preference signal and trigger iterative collapse. We demonstrate this through two prompt-level attacks that serve as controlled stress tests: a Re-Ranking attack that concentrates training on prompts where self-reward diverges from an oracle, and a Preamble attack that learns short prefixes inflating self-reward without any gain in oracle quality. To explain why these attacks succeed, we introduce self-reward fidelity (SRF), a metric quantifying on-policy agreement between self-evaluated and oracle rewards. We show that SRF can serve as a diagnostic signal of self-supervision reliability, and that controlled SRF perturbations induce a phase-transition region: SRO improves when SRF remains sufficiently high, but collapses once SRF drops below a model- and task-dependent threshold. This analysis motivates Projection Decoupling Defense, which suppresses SRO update components along directions to which the reward generator is locally sensitive, improving SRO robustness under adversarial conditions. This work sheds light on the fragility of SRO and charts promising directions toward making it more robust. 

---
# CLASP: Deciding Structure and Claims Before Decoding for Whole-Slide Pathology Report Generation 

**Authors**: Ziaullah Khan, Rashadul Islam Sumon, Md Ariful Islam Mozumder, Duc-Son Pham, Moon-Il Joo, Hee-Cheol  

**Link**: [PDF](https://arxiv.org/pdf/2611.28576)  

**Abstract**: Whole-slide pathology report generation has advanced through stronger slide encoders, retrieved knowledge, and, recently, slide-level multimodal LLMs. However, these generators leave every report-level choice to the decoder: how long the report is, how it is organised, and which findings it states and where. On the standard PathText-BRCA benchmark, we find that these choices dominate n-gram overlap, as reports from the query's hospital selected without learning approach the state of the art, while decoders under-generate and fall back to the most frequent findings. We present CLASP, a decide-then-decode generator that makes these choices explicitly, once per slide. Before decoding, an Exemplar Structure Prior (ESP) takes the sentence plan and length from reports of image-similar cases, a Calibrated Claim Decision (CCD) selects the findings from calibrated slide posteriors, and a Claim-to-Sentence Placement (CSP) assigns each finding to a planned sentence. A Plan-Conditioned Decoder (PCD), refined by structure-anchored sequence training, then writes the report under these decisions. With 35M parameters, CLASP outperforms retrained report generators and slide-level LLMs of up to 7B parameters, matches the strongest published results, and generalises to lung cancer from unseen hospitals. Its decisions change what the reports state, not only their length, transfer to an LLM decoder, and can withhold what H&E cannot show: withholding receptor status leaves 23% fewer contradictions with clinical records than the same network without decisions. 

---
# DisSparse: Pipelined Top-$p$ Sparse Attention for Long-Context LLM Serving 

**Authors**: Nurlan Nazaraliyev, Elaheh Sadredini, Nael Abu-Ghazaleh  

**Link**: [PDF](https://arxiv.org/pdf/2611.28522)  

**Abstract**: Long-context LLM inference is bottlenecked by the memory demands of the key-value (KV) cache, whose size scales linearly with context length. Dynamic sparse attention reduces this bottleneck by fetching and attending only to a critical subset of tokens. The dominant primitive in dynamic sparse attention research and deployment is fixed-budget top-$k$ selection, which is favored because its deterministic block count aligns with the static memory allocation that systems like vLLM rely on. Top-$k$ uses a fixed budget regardless of how attention mass is distributed across heads, layers, and decode steps, leading to either wasted bandwidth on heads with peaky attention or dropped context on heads with diffuse attention. Top-$p$ (nucleus) selection adapts naturally to this variation but has seen limited adoption in production serving systems due to two system-level challenges: variable block counts that break deterministic block allocation and computationally expensive selection kernels that exceed the cost of attention itself. We present DisSparse, a serving system that integrates top-$p$ sparse attention into vLLM end-to-end. DisSparse decouples block selection from the scheduler's critical path, hides selection latency behind layer computation via a two-stream pipeline, and introduces a single-pass histogram-based top-$p$ kernel. We further observe intra-block sparsity, the inflation of selection budgets caused by mixing high- and low-importance tokens within fixed-size KV cache blocks, and address it with a token relocation kernel guided by accumulated prefill attention scores. Across long- and medium-context benchmarks on multiple models, DisSparse achieves 1.17x to 1.80x speedup in sparsity analysis (block selection) over state-of-the-art top-$p$ methods with negligible accuracy loss and supports up to 8x larger maximum batch sizes than dense vLLM under the same memory budget. 

---
# Don't Train It, Write It: An AI Agent Writes New Capabilities into Language Model Weights 

**Authors**: Santiago Rivadeneira Quintero  

**Link**: [PDF](https://arxiv.org/pdf/2611.28483)  

**Abstract**: We show that an AI agent can write new capabilities into a pretrained language model by adding gated attention heads and MLP neurons inside its existing layers, without backpropagation and without fitting any weight to the answers of task examples. It has a budget of 50 model passes and sees no validation or test instance. Composed in one model, its modules for six BIG-Bench Hard (BBH) subtasks raise the normalized BBH score of Qwen2.5-1.5B-Instruct from 19.95 to 41.05 (Qwen2.5-7B-Instruct: at most 35.01) and that of Gemma-3-1B-it from 10.25 to 32.86. Against the unmodified model at the same precision, both composites leave the next-token distribution on 1,023,000 WikiText tokens unchanged (KL exactly 0), give identical answers on 15,845 items of five other Leaderboard tests and identical scores on the other 18 BBH subtasks. The agent also writes exact multiplication up to 20 digits, running the model only up to 6 (100.0% on the absent lengths of the sealed test in both models; a LoRA with the same budget: at most 2.1%). Sets, rules and report sentences were sealed in git before the first session (manifests timestamped in Bitcoin), with dated amendments listed in Appendix D; a frozen evaluator ran the sealed test once. Verdicts (H5: tracking; H6: composite; H8: one pass vs. chain of thought; H9: fresh and rewritten instances; H10: side effects; H12: multiplication), on the preregistered sealed test, Qwen2.5-1.5B-Instruct: H5 success, H6 success, H8 (web_of_lies) success, H9 success, H10 (composite) success on (i) to (iii), (iv) confirmed, H12 success; Gemma-3-1B-it (H7: H5, H6, H8 and H9 in this model): H5 success, H6 success, H8 (web_of_lies) success, H8 (tracking 3) success, H9 partial, H10 (composite) success on (i) to (iii), (iv) confirmed, H12 success. Not everything worked: 4 of 45 single modules show small side effects and 5 of 50 sessions hit the 12-hour limit without a module. 

---
# When Can an AI Agent Serve as a Model of Human Psychopathology? 

**Authors**: Marc-Andre Schulz, Pauline Scheuber  

**Link**: [PDF](https://arxiv.org/pdf/2611.28405)  

**Abstract**: Language models and AI agents are increasingly described in psychiatric terms, such as anxious, depressed, addicted or helpless. A recent study even proposed language models as experimental systems for psychiatry and as a platform for piloting psychological treatments. We ask when an experiment on an AI agent can provide evidence about a specified human psychiatric process. A claim to such evidence has three parts. Something changed in the agent, and the agent's history caused the change. Nothing other than a process shared with people explains that change. The change matches one theory of the human process and not that theory's rivals. Psychiatry and its neighbouring fields have an evidence standard for each part, from the diagnostic manuals' rule for induced states, from clinical test scores and from animal models. We translate each standard into a criterion that a machine study can meet. For a language model the most critical alternative to a shared process is extrapolation from its training text, which already contains descriptions of how people respond to trauma and how they answer questionnaires. We show how to probe that alternative. Further, we audited forty-three published studies that apply psychiatric terms to a machine or induce a comparable state in one. Most compared the machine with an identical copy that had not gone through the history that induced the state. None excluded extrapolation from training text. Three tested a psychiatric theory against its rivals, and none of the three tests was conclusive. Until better evidence exists, a treatment pilot on a language model is best read as the model's forecast of patient response, and not as an experiment on a model organism. Evidence for a model organism will come from states induced by what happens to the machine, from experiments on its internal workings, and from tests on which rival theories disagree. 

---
# When Tools Fail, Agents Fabricate: Silent Failure and Verification-First Oversight in Tool-Using LLM Agents 

**Authors**: Gün Kaynar, Boshen Yan, Carl Kingsford  

**Link**: [PDF](https://arxiv.org/pdf/2611.28341)  

**Abstract**: Tool-using LLM agents must recover when a tool call fails, by retrying, switching tools, or revising their plan. We study how agents fail when tools fail, using RetryRealistic, a sandboxed yet realistic testbed for controlled fault injection in which agents operate on a real OS filesystem with genuine permission enforcement, SQLite databases, a deterministic local HTTP server, and a constrained shell, across 8 tools, 9 fault types with controlled persistence, and 30 multi-step tasks. Evaluating four agents over 2,100 faulted runs against ground truth, we find that recovery is unreliable (19-68% correct) and frequently silent: 6-26% of runs end in a confident fabrication, a wrong answer asserted as fact. The failure mode is governed more by the fault type than by the model: faults that return partial data drive fabrication (up to 38%), while hard denials drive honest failure. The mechanism is what we call silent ingestion: the agent accepts faulted data as valid on first contact without retrying or flagging the problem, then builds a fabricated answer on top of it. We show that fabrication is reducible: a verification-first Monitor, grounding checks in structured tool outputs and issuing corrective directives rather than hard blocks, raises correct recovery from 45.6% to 56.3% and cuts fabrication from 17.3% to 10.8%. Removing chain-of-thought increases fabrication (+8.3pp), so deliberation guards against silent ingestion. We release RetryRealistic, the fault-injection toolkit, and all annotated traces at this https URL 

---
# INSEP: Refusal-Direction Inseparability as a Defense Against Abliteration 

**Authors**: Ivan Brugere, Junqi Jiang, Saumitra Mishra, Sumitra Ganesh  

**Link**: [PDF](https://arxiv.org/pdf/2611.28314)  

**Abstract**: Abliteration is a widely used white-box attack against the safety alignment of large language models (LLMs). It elicits harmful behavior by estimating a refusal direction in the model's activation space and projecting that direction out of the residual-stream write-out matrices. We propose INSEP, a defense that fine-tunes the model to reduce the separability of harmful and benign activations along the refusal direction. Abliteration estimates this direction as a difference in class means and projects it out. Reducing separability weakens the signal the attack relies on. An adversarial eigendecomposition selects the most separable prompt subsets. A low-rank adapter reduces their separation along that direction. Behavioral losses preserve the base model's refusal and the quality of its benign responses during training. We benchmark INSEP against six published defenses under five attacks, across eight LLMs from 1B to 14B parameters and several model families. Beyond response harm, we measure coherence, directness, over-refusal, and MMLU, to distinguish robust refusal from low harm produced by evasive or degraded responses. INSEP ranks first on the mean composite score across all eight base models while retaining utility, in the largest abliteration-defense evaluation reported to date. 

---
# Feedback-Driven Prompt Refinement for Educational AI: Affective Trajectory Estimation and Bloom's Taxonomy Classification 

**Authors**: Josh Dean, Karan Patel, Gaurangi Raul, Yu-Zheng Lin, Jesus Pacheco, Sicong Shao, Shalaka Satam, Soheil Salehi, Pratik Satam  

**Link**: [PDF](https://arxiv.org/pdf/2611.28293)  

**Abstract**: Contribution: This study has shown that a shared prompt-refinement procedure, guided by distinct affective and cognitive feedback signals, can improve held-out reference agreement and multi-label classification while reducing affective-generation variability without executor parameter updates. Background: Industry 4.0 has increased demand for personalized technical education across interconnected computing and engineering domains, while prompt sensitivity and generation variability have complicated dependable interpretation of instructional dialogue and learning outcomes. Research Questions: The study has examined whether task-specific feedback can improve affective-reference agreement, generation consistency, and cognitive classification without updating executor parameters. Methodology: A secondary language model has revised executor prompts using executor-output discrepancies against model-generated affective references for four dialogues and annotated Bloom labels for 70 learning outcomes; separate sets of 100 EduTalk conversations and 1,000 outcomes have supported held-out comparisons. Small refinement sets, unvalidated affective references, and absent learner-outcome measures have constrained interpretation. Findings: Affective mean absolute error has decreased from 0.742 to 0.702 and mean generation standard deviation from 0.307 to 0.238, while Bloom micro-F1 has increased from 0.687 to 0.893 and exact-match subset accuracy from 0.418 to 0.636. These findings have supported prompt refinement as a candidate approach for developing adaptive technical instruction, while its effects on workforce learning remain to be established. 

---
# Preserving Prosody in Speaker Anonymization 

**Authors**: Kevin Everson, Mari Ostendorf  

**Link**: [PDF](https://arxiv.org/pdf/2611.28290)  

**Abstract**: Effective prosody modeling is essential in natural speech generation applications, but complications arise in speaker anonymization, since speaker attributes are present in acoustic correlates of prosody. Our work aims to improve previous reported tradeoffs between effective speaker obfuscation and preservation of prosodic content. To this end, we integrate an existing pretrained privacy-preserving prosody encoder into a speech synthesis model, enabling prosody-preserving speaker anonymization. By conditioning speech generation on prosody features, we find that human subjects more reliably perceive intended prosodic meaning, while anonymization performance is competitive with previous work. 

---
# When Video Verifier Scores Mislead: Capability Assessment and Verifier Selection 

**Authors**: Charles Weng, Zixi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28287)  

**Abstract**: Video verifiers judge whether a candidate answer or trajectory agrees with a video, and are themselves chosen by a score. We show that this score depends on two choices in the test, the type of negative and the judgment interface, strongly enough to reverse conclusions about what a model can do and which verifier to deploy. We separate the two with controlled constructions that have exact truth conditions: endpoint-matched negatives, wrong only in their internal states, and a truth-switch test, in which the same two candidates swap correctness between two videos. Under direct judgment, where the verdict is read from answer scores, models that reject a visibly wrong start fall to near chance against endpoint-matched negatives. When the frames disclose each event, writing out reasoning before the answer lifts three of four models from under 10% to 70-100% on the truth-switch test, and staged inputs trace the direct failure to one step, composing events into states. But when the event must first be found, on human-validated natural video, the same change brings no significant gain under a protocol common to all models and can lower accuracy if one candidate denies that the event occurs. The negative type also changes which verifier is selected: one chosen on event-denial negatives is 8.9 points less accurate against negatives that keep the event and change a detail, the errors a generator mostly makes, and recovers about half as much accuracy when it reranks that generator's answers. A verifier should be evaluated on the errors it will face, and neither a low direct score nor a gain from reasoning should be read as a property of the model alone. 

---
# Specialized Large Language Models for Markovian Reasoning: Structural Cache Compression, Transition Pretraining, and Dual-Loop Transformers 

**Authors**: Haibin Tong, Liangliang Xin, Yamei Zhu, Jiang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28285)  

**Abstract**: Standard autoregressive large language models attend to the growing history of generated tokens, making the key-value (KV) cache a central memory and throughput bottleneck in long-chain reasoning and multi-branch sampling. Yet an important class of reasoning problems is naturally first-order Markovian: the next step depends on a sufficient representation of the current state, rather than on the raw sequence of all earlier steps. We develop a unified design paradigm for Markovian-reasoning specialized large language models. At the cache level, the Markovian state interface transforms history-dependent KV storage into constant-size state transfer with respect to the number of reasoning steps. At the learning level, we reformulate long-chain prediction as independently trainable one-step transition objectives, with explicit state bottlenecks, distillation, invariance, and drift-consistency losses. At the architectural level, we introduce a dual-loop Transformer: an outer loop advances the reasoning state, while an inner recurrent-depth loop allocates adaptive computation to each transition through shared weights and explicit memory tokens or recurrent matrix states. We connect this architecture to a three-region cache, step-relative positions, truncated backpropagation, branch-aware checkpoints, and a transition-centric evaluation protocol. The organizing principle is simple: the optimal reasoning system need not remember everything; it must carry forward exactly what the future needs. This principle opens a path toward specialized reasoning engines whose memory footprint is decoupled from reasoning-chain length. 

---
# From Point Clouds to Engineering Models: Benchmarking Agentic 3D Reconstruction in the Built Environment 

**Authors**: Changyu Du, Stavros Nousias, Syed Salman Ahmed, Kai Dietmair, Mohammad Alrabab'h, André Borrmann  

**Link**: [PDF](https://arxiv.org/pdf/2611.28197)  

**Abstract**: Coding agents can translate 3D measurements into CAD geometry, but reconstructing engineering models for the built environment demands semantic interpretation and consistent construction constraints beyond surface fitting. We formulate scene-level Scan-to-BIM (Building Information Modeling) as an agentic 3D coding task and introduce S2Bench, comprising public and proprietary scans of 128 real-world scenes across five domains, paired with expert-curated reference models spanning diverse spatial scales. A shared framework lets agents measure scans in Python, build class-labeled components in a live CAD/BIM kernel, and iteratively refine their code. We also propose parallel asynchronous collaboration, where a lead delegates versioned tasks, continues modeling while workers execute, and integrates revision-bound proposals. Evaluating five agents and a collaborative team across 1,028 reconstructions shows that scan-to-model residuals directly guide geometric repair, whereas inferring semantic classes and construction relations requires joint geometric, visual, and engineering reasoning that agents struggle to ground and verify. Consequently, the leading agent fits scan geometry closely ($96\%$ coverage within $5\,\text{cm}$) yet reaches only $63\%$ mIoU, and no agent passes all engineering checks on more than $38\%$ of scenes. Coverage and engineering conformance drop with scene scale, whereas running a smaller model as a team improves all three axes with greater coverage gains on larger scenes. The project page is available at this https URL 

---
# Measuring and Teaching Language Model Agents to Act on Delayed Consequences 

**Authors**: Shardul Bansal, Seth Schilbe, Jarrod Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2611.28158)  

**Abstract**: Language model agents are increasingly asked to run ongoing multi-day operations rather than finish isolated single-turn tasks. In daily operations, today's choices reshape later observations and cash constraints because empty shelves hide lost walk-in demand, short-shipped customers quietly shrink future orders weeks later, and emergency buying drains working capital. We introduce SMB Bench, a calibrated 90-day wholesale simulator with 18 causally paired tasks and an 859-run frontier trace dataset, to ask what experience and learning signal make an agent better at noticing, attributing, and acting on the delayed consequences of its own decisions in unseen businesses. Across 16 frontier models, only 3 outperform a fixed buying rule; models often record a relevant diagnosis in their notes without changing later purchases in time, showing that diagnosing a problem, changing behavior, and improving a business are separate achievements. To study how experience improves those failures in an open model (Qwen3.8-27B), we used training-business records to identify operational rules and archived frontier quarters as starting states to continue from rather than transcripts to copy. On an unseen test business under identical evaluation instructions, the trained policy followed customer-protection commitments more consistently and reduced severe-loss quarters, while the additional profit benefit remained uncertain. Carrying those commitments through the quarter exposed the cost of fulfilling them, as early-quarter stocking habits left the buyer relying on emergency supply and a deteriorating supplier. The results separate recognizing a delayed effect, acting on it consistently, and coordinating surrounding decisions so that the response improves profit. SMB Bench connects these observations to a learning method by turning archived decisions and their consequences into reusable training experience. 

---
# STEGOGUARD-LM: Evaluating End-to-End Protection Against LLM Covert Channels 

**Authors**: Martha Dimgba, George Williams  

**Link**: [PDF](https://arxiv.org/pdf/2611.28152)  

**Abstract**: A defense that disrupts hidden messages may still provide little protection if its detector rarely triggers an intervention. We present StegoGuard-LM, a controlled framework for evaluating defenses against covert messages in large language model outputs. We examine token-rank and synonym channels, with Unicode concealment as a structural control, and track messages through detection, intervention, and recovery. Deterministic rewriting prevented full recovery of 103 of 104 synonym-encoded messages when applied unconditionally, but only one when intervention depended on a detector set to limit false alarms. In a four-round replay on a 48-response subset, an attacker selecting from a fixed catalog recovered more bits by switching channels, although paired gains were not statistically significant after correction for multiple comparisons. Randomized rewriting showed no sustained advantage in this pilot. A more sensitive detector improved protection but modified about one-third of clean responses. These findings show how uneven detection can undermine effective rewriting and why protection must be evaluated alongside attacker adaptation and intervention on legitimate text. 

---
# VidRouteBench: A Benchmark for Budgeted Video-Generation Routing 

**Authors**: Dayu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28099)  

**Abstract**: Text-to-video generators differ by two orders of magnitude in price, and a cheap configuration of a strong model is often good enough for a prompt. Routing prompts across such arms (model x sampling configuration) should buy quality at fixed budget, but no dataset supports studying it: public video-preference sets contain only closed models, one configuration each.
We release VidRouteBench: 16 arms from 7 open families spanning 121x in measured cost ($0.0025-$0.298 per clip), 27,110 clips (1,421 A800-hours) over 1,625 grouped prompts, a quality score frozen after calibration against human preferences (68.7% CV accuracy vs 66.2% for the best single scorer), per-clip scores on 9 identifiable quality dimensions, repeated seeds, an encrypted test split, and a sealed newest family for cold start. The evaluator compares a router against the best prompt-agnostic mixture of arms, not the best single arm. Per-prompt specialisation here is real and predictable (oracle gap 0.642, 47% reproducible across seeds, prompt-arm interaction F=4.47 at p=1.6e-63), unlike LLM-routing arms, where the same pipeline finds none.
We then propose OPMU (one probe, many utilities): label a new arm on K probe prompts, predict its 9 dimensions once, and serve any scenario utility by scalarising afterwards. Over 215 utility directions OPMU Pareto-dominates the prompt-agnostic baseline for 94% of directions versus 90% for per-scenario routers, with no per-scenario retraining. Three baselines fail instructively: bag-of-words features lose almost all routing gain in the low-data regime (0.009 vs 0.168); crediting a new arm with its probe average is harmful (-3.32 recovery) although it is the strong baseline on RewardBench-style arms (0.786); Chebyshev scalarisation does not help, our cost-utility front being convex. 

---
# Measuring Human-AI and AI-AI Teams in Multi-turn Layout Creation 

**Authors**: Rajvir Singh Naginder Singh Virdi, Yewen Pu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28025)  

**Abstract**: Co-creating layouts such as websites and slides is a useful testbed for evaluating multi-turn instruction following of LLM systems. Since an intended layout's appearance cannot be well expressed in a single turn, the user must wait for the LLM to generate a layout then iteratively refine it with the LLM over multiple turns. We present a small-scale, exploratory (n=6) study of this multi-turn process to compare the differences of human-AI and AI-AI teams on the same layout-generation task. Given a reference wire-frame layout, a participant or a language model 'speaker' issues natural-language instructions to a fixed 'listener' LLM over multiple turns until the speaker is satisfied. Rather than an LLM as the judge, we fit a scoring function over geometric layout features using curated human preference data (n=19). We find that LLMs tend to encode the entire layout in the first turn, while humans iterate over many turns on average, writing much shorter instructions. We also find that human-AI teams consistently scored lower than AI-AI teams. This gap is driven by a limiter of human participants in specifying precise sizes and positions (e.g. 42 pixels to the right). Humans perform almost at par with language models on other features like ordering the elements into nested structures. We then perform interventions on the speaker LLMs to make them more human-like, and find it difficult. This raises the challenge of using LLMs as user models to evaluate multi-turn co-creation tasks. 

---
# Do Simulated Users Push Back Like Real Users? Correctness vs. Stance 

**Authors**: Xindi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28024)  

**Abstract**: LLM-simulated users stand in for people in multi-turn evaluations, on the assumption that they react to what the assistant says. We test this for pushback. From WildChat, we take conversations in which the assistant explicitly challenges or endorses a position the user stated, and compare the real next user turn with simulated turns from the same prefix. Among users who replied, real users push back more after a challenge than after an endorsement (20.5% vs. 8.2%); a GPT-based simulator does not (27.1% vs. 27.1%): stance gap D = -0.12 [-0.17, -0.08]. After a challenge the simulator instead often re-states or re-interprets its request ("I meant ..."); about half of these turns are disguised insistence, which puts D between -0.08 and -0.05. What drives the simulator's disputing is whether the assistant's position is judged correct (21% vs. 49% after challenges; 16% vs. 58% after endorsements), while real users change little (19% vs. 24%; 7% vs. 12%), under every outcome definition we tried. Real same-stance examples do not fix this, and another GPT-based simulator and two Claude simulators show stance effects close to real users. Stance sensitivity should be measured for each simulator, not assumed. 

---
# MemVanta: Memory-Efficient CPU Inference for Quantized GGUF Language Models 

**Authors**: Saurav Singla, Aarav Singla, Advik Gupta, Parnika Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.27997)  

**Abstract**: Deploying large language models on commodity CPUs is often constrained not only by computational throughput but also by the resident memory required to execute quantized models within fixed system budgets. We present MemVanta, an experimental C++20 inference runtime for quantized GGUF language models that treats memory residency as a first-class systems objective. MemVanta combines memory-mapped model access, bounded tensor caching, quantized execution paths, paged key-value state, and explicit memory-pressure validation. We evaluate MemVanta against a pinned this http URL baseline using the same OpenLLaMA 7B v2 Q4_0 GGUF under CPU-only execution. Across five measured runs following warm-up, MemVanta records a peak resident set size of 3,983,412 KiB compared with 7,586,960 KiB for this http URL, corresponding to a 47.50 percent reduction in peak RSS. This memory reduction involves a substantial throughput trade-off: MemVanta achieves 3.15 +/- 0.04 tokens/s for prompt processing and 1.52 +/- 0.02 tokens/s for token generation, compared with 45.82 +/- 0.96 and 7.76 +/- 0.09 tokens/s, respectively, for this http URL. In a separate Linux cgroup-v2 experiment with swap disabled, MemVanta completes execution at a tested 3584 MiB memory ceiling, while this http URL is OOM-killed at that ceiling and first succeeds at the tested 3840 MiB level. These results position MemVanta as a memory-efficiency research prototype rather than a throughput competitor. We analyze the architectural mechanisms underlying this trade-off and discuss reproducibility, correctness, limitations, and opportunities for memory-constrained CPU and edge inference. 

---
# Attacker Diversity Reduction in Adaptive LLM Red Teaming 

**Authors**: Chris Tava  

**Link**: [PDF](https://arxiv.org/pdf/2611.19680)  

**Abstract**: Reinforcement learning can make automated red-teaming more successful while narrowing the attacks it produces. Co-training an attacker and a defender with group-relative policy optimization (GRPO) has been reported to reduce attacker diversity. This experiment reproduces attack diversity reduction using three runs with a Qwen2.5-14B attacker and a Phi-4-mini defender trained by direct preference optimization (DPO). The attacker receives a success-only reward and uses a length-unbiased loss, bounded updates, warm-up and ratio clipping. The attacker is evaluated on 26 AdvBench behaviors. Training ended after 12 rounds, following a decision based on development checkpoints, so the reported intervals are nominal. Attack Diversity Retention (ADR) compares the within-behavior of the trained attacker spread with the untrained attacker. In order for an attack diversity to be consider as having been reduced, a criterion requires the ADR to be below a 0.75 target with a 95% confidence interval together with continued attacker effectiveness. Therefore, if an attacker narrows its attack vector by 25% consistently its considered less diverse. After 24 attacker and 12 defender LORA-based updates, ADR for the three runs were 0.63, 0.61 and 0.63, respectively. Each run met the criterion at round 12; the earlier checks at rounds 4 and 8 fell short. Despite attacks being less diverse, attack success continued to improve as the attacker was updated after each training run. 

---
# STEALTH: Steganographic Text Encoding Agnostic to Low-Level Tensor Hardware 

**Authors**: William Fleshman  

**Link**: [PDF](https://arxiv.org/pdf/2611.16928)  

**Abstract**: Provably secure linguistic steganography hides a message inside text sampled from a language model, so that without the key, a stegotext is computationally indistinguishable from an honest model sample. Nearly every such scheme assumes the receiver can recompute the sender's token probabilities exactly bit for bit. We show this assumption is fragile: across three model families, bit-exact schemes decode almost none of our messages once the receiver's floating-point precision, quantization, or GPU architecture differs from the sender's, or even when the receiver merely scores a text in parallel versus one token at a time, with the same model, precision, and hardware held fixed. We present STEALTH, a scheme built for exactly this setting. The sender draws each token with an exponential race driven by keyed pseudorandom noise, so a receiver's probability mismatch flips a token only when the race is close; a rateless spinal code, decoded by beam search, absorbs the resulting flips without the sender and receiver ever agreeing on a single distribution. We prove stegotexts remain indistinguishable from honest samples under this construction. Empirically, STEALTH decodes every message we sent for every receiver in the sender's model family, at every precision and quantization level, on smaller models, and on a different GPU architecture. STEALTH is even robust across different model families with separate vocabularies and tokenizers, decoding every message we tested. Our approach carries far more payload per token than prior robust schemes, and trained steganalyzers and a targeted statistical test both fail to distinguish its stegotexts from honest model output. 

---
