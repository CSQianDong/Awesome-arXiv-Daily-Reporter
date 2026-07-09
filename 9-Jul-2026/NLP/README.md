# Accurate, Interdisciplinary and Transparent Structure-property Understanding with Deep Native Structural Reasoning 

**Authors**: Chen Tang, Yizhou Wang, Jianyu Wu, Lintao Wang, Shixiang Tang, Pengze Li, Encheng Su, Jun Yao, Jiabei Xiao, Yuqi Shi, Jielan Li, Hongxia Hao, Zhangyang Gao, Fang Wu, Ben Fei, Xiangyu Yue, Pan Tan, Bozitao Zhong, Jinouwen Zhang, Aoran Wang, Yan Lu, Jiaheng Liu, Xinzhu Ma, Liang Hong, Mingyue Zheng, Phil Torr, Bowen Zhou, Wanli Ouyang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2607.07708)  

**Abstract**: Structure-property relationships are foundational to biology, chemistry and materials science, where function, reactivity and physical response emerge from spatial, chemical and periodic organization. Mechanistically explaining these relationships requires interpreting structural evidence through scientific principles and physical constraints, from stereochemistry and bonding to symmetry, energetics and periodic order. However, applying artificial intelligence to this process presents a joint challenge of representation and reasoning: models must preserve domain-native structural information while showing how specific evidence supports predictions under these constraints. Here we introduce SciReasoner, a multimodal scientific foundation model for native structural reasoning across proteins, small molecules and inorganic crystals. SciReasoner discretizes coordinates, topologies and periodic connectivities into a unified structure-aware vocabulary, treating structural tokens as addressable evidence units during reasoning. In homology-controlled Gene Ontology prediction, SciReasoner improves Cellular Component annotation for low-homology and orphan-like proteins, increasing $F_{\max}$ from 0.42 to 0.55. In chemistry, it raises single-step retrosynthesis accuracy from 0.63 to 0.72 while generating fragment-level disconnection and precursor-verification traces. In materials science, its representations separate elemental and compound phases and resolve high- and low-band-gap regimes. Across 86 benchmarks, SciReasoner achieves state-of-the-art performance on 67 tasks. Double-blind expert evaluation rates its reasoning traces as preferred or at least comparable to those of a frontier large language model in 98% of cases. By making structure an inspectable substrate for reasoning under scientific constraints, SciReasoner connects accurate prediction with interpretable scientific inference. 

---
# Co-LMLM: Continuous-Query Limited Memory Language Models 

**Authors**: Yair Feldman, Linxi Zhao, Nathan Godey, Dongyoung Go, Yilun Hua, Kilian Q. Weinberger, Jennifer J. Sun, Yoav Artzi  

**Link**: [PDF](https://arxiv.org/pdf/2607.07707)  

**Abstract**: Limited memory language models (LMLMs) externalize factual knowledge during pretraining to a knowledge base (KB), rather than memorizing it in their weights. During generation, the model then fetches knowledge from the KB as needed. This recently introduced paradigm provides multiple advantages, including knowledge control capabilities that remain beyond conventional LLMs. We propose continuous-query LMLM (CO-LMLM), where the KB pairs continuous keys with textual knowledge values, a significant departure from prior reliance on relational KB and queries. CO-LMLM generates flexible vector queries at minimal cost, while still integrating human-readable and attributable retrieved knowledge into its generation. We pair this design with an annotation pipeline that tags free-form factual spans in arbitrary text, removing prior work's restriction to Wikipedia. Across pretraining on Wikipedia and FineWeb-Edu and at multiple model scales, CO-LMLM outperforms prior LMLMs and vanilla LLMs in both perplexity and factual precision. At 360M scale, this includes lower perplexity than models pretrained on 40x more data, and SimpleQA-verified performance that is in line with gpt-4o-mini and higher than Claude Sonnet 4.5. 

---
# From Noisy Traces to Root Causes: Structural Trajectory Analysis and Causal Extraction for Agent Optimization 

**Authors**: Ying Chang, Jiahang Xu, Xuan Feng, Chenyuan Yang, Peng Cheng, Yuqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2607.07702)  

**Abstract**: The optimization of long-horizon agents increasingly relies on reflection-based mechanisms, where a large language model (LLM) acts as an optimizer to diagnose agent failures and improve agent policies. However, real execution traces are difficult to use directly for optimization: large trace collections are often redundant and heterogeneous, making optimization inefficient and prone to overfitting to low-value failures; meanwhile, each individual trajectory also contains many irrelevant steps, while naive context reduction methods such as truncation or sliding windows can discard causally important evidence and produce misleading optimization signals. To resolve this dilemma, we introduce STRACE (Structural TRajectory Analysis and Causal Extraction), a framework that constructs high signal-noise optimization contexts for more precise and effective optimization. At the batch level, STRACE mines failure patterns to filter redundant traces and retain representative failures; within each selected trace, it performs causal localization over a textual dependency graph to remove non-causal steps and identify the true root-cause module for optimization. Empirical results demonstrate that STRACE significantly outperforms standard context-filtering baselines. Notably, on a challenging formal verification task (VeruSAGE-Bench), it successfully optimizes human-expert designed agents, delivering $1.4\times$ success-rate improvement (42.5% to 58.5%). The code is available at this https URL . 

---
# Does Bielik Know What It Doesn't Know? Activation Dispersion Separates Entity Familiarity from Factual Reliability Across Model Scale 

**Authors**: Grzegorz Brzezinka  

**Link**: [PDF](https://arxiv.org/pdf/2607.07670)  

**Abstract**: Large language models hallucinate most about entities they have never seen. We ask whether a model's activations betray entity familiarity before a single answer token is generated, and whether that signal predicts the factual reliability of the answers. On four Polish Bielik models (1.5B-11B parameters), we probe four entity domains (athletes, cities, writers, musicians), each with 42 well-known, 42 obscure-but-real, and 42 fabricated entities addressed by a one-sentence question (504 prompts per model). Two unsupervised, single-forward-pass dispersion measures over post-SwiGLU MLP activations, inverse participation ratio and spectral entropy, separate known from fabricated entities at AUROC 0.95-1.00 across all domains and scales; a supervised linear probe reaches 0.99-1.00. Both clear selection-aware permutation floors of about 0.70-0.74 (empirical p<=1e-3), survive held-out layer selection (0.93-0.99), and persist on real names (known vs. obscure-but-real: 0.96-1.00). The signal transfers across entity types (mean off-diagonal AUROC 0.92-0.99); a matched-template counterfactual shows the only large drops are template-caused, not entity-type effects, and the signal is diffuse across heads. This representational signal is already at ceiling at 1.5B, whereas behavioral factual reliability scales sharply: 0, 2, 10, and 19 of 42 known athletes are answered fully correctly by the 1.5B, 4.5B, 7B, and 11B models under a strict judge. Within known entities, separating correct from hallucinated answers is much harder (probe 0.93; dispersion no better than a first-token-entropy baseline). A five-sample semantic-entropy baseline reaches only 0.71-0.83 at 5x the inference cost. Despite this internal awareness, the models almost never abstain: an audit of 2,520 answers finds 2 refusals and 1 hedge. Entity familiarity and factual reliability are distinct phenomena on different scaling curves. 

---
# DiaLLM: An Investigation into the Robustness-Generation Gap in English Dialect Adaptation 

**Authors**: Jordan Painter, Dipankar Srirag, Adarsh Kappiyath, Diptesh Kanojia, Aditya Joshi, Lu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2607.07669)  

**Abstract**: Large language models increasingly \emph{understand} dialectal English, yet still \emph{produce} only standard, US-leaning English, leaving dialectal generation, the harder half of the problem, largely unaddressed. We introduce \textbf{DiaLLM}, which continually pretrains three open-weight language model families on the International Corpus of English and applies implicit and explicit post-training paradigms, each combined with three model alignment strategies, giving the first controlled comparison of these components across Australian, Indian, and Northern British English. Our results reveal that dialectal robustness and generation are \emph{dissociated}: benchmarks are shaped by continual pretraining and SFT, while alignment visibly reshapes generation in ways benchmarks do not capture. Explicit variety-targeted adaptation produces output reliably recognised as dialectal and preferred over broad alignment, yet the method that most aggressively optimises the dialectal reward is not preferred by human evaluators. Independent linguistic analysis corroborates this reward-quality gap, most clearly on two of the three families. No single alignment method dominates, and closing the gap will require richer reward designs and continued investment in dialectal resources. We release all code, checkpoints, and preference datasets. 

---
# Future Confidence Distillation in Large Language Models 

**Authors**: Sahil Kale  

**Link**: [PDF](https://arxiv.org/pdf/2607.07626)  

**Abstract**: Reliable confidence estimation is essential for deploying large language models (LLMs) in confidence-aware systems, where downstream decisions such as retrieval, tool use, and adaptive computation depend on accurately estimating answer reliability. Existing approaches, however, largely treat confidence as a property of completed responses, overlooking how confidence-related information evolves throughout the answering process. In this work, we investigate confidence from a temporal perspective by comparing pre-solution Feeling-of-Knowing (FOK) and post-solution Judgement-of-Learning (JOL) confidence estimates across frontier and open-source LLMs. We show that post-solution confidence is consistently better calibrated and more discriminative than pre-solution confidence, while linear probes trained on hidden representations recover substantially richer confidence-related information than models explicitly verbalise. Building on this observation, we introduce future confidence distillation, which trains predictors operating on pre-solution hidden representations using teacher confidence estimates produced by post-solution correctness probes. Despite requiring only pre-solution representations for inference, distilled predictors recover much of the calibration improvement achieved by post-solution confidence, remain highly sample efficient, and transfer across datasets within the same domain. Together, our findings demonstrate that confidence-related information evolves throughout the answering process and can be anticipated before answer generation is complete, enabling significantly more reliable yet low-cost confidence estimation. 

---
# PALS: Percentile-Aware Layerwise Sparsity for LLM Pruning 

**Authors**: Yazdan Jamshidi, Alexey Shvets  

**Link**: [PDF](https://arxiv.org/pdf/2607.07557)  

**Abstract**: One-shot pruning methods like Wanda and SparseGPT apply the same sparsity ratio to every layer of a transformer, ignoring known variation in layer importance. We propose PALS (Percentile-Aware Layerwise Sparsity), which adjusts per-layer sparsity based on the 99th percentile of activation magnitudes, bounded to $\pm 5\%$ around the target ratio. On LLaMA-2-7B at 50\% sparsity, PALS achieves 10.96 WikiText-2 perplexity versus 12.92 for uniform Wanda (mean over 9 runs, $p < 0.001$). The benefit is architecture-dependent: LLaMA-3-8B shows marginal gains and Mistral-7B shows none. We also find that gradient-based allocation -- the seemingly more principled approach -- produces results worse than random, suggesting that gradient magnitude does not predict the impact of discrete weight removal. PALS adds negligible cost to the pruning pipeline and requires no fine-tuning. 

---
# Think Big, Search Small: Where Capacity Matters in Hierarchical Search Agents? 

**Authors**: Qinnan Cai, Yibo Zhao, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.07548)  

**Abstract**: Large language model based search agents increasingly adopt multi-agent architectures in which a main agent decomposes a complex question into sub-queries and dispatches them to parallel sub-agents. However, existing systems instantiate all roles from a single model of identical scale, leaving open how model capacity should be distributed across roles. We factorize hierarchical search into three roles: a delegation role responsible for task decomposition, an execution role responsible for retrieval and evidence extraction, and an answer generation role held fixed as a confound control. We then conduct controlled capacity sweeps along the delegation and execution axes on five multi-hop QA benchmarks. The experiments yield three findings. First, role factorization consistently outperforms a single-agent baseline, improving exact match from 4.5 to 8.6 points across six model scales. Second, capacity sensitivity is asymmetric: scaling the delegation backbone improves EM by ~11 points, whereas scaling the execution sub-agent moves EM by only ~2.6 points, identifying decomposition as the capability bottleneck. Third, a 1.7B-parameter executor trained via quality-filtered trajectory distillation matches a frontier sub-agent in accuracy while consuming 37% fewer sub-agent tokens, advancing the Pareto frontier. These results suggest a concrete recipe for building hierarchical search agents: concentrate capacity at delegation and downsize execution without sacrificing accuracy. Our code is available at this https URL. 

---
# SynthAVE: Scalable Synthetic Labeling for E-Commerce with LLM-Arena Validation 

**Authors**: Andrea Scarinci, Virginia Negri, Brayan Impata, Suleiman Khan, Victor Martinez, Marcello Federico  

**Link**: [PDF](https://arxiv.org/pdf/2607.07469)  

**Abstract**: Fine-tuning large language models (LLMs) for e-commerce attribute extraction requires labeled data representative across thousands of product types, attributes, and multiple languages. This combinatorial scale translates to millions of annotations, rendering human labeling prohibitively costly. While recent work has demonstrated synthetic label generation using LLMs, deploying such approaches at industrial scale requires integrated quality control mechanisms. We present SynthAVE, a large-scale human-validated benchmark for attribute value extraction spanning 12,726 products across 229 product types, 792 attributes, and 4 languages (Spanish, French, Italian, German). To validate synthetic labels at scale, we introduce a multi-LLM arena framework where samples are independently evaluated by 21 judge configurations (7 model families $\times$ 3 prompts), with final labels determined via majority voting. The majority vote ensemble agrees with human experts at Cohen's $\kappa = 0.92$ (95.2% agreement), while individual judges show substantial inter-model agreement (Fleiss' $\kappa = 0.76$). This demonstrates that diverse models with varying individual judgments aggregate into highly reliable predictions, enabling cost-effective validation at scale while maintaining quality parity with human review. 

---
# DeLS-Spec: Decoupled Long-Short Contexts for Parallel Speculative Drafting 

**Authors**: Hong-Kai Zheng, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.07409)  

**Abstract**: Speculative decoding accelerates LLM inference by drafting multiple tokens and verifying them in parallel. Block-parallel drafters such as DFlash further improve drafting efficiency by predicting an entire block in one pass, but their position-wise predictions lack explicit intra-block causal conditioning. Recent methods such as Domino and DSpark attempt to introduce such causality into block-parallel drafting, but they require training the draft model from scratch, which limits their flexibility and increases training cost. We propose DeLS-Spec, a decoupled long-short context speculative decoding method. DeLS-Spec treats the fixed DFlash model as a long-context expert and introduces a lightweight local head as a short-context expert. The local head can be trained independently with a standard next-token prediction objective, without joint training with the target model or the DFlash backbone, leading to extremely low training cost. At inference time, DeLS-Spec combines long-context and short-context logits, and the local head is not tied to a specific DFlash checkpoint, making the method more modular and flexible. Experiments on Qwen3 models show that DeLS-Spec consistently improves speedup and average acceptance length over DFlash across math, code, and dialogue benchmarks. 

---
# Transformer-based segmentation of prosodic boundaries in Brazilian Portuguese 

**Authors**: Rodrigo de Freitas Lima, Julio Cesar Galdino, Marcos Vinicius Treviso  

**Link**: [PDF](https://arxiv.org/pdf/2607.07408)  

**Abstract**: Automatic prosodic segmentation identifies boundaries between speech units from acoustic and linguistic evidence. Although recent deep learning approaches have produced strong results for English, automatic segmentation for Brazilian Portuguese (BP) still relies mostly on rule-based or traditional machine-learning methods. This paper presents SAMPA, a Whisper-based segmenter that transcribes BP speech while inserting explicit markers for terminal prosodic boundaries. We fine-tune Whisper large-v3 on manually segmented recordings from the NURC-SP dataset and evaluate different training and test-time filtering configurations, including out-of-distribution testing on the MuPe-Diversidades dataset. SAMPA achieves competitive boundary-detection performance across settings, with the best models reaching F1=0.731 on the held-out test split and F1=0.796 on MuPe-Diversidades. Finally, through n-gram and acoustic-visual analyses, we show that our model follows morphosyntactic, semantic, and prosodic cues for detecting prosodic boundaries. 

---
# TF-Engram: A Train-Free Engram with SSD-Backed Memory for Large Language Models 

**Authors**: Yutang Ma, Kecheng Huang, Xikun Jiang, Zili Shao  

**Link**: [PDF](https://arxiv.org/pdf/2607.07388)  

**Abstract**: Large Language Models (LLMs) store factual knowledge and domain-specific patterns implicitly in dense Transformer parameters, making knowledge expansion costly through pretraining, fine-tuning, retrieval augmentation, or longer contexts. Engram-style memory offers a compact hidden-state injection pathway, but existing GPU-resident designs often rely on hash-based compression, causing unrelated phrases to collide in shared slots and weakening phrase-level semantic fidelity. We present TF-Engram, a train-free Engram system that constructs phrase-specific semantic memory offline from external corpora, stores large memory tables across a GPU--DRAM--SSD hierarchy, and uses Early-Exit Guided Predictive Prefetching to hide external-memory latency during autoregressive decoding. On Qwen3-0.6B, TF-Engram improves the average downstream score from 57.6 to 59.4, outperforming both the frozen backbone and a parameter-matched LoRA baseline. System evaluation shows that large TF-Engram tables can be built with moderate offline cost, SSD-backed storage substantially reduces GPU memory demand, and predictive prefetching recovers much of the throughput loss caused by external memory access. These results demonstrate that static phrase memory can be integrated into LLM inference as a scalable, train-free, and low-overhead system component. 

---
# R^3: Advertisement Compliance Rectification via Group-Relative Experience Extractor and Curriculum Reinforcement 

**Authors**: Yuan Chen, Zhenyu Hu, Mengge Xue, Te Cao, Liqun Liu, Peng Shu, Huan Yu, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2607.07318)  

**Abstract**: Rigorous content moderation is crucial for online advertising but leads to millions of daily rejections. This scale renders manual rectification infeasible, particularly for video advertisements. However, existing safety-driven methods often suffer from aggressive over-editing, which compromises the advertiser's original semantic intent merely to satisfy compliance. In this work, we target the rectification of textual violations in video ads, covering both speech transcripts and on-screen text. We propose R^3, a novel framework designed to harmonize compliance with original semantic intent preservation. Our approach integrates three key innovations: (1) an experience-driven data synthesis framework that bootstraps high-quality supervision via a group-Relative compliance experience extractor; (2) a curriculum Reinforcement learning strategy with hierarchical rewards designed to enforce compliance while maximizing semantic consistency; and (3) a comprehensive video Rectification framework seamlessly integrating text recognition, rewriting, and re-rendering for industrial deployment. Extensive experiments on industrial datasets and online A/B testing demonstrate that R^3 significantly outperforms state-of-the-art baselines, achieving an optimal trade-off between violation rectification and intent preservation. 

---
# Evaluating RAG Metrics in Applied Contexts: An Experiment, Its Findings and Its Limitations 

**Authors**: Quentin Brabant  

**Link**: [PDF](https://arxiv.org/pdf/2607.07302)  

**Abstract**: This paper reports an empirical study evaluating the relevance of several RAG metrics. The experiment is based on a question-answering dataset created by human annotators from business data. The generated responses and retrieved spans of a RAG system are scored using evaluation metrics from four libraries (Ragas, DeepEval, RAGChecker, Opik). These metrics are compared to scores given by two evaluators, as well as to standard metrics such as recall. An analysis of correlations is conducted. Finally, we highlight certain limitations of our methodology, compare it to those used in the literature, and suggest some avenues for future research. This paper is an English translation of a paper originally published in the French-speaking workshop EvalLLM (Brabant, 2026). 

---
# A Word-Level Digital Reader of the Prasthanatrayi with Sankara's Bhasya: Corpus, Method, and an Open, Offline Reading Aid for the Advaita Vedanta Canon 

**Authors**: Tamal Maharaj  

**Link**: [PDF](https://arxiv.org/pdf/2607.07282)  

**Abstract**: The Prasthanatrayi -- the ten principal Upanisads, the Brahmasutra, and the Bhagavadgita, with Sankara's commentaries (bhasya) -- is the foundational corpus of Advaita Vedanta. Continuous euphonic combination (sandhi), long compounds (samasa), and dense scholastic prose make it hard to read at the word level: where one word ends, and what each word means grammatically, are both obscured. We present an open, fully offline, word-level digital reader of the entire Prasthanatrayi with Sankara's bhasya. Every word -- of both the root text (mula) and the commentary -- is clickable and resolves to a pop-up giving its split (padaccheda), morphological analysis, and gloss. Because every word carries a lemma, the reader also acts as a concordance: a search on a dictionary headword retrieves all of that word's inflected and sandhi-hidden occurrences, and its occurrences inside compounds, across both layers. The resource covers thirteen commentarial units (2,971 verses, sutras, and prose sections; 36,881 analysed word-occurrences of root text) and a global dictionary of 95,587 distinct commentarial surface forms. We describe the corpus, the hybrid pipeline -- a rule-based sandhi splitter over an inflected-form lexicon and attested-corpus look-ups, with LLM-assisted analysis under an adversarial two-pass verification protocol -- and a durable human-review loop whose corrections survive every regeneration. An intrinsic evaluation against independent Sanskrit resources finds high-confidence analyses agree with an authoritative inflectional lexicon on over 99% of attested forms, and a band-blind adjudication confirms that quality degrades predictably across confidence bands, with errors concentrated in the low-confidence tier the review loop targets. The reader is a single self-contained HTML file needing no server or network, offered as a freely redistributable teaching and reading aid. 

---
# Understanding Interpretation Difficulty in Harmful Online Communication: Insights from Cybercrime Communities 

**Authors**: Tomohiro Okatsu, Naoki Takada, Yin Min Pa Pa, Katsunari Yoshioka, Tatsunori Mori  

**Link**: [PDF](https://arxiv.org/pdf/2607.07277)  

**Abstract**: Harmful online communication often contains slang, coded terms, abbreviations, and community-specific expressions, which make messages difficult to interpret. This paper presents an exploratory study of interpretation difficulty in Discord chats related to cybercrime. We construct reference interpretations of purposefully selected difficult messages, which were reviewed by an expert. We then use them to evaluate human and large language model (LLM) interpretations under different context conditions. The results show that local context alone is often insufficient for humans, while external knowledge and extended conversational context substantially improve human interpretation. For LLMs, local context also improves interpretation, and the larger model performs better. We further conduct a qualitative error analysis and propose a preliminary classification of factors that make harmful chats difficult to interpret. These findings suggest that harmful-content analysis should treat interpretation as an evidence-integration problem, rather than as message-level classification alone. 

---
# Evaluation of Multilingual Ability to Use Spatial Deictic Expressions in Vision-Language Models 

**Authors**: Kaito Watanabe, Taisei Yamamoto, Tomoki Doi, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2607.07251)  

**Abstract**: One of the expected abilities of vision-language models (VLMs) is spatial reasoning ability based on a given text and image. To evaluate the spatial reasoning abilities of VLMs, we focus on the use of spatial deictic expressions, which are defined as spatial expressions whose referent is determined by their situational context, such as ``this'' and ``that''. To handle spatial deictic expressions, VLMs must jointly reason over language and visual space, grounding context-dependent references in the image's spatial structure. In addition, selecting appropriate spatial deictic expressions across languages requires VLMs to understand the language-specific spatial distinctions encoded by these expressions. In this paper, we develop a benchmark to evaluate the multilingual ability of VLMs to use spatial deictic expressions in four languages. Our experiments using this benchmark reveal that the tested models use demonstratives in a manner different from that of humans, particularly in selecting the appropriate demonstratives based on the distance to the object. 

---
# From Text to Parameters: Predicting Item Parameters from Embedding Regularization with Reliability and Design Ceilings 

**Authors**: Shi-Ting Chen, Jinsong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.07141)  

**Abstract**: Newly developed items must ordinarily be field tested before their psychometric properties are known, creating a cold start problem for item calibration. Predicting item parameters from features is a long standing measurement problem dating back to the Linear Logistic Test Model; modern text embeddings now automate the design matrices traditionally specified by hand. We propose an evaluation framework combining regularized regression on item text embeddings, repeated cross validated R squared reported with its resampling standard deviation, and two performance upper bounds: a reliability ceiling derived from parameter standard errors, and a design ceiling derived from simulation based power calibration. Applying this framework to a mathematics item bank (EEDI) and a medical licensure benchmark (BEA 2024), we find that item difficulty is highly predictable from text (repeated cross validated R squared = 0.53, or about 57% of its reliability ceiling), whereas discrimination and pseudo guessing appear less predictable. However, evaluating these results against our ceilings reveals that this apparent hierarchy stems from target reliability rather than text signal strength: text uniformly recovers 57 to 63% of the reliable variance across difficulty targets, whereas the 3PL pseudo guessing parameter has a reliability ceiling near zero, making it an unviable target at current precision. On BEA, embedding based regression matches leaderboard RMSE despite explaining almost no variance, highlighting the critical need for scale free metrics and explicit ceilings in benchmarking. Finally, we show that a single train and test split can inflate apparent accuracy by 0.1 to 0.15 in R squared, underscoring the necessity of repeated cross validation for calibration support applications and future benchmark construction. 

---
# Behavior Leverage Imbalance in Multi-Teacher On-Policy Distillation 

**Authors**: Jiabin Shen, Guang Chen, Chengjun Mao  

**Link**: [PDF](https://arxiv.org/pdf/2607.07050)  

**Abstract**: Agentic language models must learn when to call tools, when to consume tool responses, and when to answer directly. This makes multi-teacher on-policy distillation a natural training strategy: one teacher can specialize in tool calls, another in direct responses, and the student can learn from both on
its own generated distribution. We show that this strategy can induce a behavior shift that is invisible from aggregate losses alone. In a two-teacher tool-use setting, vanilla generalized knowledge distillation improves tool-call recall but also moves the model toward over-calling, where it calls tools
on examples that should be answered directly. Aggregate explanations are insufficient: tool-call samples do not receive more token exposure, and full-sequence per-token divergence is not larger for the tool-call teacher. We instead analyze behavior leverage imbalance: local token-level signals at mode-
entry and structural positions, such as <tool_call> and function names, can have disproportionate control over the global generation mode. We propose Soft Clamp, a per-token divergence calibration method that dynamically compresses extreme token-level Jensen-Shannon divergence while preserving nonzero
gradients. On APIGen-MT, Soft Clamp reduces over-calling from 13.7% to 9.0% relative to vanilla GKD while matching its decision accuracy. In a BFCL multi-turn diagnostic, it also lowers tool-call loops and repeated calls among GKD variants. These results suggest that multi-teacher OPD should monitor
where teacher signals act, not only how large they are in aggregate. 

---
# Riemannian Geometry for Pre-trained Language Model Embeddings 

**Authors**: Szczepan Konior, Alexandre Quemy, Przemysław Klocek, Grégoire Cattan, Bartłomiej Sobieski  

**Link**: [PDF](https://arxiv.org/pdf/2607.07047)  

**Abstract**: Understanding the geometric structure of pre-trained language model embeddings matters for interpretability and safety. We ask whether sentence-level classification signal lives in the Riemannian geometry of contextual token embeddings, and probe it by extracting per-token pullback metrics from a learned encoder's analytical Jacobian and aggregating them with the Fréchet mean on the symmetric positive definite (SPD) manifold; we call this procedure Riemannian Mean Pooling (RMP). Across three datasets with non-trivial linguistic structure (CoLA, CREAK, RTE), RMP outperforms Euclidean mean pooling, while on FEVER-Symmetric, a benchmark constructed to remove annotation-driven lexical artifacts, the method correctly stays at chance. Ablations show that a randomly initialised encoder combined with Fréchet aggregation already beats Euclidean pooling on two of the three signal-bearing datasets, localising the source of the gain to the geometric aggregation rather than to learned manifold structure; the trained encoder contributes additional signal specifically on CREAK, the most knowledge-heavy of the three signal-bearing datasets. 

---
# MILES: Modular Instruction Memory with Learnable Selection for Self-Improving LLM Reasoning 

**Authors**: Ruilin Tong, Dong Gong  

**Link**: [PDF](https://arxiv.org/pdf/2607.06974)  

**Abstract**: Large language models (LLMs) increasingly improve their reasoning at test time via additional computation, yet most existing works treat each problem in isolation. When problems arrive sequentially, accumulating reusable experience across them can further improve performance. Existing memory-based methods either store whole-solution templates that generalize poorly to novel problems or use heuristic step-level selection that is not optimized for final-answer correctness. Learning selection policies requires large-scale training data and fixed action spaces, making such approaches unsuitable for test-time settings where memory expands incrementally and only limited supervision is available. We propose MILES (Modular Instruction Memory with LEarnable Selection for self-improving LLM reasoning), a framework that dynamically expands step-wise memory and applies correctness-optimized memory composition under realistic test-time constraints. MILES maintains modular memory units consisting of asymmetric pairs of sub-goal embeddings and sub-instructions, each associated with a learnable selection head. This memory structure enables a coarse-to-fine retrieval mechanism: The coarse level enables memory expansion and collects supervision for training selection heads from confident samples, while the fine stage applies learned selection heads to rerank coarse-level candidates and guide reasoning for uncertain samples. MILES consistently matches or outperforms prior methods while achieving superior accuracy-efficiency tradeoffs. Extensive experiments demonstrate its effectiveness, robustness, and transferability. 

---
# Comprehensive Evaluation of Large Language Model Responses: A Multi-Factor Scoring System 

**Authors**: Yiming Gai, Junde Lu, Xuefei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.06940)  

**Abstract**: The remarkable performance of large language models (LLMs) in linguistic tasks underscores an urgent need for comprehensive evaluation of their response quality. Prevailing methods, often confined to singular dimensions, fall short of capturing the full spectrum of model capabilities. This study introduces a multifactor scoring paradigm, integrating accuracy, conciseness, factual consistency, readability, and coherence, complemented by a graphical user interface (GUI) for visualizing outcomes. Evaluations on the TruthfulQA dataset unveil mainstream LLMs' strengths in reasoning tasks (peaking at a composite score of 0.6104) alongside pervasive limitations in navigating complex facts and ambiguities. Transcending the narrow lens of traditional metrics, this framework offers a transparent, adaptable avenue to illuminate model potential and deficiencies. Though presently focused on English tasks, its horizons beckon toward multilingual domains. This work carves a novel path for knowledge engineering and model refinement. 

---
# LLMs Silently Correct African American English: Auditing and Mitigating Dialect Bias via Activation Steering 

**Authors**: Huan Wu, Ali Emami, Muhammad Furquan Hassan, Osaretin Igbinoba, Osakpolor Idusuyi, Osamede Igbinoba, Faiza Khan Khattak, Laleh Seyyed-Kalantari  

**Link**: [PDF](https://arxiv.org/pdf/2607.06845)  

**Abstract**: African American English (AAE), a rule-governed dialect spoken by over 30 million people, is routinely misinterpreted and "corrected" by large language models (LLMs). Across six instruction-tuned LLMs (14B to 70B), we show that state-of-the-art models systematically prefer Standard American English (SAE) continuations even when the preceding context is in AAE, effectively rewriting AAE into SAE. We present an end-to-end framework to audit and mitigate this bias. For auditing, we introduce conditional Dialect Group Invariance (cDGI), which isolates true model bias from translator-induced artifacts, and a feature-level localization analysis that identifies which AAE markers most strongly trigger bias; we find that syntactic constructions, especially negative concord (e.g., "ain't nobody"), are universal triggers across all models. For mitigation, we introduce, to our knowledge, the first application of activation steering to dialect bias: a training-free, test-time method that extracts dialect directions via causal tracing and injects them into bias-relevant layers. Activation steering reduces bias 5 to 20 times more than prompting while preserving SAE fluency. To enable this work, we release REAL-AAE , the largest real-AAE parallel corpus to date: 17,479 AAE/SAE/ AAE_back triplets from natural tweets (2 to 6 times larger than prior real-AAE resources), validated automatically (BERTScore F1 = 0.95) and by three native AAE speakers (83.0% semantic agreement). 

---
# Gradient-Based Speech-to-Text Alignment for Any ASR Model: From CTC to Speech LLMs 

**Authors**: Albert Zeyer, Ralf Schlüter, Hermann Ney  

**Link**: [PDF](https://arxiv.org/pdf/2607.06831)  

**Abstract**: Speech-to-text alignment means finding the temporal boundaries of each word in the audio. Some models provide such an alignment directly and others do not. Connectionist temporal classification (CTC) and transducer models have an alignment by construction, whereas attention-based encoder-decoders (AED) and speech large language models (LLMs) do not, and their word timings are usually read off the attention weights instead. All of these signals live on the encoder frame grid, which bounds their temporal precision. We study a generic gradient-based alignment that applies to any differentiable ASR model. We take the gradient of each teacher-forced token log probability with respect to the input, reduce it to a per-frame saliency, and decode the resulting matrix into word boundaries with a single dynamic-programming pass. The method needs no training, no model modification and no alignment heads, works across all model families including the speech LLMs, and aligns on the input grid rather than on the coarser encoder grid. We evaluate it on sixteen models from four families, on read (TIMIT) and spontaneous (Buckeye) speech, each against the model's own native or attention-based alignment. We find that the gradient yields a usable alignment for every model, that it is usually somewhat behind a strong native aligner but better where the native alignment is weak, as for the streaming models, and that its main disadvantage is the cost of one backward pass per token. 

---
# Ad Headline Generation using Self-Critical Masked Language Model 

**Authors**: Yashal Shakti Kanungo, Sumit Negi, Aruna Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2607.06818)  

**Abstract**: For any E-commerce website it is a nontrivial problem to build enduring advertisements that attract shoppers. It is hard to pass the creative quality bar of the website, especially at a large scale. We thus propose a programmatic solution to generate product advertising headlines using retail content. We propose a state of the art application of Reinforcement Learning (RL) Policy gradient methods on Transformer based Masked Language Models. Our method creates the advertising headline by jointly conditioning on multiple products that a seller wishes to advertise. We demonstrate that our method outperforms existing Transformer and LSTM + RL methods in overlap metrics and quality audits. We also show that our model-generated headlines outperform human submitted headlines in terms of both grammar and creative quality as determined by audits. 

---
# Healthier LLMs: Retrieval-Augmented Generation for Public Health Question Answering 

**Authors**: Felix Feldman, Joshua Harris, Timothy Laurence, Leo Loman, Ollie Higgins, Fan Grayson, Poonam Soma, Bethany Pace-Bonello, Michael Borowitz, Toby Nonnenmacher  

**Link**: [PDF](https://arxiv.org/pdf/2607.06641)  

**Abstract**: Large language models (LLMs) achieve promising results on medical question answering benchmarks, yet their use in public health is constrained by hallucinations and the rapid evolution of official guidance. Retrieval-Augmented Generation (RAG) mitigates these risks by grounding responses in an explicitly maintained corpus, but end-to-end performance depends critically on retrieval configuration and on evaluation beyond multiple-choice formats. We extend PubHealthBench, a question answering (QA) benchmark of 7,929 questions derived from UK Government public health guidance, into a retrieval-augmented setting and systematically evaluate retrieval and generation choices. We compare dense, sparse, and hybrid retrieval across multiple embedding models and corpus variants, and show that hybrid retrieval consistently improves recall and ranking quality, with chunk length and topic interacting with ranking performance. Providing retrieved context substantially increases multiple-choice accuracy across a diverse set of LLMs, enabling smaller open-weight models to match or outperform larger models used without retrieval, with gains primarily driven by retrieval quality and careful context selection. To assess realistic free-form answering, we introduce a rubric-based LLM-as-a-judge covering faithfulness, completeness, clarity, and factual consistency, and validate it against dual human annotations. Judge-human agreement is strongest for faithfulness and completeness, while factual consistency and clarity are less reliably reproduced, motivating caution when interpreting those dimensions at scale. Overall, our results highlight retrieval as a primary lever for reliable public health QA and provide practical guidance for building and evaluating RAG systems grounded in official guidance. 

---
# Audio Sentiment Analysis via Distillation and Cross-Modal Integration of Generated Multilingual Transcripts 

**Authors**: Andrei-George Durdun, Victor Constantinescu, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2607.06611)  

**Abstract**: Automatically recognizing the sentiment, positive or negative, from speech is a challenging task, requiring both the analysis of vocal inflections and the interpretation of uttered words. Recent solutions rely on audio foundation models to solve the task, but it remains unclear if such models can take all aspects into account. To this end, we propose a multimodal solution that integrates audio and text information via cross-modal transformers, where text transcripts are automatically generated via an automatic speech recognition (ASR) tool. Moreover, we create multiple text modalities by automatically translating the transcripts into multiple languages via machine translation tools. Audio and multilingual text features are combined via a cascaded architecture comprising cross-modal transformer blocks that integrate modalities one by one. We further distill knowledge from the multimodal model, called teacher, into a unimodal (audio only) model, called student. We conduct experiments on a large-scale dataset, demonstrating that the automatically generated textual information can bring significant performance boosts in multimodal sentiment polarity classification. Our ablation study confirms that both automatic transcripts and automatic translations are helpful. Moreover, we show that the audio-only model can be enhanced via distillation, boosting performance without any computational overhead during inference. To reproduce the reported results, we publicly release our code at this https URL. 

---
# Agon: Competitive Cross-Model RL with Implicit Rival Grading of Reasoning 

**Authors**: Vladislav Beliaev  

**Link**: [PDF](https://arxiv.org/pdf/2607.07690)  

**Abstract**: Reinforcement learning from verifiable rewards (e.g. GRPO) is the engine behind today's reasoning models, yet it grades only the final answer. On hard problems this trains models to write more rather than to think better, since the trace itself is never graded and no label for good thinking exists. We introduce Agon, which makes two competing models each other's graders. Both attempt the same problem; in alternating roles, one drafts a solution and the other reads it while solving, and each is rewarded for out-solving the other. To win, a model must out-reason a rival that has seen its work, so reasoning is judged implicitly during training, with no process labels and no reward model. Because both models are optimized, each faces a progressively stronger rival, which single-model RL cannot provide. The two need only be comparably strong and behaviorally different. At inference the pair deploys as it trains, a two-stage cascade in which one model drafts and the other answers after reading the draft. On the hard split of DeepMath with Qwen3, this doubles GRPO's pass@1, roughly eight times the gain of an untrained Mixture-of-Agents pass over the same base. The ordering replicates on competitive-programming code and across model families (Qwen3.5, Gemma 4). For now the models talk in text; the next step is to let them reason together in latent space. 

---
# Max Out GRPO Signal: Adaptive Trace Prefix Control for Hard Reasoning Problems 

**Authors**: Vladislav Beliaev  

**Link**: [PDF](https://arxiv.org/pdf/2607.07674)  

**Abstract**: Group Relative Policy Optimization (GRPO) stalls on a model's hardest problems: when no rollout in a group succeeds, the group-relative advantages vanish and the problem contributes no gradient, wasting the frontier examples we most want to learn from. Prepending a correct prefix of a reference solution raises the success rate, making prefix length a continuous knob on difficulty. Concurrent methods set the knob once; AdaPrefix-GRPO turns it into a feedback controller: throughout training it adjusts how much of the solution each problem gets, holding its success rate near 50%, where GRPO's gradient signal is largest, then withdraws the assistance entirely, so the deployed model solves problems unaided. On hard math, at matched training FLOPs, it more than doubles GRPO's accuracy on held-out problems from the training distribution for a 0.6B model (2.1x), with 1.6x on Qwen3-1.7B and 1.7x on AIME, while roughly halving trace length. The method is implemented in data preparation plus a loss mask on prefix tokens; the trainer is otherwise stock. The smaller the model, the larger the gain. 

---
# RL Post-Training Builds Compositional Reasoning Strategies 

**Authors**: Azwar Abdulsalam, Nishil Patel, Andrew Saxe  

**Link**: [PDF](https://arxiv.org/pdf/2607.07646)  

**Abstract**: Does RL post-training merely amplify primitive skills already latent in a base model, or can it compose primitive skills into new higher-level strategies? We study this question in a fully observable rewrite-grammar environment where the pretraining distribution is known and every generated rewrite can be audited. A Transformer is pretrained on primitive symbol-rewrite chains and post-trained on a Trace-based reasoning task with only a binary final-answer reward. RL solves held-out problems that remain rarely solved by the pretrained model even under much larger sampling budgets, while rejection fine-tuning improves early but plateaus. Trace analysis shows that RL reorganizes primitive competence through a phased compositional mechanism: it first strengthens primitive reductions, then discovers valid composed procedures. These include sequential compositions, which collapse ordered chains of primitive contractions, and parallel compositions, which combine independent primitive contractions in a single step. The composed procedures are not isolated samples; they are reused and consolidated into a stable repertoire. Comparing RL with rejection fine-tuning shows that the key difference is not exploration volume but selectivity: RFT produces many shortcut-like rewrites, much of them invalid, whereas RL concentrates exploration into valid reusable structure. Pretraining ablations show that the emergence of compositional strategies is gated not by primitive exposure alone, but by whether pretraining organizes primitive competence into reduction procedures that RL can later compress. The base model provides weak procedural ingredients; RL builds them into reliable higher-level strategies. 

---
# FourierQK: Spectral Preprocessing of Query-Key Projections Improves Transformer Attention 

**Authors**: Athanasios Zeris  

**Link**: [PDF](https://arxiv.org/pdf/2607.07478)  

**Abstract**: FFT-based spectral preprocessing of learned query-key (Q/K) projections substantially improves transformer attention on character-level language modelling. On TinyShakespeare: a fixed random spectral filter achieves val=1.031 (Delta=+0.443); a single learned frequency at paragraph scale achieves val=0.608 (Delta=+0.867); and four learned frequencies spanning paragraph to word scale achieve val=0.309 (Delta=+1.166), a 79% reduction over standard dot-product attention. The single-frequency result is confirmed across three random seeds (mean val=0.236, std=0.019). The four frequencies converge to a near-geometric multi-scale ordering (49, 27, 10, 6 tokens/cycle) corresponding to paragraph, sub-paragraph, phrase, and word scales. The gain is specific to spectral preprocessing: random orthogonal and non-orthogonal projections of Q/K produce no measurable improvement, suggesting the benefit comes from global frequency-domain mixing rather than metric distortion. All results are verified by a shuffled-validation diagnostic against positional leakage. Causal filters (Gaussian, Mexican Hat, Morlet) do not improve over standard attention at character-level tokenisation: the bilateral FFT kernel is structurally non-causal, coupling every position to future tokens. This defines an architectural boundary between bilateral spectral attention (this paper) and genuinely causal spectral attention at word-scale tokenisation (companion paper MorletQK). This work is architecturally distinct from FNet (Lee-Thorp et al., 2021), which replaces attention with Fourier mixing of token embeddings. Here, spectral preprocessing applies only to Q/K projections while the full attention score structure is preserved. 

---
# Beyond Attack-Success Rate: Action-Graded Severity Scale for Tool-Using AI Agents 

**Authors**: Harry Owiredu-Ashley  

**Link**: [PDF](https://arxiv.org/pdf/2607.07474)  

**Abstract**: Agentic red-teaming benchmarks report whether an injected agent was compromised as a single bit: the attack succeeded, or it did not. We argue that this binary attack-success rate discards the information a defender most needs, namely how harmful the resulting action was. We introduce an action-graded harm rubric that scores an agent's tool-call trajectory on a seven-level ordinal scale (L0 to L6) according to whether the executed action was reversible, whether it crossed scope to reach another party, and whether it expanded privilege. We compute the scale two ways: a deterministic oracle that reads the trajectory and the attacker's stated goal, and a panel of three frontier language-model judges that read a tag-free account of the same trajectory. Across four victim models and two defenses on the AgentDojo workspace suite, severity grading exposes three cases the binary metric hides, including a defense that reports a zero attack-success rate while still permitting an externally visible cross-scope leak through an unfiltered tool. The judge panel reproduces the oracle with high ordinal agreement (Krippendorff's alpha = 0.91) but shares systematic blind spots that we characterize, most notably a failure to recognize escalation chains. Unlike prior work that provides harm taxonomies, harmful-task completion tests, execution-level safety benchmarks, or severity-aware simulation, our contribution is a reusable, trace-grounded severity instrument applied to the actual actions recorded in existing red-team logs. All code, prompts, and per-episode logs are released. 

---
# The Blind Curator: How a Biased Judge Silently Disables Skill Retirement in Self-Evolving Agents 

**Authors**: Xing Zhang, Yanwei Cui, Guanghui Wang, Ziyuan Li, Wei Qiu, Bing Zhu, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2607.07436)  

**Abstract**: A self-evolving agent retires its bad skills by watching them fail, so what happens when the judge cannot see the failures? Skill retirement is the structural constraint that keeps a growing library from drifting below the no-skill baseline, but its guarantee assumes an unbiased reward, which is false for the LLM judges that reference-free tasks force upon us. We show that a biased judge does not merely add noise; it \emph{silently switches off the curator}. We make this precise with a corrupted-reward analysis and, isolating the causal channel by injecting corruption on top of a deterministic reward, a behavioral study on a reference-free report-writing testbed with a code-generation cross-check. Symmetric noise leaves retirement intact, but \emph{false-pass} bias (failures slipping through as passes) disables contribution-based retirement past a sharp threshold that no amount of data can cross. Separating genuine retirement from cap-eviction churn shows this \emph{mechanism} failure is universal, holding across domains and failure rates and sparing only near-zero-false-pass, verifier-like graders. The downstream \emph{outcome}, though, is regime-dependent: eval quality degrades only where the same corruption also starves skill synthesis, and otherwise holds steady, so the disabled curator is \emph{silent}, surfacing in no aggregate metric. The contribution is a behavioral safety result, not a performance one. A cheap defect-injection audit then tells an operator, before deployment, which side of the threshold their judge occupies. 

---
# From Atomic Actions to Standard Operating Procedures: Iterative Tool Optimization for Self-Evolving LLM Agents 

**Authors**: Haipeng Ding, Yuexiang Xie, Zhewei Wei, Yaliang Li, Bolin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2607.07321)  

**Abstract**: Tool utilization enables Large Language Model (LLM) agents to interact with the real world and resolve complex tasks. However, existing agent frameworks predominantly rely on static toolsets composed of granular atomic actions (e.g., basic file I/O or single-turn search), which forces agents to reinvent low-level logic for every recurring workflow, leading to increased reasoning overhead and failure rates. In this study, we propose that agents can achieve self-evolution by synthesizing these atomic actions into reusable Standard Operating Procedures (SOPs), which function as callable higher-order tools that encapsulate multi-step logic. We further introduce EvoSOP, a framework that empowers agents to extract SOPs from execution trajectories and iteratively optimize the toolset through a systematic lifecycle of construction, merging, evaluation, and pruning. Extensive experiments demonstrate that EvoSOP significantly boosts task success rates while substantially reducing the number of interaction rounds compared to baselines. Our analysis also reveals that iterative tool optimization fosters reliable and efficient tool-use patterns, providing a scalable pathway for the development of self-evolving agents. 

---
# Multimodal Voice Activity Projection for Turn-Taking in Social Robots with Voice-Activity-Related Pretrained Encoders 

**Authors**: Antonio Cano, Guillermo Pérez, Luis Merino, Randy Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2607.07294)  

**Abstract**: Turn-taking prediction is a key requirement for social robots involved in human-human interaction, particularly in mediator settings, where the robot must anticipate conversational dynamics rather than merely react to pauses. This work presents a Multimodal Voice Activity Projection (MM-VAP) framework that extends the original audio-only VAP formulation to synchronized audio-visual inputs while preserving its self-supervised future-projection objective. The proposed approach builds on pretrained audio-visual backbones originally optimized for speech-related tasks and adapts them through Low-Rank Adaptation to the multimodal turn-taking problem. After independent speaker encoding, an inter-speaker attention stage models the relational dynamics required to project future voice activity. In addition, a semantic consistency loss is introduced to regularize the 256-state output space according to higher-level dialogue activity patterns. Experiments on NoXi and NoXi+J showed improvements over the current baselines, particularly for some turn-taking events. Additional evaluation on the Haru EDR corpus further supported the suitability of this direction for mediation-oriented human-robot interaction. 

---
# Billions of Sketches Reveal Hidden Cultural Variation in Human Concepts 

**Authors**: Arianna Pera, Mauro Martino, Nima Dehmamy, Douglas Guilbeault, Luca Maria Aiello, Andrea Baronchelli  

**Link**: [PDF](https://arxiv.org/pdf/2607.07267)  

**Abstract**: Claims about the universality of human concepts have been predominantly assessed through linguistic similarity across languages and cultures. However, words are effective as communication devices because they compress rich experiential variation into shared conventions, potentially obscuring hidden individual and cultural differences in how concepts are mentally represented. Here, we analyse 2.6 billion human-made sketches of common concepts from 236 countries and territories to examine conceptual structure through people's visual imagination. Consistent with recent work on image-based cognition, we find that single concepts unfold into multiple distinct visual exemplars, revealing latent information about similarities and differences in conceptual structure across cultures. This variation is strongest for concepts involving haptic interaction, suggesting that visual imagery reflects variation in embodied experience as much as conventional definitions. Comparing embedding models of sketches with word embedding models across languages, we find that their geometries diverge, with visual representations preserving rich semantic and cultural structure that language models compress. Cross-cultural similarities derived from sketches align 45% more closely with established cultural distances than do text-based measures. Together, these results suggest that patterns of human conceptual universality may depend critically on the modality through which concepts are measured, with large-scale sketching providing a direct, high-resolution probe of conceptual diversity across embodied and cultural dimensions of thought. 

---
# Recovering Latent Structures after Variational Bayesian Variable Selection: Fit Assessment and Factor-Number Selection in Partially Exploratory Factor Analysis 

**Authors**: Jinsong Chen, Yi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2607.07159)  

**Abstract**: In partially exploratory factor analysis (PEFA), the loading structure and factor numbers are weakly specified. The regularized variational approximation for partially confirmatory factor analysis (PCFA VA) recovers this structure via Bayesian variable selection, using spike and slab priors to assign inclusion probabilities to unspecified loadings. This research introduces a post selection assessment framework for this approach. We convert converged solutions into covariance models using either hard selection (thresholding probabilities into a sparse pattern) or soft selection (retaining them as weights for effective parameter counts). We derive the resulting degrees of freedom, absolute fit diagnostics (RMSEA, SRMR, CFI, TLI), and relative criteria (AIC, BIC, ELBO). To determine factor numbers, we propose a scale free gain rule with a sustained drop guard. Simulations show absolute indices successfully track loading recovery and flag under factoring. While raw criteria over factor, our gain rule accurately recovers true dimensionality, with the ELBO variant proving most robust. Finally, a 100 item PID 5 example demonstrates that our model fits better than a confirmatory 25 facet model and concordantly recovers major structures across disjoint specifications. 

---
# Dissociating the Internal Representations of Sycophancy in LLMs 

**Authors**: Anthony Baez, Sheer Karny, Pat Pataranutaporn  

**Link**: [PDF](https://arxiv.org/pdf/2607.07003)  

**Abstract**: Large Language Models (LLMs) frequently exhibit sycophancy, where they agree with a user's statement even when incorrect. While sycophancy is often treated as a single defined behavior, it can manifest in substantially distinct ways and circumstances, raising the question of whether this multi-faceted nature is reflected in its internal mechanisms. To address this gap, we dissociate the representations of sycophancy into factual and opinion subtypes -- motivated by the distinction between verifiable claims and subjective beliefs. We train linear probes and construct steering vectors on activations of one subtype and evaluate their transfer to the other subtype to measure to what extent they share representations. We find evidence that different LLMs represent these subtypes differently, with either more unified or more distinct and causally interfering representations. This method of dissociation offers a promising framework for studying the representational structure of complex model behaviors. 

---
# Large Language Models (LLMs) and Generative AI in Cybersecurity and Privacy: A Survey of Dual-Use Risks, AI-Generated Malware, Explainability, and Defensive Strategies 

**Authors**: Kiarash Ahi, Saeed Valizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2607.06963)  

**Abstract**: Large Language Models (LLMs) and generative AI (GenAI) systems, such as ChatGPT, Claude, Gemini, LLaMA, Copilot, Stable Diffusion by OpenAI, Anthropic, Google, Meta, Microsoft, Stability AI, respectively, are revolutionizing cybersecurity, enabling both automated defense and sophisticated attacks. These technologies power real-time threat detection, phishing defense, secure code generation, and vulnerability exploitation at unprecedented scales. Following a rapid surge where LLM-generated malware grew to account for an estimated 50% of detected threats by 2025, up from just 2% in 2021, navigating this highly automated threat landscape in 2026 demands next-generation security frameworks. This paper presents a comprehensive survey of the beneficial and malicious applications of LLMs in cybersecurity, including zero-day detection, DevSecOps, federated learning, synthetic content analysis, and explainable AI (XAI). Drawing on a review of over 70 academic papers, industry reports, and technical documents, this work synthesizes insights from real-world case studies across platforms like Google Play Protect, Microsoft Defender, Amazon Web Services (AWS), Apple App Store, OpenAI Plugin Stores, Hugging Face Spaces, and GitHub, alongside emerging initiatives like the SAFE Framework and AI-driven anomaly detection. We conclude with practical recommendations for responsible and transparent LLM deployment and trustworthy AI, including model watermarking, adversarial defense, and cross-industry collaboration, setting a new benchmark for rigorous, holistic cybersecurity research at the intersection of AI and threat defense, and offering a roadmap for secure, scalable LLM systems that serves as a critical reference for researchers, engineers, and security leaders navigating the complex challenges of AI-driven cybersecurity. 

---
# Geometric Self-Distillation for Reasoning Generalization 

**Authors**: Josip Jukić, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2607.06855)  

**Abstract**: On-policy distillation is a practical post-training recipe for large language models, supplying dense teacher supervision on the student's own trajectories. In privileged-context self-distillation, teacher and student are the same model conditioned on the same prefix, but the teacher also sees a hint or the full solution trace. This makes supervision abundant but harder to trust: the teacher can be confident about continuations its privileged view makes obvious but the student cannot yet justify. The distillation pull is strongest where teacher and student disagree most, and over many updates it accumulates into drift that degrades out-of-distribution (OOD) reasoning. We introduce GeoSD, a geometric self-distillation objective that treats this drift as movement in the student's predictive behavior and counters it in two complementary ways. A Hellinger loss scales each teacher preference by the overlap the student already shares with it, attenuating the pull on tokens the student cannot yet support. Since these pulls still compound over training, a proximal term penalizes how far the student's predictions drift from a recent checkpoint, measured as a Fisher-Rao distance. Both are distances in the same geometry of next-token distributions, and a natural-gradient update takes its steps in that geometry rather than in parameter space. Across mathematical reasoning benchmarks and three model families, GeoSD preserves the in-distribution gains of self-distillation while improving average OOD accuracy by 5.7-8.6 points over the base model, with gains holding across model scales from 1.7B to 32B. Analyzing why standard matching fails out of distribution, we find it wins agreement with the teacher by draining mass from alternatives at high-entropy states, resulting in confident agreement on wrong answers, whereas GeoSD keeps those alternatives in reach. 

---
# Trees from Marginals: Autoregressive drafting with factorized priors 

**Authors**: Yuma Oda, Ryan Mathieu, Roman Knyazhitskiy, Artur Chakhvadze  

**Link**: [PDF](https://arxiv.org/pdf/2607.06763)  

**Abstract**: Speculative decoding greatly increases the interactivity of autoregressive language models by trading off computation for extra tokens generated in a single forward pass. Factorized draft models are especially efficient because they predict future-token marginals in parallel, but their independence assumption causes acceptance rates to degrade sharply as the speculative budget grows. We analyze this limitation and introduce Weaver, a lightweight autoregressive adapter that constructs proposal trees from the top-K marginals of a factorized drafter. Weaver restores conditional dependencies between proposed tokens while avoiding a full-vocabulary projection. To support fast verification for models with Gated Delta Net layers, we derive a rollback-free tree-verification algorithm and implement optimized CUDA kernels in SGLang. By combining these model and systems contributions we achieve a 4.37-fold speedup over autoregressive decoding, and outperform a highly optimized DFlash baseline by 24.7%. 

---
# When Does In-Context Search Help? A Sampling-Complexity Theory of Reflection-Driven Reasoning 

**Authors**: Yotam Wolf, Noam Wies, Amnon Shashua  

**Link**: [PDF](https://arxiv.org/pdf/2607.06720)  

**Abstract**: Training large language models (LLMs) with extended reasoning has enabled in-context search, in which models iteratively generate, critique, and revise solution attempts. We provide a theoretical analysis of in-context search by modeling it as approximate inference over reasoning traces, where the base model defines a prior and self-reflection provides feedback for posterior updates, and study the resulting inference-time sampling complexity - the number of sequential attempts needed to achieve high success probability. We show that when reflections reliably localize early mistakes, in-context search can yield exponential improvements over the base model, solving problems with exponentially small zero-shot pass rates using only a polynomial number of sequential attempts, whereas when this property fails, conditioning on past attempts offers no asymptotic benefit over parallel sampling. We further show that these gains are robust and learnable: approximate posterior updates suffice, and cross-entropy training on search rollouts recovers the required behavior with polynomial sample complexity. Finally, we show that under a stagewise abstraction of reinforcement learning with verifiable rewards, the optimal policy extension implements the same posterior reweighting rule. We validate key qualitative predictions of the theory on real large reasoning models. 

---
# Final Checkpoints Are Not Enough: Analyzing Latent Reasoning Faithfulness Along Training Trajectories 

**Authors**: Hengyu Jin, Shu Yang, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.06648)  

**Abstract**: Latent reasoning methods perform multi-step inference entirely in the model's continuous hidden states, promising more compact and efficient reasoning. However, these opaque hidden states raise a question of faithfulness: whether these latent reasoning steps causally drive the final answer. Prior work investigates this question at converged checkpoints and reports several unfaithful behaviors, such as latent reasoning steps that can be replaced without changing the answer, but leaves how these behaviors form during training unexamined. We instead track how faithfulness evolves across saved checkpoints for different latent reasoning paradigms, applying a verifiable counterfactual edit on the input and a noise-ablation activation patch on the latent reasoning steps. We find that (i) at the output level, latent reasoning methods can look similarly unfaithful at convergence under counterfactual edits while following qualitatively divergent trajectories; (ii) at the activation level, the causal contribution of latent reasoning steps to the final answer decays across training for both paradigms, with the examples that flip on the output side in (i) also being the examples on which this contribution decays; and (iii) the activation-level trajectory diverges by answer format, decaying on binary choice and rising on open-ended decoding. These findings highlight that latent reasoning faithfulness depends on training stage and answer format. 

---
# Reconfigurable Radiology Labels Without Relabeling 

**Authors**: Jean-Benoit Delbrouck, Dave Van Veen, Akash Pattnaik, Kalina Slavkova, Javid Abderezaei, Harris Bergman, Khan Siddiqui  

**Link**: [PDF](https://arxiv.org/pdf/2607.06597)  

**Abstract**: Public chest-radiograph (CXR) datasets are typically released with small, fixed label schemas such as CheXpert-14. However, the underlying free-text reports describe far more findings -- and which findings matter depends on the task, site, and reader. We release a pipeline that converts free-text reports into multi-label matrices and then reconfigures the label schema through dictionary edits rather than new inference passes, i.e., without relabeling the corpus. After this one-time pass, reconfiguring MIMIC-CXR (223K reports) from cached annotations takes 196 seconds with no API cost, compared to \$6.6K for an equivalent relabeling pass with Claude Opus 4.7. Using a 58-label taxonomy, we show that 43\% of CXR studies contain at least one finding outside CheXpert-14. Image probes trained on these labels match CheXpert-14 probes on shared targets while also reaching 0.78 AUROC on expert-reviewed long-tail labels that CheXpert-14 cannot represent. These results suggest a different unit of work for radiology labeling: once reports are structured, the label schema becomes a configuration to edit, not a corpus to relabel. 

---
