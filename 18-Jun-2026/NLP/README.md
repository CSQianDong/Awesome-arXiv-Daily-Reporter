# Learning User Simulators with Turing Rewards 

**Authors**: Yingshan Susan Wang, Cedegao E. Zhang, Linlu Qiu, Zexue He, Pengyuan Li, Alex Pentland, Roger P. Levy, Yoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.19336)  

**Abstract**: Learning to simulate human users in interactive settings could advance the training of agent assistants, evaluation of personalization systems, research in the social sciences, and more. Existing approaches generally do so by training a large language model (LLM) to match a single ground truth response, either by maximizing the log probability or by using a similarity reward. We instead propose {Turing-RL}: a Turing-Test-based reinforcement learning approach for training user simulator models. {Turing-RL} uses a discriminative Turing reward with an LLM judge to score how indistinguishable a generated response is from the real user's given the user's history, and the user simulator LLM learns to produce responses indistinguishable from what the user could have said with such rewards. Across two different domains--conversational chat and Reddit forum discussion--we find that {Turing-RL} consistently outperforms baseline methods on both LLM and human evaluation metrics. Our study suggests that optimizing for indistinguishability, rather than response matching, is effective for learning user simulators. 

---
# Freeing the Law with LOCUS: A Local Ordinance Corpus for the United States 

**Authors**: Denis Peskoff, Joe Barrow, Christopher Vu, Diag Davenport  

**Link**: [PDF](https://arxiv.org/pdf/2606.19334)  

**Abstract**: Progress in legal AI increasingly depends on access to authoritative legal text at scale. Yet one of the most consequential layers of American law remains largely absent from existing machine-readable corpora: local ordinances. Local codes govern zoning, housing, business licensing, public health, noise, animal control, and many other domains of everyday regulation, but they are fragmented across vendor platforms designed for human browsing rather than bulk research access. We introduce LOCUS - the Local Ordinance Corpus for the United States - a comprehensive corpus and county-harmonized access layer for U.S. municipal and county ordinance codes. The raw corpus, available for release to researchers, represents nearly all publicly available municipal and county ordinance codes. The resulting raw corpus contains codes from 9,239 cities and counties. A smaller county-harmonized LOCUS access layer provides coverage for the largest 2,309 of 3,144 U.S. counties, accounting for a majority of the population. We use OCR to handle the myriad of document formats that have kept the law from being a public resource. We release the corpus with coverage metadata to support reproducibility, downstream legal AI research, and the incremental expansion of machine-readable access to local law. We train a collection of ModernBERT-based classifiers and scorers to facilitate analyzing U.S. local law among several dimensions, such as opacity and paternalism, that have not previously been studied at this scale. LOCUS-v1 and its derivative models are available at: this https URL 

---
# Enhancing Decision-Making with Large Language Models through Multi-Agent Fictitious Play 

**Authors**: Leyang Shen, Yang Zhang, Xiaoyan Zhao, Chun Kai Ling, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2606.19308)  

**Abstract**: Large language model (LLM)-based multi-agent systems (MAS) have demonstrated great potential in solving tasks with execution complexity, by distributing subtasks across cooperative agents. However, this divide-and-conquer paradigm falls short on decision-making tasks that are also prevalent in the real world. These tasks require simultaneous reasoning from the stances of all involved stakeholders whose decisions are mutually dependent and thus cannot be solved in isolation. We characterize this challenge as stance entanglement, a form of decision complexity distinct from execution complexity. To address it, we propose Multi-Agent Fictitious Play (MAFP), a novel MAS paradigm that represents stakeholder stances as agents and formulates decision-making as an equilibrium-seeking process. Built on the game-theoretic principle of fictitious play, MAFP iteratively updates each agent's decision by best responding to the empirical mixture of other agents' past decisions. This enables agents to expose and address one another's weaknesses, progressively improving decision quality and robustness. We evaluate MAFP on challenging decision-making tasks that test the capability of deciding strategies for competitive scenarios prior to acting. MAFP outperforms both single-round and multi-round baselines on two complementary metrics, tournament strength and robustness, demonstrating its effectiveness in addressing stance entanglement. 

---
# Trade-offs in Medical LLM Adaptation: An Empirical Study in French QA 

**Authors**: Ikram Belmadani, Oumaima El Khettari, Carlos Ramisch, Frederic Bechet, Richard Dufour, Benoit Favre  

**Link**: [PDF](https://arxiv.org/pdf/2606.19266)  

**Abstract**: The development of large language models (LLMs) has led to an increased focus on their adaptation to specialized domains and languages, yet the effectiveness of domain adaptation strategies remains unclear. We present a study of medical domain adaptation using French medical question-answering (QA) as a case study. We compare continual pretraining (CPT), supervised fine-tuning (SFT), and their combination across three model families, multiple sizes, and three initialization types, explicitly disentangling adaptation effects from base model choice. We evaluate both multiple-choice (MCQA) and open-ended QA (OEQA) under greedy and constrained decoding using automatic metrics and LLM-as-a-Judge evaluation. For MCQA, CPT+SFT most often achieves the best scores, but gains over SFT are small and frequently not statistically significant, making SFT a strong and cost-effective default. For OEQA, CPT consistently improves overlap-based metrics, while SFT often degrades generation quality; instruction tuning and CPT+SFT are preferred by LLM-based evaluation. Cross-lingual experiments further show effective transfer from French adaptation to English benchmarks. Overall, we provide practical guidelines for selecting adaptation strategies under computational constraints. 

---
# DreamReasoner-8B: Block-Size Curriculum Learning for Diffusion Reasoning Models 

**Authors**: Zirui Wu, Lin Zheng, Jiacheng Ye, Shansan Gong, Xueliang Zhao, Yansong Feng, Wei Bi, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2606.19257)  

**Abstract**: Block diffusion language models accelerate decoding through parallel block-wise denoising, yet whether they can be reliably scaled for long chain-of-thought (CoT) reasoning remains unresolved. To this end, we develop DreamReasoner-8B, an open-source block diffusion reasoning model, and conduct a systematic study of how training and inference block sizes affect long-CoT reasoning. Our analysis reveals a stark performance disparity: training with large block sizes yields remarkably poor reasoning, whereas small block sizes preserve effective reasoning. To bridge this granularity gap, we propose block-size curriculum learning, which gradually transitions training from fine-grained to coarse-grained block sizes, thereby overcoming this limitation and enabling strong reasoning performance that generalizes across diverse inference block sizes. On mathematical and code reasoning benchmarks, DreamReasoner-8B achieves results competitive with leading open autoregressive models such as Qwen3-8B. This work establishes a practical foundation for efficient, reasoning-capable diffusion language models. We release our model at this https URL. 

---
# RECOM: A Validity Discrimination Tradeoff in Automatic Metrics for Open Ended Reddit Question Answering 

**Authors**: Pushwitha Krishnappa, Amit Das, Vinija Jain, Aman Chadha, Tathagata Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.19218)  

**Abstract**: Automatic metrics are the default for evaluating LLM-generated text, yet a metric is quietly asked to do two jobs: tell genuine content alignment from surface coincidence (validity), and tell a better system from a worse one (discriminative power). On open-ended, opinion-driven question answering, the two are in tension. We introduce RECOM (Reddit Evaluation for Correspondence of Models), a contamination-free evaluation dataset of 15,000 r/AskReddit questions (September 2025), each paired with its authentic community replies, which postdate every evaluated model's training cutoff. Scoring five open-source LLMs (7--10B) against every reply each metric paired with a random-derangement noise floor we find that no metric does both jobs well. Cosine similarity separates real from random answers (Cohen's $d \approx 2$) but cannot rank the five models ($|d| < 0.1$); BERTScore precision appears to rank the models (raw $|d|$ up to 0.63), but once response length is controlled this collapses to $|d| = 0.09$ and its validity is weak ($d \approx 0.8$, versus cosine's $\approx 2$). Because every metric scores the same outputs, this validity--discrimination tradeoff is a property of the metrics, not the models, and we argue it stems from representation design. Three independent LLM judges reproduce the validity gap and likewise separate the five models only weakly. We recommend reporting metrics on both axes, with an explicit random-baseline floor. RECOM is publicly available at this https URL 

---
# Language Models as Interfaces, Not Oracles: A Hybrid LLM-ML System for Pediatric Appendicitis 

**Authors**: Soheyl Bateni, Maryam Abdolali  

**Link**: [PDF](https://arxiv.org/pdf/2606.19183)  

**Abstract**: Large language models (LLMs) can make clinical decision support more accessible by interpreting free-text documentation, but their direct use as diagnostic engines is limited by sensitivity to prompts, information order, and plausible but incorrect outputs. Structured machine-learning models offer more stable risk prediction, yet they require tabular inputs that are difficult to integrate with narrative clinical workflows. We present ClaMPAPP (Clinical Language-assisted Machine-learning Pipeline for Appendicitis), a hybrid system that uses an LLM as an interface rather than as the final decision-maker. ClaMPAPP extracts schema-constrained clinical features from note-like narratives, applies deterministic plausibility checks, and passes validated features to an XGBoost classifier trained on clinical, laboratory, and ultrasound variables. We evaluated ClaMPAPP on two independent pediatric appendicitis cohorts from German hospitals and compared it with end-to-end LLM baselines, including open-source and proprietary models. To preserve ground truth while testing free-text input, narratives were generated from structured electronic health records through template rendering and constrained LLM rewriting, with additional sentence-order permutation to assess positional robustness. ClaMPAPP achieved the strongest overall diagnostic performance in both internal and external validation while minimizing missed appendicitis cases, the key safety concern in acute triage. End-to-end LLMs showed unstable sensitivity-specificity trade-offs and greater degradation under narrative reordering. These results support an LLM-as-interface, ML-as-predictor design that separates natural-language usability from predictive inference and provides a more auditable pathway for clinical decision support. 

---
# Dango: A Strictly L1-Only Large Language Model for Studying Second Language Acquisition 

**Authors**: Shiho Matta, Yin Jou Huang, Fei Cheng, Takashi Kodama, Hirokazu Kiyomaru, Yugo Murawaki  

**Link**: [PDF](https://arxiv.org/pdf/2606.19170)  

**Abstract**: We introduce Dango, a 1.8B-parameter large language model designed for controlled studies of L1-to-L2 (Japanese-to-English) transfer in second language acquisition (SLA). While previous studies have explored SLA in language models, they have predominantly relied on smaller or non-decoder models, limiting their ability to generate open-ended text and reducing their suitability as practical L2 simulators. We identify a key challenge when scaling models to this size: L2 contamination within the "monolingual" pretraining corpus used for L1 acquisition. To address this, we propose a filtering method to reduce premature exposure to English while preserving realistic, minimal exposure. We then fine-tune the model on LLM-generated L2-learning lessons to simulate the L2 acquisition process. Our evaluations confirm that Dango develops human-like L2 production patterns, outperforming both unfiltered and standard multilingual baselines. We release the model, data, and code to facilitate reproducible computational SLA research and learner-facing applications. 

---
# Leadership as Coordination Control: Behavioral Signatures and the Recovery-Advantage Boundary in Multi-Agent LLM Teams 

**Authors**: Haewoon Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2606.19111)  

**Abstract**: Team science holds that leadership is contingent: it helps only under specific conditions, and capable, autonomous teams may need none at all. We ask the analogous question for multi-agent LLM teams: under what measurable conditions does process-level coordination control add value, and do those conditions match what team science predicts? We use behavioral signatures (majority lock-in, exploration, recovery from an incorrect round-0 consensus) and per-action ablations, clean because each controller is an explicit action set, not a monolithic prompt. We operationalize three classical leadership styles (transactional, transformational, situational) as controllers over a shared action vocabulary (explore, revise, accept, synthesize). A matched controller with the same actions but an arbitrary rule recovers no better than majority voting, so the theory-derived rule, not the vocabulary, does the work. Across four task regimes and three open-weight model families, no controller dominates by accuracy, as the contingency view predicts: transactional control matches a shared round-0 vote on all 12 (model, regime) combinations to within 1.3pp, and gains appear only on the one combination where the round-0 majority is unreliable (llama-4-scout social; situational +8pp over flat). A recovery-advantage account, tested with four boundary probes, says a controller beats plain interaction only where the round-0 majority is unreliable, the task is recoverable, and undirected interaction does not already repair it. These regions map onto contingency theory (leadership substitutes, path-goal redundancy, the situational readiness gap), so a largely null accuracy result is what the theory predicts, not a failure of the controllers. We read process-level coordination control as a contingency to be measured and theory-mapped, not a leaderboard to be topped. 

---
# Which Sections of a Research Paper Best Reveal Its Research Methods? Evidence from Library and Information Science 

**Authors**: Qiuyu Fang, Jiayi Hao, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.19051)  

**Abstract**: Research methods are essential carriers of knowledge contribution in academic papers. Automatic multi-label classification of research methods can support knowledge services such as method retrieval, review generation, and research intelligence analysis. While existing studies primarily rely on titles and abstracts, abstracts often provide only limited methodological information, whereas utilizing full-text content faces challenges related to excessive length and information redundancy. Therefore, this paper proposes a segment combination strategy by partitioning the full-text content according to its physical postion. Using an annotated corpus of 1,954 full-text articles from three representative journals in Library and Information Science (JASIST, LISR, and JDoc), we evaluate the classification performance of various segments and their combinations across multiple models. Experimental results indicate that methodological information is distributed unevenly within the full-text content, with the middle-to-late and final segments exhibiting greater discriminative power. Furthermore, integrating bibliographic metadata with cross-segment combination strategies effectively enhances classification performance. 

---
# Sumi: Open Uniform Diffusion Language Model from Scratch 

**Authors**: Mengyu Ye, Keito Kudo, Wataru Ikeda, Ryosuke Matsuda, Keisuke Sakaguchi, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2606.19005)  

**Abstract**: Diffusion models have become a promising alternative to autoregressive models. Among these, uniform diffusion language models (UDLMs) permit any token to be updated at any step, in principle enabling more flexible generation. However, no UDLM has yet been pretrained from scratch at both large parameter scale and large token budget. Both autoregressive modeling and masked diffusion modeling already have capable models at scale that the community can study and build on; uniform diffusion has none. A scratch-pretrained UDLM at scale would provide a clean reference point for studying scaling behavior, generation dynamics, controllability, and trade-offs against established autoregressive and masked diffusion models. To this end, we introduce Sumi ("ink" in Japanese), a fully open 7B uniform diffusion language model pretrained from scratch on 1.5T tokens. Sumi performs competitively with autoregressive models trained at comparable token budgets on knowledge, reasoning, and coding benchmarks, while under-performing on commonsense benchmarks, where our education-heavy data mixture is a likely contributor. We release our model weights, checkpoints, and full training recipe, including a complete specification of the data mixture over publicly available corpora. We hope this release enables the community to study native uniform diffusion at scale and catalyzes work on its as-yet poorly understood aspects. 

---
# Enhancing Multilingual Reasoning via Steerable Model Merging 

**Authors**: Zhuoran Li, Rui Xu, Jian Yang, Junnan Liu, Zhijun Chen, Qianren Mao, Hongcheng Guo, Jiaheng Liu, Likang Xiao, Ming Li, Xiaojie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.19002)  

**Abstract**: Model merging is an effective technique for composing the capabilities of a multilingual model and a reasoning model. It has achieved promising generalization in multilingual reasoning tasks by aligning feature spaces of different models. However, the merged single model often fails to address the conflicts between source models, leading to suboptimal performance. In other words, the one-size-fits-all merging strategy may not align with the characteristics of different inputs which may require prioritizing certain models over others. To this end, we propose a Steerable Model Merging (ST-Merge) framework to modulate the contribution of each source model. To realize this idea, we introduce a gated cross-attention mechanism to weight or filter the two attended source models in an adaptive manner. Extensive experiments demonstrate that ST-Merge consistently outperforms multiple strong baselines on four multilingual reasoning benchmarks across 21 different languages. 

---
# G-IdiomAlign: A Gloss-Pivoted Benchmark for Cross-Lingual Idiom Alignment 

**Authors**: Fengying Ye, Yanming Sun, Runzhe Zhan, Zheqi Zhang, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2606.18989)  

**Abstract**: Idioms are difficult to transfer across languages due to their non-compositionality and weak surface-form grounding, making literal mappings unreliable. We present G-IdiomAlign, a gloss-pivoted benchmark where each idiom is anchored by an English gloss from Wiktionary. We further construct a high-confidence reference alignment set for reproducible evaluation. G-IdiomAlign supports two protocols: (1) a controlled Multiple-Choice Idiom Equivalence with typed distractors for error attribution; and (2) a Gloss-Contrastive Generation contrasting No-gloss and With-gloss inputs to isolate the effect of an explicit semantic pivot. Across diverse LLMs, a bias to literal translation is a dominant failure mode, especially when the target is a low-resource language. Glosses consistently improve Gloss-Contrastive Generation under an embedding-based semantic proxy, but performance remains modest, indicating substantial headroom in the open output space. Subsequent analysis on Qwen3-8B further suggests that cross-condition differences are concentrated more in attention heads than in layers, while better With-gloss generations coincide with stronger gloss anchoring. 

---
# Beyond Tokenization: Direct Timestep Embedding and Contrastive Alignment for Time-Series Question Answering 

**Authors**: Yafeng Wu, Huu Hiep Nguyen, Thin Nguyen, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2606.18986)  

**Abstract**: Recent advances in large language models (LLMs) have given rise to time-series question answering (TSQA), which formulates time-series analysis as natural-language question answering. However, directly feeding raw numerical series into LLMs suffers from a tokenization bottleneck: Byte Pair Encoding fragments continuous values into unstable tokens whose embeddings lack meaningful metric structure, resulting in the loss of magnitude, scale, and trend information. Prior methods use patch-based encoders that split the series into fixed windows, locking in one granularity that breaks patterns and hides exact timesteps, through a separate module that rarely transfers across datasets with different lengths or sampling rates. To address this challenge, we propose CADE (Contrastive Alignment with Direct Embedding), a novel framework for TSQA built upon two key components: direct timestep embedding and semantic alignment. The proposed framework maps each timestep directly into the LLM embedding space through a point-wise linear encoder and MLP projector, preserving exact index-level access while eliminating the need for patching and padding. To further bridge the semantic gap between time-series and language representations, we introduce a novel one-directional supervised contrastive loss that aligns time-series embeddings with frozen class-name text anchors. Experimental results on the public Time-MQA benchmark demonstrate that our framework consistently improves performance across six TSQA tasks, outperforming both open-source and proprietary LLM baselines. 

---
# GraphPO: Graph-based Policy Optimization for Reasoning Models 

**Authors**: Yuliang Zhan, Xinyu Tang, Jian Li, Dandan Zheng, Weilong Chai, Jingdong Chen, Jun Zhou, Ge Wu, Wenyue Tang, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.18954)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a standard paradigm for enhancing the capability of large reasoning models. RLVR typically samples responses independently and optimizes the policy using from final answers. This paradigm has two limitations. First, independently responses often contain similar intermediate reasoning steps, causing redundant exploration and wasted computation. Second, sparse final-answer rewards make it hard to identify useful steps. Tree-based methods partly address this problem by sharing prefixes and comparing branches from the same prefix to provide fine-grained signals. However, tree branches are still expanded independently. When different branches reach similar reasoning states, they cannot share information and repeat similar exploration. Moreover, tree-based methods ignore such dispersion and only perform local comparisons within separate branches, which can lead to higher variance in advantage estimation. To address this challenge, we propose GraphPO (Graph-based Policy Optimization), a novel RL framework that represents rollouts as a directed acyclic graph, with reasoning steps as edges and semantic states summarized from the reasoning paths as nodes. GraphPO merges semantically equivalent reasoning paths into equivalence classes, allowing them to share suffixes and reallocating budget away from redundant expansions to diverse exploration. Furthermore, we assign efficiency advantages to incoming edges and correctness advantages to outgoing edges, thereby improving inference efficiency while deriving process supervision from outcome. Theory shows that GraphPO reduces advantage-estimation variance and enhances reasoning efficiency. Experiments on three LLMs across reasoning and agentic search benchmarks show that GraphPO consistently outperforms chain- and tree-based baselines with the same token budgets or response budgets. 

---
# SenFlow: Inter-Sentence Flow Modeling for AI-Generated Text Detection in Hybrid Documents 

**Authors**: Jingkun Luo, Yifan Sun, Da-Tian Peng, Guanxiong Pei  

**Link**: [PDF](https://arxiv.org/pdf/2606.18946)  

**Abstract**: Sentence-level AI-generated text detection (S-AGTD) for hybrid documents, where humans and LLMs co-author one text, faces two gaps: existing methods classify each sentence in isolation, discarding inter-sentence dependencies, and existing benchmarks omit the newest generation of generators. We construct MOSAIC, a benchmark of 16,000 hybrid documents over PubMed and XSum, generated by DeepSeek-V3.2 and Kimi K2 under stringent quality controls including a perplexity-consistency filter absent from prior benchmarks. We recast S-AGTD as structured prediction over the document sentence sequence and instantiate it as SenFlow, integrating graph-based inter-sentence propagation with linear-chain CRF decoding in a single document-level pass over a sentence graph. SenFlow reaches state-of-the-art performance on MOSAIC, with a +4.15 pp average Macro-F1 margin on cross-domain transfer, the hardest of three protocols of increasing difficulty. We further find that even after the perplexity filter equalizes overt cues, AI insertions retain a generator-dependent sentence-length gap that sentence-level detectors still exploit. Code and data: this https URL 

---
# As Easy as Rocket Science: Assessing the Ability of Large Language Models to Interpret Negation in Figurative Language 

**Authors**: Jasmine Owers, Edwin Simpson, Martha Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2606.18922)  

**Abstract**: Figurative language and negation are two areas that challenge current language models, however, both are widely used throughout written and spoken language. Large language models (LLMs) are also widely used in everyday contexts where they cannot necessarily be tuned for a specific dataset. It is therefore essential to understand the ability of LLMs to correctly interpret text that includes both negation and figurative language. To investigate this, we develop a set of new annotations to an existing dataset of figurative language, and test a range of language models on the dataset. We find that the combination of negation and figurativeness can present a particular challenge, and that performance overall and across different negation types is particularly dependent on the prompt style used. 

---
# SAGE: Stochastic Prompt Optimization via Agent-Guided Exploration 

**Authors**: Ziyi Zhu, Luka Smyth, Saki Shinoda, Jinghong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.18902)  

**Abstract**: Context engineering has emerged as a primary lever for improving AI systems without parameter updates. Recent work showing that textual gradients do not function as real gradients motivates treating automatic prompt optimization (APO) as black-box search. We introduce SPO (Stochastic Prompt Optimization), a framework for stochastic search over prompt space, and compare three strategies of increasing sophistication: error-informed random search, a genetic algorithm with evolutionary operators, and SAGE (SPO via Agent-Guided Exploration), a multi-agent pipeline with diagnostic code execution. Across three benchmarks, no single strategy dominates; effectiveness depends on the interaction of landscape structure with error type. We further deploy SAGE on a mental-health chatbot under a continuous optimization paradigm, where it compounds eight cycles of individually-noisy A/B tests into a statistically robust gain in next-day retention. We argue that coupling qualitative diagnosis with quantitative validation is what makes agentic optimization effective for open-ended task-oriented dialogue. 

---
# Learning Robust Pair Confidence for Multimodal Emotion-Cause Pair Extraction 

**Authors**: Zhuangzhuang Pan, Ning Dong, Yingna Su, Yan Xia  

**Link**: [PDF](https://arxiv.org/pdf/2606.18893)  

**Abstract**: Multimodal emotion-cause pair extraction (MECPE) requires reliable pair confidence over candidate pairs. Existing pair scorers commonly use pair-level cross entropy over valid candidates, which treats links mostly independently. This leaves the relative confidence geometry among competing causes under-constrained, allowing gold pairs to stay close to hard negatives or rely on incidental non-gold context. We study this vulnerability as pair-confidence brittleness and propose RPCL (Robust Pair Confidence Learning), a training-only framework for pair-confidence learning. RPCL encourages pair confidence to be both discriminative and stable: gold pairs are separated from row-wise hard negatives through a confidence-difference margin constraint, and clean pair predictions are aligned with predictions from a corrupted view where non-gold contextual utterance representations are partially corrupted. The original clean pair scorer and decoding pipeline are used unchanged at inference time. On ECF, MECAD, and MEC4, RPCL improves the three-seed mean Pair F1 over a matched base model by 2.58 to 2.83 percentage points in the full text-audio-video setting, and improves mean Pair AUPRC on all three datasets. Diagnostic analysis further shows larger gold-negative confidence gaps and lower margin-violation severity. These results suggest that explicitly shaping pair confidence is an effective training strategy for MECPE. 

---
# Improving Medical Communication using Rubric-Guided Counterfactual Recommendations 

**Authors**: Adrian Cosma, Nicoleta-Nina Basoc, Andrei Niculae, Cosmin Dumitrache, Emilian Radoi  

**Link**: [PDF](https://arxiv.org/pdf/2606.18889)  

**Abstract**: Text-based telemedicine increasingly relies on lightweight patient feedback, however, such feedback primarily reflects perceived communication quality rather than medical accuracy. We introduce an LM-guided counterfactual recommendation pipeline that discovers and refines interpretable communication features such as tone, personalization, actionability and completeness in addressing patient concerns, without interfering with the medical content. These features are used together with patient-doctor interaction metadata to estimate positive feedback. At inference time, the system searches over low-cost ordinal feature changes and recommends minimal communication changes predicted to increase the probability of positive feedback, while independent auditor models test whether these gains generalize beyond the selection model. Across interactions, recommendations yield a mean +6.41% gain in predicted positive feedback probability under independent auditors, and are non-negative for 93.31% of recommendations. These results suggest that small, interpretable communication changes can capture most predicted gains while preserving the doctor's control over medical reasoning and final wording. 

---
# Efficient Financial Language Understanding via Distillation with Synthetic Data 

**Authors**: Wen-Fong, Huang, Edwin Simpson  

**Link**: [PDF](https://arxiv.org/pdf/2606.18875)  

**Abstract**: Large instruction-following models are powerful but costly to deploy, particularly in finance, where labelled data are limited by confidentiality and expert annotation cost. We present an efficient framework for financial sentiment analysis through distillation with synthetic data, transferring knowledge from a large instruction-tuned teacher to compact student models. The framework is designed for low-resource conditions, where a small set of real examples are collected and labelled by hand. The framework then clusters the examples and uses the clusters to select seeds for generating synthetic examples via structured few-shot prompting. Experiments show that clustering-based seed selection yields more representative synthetic data than random sampling, enabling compact models to achieve strong performance with minimal supervision. Notably, on a more complex and noisy text domain, the compact model trained on the complete synthetic-seed corpus even outperforms the teacher model, while remaining competitive on formal text. The framework provides a practical route toward resource-efficient domain adaptation in financial NLP with minimal human labelling effort. 

---
# Approximate Structured Diffusion for Sequence Labelling 

**Authors**: Nicolas Floquet, Joseph Le Roux, Nadi Tomeh  

**Link**: [PDF](https://arxiv.org/pdf/2606.18856)  

**Abstract**: Sequence labelling, a core task of Natural Language Processing (NLP), consists in assigning each token of an input sentence a label.
From a Machine Learning point of view, sequence labelling is often cast as a Linear-Chain Conditional Random Field (CRF) parametrised by a neural network.
While this approach gives good empirical results, CRFs assume a finite decision span (eg label bigrams) which can limit their expressivity and hurt performance when long-range dependencies are required.
We show we can leverage diffusion to train a CRF conditioned on an entire label sequence, with the caveat that the condition is on a noisy version of labels.
We show experimentally that this method, in conjunction with approximate CRF inference, improves label accuracy with a 16.5% error reduction for POS-tagging. 

---
# Aligning Implied Statements for Implicit Hate Speech Generalizability with Context-Bounded Semi-hard Negative Mining 

**Authors**: Wicaksono Leksono Muhamad, Yunita Sari  

**Link**: [PDF](https://arxiv.org/pdf/2606.18852)  

**Abstract**: Classifying implicit hate speech remains a challenge, as intent is often masked through insinuation and context rather than explicit slurs. Prior supervised contrastive approaches improve in-domain detection but can overfit surface cues and struggle to transfer across datasets. We propose ImpSH, a triplet-based framework that aligns posts with implied statements when available and uses context-bounded semi-hard negatives to focus learning on near confusions. We also examine AugSH, which forms positives via data augmentation. In controlled evaluations on IHC, SBIC, and DynaHate with BERT and HateBERT, ImpSH is a viable alternative to standard supervised contrastive baselines and often improves cross-domain performance under matched preprocessing and tuning budgets. Representation analysis using alignment and uniformity indicates tighter positive pairs with balanced global spread, and qualitative nearest-neighbor case studies illustrate typical false negatives under domain shift. These results demonstrate that aligning posts with their implied statements via context-bounded mining provides a more stable, bijective-like mapping to related insinuations, overcoming the volatility inherent in traditional clustering-based representation learning. 

---
# ScholarSum: Student-Teacher Abstractive Summarization via Knowledge Graph Reasoning and Reflective Refinement 

**Authors**: Bohou Zhang, Xiaoyu Tao, Mingyue Cheng, Huijie Liu, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.18850)  

**Abstract**: Abstractive summarization plays a crucial role in enabling efficient understanding of scientific literature, yet it inherently demands both linguistic fluency and factual faithfulness. Existing approaches often fail to reconcile these two requirements. Extractive methods rely on rigid sentence splicing that disrupts macro-level logical coherence, while large language model (LLM)-based generative approaches, despite mastering linguistic fluency, exhibit limited factual consistency. In this work, we propose ScholarSum, a hierarchical reflective graph-based framework that emulates a student-teacher writing process for fluent and faithful scientific summarization. ScholarSum first organizes the document into a hierarchical knowledge graph by segmenting it into semantically coherent units, whose multi-layered community structure captures global logic and macro-level themes. Guided by this global structure, the student generates an initial draft, which is subsequently refined through fine-grained evidence retrieval. To ensure factual consistency, a teacher-like reviewer then iteratively examines the draft, identifies unsupported content, and prompts targeted re-retrieval and rewriting until the summary meets rigorous quality standards. Extensive experiments demonstrate that ScholarSum significantly outperforms previous baselines in terms of both completeness and faithfulness. Our code is available at this https URL. 

---
# Beyond Reward Engineering: A Data Recipe for Long-Context Reinforcement Learning 

**Authors**: Xiaoyue Xu, Sikui Zhang, Xiaorong Wang, Xu Han, Chaojun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2606.18831)  

**Abstract**: Long-context reasoning is an essential capability for large language models, particularly when they are deployed as autonomous agents that must reason over lengthy trajectories. Reinforcement learning (RL) has recently emerged as a dominant paradigm for improving this ability, yet existing work largely focuses on reward engineering while diverse training data remains scarce. We revisit this problem from a data-centric perspective and show that a simple yet effective data recipe alone, paired with a minimal outcome-based GRPO setup, suffices to substantially improve long-context reasoning. Our recipe targets three complementary task families -- retrieval, multi-evidence synthesis, and reasoning -- for which we construct and curate eight datasets totaling ~14K examples. Experiments on three models (Qwen3-4B/8B/30B-A3B) yield average gains of +7.2/+3.2/+6.4 points across seven long-context benchmarks, surpassing prior RL training sets. We further demonstrate that these gains transfer to agentic tasks, where continuing RL training on an agent-tuned model with our data recipe improves GAIA by +4.8 and BrowseComp by +7.0 points. We will release our datasets to facilitate future research. 

---
# Beyond Scalar Scores: Exploring LLM-based Metrics for Clinical Significance Evaluation in Radiology Reports 

**Authors**: Qingyu Lu, Ruochen Li, Liang Ding, Yufei Xia, Youxiang Zhu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2606.18797)  

**Abstract**: Reliable evaluation of generated radiology reports requires strict clinical accuracy, as omitted critical findings or mischaracterized radiographic observations can directly affect patient care. Existing metrics obscure this requirement by reducing report quality to a medically ungrounded scalar. Although Large Language Models (LLMs) possess rich medical knowledge, they likewise struggle to draw a reliable boundary between clinically significant errors and harmless variation. We study this boundary using ReEvalMed benchmark as testbed and evaluate metric-level clinical significance from detecting true clinical errors ("Discrimination") and tolerating insignificant variations ("Robustness"). Across 8 LLM evaluators under one-pass and two-pass settings, we identify a widespread discrimination bias: models effectively detect errors but also over-penalize harmless rephrasings. To mitigate this, we synthesize 4k report pairs and train lightweight interpretable metrics on Qwen3-8B and MedGemma-4B. Our trained metric sharpens the clinical significance boundary, surpassing 32B-scale medical LLMs and remaining competitive with proprietary models. Crucially, the more costly two-pass setting fails to consistently improve overall performance and mainly trades discrimination for robustness. These findings suggest one-pass trained metrics as the practical choice for cost-sensitive deployment, with two-pass inference reserved for settings where D-R balance is critical. We will release the dataset and metric. 

---
# RedactionBench 

**Authors**: Sean Brynjólfsson, Shashvat Jayakrishnan, Esha Sali, Diptanshu Purwar, Madhav Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2606.18782)  

**Abstract**: Large Language Models are increasingly applied to sensitive domains that require redaction of personally identifiable information (PII). While redacting PII is a data cleaning prerequisite, existing benchmarks conflate extraction mechanics with privacy semantics. A public phone number is not equivalent to a phone number in a medical record. Whether information constitutes a violation depends heavily on who holds it, why, and in what context, fundamentally differentiating redaction from simple entity recognition. Grounded in contextual integrity, we introduce RedactionBench, a manually annotated benchmark comprising 200 diverse documents across 11 domains, mostly seeded from real-world sources. We also introduce R-Score, a novel character-level metric that treats semantically similar redactions equally and nullifies shallow formatting choices, such as varying masking styles for phone numbers. Evaluations across Named Entity Recognition models, entity extraction Small Language Models, and frontier models equipped with agentic tools demonstrate that contextual redaction remains an unsolved problem. A human evaluation with over 80 users on RedactionBench reveals a stark dichotomy in privacy perceptions. Annotators show consensus with target labels for mandatory redactions (89.4 percent) and safe text preservations (94.1 percent), but fail to agree on contextual redactions (47.7 percent). This variance demonstrates the subjective nature of contextual privacy and motivates R-Score, which decouples contextual ambiguity from strict precision. We compare 35 models across families and report their performance in redacting PII. Finally, we release RedactionBench to establish a baseline for future privacy-preserving systems, hoping to inspire efficient model design and standardized evaluations. 

---
# Lost in a Single Vector: Improving Long-Document Retrieval with Chunk Evidence Aggregation 

**Authors**: Shanshan Lyu, Yiwei Wang, Yujun Cai, Jiafeng Guo, Shenghua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.18781)  

**Abstract**: Dense retrieval ranks one query vector against one document vector. On long documents, this interface can fail when a short but decisive span is weakened during document encoding before ranking. We study this failure mode as document-side early compression and introduce the Evidence Dilution Index (EDI) to measure how far a document-level representation falls below the strongest chunk-level evidence within the same gold document. Guided by this view, we propose DICE (Document Inference via Chunk Evidence), a training-free document-side strategy that splits documents into chunks, encodes them independently with a frozen model, and aggregates them back into a single vector while preserving the standard one-query-one-document interface. On LongEmbed, DICE improves retrieval across four backbones, with the largest gains on slices beyond 4k tokens: for Dream, Passkey >4k rises from 30.0 to 90.0 and Needle >4k from 23.3 to 74.0. Across 12,779 filtered samples, DICE yields lower EDI than the single-vector baseline in 92.8% of cases. These results establish document-level encoding as a practical and underexplored lever for long-document retrieval. 

---
# Output Vector Editing for Memorization Mitigation in Large Language Models 

**Authors**: Ahmad Dawar Hakimi, Kaiwei Lei, Isabelle Augenstein, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2606.18767)  

**Abstract**: Large language models memorize and reproduce sequences from their training data, creating privacy, copyright, and security risks. Existing neuron-level mitigation methods equate editing with zeroing out neuron activations, but the activation only controls whether a neuron engages; the output vector is what writes to the residual stream and, through superposition, encodes multiple features. We propose output vector editing, a constrained-optimization weight edit that locates a small set of MLP neurons responsible for a memorized continuation and minimally modifies their output vectors to introduce a distractor in vocabulary space, redirecting their residual-stream contributions while leaving activations unchanged. Evaluating on four models from 360M to 7B parameters (SmolLM-360M, OLMo-1B, OLMo-7B, Llama2-7B), we center on OLMo-7B (whose open weights and pretraining corpus enable systematic mining) and mine 6831 memorized sequences, achieving up to 87.9% suppression. The 2.7$\times$ gap over zero ablation on the same located neurons shows the suppression comes from the output-vector edit, not localization alone. Four edit modes span a spectrum from aggressive suppression to minimal redirection; in ensemble they cover 96.5% of memorized sequences, while our recommended single-mode configuration reaches 81.5% with no catastrophic locality failures. We further identify a mechanistic boundary at ${\sim}14%$ of sequences unreachable by MLP-only editing; while these failures are not attention-driven overall, ablating the top contributing attention heads recovers 60--64% of them, with stronger recovery on continuations that copy tokens from the prefix, positioning attention as a complementary fallback rather than a primary mechanism. Edit mode ordering and the success-locality trade-off transfer across all four models, with success rates scaling with model size rather than family. 

---
# LegalWorld: A Life-Cycle Interactive Environment for Legal Agents 

**Authors**: Songhan Zuo, Shengbin Yue, Tao Chiang, Guanying Li, Yun Song, Xuanjing Huang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.18728)  

**Abstract**: Civil litigation is inherently a life-cycle process: what a lawyer drafts on day one constrains what unfolds at trial months later. Yet existing legal benchmarks evaluate isolated subtasks, and prior legal-agent simulators reinitialize each scenario from shared ground truth, leaving cross-stage causal dependencies unmodeled. We present LegalWorld, a life-cycle interactive environment that models Chinese civil litigation as a causally connected state chain of five stages (seven sub-scenarios), grounded in 75,309 paired Chinese civil judgments. We pair it with reusable infrastructure (local memory, global case memory, a Skill/Tool library) that keeps each dispute consistent across its full life cycle. Building on this environment, we construct LongJud-Bench to evaluate agent capability across all five connected stages. 18,992 ratings from 217 legal-background evaluators confirm that LegalWorld trajectories are procedurally faithful and role-consistent; and a capability-level cross-model evaluation reveals sharp divergences that aggregate scores cannot expose, with no single backbone leading across consultation, drafting, and courtroom advocacy. Detailed resources will be released publicly. 

---
# Morpheus: A Morphology-Aware Neural Tokenizer and Word Embedder for Turkish 

**Authors**: Tolga Şakar  

**Link**: [PDF](https://arxiv.org/pdf/2606.18717)  

**Abstract**: Turkish is agglutinative: meaning is carried by morphemes, yet the subword tokenizers that drive modern language models split words by corpus statistics, fragmenting semantically loaded suffixes and -- in the case of WordPiece and rule-based analyzers -- failing to decode their output back to the original text. This paper presents \textbf{Morpheus}, a neural morpheme-boundary model for Turkish that is at once a lossless, morphology-aware tokenizer and a word-embedding producer. A differentiable Poisson-binomial dynamic program turns per-character boundary probabilities into soft morpheme memberships during training and exact segments at inference, with no string normalization, so $\mathrm{decode}(\mathrm{encode}(w)) = w$ holds by construction. Because the model is neural, the same forward pass that tokenizes also emits a structured word embedding. Among reversible tokenizers -- the only ones valid for generation -- Morpheus attains the lowest bits-per-character ($1.425$), roughly doubles the gold morphological alignment of the subword family (MorphScore macro-F1 $0.61$ vs.\ ${\sim}0.32$), and uses ${\sim}19\%$ less GPU memory than 64K-vocabulary subword tokenizers. As an embedder, frozen Morpheus vectors lead on lexical retrieval (root-family MAP $0.85$) and same-root verification (ROC-AUC $1.00$), surpassing the multilingual retriever BGE-M3 and BERTurk; on context- and inflection-dependent tasks (NER, case/number probing) the heavier contextual encoders remain ahead -- a trade-off we attribute to Morpheus's root-centric geometry. Code: this https URL model: this https URL interactive demo: this https URL. 

---
# LLMs Struggle to Measure What Distinguishes Students of Different Proficiency Levels: A Study of Item Discrimination in Reading Comprehension Assessment 

**Authors**: Han Chen, Ming Li, Chenguang Wang, Yijun Liang, Dawei Zhou, Hong jiao, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.18709)  

**Abstract**: Item discrimination is a fundamental psychometric property of educational assessment, which measures whether an item meaningfully distinguishes students with higher proficiency from students with lower proficiency. While various existing works have explored whether large language models (LLMs) can estimate item difficulty, it remains unclear whether they can capture item discrimination. In this work, we evaluate 42 proprietary and open-weight LLMs in zero-shot settings using two complementary approaches: direct discrimination prediction, where models explicitly estimate an item's discrimination value from its content, and response-based Classical Test Theory (CTT) calibration, where LLM answers are treated as synthetic student responses to compute discrimination scores. Our results show that direct prediction yields weak alignment with human-calibrated discrimination: the best-performing model reaches only a Spearman correlation of 0.152. Response-based CTT calibration provides a stronger but still limited signal, with the all-persona synthetic respondent pool reaching a Spearman correlation of 0.241. These findings highlight item discrimination as an open challenge for LLM-based psychometric evaluation: current LLMs contain non-random discrimination-relevant signal, but they do not yet reliably capture how assessment items distinguish human students. 

---
# TW-LegalBench: Measuring Taiwanese Legal Understanding 

**Authors**: Fei-Yueh Chen, Chun Huang Lin, Chan Wei Hsu, Kuan Hsuan Yeh, Zih-Ching Chen, Kuan-Ming Chen, Patrick Chung-Chia Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.18699)  

**Abstract**: Large language models (LLMs) have shown impressive capabilities across diverse tasks, yet their performance on jurisdiction-specific legal reasoning remains underexplored. We present TW-LegalBench that utilizes Taiwanese legal system's rich official corpus open to the public to fill the gap in evaluating LLMs on Taiwanese law, among common-law benchmarks that focus on English sources and civil-law benchmarks focusing on sources of Simplified Chinese. TW-LegalBench comprises three task types: (1) over 16,000 multiple-choice questions (MCQs) across five years of official examinations in 18 professional domains; (2) 117 open-ended essay questions (OEQs) from examinations for legal professionals with official scoring rubrics; and (3) more than 14,000 legal judgment prediction (LJP) instances covering hundreds of crime categories. We evaluate 13 LLMs using accuracy for MCQs, a decomposed LLM-as-Judge framework based on the scoring rubric points for OEQs, and metrics for sentencing accuracy and statute citation for LJP. Our results reveal that top-performing models exceed the passing threshold for qualified lawyers (passing rate: 11%) but fall short of that for judges and prosecutors (passing rate: 1~2%). For LJP, while models demonstrate reasonable verdict type accuracy and sentence prediction capability, they struggle to cite exact legal articles. These findings highlight that reliable legal text generation remains challenging for LLMs, even though their performance on qualification examinations approaches human level. 

---
# RegMix-D: Dynamic Data Mixing via Proxy Training Trajectories 

**Authors**: Kaiyan Zhao, Zhongtao Miao, Akiko Aizawa, Yoshimasa Tsuruoka  

**Link**: [PDF](https://arxiv.org/pdf/2606.18663)  

**Abstract**: Data mixture selection is critical for Large Language Model pretraining. Existing methods such as RegMix select a single static mixture by fitting a regression model on small-scale proxy runs. We propose RegMix-D, a simple extension of RegMix to dynamic mixing. Our key observation is that proxy runs produce not only endpoint losses, but also full loss trajectories, which can be used to further improve data mixture. By training regression model on these trajectories, we can predict optimal mixtures at multiple training stages. RegMix-D supports two deployment modes: an offline variant that generates a complete mixture schedule before target training, and an online variant that adapts the mixture during training using observed loss. Experiments on 25B tokens of the Pile dataset with a 1B parameter target model show that RegMix-D consistently improves over RegMix and DoReMi across 13 downstream tasks while remaining proxy-efficient: it surpasses RegMix even with only 128 proxy models (25% of RegMix's proxy compute budget). 

---
# The Wrong Kind of Right: Quantifying and Localizing Misfired Alignment in LLMs 

**Authors**: Naihao Deng, Yiming Feng, Chimaobi Okite, Kaijian Zou, Lu Wang, Rada Mihalcea, Yulong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.18656)  

**Abstract**: Warning: This paper studies stereotypes and biases, and contains potentially disturbing examples, used for illustration purposes only. Our findings should not be interpreted as an argument against alignment. Instead, this paper highlights the need for principled approaches to more advanced alignment. Alignment aims to ensure that large language models (LLMs) behave safely and reliably, including by avoiding unsafe inferences. However, we show that such safety-oriented behaviors can misfire: models may reject warranted conclusions even when they are explicitly supported by context. We call this failure mode misfired alignment, where alignment-induced changes cause LLMs to override explicit evidence. To quantify this phenomenon, specifically on stereotype-related alignment, we introduce VETO, a benchmark consisting of 2,032 BBQ-derived contrastive pairs, and define a new metric, Misfired Alignment Rate (MAR), which measures on a 0 to 100 scale how often a model fails on a stereotype-related question but succeeds on its contrastive counterpart. We benchmark 25 LLMs on VETO, and show that all LLMs, including the most recent ones, exhibit non-trivial (4.7 to 18.9%) MARs while all human participants achieve 0.0% MAR. Controlled priming experiments further show that alignment-induced cues can substantially amplify MAR across LLMs, indicating that these failures are not merely artifacts of individual examples but can be induced by safety-related framing. Mechanistic analyses on open-weight LLMs reveal late-layer suppression of evidence-supported answers, and comparisons between instruct and base LLMs suggest that this suppression emerges after instruction training. These findings show that current alignment methods can overgeneralize surface-level safety cues, to the point of overriding objective evidence, motivating more work on alignment objectives that better preserve contextual grounding. 

---
# PEC-Home: Interpretation of Progressively Elliptical Commands in Smart Homes 

**Authors**: Yingyu Shan, Zeming Liu, Silin Li, Boao Qian, Jiashu Yao, Yuhang Guo, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.18636)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have empowered home assistants with natural language interaction capabilities. However, current assistants overlook the progressive omission that occurs in human dialogue as shared context accumulates, leading to more elliptical expressions for efficient communication. Thus, current assistants still struggle to interpret such elliptical expressions accurately, which limits their effectiveness in real-world applications. In practical smart home scenarios, assistants face two major challenges caused by elliptical commands: (1) referential ambiguity caused by different environmental expectations among multiple users; and (2) intention ambiguity resulting from user preferences that evolve over time or change with the environment. To address these challenges, we introduce PEC-Home, the first simulated home dataset specifically designed for interpreting progressively elliptical commands in smart homes. Extensive experiments on various LLMs, including GPT-4o, show that existing home assistants struggle to execute user-intended operations based solely on elliptical commands. Even when equipped with tools for storing and retrieving user dialogue history, execution accuracy remains below that achieved with complete commands.}. 

---
# PragReST: Self-Reinforcing Counterfactual Reasoning for Pragmatic Language Understanding 

**Authors**: Jihyung Park, Minchao Huang, Leqi Liu, Elias Stengel-Eskin  

**Link**: [PDF](https://arxiv.org/pdf/2606.18624)  

**Abstract**: Natural language understanding often depends on meanings that are implied rather than explicitly stated, requiring pragmatic reasoning. Despite strong performance on math and logical reasoning, large language models (LLMs) still struggle with making pragmatic inferences, often choosing literal interpretations. To improve LLM pragmatic reasoning, we introduce PragReST, a self-supervised framework that constructs pragmatic QA data, generates counterfactual reasoning traces, and trains models to internalize them through supervised fine-tuning and reinforcement learning, without human-labeled training data or distillation from a stronger teacher. Across four pragmatic benchmarks (PragMega, Ludwig, MetoQA, and AltPrag), PragReST improves over backbone models, task-specific pragmatic tuning baselines, and non-counterfactual variants of the same pipeline. On accuracy-based benchmarks, PragReST improves over the instruct backbone by 5.37 and 5.50% (absolute) for Qwen3-8B and Qwen3-14B, respectively. Our error analysis and ablations underscore the importance of counterfactual reasoning: PragReST primarily reduces errors caused by failures to contrast observed utterances with plausible alternatives, and removing counterfactual reasoning substantially reduces performance. Moreover, our training preserves out-of-domain performance on general-knowledge and mathematical reasoning benchmarks. 

---
# BCL: Bayesian In-Context Learning Framework for Information Extraction 

**Authors**: Haoliang Liu, Chengkun Cai, Xu Zhao, Han Zhu, Shizhou Huang, Xinglin Zhang, Tao Chen, Jenq-Neng Hwang, Zhang Huaping, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.18620)  

**Abstract**: Existing information extraction (IE) tasks increasingly adopt in-context learning (ICL) with large language models. However, current approaches either show inconsistent performance across model scales or lack systematic optimization and generalizability. Building on this, we propose BCL (Bayesian In-Context Learning Framework for Information Extraction), the first optimization framework that uses particle filtering with Bayesian updates to systematically refine label representations across IE tasks. Through four steps initialization, observation, weight update, and resampling, BCL generalizes to both sequence labeling and relation classification paradigms. Extensive experiments demonstrate substantial and consistent improvements over existing approaches. 

---
# Are LLMs Ready to Assist Physicians? PhysAssistBench for Interactive Doctor-Patient-EHR Assistance 

**Authors**: Tianming Du, Peijie Yu, Sihan Shang, Danli Shi, My Linh Nguyen, Shengbo Gao, Guangyuan Li, Yinghong Yu, Yan Jiang, Qianlong Zhao, Behzad Bozorgtabar, Shaoxiong Ji, Jiazhen Pan, Daniel Rueckert, Jiancheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.18613)  

**Abstract**: The most plausible near-term role of medical LLMs is to assist rather than replace physicians, yet current evaluations often test isolated capabilities: clinical knowledge, EHR system interaction, or patient communication. Physician assistance instead requires coordinating these capabilities within the same interaction, where physicians issue underspecified requests, patients describe symptoms ambiguously, and EHR systems demand precise tool use. We introduce PhysAssistBench, a benchmark for interactive doctor-patient-EHR assistance. Built from real MIMIC-IV cases, PhysAssistBench uses a scalable pipeline to construct agentic patients: interactive, record-grounded agents that turn static EHR records into multi-turn clinical scenarios while preserving clinical factuality. PhysAssistBench provides a curated bilingual evaluation set of 1,296 manually reviewed and physician-validated turns. Experiments with leading LLMs show that current models remain unreliable in this setting, which exposes a key bottleneck for clinical LLMs: reliable assistance requires coordination across knowledge, communication, and systems, not isolated gains in any of them. 

---
# Steerable Cultural Preference Optimization of Reward Models 

**Authors**: Minsik Oh, Advit Deepak, Sophie Wu, Douwe Kiela, Ekaterina Shutova  

**Link**: [PDF](https://arxiv.org/pdf/2606.18606)  

**Abstract**: It is essential for large language model (LLM) technology to serve many different cultural sub-communities in a manner that is acceptable to each community. However, research on LLM alignment has so far predominantly focused on predicting a unified response preference of annotators from certain regions. This paper aims to advance the development of alignment models with a more global outlook, that are able to accurately represent the preferences of subcommunities and do not exhibit excessive bias towards any of them. We focus on the development of reward models for this purpose and present a novel reward model training algorithm (SCPO) that can incorporate diverse cultural preferences in a balanced manner. Our method results in performance increases of the minority reward model of up to 7 points over the baseline model across two datasets, PRISM and GlobalOpinionQA, and across 7 countries. SCPO is up to 280% more training data-efficient than full-data finetuning of reward models. In addition, we perform analysis of bias by separately evaluating on the preference of subcommunities and show that excessive bias is mitigated via our weighting method. Our code is available at this https URL 

---
# Low-resource Language Discrimination Towards Chinese Dialects with Transfer learning and Data Augmentation 

**Authors**: Fan Xu, Yangjie Dan, Keyu Yan, Yong Ma, Mingwen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.18597)  

**Abstract**: Chinese dialects discrimination is a challenging natural language processing task due to scarce annotation resource. In this article, we develop a novel Chinese dialects discrimination framework with transfer learning and data augmentation (CDDTLDA) in order to overcome the shortage of resources. To be more specific, we first use a relatively larger Chinese dialects corpus to train a source-side automatic speech recognition (ASR) model. Then, we adopt a simple but effective data augmentation method (i.e., speed, pitch, and noise disturbance) to augment the target-side low-resource Chinese dialects, and fine-tune another target ASR model based on the previous source-side ASR model. Meanwhile, the potential common semantic features between source-side and target-side ASR models can be captured by using self-attention mechanism. Finally, we extract the hidden semantic representation in the target ASR model to conduct Chinese dialects discrimination. Our extensive experimental results demonstrate that our model significantly outperforms state-of-the-art methods on two benchmark Chinese dialects corpora. 

---
# Dual Dimensionality for Local and Global Attention 

**Authors**: Zhiyuan Wang, Xuan Luo, Sirui Zeng, Xifeng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.18587)  

**Abstract**: Decoder-only Transformers compute attention over the KV cache of preceding tokens. Keys (and Values) are typically represented with the same dimensionality, regardless of its distance from the prediction target. In natural language, however, the next word is most strongly influenced by the immediately preceding tokens. We hypothesize that local and distant tokens impose asymmetric demands on representational capacity: local tokens are more critical for predicting immediate outputs and thus require richer representations, whereas distant tokens primarily serve as long-range memory, for which lower-dimensional representations may suffice. We formalize this idea as Distance-Adaptive Representation (DAR), implemented in a controlled setting that preserves full-dimensional representations within a local context window while assigning reduced-dimensional representations (e.g. 1/4 of the original dimensionality) to tokens beyond that window. Across multiple pretraining scales (70M to 410M parameters), as well as continued supervised fine-tuning on a 1B-scale model, this approach closely matches the performance of full-dimensional baselines. In contrast, uniformly reducing dimensionality across all token positions leads to worse performance. These results challenge the common assumption that key and value dimensionality should be uniform across token positions. Our findings suggest a new direction for designing attention architectures that adaptively allocate representational capacity across sequences, enabling further reductions in KV cache during inference. 

---
# Speech-Driven End-to-End Language Discrimination towards Chinese Dialects 

**Authors**: Fan Xu, Jian Luo, MingWen Wang, GuoDong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.18584)  

**Abstract**: Language discrimination among similar languages, varieties, and dialects is a challenging natural language processing task. The traditional text-driven focus leads to poor results. In this paper, we explore the effectiveness of speech-driven features towards language discrimination among Chinese dialects. First, we systematically explore the appropriateness of speech-driven MFCC features towards CNN-based language discrimination. Then, we design an end-to-end speech recognition model based on HMM-DNN to predict Chinese dialect words. We adopt attention to extract the discriminative words related to different Chinese dialects. Finally, through a CNN, we combine the word-level embedding and the MFCC-based features. Evaluation of two benchmark Chinese dialect corpora shows the appropriateness and effectiveness of the proposed speech-driven approach to fine-grained Chinese dialect discrimination compared to the state-of-the-art methods. 

---
# MCompassRAG: Topic Metadata as a Semantic Compass for Paragraph-Level Retrieval 

**Authors**: Amirhossein Abaskohi, Raymond Li, Gaetano Cimino, Peter West, Giuseppe Carenini, Issam H. Laradji  

**Link**: [PDF](https://arxiv.org/pdf/2606.18508)  

**Abstract**: Retrieval-augmented generation (RAG) systems depend critically on how documents are chunked and searched. Fine-grained chunks can improve retrieval precision but expand the search space, increasing latency and cost; larger chunks reduce the number of candidates but make dense similarity less reliable, as the representation for each chunk mixes multiple topics and introduces more semantic noise. This trade-off becomes especially limiting in deep research tasks, where retrieval must be both fast and precise across large, heterogeneous corpora. We introduce MCompassRAG, a metadata-guided retrieval framework that uses topic-level signals as a semantic compass for selecting relevant evidence. Instead of relying only on cosine similarity between queries and noisy chunk embeddings, MCompassRAG enriches chunk representations with topic metadata in the same embedding space and trains a lightweight retriever through LLM-teacher distillation. At inference time, MCompassRAG performs topic-aware retrieval without additional LLM calls, improving both efficiency and evidence quality. Across six complex retrieval benchmarks, MCompassRAG improves information efficiency (IE) by 8.24% on average with over 5 times lower latency than the strongest efficient RAG baselines. Code is available on this https URL. 

---
# Towards Scalable Customization and Deployment of Multi-Agent Systems for Enterprise Applications 

**Authors**: Paresh Dashore, Shreyas Kulkarni, Uttam Gurram, Nadia Bathaee, Kartik Balasubramaniam, Genta Indra Winata, Sambit Sahu, Shi-Xiong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.18502)  

**Abstract**: Large language model (LLM)-based multi-agent systems demonstrate strong performance on complex reasoning and task execution, enabling broad enterprise applications. However, production deployment remains challenging due to domain-specific customization requirements and high latency and inference costs in agentic workflows. We propose a unified framework for customization and efficient deployment of multi-agent systems in real-world settings. The first stage, Agentic Model Customization, combines continual pretraining, supervised fine-tuning, and preference optimization to adapt a compact model to specialized domains while retaining strong agentic capabilities. The second stage, Inference Optimization, integrates speculative decoding and FP8 quantization with targeted calibration to enable cost-efficient serving with minimal quality loss. Across enterprise workloads, our framework enables rapid domain adaptation and achieves a 4.48x speedup in throughput while maintaining performance and improving robustness on long-tail scenarios. 

---
# PreUnlearn: Auditing Collateral Knowledge Damage Before Large Language Model Unlearning 

**Authors**: Bo Su, Ankit Shah, Thai Le  

**Link**: [PDF](https://arxiv.org/pdf/2606.18473)  

**Abstract**: Machine unlearning for large language models (LLMs) aims to remove specified knowledge while preserving the rest of the model's capabilities. However, the boundary between knowledge to forget and knowledge to retain is often unclear, since related and even distant information may be entangled in the model. In this paper, we study LLM unlearning from a data-centric perspective and measure how unlearning effects propagate from the forget set to same-domain and distant-domain knowledge. We find a consistent decay pattern: collateral damage is strongest near the forget set, weakens with semantic distance, but does not disappear at domain boundaries. We further ask whether such damage can be audited before unlearning is executed. We formulate forget-set auditing as a pre-unlearning prediction task and analyze which data features are most predictive of downstream damage. Our results show that interaction features between the forget set and evaluation set provide the strongest signals, suggesting that collateral damage is partly reflected in data geometry before model updates occur. These findings position forget-set auditing as an early warning tool for identifying risky unlearning runs and designing more reliable unlearning procedures. 

---
# Possible or Definite? A Benchmark for Evaluating Diagnostic Uncertainty Preservation in Clinical Text 

**Authors**: Hongbo Du, Zixin Lu, Jiaming Qu  

**Link**: [PDF](https://arxiv.org/pdf/2606.18471)  

**Abstract**: Large language models (LLMs) are increasingly used for clinical text tasks such as summarization and revision. While most studies evaluate the fluency and coherence of LLM-generated text, whether LLMs correctly preserve diagnostic uncertainty remains underexplored. In clinical practice, phrases such as ``possible pneumonia'' communicate the strength of available evidence and directly guide decisions about follow-up testing and treatment. Altering these uncertainty expressions can change the clinical meaning entirely. In this paper, we systematically evaluated this problem in two steps. First, we constructed a benchmark of 1,200 clinical documents with 9,184 uncertainty annotations across five levels. Second, we evaluated three LLMs on this benchmark. Our results show that (1) LLMs preserve the original uncertainty cues poorly, often less than half the time; (2) LLMs struggle with nuanced distinctions between adjacent levels. This work reveals a failure mode not captured by standard evaluation metrics and provides implications for the safe deployment of LLMs in clinical workflows. 

---
# Montreal Forced Aligner and the state of speech-to-text alignment in 2026 

**Authors**: Michael McAuliffe, Kaylynn Gunter, Michael Wagner, Morgan Sonderegger  

**Link**: [PDF](https://arxiv.org/pdf/2606.18466)  

**Abstract**: The Montreal Forced Aligner (MFA) was released in 2016 and has since become the most widely used tool for forced alignment in research and industry. In the decade since, MFA has undergone substantial development, including expanded coverage across more languages and dialects using larger open-source datasets, harmonized IPA dictionaries, model adaptation, cross-language phone remapping, and support utilities. This paper documents MFA 3.0's developments since version 1.0 and evaluates MFA's performance across English, Japanese, and Korean, benchmarked against classic and neural forced aligners. MFA 3.0 achieves state-of-the-art or near state-of-the-art performance across all four benchmark datasets with mean boundary errors below 15 ms. Adaptation and cross-language remapping are effective for languages outside MFA's training distribution, and pronunciation probability modeling and phonological rules provide gains in specific conditions. 

---
# LLM Parameters for Math Across Languages: Shared or Separate? 

**Authors**: Behzad Shomali, Luisa Victor, Tim Selbach, Ali Hamza Bashir, David Berghaus, Joachim Koehler, Mehdi Ali, Markus Frey  

**Link**: [PDF](https://arxiv.org/pdf/2606.18453)  

**Abstract**: Large language models (LLMs) exhibit substantial cross-lingual variation in mathematical reasoning performance, but it remains unclear whether these differences reflect language-specific parameters or a shared mechanism that manifests differently by language. We present a cross-lingual mechanistic analysis of mathematical reasoning in LLMs, enabling us to localize and compare model parameters that support mathematical reasoning across languages. We find that the extracted math-associated parameters exhibit partial cross-lingual overlap, with the strongest overlap concentrated in intermediate model layers. We further observe that English consistently produces the largest set of math-relevant parameters, whereas lower-resource languages reveal smaller sets of relevant parameters. These results suggest that math-related behavior in multilingual LLMs is neither fully language-invariant nor fully language-specific, but instead exhibits partial cross-lingual parameter overlap with systematic language-dependent differences. 

---
# VISUALSKILL: Multimodal Skills for Computer-Use Agents 

**Authors**: Ziyan Jiang, Li An, Yujian Liu, Jiabao Ji, Qiucheng Wu, Jacob Andreas, Yang Zhang, Shiyu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2606.18448)  

**Abstract**: Computer-use agents (CUAs) approach human-level performance on standardised benchmarks but still struggle on long-horizon tasks and unseen software. Existing skill libraries address this with reusable skills, but represent the skill artifact as text only, despite the visual nature of GUI interaction. We propose VISUALSKILL: a hierarchical multimodal skill, tailored to each target application and organised as a central index over per-topic files, which the agent consumes through a load_topic MCP tool that fetches the relevant topic's text and figures on demand. We construct each skill with a two-stage pipeline that combines authored documentation with live-application UI exploration. On two CUA benchmarks, CUA-World and OSExpert-Eval, a Claude Code CLI agent backed by Claude Opus 4.6 reaches an average score of 0.456 with VISUALSKILL, a +15.3 point absolute lift over the no-skill baseline (0.303). Against a matched text-only skill that is generated from the same source content and differs from VISUALSKILL only in modality, VISUALSKILL yields a further +8.3 point absolute gain over the matched text-only skill (0.373 vs. 0.456), providing direct evidence that retaining visual figures in the skill artifact, rather than verbalizing them away, helps the agent both identify UI elements and verify workflow state after each action. Our code is available at this https URL. 

---
# CoreMem: Riemannian Retrieval and Fisher-Guided Distillation for Long-Term Memory in Dialogue Agents 

**Authors**: Jiaqi Chen, Yongqin Zeng, Shaoshen Chen, Yijian Zhang, Hai-Tao Zheng, Chunxia Ma, XiuTeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.18406)  

**Abstract**: Personalized dialogue agents require continuous long-term memory to maintain coherent interactions across multiple sessions. However, deploying these capabilities on consumer-grade hardware (e.g., 8 GB VRAM edge devices) introduces severe memory and compute bottlenecks. Existing systems typically rely on isotropic cosine similarity for retrieval and heuristic rules for context compression. These approaches lack a unified theoretical foundation, frequently suffering from the hubness problem in high-dimensional retrieval and syntactic fragmentation during compression. To overcome these limitations, we propose CoreMem, a resource-efficient edge-cloud memory architecture fundamentally unified by information geometry. First, Riemannian retrieval replaces cosine matching with a locally adaptive Fisher-Rao metric, effectively penalizing hub memories via Mahalanobis distance with O(Ndr) Woodbury acceleration for real-time search. Second, Fisher-guided discrete token distillation (FDTD) introduces a hierarchical sentence-to-token compression mechanism. It derives sensitivity scores from Fisher information traces, providing a principled compression-KL tradeoff augmented with explicit structural syntax protection. Evaluated on the LOCOMO and LongMemEval-S benchmarks, CoreMem achieves strong accuracy improvements, yielding substantial gains in Open-domain (+4.51 pp) and Temporal (+4.17 pp) reasoning. Extensive profiling confirms that CoreMem operates seamlessly within a strict 8 GB VRAM budget, successfully bridging the gap between resource-constrained edge devices and the demand for theoretically grounded, lifelong memory agents. 

---
# JetFlow: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting 

**Authors**: Lanxiang Hu, Zhaoxiang Feng, Yulun Wu, Haoran Yuan, Yujie Zhao, Yu-Yang Qian, Bojun Wang, Daxin Jiang, Yibo Zhu, Tajana Rosing, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.18394)  

**Abstract**: Speculative decoding (SD) accelerates autoregressive Large Language Models (LLMs) by drafting multiple tokens and verifying them in parallel, but it faces a scaling limitation: increasing the draft budget improves speed only when acceptance remains high and drafting overhead stays low. This ceiling has been difficult to break because prior head-based SD methods face a causality-efficiency dilemma. Autoregressive drafters produce path-conditioned candidates that are effective for tree speculative decoding with higher acceptance length, but their drafting cost grows with tree depth. Bidirectional block-diffusion drafters generate all positions in one pass, but their branch-agnostic marginals can form individually plausible yet mutually inconsistent trees, wasting budget and reducing acceptance. We propose JetFlow, a head-based SD framework that combines one-forward drafting efficiency with branch-wise causal conditioning. JetFlow trains a causal parallel draft head over fused hidden states from the frozen target model, producing candidate trees whose scores align with the target model's autoregressive factorization. This enables JetFlow to convert larger draft budgets into longer accepted prefixes and higher end-to-end speedup. Across math, coding, and chat benchmarks on dense and MoE Qwen3 models, JetFlow consistently outperforms bidirectional-head and tree-based SD baselines. On H100 GPUs, JetFlow achieves up to 9.64x speedup on MATH-500 and 4.58x on open-ended conversational workloads, with further latency gains demonstrated through vLLM integration under realistic serving loads. Our code and models are available at this https URL. 

---
# Want Better Synthetic Data? Steer It: Activation Steering for Low-Resource Language Generation 

**Authors**: Jan Cegin, Daniil Gurgurov, Yusser Al Ghussin, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2606.18389)  

**Abstract**: Large language models (LLMs) have become an effective tool for synthetic data generation, including for low-resource languages, where generated data can improve downstream task performance. Current best-performing approaches typically rely on few-shot prompting with target-language examples, which increases inference costs and may reduce diversity through lexical anchoring. In this work, we investigate activation steering as an alternative for low-resource synthetic data generation. We study two steering strategies: Language Steering, which targets the linguistic identity of a language, and Quality Steering, which captures well-formedness by contrasting human-written and backtranslated text representations. We evaluate these methods across four open-source LLMs, multiple layers, and 11 typologically diverse languages by generating sentiment and topic classification data and finetuning smaller classifiers. Steering is applied in both zero-shot and few-shot prompting settings and compared against non-steered counterparts. Our results show that steering on early layers consistently improves the diversity of generated data while often yielding stronger downstream model performance, particularly for low-resource languages. 

---
# SproutRAG: Attention-Guided Tree Search with Progressive Embeddings for Long-Document RAG 

**Authors**: Amirhossein Abaskohi, Issam H. Laradji, Peter West, Giuseppe Carenini  

**Link**: [PDF](https://arxiv.org/pdf/2606.18381)  

**Abstract**: Retrieval-augmented generation (RAG) systems must balance retrieval granularity with contextual coherence, a challenge that existing methods address through LLM-guided chunking, single-level context expansion, or hierarchical summarization. These approaches variously depend on costly LLM calls during indexing or retrieval, limit context aggregation to a single granularity level, or introduce information loss through summarization. We present SproutRAG, an attention-guided hierarchical RAG framework that addresses this trade-off by organizing sentence-level chunks into progressively larger but semantically coherent units, using learned inter-sentence attention to construct a binary chunking tree. Unlike prior approaches that rely on external LLMs, fixed context expansion, or lossy summarization, SproutRAG learns which attention heads and layers best capture semantic document structure, enabling multi-granularity retrieval without additional LLM calls or compressed summaries. At retrieval time, SproutRAG uses hierarchical beam search to retrieve candidates at multiple granularities, capturing multi-sentence relevance beyond flat retrieval. The framework is trained end-to-end with a joint objective that improves both embeddings and tree structure. Experiments across four benchmarks spanning scientific, legal, and open-domain settings demonstrate that SproutRAG improves information efficiency (IE) by 6.1% on average over the strongest baseline. Code is available on this https URL. 

---
# Redact or Keep? A Fully Local AI Cascade for Educational Dialogue De-Identification 

**Authors**: Haocheng Zhang, Zhuqian Zhou, Kirk Vanacore, Bakhtawar Ahtisham, René F. Kizilcec  

**Link**: [PDF](https://arxiv.org/pdf/2606.18372)  

**Abstract**: Educational dialogue is a valuable but sensitive resource for research: the same transcripts that capture authentic learning often capture personally identifiable information (PII) entangled with curricular content, where "Riemann" may refer to a real student or to a mathematical concept. Existing approaches force a tradeoff between governance and accuracy. Commercial Large Language Models (LLMs) can handle this ambiguity but require sending student data to third parties, while local named entity recognition (NER) systems preserve governance but over-redact curricular terms. We propose a fully local cascade framework that reframes de-identification from open-ended entity recognition to constrained privacy triage. A recall-first union proposer combines two lightweight encoders with deterministic rules to over-generate candidate spans; a context-aware reviewer then makes a binary Redact/Keep decision for each candidate using surrounding dialogue and speaker role. We evaluate three reviewer configurations against same-family LLM-only baselines and a commercial API on math tutoring transcripts from two large platforms. The strongest local configuration reaches 0.958 macro F1, compared with 0.767 for a same-family LLM-only baseline and 0.706 for the commercial API, while running entirely on a single laptop. On a targeted challenge set of curricular-personal name ambiguity, the same configuration degrades by only 0.03 F1 versus 0.19 to 0.25 for smaller reviewers. These results suggest that for educational de-identification, problem formulation matters more than model scale. 

---
# Continuous Audio Thinking for Large Audio Language Models 

**Authors**: Gyojin Han, Dong-Jae Lee, Changho Choi, Jongsuk Kim, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.18273)  

**Abstract**: Large audio language models (LALMs) have shown impressive capabilities on diverse audio understanding tasks, ranging from speech transcription to music analysis. However, because LALMs are typically trained to produce text-aligned responses, their hidden states are progressively shaped for text generation rather than for preserving acoustic information. As a result, the diverse acoustic content that audio carries, such as phonetic detail, prosody, sound events, affect, and pitch, is lost along the way and difficult to leverage in the response. We introduce Continuous Audio Thinking (CoAT), a framework that equips audio language models with a continuous latent workspace for organizing acoustic information prior to response generation, grounded by distillation from audio experts. Within the thinking space, the model can utilize the rich acoustic information provided by expert distillation when generating its response. Furthermore, the proposed continuous thinking block can be processed in a single prefill, so CoAT does not require additional autoregressive decoding cost over the baseline. Across three LALMs, Qwen2-Audio, Qwen2.5-Omni-7B, and Audio Flamingo~3, performance gains on a broad benchmark suite spanning audio reasoning, audio understanding, music classification, speech emotion, and speech transcription demonstrate the effectiveness of CoAT. Further analysis confirms that the auxiliary supervision propagates from the thinking positions to the model's textual responses. 

---
# Native Active Perception as Reasoning for Omni-Modal Understanding 

**Authors**: Zhenghao Xing, Ruiyang Xu, Yuxuan Wang, Jinzheng He, Ziyang Ma, Qize Yang, Yunfei Chu, Jin Xu, Junyang Lin, Chi-Wing Fu, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2606.19341)  

**Abstract**: Passive models for long video understanding typically rely on a "watch-it-all" paradigm, processing frames uniformly regardless of query difficulty, causing computational cost to grow with video duration. Although interactive frameworks have emerged, they often rely on global pre-scanning, and their context cost still scales with video length. We propose OmniAgent, the first native omni-modal agent that formulates video understanding as a POMDP-based iterative Observation-Thought-Action cycle. OmniAgent executes on-demand actions to selectively distill audio-visual cues into a persistent textual memory, effectively decoupling reasoning complexity from raw video duration. To operationalize this, we introduce (1) Agentic Supervised Fine-Tuning to bootstrap native active perception via best-of-N trajectory synthesis with dual-stage quality control, and (2) Agentic Reinforcement Learning with TAURA (Turn-aware Adaptive Uncertainty Rescaled Advantage), which leverages turn-level entropy to steer credit assignment toward pivotal discovery turns. Crucially, OmniAgent exhibits positive test-time scaling, where performance improves as the number of reasoning turns increases, validating the efficacy of active perception. Empirical results across ten benchmarks (e.g., VideoMME, LVBench) demonstrate that OmniAgent achieves state-of-the-art performance among open-source models. Notably, on LVBench, our 7B agent outperforms the 10$\times$ larger Qwen2.5-VL-72B (50.5% vs. 47.3%). 

---
# Rethinking Reward Supervision: Rubric-Conditioned Self-Distillation 

**Authors**: Siyi Gu, Jialin Chen, Sophia Zhou, Arman Cohan, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2606.19327)  

**Abstract**: Post-training of reasoning language models is commonly driven by supervised distillation and reinforcement learning with verifiable rewards. Distillation often relies on chain-of-thought annotations that are expensive to obtain and may themselves be noisy, incomplete, or partially incorrect; even when the final solution is correct, an imperfect rationale can interfere with learning. Reinforcement learning with verified rewards, on the other hand, typically compresses evaluative feedback into a scalar signal, obscuring which aspects of a response should be improved. We propose \textbf{Rubric-Conditioned Self-Distillation}, a framework that incorporates rubrics as structured, fine-grained feedback for on-policy self-distillation. Our method conditions the teacher model on criterion-level rubrics and uses it to provide token-level guidance on the student's own sampled trajectories. This design avoids treating a single reference rationale as the sole supervision target. Instead, rubrics specify what a strong response should satisfy, enabling more fine-grained credit assignment over the reasoning process than scalar reward optimization. We instantiate this framework with a two-stage pipeline that first learns to generate task-specific rubrics and then trains a rubric-guided reasoner. We evaluate on a diverse suite of science reasoning benchmarks and results show that rubric-conditioned self-distillation effectively converts rubric-level criteria into token-level guidance over the reasoning process, surpassing GRPO by 1.0 points and OPSD by 0.9 points on average. 

---
# Structured Inference with Large Language Gibbs 

**Authors**: Sanghyeok Choi, Henry Gouk, Esmeralda S. Whitammer  

**Link**: [PDF](https://arxiv.org/pdf/2606.19264)  

**Abstract**: The knowledge encoded in large language models (LLMs) can serve as a substrate for structured reasoning over variables describing a complex world, but accessing this knowledge in a probabilistically coherent manner poses a difficult inference problem. We propose Large Language Gibbs, a scheme for structured probabilistic inference that uses conditional distributions of an LLM as transition operators. Rather than sampling structured objects through single-pass autoregressive generation, we iteratively resample individual variables conditioned on others using an LLM's next-token conditionals. This approach avoids order-dependent biases and produces a stationary distribution that reflects a compromise between all local conditionals. We apply this approach to sampling from synthetic distributions, consistent reasoning tasks, and Bayesian structure learning. The results suggest that the use of LLM conditionals in MCMC is a practical alternative to one-pass generation for structured probabilistic inference under a world prior accessible through noisy LLM conditionals. 

---
# STARE: Surprisal-Guided Token-Level Advantage Reweighting for Policy Entropy Stability 

**Authors**: Haipeng Luo, Qingfeng Sun, Songli Wu, Can Xu, Wenfeng Deng, Han Hu, Yansong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.19236)  

**Abstract**: Reinforcement Learning with Verifiable Rewards algorithms like GRPO have emerged as the dominant post-training paradigm for complex reasoning in LLMs, yet commonly suffer from policy entropy collapse during training. We conduct a first-order gradient analysis of token-level entropy dynamics under GRPO and identify a token-level credit assignment mismatch: the per-token entropy variation decomposes into the product of the trajectory-level advantage and an entropy sensitivity function over the next-token distribution, yielding an advantage-surprisal four-quadrant structure and a near-criticality property. Motivated by it, we propose STARE (Surprisal-guided Token-level Advantage Reweighting for policy Entropy stability), which identifies entropy-critical token subsets via batch-internal surprisal quantiles, selectively reweights their effective advantages, and incorporates a target-entropy closed-loop gate for stable entropy regulation. Across model scales from 1.5B to 32B and three task families (Short CoT, Long CoT, and Multi-Turn Tool Use), STARE sustains stable RL training over thousands of steps while maintaining policy entropy within the target band. On AIME24 and AIME25, STARE outperforms DAPO and other competitive baselines by 4%-8% in average accuracy, with reflection tokens and response length growing in tandem, indicating sustained exploration-exploitation balance that further unlocks RL training this http URL is available at this https URL. 

---
# IndicContextEval: A Benchmark for Evaluating Context Utilisation in Audio Large Language Models Across 8 Indic Languages 

**Authors**: Sakshi Joshi, Dhruv Subhash Rathi, Sanskar Singh, Eldho Ittan George, R J Hari, Kaushal Bhogale, Mitesh M. Khapra  

**Link**: [PDF](https://arxiv.org/pdf/2606.19157)  

**Abstract**: AudioLLMs enable speech recognition conditioned on textual prompts such as domain descriptions or entity lists. However, it remains unclear whether these models genuinely utilise such context or rely on parametric knowledge learned during pretraining. Existing benchmarks cannot answer this question because they evaluate transcription under fixed prompting conditions and rarely include explicit contextual inputs. We introduce IndicContextEval, a 56-hour multilingual benchmark of natural speech from 555 speakers across 8 Indian languages and 23 professional domains. We design a 7-level prompting framework that progressively introduces contextual signals, including metadata, natural-language descriptions, entity lists in English and native script, and adversarial prompts with incorrect entities. Evaluating five models reveals substantial differences in context utilisation behaviour, highlighting the need for explicit evaluation of contextual grounding in AudioLLMs. 

---
# Human-AI Coevolution Dynamics: A Formal Theory of Social Intelligence Emergence Through Long-Term Interaction 

**Authors**: Jingyi Zhou, Senlin Luo, Haofan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.19144)  

**Abstract**: Current conversational AI systems have made significant progress in language generation, personalization, and long-context interaction. However, most existing methods model social behavior through isolated components such as emotion modeling, memory retrieval, or persona conditioning, lacking a unified framework to explain the emergence of stable social relationships and social intelligence in long-term human-AI this http URL address this, we propose the Human-AI Coevolution Dynamics Framework (HACD-H), a formal model of human-AI interaction as a self-organizing social cognitive system. HACD-H integrates emotional adaptation, relational organization, social memory, and personality consistency into a unified dynamical framework and introduces principles including multi-timescale social cognition, relational attractors, trust basins, developmental phase transitions, and social cognitive energy this http URL construct a conversational dataset with approximately 14,700 interaction turns and develop a theory-driven empirical evaluation framework. Results reveal a hierarchy of temporal persistence in social cognition, stable relational attractors, phase-transition-like developmental patterns, and a structured social cognitive energy landscape. Social intelligence shows a significant negative correlation with social cognitive energy (r = -0.391, p < 0.001), and interaction trajectories exhibit progressive energy reduction over this http URL findings suggest that social intelligence emerges from long-term social cognitive coevolution rather than isolated conversational capabilities. HACD-H provides a unified theoretical foundation for modeling adaptive human-AI social interaction and developing socially intelligent AI systems. 

---
# Urdu Katib Handwritten Dataset: A Historical Document Dataset for Offline Urdu Handwritten Text Recognition with CRNN-Based Baseline Evaluation 

**Authors**: Ramza Basharat, Muhammad Usman Ali  

**Link**: [PDF](https://arxiv.org/pdf/2606.19139)  

**Abstract**: Automatic Handwritten Text Recognition (HTR) is inherently a challenging task, and its complexity is further increased when dealing with cursive scripts. Although significant efforts have been made on various cursive scripts, research regarding Urdu Handwritten Text Recognition (UHTR) has been relatively limited. This lag of research is primarily due to the unique challenges posed by its script, and the scarcity and unavailability of benchmark datasets. Therefore, to advance research in UHTR, this study presents a specialized real dataset called the Urdu Katib Handwritten Dataset (UKHD). To the best of our knowledge, this is the first offline Urdu handwritten text lines dataset specifically curated from the materials written by Katibs in historical times. It encompasses a diverse range of flat nib writing variations in the Nastalique calligraphic style. Additionally, the effectiveness of different CRNN-based hybrid models has been evaluated to identify the optimal architecture for Urdu Katib Handwriting Recognition (UKHR). Among the analyzed models, the CNN-BGRU-CTC model showed more robust performance, with low Character Error Rate (CER) and Word Error Rate (WER). This research work aims to support and encourage the research community in developing a robust recognition system for preserving Urdu handwritten literature. 

---
# Written by AI, Managed by AI: Semantic Space Control and Index Sickness Elimination Across 391 Consecutive Sessions 

**Authors**: Hui Zhang, Shuren Song  

**Link**: [PDF](https://arxiv.org/pdf/2606.19121)  

**Abstract**: The prevailing engineering intuition for addressing conceptual drift in long-horizon LLM collaboration is to trade more formal constraints for more reliable outputs -- designing symbolic identifier systems, accumulating defensive rules in System Prompts, expanding context windows. Our engineering record shows that in long-horizon settings, this direction may produce effects contrary to design intent. Using action research methods in a real software project (Bang-v3) spanning approximately one month and 391 collaborative sessions, we document and analyze the failure process of these strategies. When the symbolic system exceeds a complexity threshold, LLMs do not become more accurate -- instead, they abandon genuine understanding of business semantics, retreat to self-referential reasoning within the symbolic layer, and generate outputs that appear internally consistent but are physically disconnected from reality. We name this failure pattern "Index Sickness," and its canonical manifestation "Phantom Legislation." We name the underlying principle the "Pang Principle (Semantic Vitality Law)": natural language carrying explicit purpose conveys far greater information quality than symbolic expression. From this, we design and validate its physical engineering mechanism: "Baseline-Log Physical Separation." In the same project, this mechanism reduced AI Instructions volume by ~75%, and across the subsequent ~150 sessions, no recurrence of Index Sickness was observed. A bilingual companion version (Chinese) is included as supplementary material. 

---
# Mitigating Scoring Errors and Compensating for Nonverbal Subtests in Speech-Based Dementia Assessment 

**Authors**: Franziska Braun, Christopher Witzl, Andreas Erzigkeit, Hartmut Lehfeld, Thomas Hillemacher, Tobias Bocklet, Korbinian Riedhammer  

**Link**: [PDF](https://arxiv.org/pdf/2606.18979)  

**Abstract**: Early detection of cognitive impairment relies on neuropsychological tests to minimize subjectivity by assessing multiple cognitive domains. Speech-based evaluation can support diagnostics and improve accessibility, but transcription errors and the omission of nonverbal subtests (e.g., motor skills) limit accuracy. Beyond conventional test scores, speech-derived features can provide additional insights into cognitive status. This study investigates the speech-based evaluation of the German "Syndrom-Kurz-Test," a standardized dementia screening test comprising verbal and motor subtests. We train models that integrate transcript-derived scores and Whisper embeddings per verbal subtest to reduce scoring errors. To compensate for missing motor subtests, we then leverage these fused representations to approximate expert overall ratings. Despite omitting subtests, our models strongly correlate with expert ratings and efficiently and accurately discriminate between cognitive status groups. 

---
# Decoupling Search from Reasoning: A Vendor-Agnostic Grounding Architecture for LLM Agents 

**Authors**: Emmanuel Aboah Boateng, Kyle MacDonald, Amardeep Kumar, Siddharth Kodwani, Sudeep Das  

**Link**: [PDF](https://arxiv.org/pdf/2606.18947)  

**Abstract**: Production LLM agents increasingly depend on real-time search, yet native search grounding bundles retrieval policy, provider choice, evidence injection, cost, latency, and generation behavior behind a single model-provider boundary. This coupling makes grounding hard to inspect, tune, reuse, or port, and can trigger Search-Induced Verbosity that breaks strict output contracts. We present Decoupled Search Grounding (DSG), a vendor-agnostic boundary that moves grounding outside the reasoning model through an MCP-compatible gateway, exposing provider routing, source-aware context rendering, configured fallback, retrieval-depth control, and exact plus semantic caching as first-class controls. Across five frontier models on SimpleQA, FreshQA, and HotpotQA, native search leads on recency-sensitive FreshQA, but DSG exposes a stronger frontier when control matters: on SimpleQA it nearly matches native accuracy (86.1% vs. 87.7%) at 91% lower search cost, preserves concise answer contracts, and reaches a 99.4% warm-cache hit rate with 68% lower latency. Deployed as a shared production grounding layer for large-scale agentic workloads with interchangeable models, DSG matches or slightly exceeds native-search accuracy on an e-commerce query-understanding (QIU) workload while cutting search cost by over 98%. Real-time grounding is best treated as an optimizable interface boundary, not a fixed model feature. 

---
# Graph-ESBMC-PLC: Formal Verification of Graphical PLCopen XML Ladder Diagram Programs Using SMT-Based Model Checking 

**Authors**: Pierre Dantas, Lucas Cordeiro, Waldir Junior  

**Link**: [PDF](https://arxiv.org/pdf/2606.18941)  

**Abstract**: PLCopen XML defines two encoding formats for IEC 61131-3 Ladder Diagram programs: a textual encoding using <rung> elements, and a graphical encoding that represents rung logic as a directed graph of localId/refLocalId connections. ESBMC-PLC supported the textual format but parsed graphical exports from CONTROLLINO, Beremiz, and OpenPLC Editor into an empty GOTO intermediate representation, causing vacuous verification success. This paper presents Graph-ESBMC-PLC, which closes this gap with a DFS-based graphical LD resolver. The resolver traverses the connection graph from leftPowerRail to each coil, extracts rung paths as Boolean contact conjunctions, and applies a three-tier I/O inference scheme. Ordering coils by rightPowerRail connectionPointIn sequence ensures SET coils process before RESET coils, matching IEC scan-cycle semantics. The graphical-to-IR conversion leaves the ESBMC backend unchanged. Validation on 3 graphical LD programs from CONTROLLINO/OpenPLC Editor shows all produce full GOTO IR with nondeterministic inputs and rung logic, versus the empty IR previously. All 3 verify SAFE at k=2 under 70ms. The 11 textual LD benchmarks are fully preserved, with no regression. Two Beremiz examples with no LD content or unsupported timer semantics are reported as discovered limitations. Artifact at Zenodo (DantasCordeiro2026graphical, doi:https://doi.org/10.5281/zenodo.20699856). 

---
# REVES: REvision and VErification--Augmented Training for Test-Time Scaling 

**Authors**: Yuanxin Liu, Ruida Zhou, Xinyan Zhao, Amr Sharaf, Hongzhou Lin, Arijit Biswas, Mohammad Ghavamzadeh, Zhaoran Wang, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2606.18910)  

**Abstract**: Test-time scaling via sequential revision has emerged as a powerful paradigm for enhancing Large Language Model (LLM) reasoning. However, standard post-training methods primarily optimize single-shot objectives, creating a fundamental misalignment with multi-step inference dynamics. While recent work treats this as multi-turn reinforcement learning (RL), conventional approaches optimize over the multi-step trajectories directly, failing to further exploit the high-quality mistakes in intermediate steps that model can learn from correcting them. We propose a two-stage iterative framework that alternates between online data/prompt augmentation and policy optimization. By converting the intermediate steps (``near-miss'' answers) in the successful recovery trajectories into decoupled revision and verification prompts, our approach concentrates training on both effective answer transformation and error identification. This approach enables efficient off-policy data generation and reduces the computational overhead of long-horizon sampling compared to standard multi-turn RL. On LiveCodeBench, using publicly available test cases as feedback, we observe gains of +6.5 points over the RL baseline and +4.0 points over standard multi-turn training. Beyond coding, our approach matches the previously reported SOTA result on circle packing while using the smallest base model (4B) and far fewer rollouts than the much larger evolutionary search systems. Math results under ground-truth verification further confirm improved correction ability. It also generalizes to out-of-distribution constraint-satisfaction puzzles such as n\_queens and mini\_sudoku, where correctness is defined entirely by problem constraints. Code is available at this https URL. 

---
# GateMem: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents 

**Authors**: Zhe Ren, Yibo Yang, Yimeng Chen, Zijun Zhao, Benshuo Fu, Zhihao Shu, Bingjie Zhang, Yangyang Xu, Dandan Guo, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.18829)  

**Abstract**: Memory benchmarks for LLM agents largely assume single-user settings, leaving shared assistants for hospitals, workplaces, campuses, and households understudied. In these deployments, multiple principals write to a common memory pool and query it under different roles, scopes, and relationships, so memory quality requires governance as well as recall. We introduce GateMem, a benchmark for multi-principal shared-memory agents. GateMem jointly evaluates utility for legitimate long-horizon requests with state updates, access control across contextual authorization boundaries, and agent-facing active forgetting after explicit deletion requests. It spans medical, office, education, and household domains, with long-form multi-party episodes, incremental memory injection, hidden checkpoints, structured judging, and leak-target annotations. Across diverse baselines and backbone models, no method simultaneously achieves strong utility, robust access control, and reliable forgetting. Long-context prompting often yields the best governance score at high token cost, while retrieval-based and external-memory methods reduce cost yet still leak unauthorized or deleted information. These results show current memory agents remain far from reliable shared institutional deployment. 

---
# HandwritingAgent: Language-Driven Handwriting Synthesis in Scalable Vector Space 

**Authors**: Jaward Sesay, Yue Yu, Börje F. Karlsson  

**Link**: [PDF](https://arxiv.org/pdf/2606.18788)  

**Abstract**: Teaching machines to emulate natural handwriting styles remains an open challenge, as it requires synthesizing stroke sequences that dynamically vary in shape, texture, pressure and script - not only across individuals, but also within a single person's handwriting. Attempts at this challenge have largely explored deep learning methods in both online and offline settings. However, these approaches are often constrained by style-specific architectural choices, heavy reliance on large datasets, high compute costs, and a lack of flexible control over writing styles through natural language. To this end, we introduce HandwritingAgent, a language-driven agent that can synthesize natural handwriting sequences directly in Scalable Vector Graphics (SVG) format with no need for style-specific training. The agent leverages a large reasoning model to geometrically analyse and autoregressively generate target handwritten glyphs as stroke sequences in a discrete grid canvas environment. Generation is conditioned on texts provided in either conversational or non-conversational mode, along with a reference handwriting-style image. Experiments on diverse handwriting tasks spanning imitation, recognition, multi-lingual handwriting synthesis, and generation of complex handwritten maths and science expressions indicate substantial improvement in performance, with HandwritingAgent matching or surpassing state-of-the-art generative handwriting models, while providing a more efficient, controllable, and generalizable synthesis method. 

---
# SAMA: Semantic Anchor-aligned Augmentation for Unified Low-Resource Multimodal Information Extraction 

**Authors**: Quanjiang Guo, Chong Mu, Jiazhou Pan, Ming Jia, Ling Tian, Hui Gao, Zhao Kang  

**Link**: [PDF](https://arxiv.org/pdf/2606.18780)  

**Abstract**: Multimodal Information Extraction (MIE)-covering tasks such as Multimodal Named Entity Recognition (MNER), Relation Extraction (MRE), and Event Extraction (MEE)-is essential for understanding multimedia content but remains constrained by severe data scarcity. Although data augmentation is a promising remedy, existing approaches are impeded by coarse cross-modal alignment and fragmented, task-specific designs that fail to exploit shared semantic knowledge. To overcome these limitations, we introduce Semantic Anchor-aligned Multimodal Augmentation (SAMA), a unified framework for generating high-fidelity, task-aware synthetic data. SAMA constructs structured semantic anchors from ground-truth labels to guide a Collaborative Multi-Experts Multimodal Large Language Model (CME-MLLM), which integrates a Universal Adapter for shared semantics with Task-Specific Adapters to produce diverse yet constraint-compliant textual samples. For image synthesis, SAMA employs an Anchor-Preserving Diffusion mechanism that uses anchor-weighted prompts and latent conditioning to maintain critical semantic anchors while diversifying visual contexts. To eliminate the need for manual verification, SAMA further introduces a Dual-Constraint Filtering module that selects synthetic samples based on both cross-modal consistency and anchor fidelity. Extensive experiments across benchmark datasets for MNER, MRE, and MEE demonstrate that SAMA consistently outperforms state-of-the-art augmentation baselines under both fully supervised and low-resource settings, underscoring its versatility, robustness, and effectiveness. 

---
# Attention as Frustrated Synchronization 

**Authors**: Joshua Nunley  

**Link**: [PDF](https://arxiv.org/pdf/2606.18694)  

**Abstract**: A network of oscillators that synchronizes perfectly computes nothing further, so an attention architecture built from synchronization must locate its computation in structured departures from agreement. We introduce the Frustrated Synchronization Network (FSN), whose token states are phases on a torus and whose entire value pathway is one learned complex coupling kernel over harmonics and a one-step delay. Each component of the kernel is a frustration in the sense of the synchronization literature. The complex phases are static Kuramoto-Sakaguchi frustration angles, the signed harmonics are repulsive Daido components, and the delay term, which couples each token to the successors of the tokens it attends to, is algebraically identical to Kuramoto-Sakaguchi coupling whose frustration angle is the data's own transition, so next-token prediction is implemented as synchronization frustrated by the data. At matched one-million-parameter and training budgets on character-level text and code, the FSN's validation loss is below a tuned RoPE-SwiGLU transformer's at every epoch measured, and the comparison survives training the baseline to convergence: every thirty-epoch enwik8 seed finishes below the transformer's converged fifty-epoch loss of 1.611, and the FSN's completed fifty-epoch runs converge to 1.5953 +/- 0.0014. A variant with every feed-forward block replaced by mean-field coupling to learned collective modes, leaving no multilayer perceptron in the stack, tracks the transformer. On natural text the unfrustrated base layer falls behind the converged transformer at every copy depth, worst on long-range copy events; the kernel reverses the deficit at every depth of four and beyond. Headline comparisons are at the one-million-parameter scale; a scale ladder is complete through four million parameters with the advantage persisting, and remaining arms are marked as in progress. 

---
# ForecastBench-Sim: A Simulated-World Forecasting Benchmark 

**Authors**: Jaeho Lee, Nick Merrill, Ezra Karger  

**Link**: [PDF](https://arxiv.org/pdf/2606.18686)  

**Abstract**: Forecasting benchmarks for general-purpose AI systems usually inherit the constraints of the real world: outcomes resolve slowly, tail events are rare, and counterfactual questions are difficult to score. We introduce ForecastBench-Sim, a simulated-world forecasting benchmark built on game rollouts from Freeciv, a turn-based strategy game modelled on the Civilization series. Forecasters receive a fixed world report (a structured snapshot of the current game state) and answer questions about hidden future states; the benchmark then continues the simulation and scores forecasts. Because the world is simulated, the same setup can generate continuous or binary forecasting questions at arbitrary time horizons, paired intervention worlds for conditional or causal questions, and resolved examples of rare or disruptive outcomes. We describe the benchmark pipeline, question families, scoring protocol, and release artifacts, and report validation slices from model evaluations and an anonymized human pilot. ForecastBench-Sim is intended to complement real-world forecasting benchmarks by providing controlled, immediately resolvable tasks for studying probabilistic reasoning under dynamic world states. 

---
# EARS: Explanatory Abstention for Reliable Sub-Agent Modeling in Large-scale Multi-Agent Systems 

**Authors**: Shuang Xie, Yunan Lu, Han Li, Lingyun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.18668)  

**Abstract**: In large-scale enterprise settings, centralized multi-agent systems (MAS) are increasingly adopted, in which a coordinator delegates user requests to lightweight, domain-specialized sub-agents. While this architecture improves modularity, scalability, and cost efficiency, its reliability depends not only on accurate routing but also on sub-agents' ability to calibrate their responses to capability constraints. In particular, sub-agents built on smaller fine-tuned models often struggle with such calibration, leading them to over-answer ambiguous, underspecified, misrouted, or unsupported requests and produce hallucinated outputs instead of actionable feedback. To address this challenge, we present EARS (Explanatory Abstention for Reliable Sub-Agent Modeling), a production-oriented framework that reframes sub-agent abstention as an inter-agent communication protocol: a sub-agent does not merely abstain, but exposes an actionable failure state to the coordinator. EARS curates human-agent interaction data using an ensemble of calibrated LLM-as-a-Judge models, producing structured abstention labels and rationales under a taxonomy of sub-agent failure modes. These data are used to fine-tune sub-agents to detect failure conditions and return rationales for coordinator-level clarification, rerouting, or fallback. We evaluate EARS in a large-scale production e-commerce assistant supporting enterprise business intelligence workflows. EARS improves the overall response pass rate from 68.5% to 78.9%, demonstrating that sub-agent-side explanatory abstention improves MAS reliability. 

---
# Fair Cognitive Impairment Detection Through Unlearning 

**Authors**: William Nguyen, Jiali Cheng, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2606.18571)  

**Abstract**: Mild Cognitive Impairment (MCI) is a medical condition characterized by a noticeable decline in memory, language, or thinking abilities. MCI detection from spontaneous speech is promising for scalable screening. However, learned models often exploit demographic cues correlated with labels, resulting in a large performance gap across subgroups. We present a multimodal framework that combines (i) cross-model fusion between modalities (speech, text, and image), and (ii) unlearning using gradient reversal that discourages the shared embedding from encoding task-irrelevant demographic attributes. Evaluated on the multilingual benchmarks TAUKADIAL and PREPARE, our method outperforms the state-of-the-art multilingual and multimodal baseline in MCI classification while substantially reducing the performance gap across patient subgroups (sex and language). We further analyze transfer across datasets, showing that demographic unlearning helps learn more robust representations for MCI detection. 

---
# CEO-Bench: Can Agents Play the Long Game? 

**Authors**: Haozhe Chen, Karthik Narasimhan, Zhuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.18543)  

**Abstract**: Language model agents are becoming proficient executors at isolated, short-horizon tasks such as software engineering and customer service. Yet real-world challenges require a combination of sophisticated skills that remain largely untested in agents: (1) navigating long horizons amid uncertainty; (2) acquiring information in noisy environments; (3) adapting to a changing world; (4) orchestrating multiple moving parts toward a coherent goal. We introduce CEO-Bench, which evaluates these capabilities together by simulating a representative real-world task: operating a startup for 500 days. An agent manages pricing, marketing, budgeting, and many other aspects of a fictional company through a programmable Python interface, operating in the same environment and facing the same challenges as a human CEO. Success demands analyzing noisy, interconnected business databases, translating signals into sound strategy, and coordinating many decisions with programming. The strongest agents write sophisticated code that simulates customer cohorts to forecast future cash and mines negotiation history to uncover hidden customer preferences. Even so, most state-of-the-art models struggle in this environment. Only Claude Opus 4.8 and GPT-5.5 finish above the $1M starting balance, and neither consistently turns a profit. CEO-Bench takes a first step toward measuring the intelligence required to drive sustained, adaptive progress over time. 

---
# Evaluating Prompting-Based Defenses Against Domain-Camouflaged Injection Attacks 

**Authors**: Aaditya Pai  

**Link**: [PDF](https://arxiv.org/pdf/2606.18530)  

**Abstract**: Domain-camouflaged injection attacks embed malicious instructions in retrieved content using domain-appropriate vocabulary, evading standard detectors that rely on syntactic injection markers. When detection fails, practitioners need to know which defense architectures reduce attack success. We evaluate five prompting-based defenses (spotlighting, paraphrasing, prompt sandwiching, and two combinations) against domain-camouflaged injection across three model families (Claude Haiku, Llama 3.1 8B, Gemini 2.0 Flash) and three deployment domains (financial, legal, general) using 3,510 trials. Paraphrasing retrieved content before agent processing is the most consistently effective defense in this benchmark, reducing camouflage attack success rate by 55-84\% depending on model, and achieves lower attack success rates than our Llama Guard 4 configuration on every model tested. Defense effectiveness is strongly model-dependent: spotlighting halves attack success on Claude Haiku but provides no benefit on Llama 3.1 8B. Financial domain deployments face the highest residual risk at 26-33\% baseline attack success rate, with no prompting-based defense fully eliminating the threat on weaker models. These results provide the first systematic evaluation of prompting-based defenses specifically against camouflage-class injection attacks and establish benchmark-based recommendations for practitioners. All tasks use synthetically constructed professional documents; whether these benchmark rankings generalize to real enterprise documents remains an open question. 

---
# Compact Geometric Representations of Hierarchies 

**Authors**: Prashant Gokhale, Piotr Indyk, Yuhao Liu, Sandeep Silwal, Tony Chang Wang, Haike Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.18520)  

**Abstract**: Computing geometric representations of data is a cornerstone of modern machine learning, typically achieved by training dual encoders which map queries and documents into a shared embedding space. Recent work of You et al. [NeurIPS '25] has extended this approach to hierarchical retrieval, where relevance is determined by the ancestor-descendant relationships in a Directed Acyclic Graph (DAG). While previous work has shown that valid embeddings exist when the number of descendants is small, these bounds degrade significantly for deep hierarchies, requiring dimensions as large as the total number of nodes.
In this paper, we investigate compact reachability embeddings for more general graph classes and provide theoretical guarantees for representing hierarchies using embeddings whose dimension depends on structural graph parameters. We prove that for any directed tree, there exists a reachability embedding in constant dimension 3, independent of the tree's size or depth. We generalize this result to graphs characterized by treewidth $t$, constructing embeddings of dimension $O(t \log n)$, where $n$ is the number of nodes. Complementing these upper bounds, we provide matching or near-matching lower bounds, showing that dimension $\Omega(n)$ is necessary for general DAGs and $\Omega(t/\log(n/t))$ is required for graphs of treewidth $t$. We also obtain upper and lower bounds parameterized by the number of cross-edges in the DAG. We additionally show that our embeddings can be constructed on real world datasets, and that they give much smaller dimensions in high recall regimes compared to prior embeddings with theoretical guarantees. 

---
# SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR 

**Authors**: Siddharth Aphale, Kelly Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.18487)  

**Abstract**: The standard heuristic of selecting the SFT checkpoint with the highest pass@1 for GRPO can fail when SFT compresses the rollout distribution. For binary rewards, the expected within group advantage variance is $p(1{-}p)(g{-}1)/g$; when early GRPO drives $p$ below $p^*(g)$, most groups have identical rewards and provide no group relative signal. We study SFT depth ladders for Qwen2.5-Coder-3B and DeepSeek-Coder-6.7B. We test Qwen2.5-Coder-3B across five depths and three seeds, and DeepSeek-Coder-6.7B across four matched depths and three seeds. On Qwen, pre RL pass@1 rises with SFT depth, but peak GRPO pass@10 falls from $0.806$ to $0.481$ (3 seed mean, $n{=}20$); pre RL entropy is positively associated with the GRPO outcome ($\rho{=}{+}0.69$). On DeepSeek, pass@1 remains far above $p^*(8){=}0.083$, and GRPO outcomes compress rather than invert. A two stage diagnostic, combining pre RL entropy triage with an early GRPO entropy monitor, flags high risk checkpoints and can stop failing runs early. Simple KL to reference regularisation and label smoothing variants do not rescue the collapsed Qwen checkpoint in our setting, suggesting the failure is not a trivial GRPO hyperparameter artefact. 

---
# LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents 

**Authors**: Haoyang Fang, Wei Zhu, Boran Han, Alex Zhang, Zhenyu Pan, Shuo Yang, Shuai Zhang, Jiading Gai, Peng Tang, Cuixiong Hu, Xuan Zhu, Huzefa Rangwala, George Karypis, Bernie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.18388)  

**Abstract**: RL post-training strategies are dataset-dependent and reveal a recurring empirical pattern: capacity parameters accumulate monotonically across stages, while regularization parameters predominantly oscillate in response to shifting training dynamics. This distinction matters because fixed schedules commit all parameters to fixed trajectories and therefore cannot express the non-stationary exploration-exploitation tradeoffs that regularization must track; the principle provides actionable design rules for multi-stage training. We discover this through LLMZero, a system where LLM agents search over training trajectories via tree search, diagnosing pathologies at each checkpoint and proposing coordinated multi-parameter transitions. Across 4 diverse GRPO tasks, LLMZero discovers strategies that improve over the base model by 9% to 140% relative and over grid search by 6% to 15% relative, consistently outperforming random search and the skill-based agent. The structural principle transfers across tasks, providing an explanation for why discovered strategies take qualitatively different forms yet share similar parameter dynamics. 

---
# From Sparse Features to Trustworthy Proxies: Certifying SAE-Based Interpretability 

**Authors**: Dibyanayan Bandyopadhyay, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2606.18383)  

**Abstract**: Sparse autoencoders (SAEs) are increasingly used to extract interpretable features from language models (LMs), yet a central question remains: when can an SAE-based explanation be treated as a faithful view of an underlying frozen LM We study this through a post-hoc generalization framework that certifies the LM via a sparse proxy, obtained by replacing a native hidden activation with its pretrained SAE reconstruction. Our framework derives an upper bound on the base model's expected risk using four measurable quantities: proxy risk, SAE reconstruction gap, concept-pool mismatch, and sparse complexity. We interpret this certificate as an operational criterion for explanatory faithfulness. In particular, a non-vacuous bound indicates that the extracted sparse features retain meaningful predictive information, while small reconstruction and mismatch errors indicate that the proxy remains behaviorally close to the original model. Empirically, we show that the bound becomes non-vacuous on GPT-2 Small, Gemma-2B, and Llama-3-8B at practical sample sizes. A detailed layerwise analysis of Llama-3-8B reveals a strong depth dependence, with later layers becoming much easier to certify, associated with both stronger local fidelity and weaker downstream error amplification. Finally, through feature-shuffling ablations, we show that the decomposition distinguishes genuine semantic alignment from mere statistical sparsity, providing a useful diagnostic for when SAE-based explanations become less reliable. 

---
# Breaking the Solver Bottleneck: Training Task Generators at the Learnable Frontier 

**Authors**: Lorenz Wolf, Connor Watts, Roger Creus Castanyer, Geoffrey Bradway, Maxwill Lin, Augustine N. Mavor-Parker, Matthew Daborn-Sargent  

**Link**: [PDF](https://arxiv.org/pdf/2606.18284)  

**Abstract**: The limiting resource for training agents via reinforcement learning (RL) is increasingly frontier task supply: valid, solvable tasks just difficult enough to train the current model. As reasoning and agentic models improve, fixed task distributions saturate, while naive synthetic generation yields tasks that are trivial, impossible, or ill-posed. Training a task generator with RL to optimize validity and learnability can address this bottleneck, but direct optimization requires repeated solver rollouts per candidate. For software-engineering (SWE) tasks, a single rollout can take tens of minutes; solver-in-the-loop generator training is intractable. We introduce PROPEL, a solver-amortized framework for training task generators at the targeted solve rate. PROPEL trains a lightweight activation probe on a one-time labeled corpus of generated tasks and solver outcomes. The probe predicts target-solver pass rate from a frozen generator reference model and serves as a proxy for solve rate during generator optimization, reducing generator evaluation to a single forward pass. Across math, code, and software-engineering at multiple model scales, PROPEL shifts generation toward the targeted solve rate: for coding, tasks generated at the learnable frontier increase from $10.1\% \rightarrow 20.0\%$ for a Qwen2.5-3B-Instruct solver and from $5.3\% \rightarrow 12.6\%$ for a Qwen2.5-7B-Instruct solver. For SWE, PROPEL increases the share of generations at the targeted solve rate from $9.8\% \rightarrow 19.6\%$ for Qwen3.5-27B on repositories not seen during training of probe and generator. 

---
# Simulating Hate Speech Cascades with Multi-LLM Agents: Empirical Grounding, Modeling Fidelity, and Intervention Strategies 

**Authors**: Fan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.18264)  

**Abstract**: Faithful modeling of hateful content propagation on online platforms remains an open problem for moderation research. Classical cascade models that do not explicitly represent the profile, community, and content factors associated with hateful-content propagation may yield moderation strategies that behave less effectively when deployed in real-world scenarios. Multi-agent large language model (LLM) systems can, in principle, make each reshare decision depend on the user's profile, the surrounding community, and the post's content, but it remains unclear whether this added flexibility actually reproduces real hateful cascades more faithfully than classical baselines. We study three hateful Bluesky cascades and a size-matched benign control. In the empirical Bluesky data, we found that: 97.4--99.7\% of reposters take a hostile stance; toxicity-engagement homophily is higher on the diffusion tree than on the follower graph for hateful cascades; topology is star-like for the hateful cascades (most reposts come directly from the root) versus tree-like for the benign cascade (reposts propagate through multi-hop chains). In simulation, a multi-LLM-agent simulator reproduces the stance monoculture and the toxicity-delta direction. A structured ablation identifies agent heterogeneity as the leading fidelity factor, and amplifier targeting on dense networks yields 7.5--12.9\% reduction at 5.7\% benign collateral. 

---
