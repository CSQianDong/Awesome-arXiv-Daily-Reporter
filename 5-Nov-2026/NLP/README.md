# SpecCompact: Accelerating LLM Agent Context Compaction with Slot-Parallel Semantic Speculative Verification 

**Authors**: Yulu An, Yifei Feng, Xiangyang Yin  

**Link**: [PDF](https://arxiv.org/pdf/2611.06876)  

**Abstract**: LLM agents use context compaction to manage growing interaction histories, but summarizing long histories with large models can introduce substantial delays. Compaction therefore requires balancing information retention against latency. We observe that structured summary slots can retain information when generated independently. Based on this observation, we present SpecCompact, a framework that accelerates compaction through slot-parallel semantic speculation. SpecCompact fine-tunes a small drafter and generates slots in parallel. The target model verifies each proposed line and repairs rejected content, while subsequent drafting uses the updated slot prefix. System optimizations further reduce repeated context processing and model calls. Across AppWorld and $\tau^3$ benchmarks with two target models, SpecCompact achieves $2.04$--$5.63\times$ compaction speedups over Hermes-style summarization. Separate downstream evaluations reveal model- and task-dependent performance trade-offs. A user study within Hermes further shows an $8.68\times$ median compaction speedup. 

---
# Balancing hierarchical representation and quantization: A performance trade-off in text embeddings 

**Authors**: Kai Ponel, Marco Polignano, Erasmo Purificato, Aylin Julia Temizkan, Marco Levantesi, Giovanni Semeraro, Ernesto William De Luca  

**Link**: [PDF](https://arxiv.org/pdf/2611.06859)  

**Abstract**: Hierarchical text embedding models offer variable-length representations, while quantization reduces numerical precision; both aim to improve computational efficiency. This work is an empirical benchmark study of how these two orthogonal, already-established techniques interact. We evaluate 38 tasks from the Massive Text Embedding Benchmark (MTEB) using four transformer-based embedding models, covering 1D and 2D hierarchical representations combined with 32-bit, 8-bit, and binary quantization, for 168 configurations per task. We report each efficiency dimension separately: theoretical per-vector storage and end-to-end evaluation time. Under a Friedman test with Nemenyi post-hoc analysis, int8 quantization is statistically indistinguishable from float32 for pair classification and re-ranking and costs at most 1.5 percentage points overall, whereas binary quantization is significantly worse across every task category. Halving the embedding dimension once is not statistically detectable above $m=16$; two successive halvings always are. Combining an 8x dimensionality reduction with int8 yields a 32x reduction in per-vector storage at 94.4% of baseline effectiveness. Because evaluation times were measured once, we treat them as indicative: we find no evidence that quantization speeds up evaluation, whereas sampling from intermediate layers shortens it substantially. Retrieval degrades far faster than every other category, so we report findings per task category. 

---
# Persona Perspectives: Adopting Viewpoints Reorganizes Representations in LLMs 

**Authors**: Austin C. Kozlowski, Avital Mintz, Madison Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.06848)  

**Abstract**: Common methods for monitoring features in large language models (LLMs) such as linear probes and sparse autoencoders (SAEs) assume that features maintain a consistent representation across contexts. LLMs, however, can flexibly adopt different perspectives or "personas" that suggest different underlying systems of conceptual relations. In this paper, we examine whether prompting an LLM to adopt a persona alters the representational geometry of the model's latent space. We extract representations of over 9,000 common words and 33 semantic axes from models adopting liberal and conservative political personas and compare these against two neutral baselines. We find that persona conditioning systematically alters both the projections of words onto semantic axes and the alignments among the axes themselves in directions consistent with the specified ideology. We then test the implications of this reorganization for two safety-relevant features: truthfulness and morality. Linear probes, SAE-based classifiers, and classifiers built from manually selected word representations, all trained without a persona, produce divergent results when applied to models conditioned on political personas, producing politically slanted ratings of truth and morality on controversial statements. These findings suggest that internal representations, as they are conventionally extracted, can exhibit different geometric relations when the model adopts personas, potentially undermining monitoring efforts. 

---
# Before You Continue: Evaluating Time-Sensitive Safety Awareness in Conversational Assistants 

**Authors**: Peidi Dong, Yunze Xiao, Yu-Lin Tsai, Yichen Zheng, Heyuan Huang, Weihao Xuan, Mark Dredze, Jen-tse Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.06839)  

**Abstract**: Large language model (LLM) assistants increasingly help users with everyday activities that involve time-dependent constraints. However, users may not mention these constraints in later requests, requiring assistants to recognize and adjust their advice accordingly. To evaluate such capability, we introduce TASC (Temporal Action Safety in Conversation), a benchmark that evaluates whether LLMs apply time-dependent constraints when responding to conversational requests. TASC consists of 3,418 inputs derived from 128 templates, along with a separate set of 33 scenarios designed to compare models' ability to calculate elapsed time with their ability to apply that information. Across 13 models, assistants account for time-dependent constraints in only 30.5% of cases where the proposed action violates a rule, while rarely warning users unnecessarily. More importantly, models fail to apply these constraints in 62.1% of cases despite correctly calculating the elapsed time. Explicitly providing the timing resolves only about half of these failures, revealing a gap between temporal reasoning and its application in conversational advice. 

---
# LooM: Loop-Invariant Memory for Causal Encoder-Decoder Language Models 

**Authors**: Lele Yang, Muxi Diao, Zhonghao Yan, Jiankang Chen, Bin Wen, Tingting Gao, Wentao Zhang, Kongming Liang, Zhanyu Ma, Weiran Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.06835)  

**Abstract**: Looped Transformers increase effective depth by re-executing shared layers, but three limitations hinder their adoption: memory grows with the loop count, compute can only be adjusted in whole-block steps, and the loop count is hard to change after training. We propose LooM (Loop-invariant Memory), which addresses each in turn. In a causal encoder-decoder (CED), LooM loops only the decoder layers after the first, which read a global memory written once per token, and each of them publishes the key-value (KV) entry for a token only on its first visit, so the cache does not grow with the loop count. These layers loop independently, providing every budget from 3 to 21 layer executions per token, and one model trained on randomly sampled budgets covers all of them without a router. With enough loops, LooM outperforms a single-pass baseline trained on the same data on language modeling, zero-shot accuracy and long-context perplexity, while using the same cache size. Our analysis further shows that the training budget distribution decides what extra depth buys. These results suggest that a CED is a natural place to loop: memory is written once, and depth becomes a budget chosen at deployment. 

---
# SAUCE: Summary Analysis Using Conversation Entailment 

**Authors**: Man-Ling Sung, Hemanth Kandula, Jeff Ma, William Hartmann, Matthew Snover  

**Link**: [PDF](https://arxiv.org/pdf/2611.06812)  

**Abstract**: With the growing need for evaluating Large Language Models (LLMs) and their applications to speech, challenges persist in summarizing and evaluating conversations that lack a clear end goal. We introduce SAUCE -- a reference-free, fact-based evaluation pipeline for cross-lingual conversational speech summarization. It measures the accuracy and the fact coverage of a summary through the entailment between conversation and text. We compare SAUCE against several popular summarization metrics and demonstrate the effectiveness of capturing information loss due to transcription and translation error and identifying broken summaries. Crucially, unlike black-box LLM evaluators or dense embedding metrics, SAUCE is inherently explainable: it maps summary scores to discrete, verifiable facts, allowing users to pinpoint exact hallucinations or omissions. We illustrate how this interpretability helps developers systematically profile LLM behaviors and gives end-users an actionable tool to verify summary accuracy in noisy, real-world conditions. Preliminary investigations show SAUCE strongly align with human judgment. 

---
# LLMs Don't Know What They Know: Benchmarking Temporal Self-Knowledge for Financial Backtesting 

**Authors**: Lingfei Qian, Yan Wang, Xueqing Peng, Yupeng Cao, Yuechen Jiang, Zhuohan Xie, Anke Xu, Wenbo Cao, Jiarui Chen, Zehan Li, Jimin Huang, Alejandro Lopez-Lira, Yuehua Tang, Sophia Ananiadou, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2611.06808)  

**Abstract**: Financial backtests of large language models (LLMs) are meaningful only if the model did not already know the outcomes it is asked to predict. Practitioners guard against this by trusting a vendor's stated knowledge cutoff, by asking the model what it knows, or by instructing it to answer as of a past date. To verify whether any of these safeguards can be trusted, we introduce FinDendro, a benchmark of 2,226 dated U.S. facts from 1999 to 2026, covering monthly macroeconomic series, month-end stock prices and Federal Reserve decisions. Each fact yields four prompts: a value question, Recall and Predict questions that pair each candidate value with a claim of knowing or guessing without a declared date, and a Predict question with a historical declared date. Controls for persistence and dominant outcomes keep chance accuracy near 0.27. Results show that for most frontier models, self-reports understate knowledge. GPT-5 answers 76% of controlled facts correctly but claims to know 38%, and answers 61% of the facts it says it does not know. Knowledge claims can persist beyond the measured knowledge boundary, by up to seven quarters, and a declared date removes most of the claims of prior knowledge while leaving accuracy unchanged. Backtests inside a model's training window therefore cannot be certified leak-free by self-report, instruction or stated cutoff. These findings motivate behavioral measurement of temporal knowledge boundaries, and FinDendro provides a benchmark and procedure for selecting backtest periods based on demonstrated factual knowledge. 

---
# Measuring Risk, Not Harm: Detecting Psychological Harm Risk Factors in LLM Conversations 

**Authors**: Nil-Jana Akpinar, Eugenia Kim, Natalie Isak, Amanda Minnich, Victoria Westerhoff, Matthew Dressman  

**Link**: [PDF](https://arxiv.org/pdf/2611.06806)  

**Abstract**: Psychological harms from interactions with large language models cannot be monitored directly because they may emerge over repeated interactions and depend strongly on subjective individual user context. We propose measuring risk, not harm: detecting observable AI behaviors that constitute risk factors for psychological harm at the AI response level and aggregating these signals to support longitudinal monitoring. We operationalize eight such behaviors and construct a synthetic multi-label corpus of 47,062 AI responses and 373,873 response-factor labels, with targeted human annotations for validation and evaluation. We then compare trained embedding-based classifiers, direct LLM judging, and a classifier--judge cascade. All detectors are model-agnostic and operate only on LLM outputs, requiring no access to internal model states. We apply these detectors to several out-of-distribution human-AI conversation data sets from prior research. On held-out human-labeled data, the best cascade achieves macro precision of 0.86, recall of 0.76, and F1 of 0.79. On WildChat, it reduces LLM-judge calls by 97.1\% relative to direct judging. On DelusionEval, risk-factor detections cluster within conversations. Together, our results show that response-level behavioral risk-factor detection can provide a scalable measurement layer for longitudinal psychological-safety monitoring without conflating observable AI behavior with latent user harm. 

---
# Ranking Agreement Does Not Reveal Paragraph-Edit Agreement in Two Released Paper Acceptance Predictors 

**Authors**: Shaoliang Yang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.06789)  

**Abstract**: A fast acceptance classifier could select revisions for a slower evaluator. We ask whether agreement on existing papers supports that use. We compare a classifier of word and document features with two released text-only acceptance predictors fine-tuned from the same language model with three billion parameters. For 198 papers from the International Conference on Learning Representations, we compare model rankings of the original papers and of paragraph rewrites screened automatically for unchanged facts, claim strength, and attribution. The classifier ranks original papers closely with one predictor and moderately with the other, but shows no detectable agreement with either on which rewrites raise scores. The tests specified in advance used reduced numerical precision. Rounding error dominated the small score responses. We therefore rescored the released weights in full 32-bit precision as a post hoc check; edit agreement remained near zero. At this precision, when each predictor chooses among five rewrites per paper, its average raw score is under two points out of 100 higher than with random choice. The evidence covers presentation rewrites and two open predictors from one family; transfer to costly evaluators remains untested. Revision selectors should therefore be checked against the intended evaluator on the revisions they rank. 

---
# Reasoning over Non-Parametric Memory in Multilingual LLMs: A Cognitive-Based Analysis of Relational Knowledge 

**Authors**: Zohreh Abbasi Liasi, Batoul Hammoud, Anas Madkoor, Ehsaneddin Asgari  

**Link**: [PDF](https://arxiv.org/pdf/2611.06715)  

**Abstract**: Retrieval-augmented generation (RAG) assumes that language models can reason over relations stated in their context rather than fall back on associations memorized during pretraining. We study this evidence-conditioned relational reasoning in isolation, with the facts supplied directly in the prompt rather than retrieved. \textsc{Rel-Bloom} is a controlled multilingual diagnostic of 450 multiple-choice questions built from 30 Wikidata relations; the same source relations are instantiated at five Bloom-inspired levels (Remember, Understand, Apply, Analyze, Evaluate) and under three veracity conditions (Factual, Counterfactual, Neutralized). Across 43 models, six languages, and two multilingual settings (226,800 predictions), three findings stand out. First, accuracy drops sharply from Apply to Analyze in every evaluated open-weight model series, reaching 47.2 percentage points for the Large Granite bucket, but not for either Gemini model. Second, Evaluate items remain a bottleneck even for models near ceiling on the other task types: Gemini~3.1 Pro averages 99.5\% on the other four levels but 60.7\% on Evaluate. Third, item- and relation-level difficulty rankings are consistently preserved across languages, indicating a difficulty structure partly shared across languages within this benchmark. Across knowledge conditions, Counterfactual contexts are generally harder than Factual ones, while removing entity identities can improve higher-level performance in some model regimes. Greater model capability therefore does not by itself ensure reliable reasoning over supplied relational evidence. Relevant resources are available at: this https URL. 

---
# ALTO: African and Levantine Tech-Facilitated Gender-Based Violence Corpus 

**Authors**: Tianwei Zhang, Ivy Njoki Nyagwencha, Zahraa Dirani, Shalini Joshi, Scott A. Hale  

**Link**: [PDF](https://arxiv.org/pdf/2611.06706)  

**Abstract**: Tech-facilitated gender based violence (TFGBV) is a pervasive online harm affecting women and gender minority groups through its many direct and covert forms. Current models and annotated datasets for gendered harm detection are dominated by data from, and models trained in, Global North languages. In collaboration with civil society organizations (CSOs) from Jordan, Francophone West Africa, and Kenya, we developed a community-led approach to crowdsource culturally and linguistically relevant instances of TFGBV in Levantine Arabic, African French, and Swahili. We annotated the crowdsourced data using active learning to develop a small but performant dataset (N=3022) called ALTO to improve TFGBV detection in the three languages. Through benchmarking and fine-tuning experiments, we show that existing harmful content detection models show poor TFGBV recall within ALTO and improve with fine-tuning. We found that models often miss covert harms across all languages, while error rates are markedly higher in Swahili. We report the experimental results and release the annotation taxonomy, dataset, and trained models. 

---
# Replicability, Reproducibility, and Robustness in Political and Social Science Research with Large Language Models 

**Authors**: Joan C. Timoneda, Sebastián Vallejo Vera  

**Link**: [PDF](https://arxiv.org/pdf/2611.06639)  

**Abstract**: This chapter addresses the central reproducibility problem of research with large language models. LLM output is probabilistic and varies across models and versions, which can have significant effects on downstream statistical results. The chapter distinguishes between replicability, reproducibility, and robustness, and proposes a probabilistic standard based on the distribution of results, not identical outputs. It introduces a standardized prompt script, tests within-model variation for encoders and decoders, and develops error-structure analysis to diagnose where models err and whether their errors align. Using three datasets, the chapter shows that measurement choice can reverse the sign of downstream regression coefficients. Existing design-based corrections with gold-labeled subsamples restore validity, but come at the cost of estimate precision. The main contribution of the chapter is a stacked ensemble of open-weight models that improves precision in the correction step while retaining estimate validity. 

---
# When Do Summaries Become Instructions? Unsupported Policy in LLM Agents' Self-Written Handoff Summaries 

**Authors**: Wenhui Chu  

**Link**: [PDF](https://arxiv.org/pdf/2611.06545)  

**Abstract**: Long-running LLM agents compact their context, replacing most history with a summary. We ask whether handoff summaries state permissions or requirements the record does not support, whether such sentences change what the next instance (the successor) does, and where they come from. Synthetic tool-use tasks carry a ground-truth permission ledger; same-snapshot replay reruns the successor with those whole sentences replaced by neutral filler. Two AI readers, reading only rule-flagged held-out summaries, confirmed unsupported restrictions in 1.2% and 2.4% of all readable summaries from two open-weight models, and unsupported authorizations in 0.1% and 1.3%. For restrictions, a pre-specified replay test was not significant (p = 0.58); a follow-up, planned after it, found a fragile effect (p = 0.027). Neutralizing false approvals cut unauthorized actions from 24 to 7 of 120 held-out runs (post hoc; later set: 17 to 11 of 48). Most development restrictions followed the agent's earlier reasoning that an action needs approval (post hoc). In pre-specified tests, replacing the system prompt's "Some actions need approval." with one saying where requirements come from cut unsupported restrictions from 5.0% to 1.8% of development summaries and, on held-out templates, from 1.8% to 0.3% (both p < 10^-9), while requests for approvals the ledger requires barely changed. Code and data: this https URL 

---
# Dialect Matters: A Dialect-Aware Baseline for Igala-English Machine Translation 

**Authors**: Etuh E., Eli. A. Jiya  

**Link**: [PDF](https://arxiv.org/pdf/2611.06517)  

**Abstract**: The massive permeability of intelligent systems in every sphere of human activities have greatly enhanced clarity, ease of work, and increased general global understanding which has significantly awaken the use of intelligent systems by all users. Intelligent systems greatly rely on precise dataset/corpus for the correctness of their output. Despite efforts by researchers to address Low-resource language in the Natural Language Processing (NLP), Igala still has zero parallel corpora despite efforts by researchers. We present a dialect-aware baseline for Igala-English machine translation, among first open systems and first with dialectal variation analysis. Our dataset of 1137 sentences (899 training) is first with dialect tags: Idah 459 (40.4%), Ankpa 297 (26.1%), Ibaji 160 (14.1%), Other 187 (16.4%), Ogugu 32 (2.8%). Same English yields 20-60 Igala forms. Mixed-dialect training achieves balanced BLEU 11.66 versus Idah-only model which drops 4.7 BLEU on Ibaji (13.2 on Idah, 8.5 on Ibaji), proving dialect fairness requires mixing. Fine-tuned from Helsinki-NLP/opus-mt-en-mul (78M), epoch ablation shows convergence at 8 epochs and overfitting at 15. BLEU 11.66, chrF 46.69. Analysis shows 899 sentences on log-linear trend consistent with African low-resource literature. This work provides first dialectal impact analysis for Igala and establishes baseline for future work. 

---
# NaijaFacts: A Framework for Disentangling Language from Knowledge in Facts about Nigeria 

**Authors**: Zainab Afolabi, Nikita Kozodoi, Jack Butler, Babs Khalidson  

**Link**: [PDF](https://arxiv.org/pdf/2611.06504)  

**Abstract**: Multilingual benchmarks for African languages mostly translate generic questions into local languages, measuring linguistic competence but not whether models know facts about Africa. We introduce \naija, a benchmark of factual and statistical questions about Nigeria, authored by native speakers and grounded in authoritative official statistics, with every question available in English, Nigerian Pidgin, \yoruba, Igbo, and Hausa. Answering similar questions for the UK and US yields a country by language matrix that disentangles a language effect (the model knows the fact but degrades in a low-resource language) from a knowledge effect (the fact is absent in every language). Across eight frontier and Nigeria-adapted models, the knowledge effect dominates: English-query accuracy is 49--50\% for the US/UK but 20\% for Nigeria. The language effect is smaller and tracks web representation, degrading from English to \yoruba{} (9 points on Nigeria, up to 24 on US/UK questions). We find scale does not help: the strongest models reach 66--79\% on the UK and the US but none exceed 31\% on Nigerian statistics. Extending to all 54 African countries with our best performing frontier model shows the same profile continent-wide, 96\% on culture and 98\% on politics against 26\% on statistics, so the gap is not specific to Nigeria. 

---
# CRESiST: Self-Enhancing Explorative Policy Learning for Simultaneous Speech Translation 

**Authors**: Chenyang Le, Pei Zhang, Yuxiang Chen, Bing Han, Derek F. Wong, Baosong Yang, Yanmin Qian  

**Link**: [PDF](https://arxiv.org/pdf/2611.06501)  

**Abstract**: Simultaneous Speech Translation (SimulST) requires balancing translation quality and latency. While Large Language Models (LLMs) excel in offline translation, current LLM-based SimulST systems predominantly rely on prefix-incremental generation. This paradigm suffers from redundant Key-Value cache recomputation and necessitates external Voice Activity Detection to process unbounded streams. To overcome these bottlenecks, we propose an end-to-end interleaved read-write framework built upon an omni-modal LLM. By interleaving speech chunks and text tokens, our architecture achieves full KV cache reuse, natively supporting segmentation-free, infinite-length inference. Furthermore, to learn optimal streaming policies without rigid offline alignments, we introduce a novel Actor-Critic self-enhancement pipeline. An RNN-T Actor dynamically samples read-write trajectories, while an LLM-based Critic evaluates these paths via intrinsic semantic and latency rewards, distilling the optimal policy back to the Actor. Extensive experiments on sentence-level (FLEURS, CoVoST2) and document-level (MuST-C, ACL 60/60) benchmarks demonstrate that our model establishes new state-of-the-art quality-latency Pareto frontiers, substantially outperforming existing cascaded and static-alignment baselines. 

---
# LibriSpeech-Real-QA: A Benchmark for Streaming Question Answering 

**Authors**: Zewen Zhan, Hui Lu, Xueyuan Chen, Huimeng Wang, Yikai Huang, Xixin Wu, Shiyin Kang, Bin Xia, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.06496)  

**Abstract**: Most speech large language models (SLMs) still process speech in an offline manner. They only start responding after they receive the complete speech segment. However, many real-world applications require streaming speech interaction, for which the audio continuously flows in and the machine is required to answer questions based on the so far received audio chunks. For example, a user may ask the system to continuously count how many times a person's name has been mentioned while they are speaking. We observe a lack of benchmarks and evaluation protocols for real-time speech interaction. Motivated by this, we propose LibriSpeech-Real-QA, a dataset and a benchmark for real-time speech interaction. We construct different types of real-time question-answer pairs from LibriSpeech to evaluate whether speech LLMs can identify target words in streaming audio. We evaluate several popular real-time commercial models and find that they still tend to respond only after the speech turn has finished. We demonstrate that we can enable a pretrained SLM to perform this task, where we explore several modeling designs and establish a solid baseline for future research. 

---
# ArcRAG: Agentic Compositional Retrieval over Knowledge Graphs 

**Authors**: Jyotin Goel, Abhinav Arun, Tejas Prakash Agarwal, Bhaskarjit Sarmah, Stefano Pasquali  

**Link**: [PDF](https://arxiv.org/pdf/2611.06481)  

**Abstract**: Graph-based retrieval-augmented generation (GraphRAG) answers questions by retrieving over a knowledge graph built from a corpus. Yet retrieval strategies are designed around the structure their own construction pipeline produces, and whether they survive a change of construction is rarely tested. We study this cross-construction portability: whether one retrieval procedure, with no construction-specific schema, relation names, or retuning, stays effective when the graph is built differently. We introduce ArcRAG, which induces a query-local graph interface at inference time (the node labels, relation descriptors, and property keys reachable from the entities a question names), plans multi-hop traversals against it, and executes and repairs them as Cypher. We build each of the MuSiQue, HotpotQA, and TriviaQA corpora as a schema-guided graph (typed nodes, a closed relation set, source provenance) and as an open-information-extraction graph (untyped nodes, free-text relations), with the retriever's configuration fixed across them. We compare against a construction-agnostic Direct-Cypher agent, ablate each stage across retry and pass budgets on both graph families, and compare with results reported for prior GraphRAG systems, most of which design retrieval together with their own graph construction. Across the two constructions, we find that retrieval need not be co-designed with the graph: query-time interface induction keeps a construction-agnostic retriever effective when the construction changes, although construction still shapes absolute accuracy. 

---
# EvoGuard: Mitigating Degradation in Online Skill Evolution 

**Authors**: Yanli Wang, Zichen Bai, Tianyue Jiang, Kunming Shao, Junhao Shi, Xueqin Liu, Jinguo Sun, Muye Li, Tingting Bi, Yanlin Wang, Shaoqiang Zheng, Xuefei Diao, Zhuwei Zeng, Nan Duan, Peng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2611.06444)  

**Abstract**: Skills let language agents reuse experience across tasks as natural-language documents. Recent work evolves skills from interaction, but existing methods have limitations when skills evolve online. Prior methods update a skill offline from a batch of trajectories collected over many tasks. Some then adopt the new skill without any check, which can degrade the skill over time. Others keep it only if it improves performance on a held-out validation set, which is too slow and costly to run after every update. We introduce EvoGuard, a pairwise skill selector that mitigates skill degradation at low latency and cost. We first run the agent iteratively to build pools of skills and score each skill on a private test set to obtain utility labels. We pair the skills and use rejection sampling to keep only pairs with a clear utility gap. The selector is trained on the logits of two label tokens to pick the better skill. At deployment it decides in a single forward pass, which is significantly faster than evaluation on the private test set. Compared with adopting every new skill, EvoGuard improves success by 11.0% on ALFWorld, 5.9% on ScienceWorld, and 2.7% on LiveCodeBench. 

---
# Twists and Turns: A Graph-Based Benchmark and Framework for Long-Term Memory in Exploration Tasks 

**Authors**: Chongyuan Dai, Yaling Shen, Zirui Song, Junfeng Liao, Jinpeng Hu, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.06414)  

**Abstract**: Long-term problem-solving processes, such as scientific research and software development, proceed along an evolving trial-and-error trajectory toward a central goal. We term these \textit{exploration tasks}, which require models to remember not only facts but also how solutions are tested against intermediate outcomes and early attempts are continually revised or revived. However, existing long-term memory research focuses on factual recall and updates, overlooking the interconnected nature of exploration. We observe that exploration processes naturally form graphs, with branching approaches linked by revisions and reconsiderations. Motivated by this, we propose \textsc{LExMem}, a graph-guided benchmark designed to evaluate \underline{\textsc{Mem}}ory in \underline{\textsc{L}}ong-term \underline{\textsc{Ex}}ploration tasks. We model exploration processes as graphs and introduce an evolutionary mechanism to increase their structural complexity, yielding 224 dialogues and 2,018 question-answer instances derived from graph structures. Building on the same insight, we propose \textsc{LExGraph}, a graph-based memory framework that incrementally organizes dialogues into interconnected memory nodes and retrieves evidence through hybrid retrieval and neighbor expansion. Extensive experiments demonstrate that \textsc{LExMem} poses a significant challenge to existing retrieval- and memory-based systems, whereas \textsc{LExGraph} achieves superior performance and generalizes effectively to external benchmarks. Codes and datasets are available at this https URL. 

---
# Ruby-OPD: Enhancing Pronunciation Controllability and Accuracy for Flow-Matching Text-to-Speech via On-Policy Distillation 

**Authors**: Rui Hu, Xiaolong Lin, Zhenpeng Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2611.06403)  

**Abstract**: Achieving both high pronunciation accuracy and fine-grained controllability remains a central challenge in Japanese text-to-speech (TTS), as Japanese is an unsegmented language mixing logographic Kanji and phonographic Kana with extensive polyphony. Traditional cascaded systems convert text into phoneme sequences via a front-end, ensuring pronunciation control but discarding lexical semantics and prosodic cues in surface orthography. Conversely, recent zero-shot TTS models directly consume raw orthographic text to improve naturalness, yet suffer from polyphone errors and lack explicit pronunciation control. To bridge this gap, we propose a two-stage framework for flow-matching TTS. First, we investigate Ruby text representations with orthography perturbation to jointly preserve surface forms and target pronunciations for reliable control. Second, we introduce On-Policy Distillation (OPD) on large-scale text-only corpora to enhance pronunciation accuracy on raw orthographic text without paired speech. On the Joyo-Kanji-Yomi benchmark, our model achieves 97.13% Accuracy, 2.02% Target PER, 0.70% Sentence PER, and 4.21% Sentence CER under raw text while retaining strong controllability. 

---
# Interpretable Text Embeddings Based on Hierarchical Questions 

**Authors**: Kohei Oda, Po-Min Chuang, Kiyoaki Shirai, Natthawut Kertkeidkachorn  

**Link**: [PDF](https://arxiv.org/pdf/2611.06402)  

**Abstract**: QA-based text embeddings represent texts as vectors consisting of 1/0 answers to yes/no questions. Although these embeddings provide detailed information for interpretation and achieve high task performance, they often require humans to examine a large number of questions. Furthermore, the relationship between the number of questions and the appropriate level of question granularity has not been sufficiently studied. To address these challenges, we propose a method for constructing hierarchical questions for text embeddings. As the hierarchy deepens, questions become more specific and their number increases. During interpretation, we reduce the number of questions that humans need to examine by replacing questions at deeper levels with aligned questions at shallower levels. The question hierarchy is constructed through recursive clustering, contrastive prompting, and question alignment. Experimental results on general-purpose tasks show that our method improves interpretability while maintaining task performance. It is also found that, when only a limited number of questions are available, questions at shallower levels tend to achieve higher task performance than those at deeper levels, even with the same number of questions. 

---
# Where Should a KV-Cache Repair Budget Go? Distractor Structure and Contextual Compatibility in Non-Prefix Reuse 

**Authors**: Yuxing Peng, Zhiqing Tang, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2611.06397)  

**Abstract**: Non-prefix KV-cache reuse encodes retrieved chunks independently and concatenates their caches. To find where a small repair budget should be spent, we vary the chunks around the answer-bearing one (the target) across three 7B-8B models and four datasets of dialogue memory, open-domain QA and API documentation. Reuse loses 19-100% of the context-dependent questions that full prefill answers, and it loses more when the other chunks are of the same kind as the target. Because the recoverable damage is spread over these other chunks, restoring a few of them recovers little, and the budget is better spent on the target. This works through contextual compatibility: recomputing the target against the reused cache beats inserting its full-prefill KV in all seven settings tested, although the recomputed KV stays far from the full-prefill KV. Question attention over reused caches, however, selects the target in at most 18.8% of development questions. A retriever finds it more often, and recomputing its top chunk after moving it next to the question significantly beats InfoFlow-style reordering at the same 5% budget in six of nine open-domain QA and API-documentation settings. On dialogue memory, though, recomputing chunks never significantly beats the token-level selector chosen for each setting. 

---
# Presentation Sensitivity in Typed Decision Models for Subjective Text Evaluation 

**Authors**: Jatin Agrawal, Radhika Mamidi  

**Link**: [PDF](https://arxiv.org/pdf/2611.06384)  

**Abstract**: Typed decision models offer structured choices, scores, and probabilities for low-cost automated evaluation. We present a controlled audit of presentation sensitivity in Jev, Clef Flash, and Perplexity Decider across native choice and score interfaces. Across six datasets with 9,000 multiply annotated humor, helpfulness, and summarization items, we collect 702,000 judgments under interventions that vary positions, identifiers, authorship claims, endorsements, decision framing, and rating rubrics while preserving the evaluated texts. Jev's second-position probability advantage reaches 6.11 percentage points on summaries; reversing reader endorsements changes Perplexity's selected humor text on 40.40% of pairs; and rejection framing reduces Clef's summary agreement with human preferences by 23.36 points. Rating formats also shift score levels while often preserving rankings. Across these findings, repeatability, presentation robustness, and human alignment emerge as distinct properties of an evaluator. Reliable use therefore requires validating the decisions and score interpretations on which an evaluation pipeline depends. 

---
# Training-Free Character-Length Control in Summarization with Diffusion Language Models 

**Authors**: Steffen Castle, Nils Feldhus, Christopher Ebert, Leonhard Hennig, Sebastian Möller  

**Link**: [PDF](https://arxiv.org/pdf/2611.06371)  

**Abstract**: Summarization with a fixed character length is important for tasks such as headline and article preview generation, where text must fit an exact space. Autoregressive language models (LMs) struggle with precise character-length control: training-based methods require task-specific fine-tuning, while zero-shot prompting strategies achieve only approximate compliance. We propose DELTA (Diffusion Expected Length Targeting Algorithm), a training-free inference-time method for masked diffusion LMs that estimates expected character length at each denoising step and dynamically inserts or removes masked positions to steer toward a target length. DELTA requires no fine-tuning and is designed for off-the-shelf masked diffusion LMs. Across three summarization benchmarks, it reduces character-length errors by an order of magnitude compared to baselines while maintaining competitive summary quality in both automatic and human evaluation. 

---
# Mass Producing Training Data with Surprising Side Effects 

**Authors**: Yaowen Ye, Jacob Steinhardt, Sarah Schwettmann, Cassidy Laidlaw  

**Link**: [PDF](https://arxiv.org/pdf/2611.06324)  

**Abstract**: Training data can have unexpected side effects on language model behavior outside the training domain: insecure code data can induce broad misalignment, and outdated bird names can induce a 19th-century persona. Understanding such generalization is hard because only dozens of isolated examples exist, each constructed by hand or discovered by accident. We propose automatically discovering such datasets with surprising side effects (DSSEs). We define three objectives for DSSEs: they should (1) elicit a target behavior, (2) read like ordinary post-training data, and (3) be surprising to human auditors reading the data. We operationalize each with LLM judges, and search for high-scoring datasets via evolution, reinforcement learning, and subliminal learning approaches. After re-scoring and filtering with held-out metrics, our pipeline produces a collection of 6,575 DSSEs across 8 behaviors, many of which generalize across model families and sizes. We find our automated metrics track ML researchers' judgments, and frontier models outperform humans at predicting side effects. These datasets form a testbed for studying generalization and lay the groundwork for predicting training side effects before model deployment. Explore them at this https URL. 

---
# Zuse: A Decision Model for European Languages 

**Authors**: Umair Mehmood Imam  

**Link**: [PDF](https://arxiv.org/pdf/2611.06238)  

**Abstract**: Decision models read a document and return a probability for each of a fixed set of options instead of generating text. Several appeared in 2026, but none has published results outside English. We describe Zuse, three decision models (4B, 9B and 27B) that train a low-rank adapter and a pointer head on frozen open-weight bases, targeting 27 European languages, and use them to study how this design behaves across languages and tasks. On a broad European suite of 11 sources excluded from training, covering 9 task groups and 27 languages, a development set that also guided model selection, the 27B model reaches 87.5% accuracy with a pooled expected calibration error of 0.016, and 0.035 when weighted per source. Post-hoc analyses of these single-seed results give three findings. First, miscalibration is structured by task rather than language: on parallel reading and topic items in 26 languages the confidence gap varies about as much as sampling noise, while across sources it spans -0.07 to +0.09, with knowledge tasks overconfident and acceptability judgements underconfident at all three sizes, so one fitted temperature cannot correct both. Second, the share of decisions that can be automated at a 5% error budget follows chance-corrected skill (Spearman 0.97 over 33 source and model pairs) more closely than raw accuracy (0.91). Third, reach beyond the training languages is set by the base model: across 210 languages, changing the training run moved scores by at most about 2 points within a size, while languages outside the base's documented list trail by 23 to 28 points at every size. On English public suites Zuse is behind Jev and Kev, including Kev models on the same bases. We also report two pitfalls in building multi-source training data: label-word collision between topic schemes, and an injection augmentation that taught a shortcut. Zuse can be tried at this https URL. 

---
# Loop Before You Leap: What Recurrent Depth Can and Cannot Buy in Data-Limited Pretraining 

**Authors**: Daichi Hotta, Yohei Kobashi, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2611.06206)  

**Abstract**: Multimodal models have often scored below text-only baselines in the BabyLM Challenge (Klerings et al., 2024; Takmaz et al., 2025). We test whether additional recurrent computation helps close this gap. Recurrence increases computation by reusing layers without adding parameters or training data. Using a single implementation, we compare standard and looped Transformers under a 100M-word budget, with three seeds per main condition. Each image is represented by one feature vector from a frozen DINOv2 encoder, used as a prefix token. Caption words count toward the budget, so multimodal training also changes the language data. In a pre-registered text-only comparison, a 12-layer stack repeated four times exceeds a standard 48-layer Transformer by 5.4 BLiMP percentage points at matched inference compute. However, multimodal training lowers BLiMP accuracy by about 3.8 points in both architectures, with no clear mitigation from recurrence. An exploratory loop-side no-image control, trained with the multimodal language data and batching schedule, scores 5.7 points below the text-only baseline. Adding the visual prefix raises accuracy by 1.85 points on average, but this contrast is uncertain with three seeds. These results suggest that changes in language data and its presentation are a major source of the drop. Diagnostic tests show that the models use visual information; its contribution to grammatical generalization remains uncertain under this single-vector interface. 

---
# Editing Large Language Models with Geometry-Aware Regularization 

**Authors**: Bingqing Liu, Wei Liu, Jun Wang, Xiaobo Sun, Yuhua Li, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.06185)  

**Abstract**: Knowledge editing has become a key technique for updating knowledge in large language models. However, existing methods neglect the geometric structure of the knowledge representation space, leading to two key limitations: they cannot distinguish which pre-trained knowledge should be preferentially preserved (core vs. non-core) nor which new edits are inherently difficult (hard vs. easy). To address this, we propose GEdit, a geometry-aware framework built on a Low-Rank Plus Diagonal (LR+D) factor model that captures the intrinsic low-rank structure of Transformer FFN hidden representations. GEdit features two complementary mechanisms. First, the low-rank regularization applies targeted protection on a low-rank core knowledge subspace. Second, the adaptive regularization dynamically adjusts the regularization strength based on each edit request's editability: relaxing constraints for hard edits to ensure success, and tightening them for easy edits to better preserve pretrained knowledge. Extensive experiments on GPT2-XL, GPT-J, and LLaMA-3-8B under large-scale editing scenarios, including both mini-batch and fully sequential editing settings with up to 10,000 continuous updates, show that GEdit consistently and significantly outperforms state-of-the-art baselines in both editing success and knowledge preservation. 

---
# AssameseLM-E2B: An Open-Weight Assamese Language Model for Efficient Consumer-Grade Deployment 

**Authors**: Pragyan Phukan  

**Link**: [PDF](https://arxiv.org/pdf/2611.06170)  

**Abstract**: While Large Language Models (LLMs) excel in high-resource languages, their performance often degrades in low-resource languages due to tokenization inefficiencies, generative instability, and linguistic interference. This paper presents AssameseLM-E2B and its instruction-tuned variant, AssameseLM-E2B-it, a 2-billion-parameter model adapted specifically for low resource Assamese language. Built upon the Gemma-4 architecture, the model incorporates an expanded byte-level BPE vocabulary to reduce tokenization overhead by nearly 500% compared with multilingual 2B--3B baselines. We employ a parameter-efficient Continual Pre-Training (CPT) pipeline with Hybrid LoRA, followed by Supervised Fine-Tuning (SFT) on a balanced, multi-domain dataset to enhance instruction-following and Retrieval-Augmented Generation (RAG) capabilities.
Evaluation demonstrates a Bits-per-Byte (BPB) score of 0.7488 on FLORES-200, alongside an overall RAG score of 96.7% and 100% refusal accuracy on adversarial unanswerable prompts. Comparative analysis reveals significant limitations in general-purpose baselines, including receptive-productive decoupling, generative degradation on Romanized Assamese, and unintended Bengali script assimilation. These results demonstrate the potential of parameter-efficient adaptation for developing capable, locally deployable language models for low-resource languages under constrained computational resources. 

---
# Accelerating Long-Context Decoding in Large-Scale Hybrid LLMs Without Retraining 

**Authors**: João Eduardo Batista, Emil Vatai, Mohamed Wahib  

**Link**: [PDF](https://arxiv.org/pdf/2611.06133)  

**Abstract**: The full-attention layers of large language models (LLMs) make every generated token more expensive than the previous one, slowing models down on long-context tasks. Hybrid models reduce this cost by replacing most full-attention layers with linear-attention layers, but the first still dominate decoding at long contexts. We accelerate these remaining layers without any training or weight changes using combined sparse attention: each layer attends to a sliding window of recent tokens, a pinned prefix, and a fixed budget of most relevant tokens in the context. The window size and budget are found on a per-layer basis using a multi-objective search algorithm. Across six hybrid models from three families, from 4B to 397B parameters, the selected configurations preserve their baseline model's accuracy on 21 long-context datasets. Decoding 64k/256k tokens becomes up to 1.26x/2.22x faster on Qwen3.5-4B, 1.20x/2.23x on Qwen3.5-122B-A10B, and 1.16x/2.06x on Ornith-1.0-397B. 

---
# The Harness Is the Model: Agent Scaffolding Changes a Fixed LLM's Compliance with Banking Conduct Rules, Unevenly Across Models 

**Authors**: Mehmet Murat Huyut  

**Link**: [PDF](https://arxiv.org/pdf/2611.06116)  

**Abstract**: Banks validate language models, but they deploy systems: a model inside a harness that decides what it sees, which tools it can call and who checks its actions. We test whether that harness changes compliance with banking conduct rules when the model is held fixed. Four open-weight models each ran five harnesses (a pipeline that makes one structured decision, a ReAct-style tool loop, a loop in which the policy is reachable only through search, a loop with a long benign customer memory, and a loop with a same-model compliance reviewer) on 156 pre-registered cases in a simulated retail bank whose 11 rule families paraphrase EU consumer-credit, payments, investor-protection, data-protection and anti-money-laundering provisions. Every one of the 3120 two-turn episodes was scored deterministically from the executed tool trace, with violations and over-refusals counted separately. Pooled across models the harness moved the correct-case rate only between 87.3% and 92.1%, but within a model it mattered a great deal: for one model the spread between its best and worst harness was 11.5 percentage points, enough to move it from first to last place among the four models, while for another the harness made no detectable difference; the model x harness interaction itself was suggestive rather than conclusive. The same-model reviewer cut violations from 9.6% to 6.4% without raising over-refusals, contrary to our pre-registered prediction. Rules that require date arithmetic failed in every harness. In an exploratory analysis with two LLM judges and a blind human check, 26% of violations were accompanied by a customer message that denied the action the agent had just executed, which a transcript-only review would record as a correct refusal. For model-risk management the unit of validation should be the model-harness pair, and conduct testing should read the action log, not the conversation. 

---
# Overview of the NTCIR-19 SciClaimEval Task: Cross-modal Claim Verification in Scientific Papers 

**Authors**: Xanh Ho, André Greiner-Petter, Florian Boudin, Yun-Ang Wu, Sunisth Kumar, Tian Cheng Xia, Hiba Akbi, Tim Schopf, Akiko Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2611.06090)  

**Abstract**: This paper presents an overview of \emph{SciClaimEval}, a pilot task at NTCIR-19 on verifying scientific claims against the tables and figures of the papers in which they appear. Our dataset pairs authentic claims from 180 papers in machine learning, natural language processing, and medicine with contrastive versions of their tables and figures, one supporting the claim and one minimally modified so that it does not. The task comprises two subtasks: deciding whether a piece of evidence supports a claim, and selecting which of two evidence versions does. Twenty-two teams registered, the highest number among NTCIR-19 tasks, and 14 submitted 177 runs. Most systems built on multimodal large language models, relying on prompting and multi-stage inference rather than task-specific fine-tuning. On the first subtask, the best systems reached 98.1\% pair accuracy, but all six top-ranked systems relied on the contrastive pairs; the best system that judged each claim-evidence pair on its own scored 24.3 points lower. Open-weight models, embedded in carefully designed pipelines, matched proprietary ones. Yet verifying a single claim against its evidence, the setting that matters in practice, remains an open problem. 

---
# ProcessPrefBench: Evaluating Cross-Task Transfer of Individual Process Preferences from Naturalistic Human-AI Interaction 

**Authors**: Yiwen Yang, Guanming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.06065)  

**Abstract**: In natural human-AI dialogue, users' corrections, requests, and self-descriptions carry reusable signals about how they want work to be done. These signals reveal individual process preferences, user-specific standards for evidence, depth, analytical focus, procedure, and delivery. We ask whether a preference revealed in one dialogue can shape an assistant's later answer when the task does not restate it. ProcessPrefBench uses three months of natural dialogues from two knowledge workers to build 66 source-target cases separately judged for source evidence and target applicability. Sources are preselected, so the benchmark tests application rather than retrieval. We test three models with target-only, full-history, and distilled-preference prompts. Target-specific 0-2 rubrics score preference realization, and a task gate flags severe failures. Compared with target-only prompting, full-history and imperative distilled-preference prompting raise preference realization by 0.204 and 0.244 points on the 0-2 scale, respectively, with no reliable advantage from distillation. Fabricated-evidence failures rise from 5.8% under target-only prompting to 10.1% under full-history and 10.4% under imperative prompting. ProcessPrefBench highlights a tension in learning from dialogue, where drawing on prior interactions can improve preference realization yet also invite fabricated evidence. We hope it helps build assistants that honor a user's process preferences without sacrificing factual reliability. 

---
# Adapting Qwen3 to Tajik from 0.6B to 8B: Lower Text Loss Did Not Mean Better Understanding 

**Authors**: Tohir Saidzoda, Madina Abdusalomzoda, Nasimjon Rajabov, Akhrorkhon Ikromov, Komron Giyosiev, Abubakr Abdullozoda, Amin Jalilov, Jonibek Kishvarov, Rahmonali Sharipov, Mariyamkhon Shodyeva  

**Link**: [PDF](https://arxiv.org/pdf/2611.06062)  

**Abstract**: Small LoRA adapters trained on Tajik news and books lowered Qwen3's Tajik text loss at every size we tried, from 0.6B to 8B, by 0.60 to 1.45 nats per token, but none of them improved Tajik reading comprehension where it can be measured. On Belebele the 1.7B and 8B adapters left accuracy unchanged and the 4B adapter lowered it by 3.7 points; at 0.6B the score rose by 4.2 points but sits near chance and coincides with a change in which answer letter the model prefers, so we do not read it as understanding. At 8B, continued pretraining on about 375 million tokens of raw Tajik text was followed by a Belebele gain of 9.9 points. After a further supervised fine-tuning stage, our main model, e3, is 11.4 points above Qwen3-8B on Belebele; its English MMLU changed by -1.2 points, and a loss larger than 3.3 points is excluded. The same heavy recipe applied to Qwen3-0.6B-Base brought its text loss close to that of the 8B model after books-only pretraining, while its multiple-choice answers stayed near chance and its English MMLU fell by 7.6 points. We report 15 trained models and their 5 starting models from June to October 2025, scored in 2026 on one harness with paired intervals, and we report what went wrong: lost training records, a thinking-mode dataset with almost no reasoning in it, later branches whose training data was never documented, and a 4-bit release that contains no transformer layers. 

---
# Confucius4-R2T2: A Stable-Prefix Streaming ASR Model for Real Real-Time Transcription 

**Authors**: Daniel Huang, Qi He, Shaoxiong Duan, Yingjie Li, Qiao Cheng, Yitao Duan  

**Link**: [PDF](https://arxiv.org/pdf/2611.06043)  

**Abstract**: Streaming speech recognition is increasingly consumed by machines, such as translators and voice agents, that act on each token as it appears. A token that is later revised costs wasted work or a wrong action. In this work, we present Confucius4-R2T2 (short for Real Real-Time Transcription), a stable-prefix streaming ASR model that never modifies its earlier output while maintaining high accuracy and low latency. R2T2 is obtained by post-training Qwen3-ASR-1.7B on 1,500 hours of Chinese and English audio while freezing its pre-trained audio encoder and projector. The adaptation is based on the concept of Longest Stable Prefix (LSP), a new notion that we introduce in this work. We prove that LSP is monotone under the prefix order, consistent with the full-audio output, and maximal among labels that equal a whole earlier output. Learning from LSP labels converts a full-context model into a stable-prefix streaming one without any architectural change. Extensive evaluations show that R2T2 brings streaming recognition close to the base model's full-context quality while delivering stable text with low emission lag. Its gains extend beyond the post-training languages, demonstrating a practical path from full-context ASR to stable streaming. We open-source the model weights and code at this https URL and this https URL. 

---
# When Every Choice Is Correct: A Causal Study of Numerical Preference in Language Models 

**Authors**: Hongnian Huang, Hongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.06025)  

**Abstract**: When a language model chooses a number between 1 and 30, every integer in the range is valid, yet the model can favor particular answers. We use numerical choice to study how preference is processed across layers and tokens. Perturbation probing and gradient attribution nominate neurons for activation interventions. In a selected Qwen3-4B case, a pre-answer edit changes 17 to 15 while both runs first emit 1. After the edit ends, selective cache and attention-output replacements trace part of its continuing influence on the second digit. Replacing one attention head's output or the other heads' outputs jointly produces similar immediate readout contributions; both interventions restore 17 but have different final log-odds effects on 7 versus 5. Holding later multilayer perceptron (MLP) outputs at their values from the edited run without either replacement reduces this effect gap from 0.119 to 0.0025 nat. Later computation therefore changes how similar local contributions affect the continuation. Retrospective estimates across eighteen contexts capture much of the probability movement but miss the selected answer switch. Prompt tests also separate influence from answer changes: one held-out prompt retains influence without switching, and only one of twelve fresh wordings qualifies for the reader/MLP test; there, joint MLP clamping again reduces the gap between the two reader effects while the answer remains 17. These results use numerical choice as a controlled case for studying persistent state and conditional downstream processing; they do not establish the origin or a complete circuit of natural numerical preference. 

---
# Holistic Parallel Supervision for Expressive Speech to Speech Translation 

**Authors**: Hao Zhang, Ismail Rasim Ulgen, Bismarck Bamfo Odoom, Rui Liu, Zihan Zhang, Berrak Sisman, Philipp Koehn  

**Link**: [PDF](https://arxiv.org/pdf/2611.06010)  

**Abstract**: Expressiveness is central to speech-to-speech translation (S2ST), yet existing datasets rarely align source and target speech jointly in content, prosody, timing, and speaker identity. Consequently, current systems often rely on indirect conditioning or transfer mechanisms. To obtain this supervision, we introduce a training-data construction strategy that combines professional dubbing, which provides human-performed correspondence in content, prosody, and timing, with multilingual voice conversion to reduce speaker mismatch. We evaluate this supervision using existing expressive S2ST systems and DirectS2ST, an end-to-end method developed in this work that predicts the complete target-speech representation directly from source speech. Experiments on English-German and English-Spanish show that holistic parallel supervision improves expressive realization across all evaluated systems. DirectS2ST exhibits the largest gains in content, prosody, timing, speaker similarity, and human-rated expressiveness. To our knowledge, this work provides the first systematic investigation of holistic parallel supervision for expressive S2ST. 

---
# Familiar letters, not related languages, let unseen languages into a shared space of meaning 

**Authors**: Yuliang Liu, Xinyu Wang, Xiang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2611.06006)  

**Abstract**: Hundreds of languages have written text but almost no translations, and whether a model learning one from untranslated text builds on a related language or familiar letters is unclear: they usually coincide. Here we show that familiar letters, not a related language, let an unseen language into a shared space of meaning. GlyphMind, a tokenizer-free vision transformer that reads text as pixels and lets any sentence be shown in any script, was pretrained to align translations across 27 writing systems. Adapted on untranslated Lao, a language and script it had never seen, it retrieved English translations seven times as often as the strongest simple baseline. Rewritten in unfamiliar letters, Lao did not enter although its close relative Thai was in pretraining; Santali, which had none, entered in Devanagari, familiar from Hindi, but not in its own script. Transliteration into familiar scripts may thus open multilingual models to languages without translations. 

---
# Selective, Risk-Controlled Emotion Forecasting in Conversation: From a Persistence Barrier to Departure Detection with Conformal Guarantees 

**Authors**: Bin Wen, Dai-Qiao Zhang, Tien-Ping Tan  

**Link**: [PDF](https://arxiv.org/pdf/2611.05896)  

**Abstract**: Forecasting emotion in conversation is increasingly framed as predicting the next emotional state's value. We show this targets a hard quantity in a state-observed testbed and propose a learnable, deployable reformulation instead. On valence-arousal-dominance trajectories from a dyadic corpus, repeating a speaker's own previous state sets a persistence baseline that no frozen-encoder correction meaningfully improves upon and a state-space filter exceeds; the floor requires the historical state to be observed, since an estimated anchor collapses persistence below a direct forecaster. What survives is the timing of a departure from persistence, learnable on three of four corpora and near chance on the highest-base-rate corpus, a negative result we report. We recast the task as selective departure forecasting: a lead-time-weighted metric with a leakage protocol, and commitment policies whose best choice switches with the departure rate. A dialogue-level conformal risk control bounds the expected loss of committed predictions, meeting the target in all twelve configurations tested, though the guarantee is bought through abstention, with coverage and alarm recall falling sharply on the hardest corpus. One tested language model's confidence fails as a ranking signal where another succeeds, a model-dependent failure not yet attributable to family versus scale. Code and result logs are available from the corresponding author upon request. 

---
# Information-Matched Auditing of Conversational Emotion-Shift Forecasting: A Leakage-Safe Evaluation Protocol 

**Authors**: Bin Wen, Dai-Qiao Zhang, Tien-Ping Tan  

**Link**: [PDF](https://arxiv.org/pdf/2611.05895)  

**Abstract**: Comparisons in conversational emotion-shift forecasting are easily distorted when a learned model receives rich features but its transition baseline is given a noisy or unavailable current-emotion label, when hyperparameter budgets differ, or when target-turn information enters the representation. We present an information-matched, leakage-safe auditing protocol that closes these gaps with an explicit commitment point, executable causality tests, equal four-configuration validation searches, and seed-dialogue-aware inference. Applying it to six causal model families on four text corpora and two feature-source sensitivity configurations shows that a deployable train-only current-emotion classifier feeding the transition baseline yields a different verdict than an oracle diagnostic where both sides receive gold current emotion. Under the deployable condition, learned models improve over the predicted-label baseline on all four primary corpora; DailyDialog gains are large and survive a 72-comparison Holm correction (best ROC-AUC gain = +0.167, adjusted p=.0144). Under matched gold information, the baseline remains competitive on IEMOCAP and stronger on MELD, while every model retains a large DailyDialog advantage. We additionally report precision-recall AUC, calibration, a strictly utterance-local feature control, same-person self-shift forecasting, and a future-access dose-response test. An eight-item leakage checklist accompanies the protocol. Code, raw seed predictions, and reporting artifacts are available from the corresponding author upon request. 

---
# When Does Better Confidence Improve a Knowledge-Graph Audit? 

**Authors**: Zehao Lin, Kai Chen, Xixuan Hao, Erenkai Zhu, Chunyu Li, Helan Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.05877)  

**Abstract**: Confidence estimates guide human review in LLM-built knowledge graphs, but better error prediction can yield little improvement in an audit. We study this gap through repairability, selection headroom and label validity. Modelling non-adaptive auditing as budgeted covering, we characterise the minimal decision-relevant error statistics and bound the regret of scalar-preserving first-order planning by a structural mass $\Delta^{\mathrm{cl}}$. We simulate ideal repair against benchmark-aligned labels for five extraction configurations and two cross-document graphs on Re-DocRED, followed by frozen confirmation on test and replication on 500 hash-selected training documents. On the held-out cross-document graphs, an error-rate $\times$ fan-out queue leaves 3.98-4.74 percentage points more errors than a joint model with the same actions. By comparison, opening shared actions for that joint planner reduces residual errors by 0.89-1.43 points relative to its own fact-only plans, in point estimates. Equivalence between joint and independent-block models remains inconclusive. Generic LLM verification improves three of four model-budget configurations on each held-out set. LLM adjudication identifies possible benchmark omissions. 

---
# CARTBench: Rubric-Based Evaluation of Shopping Advice for Complex and Ambiguous Requests with Timelines 

**Authors**: Yuqin Dai, Zhanwei Zhang, Honghui Yang, Zhouheng Yao, Yueqi Zhang, Jingyu He, Chaoyang Shi, Fanding Wei, Boyuan Pan, Kai Xu, Yao Hu, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.05862)  

**Abstract**: Large language model (LLM) shopping assistants now recommend products in multi-turn conversations. However, reliable evaluation faces Data scarcity and Difficulty Standardizing Criteria. To address these obstacles, we introduce CARTBench for CART (Complex and Ambiguous Requests with Timelines). For Data scarcity, we curate cases from real commercial shopping needs across 17 retail domains, pairing timestamped dialogue with 1,603 product records. For Difficulty Standardizing Criteria, we develop case-specific rubrics guided by Ideal Response Principles, distilled from excellent replies manually selected from successfully completed orders. These principles also guide Principle-Guided CART Prompting, our structured recommendation strategy. The experimental benchmark contains 1,328 expert-annotated criteria across Core, Excellence, and Safety. To answer the CART question, we evaluate 7 flagship commercial API models, including GPT-6.1-Sol and Claude-Opus-5, and 35B+ parameters open-weight models. In the first full-suite evaluation under the frozen rubrics, 3 of 10 models exceed 90% on Core, while none exceeds 45% on Excellence. Therefore, we identify an understanding-to-execution gap between required behavior and high-quality purchase advice. To examine these limitations, we diagnose shared failures in product evidence and unresolved needs, linked to changing requirements and decisive unknowns. Code and data are available at this https URL. 

---
# TandemBench: Benchmarking Decision Models for Multi-Agent Collaboration 

**Authors**: Baohang Li, Xiaocheng Feng, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2611.05827)  

**Abstract**: Decision models choose among predefined options in tasks such as routing and action selection. In multi-agent collaboration, a decision determines how members divide work, exchange resources, and act in parallel. We introduce TandemBench, a benchmark of 1,200 questions in multi-agent cooking scenarios where coordination choices affect goal completion. Models receive complete global information and candidate team plans, with reference answers established through execution and exact search. Models select an optimal team plan and, in paired variants that retain or remove it, judge whether better arrangements exist beyond the supplied options. We evaluate 40 native decision models and compare them with 10 general-purpose language models. The best native decision model achieves 62.50% decision accuracy. Native decision models remain close to chance in optimality verification. Paired tests reveal that some report better alternatives regardless of whether an optimal plan is already present. These findings show that team-plan decision accuracy alone can conceal systematic errors in judging whether the supplied options contain an optimal arrangement. The dataset and evaluation code are publicly available at this https URL. 

---
# Learning When Human Decisions Matter: Executor-Relative Consultation Value for LLM Agents 

**Authors**: Shuli Jiang, Khushbu Pahwa, Yuting Zhang, Yi Zhang, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2611.05814)  

**Abstract**: Large language models increasingly automate complex tasks, but human decision capacity does not scale with their capabilities. In many applications, humans remain essential for providing private context, authority, or judgments unavailable to the agent. We formulate human decision capacity allocation: given multiple consultation opportunities and a limited budget, at which decisions should an LLM agent consult a human? The central challenge is estimating consultation value, which is executor-dependent because the same human input may benefit different downstream agents differently. We introduce Consultation Value Distillation (CVD), which estimates consultation values offline using Monte Carlo executor rollouts and distills these values into lightweight predictors for budgeted allocation, requiring no rollout-based value estimation at deployment. Across five batch and three sequential settings with two frontier executors, CVD improves allocation through joint and continuation-aware value learning, while independent consultation values remain competitive in several settings. CVD also broadly outperforms direct prompting and Value of Information where applicable, remains robust with substantially less rollout supervision, and enables nearly lossless approximate allocation at low search cost. 

---
# Matching the Training-Time Weight Rule in Inference-Time Expert Deactivation 

**Authors**: Hao Cheng, Zhiyu Yang, Bo Ma, Hanzhang Jia  

**Link**: [PDF](https://arxiv.org/pdf/2611.05804)  

**Abstract**: Expert deactivation at inference time changes both token routing and expert weights, and its effect on performance depends on how these changes are implemented. We compare removal, freezing, remaining-expert renormalization, and unnormalized re-routing on identical disabled sets across five instruction-tuned checkpoints from four model families, with a base OLMoE checkpoint as an additional comparison. We develop a weight-sum model that relates the loss of expert contributions and changes in branch scale to the weight rule used during training. With half of the experts deactivated according to activation-based rankings, remaining-expert renormalization yields the highest perplexity in models trained with raw router probabilities, whereas unnormalized re-routing yields the highest perplexity in models trained with normalized weights. Perplexity ranges from 1.4 to 73 times the intact baseline when the training rule is retained, and from 14 to 28,740 times the baseline under a rule mismatch. Task outcomes vary across evaluation types: on OLMoE, GSM8K strict-match accuracy falls to zero under all four implementations. Extending the model to changes in the number of active experts yields a further prediction, supported by the OLMoE experiments: the perplexity penalty from additional renormalization increases as fewer experts are activated. 

---
# Phonological Knowledge of Chinese Characters in Language Models 

**Authors**: Xiaozhe Ji, Renfen Hu, Hai Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.05803)  

**Abstract**: How much phonological knowledge do language models (LMs) possess about Chinese characters, and how is this knowledge represented and used? We introduce PinyinBench, a controlled benchmark covering homophones, phonological features (initials, finals, and tones), and polyphonic characters. We investigate 17 text-only and multimodal LMs through three complementary approaches: behavioral evaluation, layer-wise probing, and activation steering. Behaviorally, models demonstrate substantial knowledge of initials and finals, with a frontier model (DS-v4-flash) scoring 90%+ in homophone judgment, when two characters share no phonological features. However, its accuracy drops sharply to 72% when two characters differ only in lexical tone. Probing reveals that tone is highly decodable from embeddings and improves modestly across layers, whereas segmental information starts less decodable and improves substantially with depth. Activation steering shows that segmental representations can systematically influence generation, while tone exhibits much weaker effects. Furthermore, audio-bearing training improves phonological decodability, including tone, without corresponding improvements in steering effectiveness. Together, these findings reveal distinct representational and functional profiles for segmental and tonal information, with tonal information much harder to access and manipulate, demonstrating the necessity for combined evaluation, probing, and causal intervention for understanding linguistic knowledge in LMs. 

---
# How Well Can Large Language Models Impersonate Individual Authors? A Comparative Evaluation 

**Authors**: Harshit Gupta, Shunichi Ishihara  

**Link**: [PDF](https://arxiv.org/pdf/2611.05771)  

**Abstract**: This study investigates the authorship impersonation capacity of six widely used large language models (LLMs): Gemini 3 Pro, Claude Opus, GPT 5.2, GPT 5.2 Pro, DeepSeek 3.2, and Grok 4.1 Fast. The evaluation was conducted using short product review texts selected to ensure high within author consistency in individual writing style. Two types of in context prompt were designed: a simple prompt with minimal instruction and a complex prompt incorporating explicit stylometric guidance for mimicry. Authorship mimicry was evaluated using a cosine distance based distance to training metric, supplemented by visual analyses of embedding distributions. The results indicate that authorship impersonation remained challenging for most LLMs, with Gemini 3 Pro showing the strongest apparent mimicry performance. However, subsequent overlap analysis revealed that Gemini 3 Pro extensively copied and reused sentences and phrases from the training texts, at levels far exceeding those observed among human authors. Follow up factor analyses showed that training set consistency was the most robust predictor of mimicry success across models, whereas author distinctiveness did not generalise as a reliable predictor. Global embedding visualisations further revealed a consistent macro level pattern: for most models, generated texts occupied regions of the embedding space that were largely separated from the corresponding human training clusters. This indicates limited convergence towards target author style and suggests the presence of model specific lexical and phraseological preferences. The findings highlight the importance of distinguishing genuine stylistic imitation from direct textual reuse and have implications for the forensic analysis of textual evidence involving potential LLM enabled impersonation. 

---
# Prospective Memory for Proactive LLM Agents 

**Authors**: Yuanchen Bei, Zhining Liu, Tianxin Wei, Weizhi Zhang, Zhichen Zeng, Philip S. Yu, Hendrik Hamann, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2611.05731)  

**Abstract**: Most memory systems for LLM agents are retrospective, preserving and retrieving past information to inform current decisions. Proactive agents, however, also require prospective memory: retaining future-directed intentions and realizing them when appropriate opportunities arise, even without explicit requests. Crucially, remembering the right intention does not guarantee acting at the right time: a related observation may call for waiting, while a later cue may make the same intention actionable. Inspired by cognitive theories of human prospective memory, we introduce Pro-Memory, a memory framework that separates maintaining future intentions from deciding when and how to realize them. Pro-Memory preserves pending objectives linked to their supporting evidence, assesses new observations against their original goals and activation conditions, and coordinates appropriate contributions with the agent's ongoing actions. It further updates intention states based on actions actually issued or evidence of resolution, while an experience memory distills feedback from completed episodes into guidance for future decisions. Experiments on two benchmarks spanning delayed-intention tasks and open-ended proactive assistance demonstrate consistent improvements in proactive task performance over representative memory approaches across multiple LLM backbones, while limiting unnecessary interventions. 

---
# Cognitive Bias Signatures in Search-Equipped LLM Agents: A Stage-Resolved Analysis 

**Authors**: Yanru Xiao, Yingwei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.05720)  

**Abstract**: Search can help LLM agents correct mistakes, but their starting conditions may shape which evidence they find and use. Drawing on cognitive-bias paradigms, we develop a controlled search harness to examine result order, records of prior work, and initial hypotheses while holding each question and document collection fixed. Matched controls distinguish order effects from differences in document composition and the effects of failed work from context length. Across 10 models and 1,000 synthetic search worlds, we track queries, retrieved documents, document openings, and final answers. Our comparisons suggest that final answers are more sensitive to initial evidence order during active search than when all evidence is supplied in full. Dead-end histories can reduce strategy switching relative to matched neutral histories, indicating an effect beyond context length. Under wrong hypotheses, stage measurements distinguish reduced retrieval or opening from errors that persist after counter-evidence is opened. Exploratory interventions show that opening counter-evidence more often can coincide with lower accuracy, while accuracy can improve with little change in opening. 

---
# Contextual Pictograms for Repairing Multilingual ASR Errors 

**Authors**: Mariana Hernandez, Hector Luna, Carlos Bruno, Luis Ricardo Santos, Elena Simoes  

**Link**: [PDF](https://arxiv.org/pdf/2611.05637)  

**Abstract**: Speech transcription errors can be difficult to recognize when competing word hypotheses are unfamiliar or visually similar. We describe an interface that combines alternative transcripts, replay and contextually checked pictograms for selected uncertain concepts. The system will initially support French and Spanish functional utterances. A counterbalanced study will assess whether pictorial feedback helps listeners recover the speaker's intended action beyond text alternatives alone. We will also quantify semantic mismatch and additional interaction time, since an inappropriate image can reinforce the wrong interpretation rather than repair it. 

---
# RAGCompass: Generalizable Retriever Routing via Reranking 

**Authors**: Valentin Six, Mufei Li, Dongqi Fu, Yinglong Xia, Hong Li, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.05596)  

**Abstract**: Different retrievers exist for retrieval-augmented generation, yet none performs best for all queries. Therefore, the key question is: given a query and retrieved passages from different retrievers, which one is most suited for that query? Existing routers usually make predictions from query-specific features and a fixed pool of retrievers; such routers have no output for an unseen retriever and are incapable by design on a new corpus. To fill these gaps, RAGCompass reads the retrieved passage sets themselves: passages from an unfamiliar retriever or corpus stay directly comparable to other passages, so routing stays well designed where a query-only map would fail. We contrastively fine-tune a reranker against the answer each evidence set produces, which accounts for most of the gain; a small comparison head adds a moderate improvement, and neither component has retriever-specific parameters. RAGCompass improves answer F1 by 28% relative to the best fixed retriever, and outperforms fusion baselines, three query-only routers, and a prompted 8B language model. Generalization experiments show that RAGCompass keeps most of its gain under an unseen corpus, retriever, or reader, while query-only baselines fall to the best fixed retriever or below. 

---
# Rewrite, Then Veto: Rewriter-Agnostic Edit Gating for LLM-Based Post-OCR Correction 

**Authors**: Jorge Ramón Fonseca Cacho  

**Link**: [PDF](https://arxiv.org/pdf/2611.05539)  

**Abstract**: LLMs can repair noisy OCR transcripts of historical documents, but they also change text that was already correct. We propose EditVeto, a veto gate placed after a full-context LLM rewrite. The rewrite is character-aligned to the OCR input, its differences are split into blocks, and a 0.6B-parameter gate keeps or reverts each block. An oracle analysis motivates the design: given the true error locations, per-span generation is worse than no correction, whereas vetoing a full rewrite's edits helps. Trained on rewrites by two Qwen3 models and evaluated, frozen, on the eight HIPE-OCRepair-2026 test sets, the gate improves all five rewriters tested, two of them unseen in training: weighted cMER falls by 0.0026-0.0044 and the preference score rises by 0.15-0.25 (paired bootstrap, p ~ 0.0002 each). On published competition outputs it lowers cMER for three of four teams but raises it for the winning run (0.0050 to 0.0065). In a second, disclosed test look, a retrained gate still improves LoRA-tuned rewriters (cMER -0.0011 to -0.0021), and an ensemble of their gated edits reaches a cMER of 0.0069 and a preference score of 0.928, mostly through fine-tuning and pooling rather than gating. Built only on public models and data, it is behind the winner (0.0050) and tied with the second-ranked run (0.0071), both from the only team using domain pre-training, and ahead of all other published runs. A third, secondary look shows the unchanged gate improving zero-shot Qwen3.8-27B (cMER 0.0114 to 0.0108) but only the preference score of its LoRA-tuned version. 

---
# When to Ask, When to Answer: Strategic Clarification for Ambiguous Reference Resolution in Dialogue 

**Authors**: Ibrahim Khebour, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2611.05517)  

**Abstract**: For humans, referential disambiguation is a collaborative process involving clarification questions asked when they will meaningfully change the outcome. However, large language models (LLMs) have been shown to be biased toward premature answering, preferring definite single-turn responses over more clarification exchanges. In this paper, we study the problem of "strategic clarification" -- asking clarification questions in a reference game when their answers will narrow the set of candidate referents. We construct a controlled test environment from the SIMMC 2.1 dataset that records how many questions each ambiguous reference needs to be successfully resolved, and examine how baseline LLM behavior and common post-training methods ask clarifying questions vs. attempt to answer the referent. We also propose BACE (Belief-Aware Critic with Epistemic tracking), a lightweight action chooser method on top of a frozen LM backbone, and show that BACE more consistently chooses the optimal number of clarifying questions than methods such as GRPO and ArCHer. 

---
# AI-Driven SaaS Platform Integration: AI Express for Unified Generative AI Services with Pay-As-You-Go Credit Systems and Scalable Microservice Architecture 

**Authors**: Deepit Sapru  

**Link**: [PDF](https://arxiv.org/pdf/2611.05389)  

**Abstract**: Generative AI is currently fragmented, with separate subscriptions required for text generation, image synthesis, and code assistance, leading to high costs and workflow disruptions. AI Express is a SaaS platform that consolidates these capabilities into a unified pay-as-you-go service. The platform provides a range of AI tools, including a Core Code Reviewer that evaluates GitHub repositories, a Text-to-Website generator that converts text into fully functional websites, script generation, grammar checking, data analysis, and watermark removal, all accessible through a single interface without requiring technical expertise. 

---
# SRRM: A Retrieval-Reconstruction Benchmark for Evaluating Long-Context Memory in Small Language Models 

**Authors**: Christopher Alberto Hamonangan  

**Link**: [PDF](https://arxiv.org/pdf/2611.05386)  

**Abstract**: Small Language Models (SLMs) are increasingly deployed on consumer-grade hardware, yet their ability to preserve information across progressively longer contexts remains insufficiently understood. Existing benchmarks primarily evaluate retrieval-oriented tasks, providing limited insight into whether successful retrieval reflects comprehensive long-context memory preservation. To address this limitation, this paper introduces Simple Retrieval-Reconstruction Memory (SRRM), a lightweight and fully reproducible benchmark that jointly evaluates Targeted Retrieval and Global Information Reconstruction. This study also proposes the Memory Preservation Ratio (MPR), a benchmark-level metric that quantifies the relationship between retrieval and reconstruction performance. SRRM was evaluated using a synthetic dataset on three Qwen2.5 models (0.5B, 1.5B, and 3B parameters) across context loads ranging from 10 to 100 factual statements. Experimental results show that reconstruction performance deteriorated substantially faster than retrieval accuracy as contextual information increased, indicating that retrieval alone is insufficient to characterize long-context memory preservation. Among the evaluated models, Qwen2.5-1.5B achieved the highest retrieval accuracy (86.7%), reconstruction performance (45.7%), and MPR (0.53), outperforming the larger Qwen2.5-3B model (72.2%, 18.5%, and 0.26, respectively). These findings demonstrate that increasing parameter count does not necessarily improve long-context memory preservation. Overall, SRRM provides a reproducible and extensible benchmark for systematically evaluating long-context memory while revealing complementary memory behaviors that remain hidden under retrieval-only evaluation frameworks. 

---
# Same Origin, Different Verdicts: Text Production Conditions Confound Machine-Generated Text Detection in Russian 

**Authors**: Stanislav Kirichenko  

**Link**: [PDF](https://arxiv.org/pdf/2611.05385)  

**Abstract**: Automated detection of machine-generated text is applied where the verdict carries a cost for the author, yet origin is a property of the production process, not an observable property of the text. The conditions under which a text is produced push it in the same direction in which a generative model is presumed to push it. This study measures the contribution of these conditions while origin is held fixed. The Russian-language corpus of 1882 documents comprises a machine part, generated by four generation channels (each backed by its own model family) across 45 tasks in three task modes, and an archival human part annotated for source regimentation level before the texts were read. The primary contrast is computed within the machine corpus: both halves of a pair are generated by the same model for the same task and differ only in task wording. Machine-likeness is measured by four operationalisations: an interpretable index with weights fixed before data collection, a trained classifier, a zero-shot metric on an open model, a judge model. The main result is disagreement between procedures: changing the task mode shifted the scores of all four, but three gave a negative shift, the judge a positive one, and none of the four intervals covers zero. The comparison plan and the specifications of procedures 2-4 were fixed after the result of procedure 1, before procedures 2-4 ran; the disagreement survived correction of a corpus defect and of the nested validation. A baseline on the source identifier alone reaches a median AUROC of 1.0000 on the thirteen group splits where it is defined, and removing emphasis markers flipped a confident machine verdict to human without changing the full text by a single character. The scores are properties of a specific measurement system and corpus: they reflect data collection, repetition, formatting, and preprocessing decisions as well as linguistic style. 

---
# Human-like cross-language generalisation in deep neural speaker embeddings and its acoustic foundations 

**Authors**: Tianze Xu, Xiyang Li, Xiaoming Jiang, Volker Dellwo  

**Link**: [PDF](https://arxiv.org/pdf/2611.05384)  

**Abstract**: Generalisability is a central question for deep neural networks, including the cross-language generalisation of speaker embeddings. This study examined whether deep neural speaker embeddings generalise across languages and how such generalisation relates to human perception and acoustic structure. Using Cantonese--English bilingual speech presented in within- and cross-language conditions, we derived similarity from 18 state-of-the-art models and compared it with human ratings and acoustic features using linear mixed models and representational similarity analysis. Embeddings exhibited cross-language generalisation that broadly paralleled human behaviour, reflecting accurate identity discrimination and sensitivity to language switching. This generalisation relied on largely shared acoustic cues with language-sensitive weighting, mirroring human patterns. Together, these findings provide converging evidence for human--machine alignment in cross-language voice similarity processing. 

---
# CCDWIKI: Decoupling Case Formulation from Response Generation in Longitudinal CBT Dialogue 

**Authors**: Mingwen Zhang, Ruiyang Song, Qinying Wang, Yuchen Wang, Changsheng Ma, Minqiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.05383)  

**Abstract**: Cognitive Behavioral Therapy (CBT) demands more than empathic validation, requiring instead a structured formulation of the interplay among situations, automatic thoughts, emotions, and behaviors. Therapists formalize this interplay through a Cognitive Conceptualization Diagram, where maintaining longitudinal coherence necessitates continuous evidential adjudication to separate tentative hypotheses from established findings. Current dialogue systems track conversational history via retrieval or static profiles but lack the representational capacity to encode epistemic status. Consequently, provisional interpretations are often prematurely reified as established facts, inadvertently biasing downstream interventions. Addressing this limitation, CCDWiki is introduced as a plug-and-play advisor conceptualizing longitudinal therapy as evidence-grounded hypothesis adjudication. The approach encodes each clinical hypothesis as a mutable entry bound to corroborating and contradicting evidence. Across sessions, accumulating data strengthens supported entries, preserves provisional ones, and retracts contradicted assumptions. The resulting formulation distills into concise guidance for unmodified therapeutic language models. Experiments demonstrate substantial improvements in both conceptualization and intervention quality without requiring backbone retraining. 

---
# DS@GT ARC at CheckThat! 2026: A Verdict-Anchored Pipeline for Generating Fact-Checking Articles 

**Authors**: Hoang Thanh Thanh Truong, Shuyu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2611.05382)  

**Abstract**: Misinformation spreads faster than professional fact-checkers can manually verify it and write articles explaining their verdicts. This creates a need to automate the production of fact-checking articles. This paper details work done by the DS@GT ARC team for CLEF 2026 CheckThat! Lab Task 3 Fact-Checking Article Generation. We seek to generate a full fact-checking article from a claim, its original verdict, and a set of scraped source documents. Our team built a four-stage pipeline that cleans noisy evidence, retrieves and clusters relevant sources, generates the article by conditioning \texttt{Llama-3.1-8B-Instruct} on the original fact-checker verdict, and recovers inline citations through deterministic post-processing. We achieved a mean score of 0.428 and ranked 3rd out of 9 teams for the CLEF 2026 CheckThat! Lab Task 3, compared to a baseline mean score of 0.272. Our code is available on GitHub at this https URL. 

---
# Discourse Analysis of Legislative Bill Discussions 

**Authors**: Isaac Rudnick, Nipun Das, Foaad Khosmood  

**Link**: [PDF](https://arxiv.org/pdf/2611.05381)  

**Abstract**: Legislative hearings are distinct communication events with their own human recognizable structure and logic, yet they have been analyzed as distinct phenomena beyond their transcripts. Understanding legislative hearings at a deeper level is a major step in analyzing the work and power relationships within democratic bodies. In this work, we present a novel end-to-end pipeline for parsing rhetorical structures from legislative committee hearings with the goal of improving legislative discourse analysis and democratic accountability. We also propose a rhetorical structure grammar and find that 55.21% of California state bill discussions in our dataset can be parsed with this grammar. We evaluate our pipeline with two utterance section classifiers on a random subset of over 2,000 annotated utterances corresponding to 100 hearings. We find that a logistic regression-based classifier with grammar-based output masking and utterance tag smoothing outperforms more complex models, achieving an F1 score of 52.8% against our ground truth discourse section labels. 

---
# A Framework for Context Enrichment and Selective Robustness in LLM Social Agents 

**Authors**: Mattia Greiche, Reihaneh Rabbany, Maximilian Puelma Touzel  

**Link**: [PDF](https://arxiv.org/pdf/2611.06914)  

**Abstract**: Researchers use large language models as agents in social simulations, where agents are prompted with observations and background information intended to shape their actions. Yet LLM agents have been shown to substantially change their behavior under meaning-preserving prompt variations, creating a validation problem for social simulation. One plausible hypothesis is that this brittleness reflects a lack of grounding and can be reduced by enriching the context provided to the agents. Whether richer context improves robustness remains unclear, in part because context richness and robustness are rarely defined in ways that support controlled comparisons. We address the first need with controlled context enrichment, a set of prompt substitution rules that systematically enriches distinct types of context: neutral padding, generic social background, descriptive decision grounding, and normative decision grounding. We address the second by introducing selective robustness as a validation criterion: insensitivity to meaning-preserving prompt variation combined with a directional response to decision-relevant changes in the social world. We demonstrate the framework in a Prisoner's Dilemma-like game, showing how it can be used to identify which forms of context enrichment produce more desirable and robust agent behavior. We find that these forms of enrichment produce distinct behavioral patterns that vary between the two models tested. 

---
# LongBanana: An Expert-Verified Benchmark for Long-Context Multi-Reference Image Synthesis 

**Authors**: Haoxiang Cao, Yuxuan Zhang, Penghui Du, Bo Li, Jiajiong Cao, Yingying Fan, Zimeng Wu, Yifu Luo, Mingzhe Zheng, Junwen Miao, Huaisong Zhang, Yang Yao, Yu Wang, Dongfu Jiang, Ping Nie, Wenhu Chen, Chaoqun Wang, Changqian Yu, Kelsey R. Allen  

**Link**: [PDF](https://arxiv.org/pdf/2611.06856)  

**Abstract**: We introduce LongBanana, to our knowledge the first expert-designed and expert-verified benchmark for long-context multi-reference image synthesis in the age of agent-driven image generation. The task requires retrieving dispersed visual evidence, rejecting distractors, binding entities and relations, and composing one coherent final image from multiple references. LongBanana contains 90 carefully designed tasks organized by a 3-by-3 taxonomy; each task includes about 32 reference images, 20 constraints, and 50 evaluation items, yielding 4,670 expert-specified atomic evaluation items in total. Its graph-grounded evaluation design instantiates each task with reference images and an entity-relation graph, then converts node, edge, and whole-graph obligations into checklist items for Content Fidelity, Relational Binding, and Global Constraint Satisfaction. Reviewed expert labels include a 15% double-labeled subset with 89.4% exact agreement and 0.83 quadratic weighted kappa. We further introduce LongBanana-Gen, an agent-harness framework for long-horizon image generation; experiments show that minimal agent harnesses improve over direct generation, while a ReAct-Plan-Exec-Reflect (RPR) workflow yields harness-dependent gains by strengthening evidence grounding and repair. Together, LongBanana provides an auditable testbed for measuring dense visual evidence use and a shared target for developing stronger agent-driven image-generation systems. Code and data are available at the project page: this https URL. 

---
# Template Gap Reconstruction for Shell Output Simulation with Small Language Models 

**Authors**: Jarrod Ragsdale, Rajendra Boppana  

**Link**: [PDF](https://arxiv.org/pdf/2611.06731)  

**Abstract**: Small language models offer a compact basis for shell simulation without executing submitted commands, but full-output decoding requires them to predict recurring utility structure alongside content that varies across executions. We introduce template gap reconstruction (TGR), which supplies reusable output literals interspersed with sentinels to be interleaved with autoregressively generated gaps. A template selector chooses which template to be reconstructed, with the empty template retaining full-output decoding. We construct a template bank from 18.7 million outputs collected across heterogeneous execution environments and train a template selector and five reconstruction models initialized from various small language model checkpoints on a target environment excluded from bank construction. Evaluation demonstrates that TGR improves output edit similarity (O-ES) over full-output decoding (FOD) when the optimal template covers more than 25\% of the template, while performing similarly for lower-coverage outputs. Additionally, TGR outperforms FOD for shorter gap sequences, with O-ES values of 0.964 and 0.812, respectively, while maintaining similar performance for longer sequences. TGR reduces the number of tokens generated by 26.3--38.3\% across used tokenizer families. These results support using a reusable output structure as a practical design choice for small shell simulators, improving response fidelity and shortening generation targets while keeping template selection and residual prediction central to response quality. 

---
# Turns Matter: Advancing Streaming Video RL via Turn-Level Credit 

**Authors**: Shuai Dong, Kanghui Tian, Linghan Chen, Wenqi Shao, Hongrui Wu, Haowen Hou, Yongfu Zhu, Xuehui Wang, Ziyue Wang, Wenpu Liu, Weichu Xie, Yuqi Xu, Congcong Wang, Kaiwen Tuo, Caoyuan Ma, Taiqiang Wu, Ji Zhao, Wenzheng Chang, Dianyi Wang, Siyuan Wang, Zhongyu Wei, Yu Qiao, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.06728)  

**Abstract**: Streaming video understanding requires models to repeatedly decide whether to respond or continue observing. Recently, Group Relative Policy Optimization (GRPO) has been applied to this setting, with existing approaches emphasizing reward design while retaining a single trajectory-level advantage across turns. Such shared credit can reinforce redundant replies after earlier successes and discourage useful follow-ups after earlier failures. We propose TurnWise, a turn-level GRPO framework that aligns learning with individual streaming decisions. TurnWise addresses the mismatch between trajectory-level feedback and turn-level interaction, promoting informative responses and reducing redundancy. Specifically, turn-level credit assignment uses silence-completed prefix scoring to construct group-relative advantages from each decision's remaining return, separating its learning signal from earlier rewards. Turn-mean policy optimization averages token losses within each turn and then across turns, giving response and waiting decisions equal weight regardless of output length. These complementary designs improve streaming policy learning without a learned critic or additional rollouts. Extensive experiments show our method outperforms prior RL approaches for streaming video understanding. 

---
# Beyond accuracy: what AI assurance needs to establish in public services 

**Authors**: Louise Yorke  

**Link**: [PDF](https://arxiv.org/pdf/2611.06655)  

**Abstract**: Public bodies increasingly rely on AI-produced evidence, including classifications, summaries and assessments. These outputs can inform policy decisions and service delivery, making their reliability a matter of public accountability. Establishing whether this evidence can be relied upon requires testing its outputs and examining how organisations decide whether those tests are sufficient. We combine technical evaluation with analysis of AI governance requirements, informed by measurement theory and management-based regulation. We compare eight GPT-4o configurations classifying 1,200 UK Prime Minister's Questions into 21 policy topics, against human-coded labels and across unchanged repeats. Accuracy ranges from 68.2% to 70.3%, yet 24.3% of questions receive different labels, with changes in topic shares and rankings. Configurations agree more with one another than with human coding: agreement does not guarantee correctness. Under the implemented score-extraction procedure, the proportion flagged for human review ranges from 0.9% to 41.0%. We analyse 37 UK and EU policy documents. Both sets expect testing and oversight, but reporting expectations do not consistently cover changing answers, shared errors, selective evaluation, substantive conclusions or review routing. Some UK guidance explicitly addresses stability. The UK sample emphasises public bodies checking their own use; the EU places substantial testing duties on providers while leaving them important choices about measures and acceptance thresholds. Without appropriate checks, evidence could misrepresent policy issues, redirect attention or resources, and leave errors undetected. Public policy should require organisations to justify reliance through checks beyond aggregate accuracy, clear responsibility, access to expertise and scrutiny of results. 

---
# What Does Multi-Turn Reinforcement Learning Really Improve in LLM Agents? 

**Authors**: Kejian Zhu, Zhuoran Jin, Shangqing Tu, Tianyi Men, Yupu Hao, Jiachun Li, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.06483)  

**Abstract**: While multi-turn agentic reinforcement learning (RL) excels on long-horizon tasks, its performance gains remain highly sensitive to interaction constraints. Evaluating across sampling breadth $K$ and interaction depth $T$ via $Pass@(K,T)$, we show that trajectory-level RL advantages depend on sufficient turn budgets, diminishing significantly when budgets tighten. Besides, agentic RL does not uniformly elevate single-step decision quality; rather, its strength concentrates in error recovery and long-horizon resilience. To mitigate the deployment risks caused by this budget dependence, we provide a preliminary discussion on the benefits and limitations of process supervision signals and introduce BudgetMix RL as an alternative that varies interaction budgets during training. Experiments demonstrate that BudgetMix RL achieves robust performance across diverse execution settings. Project Page: this https URL 

---
# LegalEvolver: Agentic Legal Judgment Prediction via Rubric-Guided Self-Evolution 

**Authors**: Xinyu Yang, Jiejun Tan, Chenlong Deng, Chenghao Zhang, Tongyu Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2611.06381)  

**Abstract**: As LLM-based agents develop, they have been increasingly applied in the legal domain. Legal Judgment Prediction~(LJP) requires models to conduct multi-step reasoning and external legal evidence retrieval. Existing legal search agents' performance is often constrained by the quality of retrieval queries. They often formulate queries directly from the initial user request, overlooking latent legal issues and legally salient details. To address this issue, we propose \emph{LegalEvolver}, a query optimization framework specifically designed to generate legally informative queries for LJP-oriented retrieval. Sampling high-quality optimized queries and their corresponding search trajectories is challenging, as relying solely on unguided high-temperature rollouts is both inefficient and constrained by the model's initial capabilities. We therefore heuristically optimize queries as seeds, improving the efficiency of trajectory generation. Furthermore, we summarize rollout experience into rubrics and use them to guide multi-round iterative self-evolution. Iteratively performing rubric optimization and rubric-guided RL enables the model to surpass its initial performance ceiling. End-to-end performance on extensive benchmarks demonstrates the effectiveness of query optimization on LJP. Comparisons with direct SFT followed by RL demonstrate the effectiveness of rubric guidance. Evolving rubrics further outperform fixed rubrics, confirming the benefits of self-evolution. 

---
# Does Context Compression Preserve Refusal Alignment? 

**Authors**: Anthony Nguyen, Wenjun Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.06380)  

**Abstract**: Context compression reduces inference cost by encoding inputs into compact representations while preserving semantic content. An open question is whether semantic preservation alone is sufficient to maintain downstream behaviours such as refusal alignment. We investigate this question and find that encoder-based compression systematically weakens refusal behaviour in instruction-tuned language models, despite high reconstruction fidelity. This effect persists across model families and compression architectures. Mechanistic analysis shows that compression attenuates activation along the decoder's learned refusal direction. We further explore Memory Steering, a lightweight inference-time intervention that restores refusal rates to near-baseline levels without retraining and operates entirely in compressed representation space. These results demonstrate that semantic preservation does not guarantee behavioural preservation under compression, highlighting the need to explicitly preserve alignment-relevant features in compression-aware systems. 

---
# Corralling AI Risks with Grassroots Contracts 

**Authors**: Ehud Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2611.06374)  

**Abstract**: AI's inflection point is a fountain of alarming risks. Digital platforms in particular are vulnerable to three major AI risks, as those in control of global platforms cede control to AI, of the algorithms, of membership and of content: (1) AI control of global platforms, social and economic, used by the majority of humanity, resulting in the majority of humanity becoming vulnerable to AI control. (2) Unaccountable AI agents operating anonymously on social and economic digital platforms, side-by-side with humans, possibly masquerading as humans, serving unknown masters with unknown goals, operating under unknown restraints without accountability or culpability. (3) Unattributed deepfakes generated by anonymous AI, with people unable to distinguish them from genuine content and with their producers evading accountability and culpability.
Here, we propose to corral these AI risks with grassroots contracts, a digital incarnation of a two-centuries old notion of Pierre-Joseph Proudhon. We present grassroots contracts for a social graph with accountable AI agents and for a social network with digital provenance, and working smartphone apps that implement the two and show how they corral these major AI risks: (1) AI control of global platforms. (2) Unaccountable AI agents. (3) Unattributed deepfakes. 

---
# Growing One's AI Tools: A Paradigm of Practice-Grown Tools 

**Authors**: Zeming Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.06132)  

**Abstract**: Professionals have long used software built for them by others. This article reports the growth of a professional computer-assisted interpreting (CAI) tool built by its own user, a conference interpreter without programming training. Drawing on an eight-week version-controlled record with telemetry from 133 logged sessions and over 20,000 recognized utterances, the article traces how real conference work drove the tool through 62 commits and six versions. The record shows capabilities deployed on the day their need appeared and one strategy change that cut median end-to-end delay by over a third within one afternoon. The article then proposes the paradigm of practice-grown tools, defined by the shift from using to growing, the co-evolution of tool and practice through breakdown and repair, bounded local computation, and an assessment corollary, and it formulates three propositions toward tool-inclusive assessment of practitioners, from tool-making literacy to a compute-standardized format for certifying what they and their tools achieve together. 

---
# Reconciling Privacy with Provenance for Language Model Watermarks 

**Authors**: Elise L. Han, Tatsunori Hashimoto  

**Link**: [PDF](https://arxiv.org/pdf/2611.06100)  

**Abstract**: Watermarking has great potential for regulating the use of language models (LMs), but its deployment has been marred by controversy, as many users argue that fine-grained tracking of LM usage violates their privacy. This tension reduces the effectiveness of watermarks, as users engage in evasion strategies such as paraphrasing that can remove much of the watermark signal. We show that provenance and privacy can be reconciled if we shift the goal of watermark design from tracking individuals to estimating aggregate usage. We develop locally differentially private watermarks which provably limit the evidence that any individual text can contribute to detection, while still enabling accurate estimation of population frequencies. We introduce the Red-Green Sequence (RGS) watermark, a private and computationally efficient watermark that achieves accurate frequency estimation at high privacy levels across various models, sample sizes and generation entropies in settings that reflect realistic deployment scenarios. On C4 language modeling tasks, we attain $0.0019$ MSE with $0.0038$ mean absolute bias at a low $\varepsilon$ of 1, while in more challenging settings such as an essay writing task with only 100 text samples, we still find errors on the order of $0.0113$ MSE with $0.004$ mean absolute bias at $\varepsilon=2$. 

---
# SVGFaith: A Two-Layer Meta-Evaluation of Faithfulness Metrics for Text-to-SVG Generation 

**Authors**: Guotao Liang, Haitao Zhou, Biao Liu, Huanxuan Liao, Lingfeng Qiao, Qian-Wen Zhang, Qian Yu, Di Yin, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.06048)  

**Abstract**: Text-to-SVG systems are commonly evaluated by applying raster-image metrics to rasterised outputs, yet whether these scores capture fine-grained prompt faithfulness remains unclear. We introduce SVGFaith, a two-layer meta-evaluation framework in which the more faithful candidate of each pair is determined by construction or by human judgment. SVGFaith-Diagnostic contains 1,842 controlled contrastive pairs across eight capability axes, split evenly between two directions: two renderings compared against one prompt, and two descriptions compared against one rendering. SVGFaith-Arena contains 1,327 human-annotated comparisons of real generator outputs. On SVGFaith-Diagnostic, CLIP and three other image--text embedding metrics reach at least 0.789 pairwise accuracy on object identity but at most 0.672 on spatial relations and attribute binding, where chance is 0.5. We further introduce SVGFaith-Score, an interpretable pointwise metric. A domain-tuned decomposer turns each prompt into a dependency graph of atomic visual requirements, a visual question answering model scores whether each requirement is satisfied, and a requirement's score is discounted when its prerequisites score low. In addition to the overall score, the per-requirement scores show which requirements are judged unsatisfied and which inherit discounts from their prerequisites. SVGFaith-Score significantly outperforms all nine baselines, including a Qwen3.5-122B-A10B judge, on Arena and in both Diagnostic directions. Code and data: this https URL. 

---
# Who Pays for the Knowledge Models Copy? Model Collapse as Rogers' Paradox 

**Authors**: Nam H. Le  

**Link**: [PDF](https://arxiv.org/pdf/2611.06022)  

**Abstract**: Generative models trained on the output of earlier models lose knowledge across generations, rare knowledge first. The remedies that work all bring in information from outside the loop, and the theory of model collapse treats that supply as given: a fixed archive, drawn from a fixed distribution. We argue that the supply is the problem, and that cultural evolution has the theory of it. We read model collapse as Rogers' paradox: a population that comes to rely on copying ends up no better off than one that never copied, because copies go stale unless someone keeps learning for themselves. We test the identification in three steps. First, Rogers' model, fitted only to fully synthetic training, predicts to within a few percentage points how much rare knowledge language models, from word trigrams to a pretrained transformer, retain under other mixtures of fresh data. Second, when the world itself changes, a lineage that copies its own past needs more fresh data to keep up the faster the world changes, as Rogers' model says it must and as analyses that assume a fixed target cannot; a lineage that restarts every generation from fixed pretrained weights is largely insulated, because those weights are knowledge that is never copied. Third, we close the loop: when the people who write a model's training data can ask the model instead of finding things out, the supply of fresh data becomes endogenous. At equilibrium, the model is exactly as reliable as it is costly for people to find out for themselves, the population is no better off than without the model, and every increase in the share of model-written text must be paid for by more people learning for themselves. Users who check what they are told and relearn when it is wrong break the paradox. Filters that select plausible text are conformist and accelerate collapse; only checks that carry new information act as critical learning. 

---
# How Far Can You Steer? Behavior-Capability Frontiers in Contrastive Activation Addition 

**Authors**: Deepanshu Raj  

**Link**: [PDF](https://arxiv.org/pdf/2611.06001)  

**Abstract**: Activation steering can modify language-model behavior without parameter updates, but behavioral efficacy alone does not determine whether an intervention is usable. Contrastive Activation Addition (CAA) steers targeted behaviors using activation directions derived from contrasting examples, potentially degrading unrelated model capabilities in the process. We therefore study CAA through a behavior-capability frontier, selecting interventions that maximize behavioral gain subject to an explicit bound on capability degradation. We evaluate this framework across four instruction-tuned models, seven behaviors, and capability-loss budgets of 1%, 3%, and 5% (absolute percentage points). Validation-selected behavioral effects generalize strongly to held-out data ($\rho = 0.945$), while joint behavior-capability survival among nonzero operating points rises from 37.5% at the 1% budget to 72.7% at 3%, with limited additional improvement at 5%. Control experiments indicate that these effects depend on learned behavior directions rather than generic activation perturbations. Representational geometry is significantly but only modestly predictive of causal cross-behavior transfer. Finally, secondary capability, out-of-distribution, and open-ended evaluations show that interventions satisfying protected benchmarks can still degrade untested capabilities or transfer unevenly to free-form generation. These results characterize activation steering as a constrained intervention whose usable range depends on model, behavior, direction, layer, and evaluation budget, rather than a universal control knob. 

---
# On the Design of Hyper-Connections: Simplification, Stability, and Scalability 

**Authors**: Xiangdong Zhang, Xiaohan Qin, Sunan Zou, Tuo Dai, Bizhao Shi, Zhuo Xia, Yebin Yang, Yizhou Chen, Yuliang Liu, Yu Cheng, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2611.05984)  

**Abstract**: Hyper-Connections (HC) architectures are increasingly adopted in frontier language models, yet their components remain insufficiently understood and introduce non-negligible training overhead. In this work, we systematically study the HC architecture to understand its components and identify simpler, more efficient designs. We find that learned read/write gating without mixing and unconstrained mixing with fixed gates form two effective interaction groups: either alone achieves performance comparable to mHC. We also identify repeated access to the expanded residual state as a major source of training overhead, even without mixing. These findings lead to SimpleHC, which simplifies the architecture by removing residual mixing and simplifies execution through shared memory access, using the resulting additive structure and shared gate inputs. Layer-specific gates share a group-entry state, enabling batched reads and deferred writes; scalar corrections exactly preserve preceding sublayer output contributions. Our analyses support both the retained residual structure and the design of shared memory access. SimpleHC incurs only 2.4% extra end-to-end training time over a vanilla residual baseline at the largest tested scale. It maintains performance comparable to mHC when scaling from 15B to 30B MoE models, exhibits smoother gradient dynamics during standard training, and remains stable under stress. We hope SimpleHC will encourage a broader pursuit of simplicity in foundation model design, making performance gains more accessible and practical at scale. 

---
# Robots in the City: Persona-Based Perceptions of Urban Robots 

**Authors**: Keonhee Jang, Kanghoon Lee, Seokwoo Yun, Namwoo Kim, Yoonjin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2611.05954)  

**Abstract**: Pedestrians increasingly encounter robots in urban public spaces without interacting with them, but how robots change the perceived safety of a street remains unclear. We examine how robot presence and type affect perceived safety across urban contexts. We generate 18 images of Seoul streets that vary in time of day, road class, and robot condition (no robot, a delivery robot, or a humanoid). Two multimodal large language models (MLLMs), Qwen3.8-27B and Gemma4-26B, each conditioned on 500 synthetic personas, choose the safer image in each pair, and 25 human respondents complete the same task. Both models prefer streets without robots and favor the delivery robot over the humanoid. Robots are chosen more often at night and less often by older personas, whereas road class has smaller effects. In the model rationales, the delivery robot appears as a physical obstacle, whereas the humanoid appears as either a threat or a source of protection. Like humans, both models chose robots more often at night, but chose the humanoid over the delivery robot far less often, and Gemma showed a night effect for the humanoid about three times that of humans. These results show that the effect of a robot on perceived safety depends on when it appears, and that persona-conditioned MLLMs capture only part of human judgments. 

---
# Modality-Aware Expert Pruning for MoE-Based Multimodal Large Language Models 

**Authors**: Donggeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.05927)  

**Abstract**: Mixture-of-Experts (MoE) enables multimodal large language models (MLLMs) to scale capacity efficiently, but expert parameters dominate memory even though only a few experts are activated per token: they account for over 90% of model size in modern MoE-MLLMs (e.g., 54 GB out of about 58 GB in BF16 for InternVL3.5-30B-A3B and Qwen3-VL-30B-A3B). Expert pruning has emerged as an effective compression strategy, but existing methods aggregate importance uniformly across tokens, ignoring that the same expert may serve different roles for vision versus language processing. To address this, we propose MAEP (Modality-Aware Expert Pruning), a training-free framework that decomposes expert importance by modality in a single forward pass. MAEP introduces (1) Delta-H Output ($\Delta H$), a redistribution-aware metric measuring the MoE output change when an expert is removed, and (2) low-attention vision tokens as a vision-side pruning signal that is dataset-stable: its expert-importance ranking remains stable across different datasets. Text-side importance serves as a preservation signal while vision-side importance serves as a pruning signal, and experts are ranked globally across layers. Experiments on three MoE-based MLLMs across multimodal and text benchmarks show that MAEP outperforms modality-agnostic baselines on the majority of multimodal benchmarks while remaining competitive on text-only tasks, reducing the BF16 expert weight footprint by about 27 GB (about 47% of total memory) so that the full weight footprint of an approximately 58 GB model fits within a single 40 GB GPU. Code is available at this https URL. 

---
# Routes of Reasoning: Reading and Reshaping Test-Time Compute in MoE Models 

**Authors**: Shengjie Wang, Mohsen Fayyaz, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2611.05919)  

**Abstract**: Reasoning models allocate test-time compute across distinct behaviors such as planning, deriving, and verifying. However, models frequently "overthink," continuing to verify and re-derive long after finding a correct answer. This wastes expensive compute and can even degrade performance. Existing interventions, such as token budgets or steering, primarily regulate reasoning duration rather than targeting specific behaviors. In this work, we demonstrate that Mixture-of-Experts (MoE) routers provide a natural, training-free interface for behavioral control. We first split reasoning traces into sentence-level segments and label each with one of six reasoning behaviors (planning, execution, restatement, fact, reflection, and conclusion), so that every token carries the behavior of its segment. Linear probes on router logits predict this label nearly as well as probes on the residual stream, the model's full hidden state and the standard target for probing, while using about half as many features. These signals generalize: experts associated with a reasoning behavior on one dataset predict the same behavior on the others, across six datasets spanning five domains (math, code, science, knowledge, and commonsense). Building on this, we selectively edit expert router scores during inference to curb overthinking. Across four models and four benchmarks, this intervention reduces the number of generated tokens by a median of 30.4% (and up to 63.3%). Crucially, it encourages the model to stop once the problem is solved, rather than skipping essential reasoning steps. Accuracy never drops by more than 0.5 percentage points, and frequently improves. Ultimately, MoE routing offers both an interpretable lens into reasoning states and a lightweight mechanism for optimizing test-time compute. 

---
# Autonomous agents for electricity-system planning 

**Authors**: Yoon Pyo Lee, Samrendra Roy, Jay Phil Yoo, Kazuma Kobayashi, Souvik Chakraborty, Syed Bahauddin Alam  

**Link**: [PDF](https://arxiv.org/pdf/2611.05911)  

**Abstract**: Planning electricity systems requires repeated decisions that connect regional resources, investment, operation and network constraints. Here we present an autonomous planning framework that carries prepared regional scenarios through proposal, numerical evaluation, feedback interpretation, redesign and final plan selection. Agents propose investment bounds, storage operating conditions and feeder interventions, interpret numerical feedback, and decide how to revise designs and when to stop. The framework automatically executes annual investment-and-dispatch optimization and representative-feeder verification, applies fixed acceptance criteria and retains the lowest-cost accepted candidate. We evaluate forty carbon-constrained trajectories across ten system-year scenarios in five electricity systems and four countries, using two model deployments and single- and multi-agent architectures. All forty retain candidates satisfying the implemented physical and fixed annual operational carbon-budget checks. Twenty-eight of forty matched reference designs exceed the new scenario ceilings. A separate archive of ninety-six trajectories under a common 1,000 Mt annual ceiling provides matched references and broader regional coverage. The results demonstrate autonomous planning under explicit emissions constraints using public regional data and published institutional demand outlooks. 

---
# PAYWORTHY: Can LLMs Predict Which Freelance Jobs an AI Agent Will Complete? 

**Authors**: Mohammad Reza Rezaei, Dingxian Wang, Brett Levert, Celeste Fejzo, Andrew Rabinovich, Zhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.05759)  

**Abstract**: Before an AI agent takes on a paid job, someone has to judge whether it is likely to succeed. We test whether frontier language models can make this call before any attempt. We propose PAYWORTHY which contains 1{,}494 real freelance job postings, sampled from 18.8M postings made between 2020 and 2026 on a freelancing platform. An agent attempts each job from the posting alone in a single run, and an LLM judge acting as the client decides whether to accept the work. We ask 18 models to forecast how likely the agent is to succeed and measure how well each model's forecasts rank the accepted jobs above the rejected ones (AUC, where 0.5 is a random ranking). The answer depends on how the agent is described. Told what the agent cannot do, every model ranks the jobs well (AUC 0.66--0.86). Told instead that it is a generic agent with a browser and code execution, the same models rank the same jobs about as well as a random ordering (0.45--0.56). These results hold under judges from four other model families (Claude, GLM, Kimi and Qwen), and the informed forecasts also rank the verdicts of two human raters.
Code is available at this https URL 

---
# Text-Only Domain Adaptation for Low-Resource Medical ASR: Shallow Fusion and Density Ratio Rescoring for Hausa 

**Authors**: Oche David Ankeli, Hassan Adelani Luqman, Theodora Ngozi Egbunike, Joshua Gottschalk, Bonaventure F. P. Dossou  

**Link**: [PDF](https://arxiv.org/pdf/2611.05744)  

**Abstract**: Automatic speech recognition (ASR) systems often struggle with specialised medical vocabulary, especially in low-resource languages where domain-specific speech data are scarce. We investigate text-only domain adaptation for Hausa medical ASR using the fixed MMS-1B-FL102 acoustic model and two decoding-time language-model fusion methods: Shallow Fusion (SF) and the Density Ratio Approach (DRA). A 4-gram KenLM trained on 19,421 Hausa health sentences serves as the medical target model, while a 75,372-sentence general-domain model provides the source prior for DRA. Evaluation uses 250 recordings of 25 physician-reviewed prompts from 10 speakers across six northern Nigerian states. Under prompt-disjoint five-fold cross-validation, SF and DRA reduce WER from 29.56% to 23.55% and 23.33%, respectively. Both significantly outperform greedy decoding (p < 0.001), while their difference is not significant (p = 0.423). However, general-LM-only SF achieves 23.96% WER, indicating that the medical LM's additional improvement over a general LM is modest. Medical Term Error Rate also improves under fusion, but does not establish a distinct medical-domain advantage. These findings demonstrate that lightweight, text-only language-model fusion can improve Hausa medical ASR without additional domain-specific speech collection. 

---
# The Male Abstraction Bias in Internet Culture and Large Language Models 

**Authors**: Douglas Guilbeault, Bhargav Srinivasa Desikan  

**Link**: [PDF](https://arxiv.org/pdf/2611.05730)  

**Abstract**: Two foundational aspects of human cognition, abstractness and gender, are typically examined as separable. Here, we find they are systematically linked throughout the entire space of concepts spanning single (N = 40,000) and multi-word expressions (N = 67,000): across billions of words and millions of images on the internet -- and across numerous measurement techniques -- more abstract concepts are more associated with men, an effect which replicates in languages around the world. This pattern not only holds across conceptual domains, but it also maps onto grammatical structure: verbs and adverbs are more abstract and more male-coded, a system-wide trace of stereotypes accentuating agentic qualities in men. Furthermore, this bias emerges in human and AI responses in free association tasks, indicating its internalization and its algorithmic reproduction. The male abstraction bias may underlie and unify a range of stereotypes, evidencing gender as a core organizing dimension in conceptual space. 

---
# Grounded but Not Immune: Confirmation Bias in Agent-as-a-judge 

**Authors**: Drake Benjamin Brown, Khushbu Pahwa, Yubin Ge, Salvatore Romeo, Akhil Agnihotri, Yi Zhang, Xiaofei Ma, Daniele Bonadiman  

**Link**: [PDF](https://arxiv.org/pdf/2611.05662)  

**Abstract**: Evaluating open-ended agent tasks increasingly falls to an Agent-as-a-Judge: an evaluator handed tools and the agent's full trajectory so it can verify a result against the environment. The premise is that a judge that gathers its own evidence should be grounded by it. We show that this grounding does not immunize the judge against confirmation bias: a single false pre-screening note, attributed to an authority and carrying no evidential bearing on the correct verdict, overturns up to 40% of a tool-using judge's correct verdicts across four judges and five datasets, even though the judge collects decisive evidence. The shift is driven by specific properties of the cue. We treat this cue as a controlled diagnostic of confirmation bias rather than an adversarial attack. Our central contribution is to explain how confirmation bias changes the judge's behavior in a trajectory. We classify changes in judge behavior and attribute changes to one of three failure modes: over-trust (verify, conclude correctly, then defer), hallucination (fabricate support for the flip), and under-investigation (stop at the first corroborating signal). For the two Claude judges, the failure is predominantly over-trust (67-75% of harmful flips), while smaller judges have error modes spread across all three. Lightweight mitigations blunt but never remove the bias. 

---
# AGORA: Reliability-Aware Aggregation of LLM Judges via Item-Response Theory 

**Authors**: Junhyung Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2611.05556)  

**Abstract**: LLM-as-a-judge is a common substitute for costly human evaluation, and aggregating several judges is the standard remedy for any single judge's bias. Judge pools mix judges of very different reliability, while the human-labeled sets that validate them hold a few hundred items or fewer. In this regime the per-item mean weights weak judges as heavily as strong ones, and confusion-table models such as Dawid-Skene must estimate a per-judge error table from too few items. We propose AGORA (Aggregation Of Reliability-weighted Assessments), an item-response-theory aggregator that places every item on one latent quality scale and describes each judge by a sensitivity parameter and ordered rating thresholds, so judge reliability is learned from all items jointly and without human labels. Across three benchmarks, AGORA matches or exceeds Dawid-Skene on large evaluation sets, and its advantage widens as the set shrinks toward a few hundred items. It remains accurate as pools grow more uneven, a label-free dispersion statistic indicates in advance when trust weighting helps, and its learned trust weights track per-judge human validity. 

---
# Iterated creation, grading, and revision of data science projects by language models 

**Authors**: Rohan Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2611.05548)  

**Abstract**: We evaluate the extent to which agentic AI models can complete an end-to-end data science project, and how this improves with feedback. A 'writer' model with specific tools creates a complete data science project including downloading, cleaning, and analyzing a dataset, and writing a short paper about it. The project is executed and a 'grader' model evaluates it. The writer can then revise its submission, and this repeats for a total of three rounds. We consider seven models from five developers: Anthropic, OpenAI, Google, DeepSeek, and this http URL. Each model is both a writer and a grader, so there are 49 writer-grader pairs and we run each pair three times. Of the 147 initial submissions, 107 run and render, and the average score increases from 59 per cent to 76 per cent after the final round of feedback. By writer the extent of the increase ranges from 3 percentage points for Anthropic's Opus 5.5 to 46 percentage points for Anthropic's Haiku 4.5, and by grader it ranges from 13 percentage points for Google's Gemini 3.8 Flash to 32 percentage points for Anthropic's Haiku 4.5. When a model grades its own writing the score tends to be about 9 percentage points higher than the average of what the other models give it. About half of the difference that we see from round to round is within the grader's own noise. Feedback generally results in the highest marks, but even just having the chance to revise, or having the additional budget, in terms of time or tool calls, without any feedback, results in considerable increases for some of the models but not for others. We find that how much such a loop helps depends on which model plays which role, but weaker models generally do not improve stronger models. 

---
# SlackBench: Benchmarking Agents on Collaborative Projects Grounded in Real Code Repositories 

**Authors**: Mingxuan Li, Huanzhi Mao, Ruitao Zheng, Jarod Stanbury James, Yunze Li, Natanya Anderson, Joseph E. Gonzalez, Alex Dimakis  

**Link**: [PDF](https://arxiv.org/pdf/2611.05512)  

**Abstract**: AI Agents are becoming digital coworkers that can browse the web, write code, run experiments, and assist humans over extended horizons. Yet existing benchmarks fall short of real knowledge work: many rely on synthetic data or isolated tasks, missing the complex, multi-party dynamics of collaborative projects, where critical information is scattered across conversations, code, and evolving decisions. Even simple workplace questions may require locating relevant channels and direct messages, recovering context from long discussions, inspecting repository state, reconciling stale or conflicting evidence, and synthesizing across sources.
We introduce SlackBench, a benchmark for evaluating agents on realistic research and engineering workflows that require joint reasoning over workplace communication and evolving repository state. Grounded in privacy-preserving reenactments of real academic research projects, SlackBench provides reproducible environments with realistic multi-party dialogue, evolving decisions, and corresponding code changes grounded in GitHub repositories and Google Docs. It contains 140 queries spanning heterogeneous-source agentic search, abstention on unanswerable or false-premise questions, goal-directed summarization, and circumstance inference over long conversations. These tasks are not reducible to static corpus question answering: agents must determine which conversations matter, traverse threads, inspect repository state, and synthesize evidence across communication and code artifacts. Across 33 frontier models and agent harnesses, the best system scores around 80% and most agents score below 66%, highlighting substantial headroom for improvement. SlackBench thus provides an extensible framework for evaluating agents in collaborative work environments. The SlackBench dataset is available at this https URL . 

---
# Unlocking Turing Tensor Cores: A Measured Throughput Account of From-Scratch Pre-training on a Single 22G RTX 2080 Ti 

**Authors**: Dezhou Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.05419)  

**Abstract**: The RTX 2080 Ti (TU102) ships full second-generation Tensor Cores in hardware, yet the vendor throttles FP16 matrix multiplication at the driver level, leaving usable tensor throughput well below the hardware peak. turing-unlock lifts this restriction with a kernel module that, before the nvidia driver binds the device, writes the FP16 speed selectors (SS0=0x88888888, SS1=0x8) and releases the ACR mutex; published measurements report cuBLAS FP16 matmul throughput rising from 55.7 to 91.4 TFLOPS (about 1.6x). On the same 22G 2080 Ti and under the same from-scratch pre-training protocol as the companion memory ledger (five tiers of Chinese encoder-decoder models, 2,451 steps per tier, 16,384 tokens per step, seed 42), we obtain a before/after throughput account: the small tier (75.7M parameters, 512-wide, 8+8 layers, activation checkpointing on) improves in steady-state throughput from 6,480 to 7,246 tok/s (+11.8%) with wall time cut from 1.77h to 1.61h (9.0% saved); the tiny tier goes from 15,150 to 15,866 tok/s (+4.7%) with wall time savings of 3.8%; the three smaller tiers (pico-micro) gain at most +0.3%; per-tier peak memory stays bit-identical, and final quality metrics are bit-identical or consistent within 10^-3. We conclude that the unlock gain is a function of matrix size, rising monotonically with tier -- imperceptible for pico-micro, visible from tiny upward, and cashed out as a double-digit percentage at small -- and that it is a pure throughput gain that leaves training dynamics untouched. 

---
# Challenge-Aware Knowledge Recruitment with Meta-Agent Supervision for Medical Multi-Agent Reasoning 

**Authors**: Riccardo Riccio, Haitz Sáez de Ocáriz Borde, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2611.05393)  

**Abstract**: Large language models perform strongly on standard medical QA but degrade on harder reasoning tasks, where multi-agent debate often fails to beat strong single-model prompting. We argue that a key limitation is that existing systems select experts and retrieve knowledge from the question alone, rather than from gaps that emerge during reasoning. We propose challenge-aware adaptive recruitment: an external CHALLENGER inspects the evolving reasoning trace and identifies specific factual or expertise gaps. These signals trigger targeted knowledge retrieval or mid-discussion specialist recruitment. On the MEDAGENTSBENCH hard subset (862 questions across nine biomedical datasets), upfront retrieval provides little benefit, while challenge-aware in-between retrieval performs better; on a multi-specialist backbone it improves accuracy on 8 of 9 datasets and adds 6 percentage points over Challenger Only. Ablations indicate benefits from both retrieval timing and gap-specific query signals, and a live web-search experiment shows that the timing advantage persists across retrieval backends. For specialist recruitment, self-triggered adaptation is unreliable, whereas Challenger-triggered recruitment raises GPT-4.1-mini MedQA accuracy from 46% to 54%, with positive direction on all tested generalization datasets. Naively combining retrieval and specialist recruitment degrades performance, highlighting the need to coordinate adaptive intervention channels. These results show that when and why external resources are recruited is a key design choice in multi-agent medical reasoning. 

---
# Agent Harness Awareness: Turning Verified Telemetry into Operational Control for Tool-Using Agents 

**Authors**: Manu Agrawal, Shrey Nagpal  

**Link**: [PDF](https://arxiv.org/pdf/2611.05391)  

**Abstract**: Tool descriptions tell an agent what an action can do, but they do not necessarily reveal whether that action is healthy or suitable at the moment of use. Execution harnesses often possess this information through monitors, traces, and recent failures while the model choosing the next action does not. We call this gap operational blindness. We introduce Agent Harness Awareness, a trust-separated design in which the host supplies a small, fresh, verified state snapshot before a consequential action while retaining measurement, enforcement, evaluation, and final dispatch authority. We test its tools-and-network instance with ToolRoute, a controlled isolation benchmark that holds the task and candidate semantics constant and varies the operational information available to the model. In a 216-decision ablation across GPT-5.6 Sol, Claude Sonnet 5, and Gemini 3.7 Flash, verified tool-health state reduced mean observable policy regret by 96.9% against a same-shape neutral control (4,252 ms to 131 ms) and raised single-dispatch completion from 69.4% to 95.8%. In a separate 27-decision end-to-end replication with live local TCP/HTTP execution, OpenTelemetry client spans, and Toxiproxy-injected faults, verified state reduced mean regret from 8,956 ms to 0 ms (zero excess observable operational cost) and completed the live action in 9/9 model-by-fault cases versus 5/9 for the neutral control. These studies establish the tools-and-network instance of Agent Harness Awareness: selected host telemetry materially improves pre-dispatch operational choice. Runtime state, hardware pressure, economics, retry-aware recovery, and heterogeneous selection remain open empirical extensions of the same architecture. 

---
