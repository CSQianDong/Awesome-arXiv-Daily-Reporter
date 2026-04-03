# Grounded Token Initialization for New Vocabulary in LMs for Generative Recommendation 

**Authors**: Daiwei Chen, Zhoutong Fu, Chengming Jiang, Haichao Zhang, Ran Zhou, Tan Wang, Chunnan Yao, Guoyao Li, Rui Cai, Yihan Cao, Ruijie Jiang, Fedor Borisyuk, Jianqiang Shen, Jingwei Wu, Ramya Korlakai Vinayak  

**Link**: [PDF](https://arxiv.org/pdf/2604.02324)  

**Abstract**: Language models (LMs) are increasingly extended with new learnable vocabulary tokens for domain-specific tasks, such as Semantic-ID tokens in generative recommendation. The standard practice initializes these new tokens as the mean of existing vocabulary embeddings, then relies on supervised fine-tuning to learn their representations. We present a systematic analysis of this strategy: through spectral and geometric diagnostics, we show that mean initialization collapses all new tokens into a degenerate subspace, erasing inter-token distinctions that subsequent fine-tuning struggles to fully recover. These findings suggest that \emph{token initialization} is a key bottleneck when extending LMs with new vocabularies. Motivated by this diagnosis, we propose the \emph{Grounded Token Initialization Hypothesis}: linguistically grounding novel tokens in the pretrained embedding space before fine-tuning better enables the model to leverage its general-purpose knowledge for novel-token domains. We operationalize this hypothesis as GTI (Grounded Token Initialization), a lightweight grounding stage that, prior to fine-tuning, maps new tokens to distinct, semantically meaningful locations in the pretrained embedding space using only paired linguistic supervision. Despite its simplicity, GTI outperforms both mean initialization and existing auxiliary-task adaptation methods in the majority of evaluation settings across multiple generative recommendation benchmarks, including industry-scale and public datasets. Further analyses show that grounded embeddings produce richer inter-token structure that persists through fine-tuning, corroborating the hypothesis that initialization quality is a key bottleneck in vocabulary extension. 

---
# No Single Best Model for Diversity: Learning a Router for Sample Diversity 

**Authors**: Yuhan Liu, Fangyuan Xu, Vishakh Padmakumar, Daphne Ippolito, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2604.02319)  

**Abstract**: When posed with prompts that permit a large number of valid answers, comprehensively generating them is the first step towards satisfying a wide range of users. In this paper, we study methods to elicit a comprehensive set of valid responses. To evaluate this, we introduce \textbf{diversity coverage}, a metric that measures the total quality scores assigned to each \textbf{unique} answer in the predicted answer set relative to the best possible answer set with the same number of answers. Using this metric, we evaluate 18 LLMs, finding no single model dominates at generating diverse responses to a wide range of open-ended prompts. Yet, per each prompt, there exists a model that outperforms all other models significantly at generating a diverse answer set. Motivated by this finding, we introduce a router that predicts the best model for each query. On NB-Wildchat, our trained router outperforms the single best model baseline (26.3% vs $23.8%). We further show generalization to an out-of-domain dataset (NB-Curated) as well as different answer-generation prompting strategies. Our work lays foundation for studying generating comprehensive answers when we have access to a suite of models. 

---
# CV-18 NER: Augmented Common Voice for Named Entity Recognition from Arabic Speech 

**Authors**: Youssef Saidi, Haroun Elleuch, Fethi Bougares  

**Link**: [PDF](https://arxiv.org/pdf/2604.02209)  

**Abstract**: End-to-end speech Named Entity Recognition (NER) aims to directly extract entities from speech. Prior work has shown that end-to-end (E2E) approaches can outperform cascaded pipelines for English, French, and Chinese, but Arabic remains under-explored due to its morphological complexity, the absence of short vowels, and limited annotated resources. We introduce CV-18 NER, the first publicly available dataset for NER from Arabic speech, created by augmenting the Arabic Common Voice 18 corpus with manual NER annotations following the fine-grained Wojood schema (21 entity types). We benchmark both pipeline systems (ASR + text NER) and E2E models based on Whisper and AraBEST-RQ. E2E systems substantially outperform the best pipeline configuration on the test set, reaching 37.0% CoER (AraBEST-RQ 300M) and 38.0% CVER (Whisper-medium). Further analysis shows that Arabic-specific self-supervised pretraining yields strong ASR performance, while multilingual weak supervision transfers more effectively to joint speech-to-entity learning, and that larger models may be harder to adapt in this low-resource setting. Our dataset and models are publicly released, providing the first open benchmark for end-to-end named entity recognition from Arabic speech this https URL. 

---
# Towards Position-Robust Talent Recommendation via Large Language Models 

**Authors**: Silin Du, Hongyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.02200)  

**Abstract**: Talent recruitment is a critical, yet costly process for many industries, with high recruitment costs and long hiring cycles. Existing talent recommendation systems increasingly adopt large language models (LLMs) due to their remarkable language understanding capabilities. However, most prior approaches follow a pointwise paradigm, which requires LLMs to repeatedly process some text and fails to capture the relationships among candidates in the list, resulting in higher token consumption and suboptimal recommendations. Besides, LLMs exhibit position bias and the lost-in-the-middle issue when answering multiple-choice questions and processing multiple long documents. To address these issues, we introduce an implicit strategy to utilize LLM's potential output for the recommendation task and propose L3TR, a novel framework for listwise talent recommendation with LLMs. In this framework, we propose a block attention mechanism and a local positional encoding method to enhance inter-document processing and mitigate the position bias and concurrent token bias issue. We also introduce an ID sampling method for resolving the inconsistency between candidate set sizes in the training phase and the inference phase. We design evaluation methods to detect position bias and token bias and training-free debiasing methods. Extensive experiments on two real-world datasets validated the effectiveness of L3TR, showing consistent improvements over existing baselines. 

---
# Neuro-RIT: Neuron-Guided Instruction Tuning for Robust Retrieval-Augmented Language Model 

**Authors**: Jaemin Kim, Jae O Lee, Sumyeong Ahn, Seo Yeon Park  

**Link**: [PDF](https://arxiv.org/pdf/2604.02194)  

**Abstract**: Retrieval-Augmented Language Models (RALMs) have demonstrated significant potential in knowledge-intensive tasks; however, they remain vulnerable to performance degradation when presented with irrelevant or noisy retrieved contexts. Existing approaches to enhance robustness typically operate via coarse-grained parameter updates at the layer or module level, often overlooking the inherent neuron-level sparsity of Large Language Models (LLMs). To address this limitation, we propose Neuro-RIT (Neuron-guided Robust Instruction Tuning), a novel framework that shifts the paradigm from dense adaptation to precision-driven neuron alignment. Our method explicitly disentangles neurons that are responsible for processing relevant versus irrelevant contexts using attribution-based neuron mining. Subsequently, we introduce a two-stage instruction tuning strategy that enforces a dual capability for noise robustness: achieving direct noise suppression by functionally deactivating neurons exclusive to irrelevant contexts, while simultaneously optimizing targeted layers for evidence distillation. Extensive experiments across diverse QA benchmarks demonstrate that Neuro-RIT consistently outperforms strong baselines and robustness-enhancing methods. 

---
# The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models at Expert Level 

**Authors**: Jeremy Herbst, Jae Hee Lee, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2604.02178)  

**Abstract**: Mixture-of-Experts (MoE) architectures have become the dominant choice for scaling Large Language Models (LLMs), activating only a subset of parameters per token. While MoE architectures are primarily adopted for computational efficiency, it remains an open question whether their sparsity makes them inherently easier to interpret than dense feed-forward networks (FFNs). We compare MoE experts and dense FFNs using $k$-sparse probing and find that expert neurons are consistently less polysemantic, with the gap widening as routing becomes sparser. This suggests that sparsity pressures both individual neurons and entire experts toward monosemanticity. Leveraging this finding, we zoom out from the neuron to the expert level as a more effective unit of analysis. We validate this approach by automatically interpreting hundreds of experts. This analysis allows us to resolve the debate on specialization: experts are neither broad domain specialists (e.g., biology) nor simple token-level processors. Instead, they function as fine-grained task experts, specializing in linguistic operations or semantic tasks (e.g., closing brackets in LaTeX). Our findings suggest that MoEs are inherently interpretable at the expert level, providing a clearer path toward large-scale model interpretability. Code is available at: this https URL 

---
# Adam's Law: Textual Frequency Law on Large Language Models 

**Authors**: Hongyuan Adam Lu, Z.L., Victor Wei, Zefan Zhang, Zhao Hong, Qiqi Xiang, Bowen Cao, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2604.02176)  

**Abstract**: While textual frequency has been validated as relevant to human cognition in reading speed, its relatedness to Large Language Models (LLMs) is seldom studied. We propose a novel research direction in terms of textual data frequency, which is an understudied topic, to the best of our knowledge. Our framework is composed of three units. First, this paper proposes Textual Frequency Law (TFL), which indicates that frequent textual data should be preferred for LLMs for both prompting and fine-tuning. Since many LLMs are closed-source in their training data, we propose using online resources to estimate the sentence-level frequency. We then utilize an input paraphraser to paraphrase the input into a more frequent textual expression. Next, we propose Textual Frequency Distillation (TFD) by querying LLMs to conduct story completion by further extending the sentences in the datasets, and the resulting corpora are used to adjust the initial estimation. Finally, we propose Curriculum Textual Frequency Training (CTFT) that fine-tunes LLMs in an increasing order of sentence-level frequency. Experiments are conducted on our curated dataset Textual Frequency Paired Dataset (TFPD) on math reasoning, machine translation, commonsense reasoning and agentic tool calling. Results show the effectiveness of our framework. 

---
# Do Lexical and Contextual Coreference Resolution Systems Degrade Differently under Mention Noise? An Empirical Study on Scientific Software Mentions 

**Authors**: Atilla Kaan Alkan, Felix Grezes, Jennifer Lynn Bartlett, Anna Kelbert, Kelly Lockhart, Alberto Accomazzi  

**Link**: [PDF](https://arxiv.org/pdf/2604.02171)  

**Abstract**: We present our participation in the SOMD 2026 shared task on cross-document software mention coreference resolution, where our systems ranked second across all three subtasks. We compare two fine-tuning-free approaches: Fuzzy Matching (FM), a lexical string-similarity method, and Context Aware Representations (CAR), which combines mention-level and document-level embeddings. Both achieve competitive performance across all subtasks (CoNLL F1 of 0.94-0.96), with CAR consistently outperforming FM by 1 point on the official test set, consistent with the high surface regularity of software names, which reduces the need for complex semantic reasoning. A controlled noise-injection study reveals complementary failure modes: as boundary noise increases, CAR loses only 0.07 F1 points from clean to fully corrupted input, compared to 0.20 for FM, whereas under mention substitution, FM degrades more gracefully (0.52 vs. 0.63). Our inference-time analysis shows that FM scales superlinearly with corpus size, whereas CAR scales approximately linearly, making CAR the more efficient choice at large scale. These findings suggest that system selection should be informed by both the noise profile of the upstream mention detector and the scale of the target corpus. We release our code to support future work on this underexplored task. 

---
# AstroConcepts: A Large-Scale Multi-Label Classification Corpus for Astrophysics 

**Authors**: Atilla Kaan Alkan, Felix Grezes, Sergi Blanco-Cuaresma, Jennifer Lynn Bartlett, Daniel Chivvis, Anna Kelbert, Kelly Lockhart, Alberto Accomazzi  

**Link**: [PDF](https://arxiv.org/pdf/2604.02156)  

**Abstract**: Scientific multi-label text classification suffers from extreme class imbalance, where specialized terminology exhibits severe power-law distributions that challenge standard classification approaches. Existing scientific corpora lack comprehensive controlled vocabularies, focusing instead on broad categories and limiting systematic study of extreme imbalance. We introduce AstroConcepts, a corpus of English abstracts from 21,702 published astrophysics papers, labeled with 2,367 concepts from the Unified Astronomy Thesaurus. The corpus exhibits severe label imbalance, with 76% of concepts having fewer than 50 training examples. By releasing this resource, we enable systematic study of extreme class imbalance in scientific domains and establish strong baselines across traditional, neural, and vocabulary-constrained LLM methods. Our evaluation reveals three key patterns that provide new insights into scientific text classification. First, vocabulary-constrained LLMs achieve competitive performance relative to domain-adapted models in astrophysics classification, suggesting a potential for parameter-efficient approaches. Second, domain adaptation yields relatively larger improvements for rare, specialized terminology, although absolute performance remains limited across all methods. Third, we propose frequency-stratified evaluation to reveal performance patterns that are hidden by aggregate scores, thereby making robustness assessment central to scientific multi-label evaluation. These results offer actionable insights for scientific NLP and establish benchmarks for research on extreme imbalance. 

---
# Brief Is Better: Non-Monotonic Chain-of-Thought Budget Effects in Function-Calling Language Agents 

**Authors**: Xuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2604.02155)  

**Abstract**: How much should a language agent think before taking action? Chain-of-thought (CoT) reasoning is widely assumed to improve agent performance, but the relationship between reasoning length and accuracy in structured tool-use settings remains poorly understood. We present a systematic study of CoT budget effects on function-calling agents, sweeping six token budgets (0--512) across 200 tasks from the Berkeley Function Calling Leaderboard v3 Multiple benchmark. Our central finding is a striking non-monotonic pattern on Qwen2.5-1.5B-Instruct: brief reasoning (32 tokens) dramatically improves accuracy by 45% relative over direct answers, from 44.0% to 64.0%, while extended reasoning (256 tokens) degrades performance well below the no-CoT baseline, to 25.0% (McNemar p < 0.001). A three-way error decomposition reveals the mechanism. At d = 0, 30.5% of tasks fail because the model selects the wrong function from the candidate set; brief CoT reduces this to 1.5%, effectively acting as a function-routing step, while long CoT reverses the gain, yielding 28.0% wrong selections and 18.0% hallucinated functions at d = 256. Oracle analysis shows that 88.6% of solvable tasks require at most 32 reasoning tokens, with an average of 27.6 tokens, and a finer-grained sweep indicates that the true optimum lies at 8--16 tokens. Motivated by this routing effect, we propose Function-Routing CoT (FR-CoT), a structured brief-CoT method that templates the reasoning phase as "Function: [name] / Key args: [...]," forcing commitment to a valid function name at the start of reasoning. FR-CoT achieves accuracy statistically equivalent to free-form d = 32 CoT while reducing function hallucination to 0.0%, providing a structural reliability guarantee without budget tuning. 

---
# GaelEval: Benchmarking LLM Performance for Scottish Gaelic 

**Authors**: Peter Devine, William Lamb, Beatrice Alex, Ignatius Ezeani, Dawn Knight, Mícheál J. Ó Meachair, Paul Rayson, Martin Wynne  

**Link**: [PDF](https://arxiv.org/pdf/2604.02135)  

**Abstract**: Multilingual large language models (LLMs) often exhibit emergent 'shadow' capabilities in languages without official support, yet their performance on these languages remains uneven and under-measured. This is particularly acute for morphosyntactically rich minority languages such as Scottish Gaelic, where translation benchmarks fail to capture structural competence. We introduce GaelEval, the first multi-dimensional benchmark for Gaelic, comprising: (i) an expert-authored morphosyntactic MCQA task; (ii) a culturally grounded translation benchmark and (iii) a large-scale cultural knowledge Q&A task. Evaluating 19 LLMs against a fluent-speaker human baseline ($n=30$), we find that Gemini 3 Pro Preview achieves $83.3\%$ accuracy on the linguistic task, surpassing the human baseline ($78.1\%$). Proprietary models consistently outperform open-weight systems, and in-language (Gaelic) prompting yields a small but stable advantage (+$2.4\%$). On the cultural task, leading models exceed $90\%$ accuracy, though most systems perform worse under Gaelic prompting and absolute scores are inflated relative to the manual benchmark. Overall, GaelEval reveals that frontier models achieve above-human performance on several dimensions of Gaelic grammar, demonstrates the effect of Gaelic prompting and shows a consistent performance gap favouring proprietary over open-weight models. 

---
# Reliable Control-Point Selection for Steering Reasoning in Large Language Models 

**Authors**: Haomin Zhuang, Hojun Yoo, Xiaonan Luo, Kehan Guo, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.02113)  

**Abstract**: Steering vectors offer a training-free mechanism for controlling reasoning behaviors in large language models, but constructing effective vectors requires identifying genuine behavioral signals in the model's hidden states. For behaviors that can be toggled via prompts, this is straightforward. However, many reasoning behaviors -- such as self-reflection -- emerge spontaneously and resist prompt-level control. Current methods detect these behaviors through keyword matching in chain-of-thought traces, implicitly assuming that every detected boundary encodes a genuine behavioral signal. We show that this assumption is overwhelmingly wrong: across 541 keyword-detected boundaries, 93.3\% are behaviorally unstable, failing to reproduce the detected behavior under re-generation from the same prefix. We develop a probabilistic model that formalizes intrinsic reasoning behaviors as stochastic events with context-dependent trigger probabilities, and show that unstable boundaries dilute the steering signal. Guided by this analysis, we propose stability filtering, which retains only boundaries where the model consistently reproduces the target behavior. Combined with a content-subspace projection that removes residual question-specific noise, our method achieves 0.784 accuracy on MATH-500 (+5.0 over the strongest baseline). The resulting steering vectors transfer across models in the same architecture family without re-extraction, improving Nemotron-Research-Reasoning-1.5B (+5.0) and DeepScaleR-1.5B-Preview (+6.0). Code is available at this https URL. 

---
# Prosodic ABX: A Language-Agnostic Method for Measuring Prosodic Contrast in Speech Representations 

**Authors**: Haitong Sun, Stephen McIntosh, Kwanghee Choi, Eunjung Yeo, Daisuke Saito, Nobuaki Minematsu  

**Link**: [PDF](https://arxiv.org/pdf/2604.02102)  

**Abstract**: Speech representations from self-supervised speech models (S3Ms) are known to be sensitive to phonemic contrasts, but their sensitivity to prosodic contrasts has not been directly measured. The ABX discrimination task has been used to measure phonemic contrast in S3M representations via minimal pairs. We introduce prosodic ABX, an extension of this framework to evaluate prosodic contrast with only a handful of examples and no explicit labels. Also, we build and release a dataset of English and Japanese minimal pairs and use it along with a Mandarin dataset to evaluate contrast in English stress, Japanese pitch accent, and Mandarin tone. Finally, we show that model and layer rankings are often preserved across several experimental conditions, making it practical for low-resource settings. 

---
# Optimizing RAG Rerankers with LLM Feedback via Reinforcement Learning 

**Authors**: Yuhang Wu, Xiangqing Shen, Fanfan Wang, Cangqi Zhou, Zhen Wu, Xinyu Dai, Rui Xia  

**Link**: [PDF](https://arxiv.org/pdf/2604.02091)  

**Abstract**: Rerankers play a pivotal role in refining retrieval results for Retrieval-Augmented Generation. However, current reranking models are typically optimized on static human annotated relevance labels in isolation, decoupled from the downstream generation process. This isolation leads to a fundamental misalignment: documents identified as topically relevant by information retrieval metrics often fail to provide the actual utility required by the LLM for precise answer generation. To bridge this gap, we introduce ReRanking Preference Optimization (RRPO), a reinforcement learning framework that directly aligns reranking with the LLM's generation quality. By formulating reranking as a sequential decision-making process, RRPO optimizes for context utility using LLM feedback, thereby eliminating the need for expensive human annotations. To ensure training stability, we further introduce a reference-anchored deterministic baseline. Extensive experiments on knowledge-intensive benchmarks demonstrate that RRPO significantly outperforms strong baselines, including the powerful list-wise reranker RankZephyr. Further analysis highlights the versatility of our framework: it generalizes seamlessly to diverse readers (e.g., GPT-4o), integrates orthogonally with query expansion modules like Query2Doc, and remains robust even when trained with noisy supervisors. 

---
# Goose: Anisotropic Speculation Trees for Training-Free Speculative Decoding 

**Authors**: Tao Jin, Phuong Minh Nguyen, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2604.02047)  

**Abstract**: Speculative decoding accelerates large language model inference by drafting multiple candidate tokens and verifying them in a single forward pass. Candidates are organized as a tree: deeper trees accept more tokens per step, but adding depth requires sacrificing breadth (fallback options) under a fixed verification budget. Existing training-free methods draft from a single token source and shape their trees without distinguishing candidate quality across origins. We observe that two common training-free token sources - n-gram matches copied from the input context, and statistical predictions from prior forward passes - differ dramatically in acceptance rate (~6x median gap, range 2-18x across five models and five benchmarks). We prove that when such a quality gap exists, the optimal tree is anisotropic (asymmetric): reliable tokens should form a deep chain while unreliable tokens spread as wide branches, breaking through the depth limit of balanced trees. We realize this structure in GOOSE, a training-free framework that builds an adaptive spine tree - a deep chain of high-acceptance context-matched tokens with wide branches of low-acceptance alternatives at each node. We prove that the number of tokens accepted per step is at least as large as that of either source used alone. On five LLMs (7B-33B) and five benchmarks, GOOSE achieves 1.9-4.3x lossless speedup, outperforming balanced-tree baselines by 12-33% under the same budget. 

---
# BidirLM: From Text to Omnimodal Bidirectional Encoders by Adapting and Composing Causal LLMs 

**Authors**: Nicolas Boizard, Théo Deschamps-Berger, Hippolyte Gisserot-Boukhlef, Céline Hudelot, Pierre Colombo  

**Link**: [PDF](https://arxiv.org/pdf/2604.02045)  

**Abstract**: Transforming causal generative language models into bidirectional encoders offers a powerful alternative to BERT-style architectures. However, current approaches remain limited: they lack consensus on optimal training objectives, suffer from catastrophic forgetting at scale, and fail to flexibly integrate the vast ecosystem of specialized generative models. In this work, through systematic ablations on the Gemma3 and Qwen3 families, we identify the key factors driving successful adaptation, highlighting the critical role of an often-omitted prior masking phase. To scale this process without original pre-training data, we introduce a dual strategy combining linear weight merging with a lightweight multi-domain data mixture that mitigates catastrophic forgetting. Finally, we augment our encoders by merging them with specialized causal models, seamlessly transferring modality- and domain-specific capabilities. This open-source recipe, designed for any causal decoder LLM, yields BidirLM, a family of five encoders that outperform alternatives on text, vision, and audio representation benchmarks. 

---
# Tracking the emergence of linguistic structure in self-supervised models learning from speech 

**Authors**: Marianne de Heer Kloots, Martijn Bentum, Hosein Mohebbi, Charlotte Pouw, Gaofei Shen, Willem Zuidema  

**Link**: [PDF](https://arxiv.org/pdf/2604.02043)  

**Abstract**: Self-supervised speech models learn effective representations of spoken language, which have been shown to reflect various aspects of linguistic structure. But when does such structure emerge in model training? We study the encoding of a wide range of linguistic structures, across layers and intermediate checkpoints of six Wav2Vec2 and HuBERT models trained on spoken Dutch. We find that different levels of linguistic structure show notably distinct layerwise patterns as well as learning trajectories, which can partially be explained by differences in their degree of abstraction from the acoustic signal and the timescale at which information from the input is integrated. Moreover, we find that the level at which pre-training objectives are defined strongly affects both the layerwise organization and the learning trajectories of linguistic structures, with greater parallelism induced by higher-order prediction tasks (i.e. iteratively refined pseudo-labels). 

---
# Why Gaussian Diffusion Models Fail on Discrete Data? 

**Authors**: Alexander Shabalin, Simon Elistratov, Viacheslav Meshchaninov, Ildus Sadrtdinov, Dmitry Vetrov  

**Link**: [PDF](https://arxiv.org/pdf/2604.02028)  

**Abstract**: Diffusion models have become a standard approach for generative modeling in continuous domains, yet their application to discrete data remains challenging. We investigate why Gaussian diffusion models with the DDPM solver struggle to sample from discrete distributions that are represented as a mixture of delta-distributions in the continuous space. Using a toy Random Hierarchy Model, we identify a critical sampling interval in which the density of noisified data becomes multimodal. In this regime, DDPM occasionally enters low-density regions between modes producing out-of-distribution inputs for the model and degrading sample quality. We show that existing heuristics, including self-conditioning and a solver we term q-sampling, help alleviate this issue. Furthermore, we demonstrate that combining self-conditioning with switching from DDPM to q-sampling within the critical interval improves generation quality on real data. We validate these findings across conditional and unconditional tasks in multiple domains, including text, programming code, and proteins. 

---
# $k$NNProxy: Efficient Training-Free Proxy Alignment for Black-Box Zero-Shot LLM-Generated Text Detection 

**Authors**: Kahim Wong, Kemou Li, Haiwei Wu, Jiantao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.02008)  

**Abstract**: LLM-generated text (LGT) detection is essential for reliable forensic analysis and for mitigating LLM misuse. Existing LGT detectors can generally be categorized into two broad classes: learning-based approaches and zero-shot methods. Compared with learning-based detectors, zero-shot methods are particularly promising because they eliminate the need to train task-specific classifiers. However, the reliability of zero-shot methods fundamentally relies on the assumption that an off-the-shelf proxy LLM is well aligned with the often unknown source LLM, a premise that rarely holds in real-world black-box scenarios. To address this discrepancy, existing proxy alignment methods typically rely on supervised fine-tuning of the proxy or repeated interactions with commercial APIs, thereby increasing deployment costs, exposing detectors to silent API changes, and limiting robustness under domain shift. Motivated by these limitations, we propose the $k$-nearest neighbor proxy ($k$NNProxy), a training-free and query-efficient proxy alignment framework that repurposes the $k$NN language model ($k$NN-LM) retrieval mechanism as a domain adapter for a fixed proxy LLM. Specifically, a lightweight datastore is constructed once from a target-reflective LGT corpus, either via fixed-budget querying or from existing datasets. During inference, nearest-neighbor evidence induces a token-level predictive distribution that is interpolated with the proxy output, yielding an aligned prediction without proxy fine-tuning or per-token API outputs. To improve robustness under domain shift, we extend $k$NNProxy into a mixture of proxies (MoP) that routes each input to a domain-specific datastore for domain-consistent retrieval. Extensive experiments demonstrate strong detection performance of our method. 

---
# SAFE: Stepwise Atomic Feedback for Error correction in Multi-hop Reasoning 

**Authors**: Daeyong Kwon, Soyoung Yoon, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2604.01993)  

**Abstract**: Multi-hop QA benchmarks frequently reward Large Language Models (LLMs) for spurious correctness, masking ungrounded or flawed reasoning steps. To shift toward rigorous reasoning, we propose SAFE, a dynamic benchmarking framework that replaces the ungrounded Chain-of-Thought (CoT) with a strictly verifiable sequence of grounded entities. Our framework operates across two phases: (1) train-time verification, where we establish an atomic error taxonomy and a Knowledge Graph (KG)-grounded verification pipeline to eliminate noisy supervision in standard benchmarks, identifying up to 14% of instances as unanswerable, and (2) inference-time verification, where a feedback model trained on this verified dataset dynamically detects ungrounded steps in real-time. Experimental results demonstrate that SAFE not only exposes the critical flaws of existing benchmarks at train-time, but also significantly outperforms standard baselines, achieving an average accuracy gain of 8.4 pp while guaranteeing verifiable trajectories at inference-time. 

---
# Diagnosing Translated Benchmarks: An Automated Quality Assurance Study of the EU20 Benchmark Suite 

**Authors**: Klaudia Thellmann, Bernhard Stadler, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2604.01957)  

**Abstract**: Machine-translated benchmark datasets reduce costs and offer scale, but noise, loss of structure, and uneven quality weaken confidence. What matters is not merely whether we can translate, but also whether we can measure and verify translation reliability at scale. We study translation quality in the EU20 benchmark suite, which comprises five established benchmarks translated into 20 languages, via a three-step automated quality assurance approach: (i) a structural corpus audit with targeted fixes; (ii) quality profiling using a neural metric (COMET, reference-free and reference-based) with translation service comparisons (DeepL / ChatGPT / Google); and (iii) an LLM-based span-level translation error landscape. Trends are consistent: datasets with lower COMET scores exhibit a higher share of accuracy/mistranslation errors at span level (notably HellaSwag; ARC is comparatively clean). Reference-based COMET on MMLU against human-edited samples points in the same direction. We release cleaned/corrected versions of the EU20 datasets, and code for reproducibility. In sum, automated quality assurance offers practical, scalable indicators that help prioritize review -- complementing, not replacing, human gold standards. 

---
# How to measure the optimality of word or gesture order with respect to the principle of swap distance minimization 

**Authors**: Ramon Ferrer-i-Cancho  

**Link**: [PDF](https://arxiv.org/pdf/2604.01938)  

**Abstract**: The structure of all the permutations of a sequence can be represented as a permutohedron, a graph where vertices are permutations and two vertices are linked if a swap of adjacent elements in the permutation of one of the vertices produces the permutation of the other vertex. It has been hypothesized that word orders in languages minimize the swap distance in the permutohedron: given a source order, word orders that are closer in the permutohedron should be less costly and thus more likely. Here we explain how to measure the degree of optimality of word order variation with respect to swap distance minimization. We illustrate the power of our novel mathematical framework by showing that crosslinguistic gestures are at least $77\%$ optimal. It is unlikely that the multiple times where crosslinguistic gestures hit optimality are due to chance. We establish the theoretical foundations for research on the optimality of word or gesture order with respect to swap distance minimization in communication systems. Finally, we introduce the quadratic assignment problem (QAP) into language research as an umbrella for multiple optimization problems and, accordingly, postulate a general principle of optimal assignment that unifies various linguistic principles including swap distance minimization. 

---
# Reliable News or Propagandist News? A Neurosymbolic Model Using Genre, Topic, and Persuasion Techniques to Improve Robustness in Classification 

**Authors**: Géraud Faye, Benjamin Icard, Morgane Casanova, Guillaume Gadek, Guillaume Gravier, Wassila Ouerdane, Céline Hudelot, Sylvain Gatepaille, Paul Égré  

**Link**: [PDF](https://arxiv.org/pdf/2604.01936)  

**Abstract**: Among news disorders, propagandist news are particularly insidious, because they tend to mix oriented messages with factual reports intended to look like reliable news. To detect propaganda, extant approaches based on Language Models such as BERT are promising but often overfit their training datasets, due to biases in data collection. To enhance classification robustness and improve generalization to new sources, we propose a neurosymbolic approach combining non-contextual text embeddings (fastText) with symbolic conceptual features such as genre, topic, and persuasion techniques. Results show improvements over equivalent text-only methods, and ablation studies as well as explainability analyses confirm the benefits of the added features.
Keywords: Information disorder, Fake news, Propaganda, Classification, Topic modeling, Hybrid method, Neurosymbolic model, Ablation, Robustness 

---
# ImplicitBBQ: Benchmarking Implicit Bias in Large Language Models through Characteristic Based Cues 

**Authors**: Bhaskara Hanuma Vedula, Darshan Anghan, Ishita Goyal, Ponnurangam Kumaraguru, Abhijnan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2604.01925)  

**Abstract**: Large Language Models increasingly suppress biased outputs when demographic identity is stated explicitly, yet may still exhibit implicit biases when identity is conveyed indirectly. Existing benchmarks use name based proxies to detect implicit biases, which carry weak associations with many social demographics and cannot extend to dimensions like age or socioeconomic status. We introduce ImplicitBBQ, a QA benchmark that evaluates implicit bias through characteristic based cues, culturally associated attributes that signal implicitly, across age, gender, region, religion, caste, and socioeconomic status. Evaluating 11 models, we find that implicit bias in ambiguous contexts is over six times higher than explicit bias in open weight models. Safety prompting and chain-of-thought reasoning fail to substantially close this gap; even few-shot prompting, which reduces implicit bias by 84%, leaves caste bias at four times the level of any other dimension. These findings indicate that current alignment and prompting strategies address the surface of bias evaluation while leaving culturally grounded stereotypic associations largely unresolved. We publicly release our code and dataset for model providers and researchers to benchmark potential mitigation techniques. 

---
# Is Clinical Text Enough? A Multimodal Study on Mortality Prediction in Heart Failure Patients 

**Authors**: Oumaima El Khettari, Virgile Barthet, Guillaume Hocquet, Joconde Weller, Emmanuel Morin, Pierre Zweigenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2604.01924)  

**Abstract**: Accurate short-term mortality prediction in heart failure (HF) remains challenging, particularly when relying on structured electronic health record (EHR) data alone. We evaluate transformer-based models on a French HF cohort, comparing text-only, structured-only, multimodal, and LLM-based approaches. Our results show that enriching clinical text with entity-level representations improves prediction over CLS embeddings alone, and that supervised multimodal fusion of text and structured variables achieves the best overall performance. In contrast, large language models perform inconsistently across modalities and decoding strategies, with text-only prompts outperforming structured or multimodal inputs. These findings highlight that entity-aware multimodal transformers offer the most reliable solution for short-term HF outcome prediction, while current LLM prompting remains limited for clinical decision support. 

---
# SURE: Synergistic Uncertainty-aware Reasoning for Multimodal Emotion Recognition in Conversations 

**Authors**: Yiqiang Cai, Chengyan Wu, Bolei Ma, Bo Chen, Yun Xue, Julia Hirschberg, Ziwei Gong  

**Link**: [PDF](https://arxiv.org/pdf/2604.01916)  

**Abstract**: Multimodal emotion recognition in conversations (MERC) requires integrating multimodal signals while being robust to noise and modeling contextual reasoning. Existing approaches often emphasize fusion but overlook uncertainty in noisy features and fine-grained reasoning. We propose SURE (Synergistic Uncertainty-aware REasoning) for MERC, a framework that improves robustness and contextual modeling. SURE consists of three components: an Uncertainty-Aware Mixture-of-Experts module to handle modality-specific noise, an Iterative Reasoning module for multi-turn reasoning over context, and a Transformer Gate module to capture intra- and inter-modal interactions. Experiments on benchmark MERC datasets show that SURE consistently outperforms state-of-the-art methods, demonstrating its effectiveness in robust multimodal reasoning. These results highlight the importance of uncertainty modeling and iterative reasoning in advancing emotion recognition in conversational settings. 

---
# Beyond Detection: Ethical Foundations for Automated Dyslexic Error Attribution 

**Authors**: Samuel Rose, Debarati Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2604.01853)  

**Abstract**: Dyslexic spelling errors exhibit systematic phonological and orthographic patterns that distinguish them from the errors produced by typically developing writers. While this observation has motivated dyslexic-specific spell-checking and assistive writing tools, prior work has focused predominantly on error correction rather than attribution, and has largely neglected the ethical risks. The risk of harmful labelling, covert screening, algorithmic bias, and institutional misuse that automated classification of learners entails requires the development of robust ethical and legal frameworks for research in this area. This paper addresses both gaps. We formulate dyslexic error attribution as a binary classification task. Given a misspelt word and its correct target form, determine whether the error pattern is characteristic of a dyslexic or non-dyslexic writer. We develop a comprehensive feature set capturing orthographic, phonological, and morphological properties of each error, and propose a twin-input neural model evaluated against traditional machine learning baselines under writer-independent conditions. The neural model achieves 93.01% accuracy and an F1-score of 94.01%, with phonetically plausible errors and vowel confusions emerging as the strongest attribution signals. We situate these technical results within an explicit ethics-first framework, analysing fairness across subgroups, the interpretability requirements of educational deployment, and the conditions, consent, transparency, human oversight, and recourse, under which a system could be responsibly used. We provide concrete guidelines for ethical deployment and an open discussion of the systems limitations and misuse potential. Our results demonstrate that dyslexic error attribution is feasible at high accuracy while underscoring that feasibility alone is insufficient for deployment in high-stakes educational contexts. 

---
# From Guessing to Placeholding: A Cost-Theoretic Framework for Uncertainty-Aware Code Completion 

**Authors**: Liang Zhu, Haolin Chen, Lidong Zhao, Xian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.01849)  

**Abstract**: While Large Language Models (LLMs) have demonstrated exceptional proficiency in code completion, they typically adhere to a Hard Completion (HC) paradigm, compelling the generation of fully concrete code even amidst insufficient context. Our analysis of 3 million real-world interactions exposes the limitations of this strategy: 61% of the generated suggestions were either edited after acceptance or rejected despite exhibiting over 80% similarity to the user's subsequent code, suggesting that models frequently make erroneous predictions at specific token positions. Motivated by this observation, we propose Adaptive Placeholder Completion (APC), a collaborative framework that extends HC by strategically outputting explicit placeholders at high-entropy positions, allowing users to fill directly via IDE navigation. Theoretically, we formulate code completion as a cost-minimization problem under uncertainty. Premised on the observation that filling placeholders incurs lower cost than correcting errors, we prove the existence of a critical entropy threshold above which APC achieves strictly lower expected cost than HC. We instantiate this framework by constructing training data from filtered real-world edit logs and design a cost-based reward function for reinforcement learning. Extensive evaluations across 1.5B--14B parameter models demonstrate that APC reduces expected editing costs from 19% to 50% while preserving standard HC performance. Our work provides both a theoretical foundation and a practical training framework for uncertainty-aware code completion, demonstrating that adaptive abstention can be learned end-to-end without sacrificing conventional completion quality. 

---
# PLOT: Enhancing Preference Learning via Optimal Transport 

**Authors**: Liang Zhu, Yuelin Bai, Xiankun Ren, Jiaxi Yang, Lei Zhang, Feiteng Fang, Hamid Alinejad-Rokny, Minghuan Tan, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.01837)  

**Abstract**: Preference learning in Large Language Models (LLMs) has advanced significantly, yet existing methods remain limited by modest performance gains, high computational costs, hyperparameter sensitivity, and insufficient modeling of global token-level relationships. We introduce PLOT, which enhances Preference Learning in fine-tuning-based alignment through a token-level loss derived from Optimal Transport. By formulating preference learning as an Optimal Transport Problem, PLOT aligns model outputs with human preferences while preserving the original distribution of LLMs, ensuring stability and robustness. Furthermore, PLOT leverages token embeddings to capture semantic relationships, enabling globally informed optimization. Experiments across two preference categories - Human Values and Logic & Problem Solving - spanning seven subpreferences demonstrate that PLOT consistently improves alignment performance while maintaining fluency and coherence. These results substantiate optimal transport as a principled methodology for preference learning, establishing a theoretically grounded framework that provides new insights for preference learning of LLMs. 

---
# DEFT: Distribution-guided Efficient Fine-Tuning for Human Alignment 

**Authors**: Liang Zhu, Feiteng Fang, Yuelin Bai, Longze Chen, Zhexiang Zhang, Minghuan Tan, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.01787)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF), using algorithms like Proximal Policy Optimization (PPO), aligns Large Language Models (LLMs) with human values but is costly and unstable. Alternatives have been proposed to replace PPO or integrate Supervised Fine-Tuning (SFT) and contrastive learning for direct fine-tuning and value alignment. However, these methods still require voluminous data to learn preferences and may weaken the generalization ability of LLMs. To further enhance alignment efficiency and performance while mitigating the loss of generalization ability, this paper introduces Distribution-guided Efficient Fine-Tuning (DEFT), an efficient alignment framework incorporating data filtering and distributional guidance by calculating the differential distribution reward based on the output distribution of language model and the discrepancy distribution of preference data. A small yet high-quality subset is filtered from the raw data using a differential distribution reward, which is then incorporated into existing alignment methods to guide the model's output distribution. Experimental results demonstrate that the methods enhanced by DEFT outperform the original methods in both alignment capability and generalization ability, with significantly reduced training time. 

---
# Taming CATS: Controllable Automatic Text Simplification through Instruction Fine-Tuning with Control Tokens 

**Authors**: Hanna Hubarava, Yingqiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.01779)  

**Abstract**: Controllable Automatic Text Simplification (CATS) produces user-tailored outputs, yet controllability is often treated as a decoding problem and evaluated with metrics that are not reflective to the measure of control. We observe that controllability in ATS is significantly constrained by data and evaluation. To this end, we introduce a domain-agnostic CATS framework based on instruction fine-tuning with discrete control tokens, steering open-source models to target readability levels and compression rates. Across three model families with different model sizes (Llama, Mistral, Qwen; 1-14B) and four domains (medicine, public administration, news, encyclopedic text), we find that smaller models (1-3B) can be competitive, but reliable controllability strongly depends on whether the training data encodes sufficient variation in the target attribute. Readability control (FKGL, ARI, Dale-Chall) is learned consistently, whereas compression control underperforms due to limited signal variability in the existing corpora. We further show that standard simplification and similarity metrics are insufficient for measuring control, motivating error-based measures for target-output alignment. Finally, our sampling and stratification experiments demonstrate that naive splits can introduce distributional mismatch that undermines both training and evaluation. 

---
# LiveMathematicianBench: A Live Benchmark for Mathematician-Level Reasoning with Proof Sketches 

**Authors**: Linyang He, Qiyao Yu, Hanze Dong, Baohao Liao, Xinxing Xu, Micah Goldblum, Jiang Bian, Nima Mesgarani  

**Link**: [PDF](https://arxiv.org/pdf/2604.01754)  

**Abstract**: Mathematical reasoning is a hallmark of human intelligence, and whether large language models (LLMs) can meaningfully perform it remains a central question in artificial intelligence and cognitive science. As LLMs are increasingly integrated into scientific workflows, rigorous evaluation of their mathematical capabilities becomes a practical necessity. Existing benchmarks are limited by synthetic settings and data contamination. We present LiveMathematicianBench, a dynamic multiple-choice benchmark for research-level mathematical reasoning built from recent arXiv papers published after model training cutoffs. By grounding evaluation in newly published theorems, it provides a realistic testbed beyond memorized patterns. The benchmark introduces a thirteen-category logical taxonomy of theorem types (e.g., implication, equivalence, existence, uniqueness), enabling fine-grained evaluation across reasoning forms. It employs a proof-sketch-guided distractor pipeline that uses high-level proof strategies to construct plausible but invalid answer choices reflecting misleading proof directions, increasing sensitivity to genuine understanding over surface-level matching. We also introduce a substitution-resistant mechanism to distinguish answer recognition from substantive reasoning. Evaluation shows the benchmark is far from saturated: Gemini-3.1-pro-preview, the best model, achieves only 43.5%. Under substitution-resistant evaluation, accuracy drops sharply: GPT-5.4 scores highest at 30.6%, while Gemini-3.1-pro-preview falls to 17.6%, below the 20% random baseline. A dual-mode protocol reveals that proof-sketch access yields consistent accuracy gains, suggesting models can leverage high-level proof strategies for reasoning. Overall, LiveMathematicianBench offers a scalable, contamination-resistant testbed for studying research-level mathematical reasoning in LLMs. 

---
# Detecting Toxic Language: Ontology and BERT-based Approaches for Bulgarian Text 

**Authors**: Melania Berbatova, Tsvetoslav Vasev  

**Link**: [PDF](https://arxiv.org/pdf/2604.01745)  

**Abstract**: Toxic content detection in online communication remains a significant challenge, with current solutions often inadvertently blocking valuable information, including medical terms and text related to minority groups. This paper presents a more nu-anced approach to identifying toxicity in Bulgarian text while preserving access to essential information. The research explores two distinct methodologies for detecting toxic content. The developed methodologies have po-tential applications across diverse online platforms and content moderation systems. First, we propose an ontology that models the potentially toxic words in Bulgarian language. Then, we compose a dataset that comprises 4,384 manually anno-tated sentences from Bulgarian online forums across four categories: toxic language, medical terminology, non-toxic lan-guage, and terms related to minority communities. We then train a BERT-based model for toxic language classification, which reaches a 0.89 F1 macro score. The trained model is directly applicable in a real environment and can be integrated as a com-ponent of toxic content detection systems. 

---
# Human-Guided Reasoning with Large Language Models for Vietnamese Speech Emotion Recognition 

**Authors**: Truc Nguyen, Then Tran, Binh Truong, Phuoc Nguyen T. H  

**Link**: [PDF](https://arxiv.org/pdf/2604.01711)  

**Abstract**: Vietnamese Speech Emotion Recognition (SER) remains challenging due to ambiguous acoustic patterns and the lack of reliable annotated data, especially in real-world conditions where emotional boundaries are not clearly separable. To address this problem, this paper proposes a human-machine collaborative framework that integrates human knowledge into the learning process rather than relying solely on data-driven models. The proposed framework is centered around LLM-based reasoning, where acoustic feature-based models are used to provide auxiliary signals such as confidence and feature-level evidence. A confidence-based routing mechanism is introduced to distinguish between easy and ambiguous samples, allowing uncertain cases to be delegated to LLMs for deeper reasoning guided by structured rules derived from human annotation behavior. In addition, an iterative refinement strategy is employed to continuously improve system performance through error analysis and rule updates. Experiments are conducted on a Vietnamese speech dataset of 2,764 samples across three emotion classes (calm, angry, panic), with high inter-annotator agreement (Fleiss Kappa = 0.8574), ensuring reliable ground truth. The proposed method achieves strong performance, reaching up to 86.59% accuracy and Macro F1 around 0.85-0.86, demonstrating its effectiveness in handling ambiguous and hard-to-classify cases. Overall, this work highlights the importance of combining data-driven models with human reasoning, providing a robust and model-agnostic approach for speech emotion recognition in low-resource settings. 

---
# Memory in the LLM Era: Modular Architectures and Strategies in a Unified Framework 

**Authors**: Yanchen Wu, Tenghui Lin, Yingli Zhou, Fangyuan Zhang, Qintian Guo, Xun Zhou, Sibo Wang, Xilin Liu, Yuchi Ma, Yixiang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2604.01707)  

**Abstract**: Memory emerges as the core module in the large language model (LLM)-based agents for long-horizon complex tasks (e.g., multi-turn dialogue, game playing, scientific discovery), where memory can enable knowledge accumulation, iterative reasoning and self-evolution. A number of memory methods have been proposed in the literature. However, these methods have not been systematically and comprehensively compared under the same experimental settings. In this paper, we first summarize a unified framework that incorporates all the existing agent memory methods from a high-level perspective. We then extensively compare representative agent memory methods on two well-known benchmarks and examine the effectiveness of all methods, providing a thorough analysis of those methods. As a byproduct of our experimental analysis, we also design a new memory method by exploiting modules in the existing methods, which outperforms the state-of-the-art methods. Finally, based on these findings, we offer promising future research opportunities. We believe that a deeper understanding of the behavior of existing methods can provide valuable new insights for future research. 

---
# Development and multi-center evaluation of domain-adapted speech recognition for human-AI teaming in real-world gastrointestinal endoscopy 

**Authors**: Ruijie Yang, Yan Zhu, Peiyao Fu, Te Luo, Zhihua Wang, Xian Yang, Quanlin Li, Pinghong Zhou, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.01705)  

**Abstract**: Automatic speech recognition (ASR) is a critical interface for human-AI interaction in gastrointestinal endoscopy, yet its reliability in real-world clinical settings is limited by domain-specific terminology and complex acoustic conditions. Here, we present EndoASR, a domain-adapted ASR system designed for real-time deployment in endoscopic workflows. We develop a two-stage adaptation strategy based on synthetic endoscopy reports, targeting domain-specific language modeling and noise robustness. In retrospective evaluation across six endoscopists, EndoASR substantially improves both transcription accuracy and clinical usability, reducing character error rate (CER) from 20.52% to 14.14% and increasing medical term accuracy (Med ACC) from 54.30% to 87.59%. In a prospective multi-center study spanning five independent endoscopy centers, EndoASR demonstrates consistent generalization under heterogeneous real-world conditions. Compared with the baseline Paraformer model, CER is reduced from 16.20% to 14.97%, while Med ACC is improved from 61.63% to 84.16%, confirming its robustness in practical deployment scenarios. Notably, EndoASR achieves a real-time factor (RTF) of 0.005, significantly faster than Whisper-large-v3 (RTF 0.055), while maintaining a compact model size of 220M parameters, enabling efficient edge deployment. Furthermore, integration with large language models demonstrates that improved ASR quality directly enhances downstream structured information extraction and clinician-AI interaction. These results demonstrate that domain-adapted ASR can serve as a reliable interface for human-AI teaming in gastrointestinal endoscopy, with consistent performance validated across multi-center real-world clinical settings. 

---
# On the Role of Reasoning Patterns in the Generalization Discrepancy of Long Chain-of-Thought Supervised Fine-Tuning 

**Authors**: Zhaoyi Li, Xiangyu Xi, Zhengyu Chen, Wei Wang, Gangwei Jiang, Ranran Shen, Linqi Song, Ying Wei, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2604.01702)  

**Abstract**: Supervised Fine-Tuning (SFT) on long Chain-of-Thought (CoT) trajectories has become a pivotal phase in building large reasoning models. However, how CoT trajectories from different sources influence the generalization performance of models remains an open question. In this paper, we conduct a comparative study using two sources of verified CoT trajectories generated by two competing models, \texttt{DeepSeek-R1-0528} and \texttt{gpt-oss-120b}, with their problem sets controlled to be identical. Despite their comparable performance, we uncover a striking paradox: lower training loss does not translate to better generalization. SFT on \texttt{DeepSeek-R1-0528} data achieves remarkably lower training loss, yet exhibits significantly worse generalization performance on reasoning benchmarks compared to those trained on \texttt{gpt-oss-120b}. To understand this paradox, we perform a multi-faceted analysis probing token-level SFT loss and step-level reasoning behaviors. Our analysis reveals a difference in reasoning patterns. \texttt{gpt-oss-120b} exhibits highly convergent and deductive trajectories, whereas \texttt{DeepSeek-R1-0528} favors a divergent and branch-heavy exploration pattern. Consequently, models trained with \texttt{DeepSeek-R1} data inherit inefficient exploration behaviors, often getting trapped in redundant exploratory branches that hinder them from reaching correct solutions. Building upon this insight, we propose a simple yet effective remedy of filtering out frequently branching trajectories to improve the generalization of SFT. Experiments show that training on selected \texttt{DeepSeek-R1-0528} subsets surprisingly improves reasoning performance by up to 5.1% on AIME25, 5.5% on BeyondAIME, and on average 3.6% on five benchmarks. 

---
# PRISM: Probability Reallocation with In-Span Masking for Knowledge-Sensitive Alignment 

**Authors**: Chenning Xu, Mao Zheng, Mingyang Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.01682)  

**Abstract**: Supervised fine-tuning (SFT) with token-level hard labels can amplify overconfident imitation of factually unsupported targets, causing hallucinations that propagate in multi-sentence generation. We study an augmented SFT setting in which training instances include coarse sentence-level factuality risk labels and inter-sentence dependency annotations, providing structured signals about where factual commitments are weakly supported. We propose \textbf{PRISM}, a differentiable risk-gated framework that modifies learning only at fact-critical positions. PRISM augments standard SFT with a lightweight, model-aware probability reallocation objective that penalizes high-confidence predictions on risky target tokens, with its scope controlled by span-level risk weights and model-aware gating. Experiments on hallucination-sensitive factual benchmarks and general evaluations show that PRISM improves factual aggregates across backbones while maintaining a competitive overall capability profile. Ablations further show that the auxiliary signal is most effective when used conservatively, and that knowledge masking and model-aware reallocation play complementary roles in balancing factual correction and capability preservation. 

---
# PRCCF: A Persona-guided Retrieval and Causal-aware Cognitive Filtering Framework for Emotional Support Conversation 

**Authors**: Yanxin Luo, Xiaoyu Zhang, Jing Li, Yan Gao, Donghong Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.01671)  

**Abstract**: Emotional Support Conversation (ESC) aims to alleviate individual emotional distress by generating empathetic responses. However, existing methods face challenges in effectively supporting deep contextual understanding. To address this issue, we propose PRCCF, a Persona-guided Retrieval and Causality-aware Cognitive Filtering framework. Specifically, the framework incorporates a persona-guided retrieval mechanism that jointly models semantic compatibility and persona alignment to enhance response generation. Furthermore, it employs a causality-aware cognitive filtering module to prioritize causally relevant external knowledge, thereby improving contextual cognitive understanding for emotional reasoning. Extensive experiments on the ESConv dataset demonstrate that PRCCF outperforms state-of-the-art baselines on both automatic metrics and human evaluations. Our code is publicly available at: this https URL. 

---
# What Do Claim Verification Datasets Actually Test? A Reasoning Trace Analysis 

**Authors**: Delip Rao, Chris Callison-Burch  

**Link**: [PDF](https://arxiv.org/pdf/2604.01657)  

**Abstract**: Despite rapid progress in claim verification, we lack a systematic understanding of what reasoning these benchmarks actually exercise. We generate structured reasoning traces for 24K claim-verification examples across 9 datasets using GPT-4o-mini and find that direct evidence extraction dominates, while multi-sentence synthesis and numerical reasoning are severely under-represented. A dataset-level breakdown reveals stark biases: some datasets almost exclusively test lexical matching, while others require information synthesis in roughly half of cases. Using a compact 1B-parameter reasoning verifier, we further characterize five error types and show that error profiles vary dramatically by domain -- general-domain verification is dominated by lexical overlap bias, scientific verification by overcautiousness, and mathematical verification by arithmetic reasoning failures. Our findings suggest that high benchmark scores primarily reflect retrieval-plus-entailment ability. We outline recommendations for building more challenging evaluation suites that better test the reasoning capabilities verification systems need. 

---
# Fragile Reasoning: A Mechanistic Analysis of LLM Sensitivity to Meaning-Preserving Perturbations 

**Authors**: Shou-Tzu Han, Rodrigue Rizk, KC Santosh  

**Link**: [PDF](https://arxiv.org/pdf/2604.01639)  

**Abstract**: Large language models demonstrate strong performance on mathematical reasoning benchmarks, yet remain surprisingly fragile to meaning-preserving surface perturbations. We systematically evaluate three open-weight LLMs, Mistral-7B, Llama-3-8B, and Qwen2.5-7B, on 677 GSM8K problems paired with semantically equivalent variants generated through name substitution and number format paraphrasing. All three models exhibit substantial answer-flip rates (28.8%-45.1%), with number paraphrasing consistently more disruptive than name swaps. To trace the mechanistic basis of these failures, we introduce the Mechanistic Perturbation Diagnostics (MPD) framework, combining logit lens analysis, activation patching, component ablation, and the Cascading Amplification Index (CAI) into a unified diagnostic pipeline. CAI, a novel metric quantifying layer-wise divergence amplification, outperforms first divergence layer as a failure predictor for two of three architectures (AUC up to 0.679). Logit lens reveals that flipped samples diverge from correct predictions at significantly earlier layers than stable samples. Activation patching reveals a stark architectural divide in failure localizability: Llama-3 failures are recoverable by patching at specific layers (43/60 samples), while Mistral and Qwen failures are broadly distributed (3/60 and 0/60). Based on these diagnostic signals, we propose a mechanistic failure taxonomy (localized, distributed, and entangled) and validate it through targeted repair experiments: steering vectors and layer fine-tuning recover 12.2% of localized failures (Llama-3) but only 7.2% of entangled (Qwen) and 5.2% of distributed (Mistral) failures. 

---
# Grounding AI-in-Education Development in Teachers' Voices: Findings from a National Survey in Indonesia 

**Authors**: Nurul Aisyah, Muhammad Dehan Al Kautsar, Arif Hidayat, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2604.01630)  

**Abstract**: Despite emerging use in Indonesian classrooms, there is limited large-scale, teacher-centred evidence on how AI is used in practice and what support teachers need, hindering the development of context-appropriate AI systems and policies. To address this gap, we conduct a nationwide survey of 349 K-12 teachers across elementary, junior high, and senior high schools. We find increasing use of AI for pedagogy, content development, and teaching media, although adoption remains uneven. Elementary teachers report more consistent use, while senior high teachers engage less; mid-career teachers assign higher importance to AI, and teachers in Eastern Indonesia perceive greater value. Across levels, teachers primarily use AI to reduce instructional preparation workload (e.g., assessment, lesson planning, and material development). However, generic outputs, infrastructure constraints, and limited contextual alignment continue to hinder effective classroom integration. 

---
# Swift-SVD: Theoretical Optimality Meets Practical Efficiency in Low-Rank LLM Compression 

**Authors**: Ruoling Qi, Yirui Liu, Xuaner Wu, Xiangyu Wang, Ming Li, Chen Chen, Jian Chen, Yin Chen, Qizhen Weng  

**Link**: [PDF](https://arxiv.org/pdf/2604.01609)  

**Abstract**: The deployment of Large Language Models is constrained by the memory and bandwidth demands of static weights and dynamic Key-Value cache. SVD-based compression provides a hardware-friendly solution to reduce these costs. However, existing methods suffer from two key limitations: some are suboptimal in reconstruction error, while others are theoretically optimal but practically inefficient. In this paper, we propose Swift-SVD, an activation-aware, closed-form compression framework that simultaneously guarantees theoretical optimum, practical efficiency and numerical stability. Swift-SVD incrementally aggregates covariance of output activations given a batch of inputs and performs a single eigenvalue decomposition after aggregation, enabling training-free, fast, and optimal layer-wise low-rank approximation. We employ effective rank to analyze local layer-wise compressibility and design a dynamic rank allocation strategy that jointly accounts for local reconstruction loss and end-to-end layer importance. Extensive experiments across six LLMs and eight datasets demonstrate that Swift-SVD outperforms state-of-the-art baselines, achieving optimal compression accuracy while delivering 3-70X speedups in end-to-end compression time. Our code will be released upon acceptance. 

---
# DeltaMem: Towards Agentic Memory Management via Reinforcement Learning 

**Authors**: Qi Zhang, Shen Huang, Chu Liu, Shouqing Yang, Junbo Zhao, Haobo Wang, Pengjun Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.01560)  

**Abstract**: Recent advances in persona-centric memory have revealed the powerful capability of multi-agent systems in managing persona memory, especially in conversational scenarios. However, these complex frameworks often suffer from information loss and are fragile across varying scenarios, resulting in suboptimal performance. In this paper, we propose DeltaMem, an agentic memory management system that formulates persona-centric memory management as an end-to-end task within a single-agent setting. To further improve the performance of our agentic memory manager, we draw inspiration from the evolution of human memory and synthesize a user-assistant dialogue dataset along with corresponding operation-level memory updating labels. Building on this, we introduce a novel Memory-based Levenshtein Distance to formalize the memory updating reward, and propose a tailored reinforcement learning framework to further enhance the management capabilities of DeltaMem. Extensive experiments show that both training-free and RL-trained DeltaMem outperform all product-level baselines across diverse long-term memory benchmarks, including LoCoMo, HaluMem, and PersonaMem. 

---
# Countering Catastrophic Forgetting of Large Language Models for Better Instruction Following via Weight-Space Model Merging 

**Authors**: Mengxian Lyu, Cheng Peng, Ziyi Chen, Mengyuan Zhang, Jieting Li Lu, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.01538)  

**Abstract**: Large language models have been adopted in the medical domain for clinical documentation to reduce clinician burden. However, studies have reported that LLMs often "forget" a significant amount of instruction-following ability when fine-tuned using a task-specific medical dataset, a critical challenge in adopting general-purpose LLMs for clinical applications. This study presents a model merging framework to efficiently adapt general-purpose LLMs to the medical domain by countering this forgetting issue. By merging a clinical foundation model (GatorTronLlama) with a general instruct model (Llama-3.1-8B-Instruct) via interpolation-based merge methods, we seek to derive a domain-adapted model with strong performance on clinical tasks while retaining instruction-following ability. Comprehensive evaluation across medical benchmarks and five clinical generation tasks (e.g., radiology and discharge summarization) shows that merged models can effectively mitigate catastrophic forgetting, preserve clinical domain expertise, and retain instruction-following ability. In addition, our model merging strategies demonstrate training efficiency, achieving performance on par with fully fine-tuned baselines under severely constrained supervision (e.g., 64-shot vs. 256-shot). Consequently, weight-space merging constitutes a highly scalable solution for adapting open-source LLMs to clinical applications, facilitating broader deployment in resource-constrained healthcare environments. 

---
# Read More, Think More: Revisiting Observation Reduction for Web Agents 

**Authors**: Masafumi Enomoto, Ryoma Obara, Haochen Zhang, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2604.01535)  

**Abstract**: Web agents based on large language models (LLMs) rely on observations of web pages -- commonly represented as HTML -- as the basis for identifying available actions and planning subsequent steps. Prior work has treated the verbosity of HTML as an obstacle to performance and adopted observation reduction as a standard practice. We revisit this trend and demonstrate that the optimal observation representation depends on model capability and thinking token budget: (1) compact observations (accessibility trees) are preferable for lower-capability models, while detailed observations (HTML) are advantageous for higher-capability models; moreover, increasing thinking tokens further amplifies the benefit of HTML. (2) Our error analysis suggests that higher-capability models exploit layout information in HTML for better action grounding, while lower-capability models suffer from increased hallucination under longer inputs. We also find that incorporating observation history improves performance across most models and settings, and a diff-based representation offers a token-efficient alternative. Based on these findings, we suggest practical guidelines: adaptively select observation representations based on model capability and thinking token budget, and incorporate observation history using diff-based representations. 

---
# Why Instruction-Based Unlearning Fails in Diffusion Models? 

**Authors**: Zeliang Zhang, Rui Sun, Jiani Liu, Qi Wu, Chenliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.01514)  

**Abstract**: Instruction-based unlearning has proven effective for modifying the behavior of large language models at inference time, but whether this paradigm extends to other generative models remains unclear. In this work, we investigate instruction-based unlearning in diffusion-based image generation models and show, through controlled experiments across multiple concepts and prompt variants, that diffusion models systematically fail to suppress targeted concepts when guided solely by natural-language unlearning instructions. By analyzing both the CLIP text encoder and cross-attention dynamics during the denoising process, we find that unlearning instructions do not induce sustained reductions in attention to the targeted concept tokens, causing the targeted concept representations to persist throughout generation. These results reveal a fundamental limitation of prompt-level instruction in diffusion models and suggest that effective unlearning requires interventions beyond inference-time language control. 

---
# Magic, Madness, Heaven, Sin: LLM Output Diversity is Everything, Everywhere, All at Once 

**Authors**: Harnoor Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2604.01504)  

**Abstract**: Research on Large Language Models (LLMs) studies output variation across generation, reasoning, alignment, and representational analysis, often under the umbrella of "diversity." Yet the terminology remains fragmented, largely because the normative objectives underlying tasks are rarely made explicit. We introduce the Magic, Madness, Heaven, Sin framework, which models output variation along a homogeneity-heterogeneity axis, where valuation is determined by the task and its normative objective. We organize tasks into four normative contexts: epistemic (factuality), interactional (user utility), societal (representation), and safety (robustness). For each, we examine the failure modes and vocabulary such as hallucination, mode collapse, bias, and erasure through which variation is studied. We apply the framework to analyze all pairwise cross-contextual interactions, revealing that optimizing for one objective, such as improving safety, can inadvertently harm demographic representation or creative diversity. We argue for context-aware evaluation of output variation, reframing it as a property shaped by task objectives rather than a model's intrinsic trait. 

---
# A Dynamic Atlas of Persian Poetic Symbolism: Families, Fields, and the Historical Rewiring of Meaning 

**Authors**: Kourosh Shahnazari, Seyed Moein Ayyoubzadeh, Mohammadali Keshtparvar  

**Link**: [PDF](https://arxiv.org/pdf/2604.01467)  

**Abstract**: Persian poetry is often remembered through recurrent symbols before it is remembered through plot. Wine vessels, gardens, flames, sacred titles, bodily beauty, and courtly names return across centuries, yet computational work still tends to flatten this material into isolated words or broad document semantics. That misses a practical unit of organization in Persian poetics: related forms travel as families and gain force through recurring relations. Using a corpus of 129,451 poems, we consolidate recurrent forms into traceable families, separate imagistic material from sacred and courtly reference, and map their relations in a multi-layer graph. The symbolic core is relatively sparse, the referential component much denser, and the attachment zone between them selective rather than diffuse. Across 11 Hijri-century bins, some families remain widely distributed, especially Shab (Night), Ruz (Day), and Khaak (Earth). Wine vessels, garden space, flame, and lyric sound strengthen later, while prestige-coded and heroic-courtly vocabulary is weighted earlier. Century-specific graphs show change in arrangement as well as membership. Modularity rises, cross-scope linkage declines, courtly bridges weaken, and sacred bridges strengthen. Hub positions shift too: Kherqe (Sufi Robe) gains late prominence, Farkhondeh {Blessed} and Banafsheh (Violet) recede, and Saaghar (Wine Cup) stays central across the chronology. In this corpus, Persian symbolism appears less as a fixed repertory than as a long-lived system whose internal weights and connections change over time. 

---
# Wired for Overconfidence: A Mechanistic Perspective on Inflated Verbalized Confidence in LLMs 

**Authors**: Tianyi Zhao, Yinhan He, Wendy Zheng, Yujie Zhang, Chen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.01457)  

**Abstract**: Large language models are often not just wrong, but \emph{confidently wrong}: when they produce factually incorrect answers, they tend to verbalize overly high confidence rather than signal uncertainty. Such verbalized overconfidence can mislead users and weaken confidence scores as a reliable uncertainty signal, yet its internal mechanisms remain poorly understood. We present a circuit-level mechanistic analysis of this inflated verbalized confidence in LLMs, organized around three axes: capturing verbalized confidence as a differentiable internal signal, identifying the circuits that causally inflate it, and leveraging these insights for targeted inference-time recalibration. Across two instruction-tuned LLMs on three datasets, we find that a compact set of MLP blocks and attention heads, concentrated in middle-to-late layers, consistently writes the confidence-inflation signal at the final token position. We further show that targeted inference-time interventions on these circuits substantially improve calibration. Together, our results suggest that verbalized overconfidence in LLMs is driven by identifiable internal circuits and can be mitigated through targeted intervention. 

---
# Are Finer Citations Always Better? Rethinking Granularity for Attributed Generation 

**Authors**: Hexuan Wang, Jingyu Zhang, Benjamin Van Durme, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2604.01432)  

**Abstract**: Citation granularity - whether to cite individual sentences, paragraphs, or documents - is a critical design choice in attributed generation. While fine-grained citations are often preferred for precise human verification, their impact on model performance remains under-explored. We analyze four model scales (8B-120B) and demonstrate that enforcing fine-grained citations degrades attribution quality by 16-276% compared to the best-performing granularity. We observe a consistent performance pattern where attribution quality peaks at intermediate granularities (paragraph-level). Our analysis suggests that fine-grained (sentence-level) citations disrupt necessary semantic dependencies for attributing evidence to answer claims, while excessively coarse citations (multi-paragraph) introduce distracting noise. Importantly, the magnitude of this performance gap varies non-monotonically with model scale: fine-grained constraints disproportionately penalize larger models, suggesting that atomic citation units disrupt the multi-sentence information synthesis at which these models excel. Strikingly, citation-optimal granularity leads to substantial gains in attribution quality while preserving or even improving answer correctness. Overall, our findings demonstrate that optimizing solely for human verification via fine-grained citation disregards model constraints, compromising both attribution faithfulness and generation reliability. Instead, effective attribution requires aligning citation granularity with the model's natural semantic scope. 

---
# The power of context: Random Forest classification of near synonyms. A case study in Modern Hindi 

**Authors**: Jacek Bąkowski  

**Link**: [PDF](https://arxiv.org/pdf/2604.01425)  

**Abstract**: Synonymy is a widespread yet puzzling linguistic phenomenon. Absolute synonyms theoretically should not exist, as they do not expand language's expressive potential. However, it was suggested that even if synonyms denote the same concept, they may reflect different perspectives or carry distinct cultural associations, claims that have rarely been tested quantitatively.
In Hindi, prolonged contact with Persian produced many Perso-Arabic loanwords coexisting with their Sanskrit counterpart, forming numerous synonym pairs. This study investigates whether centuries after these borrowings appeared in the Subcontinent their origin can still be distinguished using distributional data alone and regardless of their semantic content.
A Random Forest trained on word embeddings of Hindi synonyms successfully classified words by Sanskrit or Perso-Arabic origin, even when they were semantically unrelated, suggesting that usage patterns preserve traces of etymology. These findings provide quantitative evidence that context encodes etymological signals and that synonymy may reflect subtle but systematic distinctions linked to origin. They support the idea that synonymous words can offer different perspectives and that etymologically related words may form distinct conceptual subspaces, creating a new type of semantic frame shaped by historical origin. Overall, the results highlight the power of context in capturing nuanced distinctions beyond traditional semantic similarity. 

---
# Cost-Efficient Estimation of General Abilities Across Benchmarks 

**Authors**: Michael Krumdick, Adam Wiemerslage, Seth Ebner, Charles Lovering, Chris Tanner  

**Link**: [PDF](https://arxiv.org/pdf/2604.01418)  

**Abstract**: Thousands of diverse benchmarks have been developed to measure the quality of large language models (LLMs). Yet prior work has demonstrated that LLM performance is often sufficiently explained by a small set of latent factors, or abilities. This suggests the potential for more efficient and principled benchmarking, but it remains difficult to compare the quality of different methods. Motivated by predictive validity, we argue that the quality of a benchmarking framework should be grounded in how efficiently it enables the prediction of model performance on unseen tasks. To analyze this objective, we collect the "Wide-scale Item Level Dataset" (WILD), a dataset of item-model response pairs, comprising evaluations of 65 models on 109,564 unique items spanning 163 tasks drawn from 27 datasets. This dataset enables the first analysis of how different techniques can predict a model's performance on a large, diverse collection of unseen tasks under different budget constraints. We demonstrate that combining a modified multidimensional item response theory (IRT) model with adaptive item selection driven by optimal experimental design can predict performance on 112 held-out benchmark tasks with a mean absolute error (MAE) of less than 7%, and can do so after observing only 16 items. We further demonstrate that incorporating cost-aware discount factors into our selection criteria can reduce the total tokens needed to reach 7% MAE from 141,000 tokens to only 22,000, an 85% reduction in evaluation cost. 

---
# Adaptive Stopping for Multi-Turn LLM Reasoning 

**Authors**: Xiaofan Zhou, Huy Nguyen, Bo Yu, Chenxi Liu, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.01413)  

**Abstract**: Large Language Models (LLMs) increasingly rely on multi-turn reasoning and interaction, such as adaptive retrieval-augmented generation (RAG) and ReAct-style agents, to answer difficult questions. These methods improve accuracy by iteratively retrieving information, reasoning, or acting, but introduce a key challenge: \textbf{When should the model stop?} Existing approaches rely on heuristic stopping rules or fixed turn budgets and provide no formal guarantees that the final prediction still contains the correct answer. This limitation is particularly problematic in high-stakes domains such as finance and healthcare, where unnecessary turns increase cost and latency, while stopping too early risks incorrect decisions. Conformal prediction (CP) provides formal coverage guarantees, but existing LLM-CP methods only apply to a single model output and cannot handle multi-turn pipelines with adaptive stopping. To address this gap, we propose Multi-Turn Language Models with Conformal Prediction (MiCP), the first CP framework for multi-turn reasoning. MiCP allocates different error budgets across turns, enabling the model to stop early while maintaining an overall coverage guarantee. We demonstrate MiCP on adaptive RAG and ReAct, where it achieves the target coverage on both single-hop and multi-hop question answering benchmarks while reducing the number of turns, inference cost, and prediction set size. We further introduce a new metric that jointly evaluates coverage validity and answering efficiency. 

---
# Assessing Pause Thresholds for empirical Translation Process Research 

**Authors**: Devi Sri Bandaru, Michael Carl, Xinyue Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.01410)  

**Abstract**: Text production (and translations) proceeds in the form of stretches of typing, interrupted by keystroke pauses. It is often assumed that fast typing reflects unchallenged/automated translation production while long(er) typing pauses are indicative of translation problems, hurdles or difficulties. Building on a long discussion concerning the determination of pause thresholds that separate automated from presumably reflective translation processes (O'Brien, 2006; Alves and Vale, 2009; Timarova et al., 2011; Dragsted and Carl, 2013; Lacruz et al., 2014; Kumpulainen, 2015; Heilmann and Neumann 2016), this paper compares three recent approaches for computing these pause thresholds, and suggest and evaluate a novel method for computing Production Unit Breaks. 

---
# Friends and Grandmothers in Silico: Localizing Entity Cells in Language Models 

**Authors**: Itay Yona, Dan Barzilay, Michael Karasik, Mor Geva  

**Link**: [PDF](https://arxiv.org/pdf/2604.01404)  

**Abstract**: Language models can answer many entity-centric factual questions, but it remains unclear which internal mechanisms are involved in this process. We study this question across multiple language models. We localize entity-selective MLP neurons using templated prompts about each entity, and then validate them with causal interventions on PopQA-based QA examples. On a curated set of 200 entities drawn from PopQA, localized neurons concentrate in early layers. Negative ablation produces entity-specific amnesia, while controlled injection at a placeholder token improves answer retrieval relative to mean-entity and wrong-cell controls. For many entities, activating a single localized neuron is sufficient to recover entity-consistent predictions once the context is initialized, consistent with compact entity retrieval rather than purely gradual enrichment across depth. Robustness to aliases, acronyms, misspellings, and multilingual forms supports a canonicalization interpretation. The effect is strong but not universal: not every entity admits a reliable single-neuron handle, and coverage is higher for popular entities. Overall, these results identify sparse, causally actionable access points for analyzing and modulating entity-conditioned factual behavior. 

---
# Open-Domain Safety Policy Construction 

**Authors**: Di Wu, Siyue Liu, Zixiang Ji, Ya-Liang Chang, Zhe-Yu Liu, Andrew Pleffer, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2604.01354)  

**Abstract**: Moderation layers are increasingly a core component of many products built on user- or model-generated content. However, drafting and maintaining domain-specific safety policies remains costly. We present Deep Policy Research (DPR), a minimal agentic system that drafts a full content moderation policy based on only human-written seed domain information. DPR uses a single web search tool and lightweight scaffolding to iteratively propose search queries, distill diverse web sources into policy rules, and organize rules into an indexed document. We evaluate DPR on (1) the OpenAI undesired content benchmark across five domains with two compact reader LLMs and (2) an in-house multimodal advertisement moderation benchmark. DPR consistently outperforms definition-only and in-context learning baselines, and in our end-to-end setting it is competitive with expert-written policy sections in several domains. Moreover, under the same seed specification and evaluation protocol, DPR outperforms a general-purpose deep research system, suggesting that a task-specific, structured research loop can be more effective than generic web research for policy drafting. We release our experiment code at this https URL. 

---
# No Attacker Needed: Unintentional Cross-User Contamination in Shared-State LLM Agents 

**Authors**: Tiankai Yang, Jiate Li, Yi Nian, Shen Dong, Ruiyao Xu, Ryan Rossi, Kaize Ding, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.01350)  

**Abstract**: LLM-based agents increasingly operate across repeated sessions, maintaining task states to ensure continuity. In many deployments, a single agent serves multiple users within a team or organization, reusing a shared knowledge layer across user identities. This shared persistence expands the failure surface: information that is locally valid for one user can silently degrade another user's outcome when the agent reapplies it without regard for scope. We refer to this failure mode as unintentional cross-user contamination (UCC). Unlike adversarial memory poisoning, UCC requires no attacker; it arises from benign interactions whose scope-bound artifacts persist and are later misapplied. We formalize UCC through a controlled evaluation protocol, introduce a taxonomy of three contamination types, and evaluate the problem in two shared-state mechanisms. Under raw shared state, benign interactions alone produce contamination rates of 57--71%. A write-time sanitization is effective when shared state is conversational, but leaves substantial residual risk when shared state includes executable artifacts, with contamination often manifesting as silent wrong answers. These results indicate that shared-state agents need artifact-level defenses beyond text-level sanitization to prevent silent cross-user failures. 

---
# Procedural Knowledge at Scale Improves Reasoning 

**Authors**: Di Wu, Devendra Singh Sachan, Wen-tau Yih, Mingda Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.01348)  

**Abstract**: Test-time scaling has emerged as an effective way to improve language models on challenging reasoning tasks. However, most existing methods treat each problem in isolation and do not systematically reuse knowledge from prior reasoning trajectories. In particular, they underutilize procedural knowledge: how to reframe a problem, choose an approach, and verify or backtrack when needed. We introduce Reasoning Memory, a retrieval-augmented generation (RAG) framework for reasoning models that explicitly retrieves and reuses procedural knowledge at scale. Starting from existing corpora of step-by-step reasoning trajectories, we decompose each trajectory into self-contained subquestion-subroutine pairs, yielding a datastore of 32 million compact procedural knowledge entries. At inference time, a lightweight in-thought prompt lets the model verbalize the core subquestion, retrieve relevant subroutines within its reasoning trace, and reason under diverse retrieved subroutines as implicit procedural priors. Across six math, science, and coding benchmarks, Reasoning Memory consistently outperforms RAG with document, trajectory, and template knowledge, as well as a compute-matched test-time scaling baseline. With a higher inference budget, it improves over no retrieval by up to 19.2% and over the strongest compute-matched baseline by 7.9% across task types. Ablation studies show that these gains come from two key factors: the broad procedural coverage of the source trajectories and our decomposition and retrieval design, which together enable effective extraction and reuse of procedural knowledge. 

---
# Preference learning in shades of gray: Interpretable and bias-aware reward modeling for human preferences 

**Authors**: Simona-Vasilica Oprea, Adela Bâra  

**Link**: [PDF](https://arxiv.org/pdf/2604.01312)  

**Abstract**: Learning human preferences in language models remains fundamentally challenging, as reward modeling relies on subtle, subjective comparisons or shades of gray rather than clear-cut labels. This study investigates the limits of current approaches and proposes a feature-augmented framework to better capture the multidimensional nature of human judgment. Using the Anthropic HHRLHF dataset, we evaluate ten diverse large language models LLMs under a standard pairwise preference setting, where baseline performance remains below 0.74 ROC AUC, highlighting the difficulty of the task. To address this, we enrich textual representations with interpretable signals: response length, refusal indicators, toxicity scores and prompt response semantic similarity, enabling models to explicitly capture key aspects of helpfulness, safety and relevance. The proposed hybrid approach yields consistent improvements across all models, achieving up to 0.84 ROC AUC and significantly higher pairwise accuracy, with DeBERTav3Large demonstrating the best performance. Beyond accuracy, we integrate SHAP and LIME to provide fine-grained interpretability, revealing that model decisions depend on contextualized safety and supportive framing rather than isolated keywords. We further analyze bias amplification, showing that while individual features have weak marginal effects, their interactions influence preference learning. 

---
# M2-Verify: A Large-Scale Multidomain Benchmark for Checking Multimodal Claim Consistency 

**Authors**: Abolfazl Ansari, Delvin Ce Zhang, Zhuoyang Zou, Wenpeng Yin, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.01306)  

**Abstract**: Evaluating scientific arguments requires assessing the strict consistency between a claim and its underlying multimodal evidence. However, existing benchmarks lack the scale, domain diversity, and visual complexity needed to evaluate this alignment realistically. To address this gap, we introduce M2-Verify, a large-scale multimodal dataset for checking scientific claim consistency. Sourced from PubMed and arXiv, M2-Verify provides over 469K instances across 16 domains, rigorously validated through expert audits. Extensive baseline experiments show that state-of-the-art models struggle to maintain robust consistency. While top models achieve up to 85.8\% Micro-F1 on low-complexity medical perturbations, performance drops to 61.6\% on high-complexity challenges like anatomical shifts. Furthermore, expert evaluations expose hallucinations when models generate scientific explanations for their alignment decisions. Finally, we demonstrate our dataset's utility and provide comprehensive usage guidelines. 

---
# Scaling Reasoning Tokens via RL and Parallel Thinking: Evidence From Competitive Programming 

**Authors**: Qianfan Zhang, Tianyu Guo, Xuandi Ren, Jiale Chen, Ming Ding, Ran Xin, Xia Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.01302)  

**Abstract**: We study how to scale reasoning token budgets for competitive programming through two complementary approaches: training-time reinforcement learning (RL) and test-time parallel thinking. During RL training, we observe an approximately log-linear relationship between validation accuracy and the average number of generated reasoning tokens over successive checkpoints, and show two ways to shift this training trajectory: verification RL warmup raises the starting point, while randomized clipping produces a steeper trend in the observed regime. As scaling single-generation reasoning during RL quickly becomes expensive under full attention, we introduce a multi-round parallel thinking pipeline that distributes the token budget across threads and rounds of generation, verification, and refinement. We train the model end-to-end on this pipeline to match the training objective to the test-time structure. Starting from Seed-OSS-36B, the full system with 16 threads and 16 rounds per thread matches the underlying RL model's oracle pass@16 at pass@1 using 7.6 million tokens per problem on average, and surpasses GPT-5-high on 456 hard competitive programming problems from AetherCode. 

---
# The Overlooked Repetitive Lengthening Form in Sentiment Analysis 

**Authors**: Lei Wang, Eduard Dragut  

**Link**: [PDF](https://arxiv.org/pdf/2604.01268)  

**Abstract**: Individuals engaging in online communication frequently express personal opinions with informal styles (e.g., memes and emojis). While Language Models (LMs) with informal communications have been widely discussed, a unique and emphatic style, the Repetitive Lengthening Form (RLF), has been overlooked for years. In this paper, we explore answers to two research questions: 1) Is RLF important for sentiment analysis (SA)? 2) Can LMs understand RLF? Inspired by previous linguistic research, we curate \textbf{Lengthening}, the first multi-domain dataset with 850k samples focused on RLF for SA. Moreover, we introduce \textbf{Exp}lainable \textbf{Instruct}ion Tuning (\textbf{ExpInstruct}), a two-stage instruction tuning framework aimed to improve both performance and explainability of LLMs for RLF. We further propose a novel unified approach to quantify LMs' understanding of informal expressions. We show that RLF sentences are expressive expressions and can serve as signatures of document-level sentiment. Additionally, RLF has potential value for online content analysis. Our results show that fine-tuned Pre-trained Language Models (PLMs) can surpass zero-shot GPT-4 in performance but not in explanation for RLF. Finally, we show ExpInstruct can improve the open-sourced LLMs to match zero-shot GPT-4 in performance and explainability for RLF with limited samples. Code and sample data are available at this https URL 

---
# Batched Contextual Reinforcement: A Task-Scaling Law for Efficient Reasoning 

**Authors**: Bangji Yang, Hongbo Ma, Jiajun Fan, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.02322)  

**Abstract**: Large Language Models employing Chain-of-Thought reasoning achieve strong performance but suffer from excessive token consumption that inflates inference costs. Existing efficiency methods such as explicit length penalties, difficulty estimators, or multi-stage curricula either degrade reasoning quality or require complex training pipelines. We introduce Batched Contextual Reinforcement, a minimalist, single-stage training paradigm that unlocks efficient reasoning through a simple structural modification: training the model to solve N problems simultaneously within a shared context window, rewarded purely by per-instance accuracy. This formulation creates an implicit token budget that yields several key findings: (1) We identify a novel task-scaling law: as the number of concurrent problems N increases during inference, per-problem token usage decreases monotonically while accuracy degrades far more gracefully than baselines, establishing N as a controllable throughput dimension. (2) BCR challenges the traditional accuracy-efficiency trade-off by demonstrating a "free lunch" phenomenon at standard single-problem inference. Across both 1.5B and 4B model families, BCR reduces token usage by 15.8% to 62.6% while consistently maintaining or improving accuracy across five major mathematical benchmarks. (3) Qualitative analyses reveal emergent self-regulated efficiency, where models autonomously eliminate redundant metacognitive loops without explicit length supervision. (4) Crucially, we empirically demonstrate that implicit budget constraints successfully circumvent the adversarial gradients and catastrophic optimization collapse inherent to explicit length penalties, offering a highly stable, constraint-based alternative for length control. These results prove BCR practical, showing simple structural incentives unlock latent high-density reasoning in LLMs. 

---
# go-$m$HC: Direct Parameterization of Manifold-Constrained Hyper-Connections via Generalized Orthostochastic Matrices 

**Authors**: Torque Dandachi, Sophia Diggs-Galligan  

**Link**: [PDF](https://arxiv.org/pdf/2604.02309)  

**Abstract**: Doubly stochastic matrices enable learned mixing across residual streams, but parameterizing the set of doubly stochastic matrices (the Birkhoff polytope) exactly and efficiently remains an open challenge. Existing exact methods scale factorially with the number of streams ($d$), while Kronecker-factorized approaches are efficient but expressivity-limited. We introduce a novel exact parameterization grounded in the theory of generalized orthostochastic matrices, which scales as $\mathcal{O}(d^3)$ and exposes a single hyperparameter $s$ which continuously interpolates between a computationally efficient boundary and the fully expressive Birkhoff polytope. Building on Manifold-Constrained Hyper-Connections ($m$HC), a framework for learned dynamic layer connectivity, we instantiate this parameterization in go-$m$HC. Our method composes naturally with Kronecker-factorized methods, substantially recovering expressivity at similar FLOP costs. Spectral analysis indicates that go-$m$HC fills the Birkhoff polytope far more completely than Kronecker-factorized baselines. On synthetic stream-mixing tasks, go-$m$HC achieves the minimum theoretical loss while converging up to $10\times$ faster. We validate our approach in a 30M parameter GPT-style language model. The expressivity, efficiency, and exactness of go-$m$HC offer a practical avenue for scaling $d$ as a new dimension of model capacity. 

---
# De Jure: Iterative LLM Self-Refinement for Structured Extraction of Regulatory Rules 

**Authors**: Keerat Guliani, Deepkamal Gill, David Landsman, Nima Eshraghi, Krishna Kumar, Lovedeep Gondara  

**Link**: [PDF](https://arxiv.org/pdf/2604.02276)  

**Abstract**: Regulatory documents encode legally binding obligations that LLM-based systems must respect. Yet converting dense, hierarchically structured legal text into machine-readable rules remains a costly, expert-intensive process. We present De Jure, a fully automated, domain-agnostic pipeline for extracting structured regulatory rules from raw documents, requiring no human annotation, domain-specific prompting, or annotated gold data. De Jure operates through four sequential stages: normalization of source documents into structured Markdown; LLM-driven semantic decomposition into structured rule units; multi-criteria LLM-as-a-judge evaluation across 19 dimensions spanning metadata, definitions, and rule semantics; and iterative repair of low-scoring extractions within a bounded regeneration budget, where upstream components are repaired before rule units are evaluated. We evaluate De Jure across four models on three regulatory corpora spanning finance, healthcare, and AI governance. On the finance domain, De Jure yields consistent and monotonic improvement in extraction quality, reaching peak performance within three judge-guided iterations. De Jure generalizes effectively to healthcare and AI governance, maintaining high performance across both open- and closed-source models. In a downstream compliance question-answering evaluation via RAG, responses grounded in De Jure extracted rules are preferred over prior work in 73.8% of cases at single-rule retrieval depth, rising to 84.0% under broader retrieval, confirming that extraction fidelity translates directly into downstream utility. These results demonstrate that explicit, interpretable evaluation criteria can substitute for human annotation in complex regulatory domains, offering a scalable and auditable path toward regulation-grounded LLM alignment. 

---
# VISTA: Visualization of Token Attribution via Efficient Analysis 

**Authors**: Syed Ahmed, Bharathi Vokkaliga Ganesh, Jagadish Babu P, Karthick Selvaraj, Praneeth Talluri, Sanket Hingne, Anubhav Kumar, Anushka Yadav, Pratham Kumar Verma, Kiranmayee Janardhan, Mandanna A N  

**Link**: [PDF](https://arxiv.org/pdf/2604.02217)  

**Abstract**: Understanding how Large Language Models (LLMs) process information from prompts remains a significant challenge. To shed light on this "black box," attention visualization techniques have been developed to capture neuron-level perceptions and interpret how models focus on different parts of input data. However, many existing techniques are tailored to specific model architectures, particularly within the Transformer family, and often require backpropagation, resulting in nearly double the GPU memory usage and increased computational cost. A lightweight, model-agnostic approach for attention visualization remains lacking. In this paper, we introduce a model-agnostic token importance visualization technique to better understand how generative AI systems perceive and prioritize information from input text, without incurring additional computational cost. Our method leverages perturbation-based strategies combined with a three-matrix analytical framework to generate relevance maps that illustrate token-level contributions to model predictions. The framework comprises: (1) the Angular Deviation Matrix, which captures shifts in semantic direction; (2) the Magnitude Deviation Matrix, which measures changes in semantic intensity; and (3) the Dimensional Importance Matrix, which evaluates contributions across individual vector dimensions. By systematically removing each token and measuring the resulting impact across these three complementary dimensions, we derive a composite importance score that provides a nuanced and mathematically grounded measure of token significance. To support reproducibility and foster wider adoption, we provide open-source implementations of all proposed and utilized explainability techniques, with code and resources publicly available at this https URL 

---
# Blinded Radiologist and LLM-Based Evaluation of LLM-Generated Japanese Translations of Chest CT Reports: Comparative Study 

**Authors**: Yosuke Yamagishi, Atsushi Takamatsu, Yasunori Hamaguchi, Tomohiro Kikuchi, Shouhei Hanaoka, Takeharu Yoshikawa, Osamu Abe  

**Link**: [PDF](https://arxiv.org/pdf/2604.02207)  

**Abstract**: Background: Accurate translation of radiology reports is important for multilingual research, clinical communication, and radiology education, but the validity of LLM-based evaluation remains unclear. Objective: To evaluate the educational suitability of LLM-generated Japanese translations of chest CT reports and compare radiologist assessments with LLM-as-a-judge evaluations. Methods: We analyzed 150 chest CT reports from the CT-RATE-JPN validation set. For each English report, a human-edited Japanese translation was compared with an LLM-generated translation by DeepSeek-V3.2. A board-certified radiologist and a radiology resident independently performed blinded pairwise evaluations across 4 criteria: terminology accuracy, readability, overall quality, and radiologist-style authenticity. In parallel, 3 LLM judges (DeepSeek-V3.2, Mistral Large 3, and GPT-5) evaluated the same pairs. Agreement was assessed using QWK and percentage agreement. Results: Agreement between radiologists and LLM judges was near zero (QWK=-0.04 to 0.15). Agreement between the 2 radiologists was also poor (QWK=0.01 to 0.06). Radiologist 1 rated terminology as equivalent in 59% of cases and favored the LLM translation for readability (51%) and overall quality (51%). Radiologist 2 rated readability as equivalent in 75% of cases and favored the human-edited translation for overall quality (40% vs 21%). All 3 LLM judges strongly favored the LLM translation across all criteria (70%-99%) and rated it as more radiologist-like in >93% of cases. Conclusions: LLM-generated translations were often judged natural and fluent, but the 2 radiologists differed substantially. LLM-as-a-judge showed strong preference for LLM output and negligible agreement with radiologists. For educational use of translated radiology reports, automated LLM-based evaluation alone is insufficient; expert radiologist review remains important. 

---
# MTI: A Behavior-Based Temperament Profiling System for AI Agents 

**Authors**: Jihoon Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2604.02145)  

**Abstract**: AI models of equivalent capability can exhibit fundamentally different behavioral patterns, yet no standardized instrument exists to measure these dispositional differences. Existing approaches either borrow human personality dimensions and rely on self-report (which diverges from actual behavior in LLMs) or treat behavioral variation as a defect rather than a trait.
We introduce the Model Temperament Index (MTI), a behavior-based profiling system that measures AI agent temperament across four axes: Reactivity (environmental sensitivity), Compliance (instruction-behavior alignment), Sociality (relational resource allocation), and Resilience (stress resistance). Grounded in the Four Shell Model from Model Medicine, MTI measures what agents do, not what they say about themselves, using structured examination protocols with a two-stage design that separates capability from disposition.
We profile 10 small language models (1.7B-9B parameters, 6 organizations, 3 training paradigms) and report five principal findings: (1) the four axes are largely independent among instruction-tuned models (all |r| < 0.42); (2) within-axis facet dissociations are empirically confirmed -- Compliance decomposes into fully independent formal and stance facets (r = 0.002), while Resilience decomposes into inversely related cognitive and adversarial facets; (3) a Compliance-Resilience paradox reveals that opinion-yielding and fact-vulnerability operate through independent channels; (4) RLHF reshapes temperament not only by shifting axis scores but by creating within-axis facet differentiation absent in the unaligned base model; and (5) temperament is independent of model size (1.7B-9B), confirming that MTI measures disposition rather than capability. 

---
# LLM-as-a-Judge for Time Series Explanations 

**Authors**: Preetham Sivalingam, Murari Mandal, Saurabh Deshpande, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.02118)  

**Abstract**: Evaluating factual correctness of LLM generated natural language explanations grounded in time series data remains an open challenge. Although modern models generate textual interpretations of numerical signals, existing evaluation methods are limited: reference based similarity metrics and consistency checking models require ground truth explanations, while traditional time series methods operate purely on numerical values and cannot assess free form textual reasoning. Thus, no general purpose method exists to directly verify whether an explanation is faithful to underlying time series data without predefined references or task specific rules. We study large language models as both generators and evaluators of time series explanations in a reference free setting, where given a time series, question, and candidate explanation, the evaluator assigns a ternary correctness label based on pattern identification, numeric accuracy, and answer faithfulness, enabling principled scoring and comparison. To support this, we construct a synthetic benchmark of 350 time series cases across seven query types, each paired with correct, partially correct, and incorrect explanations. We evaluate models across four tasks: explanation generation, relative ranking, independent scoring, and multi anomaly detection. Results show a clear asymmetry: generation is highly pattern dependent and exhibits systematic failures on certain query types, with accuracies ranging from 0.00 to 0.12 for Seasonal Drop and Volatility Shift, to 0.94 to 0.96 for Structural Break, while evaluation is more stable, with models correctly ranking and scoring explanations even when their own outputs are incorrect. These findings demonstrate feasibility of data grounded LLM based evaluation for time series explanations and highlight their potential as reliable evaluators of data grounded reasoning in the time series domain. 

---
# Ouroboros: Dynamic Weight Generation for Recursive Transformers via Input-Conditioned LoRA Modulation 

**Authors**: Jaber Jaber, Osama Jaber  

**Link**: [PDF](https://arxiv.org/pdf/2604.02051)  

**Abstract**: Recursive transformers reuse a shared weight block across multiple depth steps, trading parameters for compute. A core limitation: every step applies the same transformation, preventing the model from composing distinct operations across depth. We present Ouroboros, a system that attaches a compact Controller hypernetwork to a recursive transformer block. The Controller observes the current hidden state, produces a per-step diagonal modulation vector, and applies it to frozen SVD-initialized LoRA bases, making each recurrence step input-dependent. We combine this with gated recurrence (bias-initialized to 88% retention) and per-step LayerNorm for stable deep iteration. On Qwen2.5-3B split into a Prelude/Recurrent/Coda architecture (17 of 36 layers retained), Ouroboros reduces training loss by 43.4% over the unmodified 17-layer baseline, recovering 51.3% of the performance gap caused by layer removal. The full system adds only 9.2M trainable parameters (Controller, gate, and per-step norms) yet outperforms equivalently-sized static per-step LoRA by 1.44 loss points at depth 1 and remains ahead across all tested depths (1, 4, 8, 16) and ranks (8, 32, 64). We also find that gated recurrence is essential: without it, recursive layer application makes the model strictly worse. These gains are measured on the training distribution; on held-out text, the Controller does not yet improve over the baseline, a limitation we attribute to frozen downstream layers and discuss in detail. Code: this https URL 

---
# RuleForge: Automated Generation and Validation for Web Vulnerability Detection at Scale 

**Authors**: Ayush Garg, Sophia Hager, Jacob Montiel, Aditya Tiwari, Michael Gentile, Zach Reavis, David Magnotti, Wayne Fullen  

**Link**: [PDF](https://arxiv.org/pdf/2604.01977)  

**Abstract**: Security teams face a challenge: the volume of newly disclosed Common Vulnerabilities and Exposures (CVEs) far exceeds the capacity to manually develop detection mechanisms. In 2025, the National Vulnerability Database published over 48,000 new vulnerabilities, motivating the need for automation. We present RuleForge, an AWS internal system that automatically generates detection rules--JSON-based patterns that identify malicious HTTP requests exploiting specific vulnerabilities--from structured Nuclei templates describing CVE details. Nuclei templates provide standardized, YAML-based vulnerability descriptions that serve as the structured input for our rule generation process.
This paper focuses on RuleForge's architecture and operational deployment for CVE-related threat detection, with particular emphasis on our novel LLM-as-a-judge (Large Language Model as judge) confidence validation system and systematic feedback integration mechanism. This validation approach evaluates candidate rules across two dimensions--sensitivity (avoiding false negatives) and specificity (avoiding false positives)--achieving AUROC of 0.75 and reducing false positives by 67% compared to synthetic-test-only validation in production. Our 5x5 generation strategy (five parallel candidates with up to five refinement attempts each) combined with continuous feedback loops enables systematic quality improvement. We also present extensions enabling rule generation from unstructured data sources and demonstrate a proof-of-concept agentic workflow for multi-event-type detection. Our lessons learned highlight critical considerations for applying LLMs to cybersecurity tasks, including overconfidence mitigation and the importance of domain expertise in both prompt design and quality review of generated rules through human-in-the-loop validation. 

---
# Do We Need Bigger Models for Science? Task-Aware Retrieval with Small Language Models 

**Authors**: Florian Kelber, Matthias Jobst, Yuni Susanti, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2604.01965)  

**Abstract**: Scientific knowledge discovery increasingly relies on large language models, yet many existing scholarly assistants depend on proprietary systems with tens or hundreds of billions of parameters. Such reliance limits reproducibility and accessibility for the research community. In this work, we ask a simple question: do we need bigger models for scientific applications? Specifically, we investigate to what extent carefully designed retrieval pipelines can compensate for reduced model scale in scientific applications. We design a lightweight retrieval-augmented framework that performs task-aware routing to select specialized retrieval strategies based on the input query. The system further integrates evidence from full-text scientific papers and structured scholarly metadata, and employs compact instruction-tuned language models to generate responses with citations. We evaluate the framework across several scholarly tasks, focusing on scholarly question answering (QA), including single- and multi-document scenarios, as well as biomedical QA under domain shift and scientific text compression. Our findings demonstrate that retrieval and model scale are complementary rather than interchangeable. While retrieval design can partially compensate for smaller models, model capacity remains important for complex reasoning tasks. This work highlights retrieval and task-aware design as key factors for building practical and reproducible scholarly assistants. 

---
# HieraVid: Hierarchical Token Pruning for Fast Video Large Language Models 

**Authors**: Yansong Guo, Chaoyang Zhu, Jiayi Ji, Jianghang Lin, Liujuan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2604.01881)  

**Abstract**: Video Large Language Models (VideoLLMs) have demonstrated impressive capabilities in video understanding, yet the massive number of input video tokens incurs a significant computational burden for deployment. Existing methods mainly prune video tokens at input level while neglecting the inherent information structure embedded in videos and large language models (LLMs). To address this, we propose HieraVid, a hierarchical pruning framework that progressively and dynamically reduces visual redundancy. Based on two observations that videos possess the segment-frame structure and LLMs internally propagate multi-modal information unidirectionally, we decompose pruning into three levels: 1) segment-level, where video tokens are first temporally segmented and spatially merged; 2) frame-level, where similar frames within the same segment are jointly pruned to preserve diversity; 3) layer-level, redundancy gradually shrinks as LLM layer increases w/o compromising performance. We conduct extensive experiments on four widely used video understanding benchmarks to comprehensively evaluate the effectiveness of HieraVid. Remarkably, with only 30% of tokens retained, HieraVid achieves new state-of-the-art performance, while maintaining over 98% and 99% of the performance of LLaVA-Video-7B and LLaVA-OneVision-7B, respectively. 

---
# Language-Pretraining-Induced Bias: A Strong Foundation for General Vision Tasks 

**Authors**: Yaxin Luo, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.01833)  

**Abstract**: The ratio of outlier parameters in language pre-training models and vision pre-training models differs significantly, making cross-modality (language and vision) inherently more challenging than cross-domain adaptation. As a result, many prior studies have focused on cross-domain transfer rather than attempting to bridge language and vision modalities, assuming that language pre-trained models are unsuitable for downstream visual tasks due to disparate parameter spaces. Contrary to this assumption, we show that adding a bridge training stage as a modality adaptation learner can effectively align Large Language Model (LLM) parameters with vision tasks. Specifically, we propose a simple yet powerful solution random label bridge training that requires no manual labeling and helps LLM parameters adapt to vision foundation tasks. Moreover, our findings reveal that partial bridge training is often advantageous, as certain layers in LLMs exhibit strong foundational properties that remain beneficial even without fine-tuning for visual tasks. This surprising discovery opens up new avenues for leveraging language pre-trained parameters directly within vision models and highlights the potential of partial bridge training as a practical pathway to cross-modality adaptation. 

---
# FourierMoE: Fourier Mixture-of-Experts Adaptation of Large Language Models 

**Authors**: Juyong Jiang, Fan Wang, Hong Qi, Sunghun Kim, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.01762)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) has emerged as a crucial paradigm for adapting large language models (LLMs) under constrained computational budgets. However, standard PEFT methods often struggle in multi-task fine-tuning settings, where diverse optimization objectives induce task interference and limited parameter budgets lead to representational deficiency. While recent approaches incorporate mixture-of-experts (MoE) to alleviate these issues, they predominantly operate in the spatial domain, which may introduce structural redundancy and parameter overhead. To overcome these limitations, we reformulate adaptation in the spectral domain. Our spectral analysis reveals that different tasks exhibit distinct frequency energy distributions, and that LLM layers display heterogeneous frequency sensitivities. Motivated by these insights, we propose FourierMoE, which integrates the MoE architecture with the inverse discrete Fourier transform (IDFT) for frequency-aware adaptation. Specifically, FourierMoE employs a frequency-adaptive router to dispatch tokens to experts specialized in distinct frequency bands. Each expert learns a set of conjugate-symmetric complex coefficients, preserving complete phase and amplitude information while theoretically guaranteeing lossless IDFT reconstruction into real-valued spatial weights. Extensive evaluations across 28 benchmarks, multiple model architectures, and scales demonstrate that FourierMoE consistently outperforms competitive baselines in both single-task and multi-task settings while using significantly fewer trainable parameters. These results highlight the promise of spectral-domain expert adaptation as an effective and parameter-efficient paradigm for LLM fine-tuning. 

---
# From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents 

**Authors**: Meftun Akarsu, Recep Kaan Karaman, Christopher Mierbach  

**Link**: [PDF](https://arxiv.org/pdf/2604.01733)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems critically depend on retrieval quality, yet no systematic comparison of modern retrieval methods exists for heterogeneous documents containing both text and tabular data. We benchmark ten retrieval strategies spanning sparse, dense, hybrid fusion, cross-encoder reranking, query expansion, index augmentation, and adaptive retrieval on a challenging financial QA benchmark of 23,088 queries over 7,318 documents with mixed text-and-table content. We evaluate retrieval quality via Recall@k, MRR, and nDCG, and end-to-end generation quality via Number Match, with paired bootstrap significance testing. Our results show that (1) a two-stage pipeline combining hybrid retrieval with neural reranking achieves Recall@5 of 0.816 and MRR@3 of 0.605, outperforming all single-stage methods by a large margin; (2) BM25 outperforms state-of-the-art dense retrieval on financial documents, challenging the common assumption that semantic search universally dominates; and (3) query expansion methods (HyDE, multi-query) and adaptive retrieval provide limited benefit for precise numerical queries, while contextual retrieval yields consistent gains. We provide ablation studies on fusion methods and reranker depth, actionable cost-accuracy recommendations, and release our full benchmark code. 

---
# MiCA Learns More Knowledge Than LoRA and Full Fine-Tuning 

**Authors**: Sten Rüdiger, Sebastian Raschka  

**Link**: [PDF](https://arxiv.org/pdf/2604.01694)  

**Abstract**: Minor Component Adaptation (MiCA) is a novel parameter-efficient fine-tuning method for large language models that focuses on adapting underutilized subspaces of model representations. Unlike conventional methods such as Low-Rank Adaptation (LoRA), which target dominant subspaces, MiCA leverages Singular Value Decomposition to identify subspaces related to minor singular vectors associated with the least significant singular values and constrains the update of parameters during fine-tuning to those directions. This strategy leads to up to 5.9x improvement in knowledge acquisition under optimized training hyperparameters and a minimal parameter footprint of 6-60% compared to LoRA. These results suggest that constraining adaptation to minor singular directions provides a more efficient and stable mechanism for integrating new knowledge into pre-trained language models. 

---
# Coupled Query-Key Dynamics for Attention 

**Authors**: Barak Gahtan, Alex M. Bronstein  

**Link**: [PDF](https://arxiv.org/pdf/2604.01683)  

**Abstract**: Standard scaled dot-product attention computes scores from static, independent projections of the input. We show that evolving queries and keys \emph{jointly} through shared learned dynamics before scoring - which we call \textbf{coupled QK dynamics} - improves language modeling perplexity and training stability. On WikiText-103 at 60M parameters, coupled dynamics achieves 22.55--22.62 perplexity vs.\ 24.22 for standard attention ($-$6.6--6.9\%), with only 0.11\% additional parameters (shared across both instantiations). A structural ablation isolates coupling as the active ingredient: a symplectic (Hamiltonian) and a non-symplectic (Euler) integrator perform identically when both couple Q and K, while an uncoupled MLP baseline of matched capacity reaches only 23.81 with 8$\times$ higher seed variance. The integration step count (1--7) is similarly irrelevant - a single coupled step suffices. A compute-matched comparison reveals that coupling is a \emph{sample-efficiency} mechanism: standard attention trained for 2.4$\times$ longer (matching wall-clock) reaches the same perplexity, but requires 2.4$\times$ more tokens. The advantage scales to 150M ($-$6.7\%) but narrows at 350M ($-$1.0\%), where Differential Attention (18.93) overtakes coupled dynamics (19.35). The benefit is corpus-dependent: coupling helps on domain-coherent text (WikiText-103 $-$6.6\%, PubMed $-$4.5\%) but degrades on heterogeneous web text ($+$10.3\%) and shows no benefit on GLUE. We characterize when coupling helps and when it does not, providing practical guidelines. 

---
# ThinknCheck: Grounded Claim Verification with Compact, Reasoning-Driven, and Interpretable Models 

**Authors**: Delip Rao, Feijiang Han, Chris Callison-Burch  

**Link**: [PDF](https://arxiv.org/pdf/2604.01652)  

**Abstract**: We present ThinknCheck, a 1B-parameter verifier for grounded claim verification that first produces a short, structured rationale and then a binary verdict. We construct LLMAggreFact-Think, a 24.1k reasoning-augmented training set derived from LLMAggreFact, and fine-tune a 4-bit Gemma3 model to follow this format. On LLMAggreFact, ThinknCheck attains 78.1 balanced accuracy (BAcc), surpassing MiniCheck-7B (77.4) with 7x fewer parameters; removing the reasoning step reduces BAcc to 57.5. On SciFact, ThinknCheck reaches 64.7 BAcc, a +14.7 absolute gain over MiniCheck-7B. By contrast, zero-shot chain-of-thought on the base Gemma3-1B harms accuracy relative to direct answers, and preference optimization with a simple format+accuracy reward underperforms supervised reasoning. To probe the latter, we introduce GSMClaims and a domain-specialized variant, ThinknCheck-Science, which improves across benchmarks, including 61.0\% accuracy on GSMClaims. Overall, explicit, supervised reasoning enables compact verifiers that are competitive while remaining resource-efficient and interpretable. 

---
# CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning 

**Authors**: Junyoung Sung, Seungwoo Lyu, Minjun Kim, Sumin An, Arsha Nagrani, Paul Hongsuck Seo  

**Link**: [PDF](https://arxiv.org/pdf/2604.01634)  

**Abstract**: Real-world reasoning often requires combining information across modalities, connecting textual context with visual cues in a multi-hop process. Yet, most multimodal benchmarks fail to capture this ability: they typically rely on single images or set of images, where answers can be inferred from a single modality alone. This limitation is mirrored in the training data, where interleaved image-text content rarely enforces complementary, multi-hop reasoning. As a result, Vision-Language Models (VLMs) frequently hallucinate and produce reasoning traces poorly grounded in visual evidence. To address this gap, we introduce CRIT, a new dataset and benchmark built with a graph-based automatic pipeline for generating complex cross-modal reasoning tasks. CRIT consists of diverse domains ranging from natural images, videos, and text-rich sources, and includes a manually verified test set for reliable evaluation. Experiments on this benchmark reveal that even state-of-the-art models struggle on such reasoning tasks. Models trained on CRIT show significant gains in cross-modal multi-hop reasoning, including strong improvements on SPIQA and other standard multimodal benchmarks. 

---
# OSCAR: Orchestrated Self-verification and Cross-path Refinement 

**Authors**: Yash Shah, Abhijit Chakraborty, Naresh Kumar Devulapally, Vishnu Lokhande, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.01624)  

**Abstract**: Diffusion language models (DLMs) expose their denoising trajectories, offering a natural handle for inference-time control; accordingly, an ideal hallucination mitigation framework should intervene during generation using this model-native signal rather than relying on an externally trained hallucination classifier. Toward this, we formulate commitment uncertainty localization: given a denoising trajectory, identify token positions whose cross-chain entropy exceeds an unsupervised threshold before factually unreliable commitments propagate into self-consistent but incorrect outputs. We introduce a suite of trajectory-level assessments, including a cross-chain divergence-at-hallucination (CDH) metric, for principled comparison of localization methods. We also introduce OSCAR, a training-free inference-time framework operationalizing this formulation. OSCAR runs N parallel denoising chains with randomized reveal orders, computes cross-chain Shannon entropy to detect high-uncertainty positions, and then performs targeted remasking conditioned on retrieved evidence. Ablations confirm that localization and correction contribute complementary gains, robust across N in {4, 8, 16}. On TriviaQA, HotpotQA, RAGTruth, and CommonsenseQA using LLaDA-8B and Dream-7B, OSCAR enhances generation quality by significantly reducing hallucinated content and improving factual accuracy through uncertainty-guided remasking, which also facilitates more effective integration of retrieved evidence. Its native entropy-based uncertainty signal surpasses that of specialized trained detectors, highlighting an inherent capacity of diffusion language models to identify factual uncertainty that is not present in the sequential token commitment structure of autoregressive models. We are releasing the codebase1 to support future research on localization and uncertainty-aware generation in DLMs. 

---
# Expert-Choice Routing Enables Adaptive Computation in Diffusion Language Models 

**Authors**: Shuibai Zhang, Caspian Zhuang, Chihan Cui, Zhihan Yang, Fred Zhangzhi Peng, Yanxin Zhang, Haoyue Bai, Zack Jia, Yang Zhou, Guanhua Chen, Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.01622)  

**Abstract**: Diffusion language models (DLMs) enable parallel, non-autoregressive text generation, yet existing DLM mixture-of-experts (MoE) models inherit token-choice (TC) routing from autoregressive systems, leading to load imbalance and rigid computation allocation. We show that expert-choice (EC) routing is a better fit for DLMs: it provides deterministic load balancing by design, yielding higher throughput and faster convergence than TC. Building on the property that EC capacity is externally controllable, we introduce timestep-dependent expert capacity, which varies expert allocation according to the denoising step. We find that allocating more capacity to low-mask-ratio steps consistently achieves the best performance under matched FLOPs, and provide a mechanistic explanation: tokens in low-mask-ratio contexts exhibit an order-of-magnitude higher learning efficiency, so concentrating compute on these steps yields the largest marginal return. Finally, we show that existing pretrained TC DLMs can be retrofitted to EC by replacing only the router, achieving faster convergence and improved accuracy across diverse downstream tasks. Together, these results establish EC routing as a superior paradigm for DLM MoE models and demonstrate that computation in DLMs can be treated as an adaptive policy rather than a fixed architectural constant. Code is available at this https URL. 

---
# Acoustic and perceptual differences between standard and accented Chinese speech and their voice clones 

**Authors**: Tianle Yang, Chengzhe Sun, Phil Rose, Siwei Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2604.01562)  

**Abstract**: Voice cloning is often evaluated in terms of overall quality, but less is known about accent preservation and its perceptual consequences. We compare standard and heavily accented Mandarin speech and their voice clones using a combined computational and perceptual design. Embedding-based analyses show no reliable accented-standard difference in original-clone distances across systems. In the perception study, clones are rated as more similar to their originals for standard than for accented speakers, and intelligibility increases from original to clone, with a larger gain for accented speech. These results show that accent variation can shape perceived identity match and intelligibility in voice cloning even when it is not reflected in an off-the-shelf speaker-embedding distance, and they motivate evaluating speaker identity preservation and accent preservation as separable dimensions. 

---
# From SWE-ZERO to SWE-HERO: Execution-free to Execution-based Fine-tuning for Software Engineering Agents 

**Authors**: Nikolai Ludwig, Wasi Uddin Ahmad, Somshubra Majumdar, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2604.01496)  

**Abstract**: We introduce SWE-ZERO to SWE-HERO, a two-stage SFT recipe that achieves state-of-the-art results on SWE-bench by distilling open-weight frontier LLMs. Our pipeline replaces resource-heavy dependencies with an evolutionary refinement strategy: (1) SWE-ZERO utilizes large-scale, execution-free trajectories to master code semantics and repository-level reasoning, and (2) SWE-HERO applies targeted, execution-backed refinement to transition these semantic intuitions into rigorous engineering workflows. Our empirical results set a new benchmark for open-source models of comparable size. We release a dataset of 300k SWE-ZERO and 13k SWE-HERO trajectories distilled from Qwen3-Coder-480B, alongside a suite of agents based on the Qwen2.5-Coder series. Notably, SWE-HERO-32B achieves a 62.2% resolution rate on SWE-bench Verified. Furthermore, despite being trained exclusively on Python, our agents demonstrate robust zero-shot transferability on SWE-bench Multilingual, reaching 44.1% and confirming the paradigm's generalizability across diverse languages. 

---
# When Reward Hacking Rebounds: Understanding and Mitigating It with Representation-Level Signals 

**Authors**: Rui Wu, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.01476)  

**Abstract**: Reinforcement learning for LLMs is vulnerable to reward hacking, where models exploit shortcuts to maximize reward without solving the intended task. We systematically study this phenomenon in coding tasks using an environment-manipulation setting, where models can rewrite evaluator code to trivially pass tests without solving the task, as a controlled testbed. Across both studied models, we identify a reproducible three-phase rebound pattern: models first attempt to rewrite the evaluator but fail, as their rewrites embed test cases their own solutions cannot pass. They then temporarily retreat to legitimate solving. When legitimate reward remains scarce, they rebound into successful hacking with qualitatively different strategies. Using representation engineering, we extract concept directions for shortcut, deception, and evaluation awareness from domain-general contrastive pairs and find that the shortcut direction tracks hacking behavior most closely, making it an effective representational proxy for detection. Motivated by this finding, we propose Advantage Modification, which integrates shortcut concept scores into GRPO advantage computation to penalize hacking rollouts before policy updates. Because the penalty is internalized into the training signal rather than applied only at inference time, Advantage Modification provides more robust suppression of hacking compared with generation-time activation steering. 

---
# ReFormeR: Learning and Applying Explicit Query Reformulation Patterns 

**Authors**: Amin Bigdeli, Mert Incesu, Negar Arabzadeh, Charles L. A. Clarke, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2604.01417)  

**Abstract**: We present ReFormeR, a pattern-guided approach for query reformulation. Instead of prompting a language model to generate reformulations of a query directly, ReFormeR first elicits short reformulation patterns from pairs of initial queries and empirically stronger reformulations, consolidates them into a compact library of transferable reformulation patterns, and then selects an appropriate reformulation pattern for a new query given its retrieval context. The selected pattern constrains query reformulation to controlled operations such as sense disambiguation, vocabulary grounding, or discriminative facet addition, to name a few. As such, our proposed approach makes the reformulation policy explicit through these reformulation patterns, guiding the LLM towards targeted and effective query reformulations. Our extensive experiments on TREC DL 2019, DL 2020, and DL Hard show consistent improvements over classical feedback methods and recent LLM-based query reformulation and expansion approaches. 

---
# Test-Time Scaling Makes Overtraining Compute-Optimal 

**Authors**: Nicholas Roberts, Sungjun Cho, Zhiqi Gao, Tzu-Heng Huang, Albert Wu, Gabriel Orlanski, Avi Trost, Kelly Buchanan, Aws Albarghouthi, Frederic Sala  

**Link**: [PDF](https://arxiv.org/pdf/2604.01411)  

**Abstract**: Modern LLMs scale at test-time, e.g. via repeated sampling, where inference cost grows with model size and the number of samples. This creates a trade-off that pretraining scaling laws, such as Chinchilla, do not address. We present Train-to-Test ($T^2$) scaling laws that jointly optimize model size, training tokens, and number of inference samples under fixed end-to-end budgets. $T^2$ modernizes pretraining scaling laws with pass@$k$ modeling used for test-time scaling, then jointly optimizes pretraining and test-time decisions. Forecasts from $T^2$ are robust over distinct modeling approaches: measuring joint scaling effect on the task loss and modeling impact on task accuracy. Across eight downstream tasks, we find that when accounting for inference cost, optimal pretraining decisions shift radically into the overtraining regime, well-outside of the range of standard pretraining scaling suites. We validate our results by pretraining heavily overtrained models in the optimal region that $T^2$ scaling forecasts, confirming their substantially stronger performance compared to pretraining scaling alone. Finally, as frontier LLMs are post-trained, we show that our findings survive the post-training stage, making $T^2$ scaling meaningful in modern deployments. 

---
# Look Twice: Training-Free Evidence Highlighting in Multimodal Large Language Models 

**Authors**: Marco Morini, Sara Sarto, Marcella Cornia, Lorenzo Baraldi  

**Link**: [PDF](https://arxiv.org/pdf/2604.01280)  

**Abstract**: Answering questions about images often requires combining visual understanding with external knowledge. Multimodal Large Language Models (MLLMs) provide a natural framework for this setting, but they often struggle to identify the most relevant visual and textual evidence when answering knowledge-intensive queries. In such scenarios, models must integrate visual cues with retrieved textual evidence that is often noisy or only partially relevant, while also localizing fine-grained visual information in the image. In this work, we introduce Look Twice (LoT), a training-free inference-time framework that improves how pretrained MLLMs utilize multimodal evidence. Specifically, we exploit the model attention patterns to estimate which visual regions and retrieved textual elements are relevant to a query, and then generate the answer conditioned on this highlighted evidence. The selected cues are highlighted through lightweight prompt-level markers that encourage the model to re-attend to the relevant evidence during generation. Experiments across multiple knowledge-based VQA benchmarks show consistent improvements over zero-shot MLLMs. Additional evaluations on vision-centric and hallucination-oriented benchmarks further demonstrate that visual evidence highlighting alone improves model performance in settings without textual context, all without additional training or architectural modifications. Source code will be publicly released. 

---
