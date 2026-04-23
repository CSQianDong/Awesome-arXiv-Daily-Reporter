# Coverage, Not Averages: Semantic Stratification for Trustworthy Retrieval Evaluation 

**Authors**: Andrew Klearman, Radu Revutchi, Rohin Garg, Rishav Chakravarti, Samuel Marc Denton, Yuan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2604.20763)  

**Abstract**: Retrieval quality is the primary bottleneck for accuracy and robustness in retrieval-augmented generation (RAG). Current evaluation relies on heuristically constructed query sets, which introduce a hidden intrinsic bias. We formalize retrieval evaluation as a statistical estimation problem, showing that metric reliability is fundamentally limited by the evaluation-set construction. We further introduce \emph{semantic stratification}, which grounds evaluation in corpus structure by organizing documents into an interpretable global space of entity-based clusters and systematically generating queries for missing strata. This yields (1) formal semantic coverage guarantees across retrieval regimes and (2) interpretable visibility into retrieval failure modes.
Experiments across multiple benchmarks and retrieval methods validate our framework. The results expose systematic coverage gaps, identify structural signals that explain variance in retrieval performance, and show that stratified evaluation yields more stable and transparent assessments while supporting more trustworthy decision-making than aggregate metrics. 

---
# Self-Aware Vector Embeddings for Retrieval-Augmented Generation: A Neuroscience-Inspired Framework for Temporal, Confidence-Weighted, and Relational Knowledge 

**Authors**: Naizhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.20598)  

**Abstract**: Modern retrieval-augmented generation (RAG) systems treat vector embeddings as static, context-free artifacts: an embedding has no notion of when it was created, how trustworthy its source is, or which other embeddings depend on it. This flattening of knowledge has a measurable cost: recent work on VersionRAG reports that conventional RAG achieves only 58% accuracy on versioned technical queries, because retrieval returns semantically similar but temporally invalid content. We propose SmartVector, a framework that augments dense embeddings with three explicit properties -- temporal awareness, confidence decay, and relational awareness -- and a five-stage lifecycle modeled on hippocampal-neocortical memory consolidation. A retrieval pipeline replaces pure cosine similarity with a four-signal score that mixes semantic relevance, temporal validity, live confidence, and graph-relational importance. A background consolidation agent detects contradictions, builds dependency edges, and propagates updates along those edges as graph-neural-network-style messages. Confidence is governed by a closed-form function combining an Ebbinghaus-style exponential decay, user-feedback reconsolidation, and logarithmic access reinforcement. We formalize the model, relate it to temporal knowledge graph embedding, agentic memory architectures, and uncertainty-aware RAG, and present a reference implementation. On a reproducible synthetic versioned-policy benchmark of 258 vectors and 138 queries, SmartVector roughly doubles top-1 accuracy over plain cosine RAG (62.0% vs. 31.0% on a held-out split), drops stale-answer rate from 35.0% to 13.3%, cuts Expected Calibration Error by nearly 2x (0.244 vs. 0.470), reduces re-embedding cost per single-word edit by 77%, and is robust across contradiction-injection rates from 0% to 75%. 

---
# Break the Optimization Barrier of LLM-Enhanced Recommenders: A Theoretical Analysis and Practical Framework 

**Authors**: Zhangchi Zhu, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20490)  

**Abstract**: Large language model (LLM)-enhanced recommendation models inject LLM representations into backbone recommenders to exploit rich item text without inference-time LLM cost. However, we find that existing LLM-enhanced methods significantly hinder the optimization of backbone models, resulting in high training losses that are difficult to reduce. To address it, we establish a comprehensive theoretical analysis of local optimization curvature and identify two key causes: 1) large norm disparity and 2) semantic-collaboration misaligned angular clustering of LLM representations. Guided by these insights, we propose Training-Friendly LLM-Enhanced Recommender (TF-LLMER), a lightweight framework with two key components. First, we highlight the necessity of item embedding normalization to eliminate norm-driven instability and achieve provable control over optimization conditioning. Second, we introduce Rec-PCA, a recommendation-aware dimensionality reduction method that injects collaborative structure into the representation transformation to resolve semantic-collaboration misaligned angular clustering. It jointly optimizes semantic information retention and alignment with an item-item co-occurrence graph constructed from interaction histories. The graph captures collaborative structure, and alignment is promoted by penalizing total variation over the graph. Both theory and extensive experiments demonstrate that TF-LLMER significantly outperforms state-of-the-art methods. Our code is available at this https URL. 

---
# HaS: Accelerating RAG through Homology-Aware Speculative Retrieval 

**Authors**: Peng Peng, Weiwei Lin, Wentai Wu, Xinyang Wang, Yongheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.20452)  

**Abstract**: Retrieval-Augmented Generation (RAG) expands the knowledge boundary of large language models (LLMs) at inference by retrieving external documents as context. However, retrieval becomes increasingly time-consuming as the knowledge databases grow in size. Existing acceleration strategies either compromise accuracy through approximate retrieval, or achieve marginal gains by reusing results of strictly identical queries. We propose HaS, a homology-aware speculative retrieval framework that performs low-latency speculative retrieval over restricted scopes to obtain candidate documents, followed by validating whether they contain the required knowledge. The validation, grounded in the homology relation between queries, is formulated as a homologous query re-identification task: once a previously observed query is identified as a homologous re-encounter of the incoming query, the draft is deemed acceptable, allowing the system to bypass slow full-database retrieval. Benefiting from the prevalence of homologous queries under real-world popularity patterns, HaS achieves substantial efficiency gains. Extensive experiments demonstrate that HaS reduces retrieval latency by 23.74% and 36.99% across datasets with only a 1-2% marginal accuracy drop. As a plug-and-play solution, HaS also significantly accelerates complex multi-hop queries in modern agentic RAG pipelines. Source code is available at: this https URL. 

---
# Discrete Preference Learning for Personalized Multimodal Generation 

**Authors**: Yuting Zhang, Ying Sun, Dazhong Shen, Ziwei Xie, Feng Liu, Changwang Zhang, Xiang Liu, Jun Wang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.20434)  

**Abstract**: The emergence of generative models enables the creation of texts and images tailored to users' preferences. Existing personalized generative models have two critical limitations: lacking a dedicated paradigm for accurate preference modeling, and generating unimodal content despite real-world multimodal-driven user interactions. Therefore, we propose personalized multimodal generation, which captures modal-specific preferences via a dedicated preference model from multimodal interactions, and then feeds them into downstream generators for personalized multimodal content. However, this task presents two challenges: (1) Gap between continuous preferences from dedicated modeling and discrete token inputs intrinsic to generator architectures; (2) Potential inconsistency between generated images and texts. To tackle these, we present a two-stage framework called Discrete Preference learning for Personalized Multimodal Generation (DPPMG). In the first stage, to accurately learn discrete modal-specific preferences, we introduce a modal-specific graph neural network (a dedicated preference model) to learn users' modal-specific preferences, which preferences are then quantized into discrete preference tokens. In the second stage, the discrete modal-specific preference tokens are injected into downstream text and image generators. To further enhance cross-modal consistency while preserving personalization, we design a cross-modal consistent and personalized reward to fine-tune token-associated parameters. Extensive experiments on two real-world datasets demonstrate the effectiveness of our model in generating personalized and consistent multimodal content. 

---
# Semantic Recall for Vector Search 

**Authors**: Leonardo Kuffo, Ioanna Tsakalidou, Roberta De Viti, Albert Angel, Jiří Iša, Rastislav Lenhardt  

**Link**: [PDF](https://arxiv.org/pdf/2604.20417)  

**Abstract**: We introduce Semantic Recall, a novel metric to assess the quality of approximate nearest neighbor search algorithms by considering only semantically relevant objects that are theoretically retrievable via exact nearest neighbor search. Unlike traditional recall, semantic recall does not penalize algorithms for failing to retrieve objects that are semantically irrelevant to the query, even if those objects are among their nearest neighbors. We demonstrate that semantic recall is particularly useful for assessing retrieval quality on queries that have few relevant results among their nearest neighbors-a scenario we uncover to be common within embedding datasets. Additionally, we introduce Tolerant Recall, a proxy metric that approximates semantic recall when semantically relevant objects cannot be identified. We empirically show that our metrics are more effective indicators of retrieval quality, and that optimizing search algorithms for these metrics can lead to improved cost-quality tradeoffs. 

---
# SAKE: Self-aware Knowledge Exploitation-Exploration for Grounded Multimodal Named Entity Recognition 

**Authors**: Jielong Tang, Xujie Yuan, Jiayang Liu, Jianxing Yu, Xiao Dong, Lin Chen, Yunlai Teng, Shimin Di, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.20146)  

**Abstract**: Grounded Multimodal Named Entity Recognition (GMNER) aims to extract named entities and localize their visual regions within image-text pairs, serving as a pivotal capability for various downstream applications. In open-world social media platforms, GMNER remains challenging due to the prevalence of long-tailed, rapidly evolving, and unseen entities. To tackle this, existing approaches typically rely on either external knowledge exploration through heuristic retrieval or internal knowledge exploitation via iterative refinement in Multimodal Large Language Models (MLLMs). However, heuristic retrieval often introduces noisy or conflicting evidence that degrades precision on known entities, while solely internal exploitation is constrained by the knowledge boundaries of MLLMs and prone to hallucinations. To address this, we propose SAKE, an end-to-end agentic framework that harmonizes internal knowledge exploitation and external knowledge exploration via self-aware reasoning and adaptive search tool invocation. We implement this via a two-stage training paradigm. First, we propose Difficulty-aware Search Tag Generation, which quantifies the model's entity-level uncertainty through multiple forward samplings to produce explicit knowledge-gap signals. Based on these signals, we construct SAKE-SeCoT, a high-quality Chain-of-Thought dataset that equips the model with basic self-awareness and tool-use capabilities through supervised fine-tuning. Second, we employ agentic reinforcement learning with a hybrid reward function that penalizes unnecessary retrieval, enabling the model to evolve from rigid search imitation to genuine self-aware decision-making about when retrieval is truly necessary. Extensive experiments on two widely used social media benchmarks demonstrate SAKE's effectiveness. 

---
# From Hidden Profiles to Governable Personalization: Recommender Systems in the Age of LLM Agents 

**Authors**: Jiahao Liu, Mingzhe Han, Guanming Liu, Weihang Wang, Dongsheng Li, Hansu Gu, Peng Zhang, Tun Lu, Ning Gu  

**Link**: [PDF](https://arxiv.org/pdf/2604.20065)  

**Abstract**: Personalization has traditionally depended on platform-specific user models that are optimized for prediction but remain largely inaccessible to the people they describe. As LLM-based assistants increasingly mediate search, shopping, travel, and content access, this arrangement may be giving way to a new personalization stack in which user representation is no longer confined to isolated platforms. In this paper, we argue that the key issue is not simply that large language models can enhance recommendation quality, but that they reconfigure where and how user representations are produced, exposed, and acted upon. We propose a shift from hidden platform profiling toward governable personalization, where user representations may become more inspectable, revisable, portable, and consequential across services. Building on this view, we identify five research fronts for recommender systems: transparent yet privacy-preserving user modeling, intent translation and alignment, cross-domain representation and memory design, trustworthy commercialization in assistant-mediated environments, and operational mechanisms for ownership, access, and accountability. We position these not as isolated technical challenges, but as interconnected design problems created by the emergence of LLM agents as intermediaries between users and digital platforms. We argue that the future of recommender systems will depend not only on better inference, but on building personalization systems that users can meaningfully understand, shape, and govern. 

---
# A Reproducibility Study of Metacognitive Retrieval-Augmented Generation 

**Authors**: Gabriel Iturra-Bocaz, Petra Galuscakova  

**Link**: [PDF](https://arxiv.org/pdf/2604.19899)  

**Abstract**: Recently, Retrieval Augmented Generation (RAG) has shifted focus to multi-retrieval approaches to tackle complex tasks such as multi-hop question answering. However, these systems struggle to decide when to stop searching once enough information has been gathered. To address this, \citet{zhou2024metacognitive} introduced Metacognitive Retrieval Augmented Generation (MetaRAG), a framework inspired by metacognition that enables Large Language Models to critique and refine their reasoning. In this reproducibility paper, we reproduce MetaRAG following its original experimental setup and extend it in two directions: (i) by evaluating the effect of PointWise and ListWise rerankers, and (ii) by comparing with SIM-RAG, which employs a lightweight critic model to stop retrieval. Our results confirm MetaRAG's relative improvements over standard RAG and reasoning-based baselines, but also reveal lower absolute scores than reported, reflecting challenges with closed-source LLM updates, missing implementation details, and unreleased prompts. We show that MetaRAG is partially reproduced, gains substantially from reranking, and is more robust than SIM-RAG when extended with additional retrieval features. 

---
# Enhancing Research Idea Generation through Combinatorial Innovation and Multi-Agent Iterative Search Strategies 

**Authors**: Shuai Chen, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.20548)  

**Abstract**: Scientific progress depends on the continual generation of innovative re-search ideas. However, the rapid growth of scientific literature has greatly increased the cost of knowledge filtering, making it harder for researchers to identify novel directions. Although existing large language model (LLM)-based methods show promise in research idea generation, the ideas they produce are often repetitive and lack depth. To address this issue, this study proposes a multi-agent iterative planning search strategy inspired by com-binatorial innovation theory. The framework combines iterative knowledge search with an LLM-based multi-agent system to generate, evaluate, and re-fine research ideas through repeated interaction, with the goal of improving idea diversity and novelty. Experiments in the natural language processing domain show that the proposed method outperforms state-of-the-art base-lines in both diversity and novelty. Further comparison with ideas derived from top-tier machine learning conference papers indicates that the quality of the generated ideas falls between that of accepted and rejected papers. These results suggest that the proposed framework is a promising approach for supporting high-quality research idea generation. The source code and dataset used in this paper are publicly available on Github repository: this https URL. The demo is available at this https URL. 

---
# Finding Duplicates in 1.1M BDD Steps: cukereuse, a Paraphrase-Robust Static Detector for Cucumber and Gherkin 

**Authors**: Ali Hassaan Mughal, Noor Fatima, Muhammad Bilal  

**Link**: [PDF](https://arxiv.org/pdf/2604.20462)  

**Abstract**: Behaviour-Driven Development (BDD) suites accumulate step-text duplication whose
maintenance cost is established in prior work. Existing detection techniques require
running the tests (Binamungu et al., 2018-2023) or are confined to a single
organisation (Irshad et al., 2020-2022), leaving a gap: a purely static,
paraphrase-robust, step-level detector usable on any repository. We fill the gap
with cukereuse, an open-source Python CLI combining exact hashing, Levenshtein
ratio, and sentence-transformer embeddings in a layered pipeline, released alongside
an empirical corpus of 347 public GitHub repositories, 23,667 parsed .feature
files, and 1,113,616 Gherkin steps. The step-weighted exact-duplicate rate is 80.2
%; the median-repository rate is 58.6 % (Spearman rho = 0.51 with size). The top
hybrid cluster groups 20.7k occurrences across 2.2k files. Against 1,020 pairs
manually labelled by the three authors under a released rubric (inter-annotator
Fleiss' kappa = 0.84 on a 60-pair overlap), we report precision, recall, and F1 with
bootstrap 95 % CIs under two protocols: the primary rubric and a score-free
second-pass relabelling. The strongest honest pair-level number is near-exact at F1
= 0.822 on score-free labels; the primary-rubric semantic F1 = 0.906 is inflated by
a stratification artefact that pins recall at 1.000. Lexical baselines
(SourcererCC-style, NiCad-style) reach primary F1 = 0.761 and 0.799. The paper also
presents a CDN-structured critique of Gherkin (Cognitive Dimensions of Notations);
eight of fourteen dimensions are rated problematic or unsupported. The tool, corpus,
labelled pairs, rubric, and pipeline are released under permissive licences. 

---
# AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning in E-commerce 

**Authors**: Biao Zhang, Lixin Chen, Bin Zhang, Zongwei Wang, Tong Liu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.20135)  

**Abstract**: Multimodal representation is crucial for E-commerce tasks such as identical product retrieval. Large representation models (e.g., VLM2Vec) demonstrate strong multimodal understanding capabilities, yet they struggle with fine-grained semantic comprehension, which is essential for distinguishing highly similar items. To address this, we propose Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning (AFMRL), which defines product fine-grained understanding as an attribute generation task. It leverages the generative power of Multimodal Large Language Models (MLLMs) to extract key attributes from product images and text, and enhances representation learning through a two-stage training framework: 1) Attribute-Guided Contrastive Learning (AGCL), where the key attributes generated by the MLLM are used in the image-text contrastive learning training process to identify hard samples and filter out noisy false negatives. 2) Retrieval-aware Attribute Reinforcement (RAR), where the improved retrieval performance of the representation model post-attribute integration serves as a reward signal to enhance MLLM's attribute generation during multimodal fine-tuning. Extensive experiments on large-scale E-commerce datasets demonstrate that our method achieves state-of-the-art performance on multiple downstream retrieval tasks, validating the effectiveness of harnessing generative models to advance fine-grained representation learning. 

---
# DR-Venus: Towards Frontier Edge-Scale Deep Research Agents with Only 10K Open Data 

**Authors**: Venus Team, Sunhao Dai, Yong Deng, Jinzhen Lin, Yusheng Song, Guoqing Wang, Xiaofeng Wu, Yuqi Zhou, Shuo Yang, Zhenzhe Ying, Zhanwei Zhang, Changhua Meng, Weiqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.19859)  

**Abstract**: Edge-scale deep research agents based on small language models are attractive for real-world deployment due to their advantages in cost, latency, and privacy. In this work, we study how to train a strong small deep research agent under limited open-data by improving both data quality and data utilization. We present DR-Venus, a frontier 4B deep research agent for edge-scale deployment, built entirely on open data. Our training recipe consists of two stages. In the first stage, we use agentic supervised fine-tuning (SFT) to establish basic agentic capability, combining strict data cleaning with resampling of long-horizon trajectories to improve data quality and utilization. In the second stage, we apply agentic reinforcement learning (RL) to further improve execution reliability on long-horizon deep research tasks. To make RL effective for small agents in this setting, we build on IGPO and design turn-level rewards based on information gain and format-aware regularization, thereby enhancing supervision density and turn-level credit assignment. Built entirely on roughly 10K open-data, DR-Venus-4B significantly outperforms prior agentic models under 9B parameters on multiple deep research benchmarks, while also narrowing the gap to much larger 30B-class systems. Our further analysis shows that 4B agents already possess surprisingly strong performance potential, highlighting both the deployment promise of small models and the value of test-time scaling in this setting. We release our models, code, and key recipes to support reproducible research on edge-scale deep research agents. 

---
# SkillGraph: Graph Foundation Priors for LLM Agent Tool Sequence Recommendation 

**Authors**: Hao Liu, Dongyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.19793)  

**Abstract**: LLM agents must select tools from large API libraries and order them correctly. Existing methods use semantic similarity for both retrieval and ordering, but ordering depends on inter-tool data dependencies that are absent from tool descriptions. As a result, semantic-only methods can produce negative Kendall-$\tau$ in structured workflow domains. We introduce SkillGraph, a directed weighted execution-transition graph mined from 49,831 successful LLM agent trajectories, which encodes workflow-precedence regularities as a reusable graph foundation prior. Building on this graph foundation prior, we propose a two-stage decoupled framework: GS-Hybrid retrieval for candidate selection and a learned pairwise reranker for ordering. On ToolBench (9,965 test instances; ~16,000 tools), the method reaches Set-F1 = 0.271 and Kendall-$\tau$ = 0.096; on API-Bank, Kendall-$\tau$ improves from -0.433 to +0.613. Under identical Stage-1 inputs, the learned reranker also outperforms LLaMA-3.1-8B Stage-2 rerankers. 

---
# Self-Describing Structured Data with Dual-Layer Guidance: A Lightweight Alternative to RAG for Precision Retrieval in Large-Scale LLM Knowledge Navigation 

**Authors**: Hung Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.19777)  

**Abstract**: Large Language Models (LLMs) exhibit a well-documented positional bias when processing long input contexts: information in the middle of a context window receives substantially less attention than content at the boundaries, a phenomenon termed the Lost-in-the-Middle effect (Liu et al., 2024). This limits knowledge-retrieval applications that embed large structured knowledge bases directly in the LLM context. Retrieval-Augmented Generation (RAG) addresses scalability by retrieving only relevant fragments, but introduces substantial infrastructure overhead and is ill-suited to libraries whose semantic boundaries are human-defined rather than statistically learned.
We propose Self-Describing Structured Retrieval (SDSR), a lightweight framework in which structured data files embed human-authored navigational metadata at the file's primacy position, thereby exploiting rather than fighting the LLM's primacy bias. We further propose a Dual-Layer Guidance strategy combining in-file metadata with explicit routing rules in the system prompt.
We validate SDSR through a four-round benchmark using a 190-skill library expanded from 36 to 119 categories via adversarial distractor injection. Four conditions are tested: (A) no guidance, (B) in-file summary only, (C) prompt hint only, (D) both combined. Version D achieves 100% primary routing accuracy (20/20) at 119 categories versus 65% for the no-guidance baseline. We identify a fundamental asymmetry: primary routing is solvable by explicit rules, while secondary cross-category routing requires architectural intent explicitly encoded in the data structure. We further extend SDSR to semi-structured corpora, showing how cross-reference encoding enables operation without vector databases in domains with recoverable document structure. 

---
# Cognis: Context-Aware Memory for Conversational AI Agents 

**Authors**: Parshva Daftari, Khush Patel, Shreyas Kapale, Jithin George, Siva Surendira  

**Link**: [PDF](https://arxiv.org/pdf/2604.19771)  

**Abstract**: LLM agents lack persistent memory, causing conversations to reset each session and preventing personalization over time. We present Lyzr Cognis, a unified memory architecture for conversational AI agents that addresses this limitation through a multi-stage retrieval pipeline. Cognis combines a dual-store backend pairing OpenSearch BM25 keyword matching with Matryoshka vector similarity search, fused via Reciprocal Rank Fusion. Its context-aware ingestion pipeline retrieves existing memories before extraction, enabling intelligent version tracking that preserves full memory history while keeping the store consistent. Temporal boosting enhances time-sensitive queries, and a BGE-2 cross-encoder reranker refines final result quality. We evaluate Cognis on two independent benchmarks -- LoCoMo and LongMemEval -- across eight answer generation models, demonstrating state-of-the-art performance on both. The system is open-source and deployed in production serving conversational AI applications. 

---
