# jina-reranker-v3.5: An Efficient Listwise Reranker with Hybrid Attention and Self-Distillation 

**Authors**: Christina Nasika, Feng Wang, Antonis Krasakis, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2607.18152)  

**Abstract**: Listwise rerankers are the discriminative core of agentic retrieval pipelines, yet production deployment demands efficiency, domain robustness, and fluency on semi-structured data at the same time. We present jina-reranker-v3.5, a 0.6B-parameter listwise reranker that meets these demands together without sacrificing the cross-document comparison that makes its predecessor jina-reranker-v3 effective. jina-reranker-v3.5 keeps the last-but-not-late (LBNL) interaction of jina-reranker-v3 and reworks it along three axes. It replaces uniform global attention with a hybrid schedule of three sliding-window layers followed by two global layers, pinning the terminal layer to global as LBNL readout requires. It trains on a curated multi-domain mixture that spans legal, medical, financial, multilingual, and structured retrieval. It transfers quality through a three-stage self-distillation recipe in which a full-attention teacher sets an upper bound that a sparse-attention student then recovers under a staged adaptation protocol. jina-reranker-v3.5 reaches 63.20 nDCG@10 on BEIR, matching a 4B model at roughly 7x fewer parameters, and improves over jina-reranker-v3 on MIRACL and RTEB as well. Its largest gains come on semi-structured retrieval, where it lifts nDCG@10 by 9.6 points over jina-reranker-v3 and leads all rerankers of comparable size. The hybrid schedule further cuts listwise inference latency by up to 1.56x. We release the model weights on Hugging Face under a non-commercial license. 

---
# FinSAgent: Corpus-Aligned Multi-Agent RAG Framework for Evidence-Grounded SEC Filing Question Answering 

**Authors**: Jijun Chi, Zhenghan Tai, Hanwei Wu, Tung Sum Thomas Kwok, Hailin He, Zixing Liao, Bohuai Xiao, Chaolong Jiang, Jianliang Lei, Jerry Huang, Peng Lu, Muzhi Li, Liheng Ma, Yihong Wu, Sicheng Lyu, Jingrui Tian, Yihan Li, Yanzhang Ma, Dingtao Hu, Yufei Cui, Ling Zhou, Lei Ding, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.18102)  

**Abstract**: Financial question answering over U.S. Securities and Exchange Commission (SEC) filings requires retrieving and synthesizing heterogeneous evidence dispersed across long, standardized, and highly redundant disclosures. Existing retrieval-augmented and multi-agent systems typically derive retrieval queries directly from the user's question and rank candidates by semantic similarity. Together, these choices create prior-corpus misalignment: a mismatch between model priors and the target filings' structure, terminology, and evidence standards. As a result, query generation misses corpus-specific evidence, while semantic reranking favors topically similar but evidentially invalid false-positive chunks. We propose FinSAgent, an evidence-grounded multi-agent framework that reframes SEC filing QA as corpus-aligned retrieval planning and corrects both ends with a single principle: inject corpus-side conditioning wherever model priors would otherwise dominate. FinSAgent combines (1) role-specialized agents anchored to the mandated 10-K item structure, (2) database-aware query decomposition that conditions each agent's sub-queries on a lightweight, summary-level view of the local corpus, and (3) multi-path retrieval with a learned feature-gated reranker that separates evidential validity from semantic similarity. Across five offline financial QA benchmarks, FinSAgent improves retrieval coverage and answer correctness over strong single-agent and multi-agent baselines; in a three-arm randomized online experiment with 1,000 anonymous user ratings, it also receives higher scores than baselines. 

---
# Evidence-in-the-Loop: Trace-Driven Optimization for Customer-Service LLM Agents 

**Authors**: Chunming Wu, Dafei Qiu, Congde Yuan, Charles Quan, Jun Wu, Suipeng Li, Mo Wu, Gavin Xie, Hope Chen, Max Yao  

**Link**: [PDF](https://arxiv.org/pdf/2607.18039)  

**Abstract**: Production customer-service bots must improve answer quality across iterative releases, yet large language models must not bypass evidence boundaries, policy rules, or human-handoff safeguards. We present an \textbf{Evidence-Grounded Customer-Service Agent Workflow} deployed in a real-world customer-service setting. BM25 recall, issue-title-vector recall, issue-description-vector recall, weighted RRF fusion, and cross-encoder reranking construct grounded FAQ evidence for controlled LLM decisions. Policy-guided orchestration then combines this RAG evidence with scenario-specific rule evidence, conversation memory, and clarification state inside a fixed LangGraph DAG~\cite{langgraph2024}. The paper contributes three reusable deployment patterns: \textbf{hybrid RAG evidence construction}, where multi-channel retrieval and reranking produce auditable FAQ candidates; \textbf{evidence-grounded issue/action decision}, where an Evidence-Grounded Decision Module selects an issue/action from typed FAQ evidence and scenario-specific rule evidence; and \textbf{trace-driven RAG and reranker improvement}, where traces diagnose whether failures come from recall, ranking, final candidate selection, clarification, rule-derived evidence, or action policy, and where reranker fine-tuning is evaluated not only for in-domain gain but also for forgetting risk. 

---
# Remote Awareness of Seafloor Images Collected by AUVs over Low-Bandwidth Communication Links 

**Authors**: Adrian Bodenmann, Cailei Liang, Miquel Massot-Campos, Samuel Simmons, Alexander B. Phillips, Alberto Consensi, Matthew Kingsland, Rashiid Sherif, Stan Brown, Adam Riese, Blair Thornton  

**Link**: [PDF](https://arxiv.org/pdf/2607.18013)  

**Abstract**: This paper introduces a method for real-time processing and transmission of autonomous underwater vehicle (AUV) imagery over low-bandwidth communication links. It leverages artificial intelligence (AI) techniques to identify a set of images that best represent an entire dataset, or automatically finds the most similar images to a given query image for transmission to operators. Combined with metadata of a larger set of images, compressed versions of the selected images can be transmitted over satellite communication links or underwater modems, and provide operators on shore with information about the type of imagery the AUV is collecting while it is still deployed. Data from three deployments off the coast of the UK and in Gran Canaria using different AUVs and imaging systems demonstrate the method in the field. It achieved an almost 400,000-fold reduction in data volume compared to the raw data size, enabling transmission of data summaries of a 2-hour 47-minute-long mapping mission in just over 34 minutes over low-bandwidth satellite communication. 

---
# MagicSelector: Joint Optimization for Agent Tool Selection via Counterfactual Decomposition and Progressive Reranking 

**Authors**: HONOR Agentic Search Team, Zhengzong Chen, Lei Tang, Lijun Liu, Chuandi Jiang, Fan Yang, Keyun Chu, Chu Zhao, Shihao Liu, Minghang Li, Bo Liang, Can Wen, Hailong Wu, Jingnan Ju, Mian Liu, Nengbin Zhang, Peiqiang Wang, Penghe Nie, Qinhui Gu, Sijia Lv, Siqi Chen, Wei Zhang, Yang Xu, Yuhao Qian, Yuxiang Zhang, Zeng Cheng, Zhen Wang, Zuan Chen, Yuanyuan Zhao, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.17751)  

**Abstract**: We present MagicSelector, a joint optimization framework integrating Counterfactual task decomposition, Progressive reranking, and Dynamic Top-K, designed to address the fundamental challenges of tool retrieval in agents. MagicSelector is a specialized framework capable of translating ambiguous user instructions into executable atomic subtasks and guiding high-precision tool retrieval, effectively mitigating redundant noise and severe context distraction in out-of-domain (OOD) this http URL empower MagicSelector with these capabilities through three key contributions: (1) a preferenceguided counterfactual task decomposition mechanism that utilizes a counterfactual reward to quantify the marginal causal gain of decomposition on retrieval ranking, effectively imposing fine-grained structural supervision on logical coherence; (2) a progressive tool reranking method driven by self-distillation hard negative mining, which optimizes both point-wise and list-wise relevance to enhance fine-grained discrimination among highly similar tools; and (3) a dual semantic boundary-aware dynamic Top-K strategy that adaptively monitors reranking score cliffs and inter-tool semantic shifts to dynamically truncate the candidate list, maximizing relevant tool recall while filtering long-tail noise. Evaluated on MTDTool, the first task decomposition benchmark we constructed tailored for mobile multi-turn interactions with process-level annotations, MagicSelector yields promising performance. Extensive experiments demonstrate that MagicSelector significantly outperforms state-of-the-art methods in terms of tool retrieval accuracy, OOD generalization capability, and overall token efficiency, thereby demonstrating the effectiveness of our proposed framework. 

---
# RAMP: Robust Ad Recommendation Under Limited Personalized-Feature Availability via Masking and Alignment Pathways 

**Authors**: Dairui Liu, Zhongyi Lu, Roger Zhe Li, Changhong Jin, Jitao Lu, Xinyang Shao, Bichen Shi, Mete Sertkan, Aghiles Salah, Aonghus Lawlor, Barry Smyth, Tri Kurniawan Wijaya, Ruihai Dong, Xingsheng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.17473)  

**Abstract**: Click-through rate (CTR) and conversion rate (CVR) prediction are fundamental tasks in online advertising, aiming to estimate the likelihood of user interactions based on various features. While personalized attributes such as age and gender can significantly enhance predictive accuracy, their use is increasingly restricted by privacy regulations, thereby limiting available data for both training and inference. To address this challenge, we propose RAMP (Robust Ad Recommendation Under Limited Personalized-Feature Availability via Masking and Alignment Pathways), which is designed to improve CTR/CVR prediction accuracy when personalized features are not accessible, thus supporting deployment in privacy-constrained this http URL consists of (i) a personalized pathway built upon a dual-tower component with identical inputs but independent parameters, where output masking separates predictions for personalized and non-personalized signals, (ii) a separate non-personalized pathway trained with non-personalized features only, and (iii) a distillation-inspired prediction-alignment architecture between (i) and (ii) that improves prediction when personalized features are unavailable. We conduct comprehensive experiments using both public benchmarks and industrial datasets to evaluate the performance of RAMP. Our evaluation spans multiple backbone models and different settings: with and without access to personalized features. The results show that RAMP consistently outperforms state-of-the-art methods when personalized features are missing, while maintaining competitive performance when all features are available. %demonstrating its effectiveness and practicality for real-world advertising systems. Our code is publicly available at this https URL. 

---
# HyCoRec: Hypergraph-Enhanced Multi-Preference Learning for Alleviating Matthew Effect in Conversational Recommendation 

**Authors**: Yongsen Zheng, Ruilin Xu, Ziliang Chen, Guohua Wang, Mingjie Qian, Jinghui Qin, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2607.17461)  

**Abstract**: The Matthew effect is a notorious issue in Recommender Systems (RSs), \emph{i.e.}, the rich get richer and the poor get poorer, wherein popular items are overexposed while less popular ones are regularly ignored. Most methods examine Matthew effect in static or nearly-static recommendation scenarios. However, the Matthew effect will be increasingly amplified when the user interacts with the system over time. To address these issues, we propose a novel paradigm, Hypergraph-Enhanced Multi-Preference Learning for Alleviating Matthew Effect in Conversational Recommendation (HyCoRec), which aims to alleviate the Matthew effect in conversational recommendation. Concretely, HyCoRec devotes to alleviate the Matthew effect by learning multi-aspect preferences, \emph{i.e.}, item-, entity-, word-, review-, and knowledge-aspect preferences, to effectively generate responses in the conversational task and accurately predict items in the recommendation task when the user chats with the system over time. Extensive experiments conducted on two benchmarks validate that HyCoRec achieves new state-of-the-art performance and the superior of alleviating Matthew effect. Our code is available at this https URL. 

---
# The Matryoshka Hypencoder 

**Authors**: Majd Alkawaas, Sean MacAvaney  

**Link**: [PDF](https://arxiv.org/pdf/2607.17457)  

**Abstract**: The Hypencoder is a recently-proposed retrieval approach that encodes queries as shallow neural networks ("Q-Nets") that estimate relevance over pre-computed document embeddings. Inspired by Matryoshka Representation Learning, we show that the Hypencoder can be extended to support multiple sizes of Q-Nets, allowing trade-offs between effectiveness and efficiency when deployed. We find that this "Matryoshka Hypencoder" achieves comparable in-domain effectiveness with approximately 7x fewer active parameters in-domain and half as many active parameters out-of-domain, which corresponds to a 1.6-3.4x increase in scoring throughput. This work paves the way for practical deployment of Hypencoders. 

---
# Adapting Embedding Models for Agent Capability Retrieval 

**Authors**: Tingwei Chen, Yunxiao Shi, Zhengdong Chu, Qingsong Wen, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.17347)  

**Abstract**: Open agent marketplaces list native agents, tool bundles, and reusable skill packages in the same search interface, yet practitioners still have little guidance on how to retrieve across this mixed catalog. We study whether off-the-shelf retrieval models, trained for general text retrieval, can be adapted to match user queries to executable agent capabilities, and whether the learned signal transfers beyond the benchmark used for tuning. We fine-tune three open retrieval backbones, BGE-base, KaLM-v1.5, and EasyRec, on AgentSelect, which represents marketplace-visible units as capability profiles derived from public metadata, and test transfer on two catalogs not seen during training: MuleRun native agents and a ClawHub benchmark of 50 skills with 1,000 queries. Adaptation helps on both catalogs. Code and data will be released upon publication. 

---
# Learning Sparse Representations of Multimodal Content for Enhanced Cold Item Recommendation 

**Authors**: Gregor Meehan, Johan Pauwels  

**Link**: [PDF](https://arxiv.org/pdf/2607.17184)  

**Abstract**: The scale and rapid growth of item catalogs in modern digital platforms present significant challenges to recommender system (RS) practitioners. Most RSs use embedding similarity to predict user-item preferences, but embedding storage and low-latency retrieval is challenging in industry-scale catalogs. Furthermore, newly added items do not have corresponding embeddings and cannot be recommended effectively; previous works often tackle this item cold-start problem by generating cold item representations from auxiliary content, such as images or descriptive text, so that user preferences can be predicted without historical interactions. In this paper, we argue that sparse embeddings have notable advantages over standard dense vectors in this content-based cold-start paradigm. We describe how existing cold-start training regimes can be adapted for sparse representation learning, and build on insights from linear attention to design a pre-sparsification activation technique that induces sharpness and denoising effects in learned item-item similarities. We show that the resulting sparse embeddings achieve significant improvements in cold-start recommendation accuracy over dense embeddings at considerably lower storage costs, especially for users with multiple interests. Through comprehensive experiments on four multimodal RS datasets, we also demonstrate the interpretability of sparse content embeddings and their robustness in the trade-off between size and accuracy. 

---
# Fenced Citation-Context Retrieval for Case Law: Temporal Leakage and Degree Control Across Two Jurisdictions 

**Authors**: Yao Liu, Tien-Ping Tan, Zhilan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.17142)  

**Abstract**: Incoming citation context-the text with which later cases describe a case when citing it-is a strong signal for legal precedent retrieval, but it is evaluated without a temporal fence: at query time the target precedent has not yet been cited by future cases, so an unfenced method credits itself with evidence unavailable at deployment. Our central contribution is to measure how much: a temporal-admission decomposition that splits the naive relax-the-fence gain into genuine future-citation leakage, legitimate pre-query admission, and an index effect, reporting the phantom fraction-the share that is future evidence. We instantiate this as a controlled deployability audit of incoming-citation retrieval across two jurisdictions, CLERC (US federal, 1.84M documents) and ECtHR-PCR (European Court of Human Rights), adding a zero-training anchor channel that admits only citers dated before the query. On ECtHR-PCR only 14.9% of the naive gain is genuine future evidence; on CLERC the over-credit grows to +4.95 R@1000 as the fence is relaxed. Under the fence, the zero-training channel gains +16.1 R@1000 over BM25 on CLERC, clears a citation-degree control, and reaches a published R@1000 estimate similar to the strongest trained system on ECtHR-PCR (79.56 vs 79.39) at zero training. 

---
# Uncertainty as Remedy: Mitigating Satisfaction Label Bias in Short Video Multi-Objective Ensemble Ranking 

**Authors**: Zonghe Shao, Tiantian He, Xiaoxiao Xu, Jiaqi Yu, Minzhi Xie, Jinfang Gu, Yongqi Liu, Kaiqiao Zhan, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2607.17092)  

**Abstract**: The core objective of short video recommendation is to model users' unobservable true satisfaction with recommended videos. As the dominant industrial framework, end-to-end multi-objective ensemble ranking models are typically trained with multi-dimensional dense user behavioral signals, such as clicks and watch time. However, these behavioral signals are partial, fragmented, and often mutually conflicting user satisfaction proxies, introducing uncertainty and label bias into satisfaction modeling. Conventional deterministic models overlook this uncertainty, which exacerbates satisfaction label bias and results in suboptimal model convergence. Meanwhile, existing uncertainty-aware methods mostly employ uncertainty for post-hoc ranking adjustments rather than leveraging it as a remedy to mitigate the inherent bias within the core optimization pipeline. This paper proposes UAME, an Uncertainty-Aware end-to-end Multi-objective Ensemble ranking framework for short video recommendation. UAME represents the model's prediction as a Gaussian scoring variable, where the mean denotes the predicted satisfaction score and the variance quantifies predictive uncertainty associated with this score. We further design a probabilistic pairwise ranking loss, and construct an uncertainty-aware sample-level weighting scheme to mitigate the bias. We further provide theoretical analysis suggesting that the weighting scheme helps mitigate satisfaction label bias. Extensive offline and online experiments on a large-scale industrial short video platform demonstrate that UAME consistently improves two state-of-the-art paradigms, EMER and EASQ, and better aligns with questionnaire-based user satisfaction. UAME has been deployed in our production short-video recommendation system and continues to deliver stable, statistically significant gains. 

---
# WHALE: A Scalable Unified Model for Recommendation with Wukong-HSTU Architecture 

**Authors**: Renqin Cai, Dawei Sun, Yuanjun Yao, Zhiyong Wang, Velvin Fu, Maggie Zhuang, Yu Shi, Zhongnan Fang, Xuan Cao, Jing Qian, Rui Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.17017)  

**Abstract**: As scalability becomes increasingly important in recommendation modeling, recent architectures have advanced the modeling of two broad sources of ranking signals along separate paths: non-sequence features, including user, item, context, and cross features; and sequence features from user behavior histories. Wukong and HSTU have emerged as representative scalable backbones for these paths: Wukong scales high-order non-sequence feature-interaction modeling, while HSTU scales long user-behavior sequence modeling. Despite their complementary strengths, practical architectures that combine these two types of feature modeling remain underexplored. We present WHALE, a scalable unified recommendation architecture that jointly models non-sequence and sequence features on top of Wukong and HSTU. Each WHALE layer contains a Wukong module, an HSTU module, and an attention-based fusion module in which Wukong-derived interaction representations query HSTU-derived behavior representations. This design keeps both backbones active throughout the network and enables progressive Wukong-HSTU exchange, allowing high-order feature crosses to repeatedly retrieve fine-grained evidence from long user histories. To make WHALE practical for industrial deployment, we introduce customized Triton kernels and other model-systems co-design techniques to improve training and inference efficiency. On large-scale industrial recommendation data, WHALE achieves consistent gains in offline experiments. Additionally, it delivers positive online gains with a modest serving-throughput trade-off. The method has been deployed in production systems. Overall, WHALE provides a practical example of how these two sources of information can be scalably unified in an industrial recommendation model. 

---
# Beyond Fixed Depths and Widths: Optimizing Textual Decoding Tries in LLM-based Generative Recommendation 

**Authors**: Jingzhe Liu, Hanbing Wang, Jiliang Tang, Liam Collins, Tong Zhao, Neil Shah, Mingxuan Ju  

**Link**: [PDF](https://arxiv.org/pdf/2607.16633)  

**Abstract**: Generative recommendation (GR) is an increasingly popular paradigm in recommender systems, with a prominent line of work using LLMs as autoregressive backbones to predict the next item's term IDs (e.g., titles or keywords). The success of autoregressive generation hinges on constrained beam search over a decoding trie to ensure that generated outputs correspond to valid items. However, current research predominantly focuses on generating more comprehensive term IDs to describe items, while largely neglecting the structural design of the decoding trie formed by these terms. This can lead to a trie that is poorly suited to beam search, which degrades performance. To address this, we examine the effectiveness of term IDs from the perspective of decoding trie optimization. Through empirical and theoretical analyses, we identify two desirable properties for a highly performant trie: (1) adaptive and variable ID length, enabling items with varying semantic richness to be represented by IDs of appropriate lengths, and (2) constrained branching factors, especially at shallow levels, which drastically improves the success rate of constrained beam search. Motivated by these properties, we introduce BONSAI: Branching-Optimized Node Structure for Adaptive Identifiers, a novel framework that co-designs textual term IDs and their underlying decoding trie. BONSAI extracts recommendation-informative words from item metadata and employs a minimum set cover formulation to recursively build a trie that satisfies the above properties. Experiments reveal that BONSAI achieves up to a 21.6% relative improvement over state-of-the-art baselines. Further analyses confirm the crucial role of our proposed properties, and demonstrate their generalizability to be applied to enhance the performance of other term ID methods. 

---
# A Quantum-Classical Hybrid Framework for Multivariate Time-Series Forecasting Complexity-Fidelity Trade-offs and Limitations 

**Authors**: Sanjay Chakraborty, Fredrik Heintz  

**Link**: [PDF](https://arxiv.org/pdf/2607.16358)  

**Abstract**: This paper presents a unified quantum-classical hybrid framework for multi-horizon time-series forecasting, introducing two model variants Quantum Reservoir Forecaster (QRC-F) and Variational Quantum Forecaster (VQF-F). The proposed framework investigates the complexity-fidelity trade-off of quantum forecasting under near-term NISQ hardware constraints. Continuous time-series signals are transformed into binary representations through uniform quantization and encoded into quantum states using angle encoding with parameterized RY rotation gates. Cross-channel entanglement layers capture dependencies among multiple variables. QRC-F utilizes a fixed random unitary quantum reservoir for stable, gradient-free temporal feature extraction, whereas VQF-F employs a trainable variational quantum circuit optimized through the parameter-shift rule to learn temporal and inter-variable patterns from Pauli expectation values. Both models replace computationally expensive quadratic self-attention with efficient linear transformations, reducing parameter complexity. A shared MIMO-based multi-horizon prediction head simultaneously generates forecasts across multiple horizons, avoiding error accumulation in recursive forecasting. Experimental evaluations on benchmark datasets, including ETTh1, ETTh2, ETTm1, ETTm2, Weather, electricity, and exchange-rate, demonstrate that VQF-F achieves superior training stability and parameter efficiency, while QRC-F provides enhanced robustness and circuit fidelity under quantum noise. The results establish a practical quantum-native forecasting framework with strong potential for deployment on near-term NISQ devices. 

---
# ANNLib: A Development Framework for Efficient Approximate Nearest Neighbor Search 

**Authors**: Zheqi Shen, Jingbo Su, Zijin Wan, Yan Gu, Yihan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.17582)  

**Abstract**: Approximate Nearest Neighbor Search (ANNS) plays a pivotal role in modern deep learning pipelines. Recently, many ANNS systems have been proposed to either provide broad functionality or reach high performance. However, it is yet difficult to achieve both with minimal programming efforts. We propose ANNLib to address the gap. ANNLib is a library that provides a programming framework for achieving high performance and flexible functionality in ANNS systems, based on popular graph-based ANNS algorithms. We carefully decouple and independently optimize both the algorithm and the data structure components of an ANNS system. In addition, we integrate state-of-the-art algorithms and data structures into ANNLib as modules, along with our new designs. Users can choose combinations of components to implement sophisticated settings with high performance, such as filter search, fully dynamic updates, and historical queries on snapshots. Our experiments show that our new solution provides a simple interface for various applications and achieves comparable or even better performance than previous work, specifically for each application. 

---
# D-NOVA: In-Storage Retrieval Accelerator via Dual-Bound 3D NAND-Optimized Similarity Search with Vector Adaptation 

**Authors**: Chang Eun Song, Sumukh Pinge, Tianqi Zhang, Sung Eun Kim, Tajana S. Rosing, Mingu Kang  

**Link**: [PDF](https://arxiv.org/pdf/2607.17538)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the factual grounding of large language model (LLM) inference by retrieving relevant information from external knowledge bases. However, its dense vector retrieval introduces significant latency and energy overhead, becoming the primary performance bottleneck. Although recent in-storage accelerators aim to reduce data movement, they still rely on host or embedded processors outside the memory, where nearly 70% of the total retrieval time is spent. As a result, they cannot fully overcome the bandwidth limitations, leading to yet another memory bottleneck. To tackle these limitations, we present D-NOVA, a hardware-software co-designed in-storage retrieval accelerator. D-NOVA executes an inverted file (IVF)-based hierarchical retrieval pipeline by deeply embedding the search functionality directly into the NAND memory array. This is achieved by incorporating a new distance metric, Dual-Bound Tight Similarity Sensing (DTS), which is specifically tailored for searching within the NAND string. In addition, we introduce a lightweight contrastive adapter that maps embedding vectors into a DTS-friendly domain, recovering near-software recall while improving performance and energy efficiency. D-NOVA is up to 41.7x faster and 71x more energy-efficient than a CPU baseline, and achieves 12.13x higher throughput while being up to 1.26x more energy-efficient than state-of-the-art in-storage RAG accelerators, demonstrating the potential of fully in-storage vector search for scalable RAG acceleration. 

---
# DRNOISE: Benchmarking Deep Research Agents in Misleading Evidence Environments 

**Authors**: Jun Nie, Zhiqin Yang, Zhenheng Tang, Yonggang Zhang, Xiaowen Chu, Xinmei Tian, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2607.17291)  

**Abstract**: Deep research agents increasingly operate over the open web, where relevant records coexist with redundant summaries, outdated reports, and misleading documents. Existing evaluations offer limited insight into whether agents preserve sound evidential standards when an ordinary-looking false document is deliberately seeded into a searchable environment and offers a direct shortcut to a conflicting answer. We introduce DRNOISE, a 100-task benchmark for answer recovery under misleading evidence. Each task has a unique gold answer supported by two corroborating indirect record chains; the paired noisy condition adds one plausible document that states a conflicting answer directly. The benchmark spans ten families of evidence operations. Across agents with strong clean-task performance, this single intervention causes 66-88 percentage-point accuracy drops. Trace analyses identify verification inertia as the dominant failure mode: agents often retrieve truthful records but stop before completing and reconciling the evidence chain, instead deferring to the answer-like document. Generic verification prompts reduce but do not close this gap. The setting is especially relevant to open-web deployment, where plausible falsehoods arrive through ordinary-looking pages rather than explicit attacks. Reliable deep research therefore requires more than retrieval and citation; it requires active reconciliation of direct claims with record-level evidence. 

---
# TurboVec: A Case Study in Cost-Efficient Private Retrieval for Enterprise RAG via Codebook-Oblivious Quantization 

**Authors**: Navnit Shukla, Kamal Pandey, Omsankar Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2607.16973)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems increasingly power enterprise LLM applications, yet the vector retrieval layer introduces two underexplored challenges: (1) trained codebook quantizers may expose corpus statistics during index construction, creating a leakage channel in multi-tenant deployments, and (2) post-hoc filtering for tenant isolation degrades recall on selective queries. We study TurboVec, an open-source vector index built on TurboQuant - a codebook-oblivious scalar quantizer requiring no corpus-dependent training. On the DBpedia OpenAI embeddings benchmark (d=1536, 100K-999K vectors), TurboQuant 4-bit outperforms trained FAISS Product Quantization at the same memory budget by 8.5-8.9 percentage points in Recall@5 across all scales. Compared to HNSW (R@5=0.991) and IVF-PQ (R@5=0.840), TurboQuant occupies a distinct design point: higher recall than IVF-PQ without training, at 4-8x less memory than HNSW. Deployed on Snowpark Container Services, TurboVec achieves 11ms median query latency at 100K vectors versus 707ms for warehouse brute-force scan. Kernel-level allowlist filtering maintains 0.86-0.93 Recall@10 across 10-1000 tenant workloads versus 0.09-0.19 for post-filter baselines. Codebook-oblivious design reduces membership inference accuracy to near-random (50.0%) versus 57.3% for PQ codebooks. Limitations include single dataset evaluation, uncompressed HNSW comparison, and privacy evaluation on synthetic data only. 

---
# Adaptive Incident Prioritization for Security Operations at Scale 

**Authors**: Scott Freitas, Amir Gharib, Maayan Magenheim  

**Link**: [PDF](https://arxiv.org/pdf/2607.16963)  

**Abstract**: Large security operations centers (SOCs) often face hundreds of active incidents per day, creating substantial cognitive and operational demands for analysts. Analysts must quickly decide which incidents deserve attention within long, constantly changing queues, yet incidents are commonly ordered by arrival time, coarse severity, or product-specific heuristics that leave their relative priority unclear. We introduce Adaptive Incident Prioritization (AIP), the ranking algorithm behind Microsoft Defender Queue Assistant, which continuously prioritizes security incidents for analyst investigation. AIP adapts BM25-style ranking to a query-less, multi-tenant queue setting by representing each incident as a collection of normalized security components extracted from alerts and metadata. The model combines saturated local component frequency, global component rarity estimated across tenants, bounded domain-prior multipliers, and component-level explanations. Deployed across tens of thousands of customers, AIP performs near-real-time inference and refreshes incident scores with a median latency of five seconds. In an expert-reviewed evaluation across 1,000 customer organizations, AIP achieves 92.8% Precision@10. In post-launch telemetry across 473,000 organization-day queues, AIP increases alert-detail interaction by 5.8% and alert-detail view events by 17.5% relative to severity ordering, providing behavioral evidence that model-ranked queues concentrate analyst engagement. We also extend the Microsoft GUIDE dataset with, to our knowledge, the first public label source for SOC queue prioritization over real-world incidents. The extension covers 499 organization queues and 9,980 incidents with expert-derived priority labels, enabling the research community to develop, compare, and advance methods for incident prioritization. 

---
# How Do You Choose Your AI Component? An Interview Study of Secure AI Integration in Practice 

**Authors**: Mahzabin Tamanna, Elizabeth Lin, Sparsha Gowda, Laurie Williams, Dominik Wermke  

**Link**: [PDF](https://arxiv.org/pdf/2607.16660)  

**Abstract**: The increasing adoption of Large Language Models (LLMs) as AI components in modern software systems introduces distinct security risks to the software supply chain. While many considerations and safety mechanisms are in place for components of the traditional software supply chain, the recent rapid adoption of AI components and platforms has overlooked these hard learned lessons. Selecting and integrating AI models without clear guidance on how these choices affect system security may leave applications vulnerable to threats, such as malicious components, data leakage, and unintended behavior. The goal of this study is to understand practitioners' decision making process and security considerations in selecting and integrating AI components through an exploratory semi-structured interview study. Toward this goal, we conducted semistructured interviews with 22 software developers, architects, and AI practitioners across diverse organizations about how they integrate AI components into their software.
Our analysis finds that practitioners' model selection is predominantly driven by functional criteria, including performance, accuracy, cost, and specific features, e.g., tool calling or multimodal support, while security is rarely considered as an evaluation criterion. We observe a consistent lack of security concern throughout the AI component integration process, with established software supply chain lessons overlooked or ignored. The industry is repeating the historically costly mistakes of early software dependency management, prioritizing rapid reuse and availability over security and provenance. We distill our findings into actionable recommendations for AI adopters, model providers, and researchers, advocating for a proactive, security-by-design approach that integrates security evaluation into component selection and sustains it throughout the software development lifecycle. 

---
# Art Beyond Semantics: Sheaf-Informed Contrastive Learning for Multi-Relational Representations 

**Authors**: Ludovica Schaerf, Antonio Purificato, Piera Riccio, Fabrizio Silvestri, Noa Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2607.16321)  

**Abstract**: Understanding a painting is never a single act. Art historians may analyze the same work through concepts of style, iconography, or historical context, dimensions that are not interchangeable, and each carries distinct semantic relationships between the visual and the textual. Vision-Language Models (VLMs) like CLIP, which learn a single shared embedding space, collapse this richness into a single homogeneous alignment, thereby losing the multi-relational structure that defines art-historical reasoning. We introduce CANVAS (Contrastive Art-aware Network for Vision-Language Alignment with Sheaves), a framework for learning relation-aware multimodal representations inspired by sheaf theory. Each artwork is projected into multiple embeddings conditioned on the type of relation (i.e., the context), and a novel contrastive loss encodes contextual information during training, with no dependency on external data at inference. We evaluate on three newly introduced benchmarks of artworks for multi-relational art understanding: WikiArt+, derived from WikiArt and Wikipedia, HertzianaDP, from the Bibliotheca Hertziana collection, and SemArt+, refined from the SemArt dataset. In multimodal retrieval and art understanding, CANVAS outperforms the baselines, supporting the view that multi-relational alignment is not just theoretically motivated but also practically essential. 

---
# Discovery by Dreaming: Cross-Domain Recombination in Artificial Memory 

**Authors**: Oliver Zahn, James Evans, David Eagleman  

**Link**: [PDF](https://arxiv.org/pdf/2607.16256)  

**Abstract**: Dreams splice together people, places, and times that never met. Neuroscience suggests this recombination is not noise, but a function driving insight and creative discovery. This reframes memory consolidation: rather than merely defending against forgetting, its measurable value lies in recombining knowledge across experiences that have not yet co-occurred. We test this directly by isolating the recombinatory-replay mechanism and implementing it in two architecturally unrelated systems: a LoRA fine-tuning pipeline (DREAMS) and a symbolic engine replaying structured knowledge objects (SAPIENCE). Both systems converge on the same finding: cross-domain consolidation creates value, while within-domain rehearsal does not. The symbolic arm surfaces novel cross-domain connections at 85.7%, a +21 percentage point (pp) gain over baseline. The neural arm improves overall by +5.64 pp, but on subtasks explicitly requiring cross-domain transfer (like unseen math reasoning on GSM8K), gains reach +14.5 pp. This effect is a genuine property of the weights--not a prompt artifact--as prepending the same material in-context to a 671B-parameter model actually reverses the gain. We validate this prediction against documented discoveries across 50,000 real papers and state a falsifiable hippocampal-recording prediction to distinguish recombination from rehearsal. Ultimately, this principle is substrate-general, tracking real discovery at scale. Reading the literature teaches a model to recall what it has seen, but producing discovery requires a separate offline phase that recombines knowledge across domains--the computational analog of dreaming. Consolidation is not for remembering, but for discovering. 

---
