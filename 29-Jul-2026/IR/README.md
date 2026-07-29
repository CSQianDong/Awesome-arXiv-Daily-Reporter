# RecoReward: Recommender-Guided Multimodal Description Generation for Recommendation 

**Authors**: Guohong Mu, Yueyang Liu, Jiangxia Cao, Changxin Lao, Zijie Zhuang, Yuhui Zhang, Jiaqi Feng, Ruochen Yang, Shuang Yang, Zhaojie Liu, Qibin Hou  

**Link**: [PDF](https://arxiv.org/pdf/2607.25901)  

**Abstract**: Multimodal large language models (MLLMs) can convert multimodal item content into structured descriptions used as semantic features for recommendation. Conventional content-only generation, however, cannot use downstream user signals to determine which semantics should be emphasized. Recent user-conditioned methods incorporate these signals through user histories or profiles, but they require user information at inference and make generation user-dependent. In this paper, we introduce RecoReward, which instead uses behavior-derived rewards during training and preserves content-only inference. To instantiate this idea in live-stream recommendation, we treat historically engaged users as a proxy for future target users and use observational non-target users to estimate affinity shared broadly across users. The Recommender Affinity Score (RAS) contrasts these signals to provide user-selective feedback for reinforcement learning, allowing the learned policy to generate a single shared description without user inputs. In our offline benchmark, RecoReward-9B outperforms its Qwen3.5-9B baseline and all other evaluated models across seven recall metrics. Online A/B testing also shows performance gains. These results show that RecoReward trains the MLLM to produce item features that benefit downstream recommendation while retaining content-only serving. 

---
# Hypothesis-Driven Shelf Generation for Personalised Recommendation 

**Authors**: Aleksandr V. Petrov, Tarun Chillara, Matthew D. Moellman, Lucas de Haas, Yabai Song, Alina Susoykina, Melissa Crawford, Gabriel Negash, Erik Franco, Tasnim Rahman, Binal Jhaveri, Shubham Bansal, Hugues Bouchard, Roberto Mirizzi, Mounia Lalmas, Aloïs Gruson  

**Link**: [PDF](https://arxiv.org/pdf/2607.25823)  

**Abstract**: Modern recommendation interfaces organise content into shelves: themed rows such as "More of What You Like" or "New Releases for You." In production systems, these shelves are typically defined through hand-crafted templates coupled with dedicated retrieval logic. While effective for broad recommendation intents, this approach does not scale to the long tail of individual taste. We present a content-hypothesis-driven shelf generation system for Spotify Home that replaces fixed templates with natural-language hypotheses describing what a personalised shelf should contain. The system has four stages hypothesis generation, catalogue fulfilment, shelf alignment, and offline serving. This decomposition decouples shelf planning from catalogue fulfilment, supports independent optimisation of planning and retrieval, and enables both constrained generative retrieval over catalogue entities and distillation of frontier LLM behaviour into compact models. Our production pipeline combines hypothesis generation, generative retrieval, candidate selection and shelf alignment, offline LLM-as-a-judge evaluation, and precomputed serving. We describe the end-to-end architecture and evaluate it through offline analyses and an early online evaluation under uniform random exposure on Spotify Home. Results show that hypothesis-driven shelves substantially expand personalised recommendation supply with engagement that varies by content type and is competitive with strong existing shelves in some settings. 

---
# LLM-as-a-Judge for Evaluating System Responses in Conversational Music Recommendation 

**Authors**: Seungheon Doh, Bruno Sguerra, Sergio Oramas, Elena V. Epure, Juhan Nam  

**Link**: [PDF](https://arxiv.org/pdf/2607.25640)  

**Abstract**: Conversational Recommendation Systems (CRS) aim to achieve two primary objectives: recommending relevant items and generating natural language responses. While recommendation accuracy is effectively measured by established ranking metrics, the evaluation of response generation poses a more fundamental challenge. Although human evaluation remains the gold standard, its cost and scalability constraints have motivated the adoption of LLM-as-a-judge as a promising proxy, whose alignment with human judgment in the context of CRS remains an open question. In this paper, we present the first user study to empirically assess the reliability of LLM-as-a-judge for evaluating CRS responses. We sample 20 multi-turn music recommendation sessions and generate candidate system responses using four instruction-tuned LLMs, inducing variance in response quality across model scales. We collect $n{=}400$ ratings from 20 domain-expert annotators, who evaluate each response across two dimensions: Personalization Quality and Explanation Quality. Through bootstrapped correlation analysis, we find that LLM-based judges exhibit moderate positive alignment with human assessments and outperform all reference-based baselines. Furthermore, we analyze how judge performance varies according to model scale and conditioning information, providing practical guidance for deploying LLM-as-a-judge. 

---
# Beyond Self-Knowledge: Propagating Uncertainty Across Reasoning and Retrieval in LLMs 

**Authors**: Chandan Kumar Sah, Xiaoli Lian, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.25600)  

**Abstract**: Retrieval-augmented generation improves knowledge-intensive question answering, but indiscriminate retrieval can introduce irrelevant evidence and unnecessary computation. We investigate whether verbalized confidence from black-box language models can serve as an actionable signal for retrieval routing. Our method, BeyondUncertainty, first elicits a structured provisional answer and confidence estimate, then applies a model-specific threshold selected on held-out validation data and frozen before test evaluation. Low-confidence questions receive top-$5$ TF--IDF retrieval followed by a second answer call, whereas high-confidence questions return the provisional answer directly. We evaluate 27,000 policy instances across six QA benchmarks, three model families, and three retrieval policies. BeyondUncertainty achieves $0.483$ mean token-level F1, compared with $0.467$ for always retrieval and $0.401$ for no retrieval, while reducing retrieved passages by $20.4\%$ relative to always retrieval. When matched on the number of questions routed to retrieval within each dataset--model cell, it outperforms a post-hoc random allocation in 17 of 18 settings, with an average gain of 0.024 F1. Although poorly calibrated as an absolute probability, probe uncertainty modestly predicts question-level retrieval benefit (AUROC $=0.628$). However, the additional probe increases total token usage by $28.2\%$, revealing a trade-off between more selective evidence acquisition and end-to-end token efficiency. 

---
# MARS: Multi-Agent Re-ranking for Repeat-Order Food Delivery Recommendation 

**Authors**: Jiahao Tian, Zhenkai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.25420)  

**Abstract**: Large language models (LLMs) are increasingly used in recommender systems, but it is often unclear how much performance can be obtained from strong pre-trained backbones alone when they are placed inside a structured recommendation pipeline. In this paper, we present MARS, a modular multi-agent re-ranking framework for repeat-order food delivery recommendation. MARS serves as a controlled hybrid framework for studying how far pre-trained LLMs can go in this setting when combined with lightweight collaborative retrieval and contextual filtering. MARS performs coarse-to-fine recommendation in two stages: cuisine prediction followed by vendor ranking. The framework combines LightGCN-based global preference signals, Swing-based local peer evidence, geospatial filtering, and prompt-driven LLM reasoning over behavioral, temporal, and geographic context. We evaluate MARS on two real-world Delivery Hero benchmarks, DHRD-SE and DHRD-SG, and compare it against heuristic, sequential, graph-based, and food-delivery-specific baselines. We also provide detailed implementation and evaluation protocols, including prompting and decoding. Our study makes three contributions. First, it presents a modular multi-agent framework for repeat-order food delivery recommendation that integrates collaborative signals and LLM-based re-ranking in a transparent pipeline. Second, it shows that strong pre-trained backbones can already be competitive in repeat-order recommendation when paired with lightweight collaborative retrieval. Third, it establishes a reproducible evaluation setting for hybrid LLM recommenders in food delivery. 

---
# Sharpness-aware Model Merging with Salience Recovery for LLM-based Cross-Domain Sequential Recommendation 

**Authors**: Huwei Ji, Jiajie Su, Yuyuan Li, Xiaohua Feng, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.25366)  

**Abstract**: LLM-based Cross-Domain Sequential Recommendation (CDSR) leverages LLMs to enhance target performance via deep semantic reasoning, alleviating the dependency on overlapping users. Among LLM-based paradigms, model merging is particularly promising for multi-domain scenarios due to its superior scalability and flexibility in integrating diverse knowledge sources. However, our empirical investigations reveal two critical bottlenecks: (1) cross-domain knowledge conflict; and (2) performance saturation in multi-domain fusion. Our analysis attributes these phenomena to parameter-level misalignment and statistical homogenization during the merging process. To address these bottlenecks, we propose SharpRec, Sharpness-aware Model Merging with Salience Recovery for LLM-based CDSR, a framework designed to lift the performance upper bound of merged models. SharpRec incorporates two synergistic modules: Sharpness-aware Geometric Alignment to establish a stable geometric foundation for interference-free fusion; and Preference Salience Activation to effectively recover the distinctive features essential for bolstering target domain performance. Extensive experiments in both dual-domain and multi-domain scenarios demonstrate that SharpRec consistently outperforms state-of-the-art baselines. 

---
# The Case Against Generation for Retrieval: Discriminative Language Models as Effective Retrievers 

**Authors**: Zhe Xu, Prachi Agrawal, Kavosh Asadi, Tianyi Chen, Carl Hu, Justin Johnson, Wuwei Lan, Mingfu Liang, Xi Liu, Tik On Lui, Oladipo Ositelu, Sandeep Pandey, Ankit Peshin, Feng Qi, Anil Ramakrishna, Kaushik Rangadurai, Frank Shyu, Luke Simon, Yang Yang, Chiyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.25346)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful assets for recommender systems. However, deploying them as generative recommenders or zero-shot rankers at web-scale remains bottlenecked by prohibitive computational overhead and grounding challenges. In this paper, we revitalize the classic, highly efficient two-tower retrieval architecture by adapting LLMs as semantic representation backbones rather than generative engines. We introduce an LLM-native two-tower framework engineered for high-throughput, large-scale retrieval. Our architecture introduces several key innovations: a shared LLM encoder for joint user-item modeling, End-Of-Sentence (EOS) token pooling for compact sequence embedding, cross-dataset transfer learning, knowledge distillation from powerful cross-encoder teachers, and latent reasoning within the user tower. Extensive evaluation across three public benchmarks demonstrates that cross-encoder architecture outperforms current state-of-the-art (SoTA) models, while the efficient two-tower student achieves SoTA-comparable retrieval performance. Furthermore, experiments on internal large-scale production systems yield substantial topline retrieval improvements along with high resilience to model staleness and superior data scaling. Our findings demonstrate that when augmented with modern representation learning, the traditional two-tower paradigm remains an exceptionally competitive and practical solution for industrial retrieval systems. 

---
# Reward Guided Decoding for Generative Recommendation 

**Authors**: Ruochen Yang, Yusheng Huang, Youfeng Zheng, Shuang Wen, Liangliang Chen, Pengbo Xu, Xiaoyu Zhang, Shijun Wang, Shuang Yang, Zhaojie Liu, Lantao Hu, Wenwu Ou, Jiawei Sheng, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.25344)  

**Abstract**: Generative recommendation formulates recommendation task into an SID sequence autoregressive generation paradigm, but the decoding process is often dominated by generation likelihood. This may conflict with real-world business objectives, where high-value candidates can receive low generation probability and be pruned early during beam search. Existing reranking or training-time alignment methods either intervene too late or require costly model retraining when business preferences change. To this end, we propose \textbf{R}eward \textbf{G}uided \textbf{D}ecoding, named \textbf{RGD}, a controllable decoding framework for industrial value-oriented generative recommendation. We formulate value-guided decoding as a KL-regularized reward maximization problem, deriving a closed-form reward guided decoding distribution that principledly combines generation probability with reward signals. RGD treats the base generator as a reference policy and introduces a reward model as a test-time controller, injecting reward into each decoding step to reshape the search trajectory without retraining the generator. Extensive offline and online experiments demonstrate the effectiveness of our approach for aligning personalization and business value. RGD has been deployed on the Kuaishou platform, bringing consistent improvements in real-world recommendation scenarios. 

---
# SPARC: Sequence-aware Progressive Attribute Routing and Compression Framework for Generative Recommendation 

**Authors**: Chang Liu, Changfa Wu, Hui Qian, Binbin Cao, Jian Wu, Yuliang Yan, Han Zhu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.25339)  

**Abstract**: Generative recommendation tokenizes items as discrete Semantic IDs (SIDs) and autoregressively generates target items from users' historical SID sequences. Although existing SIDs incorporate multimodal and structured information, they are typically statically assigned and independent of the current interaction context. In industrial scenarios, each behavior also contains heterogeneous attributes, such as category, brand, price, behavior type, and timestamp. Fully expanding these features greatly increases the input length, while directly compressing them into a single representation may prematurely discard context-relevant information.
We propose \textbf{SPARC}, \uline{\textbf{S}}equence-aware \uline{\textbf{P}}rogressive \uline{\textbf{A}}ttribute \uline{\textbf{R}}outing and \uline{\textbf{C}}ompression Framework for Generative recommendation. SPARC first models the sequential dependencies of each field type to obtain context-aware field representations. It then routes the original, contextual, and identity representations of different fields into multiple slots to preserve complementary information under a fixed capacity. Finally, lightweight cross-item interaction integrates the intermediate tokens and compresses each historical item into a single token. Following the principle of contextualizing before compression, SPARC enriches user-history representations without increasing the input length of the generative backbone.
Experiments on industrial Taobao and public Amazon datasets demonstrate that SPARC outperforms strong conventional and generative baselines. Further comparisons with static compression variants show that the improvement of SPARC comes from context-conditioned information retention rather than merely increasing the expressiveness of the compression module. 

---
# Grevo: A Unified Generative Recommendation Framework with Evolutionary Item Indexing 

**Authors**: Huanjie Wang, Liwei Guan, Zekai Sun, Hongwei Zhang, Honghui Bao  

**Link**: [PDF](https://arxiv.org/pdf/2607.25329)  

**Abstract**: Generative recommendation has recently emerged as a promising paradigm that reformulates retrieval as autoregressive generation over semantic identifiers (SIDs), achieving strong performance and drawing increasing attention as an alternative to matching. Despite this progress, SIDs are typically frozen by a content-based tokenizer before the recommender is trained, leaving a persistent gap between what best reconstructs an item's content and what a recommender can predict from user behavior. Recent end-to-end methods close this gap by jointly training the tokenizer and the recommender, but coupling the two destabilizes the identifier space and requires a second learnable model, alignment losses, and usually a delicate alternating-optimization schedule. To address this issue, we propose Grevo, a unified Generative recommendation framework with evolutionary item indexing, which treats the SID assignment itself as an evolvable discrete variable that adapts to behavioral feedback rather than as a tokenizer to be trained. Grevo builds on a single multitask recommender that unifies a behavioral SID generation task and a semantic SID grounding task, letting the recommender absorb the tokenizer's role. Through evolutionary item indexing, Grevo then uses the trained recommender itself as a posterior evaluator to reassign a budgeted set of high-risk identifiers under a fixed vocabulary and length. Together, these components turn index construction into a stable, feedback-driven search that requires no second learnable model, no alignment losses, and no alternating-optimization schedule. Extensive experiments on multiple real-world datasets demonstrate that Grevo consistently outperforms state-of-the-art generative recommendation methods. 

---
# Structure-aware Relative Policy Optimization for Ranking 

**Authors**: Yiteng Tu, Weihang Su, Zitao Su, Yiqun Liu, Min Zhang, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2607.25268)  

**Abstract**: Ranking is a fundamental component of modern information access systems. Reinforcement learning (RL) provides a flexible framework for directly optimizing coarse-grained feedback and system-level objectives defined over the complete ranking list. However, existing RL-based ranking methods typically treat each sampled permutation as an atomic output and evaluate it primarily through a scalar reward, overlooking the structural relationships among different ranking lists. Consequently, permutations with similar rewards but substantially different permutation patterns may receive comparable optimization signals, potentially leading to inaccurate credit assignment and overly aggressive policy updates. To address this limitation, we propose SRPO, a \textbf{S}tructure-aware \textbf{R}elative \textbf{P}olicy \textbf{O}ptimization framework for listwise ranking. SRPO measures the discrepancy between sampled permutations using a top-weighted Kendall-tau distance and normalizes their pairwise reward differences by the corresponding distances. It quantifies the reward improvement per unit of ranking change, thereby emphasizing efficient local refinements, particularly those involving top-ranked positions. Experimental results across two ranking scenarios demonstrate that explicitly modeling permutation-level differences improves the effectiveness and stability of listwise ranking, with particularly favorable performance in limited-feedback and complex list-level optimization settings. 

---
# TopoGR: Revealing and Preserving Latent Structure of Semantic ID in Generative Recommendation 

**Authors**: Ziyu Zheng, Zhengshun Du, Yaming Yang, Bin Tong, Guan Wang, Meng Yan, Ziyu Guan, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2607.25216)  

**Abstract**: Semantic ID-based generative recommendation tokenizes each item into a sequence of discrete semantic IDs and predicts the next item by generating semantic IDs. However, existing methods typically regard SIDs as independent discrete symbols, while often overlooking the topology of the learned semantic ID space. We identify a structural mismatch between tokenization and generation: the tokenizer learns a structured code space with semantic neighborhood relations, whereas the generator consumes semantic ID tokens as independent categorical symbols. Consequently, item relatedness is reduced to exact semantic ID overlap, making it difficult to identify semantically similar items whose semantic IDs do not overlap. To address this issue, we propose TopoGR, a topology-preserving generative recommendation framework based on Bit-decomposable Semantic ID(Binary SID). Each Binary SID is learned in a bit-decomposable form and can be deterministically converted to a standard integer SID, while exposing an explicit Hamming geometry. TopoGR exploits this topology at three stages: binary SID features preserve Hamming proximity at the input layer; Hamming soft targets inject topology-aware supervision; and Hamming-consistent reranking aligns candidate items with the predicted binary prototype during inference. We further verify that the Hamming topology can capture item relatedness beyond exact SID matching. Experiments on four benchmark datasets show that TopoGR consistently outperforms existing state-of-the-art baselines in recommendation performance. 

---
# VaLiDRec: Variable-Length LLM-Aligned Semantic IDs for Generative Recommendation 

**Authors**: Shutong Qiao, Wei Yuan, Tong Chen, Hao Wang, Quoc Viet Hung Nguyen, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2607.25209)  

**Abstract**: Generative recommendation commonly represents items using fixed-length semantic identifiers (SIDs) constructed through clustering and quantization. However, these artificial codes may overcompress item semantics, remain misaligned with pretrained LLM vocabularies, and require costly autoregressive decoding. In light of this, we propose VaLiDRec, a generative recommendation framework based on variable-length, LLM-aligned semantic identifiers. VaLiDRec constructs SIDs directly from informative native LLM vocabulary tokens via token importance estimation, semantic-quality-aware pruning, and collision-aware refinement, allowing identifier lengths to adapt to item semantic complexity. To model user preferences, VaLiDRec incorporates graph-aware soft prompts and reformulates recommendation as token-set prediction with token-level item scoring, eliminating autoregressive SID generation and beam search. Experiments on four real-world datasets show that VaLiDRec consistently outperforms strong sequential and generative recommendation baselines across all evaluation metrics. It further achieves superior zero-shot item cold-start performance and 87.49$\times$ faster inference than LC-Rec. These results demonstrate that LLM-native variable-length semantic identifiers provide a more expressive and efficient paradigm for generative recommendation. 

---
# Bekko Embedding: Parameter-Efficient Multilingual Retrieval with Ultra-Compact Encoders 

**Authors**: Yuichi Tateno  

**Link**: [PDF](https://arxiv.org/pdf/2607.25180)  

**Abstract**: How small can a competitive multilingual retrieval model be? We present Bekko Embedding: its smallest model, bekko-embedding-v1-a8m, has just under 8M Active Parameters (AP) -- the non-embedding parameters that dominate inference compute -- yet on official MMTEB Multilingual v2 Retrieval (nDCG@10) it scores 56.2, above the multilingual-e5 family and BGE-M3 (40x the AP) in our comparison. The higher-quality bekko-embedding-v1-a25m (just under 25M AP) reaches 57.5, on par with gte-multilingual-base, and Multilingual NanoBEIR (14 languages) confirms the trend. Both models handle inputs of up to 8192 tokens, and on long-input retrieval (NanoLongEmbed) a25m is the strongest dense model in our comparison. The recipe is deliberately simple. We prune the 22-layer multilingual encoder mmBERT-small to 4 / 13 layers and train the pruned models in two stages -- large-scale contrastive learning on about 1.1 billion multilingual pairs from our public corpus, followed by hard-negative fine-tuning with 8192-token long-document negatives -- with a masked contrastive loss whose direction depends on pair type, plus the Matryoshka objective. No teacher distillation is used, and all training completes on a single GPU in about 3 days for a8m. Small AP pays off directly in speed: among the compared models measured under identical conditions, a8m is the fastest on both CPU and GPU -- 1.6x multilingual-e5-small on x86 CPU -- and the fastest on a Raspberry Pi 5. The 384-dimensional output (truncatable to 256/128/64) keeps similarity search and indexing cheap, and row-wise int8 quantization of the vocabulary embedding shrinks the a8m ONNX / OpenVINO build to 124 MiB, which runs in the browser via this http URL. To support reproducible research, we release the model weights, the complete stage-1 corpus, and the independently mined stage-2 hard negatives. 

---
# HiEviDR-Bench: A Benchmark for Hierarchical Evidence Aggregation in Deep Research 

**Authors**: Yubo Sun, Chunyi Peng, Yukun Yan, Zhenghao Liu, Sen Mei, Bangrui Xu, Xuanhe Zhou, Chi Chen, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.25151)  

**Abstract**: Deep research requires models to retrieve, connect, and synthesize evidence from large-scale heterogeneous sources to answer complex queries and produce analytical reports. Existing benchmarks mainly evaluate final outcomes, such as answer correctness, report quality, or citation alignment, while providing limited visibility into whether evidence is correctly selected, linked, and aggregated into supported claims and conclusions. To address this gap, we introduce HiEviDR-Bench, a benchmark for evaluating Hierarchical Evidence Aggregation in Deep Research. HiEviDR-Bench covers open-domain and academic-domain settings under both text-only and multimodal conditions, and represents each instance with an explicit evidence graph that captures evidence selection, cross-source linking, and aggregation from evidence to intermediate claims and final conclusions. Based on this formulation, we develop a traceability-oriented evaluation framework with five dimensions: report quality, evidence traceability, citation accuracy, claim verification, and answer correctness, together with a progressive gating mechanism for fine-grained error localization. HiEviDR-Bench contains 2,000 human-validated questions with evidence graphs across multiple difficulty levels. Experiments on 16 representative multimodal large language models show that, although many systems achieve strong report quality, their performance drops markedly on citation accuracy, claim construction, and answer correctness. Further analysis shows that the main bottlenecks lie in evidence identification and intermediate claim construction, revealing that strong surface-level report quality does not necessarily imply grounded multi-stage reasoning on our benchmark. 

---
# Memory Layer: Train the In-Model Cache for Recommendation Models 

**Authors**: Liangyuan Na, Gufan Yin, Yixin Bao, Xianjie Chen, Justin Lin, Ziheng huang, Xinyuan Zhang, Wen Zhang, Hao Lin, Xiaoheng Mao, Shuo Tang, Min Yu, Lei Chen, Chao yang, Ziliang Zhao, Mengjiao Zhou, Zheng Qi, Dmitry Barablin, Chuo-Yun Yang, Kaustubh Vartak, Tingting Zhang, Arun Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2607.25110)  

**Abstract**: Early ranking stages in recommendation systems precompute item embeddings and cache them in-model for scoring within strict latency constraints. Because this cache exists only at serving time, outside the training loop, training and serving use different item representations, a structural discrepancy that limits quality and adds operational fragility. We show that co-designing the training and serving paths removes this representation discrepancy at its source. We introduce the memory layer, an in-model key-value embedding cache co-trained with the model: the item tower writes embeddings during training and the model reads them at serving, one source of truth for item representations by construction. Always-on embeddings cover items not yet cached, so every item receives a prediction, and the design consolidates three separate trainer-to-predictor update paths into a single self-contained pipeline. Deployed in production on Instagram Reels, the memory layer raises prediction coverage from 96% to 100%, improves embedding freshness from $O(5\text{ min})$ to $O(20\text{ s})$, and narrows the training-serving Normalized Entropy (NE) gap by up to 86%, yielding over $2\times$ recall for the freshest content and a 5-6% cold start engagement lift. Because embeddings are produced during training, the system needs no separate bulk-evaluation or publish-time recomputation, cutting training-and-publish computational cost by 30% at neutral serving computational cost. 

---
# On the Convergent Validity of Offline Evaluation Designs for Recommender Systems 

**Authors**: Sushobhan Parajuli, Samira Vaez Barenji, Michael D. Ekstrand  

**Link**: [PDF](https://arxiv.org/pdf/2607.25097)  

**Abstract**: Offline evaluation on historical interaction logs is the most common evaluation methodology for recommender systems. However, such evaluations depend on sparse, incomplete, or biased data, which raises concerns about whether commonly used evaluation setups reliably reflect true user preferences. In this work, we study how offline evaluation design choices affect the validity of recommender system comparisons. We evaluate a set of recommendation models across several evaluation setups that vary key factors such as data filtering thresholds and candidate set construction. To assess the validity of these configurations, we measure the correlation between model rankings obtained from conventional train-test splits on sparse interaction data and rankings from evaluations based on dense ground-truth user feedback. We use this agreement as an indication of their validity with respect to true user preferences. Our results show that the validity of sparse evaluation depends on the dataset and the specific dense evaluation targets, and that there is no uniformly best offline evaluation design. 

---
# ScoreShield: Differentially Private Release of Similarity Scores 

**Authors**: Behrooz Razeghi, Parsa Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2607.25041)  

**Abstract**: A growing number of applications, such as biometrics and retrieval-augmented generation (RAG), rely on cosine similarity scores computed between vector embeddings of text, images, or audio. These systems return similarity scores through their APIs for ranking and verification. However, such releases can leak information about individual records and enable membership inference attacks. While differential privacy (DP) provides a principled metric for quantifying attack risks, naïve application of DP mechanisms---such as adding i.i.d. Gaussian noise to vector entries---leads to excessive distortion (i.e., low utility) at a given privacy constraint that scales poorly with the number of released scores. We propose \textsc{ScoreShield}, a perturb-then-project mechanism that adds Gaussian noise calibrated to global sensitivity of the chosen score release regime and then projects the result onto the feasibility set of valid cosine objects. \textsc{ScoreShield} satisfies $(\varepsilon,\delta)$-DP for releasing similarity score vectors and Gram matrices. We provide utility guarantees for the exact Frobenius metric projection used in the risk analysis, and prove convergence to feasibility for the practical averaged alternating-projection solver used for large-scale Gram releases. For full pairwise cosine Gram release under record-level replacement adjacency, the exact-projection bound improves the $n$-dependence of squared Frobenius risk from $\Theta(n^3)$ for the naïve Gaussian baseline to $\mathcal{O}(n^2)$ for fixed privacy parameters, with sharper local bounds at low-rank Grams. We evaluate the mechanism across RAG, face recognition, semantic retrieval, image similarity, and recommender-system tasks. 

---
# Grounded in Consensus, In Step With Emerging Science: A Consensus-Anchored Multi-Corpus Clinical Chatbot for Long COVID 

**Authors**: Yining Wu, Philip DiGiacomo, Ying Ding, William Brode  

**Link**: [PDF](https://arxiv.org/pdf/2607.25038)  

**Abstract**: Long COVID (LC) poses a challenge for clinical decision support because relevant evidence is distributed across sources with different update cycles, evidentiary roles, and levels of clinical maturity. We present a clinician-facing chatbot that organizes four sources within a retrieval-augmented workflow: expert-curated consensus guidance, current PubMed literature, registered interventional trials, and evidence from living systematic reviews. Consensus guidance is always included to frame responses, while the remaining sources are retrieved in parallel when selected by the user. In an exploratory automated evaluation on 50 clinician-facing questions, our chatbot showed comparable mean ratings to OpenEvidence, with numerically higher scores and lower score variability in an LLM-judged comparison. 

---
# Agent Retrieval Bench: Evaluating Repository Context Retrieval for Coding Agents 

**Authors**: Bowen Qin, Yi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2607.24882)  

**Abstract**: Modern coding agents are usually evaluated by whether they eventually produce a correct patch, but patch generation depends on an earlier context-acquisition stage: finding the repository files needed for the task. We introduce Agent Retrieval Bench, a file-level benchmark for this upstream retrieval problem. Samples are built from real coding-workflow signals and evaluated against frozen base-commit repositories, with relevance defined by what an agent needs next rather than direct query-file semantic similarity. The benchmark covers four positive-retrieval tasks: code2test, comment2context, trace2code, and edit2ripple; a fifth subset evaluates selective retrieval using natural evidence-backed no-gold cases and counterfactual wrong-repository controls. Agent Retrieval Bench contains 427 samples across 25 repositories: 345 positive examples, 50 natural no-gold examples, and 32 counterfactual controls. The corpus includes 308 base-commit snapshots, 392,000 files, and 7.9 million chunks. We evaluate lexical retrieval, RepoMap, open-source embeddings, selective abstention, and logged agent context selection. No single retrieval family dominates: Qwen3-Embedding-4B has the best sample-weighted MRR on positive samples, Qwen3-Embedding-8B the best Recall@20, and RepoMap the best budgeted context yield at 8K tokens, with task-level winners differing substantially. Selective thresholds calibrated with counterfactual controls do not improve selective success on natural no-gold cases, revealing a calibration gap. Logged trajectories also miss every gold file on 27-35 percent of samples. A controlled seed-intervention pilot finds that retrieval-derived initial context yields higher file F1 with less post-seed exploration than random non-gold context, while oracle gold context shows substantial remaining headroom. 

---
# Ranked by Position: Order Sensitivity as an Exploitable Attack Surface in LLM Listwise Recommenders 

**Authors**: Ge Zhang, Jingru Cheng, Huiyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.24869)  

**Abstract**: Large language models (LLMs) used as listwise rerankers in recommendation systems suffer from position bias when serializing candidate sets into prompts. We show this order sensitivity creates an exploitable attack surface: an attacker can promote a label-0 target into the top-$k$ solely by reordering candidates, without changing item content, labels, or model parameters. We introduce $\mathrm{promo}@k$ to quantify this vulnerability, measuring the fraction of label-0 targets that can be elevated into top-$k$ rankings via permutation. Evaluating across three domains (MovieLens, Amazon Books, and Amazon Fashion), $\mathrm{promo}@5$ reaches up to 0.57 at an attack budget of $R$ = 50 orderings. Furthermore, ordinary permutation stability predicts vulnerability without running the attack. While a bidirectional T5 encoder scorer reduces exposure, permutation-consistency regularization and architectural invariance effectively mitigate it. Pointwise scoring avoids the bias issue but degrades ranking quality. These results demonstrate that input candidate order in listwise LLM reranking is a security-relevant attack vector. Code and data are available at this https URL. 

---
# Tokens are All You Need: Dual-purpose Semantic IDs for Achieving LLM-Level I/O Efficiency in recommendation systems 

**Authors**: Baolei Li, Yiping Yuan, Yilin Zheng, Likang Yin, Ling Liu, Fabio Soldo, Romer Rosales, Xinyang Yi, Lichan Hong  

**Link**: [PDF](https://arxiv.org/pdf/2607.24865)  

**Abstract**: Large-scale recommendation systems face "Memory Wall" bottlenecks due to massive, dense embedding tables. While generative retrieval uses discrete tokens for IDs, high-dimensional context still relies on inefficient dense formats. Inspired by computer vision data compression, we propose Dual-purpose Semantic IDs to achieve LLM-level I/O efficiency. Our methodology uses hierarchical quantization to condense continuous embeddings into discrete Semantic IDs performing two concurrent roles: (1) Collaborative Identity: modeling user-item interactions via learnable embedding table; and (2) Content Reconstruction: using a lightweight Semantic Decoder for on-the-fly embedding approximation. This approach replaces massive vector storage with on-demand reconstruction, reducing system overhead and data footprints. We demonstrate the efficacy of our framework through offline evaluations and successful online deployment in production-scale ranking and retrieval systems at a major video sharing platform, showing that discrete tokens are indeed all you need for highly efficient, content-rich recommendation. 

---
# KuaiLive-M3: A Multi-Modal, Multi-Domain, and Multi-Feedback Dataset for Live Streaming Recommendation 

**Authors**: Ke Guo, Changle Qu, Jiayaqi Cheng, Xiao Zhang, Shijun Wang, Xiaoyu Zhang, Xueliang Wang, Le Zhang, Lantao Hu, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.24862)  

**Abstract**: Existing public live streaming datasets suffer from three major limitations: they provide limited access to temporally evolving multimodal live content, overlook users' cross-domain interactions between short videos and live streams, and contain only implicit behavioral signals without explicit feedback that captures users' perceived content quality and satisfaction. These limitations prevent existing benchmarks from faithfully reflecting real-world live streaming scenarios and hinder comprehensive research on live streaming recommendation. To address these limitations, we introduce KuaiLive-M3, a multi-modal, multi-domain, and multi-feedback dataset for live streaming recommendation, collected from Kuaishou, a leading live streaming and short video platform in China. KuaiLive-M3 covers 21,938 users and contains 35 million live streaming interactions and 111 million short video interactions, with fine-grained timestamps and diverse user behaviors. It further provides approximately 88 million timestamped segment-level multi-modal embeddings that capture the temporal evolution of live streaming content, as well as 25,403 questionnaire-based feedback records that bridge implicit user behaviors and explicit user preferences. Based on these unique signals, we establish benchmarks for cross-domain recommendation, live stream highlight prediction, and questionnaire-enhanced recommendation. Extensive experiments with representative baselines demonstrate that KuaiLive-M3 provides a challenging and realistic benchmark for future live streaming recommendation research. The results further highlight the importance of modeling temporally evolving content, transferring user preferences across domains, and bridging the gap between implicit behaviors and explicit user feedback. The dataset and benchmark code are publicly available at this https URL. 

---
# HVM-GraphRAG: Holistic-View Multimodal Graph Retrieval-Augmented Generation on Complex Document 

**Authors**: Xin He, Yili Wang, Wenqi Fan, Qing Li, Qinggang Zhang, Yi Chang, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.24861)  

**Abstract**: Question answering (QA) over complex documents requires models to retrieve and integrate evidence distributed across distant document regions and modalities. Multimodal GraphRAG provides a promising direction by organizing document evidence with graph structures. However, existing methods often suffer from unreliable cross-modal evidence indexing and expensive graph traversal. To address these issues, we propose HVM-GraphRAG, a holistic-view multimodal GraphRAG framework on complex document. HVM-GraphRAG uses a holistic view to guide graph construction, thereby reducing noisy and conflicting graph updates and building reliable indices between concept-level graph nodes and supporting multimodal chunks. During retrieval, HVM-GraphRAG searches over a compact concept-level graph and directly accesses supporting evidence through the constructed index, avoiding costly traversal over dense entity-level graphs. After obtaining the retrieved evidence, HVM-GraphRAG further reorganizes chunks into modality-specific groups, enabling the answering model to better integrate heterogeneous evidence. Experiments on three datasets show that HVM-GraphRAG achieves the best answer performance in most evaluated settings while substantially improving online retrieval efficiency over representative graph-based baselines. 

---
# SearchArt: Training Long-Horizon Search Agent with Scalable Synthetic and Verified Task 

**Authors**: Lang Mei, Xiaohan Yu, Chong Chen, Liyan Liu, Xiangnan Chen, Jinchao Ma, Chao Feng, Li Huang, Siyu Mo, Sichen Kang, Yunkun Xu, Zhihan Yang, Zhujun Xue, Jingren Zhang, Qing He, Yingdi Huang, Hao Jiang, Ziao Ma, Zewei Pan, Minhao Sun, Zhuo Tao, Jinzhao Xiao, Gangtao Xin, Huanyao Zhang, Wenjian Zhang, Jiangshan Zhang, Guojie Zhu, Jiaxin Mao, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.24850)  

**Abstract**: Recent advances in large language models (LLMs) have enabled search agents to autonomously tackle complex tasks across extended search and reasoning horizons. However, training effective search agents remains challenging due to the lack of scalable and long-horizon tasks, and the difficulty of evaluating and correcting intermediate reasoning and tool-use behaviors. We introduce SearchArt, a scalable framework for training long-horizon search agents through verification-driven task synthesis and a multi-stage post-training pipeline. SearchArt constructs large-scale datasets for complex search-, research- and user-oriented tasks by synthesizing diverse information-seeking QA pairs and corresponding search trajectories from web documents and automatically generated evidence graphs. To ensure the reliability of the synthesized data, we design a verification pipeline that jointly evaluates QA consistency, trajectory quality, and the relevance of retrieved evidence. The verified trajectories are subsequently used in a multi-stage training process comprising supervised fine-tuning and reinforcement learning-based policy optimization. Search agents trained with SearchArt exhibit adaptive search planning, iterative evidence aggregation, and complex reasoning over extended interaction horizons. Experimental results demonstrate that, with only (Qwen3.5-) 27B parameters, SearchArt scores 74.39 on BrowseComp-ZH, 70.06 on BrowseComp, and 52.55 on Deepresearch-bench, matching or surpassing frontier closed-source agents on both deepsearch and deepresearch benchmarks. 

---
# Two Views, One Voice: Evidence-Grounded Conversational Music Recommendation 

**Authors**: Sungwook Yoo, Sewook Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2607.24846)  

**Abstract**: Traditional conversational recommenders entangle retrieval and response generation within a single text interface, so exact entity cues fade as the dialogue's intent evolves, which compromises explanation credibility. We address this within the ACM RecSys Challenge 2026, which mandates both top-20 ranking and evidence-grounded response generation. This paper presents the third-place solution by team "swyoo" for the Blind-B industry track. We decouple retrieval and response into separate pipelines connected strictly via ranked tracks and metadata. Retrieval combines a hybrid lexical-dense pool for exact matching with a task-adapted pool driven by fine-tuned Qwen 8B adapters. Candidates are calibrated via LightGBM, then routed to an evidence-grounded propose-assign-select (PAS) framework to structure responses. This system also ranked second on the explanation-quality leaderboard in the final blind evaluation. Our findings demonstrate that: (i) isolating retrieval and response preserves both catalog cues and fluid intent; (ii) structuring generation via explicit evidence assignment is key to this near-best-in-class explanation reliability. 

---
# REPREC: Representation Driven Parameter-Efficient Recommendation System 

**Authors**: Harshini Kavuru, Dwipam Katariya, Giri Iyengar, Pranab Mohanty, Kalanand Mishra, Kalanand Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2607.24845)  

**Abstract**: Large language models (LLMs) have been applied to sequential recommendation by formulating it as a natural language task. Previous work has improved personalization by incorporating collaborative and sequential signals through input conditioning or LLM fine-tuning. However, existing approaches often rely on one or more of the following: LLM fine-tuning, additional architectural modules, representation distillation, or item-level conditioning over long interaction histories, increasing training complexity and deployment cost. We propose REPREC, a lightweight framework that reformulates LLM-based sequential recommendation through lightweight user representation alignment. REPREC maps a fixed-size user embedding from a frozen sequential encoder into a small set of learned soft tokens through a lightweight MLP injector that conditions a frozen LLM, leaving both pretrained backbones unchanged while training only the injector. We conducted exhaustive experiments on multiple benchmark datasets and demonstrate that REPREC consistently outperforms LoRA while remaining compatible with different pretrained sequential encoders and LLM backbones, enabling a modular and production-friendly recommendation pipeline without modifying either pretrained component. The gains are particularly pronounced for casual and core users across all datasets, highlighting REPREC's effectiveness in low-data regimes. Finally, when trained on short prompt histories and evaluated with longer contexts, REPREC maintains 85-100% of LoRA's performance while reducing per-epoch training time by an average of 1.51X, demonstrating an effective balance between recommendation quality and computational efficiency for production deployment. The code is available at this https URL 

---
# MedJudgeRAG: Option-Wise Evidence Judgment with Dynamic Knowledge Graphs for Medical MCQA 

**Authors**: Seongwon Seo, Seung Hwan Cho, Young-Min Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.24838)  

**Abstract**: In medical multiple-choice question answering (MCQA), Retrieval-Augmented Generation (RAG) can supplement the domain knowledge of language models (LMs). However, since vanilla RAG indiscriminately utilizes retrieved documents, it can degrade LM performance. To address this, we propose MedJudgeRAG. Our framework represents retrieved documents as a dynamic knowledge graph (KG) composed of entities and relations. For each option, the model judges an evidence verdict from the retrieved documents and the KG. Based on the verdict combination, the model determines a knowledge utilization strategy to reason toward the final answer. These capabilities are trained via supervised fine-tuning using structured reasoning traces generated by a teacher LM. The training employs a weighted cross-entropy loss that differentially weights the KG and reasoning segments. Experiments on two medical MCQA benchmarks demonstrate that MedJudgeRAG consistently outperforms both vanilla RAG and parametric baselines. Furthermore, ablation analysis reveals that the dynamic KG is more effective as graph-conditioned supervision at training time than as an explicit output at inference time. Our code is available at this https URL, and the generated reasoning traces are released at this https URL. 

---
# Improving Rare Medication Recommendation with Counterfactual Data Augmentation and Large Language Models 

**Authors**: Shinhwan Kang, Soo Yong Lee, Jaewon Kim, Kijung Shin, Buru Chang  

**Link**: [PDF](https://arxiv.org/pdf/2607.24829)  

**Abstract**: AI-based medication recommendation systems have attracted substantial attention due to their potential to enhance patient safety and therapeutic outcomes. Despite the clinical importance of accurately recommending rarely prescribed medications (rare-meds), we observe that most existing methods show significantly lower predictive performance for rare-meds. We attribute this issue to two intrinsic limitations: (a) the inherent scarcity of data for rare-meds and (b) limited consideration of co-recommended medications. To address these limitations, we propose GenRxR, a novel framework based on large language models (LLMs). GenRxR leverages the medical knowledge and clinical reasoning capability of LLMs to generate counterfactual medical data, mitigating the data scarcity issue for rare-meds. It also integrates an LLM into the medication recommendation process to model relationships among co-recommended medications. To further enhance the clinical reasoning, we introduce an instruction tuning step that aligns the LLM's capability with the recommendation task, enabling better handling of clinical context, including rare-meds cases. In our experiments, we show that GenRxR outperforms 14 (including 5 LLM-based) baselines in most cases. Specifically, it achieves up to 30.9% higher predictive performance for rare-meds than the strongest baseline. 

---
# Aethel: A Reproducible Graph-Retrieval Framework for Multi-Hop Financial Diligence 

**Authors**: Krish Sapru  

**Link**: [PDF](https://arxiv.org/pdf/2607.24826)  

**Abstract**: Secondary private equity transactions require rapid synthesis of fragmented, unstructured financial disclosures, where critical metrics and their entity anchors are distributed across disjoint documents with limited lexical overlap. We present Aethel, a reproducible framework that combines bipartite Personalized PageRank graph retrieval with a coreference-aware Bipartite Coreference Teleportation layer and an orchestrated specialist-agent architecture. Aethel models corpora as entity-passage graphs and propagates relevance through explicit relational paths to support multi-hop financial diligence. We evaluate the retrieval layer on 200-question samples from the MuSiQue and 2WikiMultiHopQA validation sets, comparing sparse lexical, dense bi-encoder, vanilla graph, and coreference-aware graph retrieval. Aethel achieves HR@5 of 100.0% on 2WikiMultiHopQA and 88.5% on MuSiQue, improving coverage over vanilla graph retrieval while trading off top-rank precision. We also evaluate retrieval over a 4,123-chunk corpus of financial disclosures and find that graph retrieval outperforms dense retrieval on multi-hop recall but does not surpass a strong BM25 baseline at open-corpus scale. The results show that graph-based retrieval offers interpretable multi-hop evidence paths and degrades more gracefully than dense retrieval as corpus size grows, while also demonstrating that its advantage depends strongly on corpus scale and entity-index quality. Code and evaluation artifacts are released for reproducibility. 

---
# Retrieval-based and Fine-tuned LLM Approaches for Industrial Asset Health Monitoring and Decision Support 

**Authors**: Seshu Kumar Damarla, Xiuli Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2607.24824)  

**Abstract**: Industrial plants run many important machines such as pumps, turbines, and compressors. Although engineers can use their experience to identify and diagnose machine problems, transferring this reasoning ability to computer systems remains difficult. This work studies how well a retrieval-only method and an open-source large language model (LLM) perform failure-sensor diagnostic reasoning using the FailureSensorIQ benchmark, a multiple-choice question-answering task introduced by IBM Research. In the retrieval-only approach, each answer option is converted into an option-level query and scored using similar correct and incorrect records from the training data. TF-IDF, BM25, semantic search, and hybrid search are tested and compared. In the LLM-based approach, the Qwen2.5-7B-Instruct model is evaluated using zero-shot prompting, few-shot prompting, and QLoRA fine-tuning. The results show that semantic search and hybrid search perform better than pure keyword-matching techniques, indicating that meaning-based similarity is more important for industrial failure-sensor reasoning. Among the LLM-based methods, the fine-tuned model achieves the best performance and substantially improves over zero-shot and few-shot prompting. Error analysis shows that performance decreases as the number of answer options increases. Robustness analysis also shows that all methods are sensitive to option shuffling, changed labels, paraphrasing, and additional distractors. 

---
# A Position Paper on Recommender Systems in the Era of Autonomous Agents 

**Authors**: Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.24822)  

**Abstract**: For decades, recommender systems have been optimized to serve human users. However, the rapid deployment of autonomous agents introduces a paradigm shift: recommendation consumers are predicted to be increasingly a mixture of humans and authorized agents acting on their behalf. This position paper reviews insights from prior human-centric RecSys research and outlines the transition to this hybrid environment. In our discussion, we treat agents as independent, user-aligned assistants that act as end-users of recommender systems, rather than platform-built components. We characterize the resulting tripartite interactions among humans, agents, and platforms, highlighting the dynamics that can arise across these relationships. This shift presents new opportunities for RecSys research, while introducing unique challenges for evaluation, alignment, and system design. 

---
# Dual-Level Atomic and Coordination Geometry Learning for Crystal Property Prediction Using Graph Neural Networks 

**Authors**: Sanjay Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2607.24818)  

**Abstract**: Accurate prediction of crystal properties remains a key challenge in computational materials science. While graph neural networks (GNNs) such as CGCNN, MEGNet, ALIGNN, and SchNet have shown strong performance, they primarily represent crystals at the atomic level and implicitly learn local chemical environments through message passing. However, many material properties are governed by coordination polyhedra, the fundamental structural units formed by atoms and their neighboring atoms. To address this limitation, we propose the Coordination Polyhedron Graph Network (CPGN), a multi-scale GNN that jointly learns atomic, bond, and coordination-polyhedron representations. CPGN constructs three coupled graphs: an atom graph encoding elemental and bonding information, a line graph capturing angular interactions, and a coordination polyhedron graph describing Voronoi-derived local environments through corner-, edge-, and face-sharing relationships. Physically meaningful geometric descriptors are incorporated for each polyhedron, while an interleaved message-passing mechanism with bidirectional cross-attention enables effective information exchange across structural levels. Extensive evaluations on the Materials Project, JARVIS-DFT, and QM9 benchmark datasets demonstrate that CPGN outperforms existing state-of-the-art GNN models. It achieves a formation-energy MAE of 0.060 eV/atom and a band-gap MAE of 0.292 eV on the Materials Project, while providing competitive multi-property prediction on JARVIS-DFT and superior HOMO prediction on QM9. The results highlight that explicit modeling of coordination polyhedra improves crystal representation learning and enables accurate, physically interpretable prediction of material properties. 

---
# Retrieval-Augmented Generation in LLMs for Mental Health: Quantifying the Incremental Contribution of Retrieval Within a Layered Safety Architecture 

**Authors**: Anand Gupta, Akshat Surolia, Shubham Mishra, Shakil Imtiaz, Chaitali Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2607.24817)  

**Abstract**: Digital mental health interventions (DMHIs) offer scalable support, but ensuring they accurately detect users' intent during volatile situations can be challenging. Pure parametric Large Language models (LLMs) do not contain specific safety critical architecture, and can miss critical cues, or hallucinate, undermining reliability. Retrieval Augmented Generation (RAG), which supplements an LLM with retrieved context, could enhance intent detection during volatile situations. Commercially available DMHIs typically combine multiple independent safety layers like rule-based filters, symbolic escalation protocols, and neural classification. The incremental contribution of any single layer, however, remains unquantified. This paper evaluates six LLM models within a DMHI called Wysa, via a controlled comparison of RAG-enabled versus RAG-disabled modes. Anonymized real and synthetic user-chatbot exchanges were annotated by a qualified clinical team against multi-class intent categories (e.g. self-harm, abuse, panic). The study computed classification accuracy, recall, precision and F1 scores against ground truth labels and tested differences for statistical significance. Performance was also examined by risk category and inter-model agreement. While RAG caused a rise in false alarms, the trade-off is consistent with safety-critical design principles that prioritize sensitivity, where flagged cases are routed to additional review rather than acted on directly. Overall, these findings support RAG as a promising approach to improve the accuracy, consistency and safety of LLM-driven DMHIs.
Keywords: Digital Mental Health Intervention, Large Language Model, Retrieval Augmented Generation, Accuracy, Recall, Precision 

---
# Bumblebee: Interleaved Mixed-Layer Building Blocks for Large-Scale Recommendation Systems 

**Authors**: David Bauer, Cancan Zhang, Wenshun Liu, Xiaoyi Zhang, Weijia Liu, Wanli Ma, Yue Weng, Wei Li, Rui Li, Jing Qian, Huayu Li, Xiaoyi Liu, Linhong Zhu, Jerry Fu  

**Link**: [PDF](https://arxiv.org/pdf/2607.24804)  

**Abstract**: Recommendation systems have undergone significant transformations in the past years. The transition from traditional feature interaction modules to generative next-action prediction has pushed the boundaries of personalized content. Developments have largely evolved along two separate tracks. Sequence modeling approaches on the one hand and feature interaction methods on the other. In this paper, we introduce Bumblebee, a recommendation architecture that addresses the lack of interaction between the two directions through an interleaved, stackable block design. Each block implements a micro-pipeline of layers combining sequence personalization, attention-based encoding, and feature crossing into a self-contained unit. Every block produces a joint representation of both feature modalities which is consumed by the next block in the sequence. This mechanism encourages early and repeated mixture of modalities and enriches downstream features with additional contextual information. Residual connections between blocks create cross-modal information pathways and yield additional predictive performance without adding additional parameters. Blocks can be specialized by selectively dropping components, enabling flexible trade-offs between quality and throughput. We evaluate our approach on large-scale industrial data and show consistent improvements over comparable baseline models across several classification and regression tasks. Furthermore, we conduct ablation studies to confirm that the interleaved composition itself is the primary driver of these improvements. Our results suggest that interleaving heterogeneous functional units, rather than composing deep stacks, is a promising paradigm for future-generation recommendation architectures. 

---
# SciClaimSeekers at CheckThat! 2026: Retrieving Scientific Sources for Social Media Claims with LLM Reranking 

**Authors**: Mohotarema Rashid, Nansu Baniya, Anirban Saha Anik, Xiaoying Song, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2607.24803)  

**Abstract**: Scientific claims often spread on social media faster than they can be verified, while posts rarely link to the original scholarly sources. To tackle this problem this paper presents system called SciClaimSeekers, a retrieval and reranking framework by combining BM25 and zero-shot multilingual E5 retrieval with Reciprocal Rank Fusion (k=60), followed by Qwen2.5-14B-Instruct pointwise reranking. The pipeline reaches 64.36% MRR@5 on the English development set a 13.67-point jump over BM25 and 10.17 points over the unranked hybrid and 64.39% on the official test set, in the CLEF-2026 CheckThat! Task 1 evaluation. Our experiment suggests that large pre-trained models, when combined into a careful pipeline, can be competitive with fine-tuned approaches on this task. 

---
# SourceMinds at CheckThat! 2026: NLI-Grounded Citation Auditing in a Multi-Agent Pipeline for Full Fact-Checking Article Generation 

**Authors**: Farhan Sharukh Hasan, Anirban Saha Anik, Eric Liu, Xiaoying Song, Mohotarema Rashid, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2607.24802)  

**Abstract**: This paper presents our system for Task 3 of the CLEF 2026 CheckThat! Lab, which focuses on generating full fact-checking articles from claims, veracity labels, and evidence documents. We propose a multi-agent pipeline that combines evidence retrieval, structured fact planning, article generation, gated self-critique, and NLI-based citation auditing. The system retrieves claim-relevant evidence using dense retrieval, reranking, and source-balanced selection, then generates a citation-supported article from a structured plan. A gated self-critique stage revises weakly grounded drafts, while the NLI citation auditor repairs missing citations and removes unsupported or redundant ones. The approach highlights the importance of combining evidence selection, structured generation, and post-generation citation validation for source-grounded fact-checking article generation. 

---
# Influence of Prompt Engineering on Small Language Models for Guarded Query Routing 

**Authors**: Richard Šléher, William Brach, Kristián Košťál, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2607.24801)  

**Abstract**: We study the problem of guarded query routing, where we assume that a user query first meets a router that either determines the ideal endpoint for in-distribution queries or rejects out-of-distribution queries that are potentially unsafe or out of the system's scope. We investigate whether compact open-weight Small Language Models (SLMs) can jointly handle both tasks under latency constraints. We evaluate 22 models on GQR-Bench and score them with the harmonic mean of in-distribution and out-of-distribution accuracy. We find that mid-scale SLMs come close to frontier model routing quality at much lower latency. Still, many compact models fail because they do not reliably follow the required output format. However, our results show that prompt optimization techniques enable SLMs to handle such cases gracefully, without changing the models' weights. Moreover, few-shot prompt optimization raises Mistral 7B from 81.79 to 90.87 GQR-Score and lifts Qwen3.5 9B to 95.74, the best optimized score in our study and within 0.3 points of the strongest unoptimized larger model: Gemma 3 27B at 96.01. The bare DSPy signature, without in-context exemplars, is the most effective strategy for Granite 4 Tiny, raising its score from 54.29 to 83.05. These results show that prompt optimization is a useful first step for guarded query routing, while weaker models may still need weight-level adaptation or schema-aware training 

---
# When Thinking Before Retrieval Hurts: TraceBound Diagnostics for Adaptive Knowledge-Graph Retrieval 

**Authors**: Partha Sarathi Purkayastha  

**Link**: [PDF](https://arxiv.org/pdf/2607.24800)  

**Abstract**: Adaptive retrieval promises to make knowledge-graph question answering more robust by letting a controller search, inspect neighborhoods, revise actions, and stop when evidence is sufficient. We study this premise by introducing TraceBound, a lightweight profile- and trace-conditioned diagnostic protocol for an ARK-style retriever on text-rich knowledge graphs. TraceBound exposes a compact query profile before retrieval, issues short trace hints after observable failure symptoms, and logs trajectory counters, while keeping graph data, tools, gold labels, and ranking metrics fixed. Across STaRK validation and held-out subsets, the added conditioning improves inspectability but consistently reduces retrieval quality under open-weight controllers. Paired trajectory analysis localizes the degradation to repeated calls, zero-result calls, and misallocated exploration budget, while stricter interaction budgets shorten trajectories without repairing the policy. The result diagnoses the common failure mode in that "thinking before retrieval'' must be evaluated as a control problem over action selection, not as a prompt-format change. 

---
# Multimodal Hybrid Retrieval-Augmented Generation for Scientific Document Understanding using Open-Source SLMs 

**Authors**: Alexandru-Andrei Saucă, Ana-Luiza Rusnac  

**Link**: [PDF](https://arxiv.org/pdf/2607.24799)  

**Abstract**: Large Language Models tend to hallucinate when answering domain-specific ques tions from scientific documents without prior fine-tuning. Currently, methods such as Retrieval-Augmented Generation partially solve this problem but face different challenges: limited context knowledge, difference between sparse and dense retrieval, and retrieval noise. This paper presents an Advanced Multimodal Retrieval-Augmented Generation system that aims to solve those challenges and im prove the accuracy of information extraction. The proposed architecture introduces a multimodal ingestion pipeline that leverages an open-source Vision-Language Model (Qwen2-VL-2B-Instruct) to generate textual summaries of tables and fig ures. The retrieval phase integrates HNSW-based semantic search with GIN-based lexical search, unified through Reciprocal Rank Fusion and refined using Cross Encoder reranking to minimize retrieval noise. To ensure conversational coherence across multi-turn interactions, a Query Condenser module is employed. Evaluation is conducted by independently assessing the ingestion, retrieval and generation stages using the MMLongBench benchmark, a BeIR-format synthetic dataset and the DeepEval framework. Moreover, results demonstrate a 157% improvement in retrieval quality over a Naive-RAG baseline, with only 50 ms additional la tency, while Qwen2-VL-2B-Instruct achieved results comparable to cloud-based models in BERTScore. These findings validate that open-source optimized SLMs, paired with advanced retrieval strategies, can provide competitive performance for document understanding without relying on cloud-based models. 

---
# Prediction Is Not Memory: Dual-Timescale Gated Profile Writing for Persistent User Modeling 

**Authors**: Ziyide Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.24798)  

**Abstract**: Persistent user profiles increasingly serve as reusable memory in recommender systems, but common update pipelines conflate two decisions: predicting an interaction and deciding whether it should persist in the profile. After an interaction is observed, many systems treat it as evidence for updating durable user state. This assumption can be harmful when the event reflects transient context, exploration, exposure, or short-term satisfaction rather than stable preference formation. We formulate this boundary as selective profile-write control: after an observed interaction, a system should decide whether, and how strongly, to write it into the persistent profile. We introduce a chronological near/far offline protocol in which near-future evidence provides weak write-risk supervision and far-future evidence is reserved for evaluation. We instantiate the controller as SPW-Gate, a lightweight write-risk gate using long-term, short-term, and candidate-profile drift features. On MicroLens-100K, write-all updating hurts far-future profile alignment in 22.45% of test cases under the main protocol. SPW-Gate reduces far hurt to about 14.5% while preserving about 77% write coverage. Matched-coverage controls and prediction-confidence baselines show that the gain is not merely a by-product of writing less, and that next-item confidence is not a sufficient proxy for persistent write validity. 

---
# Retrieval, not hallucinations, will be the limiting factor for LLM-based clinical AI tools 

**Authors**: Kirk Roberts, Steven Bedrick, Kurt Miller, William R. Hersh, Hongfang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.24793)  

**Abstract**: Discussions around large language model (LLM) errors in clinical artificial intelligence (AI) generally center around precision errors like hallucinations. This perspective, targeting both clinicians and AI researchers, seeks to shift that discussion to recall errors, particularly in retrieval of patient-level data needed for many clinical AI tools. The perspective outlines types of errors and mitigation strategies, describes research directions in LLMs and retrieval, and provides an overview of retrieval evaluation. 

---
# AI-Assisted Knowledge Access for Legacy Enterprise Asset Management in Energy Operations: A Practical Retrieval System 

**Authors**: Dave Mercier, Mishca de Costa, Muhammad Anwar, Mark Randall, Issam Hammad  

**Link**: [PDF](https://arxiv.org/pdf/2607.24792)  

**Abstract**: Energy utilities still run engineering work management, engineering procurement, and inventory processes on long-lived enterprise asset management platforms. Replacing these platforms is often cost prohibitive and operationally disruptive, so practical improvement layers are required. This paper presents a retrieval assistant that improves day-to-day knowledge access across three operational modes: vendor documentation question answering, operational data store (ODS) schema question answering, and user interface usage and how-to question answering. The runtime method combines intent understanding, query rewriting, hybrid semantic and vector retrieval, context engineering under token limits, grounded answer generation, and deterministic hyperlink conversion for panel identifiers and cited documentation. The data preparation pipeline emphasizes semantic enrichment as the primary quality lever by adding table and field descriptions, normalizing acronyms across sources, and indexing representative row-level context when useful. A measured pilot shows consistent gains in retrieval quality and user outcomes. Precision at five improved from 0.56 to 0.72, mean reciprocal rank from 0.43 to 0.58, and normalized discounted cumulative gain (nDCG) at five from 0.51 to 0.66. Median task completion time dropped from 14.2 to 8.3 minutes, while usefulness and confidence both increased to 4.0 on a five-point scale. Results are based on a small sample and are reported as pilot findings, but they indicate that intent understanding and semantic enrichment can deliver meaningful operational value in legacy environments while also establishing reusable foundations for future analytics and automation tools. 

---
# From Naive RAG to Deep Agentic Retrieval: An Evolving Context Engineering Pipeline for Regulatory Compliance 

**Authors**: Mishca de Costa, Muhammad Saleh Anwar, Dave Mercier, Issam Hammad  

**Link**: [PDF](https://arxiv.org/pdf/2607.24791)  

**Abstract**: Retrieval-augmented generation (RAG) is the dominant paradigm for applying large language models (LLMs) to enterprise document corpora, yet naive implementations encounter hard limits as corpus scale and query complexity grow. This paper traces the evolution of a production retrieval pipeline at Ontario Power Generation (OPG) for regulatory compliance and rate case analysis under Ontario Energy Board (OEB) reporting requirements. We examine successive stages: naive RAG, hybrid retrieval with re-ranking, agentic function-calling retrieval, and a deep multi-agent architecture with code-based tool synthesis and explicit planning, and identify the failure modes and tradeoffs that motivated each transition. We formalize the mature architecture as Progressive Evidence Acquisition with Cost-Aware Escalation (PEA-CAE): begin with low-cost, high-precision retrieval and escalate to full-document reads only when the expected evidence gain justifies latency and cost. Our findings show that context engineering is a more tractable and economically viable path than domain-specific fine-tuning for large, evolving regulatory corpora. More broadly, the progression toward deep agentic retrieval mirrors classical information retrieval ideas, introducing adaptive query reformulation, progressive document discovery, and hierarchical subagent summarization as practical system primitives. Operational traces further support the search-based nature of modern retrieval systems, where iterative evidence acquisition and adaptive planning increasingly replace single-pass retrieval as the foundation for enterprise-scale question answering. 

---
# NEXT: Reasoning-Driven Video Recommendation via a Vision-Language Model 

**Authors**: Yuming Liu, Hongye Yang, Harrison Zhao, Ellie Zhu, Bokai Cao, Lei Huang, Lizhu Zhang, Xiangjun Fan  

**Link**: [PDF](https://arxiv.org/pdf/2607.24789)  

**Abstract**: We present NEXT (Next-interest EXploration Transformer), a reasoning-driven video recommendation framework that reasons over the video a user has just watched, infers the viewer's next intent, and retrieves concrete follow-up videos. Explicit continuations such as episodes are linked directly; implicit cases are handled by generating intent queries and searching for matching candidates. This Item-to-Intent-to-Item formulation produces directed recommendations beyond co-engagement correlation or semantic similarity.
To make this framework reliable at scale, we train NEXT-8B, a purpose-trained 8B vision-language model with a three-stage recipe: Perception-Enhanced Reinforcement Learning for query-agnostic evidence extraction, Distribution-Aligned Supervised Fine-Tuning over real and synthetic visual QA mixtures, and Group Relative Policy Optimization for last-mile alignment. NEXT-8B achieves the best single-model DocVQA performance, ranking second overall only behind a multi-agent system while surpassing a substantially larger 200B+ scale model, and improves next-intent logic-wise quality by 3.3% over the base model in a task-specific LLM-as-a-judge evaluation.
We deploy NEXT as an additional retrieval path in a large-scale social media recommendation system and observe statistically significant production gains, including +0.53% watch time and +0.51% distinct video exposure. Overall, NEXT shows that a carefully trained compact vision-language model can serve as a practical reasoning engine for next-interest exploration at production scale. 

---
# Unlocking Spatial Grounding in Large Audio-Visual Retrieval models 

**Authors**: Hugo Malard, Michel Olvera, Sanjeel Parekh, Gaël Richard, Slim Essid, Stéphane Lathuilière  

**Link**: [PDF](https://arxiv.org/pdf/2607.24786)  

**Abstract**: Weak supervision sets a practical regime for audio-visual sound source localization as dense spatial annotations are costly to obtain at scale. The task, however, remains challenging, as models must locate sound sources from temporally aligned audio-visual data without pixel-level supervision. Recent large-scale audio-visual retrieval models, trained at unprecedented scale, encode rich multimodal structure. We show their latent representations, though optimized for global alignment, can nonetheless enable fine-grained spatial grounding. While spatial detail is progressively lost in the upper layers of retrieval backbones due to global pooling, intermediate visual tokens retain highly structured spatial information. To exploit this, we introduce LAIP (\emph{Localization via Audio-Informed Pooling}), a framework that employs a lightweight \emph{Audio-informed Spatial Pooling} (AiSP) to replace the standard global aggregation module. By using frame-aligned audio to query intermediate visual tokens, LAIP recovers localized spatial information that is otherwise discarded by the frozen retrieval pipeline. Our approach achieves state-of-the-art performance on AVSBench and AVATAR, nearly doubling previous results on the latter. These findings prove that accurate localization does not need to be learned from scratch; instead, it can be unlocked from existing retrieval representations, providing a unified path for both retrieval and localization tasks. 

---
# Three Sides of Retrieval: Factorial Evidence for Document-Side, Query-Side, and Answer-Side Complementarity in RAG 

**Authors**: Ng S. T. Chong  

**Link**: [PDF](https://arxiv.org/pdf/2607.24781)  

**Abstract**: RAG systems rely on chunking, which destroys structural information in documents. Existing heading-based retrieval (Jeong et al., 2025) requires multiple LLM calls per document and returns sub-chunks within matched sections. We introduce ToC-guided page retrieval, which infers headings from visual formatting without LLM calls, embeds them as a parallel index, and loads full page sections. Across 1,280 conditions on 8 enterprise documents (5 to 195 pages), we find: (1) ToC is a significant main effect on answer quality (d = +0.41, p = 0.031), with the largest gains in completeness (+0.40) and usefulness (+0.40); (2) combined with answer-side verification, it outperforms query-side decomposition + verification (d = +0.32, p = 0.036); (3) ToC contributes 20% of citations despite adding only 2.9 pages per query; (4) gains are directionally larger on longer documents (up to +1.50 on 118 pages), though the trend does not reach significance with 8 documents; and (5) a 480-condition sensitivity analysis finds no significant parameter effects (all p > 0.38), confirming defaults are near-optimal. The contribution is both methodological (a new zero-LLM-cost retrieval algorithm) and empirical: factorial evidence that document-side, query-side, and answer-side enhancements are complementary, a three-way interaction not previously studied. 

---
# JKO-RAG: Distributional Retrieval as Wasserstein Free-Energy Gradient Flow 

**Authors**: Levi Segal, Murari Ambati  

**Link**: [PDF](https://arxiv.org/pdf/2607.24776)  

**Abstract**: RAG pipelines return a \emph{ranked list} of passages. We argue this is a mismatch: the downstream language model conditions on a \emph{set}, and the selection problem is fundamentally geometric. We propose \jko, which frames reranking as minimising a free-energy functional $F(p)=\text{relevance}+\text{entropy}+\text{redundancy}$ under Wasserstein-2 gradient flow via the Jordan--Kinderlehrer--Otto proximal scheme. The ground metric $C_{ij}=(1-\cos\langle z_i,z_j\rangle)^2$ encodes the semantic geometry of the embedding manifold. Our central contribution is a \emph{linear-response theory} explaining \emph{why} the Wasserstein geometry helps: the Wasserstein and KL retrieval maps differ only in their proximal Hessian -- dense and geometry-aware for $W^2$, diagonal and geometry-blind for KL -- and this difference damps the mass transport that query paraphrase induces. The theory yields a falsifiable prediction: the stability advantage is monotonically decreasing in step size $h$. We verify this empirically via free-energy descent, frequency-resolved perturbation response, the predicted $h$-dependence, and a certified-radius analysis. Four extensions are introduced: \textbf{\nmjko} (learned ground metric), \textbf{\bwjko} ($W^2$--KL interpolation), \textbf{\samjko} ($2\times$ speedup), and \textbf{\dualrank} (OT dual potentials as confidence signals). Across five BEIR benchmarks, \jko\ outperforms the cross-encoder on all five; the decisive advantage is robustness -- 22--38\% more stable under paraphrase, $2\times$ fewer leaked distractors. 

---
# The Effect of Text Chunk Size on Retrieval-Augmented Generation Performance 

**Authors**: German Garrido-Lestache Belinchon, Hugo Garrido-Lestache Belinchon  

**Link**: [PDF](https://arxiv.org/pdf/2607.24767)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have emerged as a powerful process for allowing large language models (LLMs) to retrieve relevant information to use as source material during text generation. A critical yet under-explored component of these systems is the granularity at which source documents are segmented into retrievable chunks. The size of these chunks has the potential to significantly influence generation quality, contextual correctness, retrieval precision, and computational efficiency. Despite its importance, chunk size is often selected without proper evaluation of its impact on generation quality. Smaller chunks, such as individual sentences, may allow for precise retrieval by narrowing the focus of each chunk. However, they contain less information, which may limit the model's ability to generate coherent responses. Larger chunks, such as entire chapters, contain lots of broad information that may improve correctness, but also introduce additional noise and increase computational cost. Because larger chunks contain more information, the number of chunks returned to the model must also be considered. This paper evaluates how chunk size, along with the number of retrieved segments, influences generation quality and retrieval effectiveness. By comparing these configurations, this study seeks to better understand how document segmentation affects the performance and efficiency of Retrieval-Augmented Generation systems. segmentation affects the performance and efficiency of Retrieval-Augmented Generation systems. 

---
# CHaystack: Benchmarking Chinese Document Retrieval and VQA 

**Authors**: Hanxi Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.24760)  

**Abstract**: Retrieval-augmented generation (RAG) has made substantial progress in extending the memory of large language models (LLMs), and recent advances have further pushed RAG from pure text settings toward multimodal scenarios. In the document understanding domain, document visual question answering (DocumentVQA) has evolved from question answering over a single document to retrieval-and-generation pipelines over large-scale document collections. However, a benchmark specifically designed for Chinese large-scale document retrieval and question answering is still lacking. To bridge this gap, we introduce CHaystack, a new Chinese DocumentVQA benchmark that covers four document categories, namely academic papers, advertisements, web pages, and real-world photographed documents, enabling a more comprehensive evaluation of DocumentVQA systems. In addition, we present CDocRAG, a Chinese DocumentVQA system that uses a VLM-based relevance filter to verify retrieved document images before answer generation. We evaluate representative open-source embedding and generation models on CHaystack. The results reveal a clear contrast in category-wise strengths: Qwen-family models perform best on text-rich documents such as webpages and papers, whereas other models only achieve competitive results on visually rich categories such as advertisements and degrade sharply on text-dense documents. For retrieval, Qwen3-VL reaches 71.91 Recall@1 while the best non-Qwen model achieves only 14.40. These results indicate that the core challenge of CHaystack lies in Chinese textual encoding, and that Chinese large-scale DocumentVQA still leaves substantial room for improvement. Our code and dateset is available at this https URL. 

---
# VLD-RAG: Agentic Vision-Language Retrieval-Augmented Generation for Long, Visually-Rich Multi-Page Documents 

**Authors**: Seonok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.24748)  

**Abstract**: Visually-rich documents such as reports, slides, and manuals often distribute the evidence needed to answer a question across multiple pages, mixing text with layout cues, tables, charts, and figures. This work studies multimodal retrieval-augmented generation for question answering over such visually-rich long documents, where retrieval must select evidence pages that include both textual and visual signals. We present VLD-RAG, an agentic multimodal RAG framework for multi-page evidence retrieval and cross-page reasoning over long documents. VLD-RAG builds a page-preserving multimodal index that stores parsed text, page-level metadata, and dense visual representations, and uses a hybrid retrieval strategy that combines keyword-based sparse search with dense semantic queries to identify candidate sources and evidence pages. A verifier-guided agent workflow coordinates a Retrieval Agent, Answer Agent, and Validation Agent to broaden evidence coverage, detect missing citations, and refine retrieval requests when needed. We evaluate retrieval with Top-1 and Top-5 evidence-page accuracy and generation with generalized accuracy, and show that VLD-RAG improves both evidence-page retrieval and end-task question answering on visually-rich long-document benchmarks, including LongDocURL and MMLongBench-Doc, outperforming previous vision-based retrieval baselines. These findings highlight that coordinated agent verification and multimodal hybrid retrieval are crucial for reliable grounding when correct answers depend on evidence scattered across pages. 

---
# DocAnnot -- Accelerating the Creation of Key Information Extraction Datasets with GenAI-Powered Auto-annotation 

**Authors**: Siddartha Reddy, Harikrishnan P M, Goutham Vignesh, Varun V, Vishal Vaddina  

**Link**: [PDF](https://arxiv.org/pdf/2607.24745)  

**Abstract**: Key Information Extraction (KIE) is vital for many document applications, but creating training datasets is traditionally a time-consuming manual process. We introduce DocAnnot, a framework that significantly accelerates KIE dataset generation. DocAnnot leverages a Large Vision Language Model (LVLM) for label value extraction, OCR for text/bounding box detection, and a novel Spatially Informed Contextual Matching (SICM) algorithm. SICM improves label-value association by combining spatial relationships and proximity analysis with textual matching. We evaluate our framework on the CORD and SROIE benchmarks, demonstrating its ability to auto-generate annotations with F1-scores of 0.679 and 0.846, respectively. Furthermore, we investigate the effectiveness of using auto-annotated data for fine-tuning downstream KIE models. While human-annotated data remains superior, models trained exclusively on DocAnnot's outputs attain respectable performance (e.g., LayoutLMv3 achieving an F1-score of 0.6765 on CORD). These results show that while our framework significantly reduces reliance on manual effort, it does not yet fully eliminate the need for human intervention. However, by automating the process to a point where reviewers can efficiently refine outputs, our system enables near-perfect annotations with much greater efficiency than manual annotation from scratch. This approach offers substantial time and cost savings, making it valuable for resource-constrained settings and rapid model prototyping. 

---
# Tools Are Not Islands: Set-Level Tool Retrieval for LLM Agents via Query-Conditioned Hyperedge Prediction 

**Authors**: Xinyi Hong, Pinjun Dong, Xinyang Yu, Binyan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2607.25718)  

**Abstract**: Large language model (LLM) agents increasingly rely on invoking external tools to complete real-world tasks. Tool retrieval, which selects a small task-relevant subset from a library of thousands of tools before the agent acts, has therefore become a critical component of LLM agent pipelines. However, existing retrievers either score each tool in isolation or assemble the tool set sequentially, so the joint utility of a candidate set is never evaluated as a whole. In this paper, we propose HYSET, short for HYperedge-based SEt-level Tool retrieval. Our contributions are threefold: (i) we formulate tool retrieval as query-conditioned hyperedge prediction on a tool co-invocation hypergraph, under which the tool set itself becomes the unit of scoring and most existing retrieval paradigms reduce to restricted instances; (ii) we capture size-dependent tool compatibility through cardinality-specific interactions; and (iii) we design HYSET as a pre-selection module requiring no modification to the downstream agent. Experiments on ToolBench demonstrate that HYSET consistently outperforms state-of-the-art baselines in both tool retrieval performance and end-to-end task success. Beyond the in-domain setting, HYSET further supports zero-shot/few-shot transfer, generalizing to held-out tools/categories and unseen domains with minimal supervision. 

---
# Learning Dynamics of Strategic Publishers in Generative AI Ecosystems 

**Authors**: Sagie Dekel, Omer Madmon, Moshe Tennenholtz, Oren Kurland  

**Link**: [PDF](https://arxiv.org/pdf/2607.25514)  

**Abstract**: Generative AI (GenAI) search systems are transforming how users access information. Unlike ranking-based search systems, where users observe a ranked list of documents, GenAI search systems, given a user's question, generate an answer, often accompanied by external sources (e.g., in the form of citations). Content creators (publishers) seeking to increase exposure might behave strategically and compete with other creators for users' attention. While publishers in ranking-based systems might strategically modify their content to improve its ranking, the incentives in generative systems take on a new form. Publishers may now gain exposure through generated responses and attributions to those responses. We introduce a novel game-theoretic model of the emerging GenAI ecosystem in which publishers compete for attribution-based exposure. We study the learning dynamics of strategic content creators under better-response dynamics. We associate the convergence of learning dynamics to equilibrium with ecosystem stability. Employing the notion of potential games, we study the stability of GenAI ecosystems under several known content selection mechanisms. We demonstrate the instability of mechanisms representing real-world modern systems and characterize a mechanism that induces a stable ecosystem. We conduct extensive simulations to analyze the stability and welfare of GenAI ecosystems under various mechanisms. The simulations support our theoretical findings and reveal an interplay among stability, publisher welfare, and user welfare. In particular, stable mechanisms do not necessarily maximize welfare, demonstrating an important trade-off for platform designers. We then introduce a study illustrating that the proper selection of the GenAI mechanism enables the manifestation of desired trade-offs between publisher welfare and the different sources of user welfare. 

---
# TWICE: Two-Clock, Two-Window Learning for Long-Horizon Conversion Prediction in Online Advertising 

**Authors**: Kaiyuan Li, Kun Wang, Zhongbo Wang, Teng Sha, Ming Yan, Yanhua Cheng, Xialong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.25404)  

**Abstract**: Long-horizon conversion prediction under delayed feedback creates a two-clock, two-window learning problem in online advertising. A short base observation window releases recent clicks on the click clock before their outcomes mature, whereas conversions continue to arrive on the conversion clock throughout a longer target conversion window. The click clock provides timely but partially observed status supervision. The conversion clock reveals long-tail delays, but the delay composition within an arrival-time slice is weighted by historical click cohorts with different traffic volumes and target-window conversion rates.
We present TWICE, a framework that factorizes long-horizon post-click conversion rate (CVR) into a target-window conversion probability and a grouped elapsed-delay cumulative distribution function (CDF). The two clocks provide complementary supervision. Click-clock records train the target-window CVR head through a current-status likelihood over the base observation window. Newly arrived conversions train the delay model on the conversion clock. To account for the cohort mixture, TWICE uses fixed click-time predicted CVR (pCVR) mass as cohort exposure in an arrival-conditioned likelihood. This accounts for differences in cohort traffic and conversion propensity. The resulting aggregate records are self-contained. A single learned CDF produces monotone predictions for all requested horizons up to the target conversion window. Serving requires neither historical lookup nor convolution. Experiments on a public benchmark and an industrial advertising dataset demonstrate the effectiveness of TWICE. In an online A/B test in Kwai's advertising system, TWICE increased expected revenue, revenue, and conversions by 2.486%, 1.858%, and 2.061%, respectively. It was subsequently deployed to full traffic. 

---
# FunnelAL: Retrieve-then-Rank Active Learning for Single-Class Discovery 

**Authors**: Reihaneh Rostami, Brian Goodwin  

**Link**: [PDF](https://arxiv.org/pdf/2607.25276)  

**Abstract**: We present FunnelAL, a retrieve-then-rank active learning system for single-class discovery, which adapts the multi-stage funnel architecture of industrial recommender systems to data annotation. Large-scale supervised learning faces two challenges: efficiently finding relevant samples in a massive corpus, and distinguishing true positives from visually confusable negatives when embeddings do not cleanly separate classes. Conventional active learning offers a principled framework for reducing annotation cost, yet it treats sample selection as a single-stage process that addresses neither challenge efficiently. FunnelAL decomposes the problem into cascaded stages. Starting from a single positive and negative example, the system iterates through: (1) embedding-based retrieval scoring that narrows the corpus to a manageable candidate set; (2) a precision-triggered ranking stage that exploits a learned ranker (RankNet) while batch precision remains high, then automatically blends in committee-based exploration (QBC) once returns diminish; and (3) feedback from the annotator's labels that refines both stages in subsequent iterations. We evaluate on three diverse image classification benchmarks. With a perfect annotator, FunnelAL attains the best final F1 on all three benchmarks, the best annotation efficiency (first in AULC), and the fewest annotation rounds. The most recent single-class discovery methods (GAL, PF-MA) at best match its final quality, and only at consistently higher labeling cost. Under annotator labeling errors at realistic rates, FunnelAL remains first or statistically tied for first while classical uncertainty-based methods degrade two to three times faster. Our work provides a concrete bridge between multi-stage recommender systems and active learning. 

---
# The User Asks, Platforms Compete: How Agentic Recommendation Markets Take Shape 

**Authors**: Deyao Hong, Kehan Zheng, Qian Li, Jun Zhang, Jie Jiang, Hongning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.25253)  

**Abstract**: Online recommendation has traditionally taken place after a user enters a platform, which determines the candidate pool and the ranking shown to the user. LLM-based user agents enable a different recommendation process: a user specifies a need before choosing a platform, leaving platforms to compete for the user's attention, which we refer to as an agentic recommendation market. In our controlled LLM-based experiments across three product domains, we find this new setting of recommendation creates a tension between access and attention. Compared with traditional platform-centric recommendation, user-centric recommendation greatly expands the opportunity for relevant items to enter comparison; yet broader participation does not translate directly into effective exposure. Competition directly triggers platforms' strategic play: selectively positive explanations occupy 73--78% of first-ranked positions. When the user agent relates platforms' actions to subsequent user feedback, this share falls to 36--41%, while the chance of a user purchasing the relevant item increases. A user agent is therefore more than a ranker over a larger pool of candidates: its querying, ranking, and feedback mechanism governing who can compete, how scarce attention is allocated, and how earlier outcomes shape the evaluation of platforms directly affect user utility. Designing agentic recommendation therefore requires treating access, attention, and accountability as a joint mechanism design problem. 

---
# TabRank: Chain-of-Thought Distillation for Table Re-Rankers 

**Authors**: Adarsh Singh, Kushal Raj Bhandari, Jianxi Gao, Soham Dan, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2607.25182)  

**Abstract**: The ability to retrieve relevant tables for answering questions is a key task for structured information retrieval. Multi-stage retrieval systems rely heavily on rerankers to refine candidate lists produced by efficient first-stage retrievers. As a result, neural rerankers and LLM-based reranking methods have become increasingly important due to their superior capacity for semantic understanding and reasoning compared to conventional sparse or dense retrieval models. Recently, Large Reasoning Models (LRMs) equipped with explicit chain-of-thought (CoT) reasoning have shown strong improvements in ranking quality in unstructured passage retrieval. In this work, we present TabRank, a framework for training reasoning rerankers for Tabular Retrieval. We first present a comprehensive dataset of 6728 reasoning traces for tabular reranking on the Natural Questions Tables dataset. We then explore two variants of training a compact reasoning model on these reasoning traces: explicit CoT distillation and conditioning the student reranker on the teacher's reasoning trace within the prompt. We stress-test TabRank on several out-of-distribution generalization settings on diverse domains and multi-table scenarios. Our approach significantly improves performance across a variety of table retrieval datasets, increasing Acc@10 by 30.5% on HybridQA, 15.2% on SQA, 52.9% on TabFact, and 13.1% on TATQA subsets of the Multi-Table QA Benchmark compared to the base model. Notably, TabRank generalizes effectively to multi-table reasoning. Our code, data and models are available at this https URL 

---
