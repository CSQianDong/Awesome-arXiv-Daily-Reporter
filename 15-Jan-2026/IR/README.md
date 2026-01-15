# MM-BRIGHT: A Multi-Task Multimodal Benchmark for Reasoning-Intensive Retrieval 

**Authors**: Abdelrahman Abdallah, Mohamed Darwish Mounis, Mahmoud Abdalla, Mahmoud SalahEldin Kasem, Mostafa Farouk Senussi, Mohamed Mahmoud, Mohammed Ali, Adam Jatowt, Hyun-Soo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09562)  

**Abstract**: Existing retrieval benchmarks primarily consist of text-based queries where keyword or semantic matching is usually sufficient. Many real-world queries contain multimodal elements, particularly, images such as diagrams, charts, and screenshots that require intensive reasoning to identify relevant documents. To address this gap, we introduce MM-BRIGHT, the first multimodal benchmark for reasoning-intensive retrieval. Our dataset consists of 2,803 real-world queries spanning 29 diverse technical domains, with four tasks of increasing complexity: text-to-text, multimodal-to-text, multimodal-to-image, and multimodal-to-multimodal retrieval. Extensive evaluation reveals that state-of-the-art models struggle across all tasks: BM25 achieves only 8.5 nDCG@10 on text-only retrieval, while the best multimodal model Nomic-Vision reaches just 27.6 nDCG@10 on multimodal-to-text retrieval actually underperforming the best text-only model (DiVeR: 32.2). These results highlight substantial headroom and position MM-BRIGHT as a testbed for next-generation retrieval models that better integrate visual reasoning. Our code and data are available at this https URL. See also our official website: this https URL. 

---
# Examining DOM Coordinate Effectiveness For Page Segmentation 

**Authors**: Jason Carpenter, Faaiq Bilal, Eman Ramadan, Zhi-Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09543)  

**Abstract**: Web pages form a cornerstone of available data for daily human consumption and with the rise of LLM-based search and learning systems a treasure trove of valuable data. The scale of this data and its unstructured format still continue to grow requiring ever more robust automated extraction and retrieval mechanisms. Existing work, leveraging the web pages Document Object Model (DOM), often derives clustering vectors from coordinates informed by the DOM such as visual placement or tree structure. The construction and component value of these vectors often go unexamined. Our work proposes and examines DOM coordinates in a detail to understand their impact on web page segmentation. Our work finds that there is no one-size-fits-all vector, and that visual coordinates under-perform compared to DOM coordinates by about 20-30% on average. This challenges the necessity of including visual coordinates in clustering vectors. Further, our work finds that simple vectors, comprised of single coordinates, fare better than complex vectors constituting 68.2% of the top performing vectors of the pages examined. Finally, we find that if a vector, clustering algorithm, and page are properly matched, one can achieve overall high segmentation accuracy at 74%. This constitutes a 20% improvement over a naive application of vectors. Conclusively, our results challenge the current orthodoxy for segmentation vector creation, opens up the possibility to optimize page segmentation via clustering on DOM coordinates, and highlights the importance of finding mechanisms to match the best approach for web page segmentation. 

---
# SpatCode: Rotary-based Unified Encoding Framework for Efficient Spatiotemporal Vector Retrieval 

**Authors**: Bingde Hu, Enhao Pan, Wanjing Zhou, Yang Gao, Zunlei Feng, Hao Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2601.09530)  

**Abstract**: Spatiotemporal vector retrieval has emerged as a critical paradigm in modern information retrieval, enabling efficient access to massive, heterogeneous data that evolve over both time and space. However, existing spatiotemporal retrieval methods are often extensions of conventional vector search systems that rely on external filters or specialized indices to incorporate temporal and spatial constraints, leading to inefficiency, architectural complexity, and limited flexibility in handling heterogeneous modalities. To overcome these challenges, we present a unified spatiotemporal vector retrieval framework that integrates temporal, spatial, and semantic cues within a coherent similarity space while maintaining scalability and adaptability to continuous data streams. Specifically, we propose (1) a Rotary-based Unified Encoding Method that embeds time and location into rotational position vectors for consistent spatiotemporal representation; (2) a Circular Incremental Update Mechanism that supports efficient sliding-window updates without global re-encoding or index reconstruction; and (3) a Weighted Interest-based Retrieval Algorithm that adaptively balances modality weights for context-aware and personalized retrieval. Extensive experiments across multiple real-world datasets demonstrate that our framework substantially outperforms state-of-the-art baselines in both retrieval accuracy and efficiency, while maintaining robustness under dynamic data evolution. These results highlight the effectiveness and practicality of the proposed approach for scalable spatiotemporal information retrieval in intelligent systems. 

---
# TEMPO: A Realistic Multi-Domain Benchmark for Temporal Reasoning-Intensive Retrieval 

**Authors**: Abdelrahman Abdallah, Mohammed Ali, Muhammad Abdul-Mageed, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2601.09523)  

**Abstract**: Existing temporal QA benchmarks focus on simple fact-seeking queries from news corpora, while reasoning-intensive retrieval benchmarks lack temporal grounding. However, real-world information needs often require reasoning about temporal evolution and synthesizing evidence across time periods. We introduce TEMPO, the first benchmark combining temporal reasoning with reasoning-intensive retrieval across 13 domains. TEMPO features: (1) 1,730 complex queries requiring deep temporal reasoning such as tracking changes, identifying trends, or comparing cross-period evidence; (2) step-wise retrieval planning with 3,976 decomposed steps and gold documents mapped to each step for multi-hop evaluation; and (3) novel temporal metrics including Temporal Coverage@k and Temporal Precision@k measuring whether results span required time periods. Evaluation of 12 retrieval systems reveals substantial challenges: the best model (DiVeR) achieves only 32.0 NDCG@10 and 71.4\% Temporal Coverage@10, demonstrating difficulty in retrieving temporally complete evidence. We believe TEMPO provides a challenging benchmark for improving temporal reasoning in retrieval and RAG systems. Our code and data are available at this https URL. See also our official website: this https URL. 

---
# Unifying Search and Recommendation in LLMs via Gradient Multi-Subspace Tuning 

**Authors**: Jujia Zhao, Zihan Wang, Shuaiqun Pan, Suzan Verberne, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2601.09496)  

**Abstract**: Search and recommendation (S&R) are core to online platforms, addressing explicit intent through queries and modeling implicit intent from behaviors, respectively. Their complementary roles motivate a unified modeling paradigm. Early studies to unify S&R adopt shared encoders with task-specific heads, while recent efforts reframe item ranking in both S&R as conditional generation. The latter holds particular promise, enabling end-to-end optimization and leveraging the semantic understanding of LLMs. However, existing methods rely on full fine-tuning, which is computationally expensive and limits scalability. Parameter-efficient fine-tuning (PEFT) offers a more practical alternative but faces two critical challenges in unifying S&R: (1) gradient conflicts across tasks due to divergent optimization objectives, and (2) shifts in user intent understanding caused by overfitting to fine-tuning data, which distort general-domain knowledge and weaken LLM reasoning. To address the above issues, we propose Gradient Multi-Subspace Tuning (GEMS), a novel framework that unifies S&R with LLMs while alleviating gradient conflicts and preserving general-domain knowledge. GEMS introduces (1) \textbf{Multi-Subspace Decomposition}, which disentangles shared and task-specific optimization signals into complementary low-rank subspaces, thereby reducing destructive gradient interference, and (2) \textbf{Null-Space Projection}, which constrains parameter updates to a subspace orthogonal to the general-domain knowledge space, mitigating shifts in user intent understanding. Extensive experiments on benchmark datasets show that GEMS consistently outperforms the state-of-the-art baselines across both search and recommendation tasks, achieving superior effectiveness. 

---
# Bridging Semantic Understanding and Popularity Bias with LLMs 

**Authors**: Renqiang Luo, Dong Zhang, Yupeng Gao, Wen Shi, Mingliang Hou, Jiaying Liu, Zhe Wang, Shuo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09478)  

**Abstract**: Semantic understanding of popularity bias is a crucial yet underexplored challenge in recommender systems, where popular items are often favored at the expense of niche content. Most existing debiasing methods treat the semantic understanding of popularity bias as a matter of diversity enhancement or long-tail coverage, neglecting the deeper semantic layer that embodies the causal origins of the bias itself. Consequently, such shallow interpretations limit both their debiasing effectiveness and recommendation accuracy. In this paper, we propose FairLRM, a novel framework that bridges the gap in the semantic understanding of popularity bias with Recommendation via Large Language Model (RecLLM). FairLRM decomposes popularity bias into item-side and user-side components, using structured instruction-based prompts to enhance the model's comprehension of both global item distributions and individual user preferences. Unlike traditional methods that rely on surface-level features such as "diversity" or "debiasing", FairLRM improves the model's ability to semantically interpret and address the underlying bias. Through empirical evaluation, we show that FairLRM significantly enhances both fairness and recommendation accuracy, providing a more semantically aware and trustworthy approach to enhance the semantic understanding of popularity bias. The implementation is available at this https URL. 

---
# Dissecting Judicial Reasoning in U.S. Copyright Damage Awards 

**Authors**: Pei-Chi Lo, Thomas Y. Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09459)  

**Abstract**: Judicial reasoning in copyright damage awards poses a core challenge for computational legal analysis. Although federal courts follow the 1976 Copyright Act, their interpretations and factor weightings vary widely across jurisdictions. This inconsistency creates unpredictability for litigants and obscures the empirical basis of legal decisions. This research introduces a novel discourse-based Large Language Model (LLM) methodology that integrates Rhetorical Structure Theory (RST) with an agentic workflow to extract and quantify previously opaque reasoning patterns from judicial opinions. Our framework addresses a major gap in empirical legal scholarship by parsing opinions into hierarchical discourse structures and using a three-stage pipeline, i.e., Dataset Construction, Discourse Analysis, and Agentic Feature Extraction. This pipeline identifies reasoning components and extract feature labels with corresponding discourse subtrees. In analyzing copyright damage rulings, we show that discourse-augmented LLM analysis outperforms traditional methods while uncovering unquantified variations in factor weighting across circuits. These findings offer both methodological advances in computational legal analysis and practical insights into judicial reasoning, with implications for legal practitioners seeking predictive tools, scholars studying legal principle application, and policymakers confronting inconsistencies in copyright law. 

---
# LISP -- A Rich Interaction Dataset and Loggable Interactive Search Platform 

**Authors**: Jana Isabelle Friese, Andreas Konstantin Kruff, Philipp Schaer, Norbert Fuhr, Nicola Ferro  

**Link**: [PDF](https://arxiv.org/pdf/2601.09366)  

**Abstract**: We present a reusable dataset and accompanying infrastructure for studying human search behavior in Interactive Information Retrieval (IIR). The dataset combines detailed interaction logs from 61 participants (122 sessions) with user characteristics, including perceptual speed, topic-specific interest, search expertise, and demographic information. To facilitate reproducibility and reuse, we provide a fully documented study setup, a web-based perceptual speed test, and a framework for conducting similar user studies. Our work allows researchers to investigate individual and contextual factors affecting search behavior, and to develop or validate user simulators that account for such variability. We illustrate the datasets potential through an illustrative analysis and release all resources as open-access, supporting reproducible research and resource sharing in the IIR community. 

---
# On-Device Large Language Models for Sequential Recommendation 

**Authors**: Xin Xia, Hongzhi Yin, Shane Culpepper  

**Link**: [PDF](https://arxiv.org/pdf/2601.09306)  

**Abstract**: On-device recommendation is critical for a number of real-world applications, especially in scenarios that have agreements on execution latency, user privacy, and robust functionality when internet connectivity is unstable or even impossible. While large language models (LLMs) can now provide exceptional capabilities that model user behavior for sequential recommendation tasks, their substantial memory footprint and computational overhead make the deployment on resource-constrained devices a high risk proposition. In this paper, we propose OD-LLM, the first task-adaptive compression framework explicitly designed to provide efficient and accurate on-device deployment of LLMs for sequential recommendation tasks. OD-LLM uniquely integrates two complementary compression strategies: a low-rank structural compression algorithm which uses Singular Value Decomposition (SVD) to significantly reduce parameter redundancy in the model, and a novel tokenization normalization technique that better complements the low-rank decomposition process being used. Additionally, to minimize any potential performance degradation when using higher compression ratios, a novel progressive alignment algorithm is used to iteratively refine the parameters required layerwise in the target model. Empirical evaluations conducted on sequential recommendation benchmarks show that OD-LLM exhibits no loss in effectiveness when compared to the original recommendation model, when the deployed model size is halved. These promising results demonstrate the efficacy and scalability of OD-LLM, making this novel solution a practical alternative for real-time, on-device solutions wishing to replace expensive, remotely executed LLMs. 

---
# Why not Collaborative Filtering in Dual View? Bridging Sparse and Dense Models 

**Authors**: Hanze Guo, Jianxun Lian, Xiao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.09286)  

**Abstract**: Collaborative Filtering (CF) remains the cornerstone of modern recommender systems, with dense embedding--based methods dominating current practice. However, these approaches suffer from a critical limitation: our theoretical analysis reveals a fundamental signal-to-noise ratio (SNR) ceiling when modeling unpopular items, where parameter-based dense models experience diminishing SNR under severe data sparsity. To overcome this bottleneck, we propose SaD (Sparse and Dense), a unified framework that integrates the semantic expressiveness of dense embeddings with the structural reliability of sparse interaction patterns. We theoretically show that aligning these dual views yields a strictly superior global SNR. Concretely, SaD introduces a lightweight bidirectional alignment mechanism: the dense view enriches the sparse view by injecting semantic correlations, while the sparse view regularizes the dense model through explicit structural signals. Extensive experiments demonstrate that, under this dual-view alignment, even a simple matrix factorization--style dense model can achieve state-of-the-art performance. Moreover, SaD is plug-and-play and can be seamlessly applied to a wide range of existing recommender models, highlighting the enduring power of collaborative filtering when leveraged from dual perspectives. Further evaluations on real-world benchmarks show that SaD consistently outperforms strong baselines, ranking first on the BarsMatch leaderboard. The code is publicly available at this https URL. 

---
# LLMs Meet Isolation Kernel: Lightweight, Learning-free Binary Embeddings for Fast Retrieval 

**Authors**: Zhibo Zhang, Yang Xu, Kai Ming Ting, Cam-Tu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2601.09159)  

**Abstract**: Large language models (LLMs) have recently enabled remarkable progress in text representation. However, their embeddings are typically high-dimensional, leading to substantial storage and retrieval overhead. Although recent approaches such as Matryoshka Representation Learning (MRL) and Contrastive Sparse Representation (CSR) alleviate these issues to some extent, they still suffer from retrieval accuracy degradation. This paper proposes \emph{Isolation Kernel Embedding} or IKE, a learning-free method that transforms an LLM embedding into a binary embedding using Isolation Kernel (IK). IKE is an ensemble of diverse (random) partitions, enabling robust estimation of ideal kernel in the LLM embedding space, thus reducing retrieval accuracy loss as the ensemble grows. Lightweight and based on binary encoding, it offers low memory footprint and fast bitwise computation, lowering retrieval latency. Experiments on multiple text retrieval datasets demonstrate that IKE offers up to 16.7x faster retrieval and 16x lower memory usage than LLM embeddings, while maintaining comparable or better accuracy. Compared to CSR and other compression methods, IKE consistently achieves the best balance between retrieval efficiency and effectiveness. 

---
# Fine Grained Evaluation of LLMs-as-Judges 

**Authors**: Sourav Saha, Mandar Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2601.08919)  

**Abstract**: A good deal of recent research has focused on how Large Language Models
(LLMs) may be used as `judges' in place of humans to evaluate the quality
of the output produced by various text / image processing systems. Within
this broader context, a number of studies have investigated the specific
question of how effectively LLMs can be used as relevance assessors for
the standard ad hoc task in Information Retrieval (IR). We extend these
studies by looking at additional questions. Most importantly, we use a
Wikipedia based test collection created by the INEX initiative, and
prompt LLMs to not only judge whether documents are relevant /
non-relevant, but to highlight relevant passages in documents that it
regards as useful. The human relevance assessors involved in creating
this collection were given analogous instructions, i.e., they were asked
to highlight all passages within a document that respond to the
information need expressed in a query. This enables us to evaluate the
quality of LLMs as judges not only at the document level, but to also
quantify how often these `judges' are right for the right reasons.
Our findings suggest that LLMs-as-judges work best under human
supervision. 

---
# Navigating Ideation Space: Decomposed Conceptual Representations for Positioning Scientific Ideas 

**Authors**: Yuexi Shen, Minqian Liu, Dawei Zhou, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08901)  

**Abstract**: Scientific discovery is a cumulative process and requires new ideas to be situated within an ever-expanding landscape of existing knowledge. An emerging and critical challenge is how to identify conceptually relevant prior work from rapidly growing literature, and assess how a new idea differentiates from existing research. Current embedding approaches typically conflate distinct conceptual aspects into single representations and cannot support fine-grained literature retrieval; meanwhile, LLM-based evaluators are subject to sycophancy biases, failing to provide discriminative novelty assessment. To tackle these challenges, we introduce the Ideation Space, a structured representation that decomposes scientific knowledge into three distinct dimensions, i.e., research problem, methodology, and core findings, each learned through contrastive training. This framework enables principled measurement of conceptual distance between ideas, and modeling of ideation transitions that capture the logical connections within a proposed idea. Building upon this representation, we propose a Hierarchical Sub-Space Retrieval framework for efficient, targeted literature retrieval, and a Decomposed Novelty Assessment algorithm that identifies which aspects of an idea are novel. Extensive experiments demonstrate substantial improvements, where our approach achieves Recall@30 of 0.329 (16.7% over baselines), our ideation transition retrieval reaches Hit Rate@30 of 0.643, and novelty assessment attains 0.37 correlation with expert judgments. In summary, our work provides a promising paradigm for future research on accelerating and evaluating scientific discovery. 

---
# Information Access of the Oppressed: A Problem-Posing Framework for Envisioning Emancipatory Information Access Platforms 

**Authors**: Bhaskar Mitra, Nicola Neophytou, Sireesh Gururaja  

**Link**: [PDF](https://arxiv.org/pdf/2601.09600)  

**Abstract**: Online information access (IA) platforms are targets of authoritarian capture. These concerns are particularly serious and urgent today in light of the rising levels of democratic erosion worldwide, the emerging capabilities of generative AI technologies such as AI persuasion, and the increasing concentration of economic and political power in the hands of Big Tech. This raises the question of what alternative IA infrastructure we must reimagine and build to mitigate the risks of authoritarian capture of our information ecosystems. We explore this question through the lens of Paulo Freire's theories of emancipatory pedagogy. Freire's theories provide a radically different lens for exploring IA's sociotechnical concerns relative to the current dominating frames of fairness, accountability, confidentiality, transparency, and safety. We make explicit, with the intention to challenge, the dichotomy of how we relate to technology as either technologists (who envision and build technology) and its users. We posit that this mirrors the teacher-student relationship in Freire's analysis. By extending Freire's analysis to IA, we challenge the notion that it is the burden of the (altruistic) technologists to come up with interventions to mitigate the risks that emerging technologies pose to marginalized communities. Instead, we advocate that the first task for the technologists is to pose these as problems to the marginalized communities, to encourage them to make and unmake the technology as part of their material struggle against oppression. Their second task is to redesign our online technology stacks to structurally expose spaces for community members to co-opt and co-construct the technology in aid of their emancipatory struggles. We operationalize Freire's theories to develop a problem-posing framework for envisioning emancipatory IA platforms of the future. 

---
# A Deep Dive into OpenStreetMap Research Since its Inception (2008-2024): Contributors, Topics, and Future Trends 

**Authors**: Yao Sun, Liqiu Meng, Andres Camero, Stefan Auer, Xiao Xiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09338)  

**Abstract**: OpenStreetMap (OSM) has transitioned from a pioneering volunteered geographic information (VGI) project into a global, multi-disciplinary research nexus. This study presents a bibliometric and systematic analysis of the OSM research landscape, examining its development trajectory and key driving forces. By evaluating 1,926 publications from the Web of Science (WoS) Core Collection and 782 State of the Map (SotM) presentations up to June 2024, we quantify publication growth, collaboration patterns, and thematic evolution. Results demonstrate simultaneous consolidation and diversification within the field. While a stable core of contributors continues to anchor OSM research, themes have shifted from initial concerns over data production and quality toward advanced analytical and applied uses. Comparative analysis of OSM-related research in WoS and SotM reveals distinct but complementary agendas between scholars and the OSM community. Building on these findings, we identify six emerging research directions and discuss how evolving partnerships among academia, the OSM community, and industry are poised to shape the future of OSM research. This study establishes a structured reference for understanding the state of OSM studies and offers strategic pathways for navigating its future this http URL data and code are available at this https URL. 

---
# MMR-GRPO: Accelerating GRPO-Style Training through Diversity-Aware Reward Reweighting 

**Authors**: Kangda Wei, Ruihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09085)  

**Abstract**: Group Relative Policy Optimization (GRPO) has become a standard approach for training mathematical reasoning models; however, its reliance on multiple completions per prompt makes training computationally expensive. Although recent work has reduced the number of training steps required to reach peak performance, the overall wall-clock training time often remains unchanged or even increases due to higher per-step cost. We propose MMR-GRPO, which integrates Maximal Marginal Relevance to reweigh rewards based on completion diversity. Our key insight is that semantically redundant completions contribute limited marginal learning signal; prioritizing diverse solutions yields more informative updates and accelerates convergence. Extensive evaluations across three model sizes (1.5B, 7B, 8B), three GRPO variants, and five mathematical reasoning benchmarks show that MMR-GRPO achieves comparable peak performance while requiring on average 47.9% fewer training steps and 70.2% less wall-clock time. These gains are consistent across models, methods, and benchmarks. We will release our code, trained models, and experimental protocols. 

---
# StegoStylo: Squelching Stylometric Scrutiny through Steganographic Stitching 

**Authors**: Robert Dilworth  

**Link**: [PDF](https://arxiv.org/pdf/2601.09056)  

**Abstract**: Stylometry--the identification of an author through analysis of a text's style (i.e., authorship attribution)--serves many constructive purposes: it supports copyright and plagiarism investigations, aids detection of harmful content, offers exploratory cues for certain medical conditions (e.g., early signs of dementia or depression), provides historical context for literary works, and helps uncover misinformation and disinformation. In contrast, when stylometry is employed as a tool for authorship verification--confirming whether a text truly originates from a claimed author--it can also be weaponized for malicious purposes. Techniques such as de-anonymization, re-identification, tracking, profiling, and downstream effects like censorship illustrate the privacy threats that stylometric analysis can enable. Building on these concerns, this paper further explores how adversarial stylometry combined with steganography can counteract stylometric analysis. We first present enhancements to our adversarial attack, $\textit{TraceTarnish}$, providing stronger evidence of its capacity to confound stylometric systems and reduce their attribution and verification accuracy. Next, we examine how steganographic embedding can be fine-tuned to mask an author's stylistic fingerprint, quantifying the level of authorship obfuscation achievable as a function of the proportion of words altered with zero-width Unicode characters. Based on our findings, steganographic coverage of 33% or higher seemingly ensures authorship obfuscation. Finally, we reflect on the ways stylometry can be used to undermine privacy and argue for the necessity of defensive tools like $\textit{TraceTarnish}$. 

---
# SpectraQuery: A Hybrid Retrieval-Augmented Conversational Assistant for Battery Science 

**Authors**: Sreya Vangara, Jagjit Nanda, Yan-Kai Tzeng, Eric Darve  

**Link**: [PDF](https://arxiv.org/pdf/2601.09036)  

**Abstract**: Scientific reasoning increasingly requires linking structured experimental data with the unstructured literature that explains it, yet most large language model (LLM) assistants cannot reason jointly across these modalities. We introduce SpectraQuery, a hybrid natural-language query framework that integrates a relational Raman spectroscopy database with a vector-indexed scientific literature corpus using a Structured and Unstructured Query Language (SUQL)-inspired design. By combining semantic parsing with retrieval-augmented generation, SpectraQuery translates open-ended questions into coordinated SQL and literature retrieval operations, producing cited answers that unify numerical evidence with mechanistic explanation. Across SQL correctness, answer groundedness, retrieval effectiveness, and expert evaluation, SpectraQuery demonstrates strong performance: approximately 80 percent of generated SQL queries are fully correct, synthesized answers reach 93-97 percent groundedness with 10-15 retrieved passages, and battery scientists rate responses highly across accuracy, relevance, grounding, and clarity (4.1-4.6/5). These results show that hybrid retrieval architectures can meaningfully support scientific workflows by bridging data and discourse for high-volume experimental datasets. 

---
# OpenDecoder: Open Large Language Model Decoding to Incorporate Document Quality in RAG 

**Authors**: Fengran Mo, Zhan Su, Yuchen Hui, Jinghan Zhang, Jia Ao Sun, Zheyuan Liu, Chao Zhang, Tetsuya Sakai, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2601.09028)  

**Abstract**: The development of large language models (LLMs) has achieved superior performance in a range of downstream tasks, including LLM-based retrieval-augmented generation (RAG). The quality of generated content heavily relies on the usefulness of the retrieved information and the capacity of LLMs' internal information processing mechanism to incorporate it in answer generation. It is generally assumed that the retrieved information is relevant to the question. However, the retrieved information may have a variable degree of relevance and usefulness, depending on the question and the document collection. It is important to take into account the relevance of the retrieved information in answer generation. In this paper, we propose OpenDecoder, a new approach that leverages explicit evaluation of the retrieved information as quality indicator features for generation. We aim to build a RAG model that is more robust to varying levels of noisy context. Three types of explicit evaluation information are considered: relevance score, ranking score, and QPP (query performance prediction) score. The experimental results on five benchmark datasets demonstrate the effectiveness and better robustness of OpenDecoder by outperforming various baseline methods. Importantly, this paradigm is flexible to be integrated with the post-training of LLMs for any purposes and incorporated with any type of external indicators. 

---
