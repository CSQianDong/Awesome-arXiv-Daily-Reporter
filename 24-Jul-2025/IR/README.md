# Leave No One Behind: Fairness-Aware Cross-Domain Recommender Systems for Non-Overlapping Users 

**Authors**: Weixin Chen, Yuhan Zhao, Li Chen, Weike Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17749)  

**Abstract**: Cross-domain recommendation (CDR) methods predominantly leverage overlapping users to transfer knowledge from a source domain to a target domain. However, through empirical studies, we uncover a critical bias inherent in these approaches: while overlapping users experience significant enhancements in recommendation quality, non-overlapping users benefit minimally and even face performance degradation. This unfairness may erode user trust, and, consequently, negatively impact business engagement and revenue. To address this issue, we propose a novel solution that generates virtual source-domain users for non-overlapping target-domain users. Our method utilizes a dual attention mechanism to discern similarities between overlapping and non-overlapping users, thereby synthesizing realistic virtual user embeddings. We further introduce a limiter component that ensures the generated virtual users align with real-data distributions while preserving each user's unique characteristics. Notably, our method is model-agnostic and can be seamlessly integrated into any CDR model. Comprehensive experiments conducted on three public datasets with five CDR baselines demonstrate that our method effectively mitigates the CDR non-overlapping user bias, without loss of overall accuracy. Our code is publicly available at this https URL. 

---
# Citation Recommendation using Deep Canonical Correlation Analysis 

**Authors**: Conor McNamara, Effirul Ramlan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17603)  

**Abstract**: Recent advances in citation recommendation have improved accuracy by leveraging multi-view representation learning to integrate the various modalities present in scholarly documents. However, effectively combining multiple data views requires fusion techniques that can capture complementary information while preserving the unique characteristics of each modality. We propose a novel citation recommendation algorithm that improves upon linear Canonical Correlation Analysis (CCA) methods by applying Deep CCA (DCCA), a neural network extension capable of capturing complex, non-linear relationships between distributed textual and graph-based representations of scientific articles. Experiments on the large-scale DBLP (Digital Bibliography & Library Project) citation network dataset demonstrate that our approach outperforms state-of-the-art CCA-based methods, achieving relative improvements of over 11% in Mean Average Precision@10, 5% in Precision@10, and 7% in Recall@10. These gains reflect more relevant citation recommendations and enhanced ranking quality, suggesting that DCCA's non-linear transformations yield more expressive latent representations than CCA's linear projections. 

---
# "Beyond the past": Leveraging Audio and Human Memory for Sequential Music Recommendation 

**Authors**: Viet-Tran Anh, Bruno Sguerra, Gabriel Meseguer-Brocal, Lea Briand, Manuel Moussallam  

**Link**: [PDF](https://arxiv.org/pdf/2507.17356)  

**Abstract**: On music streaming services, listening sessions are often composed of a balance of familiar and new tracks. Recently, sequential recommender systems have adopted cognitive-informed approaches, such as Adaptive Control of Thought-Rational (ACT-R), to successfully improve the prediction of the most relevant tracks for the next user session. However, one limitation of using a model inspired by human memory (or the past), is that it struggles to recommend new tracks that users have not previously listened to. To bridge this gap, here we propose a model that leverages audio information to predict in advance the ACT-R-like activation of new tracks and incorporates them into the recommendation scoring process. We demonstrate the empirical effectiveness of the proposed model using proprietary data, which we publicly release along with the model's source code to foster future research in this field. 

---
# EndoFinder: Online Lesion Retrieval for Explainable Colorectal Polyp Diagnosis Leveraging Latent Scene Representations 

**Authors**: Ruijie Yang, Yan Zhu, Peiyao Fu, Yizhe Zhang, Zhihua Wang, Quanlin Li, Pinghong Zhou, Xian Yang, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17323)  

**Abstract**: Colorectal cancer (CRC) remains a leading cause of cancer-related mortality, underscoring the importance of timely polyp detection and diagnosis. While deep learning models have improved optical-assisted diagnostics, they often demand extensive labeled datasets and yield "black-box" outputs with limited interpretability. In this paper, we propose EndoFinder, an online polyp retrieval framework that leverages multi-view scene representations for explainable and scalable CRC diagnosis. First, we develop a Polyp-aware Image Encoder by combining contrastive learning and a reconstruction task, guided by polyp segmentation masks. This self-supervised approach captures robust features without relying on large-scale annotated data. Next, we treat each polyp as a three-dimensional "scene" and introduce a Scene Representation Transformer, which fuses multiple views of the polyp into a single latent representation. By discretizing this representation through a hashing layer, EndoFinder enables real-time retrieval from a compiled database of historical polyp cases, where diagnostic information serves as interpretable references for new queries. We evaluate EndoFinder on both public and newly collected polyp datasets for re-identification and pathology classification. Results show that EndoFinder outperforms existing methods in accuracy while providing transparent, retrieval-based insights for clinical decision-making. By contributing a novel dataset and a scalable, explainable framework, our work addresses key challenges in polyp diagnosis and offers a promising direction for more efficient AI-driven colonoscopy workflows. The source code is available at this https URL. 

---
# Exploring the Potential of LLMs for Serendipity Evaluation in Recommender Systems 

**Authors**: Li Kang, Yuhan Zhao, Li Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.17290)  

**Abstract**: Serendipity plays a pivotal role in enhancing user satisfaction within recommender systems, yet its evaluation poses significant challenges due to its inherently subjective nature and conceptual ambiguity. Current algorithmic approaches predominantly rely on proxy metrics for indirect assessment, often failing to align with real user perceptions, thus creating a gap. With large language models (LLMs) increasingly revolutionizing evaluation methodologies across various human annotation tasks, we are inspired to explore a core research proposition: Can LLMs effectively simulate human users for serendipity evaluation? To address this question, we conduct a meta-evaluation on two datasets derived from real user studies in the e-commerce and movie domains, focusing on three key aspects: the accuracy of LLMs compared to conventional proxy metrics, the influence of auxiliary data on LLM comprehension, and the efficacy of recently popular multi-LLM techniques. Our findings indicate that even the simplest zero-shot LLMs achieve parity with, or surpass, the performance of conventional metrics. Furthermore, multi-LLM techniques and the incorporation of auxiliary data further enhance alignment with human perspectives. Based on our findings, the optimal evaluation by LLMs yields a Pearson correlation coefficient of 21.5\% when compared to the results of the user study. This research implies that LLMs may serve as potentially accurate and cost-effective evaluators, introducing a new paradigm for serendipity evaluation in recommender systems. 

---
# R4ec: A Reasoning, Reflection, and Refinement Framework for Recommendation Systems 

**Authors**: Hao Gu, Rui Zhong, Yu Xia, Wei Yang, Chi Lu, Peng Jiang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2507.17249)  

**Abstract**: Harnessing Large Language Models (LLMs) for recommendation systems has emerged as a prominent avenue, drawing substantial research interest. However, existing approaches primarily involve basic prompt techniques for knowledge acquisition, which resemble System-1 thinking. This makes these methods highly sensitive to errors in the reasoning path, where even a small mistake can lead to an incorrect inference. To this end, in this paper, we propose $R^{4}$ec, a reasoning, reflection and refinement framework that evolves the recommendation system into a weak System-2 model. Specifically, we introduce two models: an actor model that engages in reasoning, and a reflection model that judges these responses and provides valuable feedback. Then the actor model will refine its response based on the feedback, ultimately leading to improved responses. We employ an iterative reflection and refinement process, enabling LLMs to facilitate slow and deliberate System-2-like thinking. Ultimately, the final refined knowledge will be incorporated into a recommendation backbone for prediction. We conduct extensive experiments on Amazon-Book and MovieLens-1M datasets to demonstrate the superiority of $R^{4}$ec. We also deploy $R^{4}$ec on a large scale online advertising platform, showing 2.2\% increase of revenue. Furthermore, we investigate the scaling properties of the actor model and reflection model. 

---
# Enhancing Transferability and Consistency in Cross-Domain Recommendations via Supervised Disentanglement 

**Authors**: Yuhan Wang, Qing Xie, Zhifeng Bao, Mengzi Tang, Lin Li, Yongjian Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.17112)  

**Abstract**: Cross-domain recommendation (CDR) aims to alleviate the data sparsity by transferring knowledge across domains. Disentangled representation learning provides an effective solution to model complex user preferences by separating intra-domain features (domain-shared and domain-specific features), thereby enhancing robustness and interpretability. However, disentanglement-based CDR methods employing generative modeling or GNNs with contrastive objectives face two key challenges: (i) pre-separation strategies decouple features before extracting collaborative signals, disrupting intra-domain interactions and introducing noise; (ii) unsupervised disentanglement objectives lack explicit task-specific guidance, resulting in limited consistency and suboptimal alignment. To address these challenges, we propose DGCDR, a GNN-enhanced encoder-decoder framework. To handle challenge (i), DGCDR first applies GNN to extract high-order collaborative signals, providing enriched representations as a robust foundation for disentanglement. The encoder then dynamically disentangles features into domain-shared and -specific spaces, preserving collaborative information during the separation process. To handle challenge (ii), the decoder introduces an anchor-based supervision that leverages hierarchical feature relationships to enhance intra-domain consistency and cross-domain alignment. Extensive experiments on real-world datasets demonstrate that DGCDR achieves state-of-the-art performance, with improvements of up to 11.59% across key metrics. Qualitative analyses further validate its superior disentanglement quality and transferability. Our source code and datasets are available on GitHub for further comparison. 

---
# VL-CLIP: Enhancing Multimodal Recommendations via Visual Grounding and LLM-Augmented CLIP Embeddings 

**Authors**: Ramin Giahi, Kehui Yao, Sriram Kollipara, Kai Zhao, Vahid Mirjalili, Jianpeng Xu, Topojoy Biswas, Evren Korpeoglu, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17080)  

**Abstract**: Multimodal learning plays a critical role in e-commerce recommendation platforms today, enabling accurate recommendations and product understanding. However, existing vision-language models, such as CLIP, face key challenges in e-commerce recommendation systems: 1) Weak object-level alignment, where global image embeddings fail to capture fine-grained product attributes, leading to suboptimal retrieval performance; 2) Ambiguous textual representations, where product descriptions often lack contextual clarity, affecting cross-modal matching; and 3) Domain mismatch, as generic vision-language models may not generalize well to e-commerce-specific data. To address these limitations, we propose a framework, VL-CLIP, that enhances CLIP embeddings by integrating Visual Grounding for fine-grained visual understanding and an LLM-based agent for generating enriched text embeddings. Visual Grounding refines image representations by localizing key products, while the LLM agent enhances textual features by disambiguating product descriptions. Our approach significantly improves retrieval accuracy, multimodal retrieval effectiveness, and recommendation quality across tens of millions of items on one of the largest e-commerce platforms in the U.S., increasing CTR by 18.6%, ATC by 15.5%, and GMV by 4.0%. Additional experimental results show that our framework outperforms vision-language models, including CLIP, FashionCLIP, and GCL, in both precision and semantic alignment, demonstrating the potential of combining object-aware visual grounding and LLM-enhanced text representation for robust multimodal recommendations. 

---
# LLM4MEA: Data-free Model Extraction Attacks on Sequential Recommenders via Large Language Models 

**Authors**: Shilong Zhao, Fei Sun, Kaike Zhang, Shaoling Jing, Du Su, Zhichao Shi, Zhiyi Yin, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.16969)  

**Abstract**: Recent studies have demonstrated the vulnerability of sequential recommender systems to Model Extraction Attacks (MEAs). MEAs collect responses from recommender systems to replicate their functionality, enabling unauthorized deployments and posing critical privacy and security risks. Black-box attacks in prior MEAs are ineffective at exposing recommender system vulnerabilities due to random sampling in data selection, which leads to misaligned synthetic and real-world distributions. To overcome this limitation, we propose LLM4MEA, a novel model extraction method that leverages Large Language Models (LLMs) as human-like rankers to generate data. It generates data through interactions between the LLM ranker and target recommender system. In each interaction, the LLM ranker analyzes historical interactions to understand user behavior, and selects items from recommendations with consistent preferences to extend the interaction history, which serves as training data for MEA. Extensive experiments demonstrate that LLM4MEA significantly outperforms existing approaches in data quality and attack performance, reducing the divergence between synthetic and real-world data by up to 64.98% and improving MEA performance by 44.82% on average. From a defensive perspective, we propose a simple yet effective defense strategy and identify key hyperparameters of recommender systems that can mitigate the risk of MEAs. 

---
# You Don't Bring Me Flowers: Mitigating Unwanted Recommendations Through Conformal Risk Control 

**Authors**: Giovanni De Toni, Erasmo Purificato, Emilia Gómez, Bruno Lepri, Andrea Passerini, Cristian Consonni  

**Link**: [PDF](https://arxiv.org/pdf/2507.16829)  

**Abstract**: Recommenders are significantly shaping online information consumption. While effective at personalizing content, these systems increasingly face criticism for propagating irrelevant, unwanted, and even harmful recommendations. Such content degrades user satisfaction and contributes to significant societal issues, including misinformation, radicalization, and erosion of user trust. Although platforms offer mechanisms to mitigate exposure to undesired content, these mechanisms are often insufficiently effective and slow to adapt to users' feedback. This paper introduces an intuitive, model-agnostic, and distribution-free method that uses conformal risk control to provably bound unwanted content in personalized recommendations by leveraging simple binary feedback on items. We also address a limitation of traditional conformal risk control approaches, i.e., the fact that the recommender can provide a smaller set of recommended items, by leveraging implicit feedback on consumed items to expand the recommendation set while ensuring robust risk mitigation. Our experimental evaluation on data coming from a popular online video-sharing platform demonstrates that our approach ensures an effective and controllable reduction of unwanted recommendations with minimal effort. The source code is available here: this https URL. 

---
# A Query-Aware Multi-Path Knowledge Graph Fusion Approach for Enhancing Retrieval-Augmented Generation in Large Language Models 

**Authors**: Qikai Wei, Huansheng Ning, Chunlong Han, Jianguo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.16826)  

**Abstract**: Retrieval Augmented Generation (RAG) has gradually emerged as a promising paradigm for enhancing the accuracy and factual consistency of content generated by large language models (LLMs). However, existing RAG studies primarily focus on retrieving isolated segments using similarity-based matching methods, while overlooking the intrinsic connections between them. This limitation hampers performance in RAG tasks. To address this, we propose QMKGF, a Query-Aware Multi-Path Knowledge Graph Fusion Approach for Enhancing Retrieval Augmented Generation. First, we design prompt templates and employ general-purpose LLMs to extract entities and relations, thereby generating a knowledge graph (KG) efficiently. Based on the constructed KG, we introduce a multi-path subgraph construction strategy that incorporates one-hop relations, multi-hop relations, and importance-based relations, aiming to improve the semantic relevance between the retrieved documents and the user query. Subsequently, we designed a query-aware attention reward model that scores subgraph triples based on their semantic relevance to the query. Then, we select the highest score subgraph and enrich subgraph with additional triples from other subgraphs that are highly semantically relevant to the query. Finally, the entities, relations, and triples within the updated subgraph are utilised to expand the original query, thereby enhancing its semantic representation and improving the quality of LLMs' generation. We evaluate QMKGF on the SQuAD, IIRC, Culture, HotpotQA, and MuSiQue datasets. On the HotpotQA dataset, our method achieves a ROUGE-1 score of 64.98\%, surpassing the BGE-Rerank approach by 9.72 percentage points (from 55.26\% to 64.98\%). Experimental results demonstrate the effectiveness and superiority of the QMKGF approach. 

---
# On Function-Correcting Codes in the Lee Metric 

**Authors**: Gyanendra K. Verma, Abhay Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.17654)  

**Abstract**: Function-correcting codes are a coding framework designed to minimize redundancy while ensuring that specific functions or computations of encoded data can be reliably recovered, even in the presence of errors. The choice of metric is crucial in designing such codes, as it determines which computations must be protected and how errors are measured and corrected. Previous work by Liu and Liu [6] studied function-correcting codes over $\mathbb{Z}_{2^l},\ l\geq 2$ using the homogeneous metric, which coincides with the Lee metric over $\mathbb{Z}_4$. In this paper, we extend the study to codes over $\mathbb{Z}_m,$ for any positive integer $m\geq 2$ under the Lee metric and aim to determine their optimal redundancy. To achieve this, we introduce irregular Lee distance codes and derive upper and lower bounds on the optimal redundancy by characterizing the shortest possible length of such codes. These general bounds are then simplified and applied to specific classes of functions, including Lee-local functions, Lee weight functions, and Lee weight distribution functions, leading to improved some bounds compared to those of Liu and Liu [6] over $\mathbb{Z}_4$ and generalize the other bounds over $\mathbb{Z}_m$ in the Lee metric. 

---
# BGM-HAN: A Hierarchical Attention Network for Accurate and Fair Decision Assessment on Semi-Structured Profiles 

**Authors**: Junhua Liu, Roy Ka-Wei Lee, Kwan Hui Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.17472)  

**Abstract**: Human decision-making in high-stakes domains often relies on expertise and heuristics, but is vulnerable to hard-to-detect cognitive biases that threaten fairness and long-term outcomes. This work presents a novel approach to enhancing complex decision-making workflows through the integration of hierarchical learning alongside various enhancements. Focusing on university admissions as a representative high-stakes domain, we propose BGM-HAN, an enhanced Byte-Pair Encoded, Gated Multi-head Hierarchical Attention Network, designed to effectively model semi-structured applicant data. BGM-HAN captures multi-level representations that are crucial for nuanced assessment, improving both interpretability and predictive performance. Experimental results on real admissions data demonstrate that our proposed model significantly outperforms both state-of-the-art baselines from traditional machine learning to large language models, offering a promising framework for augmenting decision-making in domains where structure, context, and fairness matter. Source code is available at: this https URL. 

---
# Content-based 3D Image Retrieval and a ColBERT-inspired Re-ranking for Tumor Flagging and Staging 

**Authors**: Farnaz Khun Jush, Steffen Vogler, Matthias Lenga  

**Link**: [PDF](https://arxiv.org/pdf/2507.17412)  

**Abstract**: The increasing volume of medical images poses challenges for radiologists in retrieving relevant cases. Content-based image retrieval (CBIR) systems offer potential for efficient access to similar cases, yet lack standardized evaluation and comprehensive studies. Building on prior studies for tumor characterization via CBIR, this study advances CBIR research for volumetric medical images through three key contributions: (1) a framework eliminating reliance on pre-segmented data and organ-specific datasets, aligning with large and unstructured image archiving systems, i.e. PACS in clinical practice; (2) introduction of C-MIR, a novel volumetric re-ranking method adapting ColBERT's contextualized late interaction mechanism for 3D medical imaging; (3) comprehensive evaluation across four tumor sites using three feature extractors and three database configurations. Our evaluations highlight the significant advantages of C-MIR. We demonstrate the successful adaptation of the late interaction principle to volumetric medical images, enabling effective context-aware re-ranking. A key finding is C-MIR's ability to effectively localize the region of interest, eliminating the need for pre-segmentation of datasets and offering a computationally efficient alternative to systems relying on expensive data enrichment steps. C-MIR demonstrates promising improvements in tumor flagging, achieving improved performance, particularly for colon and lung tumors (p<0.05). C-MIR also shows potential for improving tumor staging, warranting further exploration of its capabilities. Ultimately, our work seeks to bridge the gap between advanced retrieval techniques and their practical applications in healthcare, paving the way for improved diagnostic processes. 

---
# HLFormer: Enhancing Partially Relevant Video Retrieval with Hyperbolic Learning 

**Authors**: Li Jun, Wang Jinpeng, Tan Chaolei, Lian Niu, Chen Long, Zhang Min, Wang Yaowei, Xia Shu-Tao, Chen Bin  

**Link**: [PDF](https://arxiv.org/pdf/2507.17402)  

**Abstract**: Partially Relevant Video Retrieval (PRVR) addresses the critical challenge of matching untrimmed videos with text queries describing only partial content. Existing methods suffer from geometric distortion in Euclidean space that sometimes misrepresents the intrinsic hierarchical structure of videos and overlooks certain hierarchical semantics, ultimately leading to suboptimal temporal modeling. To address this issue, we propose the first hyperbolic modeling framework for PRVR, namely HLFormer, which leverages hyperbolic space learning to compensate for the suboptimal hierarchical modeling capabilities of Euclidean space. Specifically, HLFormer integrates the Lorentz Attention Block and Euclidean Attention Block to encode video embeddings in hybrid spaces, using the Mean-Guided Adaptive Interaction Module to dynamically fuse features. Additionally, we introduce a Partial Order Preservation Loss to enforce "text < video" hierarchy through Lorentzian cone constraints. This approach further enhances cross-modal matching by reinforcing partial relevance between video content and text queries. Extensive experiments show that HLFormer outperforms state-of-the-art methods. Code is released at this https URL. 

---
# Millions of $\text{GeAR}$-s: Extending GraphRAG to Millions of Documents 

**Authors**: Zhili Shen, Chenxin Diao, Pascual Merita, Pavlos Vougiouklis, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17399)  

**Abstract**: Recent studies have explored graph-based approaches to retrieval-augmented generation, leveraging structured or semi-structured information -- such as entities and their relations extracted from documents -- to enhance retrieval. However, these methods are typically designed to address specific tasks, such as multi-hop question answering and query-focused summarisation, and therefore, there is limited evidence of their general applicability across broader datasets. In this paper, we aim to adapt a state-of-the-art graph-based RAG solution: $\text{GeAR}$ and explore its performance and limitations on the SIGIR 2025 LiveRAG Challenge. 

---
# Triadic First-Order Logic Queries in Temporal Networks 

**Authors**: Omkar Bhalerao, Yunjie Pan, C. Seshadhri, Nishil Talati  

**Link**: [PDF](https://arxiv.org/pdf/2507.17215)  

**Abstract**: Motif counting is a fundamental problem in network analysis, and there is a rich literature of theoretical and applied algorithms for this problem. Given a large input network $G$, a motif $H$ is a small "pattern" graph indicative of special local structure. Motif/pattern mining involves finding all matches of this pattern in the input $G$. The simplest, yet challenging, case of motif counting is when $H$ has three vertices, often called a "triadic" query. Recent work has focused on "temporal graph mining", where the network $G$ has edges with timestamps (and directions) and $H$ has time constraints.
Inspired by concepts in logic and database theory, we introduce the study of "thresholded First Order Logic (FOL) Motif Analysis" for massive temporal networks. A typical triadic motif query asks for the existence of three vertices that form a desired temporal pattern. An "FOL" motif query is obtained by having both existential and thresholded universal quantifiers. This allows for query semantics that can mine richer information from networks. A typical triadic query would be "find all triples of vertices $u,v,w$ such that they form a triangle within one hour". A thresholded FOL query can express "find all pairs $u,v$ such that for half of $w$ where $(u,w)$ formed an edge, $(v,w)$ also formed an edge within an hour".
We design the first algorithm, FOLTY, for mining thresholded FOL triadic queries. The theoretical running time of FOLTY matches the best known running time for temporal triangle counting in sparse graphs. We give an efficient implementation of FOLTY using specialized temporal data structures. FOLTY has excellent empirical behavior, and can answer triadic FOL queries on graphs with nearly 70M edges is less than hour on commodity hardware. Our work has the potential to start a new research direction in the classic well-studied problem of motif analysis. 

---
# Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems 

**Authors**: Chengxuan Xia, Qianye Wu, Sixuan Tian, Yilun Hao  

**Link**: [PDF](https://arxiv.org/pdf/2507.17061)  

**Abstract**: Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems. 

---
# Multi-Label Classification with Generative AI Models in Healthcare: A Case Study of Suicidality and Risk Factors 

**Authors**: Ming Huang, Zehan Li, Yan Hu, Wanjing Wang, Andrew Wen, Scott Lane, Salih Selek, Lokesh Shahani, Rodrigo Machado-Vieira, Jair Soares, Hua Xu, Hongfang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.17009)  

**Abstract**: Suicide remains a pressing global health crisis, with over 720,000 deaths annually and millions more affected by suicide ideation (SI) and suicide attempts (SA). Early identification of suicidality-related factors (SrFs), including SI, SA, exposure to suicide (ES), and non-suicidal self-injury (NSSI), is critical for timely intervention. While prior studies have applied AI to detect SrFs in clinical notes, most treat suicidality as a binary classification task, overlooking the complexity of cooccurring risk factors. This study explores the use of generative large language models (LLMs), specifically GPT-3.5 and GPT-4.5, for multi-label classification (MLC) of SrFs from psychiatric electronic health records (EHRs). We present a novel end to end generative MLC pipeline and introduce advanced evaluation methods, including label set level metrics and a multilabel confusion matrix for error analysis. Finetuned GPT-3.5 achieved top performance with 0.94 partial match accuracy and 0.91 F1 score, while GPT-4.5 with guided prompting showed superior performance across label sets, including rare or minority label sets, indicating a more balanced and robust performance. Our findings reveal systematic error patterns, such as the conflation of SI and SA, and highlight the models tendency toward cautious over labeling. This work not only demonstrates the feasibility of using generative AI for complex clinical classification tasks but also provides a blueprint for structuring unstructured EHR data to support large scale clinical research and evidence based medicine. 

---
# Text-to-SPARQL Goes Beyond English: Multilingual Question Answering Over Knowledge Graphs through Human-Inspired Reasoning 

**Authors**: Aleksandr Perevalov, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2507.16971)  

**Abstract**: Accessing knowledge via multilingual natural-language interfaces is one of the emerging challenges in the field of information retrieval and related ones. Structured knowledge stored in knowledge graphs can be queried via a specific query language (e.g., SPARQL). Therefore, one needs to transform natural-language input into a query to fulfill an information need. Prior approaches mostly focused on combining components (e.g., rule-based or neural-based) that solve downstream tasks and come up with an answer at the end. We introduce mKGQAgent, a human-inspired framework that breaks down the task of converting natural language questions into SPARQL queries into modular, interpretable subtasks. By leveraging a coordinated LLM agent workflow for planning, entity linking, and query refinement - guided by an experience pool for in-context learning - mKGQAgent efficiently handles multilingual KGQA. Evaluated on the DBpedia- and Corporate-based KGQA benchmarks within the Text2SPARQL challenge 2025, our approach took first place among the other participants. This work opens new avenues for developing human-like reasoning systems in multilingual semantic parsing. 

---
# EVOLVE-X: Embedding Fusion and Language Prompting for User Evolution Forecasting on Social Media 

**Authors**: Ismail Hossain, Sai Puppala, Md Jahangir Alam, Sajedul Talukder  

**Link**: [PDF](https://arxiv.org/pdf/2507.16847)  

**Abstract**: Social media platforms serve as a significant medium for sharing personal emotions, daily activities, and various life events, ensuring individuals stay informed about the latest developments. From the initiation of an account, users progressively expand their circle of friends or followers, engaging actively by posting, commenting, and sharing content. Over time, user behavior on these platforms evolves, influenced by demographic attributes and the networks they form. In this study, we present a novel approach that leverages open-source models Llama-3-Instruct, Mistral-7B-Instruct, Gemma-7B-IT through prompt engineering, combined with GPT-2, BERT, and RoBERTa using a joint embedding technique, to analyze and predict the evolution of user behavior on social media over their lifetime. Our experiments demonstrate the potential of these models to forecast future stages of a user's social evolution, including network changes, future connections, and shifts in user activities. Experimental results highlight the effectiveness of our approach, with GPT-2 achieving the lowest perplexity (8.21) in a Cross-modal configuration, outperforming RoBERTa (9.11) and BERT, and underscoring the importance of leveraging Cross-modal configurations for superior performance. This approach addresses critical challenges in social media, such as friend recommendations and activity predictions, offering insights into the trajectory of user behavior. By anticipating future interactions and activities, this research aims to provide early warnings about potential negative outcomes, enabling users to make informed decisions and mitigate risks in the long term. 

---
