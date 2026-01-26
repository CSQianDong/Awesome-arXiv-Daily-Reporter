# Explaining Group Recommendations via Counterfactuals 

**Authors**: Maria Stratigi, Nikos Bikakis  

**Link**: [PDF](https://arxiv.org/pdf/2601.16882)  

**Abstract**: Group recommender systems help users make collective choices but often lack transparency, leaving group members uncertain about why items are suggested. Existing explanation methods focus on individuals, offering limited support for groups where multiple preferences interact. In this paper, we propose a framework for group counterfactual explanations, which reveal how removing specific past interactions would change a group recommendation. We formalize this concept, introduce utility and fairness measures tailored to groups, and design heuristic algorithms, such as Pareto-based filtering and grow-and-prune strategies, for efficient explanation discovery. Experiments on MovieLens and Amazon datasets show clear trade-offs: low-cost methods produce larger, less fair explanations, while other approaches yield concise and balanced results at higher cost. Furthermore, the Pareto-filtering heuristic demonstrates significant efficiency improvements in sparse settings. 

---
# From Atom to Community: Structured and Evolving Agent Memory for User Behavior Modeling 

**Authors**: Yuxin Liao, Le Wu, Min Hou, Yu Wang, Han Wu, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.16872)  

**Abstract**: User behavior modeling lies at the heart of personalized applications like recommender systems. With LLM-based agents, user preference representation has evolved from latent embeddings to semantic memory. While existing memory mechanisms show promise in textual dialogues, modeling non-textual behaviors remains challenging, as preferences must be inferred from implicit signals like clicks without ground truth supervision. Current approaches rely on a single unstructured summary, updated through simple overwriting. However, this is suboptimal: users exhibit multi-faceted interests that get conflated, preferences evolve yet naive overwriting causes forgetting, and sparse individual interactions necessitate collaborative signals. We present STEAM (\textit{\textbf{ST}ructured and \textbf{E}volving \textbf{A}gent \textbf{M}emory}), a novel framework that reimagines how agent memory is organized and updated. STEAM decomposes preferences into atomic memory units, each capturing a distinct interest dimension with explicit links to observed behaviors. To exploit collaborative patterns, STEAM organizes similar memories across users into communities and generates prototype memories for signal propagation. The framework further incorporates adaptive evolution mechanisms, including consolidation for refining memories and formation for capturing emerging interests. Experiments on three real-world datasets demonstrate that STEAM substantially outperforms state-of-the-art baselines in recommendation accuracy, simulation fidelity, and diversity. 

---
# Navigating the Shift: A Comparative Analysis of Web Search and Generative AI Response Generation 

**Authors**: Mahe Chen, Xiaoxuan Wang, Kaiwen Chen, Nick Koudas  

**Link**: [PDF](https://arxiv.org/pdf/2601.16858)  

**Abstract**: The rise of generative AI as a primary information source presents a paradigm shift from traditional web search. This paper presents a large-scale empirical study quantifying the fundamental differences between the results returned by Google Search and leading generative AI services. We analyze multiple dimensions, demonstrating that AI-generated answers and web search results diverge significantly in their consulted source domains, the typology of these domains (e.g., earned media vs. owned, social), query intent and the freshness of the information provided. We then investigate the role of LLM pre-training as a key factor shaping these differences, analyzing how this intrinsic knowledge base interacts with and influences real-time web search when enabled. Our findings reveal the distinct mechanics of these two information ecosystems, leading to critical observations on the emergent field of Answer Engine Optimization (AEO) and its contrast with traditional Search Engine Optimization (SEO). 

---
# PI2I: A Personalized Item-Based Collaborative Filtering Retrieval Framework 

**Authors**: Shaoqing Wang, Yingcai Ma, Kairui Fu, Ziyang Wang, Dunxian Huang, Yuliang Yan, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.16815)  

**Abstract**: Efficiently selecting relevant content from vast candidate pools is a critical challenge in modern recommender systems. Traditional methods, such as item-to-item collaborative filtering (CF) and two-tower models, often fall short in capturing the complex user-item interactions due to uniform truncation strategies and overdue user-item crossing. To address these limitations, we propose Personalized Item-to-Item (PI2I), a novel two-stage retrieval framework that enhances the personalization capabilities of CF. In the first Indexer Building Stage (IBS), we optimize the retrieval pool by relaxing truncation thresholds to maximize Hit Rate, thereby temporarily retaining more items users might be interested in. In the second Personalized Retrieval Stage (PRS), we introduce an interactive scoring model to overcome the limitations of inner product calculations, allowing for richer modeling of intricate user-item interactions. Additionally, we construct negative samples based on the trigger-target (item-to-item) relationship, ensuring consistency between offline training and online inference. Offline experiments on large-scale real-world datasets demonstrate that PI2I outperforms traditional CF methods and rivals Two-Tower models. Deployed in the "Guess You Like" section on Taobao, PI2I achieved a 1.05% increase in online transaction rates. In addition, we have released a large-scale recommendation dataset collected from Taobao, containing 130 million real-world user interactions used in the experiments of this paper. The dataset is publicly available at this https URL, which could serve as a valuable benchmark for the research community. 

---
# LLM-powered Real-time Patent Citation Recommendation for Financial Technologies 

**Authors**: Tianang Deng, Yu Deng, Tianchen Gao, Yonghong Hu, Rui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2601.16775)  

**Abstract**: Rapid financial innovation has been accompanied by a sharp increase in patenting activity, making timely and comprehensive prior-art discovery more difficult. This problem is especially evident in financial technologies, where innovations develop quickly, patent collections grow continuously, and citation recommendation systems must be updated as new applications arrive. Existing patent retrieval and citation recommendation methods typically rely on static indexes or periodic retraining, which limits their ability to operate effectively in such dynamic settings. In this study, we propose a real-time patent citation recommendation framework designed for large and fast-changing financial patent corpora. Using a dataset of 428,843 financial patents granted by the China National Intellectual Property Administration (CNIPA) between 2000 and 2024, we build a three-stage recommendation pipeline. The pipeline uses large language model (LLM) embeddings to represent the semantic content of patent abstracts, applies efficient approximate nearest-neighbor search to construct a manageable candidate set, and ranks candidates by semantic similarity to produce top-k citation recommendations. In addition to improving recommendation accuracy, the proposed framework directly addresses the dynamic nature of patent systems. By using an incremental indexing strategy based on hierarchical navigable small-world (HNSW) graphs, newly issued patents can be added without rebuilding the entire index. A rolling day-by-day update experiment shows that incremental updating improves recall while substantially reducing computational cost compared with rebuild-based indexing. The proposed method also consistently outperforms traditional text-based baselines and alternative nearest-neighbor retrieval approaches. 

---
# PRISM: Purified Representation and Integrated Semantic Modeling for Generative Sequential Recommendation 

**Authors**: Dengzhao Fang, Jingtong Gao, Yu Li, Xiangyu Zhao, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2601.16556)  

**Abstract**: Generative Sequential Recommendation (GSR) has emerged as a promising paradigm, reframing recommendation as an autoregressive sequence generation task over discrete Semantic IDs (SIDs), typically derived via codebook-based quantization. Despite its great potential in unifying retrieval and ranking, existing GSR frameworks still face two critical limitations: (1) impure and unstable semantic tokenization, where quantization methods struggle with interaction noise and codebook collapse, resulting in SIDs with ambiguous discrimination; and (2) lossy and weakly structured generation, where reliance solely on coarse-grained discrete tokens inevitably introduces information loss and neglects items' hierarchical logic. To address these issues, we propose a novel generative recommendation framework, PRISM, with Purified Representation and Integrated Semantic Modeling. Specifically, to ensure high-quality tokenization, we design a Purified Semantic Quantizer that constructs a robust codebook via adaptive collaborative denoising and hierarchical semantic anchoring mechanisms. To compensate for information loss during quantization, we further propose an Integrated Semantic Recommender, which incorporates a dynamic semantic integration mechanism to integrate fine-grained semantics and enforces logical validity through a semantic structure alignment objective. PRISM consistently outperforms state-of-the-art baselines across four real-world datasets, demonstrating substantial performance gains, particularly in high-sparsity scenarios. 

---
# LLM-based Semantic Search for Conversational Queries in E-commerce 

**Authors**: Emad Siddiqui, Venkatesh Terikuti, Xuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.16492)  

**Abstract**: Conversational user queries are increasingly challenging traditional e-commerce platforms, whose search systems are typically optimized for keyword-based queries. We present an LLM-based semantic search framework that effectively captures user intent from conversational queries by combining domain-specific embeddings with structured filters. To address the challenge of limited labeled data, we generate synthetic data using LLMs to guide the fine-tuning of two models: an embedding model that positions semantically similar products close together in the representation space, and a generative model for converting natural language queries into structured constraints. By combining similarity-based retrieval with constraint-based filtering, our framework achieves strong precision and recall across various settings compared to baseline approaches on a real-world dataset. 

---
# Empowering Medical Equipment Sustainability in Low-Resource Settings: An AI-Powered Diagnostic and Support Platform for Biomedical Technicians 

**Authors**: Bernes Lorier Atabonfack, Ahmed Tahiru Issah, Mohammed Hardi Abdul Baaki, Clemence Ingabire, Tolulope Olusuyi, Maruf Adewole, Udunna C. Anazodo, Timothy X Brown  

**Link**: [PDF](https://arxiv.org/pdf/2601.16967)  

**Abstract**: In low- and middle-income countries (LMICs), a significant proportion of medical diagnostic equipment remains underutilized or non-functional due to a lack of timely maintenance, limited access to technical expertise, and minimal support from manufacturers, particularly for devices acquired through third-party vendors or donations. This challenge contributes to increased equipment downtime, delayed diagnoses, and compromised patient care. This research explores the development and validation of an AI-powered support platform designed to assist biomedical technicians in diagnosing and repairing medical devices in real-time. The system integrates a large language model (LLM) with a user-friendly web interface, enabling imaging technologists/radiographers and biomedical technicians to input error codes or device symptoms and receive accurate, step-by-step troubleshooting guidance. The platform also includes a global peer-to-peer discussion forum to support knowledge exchange and provide additional context for rare or undocumented issues. A proof of concept was developed using the Philips HDI 5000 ultrasound machine, achieving 100% precision in error code interpretation and 80% accuracy in suggesting corrective actions. This study demonstrates the feasibility and potential of AI-driven systems to support medical device maintenance, with the aim of reducing equipment downtime to improve healthcare delivery in resource-constrained environments. 

---
# Better Generalizing to Unseen Concepts: An Evaluation Framework and An LLM-Based Auto-Labeled Pipeline for Biomedical Concept Recognition 

**Authors**: Shanshan Liu, Noriki Nishida, Fei Cheng, Narumi Tokunaga, Rumana Ferdous Munne, Yuki Yamagata, Kouji Kozaki, Takehito Utsuro, Yuji Matsumoto  

**Link**: [PDF](https://arxiv.org/pdf/2601.16711)  

**Abstract**: Generalization to unseen concepts is a central challenge due to the scarcity of human annotations in Mention-agnostic Biomedical Concept Recognition (MA-BCR). This work makes two key contributions to systematically address this issue. First, we propose an evaluation framework built on hierarchical concept indices and novel metrics to measure generalization. Second, we explore LLM-based Auto-Labeled Data (ALD) as a scalable resource, creating a task-specific pipeline for its generation. Our research unequivocally shows that while LLM-generated ALD cannot fully substitute for manual annotations, it is a valuable resource for improving generalization, successfully providing models with the broader coverage and structural knowledge needed to approach recognizing unseen concepts. Code and datasets are available at this https URL. 

---
# Segregation Before Polarization: How Recommendation Strategies Shape Echo Chamber Pathways 

**Authors**: Junning Zhao, Kazutoshi Sasahara, Yu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.16457)  

**Abstract**: Social media platforms facilitate echo chambers through feedback loops between user preferences and recommendation algorithms. While algorithmic homogeneity is well-documented, the distinct evolutionary pathways driven by content-based versus link-based recommendations remain unclear. Using an extended dynamic Bounded Confidence Model (BCM), we show that content-based algorithms--unlike their link-based counterparts--steer social networks toward a segregation-before-polarization (SbP) pathway. Along this trajectory, structural segregation precedes opinion divergence, accelerating individual isolation while delaying but ultimately intensifying collective polarization. Furthermore, we reveal a paradox in information sharing: Reposting increases the number of connections in the network, yet it simultaneously reinforces echo chambers because it amplifies small, latent opinion differences that would otherwise remain inconsequential. These findings suggest that mitigating polarization requires stage-dependent algorithmic interventions, shifting from content-centric to structure-centric strategies as networks evolve. 

---
