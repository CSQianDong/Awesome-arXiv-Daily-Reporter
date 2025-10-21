# Towards Mixed-Modal Retrieval for Universal Retrieval-Augmented Generation 

**Authors**: Chenghao Zhang, Guanting Dong, Xinyu Yang, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2510.17354)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing large language models (LLMs) by retrieving relevant documents from an external corpus. However, existing RAG systems primarily focus on unimodal text documents, and often fall short in real-world scenarios where both queries and documents may contain mixed modalities (such as text and images). In this paper, we address the challenge of Universal Retrieval-Augmented Generation (URAG), which involves retrieving and reasoning over mixed-modal information to improve vision-language generation. To this end, we propose Nyx, a unified mixed-modal to mixed-modal retriever tailored for URAG scenarios. To mitigate the scarcity of realistic mixed-modal data, we introduce a four-stage automated pipeline for generation and filtering, leveraging web documents to construct NyxQA, a dataset comprising diverse mixed-modal question-answer pairs that better reflect real-world information needs. Building on this high-quality dataset, we adopt a two-stage training framework for Nyx: we first perform pre-training on NyxQA along with a variety of open-source retrieval datasets, followed by supervised fine-tuning using feedback from downstream vision-language models (VLMs) to align retrieval outputs with generative preferences. Experimental results demonstrate that Nyx not only performs competitively on standard text-only RAG benchmarks, but also excels in the more general and realistic URAG setting, significantly improving generation quality in vision-language tasks. 

---
# ScholarEval: Research Idea Evaluation Grounded in Literature 

**Authors**: Hanane Nour Moussa, Patrick Queiroz Da Silva, Daniel Adu-Ampratwum, Alyson East, Zitong Lu, Nikki Puccetti, Mingyi Xue, Huan Sun, Bodhisattwa Prasad Majumder, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.16234)  

**Abstract**: As AI tools become increasingly common for research ideation, robust evaluation is critical to ensure the validity and usefulness of generated ideas. We introduce ScholarEval, a retrieval augmented evaluation framework that assesses research ideas based on two fundamental criteria: soundness - the empirical validity of proposed methods based on existing literature, and contribution - the degree of advancement made by the idea across different dimensions relative to prior research. To evaluate ScholarEval, we introduce ScholarIdeas, the first expert-annotated dataset of multi-domain research ideas and reviews, comprised of 117 ideas across four disciplines: artificial intelligence, neuroscience, biochemistry, and ecology. Our evaluation shows that ScholarEval achieves significantly higher coverage of points mentioned in the human expert annotated rubrics in ScholarIdeas compared to all baselines. Furthermore, ScholarEval is consistently preferred over our strongest baseline o4-mini-deep-research, a reasoning and search-enabled agentic system by OpenAI, in terms of evaluation actionability, depth, and evidence support. Our large-scale user study also shows that ScholarEval significantly outperforms deep research in literature engagement, idea refinement, and usefulness. We openly release our code, dataset, and ScholarEval tool for the community to use and build on. 

---
# A Brain Cell Type Resource Created by Large Language Models and a Multi-Agent AI System for Collaborative Community Annotation 

**Authors**: Rongbin Li, Wenbo Chen, Zhao Li, Rodrigo Munoz-Castaneda, Jinbo Li, Neha S. Maurya, Arnav Solanki, Huan He, Hanwen Xing, Meaghan Ramlakhan, Zachary Wise, Zhuhao Wu, Hua Xu, Michael Hawrylycz, W. Jim Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.17064)  

**Abstract**: Single-cell RNA sequencing has transformed our ability to identify diverse cell types and their transcriptomic signatures. However, annotating these signatures-especially those involving poorly characterized genes-remains a major challenge. Traditional methods, such as Gene Set Enrichment Analysis (GSEA), depend on well-curated annotations and often perform poorly in these contexts. Large Language Models (LLMs) offer a promising alternative but struggle to represent complex biological knowledge within structured ontologies. To address this, we present BRAINCELL-AID (BRAINCELL-AID: this https URL), a novel multi-agent AI system that integrates free-text descriptions with ontology labels to enable more accurate and robust gene set annotation. By incorporating retrieval-augmented generation (RAG), we developed a robust agentic workflow that refines predictions using relevant PubMed literature, reducing hallucinations and enhancing interpretability. Using this workflow, we achieved correct annotations for 77% of mouse gene sets among their top predictions. Applying this approach, we annotated 5,322 brain cell clusters from the comprehensive mouse brain cell atlas generated by the BRAIN Initiative Cell Census Network, enabling novel insights into brain cell function by identifying region-specific gene co-expression patterns and inferring functional roles of gene ensembles. BRAINCELL-AID also identifies Basal Ganglia-related cell types with neurologically meaningful descriptions. Hence, we create a valuable resource to support community-driven cell type annotation. 

---
# Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need? 

**Authors**: Junchi Yu, Yujie Liu, Jindong Gu, Philip Torr, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.16582)  

**Abstract**: Retrieval-Augmented Generation (RAG) based on knowledge graphs (KGs) enhances large language models (LLMs) by providing structured and interpretable external knowledge. However, existing KG-based RAG methods struggle to retrieve accurate and diverse information from text-rich KGs for complex real-world queries. Process Reward Models (PRMs) offer a way to align the retrieval process of KG-based RAG with query-specific knowledge requirements, but they heavily rely on process-level supervision signals that are expensive and hard to obtain on KGs. To address this challenge, we propose GraphFlow, a framework that efficiently retrieves accurate and diverse knowledge required for real-world queries from text-rich KGs. GraphFlow employs a transition-based flow matching objective to jointly optimize a retrieval policy and a flow estimator. The flow estimator factorizes the reward of the retrieval outcome into the intermediate retrieval states. Such reward factorization guides the retrieval policy to retrieve candidates from KGs in proportion to their reward. This allows GraphFlow to explore high-quality regions of KGs that yield diverse and relevant results. We evaluate GraphFlow on the STaRK benchmark, which includes real-world queries from multiple domains over text-rich KGs. GraphFlow outperforms strong KG-RAG baselines, including GPT-4o, by 10% on average in hit rate and recall. It also shows strong generalization to unseen KGs, demonstrating its effectiveness and robustness. 

---
# Right Answer at the Right Time - Temporal Retrieval-Augmented Generation via Graph Summarization 

**Authors**: Zulun Zhu, Haoyu Liu, Mengke He, Siqiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.16715)  

**Abstract**: Question answering in temporal knowledge graphs requires retrieval that is both time-consistent and efficient. Existing RAG methods are largely semantic and typically neglect explicit temporal constraints, which leads to time-inconsistent answers and inflated token usage. We propose STAR-RAG, a temporal GraphRAG framework that relies on two key ideas: building a time-aligned rule graph and conducting propagation on this graph to narrow the search space and prioritize semantically relevant, time-consistent evidence. This design enforces temporal proximity during retrieval, reduces the candidate set of retrieval results, and lowers token consumption without sacrificing accuracy. Compared with existing temporal RAG approaches, STAR-RAG eliminates the need for heavy model training and fine-tuning, thereby reducing computational cost and significantly simplifying this http URL experiments on real-world temporal KG datasets show that our method achieves improved answer accuracy while consuming fewer tokens than strong GraphRAG baselines. 

---
