# Towards a rigorous evaluation of RAG systems: the challenge of due diligence 

**Authors**: Grégoire Martinon, Alexandra Lorenzo de Brionne, Jérôme Bohard, Antoine Lojou, Damien Hervault, Nicolas J-B. Brunel  

**Link**: [PDF](https://arxiv.org/pdf/2507.21753)  

**Abstract**: The rise of generative AI, has driven significant advancements in high-risk sectors like healthcare and finance. The Retrieval-Augmented Generation (RAG) architecture, combining language models (LLMs) with search engines, is particularly notable for its ability to generate responses from document corpora. Despite its potential, the reliability of RAG systems in critical contexts remains a concern, with issues such as hallucinations persisting. This study evaluates a RAG system used in due diligence for an investment fund. We propose a robust evaluation protocol combining human annotations and LLM-Judge annotations to identify system failures, like hallucinations, off-topic, failed citations, and abstentions. Inspired by the Prediction Powered Inference (PPI) method, we achieve precise performance measurements with statistical guarantees. We provide a comprehensive dataset for further analysis. Our contributions aim to enhance the reliability and scalability of RAG systems evaluation protocols in industrial applications. 

---
# SafeDriveRAG: Towards Safe Autonomous Driving with Knowledge Graph-based Retrieval-Augmented Generation 

**Authors**: Hao Ye, Mengshi Qi, Zhaohong Liu, Liang Liu, Huadong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.21585)  

**Abstract**: In this work, we study how vision-language models (VLMs) can be utilized to enhance the safety for the autonomous driving system, including perception, situational understanding, and path planning. However, existing research has largely overlooked the evaluation of these models in traffic safety-critical driving scenarios. To bridge this gap, we create the benchmark (SafeDrive228K) and propose a new baseline based on VLM with knowledge graph-based retrieval-augmented generation (SafeDriveRAG) for visual question answering (VQA). Specifically, we introduce SafeDrive228K, the first large-scale multimodal question-answering benchmark comprising 228K examples across 18 sub-tasks. This benchmark encompasses a diverse range of traffic safety queries, from traffic accidents and corner cases to common safety knowledge, enabling a thorough assessment of the comprehension and reasoning abilities of the models. Furthermore, we propose a plug-and-play multimodal knowledge graph-based retrieval-augmented generation approach that employs a novel multi-scale subgraph retrieval algorithm for efficient information retrieval. By incorporating traffic safety guidelines collected from the Internet, this framework further enhances the model's capacity to handle safety-critical situations. Finally, we conduct comprehensive evaluations on five mainstream VLMs to assess their reliability in safety-sensitive driving tasks. Experimental results demonstrate that integrating RAG significantly improves performance, achieving a +4.73% gain in Traffic Accidents tasks, +8.79% in Corner Cases tasks and +14.57% in Traffic Safety Commonsense across five mainstream VLMs, underscoring the potential of our proposed benchmark and methodology for advancing research in traffic safety. Our source code and data are available at this https URL. 

---
# Validating Pharmacogenomics Generative Artificial Intelligence Query Prompts Using Retrieval-Augmented Generation (RAG) 

**Authors**: Ashley Rector, Keaton Minor, Kamden Minor, Jeff McCormack, Beth Breeden, Ryan Nowers, Jay Dorris  

**Link**: [PDF](https://arxiv.org/pdf/2507.21453)  

**Abstract**: This study evaluated Sherpa Rx, an artificial intelligence tool leveraging large language models and retrieval-augmented generation (RAG) for pharmacogenomics, to validate its performance on key response metrics. Sherpa Rx integrated Clinical Pharmacogenetics Implementation Consortium (CPIC) guidelines with Pharmacogenomics Knowledgebase (PharmGKB) data to generate contextually relevant responses. A dataset (N=260 queries) spanning 26 CPIC guidelines was used to evaluate drug-gene interactions, dosing recommendations, and therapeutic implications. In Phase 1, only CPIC data was embedded. Phase 2 additionally incorporated PharmGKB content. Responses were scored on accuracy, relevance, clarity, completeness (5-point Likert scale), and recall. Wilcoxon signed-rank tests compared accuracy between Phase 1 and Phase 2, and between Phase 2 and ChatGPT-4omini. A 20-question quiz assessed the tool's real-world applicability against other models. In Phase 1 (N=260), Sherpa Rx demonstrated high performance of accuracy 4.9, relevance 5.0, clarity 5.0, completeness 4.8, and recall 0.99. The subset analysis (N=20) showed improvements in accuracy (4.6 vs. 4.4, Phase 2 vs. Phase 1 subset) and completeness (5.0 vs. 4.8). ChatGPT-4omini performed comparably in relevance (5.0) and clarity (4.9) but lagged in accuracy (3.9) and completeness (4.2). Differences in accuracy between Phase 1 and Phase 2 was not statistically significant. However, Phase 2 significantly outperformed ChatGPT-4omini. On the 20-question quiz, Sherpa Rx achieved 90% accuracy, outperforming other models. Integrating additional resources like CPIC and PharmGKB with RAG enhances AI accuracy and performance. This study highlights the transformative potential of generative AI like Sherpa Rx in pharmacogenomics, improving decision-making with accurate, personalized responses. 

---
# Efficacy of AI RAG Tools for Complex Information Extraction and Data Annotation Tasks: A Case Study Using Banks Public Disclosures 

**Authors**: Nicholas Botti, Flora Haberkorn, Charlotte Hoopes, Shaun Khan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21360)  

**Abstract**: We utilize a within-subjects design with randomized task assignments to understand the effectiveness of using an AI retrieval augmented generation (RAG) tool to assist analysts with an information extraction and data annotation task. We replicate an existing, challenging real-world annotation task with complex multi-part criteria on a set of thousands of pages of public disclosure documents from global systemically important banks (GSIBs) with heterogeneous and incomplete information content. We test two treatment conditions. First, a "naive" AI use condition in which annotators use only the tool and must accept the first answer they are given. And second, an "interactive" AI treatment condition where annotators use the tool interactively, and use their judgement to follow-up with additional information if necessary. Compared to the human-only baseline, the use of the AI tool accelerated task execution by up to a factor of 10 and enhanced task accuracy, particularly in the interactive condition. We find that when extrapolated to the full task, these methods could save up to 268 hours compared to the human-only approach. Additionally, our findings suggest that annotator skill, not just with the subject matter domain, but also with AI tools, is a factor in both the accuracy and speed of task performance. 

---
# Structured Relevance Assessment for Robust Retrieval-Augmented Language Models 

**Authors**: Aryan Raj, Astitva Veer Garg, Anitha D  

**Link**: [PDF](https://arxiv.org/pdf/2507.21287)  

**Abstract**: Retrieval-Augmented Language Models (RALMs) face significant challenges in reducing factual errors, particularly in document relevance evaluation and knowledge integration. We introduce a framework for structured relevance assessment that enhances RALM robustness through improved document evaluation, balanced intrinsic and external knowledge integration, and effective handling of unanswerable queries. Our approach employs a multi-dimensional scoring system that considers both semantic matching and source reliability, utilizing embedding-based relevance scoring and synthetic training data with mixed-quality documents. We implement specialized benchmarking on niche topics, a knowledge integration mechanism, and an "unknown" response protocol for queries with insufficient knowledge coverage. Preliminary evaluations demonstrate significant reductions in hallucination rates and improved transparency in reasoning processes. Our framework advances the development of more reliable question-answering systems capable of operating effectively in dynamic environments with variable data quality. While challenges persist in accurately distinguishing credible information and balancing system latency with thoroughness, this work represents a meaningful step toward enhancing RALM reliability. 

---
# RATE: An LLM-Powered Retrieval Augmented Generation Technology-Extraction Pipeline 

**Authors**: Karan Mirhosseini, Arya Aftab, Alireza Sheikh  

**Link**: [PDF](https://arxiv.org/pdf/2507.21125)  

**Abstract**: In an era of radical technology transformations, technology maps play a crucial role in enhancing decision making. These maps heavily rely on automated methods of technology extraction. This paper introduces Retrieval Augmented Technology Extraction (RATE), a Large Language Model (LLM) based pipeline for automated technology extraction from scientific literature. RATE combines Retrieval Augmented Generation (RAG) with multi-definition LLM-based validation. This hybrid method results in high recall in candidate generation alongside with high precision in candidate filtering. While the pipeline is designed to be general and widely applicable, we demonstrate its use on 678 research articles focused on Brain-Computer Interfaces (BCIs) and Extended Reality (XR) as a case study. Consequently, The validated technology terms by RATE were mapped into a co-occurrence network, revealing thematic clusters and structural features of the research landscape. For the purpose of evaluation, a gold standard dataset of technologies in 70 selected random articles had been curated by the experts. In addition, a technology extraction model based on Bidirectional Encoder Representations of Transformers (BERT) was used as a comparative method. RATE achieved F1-score of 91.27%, Significantly outperforming BERT with F1-score of 53.73%. Our findings highlight the promise of definition-driven LLM methods for technology extraction and mapping. They also offer new insights into emerging trends within the BCI-XR field. The source code is available this https URL 

---
# VizGenie: Toward Self-Refining, Domain-Aware Workflows for Next-Generation Scientific Visualization 

**Authors**: Ayan Biswas, Terece L. Turton, Nishath Rajiv Ranasinghe, Shawn Jones, Bradley Love, William Jones, Aric Hagberg, Han-Wei Shen, Nathan DeBardeleben, Earl Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2507.21124)  

**Abstract**: We present VizGenie, a self-improving, agentic framework that advances scientific visualization through large language model (LLM) by orchestrating of a collection of domain-specific and dynamically generated modules. Users initially access core functionalities--such as threshold-based filtering, slice extraction, and statistical analysis--through pre-existing tools. For tasks beyond this baseline, VizGenie autonomously employs LLMs to generate new visualization scripts (e.g., VTK Python code), expanding its capabilities on-demand. Each generated script undergoes automated backend validation and is seamlessly integrated upon successful testing, continuously enhancing the system's adaptability and robustness. A distinctive feature of VizGenie is its intuitive natural language interface, allowing users to issue high-level feature-based queries (e.g., ``visualize the skull"). The system leverages image-based analysis and visual question answering (VQA) via fine-tuned vision models to interpret these queries precisely, bridging domain expertise and technical implementation. Additionally, users can interactively query generated visualizations through VQA, facilitating deeper exploration. Reliability and reproducibility are further strengthened by Retrieval-Augmented Generation (RAG), providing context-driven responses while maintaining comprehensive provenance records. Evaluations on complex volumetric datasets demonstrate significant reductions in cognitive overhead for iterative visualization tasks. By integrating curated domain-specific tools with LLM-driven flexibility, VizGenie not only accelerates insight generation but also establishes a sustainable, continuously evolving visualization practice. The resulting platform dynamically learns from user interactions, consistently enhancing support for feature-centric exploration and reproducible research in scientific visualization. 

---
# SemRAG: Semantic Knowledge-Augmented RAG for Improved Question-Answering 

**Authors**: Kezhen Zhong, Basem Suleiman, Abdelkarim Erradi, Shijing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.21110)  

**Abstract**: This paper introduces SemRAG, an enhanced Retrieval Augmented Generation (RAG) framework that efficiently integrates domain-specific knowledge using semantic chunking and knowledge graphs without extensive fine-tuning. Integrating domain-specific knowledge into large language models (LLMs) is crucial for improving their performance in specialized tasks. Yet, existing adaptations are computationally expensive, prone to overfitting and limit scalability. To address these challenges, SemRAG employs a semantic chunking algorithm that segments documents based on the cosine similarity from sentence embeddings, preserving semantic coherence while reducing computational overhead. Additionally, by structuring retrieved information into knowledge graphs, SemRAG captures relationships between entities, improving retrieval accuracy and contextual understanding. Experimental results on MultiHop RAG and Wikipedia datasets demonstrate SemRAG has significantly enhances the relevance and correctness of retrieved information from the Knowledge Graph, outperforming traditional RAG methods. Furthermore, we investigate the optimization of buffer sizes for different data corpus, as optimizing buffer sizes tailored to specific datasets can further improve retrieval performance, as integration of knowledge graphs strengthens entity relationships for better contextual comprehension. The primary advantage of SemRAG is its ability to create an efficient, accurate domain-specific LLM pipeline while avoiding resource-intensive fine-tuning. This makes it a practical and scalable approach aligned with sustainability goals, offering a viable solution for AI applications in domain-specific fields. 

---
# Culinary Crossroads: A RAG Framework for Enhancing Diversity in Cross-Cultural Recipe Adaptation 

**Authors**: Tianyi Hu, Andrea Morales-Garzón, Jingyi Zheng, Maria Maistro, Daniel Hershcovich  

**Link**: [PDF](https://arxiv.org/pdf/2507.21934)  

**Abstract**: In cross-cultural recipe adaptation, the goal is not only to ensure cultural appropriateness and retain the original dish's essence, but also to provide diverse options for various dietary needs and preferences. Retrieval Augmented Generation (RAG) is a promising approach, combining the retrieval of real recipes from the target cuisine for cultural adaptability with large language models (LLMs) for relevance. However, it remains unclear whether RAG can generate diverse adaptation results. Our analysis shows that RAG tends to overly rely on a limited portion of the context across generations, failing to produce diverse outputs even when provided with varied contextual inputs. This reveals a key limitation of RAG in creative tasks with multiple valid answers: it fails to leverage contextual diversity for generating varied responses. To address this issue, we propose CARRIAGE, a plug-and-play RAG framework for cross-cultural recipe adaptation that enhances diversity in both retrieval and context organization. To our knowledge, this is the first RAG framework that explicitly aims to generate highly diverse outputs to accommodate multiple user preferences. Our experiments show that CARRIAGE achieves Pareto efficiency in terms of diversity and quality of recipe adaptation compared to closed-book LLMs. 

---
# Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning 

**Authors**: Haoran Luo, Haihong E, Guanting Chen, Qika Lin, Yikai Guo, Fangzhi Xu, Zemin Kuang, Meina Song, Xiaobao Wu, Yifan Zhu, Luu Anh Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.21892)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates hallucination in LLMs by incorporating external knowledge, but relies on chunk-based retrieval that lacks structural semantics. GraphRAG methods improve RAG by modeling knowledge as entity-relation graphs, but still face challenges in high construction cost, fixed one-time retrieval, and reliance on long-context reasoning and prompt design. To address these challenges, we propose Graph-R1, an agentic GraphRAG framework via end-to-end reinforcement learning (RL). It introduces lightweight knowledge hypergraph construction, models retrieval as a multi-turn agent-environment interaction, and optimizes the agent process via an end-to-end reward mechanism. Experiments on standard RAG datasets show that Graph-R1 outperforms traditional GraphRAG and RL-enhanced RAG methods in reasoning accuracy, retrieval efficiency, and generation quality. 

---
# HRIPBench: Benchmarking LLMs in Harm Reduction Information Provision to Support People Who Use Drugs 

**Authors**: Kaixuan Wang, Chenxin Diao, Jason T. Jacques, Zhongliang Guo, Shuai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.21815)  

**Abstract**: Millions of individuals' well-being are challenged by the harms of substance use. Harm reduction as a public health strategy is designed to improve their health outcomes and reduce safety risks. Some large language models (LLMs) have demonstrated a decent level of medical knowledge, promising to address the information needs of people who use drugs (PWUD). However, their performance in relevant tasks remains largely unexplored. We introduce HRIPBench, a benchmark designed to evaluate LLM's accuracy and safety risks in harm reduction information provision. The benchmark dataset HRIP-Basic has 2,160 question-answer-evidence pairs. The scope covers three tasks: checking safety boundaries, providing quantitative values, and inferring polysubstance use risks. We build the Instruction and RAG schemes to evaluate model behaviours based on their inherent knowledge and the integration of domain knowledge. Our results indicate that state-of-the-art LLMs still struggle to provide accurate harm reduction information, and sometimes, carry out severe safety risks to PWUD. The use of LLMs in harm reduction contexts should be cautiously constrained to avoid inducing negative health outcomes. WARNING: This paper contains illicit content that potentially induces harms. 

---
# DeepSieve: Information Sieving via LLM-as-a-Knowledge-Router 

**Authors**: Minghao Guo, Qingcheng Zeng, Xujiang Zhao, Yanchi Liu, Wenchao Yu, Mengnan Du, Haifeng Chen, Wei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.22050)  

**Abstract**: Large Language Models (LLMs) excel at many reasoning tasks but struggle with knowledge-intensive queries due to their inability to dynamically access up-to-date or domain-specific information. Retrieval-Augmented Generation (RAG) has emerged as a promising solution, enabling LLMs to ground their responses in external sources. However, existing RAG methods lack fine-grained control over both the query and source sides, often resulting in noisy retrieval and shallow reasoning. In this work, we introduce DeepSieve, an agentic RAG framework that incorporates information sieving via LLM-as-a-knowledge-router. DeepSieve decomposes complex queries into structured sub-questions and recursively routes each to the most suitable knowledge source, filtering irrelevant information through a multi-stage distillation process. Our design emphasizes modularity, transparency, and adaptability, leveraging recent advances in agentic system design. Experiments on multi-hop QA tasks across heterogeneous sources demonstrate improved reasoning depth, retrieval precision, and interpretability over conventional RAG approaches. 

---
# Analise Semantica Automatizada com LLM e RAG para Bulas Farmaceuticas 

**Authors**: Daniel Meireles do Rego  

**Link**: [PDF](https://arxiv.org/pdf/2507.21103)  

**Abstract**: The production of digital documents has been growing rapidly in academic, business, and health environments, presenting new challenges in the efficient extraction and analysis of unstructured information. This work investigates the use of RAG (Retrieval-Augmented Generation) architectures combined with Large-Scale Language Models (LLMs) to automate the analysis of documents in PDF format. The proposal integrates vector search techniques by embeddings, semantic data extraction and generation of contextualized natural language responses. To validate the approach, we conducted experiments with drug package inserts extracted from official public sources. The semantic queries applied were evaluated by metrics such as accuracy, completeness, response speed and consistency. The results indicate that the combination of RAG with LLMs offers significant gains in intelligent information retrieval and interpretation of unstructured technical texts. 

---
