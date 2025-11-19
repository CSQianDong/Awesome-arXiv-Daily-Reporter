# LiveRAG: A diverse Q&A dataset with varying difficulty level for RAG evaluation 

**Authors**: David Carmel, Simone Filice, Guy Horowitz, Yoelle Maarek, Alex Shtoff, Oren Somekh, Ran Tavory  

**Link**: [PDF](https://arxiv.org/pdf/2511.14531)  

**Abstract**: With Retrieval Augmented Generation (RAG) becoming more and more prominent in generative AI solutions, there is an emerging need for systematically evaluating their effectiveness. We introduce the LiveRAG benchmark, a publicly available dataset of 895 synthetic questions and answers designed to support systematic evaluation of RAG-based Q&A systems. This synthetic benchmark is derived from the one used during the SIGIR'2025 LiveRAG Challenge, where competitors were evaluated under strict time constraints. It is augmented with information that was not made available to competitors during the Challenge, such as the ground-truth answers, together with their associated supporting claims which were used for evaluating competitors' answers. In addition, each question is associated with estimated difficulty and discriminability scores, derived from applying an Item Response Theory model to competitors' responses. Our analysis highlights the benchmark's questions diversity, the wide range of their difficulty levels, and their usefulness in differentiating between system capabilities. The LiveRAG benchmark will hopefully help the community advance RAG research, conduct systematic evaluation, and develop more robust Q&A systems. 

---
# WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG from Web 

**Authors**: Zihuai Zhao, Yujuan Ding, Wenqi Fan, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.14182)  

**Abstract**: Recommender systems play a vital role in alleviating information overload and enriching users' online experience. In the era of large language models (LLMs), LLM-based recommender systems have emerged as a prevalent paradigm for advancing personalized recommendations. Recently, retrieval-augmented generation (RAG) has drawn growing interest to facilitate the recommendation capability of LLMs, incorporating useful information retrieved from external knowledge bases. However, as a rich source of up-to-date information, the web remains under-explored by existing RAG-based recommendations. In particular, unique challenges are posed from two perspectives: one is to generate effective queries for web retrieval, considering the inherent knowledge gap between web search and recommendations; another challenge lies in harnessing online websites that contain substantial noisy content. To tackle these limitations, we propose WebRec, a novel web-based RAG framework, which takes advantage of the reasoning capability of LLMs to interpret recommendation tasks into queries of user preferences that cater to web retrieval. Moreover, given noisy web-retrieved information, where relevant pieces of evidence are scattered far apart, an insightful MP-Head is designed to enhance LLM attentions between distant tokens of relevant information via message passing. Extensive experiments have been conducted to demonstrate the effectiveness of our proposed web-based RAG methods in recommendation scenarios. 

---
# NeuroPath: Neurobiology-Inspired Path Tracking and Reflection for Semantically Coherent Retrieval 

**Authors**: Junchen Li, Rongzheng Wang, Yihong Huang, Qizhi Chen, Jiasheng Zhang, Shuang Liang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14096)  

**Abstract**: Retrieval-augmented generation (RAG) greatly enhances large language models (LLMs) performance in knowledge-intensive tasks. However, naive RAG methods struggle with multi-hop question answering due to their limited capacity to capture complex dependencies across documents. Recent studies employ graph-based RAG to capture document connections. However, these approaches often result in a loss of semantic coherence and introduce irrelevant noise during node matching and subgraph construction. To address these limitations, we propose NeuroPath, an LLM-driven semantic path tracking RAG framework inspired by the path navigational planning of place cells in neurobiology. It consists of two steps: Dynamic Path Tracking and Post-retrieval Completion. Dynamic Path Tracking performs goal-directed semantic path tracking and pruning over the constructed knowledge graph (KG), improving noise reduction and semantic coherence. Post-retrieval Completion further reinforces these benefits by conducting second-stage retrieval using intermediate reasoning and the original query to refine the query goal and complete missing information in the reasoning path. NeuroPath surpasses current state-of-the-art baselines on three multi-hop QA datasets, achieving average improvements of 16.3% on recall@2 and 13.5% on recall@5 over advanced graph-based RAG methods. Moreover, compared to existing iter-based RAG methods, NeuroPath achieves higher accuracy and reduces token consumption by 22.8%. Finally, we demonstrate the robustness of NeuroPath across four smaller LLMs (Llama3.1, GLM4, Mistral0.3, and Gemma3), and further validate its scalability across tasks of varying complexity. Code is available at this https URL. 

---
# Streamlining Industrial Contract Management with Retrieval-Augmented LLMs 

**Authors**: Kristi Topollai, Tolga Dimlioglu, Anna Choromanska, Simon Odie, Reginald Hui  

**Link**: [PDF](https://arxiv.org/pdf/2511.14671)  

**Abstract**: Contract management involves reviewing and negotiating provisions, individual clauses that define rights, obligations, and terms of agreement. During this process, revisions to provisions are proposed and iteratively refined, some of which may be problematic or unacceptable. Automating this workflow is challenging due to the scarcity of labeled data and the abundance of unstructured legacy contracts. In this paper, we present a modular framework designed to streamline contract management through a retrieval-augmented generation (RAG) pipeline. Our system integrates synthetic data generation, semantic clause retrieval, acceptability classification, and reward-based alignment to flag problematic revisions and generate improved alternatives. Developed and evaluated in collaboration with an industry partner, our system achieves over 80% accuracy in both identifying and optimizing problematic revisions, demonstrating strong performance under real-world, low-resource conditions and offering a practical means of accelerating contract revision workflows. 

---
# Tell Me: An LLM-powered Mental Well-being Assistant with RAG, Synthetic Dialogue Generation, and Agentic Planning 

**Authors**: Trishala Jayesh Ahalpara  

**Link**: [PDF](https://arxiv.org/pdf/2511.14445)  

**Abstract**: We present Tell Me, a mental well-being system that leverages advances in large language models to provide accessible, context-aware support for users and researchers. The system integrates three components: (i) a retrieval-augmented generation (RAG) assistant for personalized, knowledge-grounded dialogue; (ii) a synthetic client-therapist dialogue generator conditioned on client profiles to facilitate research on therapeutic language and data augmentation; and (iii) a Well-being AI crew, implemented with CrewAI, that produces weekly self-care plans and guided meditation audio. The system is designed as a reflective space for emotional processing rather than a substitute for professional therapy. It illustrates how conversational assistants can lower barriers to support, complement existing care, and broaden access to mental health resources. To address the shortage of confidential therapeutic data, we introduce synthetic client-therapist dialogue generation conditioned on client profiles. Finally, the planner demonstrates an innovative agentic workflow for dynamically adaptive, personalized self-care, bridging the limitations of static well-being tools. We describe the architecture, demonstrate its functionalities, and report evaluation of the RAG assistant in curated well-being scenarios using both automatic LLM-based judgments and a human-user study. This work highlights opportunities for interdisciplinary collaboration between NLP researchers and mental health professionals to advance responsible innovation in human-AI interaction for well-being. 

---
# Knowledge-Grounded Agentic Large Language Models for Multi-Hazard Understanding from Reconnaissance Reports 

**Authors**: Chenchen Kuai, Zihao Li, Braden Rosen, Stephanie Paan, Navid Jafari, Jean-Louis Briaud, Yunlong Zhang, Youssef M. A. Hashash, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.14010)  

**Abstract**: Post-disaster reconnaissance reports contain critical evidence for understanding multi-hazard interactions, yet their unstructured narratives make systematic knowledge transfer difficult. Large language models (LLMs) offer new potential for analyzing these reports, but often generate unreliable or hallucinated outputs when domain grounding is absent. This study introduces the Mixture-of-Retrieval Agentic RAG (MoRA-RAG), a knowledge-grounded LLM framework that transforms reconnaissance reports into a structured foundation for multi-hazard reasoning. The framework integrates a Mixture-of-Retrieval mechanism that dynamically routes queries across hazard-specific databases while using agentic chunking to preserve contextual coherence during retrieval. It also includes a verification loop that assesses evidence sufficiency, refines queries, and initiates targeted searches when information remains incomplete. We construct HazardRecQA by deriving question-answer pairs from GEER reconnaissance reports, which document 90 global events across seven major hazard types. MoRA-RAG achieves up to 94.5 percent accuracy, outperforming zero-shot LLMs by 30 percent and state-of-the-art RAG systems by 10 percent, while reducing hallucinations across diverse LLM architectures. MoRA-RAG also enables open-weight LLMs to achieve performance comparable to proprietary models. It establishes a new paradigm for transforming post-disaster documentation into actionable, trustworthy intelligence for hazard resilience. 

---
# SciRAG: Adaptive, Citation-Aware, and Outline-Guided Retrieval and Synthesis for Scientific Literature 

**Authors**: Hang Ding, Yilun Zhao, Tiansheng Hu, Manasi Patwardhan, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2511.14362)  

**Abstract**: The accelerating growth of scientific publications has intensified the need for scalable, trustworthy systems to synthesize knowledge across diverse literature. While recent retrieval-augmented generation (RAG) methods have improved access to scientific information, they often overlook citation graph structure, adapt poorly to complex queries, and yield fragmented, hard-to-verify syntheses. We introduce SciRAG, an open-source framework for scientific literature exploration that addresses these gaps through three key innovations: (1) adaptive retrieval that flexibly alternates between sequential and parallel evidence gathering; (2) citation-aware symbolic reasoning that leverages citation graphs to organize and filter supporting documents; and (3) outline-guided synthesis that plans, critiques, and refines answers to ensure coherence and transparent attribution. Extensive experiments across multiple benchmarks such as QASA and ScholarQA demonstrate that SciRAG outperforms prior systems in factual accuracy and synthesis quality, establishing a new foundation for reliable, large-scale scientific knowledge aggregation. 

---
