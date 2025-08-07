# TURA: Tool-Augmented Unified Retrieval Agent for AI Search 

**Authors**: Zhejun Zhao, Yuehu Dong, Alley Liu, Lixue Zheng, Pingsheng Liu, Dongdong Shen, Long Xia, Jiashu Zhao, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.04604)  

**Abstract**: The advent of Large Language Models (LLMs) is transforming search engines into conversational AI search products, primarily using Retrieval-Augmented Generation (RAG) on web corpora. However, this paradigm has significant industrial limitations. Traditional RAG approaches struggle with real-time needs and structured queries that require accessing dynamically generated content like ticket availability or inventory. Limited to indexing static pages, search engines cannot perform the interactive queries needed for such time-sensitive data. Academic research has focused on optimizing RAG for static content, overlooking complex intents and the need for dynamic sources like databases and real-time APIs. To bridge this gap, we introduce TURA (Tool-Augmented Unified Retrieval Agent for AI Search), a novel three-stage framework that combines RAG with agentic tool-use to access both static content and dynamic, real-time information. TURA has three key components: an Intent-Aware Retrieval module to decompose queries and retrieve information sources encapsulated as Model Context Protocol (MCP) Servers, a DAG-based Task Planner that models task dependencies as a Directed Acyclic Graph (DAG) for optimal parallel execution, and a lightweight Distilled Agent Executor for efficient tool calling. TURA is the first architecture to systematically bridge the gap between static RAG and dynamic information sources for a world-class AI search product. Serving tens of millions of users, it leverages an agentic framework to deliver robust, real-time answers while meeting the low-latency demands of a large-scale industrial system. 

---
# RAIDX: A Retrieval-Augmented Generation and GRPO Reinforcement Learning Framework for Explainable Deepfake Detection 

**Authors**: Tianxiao Li, Zhenglin Huang, Haiquan Wen, Yiwei He, Shuchang Lyu, Baoyuan Wu, Guangliang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.04524)  

**Abstract**: The rapid advancement of AI-generation models has enabled the creation of hyperrealistic imagery, posing ethical risks through widespread misinformation. Current deepfake detection methods, categorized as face specific detectors or general AI-generated detectors, lack transparency by framing detection as a classification task without explaining decisions. While several LLM-based approaches offer explainability, they suffer from coarse-grained analyses and dependency on labor-intensive annotations. This paper introduces RAIDX (Retrieval-Augmented Image Deepfake Detection and Explainability), a novel deepfake detection framework integrating Retrieval-Augmented Generation (RAG) and Group Relative Policy Optimization (GRPO) to enhance detection accuracy and decision explainability. Specifically, RAIDX leverages RAG to incorporate external knowledge for improved detection accuracy and employs GRPO to autonomously generate fine-grained textual explanations and saliency maps, eliminating the need for extensive manual annotations. Experiments on multiple benchmarks demonstrate RAIDX's effectiveness in identifying real or fake, and providing interpretable rationales in both textual descriptions and saliency maps, achieving state-of-the-art detection performance while advancing transparency in deepfake identification. RAIDX represents the first unified framework to synergize RAG and GRPO, addressing critical gaps in accuracy and explainability. Our code and models will be publicly available. 

---
# AIC CTU@FEVER 8: On-premise fact checking through long context RAG 

**Authors**: Herbert Ullrich, Jan Drchal  

**Link**: [PDF](https://arxiv.org/pdf/2508.04390)  

**Abstract**: In this paper, we present our fact-checking pipeline which has scored first in FEVER 8 shared task. Our fact-checking system is a simple two-step RAG pipeline based on our last year's submission. We show how the pipeline can be redeployed on-premise, achieving state-of-the-art fact-checking performance (in sense of Ev2R test-score), even under the constraint of a single NVidia A10 GPU, 23GB of graphical memory and 60s running time per claim. 

---
# Automated Generation of Curriculum-Aligned Multiple-Choice Questions for Malaysian Secondary Mathematics Using Generative AI 

**Authors**: Rohaizah Abdul Wahid, Muhamad Said Nizamuddin Nadim, Suliana Sulaiman, Syahmi Akmal Shaharudin, Muhammad Danial Jupikil, Iqqwan Jasman Su Azlan Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.04442)  

**Abstract**: This paper addresses the critical need for scalable and high-quality educational assessment tools within the Malaysian education system. It highlights the potential of Generative AI (GenAI) while acknowledging the significant challenges of ensuring factual accuracy and curriculum alignment, especially for low-resource languages like Bahasa Melayu. This research introduces and compares four incremental pipelines for generating Form 1 Mathematics multiple-choice questions (MCQs) in Bahasa Melayu using OpenAI's GPT-4o. The methods range from non-grounded prompting (structured and basic) to Retrieval-Augmented Generation (RAG) approaches (one using the LangChain framework, one implemented manually). The system is grounded in official curriculum documents, including teacher-prepared notes and the yearly teaching plan (RPT). A dual-pronged automated evaluation framework is employed to assess the generated questions. Curriculum alignment is measured using Semantic Textual Similarity (STS) against the RPT, while contextual validity is verified through a novel RAG-based Question-Answering (RAG-QA) method. The results demonstrate that RAG-based pipelines significantly outperform non-grounded prompting methods, producing questions with higher curriculum alignment and factual validity. The study further analyzes the trade-offs between the ease of implementation of framework-based RAG and the fine-grained control offered by a manual pipeline. This work presents a validated methodology for generating curriculum-specific educational content in a low-resource language, introduces a symbiotic RAG-QA evaluation technique, and provides actionable insights for the development and deployment of practical EdTech solutions in Malaysia and similar regions. 

---
# Hallucination to Truth: A Review of Fact-Checking and Factuality Evaluation in Large Language Models 

**Authors**: Subhey Sadi Rahman, Md. Adnanul Islam, Md. Mahbub Alam, Musarrat Zeba, Md. Abdur Rahman, Sadia Sultana Chowa, Mohaimenul Azam Khan Raiaan, Sami Azam  

**Link**: [PDF](https://arxiv.org/pdf/2508.03860)  

**Abstract**: Large Language Models (LLMs) are trained on vast and diverse internet corpora that often include inaccurate or misleading content. Consequently, LLMs can generate misinformation, making robust fact-checking essential. This review systematically analyzes how LLM-generated content is evaluated for factual accuracy by exploring key challenges such as hallucinations, dataset limitations, and the reliability of evaluation metrics. The review emphasizes the need for strong fact-checking frameworks that integrate advanced prompting strategies, domain-specific fine-tuning, and retrieval-augmented generation (RAG) methods. It proposes five research questions that guide the analysis of the recent literature from 2020 to 2025, focusing on evaluation methods and mitigation techniques. The review also discusses the role of instruction tuning, multi-agent reasoning, and external knowledge access via RAG frameworks. Key findings highlight the limitations of current metrics, the value of grounding outputs with validated external evidence, and the importance of domain-specific customization to improve factual consistency. Overall, the review underlines the importance of building LLMs that are not only accurate and explainable but also tailored for domain-specific fact-checking. These insights contribute to the advancement of research toward more trustworthy and context-aware language models. 

---
# Intent Aware Context Retrieval for Multi-Turn Agricultural Question Answering 

**Authors**: Abhay Vijayvargia, Ajay Nagpal, Kundeshwar Pundalik, Atharva Savarkar, Smita Gautam, Pankaj Singh, Rohit Saluja, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2508.03719)  

**Abstract**: Indian farmers often lack timely, accessible, and language-friendly agricultural advice, especially in rural areas with low literacy. To address this gap in accessibility, this paper presents a novel AI-powered agricultural chatbot, Krishi Sathi, designed to support Indian farmers by providing personalized, easy-to-understand answers to their queries through both text and speech. The system's intelligence stems from an IFT model, subsequently refined through fine-tuning on Indian agricultural knowledge across three curated datasets. Unlike traditional chatbots that respond to one-off questions, Krishi Sathi follows a structured, multi-turn conversation flow to gradually collect the necessary details from the farmer, ensuring the query is fully understood before generating a response. Once the intent and context are extracted, the system performs Retrieval-Augmented Generation (RAG) by first fetching information from a curated agricultural database and then generating a tailored response using the IFT model. The chatbot supports both English and Hindi languages, with speech input and output features (via ASR and TTS) to make it accessible for users with low literacy or limited digital skills. This work demonstrates how combining intent-driven dialogue flows, instruction-tuned models, and retrieval-based generation can improve the quality and accessibility of digital agricultural support in India.
This approach yielded strong results, with the system achieving a query response accuracy of 97.53%, 91.35% contextual relevance and personalization, and a query completion rate of 97.53%. The average response time remained under 6 seconds, ensuring timely support for users across both English and Hindi interactions. 

---
# RAVID: Retrieval-Augmented Visual Detection: A Knowledge-Driven Approach for AI-Generated Image Identification 

**Authors**: Mamadou Keita, Wassim Hamidouche, Hessen Bougueffa Eutamene, Abdelmalik Taleb-Ahmed, Abdenour Hadid  

**Link**: [PDF](https://arxiv.org/pdf/2508.03967)  

**Abstract**: In this paper, we introduce RAVID, the first framework for AI-generated image detection that leverages visual retrieval-augmented generation (RAG). While RAG methods have shown promise in mitigating factual inaccuracies in foundation models, they have primarily focused on text, leaving visual knowledge underexplored. Meanwhile, existing detection methods, which struggle with generalization and robustness, often rely on low-level artifacts and model-specific features, limiting their adaptability. To address this, RAVID dynamically retrieves relevant images to enhance detection. Our approach utilizes a fine-tuned CLIP image encoder, RAVID CLIP, enhanced with category-related prompts to improve representation learning. We further integrate a vision-language model (VLM) to fuse retrieved images with the query, enriching the input and improving accuracy. Given a query image, RAVID generates an embedding using RAVID CLIP, retrieves the most relevant images from a database, and combines these with the query image to form an enriched input for a VLM (e.g., Qwen-VL or Openflamingo). Experiments on the UniversalFakeDetect benchmark, which covers 19 generative models, show that RAVID achieves state-of-the-art performance with an average accuracy of 93.85%. RAVID also outperforms traditional methods in terms of robustness, maintaining high accuracy even under image degradations such as Gaussian blur and JPEG compression. Specifically, RAVID achieves an average accuracy of 80.27% under degradation conditions, compared to 63.44% for the state-of-the-art model C2P-CLIP, demonstrating consistent improvements in both Gaussian blur and JPEG compression scenarios. The code will be publicly available upon acceptance. 

---
# A Few Words Can Distort Graphs: Knowledge Poisoning Attacks on Graph-based Retrieval-Augmented Generation of Large Language Models 

**Authors**: Jiayi Wen, Tianxin Chen, Zhirun Zheng, Cheng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.04276)  

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) has recently emerged as a promising paradigm for enhancing large language models (LLMs) by converting raw text into structured knowledge graphs, improving both accuracy and explainability. However, GraphRAG relies on LLMs to extract knowledge from raw text during graph construction, and this process can be maliciously manipulated to implant misleading information. Targeting this attack surface, we propose two knowledge poisoning attacks (KPAs) and demonstrate that modifying only a few words in the source text can significantly change the constructed graph, poison the GraphRAG, and severely mislead downstream reasoning. The first attack, named Targeted KPA (TKPA), utilizes graph-theoretic analysis to locate vulnerable nodes in the generated graphs and rewrites the corresponding narratives with LLMs, achieving precise control over specific question-answering (QA) outcomes with a success rate of 93.1\%, while keeping the poisoned text fluent and natural. The second attack, named Universal KPA (UKPA), exploits linguistic cues such as pronouns and dependency relations to disrupt the structural integrity of the generated graph by altering globally influential words. With fewer than 0.05\% of full text modified, the QA accuracy collapses from 95\% to 50\%. Furthermore, experiments show that state-of-the-art defense methods fail to detect these attacks, highlighting that securing GraphRAG pipelines against knowledge poisoning remains largely unexplored. 

---
# PAIRS: Parametric-Verified Adaptive Information Retrieval and Selection for Efficient RAG 

**Authors**: Wang Chen, Guanqiang Qi, Weikang Li, Yang Li, Deguo Xia, Jizhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.04057)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a cornerstone technique for enhancing large language models (LLMs) with external knowledge. However, current RAG systems face two critical limitations: (1) they inefficiently retrieve information for every query, including simple questions that could be resolved using the LLM's parametric knowledge alone, and (2) they risk retrieving irrelevant documents when queries contain sparse information signals. To address these gaps, we introduce Parametric-verified Adaptive Information Retrieval and Selection (PAIRS), a training-free framework that integrates parametric and retrieved knowledge to adaptively determine whether to retrieve and how to select external information. Specifically, PAIRS employs a dual-path generation mechanism: First, the LLM produces both a direct answer and a context-augmented answer using self-generated pseudo-context. When these outputs converge, PAIRS bypasses external retrieval entirely, dramatically improving the RAG system's efficiency. For divergent cases, PAIRS activates a dual-path retrieval (DPR) process guided by both the original query and self-generated contextual signals, followed by an Adaptive Information Selection (AIS) module that filters documents through weighted similarity to both sources. This simple yet effective approach can not only enhance efficiency by eliminating unnecessary retrievals but also improve accuracy through contextually guided retrieval and adaptive information selection. Experimental results on six question-answering (QA) benchmarks show that PAIRS reduces retrieval costs by around 25% (triggering for only 75% of queries) while still improving accuracy-achieving +1.1% EM and +1.0% F1 over prior baselines on average. 

---
