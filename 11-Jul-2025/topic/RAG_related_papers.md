# DrugMCTS: a drug repurposing framework combining multi-agent, RAG and Monte Carlo Tree Search 

**Authors**: Zerui Yang, Yuwei Wan, Yinqiao Li, Yudai Matsuda, Tong Xie, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.07426)  

**Abstract**: Recent advances in large language models have demonstrated considerable potential in scientific domains such as drug discovery. However, their effectiveness remains constrained when reasoning extends beyond the knowledge acquired during pretraining. Conventional approaches, such as fine-tuning or retrieval-augmented generation, face limitations in either imposing high computational overhead or failing to fully exploit structured scientific data. To overcome these challenges, we propose DrugMCTS, a novel framework that synergistically integrates RAG, multi-agent collaboration, and Monte Carlo Tree Search for drug repurposing. The framework employs five specialized agents tasked with retrieving and analyzing molecular and protein information, thereby enabling structured and iterative reasoning. Without requiring domain-specific fine-tuning, DrugMCTS empowers Qwen2.5-7B-Instruct to outperform Deepseek-R1 by over 20\%. Extensive experiments on the DrugBank and KIBA datasets demonstrate that DrugMCTS achieves substantially higher recall and robustness compared to both general-purpose LLMs and deep learning baselines. Our results highlight the importance of structured reasoning, agent-based collaboration, and feedback-driven search mechanisms in advancing LLM applications for drug discovery. 

---
# From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on Retrieval-Augmented Generation systems 

**Authors**: Youngjoon Jang, Seongtae Hong, Junyoung Son, Sungjin Park, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.07847)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a crucial framework in natural language processing (NLP), improving factual consistency and reducing hallucinations by integrating external document retrieval with large language models (LLMs). However, the effectiveness of RAG is often hindered by coreferential complexity in retrieved documents, introducing ambiguity that disrupts in-context learning. In this study, we systematically investigate how entity coreference affects both document retrieval and generative performance in RAG-based systems, focusing on retrieval relevance, contextual understanding, and overall response quality. We demonstrate that coreference resolution enhances retrieval effectiveness and improves question-answering (QA) performance. Through comparative analysis of different pooling strategies in retrieval tasks, we find that mean pooling demonstrates superior context capturing ability after applying coreference resolution. In QA tasks, we discover that smaller models benefit more from the disambiguation process, likely due to their limited inherent capacity for handling referential ambiguity. With these findings, this study aims to provide a deeper understanding of the challenges posed by coreferential complexity in RAG, providing guidance for improving retrieval and generation in knowledge-intensive AI applications. 

---
# Performance and Practical Considerations of Large and Small Language Models in Clinical Decision Support in Rheumatology 

**Authors**: Sabine Felde, Rüdiger Buchkremer, Gamal Chehab, Christian Thielscher, Jörg HW Distler, Matthias Schneider, Jutta G. Richter  

**Link**: [PDF](https://arxiv.org/pdf/2507.07983)  

**Abstract**: Large language models (LLMs) show promise for supporting clinical decision-making in complex fields such as rheumatology. Our evaluation shows that smaller language models (SLMs), combined with retrieval-augmented generation (RAG), achieve higher diagnostic and therapeutic performance than larger models, while requiring substantially less energy and enabling cost-efficient, local deployment. These features are attractive for resource-limited healthcare. However, expert oversight remains essential, as no model consistently reached specialist-level accuracy in rheumatology. 

---
# The Cross-Lingual Cost: Retrieval Biases in RAG over Arabic-English Corpora 

**Authors**: Chen Amiraz, Yaroslav Fyodorov, Elad Haramaty, Zohar Karnin, Liane Lewin-Eytan  

**Link**: [PDF](https://arxiv.org/pdf/2507.07543)  

**Abstract**: Cross-lingual retrieval-augmented generation (RAG) is a critical capability for retrieving and generating answers across languages. Prior work in this context has mostly focused on generation and relied on benchmarks derived from open-domain sources, most notably Wikipedia. In such settings, retrieval challenges often remain hidden due to language imbalances, overlap with pretraining data, and memorized content. To address this gap, we study Arabic-English RAG in a domain-specific setting using benchmarks derived from real-world corporate datasets. Our benchmarks include all combinations of languages for the user query and the supporting document, drawn independently and uniformly at random. This enables a systematic study of multilingual retrieval behavior.
Our findings reveal that retrieval is a critical bottleneck in cross-lingual domain-specific scenarios, with significant performance drops occurring when the user query and supporting document languages differ. A key insight is that these failures stem primarily from the retriever's difficulty in ranking documents across languages. Finally, we propose a simple retrieval strategy that addresses this source of failure by enforcing equal retrieval from both languages, resulting in substantial improvements in cross-lingual and overall performance. These results highlight meaningful opportunities for improving multilingual retrieval, particularly in practical, real-world RAG applications. 

---
# Evaluating Retrieval-Augmented Generation Agents for Autonomous Scientific Discovery in Astrophysics 

**Authors**: Xueqing Xu, Boris Bolliet, Adrian Dimitrov, Andrew Laverick, Francisco Villaescusa-Navarro, Licong Xu, Íñigo Zubeldia  

**Link**: [PDF](https://arxiv.org/pdf/2507.07155)  

**Abstract**: We evaluate 9 Retrieval Augmented Generation (RAG) agent configurations on 105 Cosmology Question-Answer (QA) pairs that we built specifically for this this http URL RAG configurations are manually evaluated by a human expert, that is, a total of 945 generated answers were assessed. We find that currently the best RAG agent configuration is with OpenAI embedding and generative model, yielding 91.4\% accuracy. Using our human evaluation results we calibrate LLM-as-a-Judge (LLMaaJ) system which can be used as a robust proxy for human evaluation. These results allow us to systematically select the best RAG agent configuration for multi-agent system for autonomous scientific discovery in astrophysics (e.g., cmbagent presented in a companion paper) and provide us with an LLMaaJ system that can be scaled to thousands of cosmology QA pairs. We make our QA dataset, human evaluation results, RAG pipelines, and LLMaaJ system publicly available for further use by the astrophysics community. 

---
# KeyKnowledgeRAG (K^2RAG): An Enhanced RAG method for improved LLM question-answering capabilities 

**Authors**: Hruday Markondapatnaikuni, Basem Suleiman, Abdelkarim Erradi, Shijing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.07695)  

**Abstract**: Fine-tuning is an immensely resource-intensive process when retraining Large Language Models (LLMs) to incorporate a larger body of knowledge. Although many fine-tuning techniques have been developed to reduce the time and computational cost involved, the challenge persists as LLMs continue to grow in size and complexity. To address this, a new approach to knowledge expansion in LLMs is needed. Retrieval-Augmented Generation (RAG) offers one such alternative by storing external knowledge in a database and retrieving relevant chunks to support question answering. However, naive implementations of RAG face significant limitations in scalability and answer accuracy. This paper introduces KeyKnowledgeRAG (K2RAG), a novel framework designed to overcome these limitations. Inspired by the divide-and-conquer paradigm, K2RAG integrates dense and sparse vector search, knowledge graphs, and text summarization to improve retrieval quality and system efficiency. The framework also includes a preprocessing step that summarizes the training data, significantly reducing the training time. K2RAG was evaluated using the MultiHopRAG dataset, where the proposed pipeline was trained on the document corpus and tested on a separate evaluation set. Results demonstrated notable improvements over common naive RAG implementations. K2RAG achieved the highest mean answer similarity score of 0.57, and reached the highest third quartile (Q3) similarity of 0.82, indicating better alignment with ground-truth answers. In addition to improved accuracy, the framework proved highly efficient. The summarization step reduced the average training time of individual components by 93%, and execution speed was up to 40% faster than traditional knowledge graph-based RAG systems. K2RAG also demonstrated superior scalability, requiring three times less VRAM than several naive RAG implementations tested in this study. 

---
# FrugalRAG: Learning to retrieve and reason for multi-hop QA 

**Authors**: Abhinav Java, Srivathsan Koundinyan, Nagarajan Natarajan, Amit Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.07634)  

**Abstract**: We consider the problem of answering complex questions, given access to a large unstructured document corpus. The de facto approach to solving the problem is to leverage language models that (iteratively) retrieve and reason through the retrieved documents, until the model has sufficient information to generate an answer. Attempts at improving this approach focus on retrieval-augmented generation (RAG) metrics such as accuracy and recall and can be categorized into two types: (a) fine-tuning on large question answering (QA) datasets augmented with chain-of-thought traces, and (b) leveraging RL-based fine-tuning techniques that rely on question-document relevance signals. However, efficiency in the number of retrieval searches is an equally important metric, which has received less attention. In this work, we show that: (1) Large-scale fine-tuning is not needed to improve RAG metrics, contrary to popular claims in recent literature. Specifically, a standard ReAct pipeline with improved prompts can outperform state-of-the-art methods on benchmarks such as HotPotQA. (2) Supervised and RL-based fine-tuning can help RAG from the perspective of frugality, i.e., the latency due to number of searches at inference time. For example, we show that we can achieve competitive RAG metrics at nearly half the cost (in terms of number of searches) on popular RAG benchmarks, using the same base model, and at a small training cost (1000 examples). 

---
# Multi-Agent Retrieval-Augmented Framework for Evidence-Based Counterspeech Against Health Misinformation 

**Authors**: Anirban Saha Anik, Xiaoying Song, Elliott Wang, Bryan Wang, Bengisu Yarimbas, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.07307)  

**Abstract**: Large language models (LLMs) incorporated with Retrieval-Augmented Generation (RAG) have demonstrated powerful capabilities in generating counterspeech against misinformation. However, current studies rely on limited evidence and offer less control over final outputs. To address these challenges, we propose a Multi-agent Retrieval-Augmented Framework to generate counterspeech against health misinformation, incorporating multiple LLMs to optimize knowledge retrieval, evidence enhancement, and response refinement. Our approach integrates both static and dynamic evidence, ensuring that the generated counterspeech is relevant, well-grounded, and up-to-date. Our method outperforms baseline approaches in politeness, relevance, informativeness, and factual accuracy, demonstrating its effectiveness in generating high-quality counterspeech. To further validate our approach, we conduct ablation studies to verify the necessity of each component in our framework. Furthermore, human evaluations reveal that refinement significantly enhances counterspeech quality and obtains human preference. 

---
