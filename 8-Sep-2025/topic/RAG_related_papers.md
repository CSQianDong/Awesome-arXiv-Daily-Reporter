# KERAG: Knowledge-Enhanced Retrieval-Augmented Generation for Advanced Question Answering 

**Authors**: Yushi Sun, Kai Sun, Yifan Ethan Xu, Xiao Yang, Xin Luna Dong, Nan Tang, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.04716)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates hallucination in Large Language Models (LLMs) by incorporating external data, with Knowledge Graphs (KGs) offering crucial information for question answering. Traditional Knowledge Graph Question Answering (KGQA) methods rely on semantic parsing, which typically retrieves knowledge strictly necessary for answer generation, thus often suffer from low coverage due to rigid schema requirements and semantic ambiguity. We present KERAG, a novel KG-based RAG pipeline that enhances QA coverage by retrieving a broader subgraph likely to contain relevant information. Our retrieval-filtering-summarization approach, combined with fine-tuned LLMs for Chain-of-Thought reasoning on knowledge sub-graphs, reduces noises and improves QA for both simple and complex questions. Experiments demonstrate that KERAG surpasses state-of-the-art solutions by about 7% in quality and exceeds GPT-4o (Tool) by 10-21%. 

---
# VaccineRAG: Boosting Multimodal Large Language Models' Immunity to Harmful RAG Samples 

**Authors**: Qixin Sun, Ziqin Wang, Hengyuan Zhao, Yilin Li, Kaiyou Song, Linjiang Huang, Xiaolin Hu, Qingpei Guo, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04502)  

**Abstract**: Retrieval Augmented Generation enhances the response accuracy of Large Language Models (LLMs) by integrating retrieval and generation modules with external knowledge, demonstrating particular strength in real-time queries and Visual Question Answering tasks. However, the effectiveness of RAG is frequently hindered by the precision of the retriever: many retrieved samples fed into the generation phase are irrelevant or misleading, posing a critical bottleneck to LLMs' performance. To address this challenge, we introduce VaccineRAG, a novel Chain-of-Thought-based retrieval-augmented generation dataset. On one hand, VaccineRAG employs a benchmark to evaluate models using data with varying positive/negative sample ratios, systematically exposing inherent weaknesses in current LLMs. On the other hand, it enhances models' sample-discrimination capabilities by prompting LLMs to generate explicit Chain-of-Thought (CoT) analysis for each sample before producing final answers. Furthermore, to enhance the model's ability to learn long-sequence complex CoT content, we propose Partial-GRPO. By modeling the outputs of LLMs as multiple components rather than a single whole, our model can make more informed preference selections for complex sequences, thereby enhancing its capacity to learn complex CoT. Comprehensive evaluations and ablation studies on VaccineRAG validate the effectiveness of the proposed scheme. The code and dataset will be publicly released soon. 

---
# Evaluating Large Language Models for Financial Reasoning: A CFA-Based Benchmark Study 

**Authors**: Xuan Yao, Qianteng Wang, Xinbo Liu, Ke-Wei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04468)  

**Abstract**: The rapid advancement of large language models presents significant opportunities for financial applications, yet systematic evaluation in specialized financial contexts remains limited. This study presents the first comprehensive evaluation of state-of-the-art LLMs using 1,560 multiple-choice questions from official mock exams across Levels I-III of CFA, most rigorous professional certifications globally that mirror real-world financial analysis complexity. We compare models distinguished by core design priorities: multi-modal and computationally powerful, reasoning-specialized and highly accurate, and lightweight efficiency-optimized.
We assess models under zero-shot prompting and through a novel Retrieval-Augmented Generation pipeline that integrates official CFA curriculum content. The RAG system achieves precise domain-specific knowledge retrieval through hierarchical knowledge organization and structured query generation, significantly enhancing reasoning accuracy in professional financial certification evaluation.
Results reveal that reasoning-oriented models consistently outperform others in zero-shot settings, while the RAG pipeline provides substantial improvements particularly for complex scenarios. Comprehensive error analysis identifies knowledge gaps as the primary failure mode, with minimal impact from text readability. These findings provide actionable insights for LLM deployment in finance, offering practitioners evidence-based guidance for model selection and cost-performance optimization. 

---
# Mentalic Net: Development of RAG-based Conversational AI and Evaluation Framework for Mental Health Support 

**Authors**: Anandi Dutta, Shivani Mruthyunjaya, Jessica Saddington, Kazi Sifatul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2509.04456)  

**Abstract**: The emergence of large language models (LLMs) has unlocked boundless possibilities, along with significant challenges. In response, we developed a mental health support chatbot designed to augment professional healthcare, with a strong emphasis on safe and meaningful application. Our approach involved rigorous evaluation, covering accuracy, empathy, trustworthiness, privacy, and bias. We employed a retrieval-augmented generation (RAG) framework, integrated prompt engineering, and fine-tuned a pre-trained model on novel datasets. The resulting system, Mentalic Net Conversational AI, achieved a BERT Score of 0.898, with other evaluation metrics falling within satisfactory ranges. We advocate for a human-in-the-loop approach and a long-term, responsible strategy in developing such transformative technologies, recognizing both their potential to change lives and the risks they may pose if not carefully managed. 

---
# Maestro: Joint Graph & Config Optimization for Reliable AI Agents 

**Authors**: Wenxiao Wang, Priyatham Kattakinda, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04642)  

**Abstract**: Building reliable LLM agents requires decisions at two levels: the graph (which modules exist and how information flows) and the configuration of each node (models, prompts, tools, control knobs). Most existing optimizers tune configurations while holding the graph fixed, leaving structural failure modes unaddressed. We introduce Maestro, a framework-agnostic holistic optimizer for LLM agents that jointly searches over graphs and configurations to maximize agent quality, subject to explicit rollout/token budgets. Beyond numeric metrics, Maestro leverages reflective textual feedback from traces to prioritize edits, improving sample efficiency and targeting specific failure modes. On the IFBench and HotpotQA benchmarks, Maestro consistently surpasses leading prompt optimizers--MIPROv2, GEPA, and GEPA+Merge--by an average of 12%, 4.9%, and 4.86%, respectively; even when restricted to prompt-only optimization, it still leads by 9.65%, 2.37%, and 2.41%. Maestro achieves these results with far fewer rollouts than GEPA. We further show large gains on two applications (interviewer & RAG agents), highlighting that joint graph & configuration search addresses structural failure modes that prompt tuning alone cannot fix. 

---
# Energy Landscapes Enable Reliable Abstention in Retrieval-Augmented Large Language Models for Healthcare 

**Authors**: Ravi Shankar, Sheng Wong, Lin Li, Magdalena Bachmann, Alex Silverthorne, Beth Albert, Gabriel Davis Jones  

**Link**: [PDF](https://arxiv.org/pdf/2509.04482)  

**Abstract**: Reliable abstention is critical for retrieval-augmented generation (RAG) systems, particularly in safety-critical domains such as women's health, where incorrect answers can lead to harm. We present an energy-based model (EBM) that learns a smooth energy landscape over a dense semantic corpus of 2.6M guideline-derived questions, enabling the system to decide when to generate or abstain. We benchmark the EBM against a calibrated softmax baseline and a k-nearest neighbour (kNN) density heuristic across both easy and hard abstention splits, where hard cases are semantically challenging near-distribution queries. The EBM achieves superior abstention performance abstention on semantically hard cases, reaching AUROC 0.961 versus 0.950 for softmax, while also reducing FPR@95 (0.235 vs 0.331). On easy negatives, performance is comparable across methods, but the EBM's advantage becomes most pronounced in safety-critical hard distributions. A comprehensive ablation with controlled negative sampling and fair data exposure shows that robustness stems primarily from the energy scoring head, while the inclusion or exclusion of specific negative types (hard, easy, mixed) sharpens decision boundaries but is not essential for generalisation to hard cases. These results demonstrate that energy-based abstention scoring offers a more reliable confidence signal than probability-based softmax confidence, providing a scalable and interpretable foundation for safe RAG systems. 

---
# Fishing for Answers: Exploring One-shot vs. Iterative Retrieval Strategies for Retrieval Augmented Generation 

**Authors**: Huifeng Lin, Gang Su, Jintao Liang, You Wu, Rui Zhao, Ziyue Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.04820)  

**Abstract**: Retrieval-Augmented Generation (RAG) based on Large Language Models (LLMs) is a powerful solution to understand and query the industry's closed-source documents. However, basic RAG often struggles with complex QA tasks in legal and regulatory domains, particularly when dealing with numerous government documents. The top-$k$ strategy frequently misses golden chunks, leading to incomplete or inaccurate answers. To address these retrieval bottlenecks, we explore two strategies to improve evidence coverage and answer quality. The first is a One-SHOT retrieval method that adaptively selects chunks based on a token budget, allowing as much relevant content as possible to be included within the model's context window. Additionally, we design modules to further filter and refine the chunks. The second is an iterative retrieval strategy built on a Reasoning Agentic RAG framework, where a reasoning LLM dynamically issues search queries, evaluates retrieved results, and progressively refines the context over multiple turns. We identify query drift and retrieval laziness issues and further design two modules to tackle them. Through extensive experiments on a dataset of government documents, we aim to offer practical insights and guidance for real-world applications in legal and regulatory domains. 

---
