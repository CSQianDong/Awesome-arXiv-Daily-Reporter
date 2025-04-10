# Tuning LLMs by RAG Principles: Towards LLM-native Memory 

**Authors**: Jiale Wei, Shuchi Wu, Ruochen Liu, Xiang Ying, Jingbo Shang, Fangbo Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16071)  

**Abstract**: Memory, additional information beyond the training of large language models (LLMs), is crucial to various real-world applications, such as personal assistant. The two mainstream solutions to incorporate memory into the generation process are long-context LLMs and retrieval-augmented generation (RAG). In this paper, we first systematically compare these two types of solutions on three renovated/new datasets and show that (1) long-context solutions, although more expensive, shall be easier to capture the big picture and better answer queries which require considering the memory as a whole; and (2) when the queries concern specific information, RAG solutions shall be more competitive especially when the keywords can be explicitly matched. Therefore, we propose a novel method RAG-Tuned-LLM which fine-tunes a relative small (e.g., 7B) LLM using the data generated following the RAG principles, so it can combine the advantages of both solutions. Extensive experiments on three datasets demonstrate that RAG-Tuned-LLM can beat long-context LLMs and RAG methods across a wide range of query types. 

---
# Parameters vs. Context: Fine-Grained Control of Knowledge Reliance in Language Models 

**Authors**: Baolong Bi, Shenghua Liu, Yiwei Wang, Yilong Xu, Junfeng Fang, Lingrui Mei, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.15888)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates hallucinations in Large Language Models (LLMs) by integrating external knowledge. However, conflicts between parametric knowledge and retrieved context pose challenges, particularly when retrieved information is unreliable or the model's internal knowledge is outdated. In such cases, LLMs struggle to determine whether to rely more on their own parameters or the conflicted context. To address this, we propose **CK-PLUG**, a plug-and-play method for controlling LLMs' reliance on parametric and contextual knowledge. We introduce a novel knowledge consistency metric, Confidence Gain, which detects knowledge conflicts by measuring entropy shifts in token probability distributions after context insertion. CK-PLUG then enables fine-grained control over knowledge preference by adjusting the probability distribution of tokens with negative confidence gain through a single tuning parameter. Experiments demonstrate CK-PLUG's ability to significantly regulate knowledge reliance in counterfactual RAG scenarios while maintaining generation fluency and knowledge accuracy. For instance, on Llama3-8B, memory recall (MR) of RAG response can be adjusted within a broad range (9.9%-71.9%), compared to the baseline of 42.1%. Moreover, CK-PLUG supports adaptive control based on the model's confidence in both internal and external knowledge, achieving consistent performance improvements across various general RAG tasks. Our code is available at: $\href{this https URL}{\text{this https URL}}$. 

---
# Towards Lighter and Robust Evaluation for Retrieval Augmented Generation 

**Authors**: Alex-Razvan Ispas, Charles-Elie Simon, Fabien Caspani, Vincent Guigue  

**Link**: [PDF](https://arxiv.org/pdf/2503.16161)  

**Abstract**: Large Language Models are prompting us to view more NLP tasks from a generative perspective. At the same time, they offer a new way of accessing information, mainly through the RAG framework. While there have been notable improvements for the autoregressive models, overcoming hallucination in the generated answers remains a continuous problem. A standard solution is to use commercial LLMs, such as GPT4, to evaluate these algorithms. However, such frameworks are expensive and not very transparent. Therefore, we propose a study which demonstrates the interest of open-weight models for evaluating RAG hallucination. We develop a lightweight approach using smaller, quantized LLMs to provide an accessible and interpretable metric that gives continuous scores for the generated answer with respect to their correctness and faithfulness. This score allows us to question decisions' reliability and explore thresholds to develop a new AUC metric as an alternative to correlation with human judgment. 

---
# PersonaAI: Leveraging Retrieval-Augmented Generation and Personalized Context for AI-Driven Digital Avatars 

**Authors**: Elvis Kimara, Kunle S. Oguntoye, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.15489)  

**Abstract**: This paper introduces PersonaAI, a cutting-edge application that leverages Retrieval-Augmented Generation (RAG) and the LLAMA model to create highly personalized digital avatars capable of accurately mimicking individual personalities. Designed as a cloud-based mobile application, PersonaAI captures user data seamlessly, storing it in a secure database for retrieval and analysis. The result is a system that provides context-aware, accurate responses to user queries, enhancing the potential of AI-driven personalization.
Why should you care? PersonaAI combines the scalability of RAG with the efficiency of prompt-engineered LLAMA3, offering a lightweight, sustainable alternative to traditional large language model (LLM) training methods. The system's novel approach to data collection, utilizing real-time user interactions via a mobile app, ensures enhanced context relevance while maintaining user privacy. By open-sourcing our implementation, we aim to foster adaptability and community-driven development.
PersonaAI demonstrates how AI can transform interactions by merging efficiency, scalability, and personalization, making it a significant step forward in the future of digital avatars and personalized AI. 

---
# Privacy-Aware RAG: Secure and Isolated Knowledge Retrieval 

**Authors**: Pengcheng Zhou, Yinglun Feng, Zhongliang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.15548)  

**Abstract**: The widespread adoption of Retrieval-Augmented Generation (RAG) systems in real-world applications has heightened concerns about the confidentiality and integrity of their proprietary knowledge bases. These knowledge bases, which play a critical role in enhancing the generative capabilities of Large Language Models (LLMs), are increasingly vulnerable to breaches that could compromise sensitive information. To address these challenges, this paper proposes an advanced encryption methodology designed to protect RAG systems from unauthorized access and data leakage. Our approach encrypts both textual content and its corresponding embeddings prior to storage, ensuring that all data remains securely encrypted. This mechanism restricts access to authorized entities with the appropriate decryption keys, thereby significantly reducing the risk of unintended data exposure. Furthermore, we demonstrate that our encryption strategy preserves the performance and functionality of RAG pipelines, ensuring compatibility across diverse domains and applications. To validate the robustness of our method, we provide comprehensive security proofs that highlight its resilience against potential threats and vulnerabilities. These proofs also reveal limitations in existing approaches, which often lack robustness, adaptability, or reliance on open-source models. Our findings suggest that integrating advanced encryption techniques into the design and deployment of RAG systems can effectively enhance privacy safeguards. This research contributes to the ongoing discourse on improving security measures for AI-driven services and advocates for stricter data protection standards within RAG architectures. 

---
# Enhancing Pancreatic Cancer Staging with Large Language Models: The Role of Retrieval-Augmented Generation 

**Authors**: Hisashi Johno, Yuki Johno, Akitomo Amakawa, Junichi Sato, Ryota Tozuka, Atsushi Komaba, Hiroaki Watanabe, Hiroki Watanabe, Chihiro Goto, Hiroyuki Morisaka, Hiroshi Onishi, Kazunori Nakamoto  

**Link**: [PDF](https://arxiv.org/pdf/2503.15664)  

**Abstract**: Purpose: Retrieval-augmented generation (RAG) is a technology to enhance the functionality and reliability of large language models (LLMs) by retrieving relevant information from reliable external knowledge (REK). RAG has gained interest in radiology, and we previously reported the utility of NotebookLM, an LLM with RAG (RAG-LLM), for lung cancer staging. However, since the comparator LLM differed from NotebookLM's internal model, it remained unclear whether its advantage stemmed from RAG or inherent model differences. To better isolate RAG's impact and assess its utility across different cancers, we compared NotebookLM with its internal LLM, Gemini 2.0 Flash, in a pancreatic cancer staging experiment.
Materials and Methods: A summary of Japan's pancreatic cancer staging guidelines was used as REK. We compared three groups - REK+/RAG+ (NotebookLM with REK), REK+/RAG- (Gemini 2.0 Flash with REK), and REK-/RAG- (Gemini 2.0 Flash without REK) - in staging 100 fictional pancreatic cancer cases based on CT findings. Staging criteria included TNM classification, local invasion factors, and resectability classification. In REK+/RAG+, retrieval accuracy was quantified based on the sufficiency of retrieved REK excerpts.
Results: REK+/RAG+ achieved a staging accuracy of 70%, outperforming REK+/RAG- (38%) and REK-/RAG- (35%). For TNM classification, REK+/RAG+ attained 80% accuracy, exceeding REK+/RAG- (55%) and REK-/RAG- (50%). Additionally, REK+/RAG+ explicitly presented retrieved REK excerpts, achieving a retrieval accuracy of 92%.
Conclusion: NotebookLM, a RAG-LLM, outperformed its internal LLM, Gemini 2.0 Flash, in a pancreatic cancer staging experiment, suggesting that RAG may improve LLM's staging accuracy. Furthermore, its ability to retrieve and present REK excerpts provides transparency for physicians, highlighting its applicability for clinical diagnosis and classification. 

---
# Typed-RAG: Type-aware Multi-Aspect Decomposition for Non-Factoid Question Answering 

**Authors**: DongGeon Lee, Ahjeong Park, Hyeri Lee, Hyeonseo Nam, Yunho Maeng  

**Link**: [PDF](https://arxiv.org/pdf/2503.15879)  

**Abstract**: Non-factoid question-answering (NFQA) poses a significant challenge due to its open-ended nature, diverse intents, and the need for multi-aspect reasoning, which renders conventional factoid QA approaches, including retrieval-augmented generation (RAG), inadequate. Unlike factoid questions, non-factoid questions (NFQs) lack definitive answers and require synthesizing information from multiple sources across various reasoning dimensions. To address these limitations, we introduce Typed-RAG, a type-aware multi-aspect decomposition framework within the RAG paradigm for NFQA. Typed-RAG classifies NFQs into distinct types -- such as debate, experience, and comparison -- and applies aspect-based decomposition to refine retrieval and generation strategies. By decomposing multi-aspect NFQs into single-aspect sub-queries and aggregating the results, Typed-RAG generates more informative and contextually relevant responses. To evaluate Typed-RAG, we introduce Wiki-NFQA, a benchmark dataset covering diverse NFQ types. Experimental results demonstrate that Typed-RAG outperforms baselines, thereby highlighting the importance of type-aware decomposition for effective retrieval and generation in NFQA. Our code and dataset are available at \href{this https URL}{this https URL}. 

---
