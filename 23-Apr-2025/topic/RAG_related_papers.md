# CiteFix: Enhancing RAG Accuracy Through Post-Processing Citation Correction 

**Authors**: Harsh Maheshwari, Srikanth Tenneti, Alwarappan Nakkiran  

**Link**: [PDF](https://arxiv.org/pdf/2504.15629)  

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as a powerful application of Large Language Models (LLMs), revolutionizing information search and consumption. RAG systems combine traditional search capabilities with LLMs to generate comprehensive answers to user queries, ideally with accurate citations. However, in our experience of developing a RAG product, LLMs often struggle with source attribution, aligning with other industry studies reporting citation accuracy rates of only about 74% for popular generative search engines. To address this, we present efficient post-processing algorithms to improve citation accuracy in LLM-generated responses, with minimal impact on latency and cost. Our approaches cross-check generated citations against retrieved articles using methods including keyword + semantic matching, fine tuned model with BERTScore, and a lightweight LLM-based technique. Our experimental results demonstrate a relative improvement of 15.46% in the overall accuracy metrics of our RAG system. This significant enhancement potentially enables a shift from our current larger language model to a relatively smaller model that is approximately 12x more cost-effective and 3x faster in inference time, while maintaining comparable performance. This research contributes to enhancing the reliability and trustworthiness of AI-generated content in information retrieval and summarization tasks which is critical to gain customer trust especially in commercial products. 

---
# Synergizing RAG and Reasoning: A Systematic Review 

**Authors**: Yunfan Gao, Yun Xiong, Yijie Zhong, Yuxi Bi, Ming Xue, Haofen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15909)  

**Abstract**: Recent breakthroughs in large language models (LLMs), particularly in reasoning capabilities, have propelled Retrieval-Augmented Generation (RAG) to unprecedented levels. By synergizing retrieval mechanisms with advanced reasoning, LLMs can now tackle increasingly complex problems. This paper presents a systematic review of the collaborative interplay between RAG and reasoning, clearly defining "reasoning" within the RAG context. It construct a comprehensive taxonomy encompassing multi-dimensional collaborative objectives, representative paradigms, and technical implementations, and analyze the bidirectional synergy methods. Additionally, we critically evaluate current limitations in RAG assessment, including the absence of intermediate supervision for multi-step reasoning and practical challenges related to cost-risk trade-offs. To bridge theory and practice, we provide practical guidelines tailored to diverse real-world applications. Finally, we identify promising research directions, such as graph-based knowledge integration, hybrid model collaboration, and RL-driven optimization. Overall, this work presents a theoretical framework and practical foundation to advance RAG systems in academia and industry, fostering the next generation of RAG solutions. 

---
# FinDER: Financial Dataset for Question Answering and Evaluating Retrieval-Augmented Generation 

**Authors**: Chanyeol Choi, Jihoon Kwon, Jaeseon Ha, Hojun Choi, Chaewoon Kim, Yongjae Lee, Jy-yong Sohn, Alejandro Lopez-Lira  

**Link**: [PDF](https://arxiv.org/pdf/2504.15800)  

**Abstract**: In the fast-paced financial domain, accurate and up-to-date information is critical to addressing ever-evolving market conditions. Retrieving this information correctly is essential in financial Question-Answering (QA), since many language models struggle with factual accuracy in this domain. We present FinDER, an expert-generated dataset tailored for Retrieval-Augmented Generation (RAG) in finance. Unlike existing QA datasets that provide predefined contexts and rely on relatively clear and straightforward queries, FinDER focuses on annotating search-relevant evidence by domain experts, offering 5,703 query-evidence-answer triplets derived from real-world financial inquiries. These queries frequently include abbreviations, acronyms, and concise expressions, capturing the brevity and ambiguity common in the realistic search behavior of professionals. By challenging models to retrieve relevant information from large corpora rather than relying on readily determined contexts, FinDER offers a more realistic benchmark for evaluating RAG systems. We further present a comprehensive evaluation of multiple state-of-the-art retrieval models and Large Language Models, showcasing challenges derived from a realistic benchmark to drive future research on truthful and precise RAG in the financial domain. 

---
# The Viability of Crowdsourcing for RAG Evaluation 

**Authors**: Lukas Gienapp, Tim Hagen, Maik Fröbe, Matthias Hagen, Benno Stein, Martin Potthast, Harrisen Scells  

**Link**: [PDF](https://arxiv.org/pdf/2504.15689)  

**Abstract**: How good are humans at writing and judging responses in retrieval-augmented generation (RAG) scenarios? To answer this question, we investigate the efficacy of crowdsourcing for RAG through two complementary studies: response writing and response utility judgment. We present the Crowd RAG Corpus 2025 (CrowdRAG-25), which consists of 903 human-written and 903 LLM-generated responses for the 301 topics of the TREC RAG'24 track, across the three discourse styles 'bulleted list', 'essay', and 'news'. For a selection of 65 topics, the corpus further contains 47,320 pairwise human judgments and 10,556 pairwise LLM judgments across seven utility dimensions (e.g., coverage and coherence). Our analyses give insights into human writing behavior for RAG and the viability of crowdsourcing for RAG evaluation. Human pairwise judgments provide reliable and cost-effective results compared to LLM-based pairwise or human/LLM-based pointwise judgments, as well as automated comparisons with human-written reference responses. All our data and tools are freely available. 

---
