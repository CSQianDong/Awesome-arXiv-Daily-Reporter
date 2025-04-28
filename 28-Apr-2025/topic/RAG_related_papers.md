# SMARTFinRAG: Interactive Modularized Financial RAG Benchmark 

**Authors**: Yiwei Zha  

**Link**: [PDF](https://arxiv.org/pdf/2504.18024)  

**Abstract**: Financial sectors are rapidly adopting language model technologies, yet evaluating specialized RAG systems in this domain remains challenging. This paper introduces SMARTFinRAG, addressing three critical gaps in financial RAG assessment: (1) a fully modular architecture where components can be dynamically interchanged during runtime; (2) a document-centric evaluation paradigm generating domain-specific QA pairs from newly ingested financial documents; and (3) an intuitive interface bridging research-implementation divides. Our evaluation quantifies both retrieval efficacy and response quality, revealing significant performance variations across configurations. The platform's open-source architecture supports transparent, reproducible research while addressing practical deployment challenges faced by financial institutions implementing RAG systems. 

---
# LLMpatronous: Harnessing the Power of LLMs For Vulnerability Detection 

**Authors**: Rajesh Yarra  

**Link**: [PDF](https://arxiv.org/pdf/2504.18423)  

**Abstract**: Despite the transformative impact of Artificial Intelligence (AI) across various sectors, cyber security continues to rely on traditional static and dynamic analysis tools, hampered by high false positive rates and superficial code comprehension. While generative AI offers promising automation capabilities for software development, leveraging Large Language Models (LLMs) for vulnerability detection presents unique challenges. This paper explores the potential and limitations of LLMs in identifying vulnerabilities, acknowledging inherent weaknesses such as hallucinations, limited context length, and knowledge cut-offs. Previous attempts employing machine learning models for vulnerability detection have proven ineffective due to limited real-world applicability, feature engineering challenges, lack of contextual understanding, and the complexities of training models to keep pace with the evolving threat landscape. Therefore, we propose a robust AI-driven approach focused on mitigating these limitations and ensuring the quality and reliability of LLM based vulnerability detection. Through innovative methodologies combining Retrieval-Augmented Generation (RAG) and Mixtureof-Agents (MoA), this research seeks to leverage the strengths of LLMs while addressing their weaknesses, ultimately paving the way for dependable and efficient AI-powered solutions in securing the ever-evolving software landscape. 

---
# PropRAG: Guiding Retrieval with Beam Search over Proposition Paths 

**Authors**: Jingjin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18070)  

**Abstract**: Retrieval Augmented Generation (RAG) has become the standard non-parametric approach for equipping Large Language Models (LLMs) with up-to-date knowledge and mitigating catastrophic forgetting common in continual learning. However, standard RAG, relying on independent passage retrieval, fails to capture the interconnected nature of human memory crucial for complex reasoning (associativity) and contextual understanding (sense-making). While structured RAG methods like HippoRAG utilize knowledge graphs (KGs) built from triples, the inherent context loss limits fidelity. We introduce PropRAG, a framework leveraging contextually rich propositions and a novel beam search algorithm over proposition paths to explicitly discover multi-step reasoning chains. Crucially, PropRAG's online retrieval process operates entirely without invoking generative LLMs, relying instead on efficient graph traversal and pre-computed embeddings. This avoids online LLM inference costs and potential inconsistencies during evidence gathering. LLMs are used effectively offline for high-quality proposition extraction and post-retrieval for answer generation. PropRAG achieves state-of-the-art zero-shot Recall@5 results on PopQA (55.3%), 2Wiki (93.7%), HotpotQA (97.0%), and MuSiQue (77.3%), alongside top F1 scores (e.g., 52.4% on MuSiQue). By improving evidence retrieval through richer representation and explicit, LLM-free online path finding, PropRAG advances non-parametric continual learning. 

---
# RAG LLMs are Not Safer: A Safety Analysis of Retrieval-Augmented Generation for Large Language Models 

**Authors**: Bang An, Shiyue Zhang, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2504.18041)  

**Abstract**: Efforts to ensure the safety of large language models (LLMs) include safety fine-tuning, evaluation, and red teaming. However, despite the widespread use of the Retrieval-Augmented Generation (RAG) framework, AI safety work focuses on standard LLMs, which means we know little about how RAG use cases change a model's safety profile. We conduct a detailed comparative analysis of RAG and non-RAG frameworks with eleven LLMs. We find that RAG can make models less safe and change their safety profile. We explore the causes of this change and find that even combinations of safe models with safe documents can cause unsafe generations. In addition, we evaluate some existing red teaming methods for RAG settings and show that they are less effective than when used for non-RAG settings. Our work highlights the need for safety research and red-teaming methods specifically tailored for RAG LLMs. 

---
# Even Small Reasoners Should Quote Their Sources: Introducing the Pleias-RAG Model Family 

**Authors**: Pierre-Carl Langlais, Pavel Chizhov, Mattia Nee, Carlos Rosas Hinostroza, Matthieu Delsart, Irène Girard, Othman Hicheur, Anastasia Stasenko, Ivan P. Yamshchikov  

**Link**: [PDF](https://arxiv.org/pdf/2504.18225)  

**Abstract**: We introduce a new generation of small reasoning models for RAG, search, and source summarization. Pleias-RAG-350m and Pleias-RAG-1B are mid-trained on a large synthetic dataset emulating the retrieval of a wide variety of multilingual open sources from the Common Corpus. They provide native support for citation and grounding with literal quotes and reintegrate multiple features associated with RAG workflows, such as query routing, query reformulation, and source reranking. Pleias-RAG-350m and Pleias-RAG-1B outperform SLMs below 4 billion parameters on standardized RAG benchmarks (HotPotQA, 2wiki) and are competitive with popular larger models, including Qwen-2.5-7B, Llama-3.1-8B, and Gemma-3-4B. They are the only SLMs to date maintaining consistent RAG performance across leading European languages and ensuring systematic reference grounding for statements. Due to their size and ease of deployment on constrained infrastructure and higher factuality by design, the models unlock a range of new use cases for generative AI. 

---
