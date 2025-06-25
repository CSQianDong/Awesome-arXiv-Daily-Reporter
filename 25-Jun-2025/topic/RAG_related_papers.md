# Conversational Intent-Driven GraphRAG: Enhancing Multi-Turn Dialogue Systems through Adaptive Dual-Retrieval of Flow Patterns and Context Semantics 

**Authors**: Ziqi Zhu, Tao Hu, Honglong Zhang, Dan Yang, HanGeng Chen, Mengran Zhang, Xilun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19385)  

**Abstract**: We present CID-GraphRAG (Conversational Intent-Driven Graph Retrieval Augmented Generation), a novel framework that addresses the limitations of existing dialogue systems in maintaining both contextual coherence and goal-oriented progression in multi-turn customer service conversations. Unlike traditional RAG systems that rely solely on semantic similarity (Conversation RAG) or standard knowledge graphs (GraphRAG), CID-GraphRAG constructs dynamic intent transition graphs from goal achieved historical dialogues and implements a dual-retrieval mechanism that adaptively balances intent-based graph traversal with semantic search. This approach enables the system to simultaneously leverage both conversional intent flow patterns and contextual semantics, significantly improving retrieval quality and response quality. In extensive experiments on real-world customer service dialogues, we employ both automatic metrics and LLM-as-judge assessments, demonstrating that CID-GraphRAG significantly outperforms both semantic-based Conversation RAG and intent-based GraphRAG baselines across all evaluation criteria. Quantitatively, CID-GraphRAG demonstrates substantial improvements over Conversation RAG across automatic metrics, with relative gains of 11% in BLEU, 5% in ROUGE-L, 6% in METEOR, and most notably, a 58% improvement in response quality according to LLM-as-judge evaluations. These results demonstrate that the integration of intent transition structures with semantic retrieval creates a synergistic effect that neither approach achieves independently, establishing CID-GraphRAG as an effective framework for addressing the challenges of maintaining contextual coherence and goal-oriented progression in knowledge-intensive multi-turn dialogues. 

---
# Dialogic Pedagogy for Large Language Models: Aligning Conversational AI with Proven Theories of Learning 

**Authors**: Russell Beale  

**Link**: [PDF](https://arxiv.org/pdf/2506.19484)  

**Abstract**: Large Language Models (LLMs) are rapidly transforming education by enabling rich conversational learning experiences. This article provides a comprehensive review of how LLM-based conversational agents are being used in higher education, with extensions to secondary and lifelong learning contexts. We synthesize existing literature on LLMs in education and theories of conversational and dialogic pedagogy - including Vygotsky's sociocultural learning (scaffolding and the Zone of Proximal Development), the Socratic method, and Laurillard's conversational framework - and examine how prompting strategies and retrieval-augmented generation (RAG) can align LLM behaviors with these pedagogical theories, and how it can support personalized, adaptive learning. We map educational theories to LLM capabilities, highlighting where LLM-driven dialogue supports established learning principles and where it challenges or falls short of traditional pedagogical assumptions. Notable gaps in applying prior theories to LLMs are identified, such as the models tendency to provide direct answers instead of fostering co-construction of knowledge, and the need to account for the constant availability and broad but non-human expertise of LLM tutors. In response, we propose practical strategies to better align LLM interactions with sound pedagogy - for example, designing prompts that encourage Socratic questioning, scaffolded guidance, and student reflection, as well as integrating retrieval mechanisms to ensure accuracy and contextual relevance. Our aim is to bridge the gap between educational theory and the emerging practice of AI-driven conversational learning, offering insights and tools for making LLM-based dialogues more educationally productive and theory-aligned. 

---
# heiDS at ArchEHR-QA 2025: From Fixed-k to Query-dependent-k for Retrieval Augmented Generation 

**Authors**: Ashish Chouhan, Michael Gertz  

**Link**: [PDF](https://arxiv.org/pdf/2506.19512)  

**Abstract**: This paper presents the approach of our team called heiDS for the ArchEHR-QA 2025 shared task. A pipeline using a retrieval augmented generation (RAG) framework is designed to generate answers that are attributed to clinical evidence from the electronic health records (EHRs) of patients in response to patient-specific questions. We explored various components of a RAG framework, focusing on ranked list truncation (RLT) retrieval strategies and attribution approaches. Instead of using a fixed top-k RLT retrieval strategy, we employ a query-dependent-k retrieval strategy, including the existing surprise and autocut methods and two new methods proposed in this work, autocut* and elbow. The experimental results show the benefits of our strategy in producing factual and relevant answers when compared to a fixed-$k$. 

---
