# Improving Scientific Document Retrieval with Academic Concept Index 

**Authors**: Jeyun Lee, Junhyoung Lee, Wonbin Kweon, Bowen Jin, Yu Zhang, Susik Yoon, Dongha Lee, Hwanjo Yu, Jiawei Han, Seongku Kang  

**Link**: [PDF](https://arxiv.org/pdf/2601.00567)  

**Abstract**: Adapting general-domain retrievers to scientific domains is challenging due to the scarcity of large-scale domain-specific relevance annotations and the substantial mismatch in vocabulary and information needs. Recent approaches address these issues through two independent directions that leverage large language models (LLMs): (1) generating synthetic queries for fine-tuning, and (2) generating auxiliary contexts to support relevance matching. However, both directions overlook the diverse academic concepts embedded within scientific documents, often producing redundant or conceptually narrow queries and contexts. To address this limitation, we introduce an academic concept index, which extracts key concepts from papers and organizes them guided by an academic taxonomy. This structured index serves as a foundation for improving both directions. First, we enhance the synthetic query generation with concept coverage-based generation (CCQGen), which adaptively conditions LLMs on uncovered concepts to generate complementary queries with broader concept coverage. Second, we strengthen the context augmentation with concept-focused auxiliary contexts (CCExpand), which leverages a set of document snippets that serve as concise responses to the concept-aware CCQGen queries. Extensive experiments show that incorporating the academic concept index into both query generation and context augmentation leads to higher-quality queries, better conceptual alignment, and improved retrieval performance. 

---
# A Chain-of-Thought Approach to Semantic Query Categorization in e-Commerce Taxonomies 

**Authors**: Jetlir Duraj, Ishita Khan, Kilian Merkelbach, Mehran Elyasi  

**Link**: [PDF](https://arxiv.org/pdf/2601.00510)  

**Abstract**: Search in e-Commerce is powered at the core by a structured representation of the inventory, often formulated as a category taxonomy. An important capability in e-Commerce with hierarchical taxonomies is to select a set of relevant leaf categories that are semantically aligned with a given user query. In this scope, we address a fundamental problem of search query categorization in real-world e-Commerce taxonomies. A correct categorization of a query not only provides a way to zoom into the correct inventory space, but opens the door to multiple intent understanding capabilities for a query. A practical and accurate solution to this problem has many applications in e-commerce, including constraining retrieved items and improving the relevance of the search results. For this task, we explore a novel Chain-of-Thought (CoT) paradigm that combines simple tree-search with LLM semantic scoring. Assessing its classification performance on human-judged query-category pairs, relevance tests, and LLM-based reference methods, we find that the CoT approach performs better than a benchmark that uses embedding-based query category predictions. We show how the CoT approach can detect problems within a hierarchical taxonomy. Finally, we also propose LLM-based approaches for query-categorization of the same spirit, but which scale better at the range of millions of queries. 

---
# TeleDoCTR: Domain-Specific and Contextual Troubleshooting for Telecommunications 

**Authors**: Mohamed Trabelsi, Huseyin Uzunalioglu  

**Link**: [PDF](https://arxiv.org/pdf/2601.00691)  

**Abstract**: Ticket troubleshooting refers to the process of analyzing and resolving problems that are reported through a ticketing system. In large organizations offering a wide range of services, this task is highly complex due to the diversity of submitted tickets and the need for specialized domain knowledge. In particular, troubleshooting in telecommunications (telecom) is a very time-consuming task as it requires experts to interpret ticket content, consult documentation, and search historical records to identify appropriate resolutions. This human-intensive approach not only delays issue resolution but also hinders overall operational efficiency. To enhance the effectiveness and efficiency of ticket troubleshooting in telecom, we propose TeleDoCTR, a novel telecom-related, domain-specific, and contextual troubleshooting system tailored for end-to-end ticket resolution in telecom. TeleDoCTR integrates both domain-specific ranking and generative models to automate key steps of the troubleshooting workflow which are: routing tickets to the appropriate expert team responsible for resolving the ticket (classification task), retrieving contextually and semantically similar historical tickets (retrieval task), and generating a detailed fault analysis report outlining the issue, root cause, and potential solutions (generation task). We evaluate TeleDoCTR on a real-world dataset from a telecom infrastructure and demonstrate that it achieves superior performance over existing state-of-the-art methods, significantly enhancing the accuracy and efficiency of the troubleshooting process. 

---
# Noise-Aware Named Entity Recognition for Historical VET Documents 

**Authors**: Alexander M. Esser, Jens Dörpinghaus  

**Link**: [PDF](https://arxiv.org/pdf/2601.00488)  

**Abstract**: This paper addresses Named Entity Recognition (NER) in the domain of Vocational Education and Training (VET), focusing on historical, digitized documents that suffer from OCR-induced noise. We propose a robust NER approach leveraging Noise-Aware Training (NAT) with synthetically injected OCR errors, transfer learning, and multi-stage fine-tuning. Three complementary strategies, training on noisy, clean, and artificial data, are systematically compared. Our method is one of the first to recognize multiple entity types in VET documents. It is applied to German documents but transferable to arbitrary languages. Experimental results demonstrate that domain-specific and noise-aware fine-tuning substantially increases robustness and accuracy under noisy conditions. We provide publicly available code for reproducible noise-aware NER in domain-specific contexts. 

---
# Reinforcement-Learned Unequal Error Protection for Quantized Semantic Embeddings 

**Authors**: Moirangthem Tiken Singh, Adnan Arif  

**Link**: [PDF](https://arxiv.org/pdf/2601.00186)  

**Abstract**: This paper tackles the pressing challenge of preserving semantic meaning in communication systems constrained by limited bandwidth. We introduce a novel reinforcement learning framework that achieves per-dimension unequal error protection via adaptive repetition coding. Central to our approach is a composite semantic distortion metric that balances global embedding similarity with entity-level preservation, empowering the reinforcement learning agent to allocate protection in a context-aware manner. Experiments show statistically significant gains over uniform protection, achieving 6.8% higher chrF scores and 9.3% better entity preservation at 1 dB SNR. The key innovation of our framework is the demonstration that simple, intelligently allocated repetition coding enables fine-grained semantic protection -- an advantage unattainable with conventional codes such as LDPC or Reed-Solomon. Our findings challenge traditional channel coding paradigms by establishing that code structure must align with semantic granularity. This approach is particularly suited to edge computing and IoT scenarios, where bandwidth is scarce, but semantic fidelity is critical, providing a practical pathway for next-generation semantic-aware networks. 

---
# The Agentic Leash: Extracting Causal Feedback Fuzzy Cognitive Maps with LLMs 

**Authors**: Akash Kumar Panda, Olaoluwa Adigun, Bart Kosko  

**Link**: [PDF](https://arxiv.org/pdf/2601.00097)  

**Abstract**: We design a large-language-model (LLM) agent that extracts causal feedback fuzzy cognitive maps (FCMs) from raw text. The causal learning or extraction process is agentic both because of the LLM's semi-autonomy and because ultimately the FCM dynamical system's equilibria drive the LLM agents to fetch and process causal text. The fetched text can in principle modify the adaptive FCM causal structure and so modify the source of its quasi-autonomy--its equilibrium limit cycles and fixed-point attractors. This bidirectional process endows the evolving FCM dynamical system with a degree of autonomy while still staying on its agentic leash. We show in particular that a sequence of three finely tuned system instructions guide an LLM agent as it systematically extracts key nouns and noun phrases from text, as it extracts FCM concept nodes from among those nouns and noun phrases, and then as it extracts or infers partial or fuzzy causal edges between those FCM nodes. We test this FCM generation on a recent essay about the promise of AI from the late diplomat and political theorist Henry Kissinger and his colleagues. This three-step process produced FCM dynamical systems that converged to the same equilibrium limit cycles as did the human-generated FCMs even though the human-generated FCM differed in the number of nodes and edges. A final FCM mixed generated FCMs from separate Gemini and ChatGPT LLM agents. The mixed FCM absorbed the equilibria of its dominant mixture component but also created new equilibria of its own to better approximate the underlying causal dynamical system. 

---
