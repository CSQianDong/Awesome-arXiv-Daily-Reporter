# From Answers to Guidance: A Proactive Dialogue System for Legal Documents 

**Authors**: Ashish Chouhan, Michael Gertz  

**Link**: [PDF](https://arxiv.org/pdf/2510.19723)  

**Abstract**: The accessibility of legal information remains a constant challenge, particularly for laypersons seeking to understand and apply complex institutional texts. While the European Union provides open access to legislation, parliamentary responses, and regulatory documents, these resources can be challenging for laypeople to explore. In this paper, we introduce EUDial, a proactive multi-turn dialogue dataset constructed from 204 blogs curated by the Citizens' Enquiries Unit (AskEP) of the European Parliamentary Research Service. EUDial contains 880 dialogue turns (averaging 4.3 turns per dialogue), where each dialogue includes initial questions, structured answers, and follow-up questions. Beyond dataset construction, we propose the LexGuide framework that leverages retrieval-augmented generation with hierarchical topic organization to structure dialogue progression, ensuring both comprehensive coverage of legal aspects and coherence across conversational turns. The results demonstrate that proactive, structured navigation closes the gap between the availability of legal information and citizen comprehension, establishing EUDial and LexGuide as practical resources for advancing proactive legal dialogue systems. 

---
# Algorithmic Fairness in NLP: Persona-Infused LLMs for Human-Centric Hate Speech Detection 

**Authors**: Ewelina Gajewska, Arda Derbent, Jaroslaw A Chudziak, Katarzyna Budzynska  

**Link**: [PDF](https://arxiv.org/pdf/2510.19331)  

**Abstract**: In this paper, we investigate how personalising Large Language Models (Persona-LLMs) with annotator personas affects their sensitivity to hate speech, particularly regarding biases linked to shared or differing identities between annotators and targets. To this end, we employ Google's Gemini and OpenAI's GPT-4.1-mini models and two persona-prompting methods: shallow persona prompting and a deeply contextualised persona development based on Retrieval-Augmented Generation (RAG) to incorporate richer persona profiles. We analyse the impact of using in-group and out-group annotator personas on the models' detection performance and fairness across diverse social groups. This work bridges psychological insights on group identity with advanced NLP techniques, demonstrating that incorporating socio-demographic attributes into LLMs can address bias in automated hate speech detection. Our results highlight both the potential and limitations of persona-based approaches in reducing bias, offering valuable insights for developing more equitable hate speech detection systems. 

---
# Think Straight, Stop Smart: Structured Reasoning for Efficient Multi-Hop RAG 

**Authors**: Jihwan Bang, Juntae Lee, Seunghan Yang, Sungha Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.19171)  

**Abstract**: Multi-hop retrieval-augmented generation (RAG) is a promising strategy for complex reasoning, yet existing iterative prompting approaches remain inefficient. They often regenerate predictable token sequences at every step and rely on stochastic stopping, leading to excessive token usage and unstable termination. We propose TSSS (Think Straight, Stop Smart), a structured multi-hop RAG framework designed for efficiency. TSSS introduces (i) a template-based reasoning that caches recurring prefixes and anchors sub-queries to the main question, reducing token generation cost while promoting stable reasoning, and (ii) a retriever-based terminator, which deterministically halts reasoning once additional sub-queries collapse into repetition. This separation of structured reasoning and termination control enables both faster inference and more reliable answers. On HotpotQA, 2WikiMultiHop, and MuSiQue, TSSS achieves state-of-the-art accuracy and competitive efficiency among RAG-CoT approaches, highlighting its effectiveness in efficiency-constrained scenarios such as on-device inference. 

---
# XGen-Q: An Explainable Domain-Adaptive LLM Framework with Retrieval-Augmented Generation for Software Security 

**Authors**: Hamed Jelodar, Mohammad Meymani, Roozbeh Razavi-Far, Ali A. Ghorbani  

**Link**: [PDF](https://arxiv.org/pdf/2510.19006)  

**Abstract**: Generative AI and large language models (LLMs) have shown strong capabilities in code understanding, but their use in cybersecurity, particularly for malware detection and analysis, remains limited. Existing detection systems often fail to generalize to obfuscated or previously unseen threats, underscoring the need for more adaptable and explainable models. To address this challenge, we introduce XGen-Q, a domain-adapted LLM built on the Qwen-Coder architecture and pretrained on a large-scale corpus of over one million malware samples, spanning both source and assembly code. XGen-Q uses a multi-stage prompt strategy combined with retrieval-augmented generation (RAG) to deliver reliable malware identification and detailed forensic reporting, even in the presence of complex code obfuscation. To further enhance generalization, we design a training pipeline that systematically exposes the model to diverse obfuscation patterns. Experimental results show that XGen-Q achieves significantly lower perplexity than competitive baselines and exhibits strong performance on novel malware samples, demonstrating the promise of LLM-based approaches for interpretable and robust malware analysis. 

---
