# RippleBench: Capturing Ripple Effects Using Existing Knowledge Repositories 

**Authors**: Roy Rinberg, Usha Bhalla, Igor Shilov, Flavio P. Calmon, Rohit Gandikota  

**Link**: [PDF](https://arxiv.org/pdf/2512.04144)  

**Abstract**: Targeted interventions on language models, such as unlearning, debiasing, or model editing, are a central method for refining model behavior and keeping knowledge up to date. While these interventions aim to modify specific information within models (e.g., removing virology content), their effects often propagate to related but unintended areas (e.g., allergies); these side-effects are commonly referred to as the ripple effect. In this work, we present RippleBench-Maker, an automatic tool for generating Q&A datasets that allow for the measurement of ripple effects in any model-editing task. RippleBench-Maker builds on a Wikipedia-based RAG pipeline (WikiRAG) to generate multiple-choice questions at varying semantic distances from the target concept (e.g., the knowledge being unlearned). Using this framework, we construct RippleBench-Bio, a benchmark derived from the WMDP (Weapons of Mass Destruction Paper) dataset, a common unlearning benchmark. We evaluate eight state-of-the-art unlearning methods and find that all exhibit non-trivial accuracy drops on topics increasingly distant from the unlearned knowledge, each with distinct propagation profiles. To support ongoing research, we release our codebase for on-the-fly ripple evaluation, along with the benchmark, RippleBench-Bio. 

---
# Spatially-Enhanced Retrieval-Augmented Generation for Walkability and Urban Discovery 

**Authors**: Maddalena Amendola, Chiara Pugliese, Raffaele Perego, Chiara Renso  

**Link**: [PDF](https://arxiv.org/pdf/2512.04790)  

**Abstract**: Large Language Models (LLMs) have become foundational tools in artificial intelligence, supporting a wide range of applications beyond traditional natural language processing, including urban systems and tourist recommendations. However, their tendency to hallucinate and their limitations in spatial retrieval and reasoning are well known, pointing to the need for novel solutions. Retrieval-augmented generation (RAG) has recently emerged as a promising way to enhance LLMs with accurate, domain-specific, and timely information. Spatial RAG extends this approach to tasks involving geographic understanding. In this work, we introduce WalkRAG, a spatial RAG-based framework with a conversational interface for recommending walkable urban itineraries. Users can request routes that meet specific spatial constraints and preferences while interactively retrieving information about the path and points of interest (POIs) along the way. Preliminary results show the effectiveness of combining information retrieval, spatial reasoning, and LLMs to support urban discovery. 

---
# The Personalization Paradox: Semantic Loss vs. Reasoning Gains in Agentic AI Q&A 

**Authors**: Satyajit Movidi, Stephen Russell  

**Link**: [PDF](https://arxiv.org/pdf/2512.04343)  

**Abstract**: AIVisor, an agentic retrieval-augmented LLM for student advising, was used to examine how personalization affects system performance across multiple evaluation dimensions. Using twelve authentic advising questions intentionally designed to stress lexical precision, we compared ten personalized and non-personalized system configurations and analyzed outcomes with a Linear Mixed-Effects Model across lexical (BLEU, ROUGE-L), semantic (METEOR, BERTScore), and grounding (RAGAS) metrics. Results showed a consistent trade-off: personalization reliably improved reasoning quality and grounding, yet introduced a significant negative interaction on semantic similarity, driven not by poorer answers but by the limits of current metrics, which penalize meaningful personalized deviations from generic reference texts. This reveals a structural flaw in prevailing LLM evaluation methods, which are ill-suited for assessing user-specific responses. The fully integrated personalized configuration produced the highest overall gains, suggesting that personalization can enhance system effectiveness when evaluated with appropriate multidimensional metrics. Overall, the study demonstrates that personalization produces metric-dependent shifts rather than uniform improvements and provides a methodological foundation for more transparent and robust personalization in agentic AI. 

---
