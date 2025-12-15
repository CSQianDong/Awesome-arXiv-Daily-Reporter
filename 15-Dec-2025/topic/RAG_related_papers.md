# MedAI: Evaluating TxAgent's Therapeutic Agentic Reasoning in the NeurIPS CURE-Bench Competition 

**Authors**: Tim Cofala, Christian Kalfar, Jingge Xiao, Johanna Schrader, Michelle Tang, Wolfgang Nejdl  

**Link**: [PDF](https://arxiv.org/pdf/2512.11682)  

**Abstract**: Therapeutic decision-making in clinical medicine constitutes a high-stakes domain in which AI guidance interacts with complex interactions among patient characteristics, disease processes, and pharmacological agents. Tasks such as drug recommendation, treatment planning, and adverse-effect prediction demand robust, multi-step reasoning grounded in reliable biomedical knowledge. Agentic AI methods, exemplified by TxAgent, address these challenges through iterative retrieval-augmented generation (RAG). TxAgent employs a fine-tuned Llama-3.1-8B model that dynamically generates and executes function calls to a unified biomedical tool suite (ToolUniverse), integrating FDA Drug API, OpenTargets, and Monarch resources to ensure access to current therapeutic information. In contrast to general-purpose RAG systems, medical applications impose stringent safety constraints, rendering the accuracy of both the reasoning trace and the sequence of tool invocations critical. These considerations motivate evaluation protocols treating token-level reasoning and tool-usage behaviors as explicit supervision signals. This work presents insights derived from our participation in the CURE-Bench NeurIPS 2025 Challenge, which benchmarks therapeutic-reasoning systems using metrics that assess correctness, tool utilization, and reasoning quality. We analyze how retrieval quality for function (tool) calls influences overall model performance and demonstrate performance gains achieved through improved tool-retrieval strategies. Our work was awarded the Excellence Award in Open Science. Complete information can be found at this https URL. 

---
# EmeraldMind: A Knowledge Graph-Augmented Framework for Greenwashing Detection 

**Authors**: Georgios Kaoukis, Ioannis Aris Koufopoulos, Psaroudaki Eleni, Danae Pla Karidi, Evaggelia Pitoura, George Papastefanatos, Panayiotis Tsaparas  

**Link**: [PDF](https://arxiv.org/pdf/2512.11506)  

**Abstract**: As AI and web agents become pervasive in decision-making, it is critical to design intelligent systems that not only support sustainability efforts but also guard against misinformation. Greenwashing, i.e., misleading corporate sustainability claims, poses a major challenge to environmental progress. To address this challenge, we introduce EmeraldMind, a fact-centric framework integrating a domain-specific knowledge graph with retrieval-augmented generation to automate greenwashing detection. EmeraldMind builds the EmeraldGraph from diverse corporate ESG (environmental, social, and governance) reports, surfacing verifiable evidence, often missing in generic knowledge bases, and supporting large language models in claim assessment. The framework delivers justification-centric classifications, presenting transparent, evidence-backed verdicts and abstaining responsibly when claims cannot be verified. Experiments on a new greenwashing claims dataset demonstrate that EmeraldMind achieves competitive accuracy, greater coverage, and superior explanation quality compared to generic LLMs, without the need for fine-tuning or retraining. 

---
# From Signal to Turn: Interactional Friction in Modular Speech-to-Speech Pipelines 

**Authors**: Titaya Mairittha, Tanakon Sawanglok, Panuwit Raden, Jirapast Buntub, Thanapat Warunee, Napat Asawachaisuvikrom, Thanaphum Saiwongin  

**Link**: [PDF](https://arxiv.org/pdf/2512.11724)  

**Abstract**: While voice-based AI systems have achieved remarkable generative capabilities, their interactions often feel conversationally broken. This paper examines the interactional friction that emerges in modular Speech-to-Speech Retrieval-Augmented Generation (S2S-RAG) pipelines. By analyzing a representative production system, we move beyond simple latency metrics to identify three recurring patterns of conversational breakdown: (1) Temporal Misalignment, where system delays violate user expectations of conversational rhythm; (2) Expressive Flattening, where the loss of paralinguistic cues leads to literal, inappropriate responses; and (3) Repair Rigidity, where architectural gating prevents users from correcting errors in real-time. Through system-level analysis, we demonstrate that these friction points should not be understood as defects or failures, but as structural consequences of a modular design that prioritizes control over fluidity. We conclude that building natural spoken AI is an infrastructure design challenge, requiring a shift from optimizing isolated components to carefully choreographing the seams between them. 

---
# Bounding Hallucinations: Information-Theoretic Guarantees for RAG Systems via Merlin-Arthur Protocols 

**Authors**: Björn Deiseroth, Max Henning Höth, Kristian Kersting, Letitia Parcalabescu  

**Link**: [PDF](https://arxiv.org/pdf/2512.11614)  

**Abstract**: Retrieval-augmented generation (RAG) models rely on retrieved evidence to guide large language model (LLM) generators, yet current systems treat retrieval as a weak heuristic rather than verifiable evidence. As a result, LLMs answer without support, hallucinate under incomplete or misleading context, and rely on spurious evidence. We introduce a training framework that treats the entire RAG pipeline -- both the retriever and the generator -- as an interactive proof system via an adaptation of the Merlin-Arthur (M/A) protocol. Arthur (the generator LLM) trains on questions of unkown provenance: Merlin provides helpful evidence, while Morgana injects adversarial, misleading context. Both use a linear-time XAI method to identify and modify the evidence most influential to Arthur. Consequently, Arthur learns to (i) answer when the context support the answer, (ii) reject when evidence is insufficient, and (iii) rely on the specific context spans that truly ground the answer. We further introduce a rigorous evaluation framework to disentangle explanation fidelity from baseline predictive errors. This allows us to introduce and measure the Explained Information Fraction (EIF), which normalizes M/A certified mutual-information guarantees relative to model capacity and imperfect benchmarks. Across three RAG datasets and two model families of varying sizes, M/A-trained LLMs show improved groundedness, completeness, soundness, and reject behavior, as well as reduced hallucinations -- without needing manually annotated unanswerable questions. The retriever likewise improves recall and MRR through automatically generated M/A hard positives and negatives. Our results demonstrate that autonomous interactive-proof-style supervision provides a principled and practical path toward reliable RAG systems that treat retrieved documents not as suggestions, but as verifiable evidence. 

---
# Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs 

**Authors**: Mohor Banerjee, Nadya Yuki Wangsajaya, Syed Ali Redha Alsagoff, Min Sen Tan, Zachary Choy Kit Chun, Alvin Chan Guo Wei  

**Link**: [PDF](https://arxiv.org/pdf/2512.11509)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities in natural language understanding and reasoning, but suffer from hallucination: the generation of factually incorrect content. While numerous methods have been developed to reduce hallucinations, their impact on creative generations remains unexplored. This gap is particularly critical for AI-assisted scientific discovery, which requires both factual accuracy and creative hypothesis generation. We investigate how three hallucination-reduction techniques: Chain of Verification (CoVe), Decoding by Contrasting Layers (DoLa), and Retrieval-Augmented Generation (RAG), affect creativity in LLMs. Evaluating multiple model families (LLaMA, Qwen, Mistral) at varying scales (1B - 70B parameters) on two creativity benchmarks (NeoCoder and CS4), we find that these methods have opposing effects on divergent creativity. CoVe enhances divergent thinking, DoLa suppresses it, and RAG shows minimal impact. Our findings provide guidance for selecting appropriate hallucination-reduction methods in scientific applications, where the balance between factual accuracy and creative exploration is crucial. 

---
# MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA 

**Authors**: Seonok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.10996)  

**Abstract**: Recent advancements in retrieval-augmented generation (RAG) have significantly enhanced the ability of large language models (LLMs) to perform complex question-answering (QA) tasks. In this paper, we introduce MedBioRAG, a retrieval-augmented model designed to improve biomedical QA performance through a combination of semantic and lexical search, document retrieval, and supervised fine-tuning. MedBioRAG efficiently retrieves and ranks relevant biomedical documents, enabling precise and context-aware response generation. We evaluate MedBioRAG across text retrieval, close-ended QA, and long-form QA tasks using benchmark datasets such as NFCorpus, TREC-COVID, MedQA, PubMedQA, and BioASQ. Experimental results demonstrate that MedBioRAG outperforms previous state-of-the-art (SoTA) models and the GPT-4o base model in all evaluated tasks. Notably, our approach improves NDCG and MRR scores for document retrieval, while achieving higher accuracy in close-ended QA and ROUGE scores in long-form QA. Our findings highlight the effectiveness of semantic search-based retrieval and LLM fine-tuning in biomedical applications. 

---
