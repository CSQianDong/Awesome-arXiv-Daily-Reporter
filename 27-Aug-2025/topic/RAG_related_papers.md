# Retrieval-Augmented Generation for Natural Language Art Provenance Searches in the Getty Provenance Index 

**Authors**: Mathew Henrickson  

**Link**: [PDF](https://arxiv.org/pdf/2508.19093)  

**Abstract**: This research presents a Retrieval-Augmented Generation (RAG) framework for art provenance studies, focusing on the Getty Provenance Index. Provenance research establishes the ownership history of artworks, which is essential for verifying authenticity, supporting restitution and legal claims, and understanding the cultural and historical context of art objects. The process is complicated by fragmented, multilingual archival data that hinders efficient retrieval. Current search portals require precise metadata, limiting exploratory searches. Our method enables natural-language and multilingual searches through semantic retrieval and contextual summarization, reducing dependence on metadata structures. We assess RAG's capability to retrieve and summarize auction records using a 10,000-record sample from the Getty Provenance Index - German Sales. The results show this approach provides a scalable solution for navigating art market archives, offering a practical tool for historians and cultural heritage professionals conducting historically sensitive research. 

---
# Diverse And Private Synthetic Datasets Generation for RAG evaluation: A multi-agent framework 

**Authors**: Ilias Driouich, Hongliu Cao, Eoin Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2508.18929)  

**Abstract**: Retrieval-augmented generation (RAG) systems improve large language model outputs by incorporating external knowledge, enabling more informed and context-aware responses. However, the effectiveness and trustworthiness of these systems critically depends on how they are evaluated, particularly on whether the evaluation process captures real-world constraints like protecting sensitive information. While current evaluation efforts for RAG systems have primarily focused on the development of performance metrics, far less attention has been given to the design and quality of the underlying evaluation datasets, despite their pivotal role in enabling meaningful, reliable assessments. In this work, we introduce a novel multi-agent framework for generating synthetic QA datasets for RAG evaluation that prioritize semantic diversity and privacy preservation. Our approach involves: (1) a Diversity agent leveraging clustering techniques to maximize topical coverage and semantic variability, (2) a Privacy Agent that detects and mask sensitive information across multiple domains and (3) a QA curation agent that synthesizes private and diverse QA pairs suitable as ground truth for RAG evaluation. Extensive experiments demonstrate that our evaluation sets outperform baseline methods in diversity and achieve robust privacy masking on domain-specific datasets. This work offers a practical and ethically aligned pathway toward safer, more comprehensive RAG system evaluation, laying the foundation for future enhancements aligned with evolving AI regulations and compliance standards. 

---
# Chronological Passage Assembling in RAG framework for Temporal Question Answering 

**Authors**: Byeongjeong Kim, Jeonghyun Park, Joonho Yang, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.18748)  

**Abstract**: Long-context question answering over narrative tasks is challenging because correct answers often hinge on reconstructing a coherent timeline of events while preserving contextual flow in a limited context window. Retrieval-augmented generation (RAG) indexing methods aim to address this challenge by selectively retrieving only necessary document segments. However, narrative texts possess unique characteristics that limit the effectiveness of these existing approaches. Specifically, understanding narrative texts requires more than isolated segments, as the broader context and sequential relationships between segments are crucial for comprehension. To address these limitations, we propose ChronoRAG, a novel RAG framework specialized for narrative texts. This approach focuses on two essential aspects: refining dispersed document information into coherent and structured passages, and preserving narrative flow by explicitly capturing and maintaining the temporal order among retrieved passages. We empirically demonstrate the effectiveness of ChronoRAG through experiments on the NarrativeQA dataset, showing substantial improvements in tasks requiring both factual identification and comprehension of complex sequential relationships, underscoring that reasoning over temporal order is crucial in resolving narrative QA. 

---
# Beyond the Textual: Generating Coherent Visual Options for MCQs 

**Authors**: Wanqiang Wang, Longzhu He, Wei Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.18772)  

**Abstract**: Multiple-choice questions (MCQs) play a crucial role in fostering deep thinking and knowledge integration in education. However, previous research has primarily focused on generating MCQs with textual options, but it largely overlooks the visual options. Moreover, generating high-quality distractors remains a major challenge due to the high cost and limited scalability of manual authoring. To tackle these problems, we propose a Cross-modal Options Synthesis (CmOS), a novel framework for generating educational MCQs with visual options. Our framework integrates Multimodal Chain-of-Thought (MCoT) reasoning process and Retrieval-Augmented Generation (RAG) to produce semantically plausible and visually similar answer and distractors. It also includes a discrimination module to identify content suitable for visual options. Experimental results on test tasks demonstrate the superiority of CmOS in content discrimination, question generation and visual option generation over existing methods across various subjects and educational levels. 

---
# UniC-RAG: Universal Knowledge Corruption Attacks to Retrieval-Augmented Generation 

**Authors**: Runpeng Geng, Yanting Wang, Ying Chen, Jinyuan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2508.18652)  

**Abstract**: Retrieval-augmented generation (RAG) systems are widely deployed in real-world applications in diverse domains such as finance, healthcare, and cybersecurity. However, many studies showed that they are vulnerable to knowledge corruption attacks, where an attacker can inject adversarial texts into the knowledge database of a RAG system to induce the LLM to generate attacker-desired outputs. Existing studies mainly focus on attacking specific queries or queries with similar topics (or keywords). In this work, we propose UniC-RAG, a universal knowledge corruption attack against RAG systems. Unlike prior work, UniC-RAG jointly optimizes a small number of adversarial texts that can simultaneously attack a large number of user queries with diverse topics and domains, enabling an attacker to achieve various malicious objectives, such as directing users to malicious websites, triggering harmful command execution, or launching denial-of-service attacks. We formulate UniC-RAG as an optimization problem and further design an effective solution to solve it, including a balanced similarity-based clustering method to enhance the attack's effectiveness. Our extensive evaluations demonstrate that UniC-RAG is highly effective and significantly outperforms baselines. For instance, UniC-RAG could achieve over 90% attack success rate by injecting 100 adversarial texts into a knowledge database with millions of texts to simultaneously attack a large set of user queries (e.g., 2,000). Additionally, we evaluate existing defenses and show that they are insufficient to defend against UniC-RAG, highlighting the need for new defense mechanisms in RAG systems. 

---
# Reflection-Enhanced Meta-Optimization Integrating TextGrad-style Prompt Optimization with Memory-Driven Self-Evolution 

**Authors**: Chunlong Wu, Zhibo Qu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18749)  

**Abstract**: Recent advances in prompt optimization, exemplified by methods such as TextGrad, enable automatic, gradient-like refinement of textual prompts to enhance the performance of large language models (LLMs) on specific downstream tasks. However, current approaches are typically stateless and operate independently across optimization runs, lacking mechanisms to preserve and leverage historical optimization experience. Furthermore, they are susceptible to overfitting, often yielding prompt updates that generalize poorly beyond the immediate task context.
To address these limitations, we propose Reflection-Enhanced Meta-Optimization (REMO), a novel framework that integrates (1) a memory-augmented Reflection Retrieval-Augmented Generation (RAG) module - structured as a "mistake notebook" and (2) a Self-Adaptive Optimizer, implemented via an LLM-driven meta-controller that synthesizes epoch-level reflective insights to iteratively improve system-level prompting strategies. This architecture enables not only local, fine-grained prompt tuning akin to TextGrad, but also the systematic accumulation and reuse of cross-run optimization knowledge, thereby supporting continual improvement over time.
We instantiate the REMO framework using Qwen3-32B in standard inference mode - without explicit chain-of-thought prompting - and evaluate its efficacy on the GSM8K benchmark for mathematical reasoning. Experimental results demonstrate that, compared to a TextGrad baseline, REMO achieves more stable and robust generalization, albeit at the cost of increased computational overhead. We provide a detailed exposition of the algorithmic design, conduct a qualitative and quantitative analysis of optimization dynamics, and present a comprehensive ablation study to elucidate the contributions of each component. 

---
