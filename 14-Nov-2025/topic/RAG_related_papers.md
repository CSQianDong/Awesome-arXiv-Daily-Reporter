# TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs 

**Authors**: Shuyi Liu, Yuming Shang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10375)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful framework for enhancing the capabilities of Large Language Models (LLMs) by integrating retrieval-based methods with generative models. As external knowledge repositories continue to expand and the parametric knowledge within models becomes outdated, a critical challenge for RAG systems is resolving conflicts between retrieved external information and LLMs' internal knowledge, which can significantly compromise the accuracy and reliability of generated content. However, existing approaches to conflict resolution typically operate at the token or semantic level, often leading to fragmented and partial understanding of factual discrepancies between LLMs' knowledge and context, particularly in knowledge-intensive tasks. To address this limitation, we propose TruthfulRAG, the first framework that leverages Knowledge Graphs (KGs) to resolve factual-level knowledge conflicts in RAG systems. Specifically, TruthfulRAG constructs KGs by systematically extracting triples from retrieved content, utilizes query-based graph retrieval to identify relevant knowledge, and employs entropy-based filtering mechanisms to precisely locate conflicting elements and mitigate factual inconsistencies, thereby enabling LLMs to generate faithful and accurate responses. Extensive experiments reveal that TruthfulRAG outperforms existing methods, effectively alleviating knowledge conflicts and improving the robustness and trustworthiness of RAG systems. 

---
# Language Drift in Multilingual Retrieval-Augmented Generation: Characterization and Decoding-Time Mitigation 

**Authors**: Bo Li, Zhenghua Xu, Rui Xie  

**Link**: [PDF](https://arxiv.org/pdf/2511.09984)  

**Abstract**: Multilingual Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to perform knowledge-intensive tasks in multilingual settings by leveraging retrieved documents as external evidence. However, when the retrieved evidence differs in language from the user query and in-context exemplars, the model often exhibits language drift by generating responses in an unintended language. This phenomenon is especially pronounced during reasoning-intensive decoding, such as Chain-of-Thought (CoT) generation, where intermediate steps introduce further language instability. In this paper, we systematically study output language drift in multilingual RAG across multiple datasets, languages, and LLM backbones. Our controlled experiments reveal that the drift results not from comprehension failure but from decoder-level collapse, where dominant token distributions and high-frequency English patterns dominate the intended generation language. We further observe that English serves as a semantic attractor under cross-lingual conditions, emerging as both the strongest interference source and the most frequent fallback language.
To mitigate this, we propose Soft Constrained Decoding (SCD), a lightweight, training-free decoding strategy that gently steers generation toward the target language by penalizing non-target-language tokens. SCD is model-agnostic and can be applied to any generation algorithm without modifying the architecture or requiring additional data. Experiments across three multilingual datasets and multiple typologically diverse languages show that SCD consistently improves language alignment and task performance, providing an effective and generalizable solution in multilingual RAG. 

---
# REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering 

**Authors**: Yijie Zhu, Haojie Zhou, Wanting Hong, Tailin Liu, Ning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.09966)  

**Abstract**: Retrieval-augmented generation (RAG) has been extensively employed to mitigate hallucinations in large language models (LLMs). However, existing methods for multi-hop reasoning tasks often lack global planning, increasing the risk of falling into local reasoning impasses. Insufficient exploitation of retrieved content and the neglect of latent clues fail to ensure the accuracy of reasoning outcomes. To overcome these limitations, we propose Recursive Evaluation and Adaptive Planning (REAP), whose core idea is to explicitly maintain structured sub-tasks and facts related to the current task through the Sub-task Planner (SP) and Fact Extractor (FE) modules. SP maintains a global perspective, guiding the overall reasoning direction and evaluating the task state based on the outcomes of FE, enabling dynamic optimization of the task-solving trajectory. FE performs fine-grained analysis over retrieved content to extract reliable answers and clues. These two modules incrementally enrich a logically coherent representation of global knowledge, enhancing the reliability and the traceability of the reasoning process. Furthermore, we propose a unified task paradigm design that enables effective multi-task fine-tuning, significantly enhancing SP's performance on complex, data-scarce tasks. We conduct extensive experiments on multiple public multi-hop datasets, and the results demonstrate that our method significantly outperforms existing RAG methods in both in-domain and out-of-domain settings, validating its effectiveness in complex multi-hop reasoning tasks. 

---
# MINDS: A Cross-cultural Dialogue Corpus for Social Norm Classification and Adherence Detection 

**Authors**: Pritish Sahu, Anirudh Som, Dimitra Vergyri, Ajay Divakaran  

**Link**: [PDF](https://arxiv.org/pdf/2511.09918)  

**Abstract**: Social norms are implicit, culturally grounded expectations that guide interpersonal communication. Unlike factual commonsense, norm reasoning is subjective, context-dependent, and varies across cultures, posing challenges for computational models. Prior works provide valuable normative annotations but mostly target isolated utterances or synthetic dialogues, limiting their ability to capture the fluid, multi-turn nature of real-world conversations. In this work, we present Norm-RAG, a retrieval-augmented, agentic framework for nuanced social norm inference in multi-turn dialogues. Norm-RAG models utterance-level attributes including communicative intent, speaker roles, interpersonal framing, and linguistic cues and grounds them in structured normative documentation retrieved via a novel Semantic Chunking approach. This enables interpretable and context-aware reasoning about norm adherence and violation across multilingual dialogues. We further introduce MINDS (Multilingual Interactions with Norm-Driven Speech), a bilingual dataset comprising 31 multi-turn Mandarin-English and Spanish-English conversations. Each turn is annotated for norm category and adherence status using multi-annotator consensus, reflecting cross-cultural and realistic norm expression. Our experiments show that Norm-RAG improves norm detection and generalization, demonstrates improved performance for culturally adaptive and socially intelligent dialogue systems. 

---
# Answering Students' Questions on Course Forums Using Multiple Chain-of-Thought Reasoning and Finetuning RAG-Enabled LLM 

**Authors**: Neo Wang, Sonit Singh  

**Link**: [PDF](https://arxiv.org/pdf/2511.09831)  

**Abstract**: The course forums are increasingly significant and play vital role in facilitating student discussions and answering their questions related to the course. It provides a platform for students to post their questions related to the content and admin issues related to the course. However, there are several challenges due to the increase in the number of students enrolled in the course. The primary challenge is that students' queries cannot be responded immediately and the instructors have to face lots of repetitive questions. To mitigate these issues, we propose a question answering system based on large language model with retrieval augmented generation (RAG) method. This work focuses on designing a question answering system with open source Large Language Model (LLM) and fine-tuning it on the relevant course dataset. To further improve the performance, we use a local knowledge base and applied RAG method to retrieve relevant documents relevant to students' queries, where the local knowledge base contains all the course content. To mitigate the hallucination of LLMs, We also integrate it with multi chain-of-thought reasoning to overcome the challenge of hallucination in LLMs. In this work, we experiment fine-tuned LLM with RAG method on the HotpotQA dataset. The experimental results demonstrate that the fine-tuned LLM with RAG method has a strong performance on question answering task. 

---
# Modeling Uncertainty Trends for Timely Retrieval in Dynamic RAG 

**Authors**: Bo Li, Tian Tian, Zhenghua Xu, Hao Cheng, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2511.09980)  

**Abstract**: Dynamic retrieval-augmented generation (RAG) allows large language models (LLMs) to fetch external knowledge on demand, offering greater adaptability than static RAG. A central challenge in this setting lies in determining the optimal timing for retrieval. Existing methods often trigger retrieval based on low token-level confidence, which may lead to delayed intervention after errors have already propagated. We introduce Entropy-Trend Constraint (ETC), a training-free method that determines optimal retrieval timing by modeling the dynamics of token-level uncertainty. Specifically, ETC utilizes first- and second-order differences of the entropy sequence to detect emerging uncertainty trends, enabling earlier and more precise retrieval. Experiments on six QA benchmarks with three LLM backbones demonstrate that ETC consistently outperforms strong baselines while reducing retrieval frequency. ETC is particularly effective in domain-specific scenarios, exhibiting robust generalization capabilities. Ablation studies and qualitative analyses further confirm that trend-aware uncertainty modeling yields more effective retrieval timing. The method is plug-and-play, model-agnostic, and readily integrable into existing decoding pipelines. Implementation code is included in the supplementary materials. 

---
# TARG: Training-Free Adaptive Retrieval Gating for Efficient RAG 

**Authors**: Yufeng Wang, Lu wei, Haibin Ling  

**Link**: [PDF](https://arxiv.org/pdf/2511.09803)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves factuality but retrieving for every query often hurts quality while inflating tokens and latency. We propose Training-free Adaptive Retrieval Gating (TARG), a single-shot policy that decides when to retrieve using only a short, no-context draft from the base model. From the draft's prefix logits, TARG computes lightweight uncertainty scores: mean token entropy, a margin signal derived from the top-1/top-2 logit gap via a monotone link, or small-N variance across a handful of stochastic prefixes, and triggers retrieval only when the score exceeds a threshold. The gate is model agnostic, adds only tens to hundreds of draft tokens, and requires no additional training or auxiliary heads. On NQ-Open, TriviaQA, and PopQA, TARG consistently shifts the accuracy-efficiency frontier: compared with Always-RAG, TARG matches or improves EM/F1 while reducing retrieval by 70-90% and cutting end-to-end latency, and it remains close to Never-RAG in overhead. A central empirical finding is that under modern instruction-tuned LLMs the margin signal is a robust default (entropy compresses as backbones sharpen), with small-N variance offering a conservative, budget-first alternative. We provide ablations over gate type and prefix length and use a delta-latency view to make budget trade-offs explicit. 

---
# RAGFort: Dual-Path Defense Against Proprietary Knowledge Base Extraction in Retrieval-Augmented Generation 

**Authors**: Qinfeng Li, Miao Pan, Ke Xiong, Ge Su, Zhiqiang Shen, Yan Liu, Bing Sun, Hao Peng, Xuhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10128)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems deployed over proprietary knowledge bases face growing threats from reconstruction attacks that aggregate model responses to replicate knowledge bases. Such attacks exploit both intra-class and inter-class paths, progressively extracting fine-grained knowledge within topics and diffusing it across semantically related ones, thereby enabling comprehensive extraction of the original knowledge base. However, existing defenses target only one path, leaving the other unprotected. We conduct a systematic exploration to assess the impact of protecting each path independently and find that joint protection is essential for effective defense. Based on this, we propose RAGFort, a structure-aware dual-module defense combining "contrastive reindexing" for inter-class isolation and "constrained cascade generation" for intra-class protection. Experiments across security, performance, and robustness confirm that RAGFort significantly reduces reconstruction success while preserving answer quality, offering comprehensive defense against knowledge base extraction attacks. 

---
# fastbmRAG: A Fast Graph-Based RAG Framework for Efficient Processing of Large-Scale Biomedical Literature 

**Authors**: Guofeng Meng, Li Shen, Qiuyan Zhong, Wei Wang, Haizhou Zhang, Xiaozhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10014)  

**Abstract**: Large language models (LLMs) are rapidly transforming various domains, including biomedicine and healthcare, and demonstrate remarkable potential from scientific research to new drug discovery. Graph-based retrieval-augmented generation (RAG) systems, as a useful application of LLMs, can improve contextual reasoning through structured entity and relationship identification from long-context knowledge, e.g. biomedical literature. Even though many advantages over naive RAGs, most of graph-based RAGs are computationally intensive, which limits their application to large-scale dataset. To address this issue, we introduce fastbmRAG, an fast graph-based RAG optimized for biomedical literature. Utilizing well organized structure of biomedical papers, fastbmRAG divides the construction of knowledge graph into two stages, first drafting graphs using abstracts; and second, refining them using main texts guided by vector-based entity linking, which minimizes redundancy and computational load. Our evaluations demonstrate that fastbmRAG is over 10x faster than existing graph-RAG tools and achieve superior coverage and accuracy to input knowledge. FastbmRAG provides a fast solution for quickly understanding, summarizing, and answering questions about biomedical literature on a large scale. FastbmRAG is public available in this https URL. 

---
