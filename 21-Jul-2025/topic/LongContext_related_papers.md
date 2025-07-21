# DENSE: Longitudinal Progress Note Generation with Temporal Modeling of Heterogeneous Clinical Notes Across Hospital Visits 

**Authors**: Garapati Keerthana, Manik Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.14079)  

**Abstract**: Progress notes are among the most clinically meaningful artifacts in an Electronic Health Record (EHR), offering temporally grounded insights into a patient's evolving condition, treatments, and care decisions. Despite their importance, they are severely underrepresented in large-scale EHR datasets. For instance, in the widely used Medical Information Mart for Intensive Care III (MIMIC-III) dataset, only about $8.56\%$ of hospital visits include progress notes, leaving gaps in longitudinal patient narratives. In contrast, the dataset contains a diverse array of other note types, each capturing different aspects of care.
We present DENSE (Documenting Evolving Progress Notes from Scattered Evidence), a system designed to align with clinical documentation workflows by simulating how physicians reference past encounters while drafting progress notes. The system introduces a fine-grained note categorization and a temporal alignment mechanism that organizes heterogeneous notes across visits into structured, chronological inputs. At its core, DENSE leverages a clinically informed retrieval strategy to identify temporally and semantically relevant content from both current and prior visits. This retrieved evidence is used to prompt a large language model (LLM) to generate clinically coherent and temporally aware progress notes.
We evaluate DENSE on a curated cohort of patients with multiple visits and complete progress note documentation. The generated notes demonstrate strong longitudinal fidelity, achieving a temporal alignment ratio of $1.089$, surpassing the continuity observed in original notes. By restoring narrative coherence across fragmented documentation, our system supports improved downstream tasks such as summarization, predictive modeling, and clinical decision support, offering a scalable solution for LLM-driven note synthesis in real-world healthcare settings. 

---
# LoopServe: An Adaptive Dual-phase LLM Inference Acceleration System for Multi-Turn Dialogues 

**Authors**: Haoyang Li, Zhanchao Xu, Yiming Li, Xuejia Chen, Darian Li, Anxin Tian, Qingfa Xiao, Cheng Deng, Jun Wang, Qing Li, Lei Chen, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.13681)  

**Abstract**: Multi-turn dialogues are essential in many real-world applications of large language models, such as chatbots and virtual assistants. As conversation histories become longer, existing large language models face increasing computational and memory challenges, which hinder their ability to provide efficient and responsive interactions. Most current acceleration methods either compress the context or optimize key value caching, but they often rely on fixed or position-based heuristics that do not adapt well to the dynamic and unpredictable patterns found in actual multi-turn conversations. In this paper, we present LoopServe, an adaptive dual-phase inference acceleration framework for large language models in multi-turn dialogues. LoopServe introduces two main innovations. First, it performs online sparsification during the prefilling phase by dynamically selecting the most important parts of the attention matrix for each new input. Second, it uses progressive key value compression during decoding by adaptively maintaining a relevant and efficient cache based on the most recently generated output tokens. We also propose a \href{this https URL}{new benchmark} with eleven multi-turn datasets that reflect realistic query positions and conversational dependencies. Extensive experiments demonstrate that LoopServe consistently achieves superior effectiveness compared to existing baselines and significantly accelerates LLM inference across a wide range of long-context dialogue tasks. 

---
