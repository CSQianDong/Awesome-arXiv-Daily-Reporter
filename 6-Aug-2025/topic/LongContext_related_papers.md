# Long Story Generation via Knowledge Graph and Literary Theory 

**Authors**: Ge Shi, Kaiyu Huang, Guochen Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.03137)  

**Abstract**: The generation of a long story consisting of several thousand words is a sub-task in the field of long text generation~(LTG). Previous research has addressed this challenge through outline-based generation, which employs a multi-stage method for generating outlines into stories. However, this approach suffers from two common issues: almost inevitable theme drift caused by the loss of memory of previous outlines, and tedious plots with incoherent logic that are less appealing to human readers.
In this paper, we propose the multi-agent Story Generator structure to improve the multi-stage method, using large language models~(LLMs) as the core components of agents. To avoid theme drift, we introduce a memory storage model comprising two components: a long-term memory storage that identifies the most important memories, thereby preventing theme drift; and a short-term memory storage that retains the latest outlines from each generation round. To incorporate engaging elements into the story, we design a story theme obstacle framework based on literary narratology theory that introduces uncertain factors and evaluation criteria to generate outline. This framework calculates the similarity of the former storyline and enhances the appeal of the story by building a knowledge graph and integrating new node content. Additionally, we establish a multi-agent interaction stage to simulate writer-reader interaction through dialogue and revise the story text according to feedback, to ensure it remains consistent and logical. Evaluations against previous methods demonstrate that our approach can generate higher-quality long stories. 

---
# PyLate: Flexible Training and Retrieval for Late Interaction Models 

**Authors**: Antoine Chaffin, Raphaël Sourty  

**Link**: [PDF](https://arxiv.org/pdf/2508.03555)  

**Abstract**: Neural ranking has become a cornerstone of modern information retrieval. While single vector search remains the dominant paradigm, it suffers from the shortcoming of compressing all the information into a single vector. This compression leads to notable performance degradation in out-of-domain, long-context, and reasoning-intensive retrieval tasks. Multi-vector approaches pioneered by ColBERT aim to address these limitations by preserving individual token embeddings and computing similarity via the MaxSim operator. This architecture has demonstrated superior empirical advantages, including enhanced out-of-domain generalization, long-context handling, and performance in complex retrieval scenarios. Despite these compelling empirical results and clear theoretical advantages, the practical adoption and public availability of late interaction models remain low compared to their single-vector counterparts, primarily due to a lack of accessible and modular tools for training and experimenting with such models. To bridge this gap, we introduce PyLate, a streamlined library built on top of Sentence Transformers to support multi-vector architectures natively, inheriting its efficient training, advanced logging, and automated model card generation while requiring minimal code changes to code templates users are already familiar with. By offering multi-vector-specific features such as efficient indexes, PyLate aims to accelerate research and real-world application of late interaction models, thereby unlocking their full potential in modern IR systems. Finally, PyLate has already enabled the development of state-of-the-art models, including GTE-ModernColBERT and Reason-ModernColBERT, demonstrating its practical utility for both research and production environments. 

---
# Nemori: Self-Organizing Agent Memory Inspired by Cognitive Science 

**Authors**: Jiayan Nan, Wenquan Ma, Wenlong Wu, Yize Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.03341)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities, yet their inability to maintain persistent memory in long contexts limits their effectiveness as autonomous agents in long-term interactions. While existing memory systems have made progress, their reliance on arbitrary granularity for defining the basic memory unit and passive, rule-based mechanisms for knowledge extraction limits their capacity for genuine learning and evolution. To address these foundational limitations, we present Nemori, a novel self-organizing memory architecture inspired by human cognitive principles. Nemori's core innovation is twofold: First, its Two-Step Alignment Principle, inspired by Event Segmentation Theory, provides a principled, top-down method for autonomously organizing the raw conversational stream into semantically coherent episodes, solving the critical issue of memory granularity. Second, its Predict-Calibrate Principle, inspired by the Free-energy Principle, enables the agent to proactively learn from prediction gaps, moving beyond pre-defined heuristics to achieve adaptive knowledge evolution. This offers a viable path toward handling the long-term, dynamic workflows of autonomous agents. Extensive experiments on the LoCoMo and LongMemEval benchmarks demonstrate that Nemori significantly outperforms prior state-of-the-art systems, with its advantage being particularly pronounced in longer contexts. 

---
# Semantic-aware Graph-guided Behavior Sequences Generation with Large Language Models for Smart Homes 

**Authors**: Zhiyao Xu, Dan Zhao, Qingsong Zou, Qing Li, Yong Jiang, Yuhang Wang, Jingyu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.03484)  

**Abstract**: As smart homes become increasingly prevalent, intelligent models are widely used for tasks such as anomaly detection and behavior prediction. These models are typically trained on static datasets, making them brittle to behavioral drift caused by seasonal changes, lifestyle shifts, or evolving routines. However, collecting new behavior data for retraining is often impractical due to its slow pace, high cost, and privacy concerns. In this paper, we propose SmartGen, an LLM-based framework that synthesizes context-aware user behavior data to support continual adaptation of downstream smart home models. SmartGen consists of four key components. First, we design a Time and Semantic-aware Split module to divide long behavior sequences into manageable, semantically coherent subsequences under dual time-span constraints. Second, we propose Semantic-aware Sequence Compression to reduce input length while preserving representative semantics by clustering behavior mapping in latent space. Third, we introduce Graph-guided Sequence Synthesis, which constructs a behavior relationship graph and encodes frequent transitions into prompts, guiding the LLM to generate data aligned with contextual changes while retaining core behavior patterns. Finally, we design a Two-stage Outlier Filter to identify and remove implausible or semantically inconsistent outputs, aiming to improve the factual coherence and behavioral validity of the generated sequences. Experiments on three real-world datasets demonstrate that SmartGen significantly enhances model performance on anomaly detection and behavior prediction tasks under behavioral drift, with anomaly detection improving by 85.43% and behavior prediction by 70.51% on average. The code is available at this https URL. 

---
# Beyond the Wavefunction: Qualia Abstraction Language Mechanics and the Grammar of Awareness 

**Authors**: Mikołaj Sienicki, Krzysztof Sienicki  

**Link**: [PDF](https://arxiv.org/pdf/2508.02755)  

**Abstract**: We propose a formal reconstruction of quantum mechanics grounded not in external mathematical abstractions, but in the structured dynamics of subjective experience. The Qualia Abstraction Language (QAL) models physical systems as evolving streams of introspective units, structured sequences of modality, shape, and functional effect, rather than as state vectors in Hilbert space. This approach reimagines core quantum concepts: superposition becomes a form of structured ambiguity; collapse is reframed as an introspective contraction; and entanglement is modeled as semantic resonance across streams of qualia. Drawing on insights from nominalist philosophy and oversight theoretic limits in AI, we argue that the observer paradox in quantum mechanics reflects not an ontological lacuna, but a linguistic one: the absence of a formal vocabulary for modeling first person structure. QAL introduces such a vocabulary, providing a morphodynamic framework that embeds the observer within the system and replaces abstract projection with endogenous transformation. We analyze the alignment of QAL with endophysical approaches, contrast it with standard interpretations of quantum theory, and explore its implications for a post Platonist, introspectively grounded physics. 

---
# SmallKV: Small Model Assisted Compensation of KV Cache Compression for Efficient LLM Inference 

**Authors**: Yi Zhao, Yajuan Peng, Cam-Tu Nguyen, Zuchao Li, Xiaoliang Wang, Hai Zhao, Xiaoming Fu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02751)  

**Abstract**: KV cache eviction has emerged as an effective solution to alleviate resource constraints faced by LLMs in long-context scenarios. However, existing token-level eviction methods often overlook two critical aspects: (1) their irreversible eviction strategy fails to adapt to dynamic attention patterns during decoding (the saliency shift problem), and (2) they treat both marginally important tokens and truly unimportant tokens equally, despite the collective significance of marginal tokens to model performance (the marginal information over-compression problem). To address these issues, we design two compensation mechanisms based on the high similarity of attention matrices between LLMs of different scales. We propose SmallKV, a small model assisted compensation method for KV cache compression. SmallKV can maintain attention matching between different-scale LLMs to: 1) assist the larger model in perceiving globally important information of attention; and 2) use the smaller model's attention scores to approximate those of marginal tokens in the larger model. Extensive experiments on benchmarks including GSM8K, BBH, MT-Bench, and LongBench demonstrate the effectiveness of SmallKV. Moreover, efficiency evaluations show that SmallKV achieves 1.75 - 2.56 times higher throughput than baseline methods, highlighting its potential for efficient and performant LLM inference in resource constrained environments. 

---
