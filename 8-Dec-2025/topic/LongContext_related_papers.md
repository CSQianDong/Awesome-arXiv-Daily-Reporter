# ChipMind: Retrieval-Augmented Reasoning for Long-Context Circuit Design Specifications 

**Authors**: Changwen Xing, SamZaak Wong, Xinlai Wan, Yanfeng Lu, Mengli Zhang, Zebin Ma, Lei Qi, Zhengxiong Li, Nan Guan, Zhe Jiang, Xi Wang, Jun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.05371)  

**Abstract**: While Large Language Models (LLMs) demonstrate immense potential for automating integrated circuit (IC) development, their practical deployment is fundamentally limited by restricted context windows. Existing context-extension methods struggle to achieve effective semantic modeling and thorough multi-hop reasoning over extensive, intricate circuit specifications. To address this, we introduce ChipMind, a novel knowledge graph-augmented reasoning framework specifically designed for lengthy IC specifications. ChipMind first transforms circuit specifications into a domain-specific knowledge graph ChipKG through the Circuit Semantic-Aware Knowledge Graph Construction methodology. It then leverages the ChipKG-Augmented Reasoning mechanism, combining information-theoretic adaptive retrieval to dynamically trace logical dependencies with intent-aware semantic filtering to prune irrelevant noise, effectively balancing retrieval completeness and precision. Evaluated on an industrial-scale specification reasoning benchmark, ChipMind significantly outperforms state-of-the-art baselines, achieving an average improvement of 34.59% (up to 72.73%). Our framework bridges a critical gap between academic research and practical industrial deployment of LLM-aided Hardware Design (LAD). 

---
# Active Video Perception: Iterative Evidence Seeking for Agentic Long Video Understanding 

**Authors**: Ziyang Wang, Honglu Zhou, Shijie Wang, Junnan Li, Caiming Xiong, Silvio Savarese, Mohit Bansal, Michael S. Ryoo, Juan Carlos Niebles  

**Link**: [PDF](https://arxiv.org/pdf/2512.05774)  

**Abstract**: Long video understanding (LVU) is challenging because answering real-world queries often depends on sparse, temporally dispersed cues buried in hours of mostly redundant and irrelevant content. While agentic pipelines improve video reasoning capabilities, prevailing frameworks rely on a query-agnostic captioner to perceive video information, which wastes computation on irrelevant content and blurs fine-grained temporal and spatial information. Motivated by active perception theory, we argue that LVU agents should actively decide what, when, and where to observe, and continuously assess whether the current observation is sufficient to answer the query. We present Active Video Perception (AVP), an evidence-seeking framework that treats the video as an interactive environment and acquires compact, queryrelevant evidence directly from pixels. Concretely, AVP runs an iterative plan-observe-reflect process with MLLM agents. In each round, a planner proposes targeted video interactions, an observer executes them to extract time-stamped evidence, and a reflector evaluates the sufficiency of the evidence for the query, either halting with an answer or triggering further observation. Across five LVU benchmarks, AVP achieves highest performance with significant improvements. Notably, AVP outperforms the best agentic method by 5.7% in average accuracy while only requires 18.4% inference time and 12.4% input tokens. 

---
