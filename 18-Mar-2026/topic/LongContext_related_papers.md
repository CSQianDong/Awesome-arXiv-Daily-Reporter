# Chronos: Temporal-Aware Conversational Agents with Structured Event Retrieval for Long-Term Memory 

**Authors**: Sahil Sen, Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah  

**Link**: [PDF](https://arxiv.org/pdf/2603.16862)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled conversational AI agents to engage in extended multi-turn interactions spanning weeks or months. However, existing memory systems struggle to reason over temporally grounded facts and preferences that evolve across months of interaction and lack effective retrieval strategies for multi-hop, time-sensitive queries over long dialogue histories. We introduce Chronos, a novel temporal-aware memory framework that decomposes raw dialogue into subject-verb-object event tuples with resolved datetime ranges and entity aliases, indexing them in a structured event calendar alongside a turn calendar that preserves full conversational context. At query time, Chronos applies dynamic prompting to generate tailored retrieval guidance for each question, directing the agent on what to retrieve, how to filter across time ranges, and how to approach multi-hop reasoning through an iterative tool-calling loop over both calendars. We evaluate Chronos with 8 LLMs, both open-source and closed-source, on the LongMemEvalS benchmark comprising 500 questions spanning six categories of dialogue history tasks. Chronos Low achieves 92.60% and Chronos High scores 95.60% accuracy, setting a new state of the art with an improvement of 7.67% over the best prior system. Ablation results reveal the events calendar accounts for a 58.9% gain on the baseline while all other components yield improvements between 15.5% and 22.3%. Notably, Chronos Low alone surpasses prior approaches evaluated under their strongest model configurations. 

---
# DanceHA: A Multi-Agent Framework for Document-Level Aspect-Based Sentiment Analysis 

**Authors**: Lei Wang, Min Huang, Eduard Dragut  

**Link**: [PDF](https://arxiv.org/pdf/2603.16546)  

**Abstract**: Aspect-Based Sentiment Intensity Analysis (ABSIA) has garnered increasing attention, though research largely focuses on domain-specific, sentence-level settings. In contrast, document-level ABSIA--particularly in addressing complex tasks like extracting Aspect-Category-Opinion-Sentiment-Intensity (ACOSI) tuples--remains underexplored. In this work, we introduce DanceHA, a multi-agent framework designed for open-ended, document-level ABSIA with informal writing styles. DanceHA has two main components: Dance, which employs a divide-and-conquer strategy to decompose the long-context ABSIA task into smaller, manageable sub-tasks for collaboration among specialized agents; and HA, Human-AI collaboration for annotation. We release Inf-ABSIA, a multi-domain document-level ABSIA dataset featuring fine-grained and high-accuracy labels from DanceHA. Extensive experiments demonstrate the effectiveness of our agentic framework and show that the multi-agent knowledge in DanceHA can be effectively transferred into student models. Our results highlight the importance of the overlooked informal styles in ABSIA, as they often intensify opinions tied to specific aspects. 

---
# AdaMem: Adaptive User-Centric Memory for Long-Horizon Dialogue Agents 

**Authors**: Shannan Yan, Jingchen Ni, Leqi Zheng, Jiajun Zhang, Peixi Wu, Dacheng Yin, Jing Lyu, Chun Yuan, Fengyun Rao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16496)  

**Abstract**: Large language model (LLM) agents increasingly rely on external memory to support long-horizon interaction, personalized assistance, and multi-step reasoning. However, existing memory systems still face three core challenges: they often rely too heavily on semantic similarity, which can miss evidence crucial for user-centric understanding; they frequently store related experiences as isolated fragments, weakening temporal and causal coherence; and they typically use static memory granularities that do not adapt well to the requirements of different questions. We propose AdaMem, an adaptive user-centric memory framework for long-horizon dialogue agents. AdaMem organizes dialogue history into working, episodic, persona, and graph memories, enabling the system to preserve recent context, structured long-term experiences, stable user traits, and relation-aware connections within a unified framework. At inference time, AdaMem first resolves the target participant, then builds a question-conditioned retrieval route that combines semantic retrieval with relation-aware graph expansion only when needed, and finally produces the answer through a role-specialized pipeline for evidence synthesis and response generation. We evaluate AdaMem on the LoCoMo and PERSONAMEM benchmarks for long-horizon reasoning and user modeling. Experimental results show that AdaMem achieves state-of-the-art performance on both benchmarks. The code will be released upon acceptance. 

---
# Recursive Language Models Meet Uncertainty: The Surprising Effectiveness of Self-Reflective Program Search for Long Context 

**Authors**: Keivan Alizadeh, Parshin Shojaee, Minsik Cho, Mehrdad Farajtabar  

**Link**: [PDF](https://arxiv.org/pdf/2603.15653)  

**Abstract**: Long-context handling remains a core challenge for language models: even with extended context windows, models often fail to reliably extract, reason over, and use the information across long contexts. Recent works like Recursive Language Models (RLM) have approached this challenge by agentic way of decomposing long contexts into recursive sub-calls through programmatic interaction at inference. While promising, the success of RLM critically depends on how these context-interaction programs are selected, which has remained largely unexplored. In this paper, we study this problem and introduce SRLM, a framework that augments programmatic context interaction with uncertainty-aware Self-Reflection. SRLM leverages three intrinsic signals: self consistency, reasoning length, and verbalized confidence. These serve as complementary indicators of a model's internal uncertainty, and the model uses them to evaluate and compare candidate context-interaction programs. Extensive experiments across diverse benchmark datasets, context lengths, and backbone models, show that SRLM consistently outperforms state-of-the-art baselines, yielding up to 22% improvement over RLM under the same time budget. Our findings show that recursion itself is not the primary driver of performance in RLM, and a simple self-reflective program search can match or surpass RLM without requiring self-query or explicit recursion mechanisms. We find that for context lengths within the model's window, RLMs with recursion often degrade performance relative to the base model, whereas SRLM yields consistent gains across both short and long contexts. We also find that RLM is less effective in tasks with semantically intensive nature, where heuristic program search is insufficient and broader contextual understanding is required, while self-reflection in SRLM provides a semantic signal that better steers reasoning in these scenarios. 

---
# A Context Alignment Pre-processor for Enhancing the Coherence of Human-LLM Dialog 

**Authors**: Ding Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.16052)  

**Abstract**: Large language models (LLMs) have made remarkable progress in generating fluent text, but they still face a critical challenge of contextual misalignment in long-term and dynamic dialogue. When human users omit premises, simplify references, or shift context abruptly during interactions with LLMs, the models may fail to capture their actual intentions, producing mechanical or off-topic responses that weaken the collaborative potential of dialogue. To address this problem, this paper proposes a computational framework called the Context Alignment Pre-processor (C.A.P.). Rather than operating during generation, C.A.P. functions as a pre-processing module between user input and response generation. The framework includes three core processes: (1) semantic expansion, which extends a user instruction to a broader semantic span including its premises, literal meaning, and implications; (2) time-weighted context retrieval, which prioritizes recent dialogue history through a temporal decay function approximating human conversational focus; and (3) alignment verification and decision branching, which evaluates whether the dialogue remains on track by measuring the semantic similarity between the current prompt and the weighted historical context. When a significant deviation is detected, C.A.P. initiates a structured clarification protocol to help users and the system recalibrate the conversation. This study presents the architecture and theoretical basis of C.A.P., drawing on cognitive science and Common Ground theory in human-computer interaction. We argue that C.A.P. is not only a technical refinement but also a step toward shifting human-computer dialogue from one-way command-execution patterns to two-way, self-correcting, partnership-based collaboration. Finally, we discuss implementation paths, evaluation methods, and implications for the future design of interactive intelligent systems. 

---
# InCoder-32B: Code Foundation Model for Industrial Scenarios 

**Authors**: Jian Yang, Wei Zhang, Jiajun Wu, Junhang Cheng, Shawn Guo, Haowen Wang, Weicheng Gu, Yaxin Du, Joseph Li, Fanglin Xu, Yizhi Li, Lin Jing, Yuanbo Wang, Yuhan Gao, Ruihao Gong, Chuan Hao, Ran Tao, Aishan Liu, Tuney Zheng, Ganqu Cui, Zhoujun Li, Mingjie Tang, Chenghua Lin, Wayne Xin Zhao, Xianglong Liu, Ming Zhou, Bryan Dai, Weifeng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2603.16790)  

**Abstract**: Recent code large language models have achieved remarkable progress on general programming tasks. Nevertheless, their performance degrades significantly in industrial scenarios that require reasoning about hardware semantics, specialized language constructs, and strict resource constraints. To address these challenges, we introduce InCoder-32B (Industrial-Coder-32B), the first 32B-parameter code foundation model unifying code intelligence across chip design, GPU kernel optimization, embedded systems, compiler optimization, and 3D modeling. By adopting an efficient architecture, we train InCoder-32B from scratch with general code pre-training, curated industrial code annealing, mid-training that progressively extends context from 8K to 128K tokens with synthetic industrial reasoning data, and post-training with execution-grounded verification. We conduct extensive evaluation on 14 mainstream general code benchmarks and 9 industrial benchmarks spanning 4 specialized domains. Results show InCoder-32B achieves highly competitive performance on general tasks while establishing strong open-source baselines across industrial domains. 

---
# MobileLLM-Flash: Latency-Guided On-Device LLM Design for Industry Scale 

**Authors**: Hanxian Huang, Igor Fedorov, Andrey Gromov, Bernard Beckerman, Naveen Suda, David Eriksson, Maximilian Balandat, Rylan Conway, Patrick Huber, Chinnadhurai Sankar, Ayushi Dalmia, Zechun Liu, Lemeng Wu, Tarek Elgamal, Adithya Sagar, Vikas Chandra, Raghuraman Krishnamoorthi  

**Link**: [PDF](https://arxiv.org/pdf/2603.15954)  

**Abstract**: Real-time AI experiences call for on-device large language models (OD-LLMs) optimized for efficient deployment on resource-constrained hardware. The most useful OD-LLMs produce near-real-time responses and exhibit broad hardware compatibility, maximizing user reach. We present a methodology for designing such models using hardware-in-the-loop architecture search under mobile latency constraints. This system is amenable to industry-scale deployment: it generates models deployable without custom kernels and compatible with standard mobile runtimes like Executorch. Our methodology avoids specialized attention mechanisms and instead uses attention skipping for long-context acceleration.
Our approach jointly optimizes model architecture (layers, dimensions) and attention pattern. To efficiently evaluate candidates, we treat each as a pruned version of a pretrained backbone with inherited weights, thereby achieving high accuracy with minimal continued pretraining. We leverage the low cost of latency evaluation in a staged process: learning an accurate latency model first, then searching for the Pareto-frontier across latency and quality.
This yields MobileLLM-Flash, a family of foundation models (350M, 650M, 1.4B) for efficient on-device use with strong capabilities, supporting up to 8k context length. MobileLLM-Flash delivers up to 1.8x and 1.6x faster prefill and decode on mobile CPUs with comparable or superior quality. Our analysis of Pareto-frontier design choices offers actionable principles for OD-LLM design. 

---
