# VehicleMemBench: An Executable Benchmark for Multi-User Long-Term Memory in In-Vehicle Agents 

**Authors**: Yuhao Chen, Yi Xu, Xinyun Ding, Xiang Fang, Shuochen Liu, Luxi Lin, Qingyu Zhang, Ya Li, Quan Liu, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.23840)  

**Abstract**: With the growing demand for intelligent in-vehicle experiences, vehicle-based agents are evolving from simple assistants to long-term companions. This evolution requires agents to continuously model multi-user preferences and make reliable decisions in the face of inter-user preference conflicts and changing habits over time. However, existing benchmarks are largely limited to single-user, static question-answer settings, failing to capture the temporal evolution of preferences and the multi-user, tool-interactive nature of real vehicle environments. To address this gap, we introduce VehicleMemBench, a multi-user long-context memory benchmark built on an executable in-vehicle simulation environment. The benchmark evaluates tool use and memory by comparing the post-action environment state with a predefined target state, enabling objective and reproducible evaluation without LLM-based or human scoring. VehicleMemBench includes 23 tool modules, and each sample contains over 80 historical memory events. Experiments show that powerful models perform well on direct instruction tasks but struggle in scenarios involving memory evolution, particularly when user preferences change dynamically. Even advanced memory systems struggle to handle domain-specific memory requirements in this environment. These findings highlight the need for more robust and specialized memory management mechanisms to support long-term adaptive decision-making in real-world in-vehicle systems. To facilitate future research, we release the data and code. 

---
# CliPPER: Contextual Video-Language Pretraining on Long-form Intraoperative Surgical Procedures for Event Recognition 

**Authors**: Florian Stilz, Vinkle Srivastav, Nassir Navab, Nicolas Padoy  

**Link**: [PDF](https://arxiv.org/pdf/2603.24539)  

**Abstract**: Video-language foundation models have proven to be highly effective in zero-shot applications across a wide range of tasks. A particularly challenging area is the intraoperative surgical procedure domain, where labeled data is scarce, and precise temporal understanding is often required for complex downstream tasks. To address this challenge, we introduce CliPPER (Contextual Video-Language Pretraining on Long-form Intraoperative Surgical Procedures for Event Recognition), a novel video-language pretraining framework trained on surgical lecture videos. Our method is designed for fine-grained temporal video-text recognition and introduces several novel pretraining strategies to improve multimodal alignment in long-form surgical videos. Specifically, we propose Contextual Video-Text Contrastive Learning (VTC_CTX) and Clip Order Prediction (COP) pretraining objectives, both of which leverage temporal and contextual dependencies to enhance local video understanding. In addition, we incorporate a Cycle-Consistency Alignment over video-text matches within the same surgical video to enforce bidirectional consistency and improve overall representation coherence. Moreover, we introduce a more refined alignment loss, Frame-Text Matching (FTM), to improve the alignment between video frames and text. As a result, our model establishes a new state-of-the-art across multiple public surgical benchmarks, including zero-shot recognition of phases, steps, instruments, and triplets. The source code and pretraining captions can be found at this https URL. 

---
# StateLinFormer: Stateful Training Enhancing Long-term Memory in Navigation 

**Authors**: Zhiyuan Chen, Yuxuan Zhong, Fan Wang, Bo Yu, Pengtao Shao, Shaoshan Liu, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2603.23571)  

**Abstract**: Effective navigation intelligence relies on long-term memory to support both immediate generalization and sustained adaptation. However, existing approaches face a dilemma: modular systems rely on explicit mapping but lack flexibility, while Transformer-based end-to-end models are constrained by fixed context windows, limiting persistent memory across extended interactions. We introduce StateLinFormer, a linear-attention navigation model trained with a stateful memory mechanism that preserves recurrent memory states across consecutive training segments instead of reinitializing them at each batch boundary. This training paradigm effectively approximates learning on infinitely long sequences, enabling the model to achieve long-horizon memory retention. Experiments across both MAZE and ProcTHOR environments demonstrate that StateLinFormer significantly outperforms its stateless linear-attention counterpart and standard Transformer baselines with fixed context windows. Notably, as interaction length increases, persistent stateful training substantially improves context-dependent adaptation, suggesting an enhancement in the model's In-Context Learning (ICL) capabilities for navigation tasks. 

---
# MedMT-Bench: Can LLMs Memorize and Understand Long Multi-Turn Conversations in Medical Scenarios? 

**Authors**: Lin Yang, Yuancheng Yang, Xu Wang, Changkun Liu, Haihua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.23519)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities across various specialist domains and have been integrated into high-stakes areas such as medicine. However, as existing medical-related benchmarks rarely stress-test the long-context memory, interference robustness, and safety defense required in practice. To bridge this gap, we introduce MedMT-Bench, a challenging medical multi-turn instruction following benchmark that simulates the entire diagnosis and treatment process. We construct the benchmark via scene-by-scene data synthesis refined by manual expert editing, yielding 400 test cases that are highly consistent with real-world application scenarios. Each test case has an average of 22 rounds (maximum of 52 rounds), covering 5 types of difficult instruction following issues. For evaluation, we propose an LLM-as-judge protocol with instance-level rubrics and atomic test points, validated against expert annotations with a human-LLM agreement of 91.94\%. We test 17 frontier models, all of which underperform on MedMT-Bench (overall accuracy below 60.00\%), with the best model reaching 59.75\%. MedMT-Bench can be an essential tool for driving future research towards safer and more reliable medical AI. The benchmark is available in this https URL 

---
# MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens 

**Authors**: Yu Chen, Runkai Chen, Sheng Yi, Xinda Zhao, Xiaohong Li, Jianjin Zhang, Jun Sun, Chuanrui Hu, Yunyun Han, Lidong Bing, Yafeng Deng, Tianqiao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.23516)  

**Abstract**: Long-term memory is a cornerstone of human intelligence. Enabling AI to process lifetime-scale information remains a long-standing pursuit in
the field. Due to the constraints of full-attention architectures, the effective context length of large language models (LLMs) is typically
limited to 1M tokens. Existing approaches, such as hybrid linear attention, fixed-size memory states (e.g., RNNs), and external storage
methods like RAG or agent systems, attempt to extend this limit. However, they often suffer from severe precision degradation and rapidly
increasing latency as context length grows, an inability to dynamically modify memory content, or a lack of end-to-end optimization. These
bottlenecks impede complex scenarios like large-corpus summarization, Digital Twins, and long-history agent reasoning, while limiting memory
capacity and slowing inference. We present Memory Sparse Attention (MSA), an end-to-end trainable, efficient, and massively scalable memory
model framework. Through core innovations including scalable sparse attention and document-wise RoPE, MSA achieves linear complexity in both
training and inference while maintaining exceptional stability, exhibiting less than 9% degradation when scaling from 16K to 100M tokens.
Furthermore, KV cache compression, combined with Memory Parallel, enables 100M-token inference on 2xA800 GPUs. We also propose Memory
Interleaving to facilitate complex multi-hop reasoning across scattered memory segments. MSA significantly surpasses frontier LLMs,
state-of-the-art RAG systems, and leading memory agents in long-context benchmarks. These results demonstrate that by decoupling memory
capacity from reasoning, MSA provides a scalable foundation to endow general-purpose models with intrinsic, lifetime-scale memory. 

---
# Fast and Faithful: Real-Time Verification for Long-Document Retrieval-Augmented Generation Systems 

**Authors**: Xunzhuo Liu, Bowei He, Xue Liu, Haichen Zhang, Huamin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.23508)  

**Abstract**: Retrieval-augmented generation (RAG) is increasingly deployed in enterprise search and document-centric assistants, where responses must be grounded in long and complex source materials. In practice, verifying that generated answers faithfully reflect retrieved documents is difficult: large language models can check long contexts but are too slow and costly for interactive services, while lightweight classifiers operate within strict context limits and frequently miss evidence outside truncated passages. We present the design of a real-time verification component integrated into a production RAG pipeline that enables full-document grounding under latency constraints. The system processes documents up to 32K tokens and employs adaptive inference strategies to balance response time and verification coverage across workloads. We describe the architectural decisions, operational trade-offs, and evaluation methodology used to deploy the verifier, and show that full-context verification substantially improves detection of unsupported responses compared with truncated validation. Our experience highlights when long-context verification is necessary, why chunk-based checking often fails in real documents, and how latency budgets shape model design. These findings provide practical guidance for practitioners building reliable large-scale retrieval-augmented applications. (Model, benchmark, and code: this https URL) 

---
