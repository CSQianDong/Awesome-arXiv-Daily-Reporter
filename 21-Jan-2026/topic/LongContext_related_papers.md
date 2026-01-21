# Failure Modes in Multi-Hop QA: The Weakest Link Law and the Recognition Bottleneck 

**Authors**: Meiru Zhang, Zaiqiao Meng, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2601.12499)  

**Abstract**: Despite scaling to massive context windows, Large Language Models (LLMs) struggle with multi-hop reasoning due to inherent position bias, which causes them to overlook information at certain positions. Whether these failures stem from an inability to locate evidence (recognition failure) or integrate it (synthesis failure) is unclear. We introduce Multi-Focus Attention Instruction (MFAI), a semantic probe to disentangle these mechanisms by explicitly steering attention towards selected positions. Across 5 LLMs on two multi-hop QA tasks (MuSiQue and NeoQA), we establish the "Weakest Link Law": multi-hop reasoning performance collapses to the performance level of the least visible evidence. Crucially, this failure is governed by absolute position rather than the linear distance between facts (performance variance $<3%$). We further identify a duality in attention steering: while matched MFAI resolves recognition bottlenecks, improving accuracy by up to 11.5% in low-visibility positions, misleading MFAI triggers confusion in real-world tasks but is successfully filtered in synthetic tasks. Finally, we demonstrate that "thinking" models that utilize System-2 reasoning, effectively locate and integrate the required information, matching gold-only baselines even in noisy, long-context settings. 

---
# ARC: Active and Reflection-driven Context Management for Long-Horizon Information Seeking Agents 

**Authors**: Yilun Yao, Shan Huang, Elsie Dai, Zhewen Tan, Zhenyu Duan, Shousheng Jia, Yanbing Jiang, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.12030)  

**Abstract**: Large language models are increasingly deployed as research agents for deep search and long-horizon information seeking, yet their performance often degrades as interaction histories grow. This degradation, known as context rot, reflects a failure to maintain coherent and task-relevant internal states over extended reasoning horizons. Existing approaches primarily manage context through raw accumulation or passive summarization, treating it as a static artifact and allowing early errors or misplaced emphasis to persist. Motivated by this perspective, we propose ARC, which is the first framework to systematically formulate context management as an active, reflection-driven process that treats context as a dynamic internal reasoning state during execution. ARC operationalizes this view through reflection-driven monitoring and revision, allowing agents to actively reorganize their working context when misalignment or degradation is detected. Experiments on challenging long-horizon information-seeking benchmarks show that ARC consistently outperforms passive context compression methods, achieving up to an 11% absolute improvement in accuracy on BrowseComp-ZH with Qwen2.5-32B-Instruct. 

---
# Docs2Synth: A Synthetic Data Trained Retriever Framework for Scanned Visually Rich Documents Understanding 

**Authors**: Yihao Ding, Qiang Sun, Puzhen Wu, Sirui Li, Siwen Luo, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.12260)  

**Abstract**: Document understanding (VRDU) in regulated domains is particularly challenging, since scanned documents often contain sensitive, evolving, and domain specific knowledge. This leads to two major challenges: the lack of manual annotations for model adaptation and the difficulty for pretrained models to stay up-to-date with domain-specific facts. While Multimodal Large Language Models (MLLMs) show strong zero-shot abilities, they still suffer from hallucination and limited domain grounding. In contrast, discriminative Vision-Language Pre-trained Models (VLPMs) provide reliable grounding but require costly annotations to cover new domains. We introduce Docs2Synth, a synthetic-supervision framework that enables retrieval-guided inference for private and low-resource domains. Docs2Synth automatically processes raw document collections, generates and verifies diverse QA pairs via an agent-based system, and trains a lightweight visual retriever to extract domain-relevant evidence. During inference, the retriever collaborates with an MLLM through an iterative retrieval--generation loop, reducing hallucination and improving response consistency. We further deliver Docs2Synth as an easy-to-use Python package, enabling plug-and-play deployment across diverse real-world scenarios. Experiments on multiple VRDU benchmarks show that Docs2Synth substantially enhances grounding and domain generalization without requiring human annotations. 

---
# Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search 

**Authors**: Xinlei Yin, Xiulian Peng, Xiao Li, Zhiwei Xiong, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.13719)  

**Abstract**: Long video understanding presents significant challenges for vision-language models due to extremely long context windows. Existing solutions relying on naive chunking strategies with retrieval-augmented generation, typically suffer from information fragmentation and a loss of global coherence. We present HAVEN, a unified framework for long-video understanding that enables coherent and comprehensive reasoning by integrating audiovisual entity cohesion and hierarchical video indexing with agentic search. First, we preserve semantic consistency by integrating entity-level representations across visual and auditory streams, while organizing content into a structured hierarchy spanning global summary, scene, segment, and entity levels. Then we employ an agentic search mechanism to enable dynamic retrieval and reasoning across these layers, facilitating coherent narrative reconstruction and fine-grained entity tracking. Extensive experiments demonstrate that our method achieves good temporal coherence, entity consistency, and retrieval efficiency, establishing a new state-of-the-art with an overall accuracy of 84.1% on LVBench. Notably, it achieves outstanding performance in the challenging reasoning category, reaching 80.1%. These results highlight the effectiveness of structured, multimodal reasoning for comprehensive and context-consistent understanding of long-form videos. 

---
# Towards robust long-context understanding of large language model via active recap learning 

**Authors**: Chenyu Hui  

**Link**: [PDF](https://arxiv.org/pdf/2601.13734)  

**Abstract**: In this paper, we propose active recap learning (ARL), a framework for enhancing large language model (LLM) in understanding long contexts. ARL enables models to revisit and summarize earlier content through targeted sequence construction during contined pretraining and retrospective summarization at inference. First, we identify key tokens in prepared long context based on loss gaps between long and short forward contexts and find most revant preceding paragraphs, then summarize them using an LLM. Second, ARL equips models with the ability to autonomously generate and utilize these retrospective summaries during inference, thereby establishing a recursive memory mechanism across paragraphs. Experimental results show substantial gains, with ARL achieving a 26.8% improvement on RULER and a 9.44% improvement on LongBench. Overall, ARL offers a simple yet effective continued pretraining-based approach to strengthen long-context understanding, advancing scalable memory augmentation in LLM 

---
# HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference 

**Authors**: Zhiyuan Shi, Qibo Qiu, Feng Xue, Zhonglin Jiang, Li Yu, Jian Jiang, Xiaofei He, Wenxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13684)  

**Abstract**: The linear memory growth of the KV cache poses a significant bottleneck for LLM inference in long-context tasks. Existing static compression methods often fail to preserve globally important information, principally because they overlook the attention drift phenomenon where token significance evolves dynamically. Although recent dynamic retrieval approaches attempt to address this issue, they typically suffer from coarse-grained caching strategies and incur high I/O overhead due to frequent data transfers. To overcome these limitations, we propose HeteroCache, a training-free dynamic compression framework. Our method is built on two key insights: attention heads exhibit diverse temporal heterogeneity, and there is significant spatial redundancy among heads within the same layer. Guided by these insights, HeteroCache categorizes heads based on stability and redundancy. Consequently, we apply a fine-grained weighting strategy that allocates larger cache budgets to heads with rapidly shifting attention to capture context changes, thereby addressing the inefficiency of coarse-grained strategies. Furthermore, we employ a hierarchical storage mechanism in which a subset of representative heads monitors attention shift, and trigger an asynchronous, on-demand retrieval of contexts from the CPU, effectively hiding I/O latency. Finally, experiments demonstrate that HeteroCache achieves state-of-the-art performance on multiple long-context benchmarks and accelerates decoding by up to $3\times$ compared to the original model in the 224K context. Our code will be open-source. 

---
# A Unified Variational Imputation Framework for Electric Vehicle Charging Data Using Retrieval-Augmented Language Model 

**Authors**: Jinhao Li, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13476)  

**Abstract**: The reliability of data-driven applications in electric vehicle (EV) infrastructure, such as charging demand forecasting, hinges on the availability of complete, high-quality charging data. However, real-world EV datasets are often plagued by missing records, and existing imputation methods are ill-equipped for the complex, multimodal context of charging data, often relying on a restrictive one-model-per-station paradigm that ignores valuable inter-station correlations. To address these gaps, we develop a novel PRobabilistic variational imputation framework that leverages the power of large lAnguage models and retrIeval-augmented Memory (PRAIM). PRAIM employs a pre-trained language model to encode heterogeneous data, spanning time-series demand, calendar features, and geospatial context, into a unified, semantically rich representation. This is dynamically fortified by retrieval-augmented memory that retrieves relevant examples from the entire charging network, enabling a single, unified imputation model empowered by variational neural architecture to overcome data sparsity. Extensive experiments on four public datasets demonstrate that PRAIM significantly outperforms established baselines in both imputation accuracy and its ability to preserve the original data's statistical distribution, leading to substantial improvements in downstream forecasting performance. 

---
# Incentivizing In-depth Reasoning over Long Contexts with Process Advantage Shaping 

**Authors**: Miao Peng, Weizhou Shen, Nuo Chen, Chenliang Li, Ming Yan, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.12465)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective in enhancing LLMs short-context reasoning, but its performance degrades in long-context scenarios that require both precise grounding and robust long-range reasoning. We identify the "almost-there" phenomenon in long-context reasoning, where trajectories are largely correct but fail at the final step, and attribute this failure to two factors: (1) the lack of high reasoning density in long-context QA data that push LLMs beyond mere grounding toward sophisticated multi-hop reasoning; and (2) the loss of valuable learning signals during long-context RL training due to the indiscriminate penalization of partially correct trajectories with incorrect outcomes. To overcome this bottleneck, we propose DeepReasonQA, a KG-driven synthesis framework that controllably constructs high-difficulty, multi-hop long-context QA pairs with inherent reasoning chains. Building on this, we introduce Long-context Process Advantage Shaping (LongPAS), a simple yet effective method that performs fine-grained credit assignment by evaluating reasoning steps along Validity and Relevance dimensions, which captures critical learning signals from "almost-there" trajectories. Experiments on three long-context reasoning benchmarks show that our approach substantially outperforms RLVR baselines and matches frontier LLMs while using far fewer parameters. Further analysis confirms the effectiveness of our methods in strengthening long-context reasoning while maintaining stable RL training. 

---
# $\texttt{MemoryRewardBench}$: Benchmarking Reward Models for Long-Term Memory Management in Large Language Models 

**Authors**: Zecheng Tang, Baibei Ji, Ruoxi Sun, Haitian Wang, WangJie You, Zhang Yijun, Wenpeng Zhu, Ji Qi, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11969)  

**Abstract**: Existing works increasingly adopt memory-centric mechanisms to process long contexts in a segment manner, and effective memory management is one of the key capabilities that enables large language models to effectively propagate information across the entire sequence. Therefore, leveraging reward models (RMs) to automatically and reliably evaluate memory quality is critical. In this work, we introduce $\texttt{MemoryRewardBench}$, the first benchmark to systematically study the ability of RMs to evaluate long-term memory management processes. $\texttt{MemoryRewardBench}$ covers both long-context comprehension and long-form generation tasks, featuring 10 distinct settings with different memory management patterns, with context length ranging from 8K to 128K tokens. Evaluations on 13 cutting-edge RMs indicate a diminishing performance gap between open-source and proprietary models, with newer-generation models consistently outperforming their predecessors regardless of parameter count. We further expose the capabilities and fundamental limitations of current RMs in evaluating LLM memory management across diverse settings. 

---
# LSTM-MAS: A Long Short-Term Memory Inspired Multi-Agent System for Long-Context Understanding 

**Authors**: Yichen Jiang, Peng Ye, Jiakang Yuan, Chongjun Tu, Lei Bai, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.11913)  

**Abstract**: Effectively processing long contexts remains a fundamental yet unsolved challenge for large language models (LLMs). Existing single-LLM-based methods primarily reduce the context window or optimize the attention mechanism, but they often encounter additional computational costs or constrained expanded context length. While multi-agent-based frameworks can mitigate these limitations, they remain susceptible to the accumulation of errors and the propagation of hallucinations. In this work, we draw inspiration from the Long Short-Term Memory (LSTM) architecture to design a Multi-Agent System called LSTM-MAS, emulating LSTM's hierarchical information flow and gated memory mechanisms for long-context understanding. Specifically, LSTM-MAS organizes agents in a chained architecture, where each node comprises a worker agent for segment-level comprehension, a filter agent for redundancy reduction, a judge agent for continuous error detection, and a manager agent for globally regulates information propagation and retention, analogous to LSTM and its input gate, forget gate, constant error carousel unit, and output gate. These novel designs enable controlled information transfer and selective long-term dependency modeling across textual segments, which can effectively avoid error accumulation and hallucination propagation. We conducted an extensive evaluation of our method. Compared with the previous best multi-agent approach, CoA, our model achieves improvements of 40.93%, 43.70%,121.57% and 33.12%, on NarrativeQA, Qasper, HotpotQA, and MuSiQue, respectively. 

---
# Geometric Attention: A Regime-Explicit Operator Semantics for Transformer Attention 

**Authors**: Luis Rosario Freytes  

**Link**: [PDF](https://arxiv.org/pdf/2601.11618)  

**Abstract**: Geometric Attention (GA) specifies an attention layer by four independent inputs: a finite carrier (what indices are addressable), an evidence-kernel rule (how masked proto-scores and a link induce nonnegative weights), a probe family (which observables are treated as admissible), and an anchor/update rule (which representative kernel is selected and how it is applied). Probe families induce an operational equivalence relation on kernels and therefore a gauge; anchors select representatives relative to that probe. Under a scalar relational-work representation and a multiplicative compositionality law for evidence, the admissible link family is exponential, yielding Gibbs weights; with row anchoring this includes the softmax kernel family as a subregime. After quotienting unary row/column score fields, the remaining interaction component admits a canonical rank-r normal form (Eckart-Young/SVD); dot-product score charts implement the corresponding low-rank interaction regime. Fixing the carrier and extensionalizing the update yields the standard fixed-token Transformer attention operator; allowing carrier updates yields adaptive-carrier and staged-depth regimes. The operator language also supports multihead/mixed kernels, plan-based anchors (e.g., entropic OT/Sinkhorn), and unary operators (e.g., FFN-style fields) as explicit regime choices. This separates invariant structure from modeling choice, enabling principled comparison and extension of attention mechanisms, and attention-based architectures. 

---
# Context Discipline and Performance Correlation: Analyzing LLM Performance and Quality Degradation Under Varying Context Lengths 

**Authors**: Ahilan Ayyachamy Nadar Ponnusamy, Karthic Chandran, M Maruf Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2601.11564)  

**Abstract**: The scaling trend in Large Language Models (LLMs) has prioritized increasing the maximum context window to facilitate complex, long-form reasoning and document analysis. However, managing this expanded context introduces severe computational overhead. This paper investigates the critical trade-off between system performance and model quality when dense transformer architectures--specifically Llama-3.1-70B and Qwen1.5-14B--are exposed to large volumes of irrelevant and distracting context. The research identifies a non-linear performance degradation tied to the growth of the Key-Value (KV) cache. Furthermore, an extended analysis of the Mixture-of-Experts (MoE) architecture reveals unique behavioral anomalies at varying context scales, suggesting that architectural benefits may be masked by infrastructure bottlenecks at high token volumes. 

---
# AgentEHR: Advancing Autonomous Clinical Decision-Making via Retrospective Summarization 

**Authors**: Yusheng Liao, Chuan Xuan, Yutong Cai, Lina Yang, Zhe Chen, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13918)  

**Abstract**: Large Language Models have demonstrated profound utility in the medical domain. However, their application to autonomous Electronic Health Records~(EHRs) navigation remains constrained by a reliance on curated inputs and simplified retrieval tasks. To bridge the gap between idealized experimental settings and realistic clinical environments, we present AgentEHR. This benchmark challenges agents to execute complex decision-making tasks, such as diagnosis and treatment planning, requiring long-range interactive reasoning directly within raw and high-noise databases. In tackling these tasks, we identify that existing summarization methods inevitably suffer from critical information loss and fractured reasoning continuity. To address this, we propose RetroSum, a novel framework that unifies a retrospective summarization mechanism with an evolving experience strategy. By dynamically re-evaluating interaction history, the retrospective mechanism prevents long-context information loss and ensures unbroken logical coherence. Additionally, the evolving strategy bridges the domain gap by retrieving accumulated experience from a memory bank. Extensive empirical evaluations demonstrate that RetroSum achieves performance gains of up to 29.16% over competitive baselines, while significantly decreasing total interaction errors by up to 92.3%. 

---
# Probe and Skip: Self-Predictive Token Skipping for Efficient Long-Context LLM Inference 

**Authors**: Zimeng Wu, Donghao Wang, Chaozhe Jin, Jiaxin Chen, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.13155)  

**Abstract**: Long-context inference enhances the reasoning capability of Large Language Models (LLMs) while incurring significant computational overhead. Token-oriented methods, such as pruning and skipping, have shown promise in reducing inference latency, but still suffer from inherently limited acceleration potential, outdated proxy signals, and redundancy interference, thus yielding suboptimal speed-accuracy trade-offs. To address these challenges, we propose SPTS (Self-Predictive Token Skipping), a training-free framework for efficient long-context LLM inference. Specifically, motivated by the thought of probing the influence of targeted skipping layers, we design two component-specific strategies for selective token skipping: Partial Attention Probing (PAP) for multi-head attention, which selects informative tokens by performing partial forward attention computation, and Low-rank Transformation Probing (LTP) for feed forward network, which constructs a low-rank proxy network to predict token transformations. Furthermore, a Multi-Stage Delayed Pruning (MSDP) strategy reallocates the skipping budget and progressively prunes redundant tokens across layers. Extensive experiments demonstrate the effectiveness of our method, achieving up to 2.46$\times$ and 2.29$\times$ speedups for prefilling and end-to-end generation, respectively, while maintaining state-of-the-art model performance. The source code will be publicly available upon paper acceptance. 

---
# Gated Differentiable Working Memory for Long-Context Language Modeling 

**Authors**: Lingrui Mei, Shenghua Liu, Yiwei Wang, Yuyao Ge, Baolong Bi, Jiayu Yao, Jun Wan, Ziling Yin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.12906)  

**Abstract**: Long contexts challenge transformers: attention scores dilute across thousands of tokens, critical information is often lost in the middle, and models struggle to adapt to novel patterns at inference time. Recent work on test-time adaptation addresses this by maintaining a form of working memory -- transient parameters updated on the current context -- but existing approaches rely on uniform write policies that waste computation on low-utility regions and suffer from high gradient variance across semantically heterogeneous contexts. In this work, we reframe test-time adaptation as a budget-constrained memory consolidation problem, focusing on which parts of the context should be consolidated into working memory under limited computation. We propose Gdwm (Gated Differentiable Working Memory), a framework that introduces a write controller to gate the consolidation process. The controller estimates Contextual Utility, an information-theoretic measure of long-range contextual dependence, and allocates gradient steps accordingly while maintaining global coverage. Experiments on ZeroSCROLLS and LongBench v2 demonstrate that Gdwm achieves comparable or superior performance with 4$\times$ fewer gradient steps than uniform baselines, establishing a new efficiency-performance Pareto frontier for test-time adaptation. 

---
# SciCoQA: Quality Assurance for Scientific Paper--Code Alignment 

**Authors**: Tim Baumgärtner, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2601.12910)  

**Abstract**: We present SciCoQA, a dataset for detecting discrepancies between scientific publications and their codebases to ensure faithful implementations. We construct SciCoQA from GitHub issues and reproducibility papers, and to scale our dataset, we propose a synthetic data generation method for constructing paper-code discrepancies. We analyze the paper-code discrepancies in detail and propose discrepancy types and categories to better understand the occurring mismatches. In total, our dataset consists of 611 paper-code discrepancies (81 real, 530 synthetic), spanning diverse computational science disciplines, including AI, Physics, Quantitative Biology, and others. Our evaluation of 21 LLMs highlights the difficulty of SciCoQA, particularly for instances involving omitted paper details, long-context inputs, and data outside the models' pre-training corpus. The best performing model in our evaluation, GPT-5, can only detect 45.7\% of real-world paper-code discrepancies. 

---
# PPA-Plan: Proactive Pitfall Avoidance for Reliable Planning in Long-Context LLM Reasoning 

**Authors**: Byeongjin Kim, Gyuwan Kim, Seo Yeon Park  

**Link**: [PDF](https://arxiv.org/pdf/2601.11908)  

**Abstract**: Large language models (LLMs) struggle with reasoning over long contexts where relevant information is sparsely distributed. Although plan-and-execute frameworks mitigate this by decomposing tasks into planning and execution, their effectiveness is often limited by unreliable plan generation due to dependence on surface-level cues. Consequently, plans may be based on incorrect assumptions, and once a plan is formed, identifying what went wrong and revising it reliably becomes difficult, limiting the effectiveness of reactive refinement. To address this limitation, we propose PPA-Plan, a proactive planning strategy for long-context reasoning that focuses on preventing such failures before plan generation. PPA-Plan identifies potential logical pitfalls and false assumptions, formulates them as negative constraints, and conditions plan generation on explicitly avoiding these constraints. Experiments on long-context QA benchmarks show that executing plans generated by PPA-Plan consistently outperforms existing plan-and-execute methods and direct prompting. 

---
