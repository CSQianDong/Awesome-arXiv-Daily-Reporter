# A Progressive Visual-Logic-Aligned Framework for Ride-Hailing Adjudication 

**Authors**: Weiming Wu, Zi-Jian Cheng, Jie Meng, Peng Zhen, Shan Huang, Qun Li, Guobin Wu, Lan-Zhe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.17328)  

**Abstract**: The efficient adjudication of responsibility disputes is pivotal for maintaining marketplace fairness. However, the exponential surge in ride-hailing volume renders manual review intractable, while conventional automated methods lack the reasoning transparency required for quasi-judicial decisions. Although Multimodal LLMs offer a promising paradigm, they fundamentally struggle to bridge the gap between general visual semantics and rigorous evidentiary protocols, often leading to perceptual hallucinations and logical looseness. To address these systemic misalignments, we introduce RideJudge, a Progressive Visual-Logic-Aligned Framework. Instead of relying on generic pre-training, we bridge the semantic gap via SynTraj, a synthesis engine that grounds abstract liability concepts into concrete trajectory patterns. To resolve the conflict between massive regulation volume and limited context windows, we propose an Adaptive Context Optimization strategy that distills expert knowledge, coupled with a Chain-of-Adjudication mechanism to enforce active evidentiary inquiry. Furthermore, addressing the inadequacy of sparse binary feedback for complex liability assessment, we implement a novel Ordinal-Sensitive Reinforcement Learning mechanism that calibrates decision boundaries against hierarchical severity. Extensive experiments show that our RideJudge-8B achieves 88.41\% accuracy, surpassing 32B-scale baselines and establishing a new standard for interpretable adjudication. 

---
# Anchoring and Rescaling Attention for Semantically Coherent Inbetweening 

**Authors**: Tae Eun Choi, Sumin Shim, Junhyeok Kim, Seong Jae Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17651)  

**Abstract**: Generative inbetweening (GI) seeks to synthesize realistic intermediate frames between the first and last keyframes beyond mere interpolation. As sequences become sparser and motions larger, previous GI models struggle with inconsistent frames with unstable pacing and semantic misalignment. Since GI involves fixed endpoints and numerous plausible paths, this task requires additional guidance gained from the keyframes and text to specify the intended path. Thus, we give semantic and temporal guidance from the keyframes and text onto each intermediate frame through Keyframe-anchored Attention Bias. We also better enforce frame consistency with Rescaled Temporal RoPE, which allows self-attention to attend to keyframes more faithfully. TGI-Bench, the first benchmark specifically designed for text-conditioned GI evaluation, enables challenge-targeted evaluation to analyze GI models. Without additional training, our method achieves state-of-the-art frame consistency, semantic fidelity, and pace stability for both short and long sequences across diverse challenges. 

---
# Symphony: A Cognitively-Inspired Multi-Agent System for Long-Video Understanding 

**Authors**: Haiyang Yan, Hongyun Zhou, Peng Xu, Xiaoxue Feng, Mengyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.17307)  

**Abstract**: Despite rapid developments and widespread applications of MLLM agents, they still struggle with long-form video understanding (LVU) tasks, which are characterized by high information density and extended temporal spans. Recent research on LVU agents demonstrates that simple task decomposition and collaboration mechanisms are insufficient for long-chain reasoning tasks. Moreover, directly reducing the time context through embedding-based retrieval may lose key information of complex problems. In this paper, we propose Symphony, a multi-agent system, to alleviate these limitations. By emulating human cognition patterns, Symphony decomposes LVU into fine-grained subtasks and incorporates a deep reasoning collaboration mechanism enhanced by reflection, effectively improving the reasoning capability. Additionally, Symphony provides a VLM-based grounding approach to analyze LVU tasks and assess the relevance of video segments, which significantly enhances the ability to locate complex problems with implicit intentions and large temporal spans. Experimental results show that Symphony achieves state-of-the-art performance on LVBench, LongVideoBench, VideoMME, and MLVU, with a 5.0% improvement over the prior state-of-the-art method on LVBench. Code is available at this https URL. 

---
# The Phasor Transformer: Resolving Attention Bottlenecks on the Unit Circle 

**Authors**: Dibakar Sigdel  

**Link**: [PDF](https://arxiv.org/pdf/2603.17433)  

**Abstract**: Transformer models have redefined sequence learning, yet dot-product self-attention introduces a quadratic token-mixing bottleneck for long-context time-series. We introduce the \textbf{Phasor Transformer} block, a phase-native alternative representing sequence states on the unit-circle manifold $S^1$. Each block combines lightweight trainable phase-shifts with parameter-free Discrete Fourier Transform (DFT) token coupling, achieving global $\mathcal{O}(N\log N)$ mixing without explicit attention maps. Stacking these blocks defines the \textbf{Large Phasor Model (LPM)}. We validate LPM on autoregressive time-series prediction over synthetic multi-frequency benchmarks. Operating with a highly compact parameter budget, LPM learns stable global dynamics and achieves competitive forecasting behavior compared to conventional self-attention baselines. Our results establish an explicit efficiency-performance frontier, demonstrating that large-model scaling for time-series can emerge from geometry-constrained phase computation with deterministic global coupling, offering a practical path toward scalable temporal modeling in oscillatory domains. 

---
# Learning When to Attend: Conditional Memory Access for Long-Context LLMs 

**Authors**: Sakshi Choudhary, Aditya Chattopadhyay, Luca Zancato, Elvis Nunez, Matthew Trager, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2603.17484)  

**Abstract**: Language models struggle to generalize beyond pretraining context lengths, limiting long-horizon reasoning and retrieval. Continued pretraining on long-context data can help but is expensive due to the quadratic scaling of Attention. We observe that most tokens do not require (Global) Attention over the entire sequence and can rely on local context. Based on this, we propose L2A (Learning To Attend), a layer that enables conditional (token-wise) long-range memory access by deciding when to invoke global attention. We evaluate L2A on Qwen 2.5 and Qwen 3 models, extending their effective context length from 32K to 128K tokens. L2A matches the performance of standard long-context training to within 3% while skipping Global Attention for $\sim$80% of tokens, outperforming prior baselines. We also design custom Triton kernels to efficiently implement this token-wise conditional Attention on GPUs, achieving up to $\sim$2x improvements in training throughput and time-to-first-token over FlashAttention. Moreover, L2A enables post-training pruning of highly sparse Global Attention layers, reducing KV cache memory by up to 50% with negligible performance loss. 

---
