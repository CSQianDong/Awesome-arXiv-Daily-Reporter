# Memory as Resonance: A Biomimetic Architecture for Infinite Context Memory on Ergodic Phonetic Manifolds 

**Authors**: Tarik Houichime, Abdelghani Souhar, Younes El Amrani  

**Link**: [PDF](https://arxiv.org/pdf/2512.20245)  

**Abstract**: The memory of contemporary Large Language Models is bound by a physical paradox: as they learn, they fill up. The linear accumulation (O(N)) of Key-Value states treats context as a warehouse of static artifacts, eventually forcing a destructive choice between amnesia and latency. We challenge this discrete orthodoxy, proposing that long-term memory is not the storage of items, but the persistence of a trajectory. We introduce Phonetic Trajectory Memory (PTM), a neuro-symbolic architecture that encodes language not as a sequence of tensors, but as a continuous path on an ergodic manifold governed by irrational rotation matrices. By decoupling the navigation (an invariant O(1) geometric signal) from the reconstruction (a probabilistic generative act), PTM achieves a compression magnitude of greater than 3,000x relative to dense caches. We demonstrate that retrieval becomes a process of resonance: the phonetic trace stabilizes the model against hallucination via "Signal Consensus" mechanism, securing up to approximately 92% factual accuracy. While this aggressive abstraction alters generative texture, it unlocks immediate access latency (approximately 34ms) independent of depth. Our results suggest that infinite context does not require infinite silicon; it requires treating memory not as data to be stored, but as a reconstructive process acting on a conserved, undying physical signal. 

---
# MoE-DiffuSeq: Enhancing Long-Document Diffusion Models with Sparse Attention and Mixture of Experts 

**Authors**: Alexandros Christoforos, Chadbourne Davis  

**Link**: [PDF](https://arxiv.org/pdf/2512.20604)  

**Abstract**: We present MoE-DiffuSeq, a mixture of experts based framework for enhancing diffusion models in long document generation. Existing diffusion based text generation models, such as DiffuSeq, suffer from high computational cost and memory overhead when applied to extended sequences. To address these challenges, MoE-DiffuSeq integrates sparse attention with a mixture of experts architecture, enabling efficient and scalable long sequence modeling. Our approach introduces a customized sparse attention mechanism designed to reduce computational complexity while preserving text quality and coherence. In addition, we incorporate a soft absorbing state within the diffusion process to accelerate sequence reconstruction and improve generation precision. Extensive experiments demonstrate that MoE-DiffuSeq significantly improves training efficiency and sampling speed compared to existing diffusion models. These advantages are particularly effective for long document scenarios, including scientific article generation, code repository modeling, and long form dialogue generation. Benchmark results further show that MoE-DiffuSeq improves efficiency, speed, accuracy, and expressiveness, advancing the practical applicability of diffusion models for high quality long form text generation. 

---
# Memory-T1: Reinforcement Learning for Temporal Reasoning in Multi-session Agents 

**Authors**: Yiming Du, Baojun Wang, Yifan Xiang, Zhaowei Wang, Wenyu Huang, Boyang Xue, Bin Liang, Xingshan Zeng, Fei Mi, Haoli Bai, Lifeng Shang, Jeff Z. Pan, Yuxin Jiang, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2512.20092)  

**Abstract**: Temporal reasoning over long, multi-session dialogues is a critical capability for conversational agents. However, existing works and our pilot study have shown that as dialogue histories grow in length and accumulate noise, current long-context models struggle to accurately identify temporally pertinent information, significantly impairing reasoning performance. To address this, we introduce Memory-T1, a framework that learns a time-aware memory selection policy using reinforcement learning (RL). It employs a coarse-to-fine strategy, first pruning the dialogue history into a candidate set using temporal and relevance filters, followed by an RL agent that selects the precise evidence sessions. The RL training is guided by a multi-level reward function optimizing (i) answer accuracy, (ii) evidence grounding, and (iii) temporal consistency. In particular, the temporal consistency reward provides a dense signal by evaluating alignment with the query time scope at both the session-level (chronological proximity) and the utterance-level (chronological fidelity), enabling the agent to resolve subtle chronological ambiguities. On the Time-Dialog benchmark, Memory-T1 boosts a 7B model to an overall score of 67.0\%, establishing a new state-of-the-art performance for open-source models and outperforming a 14B baseline by 10.2\%. Ablation studies show temporal consistency and evidence grounding rewards jointly contribute to a 15.0\% performance gain. Moreover, Memory-T1 maintains robustness up to 128k tokens, where baseline models collapse, proving effectiveness against noise in extensive dialogue histories. The code and datasets are publicly available at this https URL 

---
