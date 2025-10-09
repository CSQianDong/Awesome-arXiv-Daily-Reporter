# Artificial Hippocampus Networks for Efficient Long-Context Modeling 

**Authors**: Yunhao Fang, Weihao Yu, Shu Zhong, Qinghao Ye, Xuehan Xiong, Lai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.07318)  

**Abstract**: Long-sequence modeling faces a fundamental trade-off between the efficiency of compressive fixed-size memory in RNN-like models and the fidelity of lossless growing memory in attention-based Transformers. Inspired by the Multi-Store Model in cognitive science, we introduce a memory framework of artificial neural networks. Our method maintains a sliding window of the Transformer's KV cache as lossless short-term memory, while a learnable module termed Artificial Hippocampus Network (AHN) recurrently compresses out-of-window information into a fixed-size compact long-term memory. To validate this framework, we instantiate AHNs using modern RNN-like architectures, including Mamba2, DeltaNet, and Gated DeltaNet. Extensive experiments on long-context benchmarks LV-Eval and InfiniteBench demonstrate that AHN-augmented models consistently outperform sliding window baselines and achieve performance comparable or even superior to full-attention models, while substantially reducing computational and memory requirements. For instance, augmenting the Qwen2.5-3B-Instruct with AHNs reduces inference FLOPs by 40.5% and memory cache by 74.0%, while improving its average score on LV-Eval (128k sequence length) from 4.41 to 5.88. Code is available at: this https URL. 

---
# Native Hybrid Attention for Efficient Sequence Modeling 

**Authors**: Jusen Du, Jiaxi Hu, Tao Zhang, Weigao Sun, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.07019)  

**Abstract**: Transformers excel at sequence modeling but face quadratic complexity, while linear attention offers improved efficiency but often compromises recall accuracy over long contexts. In this work, we introduce Native Hybrid Attention (NHA), a novel hybrid architecture of linear and full attention that integrates both intra \& inter-layer hybridization into a unified layer design. NHA maintains long-term context in key-value slots updated by a linear RNN, and augments them with short-term tokens from a sliding window. A single \texttt{softmax attention} operation is then applied over all keys and values, enabling per-token and per-head context-dependent weighting without requiring additional fusion parameters. The inter-layer behavior is controlled through a single hyperparameter, the sliding window size, which allows smooth adjustment between purely linear and full attention while keeping all layers structurally uniform. Experimental results show that NHA surpasses Transformers and other hybrid baselines on recall-intensive and commonsense reasoning tasks. Furthermore, pretrained LLMs can be structurally hybridized with NHA, achieving competitive accuracy while delivering significant efficiency gains. Code is available at this https URL. 

---
# Mid-Training of Large Language Models: A Survey 

**Authors**: Kaixiang Mo, Yuxin Shi, Weiwei Weng, Zhiqiang Zhou, Shuman Liu, Haibo Zhang, Anxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.06826)  

**Abstract**: Large language models (LLMs) are typically developed through large-scale pre-training followed by task-specific fine-tuning. Recent advances highlight the importance of an intermediate mid-training stage, where models undergo multiple annealing-style phases that refine data quality, adapt optimization schedules, and extend context length. This stage mitigates diminishing returns from noisy tokens, stabilizes convergence, and expands model capability in late training. Its effectiveness can be explained through gradient noise scale, the information bottleneck, and curriculum learning, which together promote generalization and abstraction. Despite widespread use in state-of-the-art systems, there has been no prior survey of mid-training as a unified paradigm. We introduce the first taxonomy of LLM mid-training spanning data distribution, learning-rate scheduling, and long-context extension. We distill practical insights, compile evaluation benchmarks, and report gains to enable structured comparisons across models. We also identify open challenges and propose avenues for future research and practice. 

---
# Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management 

**Authors**: Miao Lu, Weiwei Sun, Weihua Du, Zhan Ling, Xuesong Yao, Kang Liu, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.06727)  

**Abstract**: We study reinforcement learning (RL) fine-tuning of large language model (LLM) agents for long-horizon multi-turn tool use, where context length quickly becomes a fundamental bottleneck. Existing RL pipelines can suffer from degraded instruction following, excessive rollout costs, and most importantly, strict context limits. To address these challenges, we introduce summarization-based context management to training. In specific, it periodically compresses the tool using history by LLM-generated summaries that retain task-relevant information to keep a compact context while enabling the agent to scale beyond the fixed context window. Building on this formulation, we derive a policy gradient representation that seamlessly enables standard LLM RL infrastructures to optimize both tool-use behaviors as well as summarization strategies in an end-to-end fashion. We instantiate this framework with \underline{SU}mmarization augmented \underline{P}olicy \underline{O}ptimization (\texttt{SUPO}), an LLM RL algorithm that enables long-horizon training beyond a fixed context limit. Experiments on interactive function calling and searching tasks demonstrate that \texttt{SUPO} significantly improves the success rate while maintaining the same or even lower working context length compared to baselines. We also demonstrate that for complex searching tasks, \texttt{SUPO} can further improve the evaluation performance when scaling test-time maximum round of summarization beyond that of training time. Our results establish summarization-based context management as a principled and scalable approach for training RL agents beyond a fixed context length limit. 

---
# FinLFQA: Evaluating Attributed Text Generation of LLMs in Financial Long-Form Question Answering 

**Authors**: Yitao Long, Tiansheng Hu, Yilun Zhao, Arman Cohan, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06426)  

**Abstract**: Large Language Models (LLMs) frequently hallucinate to long-form questions, producing plausible yet factually incorrect answers. A common mitigation strategy is to provide attribution to LLM outputs. However, existing benchmarks primarily focus on simple attribution that retrieves supporting textual evidence as references. We argue that in real-world scenarios such as financial applications, attribution goes beyond reference retrieval. We introduce FinLFQA, a benchmark designed to evaluate the ability of LLMs to generate long-form answers to complex financial questions with reliable and nuanced attributions. FinLFQA evaluates three critical aspects of attribution through human annotations: (1) supporting evidence extracted from financial reports, (2) intermediate numerical reasoning steps, and (3) domain-specific financial knowledge that informs the reasoning process. We further provide an automatic evaluation framework covering both answer quality and attribution quality. Through extensive experiments on eight LLMs across multiple attribution-generation paradigms, we find that fine-grained metrics are important to distinguish model capabilities, that end-to-end generation achieves comparable performance to post-hoc approaches, and that iterative refinement only helps when guided by external feedback. 

---
# AudioMarathon: A Comprehensive Benchmark for Long-Context Audio Understanding and Efficiency in Audio LLMs 

**Authors**: Peize He, Zichen Wen, Yubo Wang, Yuxuan Wang, Xiaoqian Liu, Jiajie Huang, Zehui Lei, Zhuangcheng Gu, Xiangqi Jin, Jiabing Yang, Kai Li, Zhifei Liu, Weijia Li, Cunxiang Wang, Conghui He, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07293)  

**Abstract**: Processing long-form audio is a major challenge for Large Audio Language models (LALMs). These models struggle with the quadratic cost of attention ($O(N^2)$) and with modeling long-range temporal dependencies. Existing audio benchmarks are built mostly from short clips and do not evaluate models in realistic long context settings. To address this gap, we introduce AudioMarathon, a benchmark designed to evaluate both understanding and inference efficiency on long-form audio. AudioMarathon provides a diverse set of tasks built upon three pillars: long-context audio inputs with durations ranging from 90.0 to 300.0 seconds, which correspond to encoded sequences of 2,250 to 7,500 audio tokens, respectively, full domain coverage across speech, sound, and music, and complex reasoning that requires multi-hop inference. We evaluate state-of-the-art LALMs and observe clear performance drops as audio length grows. We also study acceleration techniques and analyze the trade-offs of token pruning and KV cache eviction. The results show large gaps across current LALMs and highlight the need for better temporal reasoning and memory-efficient architectures. We believe AudioMarathon will drive the audio and multimodal research community to develop more advanced audio understanding models capable of solving complex audio tasks. 

---
# h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning 

**Authors**: Sumeet Ramesh Motwani, Alesia Ivanova, Ziyang Cai, Philip Torr, Riashat Islam, Shital Shah, Christian Schroeder de Witt, Charles London  

**Link**: [PDF](https://arxiv.org/pdf/2510.07312)  

**Abstract**: Large language models excel at short-horizon reasoning tasks, but performance drops as reasoning horizon lengths increase. Existing approaches to combat this rely on inference-time scaffolding or costly step-level supervision, neither of which scales easily. In this work, we introduce a scalable method to bootstrap long-horizon reasoning capabilities using only existing, abundant short-horizon data. Our approach synthetically composes simple problems into complex, multi-step dependency chains of arbitrary length. We train models on this data using outcome-only rewards under a curriculum that automatically increases in complexity, allowing RL training to be scaled much further without saturating. Empirically, our method generalizes remarkably well: curriculum training on composed 6th-grade level math problems (GSM8K) boosts accuracy on longer, competition-level benchmarks (GSM-Symbolic, MATH-500, AIME) by up to 2.06x. Importantly, our long-horizon improvements are significantly higher than baselines even at high pass@k, showing that models can learn new reasoning paths under RL. Theoretically, we show that curriculum RL with outcome rewards achieves an exponential improvement in sample complexity over full-horizon training, providing training signal comparable to dense supervision. h1 therefore introduces an efficient path towards scaling RL for long-horizon problems using only existing data. 

---
# LongRM: Revealing and Unlocking the Context Boundary of Reward Modeling 

**Authors**: Zecheng Tang, Baibei Ji, Quantong Qiu, Haitian Wang, Xiaobo Liang, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06915)  

**Abstract**: Reward model (RM) plays a pivotal role in aligning large language model (LLM) with human preferences. As real-world applications increasingly involve long history trajectories, e.g., LLM agent, it becomes indispensable to evaluate whether a model's responses are not only high-quality but also grounded in and consistent with the provided context. Yet, current RMs remain confined to short-context settings and primarily focus on response-level attributes (e.g., safety or helpfulness), while largely neglecting the critical dimension of long context-response consistency. In this work, we introduce Long-RewardBench, a benchmark specifically designed for long-context RM evaluation, featuring both Pairwise Comparison and Best-of-N tasks. Our preliminary study reveals that even state-of-the-art generative RMs exhibit significant fragility in long-context scenarios, failing to maintain context-aware preference judgments. Motivated by the analysis of failure patterns observed in model outputs, we propose a general multi-stage training strategy that effectively scales arbitrary models into robust Long-context RMs (LongRMs). Experiments show that our approach not only substantially improves performance on long-context evaluation but also preserves strong short-context capability. Notably, our 8B LongRM outperforms much larger 70B-scale baselines and matches the performance of the proprietary Gemini 2.5 Pro model. 

---
