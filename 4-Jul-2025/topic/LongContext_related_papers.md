# MOTIF: Modular Thinking via Reinforcement Fine-tuning in LLMs 

**Authors**: Purbesh Mitra, Sennur Ulukus  

**Link**: [PDF](https://arxiv.org/pdf/2507.02851)  

**Abstract**: Recent advancements in the reasoning capabilities of large language models (LLMs) show that employing group relative policy optimization (GRPO) algorithm for reinforcement learning (RL) training allows the models to use more thinking/reasoning tokens for generating better responses. However, LLMs can generate only a finite amount of tokens while maintaining attention to the previously generated tokens. This limit, also known as the context size of an LLM, is a bottleneck in LLM reasoning with arbitrarily large number of tokens. To think beyond the limit of context size, an LLM must employ a modular thinking strategy to reason over multiple rounds. In this work, we propose $\textbf{MOTIF: Modular Thinking via Reinforcement Finetuning}$ -- an RL training method for generating thinking tokens in multiple rounds, effectively allowing the model to think with additional context size. We trained the open-source model Qwen2.5-3B-Instruct on GSM8K dataset via parameter efficient fine-tuning and tested its accuracy on MATH500 and AIME2024 benchmarks. Our experiments show 3.8\% and 3.3\% improvements over vanilla GRPO based training in the respective benchmarks. Furthermore, this improvement was achieved with only 15\% of samples, thus demonstrating sample efficiency of MOTIF. Our code and models are available at this https URL and this https URL, respectively. 

---
# MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent 

**Authors**: Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02259)  

**Abstract**: Despite improvements by length extrapolation, efficient attention and memory modules, handling infinitely long documents with linear complexity without performance degradation during extrapolation remains the ultimate challenge in long-text processing. We directly optimize for long-text tasks in an end-to-end fashion and introduce a novel agent workflow, MemAgent, which reads text in segments and updates the memory using an overwrite strategy. We extend the DAPO algorithm to facilitate training via independent-context multi-conversation generation. MemAgent has demonstrated superb long-context capabilities, being able to extrapolate from an 8K context trained on 32K text to a 3.5M QA task with performance loss < 5% and achieves 95%+ in 512K RULER test. 

---
