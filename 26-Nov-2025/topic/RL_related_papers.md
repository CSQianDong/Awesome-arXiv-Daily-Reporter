# QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation 

**Authors**: Xinguo Zhu, Shaohui Peng, Jiaming Guo, Yunji Chen, Qi Guo, Yuanbo Wen, Hang Qin, Ruizhi Chen, Qirui Zhou, Ke Gao, Yanjun Wu, Chen Zhao, Ling Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.20100)  

**Abstract**: Developing high-performance GPU kernels is critical for AI and scientific computing, but remains challenging due to its reliance on expert crafting and poor portability. While LLMs offer promise for automation, both general-purpose and finetuned LLMs suffer from two fundamental and conflicting limitations: correctness and efficiency. The key reason is that existing LLM-based approaches directly generate the entire optimized low-level programs, requiring exploration of an extremely vast space encompassing both optimization policies and implementation codes. To address the challenge of exploring an intractable space, we propose Macro Thinking Micro Coding (MTMC), a hierarchical framework inspired by the staged optimization strategy of human experts. It decouples optimization strategy from implementation details, ensuring efficiency through high-level strategy and correctness through low-level implementation. Specifically, Macro Thinking employs reinforcement learning to guide lightweight LLMs in efficiently exploring and learning semantic optimization strategies that maximize hardware utilization. Micro Coding leverages general-purpose LLMs to incrementally implement the stepwise optimization proposals from Macro Thinking, avoiding full-kernel generation errors. Together, they effectively navigate the vast optimization space and intricate implementation details, enabling LLMs for high-performance GPU kernel generation. Comprehensive results on widely adopted benchmarks demonstrate the superior performance of MTMC on GPU kernel generation in both accuracy and running time. On KernelBench, MTMC achieves near 100% and 70% accuracy at Levels 1-2 and 3, over 50% than SOTA general-purpose and domain-finetuned LLMs, with up to 7.3x speedup over LLMs, and 2.2x over expert-optimized PyTorch Eager kernels. On the more challenging TritonBench, MTMC attains up to 59.64% accuracy and 34x speedup. 

---
# Soft Adaptive Policy Optimization 

**Authors**: Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.20347)  

**Abstract**: Reinforcement learning (RL) plays an increasingly important role in enhancing the reasoning capabilities of large language models (LLMs), yet stable and performant policy optimization remains challenging. Token-level importance ratios often exhibit high variance-a phenomenon exacerbated in Mixture-of-Experts models-leading to unstable updates. Existing group-based policy optimization methods, such as GSPO and GRPO, alleviate this problem via hard clipping, making it difficult to maintain both stability and effective learning. We propose Soft Adaptive Policy Optimization (SAPO), which replaces hard clipping with a smooth, temperature-controlled gate that adaptively attenuates off-policy updates while preserving useful learning signals. Compared with GSPO and GRPO, SAPO is both sequence-coherent and token-adaptive. Like GSPO, SAPO maintains sequence-level coherence, but its soft gating forms a continuous trust region that avoids the brittle hard clipping band used in GSPO. When a sequence contains a few highly off-policy tokens, GSPO suppresses all gradients for that sequence, whereas SAPO selectively down-weights only the offending tokens and preserves the learning signal from the near-on-policy ones, improving sample efficiency. Relative to GRPO, SAPO replaces hard token-level clipping with smooth, temperature-controlled scaling, enabling more informative and stable updates. Empirical results on mathematical reasoning benchmarks indicate that SAPO exhibits improved training stability and higher Pass@1 performance under comparable training budgets. Moreover, we employ SAPO to train the Qwen3-VL model series, demonstrating that SAPO yields consistent performance gains across diverse tasks and different model sizes. Overall, SAPO provides a more reliable, scalable, and effective optimization strategy for RL training of LLMs. 

---
# Scaling Agentic Reinforcement Learning for Tool-Integrated Reasoning in VLMs 

**Authors**: Meng Lu, Ran Xu, Yi Fang, Wenxuan Zhang, Yue Yu, Gaurav Srivastava, Yuchen Zhuang, Mohamed Elhoseiny, Charles Fleming, Carl Yang, Zhengzhong Tu, Yang Xie, Guanghua Xiao, Hanrui Wang, Di Jin, Wenqi Shi, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19773)  

**Abstract**: While recent vision-language models (VLMs) demonstrate strong image understanding, their ability to "think with images", i.e., to reason through multi-step visual interactions, remains limited. We introduce VISTA-Gym, a scalable training environment for incentivizing tool-integrated visual reasoning capabilities in VLMs. VISTA-Gym unifies diverse real-world multimodal reasoning tasks (7 tasks from 13 datasets in total) with a standardized interface for visual tools (e.g., grounding, parsing), executable interaction loops, verifiable feedback signals, and efficient trajectory logging, enabling visual agentic reinforcement learning at scale. While recent VLMs exhibit strong text-only reasoning, both proprietary and open-source models still struggle with tool selection, invocation, and coordination. With VISTA-Gym, we train VISTA-R1 to interleave tool-use with agentic reasoning via multi-turn trajectory sampling and end-to-end reinforcement learning. Extensive experiments across 11 public reasoning-intensive VQA benchmarks show that VISTA-R1-8B outperforms state-of-the-art baselines with similar sizes by 9.51%-18.72%, demonstrating VISTA-Gym as an effective training ground to unlock the tool-integrated reasoning capabilities for VLMs. 

---
# DRAFT-RL: Multi-Agent Chain-of-Draft Reasoning for Reinforcement Learning-Enhanced LLMs 

**Authors**: Yuanhao Li, Mingshan Liu, Hongbo Wang, Yiding Zhang, Yifei Ma, Wei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20468)  

**Abstract**: Large Language Models (LLMs) have shown impressive capabilities in multi-step reasoning and this http URL works introduce multi-agent reflection frameworks where multiple LLM agents critique and refine each other's outputs using reinforcement learning (RL). However, these approaches often rely on single-shot responses and lack structural diversity in reasoning exploration. In this paper, we propose DRAFT-RL, a novel framework that integrates Chain-of-Draft (CoD) reasoning into multi-agent RL training. Instead of generating single responses, each agent produces multiple drafts per query, which are then evaluated by peer agents and a learned reward model to identify the most promising trajectory. These selected drafts are used to refine future reasoning strategies through actor-critic this http URL-RL enables explicit multi-path exploration, peer-guided reflection, and reward-aligned selection, resulting in more robust and interpretable LLM agent behavior. We evaluate our method on complex reasoning tasks including code synthesis, symbolic math, and knowledge-intensive QA,demonstrating that DRAFT-RL outperforms existing reflective and RL-based agents by significant margins in both accuracy and convergence speed 

---
# Interactive AI NPCs Powered by LLMs: Technical Report for the CPDC Challenge 2025 

**Authors**: Yitian Huang, Yuxuan Lei, Jianxun Lian, Hao Liao  

**Link**: [PDF](https://arxiv.org/pdf/2511.20200)  

**Abstract**: This report presents the solution and results of our team MSRA\_SC in the Commonsense Persona-Grounded Dialogue Challenge (CPDC 2025). We propose a simple yet effective framework that unifies improvements across both GPU Track and API Track. Our method centers on two key components. First, Context Engineering applies dynamic tool pruning and persona clipping for input compression, combined with post-processing techniques such as parameter normalization and function merging. Together with manually refined prompts, this design improves tool call stability, execution reliability, and role-playing guidance. Second, in the GPU Track, we further adopt GRPO training, replacing supervised fine-tuning with reinforcement learning directly optimized by reward signals. This mitigates small-sample overfitting and significantly enhances task-oriented dialogue performance. In the final evaluation, our team ranks 1st in Task 2 API, 2nd in Task 1 API, and 3rd in both Task 3 API and GPU track, demonstrating the effectiveness of our approach. Our code is publicly available at this https URL 

---
# MapReduce LoRA: Advancing the Pareto Front in Multi-Preference Optimization for Generative Models 

**Authors**: Chieh-Yun Chen, Zhonghao Wang, Qi Chen, Zhifan Ye, Min Shi, Yue Zhao, Yinan Zhao, Hui Qu, Wei-An Lin, Yiru Shen, Ajinkya Kale, Irfan Essa, Humphrey Shi  

**Link**: [PDF](https://arxiv.org/pdf/2511.20629)  

**Abstract**: Reinforcement learning from human feedback (RLHF) with reward models has advanced alignment of generative models to human aesthetic and perceptual preferences. However, jointly optimizing multiple rewards often incurs an alignment tax, improving one dimension while degrading others. To address this, we introduce two complementary methods: MapReduce LoRA and Reward-aware Token Embedding (RaTE). MapReduce LoRA trains preference-specific LoRA experts in parallel and iteratively merges them to refine a shared base model; RaTE learns reward-specific token embeddings that compose at inference for flexible preference control. Experiments on Text-to-Image generation (Stable Diffusion 3.5 Medium and FLUX.1-dev) show improvements of 36.1%, 4.6%, and 55.7%, and 32.7%, 4.3%, and 67.1% on GenEval, PickScore, and OCR, respectively. On Text-to-Video generation (HunyuanVideo), visual and motion quality improve by 48.1% and 90.0%, respectively. On the language task, Helpful Assistant, with Llama-2 7B, helpful and harmless improve by 43.4% and 136.7%, respectively. Our framework sets a new state-of-the-art multi-preference alignment recipe across modalities. 

---
# Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning 

**Authors**: Jiaqi Liu, Kaiwen Xiong, Peng Xia, Yiyang Zhou, Haonian Ji, Lu Feng, Siwei Han, Mingyu Ding, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.19900)  

**Abstract**: Vision-language agents have achieved remarkable progress in a variety of multimodal reasoning tasks; however, their learning remains constrained by the limitations of human-annotated supervision. Recent self-rewarding approaches attempt to overcome this constraint by allowing models to act as their own critics or reward providers. Yet, purely text-based self-evaluation struggles to verify complex visual reasoning steps and often suffers from evaluation hallucinations. To address these challenges, inspired by recent advances in tool-integrated reasoning, we propose Agent0-VL, a self-evolving vision-language agent that achieves continual improvement with tool-integrated reasoning. Agent0-VL incorporates tool usage not only into reasoning but also into self-evaluation and self-repair, enabling the model to introspect, verify, and refine its reasoning through evidence-grounded analysis. It unifies two synergistic roles within a single LVLM: a Solver that performs multi-turn tool-integrated reasoning, and a Verifier that generates structured feedback and fine-grained self-rewards through tool-grounded critique. These roles interact through a Self-Evolving Reasoning Cycle, where tool-based verification and reinforcement learning jointly align the reasoning and evaluation distributions for stable self-improvement. Through this zero-external-reward evolution, Agent0-VL aligns its reasoning and verification behaviors without any human annotation or external reward models, achieving continual self-improvement. Experiments on geometric problem solving and visual scientific analysis show that Agent0-VL achieves an 12.5% improvement over the base model. Our code is available at \href{this https URL}{this https URL}. 

---
