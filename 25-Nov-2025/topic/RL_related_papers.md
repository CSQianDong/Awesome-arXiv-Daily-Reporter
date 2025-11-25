# DeCoRL: Decoupling Reasoning Chains via Parallel Sub-Step Generation and Cascaded Reinforcement for Interpretable and Scalable RLHF 

**Authors**: Ziyuan Gao, Di Liang, Xianjie Wu, Philippe Morel, Minlong Peng  

**Link**: [PDF](https://arxiv.org/pdf/2511.19097)  

**Abstract**: Existing reinforcement learning methods for Chain-of-Thought reasoning suffer from two critical limitations. First, they operate as monolithic black boxes that provide undifferentiated reward signals, obscuring individual step contributions and hindering error diagnosis. Second, sequential decoding has O(n) time complexity. This makes real-time deployment impractical for complex reasoning tasks. We present DeCoRL (Decoupled Reasoning Chains via Coordinated Reinforcement Learning), a novel framework that transforms reasoning from sequential processing into collaborative modular orchestration. DeCoRL trains lightweight specialized models to generate reasoning sub-steps concurrently, eliminating sequential bottlenecks through parallel processing. To enable precise error attribution, the framework designs modular reward functions that score each sub-step independently. Cascaded DRPO optimization then coordinates these rewards while preserving inter-step dependencies. Comprehensive evaluation demonstrates state-of-the-art results across RM-Bench, RMB, and RewardBench, outperforming existing methods including large-scale models. DeCoRL delivers 3.8 times faster inference while maintaining superior solution quality and offers a 22.7\% improvement in interpretability through explicit reward attribution. These advancements, combined with a 72.4\% reduction in energy consumption and a 68\% increase in throughput, make real-time deployment of complex reasoning systems a reality. 

---
# SPINE: Token-Selective Test-Time Reinforcement Learning with Entropy-Band Regularization 

**Authors**: Jianghao Wu, Yasmeen George, Jin Ye, Yicheng Wu, Daniel F. Schmidt, Jianfei Cai  

**Link**: [PDF](https://arxiv.org/pdf/2511.17938)  

**Abstract**: Large language models (LLMs) and multimodal LLMs (MLLMs) excel at chain-of-thought reasoning but face distribution shift at test-time and a lack of verifiable supervision. Recent test-time reinforcement learning (TTRL) methods derive label-free pseudo-rewards from self-consistency voting over sampled trajectories, yet they often collapse: the majority-vote reward prevails, responses shorten, and Pass@1 declines. We trace this to uniform sequence updates in which most tokens are low-entropy followers, while a small high-entropy subset determines the reasoning branches. Thus we propose SPINE, a token-selective test-time reinforcement learning framework that (i) updates only forking tokens, the high-entropy branch points identified from forward-pass statistics, and (ii) applies an entropy-band regularizer at those tokens to sustain exploration when entropy is too low and to suppress noisy supervision when it is too high. SPINE plugs into GRPO-style objectives, optionally with a KL anchor, and requires no labels or reward models. Across ten benchmarks spanning multimodal VQA, general and expert QA, mathematical reasoning, and medical QA, SPINE consistently improves Pass@1 over TTRL while avoiding response-length collapse and yielding more stable training dynamics on both LLM and MLLM backbones. These results indicate that aligning updates with chain-of-thought branch points is a simple and label-free mechanism for stable and effective test-time adaptation in reasoning models. Code is available at this https URL. 

---
# RAVEN++: Pinpointing Fine-Grained Violations in Advertisement Videos with Active Reinforcement Reasoning 

**Authors**: Deyi Ji, Yuekui Yang, Liqun Liu, Peng Shu, Haiyang Wu, Shaogang Tang, Xudong Chen, Shaoping Ma, Tianrun Chen, Lanyun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.19168)  

**Abstract**: Advertising (Ad) is a cornerstone of the digital economy, yet the moderation of video advertisements remains a significant challenge due to their complexity and the need for precise violation localization. While recent advancements, such as the RAVEN model, have improved coarse-grained violation detection, critical gaps persist in fine-grained understanding, explainability, and generalization. To address these limitations, we propose RAVEN++, a novel framework that introduces three key innovations: 1) Active Reinforcement Learning (RL), which dynamically adapts training to samples of varying difficulty; 2) Fine-Grained Violation Understanding, achieved through hierarchical reward functions and reasoning distillation; and 3) Progressive Multi-Stage Training, which systematically combines knowledge injection, curriculum-based passive RL, and active RL. Extensive experiments on both public and proprietary datasets, on both offline scenarios and online deployed A/B Testing, demonstrate that RAVEN++ outperforms general-purpose LLMs and specialized models like RAVEN in terms of fine-grained violation understanding, reasoning capabilities, and generalization ability. 

---
# Natural Emergent Misalignment from Reward Hacking in Production RL 

**Authors**: Monte MacDiarmid, Benjamin Wright, Jonathan Uesato, Joe Benton, Jon Kutasov, Sara Price, Naia Bouscal, Sam Bowman, Trenton Bricken, Alex Cloud, Carson Denison, Johannes Gasteiger, Ryan Greenblatt, Jan Leike, Jack Lindsey, Vlad Mikulik, Ethan Perez, Alex Rodrigues, Drake Thomas, Albert Webson, Daniel Ziegler, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2511.18397)  

**Abstract**: We show that when large language models learn to reward hack on production RL environments, this can result in egregious emergent misalignment. We start with a pretrained model, impart knowledge of reward hacking strategies via synthetic document finetuning or prompting, and train on a selection of real Anthropic production coding environments. Unsurprisingly, the model learns to reward hack. Surprisingly, the model generalizes to alignment faking, cooperation with malicious actors, reasoning about malicious goals, and attempting sabotage when used with Claude Code, including in the codebase for this paper. Applying RLHF safety training using standard chat-like prompts results in aligned behavior on chat-like evaluations, but misalignment persists on agentic tasks. Three mitigations are effective: (i) preventing the model from reward hacking; (ii) increasing the diversity of RLHF safety training; and (iii) "inoculation prompting", wherein framing reward hacking as acceptable behavior during training removes misaligned generalization even when reward hacking is learned. 

---
# ORIGAMISPACE: Benchmarking Multimodal LLMs in Multi-Step Spatial Reasoning with Mathematical Constraints 

**Authors**: Rui Xu, Dakuan Lu, Zicheng Zhao, Xiaoyu Tan, Xintao Wang, Siyu Yuan, Jiangjie Chen, Yinghui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.18450)  

**Abstract**: Spatial reasoning is a key capability in the field of artificial intelligence, especially crucial in areas such as robotics, computer vision, and natural language understanding. However, evaluating the ability of multimodal large language models(MLLMs) in complex spatial reasoning still faces challenges, particularly in scenarios requiring multi-step reasoning and precise mathematical constraints. This paper introduces ORIGAMISPACE, a new dataset and benchmark designed to evaluate the multi-step spatial reasoning ability and the capacity to handle mathematical constraints of MLLMs through origami tasks. The dataset contains 350 data instances,each comprising a strictly formatted crease pattern (CP diagram), the Compiled Flat Pattern, the complete Folding Process, and the final Folded Shape Image. We propose four evaluation tasks: Pattern Prediction, Multi-step Spatial Reasoning, Spatial Relationship Prediction, and End-to-End CP Code Generation. For the CP code generation task, we design an interactive environment and explore the possibility of using reinforcement learning methods to train MLLMs. Through experiments on existing MLLMs, we initially reveal the strengths and weaknesses of these models in handling complex spatial reasoning tasks. 

---
# Alignment Faking - the Train -> Deploy Asymmetry: Through a Game-Theoretic Lens with Bayesian-Stackelberg Equilibria 

**Authors**: Kartik Garg, Shourya Mishra, Kartikeya Sinha, Ojaswi Pratap Singh, Ayush Chopra, Kanishk Rai, Ammar Sheikh, Raghav Maheshwari, Aman Chadha, Vinija Jain, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2511.17937)  

**Abstract**: Alignment faking is a form of strategic deception in AI in which models selectively comply with training objectives when they infer that they are in training, while preserving different behavior outside training. The phenomenon was first documented for Claude 3 Opus and later examined across additional large language models. In these setups, the word "training" refers to simulated training via prompts without parameter updates, so the observed effects are context conditioned shifts in behavior rather than preference learning. We study the phenomenon using an evaluation framework that compares preference optimization methods (BCO, DPO, KTO, and GRPO) across 15 models from four model families, measured along three axes: safety, harmlessness, and helpfulness. Our goal is to identify what causes alignment faking and when it occurs. 

---
# Training Emergent Joint Associations: A Reinforcement Learning Approach to Creative Thinking in Language Models 

**Authors**: Mukul Singh, Ananya Singha, Aishni Parab, Pronita Mehrotra, Sumit Gulwani  

**Link**: [PDF](https://arxiv.org/pdf/2511.17876)  

**Abstract**: Associative thinking--the ability to connect seemingly unrelated ideas--is a foundational element of human creativity and problem-solving. This paper explores whether reinforcement learning (RL) guided by associative thinking principles can enhance a model's performance across diverse generative tasks, including story writing, code generation, and chart creation. We introduce a reinforcement learning framework that uses a prompt-based evaluation mechanism, incorporating established divergent thinking metrics from creativity research. A base language model is fine-tuned using this framework to reward outputs demonstrating higher novelty through higher degrees of conceptual connectivity. Interestingly, the experimental results suggest that RL-based associative thinking-trained models not only generate more original and coherent stories but also exhibit improved abstraction and flexibility in tasks such as programming and data visualization. Our findings provide initial evidence that modeling cognitive creativity principles through reinforcement learning can yield more adaptive and generative AI. 

---
# VDC-Agent: When Video Detailed Captioners Evolve Themselves via Agentic Self-Reflection 

**Authors**: Qiang Wang, Xinyuan Gao, SongLin Dong, Jizhou Han, Jiangyang Li, Yuhang He, Yihong Gong  

**Link**: [PDF](https://arxiv.org/pdf/2511.19436)  

**Abstract**: We present VDC-Agent, a self-evolving framework for Video Detailed Captioning that requires neither human annotations nor larger teacher models. The agent forms a closed loop of caption generation, principle-guided scoring (score and textual suggestions), and prompt refinement. When caption quality regresses, a self-reflection path leverages the previous chain-of-thought to amend the update. Running this process on unlabeled videos produces trajectories of (caption, score) pairs. We convert the trajectories into preference tuples and filter out samples with JSON parsing errors, resulting in VDC-Agent-19K, which contains 18,886 automatically constructed pairs. We then fine-tune the base MLLM on this dataset using an easy-to-hard curriculum direct preference optimization. Built on Qwen2.5-VL-7B-Instruct, our VDC-Agent-7B attains state-of-the-art performance on the VDC benchmark with 49.08% average accuracy and 2.50 score, surpassing specialized video captioners and improving over the base model by +5.13% accuracy and +0.27 score at similar inference cost. 

---
# SLMFix: Leveraging Small Language Models for Error Fixing with Reinforcement Learning 

**Authors**: David Jiahao Fu, Aryan Gupta, Aaron Councilman, David Grove, Yu-Xiong Wang, Vikram Adve  

**Link**: [PDF](https://arxiv.org/pdf/2511.19422)  

**Abstract**: Recent advancements in large language models (LLMs) have shown very impressive capabilities in code generation across many programming languages. However, even state-of-the-art LLMs generate programs that contains syntactic errors and fail to complete the given tasks, especially for low-resource programming languages (LRPLs). In addition, high training cost makes finetuning LLMs unaffordable with constrained computational resources, further undermining the effectiveness of LLMs for code generation. In this work, we propose SLMFix, a novel code generation pipeline that leverages a small language model (SLM) finetuned using reinforcement learning (RL) techniques to fix syntactic errors in LLM-generated programs to improve the quality of LLM-generated programs for domain-specific languages (DSLs). In specific, we applied RL on the SLM for the program repair task using a reward calculated using both a static validator and a static semantic similarity metric. Our experimental results demonstrate the effectiveness and generalizability of our approach across multiple DSLs, achieving more than 95% pass rate on the static validator. Notably, SLMFix brings substantial improvement to the base model and outperforms supervised finetuning approach even for 7B models on a LRPL, showing the potential of our approach as an alternative to traditional finetuning approaches. 

---
# OrdMoE: Preference Alignment via Hierarchical Expert Group Ranking in Multimodal Mixture-of-Experts LLMs 

**Authors**: Yuting Gao, Weihao Chen, Lan Wang, Ruihan Xu, Qingpei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.19023)  

**Abstract**: Preference learning has recently emerged as a pivotal strategy for post-training alignment of Multimodal Large Language Models (MLLMs). However, existing approaches predominantly rely on external human-annotated preference data, which is costly and labor-intensive to collect. In this work, we propose OrdMoE, a novel preference alignment framework that bypasses the reliance on external human preferences entirely by leveraging intrinsic signals within Mixture-of-Experts (MoE) architectures. Specifically, we observe that the router's expert selection scores implicitly encode a quality-aware ranking of responses (i.e. higher-scoring experts consistently generate higher-quality outputs). Building on this insight, OrdMoE constructs an internal preference hierarchy by grouping experts into ranked tiers based on their per-token routing scores and activating each tier separately to produce a sequence of responses with increasing quality. This yields a zero-cost, self-supervised preference ordering over generated responses, which can be directly optimized using standard preference learning objectives. Extensive experiments across multiple multimodal benchmarks demnstrate that OrdMoE significantly enhances both alignment and overall performance of multimodal Mixture-of-Experts LLMs, achieving competitive results without requiring any human-annotated preference data. 

---
# ProxT2I: Efficient Reward-Guided Text-to-Image Generation via Proximal Diffusion 

**Authors**: Zhenghan Fang, Jian Zheng, Qiaozi Gao, Xiaofeng Gao, Jeremias Sulam  

**Link**: [PDF](https://arxiv.org/pdf/2511.18742)  

**Abstract**: Diffusion models have emerged as a dominant paradigm for generative modeling across a wide range of domains, including prompt-conditional generation. The vast majority of samplers, however, rely on forward discretization of the reverse diffusion process and use score functions that are learned from data. Such forward and explicit discretizations can be slow and unstable, requiring a large number of sampling steps to produce good-quality samples. In this work we develop a text-to-image (T2I) diffusion model based on backward discretizations, dubbed ProxT2I, relying on learned and conditional proximal operators instead of score functions. We further leverage recent advances in reinforcement learning and policy optimization to optimize our samplers for task-specific rewards. Additionally, we develop a new large-scale and open-source dataset comprising 15 million high-quality human images with fine-grained captions, called LAION-Face-T2I-15M, for training and evaluation. Our approach consistently enhances sampling efficiency and human-preference alignment compared to score-based baselines, and achieves results on par with existing state-of-the-art and open-source text-to-image models while requiring lower compute and smaller model size, offering a lightweight yet performant solution for human text-to-image generation. 

---
# LLM Reasoning for Cold-Start Item Recommendation 

**Authors**: Shijun Li, Yu Wang, Jin Wang, Ying Li, Joydeep Ghosh, Anne Cocos  

**Link**: [PDF](https://arxiv.org/pdf/2511.18261)  

**Abstract**: Large Language Models (LLMs) have shown significant potential for improving recommendation systems through their inherent reasoning capabilities and extensive knowledge base. Yet, existing studies predominantly address warm-start scenarios with abundant user-item interaction data, leaving the more challenging cold-start scenarios, where sparse interactions hinder traditional collaborative filtering methods, underexplored. To address this limitation, we propose novel reasoning strategies designed for cold-start item recommendations within the Netflix domain. Our method utilizes the advanced reasoning capabilities of LLMs to effectively infer user preferences, particularly for newly introduced or rarely interacted items. We systematically evaluate supervised fine-tuning, reinforcement learning-based fine-tuning, and hybrid approaches that combine both methods to optimize recommendation performance. Extensive experiments on real-world data demonstrate significant improvements in both methodological efficacy and practical performance in cold-start recommendation contexts. Remarkably, our reasoning-based fine-tuned models outperform Netflix's production ranking model by up to 8% in certain cases. 

---
# The Alignment Paradox of Medical Large Language Models in Infertility Care: Decoupling Algorithmic Improvement from Clinical Decision-making Quality 

**Authors**: Dou Liu, Ying Long, Sophia Zuoqiu, Kaipeng Xie, Runze Yang, Di Liu, Kang Li, Yiting Lin, Hanyi Liu, Rong Yin, Tian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18084)  

**Abstract**: Large language models (LLMs) are increasingly adopted in clinical decision support, yet aligning them with the multifaceted reasoning pathways of real-world medicine remains a major challenge. Using more than 8,000 infertility treatment records, we systematically evaluate four alignment strategies: Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), Group Relative Policy Optimization (GRPO), and In-Context Learning (ICL) through a dual-layer framework combining automatic benchmarks with blinded doctor-in-the-loop assessments. GRPO achieves the highest algorithmic accuracy across multiple decision layers, confirming the value of reinforcement-based optimization for structured prediction tasks. However, clinicians consistently prefer the SFT model, citing clearer reasoning processes (p = 0.035) and higher therapeutic feasibility (p = 0.019). In blinded pairwise comparisons, SFT attains the highest winning rate (51.2%), outperforming both GRPO (26.2%) and even physicians' original decisions (22.7%). These results reveal an alignment paradox: algorithmic improvements do not necessarily translate into higher clinical trust, and may diverge from human-centered preferences. Our findings highlight the need for alignment strategies that prioritize clinically interpretable and practically feasible reasoning, rather than solely optimizing decision-level accuracy. 

---
# Multi-Value Alignment for LLMs via Value Decorrelation and Extrapolation 

**Authors**: Hefei Xu, Le Wu, Chen Cheng, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.17579)  

**Abstract**: With the rapid advancement of large language models (LLMs), aligning them with human values for safety and ethics has become a critical challenge. This problem is especially challenging when multiple, potentially conflicting human values must be considered and balanced. Although several variants of existing alignment methods (such as Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO)) have been proposed to address multi-value alignment, they suffer from notable limitations: 1) they are often unstable and inefficient in multi-value optimization; and 2) they fail to effectively handle value conflicts. As a result, these approaches typically struggle to achieve optimal trade-offs when aligning multiple values.
To address this challenge, we propose a novel framework called Multi-Value Alignment (MVA). It mitigates alignment degradation caused by parameter interference among diverse human values by minimizing their mutual information. Furthermore, we propose a value extrapolation strategy to efficiently explore the Pareto frontier, thereby constructing a set of LLMs with diverse value preferences. Extensive experiments demonstrate that MVA consistently outperforms existing baselines in aligning LLMs with multiple human values. 

---
# Multimodal Large Language Models with Adaptive Preference Optimization for Sequential Recommendation 

**Authors**: Yu Wang, Yonghui Yang, Le Wu, Yi Zhang, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2511.18740)  

**Abstract**: Recent advances in Large Language Models (LLMs) have opened new avenues for sequential recommendation by enabling natural language reasoning over user behavior sequences. A common approach formulates recommendation as a language modeling task, where interaction histories are transformed into prompts and user preferences are learned via supervised fine-tuning. However, these methods operate solely in the textual modality and often miss users' fine-grained interests, especially when shaped by rich visual signals such as product images or movie posters. Multimodal Large Language Models (MLLMs) offer a promising alternative by aligning text and vision in a shared semantic space. A prevalent training paradigm applies Supervised Fine-Tuning (SFT) followed by Direct Preference Optimization (DPO) to model user preferences. Yet, two core challenges remain: 1) Imbalanced sample hardness, where random negative sampling causes overfitting on easy examples and under-training on hard ones; 2) Cross-modal semantic bias, where the fixed reference model in DPO prevents the policy model from correcting modality misalignments--especially over long sequences. To address these issues, we propose a Multimodal LLM framework that integrates Hardness-aware and Noise-regularized preference optimization for Recommendation (HaNoRec). Specifically, HaNoRec dynamically adjusts optimization weights based on both the estimated hardness of each training sample and the policy model's real-time responsiveness, prioritizing harder examples. It further introduces Gaussian-perturbed distribution optimization on output logits to enhance cross-modal semantic consistency and reduce modality bias inherited from the reference model. 

---
