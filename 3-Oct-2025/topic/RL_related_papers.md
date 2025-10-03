# More Than One Teacher: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration 

**Authors**: Xiaoyang Yuan, Yujuan Ding, Yi Bin, Wenqi Shao, Jinyu Cai, Jingkuan Song, Yang Yang, Hengtao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.02227)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is a promising paradigm for enhancing the reasoning ability in Large Language Models (LLMs). However, prevailing methods primarily rely on self-exploration or a single off-policy teacher to elicit long chain-of-thought (LongCoT) reasoning, which may introduce intrinsic model biases and restrict exploration, ultimately limiting reasoning diversity and performance. Drawing inspiration from multi-teacher strategies in knowledge distillation, we introduce Adaptive Multi-Guidance Policy Optimization (AMPO), a novel framework that adaptively leverages guidance from multiple proficient teacher models, but only when the on-policy model fails to generate correct solutions. This "guidance-on-demand" approach expands exploration while preserving the value of self-discovery. Moreover, AMPO incorporates a comprehension-based selection mechanism, prompting the student to learn from the reasoning paths that it is most likely to comprehend, thus balancing broad exploration with effective exploitation. Extensive experiments show AMPO substantially outperforms a strong baseline (GRPO), with a 4.3% improvement on mathematical reasoning tasks and 12.2% on out-of-distribution tasks, while significantly boosting Pass@k performance and enabling more diverse exploration. Notably, using four peer-sized teachers, our method achieves comparable results to approaches that leverage a single, more powerful teacher (e.g., DeepSeek-R1) with more data. These results demonstrate a more efficient and scalable path to superior reasoning and generalizability. Our code is available at this https URL. 

---
# Learning to Reason for Hallucination Span Detection 

**Authors**: Hsuan Su, Ting-Yao Hu, Hema Swetha Koppula, Kundan Krishna, Hadi Pouransari, Cheng-Yu Hsieh, Cem Koc, Joseph Yitan Cheng, Oncel Tuzel, Raviteja Vemulapalli  

**Link**: [PDF](https://arxiv.org/pdf/2510.02173)  

**Abstract**: Large language models (LLMs) often generate hallucinations -- unsupported content that undermines reliability. While most prior works frame hallucination detection as a binary task, many real-world applications require identifying hallucinated spans, which is a multi-step decision making process. This naturally raises the question of whether explicit reasoning can help the complex task of detecting hallucination spans. To answer this question, we first evaluate pretrained models with and without Chain-of-Thought (CoT) reasoning, and show that CoT reasoning has the potential to generate at least one correct answer when sampled multiple times. Motivated by this, we propose RL4HS, a reinforcement learning framework that incentivizes reasoning with a span-level reward function. RL4HS builds on Group Relative Policy Optimization and introduces Class-Aware Policy Optimization to mitigate reward imbalance issue. Experiments on the RAGTruth benchmark (summarization, question answering, data-to-text) show that RL4HS surpasses pretrained reasoning models and supervised fine-tuning, demonstrating the necessity of reinforcement learning with span-level rewards for detecting hallucination spans. 

---
# RESTRAIN: From Spurious Votes to Signals -- Self-Driven RL with Self-Penalization 

**Authors**: Zhaoning Yu, Will Su, Leitian Tao, Haozhu Wang, Aashu Singh, Hanchao Yu, Jianyu Wang, Hongyang Gao, Weizhe Yuan, Jason Weston, Ping Yu, Jing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02172)  

**Abstract**: Reinforcement learning with human-annotated data has boosted chain-of-thought reasoning in large reasoning models, but these gains come at high costs in labeled data while faltering on harder tasks. A natural next step is experience-driven learning, where models improve without curated labels by adapting to unlabeled data. We introduce RESTRAIN (REinforcement learning with Self-restraint), a self-penalizing RL framework that converts the absence of gold labels into a useful learning signal. Instead of overcommitting to spurious majority votes, RESTRAIN exploits signals from the model's entire answer distribution: penalizing overconfident rollouts and low-consistency examples while preserving promising reasoning chains. The self-penalization mechanism integrates seamlessly into policy optimization methods such as GRPO, enabling continual self-improvement without supervision. On challenging reasoning benchmarks, RESTRAIN delivers large gains using only unlabeled data. With Qwen3-4B-Base and OctoThinker Hybrid-8B-Base, it improves Pass@1 by up to +140.7 percent on AIME25, +36.2 percent on MMLU_STEM, and +19.6 percent on GPQA-Diamond, nearly matching gold-label training while using no gold labels. These results demonstrate that RESTRAIN establishes a scalable path toward stronger reasoning without gold labels. 

---
# Enhancing Large Language Model Reasoning with Reward Models: An Analytical Survey 

**Authors**: Qiyuan Liu, Hao Xu, Xuhong Chen, Wei Chen, Yee Whye Teh, Ning Miao  

**Link**: [PDF](https://arxiv.org/pdf/2510.01925)  

**Abstract**: Reward models (RMs) play a critical role in enhancing the reasoning performance of LLMs. For example, they can provide training signals to finetune LLMs during reinforcement learning (RL) and help select the best answer from multiple candidates during inference. In this paper, we provide a systematic introduction to RMs, along with a comprehensive survey of their applications in LLM reasoning. We first review fundamental concepts of RMs, including their architectures, training methodologies, and evaluation techniques. Then, we explore their key applications: (1) guiding generation and selecting optimal outputs during LLM inference, (2) facilitating data synthesis and iterative self-improvement for LLMs, and (3) providing training signals in RL-based finetuning. Finally, we address critical open questions regarding the selection, generalization, evaluation, and enhancement of RMs, based on existing research and our own empirical findings. Our analysis aims to provide actionable insights for the effective deployment and advancement of RMs for LLM reasoning. 

---
# Veri-R1: Toward Precise and Faithful Claim Verification via Online Reinforcement Learning 

**Authors**: Qi He, Cheng Qian, Xiusi Chen, Bingxiang He, Yi R., Fung, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.01932)  

**Abstract**: Claim verification with large language models (LLMs) has recently attracted considerable attention, owing to their superior reasoning capabilities and transparent verification pathways compared to traditional answer-only judgments. Online claim verification requires iterative evidence retrieval and reasoning, yet existing approaches mainly rely on prompt engineering or predesigned reasoning workflows without offering a unified training paradigm to improve necessary skills. Therefore, we introduce Veri-R1, an online reinforcement learning (RL) framework that enables an LLM to interact with a search engine and to receive reward signals that explicitly shape its planning, retrieval, and reasoning behaviors. The dynamic interaction between models and retrieval systems more accurately reflects real-world verification scenarios and fosters comprehensive verification skills. Empirical results show that Veri-R1 improves joint accuracy by up to 30% and doubles evidence score, often surpassing larger-scale counterparts. Ablation studies further reveal the impact of reward components and the link between output logits and label accuracy. Our results highlight the effectiveness of online RL for precise and faithful claim verification and provide a foundation for future research. We release our code to support community progress in LLM empowered claim verification. 

---
# Efficient Training of Robust Traditional Chinese LLaMA-1B on a Single Consumer GPU: Continual Pre-training, SFT, and DPO 

**Authors**: Yu-Cheng Chih, Ming-Tao Duan, Yong-Hao Hou  

**Link**: [PDF](https://arxiv.org/pdf/2510.01616)  

**Abstract**: Small Language Models (SLMs) enable cost-effective, on-device and latency-sensitive AI applications, yet their deployment in Traditional Chinese (TC) remains hindered by token-level instability - models unpredictably emit non-TC characters or code-switch into other languages. We address this practical reliability gap by creating PureTC-1B, a three-stage stabilization pipeline for Llama-3.2-1B-Instruct (an open-weight, instruction-tuned model released by Meta) using parameter-efficient LoRA adapters. Our method combines Continual Pre-Training (CPT) on TC-centric corpora, Supervised Fine-Tuning (SFT) with instruction data, and Direct Preference Optimization (DPO) using TC-adherence preferences to improve monolingual robustness without full-model retraining. On a benchmark designed to simulate real-world usage, PureTC-1B achieves a 51.3% relative reduction (micro-average) in non-TC output tokens versus the base model. On a Named Entity Translation (NET) task, PureTC-1B further reduces incorrect-language tokens by 77.2% relative to Llama-3B and 57.2% relative to Qwen-1.5B, indicating that robust TC adherence is attainable even at the 1B scale. The pipeline is reproducible, adapter-only, and hardware-friendly, offering practitioners a practical recipe to enhance language stability for TC and potentially other non-English languages. 

---
# Detoxifying Large Language Models via Autoregressive Reward Guided Representation Editing 

**Authors**: Yisong Xiao, Aishan Liu, Siyuan Liang, Zonghao Ying, Xianglong Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.01243)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance across various tasks, yet they remain vulnerable to generating toxic content, necessitating detoxification strategies to ensure safe and responsible deployment. Test-time detoxification methods, which typically introduce static or dynamic interventions into LLM representations, offer a promising solution due to their flexibility and minimal invasiveness. However, current approaches often suffer from imprecise interventions, primarily due to their insufficient exploration of the transition space between toxic and non-toxic outputs. To address this challenge, we propose \textsc{A}utoregressive \textsc{R}eward \textsc{G}uided \textsc{R}epresentation \textsc{E}diting (ARGRE), a novel test-time detoxification framework that explicitly models toxicity transitions within the latent representation space, enabling stable and precise reward-guided editing. ARGRE identifies non-toxic semantic directions and interpolates between toxic and non-toxic representations to reveal fine-grained transition trajectories. These trajectories transform sparse toxicity annotations into dense training signals, enabling the construction of an autoregressive reward model that delivers stable and precise editing guidance. At inference, the reward model guides an adaptive two-step editing process to obtain detoxified representations: it first performs directional steering based on expected reward gaps to shift representations toward non-toxic regions, followed by lightweight gradient-based refinements. Extensive experiments across 8 widely used LLMs show that ARGRE significantly outperforms leading baselines in effectiveness (-62.21% toxicity) and efficiency (-47.58% inference time), while preserving the core capabilities of the original model with minimal degradation. Our code is available at the website. 

---
# RLAD: Training LLMs to Discover Abstractions for Solving Reasoning Problems 

**Authors**: Yuxiao Qu, Anikait Singh, Yoonho Lee, Amrith Setlur, Ruslan Salakhutdinov, Chelsea Finn, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.02263)  

**Abstract**: Reasoning requires going beyond pattern matching or memorization of solutions to identify and implement "algorithmic procedures" that can be used to deduce answers to hard problems. Doing so requires realizing the most relevant primitives, intermediate results, or shared procedures, and building upon them. While RL post-training on long chains of thought ultimately aims to uncover this kind of algorithmic behavior, most reasoning traces learned by large models fail to consistently capture or reuse procedures, instead drifting into verbose and degenerate exploration. To address more effective reasoning, we introduce reasoning abstractions: concise natural language descriptions of procedural and factual knowledge that guide the model toward learning successful reasoning. We train models to be capable of proposing multiple abstractions given a problem, followed by RL that incentivizes building a solution while using the information provided by these abstractions. This results in a two-player RL training paradigm, abbreviated as RLAD, that jointly trains an abstraction generator and a solution generator. This setup effectively enables structured exploration, decouples learning signals of abstraction proposal and solution generation, and improves generalization to harder problems. We also show that allocating more test-time compute to generating abstractions is more beneficial for performance than generating more solutions at large test budgets, illustrating the role of abstractions in guiding meaningful exploration. 

---
# GRPO++: Enhancing Dermatological Reasoning under Low Resource Settings 

**Authors**: Ismam Nur Swapnil, Aranya Saha, Tanvir Ahmed Khan, Mohammad Ariful Haque  

**Link**: [PDF](https://arxiv.org/pdf/2510.01236)  

**Abstract**: Vision-Language Models (VLMs) show promise in medical image analysis, yet their capacity for structured reasoning in complex domains like dermatology is often limited by data scarcity and the high computational cost of advanced training techniques. To address these challenges, we introduce DermIQ-VLM, a VLM developed through a multi-stage, resource-efficient methodology designed to emulate a dermatologist's diagnostic process. Our primary contribution is a modified version of Grouped Relative Policy Optimization (GRPO), called GRPO++, which stabilizes the powerful but data-intensive GRPO framework. Our proposed training pipeline first employs GRPO++ for reasoning-oriented disease recognition, followed by supervised fine-tuning for conversational ability. To mitigate factual errors introduced during this step, we then align the model using Direct Preference Optimization (DPO), leveraging a Knowledge Graph-based system as a scalable proxy for expert preference. A preliminary evaluation on a curated dermatological dataset demonstrates that our proposed methodology yields notable performance gains over standard fine-tuning approaches. These findings validate the potential of our pipeline as a feasible pathway for developing specialized, reliable VLMs in resource-constrained environments. 

---
# Plan Then Action:High-Level Planning Guidance Reinforcement Learning for LLM Reasoning 

**Authors**: Zhihao Dou, Qinjian Zhao, Zhongwei Wan, Dinggen Zhang, Weida Wang, Towsif Raiyan, Benteng Chen, Qingtao Pan, Yang Ouyang, Zhiqiang Gao, Shufei Zhang, Sumon Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2510.01833)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable reasoning abilities in complex tasks, often relying on Chain-of-Thought (CoT) reasoning. However, due to their autoregressive token-level generation, the reasoning process is largely constrained to local decision-making and lacks global planning. This limitation frequently results in redundant, incoherent, or inaccurate reasoning, which significantly degrades overall performance. Existing approaches, such as tree-based algorithms and reinforcement learning (RL), attempt to address this issue but suffer from high computational costs and often fail to produce optimal reasoning trajectories. To tackle this challenge, we propose Plan-Then-Action Enhanced Reasoning with Group Relative Policy Optimization PTA-GRPO, a two-stage framework designed to improve both high-level planning and fine-grained CoT reasoning. In the first stage, we leverage advanced LLMs to distill CoT into compact high-level guidance, which is then used for supervised fine-tuning (SFT). In the second stage, we introduce a guidance-aware RL method that jointly optimizes the final output and the quality of high-level guidance, thereby enhancing reasoning effectiveness. We conduct extensive experiments on multiple mathematical reasoning benchmarks, including MATH, AIME2024, AIME2025, and AMC, across diverse base models such as Qwen2.5-7B-Instruct, Qwen3-8B, Qwen3-14B, and LLaMA3.2-3B. Experimental results demonstrate that PTA-GRPO consistently achieves stable and significant improvements across different models and tasks, validating its effectiveness and generalization. 

---
# ExGRPO: Learning to Reason from Experience 

**Authors**: Runzhe Zhan, Yafu Li, Zhi Wang, Xiaoye Qu, Dongrui Liu, Jing Shao, Derek F. Wong, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.02245)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) is an emerging paradigm for improving the reasoning ability of large language models. However, standard on-policy training discards rollout experiences after a single update, leading to computational inefficiency and instability. While prior work on RL has highlighted the benefits of reusing past experience, the role of experience characteristics in shaping learning dynamics of large reasoning models remains underexplored. In this paper, we are the first to investigate what makes a reasoning experience valuable and identify rollout correctness and entropy as effective indicators of experience value. Based on these insights, we propose ExGRPO (Experiential Group Relative Policy Optimization), a framework that organizes and prioritizes valuable experiences, and employs a mixed-policy objective to balance exploration with experience exploitation. Experiments on five backbone models (1.5B-8B parameters) show that ExGRPO consistently improves reasoning performance on mathematical/general benchmarks, with an average gain of +3.5/7.6 points over on-policy RLVR. Moreover, ExGRPO stabilizes training on both stronger and weaker models where on-policy methods fail. These results highlight principled experience management as a key ingredient for efficient and scalable RLVR. 

---
# The Reasoning Boundary Paradox: How Reinforcement Learning Constrains Language Models 

**Authors**: Phuc Minh Nguyen, Chinh D. La, Duy M. H. Nguyen, Nitesh V. Chawla, Binh T. Nguyen, Khoa D. Doan  

**Link**: [PDF](https://arxiv.org/pdf/2510.02230)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a key method for improving Large Language Models' reasoning capabilities, yet recent evidence suggests it may paradoxically shrink the reasoning boundary rather than expand it. This paper investigates the shrinkage issue of RLVR by analyzing its learning dynamics and reveals two critical phenomena that explain this failure. First, we expose negative interference in RLVR, where learning to solve certain training problems actively reduces the likelihood of correct solutions for others, leading to the decline of Pass@$k$ performance, or the probability of generating a correct solution within $k$ attempts. Second, we uncover the winner-take-all phenomenon: RLVR disproportionately reinforces problems with high likelihood, correct solutions, under the base model, while suppressing other initially low-likelihood ones. Through extensive theoretical and empirical analysis on multiple mathematical reasoning benchmarks, we show that this effect arises from the inherent on-policy sampling in standard RL objectives, causing the model to converge toward narrow solution strategies. Based on these insights, we propose a simple yet effective data curation algorithm that focuses RLVR learning on low-likelihood problems, achieving notable improvement in Pass@$k$ performance. Our code is available at this https URL. 

---
# Think Right: Learning to Mitigate Under-Over Thinking via Adaptive, Attentive Compression 

**Authors**: Joykirat Singh, Justin Chih-Yao Chen, Archiki Prasad, Elias Stengel-Eskin, Akshay Nambi, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2510.01581)  

**Abstract**: Recent thinking models solve complex reasoning tasks by scaling test-time compute, but this scaling must be allocated in line with task difficulty. On one hand, short reasoning (underthinking) leads to errors on harder problems that require extended reasoning steps; but, excessively long reasoning (overthinking) can be token-inefficient, generating unnecessary steps even after reaching a correct intermediate solution. We refer to this as under-adaptivity, where the model fails to modulate its response length appropriately given problems of varying difficulty. To address under-adaptivity and strike a balance between under- and overthinking, we propose TRAAC (Think Right with Adaptive, Attentive Compression), an online post-training RL method that leverages the model's self-attention over a long reasoning trajectory to identify important steps and prune redundant ones. TRAAC also estimates difficulty and incorporates it into training rewards, thereby learning to allocate reasoning budget commensurate with example difficulty. Our approach improves accuracy, reduces reasoning steps, and enables adaptive thinking compared to base models and other RL baselines. Across a variety of tasks (AIME, AMC, GPQA-D, BBEH), TRAAC (Qwen3-4B) achieves an average absolute accuracy gain of 8.4% with a relative reduction in reasoning length of 36.8% compared to the base model, and a 7.9% accuracy gain paired with a 29.4% length drop compared to the best RL baseline. TRAAC also shows strong generalization: although our models are trained on math datasets, they show accuracy and efficiency gains on out-of-distribution non-math datasets like GPQA-D, BBEH, and OptimalThinkingBench. Our analysis further verifies that TRAAC provides fine-grained adjustments to thinking budget based on difficulty and that a combination of task-difficulty calibration and attention-based compression yields gains across diverse tasks. 

---
# InvThink: Towards AI Safety via Inverse Reasoning 

**Authors**: Yubin Kim, Taehan Kim, Eugene Park, Chunjong Park, Cynthia Breazeal, Daniel McDuff, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.01569)  

**Abstract**: We present InvThink, a simple yet powerful approach that gives large language models (LLMs) the capability of inverse thinking: reasoning through failure modes before generating responses. Unlike existing safety alignment methods that optimize directly for safe response, InvThink instructs models to 1) enumerate potential harms, 2) analyze their consequences, and 3) generate safe outputs that proactively avoid these risks. Our method reveals three key findings: (i) safety improvements show stronger scaling with model size compared to existing safety methods. (ii) InvThink mitigates safety tax; by training models to systematically consider failure modes, it preserves general reasoning capabilities on standard benchmarks. (iii) beyond general safety tasks, InvThink excels in high-stakes domains including external-facing (medicine, finance, law) and agentic (blackmail, murder) risk scenarios, achieving up to 15.7% reduction in harmful responses compared to baseline methods like SafetyPrompt. We further implement InvThink via supervised fine-tuning, and reinforcement learning across three LLM families. These results suggest that inverse reasoning provides a scalable and generalizable path toward safer, more capable language models. 

---
# VOGUE: Guiding Exploration with Visual Uncertainty Improves Multimodal Reasoning 

**Authors**: Rui Liu, Dian Yu, Tong Zheng, Runpeng Dai, Zongxia Li, Wenhao Yu, Zhenwen Liang, Linfeng Song, Haitao Mi, Pratap Tokekar, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01444)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) improves reasoning in large language models (LLMs) but struggles with exploration, an issue that still persists for multimodal LLMs (MLLMs). Current methods treat the visual input as a fixed, deterministic condition, overlooking a critical source of ambiguity and struggling to build policies robust to plausible visual variations. We introduce $\textbf{VOGUE (Visual Uncertainty Guided Exploration)}$, a novel method that shifts exploration from the output (text) to the input (visual) space. By treating the image as a stochastic context, VOGUE quantifies the policy's sensitivity to visual perturbations using the symmetric KL divergence between a "raw" and "noisy" branch, creating a direct signal for uncertainty-aware exploration. This signal shapes the learning objective via an uncertainty-proportional bonus, which, combined with a token-entropy bonus and an annealed sampling schedule, effectively balances exploration and exploitation. Implemented within GRPO on two model scales (Qwen2.5-VL-3B/7B), VOGUE boosts pass@1 accuracy by an average of 2.6% on three visual math benchmarks and 3.7% on three general-domain reasoning benchmarks, while simultaneously increasing pass@4 performance and mitigating the exploration decay commonly observed in RL fine-tuning. Our work shows that grounding exploration in the inherent uncertainty of visual inputs is an effective strategy for improving multimodal reasoning. 

---
# RLP: Reinforcement as a Pretraining Objective 

**Authors**: Ali Hatamizadeh, Syeda Nahida Akter, Shrimai Prabhumoye, Jan Kautz, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.01265)  

**Abstract**: The dominant paradigm for training large reasoning models starts with pre-training using next-token prediction loss on vast amounts of data. Reinforcement learning, while powerful in scaling reasoning, is introduced only as the very last phase of post-training, preceded by supervised fine-tuning. While dominant, is this an optimal way of training? In this paper, we present RLP, an information-driven reinforcement pretraining objective, that brings the core spirit of reinforcement learning -- exploration -- to the last phase of pretraining. The key idea is to treat chain-of-thought as an exploratory action, with rewards computed based on the information gain it provides for predicting future tokens. This training objective essentially encourages the model to think for itself before predicting what comes next, thus teaching an independent thinking behavior earlier in the pretraining. More concretely, the reward signal measures the increase in log-likelihood of the next token when conditioning on both context and a sampled reasoning chain, compared to conditioning on context alone. This approach yields a verifier-free dense reward signal, allowing for efficient training for the full document stream during pretraining. Specifically, RLP reframes reinforcement learning for reasoning as a pretraining objective on ordinary text, bridging the gap between next-token prediction and the emergence of useful chain-of-thought reasoning. Pretraining with RLP on Qwen3-1.7B-Base lifts the overall average across an eight-benchmark math-and-science suite by 19%. With identical post-training, the gains compound, with the largest improvements on reasoning-heavy tasks such as AIME25 and MMLU-Pro. Applying RLP to the hybrid Nemotron-Nano-12B-v2 increases the overall average from 42.81% to 61.32% and raises the average on scientific reasoning by 23%, demonstrating scalability across architectures and model sizes. 

---
# Learning a Dense Reasoning Reward Model from Expert Demonstration via Inverse Reinforcement Learning 

**Authors**: Claudio Fanconi, Nicolás Astorga, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2510.01857)  

**Abstract**: We reframe and operationalise adversarial inverse reinforcement learning (IRL) to large language model reasoning, learning a dense, token-level reward model for process supervision directly from expert demonstrations rather than imitating style via supervised fine-tuning. The learned reasoning reward serves two complementary roles: (i) it provides step-level feedback to optimise a reasoning policy during training; and (ii) it functions at inference as a critic to rerank sampled traces under fixed compute budgets. We demonstrate that our approach prioritises correctness over surface form, yielding scores that correlate with eventual answer validity and enabling interpretable localisation of errors within a trace. Empirically, on GSM8K with Llama3 and Qwen2.5 backbones, we demonstrate: (i) dense reasoning rewards can be used as a learning signal to elicit reasoning, and (ii) predictive performance is improved from reward-guided reranking (notably for Llama-based policies). By unifying training signals, inference-time selection, and token-level diagnostics into a single reasoning reward, this work suggests reusable process-level rewards with broad potential to enhance multi-step reasoning in language models. 

---
# VaPR -- Vision-language Preference alignment for Reasoning 

**Authors**: Rohan Wadhawan, Fabrice Y Harel-Canada, Zi-Yi Dou, Suhaila Shakiah, Robinson Piramuthu, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.01700)  

**Abstract**: Preference finetuning methods like Direct Preference Optimization (DPO) with AI-generated feedback have shown promise in aligning Large Vision-Language Models (LVLMs) with human preferences. However, existing techniques overlook the prevalence of noise in synthetic preference annotations in the form of stylistic and length biases. To this end, we introduce a hard-negative response generation framework based on LLM-guided response editing, that produces rejected responses with targeted errors, maintaining stylistic and length similarity to the accepted ones. Using this framework, we develop the VaPR dataset, comprising 30K high-quality samples, to finetune three LVLM families: LLaVA-V1.5, Qwen2VL & Qwen2.5VL (2B-13B sizes). Our VaPR models deliver significant performance improvements across ten benchmarks, achieving average gains of 6.5% (LLaVA), 4.0% (Qwen2VL), and 1.5% (Qwen2.5VL), with notable improvements on reasoning tasks. A scaling analysis shows that performance consistently improves with data size, with LLaVA models benefiting even at smaller scales. Moreover, VaPR reduces the tendency to answer "Yes" in binary questions - addressing a common failure mode in LVLMs like LLaVA. Lastly, we show that the framework generalizes to open-source LLMs as editors, with models trained on VaPR-OS achieving ~99% of the performance of models trained on \name, which is synthesized using GPT-4o. Our data, models, and code can be found on the project page this https URL 

---
# Step-Aware Policy Optimization for Reasoning in Diffusion Large Language Models 

**Authors**: Shaoan Xie, Lingjing Kong, Xiangchen Song, Xinshuai Dong, Guangyi Chen, Eric P.Xing, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01544)  

**Abstract**: Diffusion language models (dLLMs) offer a promising, non-autoregressive paradigm for text generation, yet training them for complex reasoning remains a key challenge. Current reinforcement learning approaches often rely on sparse, outcome-based rewards, which can reinforce flawed reasoning paths that lead to coincidentally correct answers. We argue that this stems from a fundamental mismatch with the natural structure of reasoning. We first propose a theoretical framework that formalizes complex problem solving as a hierarchical selection process, where an intractable global constraint is decomposed into a series of simpler, localized logical steps. This framework provides a principled foundation for algorithm design, including theoretical insights into the identifiability of this latent reasoning structure. Motivated by this theory, we identify unstructured refinement -- a failure mode where a model's iterative steps do not contribute meaningfully to the solution -- as a core deficiency in existing methods. We then introduce Step-Aware Policy Optimization (SAPO), a novel RL algorithm that aligns the dLLM's denoising process with the latent reasoning hierarchy. By using a process-based reward function that encourages incremental progress, SAPO guides the model to learn structured, coherent reasoning paths. Our empirical results show that this principled approach significantly improves performance on challenging reasoning benchmarks and enhances the interpretability of the generation process. 

---
# RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning 

**Authors**: Sicheng Feng, Kaiwen Tuo, Song Wang, Lingdong Kong, Jianke Zhu, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02240)  

**Abstract**: Fine-grained visual reasoning remains a core challenge for multimodal large language models (MLLMs). The recently introduced ReasonMap highlights this gap by showing that even advanced MLLMs struggle with spatial reasoning in structured and information-rich settings such as transit maps, a task of clear practical and scientific importance. However, standard reinforcement learning (RL) on such tasks is impeded by sparse rewards and unstable optimization. To address this, we first construct ReasonMap-Plus, an extended dataset that introduces dense reward signals through Visual Question Answering (VQA) tasks, enabling effective cold-start training of fine-grained visual understanding skills. Next, we propose RewardMap, a multi-stage RL framework designed to improve both visual understanding and reasoning capabilities of MLLMs. RewardMap incorporates two key designs. First, we introduce a difficulty-aware reward design that incorporates detail rewards, directly tackling the sparse rewards while providing richer supervision. Second, we propose a multi-stage RL scheme that bootstraps training from simple perception to complex reasoning tasks, offering a more effective cold-start strategy than conventional Supervised Fine-Tuning (SFT). Experiments on ReasonMap and ReasonMap-Plus demonstrate that each component of RewardMap contributes to consistent performance gains, while their combination yields the best results. Moreover, models trained with RewardMap achieve an average improvement of 3.47% across 6 benchmarks spanning spatial reasoning, fine-grained visual reasoning, and general tasks beyond transit maps, underscoring enhanced visual understanding and reasoning capabilities. 

---
# DiFFPO: Training Diffusion LLMs to Reason Fast and Furious via Reinforcement Learning 

**Authors**: Hanyang Zhao, Dawen Liang, Wenpin Tang, David Yao, Nathan Kallus  

**Link**: [PDF](https://arxiv.org/pdf/2510.02212)  

**Abstract**: We propose DiFFPO, Diffusion Fast and Furious Policy Optimization, a unified framework for training masked diffusion large language models (dLLMs) to reason not only better (furious), but also faster via reinforcement learning (RL). We first unify the existing baseline approach such as d1 by proposing to train surrogate policies via off-policy RL, whose likelihood is much more tractable as an approximation to the true dLLM policy. This naturally motivates a more accurate and informative two-stage likelihood approximation combined with importance sampling correction, which leads to generalized RL algorithms with better sample efficiency and superior task performance. Second, we propose a new direction of joint training efficient samplers/controllers of dLLMs policy. Via RL, we incentivize dLLMs' natural multi-token prediction capabilities by letting the model learn to adaptively allocate an inference threshold for each prompt. By jointly training the sampler, we yield better accuracies with lower number of function evaluations (NFEs) compared to training the model only, obtaining the best performance in improving the Pareto frontier of the inference-time compute of dLLMs. We showcase the effectiveness of our pipeline by training open source large diffusion language models over benchmark math and planning tasks. 

---
# Look Less, Reason More: Rollout-Guided Adaptive Pixel-Space Reasoning 

**Authors**: Xuchen Li, Xuzhao Li, Jiahui Gao, Renjie Pi, Shiyu Hu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01681)  

**Abstract**: Vision-Language Models (VLMs) excel at many multimodal tasks, yet they frequently struggle with tasks requiring precise understanding and handling of fine-grained visual elements. This is mainly due to information loss during image encoding or insufficient attention to critical regions. Recent work has shown promise by incorporating pixel-level visual information into the reasoning process, enabling VLMs to access high-resolution visual details during their thought process. However, this pixel-level information is often overused, leading to inefficiency and distraction from irrelevant visual details. To address these challenges, we propose the first framework for adaptive pixel reasoning that dynamically determines necessary pixel-level operations based on the input query. Specifically, we first apply operation-aware supervised fine-tuning to establish baseline competence in textual reasoning and visual operations, then design a novel rollout-guided reinforcement learning framework relying on feedback of the model's own responses, which enables the VLM to determine when pixel operations should be invoked based on query difficulty. Experiments on extensive multimodal reasoning benchmarks show that our model achieves superior performance while significantly reducing unnecessary visual operations. Impressively, our model achieves 73.4\% accuracy on HR-Bench 4K while maintaining a tool usage ratio of only 20.1\%, improving accuracy and simultaneously reducing tool usage by 66.5\% compared to the previous methods. 

---
# Asymmetric Proximal Policy Optimization: mini-critics boost LLM reasoning 

**Authors**: Jiashun Liu, Johan Obando-Ceron, Han Lu, Yancheng He, Weixun Wang, Wenbo Su, Bo Zheng, Pablo Samuel Castro, Aaron Courville, Ling Pan  

**Link**: [PDF](https://arxiv.org/pdf/2510.01656)  

**Abstract**: Most recent RL for LLMs (RL4LLM) methods avoid explicit critics, replacing them with average advantage baselines. This shift is largely pragmatic: conventional value functions are computationally expensive to train at LLM scale and often fail under sparse rewards and long reasoning horizons. We revisit this bottleneck from an architectural perspective and introduce Asymmetric Proximal Policy Optimization (AsyPPO), a simple and scalable framework that restores the critics role while remaining efficient in large-model settings. AsyPPO employs a set of lightweight mini-critics, each trained on disjoint prompt shards. This design encourages diversity while preserving calibration, reducing value-estimation bias. Beyond robust estimation, AsyPPO leverages inter-critic uncertainty to refine the policy update: (i) masking advantages in states where critics agree and gradients add little learning signal, and (ii) filtering high-divergence states from entropy regularization, suppressing spurious exploration. After training on open-source data with only 5,000 samples, AsyPPO consistently improves learning stability and performance across multiple benchmarks over strong baselines, such as GRPO, achieving performance gains of more than six percent on Qwen3-4b-Base and about three percent on Qwen3-8b-Base and Qwen3-14b-Base over classic PPO, without additional tricks. These results highlight the importance of architectural innovations for scalable, efficient algorithms. 

---
# Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization 

**Authors**: Kezhao Liu, Jason Klein Liu, Mingtao Chen, Yiming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01555)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) leverages a Kullback-Leibler (KL) divergence loss to stabilize training and prevent overfitting. However, in methods such as GRPO, its implementation may be guided by principles from numerical value estimation-a practice that overlooks the term's functional role as an optimization loss. To analyze this issue, we establish a unified framework that connects two seemingly distinct implementation styles: using the mathematical term $k_n$ as a detached coefficient for the policy's score function ('$k_n$ in reward') or as a direct loss function through which gradients are propagated ('$k_n$ as loss'). We show that the latter can always be analyzed via an equivalent gradient coefficient in the former, unifying the two perspectives. Through this framework, we prove that the conventional '$k_1$ in reward' (like in PPO) is the principled loss for Reverse KL (RKL) regularization. We further establish a key finding: under on-policy conditions, the '$k_2$ as loss' formulation is, in fact, gradient-equivalent to '$k_1$ in reward'. This equivalence, first proven in our work, identifies both as the theoretically sound implementations of the RKL objective. In contrast, we show that the recently adopted '$k_3$ as loss' (like in GRPO) is merely a first-order, biased approximation of the principled loss. Furthermore, we argue that common off-policy implementations of '$k_n$ as loss' methods are biased due to neglected importance sampling, and we propose a principled correction. Our findings provide a comprehensive, gradient-based rationale for choosing and correctly implementing KL regularization, paving the way for more robust and effective RLHF systems. 

---
