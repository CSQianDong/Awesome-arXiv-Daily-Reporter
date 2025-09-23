# Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models 

**Authors**: Jun Ling, Yao Qi, Tao Huang, Shibo Zhou, Yanqin Huang, Jiang Yang, Ziqi Song, Ying Zhou, Yang Yang, Heng Tao Shen, Peng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17589)  

**Abstract**: In this work, we address the task of table image to LaTeX code generation, with the goal of automating the reconstruction of high-quality, publication-ready tables from visual inputs. A central challenge of this task lies in accurately handling complex tables -- those with large sizes, deeply nested structures, and semantically rich or irregular cell content -- where existing methods often fail. We begin with a comprehensive analysis, identifying key challenges and highlighting the limitations of current evaluation protocols. To overcome these issues, we propose a reinforced multimodal large language model (MLLM) framework, where a pre-trained MLLM is fine-tuned on a large-scale table-to-LaTeX dataset. To further improve generation quality, we introduce a dual-reward reinforcement learning strategy based on Group Relative Policy Optimization (GRPO). Unlike standard approaches that optimize purely over text outputs, our method incorporates both a structure-level reward on LaTeX code and a visual fidelity reward computed from rendered outputs, enabling direct optimization of the visual output quality. We adopt a hybrid evaluation protocol combining TEDS-Structure and CW-SSIM, and show that our method achieves state-of-the-art performance, particularly on structurally complex tables, demonstrating the effectiveness and robustness of our approach. 

---
# Reasoning Core: A Scalable RL Environment for LLM Symbolic Reasoning 

**Authors**: Valentin Lacombe, Valentin Quesnel, Damien Sileo  

**Link**: [PDF](https://arxiv.org/pdf/2509.18083)  

**Abstract**: We introduce Reasoning Core, a new scalable environment for Reinforcement Learning with Verifiable Rewards (RLVR), designed to advance foundational symbolic reasoning in Large Language Models (LLMs). Unlike existing benchmarks that focus on games or isolated puzzles, Reasoning Core procedurally generates problems across core formal domains, including PDDL planning, first-order logic, context-free grammar parsing, causal reasoning, and system equation solving. The environment is built on key design principles of high-generality problem distributions, verification via external tools, and continuous difficulty control, which together provide a virtually infinite supply of novel training instances. Initial zero-shot evaluations with frontier LLMs confirm the difficulty of Reasoning Core's tasks, positioning it as a promising resource to improve the reasoning capabilities of future models. 

---
# Correlation or Causation: Analyzing the Causal Structures of LLM and LRM Reasoning Process 

**Authors**: Zhizhang FU, Guangsheng Bao, Hongbo Zhang, Chenkai Hu, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17380)  

**Abstract**: LLMs suffer from critical reasoning issues such as unfaithfulness, bias, and inconsistency, since they lack robust causal underpinnings and may rely on superficial correlations rather than genuine understanding. Successive LRMs have emerged as a promising alternative, leveraging advanced training techniques such as reinforcement learning (RL) and distillation to improve task accuracy. However, the impact of these training methods on causality remains largely unexplored. In this study, we conduct a systematic causal analysis on LLMs and LRMs, examining structural causal models (SCMs) of four key variables: problem instruction (Z), thinking process (T), reasoning steps (X), and answer (Y). Our findings reveal that RLVR-trained LRMs exhibit enhanced causal reasoning capabilities, aligning more closely with ideal causal structures, while LLMs and distilled LRMs fail to address causality-related deficiencies. Our further investigation indicates that RLVR reduces spurious correlations and strengthens genuine causal patterns, thereby mitigating unfaithfulness and bias. In addition, our inspection on the dynamics of the RLVR training process observes a high correlation between reduced spurious features and improved causal structures, where the causal relationships consistently improve in the training process. This study contributes to the understanding of causality in reasoning models, highlights the critical role of RLVR in enhancing causal reasoning, and provides insights for designing future AI systems with stronger causal foundations. We release our code and data at this https URL. 

---
# Medical AI Consensus: A Multi-Agent Framework for Radiology Report Generation and Evaluation 

**Authors**: Ahmed T. Elboardy, Ghada Khoriba, Essam A. Rashed  

**Link**: [PDF](https://arxiv.org/pdf/2509.17353)  

**Abstract**: Automating radiology report generation poses a dual challenge: building clinically reliable systems and designing rigorous evaluation protocols. We introduce a multi-agent reinforcement learning framework that serves as both a benchmark and evaluation environment for multimodal clinical reasoning in the radiology ecosystem. The proposed framework integrates large language models (LLMs) and large vision models (LVMs) within a modular architecture composed of ten specialized agents responsible for image analysis, feature extraction, report generation, review, and evaluation. This design enables fine-grained assessment at both the agent level (e.g., detection and segmentation accuracy) and the consensus level (e.g., report quality and clinical relevance). We demonstrate an implementation using chatGPT-4o on public radiology datasets, where LLMs act as evaluators alongside medical radiologist feedback. By aligning evaluation protocols with the LLM development lifecycle, including pretraining, finetuning, alignment, and deployment, the proposed benchmark establishes a path toward trustworthy deviance-based radiology report generation. 

---
# LLMs as Layout Designers: A Spatial Reasoning Perspective 

**Authors**: Sha Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.16891)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive reasoning and planning abilities in textual domains and can effectively follow instructions for complex tasks, their capacity for spatial understanding and reasoning remains limited. Such capabilities, however, are critical for applications like content-aware graphic layout design, which demands precise placement, alignment, and structural organization of multiple elements within constrained visual spaces. To address this gap, we propose LaySPA, a reinforcement learning-based framework that augments LLM agents with explicit spatial reasoning capabilities. LaySPA leverages hybrid reward signals that capture geometric validity, structural fidelity, and visual quality, enabling agents to model inter-element relationships, navigate the canvas, and optimize spatial arrangements. Through iterative self-exploration and adaptive policy optimization, LaySPA produces both interpretable reasoning traces and structured layouts. Experimental results demonstrate that LaySPA generates structurally sound and visually appealing layouts, outperforming larger general-purpose LLMs and achieving results on par with state-of-the-art specialized layout models. 

---
# Large Language Models as End-to-end Combinatorial Optimization Solvers 

**Authors**: Xia Jiang, Yaoxin Wu, Minshuo Li, Zhiguang Cao, Yingqian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16865)  

**Abstract**: Combinatorial optimization (CO) problems, central to decision-making scenarios like logistics and manufacturing, are traditionally solved using problem-specific algorithms requiring significant domain expertise. While large language models (LLMs) have shown promise in automating CO problem solving, existing approaches rely on intermediate steps such as code generation or solver invocation, limiting their generality and accessibility. This paper introduces a novel framework that empowers LLMs to serve as end-to-end CO solvers by directly mapping natural language problem descriptions to solutions. We propose a two-stage training strategy: supervised fine-tuning (SFT) imparts LLMs with solution generation patterns from domain-specific solvers, while a feasibility-and-optimality-aware reinforcement learning (FOARL) process explicitly mitigates constraint violations and refines solution quality. Evaluation across seven NP-hard CO problems shows that our method achieves a high feasibility rate and reduces the average optimality gap to 1.03-8.20% by tuning a 7B-parameter LLM, surpassing both general-purpose LLMs (e.g., GPT-4o), reasoning models (e.g., DeepSeek-R1), and domain-specific heuristics. Our method establishes a unified language-based pipeline for CO without extensive code execution or manual architectural adjustments for different problems, offering a general and language-driven alternative to traditional solver design while maintaining relative feasibility guarantees. 

---
# GPO: Learning from Critical Steps to Improve LLM Reasoning 

**Authors**: Jiahao Yu, Zelei Cheng, Xian Wu, Xinyu Xing  

**Link**: [PDF](https://arxiv.org/pdf/2509.16456)  

**Abstract**: Large language models (LLMs) are increasingly used in various domains, showing impressive potential on different tasks. Recently, reasoning LLMs have been proposed to improve the \textit{reasoning} or \textit{thinking} capabilities of LLMs to solve complex problems. Despite the promising results of reasoning LLMs, enhancing the multi-step reasoning capabilities of LLMs still remains a significant challenge. While existing optimization methods have advanced the LLM reasoning capabilities, they often treat reasoning trajectories as a whole, without considering the underlying critical steps within the trajectory. In this paper, we introduce \textbf{G}uided \textbf{P}ivotal \textbf{O}ptimization (GPO), a novel fine-tuning strategy that dives into the reasoning process to enable more effective improvements. GPO first identifies the `critical step' within a reasoning trajectory - a point that the model must carefully proceed to succeed at the problem. We locate the critical step by estimating the advantage function. GPO then resets the policy to the critical step, samples the new rollout and prioritizes the learning process on those rollouts. This focus allows the model to learn more effectively from pivotal moments within the reasoning process to improve the reasoning performance. We demonstrate that GPO is a general strategy that can be integrated with various optimization methods to improve reasoning performance. Besides theoretical analysis, our experiments across challenging reasoning benchmarks show that GPO can consistently and significantly enhance the performance of existing optimization methods, showcasing its effectiveness and generalizability in improving LLM reasoning by concentrating on pivotal moments within the generation process. 

---
# Sycophancy Mitigation Through Reinforcement Learning with Uncertainty-Aware Adaptive Reasoning Trajectories 

**Authors**: Mohammad Beigi, Ying Shen, Parshin Shojaee, Qifan Wang, Zichao Wang, Chandan Reddy, Ming Jin, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16742)  

**Abstract**: Despite the remarkable capabilities of large language models, current training paradigms inadvertently foster \textit{sycophancy}, i.e., the tendency of a model to agree with or reinforce user-provided information even when it's factually incorrect. To address this challenge, we introduce \textbf{SMART} (Sycophancy Mitigation through Adaptive Reasoning Trajectories), which reframes sycophancy as a \textit{reasoning optimization problem} rather than an output alignment issue. SMART is a two-stage framework comprising: (1) Uncertainty-Aware Adaptive Monte Carlo Tree Search (UA-MCTS), which dynamically adjusts model exploration based on state-level uncertainty to collect high-quality, diverse reasoning trajectories alongside both stepwise progress and final outcome rewards; and (2) progress-based reinforcement learning, which fine-tunes the model using the collected trajectories and reward signals to reinforce effective reasoning patterns. Through extensive experiments, we show that SMART significantly reduces sycophantic behavior while preserving strong performance on out-of-distribution inputs and maintaining general capabilities. These results underscore the importance of optimizing internal reasoning mechanisms to build more truthful and aligned AI assistants. 

---
# VORTEX: Aligning Task Utility and Human Preferences through LLM-Guided Reward Shaping 

**Authors**: Guojun Xiong, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2509.16399)  

**Abstract**: In social impact optimization, AI decision systems often rely on solvers that optimize well-calibrated mathematical objectives. However, these solvers cannot directly accommodate evolving human preferences, typically expressed in natural language rather than formal constraints. Recent approaches address this by using large language models (LLMs) to generate new reward functions from preference descriptions. While flexible, they risk sacrificing the system's core utility guarantees. In this paper, we propose \texttt{VORTEX}, a language-guided reward shaping framework that preserves established optimization goals while adaptively incorporating human feedback. By formalizing the problem as multi-objective optimization, we use LLMs to iteratively generate shaping rewards based on verbal reinforcement and text-gradient prompt updates. This allows stakeholders to steer decision behavior via natural language without modifying solvers or specifying trade-off weights. We provide theoretical guarantees that \texttt{VORTEX} converges to Pareto-optimal trade-offs between utility and preference satisfaction. Empirical results in real-world allocation tasks demonstrate that \texttt{VORTEX} outperforms baselines in satisfying human-aligned coverage goals while maintaining high task performance. This work introduces a practical and theoretically grounded paradigm for human-AI collaborative optimization guided by natural language. 

---
# One Agent to Serve All: a Lite-Adaptive Stylized AI Assistant for Millions of Multi-Style Official Accounts 

**Authors**: Xingyu Fan, Feifei Li, Wenhui Que, Hailong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.17788)  

**Abstract**: Conversational agents deployed in industrial-scale official account platforms must generate responses that are both contextually grounded and stylistically aligned-requirements that existing methods struggle to meet. Chain-of-thought (CoT) prompting induces significant latency due to multi-turn reasoning; per-account fine-tuning is computationally prohibitive; and long prompt-based methods degrade the model's ability to grasp injected context and style. In this paper, we propose WeStar, a lite-adaptive framework for stylized contextual question answering that scales to millions of official accounts. WeStar combines context-grounded generation via RAG with style-aware generation using Parametric RAG (PRAG), where LoRA modules are dynamically activated per style cluster. Our contributions are fourfold: (1) We introduce WeStar, a unified framework capable of serving large volumes of official accounts with minimal overhead. (2) We propose a multi-dimensional, cluster-based parameter sharing scheme that enables compact style representation while preserving stylistic diversity. (3) We develop a style-enhanced Direct Preference Optimization (SeDPO) method to optimize each style cluster's parameters for improved generation quality. (4) Experiments on a large-scale industrial dataset validate the effectiveness and efficiency of WeStar, underscoring its pracitical value in real-world deployment. 

---
# LifeAlign: Lifelong Alignment for Large Language Models with Memory-Augmented Focalized Preference Optimization 

**Authors**: Junsong Li, Jie Zhou, Bihao Zhan, Yutao Yang, Qianjun Pan, Shilian Chen, Tianyu Huai, Xin Li, Qin Chen, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2509.17183)  

**Abstract**: Alignment plays a crucial role in Large Language Models (LLMs) in aligning with human preferences on a specific task/domain. Traditional alignment methods suffer from catastrophic forgetting, where models lose previously acquired knowledge when adapting to new preferences or domains. We introduce LifeAlign, a novel framework for lifelong alignment that enables LLMs to maintain consistent human preference alignment across sequential learning tasks without forgetting previously learned knowledge. Our approach consists of two key innovations. First, we propose a focalized preference optimization strategy that aligns LLMs with new preferences while preventing the erosion of knowledge acquired from previous tasks. Second, we develop a short-to-long memory consolidation mechanism that merges denoised short-term preference representations into stable long-term memory using intrinsic dimensionality reduction, enabling efficient storage and retrieval of alignment patterns across diverse domains. We evaluate LifeAlign across multiple sequential alignment tasks spanning different domains and preference types. Experimental results demonstrate that our method achieves superior performance in maintaining both preference alignment quality and knowledge retention compared to existing lifelong learning approaches. The codes and datasets will be released on GitHub. 

---
# Zero-Shot Human Mobility Forecasting via Large Language Model with Hierarchical Reasoning 

**Authors**: Wenyao Li, Ran Zhang, Pengyang Wang, Yuanchun Zhou, Pengfei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16578)  

**Abstract**: Human mobility forecasting is important for applications such as transportation planning, urban management, and personalized recommendations. However, existing methods often fail to generalize to unseen users or locations and struggle to capture dynamic intent due to limited labeled data and the complexity of mobility patterns. We propose ZHMF, a framework for zero-shot human mobility forecasting that combines a semantic enhanced retrieval and reflection mechanism with a hierarchical language model based reasoning system. The task is reformulated as a natural language question answering paradigm. Leveraging LLMs semantic understanding of user histories and context, our approach handles previously unseen prediction scenarios. We further introduce a hierarchical reflection mechanism for iterative reasoning and refinement by decomposing forecasting into an activity level planner and a location level selector, enabling collaborative modeling of long term user intentions and short term contextual preferences. Experiments on standard human mobility datasets show that our approach outperforms existing models. Ablation studies reveal the contribution of each module, and case studies illustrate how the method captures user intentions and adapts to diverse contextual scenarios. 

---
# Can GRPO Boost Complex Multimodal Table Understanding? 

**Authors**: Xiaoqiang Kang, Shengen Wu, Zimu Wang, Yilin Liu, Xiaobo Jin, Kaizhu Huang, Wei Wang, Yutao Yue, Xiaowei Huang, Qiufeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16889)  

**Abstract**: Existing table understanding methods face challenges due to complex table structures and intricate logical reasoning. While supervised finetuning (SFT) dominates existing research, reinforcement learning (RL), such as Group Relative Policy Optimization (GRPO), has shown promise but struggled with low initial policy accuracy and coarse rewards in tabular contexts. In this paper, we introduce Table-R1, a three-stage RL framework that enhances multimodal table understanding through: (1) Warm-up that prompts initial perception and reasoning capabilities, (2) Perception Alignment GRPO (PA-GRPO), which employs continuous Tree-Edit-Distance Similarity (TEDS) rewards for recognizing table structures and contents, and (3) Hint-Completion GRPO (HC-GRPO), which utilizes fine-grained rewards of residual steps based on the hint-guided question. Extensive experiments demonstrate that Table-R1 can boost the model's table reasoning performance obviously on both held-in and held-out datasets, outperforming SFT and GRPO largely. Notably, Qwen2-VL-7B with Table-R1 surpasses larger specific table understanding models (e.g., Table-LLaVA 13B), even achieving comparable performance to the closed-source model GPT-4o on held-in datasets, demonstrating the efficacy of each stage of Table-R1 in overcoming initialization bottlenecks and reward sparsity, thereby advancing robust multimodal table understanding. 

---
# Preference Distillation via Value based Reinforcement Learning 

**Authors**: Minchan Kwon, Junwon Ko, Kangil Kim, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.16965)  

**Abstract**: Direct Preference Optimization (DPO) is a powerful paradigm to align language models with human preferences using pairwise comparisons. However, its binary win-or-loss supervision often proves insufficient for training small models with limited capacity. Prior works attempt to distill information from large teacher models using behavior cloning or KL divergence. These methods often focus on mimicking current behavior and overlook distilling reward modeling. To address this issue, we propose \textit{Teacher Value-based Knowledge Distillation} (TVKD), which introduces an auxiliary reward from the value function of the teacher model to provide a soft guide. This auxiliary reward is formulated to satisfy potential-based reward shaping, ensuring that the global reward structure and optimal policy of DPO are preserved. TVKD can be integrated into the standard DPO training framework and does not require additional rollouts. Our experimental results show that TVKD consistently improves performance across various benchmarks and model sizes. 

---
# Reinforcement Learning Meets Large Language Models: A Survey of Advancements and Applications Across the LLM Lifecycle 

**Authors**: Keliang Liu, Dingkang Yang, Ziyun Qian, Weijie Yin, Yuchi Wang, Hongsheng Li, Jun Liu, Peng Zhai, Yang Liu, Lihua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16679)  

**Abstract**: In recent years, training methods centered on Reinforcement Learning (RL) have markedly enhanced the reasoning and alignment performance of Large Language Models (LLMs), particularly in understanding human intents, following user instructions, and bolstering inferential strength. Although existing surveys offer overviews of RL augmented LLMs, their scope is often limited, failing to provide a comprehensive summary of how RL operates across the full lifecycle of LLMs. We systematically review the theoretical and practical advancements whereby RL empowers LLMs, especially Reinforcement Learning with Verifiable Rewards (RLVR). First, we briefly introduce the basic theory of RL. Second, we thoroughly detail application strategies for RL across various phases of the LLM lifecycle, including pre-training, alignment fine-tuning, and reinforced reasoning. In particular, we emphasize that RL methods in the reinforced reasoning phase serve as a pivotal driving force for advancing model reasoning to its limits. Next, we collate existing datasets and evaluation benchmarks currently used for RL fine-tuning, spanning human-annotated datasets, AI-assisted preference data, and program-verification-style corpora. Subsequently, we review the mainstream open-source tools and training frameworks available, providing clear practical references for subsequent research. Finally, we analyse the future challenges and trends in the field of RL-enhanced LLMs. This survey aims to present researchers and practitioners with the latest developments and frontier trends at the intersection of RL and LLMs, with the goal of fostering the evolution of LLMs that are more intelligent, generalizable, and secure. 

---
# ConfClip: Confidence-Weighted and Clipped Reward for Reinforcement Learning in LLMs 

**Authors**: Bonan Zhang, Zhongqi Chen, Bowen Song, Qinya Li, Fan Wu, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.17730)  

**Abstract**: Reinforcement learning (RL) has become a standard paradigm for refining large language models (LLMs) beyond pre-training and instruction tuning. A prominent line of work is RL with verifiable rewards (RLVR), which leverages automatically verifiable outcomes (e.g., correctness or executability) to generate reward signals. While efficient, this framework faces two key limitations: First, its binary feedback is too sparse to capture the quality of the reasoning process. Second, its coarse-grained rewards potentially lead to vanishing gradients. Inspired by observations from human learning, we introduce a RL technique that integrates verifiable outcomes with the model's own confidence estimates. This joint design enriches the reward signal, providing finer-grained feedback and implicitly supervising the reasoning process. Experimental results demonstrate that our proposed method enhances RL performance across multiple datasets and reduces token consumption during inference, while incurring negligible additional training cost. Moreover, it can be used as a plug-in module to enhance other state-of-the-art RL methods. 

---
# Towards Universal Debiasing for Language Models-based Tabular Data Generation 

**Authors**: Tianchun Li, Tianci Liu, Xingchen Wang, Rongzhe Wei, Pan Li, Lu Su, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.16475)  

**Abstract**: Large language models (LLMs) have achieved promising results in tabular data generation. However, inherent historical biases in tabular datasets often cause LLMs to exacerbate fairness issues, particularly when multiple advantaged and protected features are involved. In this work, we introduce a universal debiasing framework that minimizes group-level dependencies by simultaneously reducing the mutual information between advantaged and protected attributes. By leveraging the autoregressive structure and analytic sampling distributions of LLM-based tabular data generators, our approach efficiently computes mutual information, reducing the need for cumbersome numerical estimations. Building on this foundation, we propose two complementary methods: a direct preference optimization (DPO)-based strategy, namely UDF-DPO, that integrates seamlessly with existing models, and a targeted debiasing technique, namely UDF-MIX, that achieves debiasing without tuning the parameters of LLMs. Extensive experiments demonstrate that our framework effectively balances fairness and utility, offering a scalable and practical solution for debiasing in high-stakes applications. 

---
