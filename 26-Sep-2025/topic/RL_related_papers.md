# RL Squeezes, SFT Expands: A Comparative Study of Reasoning LLMs 

**Authors**: Kohsei Matsutani, Shota Takashiro, Gouki Minegishi, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2509.21128)  

**Abstract**: Large language models (LLMs) are typically trained by reinforcement learning (RL) with verifiable rewards (RLVR) and supervised fine-tuning (SFT) on reasoning traces to improve their reasoning abilities. However, how these methods shape reasoning capabilities remains largely elusive. Going beyond an accuracy-based investigation of how these two components sculpt the reasoning process, this paper introduces a novel analysis framework that quantifies reasoning paths and captures their qualitative changes under each training process (with models of 1.5B, 7B, and 14B parameters on mathematical domains). Specifically, we investigate the reasoning process at two levels of granularity: the trajectory-level, which examines complete reasoning outputs, and the step-level, which analyzes reasoning graphs whose nodes correspond to individual reasoning steps. Notably, clustering of unique reasoning trajectories shows complementary effects: RL compresses incorrect trajectories, whereas SFT expands correct ones. Step-level analysis reveals that RL steepens (about 2.5 times), while SFT flattens (reduced to about one-third), the decay rates of node visitation frequency, degree, and betweenness centrality distributions in the reasoning graph. This indicates that RL concentrates reasoning functionality into a small subset of steps, while SFT homogenizes it across many steps. Furthermore, by evaluating the reasoning graph topologies from multiple perspectives, we delineate the shared and distinct characteristics of RL and SFT. Our work presents a novel reasoning path perspective that explains why the current best practice of two-stage training, with SFT followed by RL, is successful, and offers practical implications for data construction and more efficient learning approaches. 

---
# ToMPO: Training LLM Strategic Decision Making from a Multi-Agent Perspective 

**Authors**: Yiwen Zhang, Ziang Chen, Fanqi Kong, Yizhe Huang, Xue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2509.21134)  

**Abstract**: Large Language Models (LLMs) have been used to make decisions in complex scenarios, where they need models to think deeply, reason logically, and decide wisely. Many existing studies focus solely on multi-round conversations in social tasks or simulated environments, neglecting the various types of decisions and their interdependence. Current reinforcement learning methods struggle to consider the strategies of others during training. To address these issues, we first define a strategic decision-making problem that includes two types of decisions and their temporal dependencies. Furthermore, we propose **T**heory **o**f **M**ind **P**olicy **O**ptimization **(ToMPO)** algorithm to optimize the perception of other individual strategies and the game situation trends. Compared to the Group Relative Policy Optimization (GRPO) algorithm, ToMPO enhances the LLM's strategic decision-making mainly by: 1) generating rollouts based on reasoning the strategies of other individuals, 2) estimating advantages at both the graph-level and sample-level, and 3) balancing global and partial rewards. The ToMPO algorithm outperforms the GRPO method by 35% in terms of model output compliance and cooperative outcomes. Additionally, when compared to models with parameter sizes 100 times larger, it shows an 18% improvement. This demonstrates the effectiveness of the ToMPO algorithm in enhancing the model's strategic decision-making capabilities. 

---
# Expanding Reasoning Potential in Foundation Model by Learning Diverse Chains of Thought Patterns 

**Authors**: Xuemiao Zhang, Can Ren, Chengying Tu, Rongxiang Weng, Shuo Wang, Hongfei Yan, Jingang Wang, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.21124)  

**Abstract**: Recent progress in large reasoning models for challenging mathematical reasoning has been driven by reinforcement learning (RL). Incorporating long chain-of-thought (CoT) data during mid-training has also been shown to substantially improve reasoning depth. However, current approaches often utilize CoT data indiscriminately, leaving open the critical question of which data types most effectively enhance model reasoning capabilities. In this paper, we define the foundation model's reasoning potential for the first time as the inverse of the number of independent attempts required to correctly answer the question, which is strongly correlated with the final model performance. We then propose utilizing diverse data enriched with high-value reasoning patterns to expand the reasoning potential. Specifically, we abstract atomic reasoning patterns from CoT sequences, characterized by commonality and inductive capabilities, and use them to construct a core reference set enriched with valuable reasoning patterns. Furthermore, we propose a dual-granularity algorithm involving chains of reasoning patterns and token entropy, efficiently selecting high-value CoT data (CoTP) from the data pool that aligns with the core set, thereby training models to master reasoning effectively. Only 10B-token CoTP data enables the 85A6B Mixture-of-Experts (MoE) model to improve by 9.58% on the challenging AIME 2024 and 2025, and to raise the upper bound of downstream RL performance by 7.81%. 

---
# DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning 

**Authors**: Tianrun Xu, Haoda Jing, Ye Li, Yuquan Wei, Jun Feng, Guanyu Chen, Haichuan Gao, Tianren Zhang, Feng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.20912)  

**Abstract**: Recent advances in multimodal language models (MLLMs) have achieved remarkable progress in vision-language reasoning, especially with the emergence of "thinking with images," which integrates explicit visual steps into the reasoning process. While this paradigm strengthens image-based reasoning, a significant challenge remains: models may arrive at correct answers by relying on irrelevant or spurious regions, driven by prior knowledge or dataset biases. Even when the answer is correct, flawed reasoning indicates that the model has not truly understood the image, highlighting the critical importance of reasoning fidelity in multimodal tasks. To address this issue, we propose DeFacto, a counterfactual reasoning framework that jointly enforces accurate answering and faithful reasoning. A key component of our approach is the design of three complementary training paradigms: (i) positive, (ii) counterfactual, and (iii) random-masking. To enable these paradigms, we develop a pipeline that automatically localizes question-relevant evidence and constructs positive, counterfactual, and random variants, resulting in a dataset of about 100k images. Building on this framework, we train multimodal language models with GRPO-based reinforcement learning, where we design three complementary rewards to guide the model toward accurate answering and evidence-grounded reasoning. Experiments on diverse benchmarks demonstrate that DeFacto substantially improves both answer accuracy and reasoning faithfulness, establishing a stronger foundation for interpretable multimodal reasoning. The code is available on GitHub and the dataset is released on HuggingFace. 

---
# LogReasoner: Empowering LLMs with Expert-like Coarse-to-Fine Reasoning for Log Analysis Tasks 

**Authors**: Lipeng Ma, Yixuan Li, Weidong Yang, Mingjie Zhou, Xinyi Liu, Ben Fei, Shuhao Li, Xiaoyan Sun, Sihang Jiang, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20798)  

**Abstract**: Log analysis is crucial for monitoring system health and diagnosing failures in complex systems. Recent advances in large language models (LLMs) offer new opportunities for automated log analysis, leveraging their reasoning capabilities to perform tasks such as anomaly detection and failure prediction. However, general-purpose LLMs struggle to formulate structured reasoning workflows that align with expert cognition and deliver precise details of reasoning steps. To address these challenges, we propose LogReasoner, a coarse-to-fine reasoning enhancement framework designed to enable LLMs to reason log analysis tasks like experts. LogReasoner consists of two stages: (1) coarse-grained enhancement of expert thinking, where high-level expert thoughts are constructed from collected troubleshooting flowcharts and existing tasks to enable LLMs to formulate structured reasoning workflows and (2) fine-grained enhancement of specific steps, where we first fine-tune the LLM with task-specific stepwise solutions to enhance the LLM for instantiated reasoning, then employ the preference learning to calibrate the LLM's reasoning details from its mistakes, further strengthen the LLM's analytical granularity and correctness. We evaluate LogReasoner on four distinct log analysis tasks using open-source LLMs such as Qwen-2.5 and Llama-3. Experimental results show that LogReasoner significantly outperforms existing LLMs, achieving state-of-the-art performance and demonstrating its effectiveness in enhancing the reasoning capabilities of LLMs for log analysis. 

---
# GALAX: Graph-Augmented Language Model for Explainable Reinforcement-Guided Subgraph Reasoning in Precision Medicine 

**Authors**: Heming Zhang, Di Huang, Wenyu Li, Michael Province, Yixin Chen, Philip Payne, Fuhai Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.20935)  

**Abstract**: In precision medicine, quantitative multi-omic features, topological context, and textual biological knowledge play vital roles in identifying disease-critical signaling pathways and targets. Existing pipelines capture only part of these-numerical omics ignore topological context, text-centric LLMs lack quantitative grounded reasoning, and graph-only models underuse node semantics and the generalization of LLMs-limiting mechanistic interpretability. Although Process Reward Models (PRMs) aim to guide reasoning in LLMs, they remain limited by unreliable intermediate evaluation, and vulnerability to reward hacking with computational cost. These gaps motivate integrating quantitative multi-omic signals, topological structure with node annotations, and literature-scale text via LLMs, using subgraph reasoning as the principle bridge linking numeric evidence, topological knowledge and language context. Therefore, we propose GALAX (Graph Augmented LAnguage model with eXplainability), an innovative framework that integrates pretrained Graph Neural Networks (GNNs) into Large Language Models (LLMs) via reinforcement guided by a Graph Process Reward Model (GPRM), which generates disease-relevant subgraphs in a step-wise manner initiated by an LLM and iteratively evaluated by a pretrained GNN, enabling process-level supervision without explicit intermediate reasoning annotations. As an application, we also introduced Target-QA, a benchmark combining CRISPR-identified targets, multi-omic profiles, and biomedical graph knowledge across diverse cancer cell lines, which enables GNN pretraining for supervising step-wise graph construction and supports long-context reasoning over text-numeric graphs (TNGs), providing a scalable and biologically grounded framework for explainable, reinforcement-guided subgraph reasoning toward reliable and interpretable target and pathway discovery in precision medicine. 

---
# It's Not You, It's Clipping: A Soft Trust-Region via Probability Smoothing for LLM RL 

**Authors**: Madeleine Dwyer, Adam Sobey, Adriane Chapman  

**Link**: [PDF](https://arxiv.org/pdf/2509.21282)  

**Abstract**: Training large language models (LLMs) with reinforcement learning (RL) methods such as PPO and GRPO commonly relies on ratio clipping to stabilise updates. While effective at preventing instability, clipping discards information and introduces gradient discontinuities. We propose Probability Smoothing Policy Optimisation (PSPO), which smooths the current policy's probabilities toward the old (behaviour) policy before computing the importance ratio, analogous to label smoothing. Unlike clipping, PSPO preserves gradient signal, while interpolation toward the old policy creates a soft trust region that discourages large, destabilising updates, with formal guarantees.
We instantiate PSPO within GRPO (GR-PSPO) and fine-tune Qwen2.5-0.5B and Qwen2.5-1.5B on GSM8K, evaluating on GSM8K test and the cross-dataset generalisation on SVAMP, ASDiv, and MATH-500. Relative to unclipped GRPO (single iteration; no data reuse, ratio always = 1), GR-PSPO achieves similar performance but improves the reasoning leading to clearer and more concise responses which are more logical. Compared to clipped GRPO, GR-PSPO substantially improves performance both the 0.5B and 1.5B models, with a boost of over 20% on GSM8K (39.7% vs. 17.6% for 0.5B, 59.4% vs. 37.8% for 1.5B). 

---
# RLBFF: Binary Flexible Feedback to bridge between Human Feedback & Verifiable Rewards 

**Authors**: Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Ellie Evans, Daniel Egert, Hoo-Chang Shin, Felipe Soares, Yi Dong, Oleksii Kuchaiev  

**Link**: [PDF](https://arxiv.org/pdf/2509.21319)  

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) are the main RL paradigms used in LLM post-training, each offering distinct advantages. However, RLHF struggles with interpretability and reward hacking because it relies on human judgments that usually lack explicit criteria, whereas RLVR is limited in scope by its focus on correctness-based verifiers. We propose Reinforcement Learning with Binary Flexible Feedback (RLBFF), which combines the versatility of human-driven preferences with the precision of rule-based verification, enabling reward models to capture nuanced aspects of response quality beyond mere correctness. RLBFF extracts principles that can be answered in a binary fashion (e.g. accuracy of information: yes, or code readability: no) from natural language feedback. Such principles can then be used to ground Reward Model training as an entailment task (response satisfies or does not satisfy an arbitrary principle). We show that Reward Models trained in this manner can outperform Bradley-Terry models when matched for data and achieve top performance on RM-Bench (86.2%) and JudgeBench (81.4%, #1 on leaderboard as of September 24, 2025). Additionally, users can specify principles of interest at inference time to customize the focus of our reward models, in contrast to Bradley-Terry models. Finally, we present a fully open source recipe (including data) to align Qwen3-32B using RLBFF and our Reward Model, to match or exceed the performance of o3-mini and DeepSeek R1 on general alignment benchmarks of MT-Bench, WildBench, and Arena Hard v2 (at <5% of the inference cost). 

---
# Tree Search for LLM Agent Reinforcement Learning 

**Authors**: Yuxiang Ji, Ziyu Ma, Yong Wang, Guanhua Chen, Xiangxiang Chu, Liaoni Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21240)  

**Abstract**: Recent advances in reinforcement learning (RL) have significantly enhanced the agentic capabilities of large language models (LLMs). In long-term and multi-turn agent tasks, existing approaches driven solely by outcome rewards often suffer from the problem of sparse supervision. To address the challenge, we propose Tree-based Group Relative Policy Optimization (Tree-GRPO), a grouped agent RL method based on tree search, where each tree node represents the complete agent interaction step. By sharing common prefixes, the tree search sampling increases the number of rollouts achievable within a fixed budget of tokens or tool calls. Moreover, we find that the tree-structured trajectory naturally allows the construction of step-wise process supervised signals even using only the outcome reward. Based on this, Tree-GRPO estimates the grouped relative advantages both on intra-tree and inter-tree levels. Through theoretical analysis, we demonstrate that the objective of intra-tree level group relative policy optimization is equivalent to that of step-level direct preference learning. Experiments across 11 datasets and 3 types of QA tasks demonstrate the superiority of the proposed tree-based RL over the chain-based RL method. 

---
# GRPO is Secretly a Process Reward Model 

**Authors**: Michael Sullivan  

**Link**: [PDF](https://arxiv.org/pdf/2509.21154)  

**Abstract**: We prove theoretically that the GRPO RL algorithm induces a non-trivial process reward model (PRM), under certain assumptions regarding within-group overlap of token sequences across completions. We then show empirically that these assumptions are met under real-world conditions: GRPO does in fact induce a non-trivial PRM. Leveraging the framework of GRPO-as-a-PRM, we identify a flaw in the GRPO objective: non-uniformly distributed process steps hinder both exploration and exploitation (under different conditions). We propose a simple modification to the algorithm to mitigate this defect ($\lambda$-GRPO), and show that LLMs trained with $\lambda$-GRPO achieve higher validation accuracy and performance on downstream reasoning tasks$-$and reach peak performance more rapidly$-$than LLMs trained with standard GRPO. Our results call into question the advantage of costly, explicitly-defined PRMs for GRPO: we show that it is possible to instead leverage the hidden, built-in PRM structure within the vanilla GRPO algorithm to boost model performance with a negligible impact on training time and cost. 

---
# Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs 

**Authors**: Honglin Zhang, Qianyue Hao, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.21044)  

**Abstract**: Large language models (LLMs) acquire extensive prior knowledge through large-scale pretraining and can be further enhanced via supervised fine-tuning (SFT) or reinforcement learning (RL)-based post-training. A growing body of evidence has shown that RL fine-tuning improves the capability of LLMs beyond what SFT alone achieves. However, the underlying mechanisms why RL fine-tuning is able to enhance the capability of various LLMs with distinct intrinsic characteristics remain underexplored. In this study, we draw inspiration from prior work on edge attribution patching (EAP) to investigate the internal differences of LLMs before and after RL fine-tuning. Our analysis across multiple model families shows two robust effects of online RL post-training: (i) an overall increase in activation intensity, indicating that more internal pathways are engaged and their signals become stronger, and (ii) greater diversity in activation patterns, reflected by higher entropy and less concentrated edge distributions. These changes suggest that RL reshapes information flow to be both more redundant and more flexible, which may explain its advantage in generalization. Notably, models fine-tuned with Direct Preference Optimization (DPO) deviate from these trends, exhibiting substantially weaker or inconsistent internal changes compared to PPO- and GRPO-based training. Together, our findings provide a unified view of how RL fine-tuning systematically alters the internal circuitry of LLMs and highlight the methodological distinctions between online RL and preference-based approaches. Our code is open source at this https URL. 

---
# GeoRef: Referring Expressions in Geometry via Task Formulation, Synthetic Supervision, and Reinforced MLLM-based Solutions 

**Authors**: Bing Liu, Wenqiang Yv, Xuzheng Yang, Shichang Wang, Junzhuo Liu, Peng Wang, Guoqing Wang, Yang Yang, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21050)  

**Abstract**: AI-driven geometric problem solving is a complex vision-language task that requires accurate diagram interpretation, mathematical reasoning, and robust cross-modal grounding. A foundational yet underexplored capability for this task is the ability to identify and interpret geometric elements based on natural language queries. To address this, we introduce the task of Referring Expression Comprehension (REC) for geometric problems, which evaluates whether models can localize points, shapes, and spatial relations in diagrams in response to textual prompts. We present GeoRef, a benchmark dataset constructed from existing geometric problem corpora, featuring diverse, high-quality annotations and queries. Due to the lack of annotated data for this task, we generate a large-scale synthetic training dataset using a structured geometric formal language, enabling broad coverage of geometric concepts and facilitating model adaptation. We explore two fine-tuning approaches: Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO). Our results show that GRPO significantly outperforms SFT by better aligning model behavior with task-specific rewards. Furthermore, we propose a verify-and-regenerate mechanism that detects incorrect predictions and re-infers answers using contextual reasoning history, further boosting accuracy. Notably, even state-of-the-art Multimodal Large Language Models (MLLMs) struggle with this task, underscoring the necessity of explicitly evaluating and strengthening geometric grounding as a prerequisite for robust geometric problem solving. Moreover, models trained on GeoRef demonstrate measurable improvements on downstream geometric reasoning tasks, highlighting the broader value of REC as a foundation for multimodal mathematical understanding. 

---
# Unlocking Financial Insights: An advanced Multimodal Summarization with Multimodal Output Framework for Financial Advisory Videos 

**Authors**: Sarmistha Das, R E Zera Marveen Lyngkhoi, Sriparna Saha, Alka Maurya  

**Link**: [PDF](https://arxiv.org/pdf/2509.20961)  

**Abstract**: The dynamic propagation of social media has broadened the reach of financial advisory content through podcast videos, yet extracting insights from lengthy, multimodal segments (30-40 minutes) remains challenging. We introduce FASTER (Financial Advisory Summariser with Textual Embedded Relevant images), a modular framework that tackles three key challenges: (1) extracting modality-specific features, (2) producing optimized, concise summaries, and (3) aligning visual keyframes with associated textual points. FASTER employs BLIP for semantic visual descriptions, OCR for textual patterns, and Whisper-based transcription with Speaker diarization as BOS features. A modified Direct Preference Optimization (DPO)-based loss function, equipped with BOS-specific fact-checking, ensures precision, relevance, and factual consistency against the human-aligned summary. A ranker-based retrieval mechanism further aligns keyframes with summarized content, enhancing interpretability and cross-modal coherence. To acknowledge data resource scarcity, we introduce Fin-APT, a dataset comprising 470 publicly accessible financial advisory pep-talk videos for robust multimodal research. Comprehensive cross-domain experiments confirm FASTER's strong performance, robustness, and generalizability when compared to Large Language Models (LLMs) and Vision-Language Models (VLMs). By establishing a new standard for multimodal summarization, FASTER makes financial advisory content more accessible and actionable, thereby opening new avenues for research. The dataset and code are available at: this https URL 

---
# USB-Rec: An Effective Framework for Improving Conversational Recommendation Capability of Large Language Model 

**Authors**: Jianyu Wen, Jingyun Wang, Cilin Yan, Jiayin Cai, Xiaolong Jiang, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20381)  

**Abstract**: Recently, Large Language Models (LLMs) have been widely employed in Conversational Recommender Systems (CRSs). Unlike traditional language model approaches that focus on training, all existing LLMs-based approaches are mainly centered around how to leverage the summarization and analysis capabilities of LLMs while ignoring the issue of training. Therefore, in this work, we propose an integrated training-inference framework, User-Simulator-Based framework (USB-Rec), for improving the performance of LLMs in conversational recommendation at the model level. Firstly, we design a LLM-based Preference Optimization (PO) dataset construction strategy for RL training, which helps the LLMs understand the strategies and methods in conversational recommendation. Secondly, we propose a Self-Enhancement Strategy (SES) at the inference stage to further exploit the conversational recommendation potential obtained from RL training. Extensive experiments on various datasets demonstrate that our method consistently outperforms previous state-of-the-art methods. 

---
# SciReasoner: Laying the Scientific Reasoning Ground Across Disciplines 

**Authors**: Yizhou Wang, Chen Tang, Han Deng, Jiabei Xiao, Jiaqi Liu, Jianyu Wu, Jun Yao, Pengze Li, Encheng Su, Lintao Wang, Guohang Zhuang, Yuchen Ren, Ben Fei, Ming Hu, Xin Chen, Dongzhan Zhou, Junjun He, Xiangyu Yue, Zhenfei Yin, Jiamin Wu, Qihao Zheng, Yuhao Zhou, Huihui Xu, Chenglong Ma, Yan Lu, Wenlong Zhang, Chunfeng Song, Philip Torr, Shixiang Tang, Xinzhu Ma, Wanli Ouyang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2509.21320)  

**Abstract**: We present a scientific reasoning foundation model that aligns natural language with heterogeneous scientific representations. The model is pretrained on a 206B-token corpus spanning scientific text, pure sequences, and sequence-text pairs, then aligned via SFT on 40M instructions, annealed cold-start bootstrapping to elicit long-form chain-of-thought, and reinforcement learning with task-specific reward shaping, which instills deliberate scientific reasoning. It supports four capability families, covering up to 103 tasks across workflows: (i) faithful translation between text and scientific formats, (ii) text/knowledge extraction, (iii) property prediction, (iv) property classification, (v) unconditional and conditional sequence generation and design. Compared with specialist systems, our approach broadens instruction coverage, improves cross-domain generalization, and enhances fidelity. We detail data curation and training and show that cross-discipline learning strengthens transfer and downstream reliability. The model, instruct tuning datasets and the evaluation code are open-sourced at this https URL and this https URL. 

---
# CE-GPPO: Controlling Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning 

**Authors**: Zhenpeng Su, Leiyu Pan, Minxuan Lv, Yuntao Li, Wenping Hu, Fuzheng Zhang, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.20712)  

**Abstract**: Reinforcement learning (RL) has become a powerful paradigm for optimizing large language models (LLMs) to handle complex reasoning tasks. A core challenge in this process lies in managing policy entropy, which reflects the balance between exploration and exploitation during training. Existing methods, such as proximal policy optimization (PPO) and its variants, discard valuable gradient signals from low-probability tokens due to the clipping mechanism. We systematically analyze the entropy dynamics and reveal that these clipped tokens play a critical yet overlooked role in regulating entropy evolution. We propose \textbf{C}ontrolling \textbf{E}ntropy via \textbf{G}radient-\textbf{P}reserving \textbf{P}olicy \textbf{O}ptimization (CE-GPPO), a novel algorithm that reintroduces gradients from clipped tokens in native PPO in a gentle and bounded manner. By controlling the magnitude of gradients from tokens outside the clipping interval, CE-GPPO is able to achieve an exploration-exploitation trade-off. We provide theoretical justification and empirical evidence showing that CE-GPPO effectively mitigates entropy instability. Extensive experiments on mathematical reasoning benchmarks show that CE-GPPO consistently outperforms strong baselines across different model scales. 

---
# DELTA-Code: How Does RL Unlock and Transfer New Programming Algorithms in LLMs? 

**Authors**: Yiyou Sun, Yuhan Cao, Pohao Huang, Haoyue Bai, Hannaneh Hajishirzi, Nouha Dziri, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.21016)  

**Abstract**: It remains an open question whether LLMs can acquire or generalize genuinely new reasoning strategies, beyond the sharpened skills encoded in their parameters during pre-training or post-training. To attempt to answer this debate, we introduce DELTA-Code--Distributional Evaluation of Learnability and Transferrability in Algorithmic Coding, a controlled benchmark of synthetic coding problem families designed to probe two fundamental aspects: learnability -- can LLMs, through reinforcement learning (RL), solve problem families where pretrained models exhibit failure with large enough attempts (pass@K=0)? --and transferrability -- if learnability happens, can such skills transfer systematically to out-of-distribution (OOD) test sets? Unlike prior public coding datasets, DELTA isolates reasoning skills through templated problem generators and introduces fully OOD problem families that demand novel strategies rather than tool invocation or memorized patterns. Our experiments reveal a striking grokking phase transition: after an extended period with near-zero reward, RL-trained models abruptly climb to near-perfect accuracy. To enable learnability on previously unsolvable problem families, we explore key training ingredients such as staged warm-up with dense rewards, experience replay, curriculum training, and verification-in-the-loop. Beyond learnability, we use DELTA to evaluate transferability or generalization along exploratory, compositional, and transformative axes, as well as cross-family transfer. Results show solid gains within families and for recomposed skills, but persistent weaknesses in transformative cases. DELTA thus offers a clean testbed for probing the limits of RL-driven reasoning and for understanding how models can move beyond existing priors to acquire new algorithmic skills. 

---
