# EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning in Multimodal Large Language Models 

**Authors**: Yiyang Fang, Wenke Huang, Pei Fu, Yihao Yang, Kehua Su, Zhenbo Luo, Jian Luan, Mang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2602.23802)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown remarkable progress in visual reasoning and understanding tasks but still struggle to capture the complexity and subjectivity of human emotions. Existing approaches based on supervised fine-tuning often suffer from limited generalization and poor interpretability, while reinforcement learning methods such as Group Relative Policy Optimization fail to align with the intrinsic characteristics of emotional cognition. To address these challenges, we propose Reflective Reinforcement Learning for Emotional Reasoning (EMO-R3), a framework designed to enhance the emotional reasoning ability of MLLMs. Specifically, we introduce Structured Emotional Thinking to guide the model to perform step-by-step emotional reasoning in a structured and interpretable manner, and design a Reflective Emotional Reward that enables the model to re-evaluate its reasoning based on visual-text consistency and emotional coherence. Extensive experiments demonstrate that EMO-R3 significantly improves both the interpretability and emotional intelligence of MLLMs, achieving superior performance across multiple visual emotional understanding benchmarks. 

---
# RUMAD: Reinforcement-Unifying Multi-Agent Debate 

**Authors**: Chao Wang, Han Lin, Huaze Tang, Huijing Lin, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2602.23864)  

**Abstract**: Multi-agent debate (MAD) systems leverage collective intelligence to enhance reasoning capabilities, yet existing approaches struggle to simultaneously optimize accuracy, consensus formation, and computational efficiency. Static topology methods lack adaptability to task complexity variations, while external LLM-based coordination risks introducing privileged knowledge that compromises debate neutrality. This work presents RUMAD (Reinforcement-Unifying Multi-Agent Debate), a novel framework that formulates dynamic communication topology control in MAD as a reinforcement learning (RL) problem.
RUMAD employs a content-agnostic observation scheme that captures high-level debate dynamics avoiding access to raw agent reasoning content. RUMAD uses a multi-objective reward to model solution quality, cohesion and efficiency. A PPO-trained controller dynamically adjusts edge weights in the communication graph, while a dual-threshold mechanism enables fine-grained control over both agent activation and information visibility.
Experimental evaluation across MMLU, GSM8K, and GPQA benchmarks demonstrates that RUMAD achieves substantial efficiency gains, reducing token costs by over 80\%, while still improving reasoning accuracy compared to single LLM model and multiple MAD baselines. Notably, RUMAD trained exclusively on MMLU exhibits robust zero-shot generalization to out-of-domain (OOD) tasks, indicating that the learned communication strategies capture task-independent principles of effective multi-agent coordination. These results establish RUMAD as a efficient and robust approach for deploying multi-agent reasoning application with practical resource constraints. 

---
# CoME: Empowering Channel-of-Mobile-Experts with Informative Hybrid-Capabilities Reasoning 

**Authors**: Yuxuan Liu, Weikai Xu, Kun Huang, Changyu Chen, Jiankun Zhao, Pengzhi Gao, Wei Liu, Jian Luan, Shuo Shang, Bo Du, Ji-Rong Wen, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2602.24142)  

**Abstract**: Mobile Agents can autonomously execute user instructions, which requires hybrid-capabilities reasoning, including screen summary, subtask planning, action decision and action function. However, existing agents struggle to achieve both decoupled enhancement and balanced integration of these capabilities. To address these challenges, we propose Channel-of-Mobile-Experts (CoME), a novel agent architecture consisting of four distinct experts, each aligned with a specific reasoning stage, CoME activates the corresponding expert to generate output tokens in each reasoning stage via output-oriented activation. To empower CoME with hybrid-capabilities reasoning, we introduce a progressive training strategy: Expert-FT enables decoupling and enhancement of different experts' capability; Router-FT aligns expert activation with the different reasoning stage; CoT-FT facilitates seamless collaboration and balanced optimization across multiple capabilities. To mitigate error propagation in hybrid-capabilities reasoning, we propose InfoGain-Driven DPO (Info-DPO), which uses information gain to evaluate the contribution of each intermediate step, thereby guiding CoME toward more informative reasoning. Comprehensive experiments show that CoME outperforms dense mobile agents and MoE methods on both AITZ and AMEX datasets. 

---
# SafeGen-LLM: Enhancing Safety Generalization in Task Planning for Robotic Systems 

**Authors**: Jialiang Fan, Weizhe Xu, Mengyu Liu, Oleg Sokolsky, Insup Lee, Fangxin Kong  

**Link**: [PDF](https://arxiv.org/pdf/2602.24235)  

**Abstract**: Safety-critical task planning in robotic systems remains challenging: classical planners suffer from poor scalability, Reinforcement Learning (RL)-based methods generalize poorly, and base Large Language Models (LLMs) cannot guarantee safety. To address this gap, we propose safety-generalizable large language models, named SafeGen-LLM. SafeGen-LLM can not only enhance the safety satisfaction of task plans but also generalize well to novel safety properties in various domains. We first construct a multi-domain Planning Domain Definition Language 3 (PDDL3) benchmark with explicit safety constraints. Then, we introduce a two-stage post-training framework: Supervised Fine-Tuning (SFT) on a constraint-compliant planning dataset to learn planning syntax and semantics, and Group Relative Policy Optimization (GRPO) guided by fine-grained reward machines derived from formal verification to enforce safety alignment and by curriculum learning to better handle complex tasks. Extensive experiments show that SafeGen-LLM achieves strong safety generalization and outperforms frontier proprietary baselines across multi-domain planning tasks and multiple input formats (e.g., PDDLs and natural language). 

---
# DesignSense: A Human Preference Dataset and Reward Modeling Framework for Graphic Layout Generation 

**Authors**: Varun Gopal, Rishabh Jain, Aradhya Mathur, Nikitha SR, Sohan Patnaik, Sudhir Yarram, Mayur Hemani, Balaji Krishnamurthy, Mausoom Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2602.23438)  

**Abstract**: Graphic layouts serve as an important and engaging medium for visual communication across different channels. While recent layout generation models have demonstrated impressive capabilities, they frequently fail to align with nuanced human aesthetic judgment. Existing preference datasets and reward models trained on text-to-image generation do not generalize to layout evaluation, where the spatial arrangement of identical elements determines quality. To address this critical gap, we introduce DesignSense-10k, a large-scale dataset of 10,235 human-annotated preference pairs for graphic layout evaluation. We propose a five-stage curation pipeline that generates visually coherent layout transformations across diverse aspect ratios, using semantic grouping, layout prediction, filtering, clustering, and VLM-based refinement to produce high-quality comparison pairs. Human preferences are annotated using a 4-class scheme (left, right, both good, both bad) to capture subjective ambiguity. Leveraging this dataset, we train DesignSense, a vision-language model-based classifier that substantially outperforms existing open-source and proprietary models across comprehensive evaluation metrics (54.6% improvement in Macro F1 over the strongest proprietary baseline). Our analysis shows that frontier VLMs remain unreliable overall and fail catastrophically on the full four-class task, underscoring the need for specialized, preference-aware models. Beyond the dataset, our reward model DesignSense yields tangible downstream gains in layout generation. Using our judge during RL based training improves generator win rate by about 3%, while inference-time scaling, which involves generating multiple candidates and selecting the best one, provides a 3.6% improvement. These results highlight the practical impact of specialized, layout-aware preference modeling on real-world layout generation quality. 

---
# Learning to Generate Secure Code via Token-Level Rewards 

**Authors**: Jiazheng Quan, Xiaodong Li, Bin Wang, Guo An, Like Liu, Degen Huang, Lin Liu, Chengbin Hou  

**Link**: [PDF](https://arxiv.org/pdf/2602.23407)  

**Abstract**: Large language models (LLMs) have demonstrated strong capabilities in code generation, yet they remain prone to producing security vulnerabilities. Existing approaches commonly suffer from two key limitations: the scarcity of high-quality security data and coarse-grained reinforcement learning reward signals. To address these challenges, we propose Vul2Safe, a new secure code generation framework that leverages LLM self-reflection to construct high-confidence repair pairs from real-world vulnerabilities, and further generates diverse implicit prompts to build the PrimeVul+ dataset. Meanwhile, we introduce SRCode, a novel training framework that pioneers the use of token-level rewards in reinforcement learning for code security, which enables the model to continuously attend to and reinforce critical fine-grained security patterns during training. Compared with traditional instance-level reward schemes, our approach allows for more precise optimization of local security implementations. Extensive experiments show that PrimeVul+ and SRCode substantially reduce security vulnerabilities in generated code while improving overall code quality across multiple benchmarks. 

---
# LLM-Driven Multi-Turn Task-Oriented Dialogue Synthesis for Realistic Reasoning 

**Authors**: Yu Zhu, Kai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.23610)  

**Abstract**: The reasoning capability of large language models (LLMs), defined as their ability to analyze, infer, and make decisions based on input information, is essential for building intelligent task-oriented dialogue systems. However, existing benchmarks do not sufficiently reflect the complexity of real-world scenarios, which limits their effectiveness in evaluating and enhancing LLM reasoning in practical contexts. Many current reasoning datasets are overly simplistic and abstract, often disconnected from realistic task flows, domain constraints, and operational rules, making it difficult to effectively evaluate LLMs' logical reasoning ability. In addition, data contamination from pretraining corpora undermines the reliability of evaluation results, and traditional crowdsourcing methods for dataset construction are labor-intensive and difficult to scale. To address these challenges, we propose a LLM-driven framework for synthesizing multi-turn, task-oriented dialogues grounded in realistic reasoning scenarios, leveraging trilevel optimization to enhance dialogue quality. Our method generates dialogues grounded in authentic task scenarios, enriched with real-world information, and exhibiting strong contextual coherence. Corresponding reasoning tasks are carefully designed around these dialogues and iteratively refined to continuously improve the tasks' quality and challenge. The resulting dataset serves as a valuable benchmark for assessing and advancing the realistic logical reasoning capabilities of LLMs. Experimental results show that our synthetic data-based reasoning tasks introduce non-trivial reasoning challenges and provide meaningful support for improving the reasoning capabilities of LLMs. 

---
# Truncated Step-Level Sampling with Process Rewards for Retrieval-Augmented Reasoning 

**Authors**: Chris Samarinas, Haw-Shiuan Chang, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2602.23440)  

**Abstract**: Training large language models to reason with search engines via reinforcement learning is hindered by a fundamental credit assignment problem: existing methods such as Search-R1 provide only a sparse outcome reward after an entire multi-step trajectory, making it infeasible to attribute success or failure to individual reasoning and retrieval decisions. Process-reward methods like StepSearch alleviate this by introducing step-level supervision, but rely on heuristic rewards such as TF-IDF overlap with gold documents, and still sample k complete trajectories per example, retaining high gradient variance. We propose SLATE, a framework built on two complementary ideas: (1) truncated step-level sampling, which generates k trajectories that share a common prefix and differ only at the next step, and (2) dense LLM-as-judge rewards, which replace heuristic scoring with a capable LLM evaluator that assesses the quality of each reasoning step, search query, and answer, providing richer and more reliable supervision. We theoretically prove that under the same dense reward structure, truncated sampling reduces the variance of advantage estimates by up to a factor of T compared to full-trajectory sampling for T-step trajectories, yielding lower-variance, better-targeted policy gradients. Experiments on seven QA benchmarks confirm that SLATE consistently outperforms both sparse-reward and process-reward baselines, with the largest gains on harder multi-hop tasks and smaller models. 

---
# Learning to Reflect and Correct: Towards Better Decoding Trajectories for Large-Scale Generative Recommendation 

**Authors**: Haibo Xing, Hao Deng, Lingyu Mu, Jinxin Hu, Yu Zhang, Xiaoyi Zeng, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.23639)  

**Abstract**: Generative Recommendation (GR) has become a promising paradigm for large-scale recommendation systems. However, existing GR models typically perform single-pass decoding without explicit refinement, causing early deviations to accumulate and ultimately degrade recommendation quality. To tackle this problem, we propose GRC, which is, to our knowledge, the first structured reflection-correction framework for GR that extends standard decoding into a Generation-Reflection-Correction (GRC) process. Concretely, GRC introduces a supervised reflection-correction template that decomposes the decoding process into initial draft generation, multi-granular reflection, and reflection-guided correction, thereby enabling structured reflection and correction in the semantic token space. To further explore the enlarged refinement space introduced by the GRC process, we optimize the entire GRC trajectory with GRPO-based reinforcement learning, under a carefully designed reward function with token-level and trajectory-level signals. For efficient online serving, we propose an Entropy-Guided Reflection Scheduling (EGRS) strategy that dynamically allocates more correction budget to high-uncertainty decoding trajectories during beam search. Extensive experiments on real-world datasets show that GRC consistently outperforms six state-of-the-art baselines by up to 15.74%, and online A/B tests demonstrate its substantial practical value in large-scale industrial recommendation, delivering a 1.79% lift in advertising revenue with only modest latency overhead. 

---
