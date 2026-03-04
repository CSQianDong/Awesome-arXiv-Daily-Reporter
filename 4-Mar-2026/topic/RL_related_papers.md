# PrivMedChat: End-to-End Differentially Private RLHF for Medical Dialogue Systems 

**Authors**: Sudip Bhujel  

**Link**: [PDF](https://arxiv.org/pdf/2603.03054)  

**Abstract**: Large language models are increasingly used for patient-facing medical assistance and clinical decision support, but adapting them to clinical dialogue often requires supervision derived from doctor-patient conversations that may contain sensitive information. Conventional supervised fine-tuning and reinforcement learning from human feedback (RLHF) can amplify memorization risks, enabling empirical membership inference and extraction of rare training-set content. We present PrivMedChat, an end-to-end framework for differentially private RLHF (DP-RLHF) for medical dialogue. Our design enforces differential privacy at every training stage that directly accesses dialogue-derived supervision: (i) Differential Private Stochastic Gradient Descent (DP-SGD) for medical SFT and (ii) DP-SGD for reward model learning from preference pairs. To limit additional privacy expenditure during alignment, we apply DP-SGD to the PPO actor and critic when operating on dialogue-derived prompts, while the reward model remains fixed after DP training.
We also introduce an annotation-free preference construction strategy that pairs physician responses with filtered non-expert generations to produce scalable preference data without clinician labeling. Experiments on medical dialogue benchmarks show that PrivMedChat at $\varepsilon=7$ achieves the highest ROUGE-L of 0.156 among all DP models, reduces clinical hallucinations to 1.4% and harmful advice to 0.4%, and obtains the highest overall score of 2.86 in a 3-model LLM-jury evaluation, while producing membership-inference signals that are near chance (AUC 0.510-0.555). We open-source our code at this https URL. 

---
# Learning When to Act or Refuse: Guarding Agentic Reasoning Models for Safe Multi-Step Tool Use 

**Authors**: Aradhye Agarwal, Gurdit Siyan, Yash Pandya, Joykirat Singh, Akshay Nambi, Ahmed Awadallah  

**Link**: [PDF](https://arxiv.org/pdf/2603.03205)  

**Abstract**: Agentic language models operate in a fundamentally different safety regime than chat models: they must plan, call tools, and execute long-horizon actions where a single misstep, such as accessing files or entering credentials, can cause irreversible harm. Existing alignment methods, largely optimized for static generation and task completion, break down in these settings due to sequential decision-making, adversarial tool feedback, and overconfident intermediate reasoning. We introduce MOSAIC, a post-training framework that aligns agents for safe multi-step tool use by making safety decisions explicit and learnable. MOSAIC structures inference as a plan, check, then act or refuse loop, with explicit safety reasoning and refusal as first-class actions. To train without trajectory-level labels, we use preference-based reinforcement learning with pairwise trajectory comparisons, which captures safety distinctions often missed by scalar rewards. We evaluate MOSAIC zero-shot across three model families, Qwen2.5-7B, Qwen3-4B-Thinking, and Phi-4, and across out-of-distribution benchmarks spanning harmful tasks, prompt injection, benign tool use, and cross-domain privacy leakage. MOSAIC reduces harmful behavior by up to 50%, increases harmful-task refusal by over 20% on injection attacks, cuts privacy leakage, and preserves or improves benign task performance, demonstrating robust generalization across models, domains, and agentic settings. 

---
# MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization 

**Authors**: Ashutosh Chaubey, Jiacheng Pang, Mohammad Soleymani  

**Link**: [PDF](https://arxiv.org/pdf/2603.03192)  

**Abstract**: Omni-modal large language models (omni LLMs) have recently achieved strong performance across audiovisual understanding tasks, yet they remain highly susceptible to cross-modal hallucinations arising from spurious correlations and dominant language priors. In this work, we propose Modality-Decoupled Direct Preference Optimization (MoD-DPO), a simple and effective framework for improving modality grounding in omni LLMs. MoD-DPO introduces modality-aware regularization terms that explicitly enforce invariance to corruptions in irrelevant modalities and sensitivity to perturbations in relevant modalities, thereby reducing unintended cross-modal interactions. To further mitigate over-reliance on textual priors, we incorporate a language-prior debiasing penalty that discourages hallucination-prone text-only responses. Extensive experiments across multiple audiovisual hallucination benchmarks demonstrate that MoD-DPO consistently improves perception accuracy and hallucination resistance, outperforming previous preference optimization baselines under similar training budgets. Our findings underscore the importance of modality-faithful alignment and demonstrate a scalable path toward more reliable and resilient multimodal foundation models. 

---
# StitchCUDA: An Automated Multi-Agents End-to-End GPU Programing Framework with Rubric-based Agentic Reinforcement Learning 

**Authors**: Shiyang Li, Zijian Zhang, Winson Chen, Yuebo Luo, Mingyi Hong, Caiwen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2603.02637)  

**Abstract**: Modern machine learning (ML) workloads increasingly rely on GPUs, yet achieving high end-to-end performance remains challenging due to dependencies on both GPU kernel efficiency and host-side settings. Although LLM-based methods show promise on automated GPU kernel generation, prior works mainly focus on single-kernel optimization and do not extend to end-to-end programs, hindering practical deployment.
To address the challenge, in this work, we propose StitchCUDA, a multi-agent framework for end-to-end GPU program generation, with three specialized agents: a Planner to orchestrate whole system design, a Coder dedicated to implementing it step-by-step, and a Verifier for correctness check and performance profiling using Nsys/NCU. To fundamentally improve the Coder's ability in end-to-end GPU programming, StitchCUDA integrates rubric-based agentic reinforcement learning over two atomic skills, task-to-code generation and feedback-driven code optimization, with combined rubric reward and rule-based reward from real executions. Therefore, the Coder learns how to implement advanced CUDA programming techniques (e.g., custom kernel fusion, cublas epilogue), and we also effectively prevent Coder's reward hacking (e.g., just copy PyTorch code or hardcoding output) during benchmarking. Experiments on KernelBench show that StitchCUDA achieves nearly 100% success rate on end-to-end GPU programming tasks, with 1.72x better speedup over the multi-agent baseline and 2.73x than the RL model baselines. 

---
# Contextualized Privacy Defense for LLM Agents 

**Authors**: Yule Wen, Yanzhe Zhang, Jianxun Lian, Xiaoyuan Yi, Xing Xie, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.02983)  

**Abstract**: LLM agents increasingly act on users' personal information, yet existing privacy defenses remain limited in both design and adaptability. Most prior approaches rely on static or passive defenses, such as prompting and guarding. These paradigms are insufficient for supporting contextual, proactive privacy decisions in multi-step agent execution. We propose Contextualized Defense Instructing (CDI), a new privacy defense paradigm in which an instructor model generates step-specific, context-aware privacy guidance during execution, proactively shaping actions rather than merely constraining or vetoing them. Crucially, CDI is paired with an experience-driven optimization framework that trains the instructor via reinforcement learning (RL), where we convert failure trajectories with privacy violations into learning environments. We formalize baseline defenses and CDI as distinct intervention points in a canonical agent loop, and compare their privacy-helpfulness trade-offs within a unified simulation framework. Results show that our CDI consistently achieves a better balance between privacy preservation (94.2%) and helpfulness (80.6%) than baselines, with superior robustness to adversarial conditions and generalization. 

---
# Safety Training Persists Through Helpfulness Optimization in LLM Agents 

**Authors**: Benjamin Plaut  

**Link**: [PDF](https://arxiv.org/pdf/2603.02229)  

**Abstract**: Safety post-training has been studied extensively in single-step "chat" settings where safety typically refers to refusing harmful requests. We study an "agentic" (i.e., multi-step, tool-use) setting where safety refers to harmful actions directly taken by the LLM. We compare the effects of running direct preference optimization (DPO) on safety or helpfulness alone vs both metrics sequentially. As expected, training on one metric alone results in an extreme point along this frontier. However, unlike prior work, we find that safety training persists through subsequent helpfulness training. We also find that all training configurations end up near a linear Pareto frontier with $R^2 = 0.77$. Even post-training on both metrics simultaneously simply results in another point on the frontier rather than finding a "best of both worlds" strategy, despite the presence of such strategies in our DPO dataset. Overall, our findings underscore the need for better understanding of post-training dynamics. 

---
# TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning 

**Authors**: Christian Greisinger, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2603.03072)  

**Abstract**: Large language models (LLMs) are increasingly used to assist scientists across diverse workflows. A key challenge is generating high-quality figures from textual descriptions, often represented as TikZ programs that can be rendered as scientific images. Prior research has proposed a variety of datasets and modeling approaches for this task. However, existing datasets for Text-to-TikZ are too small and noisy to capture the complexity of TikZ, causing mismatches between text and rendered figures. Moreover, prior approaches rely solely on supervised fine-tuning (SFT), which does not expose the model to the rendered semantics of the figure, often resulting in errors such as looping, irrelevant content, and incorrect spatial relations. To address these issues, we construct DaTikZ-V4, a dataset more than four times larger and substantially higher in quality than DaTikZ-V3, enriched with LLM-generated figure descriptions. Using this dataset, we train TikZilla, a family of small open-source Qwen models (3B and 8B) with a two-stage pipeline of SFT followed by reinforcement learning (RL). For RL, we leverage an image encoder trained via inverse graphics to provide semantically faithful reward signals. Extensive human evaluations with over 1,000 judgments show that TikZilla improves by 1.5-2 points over its base models on a 5-point scale, surpasses GPT-4o by 0.5 points, and matches GPT-5 in the image-based evaluation, while operating at much smaller model sizes. Code, data, and models will be made available. 

---
# ShipTraj-R1: Reinforcing Ship Trajectory Prediction in Large Language Models via Group Relative Policy Optimization 

**Authors**: Yang Zhan, Yunhao Li, Zhang Chao, Yuxu Lu, Yan Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.02939)  

**Abstract**: Recent advancements in reinforcement fine-tuning have significantly improved the reasoning ability of large language models (LLMs). In particular, methods such as group relative policy optimization (GRPO) have demonstrated strong capabilities across various fields. However, applying LLMs to ship trajectory prediction remains largely unexplored. In this paper, we propose ShipTraj-R1, a novel LLM-based framework that reformulates ship trajectory prediction as a text-to-text generation problem. (1) We design a dynamic prompt containing trajectory information about conflicting ships to guide the model to achieve adaptive chain-of-thought (CoT) reasoning. (2) We introduce a comprehensive rule-based reward mechanism to incentivize the reasoning format and prediction accuracy of the model. (3) Our ShipTraj-R1 is reinforced through the GRPO mechanism guided by domain-specific prompts and rewards, and utilizes the Qwen3 as the model backbone. Extensive experimental results on two complex and real-world maritime datasets show that the proposed ShipTraj-R1 achieves the least error compared with state-of-the-art deep learning and LLM-based baselines. 

---
# LLMs for High-Frequency Decision-Making: Normalized Action Reward-Guided Consistency Policy Optimization 

**Authors**: Yang Zhao, Zihao Li, Zhiyu Jiang, Dandan Ma, Ganchao Liu, Wenzhe Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.02680)  

**Abstract**: While Large Language Models (LLMs) form the cornerstone of sequential decision-making agent development, they have inherent limitations in high-frequency decision tasks. Existing research mainly focuses on discrete embodied decision scenarios with low-frequency and significant semantic differences in state space (e.g., household planning). These methods suffer from limited performance in high-frequency decision-making tasks, since high-precision numerical state information in such tasks undergoes frequent updates with minimal fluctuations, and exhibiting policy misalignment between the learned sub-tasks and composite tasks. To address these issues, this paper proposes Normalized Action Reward guided Consistency Policy Optimization (NAR-CP). 1) Our method first acquires predefined dense rewards from environmental feedback of candidate actions via reward functions, then completes reward shaping through normalization, and theoretically verifies action reward normalization does not impair optimal policy. 2) To reduce policy misalignment in composite tasks, we use LLMs to infer sub-observation candidate actions and generate joint policies, with consistency loss ensuring precise alignment between global semantic policies and sub-semantic policies. Experiments on UAV pursuit, a typical high-frequency task, show our method delivers superior performance on independent and composite tasks with excellent generalization to unseen tasks. 

---
# PRISM: Pushing the Frontier of Deep Think via Process Reward Model-Guided Inference 

**Authors**: Rituraj Sharma, Weiyuan Chen, Noah Provenzano, Tu Vu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02479)  

**Abstract**: DEEPTHINK methods improve reasoning by generating, refining, and aggregating populations of candidate solutions, which enables strong performance on complex mathematical and scientific tasks. However, existing frameworks often lack reliable correctness signals during inference, which creates a population-enhancement bottleneck where deeper deliberation amplifies errors, suppresses correct minority solutions, and yields weak returns to additional compute. In this paper, we introduce a functional decomposition of DEEPTHINK systems and propose PRISM, a Process Reward Model (PRM)-guided inference algorithm that uses step-level verification to guide both population refinement and solution aggregation. During refinement, PRISM treats candidate solutions as particles in a PRM-defined energy landscape and reshapes the population through score-guided resampling and stochastic refinement, which concentrates probability mass on higher-quality reasoning while preserving diversity. Across mathematics and science benchmarks, PRISM is competitive with or outperforms existing DEEPTHINK methods, reaching 90.0%, 75.4%, and 71.4% with gpt-oss-20b on AIME25, HMMT25, and GPQA Diamond, respectively, while matching or exceeding gpt-oss-120b. Additionally, our analysis shows that PRISM produces consistent net-directional correction during refinement, remains reliable when the initial population contains few correct candidates, and often lies on the compute-accuracy Pareto frontier. 

---
# How to Peel with a Knife: Aligning Fine-Grained Manipulation with Human Preference 

**Authors**: Toru Lin, Shuying Deng, Zhao-Heng Yin, Pieter Abbeel, Jitendra Malik  

**Link**: [PDF](https://arxiv.org/pdf/2603.03280)  

**Abstract**: Many essential manipulation tasks - such as food preparation, surgery, and craftsmanship - remain intractable for autonomous robots. These tasks are characterized not only by contact-rich, force-sensitive dynamics, but also by their "implicit" success criteria: unlike pick-and-place, task quality in these domains is continuous and subjective (e.g. how well a potato is peeled), making quantitative evaluation and reward engineering difficult. We present a learning framework for such tasks, using peeling with a knife as a representative example. Our approach follows a two-stage pipeline: first, we learn a robust initial policy via force-aware data collection and imitation learning, enabling generalization across object variations; second, we refine the policy through preference-based finetuning using a learned reward model that combines quantitative task metrics with qualitative human feedback, aligning policy behavior with human notions of task quality. Using only 50-200 peeling trajectories, our system achieves over 90% average success rates on challenging produce including cucumbers, apples, and potatoes, with performance improving by up to 40% through preference-based finetuning. Remarkably, policies trained on a single produce category exhibit strong zero-shot generalization to unseen in-category instances and to out-of-distribution produce from different categories while maintaining over 90% success rates. 

---
# Why Does RLAIF Work At All? 

**Authors**: Robin Young  

**Link**: [PDF](https://arxiv.org/pdf/2603.03000)  

**Abstract**: Reinforcement Learning from AI Feedback (RLAIF) enables language models to improve by training on their own preference judgments, yet no theoretical account explains why this self-improvement seemingly works for value learning. We propose the latent value hypothesis, that pretraining on internet-scale data encodes human values as directions in representation space, and constitutional prompts elicit these latent values into preference judgments. We formalize this intuition under a linear model where the constitution acts as a projection operator selecting value-relevant directions. Our analysis yields several results. RLAIF improves alignment when the constitution-activated direction correlates with true values better than the model's default generation direction thus explaining the generation-judgment gap; the ceiling on RLAIF quality is determined by how well representations encode values, which scales with model capacity; and adversarial constitutions exist that can activate anti-social value directions encoded from harmful pretraining data. Our account unifies scattered empirical findings including the refusal direction, low-rank safety subspaces, and RLAIF scaling behavior. 

---
# When Scaling Fails: Mitigating Audio Perception Decay of LALMs via Multi-Step Perception-Aware Reasoning 

**Authors**: Ruixiang Mao, Xiangnan Ma, Dan Chen, Ziming Zhu, Yuan Ge, Aokai Hao, Haishu Zhao, Yifu Huo, Qing Yang, Kaiyan Chang, Xiaoqian Liu, Chenglong Wang, Qiaozhi He, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.02266)  

**Abstract**: Test-Time Scaling has shown notable efficacy in addressing complex problems through scaling inference compute. However, within Large Audio-Language Models (LALMs), an unintuitive phenomenon exists: post-training models for structured reasoning trajectories results in marginal or even negative gains compared to post-training for direct answering. To investigate it, we introduce CAFE, an evaluation framework designed to precisely quantify audio reasoning errors. Evaluation results reveal LALMs struggle with perception during reasoning and encounter a critical bottleneck: reasoning performance suffers from audio perception decay as reasoning length extends. To address it, we propose MPAR$^2$, a paradigm that encourages dynamic perceptual reasoning and decomposes complex questions into perception-rich sub-problems. Leveraging reinforcement learning, MPAR$^2$ improves perception performance on CAFE from 31.74% to 63.51% and effectively mitigates perception decay, concurrently enhancing reasoning capabilities to achieve a significant 74.59% accuracy on the MMAU benchmark. Further analysis demonstrates that MPAR$^2$ reinforces LALMs to attend to audio input and dynamically adapts reasoning budget to match task complexity. 

---
# Beyond Binary Preferences: A Principled Framework for Reward Modeling with Ordinal Feedback 

**Authors**: Amirhossein Afsharrad, Ruida Zhou, Luca Viano, Sanjay Lall, Mohammad Ghavamzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2603.02232)  

**Abstract**: Reward modeling is crucial for aligning large language models with human preferences, yet current approaches lack a principled mathematical framework for leveraging ordinal preference data. When human annotators provide graded preferences on a Likert scale (e.g., significantly better, better, slightly better, negligibly better), existing methods typically apply ad-hoc heuristics, such as margin terms or scaling factors, to loss functions derived from binary preference models like Bradley-Terry. These approaches lack an underlying mathematical model for how ordinal preference data is generated. We present a theoretically grounded framework that formulates reward modeling with Likert scale preferences as a discrete ordinal regression problem. We derive two loss functions from this formulation: a negative log-likelihood loss and an all-threshold loss, both of which learn threshold parameters that naturally capture the ordinal structure of preferences. Unlike existing heuristic methods that manually specify fixed margins or scaling weights, our approach learns these parameters directly from data within a coherent probabilistic framework. Experimental results on multiple benchmarks demonstrate that our ordinal regression approach consistently achieves competitive or superior performance compared to existing heuristic methods across diverse evaluation categories including chat, reasoning, and safety tasks. Our work provides the first principled mathematical framework for incorporating Likert scale preferences into reward model training, moving beyond ad-hoc modifications of binary preference models to enable more effective utilization of fine-grained human feedback. 

---
