# Stable Preference Optimization for LLMs: A Bilevel Approach Beyond Direct Preference Optimization 

**Authors**: Chengtao Jian, Kai Yang, Ye Ouyang, Xiaozhou Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.07723)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as a popular and efficient alternative to reward modeling and reinforcement learning for aligning language models with human preferences. Despite its empirical success, the theoretical properties and intrinsic limitations of DPO remain underexplored. In this work, we first present a comprehensive analysis of DPO's dynamics from a probability evolution perspective. Our analysis reveals that DPO is highly sensitive to initialization. It also tends to misallocate probability mass, which can inadvertently shift probability toward irrelevant or undesired responses. This misallocation may unintentionally reinforce model bias, thereby compromising both the stability of model alignment and the consistency with intended preferences. Motivated by these theoretical findings, we propose a theoretically grounded bilevel optimization framework that tightly integrate supervised fine-tuning with an enhanced DPO objective a.k.a. stable preference optimization. Our approach introduces a principled regularization scheme to explicitly encourage absolute probability improvement for preferred outputs, while maintaining stable optimization dynamics. Experiments on challenging reasoning and summarization benchmarks elucidate that our method consistently improves reasoning accuracy and better aligns output distributions with intended preferences, outperforming standard DPO. Stable preference optimization provides new insights into the design of preference-based alignment objectives and opens up new avenues towards more reliable and interpretable language model alignment. 

---
# Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology 

**Authors**: Haochen Wang, Xiangtai Li, Zilong Huang, Anran Wang, Jiacong Wang, Tao Zhang, Jiani Zheng, Sule Bai, Zijian Kang, Jiashi Feng, Zhuochen Wang, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07999)  

**Abstract**: Models like OpenAI-o3 pioneer visual grounded reasoning by dynamically referencing visual regions, just like human "thinking with images". However, no benchmark exists to evaluate these capabilities holistically. To bridge this gap, we propose TreeBench (Traceable Evidence Evaluation Benchmark), a diagnostic benchmark built on three principles: (1) focused visual perception of subtle targets in complex scenes, (2) traceable evidence via bounding box evaluation, and (3) second-order reasoning to test object interactions and spatial hierarchies beyond simple object localization. Prioritizing images with dense objects, we initially sample 1K high-quality images from SA-1B, and incorporate eight LMM experts to manually annotate questions, candidate options, and answers for each image. After three stages of quality control, TreeBench consists of 405 challenging visual question-answering pairs, even the most advanced models struggle with this benchmark, where none of them reach 60% accuracy, e.g., OpenAI-o3 scores only 54.87. Furthermore, we introduce TreeVGR (Traceable Evidence Enhanced Visual Grounded Reasoning), a training paradigm to supervise localization and reasoning jointly with reinforcement learning, enabling accurate localizations and explainable reasoning pathways. Initialized from Qwen2.5-VL-7B, it improves V* Bench (+16.8), MME-RealWorld (+12.6), and TreeBench (+13.4), proving traceability is key to advancing vision-grounded reasoning. The code is available at this https URL. 

---
# Not All Preferences are What You Need for Post-Training: Selective Alignment Strategy for Preference Optimization 

**Authors**: Zhijin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.07725)  

**Abstract**: Post-training alignment of large language models (LLMs) is a critical challenge, as not all tokens contribute equally to model performance. This paper introduces a selective alignment strategy that prioritizes high-impact tokens within preference pairs, leveraging token-level log-probability differences between the current policy and a reference model. By focusing on these informative tokens, our approach reduces computational overhead and enhances alignment fidelity. We further explore the role of reference model quality, demonstrating that stronger reference models significantly improve token selection accuracy and overall optimization effectiveness. Comprehensive experiments on benchmarks such as Arena-Hard and MT-Bench validate the superiority of our Selective-DPO method over standard DPO and distillation-based baselines. Our findings highlight the importance of token-level optimization and reference model selection in advancing preference alignment for LLMs. The code is available at this https URL. 

---
# PLAN-TUNING: Post-Training Language Models to Learn Step-by-Step Planning for Complex Problem Solving 

**Authors**: Mihir Parmar, Palash Goyal, Xin Liu, Yiwen Song, Mingyang Ling, Chitta Baral, Hamid Palangi, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2507.07495)  

**Abstract**: Recently, decomposing complex problems into simple subtasks--a crucial part of human-like natural planning--to solve the given problem has significantly boosted the performance of large language models (LLMs). However, leveraging such planning structures during post-training to boost the performance of smaller open-source LLMs remains underexplored. Motivated by this, we introduce PLAN-TUNING, a unified post-training framework that (i) distills synthetic task decompositions (termed "planning trajectories") from large-scale LLMs and (ii) fine-tunes smaller models via supervised and reinforcement-learning objectives designed to mimic these planning processes to improve complex reasoning. On GSM8k and the MATH benchmarks, plan-tuned models outperform strong baselines by an average $\sim7\%$. Furthermore, plan-tuned models show better generalization capabilities on out-of-domain datasets, with average $\sim10\%$ and $\sim12\%$ performance improvements on OlympiadBench and AIME 2024, respectively. Our detailed analysis demonstrates how planning trajectories improves complex reasoning capabilities, showing that PLAN-TUNING is an effective strategy for improving task-specific performance of smaller LLMs. 

---
# Machine Bullshit: Characterizing the Emergent Disregard for Truth in Large Language Models 

**Authors**: Kaiqu Liang, Haimin Hu, Xuandong Zhao, Dawn Song, Thomas L. Griffiths, Jaime Fernández Fisac  

**Link**: [PDF](https://arxiv.org/pdf/2507.07484)  

**Abstract**: Bullshit, as conceptualized by philosopher Harry Frankfurt, refers to statements made without regard to their truth value. While previous work has explored large language model (LLM) hallucination and sycophancy, we propose machine bullshit as an overarching conceptual framework that can allow researchers to characterize the broader phenomenon of emergent loss of truthfulness in LLMs and shed light on its underlying mechanisms. We introduce the Bullshit Index, a novel metric quantifying LLMs' indifference to truth, and propose a complementary taxonomy analyzing four qualitative forms of bullshit: empty rhetoric, paltering, weasel words, and unverified claims. We conduct empirical evaluations on the Marketplace dataset, the Political Neutrality dataset, and our new BullshitEval benchmark (2,400 scenarios spanning 100 AI assistants) explicitly designed to evaluate machine bullshit. Our results demonstrate that model fine-tuning with reinforcement learning from human feedback (RLHF) significantly exacerbates bullshit and inference-time chain-of-thought (CoT) prompting notably amplify specific bullshit forms, particularly empty rhetoric and paltering. We also observe prevalent machine bullshit in political contexts, with weasel words as the dominant strategy. Our findings highlight systematic challenges in AI alignment and provide new insights toward more truthful LLM behavior. 

---
# Robust Multimodal Large Language Models Against Modality Conflict 

**Authors**: Zongmeng Zhang, Wengang Zhou, Jie Zhao, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.07151)  

**Abstract**: Despite the impressive capabilities of multimodal large language models (MLLMs) in vision-language tasks, they are prone to hallucinations in real-world scenarios. This paper investigates the hallucination phenomenon in MLLMs from the perspective of modality conflict. Unlike existing works focusing on the conflicts between model responses and inputs, we study the inherent conflicts in inputs from different modalities that place MLLMs in a dilemma and directly lead to hallucinations. We formally define the modality conflict and construct a dataset named Multimodal Modality Conflict (MMMC) to simulate this phenomenon in vision-language tasks. Three methods based on prompt engineering, supervised fine-tuning, and reinforcement learning are proposed to alleviate the hallucination caused by modality conflict. Extensive experiments are conducted on the MMMC dataset to analyze the merits and demerits of these methods. Our results show that the reinforcement learning method achieves the best performance in mitigating the hallucination under modality conflict, while the supervised fine-tuning method shows promising and stable performance. Our work sheds light on the unnoticed modality conflict that leads to hallucinations and provides more insights into the robustness of MLLMs. 

---
# SAGE: A Visual Language Model for Anomaly Detection via Fact Enhancement and Entropy-aware Alignment 

**Authors**: Guoxin Zang, Xue Li, Donglin Di, Lanshun Nie, Dechen Zhan, Yang Song, Lei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.07939)  

**Abstract**: While Vision-Language Models (VLMs) have shown promising progress in general multimodal tasks, they often struggle in industrial anomaly detection and reasoning, particularly in delivering interpretable explanations and generalizing to unseen categories. This limitation stems from the inherently domain-specific nature of anomaly detection, which hinders the applicability of existing VLMs in industrial scenarios that require precise, structured, and context-aware analysis. To address these challenges, we propose SAGE, a VLM-based framework that enhances anomaly reasoning through Self-Guided Fact Enhancement (SFE) and Entropy-aware Direct Preference Optimization (E-DPO). SFE integrates domain-specific knowledge into visual reasoning via fact extraction and fusion, while E-DPO aligns model outputs with expert preferences using entropy-aware optimization. Additionally, we introduce AD-PL, a preference-optimized dataset tailored for industrial anomaly reasoning, consisting of 28,415 question-answering instances with expert-ranked responses. To evaluate anomaly reasoning models, we develop Multiscale Logical Evaluation (MLE), a quantitative framework analyzing model logic and consistency. SAGE demonstrates superior performance on industrial anomaly datasets under zero-shot and one-shot settings. The code, model and dataset are available at this https URL. 

---
# Teaching LLM to Reason: Reinforcement Learning from Algorithmic Problems without Code 

**Authors**: Keqin Bao, Nuo Chen, Xiaoyuan Li, Binyuan Hui, Bowen Yu, Fuli Feng, Junyang Lin, Xiangnan He, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07498)  

**Abstract**: Enhancing reasoning capabilities remains a central focus in the LLM reasearch community. A promising direction involves requiring models to simulate code execution step-by-step to derive outputs for given inputs. However, as code is often designed for large-scale systems, direct application leads to over-reliance on complex data structures and algorithms, even for simple cases, resulting in overfitting to algorithmic patterns rather than core reasoning structures. To address this, we propose TeaR, which aims at teaching LLMs to reason better. TeaR leverages careful data curation and reinforcement learning to guide models in discovering optimal reasoning paths through code-related tasks, thereby improving general reasoning abilities. We conduct extensive experiments using two base models and three long-CoT distillation models, with model sizes ranging from 1.5 billion to 32 billion parameters, and across 17 benchmarks spanning Math, Knowledge, Code, and Logical Reasoning. The results consistently show significant performance improvements. Notably, TeaR achieves a 35.9% improvement on Qwen2.5-7B and 5.9% on R1-Distilled-7B. 

---
# SAND: Boosting LLM Agents with Self-Taught Action Deliberation 

**Authors**: Yu Xia, Yiran Jenny Shen, Junda Wu, Tong Yu, Sungchul Kim, Ryan A. Rossi, Lina Yao, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2507.07441)  

**Abstract**: Large Language Model (LLM) agents are commonly tuned with supervised finetuning on ReAct-style expert trajectories or preference optimization over pairwise rollouts. Most of these methods focus on imitating specific expert behaviors or promoting chosen reasoning thoughts and actions over rejected ones. However, without reasoning and comparing over alternatives actions, LLM agents finetuned with these methods may over-commit towards seemingly plausible but suboptimal actions due to limited action space exploration. To address this, in this paper we propose Self-taught ActioN Deliberation (SAND) framework, enabling LLM agents to explicitly deliberate over candidate actions before committing to one. To tackle the challenges of when and what to deliberate given large action space and step-level action evaluation, we incorporate self-consistency action sampling and execution-guided action critique to help synthesize step-wise action deliberation thoughts using the base model of the LLM agent. In an iterative manner, the deliberation trajectories are then used to finetune the LLM agent itself. Evaluating on two representative interactive agent tasks, SAND achieves an average 20% improvement over initial supervised finetuning and also outperforms state-of-the-art agent tuning approaches. 

---
# RLEP: Reinforcement Learning with Experience Replay for LLM Reasoning 

**Authors**: Hongzhi Zhang, Jia Fu, Jingyuan Zhang, Kai Fu, Qi Wang, Fuzheng Zhang, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.07451)  

**Abstract**: Reinforcement learning (RL) for large language models is an energy-intensive endeavor: training can be unstable, and the policy may gradually drift away from its pretrained weights. We present \emph{RLEP}\, -- \,Reinforcement Learning with Experience rePlay\, -- \,a two-phase framework that first collects verified trajectories and then replays them during subsequent training. At every update step, the policy is optimized on mini-batches that blend newly generated rollouts with these replayed successes. By replaying high-quality examples, RLEP steers the model away from fruitless exploration, focuses learning on promising reasoning paths, and delivers both faster convergence and stronger final performance. On the Qwen2.5-Math-7B base model, RLEP reaches baseline peak accuracy with substantially fewer updates and ultimately surpasses it, improving accuracy on AIME-2024 from 38.2% to 39.9%, on AIME-2025 from 19.8% to 22.3%, and on AMC-2023 from 77.0% to 82.2%. Our code, datasets, and checkpoints are publicly available at this https URL to facilitate reproducibility and further research. 

---
# Bradley-Terry and Multi-Objective Reward Modeling Are Complementary 

**Authors**: Zhiwei Zhang, Hui Liu, Xiaomin Li, Zhenwei Dai, Jingying Zeng, Fali Wang, Minhua Lin, Ramraj Chandradevan, Zhen Li, Chen Luo, Xianfeng Tang, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07375)  

**Abstract**: Reward models trained on human preference data have demonstrated strong effectiveness in aligning Large Language Models (LLMs) with human intent under the framework of Reinforcement Learning from Human Feedback (RLHF). However, RLHF remains vulnerable to reward hacking, where the policy exploits imperfections in the reward function rather than genuinely learning the intended behavior. Although significant efforts have been made to mitigate reward hacking, they predominantly focus on and evaluate in-distribution scenarios, where the training and testing data for the reward model share the same distribution. In this paper, we empirically show that state-of-the-art methods struggle in more challenging out-of-distribution (OOD) settings. We further demonstrate that incorporating fine-grained multi-attribute scores helps address this challenge. However, the limited availability of high-quality data often leads to weak performance of multi-objective reward functions, which can negatively impact overall performance and become the bottleneck. To address this issue, we propose a unified reward modeling framework that jointly trains Bradley--Terry (BT) single-objective and multi-objective regression-based reward functions using a shared embedding space. We theoretically establish a connection between the BT loss and the regression objective and highlight their complementary benefits. Specifically, the regression task enhances the single-objective reward function's ability to mitigate reward hacking in challenging OOD settings, while BT-based training improves the scoring capability of the multi-objective reward function, enabling a 7B model to outperform a 70B baseline. Extensive experimental results demonstrate that our framework significantly improves both the robustness and the scoring performance of reward models. 

---
