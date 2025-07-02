# SAFER: Probing Safety in Reward Models with Sparse Autoencoder 

**Authors**: Sihang Li, Wei Shi, Ziyuan Xie, Tao Liang, Guojun Ma, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00665)  

**Abstract**: Reinforcement learning from human feedback (RLHF) is a key paradigm for aligning large language models (LLMs) with human values, yet the reward models at its core remain largely opaque. In this work, we present sparse Autoencoder For Enhanced Reward model (\textbf{SAFER}), a novel framework for interpreting and improving reward models through mechanistic analysis. Leveraging Sparse Autoencoders (SAEs), we uncover human-interpretable features in reward model activations, enabling insight into safety-relevant decision-making. We apply SAFER to safety-oriented preference datasets and quantify the salience of individual features by activation differences between chosen and rejected responses. Using these feature-level signals, we design targeted data poisoning and denoising strategies. Experiments show that SAFER can precisely degrade or enhance safety alignment with minimal data modification, without sacrificing general chat performance. Our approach contributes to interpreting, auditing and refining reward models in high-stakes LLM alignment tasks. Our codes are available at this https URL. \textit{This paper discusses topics related to large language model safety and may include discussions or examples that highlight potential risks or unsafe outcomes.} 

---
# Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning 

**Authors**: Maggie Huan, Yuetai Li, Tuney Zheng, Xiaoyu Xu, Seungone Kim, Minxin Du, Radha Poovendran, Graham Neubig, Xiang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2507.00432)  

**Abstract**: Math reasoning has become the poster child of progress in large language models (LLMs), with new models rapidly surpassing human-level performance on benchmarks like MATH and AIME. But as math leaderboards improve week by week, it is worth asking: do these gains reflect broader problem-solving ability or just narrow overfitting? To answer this question, we evaluate over 20 open-weight reasoning-tuned models across a broad suite of tasks, including math, scientific QA, agent planning, coding, and standard instruction-following. We surprisingly find that most models that succeed in math fail to transfer their gains to other domains. To rigorously study this phenomenon, we conduct controlled experiments on Qwen3-14B models using math-only data but different tuning methods. We find that reinforcement learning (RL)-tuned models generalize well across domains, while supervised fine-tuning (SFT)-tuned models often forget general capabilities. Latent-space representation and token-space distribution shift analyses reveal that SFT induces substantial representation and output drift, while RL preserves general-domain structure. Our results suggest a need to rethink standard post-training recipes, particularly the reliance on SFT-distilled data for advancing reasoning models. 

---
# $μ^2$Tokenizer: Differentiable Multi-Scale Multi-Modal Tokenizer for Radiology Report Generation 

**Authors**: Siyou Li, Pengyao Qin, Huanan Wu, Dong Nie, Arun J. Thirunavukarasu, Juntao Yu, Le Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00316)  

**Abstract**: Automated radiology report generation (RRG) aims to produce detailed textual reports from clinical imaging, such as computed tomography (CT) scans, to improve the accuracy and efficiency of diagnosis and provision of management advice. RRG is complicated by two key challenges: (1) inherent complexity in extracting relevant information from imaging data under resource constraints, and (2) difficulty in objectively evaluating discrepancies between model-generated and expert-written reports. To address these challenges, we propose $\mu^2$LLM, a $\underline{\textbf{mu}}$ltiscale $\underline{\textbf{mu}}$ltimodal large language models for RRG tasks. The novel ${\mu}^2$Tokenizer, as an intermediate layer, integrates multi-modal features from the multiscale visual tokenizer and the text tokenizer, then enhances report generation quality through direct preference optimization (DPO), guided by GREEN-RedLlama. Experimental results on four large CT image-report medical datasetdemonstrate that our method outperforms existing approaches, highlighting the potential of our fine-tuned $\mu^2$LLMs on limited data for RRG tasks. 

---
# Enhancing Reasoning Capabilities in SLMs with Reward Guided Dataset Distillation 

**Authors**: Shreyansh Padarha  

**Link**: [PDF](https://arxiv.org/pdf/2507.00054)  

**Abstract**: The push to compress and impart the proficiency of Large Language Models (LLMs) into more deployable and efficient Small Language Models (SLMs) has benefited from improvements in knowledge distillation (KD) techniques. These techniques allow a smaller student model to learn from a more capable and larger teacher model's responses. However, distillation often revolves around the student model merely copying the teacher's in-distribution responses, limiting its generalisability. This limitation is amplified on reasoning tasks and can be computationally expensive. In this study, we propose AdvDistill, a reward-guided dataset distillation framework. We utilise multiple generations (responses) from a teacher for each prompt and assign rewards based on rule-based verifiers. These varying and normally distributed rewards serve as weights when training student models. Our methods and their subsequent behavioural analysis demonstrate a significant improvement in student model performance for mathematical and complex reasoning tasks, showcasing the efficacy and benefits of incorporating a rewarding mechanism in dataset distillation processes. 

---
# Implicit Reward as the Bridge: A Unified View of SFT and DPO Connections 

**Authors**: Bo Wang, Qinyuan Cheng, Runyu Peng, Rong Bao, Peiji Li, Qipeng Guo, Linyang Li, Zhiyuan Zeng, Yunhua Zhou, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.00018)  

**Abstract**: Post-training processes are essential phases in grounding pre-trained language models to real-world tasks, with learning from demonstrations or preference signals playing a crucial role in this adaptation. We present a unified theoretical framework bridging Supervised Fine-Tuning (SFT) and preference learning in Large Language Model (LLM) post-training. Through rigorous mathematical derivation, we demonstrate that both SFT and preference learning methods like Direct Preference Optimization (DPO) operate within the same optimal policy-reward subspace, with SFT representing a special case of implicit reward learning. Our analysis reveals a critical limitation in conventional SFT: the KL divergence term in distribution matching becomes constant with respect to the policy during optimization, failing to constrain model updates. To address this, we propose a simple yet effective learning rate reduction approach that yields significant performance improvements (up to \textbf{25\%} relative gain and \textbf{6\%} absolute win rate increase in instruction following tasks. Additionally, we derive alternative SFT objectives from various f-divergence functions that preserve the KL term during optimization, further enhancing post-DPO model performance. Finally, we extend the theoretical relationship between LLM logits and Q-functions from preference learning to the SFT context, providing mathematical derivations and experimental validation. 

---
# ASTRO: Teaching Language Models to Reason by Reflecting and Backtracking In-Context 

**Authors**: Joongwon Kim, Anirudh Goyal, Liang Tan, Hannaneh Hajishirzi, Srinivasan Iyer, Tianlu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00417)  

**Abstract**: We introduce ASTRO, the "Autoregressive Search-Taught Reasoner", a framework for training language models to reason like search algorithms, explicitly leveraging self-reflection, backtracking, and exploration in their outputs. Recently, training large language models (LLMs) via reinforcement learning (RL) has led to the advent of reasoning models with greatly enhanced reasoning capabilities. Open-source replications of reasoning models, while successful, build upon models that already exhibit strong reasoning capabilities along with search behavior observed even before RL. As a result, it is yet unclear how to boost the reasoning capabilities of other non-reasoner models including Llama 3. ASTRO teaches such models to internalize structured search behavior through a synthetic dataset derived from Monte Carlo Tree Search (MCTS) over mathematical problem-solving trajectories. By converting search traces into natural language chain-of-thoughts that capture both successes and recoveries from failure, ASTRO bootstraps models with a rich prior for exploration during RL. We finetune our models on these search-derived traces and further improve performance via RL with verifiable rewards. We apply ASTRO to the Llama 3 family of models and achieve absolute performance gains of 16.0% on MATH-500, 26.9% on AMC 2023, and 20.0% on AIME 2024, especially improving upon challenging problems that require iterative correction. Our results demonstrate that search-inspired training offers a principled way to instill robust reasoning capabilities into open LLMs. 

---
# ROSE: Toward Reality-Oriented Safety Evaluation of Large Language Models 

**Authors**: Jiale Ding, Xiang Zheng, Cong Wang, Wei-Bin Lee, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00026)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed as black-box components in real-world applications, evaluating their safety-especially under adversarial prompting-has become critical. Arguably, effective safety evaluations should be adaptive, evolving with LLM capabilities, and also cover a broad spectrum of harmful topics and real-world scenarios to fully expose potential vulnerabilities. Existing manual safety benchmarks, built on handcrafted adversarial prompts, are limited by their static nature and the intensive labor required to update them, making it difficult to keep pace with rapidly advancing LLMs. In contrast, automated adversarial prompt generation offers a promising path toward adaptive evaluation. However, current methods often suffer from insufficient adversarial topic coverage (topic-level diversity) and weak alignment with real-world contexts. These shortcomings stem from the exploration-exploitation dilemma in black-box optimization and a lack of real-world contextualization, resulting in adversarial prompts that are both topically narrow and scenario-repetitive. To address these issues, we propose Reality-Oriented Safety Evaluation (ROSE), a novel framework that uses multi-objective reinforcement learning to fine-tune an adversarial LLM for generating topically diverse and contextually rich adversarial prompts. Experiments show that ROSE outperforms existing methods in uncovering safety vulnerabilities in state-of-the-art LLMs, with notable improvements in integrated evaluation metrics. We hope ROSE represents a step toward more practical and reality-oriented safety evaluation of LLMs. WARNING: This paper contains examples of potentially harmful text. 

---
# GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning 

**Authors**: Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Leyi Pan, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianle Gong, Wenkai Li, Wei Jia, Xin Lyu, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuxuan Zhang, Zhanxiao Du, Zhenyu Hou, Zhao Xue, Zhengxiao Du, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.01006)  

**Abstract**: We present GLM-4.1V-Thinking, a vision-language model (VLM) designed to advance general-purpose multimodal reasoning. In this report, we share our key findings in the development of the reasoning-centric training framework. We first develop a capable vision foundation model with significant potential through large-scale pre-training, which arguably sets the upper bound for the final performance. Reinforcement Learning with Curriculum Sampling (RLCS) then unlocks the full potential of the model, leading to comprehensive capability enhancement across a diverse range of tasks, including STEM problem solving, video understanding, content recognition, coding, grounding, GUI-based agents, and long document understanding, among others. To facilitate research in this field, we open-source GLM-4.1V-9B-Thinking, which achieves state-of-the-art performance among models of comparable size. In a comprehensive evaluation across 28 public benchmarks, our model outperforms Qwen2.5-VL-7B on nearly all tasks and achieves comparable or even superior performance on 18 benchmarks relative to the significantly larger Qwen2.5-VL-72B. Notably, GLM-4.1V-9B-Thinking also demonstrates competitive or superior performance compared to closed-source models such as GPT-4o on challenging tasks including long document understanding and STEM reasoning, further underscoring its strong capabilities. Code, models and more information are released at this https URL. 

---
# Can Large Language Models Develop Strategic Reasoning? Post-training Insights from Learning Chess 

**Authors**: Dongyoon Hwang, Hojoon Lee, Jaegul Choo, Dongmin Park, Jongho Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.00726)  

**Abstract**: While reinforcement learning (RL) for large language models (LLMs) has shown promise in mathematical reasoning, strategic reasoning for LLMs using RL remains largely unexplored. We investigate whether LLMs can develop strategic reasoning capabilities through RL in chess. To this end, we leverage a chess-pretrained action-value network to provide dense reward on the LLM's output move quality, which can be seen as a form of knowledge distillation. Our experiments show that our distillation-based dense rewards often outperform sparse binary rewards. However, surprisingly, all models plateau far below expert levels. We provide SFT and RL ablations on chess reasoning training and find evidence that this limitation stems from a deficit in the pretrained models' internal understanding of chess--a deficit which RL alone may not be able to fully overcome. 

---
# Reasoning as an Adaptive Defense for Safety 

**Authors**: Taeyoun Kim, Fahim Tajwar, Aditi Raghunathan, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.00971)  

**Abstract**: Reasoning methods that adaptively allocate test-time compute have advanced LLM performance on easy to verify domains such as math and code. In this work, we study how to utilize this approach to train models that exhibit a degree of robustness to safety vulnerabilities, and show that doing so can provide benefits. We build a recipe called $\textit{TARS}$ (Training Adaptive Reasoners for Safety), a reinforcement learning (RL) approach that trains models to reason about safety using chain-of-thought traces and a reward signal that balances safety with task completion. To build TARS, we identify three critical design choices: (1) a "lightweight" warmstart SFT stage, (2) a mix of harmful, harmless, and ambiguous prompts to prevent shortcut behaviors such as too many refusals, and (3) a reward function to prevent degeneration of reasoning capabilities during training. Models trained with TARS exhibit adaptive behaviors by spending more compute on ambiguous queries, leading to better safety-refusal trade-offs. They also internally learn to better distinguish between safe and unsafe prompts and attain greater robustness to both white-box (e.g., GCG) and black-box attacks (e.g., PAIR). Overall, our work provides an effective, open recipe for training LLMs against jailbreaks and harmful requests by reasoning per prompt. 

---
# Residual Reward Models for Preference-based Reinforcement Learning 

**Authors**: Chenyang Cao, Miguel Rogel-García, Mohamed Nabail, Xueqian Wang, Nicholas Rhinehart  

**Link**: [PDF](https://arxiv.org/pdf/2507.00611)  

**Abstract**: Preference-based Reinforcement Learning (PbRL) provides a way to learn high-performance policies in environments where the reward signal is hard to specify, avoiding heuristic and time-consuming reward design. However, PbRL can suffer from slow convergence speed since it requires training in a reward model. Prior work has proposed learning a reward model from demonstrations and fine-tuning it using preferences. However, when the model is a neural network, using different loss functions for pre-training and fine-tuning can pose challenges to reliable optimization. In this paper, we propose a method to effectively leverage prior knowledge with a Residual Reward Model (RRM). An RRM assumes that the true reward of the environment can be split into a sum of two parts: a prior reward and a learned reward. The prior reward is a term available before training, for example, a user's ``best guess'' reward function, or a reward function learned from inverse reinforcement learning (IRL), and the learned reward is trained with preferences. We introduce state-based and image-based versions of RRM and evaluate them on several tasks in the Meta-World environment suite. Experimental results show that our method substantially improves the performance of a common PbRL method. Our method achieves performance improvements for a variety of different types of prior rewards, including proxy rewards, a reward obtained from IRL, and even a negated version of the proxy reward. We also conduct experiments with a Franka Panda to show that our method leads to superior performance on a real robot. It significantly accelerates policy learning for different tasks, achieving success in fewer steps than the baseline. The videos are presented at this https URL. 

---
