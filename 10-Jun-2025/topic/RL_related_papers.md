# LUCIFER: Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement 

**Authors**: Dimitris Panagopoulos, Adolfo Perrusquia, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07915)  

**Abstract**: In dynamic environments, the rapid obsolescence of pre-existing environmental knowledge creates a gap between an agent's internal model and the evolving reality of its operational context. This disparity between prior and updated environmental valuations fundamentally limits the effectiveness of autonomous decision-making. To bridge this gap, the contextual bias of human domain stakeholders, who naturally accumulate insights through direct, real-time observation, becomes indispensable. However, translating their nuanced, and context-rich input into actionable intelligence for autonomous systems remains an open challenge. To address this, we propose LUCIFER (Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement), a domain-agnostic framework that integrates a hierarchical decision-making architecture with reinforcement learning (RL) and large language models (LLMs) into a unified system. This architecture mirrors how humans decompose complex tasks, enabling a high-level planner to coordinate specialised sub-agents, each focused on distinct objectives and temporally interdependent actions. Unlike traditional applications where LLMs are limited to single role, LUCIFER integrates them in two synergistic roles: as context extractors, structuring verbal stakeholder input into domain-aware representations that influence decision-making through an attention space mechanism aligning LLM-derived insights with the agent's learning process, and as zero-shot exploration facilitators guiding the agent's action selection process during exploration. We benchmark various LLMs in both roles and demonstrate that LUCIFER improves exploration efficiency and decision quality, outperforming flat, goal-conditioned policies. Our findings show the potential of context-driven decision-making, where autonomous systems leverage human contextual knowledge for operational success. 

---
# RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards 

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.07736)  

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements. 

---
# Coordinating Search-Informed Reasoning and Reasoning-Guided Search in Claim Verification 

**Authors**: Qisheng Hu, Quanyu Long, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07528)  

**Abstract**: Multi-hop claim verification is inherently challenging, requiring multi-step reasoning to construct verification chains while iteratively searching for information to uncover hidden bridging facts. This process is fundamentally interleaved, as effective reasoning relies on dynamically retrieved evidence, while effective search demands reasoning to refine queries based on partial information. To achieve this, we propose Hierarchical Agent Reasoning and Information Search (HARIS), explicitly modeling the coordinated process of reasoning-driven searching and search-informed reasoning. HARIS consists of a high-level reasoning agent that focuses on constructing the main verification chain, generating factual questions when more information is needed, and a low-level search agent that iteratively retrieves more information, refining its search based on intermediate findings. This design allows each agent to specialize in its respective task, enhancing verification accuracy and interpretability. HARIS is trained using reinforcement learning with outcome-based rewards. Experimental results on the EX-FEVER and HOVER benchmarks demonstrate that HARIS achieves strong performance, greatly advancing multi-hop claim verification. 

---
# Mathesis: Towards Formal Theorem Proving from Natural Languages 

**Authors**: Yu Xuejun, Jianyuan Zhong, Zijin Feng, Pengyi Zhai, Roozbeh Yousefzadeh, Wei Chong Ng, Haoxiong Liu, Ziyi Shou, Jing Xiong, Yudong Zhou, Claudia Beth Ong, Austen Jeremy Sugiarto, Yaoxi Zhang, Wai Ming Tai, Huan Cao, Dongcai Lu, Jiacheng Sun, Qiang Xu, Shen Xin, Zhenguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.07047)  

**Abstract**: Recent advances in large language models show strong promise for formal reasoning. However, most LLM-based theorem provers have long been constrained by the need for expert-written formal statements as inputs, limiting their applicability to real-world problems expressed in natural language. We tackle this gap with Mathesis, the first end-to-end theorem proving pipeline processing informal problem statements. It contributes Mathesis-Autoformalizer, the first autoformalizer using reinforcement learning to enhance the formalization ability of natural language problems, aided by our novel LeanScorer framework for nuanced formalization quality assessment. It also proposes a Mathesis-Prover, which generates formal proofs from the formalized statements. To evaluate the real-world applicability of end-to-end formal theorem proving, we introduce Gaokao-Formal, a benchmark of 488 complex problems from China's national college entrance exam. Our approach is carefully designed, with a thorough study of each component. Experiments demonstrate Mathesis's effectiveness, with the autoformalizer outperforming the best baseline by 22% in pass-rate on Gaokao-Formal. The full system surpasses other model combinations, achieving 64% accuracy on MiniF2F with pass@32 and a state-of-the-art 18% on Gaokao-Formal. 

---
# Boosting LLM Reasoning via Spontaneous Self-Correction 

**Authors**: Xutong Zhao, Tengyu Xu, Xuewei Wang, Zhengxing Chen, Di Jin, Liang Tan, Yen-Ting, Zishun Yu, Zhuokai Zhao, Yun He, Sinong Wang, Han Fang, Sarath Chandar, Chen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06923)  

**Abstract**: While large language models (LLMs) have demonstrated remarkable success on a broad range of tasks, math reasoning remains a challenging one. One of the approaches for improving math reasoning is self-correction, which designs self-improving loops to let the model correct its own mistakes. However, existing self-correction approaches treat corrections as standalone post-generation refinements, relying on extra prompt and system designs to elicit self-corrections, instead of performing real-time, spontaneous self-corrections in a single pass. To address this, we propose SPOC, a spontaneous self-correction approach that enables LLMs to generate interleaved solutions and verifications in a single inference pass, with generation dynamically terminated based on verification outcomes, thereby effectively scaling inference time compute. SPOC considers a multi-agent perspective by assigning dual roles -- solution proposer and verifier -- to the same model. We adopt a simple yet effective approach to generate synthetic data for fine-tuning, enabling the model to develop capabilities for self-verification and multi-agent collaboration. We further improve its solution proposal and verification accuracy through online reinforcement learning. Experiments on mathematical reasoning benchmarks show that SPOC significantly improves performance. Notably, SPOC boosts the accuracy of Llama-3.1-8B and 70B Instruct models, achieving gains of 8.8% and 11.6% on MATH500, 10.0% and 20.0% on AMC23, and 3.3% and 6.7% on AIME24, respectively. 

---
# Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking 

**Authors**: Silin Gao, Antoine Bosselut, Samy Bengio, Emmanuel Abbe  

**Link**: [PDF](https://arxiv.org/pdf/2506.07751)  

**Abstract**: Recent studies have shown that large language models (LLMs), especially smaller ones, often lack robustness in their reasoning. I.e., they tend to experience performance drops when faced with distribution shifts, such as changes to numerical or nominal variables, or insertions of distracting clauses. A possible strategy to address this involves generating synthetic data to further "instantiate" reasoning problems on potential variations. In contrast, our approach focuses on "abstracting" reasoning problems. This not only helps counteract distribution shifts but also facilitates the connection to symbolic tools for deriving solutions. We find that this abstraction process is better acquired through reinforcement learning (RL) than just supervised fine-tuning, which often fails to produce faithful abstractions. Our method, AbstraL -- which promotes abstract reasoning in LLMs using RL on granular abstraction data -- significantly mitigates performance degradation on recent GSM perturbation benchmarks. 

---
# LLM-driven Indoor Scene Layout Generation via Scaled Human-aligned Data Synthesis and Multi-Stage Preference Optimization 

**Authors**: Yixuan Yang, Zhen Luo, Tongsheng Ding, Junru Lu, Mingqi Gao, Jinyu Yang, Victor Sanchez, Feng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.07570)  

**Abstract**: Automatic indoor layout generation has attracted increasing attention due to its potential in interior design, virtual environment construction, and embodied AI. Existing methods fall into two categories: prompt-driven approaches that leverage proprietary LLM services (e.g., GPT APIs) and learning-based methods trained on layout data upon diffusion-based models. Prompt-driven methods often suffer from spatial inconsistency and high computational costs, while learning-based methods are typically constrained by coarse relational graphs and limited datasets, restricting their generalization to diverse room categories. In this paper, we revisit LLM-based indoor layout generation and present 3D-SynthPlace, a large-scale dataset that combines synthetic layouts generated via a 'GPT synthesize, Human inspect' pipeline, upgraded from the 3D-Front dataset. 3D-SynthPlace contains nearly 17,000 scenes, covering four common room types -- bedroom, living room, kitchen, and bathroom -- enriched with diverse objects and high-level spatial annotations. We further introduce OptiScene, a strong open-source LLM optimized for indoor layout generation, fine-tuned based on our 3D-SynthPlace dataset through our two-stage training. For the warum-up stage I, we adopt supervised fine-tuning (SFT), which is taught to first generate high-level spatial descriptions then conditionally predict concrete object placements. For the reinforcing stage II, to better align the generated layouts with human design preferences, we apply multi-turn direct preference optimization (DPO), which significantly improving layout quality and generation success rates. Extensive experiments demonstrate that OptiScene outperforms traditional prompt-driven and learning-based baselines. Moreover, OptiScene shows promising potential in interactive tasks such as scene editing and robot navigation. 

---
# LeVo: High-Quality Song Generation with Multi-Preference Alignment 

**Authors**: Shun Lei, Yaoxun Xu, Zhiwei Lin, Huaicheng Zhang, Wei Tan, Hangting Chen, Jianwei Yu, Yixuan Zhang, Chenyu Yang, Haina Zhu, Shuai Wang, Zhiyong Wu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07520)  

**Abstract**: Recent advances in large language models (LLMs) and audio language models have significantly improved music generation, particularly in lyrics-to-song generation. However, existing approaches still struggle with the complex composition of songs and the scarcity of high-quality data, leading to limitations in sound quality, musicality, instruction following, and vocal-instrument harmony. To address these challenges, we introduce LeVo, an LM-based framework consisting of LeLM and a music codec. LeLM is capable of parallelly modeling two types of tokens: mixed tokens, which represent the combined audio of vocals and accompaniment to achieve vocal-instrument harmony, and dual-track tokens, which separately encode vocals and accompaniment for high-quality song generation. It employs two decoder-only transformers and a modular extension training strategy to prevent interference between different token types. To further enhance musicality and instruction following, we introduce a multi-preference alignment method based on Direct Preference Optimization (DPO). This method handles diverse human preferences through a semi-automatic data construction process and DPO post-training. Experimental results demonstrate that LeVo consistently outperforms existing methods on both objective and subjective metrics. Ablation studies further justify the effectiveness of our designs. Audio examples are available at this https URL. 

---
# DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO 

**Authors**: Jinyoung Park, Jeehye Na, Jinyoung Kim, Hyunwoo J. Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07464)  

**Abstract**: Recent works have demonstrated the effectiveness of reinforcement learning (RL)-based post-training in enhancing the reasoning capabilities of large language models (LLMs). In particular, Group Relative Policy Optimization (GRPO) has shown impressive success by employing a PPO-style reinforcement algorithm with group-based normalized rewards. However, the application of GRPO to Video Large Language Models (Video LLMs) has been less studied. In this paper, we explore GRPO for video LLMs and identify two primary issues that impede its effective learning: (1) reliance on safeguards, and (2) the vanishing advantage problem. To mitigate these challenges, we propose DeepVideo-R1, a video large language model trained with our proposed Reg-GRPO (Regressive GRPO) and difficulty-aware data augmentation strategy. Reg-GRPO reformulates the GRPO objective as a regression task, directly predicting the advantage in GRPO. This design eliminates the need for safeguards like clipping and min functions, thereby facilitating more direct policy guidance by aligning the model with the advantage values. We also design the difficulty-aware data augmentation strategy that dynamically augments training samples at solvable difficulty levels, fostering diverse and informative reward signals. Our comprehensive experiments show that DeepVideo-R1 significantly improves video reasoning performance across multiple video reasoning benchmarks. 

---
# Advancing Multimodal Reasoning Capabilities of Multimodal Large Language Models via Visual Perception Reward 

**Authors**: Tong Xiao, Xin Xu, Zhenya Huang, Hongyu Gao, Quan Liu, Qi Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07218)  

**Abstract**: Enhancing the multimodal reasoning capabilities of Multimodal Large Language Models (MLLMs) is a challenging task that has attracted increasing attention in the community. Recently, several studies have applied Reinforcement Learning with Verifiable Rewards (RLVR) to the multimodal domain in order to enhance the reasoning abilities of MLLMs. However, these works largely overlook the enhancement of multimodal perception capabilities in MLLMs, which serve as a core prerequisite and foundational component of complex multimodal reasoning. Through McNemar's test, we find that existing RLVR method fails to effectively enhance the multimodal perception capabilities of MLLMs, thereby limiting their further improvement in multimodal reasoning. To address this limitation, we propose Perception-R1, which introduces a novel visual perception reward that explicitly encourages MLLMs to perceive the visual content accurately, thereby can effectively incentivizing both their multimodal perception and reasoning capabilities. Specifically, we first collect textual visual annotations from the CoT trajectories of multimodal problems, which will serve as visual references for reward assignment. During RLVR training, we employ a judging LLM to assess the consistency between the visual annotations and the responses generated by MLLM, and assign the visual perception reward based on these consistency judgments. Extensive experiments on several multimodal reasoning benchmarks demonstrate the effectiveness of our Perception-R1, which achieves state-of-the-art performance on most benchmarks using only 1,442 training data. 

---
# AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models 

**Authors**: Qi Liu, Jingqing Ruan, Hao Li, Haodong Zhao, Desheng Wang, Jiansong Chen, Wan Guanglu, Xunliang Cai, Zhi Zheng, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07165)  

**Abstract**: Existing multi-objective preference alignment methods for large language models (LLMs) face limitations: (1) the inability to effectively balance various preference dimensions, and (2) reliance on auxiliary reward/reference models introduces computational complexity. To address these challenges, we propose Adaptive Multi-objective Preference Optimization (AMoPO), a novel framework that achieves dynamic balance across preference dimensions. By introducing the multi-objective optimization paradigm to use the dimension-aware generation metrics as implicit rewards, AMoPO aligns LLMs with diverse preferences without additional reward models or reference models. We introduce an adaptive weight assignment mechanism that models the generation space as a Gaussian distribution, allowing dynamic prioritization of preference dimensions. Empirical results demonstrate that AMoPO outperforms state-of-the-art baselines by 28.5%, and the experiments on 7B, 14B, and 32B models reveal the scaling ability of AMoPO. Moreover, additional analysis of multiple dimensions verifies its adaptability and effectiveness. These findings validate AMoPO's capability to achieve dimension-aware preference alignment, highlighting its superiority. Our codes and datasets are available at this https URL. 

---
# AnnoDPO: Protein Functional Annotation Learning with Direct Preference Optimization 

**Authors**: Zixuan Jiang, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07035)  

**Abstract**: Deciphering protein function remains a fundamental challenge in protein representation learning. The task presents significant difficulties for protein language models (PLMs) due to the sheer volume of functional annotation categories and the highly imbalanced distribution of annotated instances across biological ontologies. Inspired by the remarkable success of reinforcement learning from human feedback (RLHF) in large language model (LLM) alignment, we propose AnnoDPO, a novel multi-modal framework for protein function prediction that leverages Direct Preference Optimization (DPO) to enhance annotation learning. Our methodology addresses the dual challenges of annotation scarcity and category imbalance through preference-aligned training objectives, establishing a new paradigm for biological knowledge integration in protein representation learning. 

---
# Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning 

**Authors**: Shubham Parashar, Shurui Gui, Xiner Li, Hongyi Ling, Sushil Vemuri, Blake Olson, Eric Li, Yu Zhang, James Caverlee, Dileep Kalathil, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.06632)  

**Abstract**: We aim to improve the reasoning capabilities of language models via reinforcement learning (RL). Recent RL post-trained models like DeepSeek-R1 have demonstrated reasoning abilities on mathematical and coding tasks. However, prior studies suggest that using RL alone to improve reasoning on inherently difficult tasks is less effective. Here, we draw inspiration from curriculum learning and propose to schedule tasks from easy to hard (E2H), allowing LLMs to build reasoning skills gradually. Our method is termed E2H Reasoner. Empirically, we observe that, although easy tasks are important initially, fading them out through appropriate scheduling is essential in preventing overfitting. Theoretically, we establish convergence guarantees for E2H Reasoner within an approximate policy iteration framework. We derive finite-sample complexity bounds and show that when tasks are appropriately decomposed and conditioned, learning through curriculum stages requires fewer total samples than direct learning. Experiments across multiple domains show that E2H Reasoner significantly improves the reasoning ability of small LLMs (1.5B to 3B), which otherwise struggle when trained with vanilla RL alone, highlighting the effectiveness of our method. 

---
# Saffron-1: Towards an Inference Scaling Paradigm for LLM Safety Assurance 

**Authors**: Ruizhong Qiu, Gaotang Li, Tianxin Wei, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2506.06444)  

**Abstract**: Existing safety assurance research has primarily focused on training-phase alignment to instill safe behaviors into LLMs. However, recent studies have exposed these methods' susceptibility to diverse jailbreak attacks. Concurrently, inference scaling has significantly advanced LLM reasoning capabilities but remains unexplored in the context of safety assurance. Addressing this gap, our work pioneers inference scaling for robust and effective LLM safety against emerging threats. We reveal that conventional inference scaling techniques, despite their success in reasoning tasks, perform poorly in safety contexts, even falling short of basic approaches like Best-of-N Sampling. We attribute this inefficiency to a newly identified challenge, the exploration--efficiency dilemma, arising from the high computational overhead associated with frequent process reward model (PRM) evaluations. To overcome this dilemma, we propose SAFFRON, a novel inference scaling paradigm tailored explicitly for safety assurance. Central to our approach is the introduction of a multifurcation reward model (MRM) that significantly reduces the required number of reward model evaluations. To operationalize this paradigm, we further propose: (i) a partial supervision training objective for MRM, (ii) a conservative exploration constraint to prevent out-of-distribution explorations, and (iii) a Trie-based key--value caching strategy that facilitates cache sharing across sequences during tree search. Extensive experiments validate the effectiveness of our method. Additionally, we publicly release our trained multifurcation reward model (Saffron-1) and the accompanying token-level safety reward dataset (Safety4M) to accelerate future research in LLM safety. Our code, model, and data are publicly available at this https URL , and our project homepage is at this https URL . 

---
# Enhancing Decision-Making of Large Language Models via Actor-Critic 

**Authors**: Heng Dong, Kefei Duan, Chongjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06376)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable advancements in natural language processing tasks, yet they encounter challenges in complex decision-making scenarios that require long-term reasoning and alignment with high-level objectives. Existing methods either rely on short-term auto-regressive action generation or face limitations in accurately simulating rollouts and assessing outcomes, leading to sub-optimal decisions. This paper introduces a novel LLM-based Actor-Critic framework, termed LAC, that effectively improves LLM policies with long-term action evaluations in a principled and scalable way. Our approach addresses two key challenges: (1) extracting robust action evaluations by computing Q-values via token logits associated with positive/negative outcomes, enhanced by future trajectory rollouts and reasoning; and (2) enabling efficient policy improvement through a gradient-free mechanism. Experiments across diverse environments -- including high-level decision-making (ALFWorld), low-level action spaces (BabyAI-Text), and large action spaces (WebShop) -- demonstrate the framework's generality and superiority over state-of-the-art methods. Notably, our approach achieves competitive performance using 7B/8B parameter LLMs, even outperforming baseline methods employing GPT-4 in complex tasks. These results underscore the potential of integrating structured policy optimization with LLMs' intrinsic knowledge to advance decision-making capabilities in multi-step environments. 

---
# Pairwise Calibrated Rewards for Pluralistic Alignment 

**Authors**: Daniel Halpern, Evi Micha, Ariel D. Procaccia, Itai Shapira  

**Link**: [PDF](https://arxiv.org/pdf/2506.06298)  

**Abstract**: Current alignment pipelines presume a single, universal notion of desirable behavior. However, human preferences often diverge across users, contexts, and cultures. As a result, disagreement collapses into the majority signal and minority perspectives are discounted. To address this, we propose reflecting diverse human preferences through a distribution over multiple reward functions, each inducing a distinct aligned policy. The distribution is learned directly from pairwise preference without annotator identifiers or predefined groups. Instead, annotator disagreements are treated as informative soft labels. Our central criterion is pairwise calibration: for every pair of candidate responses, the proportion of reward functions preferring one response matches the fraction of annotators with that preference. We prove that even a small outlier-free ensemble can accurately represent diverse preference distributions. Empirically, we introduce and validate a practical training heuristic to learn such ensembles, and demonstrate its effectiveness through improved calibration, implying a more faithful representation of pluralistic values. 

---
# Reward Is Enough: LLMs Are In-Context Reinforcement Learners 

**Authors**: Kefan Song, Amir Moeini, Peng Wang, Lei Gong, Rohan Chandra, Yanjun Qi, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06303)  

**Abstract**: Reinforcement learning (RL) is a human-designed framework for solving sequential decision making problems. In this work, we demonstrate that, surprisingly, RL emerges in LLM's (Large Language Model) inference time -- a phenomenon known as in-context RL (ICRL). Specifically, we propose a novel multi-round prompting framework called ICRL prompting. The goal is to prompt the LLM to complete a task. After the LLM generates a response at the current round, we give numerical scalar feedbacks for the response, called the rewards. At the next round, we prompt the LLM again with the same task and a context consisting of all previous responses and rewards. We observe that the quality of the LLM's response increases as the context grows. In other words, the LLM is able to maximize the scalar reward signal in the inference time, just like an RL algorithm. We evaluate ICRL prompting in three benchmarks (Game of 24, creative writing, and ScienceWorld) and demonstrate significant performance improvements over baseline methods such as Self-Refine and Reflexion. Surprisingly, in some experiments the reward signals are generated by the LLM itself, yet performance improvements are still observed from ICRL prompting, offering a promising paradigm for scaling test-time compute. 

---
# Mutual-Taught for Co-adapting Policy and Reward Models 

**Authors**: Tianyuan Shi, Canbin Huang, Fanqi Wan, Longguang Zhong, Ziyi Yang, Weizhou Shen, Xiaojun Quan, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06292)  

**Abstract**: During the preference optimization of large language models (LLMs), distribution shifts may arise between newly generated model samples and the data used to train the reward model (RM). This shift reduces the efficacy of the RM, which in turn negatively impacts the performance of the policy model (PM). To address this challenge, we propose Mutual-Taught, a self-training method that iteratively improves both the PM and RM without requiring additional human annotation. Our approach mirrors the expectation-maximization (EM) algorithm. In the E-step, the PM is updated using feedback from the current RM, guiding the PM toward a better approximation of the latent optimal preference distribution. In the M-step, we update the RM by constructing training data from the outputs of the PM before and after the E-step update. This process ensures that the RM adapts to the evolving policy distribution. Experimental results demonstrate that this iterative approach leads to consistent improvements in both models. Specifically, our 8B policy model, LLaMA-3-8B-Instruct-MT, achieves a length-controlled win rate of 54.1\% on AlpacaEval-2, while our 8B reward model, FsfairX-LLaMA3-RM-MT, performs on par with GPT-4o-2024-08-06 on RewardBench. 

---
# Towards Large Language Models with Self-Consistent Natural Language Explanations 

**Authors**: Sahar Admoni, Ofra Amir, Assaf Hallak, Yftah Ziser  

**Link**: [PDF](https://arxiv.org/pdf/2506.07523)  

**Abstract**: Large language models (LLMs) seem to offer an easy path to interpretability: just ask them to explain their decisions. Yet, studies show that these post-hoc explanations often misrepresent the true decision process, as revealed by mismatches in feature importance. Despite growing evidence of this inconsistency, no systematic solutions have emerged, partly due to the high cost of estimating feature importance, which limits evaluations to small datasets. To address this, we introduce the Post-hoc Self-Consistency Bank (PSCB) - a large-scale benchmark of decisions spanning diverse tasks and models, each paired with LLM-generated explanations and corresponding feature importance scores. Analysis of PSCB reveals that self-consistency scores barely differ between correct and incorrect predictions. We also show that the standard metric fails to meaningfully distinguish between explanations. To overcome this limitation, we propose an alternative metric that more effectively captures variation in explanation quality. We use it to fine-tune LLMs via Direct Preference Optimization (DPO), leading to significantly better alignment between explanations and decision-relevant features, even under domain shift. Our findings point to a scalable path toward more trustworthy, self-consistent LLMs. 

---
# GeometryZero: Improving Geometry Solving for LLM with Group Contrastive Policy Optimization 

**Authors**: Yikun Wang, Yibin Wang, Dianyi Wang, Zimian Peng, Qipeng Guo, Dacheng Tao, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07160)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable capabilities across diverse domains, particularly in mathematical reasoning, amid which geometry problem solving remains a challenging area where auxiliary construction plays a enssential role. Existing approaches either achieve suboptimal performance or rely on massive LLMs (e.g., GPT-4o), incurring massive computational costs. We posit that reinforcement learning with verifiable reward (e.g., GRPO) offers a promising direction for training smaller models that effectively combine auxiliary construction with robust geometric reasoning. However, directly applying GRPO to geometric reasoning presents fundamental limitations due to its dependence on unconditional rewards, which leads to indiscriminate and counterproductive auxiliary constructions. To address these challenges, we propose Group Contrastive Policy Optimization (GCPO), a novel reinforcement learning framework featuring two key innovations: (1) Group Contrastive Masking, which adaptively provides positive or negative reward signals for auxiliary construction based on contextual utility, and a (2) length reward that promotes longer reasoning chains. Building on GCPO, we develop GeometryZero, a family of affordable-size geometric reasoning models that judiciously determine when to employ auxiliary construction. Our extensive empirical evaluation across popular geometric benchmarks (Geometry3K, MathVista) demonstrates that GeometryZero models consistently outperform baselines (e.g. GRPO), achieving an average improvement of 4.29% across all benchmarks. 

---
# Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning 

**Authors**: LASA Team, Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, Yu Sun, Junao Shen, Chaojun Wang, Jie Tan, Deli Zhao, Tingyang Xu, Hao Zhang, Yu Rong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07044)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in understanding common visual elements, largely due to their large-scale datasets and advanced training strategies. However, their effectiveness in medical applications remains limited due to the inherent discrepancies between data and tasks in medical scenarios and those in the general domain. Concretely, existing medical MLLMs face the following critical limitations: (1) limited coverage of medical knowledge beyond imaging, (2) heightened susceptibility to hallucinations due to suboptimal data curation processes, (3) lack of reasoning capabilities tailored for complex medical scenarios. To address these challenges, we first propose a comprehensive data curation procedure that (1) efficiently acquires rich medical knowledge data not only from medical imaging but also from extensive medical texts and general-domain data; and (2) synthesizes accurate medical captions, visual question answering (VQA), and reasoning samples. As a result, we build a multimodal dataset enriched with extensive medical knowledge. Building on the curated data, we introduce our medical-specialized MLLM: Lingshu. Lingshu undergoes multi-stage training to embed medical expertise and enhance its task-solving capabilities progressively. Besides, we preliminarily explore the potential of applying reinforcement learning with verifiable rewards paradigm to enhance Lingshu's medical reasoning ability. Additionally, we develop MedEvalKit, a unified evaluation framework that consolidates leading multimodal and textual medical benchmarks for standardized, fair, and efficient model assessment. We evaluate the performance of Lingshu on three fundamental medical tasks, multimodal QA, text-based QA, and medical report generation. The results show that Lingshu consistently outperforms the existing open-source multimodal models on most tasks ... 

---
# Learning to Clarify by Reinforcement Learning Through Reward-Weighted Fine-Tuning 

**Authors**: Subhojyoti Mukherjee, Viet Dac Lai, Raghavendra Addanki, Ryan Rossi, Seunghyun Yoon, Trung Bui, Anup Rao, Jayakumar Subramanian, Branislav Kveton  

**Link**: [PDF](https://arxiv.org/pdf/2506.06964)  

**Abstract**: Question answering (QA) agents automatically answer questions posed in natural language. In this work, we learn to ask clarifying questions in QA agents. The key idea in our method is to simulate conversations that contain clarifying questions and learn from them using reinforcement learning (RL). To make RL practical, we propose and analyze offline RL objectives that can be viewed as reward-weighted supervised fine-tuning (SFT) and easily optimized in large language models. Our work stands in a stark contrast to recently proposed methods, based on SFT and direct preference optimization, which have additional hyper-parameters and do not directly optimize rewards. We compare to these methods empirically and report gains in both optimized rewards and language quality. 

---
# On the Adaptive Psychological Persuasion of Large Language Models 

**Authors**: Tianjie Ju, Yujia Chen, Hao Fei, Mong-Li Lee, Wynne Hsu, Pengzhou Cheng, Zongru Wu, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06800)  

**Abstract**: Previous work has showcased the intriguing capabilities of Large Language Models (LLMs) in instruction-following and rhetorical fluency. However, systematic exploration of their dual capabilities to autonomously persuade and resist persuasion, particularly in contexts involving psychological rhetoric, remains unexplored. In this paper, we first evaluate four commonly adopted LLMs by tasking them to alternately act as persuaders and listeners in adversarial dialogues. Empirical results show that persuader LLMs predominantly employ repetitive strategies, leading to low success rates. Then we introduce eleven comprehensive psychological persuasion strategies, finding that explicitly instructing LLMs to adopt specific strategies such as Fluency Effect and Repetition Effect significantly improves persuasion success rates. However, no ``one-size-fits-all'' strategy proves universally effective, with performance heavily dependent on contextual counterfactuals. Motivated by these observations, we propose an adaptive framework based on direct preference optimization that trains LLMs to autonomously select optimal strategies by leveraging persuasion results from strategy-specific responses as preference pairs. Experiments on three open-source LLMs confirm that the proposed adaptive psychological persuasion method effectively enables persuader LLMs to select optimal strategies, significantly enhancing their success rates while maintaining general capabilities. Our code is available at this https URL. 

---
# Play to Generalize: Learning to Reason Through Game Play 

**Authors**: Yunfei Xie, Yinsong Ma, Shiyi Lan, Alan Yuille, Junfei Xiao, Chen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.08011)  

**Abstract**: Developing generalizable reasoning capabilities in multimodal large language models (MLLMs) remains challenging. Motivated by cognitive science literature suggesting that gameplay promotes transferable cognitive skills, we propose a novel post-training paradigm, Visual Game Learning, or ViGaL, where MLLMs develop out-of-domain generalization of multimodal reasoning through playing arcade-like games. Specifically, we show that post-training a 7B-parameter MLLM via reinforcement learning (RL) on simple arcade-like games, e.g. Snake, significantly enhances its downstream performance on multimodal math benchmarks like MathVista, and on multi-discipline questions like MMMU, without seeing any worked solutions, equations, or diagrams during RL, suggesting the capture of transferable reasoning skills. Remarkably, our model outperforms specialist models tuned on multimodal reasoning data in multimodal reasoning benchmarks, while preserving the base model's performance on general visual benchmarks, a challenge where specialist models often fall short. Our findings suggest a new post-training paradigm: synthetic, rule-based games can serve as controllable and scalable pre-text tasks that unlock generalizable multimodal reasoning abilities in MLLMs. 

---
# Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models 

**Authors**: Pengyi Li, Matvey Skripkin, Alexander Zubrey, Andrey Kuznetsov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2506.06395)  

**Abstract**: Large language models (LLMs) excel at reasoning, yet post-training remains critical for aligning their behavior with task goals. Existing reinforcement learning (RL) methods often depend on costly human annotations or external reward models. We propose Reinforcement Learning via Self-Confidence (RLSC), which uses the model's own confidence as reward signals-eliminating the need for labels, preference models, or reward engineering. Applied to Qwen2.5-Math-7B with only 8 samples per question and 4 training epochs, RLSC improves accuracy by +20.10% on AIME2024, +49.40% on MATH500, and +52.50% on AMC23. RLSC offers a simple, scalable post-training method for reasoning models with minimal supervision. 

---
# Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models 

**Authors**: Mickel Liu, Liwei Jiang, Yancheng Liang, Simon Shaolei Du, Yejin Choi, Tim Althoff, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2506.07468)  

**Abstract**: Conventional language model (LM) safety alignment relies on a reactive, disjoint procedure: attackers exploit a static model, followed by defensive fine-tuning to patch exposed vulnerabilities. This sequential approach creates a mismatch -- attackers overfit to obsolete defenses, while defenders perpetually lag behind emerging threats. To address this, we propose Self-RedTeam, an online self-play reinforcement learning algorithm where an attacker and defender agent co-evolve through continuous interaction. We cast safety alignment as a two-player zero-sum game, where a single model alternates between attacker and defender roles -- generating adversarial prompts and safeguarding against them -- while a reward LM adjudicates outcomes. This enables dynamic co-adaptation. Grounded in the game-theoretic framework of zero-sum games, we establish a theoretical safety guarantee which motivates the design of our method: if self-play converges to a Nash Equilibrium, the defender will reliably produce safe responses to any adversarial input. Empirically, Self-RedTeam uncovers more diverse attacks (+21.8% SBERT) compared to attackers trained against static defenders and achieves higher robustness on safety benchmarks (e.g., +65.5% on WildJailBreak) than defenders trained against static attackers. We further propose hidden Chain-of-Thought, allowing agents to plan privately, which boosts adversarial diversity and reduces over-refusals. Our results motivate a shift from reactive patching to proactive co-evolution in LM safety training, enabling scalable, autonomous, and robust self-improvement of LMs via multi-agent reinforcement learning (MARL). 

---
# Multi-Step Visual Reasoning with Visual Tokens Scaling and Verification 

**Authors**: Tianyi Bai, Zengjie Hu, Fupeng Sun, Jiantao Qiu, Yizhen Jiang, Guangxin He, Bohan Zeng, Conghui He, Binhang Yuan, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07235)  

**Abstract**: Multi-modal large language models (MLLMs) have achieved remarkable capabilities by integrating visual perception with language understanding, enabling applications such as image-grounded dialogue, visual question answering, and scientific analysis. However, most MLLMs adopt a static inference paradigm, encoding the entire image into fixed visual tokens upfront, which limits their ability to iteratively refine understanding or adapt to context during inference. This contrasts sharply with human perception, which is dynamic, selective, and feedback-driven. In this work, we introduce a novel framework for inference-time visual token scaling that enables MLLMs to perform iterative, verifier-guided reasoning over visual content. We formulate the problem as a Markov Decision Process, involving a reasoner that proposes visual actions and a verifier, which is trained via multi-step Direct Preference Optimization (DPO), that evaluates these actions and determines when reasoning should terminate. To support this, we present a new dataset, VTS, comprising supervised reasoning trajectories (VTS-SFT) and preference-labeled reasoning comparisons (VTS-DPO). Our method significantly outperforms existing approaches across diverse visual reasoning benchmarks, offering not only improved accuracy but also more interpretable and grounded reasoning processes. These results demonstrate the promise of dynamic inference mechanisms for enabling fine-grained, context-aware visual reasoning in next-generation MLLMs. 

---
