# Multi-Agent Evolve: LLM Self-Improve through Co-evolution 

**Authors**: Yixing Chen, Yiding Wang, Siqi Zhu, Haofei Yu, Tao Feng, Muhan Zhan, Mostofa Patwary, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2510.23595)  

**Abstract**: Reinforcement Learning (RL) has demonstrated significant potential in enhancing the reasoning capabilities of large language models (LLMs). However, the success of RL for LLMs heavily relies on human-curated datasets and verifiable rewards, which limit their scalability and generality. Recent Self-Play RL methods, inspired by the success of the paradigm in games and Go, aim to enhance LLM reasoning capabilities without human-annotated data. However, their methods primarily depend on a grounded environment for feedback (e.g., a Python interpreter or a game engine); extending them to general domains remains challenging. To address these challenges, we propose Multi-Agent Evolve (MAE), a framework that enables LLMs to self-evolve in solving diverse tasks, including mathematics, reasoning, and general knowledge Q&A. The core design of MAE is based on a triplet of interacting agents (Proposer, Solver, Judge) that are instantiated from a single LLM, and applies reinforcement learning to optimize their behaviors. The Proposer generates questions, the Solver attempts solutions, and the Judge evaluates both while co-evolving. Experiments on Qwen2.5-3B-Instruct demonstrate that MAE achieves an average improvement of 4.54% on multiple benchmarks. These results highlight MAE as a scalable, data-efficient method for enhancing the general reasoning abilities of LLMs with minimal reliance on human-curated supervision. 

---
# Emotion-Coherent Reasoning for Multimodal LLMs via Emotional Rationale Verifier 

**Authors**: Hyeongseop Rha, Jeong Hun Yeo, Yeonju Kim, Yong Man Ro  

**Link**: [PDF](https://arxiv.org/pdf/2510.23506)  

**Abstract**: The recent advancement of Multimodal Large Language Models (MLLMs) is transforming human-computer interaction (HCI) from surface-level exchanges into more nuanced and emotionally intelligent communication. To realize this shift, emotion understanding becomes essential allowing systems to capture subtle cues underlying user intent. Furthermore, providing faithful explanations for predicted emotions is crucial to ensure interpretability and build user trust. However, current MLLM-based methods often generate emotion explanations that diverge from the target labels and sometimes even contradict their own predicted emotions. This inconsistency poses a critical risk for misunderstanding and erodes reliability in interactive settings. To address this, we propose a novel approach: the Emotional Rationale Verifier (ERV) and an Explanation Reward. Our method guides the model to produce reasoning that is explicitly consistent with the target emotion during multimodal emotion recognition without modifying the model architecture or requiring additional paired video-description annotations. Our method significantly improves faithful explanation-prediction consistency and explanation emotion accuracy on the MAFW and DFEW datasets. Through extensive experiments and human evaluations, we show that our approach not only enhances alignment between explanation and prediction but also empowers MLLMs to deliver emotionally coherent, trustworthy interactions, marking a key step toward truly human-like HCI systems. 

---
# HRM-Agent: Training a recurrent reasoning model in dynamic environments using reinforcement learning 

**Authors**: Long H Dang, David Rawlinson  

**Link**: [PDF](https://arxiv.org/pdf/2510.22832)  

**Abstract**: The Hierarchical Reasoning Model (HRM) has impressive reasoning abilities given its small size, but has only been applied to supervised, static, fully-observable problems. One of HRM's strengths is its ability to adapt its computational effort to the difficulty of the problem. However, in its current form it cannot integrate and reuse computation from previous time-steps if the problem is dynamic, uncertain or partially observable, or be applied where the correct action is undefined, characteristics of many real-world problems.
This paper presents HRM-Agent, a variant of HRM trained using only reinforcement learning. We show that HRM can learn to navigate to goals in dynamic and uncertain maze environments. Recent work suggests that HRM's reasoning abilities stem from its recurrent inference process. We explore the dynamics of the recurrent inference process and find evidence that it is successfully reusing computation from earlier environment time-steps. 

---
# Learning "Partner-Aware" Collaborators in Multi-Party Collaboration 

**Authors**: Abhijnan Nath, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2510.22462)  

**Abstract**: Large Language Models (LLMs) are increasingly bring deployed in agentic settings where they act as collaborators with humans. Therefore, it is increasingly important to be able to evaluate their abilities to collaborate effectively in multi-turn, multi-party tasks. In this paper, we build on the AI alignment and safe interruptability literature to offer novel theoretical insights on collaborative behavior between LLM-driven collaborator agents and an intervention agent. Our goal is to learn an ideal partner-aware collaborator that increases the group's common-ground (CG)-alignment on task-relevant propositions-by intelligently collecting information provided in interventions by a partner this http URL show how LLM agents trained using standard RLHF and related approaches are naturally inclined to ignore possibly well-meaning interventions, which makes increasing group common ground non-trivial in this setting. We employ a two-player Modified-Action MDP to examine this suboptimal behavior of standard AI agents, and propose Interruptible Collaborative Roleplayer (ICR)-a novel partner-aware learning algorithm to train CG-optimal collaborators. Experiments on multiple collaborative task environments show that ICR, on average, is more capable of promoting successful CG convergence and exploring more diverse solutions in such tasks. 

---
# PACR: Progressively Ascending Confidence Reward for LLM Reasoning 

**Authors**: Eunseop Yoon, Hee Suk Yoon, Jaehyun Jang, SooHwan Eom, Qi Dai, Chong Luo, Mark A. Hasegawa-Johnson, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2510.22255)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has significantly improved LLM reasoning, but its sparse, outcome-based reward provides no guidance for intermediate steps, slowing exploration. We propose Progressively Ascending Confidence Reward (PACR), a dense, model-intrinsic reward computed directly from the model's evolving belief in the correct answer. PACR encodes the inductive bias that, along a well-formed reasoning trajectory, the probability of the ground-truth answer should have a generally ascending trend. We provide empirical and theoretical analysis validating that such an inductive bias constrains the exploration search space to regions richer in logically sound reasoning. We demonstrate that PACR accelerates exploration, reaches reward saturation with fewer trajectories, and yields improvements on multiple benchmarks. Our results suggest that dense, model-intrinsic shaping signals can make RLVR training more effective and reliable. 

---
# Foundation of Intelligence: Review of Math Word Problems from Human Cognition Perspective 

**Authors**: Zhenya Huang, Jiayu Liu, Xin Lin, Zhiyuan Ma, Shangzi Xue, Tong Xiao, Qi Liu, Yee Whye Teh, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.21999)  

**Abstract**: Math word problem (MWP) serves as a fundamental research topic in artificial intelligence (AI) dating back to 1960s. This research aims to advance the reasoning abilities of AI by mirroring the human-like cognitive intelligence. The mainstream technological paradigm has evolved from the early rule-based methods, to deep learning models, and is rapidly advancing towards large language models. However, the field still lacks a systematic taxonomy for the MWP survey along with a discussion of current development trends. Therefore, in this paper, we aim to comprehensively review related research in MWP solving through the lens of human cognition, to demonstrate how recent AI models are advancing in simulating human cognitive abilities. Specifically, we summarize 5 crucial cognitive abilities for MWP solving, including Problem Understanding, Logical Organization, Associative Memory, Critical Thinking, and Knowledge Learning. Focused on these abilities, we review two mainstream MWP models in recent 10 years: neural network solvers, and LLM based solvers, and discuss the core human-like abilities they demonstrated in their intricate problem-solving process. Moreover, we rerun all the representative MWP solvers and supplement their performance on 5 mainstream benchmarks for a unified comparison. To the best of our knowledge, this survey first comprehensively analyzes the influential MWP research of the past decade from the perspective of human reasoning cognition and provides an integrative overall comparison across existing approaches. We hope it can inspire further research in AI reasoning. Our repository is released on this https URL. 

---
# Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning 

**Authors**: Ran Xu, Jingjing Chen, Jiayu Ye, Yu Wu, Jun Yan, Carl Yang, Hongkun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23038)  

**Abstract**: Large Language Models (LLMs) are widely used as judges to evaluate response quality, providing a scalable alternative to human evaluation. However, most LLM judges operate solely on intrinsic text-based reasoning, limiting their ability to verify complex constraints or perform accurate computation. Motivated by the success of tool-integrated reasoning (TIR) in numerous tasks, we propose TIR-Judge, an end-to-end RL framework for training LLM judges that integrates a code executor for precise evaluation. TIR-Judge is built on three principles: (i) diverse training across verifiable and non-verifiable domains, (ii) flexible judgment formats (pointwise, pairwise, listwise), and (iii) iterative RL that bootstraps directly from the initial model without distillation. On seven public benchmarks, TIR-Judge surpasses strong reasoning-based judges by up to 6.4% (pointwise) and 7.7% (pairwise), and achieves listwise performance comparable to Claude-Opus-4 despite having only 8B parameters. Remarkably, TIR-Judge-Zero - trained entirely without distilled judge trajectories, matches the performance of distilled variants, demonstrating that tool-augmented judges can self-evolve through iterative reinforcement learning. 

---
# Think before Recommendation: Autonomous Reasoning-enhanced Recommender 

**Authors**: Xiaoyu Kong, Junguang Jiang, Bin Liu, Ziru Xu, Han Zhu, Jian Xu, Bo Zheng, Jiancan Wu, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23077)  

**Abstract**: The core task of recommender systems is to learn user preferences from historical user-item interactions. With the rapid development of large language models (LLMs), recent research has explored leveraging the reasoning capabilities of LLMs to enhance rating prediction tasks. However, existing distillation-based methods suffer from limitations such as the teacher model's insufficient recommendation capability, costly and static supervision, and superficial transfer of reasoning ability. To address these issues, this paper proposes RecZero, a reinforcement learning (RL)-based recommendation paradigm that abandons the traditional multi-model and multi-stage distillation approach. Instead, RecZero trains a single LLM through pure RL to autonomously develop reasoning capabilities for rating prediction. RecZero consists of two key components: (1) "Think-before-Recommendation" prompt construction, which employs a structured reasoning template to guide the model in step-wise analysis of user interests, item features, and user-item compatibility; and (2) rule-based reward modeling, which adopts group relative policy optimization (GRPO) to compute rewards for reasoning trajectories and optimize the LLM. Additionally, the paper explores a hybrid paradigm, RecOne, which combines supervised fine-tuning with RL, initializing the model with cold-start reasoning samples and further optimizing it with RL. Experimental results demonstrate that RecZero and RecOne significantly outperform existing baseline methods on multiple benchmark datasets, validating the superiority of the RL paradigm in achieving autonomous reasoning-enhanced recommender systems. 

---
# CityRiSE: Reasoning Urban Socio-Economic Status in Vision-Language Models via Reinforcement Learning 

**Authors**: Tianhui Liu, Hetian Pang, Xin Zhang, Jie Feng, Yong Li, Pan Hui  

**Link**: [PDF](https://arxiv.org/pdf/2510.22282)  

**Abstract**: Harnessing publicly available, large-scale web data, such as street view and satellite imagery, urban socio-economic sensing is of paramount importance for achieving global sustainable development goals. With the emergence of Large Vision-Language Models (LVLMs), new opportunities have arisen to solve this task by treating it as a multi-modal perception and understanding problem. However, recent studies reveal that LVLMs still struggle with accurate and interpretable socio-economic predictions from visual data. To address these limitations and maximize the potential of LVLMs, we introduce \textbf{CityRiSE}, a novel framework for \textbf{R}eason\textbf{i}ng urban \textbf{S}ocio-\textbf{E}conomic status in LVLMs through pure reinforcement learning (RL). With carefully curated multi-modal data and verifiable reward design, our approach guides the LVLM to focus on semantically meaningful visual cues, enabling structured and goal-oriented reasoning for generalist socio-economic status prediction. Experiments demonstrate that CityRiSE with emergent reasoning process significantly outperforms existing baselines, improving both prediction accuracy and generalization across diverse urban contexts, particularly for prediction on unseen cities and unseen indicators. This work highlights the promise of combining RL and LVLMs for interpretable and generalist urban socio-economic sensing. 

---
# Token-Level Inference-Time Alignment for Vision-Language Models 

**Authors**: Kejia Chen, Jiawen Zhang, Jiacong Hu, Kewei Gao, Jian Lou, Zunlei Feng, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.21794)  

**Abstract**: Vision-Language Models (VLMs) have become essential backbones of modern multimodal intelligence, yet their outputs remain prone to hallucination-plausible text misaligned with visual inputs. Existing alignment approaches often rely on expensive fine-tuning with annotated preference data or sequence-level inference strategies that provide only coarse, delayed feedback. To overcome these limitations, we present TITA (Token-level Inference-Time Alignment), a lightweight framework that freezes the base VLM and instead trains a reward model to approximate its distribution. During inference, implicit preference signals are extracted as log-probability ratios between the reward model and the target VLM, yielding dense autoregressive feedback. This formulation can be viewed as an inference-time variant of Direct Preference Optimization (DPO), providing token-level corrective signals without retraining the backbone. Extensive evaluations on LLaVA-1.5-7B and 13B show consistent gains across 12 benchmarks, with improvements of 8.6% on MMVet and 6.7% on POPE, indicating stronger general understanding and reduced hallucinations. Additional experiments on Qwen2.5-VL-7B and DeepSeek-VL2-27.5B show comparable gains, especially in hallucination reduction and VQA accuracy, while incurring negligible inference overhead. 

---
# Think Twice: Branch-and-Rethink Reasoning Reward Model 

**Authors**: Yizhu Jiao, Jiaqi Zeng, Julien Veron Vialard, Oleksii Kuchaiev, Jiawei Han, Olivier Delalleau  

**Link**: [PDF](https://arxiv.org/pdf/2510.23596)  

**Abstract**: Large language models (LLMs) increasingly rely on thinking models that externalize intermediate steps and allocate extra test-time compute, with think-twice strategies showing that a deliberate second pass can elicit stronger reasoning. In contrast, most reward models (RMs) still compress many quality dimensions into a single scalar in one shot, a design that induces judgment diffusion: attention spreads across evaluation criteria, yielding diluted focus and shallow analysis. We introduce branch-and-rethink (BR-RM), a two-turn RM that transfers the think-twice principle to reward modeling. Turn 1 performs adaptive branching, selecting a small set of instance-critical dimensions (such as factuality and safety) and sketching concise, evidence-seeking hypotheses. Turn 2 executes branch-conditioned rethinking, a targeted reread that tests those hypotheses and scrutinizes only what matters most. We train with GRPO-style reinforcement learning over structured two-turn traces using a simple binary outcome reward with strict format checks, making the approach compatible with standard RLHF pipelines. By converting all-at-oncescoringintofocused, second-lookreasoning, BR-RMreducesjudgmentdiffusionandimproves sensitivity to subtle yet consequential errors while remaining practical and scalable. Experimental results demonstrate that our model achieves state-of-the-art performance on three challenging reward modeling benchmarks across diverse domains. The code and the model will be released soon. 

---
# Code Aesthetics with Agentic Reward Feedback 

**Authors**: Bang Xiao, Lingjie Jiang, Shaohan Huang, Tengchao Lv, Yupan Huang, Xun Wu, Lei Cui, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.23272)  

**Abstract**: Large Language Models (LLMs) have become valuable assistants for developers in code-related tasks. While LLMs excel at traditional programming tasks such as code generation and bug fixing, they struggle with visually-oriented coding tasks, often producing suboptimal aesthetics. In this paper, we introduce a new pipeline to enhance the aesthetic quality of LLM-generated code. We first construct AesCode-358K, a large-scale instruction-tuning dataset focused on code aesthetics. Next, we propose agentic reward feedback, a multi-agent system that evaluates executability, static aesthetics, and interactive aesthetics. Building on this, we develop GRPO-AR, which integrates these signals into the GRPO algorithm for joint optimization of functionality and code aesthetics. Finally, we develop OpenDesign, a benchmark for assessing code aesthetics. Experimental results show that combining supervised fine-tuning on AesCode-358K with reinforcement learning using agentic reward feedback significantly improves performance on OpenDesign and also enhances results on existing benchmarks such as PandasPlotBench. Notably, our AesCoder-4B surpasses GPT-4o and GPT-4.1, and achieves performance comparable to large open-source models with 480B-685B parameters, underscoring the effectiveness of our approach. 

---
# VEHME: A Vision-Language Model For Evaluating Handwritten Mathematics Expressions 

**Authors**: Thu Phuong Nguyen, Duc M. Nguyen, Hyotaek Jeon, Hyunwook Lee, Hyunmin Song, Sungahn Ko, Taehwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.22798)  

**Abstract**: Automatically assessing handwritten mathematical solutions is an important problem in educational technology with practical applications, but it remains a significant challenge due to the diverse formats, unstructured layouts, and symbolic complexity of student work. To address this challenge, we introduce VEHME-a Vision-Language Model for Evaluating Handwritten Mathematics Expressions-designed to assess open-form handwritten math responses with high accuracy and interpretable reasoning traces. VEHME integrates a two-phase training pipeline: (i) supervised fine-tuning using structured reasoning data, and (ii) reinforcement learning that aligns model outputs with multi-dimensional grading objectives, including correctness, reasoning depth, and error localization. To enhance spatial understanding, we propose an Expression-Aware Visual Prompting Module, trained on our synthesized multi-line math expressions dataset to robustly guide attention in visually heterogeneous inputs. Evaluated on AIHub and FERMAT datasets, VEHME achieves state-of-the-art performance among open-source models and approaches the accuracy of proprietary systems, demonstrating its potential as a scalable and accessible tool for automated math assessment. Our training and experiment code is publicly available at our GitHub repository. 

---
# Scalable Supervising Software Agents with Patch Reasoner 

**Authors**: Junjielong Xu, Boyin Tan, Xiaoyuan Liu, Chao Peng, Pengfei Gao, Pinjia He  

**Link**: [PDF](https://arxiv.org/pdf/2510.22775)  

**Abstract**: While large language model agents have advanced software engineering tasks, the unscalable nature of existing test-based supervision is limiting the potential improvement of data scaling. The reason is twofold: (1) building and running test sandbox is rather heavy and fragile, and (2) data with high-coverage tests is naturally rare and threatened by test hacking via edge cases. In this paper, we propose R4P, a patch verifier model to provide scalable rewards for training and testing SWE agents via reasoning. We consider that patch verification is fundamentally a reasoning task, mirroring how human repository maintainers review patches without writing and running new reproduction tests. To obtain sufficient reference and reduce the risk of reward hacking, R4P uses a group-wise objective for RL training, enabling it to verify multiple patches against each other's modification and gain a dense reward for stable training. R4P achieves 72.2% Acc. for verifying patches from SWE-bench-verified, surpassing OpenAI o3. To demonstrate R4P's practicality, we design and train a lite scaffold, Mini-SE, with pure reinforcement learning where all rewards are derived from R4P. As a result, Mini-SE achieves 26.2% Pass@1 on SWE-bench-verified, showing a 10.0% improvement over the original Qwen3-32B. This can be further improved to 32.8% with R4P for test-time scaling. Furthermore, R4P verifies patches within a second, 50x faster than testing on average. The stable scaling curves of rewards and accuracy along with high efficiency reflect R4P's practicality. 

---
# OlaMind: Towards Human-Like and Hallucination-Safe Customer Service for Retrieval-Augmented Dialogue 

**Authors**: Tianhong Gao, Jundong Shen, Bei Shi, Jiapeng Wang, Ying Ju, Junfeng Yao, Jiao Ran, Yong Zhang, Lin Dong, Huiyu Yu, Tingting Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.22143)  

**Abstract**: Intelligent customer service (ICS) systems via retrieval-augmented generation (RAG) have been widely adopted in Web-based domains such as social platforms and e-commerce, achieving remarkable improvements in automation and efficiency. However, notable limitations still remain: these systems are prone to hallucinations and often generate rigid, mechanical responses, which can introduce business risks and undermine user experience, especially in Web-based customer service interactions under the RAG scenarios. In this paper, we introduce OlaMind, a human-like and hallucination-safe customer service framework for retrieval-augmented dialogue. Specifically, it first leverages a Learn-to-Think stage to learn the reasoning processes and response strategies from human experts, and then employs a Learn-to-Respond stage to perform cold-start supervised fine-tuning (SFT) combined with reinforcement learning (RL) for basic-to-hard self-refinement. Our method significantly enhances human-likeness and naturalness while effectively mitigating hallucinations and critical business risks. We have conducted large-scale online A/B experiments in an industry-level social customer service setting, and extensive experimental results show that OlaMind achieves significant cumulative relative improvements with intelligent resolution rates +28.92%/+18.42% and human takeover rate -6.08%/-7.12% in community-support/livestream-interaction scenarios, respectively, which highlights its consistent effectiveness across diverse real-world applications. The code and data will be publicly available. 

---
# Offline Preference Optimization via Maximum Marginal Likelihood Estimation 

**Authors**: Saeed Najafi, Alona Fyshe  

**Link**: [PDF](https://arxiv.org/pdf/2510.22881)  

**Abstract**: Aligning Large Language Models (LLMs) with human preferences is crucial, but standard methods like Reinforcement Learning from Human Feedback (RLHF) are often complex and unstable. In this work, we propose a new, simpler approach that recasts alignment through the lens of Maximum Marginal Likelihood (MML) estimation. Our new MML based Preference Optimization (MMPO) maximizes the marginal log-likelihood of a preferred text output, using the preference pair as samples for approximation, and forgoes the need for both an explicit reward model and entropy maximization. We theoretically demonstrate that MMPO implicitly performs preference optimization, producing a weighted gradient that naturally up-weights chosen responses over rejected ones. Across models ranging from 135M to 8B parameters, we empirically show that MMPO: 1) is more stable with respect to the hyperparameter $\beta$ compared to alternative baselines, and 2) achieves competitive or superior preference alignment while better preserving the base model's general language capabilities. Through a series of ablation experiments, we show that this improved performance is indeed attributable to MMPO's implicit preference optimization within the gradient updates. 

---
