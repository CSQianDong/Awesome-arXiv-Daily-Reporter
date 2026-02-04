# Search-R2: Enhancing Search-Integrated Reasoning via Actor-Refiner Collaboration 

**Authors**: Bowei He, Minda Hu, Zenan Xu, Hongru Wang, Licheng Zong, Yankai Chen, Chen Ma, Xue Liu, Pluto Zhou, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2602.03647)  

**Abstract**: Search-integrated reasoning enables language agents to transcend static parametric knowledge by actively querying external sources. However, training these agents via reinforcement learning is hindered by the multi-scale credit assignment problem: existing methods typically rely on sparse, trajectory-level rewards that fail to distinguish between high-quality reasoning and fortuitous guesses, leading to redundant or misleading search behaviors. To address this, we propose Search-R2, a novel Actor-Refiner collaboration framework that enhances reasoning through targeted intervention, with both components jointly optimized during training. Our approach decomposes the generation process into an Actor, which produces initial reasoning trajectories, and a Meta-Refiner, which selectively diagnoses and repairs flawed steps via a 'cut-and-regenerate' mechanism. To provide fine-grained supervision, we introduce a hybrid reward design that couples outcome correctness with a dense process reward quantifying the information density of retrieved evidence. Theoretically, we formalize the Actor-Refiner interaction as a smoothed mixture policy, proving that selective correction yields strict performance gains over strong baselines. Extensive experiments across various general and multi-hop QA datasets demonstrate that Search-R2 consistently outperforms strong RAG and RL-based baselines across model scales, achieving superior reasoning accuracy with minimal overhead. 

---
# IntentRL: Training Proactive User-intent Agents for Open-ended Deep Research via Reinforcement Learning 

**Authors**: Haohao Luo, Zexi Li, Yuexiang Xie, Wenhao Zhang, Yaliang Li, Ying Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.03468)  

**Abstract**: Deep Research (DR) agents extend Large Language Models (LLMs) beyond parametric knowledge by autonomously retrieving and synthesizing evidence from large web corpora into long-form reports, enabling a long-horizon agentic paradigm. However, unlike real-time conversational assistants, DR is computationally expensive and time-consuming, creating an autonomy-interaction dilemma: high autonomy on ambiguous user queries often leads to prolonged execution with unsatisfactory outcomes. To address this, we propose IntentRL, a framework that trains proactive agents to clarify latent user intents before starting long-horizon research. To overcome the scarcity of open-ended research data, we introduce a scalable pipeline that expands a few seed samples into high-quality dialogue turns via a shallow-to-deep intent refinement graph. We further adopt a two-stage reinforcement learning (RL) strategy: Stage I applies RL on offline dialogues to efficiently learn general user-interaction behavior, while Stage II uses the trained agent and a user simulator for online rollouts to strengthen adaptation to diverse user feedback. Extensive experiments show that IntentRL significantly improves both intent hit rate and downstream task performance, outperforming the built-in clarify modules of closed-source DR agents and proactive LLM baselines. 

---
# MentalSeek-Dx: Towards Progressive Hypothetico-Deductive Reasoning for Real-world Psychiatric Diagnosis 

**Authors**: Xiao Sun, Yuming Yang, Junnan Zhu, Jiang Zhong, Xinyu Zhou, Kaiwen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.03340)  

**Abstract**: Mental health disorders represent a burgeoning global public health challenge. While Large Language Models (LLMs) have demonstrated potential in psychiatric assessment, their clinical utility is severely constrained by benchmarks that lack ecological validity and fine-grained diagnostic supervision. To bridge this gap, we introduce \textbf{MentalDx Bench}, the first benchmark dedicated to disorder-level psychiatric diagnosis within real-world clinical settings. Comprising 712 de-identified electronic health records annotated by board-certified psychiatrists under ICD-11 guidelines, the benchmark covers 76 disorders across 16 diagnostic categories. Evaluation of 18 LLMs reveals a critical \textit{paradigm misalignment}: strong performance at coarse diagnostic categorization contrasts with systematic failure at disorder-level diagnosis, underscoring a gap between pattern-based modeling and clinical hypothetico-deductive reasoning. In response, we propose \textbf{MentalSeek-Dx}, a medical-specialized LLM trained to internalize this clinical reasoning process through supervised trajectory construction and curriculum-based reinforcement learning. Experiments on MentalDx Bench demonstrate that MentalSeek-Dx achieves state-of-the-art (SOTA) performance with only 14B parameters, establishing a clinically grounded framework for reliable psychiatric diagnosis. 

---
# Accordion-Thinking: Self-Regulated Step Summaries for Efficient and Readable LLM Reasoning 

**Authors**: Zhicheng Yang, Zhijiang Guo, Yinya Huang, Yongxin Wang, Wenlei Shi, Yiwei Wang, Xiaodan Liang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03249)  

**Abstract**: Scaling test-time compute via long Chain-ofThought unlocks remarkable gains in reasoning capabilities, yet it faces practical limits due to the linear growth of KV cache and quadratic attention complexity. In this paper, we introduce Accordion-Thinking, an end-to-end framework where LLMs learn to self-regulate the granularity of the reasoning steps through dynamic summarization. This mechanism enables a Fold inference mode, where the model periodically summarizes its thought process and discards former thoughts to reduce dependency on historical tokens. We apply reinforcement learning to incentivize this capability further, uncovering a critical insight: the accuracy gap between the highly efficient Fold mode and the exhaustive Unfold mode progressively narrows and eventually vanishes over the course of training. This phenomenon demonstrates that the model learns to encode essential reasoning information into compact summaries, achieving effective compression of the reasoning context. Our Accordion-Thinker demonstrates that with learned self-compression, LLMs can tackle complex reasoning tasks with minimal dependency token overhead without compromising solution quality, and it achieves a 3x throughput while maintaining accuracy on a 48GB GPU memory configuration, while the structured step summaries provide a human-readable account of the reasoning process. 

---
# Agentic Proposing: Enhancing Large Language Model Reasoning via Compositional Skill Synthesis 

**Authors**: Zhengbo Jiao, Shaobo Wang, Zifan Zhang, Xuan Ren, Wei Wang, Bing Zhao, Hu Wei, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03279)  

**Abstract**: Advancing complex reasoning in large language models relies on high-quality, verifiable datasets, yet human annotation remains cost-prohibitive and difficult to scale. Current synthesis paradigms often face a recurring trade-off: maintaining structural validity typically restricts problem complexity, while relaxing constraints to increase difficulty frequently leads to inconsistent or unsolvable instances. To address this, we propose Agentic Proposing, a framework that models problem synthesis as a goal-driven sequential decision process where a specialized agent dynamically selects and composes modular reasoning skills. Through an iterative workflow of internal reflection and tool-use, we develop the Agentic-Proposer-4B using Multi-Granularity Policy Optimization (MGPO) to generate high-precision, verifiable training trajectories across mathematics, coding, and science. Empirical results demonstrate that downstream solvers trained on agent-synthesized data significantly outperform leading baselines and exhibit robust cross-domain generalization. Notably, a 30B solver trained on only 11,000 synthesized trajectories achieves a state-of-the-art 91.6% accuracy on AIME25, rivaling frontier-scale proprietary models such as GPT-5 and proving that a small volume of high-quality synthetic signals can effectively substitute for massive human-curated datasets. 

---
# VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models 

**Authors**: Woojin Kim, Sieun Hyeon, Jusang Oh, Jaeyoung Do  

**Link**: [PDF](https://arxiv.org/pdf/2602.03160)  

**Abstract**: Aligning Large Language Models (LLMs) with the diverse spectrum of human values remains a central challenge: preference-based methods often fail to capture deeper motivational principles. Value-based approaches offer a more principled path, yet three gaps persist: extraction often ignores hierarchical structure, evaluation detects presence but not calibrated intensity, and the steerability of LLMs at controlled intensities remains insufficiently understood. To address these limitations, we introduce VALUEFLOW, the first unified framework that spans extraction, evaluation, and steering with calibrated intensity control. The framework integrates three components: (i) HIVES, a hierarchical value embedding space that captures intra- and cross-theory value structure; (ii) the Value Intensity DataBase (VIDB), a large-scale resource of value-labeled texts with intensity estimates derived from ranking-based aggregation; and (iii) an anchor-based evaluator that produces consistent intensity scores for model outputs by ranking them against VIDB panels. Using VALUEFLOW, we conduct a comprehensive large-scale study across ten models and four value theories, identifying asymmetries in steerability and composition laws for multi-value control. This paper establishes a scalable infrastructure for evaluating and controlling value intensity, advancing pluralistic alignment of LLMs. 

---
# RC-GRPO: Reward-Conditioned Group Relative Policy Optimization for Multi-Turn Tool Calling Agents 

**Authors**: Haitian Zhong, Jixiu Zhai, Lei Song, Jiang Bian, Qiang Liu, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2602.03025)  

**Abstract**: Multi-turn tool calling is challenging for Large Language Models (LLMs) because rewards are sparse and exploration is expensive. A common recipe, SFT followed by GRPO, can stall when within-group reward variation is low (e.g., more rollouts in a group receive the all 0 or all 1 reward), making the group-normalized advantage uninformative and yielding vanishing updates. To address this problem, we propose RC-GRPO (Reward-Conditioned Group Relative Policy Optimization), which treats exploration as a controllable steering problem via discrete reward tokens. We first fine-tune a Reward-Conditioned Trajectory Policy (RCTP) on mixed-quality trajectories with reward goal special tokens (e.g., <|high_reward|>, <|low_reward|>) injected into the prompts, enabling the model to learn how to generate distinct quality trajectories on demand. Then during RL, we sample diverse reward tokens within each GRPO group and condition rollouts on the sampled token to improve within-group diversity, improving advantage gains. On the Berkeley Function Calling Leaderboard v4 (BFCLv4) multi-turn benchmark, our method yields consistently improved performance than baselines, and the performance on Qwen-2.5-7B-Instruct even surpasses all closed-source API models. 

---
# ATLAS : Adaptive Self-Evolutionary Research Agent with Task-Distributed Multi-LLM Supporters 

**Authors**: Ujin Jeon, Jiyong Kwon, Madison Ann Sullivan, Caleb Eunho Lee, Guang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.02709)  

**Abstract**: Recent multi-LLM agent systems perform well in prompt optimization and automated problem-solving, but many either keep the solver frozen after fine-tuning or rely on a static preference-optimization loop, which becomes intractable for long-horizon tasks. We propose ATLAS (Adaptive Task-distributed Learning for Agentic Self-evolution), a task-distributed framework that iteratively develops a lightweight research agent while delegating complementary roles to specialized supporter agents for exploration, hyperparameter tuning, and reference policy management. Our core algorithm, Evolving Direct Preference Optimization (EvoDPO), adaptively updates the phase-indexed reference policy. We provide a theoretical regret analysis for a preference-based contextual bandit under concept drift. In addition, experiments were conducted on non-stationary linear contextual bandits and scientific machine learning (SciML) loss reweighting for the 1D Burgers' equation. Both results show that ATLAS improves stability and performance over a static single-agent baseline. 

---
# Bridging Online and Offline RL: Contextual Bandit Learning for Multi-Turn Code Generation 

**Authors**: Ziru Chen, Dongdong Chen, Ruinan Jin, Yingbin Liang, Yujia Xie, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.03806)  

**Abstract**: Recently, there have been significant research interests in training large language models (LLMs) with reinforcement learning (RL) on real-world tasks, such as multi-turn code generation. While online RL tends to perform better than offline RL, its higher training cost and instability hinders wide adoption. In this paper, we build on the observation that multi-turn code generation can be formulated as a one-step recoverable Markov decision process and propose contextual bandit learning with offline trajectories (Cobalt), a new method that combines the benefits of online and offline RL. Cobalt first collects code generation trajectories using a reference LLM and divides them into partial trajectories as contextual prompts. Then, during online bandit learning, the LLM is trained to complete each partial trajectory prompt through single-step code generation. Cobalt outperforms two multi-turn online RL baselines based on GRPO and VeRPO, and substantially improves R1-Distill 8B and Qwen3 8B by up to 9.0 and 6.2 absolute Pass@1 scores on LiveCodeBench. Also, we analyze LLMs' in-context reward hacking behaviors and augment Cobalt training with perturbed trajectories to mitigate this issue. Overall, our results demonstrate Cobalt as a promising solution for iterative decision-making tasks like multi-turn code generation. Our code and data are available at this https URL. 

---
# Not All Negative Samples Are Equal: LLMs Learn Better from Plausible Reasoning 

**Authors**: Zixiang Di, Jinyi Han, Shuo Zhang, Ying Liao, Zhi Li, Xiaofeng Ji, Yongqi Wang, Zheming Yang, Ming Gao, Bingdong Li, Jie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03516)  

**Abstract**: Learning from negative samples holds great promise for improving Large Language Model (LLM) reasoning capability, yet existing methods treat all incorrect responses as equally informative, overlooking the crucial role of sample quality. To address this, we propose Plausible Negative Samples (PNS), a method that synthesizes high-quality negative samples exhibiting expected format and structural coherence while ultimately yielding incorrect answers. PNS trains a dedicated model via reverse reinforcement learning (RL) guided by a composite reward combining format compliance, accuracy inversion, reward model assessment, and chain-of-thought evaluation, generating responses nearly indistinguishable from correct solutions. We further validate PNS as a plug-and-play data source for preference optimization across three backbone models on seven mathematical reasoning benchmarks. Results demonstrate that PNS consistently outperforms other negative sample synthesis methods, achieving an average improvement of 2.03% over RL-trained models. 

---
# Socratic-Geo: Synthetic Data Generation and Geometric Reasoning via Multi-Agent Interaction 

**Authors**: Zhengbo Jiao, Shaobo Wang, Zifan Zhang, Wei Wang, Bing Zhao, Hu Wei, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03414)  

**Abstract**: Multimodal Large Language Models (MLLMs) have significantly advanced vision-language understanding. However, even state-of-the-art models struggle with geometric reasoning, revealing a critical bottleneck: the extreme scarcity of high-quality image-text pairs. Human annotation is prohibitively expensive, while automated methods fail to ensure fidelity and training effectiveness. Existing approaches either passively adapt to available images or employ inefficient random exploration with filtering, decoupling generation from learning needs. We propose Socratic-Geo, a fully autonomous framework that dynamically couples data synthesis with model learning through multi-agent interaction. The Teacher agent generates parameterized Python scripts with reflective feedback (Reflect for solvability, RePI for visual validity), ensuring image-text pair purity. The Solver agent optimizes reasoning through preference learning, with failure paths guiding Teacher's targeted augmentation. Independently, the Generator learns image generation capabilities on accumulated "image-code-instruction" triplets, distilling programmatic drawing intelligence into visual generation. Starting from only 108 seed problems, Socratic-Solver achieves 49.11 on six benchmarks using one-quarter of baseline data, surpassing strong baselines by 2.43 points. Socratic-Generator achieves 42.4% on GenExam, establishing new state-of-the-art for open-source models, surpassing Seedream-4.0 (39.8%) and approaching Gemini-2.5-Flash-Image (43.1%). 

---
# Entropy-Gated Selective Policy Optimization:Token-Level Gradient Allocation for Hybrid Training of Large Language Models 

**Authors**: Yuelin Hu, Zhengxue Cheng, Wei Liu, Li Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.03309)  

**Abstract**: Hybrid training methods for large language models combine supervised fine tuning (SFT) on expert demonstrations with reinforcement learning (RL) on model rollouts, typically at the sample level. We propose Entropy Gated Selective Policy Optimization (EGSPO), a three stage framework that extends sample level mixing with token level gradient modulation.
Stage 1, SFT expert learning, establishes a reliable warm up policy using expert demonstrations with a pure SFT loss. Stage 2, RL rollout generation, samples trajectories from the current policy and computes per token predictive entropy. Stage 3, the EGSPO mechanism, applies entropy gated gradient allocation: a predictive entropy module routes high entropy tokens to full PPO updates to encourage exploration, and low entropy tokens to attenuated PPO updates to reduce variance and preserve knowledge. Critically, both branches incorporate the advantage function A_t, ensuring that incorrect trajectories receive consistent negative learning signals and preventing reinforcement of confident errors.
EGSPO achieves consistent improvements on mathematical reasoning benchmarks, with gains of 3.8 percent on AIME and 2.9 percent on MATH over the CHORD phi baseline, while incurring only 3.4 percent additional computational overhead. 

---
# Reinforcement Learning with Promising Tokens for Large Language Models 

**Authors**: Jing-Cheng Pang, Liang Lu, Xian Tang, Kun Jiang, Sijie Wu, Kai Zhang, Xubin Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.03195)  

**Abstract**: Reinforcement learning (RL) has emerged as a key paradigm for aligning and optimizing large language models (LLMs). Standard approaches treat the LLM as the policy and apply RL directly over the full vocabulary space. However, this formulation includes the massive tail of contextually irrelevant tokens in the action space, which could distract the policy from focusing on decision-making among the truly reasonable tokens. In this work, we verify that valid reasoning paths could inherently concentrate within a low-rank subspace. Based on this insight, we introduce Reinforcement Learning with Promising Tokens (RLPT), a framework that mitigates the action space issue by decoupling strategic decision-making from token generation. Specifically, RLPT leverages the semantic priors of the base model to identify a dynamic set of \emph{promising tokens} and constrains policy optimization exclusively to this refined subset via masking. Theoretical analysis and empirical results demonstrate that RLPT effectively reduces gradient variance, stabilizes the training process, and improves sample efficiency. Experiment results on math, coding, and telecom reasoning show that RLPT outperforms standard RL baselines and integrates effectively across various model sizes (4B and 8B) and RL algorithms (GRPO and DAPO). 

---
# Prompt Augmentation Scales up GRPO Training on Mathematical Reasoning 

**Authors**: Wenquan Lu, Hai Huang, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2602.03190)  

**Abstract**: Reinforcement learning algorithms such as group-relative policy optimization (GRPO) have demonstrated strong potential for improving the mathematical reasoning capabilities of large language models. However, prior work has consistently observed an entropy collapse phenomenon during reinforcement post-training, characterized by a monotonic decrease in policy entropy that ultimately leads to training instability and collapse. As a result, most existing approaches restrict training to short horizons (typically 5-20 epochs), limiting sustained exploration and hindering further policy improvement. In addition, nearly all prior work relies on a single, fixed reasoning prompt or template during training. In this work, we introduce prompt augmentation, a training strategy that instructs the model to generate reasoning traces under diverse templates and formats, thereby increasing rollout diversity. We show that, without a KL regularization term, prompt augmentation enables stable scaling of training duration under a fixed dataset and allows the model to tolerate low-entropy regimes without premature collapse. Empirically, a Qwen2.5-Math-1.5B model trained with prompt augmentation on the MATH Level 3-5 dataset achieves state-of-the-art performance, reaching 44.5 per-benchmark accuracy and 51.3 per-question accuracy on standard mathematical reasoning benchmarks, including AIME24, AMC, MATH500, Minerva, and OlympiadBench. The code and model checkpoints are available at this https URL. 

---
# Self-Hinting Language Models Enhance Reinforcement Learning 

**Authors**: Baohao Liao, Hanze Dong, Xinxing Xu, Christof Monz, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2602.03143)  

**Abstract**: Group Relative Policy Optimization (GRPO) has recently emerged as a practical recipe for aligning large language models with verifiable objectives. However, under sparse terminal rewards, GRPO often stalls because rollouts within a group frequently receive identical rewards, causing relative advantages to collapse and updates to vanish. We propose self-hint aligned GRPO with privileged supervision (SAGE), an on-policy reinforcement learning framework that injects privileged hints during training to reshape the rollout distribution under the same terminal verifier reward. For each prompt $x$, the model samples a compact hint $h$ (e.g., a plan or decomposition) and then generates a solution $\tau$ conditioned on $(x,h)$. Crucially, the task reward $R(x,\tau)$ is unchanged; hints only increase within-group outcome diversity under finite sampling, preventing GRPO advantages from collapsing under sparse rewards. At test time, we set $h=\varnothing$ and deploy the no-hint policy without any privileged information. Moreover, sampling diverse self-hints serves as an adaptive curriculum that tracks the learner's bottlenecks more effectively than fixed hints from an initial policy or a stronger external model. Experiments over 6 benchmarks with 3 LLMs show that SAGE consistently outperforms GRPO, on average +2.0 on Llama-3.2-3B-Instruct, +1.2 on Qwen2.5-7B-Instruct and +1.3 on Qwen3-4B-Instruct. The code is available at this https URL. 

---
# Reward Shaping for Inference-Time Alignment: A Stackelberg Game Perspective 

**Authors**: Haichuan Wang, Tao Lin, Lingkai Kong, Ce Li, Hezi Jiang, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2602.02572)  

**Abstract**: Existing alignment methods directly use the reward model learned from user preference data to optimize an LLM policy, subject to KL regularization with respect to the base policy. This practice is suboptimal for maximizing user's utility because the KL regularization may cause the LLM to inherit the bias in the base policy that conflicts with user preferences. While amplifying rewards for preferred outputs can mitigate this bias, it also increases the risk of reward hacking. This tradeoff motivates the problem of optimally designing reward models under KL regularization. We formalize this reward model optimization problem as a Stackelberg game, and show that a simple reward shaping scheme can effectively approximate the optimal reward model. We empirically evaluate our method in inference-time alignment settings and demonstrate that it integrates seamlessly into existing alignment methods with minimal overhead. Our method consistently improves average reward and achieves win-tie rates exceeding 66% against all baselines, averaged across evaluation settings. 

---
# BatCoder: Self-Supervised Bidirectional Code-Documentation Learning via Back-Translation 

**Authors**: Jingwen Xu, Yiyang Lu, Zisu Huang, Changze Lv, Xiaohua Wang, Shizheng Li, Zhibo Xu, Zhengkang Guo, Zhengyuan Wang, Muzhao Tian, Xuanjing Huang, Xiaoqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.02554)  

**Abstract**: Training LLMs for code-related tasks typically depends on high-quality code-documentation pairs, which are costly to curate and often scarce for niche programming languages. We introduce BatCoder, a self-supervised reinforcement learning framework designed to jointly optimize code generation and documentation production. BatCoder employs a back-translation strategy: a documentation is first generated from code, and then the generated documentation is used to reconstruct the original code. The semantic similarity between the original and reconstructed code serves as an implicit reward, enabling reinforcement learning to improve the model's performance both in generating code from documentation and vice versa. This approach allows models to be trained using only code, substantially increasing the available training examples. Evaluated on HumanEval and MBPP with a 7B model, BatCoder achieved 83.5% and 81.0% pass@1, outperforming strong open-source baselines. Moreover, the framework demonstrates consistent scaling with respect to both training corpus size and model capacity. 

---
# Beyond Alignment: Expanding Reasoning Capacity via Manifold-Reshaping Policy Optimization 

**Authors**: Dayu Wang, Jiaye Yang, Weikang Li, Jiahui Liang, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.02545)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated remarkable success in enhancing the reasoning capabilities of Large Language Models (LLMs). However, recent studies question whether RL genuinely expands reasoning capacity or merely aligns existing latent capabilities, arguing that exploration remains confined within the pre-trained model's low-rank bias manifold. In this work, we challenge this accessibility boundary hypothesis by demonstrating that the latent reasoning space can be fundamentally expanded through targeted geometric interventions. We propose Manifold-Reshaping Policy Optimization (MRPO), a geometric framework designed to fundamentally restructure the inference space of LLMs. MRPO operates in two stages: first, we employ Spectral Orthogonal Exploration (SOE) to eject the policy initialization into the null space of the bias manifold; second, we integrate an Effective Rank regularization term into the policy optimization objective. This approach incentivizes the discovery and maintenance of high-dimensional reasoning trajectories against the entropy-reducing tendency of standard RL. Empirically, our 4B-parameter method achieves state-of-the-art performance on mathematical tasks, significantly outperforming larger models (e.g., Qwen3-32B) and expanding the capability boundary beyond standard GRPO. Our code is available at this https URL 

---
# GraphDancer: Training LLMs to Explore and Reason over Graphs via Curriculum Reinforcement Learning 

**Authors**: Yuyang Bai, Zhuofeng Li, Ping Nie, Jianwen Xie, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02518)  

**Abstract**: Large language models (LLMs) increasingly rely on external knowledge to improve factuality, yet many real-world knowledge sources are organized as heterogeneous graphs rather than plain text. Reasoning over such graph-structured knowledge poses two key challenges: (1) navigating structured, schema-defined relations requires precise function calls rather than similarity-based retrieval, and (2) answering complex questions often demands multi-hop evidence aggregation through iterative information seeking. We propose GraphDancer, a reinforcement learning (RL) framework that teaches LLMs to navigate graphs by interleaving reasoning and function execution. To make RL effective for moderate-sized LLMs, we introduce a graph-aware curriculum that schedules training by the structural complexity of information-seeking trajectories using an easy-to-hard biased sampler. We evaluate GraphDancer on a multi-domain benchmark by training on one domain only and testing on unseen domains and out-of-distribution question types. Despite using only a 3B backbone, GraphDancer outperforms baselines equipped with either a 14B backbone or GPT-4o-mini, demonstrating robust cross-domain generalization of graph exploration and reasoning skills. Our code and models can be found at this https URL . 

---
# Kimi K2.5: Visual Agentic Intelligence 

**Authors**: Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, S.H. Cai, Yuan Cao, Y. Charles, H.S. Che, Cheng Chen, Guanduo Chen, Huarong Chen, Jia Chen, Jiahao Chen, Jianlong Chen, Jun Chen, Kefan Chen, Liang Chen, Ruijue Chen, Xinhao Chen, Yanru Chen, Yanxu Chen, Yicun Chen, Yimin Chen, Yingjiang Chen, Yuankun Chen, Yujie Chen, Yutian Chen, Zhirong Chen, Ziwei Chen, Dazhi Cheng, Minghan Chu, Jialei Cui, Jiaqi Deng, Muxi Diao, Hao Ding, Mengfan Dong, Mengnan Dong, Yuxin Dong, Yuhao Dong, Angang Du, Chenzhuang Du, Dikang Du, Lingxiao Du, Yulun Du, Yu Fan, Shengjun Fang, Qiulin Feng, Yichen Feng, Garimugai Fu, Kelin Fu, Hongcheng Gao, Tong Gao, Yuyao Ge, Shangyi Geng, Chengyang Gong, Xiaochen Gong, Zhuoma Gongque, Qizheng Gu, Xinran Gu, Yicheng Gu, Longyu Guan, Yuanying Guo, Xiaoru Hao, Weiran He, Wenyang He, Yunjia He, Chao Hong, Hao Hu, Jiaxi Hu, Yangyang Hu, Zhenxing Hu, Ke Huang, Ruiyuan Huang, Weixiao Huang, Zhiqi Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yu Jing, Guokun Lai, Aidi Li, C. Li, Cheng Li, Fang Li, Guanghe Li, Guanyu Li, Haitao Li, Haoyang Li, Jia Li, Jingwei Li, Junxiong Li, Lincan Li, Mo Li, Weihong Li, Wentao Li, Xinhang Li, Xinhao Li, Yang Li, Yanhao Li, Yiwei Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.02276)  

**Abstract**: We introduce Kimi K2.5, an open-source multimodal agentic model designed to advance general agentic intelligence. K2.5 emphasizes the joint optimization of text and vision so that two modalities enhance each other. This includes a series of techniques such as joint text-vision pre-training, zero-vision SFT, and joint text-vision reinforcement learning. Building on this multimodal foundation, K2.5 introduces Agent Swarm, a self-directed parallel agent orchestration framework that dynamically decomposes complex tasks into heterogeneous sub-problems and executes them concurrently. Extensive evaluations show that Kimi K2.5 achieves state-of-the-art results across various domains including coding, vision, reasoning, and agentic tasks. Agent Swarm also reduces latency by up to $4.5\times$ over single-agent baselines. We release the post-trained Kimi K2.5 model checkpoint to facilitate future research and real-world applications of agentic intelligence. 

---
# RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System 

**Authors**: Yinjie Wang, Tianbao Xie, Ke Shen, Mengdi Wang, Ling Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02488)  

**Abstract**: We propose RLAnything, a reinforcement learning framework that dynamically forges environment, policy, and reward models through closed-loop optimization, amplifying learning signals and strengthening the overall RL system for any LLM or agentic scenarios. Specifically, the policy is trained with integrated feedback from step-wise and outcome signals, while the reward model is jointly optimized via consistency feedback, which in turn further improves policy training. Moreover, our theory-motivated automatic environment adaptation improves training for both the reward and policy models by leveraging critic feedback from each, enabling learning from experience. Empirically, each added component consistently improves the overall system, and RLAnything yields substantial gains across various representative LLM and agentic tasks, boosting Qwen3-VL-8B-Thinking by 9.1% on OSWorld and Qwen2.5-7B-Instruct by 18.7% and 11.9% on AlfWorld and LiveBench, respectively. We also that optimized reward-model signals outperform outcomes that rely on human labels. Code: this https URL 

---
# Learning to Reason Faithfully through Step-Level Faithfulness Maximization 

**Authors**: Runquan Gui, Yafu Li, Xiaoye Qu, Ziyan Liu, Yeqiu Cheng, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.03507)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has markedly improved the performance of Large Language Models (LLMs) on tasks requiring multi-step reasoning. However, most RLVR pipelines rely on sparse outcome-based rewards, providing little supervision over intermediate steps and thus encouraging over-confidence and spurious reasoning, which in turn increases hallucinations. To address this, we propose FaithRL, a general reinforcement learning framework that directly optimizes reasoning faithfulness. We formalize a faithfulness-maximization objective and theoretically show that optimizing it mitigates over-confidence. To instantiate this objective, we introduce a geometric reward design and a faithfulness-aware advantage modulation mechanism that assigns step-level credit by penalizing unsupported steps while preserving valid partial derivations. Across diverse backbones and benchmarks, FaithRL consistently reduces hallucination rates while maintaining (and often improving) answer correctness. Further analysis confirms that FaithRL increases step-wise reasoning faithfulness and generalizes robustly. Our code is available at this https URL. 

---
# Learning Query-Specific Rubrics from Human Preferences for DeepResearch Report Generation 

**Authors**: Changze Lv, Jie Zhou, Wentao Zhao, Jingwen Xu, Zisu Huang, Muzhao Tian, Shihan Dou, Tao Gui, Le Tian, Xiao Zhou, Xiaoqing Zheng, Xuanjing Huang, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.03619)  

**Abstract**: Nowadays, training and evaluating DeepResearch-generated reports remain challenging due to the lack of verifiable reward signals. Accordingly, rubric-based evaluation has become a common practice. However, existing approaches either rely on coarse, pre-defined rubrics that lack sufficient granularity, or depend on manually constructed query-specific rubrics that are costly and difficult to scale. In this paper, we propose a pipeline to train human-preference-aligned query-specific rubric generators tailored for DeepResearch report generation. We first construct a dataset of DeepResearch-style queries annotated with human preferences over paired reports, and train rubric generators via reinforcement learning with a hybrid reward combining human preference supervision and LLM-based rubric evaluation. To better handle long-horizon reasoning, we further introduce a Multi-agent Markov-state (MaMs) workflow for report generation. We empirically show that our proposed rubric generators deliver more discriminative and better human-aligned supervision than existing rubric design strategies. Moreover, when integrated into the MaMs training framework, DeepResearch systems equipped with our rubric generators consistently outperform all open-source baselines on the DeepResearch Bench and achieve performance comparable to that of leading closed-source models. 

---
# Verified Critical Step Optimization for LLM Agents 

**Authors**: Mukai Li, Qingcheng Zeng, Tianqing Fang, Zhenwen Liang, Linfeng Song, Qi Liu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03412)  

**Abstract**: As large language model agents tackle increasingly complex long-horizon tasks, effective post-training becomes critical. Prior work faces fundamental challenges: outcome-only rewards fail to precisely attribute credit to intermediate steps, estimated step-level rewards introduce systematic noise, and Monte Carlo sampling approaches for step reward estimation incur prohibitive computational cost. Inspired by findings that only a small fraction of high-entropy tokens drive effective RL for reasoning, we propose Critical Step Optimization (CSO), which focuses preference learning on verified critical steps, decision points where alternate actions demonstrably flip task outcomes from failure to success. Crucially, our method starts from failed policy trajectories rather than expert demonstrations, directly targeting the policy model's weaknesses. We use a process reward model (PRM) to identify candidate critical steps, leverage expert models to propose high-quality alternatives, then continue execution from these alternatives using the policy model itself until task completion. Only alternatives that the policy successfully executes to correct outcomes are verified and used as DPO training data, ensuring both quality and policy reachability. This yields fine-grained, verifiable supervision at critical decisions while avoiding trajectory-level coarseness and step-level noise. Experiments on GAIA-Text-103 and XBench-DeepSearch show that CSO achieves 37% and 26% relative improvement over the SFT baseline and substantially outperforms other post-training methods, while requiring supervision at only 16% of trajectory steps. This demonstrates the effectiveness of selective verification-based learning for agent post-training. 

---
# One Model, All Roles: Multi-Turn, Multi-Agent Self-Play Reinforcement Learning for Conversational Social Intelligence 

**Authors**: Bowen Jiang, Taiwei Shi, Ryo Kamoi, Yuan Yuan, Camillo J. Taylor, Longqi Yang, Pei Zhou, Sihao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.03109)  

**Abstract**: This paper introduces OMAR: One Model, All Roles, a reinforcement learning framework that enables AI to develop social intelligence through multi-turn, multi-agent conversational self-play. Unlike traditional paradigms that rely on static, single-turn optimizations, OMAR allows a single model to role-play all participants in a conversation simultaneously, learning to achieve long-term goals and complex social norms directly from dynamic social interaction. To ensure training stability across long dialogues, we implement a hierarchical advantage estimation that calculates turn-level and token-level advantages. Evaluations in the SOTOPIA social environment and Werewolf strategy games show that our trained models develop fine-grained, emergent social intelligence, such as empathy, persuasion, and compromise seeking, demonstrating the effectiveness of learning collaboration even under competitive scenarios. While we identify practical challenges like reward hacking, our results show that rich social intelligence can emerge without human supervision. We hope this work incentivizes further research on AI social intelligence in group conversations. 

---
# Short Chains, Deep Thoughts: Balancing Reasoning Efficiency and Intra-Segment Capability via Split-Merge Optimization 

**Authors**: Runquan Gui, Jie Wang, Zhihai Wang, Chi Ma, Jianye Hao, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03141)  

**Abstract**: While Large Reasoning Models (LRMs) have demonstrated impressive capabilities in solving complex tasks through the generation of long reasoning chains, this reliance on verbose generation results in significant latency and computational overhead. To address these challenges, we propose \textbf{CoSMo} (\textbf{Co}nsistency-Guided \textbf{S}plit-\textbf{M}erge \textbf{O}ptimization), a framework designed to eliminate structural redundancy rather than indiscriminately restricting token volume. Specifically, CoSMo utilizes a split-merge algorithm that dynamically refines reasoning chains by merging redundant segments and splitting logical gaps to ensure coherence. We then employ structure-aligned reinforcement learning with a novel segment-level budget to supervise the model in maintaining efficient reasoning structures throughout training. Extensive experiments across multiple benchmarks and backbones demonstrate that CoSMo achieves superior performance, improving accuracy by \textbf{3.3} points while reducing segment usage by \textbf{28.7\%} on average compared to reasoning efficiency baselines. 

---
# ReMiT: RL-Guided Mid-Training for Iterative LLM Evolution 

**Authors**: Junjie Huang, Jiarui Qin, Di Yin, Weiwen Liu, Yong Yu, Xing Sun, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03075)  

**Abstract**: Standard training pipelines for large language models (LLMs) are typically unidirectional, progressing from pre-training to post-training. However, the potential for a bidirectional process--where insights from post-training retroactively improve the pre-trained foundation--remains unexplored. We aim to establish a self-reinforcing flywheel: a cycle in which reinforcement learning (RL)-tuned model strengthens the base model, which in turn enhances subsequent post-training performance, requiring no specially trained teacher or reference model. To realize this, we analyze training dynamics and identify the mid-training (annealing) phase as a critical turning point for model capabilities. This phase typically occurs at the end of pre-training, utilizing high-quality corpora under a rapidly decaying learning rate. Building upon this insight, we introduce ReMiT (Reinforcement Learning-Guided Mid-Training). Specifically, ReMiT leverages the reasoning priors of RL-tuned models to dynamically reweight tokens during the mid-training phase, prioritizing those pivotal for reasoning. Empirically, ReMiT achieves an average improvement of 3\% on 10 pre-training benchmarks, spanning math, code, and general reasoning, and sustains these gains by over 2\% throughout the post-training pipeline. These results validate an iterative feedback loop, enabling continuous and self-reinforcing evolution of LLMs. 

---
# Test-time Recursive Thinking: Self-Improvement without External Feedback 

**Authors**: Yufan Zhuang, Chandan Singh, Liyuan Liu, Yelong Shen, Dinghuai Zhang, Jingbo Shang, Jianfeng Gao, Weizhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.03094)  

**Abstract**: Modern Large Language Models (LLMs) have shown rapid improvements in reasoning capabilities, driven largely by reinforcement learning (RL) with verifiable rewards. Here, we ask whether these LLMs can self-improve without the need for additional training. We identify two core challenges for such systems: (i) efficiently generating diverse, high-quality candidate solutions, and (ii) reliably selecting correct answers in the absence of ground-truth supervision. To address these challenges, we propose Test-time Recursive Thinking (TRT), an iterative self-improvement framework that conditions generation on rollout-specific strategies, accumulated knowledge, and self-generated verification signals. Using TRT, open-source models reach 100% accuracy on AIME-25/24, and on LiveCodeBench's most difficult problems, closed-source models improve by 10.4-14.8 percentage points without external feedback. 

---
# CPMobius: Iterative Coach-Player Reasoning for Data-Free Reinforcement Learning 

**Authors**: Ran Li, Zeyuan Liu, Yinghao chen, Bingxiang He, Jiarui Yuan, Zixuan Fu, Weize Chen, Jinyi Hu, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.02979)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong potential in complex reasoning, yet their progress remains fundamentally constrained by reliance on massive high-quality human-curated tasks and labels, either through supervised fine-tuning (SFT) or reinforcement learning (RL) on reasoning-specific data. This dependence renders supervision-heavy training paradigms increasingly unsustainable, with signs of diminishing scalability already evident in practice. To overcome this limitation, we introduce CPMöbius (CPMobius), a collaborative Coach-Player paradigm for data-free reinforcement learning of reasoning models. Unlike traditional adversarial self-play, CPMöbius, inspired by real world human sports collaboration and multi-agent collaboration, treats the Coach and Player as independent but cooperative roles. The Coach proposes instructions targeted at the Player's capability and receives rewards based on changes in the Player's performance, while the Player is rewarded for solving the increasingly instructive tasks generated by the Coach. This cooperative optimization loop is designed to directly enhance the Player's mathematical reasoning ability. Remarkably, CPMöbius achieves substantial improvement without relying on any external training data, outperforming existing unsupervised approaches. For example, on Qwen2.5-Math-7B-Instruct, our method improves accuracy by an overall average of +4.9 and an out-of-distribution average of +5.4, exceeding RENT by +1.5 on overall accuracy and R-zero by +4.2 on OOD accuracy. 

---
