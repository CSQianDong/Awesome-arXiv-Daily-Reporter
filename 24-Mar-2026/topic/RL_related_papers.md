# TiCo: Time-Controllable Training for Spoken Dialogue Models 

**Authors**: Kai-Wei Chang, Wei-Chih Chen, En-Pei Hu, Hung-yi Lee, James Glass  

**Link**: [PDF](https://arxiv.org/pdf/2603.22267)  

**Abstract**: We propose TiCo, a simple post-training method for enabling spoken dialogue models (SDMs) to follow time-constrained instructions and generate responses with controllable duration. This capability is valuable for real-world spoken language systems such as voice assistants and interactive agents, where controlling response duration can improve interaction quality. However, despite their strong ability to generate natural spoken responses, existing models lack time awareness and struggle to follow duration-related instructions (e.g., "Please generate a response lasting about 15 seconds"). Through an empirical evaluation of both open-source and commercial SDMs, we show that they frequently fail to satisfy such time-control requirements. TiCo addresses this limitation by enabling models to estimate elapsed speaking time during generation through Spoken Time Markers (STM) (e.g., <10.6 seconds>). These markers help the model maintain awareness of time and adjust the remaining content to meet the target duration. TiCo is simple and efficient: it requires only a small amount of data and no additional question-answer pairs, relying instead on self-generation and reinforcement learning. Experimental results show that TiCo significantly improves adherence to duration constraints while preserving response quality. 

---
# Probing How Scalable Table Data Enhances General Long-Context Reasoning 

**Authors**: Huaibing Xie, Guoliang Zhao, Yang Liu, Shihan Dou, Siming Huang, Yanling Xiao, Shaolei Wang, Yiting Liu, Cheng Zhang, Shaofan Liu, Pluto Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.21719)  

**Abstract**: As real-world tasks grow increasingly complex, long-context reasoning has become a core capability for Large Language Models (LLMs). However, few studies explore which data types are effective for long-context reasoning and why. We find that structured table data with periodic structures shows strong potential for long-context reasoning. Motivated by this observation, we mathematically analyze tabular dependency structures using mutual information, revealing periodic non-vanishing dependencies in table data. Furthermore, we systematically analyze the capabilities of structured table data, conduct relevant scaling experiments, and validate its underlying mechanisms for enhancing long-context reasoning, yielding several meaningful insights. Leveraging these insights, we propose a simple yet scalable pipeline(TableLong) for synthesizing high-quality, diverse, and verifiable structured table data to boost long-context reasoning via RL. Extensive experimental results demonstrate that table data significantly enhances the long-context reasoning capability of LLMs across multiple long-context benchmarks (+8.24\% on average), and even improves performance on out-of-domain benchmarks (+8.06\% on average). We hope that our insights provide practical guidance for effective post-training data to enhance long-context reasoning in LLMs. 

---
# KG-Hopper: Empowering Compact Open LLMs with Knowledge Graph Reasoning via Reinforcement Learning 

**Authors**: Shuai Wang, Yinan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.21440)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive natural language capabilities but often struggle with knowledge-intensive reasoning tasks. Knowledge Base Question Answering (KBQA), which leverages structured Knowledge Graphs (KGs) exemplifies this challenge due to the need for accurate multi-hop reasoning. Existing approaches typically perform sequential reasoning steps guided by predefined pipelines, restricting flexibility and causing error cascades due to isolated reasoning at each step. To address these limitations, we propose KG-Hopper, a novel Reinforcement Learning (RL) framework that empowers compact open LLMs with the ability to perform integrated multi-hop KG reasoning within a single inference round. Rather than reasoning step-by-step, we train a Reasoning LLM that embeds the entire KG traversal and decision process into a unified ``thinking'' stage, enabling global reasoning over cross-step dependencies and dynamic path exploration with backtracking. Experimental results on eight KG reasoning benchmarks show that KG-Hopper, based on a 7B-parameter LLM, consistently outperforms larger multi-step systems (up to 70B) and achieves competitive performance with proprietary models such as GPT-3.5-Turbo and GPT-4o-mini, while remaining compact, open, and data-efficient. The code is publicly available at: this https URL. 

---
# Benchmarking Bengali Dialectal Bias: A Multi-Stage Framework Integrating RAG-Based Translation and Human-Augmented RLAIF 

**Authors**: K. M. Jubair Sami, Dipto Sumit, Ariyan Hossain, Farig Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2603.21359)  

**Abstract**: Large language models (LLMs) frequently exhibit performance biases against regional dialects of low-resource languages. However, frameworks to quantify these disparities remain scarce. We propose a two-phase framework to evaluate dialectal bias in LLM question-answering across nine Bengali dialects. First, we translate and gold-label standard Bengali questions into dialectal variants adopting a retrieval-augmented generation (RAG) pipeline to prepare 4,000 question sets. Since traditional translation quality evaluation metrics fail on unstandardized dialects, we evaluate fidelity using an LLM-as-a-judge, which human correlation confirms outperforms legacy metrics. Second, we benchmark 19 LLMs across these gold-labeled sets, running 68,395 RLAIF evaluations validated through multi-judge agreement and human fallback. Our findings reveal severe performance drops linked to linguistic divergence. For instance, responses to the highly divergent Chittagong dialect score 5.44/10, compared to 7.68/10 for Tangail. Furthermore, increased model scale does not consistently mitigate this bias. We contribute a validated translation quality evaluation method, a rigorous benchmark dataset, and a Critical Bias Sensitivity (CBS) metric for safety-critical applications. 

---
# enhancing reasoning accuracy in large language models during inference time 

**Authors**: Vinay Sharma, Manish Jain  

**Link**: [PDF](https://arxiv.org/pdf/2603.21301)  

**Abstract**: Large Language Models (LLMs) often exhibit strong linguistic abilities while remaining unreliable on multi-step reasoning tasks, particularly when deployed without additional training or fine-tuning. In this work, we study inference-time techniques to improve the reasoning accuracy of LLMs. We systematically evaluate three classes of inference-time strategies: (i) self-consistency via stochastic decoding, where the model is sampled multiple times using controlled temperature and nucleus sampling and the most frequent final answer is selected; (ii) dual-model reasoning agreement, where outputs from two independent models are compared and only consistent reasoning traces are trusted; and (iii) self-reflection, where the model critiques and revises its own reasoning. Across all evaluated methods, we employ Chain-of-Thought (CoT) [1] prompting to elicit explicit intermediate reasoning steps before generating final answers. In this work, we provide a controlled comparative evaluation across three inference-time strategies under identical prompting and verification settings. Our experiments on LLM [2] show that self-consistency with nucleus sampling and controlled temperature value yields the substantial gains, achieving a 9% to 15% absolute improvement in accuracy over greedy single-pass decoding, well-suited for low-risk domains, offering meaningful gains with minimal overhead. The dual-model approach provides additional confirmation for model reasoning steps thus more appropriate for moderate-risk domains, where higher reliability justifies additional compute. Self-reflection offers only marginal improvements, suggesting limited effectiveness for smaller non-reasoning models at inference time. 

---
# Fast-Slow Thinking RM: Efficient Integration of Scalar and Generative Reward Models 

**Authors**: Jiayun Wu, Peixu Hou, Shan Qu, Peng Zhang, Ning Gu, Tun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.20212)  

**Abstract**: Reward models (RMs) are critical for aligning Large Language Models via Reinforcement Learning from Human Feedback (RLHF). While Generative Reward Models (GRMs) achieve superior accuracy through chain-of-thought (CoT) reasoning, they incur substantial computational costs. Conversely, Scalar Reward Models (SRMs) offer efficiency but suffer from limited performance and adaptability in complex scenarios.
We introduce Fast-Slow Thinking Reward Models (F/S-RM), a hybrid RM architecture inspired by Dual Process Theory. It trains a single model to integrate two distinct reward paradigms: first-token prediction as a scalar score (fast thinking) and CoT-based judgment (slow thinking), regulated by a dual-confidence activation mechanism that determines when to activate slow thinking.
F/S-RM achieves a 1.2% relative performance improvement over state-of-the-art models while reducing token consumption by 20.8%. Code and data will be publicly available. 

---
# EvoIdeator: Evolving Scientific Ideas through Checklist-Grounded Reinforcement Learning 

**Authors**: Andreas Sauter, Yuyue Zhao, Jacopo Urbani, Wenxiang Hu, Zaiqiao Meng, Lun Zhou, Xiaohui Yan, Yougang Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2603.21728)  

**Abstract**: Scientific idea generation is a cornerstone of autonomous knowledge discovery, yet the iterative evolution required to transform initial concepts into high-quality research proposals remains a formidable challenge for Large Language Models (LLMs). Existing Reinforcement Learning (RL) paradigms often rely on rubric-based scalar rewards that provide global quality scores but lack actionable granularity. Conversely, language-based refinement methods are typically confined to inference-time prompting, targeting models that are not explicitly optimized to internalize such critiques. To bridge this gap, we propose \textbf{EvoIdeator}, a framework that facilitates the evolution of scientific ideas by aligning the RL training objective with \textbf{checklist-grounded feedback}. EvoIdeator leverages a structured judge model to generate two synergistic signals: (1) \emph{lexicographic rewards} for multi-dimensional optimization, and (2) \emph{fine-grained language feedback} that offers span-level critiques regarding grounding, feasibility, and methodological rigor. By integrating these signals into the RL loop, we condition the policy to systematically utilize precise feedback during both optimization and inference. Extensive experiments demonstrate that EvoIdeator, built on Qwen3-4B, significantly outperforms much larger frontier models across key scientific metrics. Crucially, the learned policy exhibits strong generalization to diverse external feedback sources without further fine-tuning, offering a scalable and rigorous path toward self-refining autonomous ideation. 

---
# AdaRubric: Task-Adaptive Rubrics for LLM Agent Evaluation 

**Authors**: Liang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2603.21362)  

**Abstract**: LLM-as-Judge evaluation fails agent tasks because a fixed rubric cannot capture what matters for this task: code debugging demands Correctness and Error Handling; web navigation demands Goal Alignment and Action Efficiency. We present ADARUBRIC, which closes this gap by generating task-specific evaluation rubrics on the fly from task descriptions, scoring trajectories step-by-step with confidence-weighted per-dimension feedback, and filtering preference pairs with the novel DimensionAwareFilter - a provably necessary condition for preventing high-scoring dimensions from masking dimension-level failures. On WebArena and ToolBench, ADARUBRIC achieves Pearson r=0.79 human correlation (+0.16 over the best static baseline) with deployment-grade reliability (Krippendorff's $\alpha$=0.83). DPO agents trained on ADARUBRIC preference pairs gain +6.8 to +8.5 pp task success over Prometheus across three benchmarks; gains transfer to SWE-bench code repair (+4.9 pp) and accelerate PPO convergence by +6.6 pp at 5K steps - both without any rubric engineering. Code: this https URL. 

---
# LongCat-Flash-Prover: Advancing Native Formal Reasoning via Agentic Tool-Integrated Reinforcement Learning 

**Authors**: Jianing Wang, Jianfei Zhang, Qi Guo, Linsen Guo, Rumei Li, Chao Zhang, Chong Peng, Cunguang Wang, Dengchang Zhao, Jiarong Shi, Jingang Wang, Liulin Feng, Mengxia Shen, Qi Li, Shengnan An, Shun Wang, Wei Shi, Xiangyu Xi, Xiaoyu Li, Xuezhi Cao, Yi Lu, Yunke Zhao, Zhengyu Chen, Zhimin Lin, Wei Wang, Peng Pei, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2603.21065)  

**Abstract**: We introduce LongCat-Flash-Prover, a flagship 560-billion-parameter open-source Mixture-of- Experts (MoE) model that advances Native Formal Reasoning in Lean4 through agentic tool-integrated reasoning (TIR). We decompose the native formal reasoning task into three independent formal capabilities, i.e., auto-formalization, sketching, and proving. To facilitate these capabilities, we propose a Hybrid-Experts Iteration Framework to expand high-quality task trajectories, including generating a formal statement based on a given informal problem, producing a whole-proof directly from the statement, or a lemma-style sketch. During agentic RL, we present a Hierarchical Importance Sampling Policy Optimization (HisPO) algorithm, which aims to stabilize the MoE model training on such long-horizon tasks. It employs a gradient masking strategy that accounts for the policy staleness and the inherent train-inference engine discrepancies at both sequence and token levels. Additionally, we also incorporate theorem consistency and legality detection mechanisms to eliminate reward hacking issues. Extensive evaluations show that our LongCat-Flash-Prover sets a new state-of-the-art for open-weights models in both auto-formalization and theorem proving. Demonstrating remarkable sample efficiency, it achieves a 97.1% pass rate on MiniF2F-Test using only 72 inference budget per problem. On more challenging benchmarks, it solves 70.8% of ProverBench and 41.5% of PutnamBench with no more than 220 attempts per problem, significantly outperforming existing open-weights baselines. 

---
# Clinical Cognition Alignment for Gastrointestinal Diagnosis with Multimodal LLMs 

**Authors**: Huan Zheng, Yucheng Zhou, Tianyi Yan, Dubing Chen, Hongbo Lu, Wenlong Liao, Tao He, Pai Peng, Jianbing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.20698)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated remarkable potential in medical image analysis. However, their application in gastrointestinal endoscopy is currently hindered by two critical limitations: the misalignment between general model reasoning and standardized clinical cognitive pathways, and the lack of causal association between visual features and diagnostic outcomes. In this paper, we propose a novel Clinical-Cognitive-Aligned (CogAlign) framework to address these challenges. First, we endow the model with rigorous clinical analytical capabilities by constructing the hierarchical clinical cognition dataset and employing Supervised Fine-Tuning (SFT). Unlike conventional approaches, this strategy internalizes the hierarchical diagnostic logic of experts, ranging from anatomical localization and morphological evaluation to microvascular analysis, directly into the model. Second, to eliminate visual bias, we provide a theoretical analysis demonstrating that standard supervised tuning inevitably converges to spurious background correlations. Guided by this insight, we propose a counterfactual-driven reinforcement learning strategy to enforce causal rectification. By generating counterfactual normal samples via lesion masking and optimizing through clinical-cognition-centric rewards, we constrain the model to strictly ground its diagnosis in causal lesion features. Extensive experiments demonstrate that our approach achieves State-of-the-Art (SoTA) performance across multiple benchmarks, significantly enhancing diagnostic accuracy in complex clinical scenarios. All source code and datasets will be made publicly available. 

---
# Agentic Personas for Adaptive Scientific Explanations with Knowledge Graphs 

**Authors**: Susana Nunes, Tiago Guerreiro, Catia Pesquita  

**Link**: [PDF](https://arxiv.org/pdf/2603.21846)  

**Abstract**: AI explanation methods often assume a static user model, producing non-adaptive explanations regardless of expert goals, reasoning strategies, or decision contexts. Knowledge graph-based explanations, despite their capacity for grounded, path-based reasoning, inherit this limitation. In complex domains such as scientific discovery, this assumption fails to capture the diversity of cognitive strategies and epistemic stances among experts, preventing explanations that foster deeper understanding and informed decision-making. However, the scarcity of human experts limits the use of direct human feedback to produce adaptive explanations.
We present a reinforcement learning approach for scientific explanation generation that incorporates agentic personas, structured representations of expert reasoning strategies, that guide the explanation agent towards specific epistemic preferences. In an evaluation of knowledge graph-based explanations for drug discovery, we tested two personas that capture distinct epistemic stances derived from expert feedback.
Results show that persona-driven explanations match state-of-the-art predictive performance while persona preferences closely align with those of their corresponding experts. Adaptive explanations were consistently preferred over non-adaptive baselines (n = 22), and persona-based training reduces feedback requirements by two orders of magnitude. These findings demonstrate how agentic personas enable scalable adaptive explainability for AI systems in complex and high-stakes domains. 

---
# A Context Engineering Framework for Improving Enterprise AI Agents based on Digital-Twin MDP 

**Authors**: Xi Yang, Aurelie Lozano, Naoki Abe, Bhavya, Saurabh Jha, Noah Zheutlin, Rohan R. Arora, Yu Deng, Daby M. Sow  

**Link**: [PDF](https://arxiv.org/pdf/2603.22083)  

**Abstract**: Despite rapid progress in AI agents for enterprise automation and decision-making, their real-world deployment and further performance gains remain constrained by limited data quality and quantity, complex real-world reasoning demands, difficulties with self-play, and the lack of reliable feedback signals. To address these challenges, we propose a lightweight, model-agnostic framework for improving LLM-based enterprise agents via offline reinforcement learning (RL). The proposed Context Engineering via DT-MDP (DT-MDP-CE) framework comprises three key components: (1) A Digital-Twin Markov Decision Process (DT-MDP), which abstracts the agent's reasoning behavior as a finite MDP; (2) A robust contrastive inverse RL, which, armed with the DT-MDP, to efficiently estimate a well-founded reward function and induces policies from mixed-quality offline trajectories; and (3) RL-guided context engineering, which uses the policy obtained from the integrated process of (1) and (2), to improve the agent's decision-making behavior. As a case study, we apply the framework to a representative task in the enterprise-oriented domain of IT automation. Extensive experimental results demonstrate consistent and significant improvements over baseline agents across a wide range of evaluation settings, suggesting that the framework can generalize to other agents sharing similar characteristics in enterprise environments. 

---
# Stabilizing Iterative Self-Training with Verified Reasoning via Symbolic Recursive Self-Alignment 

**Authors**: Xinyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.21558)  

**Abstract**: Recursive self-improvement--where a model iteratively trains on its own outputs--promises sustained capability growth but faces a fundamental obstacle: recursive drift. As models train on self-generated data across multiple iterations, errors in intermediate reasoning compound, leading to mode collapse and performance degradation. We propose Neuro-Symbolic Recursive Self-Alignment (NSRSA), which stabilizes iterative self-training by embedding a symbolic verification subsystem that gates training data quality at the reasoning step level. Unlike outcome-only filtering (which admits "lucky guesses" with flawed reasoning), NSRSA verifies each arithmetic operation via sympy, checks logical flow consistency across reasoning steps, and enforces domain constraints. We evaluate NSRSA on GSM8K using Qwen3-4B-Thinking across 5 self-training iterations under five conditions: no verification, outcome verification, majority voting, full NSRSA symbolic verification, and NSRSA with DPO. Our filtering analysis shows that NSRSA rejects approximately 34% of correct-answer solutions that pass outcome verification, eliminating "lucky guesses" with flawed reasoning from the training set. We further demonstrate that constructing DPO preference pairs from NSRSA verification teaches the model to distinguish sound from flawed reasoning (reward accuracy 46% to 63%). NSRSA provides an extensible framework that demonstrates how external symbolic verification can make recursive self-improvement measurable and reliable within domains where automated verification is available. 

---
# RoboAlign: Learning Test-Time Reasoning for Language-Action Alignment in Vision-Language-Action Models 

**Authors**: Dongyoung Kim, Sumin Park, Woomin Song, Seungku Kim, Taeyoung Kim, Huiwon Jang, Jinwoo Shin, Jaehyung Kim, Younggyo Seo  

**Link**: [PDF](https://arxiv.org/pdf/2603.21341)  

**Abstract**: Improving embodied reasoning in multimodal-large-language models (MLLMs) is essential for building vision-language-action models (VLAs) on top of them to readily translate multimodal understanding into low-level actions. Accordingly, recent work has explored enhancing embodied reasoning in MLLMs through supervision of vision-question-answering type. However, these approaches have been reported to result in unstable VLA performance, often yielding only marginal or even negative gains. In this paper, we propose a more systematic MLLM training framework RoboAlign that reliably improves VLA performance. Our key idea is to sample action tokens via zero-shot natural language reasoning and refines this reasoning using reinforcement learning (RL) to improve action accuracy. As a result, RoboAlign bridges the modality gap between language and low-level actions in MLLMs, and facilitate knowledge transfer from MLLM to VLA. To validate the effectiveness of RoboAlign, we train VLAs by adding a diffusion-based action head on top of an MLLM backbone and evaluate them on major robotics benchmarks. Remarkably, by performing RL-based alignment after SFT using less than 1\% of the data, RoboAlign achieves performance improvements of 17.5\%, 18.9\%, and 106.6\% over SFT baselines on LIBERO, CALVIN, and real-world environments, respectively. 

---
# A transformer architecture alteration to incentivise externalised reasoning 

**Authors**: Elizabeth Pavlova, Mariia Koroliuk, Karthik Viswanathan, Cameron Tice, Edward James Young, Puria Radmard  

**Link**: [PDF](https://arxiv.org/pdf/2603.21376)  

**Abstract**: We propose a new architectural change, and post-training pipeline, for making LLMs more verbose reasoners by teaching a model to truncate forward passes early. We augment an existing transformer architecture with an early-exit mechanism at intermediate layers and train the model to exit at shallower layers when the next token can be predicted without deep computation. After a calibration stage, we incentivise the model to exit as early as possible while maintaining task performance using reinforcement learning. We provide preliminary results to this effect for small reasoning models, showing that they learn to adaptively reduce computations across tokens. We predict that, applied at the right scale, our approach can minimise the amount of excess computation that reasoning models have at their disposal to perform non-myopic planning using their internal activations, reserving this only for difficult-to-predict tokens. 

---
# On the Direction of RLVR Updates for LLM Reasoning: Identification and Exploitation 

**Authors**: Kexin Huang, Haoming Meng, Junkang Wu, Jinda Lu, Chiyu Ma, Ziqian Chen, Xue Wang, Bolin Ding, Jiancan Wu, Xiang Wang, Xiangnan He, Guoyin Wang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.22117)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has substantially improved the reasoning capabilities of large language models. While existing analyses identify that RLVR-induced changes are sparse, they primarily focus on the \textbf{magnitude} of these updates, largely overlooking their \textbf{direction}. In this work, we argue that the direction of updates is a more critical lens for understanding RLVR's effects, which can be captured by the signed, token-level log probability difference $\Delta\log p$ between the base and final RLVR models. Through statistical analysis and token-replacement interventions, we demonstrate that $\Delta\log p$ more effectively identifies sparse, yet reasoning-critical updates than magnitude-based metrics (\eg divergence or entropy). Building on this insight, we propose two practical applications: (1) a \textit{test-time extrapolation} method that amplifies the policy along the learned $\Delta\log p$ direction to improve reasoning accuracy without further training; (2) a \textit{training-time reweighting} method that focuses learning on low-probability (corresponding to higher $\Delta\log p$) tokens, which improves reasoning performance across models and benchmarks. Our work establishes the direction of change as a key principle for analyzing and improving RLVR. 

---
# Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement 

**Authors**: Junrong Guo, Shancheng Fang, Yadong Qu, Hongtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2603.22187)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have enabled automated generation of structured layouts from natural language descriptions. Existing methods typically follow a code-only paradigm that generates code to represent layouts, which are then rendered by graphic engines to produce final images. However, they are blind to the rendered visual outcome, making it difficult to guarantee readability and aesthetics. In this paper, we identify visual feedback as a critical factor in layout generation and propose Visual Feedback Layout Model (VFLM), a self-improving framework that leverages visual feedback iterative refinement. VFLM is capable of performing adaptive reflective generation, which leverages visual information to reflect on previous issues and iteratively generates outputs until satisfactory quality is achieved. It is achieved through reinforcement learning with a visually grounded reward model that incorporates OCR accuracy. By rewarding only the final generated outcome, we can effectively stimulate the model's iterative and reflective generative capabilities. Experiments across multiple benchmarks show that VFLM consistently outperforms advanced MLLMs, existing layout models, and code-only baselines, establishing visual feedback as critical for design-oriented MLLMs. Our code and data are available at this https URL. 

---
# P^2O: Joint Policy and Prompt Optimization 

**Authors**: Xinyu Lu, Kaiqi Zhang, Jinglin Yang, Boxi Cao, Yaojie Lu, Hongyu Lin, Min He, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.21877)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs). However, vanilla RLVR suffers from inefficient exploration, particularly when confronting "hard samples" that yield nearzero success rates. In such scenarios, the reliance on sparse outcome rewards typically results in zero-advantage estimates, effectively starving the model of supervision signals despite the high informational value of these instances. To address this, we propose P^2O, a novel framework that synergizes Prompt Optimization with Policy Optimization. P^2O identifies hard samples during training iterations and leverages the GeneticPareto (GEPA) prompt optimization algorithm to evolve prompt templates that guide the model toward discovering successful trajectories. Crucially, unlike traditional prompt engineering methods that rely on input augmentation, P^2O distills the reasoning gains induced by these optimized prompts directly into the model parameters. This mechanism provides denser positive supervision signals for hard samples and accelerates convergence. Extensive experiments demonstrate that P^2O not only achieves superior performance on in-distribution datasets but also exhibits strong generalization, yielding substantial improvements on out-of-distribution benchmarks (+4.7% avg.). 

---
# When Models Judge Themselves: Unsupervised Self-Evolution for Multimodal Reasoning 

**Authors**: Zhengxian Wu, Kai Shi, Chuanrui Zhang, Zirui Liao, Jun Yang, Ni Yang, Qiuying Peng, Luyuan Zhang, Hangrui Xu, Tianhuang Su, Zhenyu Yang, Haonan Lu, Haoqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.21289)  

**Abstract**: Recent progress in multimodal large language models has led to strong performance on reasoning tasks, but these improvements largely rely on high-quality annotated data or teacher-model distillation, both of which are costly and difficult to this http URL address this, we propose an unsupervised self-evolution training framework for multimodal reasoning that achieves stable performance improvements without using human-annotated answers or external reward models. For each input, we sample multiple reasoning trajectories and jointly model their within group this http URL use the Actor's self-consistency signal as a training prior, and introduce a bounded Judge based modulation to continuously reweight trajectories of different this http URL further model the modulated scores as a group level distribution and convert absolute scores into relative advantages within each group, enabling more robust policy updates. Trained with Group Relative Policy Optimization (GRPO) on unlabeled data, our method consistently improves reasoning performance and generalization on five mathematical reasoning benchmarks, offering a scalable path toward self-evolving multimodal this http URL code are available at this https URL. 

---
# Reward Sharpness-Aware Fine-Tuning for Diffusion Models 

**Authors**: Kwanyoung Kim, Byeongsu Sim  

**Link**: [PDF](https://arxiv.org/pdf/2603.21175)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has proven effective in aligning large language models with human preferences, inspiring the development of reward-centric diffusion reinforcement learning (RDRL) to achieve similar alignment and controllability. While diffusion models can generate high-quality outputs, RDRL remains susceptible to reward hacking, where the reward score increases without corresponding improvements in perceptual quality. We demonstrate that this vulnerability arises from the non-robustness of reward model gradients, particularly when the reward landscape with respect to the input image is sharp. To mitigate this issue, we introduce methods that exploit gradients from a robustified reward model without requiring its retraining. Specifically, we employ gradients from a flattened reward model, obtained through parameter perturbations of the diffusion model and perturbations of its generated samples. Empirically, each method independently alleviates reward hacking and improves robustness, while their joint use amplifies these benefits. Our resulting framework, RSA-FT (Reward Sharpness-Aware Fine-Tuning), is simple, broadly compatible, and consistently enhances the reliability of RDRL. 

---
